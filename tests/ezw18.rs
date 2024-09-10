#![cfg(target_arch = "wasm32")]
mod utils;

use num_traits::Pow;
use std::ops::{Add, Mul};
use wasm_bindgen_test::*;
use web_sys::console;
use num_bigint::{BigUint, RandomBits};
use crate::utils::gen_seeded_rng;
use rand::Rng;

wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

/// These test functions implement the algorithms describe in EZW18:
/// "Faster Modular Exponentiation Using Double Precision Floating Point Arithmetic on the GPU"
/// by Niall Emmart, Fangyu Zheng, and Charles Weems
/// https://ieeexplore.ieee.org/document/8464792/similar#similar

/// Fused-multiply-add is the Rust f64 data type's mul_add function. It is very important to note
/// that its rounding behaviour is not necessarily the same as that the CUDA __fma_rz() function
/// that EZW18 uses, which always rounds down to zero.
/// https://doc.rust-lang.org/std/primitive.f64.html#method.mul_add
fn fma(a: f64, b: f64, c: f64) -> f64 {
    let r = a.mul_add(b, c);
    r
}

/// The algorithm described in Figure 2, which is flawed because the results suffer from alignment
/// and round-off problems.
fn full_product(a: f64, b: f64) -> (f64, f64) {
    let p_hi = fma(a, b, 0.0);
    let p_lo = fma(a, b, -p_hi);
    (p_hi, p_lo)
}

/// This test illustrates the alignment and round-off problem that EZW18 describes (p131-2).
#[test]
#[wasm_bindgen_test]
fn test_fig_2() {
    let a0 = 2f64.powi(50i32);
    let a1 = 1f64;
    let b0 = 2f64.powi(50i32);
    let b1 = 1f64;

    let p0 = full_product(a0, b0);
    let p1 = full_product(a1, b1);

    let sum = (p0.0 + p1.0, p0.1 + p1.1);

    //console::log_1(&format!("Figure 2:").into());
    //console::log_1(&format!("p0:  {:?}", p0).into());
    //console::log_1(&format!("p1:  {:?}", p1).into());
    //console::log_1(&format!("sum: {:?}", sum).into());

    // "Adding the high products and low products of p0 and p1, yields (2^100, 0) rather than the
    // desired sum (2^100, 1), due to alignment and round-off problems." (p131-2)
    let expected_0 = 2f64.pow(100);
    let expected_1 = 0f64;
    assert_eq!(sum.0, expected_0);
    assert_eq!(sum.1, expected_1);
}

/// The algorithm described in Figure 3.
fn dpf_full_product(a: f64, b: f64) -> (f64, f64) {
    let c1: f64 = 2f64.powi(104i32);
    let c2: f64 = 2f64.powi(104i32).add(2f64.powi(52i32));
    let c3: f64 = 2f64.powi(52i32);

    let p_hi = fma(a, b, c1);
    let sub = c2 - p_hi;
    let p_lo = fma(a, b, sub);

    (p_hi - c1, p_lo - c3)
}

/// Shows how adding 2^104 to the high product and subtracting 2^52 from the low product "forces
/// alignment of the decimal places" (p132). Note that this is algorithm is not suitable for
/// platforms that do not have the equivalent of the __fma_rz() function. See below for how Emmart
/// resolves this issue when one does not have control over the rounding behaviour.
#[test]
#[wasm_bindgen_test]
fn test_fig_3() {
    let a0 = 2f64.pow(50);
    let a1 = 1f64;
    let b0 = 2f64.pow(50);
    let b1 = 1f64;

    let p0 = dpf_full_product(a0, b0);
    let p1 = dpf_full_product(a1, b1);

    let sum = (p0.0 + p1.0, p0.1 + p1.1);

    let expected_0 = 2f64.pow(100);
    let expected_1 = 1f64;
    assert_eq!(sum.0, expected_0);
    assert_eq!(sum.1, expected_1);

    //console::log_1(&format!("Figure 3:").into());
    //console::log_1(&format!("p0:  {:?}", p0).into());
    //console::log_1(&format!("p1:  {:?}", p1).into());
    //console::log_1(&format!("sum: {:?}", sum).into());
}

/// The algorithm described in Figure 4. Again, it does not always work correctly on platforms
/// which do not give the developer control over the rounding mode.
fn int_full_product(a: f64, b: f64) -> (u64, u64) {
    // The paper uses c1 = 2^104 and c2 = 2^104 + 2^52
    let c1 = 2f64.pow(104);
    let c2 = 2f64.pow(104) + 2f64.pow(52);
    // 52-bit mask
    let mask = 2u64.pow(52) - 1;

    let p_hi = fma(a, b, c1);
    let sub = c2 - p_hi;
    let p_lo = fma(a, b, sub);

    (p_hi.to_bits() & mask, p_lo.to_bits() & mask)
}

fn print_f64(label: &str, a: f64) {
    let bits = a.to_bits();
    
    // Extract the sign bit (most significant bit)
    let sign = (bits >> 63) & 1;
    
    // Extract the exponent bits (next 11 bits)
    let exponent = ((bits >> 52) & 0x7FF) as i16;
    
    // Subtract the bias (1023) from the exponent
    let unbiased_exponent = exponent - 1023;
    
    // Extract the mantissa bits (last 52 bits)
    let mantissa = bits & 0x000F_FFFF_FFFF_FFFF;

    // Print the components
    console::log_1(&format!("{}: ({}, {}, 0x{:013X})", label, sign, unbiased_exponent, mantissa).into());
}

/// In Zprize 2023, Emmart found a way to use f64s in WASM (to be precise, dual-lane f64 relaxed
/// SIMD values) to perform multiprecision arithmetic with 51-bit limbs.
/// This function modifies int_full_product() of EZW18 with his technique.
/// The key is to subtract 1 from the masked high product if the (unmasked) low product is
/// negative.
/// Adapted from https://github.com/z-prize/2023-entries/blob/a568b9eb1e73615f2cee6d0c641bb6a7bc5a428f/prize-2-msm-wasm/prize-2b-twisted-edwards/yrrid-snarkify/yrrid/SupportingFiles/FP51.java#L47
/// a and b should have at most 51 bits.
fn niall_full_product(a: f64, b: f64) -> (u64, u64) {
    console::log_1(&format!("a: {}", a).into());
    console::log_1(&format!("b: {}", b).into());

    let mask = 2u64.pow(51) - 1;
    //// c1 = 2^103
    //// c2 = 2^103 + 3 * 2^51

    //let c1 = f64::from_bits(0x4660000000000000u64);
    //let c2 = f64::from_bits(0x4660000000000003u64);
    //let tt = 0x4338000000000000u64;

    let c1: f64 = 2f64.powi(103i32);
    let c2: f64 = c1.add(2f64.powi(51i32).mul(3f64));
    let tt: f64 = 2f64.powi(51i32).mul(3f64);

    print_f64("tt", tt);
    let tt: u64 = tt.to_bits();

    let mut hi = a.mul_add(b, c1);
    print_f64("hi = a.mul_add(b, c1)", hi);

    let sub = c2 - hi;
    print_f64("sub = c2 - h1", sub);

    let mut lo = a.mul_add(b, sub);
    print_f64("lo = a.mul_add(b, sub)", lo);

    console::log_1(&format!("lo: {:013X}", lo.to_bits()).into());

    let mut hi = hi.to_bits() - c1.to_bits();
    console::log_1(&format!("hi = hi - c1: {:013X}", hi).into());

    let mut lo = lo.to_bits() - tt;
    console::log_1(&format!("lo = lo - tt: {:013X}", lo).into());

    // If the lower word is negative, subtract 1 from the higher word
    if lo & 0x8000000000000000u64 > 0 {
        hi -= 1;
    } else {
    }

    lo = lo & mask;
    console::log_1(&format!("(high, lo): {:013X}, {:013X}", hi, lo).into());

    (hi, lo)
}

#[test]
#[wasm_bindgen_test]
fn test_niall_zprize() {
    // These values would cause an incorrect result from int_full_product(), but a correct result
    // from niall_full_product().
    //let a = 55266722.0f64;
    //let b = 62775409.0f64;

    // With these values, a conditional reduction is performed
    let a = 1596695558896492f64;
    let b = 1049164860932151f64;

    // With these values, a conditional reduction is not performed
    //let a = 1574331031398118f64;
    //let b = 135495742621820f64;

    let (hi, lo) = niall_full_product(a, b);
    let a = num_bigint::BigUint::from(a as u64);
    let b = num_bigint::BigUint::from(b as u64);
    let expected = &a * &b;

    let x = num_bigint::BigUint::from(2u32).pow(51u32);
    let s = x * hi + lo;

    //console::log_1(&format!("expected: {:?}", expected).into());
    //console::log_1(&format!("integer:  {:?}", s).into());
    assert_eq!(s, expected)
}
/*
*/

/// Run tests for niall_full_product() on a large number of random inputs.
const NUM_RUNS: u32 = 3;
#[test]
#[wasm_bindgen_test]
fn test_niall_zprize_multi() {
    let limb_size = 51;
    let mut rng = gen_seeded_rng(0);

    for _ in 0..NUM_RUNS {
        let a: BigUint = rng.sample(RandomBits::new(limb_size));
        let b: BigUint = rng.sample(RandomBits::new(limb_size));

        let a = a.to_u64_digits()[0] as f64;
        let b = b.to_u64_digits()[0] as f64;

        console::log_1(&format!("a: {}", a).into());
        console::log_1(&format!("b: {}", b).into());

        let (hi, lo) = niall_full_product(a, b);
        let a = num_bigint::BigUint::from(a as u64);
        let b = num_bigint::BigUint::from(b as u64);
        let expected = &a * &b;

        let x = num_bigint::BigUint::from(2u32).pow(limb_size);
        let s = x * hi + lo;

        //console::log_1(&format!("expected: {:?}", expected).into());
        //console::log_1(&format!("integer:  {:?}", s).into());
        assert_eq!(s, expected);
        console::log_1(&format!("").into());
    }
}

#[test]
#[wasm_bindgen_test]
fn test_misc() {
    let a = 1596695558896492f64;
    let b = 1049164860932151f64;

    let c1: f64 = 2f64.powi(103i32);
    let c2: f64 = c1.add(2f64.powi(51i32).mul(3f64));
    print_f64("c1 ", c1);
    print_f64("c2 ", c2);

    // hi : (0, 103, 0x2A49B4C0B53D7)
    let mut hi = f64::from_bits(5071797117275296727u64);
    print_f64("hi ", hi);

    let sub = c2 - hi;
    print_f64("sub", sub);

    // hi : (0, 103, 0x2A49B4C0B53D8)
    let mut hi = f64::from_bits(5071797117275296728u64);
    print_f64("hi ", hi);

    let sub = c2 - hi;
    print_f64("sub", sub);

    let mut lo = a.mul_add(b, sub);
    print_f64("lo ", lo);
}
