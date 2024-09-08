#![cfg(target_arch = "wasm32")]
mod utils;

use num_traits::Pow;
use wasm_bindgen_test::*;
use web_sys::console;

wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

/// "Faster Modular Exponentiation Using Double Precision Floating Point Arithmetic on the GPU"
/// by Niall Emmart, Fangyu Zheng, and Charles Weems
/// https://ieeexplore.ieee.org/document/8464792/similar#similar

fn fma_rz(a: f64, b: f64, c: f64) -> f64 {
    let r = a.mul_add(b, c);
    r
}

fn full_product(a: f64, b: f64) -> (f64, f64) {
    let p_hi = fma_rz(a, b, 0.0);
    let p_lo = fma_rz(a, b, -p_hi);
    (p_hi, p_lo)
}

/*
#[test]
#[wasm_bindgen_test]
fn test_fig_2() {
    let a0 = 2f64.pow(50);
    let a1 = 1f64;
    let b0 = 2f64.pow(50);
    let b1 = 1f64;

    let p0 = full_product(a0, b0);
    let p1 = full_product(a1, b1);

    let sum = (p0.0 + p1.0, p0.1 + p1.1);

    //console::log_1(&format!("Figure 2:").into());
    //console::log_1(&format!("p0:  {:?}", p0).into());
    //console::log_1(&format!("p1:  {:?}", p1).into());
    //console::log_1(&format!("sum: {:?}", sum).into());

    // p131-2:
    // "Adding the high products and low products of p0 and p1, yields (2^100, 0) rather than the
    // desired sum (2^100, 1), due to alignment and round-off problems."
    let expected_0 = 2f64.pow(100);
    let expected_1 = 0f64;
    assert_eq!(sum.0, expected_0);
    assert_eq!(sum.1, expected_1);
}

fn dpf_full_product(a: f64, b: f64) -> (f64, f64) {
    let c1 = 2f64.pow(104);
    let c2 = 2f64.pow(104) + 2f64.pow(52);
    let c3 = 2f64.pow(52);

    let p_hi = fma_rz(a, b, c1);
    let sub = c2 - p_hi;
    let p_lo = fma_rz(a, b, sub);

    (p_hi - c1, p_lo - c3)
}

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
*/

use std::ops::{Add, Mul};

fn int_full_product(a: f64, b: f64) -> (u64, u64) {
    // This code uses slightly different values: https://github.com/z-prize/2023-entries/blob/a568b9eb1e73615f2cee6d0c641bb6a7bc5a428f/prize-2-msm-wasm/prize-2b-twisted-edwards/yrrid-snarkify/yrrid/SupportingFiles/FP51.java#L47
    //let c1: f64 = 2f64.pow(103);
    //let t: f64 = 3f64.mul(2f64.pow(51));
    ////let c2 = c1.add(t);
    //let c2 = 2f64.pow(103) + 3f64 * 2f64.pow(51);

    let c1: f64 = 2f64.pow(103);
    let t: f64 = 3f64.mul(2f64.pow(51));
    let c2: f64 = c1.add(3f64.mul(2f64.pow(51)));

    // The paper uses c1 = 2^104 and c2 = 2^104 + 2^52
    //let c1 = 2f64.pow(104);
    //let c2 = 2f64.pow(104) + 2f64.pow(52);
    //// 52-bit mask
    //let mask = 2u64.pow(52) - 1;

    //let c1: f64 = 2f64.pow(104);
    //let c2: f64 = 2f64.pow(104) + 2f64.pow(52);
    let mask = 2u64.pow(51) - 1;

    let p_hi = fma_rz(a, b, c1);
    let sub = c2 - p_hi;
    let p_lo = fma_rz(a, b, sub);

    //console::log_1(&format!("c2:   {}", c2).into());
    //console::log_1(&format!("c2:   {:?}", c2).into());
    console::log_1(&format!("t.to_bits(): {}", t.to_bits()).into());
    console::log_1(&format!("t:   {}", t).into());
    //console::log_1(&format!("c2:   {}", f64::from_bits(5075556780046548993u64)).into());
    //console::log_1(&format!("c1:   {}", c1.to_bits()).into());
    //console::log_1(&format!("c2:   {}", c2.to_bits()).into());
    console::log_1(&format!("p_hi: {}", p_hi).into());
    console::log_1(&format!("sub:  {}", sub).into());
    //console::log_1(&format!("p_lo: {}", p_lo.to_bits()).into());

    console::log_1(&format!("{}", (p_hi.to_bits() - c1.to_bits() & mask)).into());
    console::log_1(&format!("{}", (p_lo.to_bits() - t.to_bits() & mask)).into());
    (p_hi.to_bits() & mask, p_lo.to_bits() & mask)
}

#[test]
#[wasm_bindgen_test]
fn test_fig_4() {
    let mask = 2u64.pow(51) - 1;

    let a = 55266722.0f64;
    let b = 62775409.0f64;

    let c1: f64 = 2f64.powi(103i32);
    let c2 = f64::from_bits(5071053180419178499u64);

    //console::log_1(&format!("a: {:?}", a).into());
    //console::log_1(&format!("b: {:?}", b).into());

    let hi = a.mul_add(b, c1);
    let sub = c2 - hi;
    let lo = a.mul_add(b, sub);

    let mut hi = hi.to_bits() - c1.to_bits();
    //console::log_1(&format!("lo bits: {:?}", lo.to_bits()).into());
    let mut lo = lo.to_bits() - 0x4338000000000000u64;

    if lo >> 63 == 1 {
        hi -= 1;
    }
    lo = lo & mask;

    console::log_1(&format!("hi bits: {:?}", hi).into());
    console::log_1(&format!("lo bits: {:?}", lo).into());
    //console::log_1(&format!("lo as f64: {}", f64::from_bits(lo)).into());
    //console::log_1(&format!("lo is positive: {}", f64::from_bits(lo).is_sign_positive()).into());

    let a = num_bigint::BigUint::from(a as u64);
    let b = num_bigint::BigUint::from(b as u64);
    let expected = &a * &b;
    console::log_1(&format!("expected: {:?}", expected).into());

    let x = num_bigint::BigUint::from(2u32).pow(51u32);
    let s = x * hi + lo;

    console::log_1(&format!("integer:  {:?}", s).into());
    assert_eq!(s, expected)

    // It appears that f64::mul_add() does not have the same round-to-zero behaviour as CUDA's
    // __fma_rz. This causes an incorrect result for 52-bit limbs. Apparently, the relaxed SIMD
    // function f64x2_relaxed_madd() will work, but I'm having trouble getting it running in Rust.
    // The next step is to find out if the rounding mode really matters for slightly smaller limbs
    // (e.g. 48 bits).
}

/*
use num_bigint::{BigUint, RandomBits};
use crate::utils::gen_seeded_rng;
use rand::Rng;

const NUM_RUNS: u32 = 100;

// This tests fails, which shows that Rust's f64::mul_add() rounding mode is not compatible with
// Figure 4.
#[test]
#[wasm_bindgen_test]
fn test_fuzz_fig_4() {
    let limb_size = 26;

    // Generate random pairs of f64s
    // Compute the integer product (using int_full_product) and compare against the BigUint product
    let mut rng = gen_seeded_rng(0);

    for _ in 0..NUM_RUNS {
        let a: BigUint = rng.sample(RandomBits::new(limb_size));
        let b: BigUint = rng.sample(RandomBits::new(limb_size));
        let expected = &a * &b;

        console::log_1(&format!("a: {:?}", a).into());
        console::log_1(&format!("b: {:?}", b).into());

        let a = (a.to_u64_digits()[0] as f64).trunc();
        let b = (b.to_u64_digits()[0] as f64).trunc();

        let p = int_full_product(a, b);
        console::log_1(&format!("p: {:?}", p).into());

        let x = num_bigint::BigUint::from(2u32).pow(51u32);
        let ph = num_bigint::BigUint::from(p.0);
        let pl = num_bigint::BigUint::from(p.1);

        //console::log_1(&format!("{:?}", p).into());
        assert_eq!(x * ph + pl, expected);
        console::log_1(&format!("").into());
    }
}
*/
