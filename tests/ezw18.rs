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
    /*
     * ## Introduction to 64-bit floating-point values
     *
     * By the IEEE-754 standard, 64-bit floating-point values have the following bit layout:
     * s = sign     (1 bit)
     * e = exponent (11 bits), with a bias of 1023 (i.e. 1023 is zero)
     * m = mantissa (52 bits, starting with an implicit bit)
     *
     * c1 in binary:
     * 0100011001100000000000000000000000000000000000000000000000000000
     *
     * i.e. the sign is 0, so the value is positive, the exponent is 1126 - 1023 = 103, and the
     * mantissa is 0.
     *
     * c2 in binary:
     * 0100011001100000000000000000000000000000000000000000000000000011
     *
     * i.e. the sign is 0, so the value is positive, the exponent is 1126 - 1023 = 104, and the
     * mantissa is 3.
     *
     * ## Extracting the high bits
     *
     * To paraphase EZW18, since a * b is guaranteed to be less than 2^103 (as we use 51-bit
     * limbs), adding 2^103 in the floating-point domain ensures that 2^103 is the most significant
     * bit and also the implicit bit in its binary representation. The last 51 bits of the mantissa
     * is therefore the high product of a * b in the integer domain.
     *
     * e.g. a = 73061669485388f64 and b = 571904368095603f64.
     *
     * Let us examine the binary representation of a * b as a floating point value:
     *
     * 0100011000010101000110000111111000011101011100110011000010100100
     *     - sign: 0 (positive)
     *     - exponent: 10000110010 = 1121 - 1023 = 98
     *     - mantissa: 0101000110000111111000011101011100110011000010100100 (52 bits)
     * 
     * In order to make the higher bits show up in the mantissa, we need to sacrifice accuracy,
     * such that the binary representation loses the lower bits and exposes the higher bits. To do
     * so, we add 2^103, which sets the implicit bit to 2^103. The binary representation is now:
     *
     * 0100011001100000101010001100001111110000111010111001100110000101
     *     - sign: 0 (positive)
     *     - exponent: 1126 - 1023 = 103
     *     - mantissa: 0000101010001100001111110000111010111001100110000101
     *
     * Since the exponent is 2^103, the floating-point representation omits the lower bits and
     * shows the higher bits. To get the high bits as an integer, we convert the f64 value
     * into a u64 using the to_bits() function, and then subtract c1.to_bits(). 
     *
     * Below, we will discuss cases where 1 must be subtracted from the high term to get the
     * correct result.
     *
     * ## Extracting the low bits
     *
     * To extract the low bits, we perform these floating-point operations:
     *
     * sub = c2 - hi
     *     = (2^103 + 3 * 2^51) - hi
     * lo = fma(a, b, sub)
     *
     * Recall that hi = 2 ^ 103 + 2. Subtracting hi from c2 leaves us with a floating
     * point (sub) with this binary representation:
     *
     * 1100011000010101000110000111111000011101011100110011000001000000
     *     - sign: 1 (negative)
     *     - exponent: 1121 - 1023 = 98
     *     - mantissa: 0101000110000111111000011101011100110011000001000000
     *
     * After performing fma(a, b, sub), the mantissa will contain the lower
     * 51 bits of the integer product:
     *
     * 0100001100111000111101011100000110011000110101010110001100000000
     *     - sign: 0 (positive)
     *     - exponent: 1075 - 1023 = 52
     *     - mantissa: 2522011655299840
     *
     * TODO: explain why this is the case:
     * - c2 has exponent 103; hi has exponent 52; sub has exponent -98; and fma(a, b, sub) has
     *   exponent 52
     *
     * Finally, we extract the lower bits by subtracting 3 * 2 ^ 51 (which is 0x4338000000000000)
     * and then mask to get the lowest 51 bits.
     * 
     * lo = lo & (2^51 - 1)
     *    = 2522011655299840 & 0x7ffffffffffff
     *    = 270211841614592
     *
     * ## The conditional subtraction
     *
     * In another example (a = 55266722, b = 62775409), the high term needs to be subtracted by 1.
     * The condition for this subtraction is whether (lo - 3 * 2^51) is negative (by checking the
     * sign bit).
     *
     * ## The final normalisation steps
     *
     * To normalise the high term, we simply compute hi.to_bits() - c1.to_bits(). This works
     * because the first 12 bits of hi equal that of c1, while the remaining bits of c1 are all 0.
     *
     * A conditional subtraction may then be needed, as described below.
     *
     * To normalise the low bits, compute lo.to_bits() - (3 * 2^51).to_bits(). Recall that 3 * 2^51
     * as a floating-point value is positive, with an exponent of 52, and a mantissa of
     * 0x8000000000000 (which is 2^51).
     *
     * Since the exponent of lo is also 52, this operation
     * produces the subtraction of 2^51 from the 52-bit mantissa of lo. Finally, a 51-bit mask is
     * applied to obtain the lower 51 bits.
     *
     * e.g.:
     *
     * c1: (0, 103, 0000000000000)
     * c2: (0, 103, 0000000000003)
     * tt: (0, 52, 8000000000000)
     * hi : (0, 103, 0A8C3F0EB9985)
     * sub: (1, 98, 5187E1D733040)
     * lo : (0, 52, 8F5C198D56300)
     * (high, lo): 0A8C3F0EB9985, 0F5C198D56300
     *
     * If lo is smaller than 2^51, then an overflow will occur, causing the 64th bit of
     * lo.to_bits() - tt.to_bits() to equal 1. In this case, we subtract (aka borrow) 1 from the
     * high term, a la grade-school subtraction.
     */

    let mask = 2u64.pow(51) - 1;
    //// c1 = 2^103
    //// c2 = 2^103 + 3 * 2^51

    //let c1 = f64::from_bits(0x4660000000000000u64);
    //let c2 = f64::from_bits(0x4660000000000003u64);
    //let tt = 0x4338000000000000u64;

    let c1: f64 = 2f64.powi(103i32);
    let c2: f64 = c1.add(2f64.powi(51i32).mul(3f64));
    let tt: f64 = 2f64.powi(51i32).mul(3f64);

    print_f64("c1", c1);
    print_f64("c2", c2);
    console::log_1(&format!("c2: {:064b}", c2.to_bits()).into());
    print_f64("tt", tt);
    console::log_1(&format!("tt: {:064b}", tt.to_bits()).into());

    let tt: u64 = tt.to_bits();

    let mut hi = a.mul_add(b, c1);
    print_f64("hi ", hi);
    console::log_1(&format!("hi: {:064b}", hi.to_bits()).into());

    let sub = c2 - hi;
    print_f64("sub", sub);

    let mut lo = a.mul_add(b, sub);
    print_f64("lo ", lo);

    //console::log_1(&format!("a:  {:64b}", a.to_bits()).into());
    //console::log_1(&format!("b:  {:64b}", b.to_bits()).into());
    //console::log_1(&format!("hi:  {:64b}", lo.to_bits()).into());
    //console::log_1(&format!("c2:  {:64b}", c2.to_bits()).into());
    //console::log_1(&format!("sub: {:64b}", sub.to_bits()).into());
    //console::log_1(&format!("lo:  {:64b}", lo.to_bits()).into());

    //console::log_1(&format!("c2 fp:  {}", c2).into());
    //console::log_1(&format!("hi fp:  {}", hi).into());
    //console::log_1(&format!("sub fp: {}", sub).into());

    //console::log_1(&format!("c2 bits:  {:64b}", c2.to_bits()).into());
    //console::log_1(&format!("hi bits:  {:64b}", hi.to_bits()).into());
    //console::log_1(&format!("sub bits: {:64b}", sub.to_bits()).into());

    let mut hi = hi.to_bits() - c1.to_bits();
    let mut lo = lo.to_bits() - tt;

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

    //// With these values, sub goes negative
    let a = 730616694853888f64;
    let b = 571904368095603f64;

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

        //console::log_1(&format!("a: {}", a).into());
        //console::log_1(&format!("b: {}", b).into());

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
