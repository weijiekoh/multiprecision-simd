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

/// In Zprize 2023, Emmart found a way to use f64s in WASM (to be precise, dual-lane f64 relaxed
/// SIMD values) to perform multiprecision arithmetic with 51-bit limbs.
/// This function modifies int_full_product() of EZW18 with his technique.
/// The key is to subtract 1 from the masked high product if the (unmasked) low product is
/// negative.
/// Adapted from https://github.com/z-prize/2023-entries/blob/a568b9eb1e73615f2cee6d0c641bb6a7bc5a428f/prize-2-msm-wasm/prize-2b-twisted-edwards/yrrid-snarkify/yrrid/SupportingFiles/FP51.java#L47
/// a and b should have at most 51 bits.
fn niall_full_product(a: f64, b: f64) -> (u64, u64) {
    let mask = 2u64.pow(51) - 1;
    //// c1 = 2^103
    //// c2 = 2^103 + 3 * 2^51

    //let c1 = f64::from_bits(0x4660000000000000u64);
    //let c2 = f64::from_bits(0x4660000000000003u64);
    //let tt = 0x4338000000000000u64;

    let c1: f64 = 2f64.powi(103i32);
    let c2: f64 = c1.add(2f64.powi(51i32).mul(3f64));
    let tt = 2f64.powi(51i32).mul(3f64).to_bits();

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
     * i.e. the sign is 0, so the value is positive, the exponent is 1126 - 1023 = 104, and the
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
     * e.g. a = 55266722.0f64 and b = 62775409.0f64. The high term of their product is 1, and the
     * low term is 1217591263954050. In binary:
     * hi: 000000000000000000000000000000000000000000000000001
     * lo: 100010100110110010010001110110001010000010010000010
     *
     * Let us examine the binary representation of a * b as a floating point value:
     *
     * 0100001100101000101001101100100100011101100010100000100100000100
     *     - sign: 0 (positive)
     *     - exponent: 10000110010 = 1074 - 1023 = 51
     *     - mantissa: 1000101001101100100100011101100010100000100100000100 (52 bits)
     * 
     * We see that the 52-bit mantissa only encodes the lower 51 bits, albeit in an unaligned
     * manner, since it is shifted left by 1.
     *
     * In order to make the higher bits show up in the mantissa, we need to sacrifice accuracy,
     * such that the binary representation loses the lower bits and exposes the higher bits. To do
     * so, we add 2^103, which sets the implicit bit to 2^103. The binary representation is now:
     *
     * 0100011001100000000000000000000000000000000000000000000000000010
     *     - sign: 0 (positive)
     *     - exponent: 1126 - 1023 = 103
     *     - mantissa: 10 in binary, which is 2
     *
     * (We will conditionally subtract 1 because, as we will find out later, the low term is
     * negative.)
     *
     * ## Extracting the low bits
     *
     * To extract the low bits, we perform this in the floating-point domain:
     *
     * sub = c2 - hi
     *     = (2^103 + 3 * 2^51) - hi
     * lo = (a * b) + sub
     *
     * Recall that hi = 2 ^ 103 + 2. Subtracting hi from c2 leaves us with a floating
     * point (sub) with this binary representation:
     *
     * 0100001100100000000000000000000000000000000000000000000000000000
     *     - sign: 0 (positive)
     *     - exponent: 1074 - 1023 = 51
     *     - mantissa: 0
     *
     * Adding sub to (a * b) sets the implicit bit to 2^51, so the mantissa will contain the lower
     * 51 bits of the integer product:
     *
     * 0100001100110100010100110110010010001110110001010000010010000010
     *     - sign: 0 (positive)
     *     - exponent: 1075 - 1023 = 52
     *     - mantissa: 1217591263954050
     * 
     * ### Why should c2 equal 2^103 + 3 * 2^51?
     *
     * First of all, we know that the exponent of hi is 2^103, so we need to cancel it out in the
     * floating-point domain.
     *
     * Next, we know that the binary representation of the floating-point value 2^103 + 3 * 2^51
     * is:
     *
     * 0100011001100000000000000000000000000000000000000000000000000011
     *     - sign: 0 (positive)
     *     - exponent: 103
     *     - mantissa: 3
     *
     *
     * sub = (2^103 + 3 * 2^51) - hi;
     * lo = fma(a, b, sub)
     *
     * ### A side note
     *
     * TODO: Figure out why c2 is 2^103 + 3 * 2^51.
     * TODO: Figure out why sub can be negative
     *
     * Why not just add 2^51 or 3 * 2^51 to (a * b)?
     *
     * #### Performing a * b + 2^51:
     *
     * 0100001100110100010100110110010010001110110001010000010010000010
     *     - sign: 0 (positive)
     *     - exponent: 1075 - 1023 = 52
     *     - mantissa: 1217591263954050
     *
     *  This leads to a correct result for this particular example, but incorrect results for many
     *  other inputs, so it is wrong.
     *  For example, with a = 1596695558896492 and b: 1049164860932151, this method gives us:
     *
     *  a * b + 2^51:
     *  1111111111111100010100110110010010001110110001010000010010000010
     *      - sign: 1 (negative)
     *      - exponent: 2047 - 1024 = 1023
     *      - mantissa: 3469391077639298
     *
     * It appears that an overflow occured, which led to a completely unusable mantissa.
     *
     * #### Performing a * b + 3 * 2^51:
     *
     * Simply adding 3 * 2^51 (which is a 53-bit value) causes us to lose the lower bits. With our
     * original example, we get:
     *
     * 0100011001100000000000000000000000000000000000000000000000000101
     *     - sign: 0 (positive)
     *     - exponent: 1126 - 1023 = 103n
     *     - mantissa: 5
     *
     * It appears that 
     */

    let mut hi = a.mul_add(b, c1);
    let sub = c2 - hi;
    let mut lo = a.mul_add(b, sub);

    //console::log_1(&format!("c2 fp:  {}", c2).into());
    //console::log_1(&format!("hi fp:  {}", hi).into());
    //console::log_1(&format!("sub fp: {}", sub).into());

    //console::log_1(&format!("c2 bits:  {:64b}", c2.to_bits()).into());
    //console::log_1(&format!("hi bits:  {:64b}", hi.to_bits()).into());
    //console::log_1(&format!("sub bits: {:64b}", sub.to_bits()).into());

    /*
     * ## Normalising the high and low bits
     *
     * Recall that we extracted the high bits by adding 2^103 to a * b. We can get the high bits as
     * an integer by converting the f64 value into a u64 using the to_bits() function, and then
     * subtracting c1.to_bits(). Next, we do a conditional subtraction of 1, depending on whether
     * (lo - 3 * 2^51) is negative (by checking the sign bit). Finally, the lower bits are
     * extracted by subtracting 3 * 2 ^ 51 (which is 0x4338000000000000) and then masking to get
     * the lowest 51 bits.
     */

    let mut hi = hi.to_bits() - c1.to_bits();
    let mut lo = lo.to_bits() - tt;

    // If the lower word is negative, subtract 1 from the higher word
    if lo & 0x8000000000000000u64 > 0 {
        hi -= 1;
    } else {
    }

    lo = lo & mask;

    (hi, lo)
}

#[test]
#[wasm_bindgen_test]
fn test_niall_zprize() {
    // These values would cause an incorrect result from int_full_product(), but a correct result
    // from niall_full_product().
    let a = 55266722.0f64;
    let b = 62775409.0f64;

    // With these values, sub goes negative
    //let a = 730616694853888f64;
    //let b = 571904368095603f64;

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
const NUM_RUNS: u32 = 1000;
#[test]
#[wasm_bindgen_test]
fn test_niall_zprize_multi() {
    let limb_size = 51;
    let mut rng = gen_seeded_rng(0);

    for _ in 0..NUM_RUNS {
        let a: BigUint = rng.sample(RandomBits::new(limb_size));
        let b: BigUint = rng.sample(RandomBits::new(limb_size));

        let a = (a.to_u64_digits()[0] as f64).trunc();
        let b = (b.to_u64_digits()[0] as f64).trunc();

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
        assert_eq!(s, expected)
    }
}
