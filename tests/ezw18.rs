#![cfg(target_arch = "wasm32")]
mod utils;

use crate::utils::gen_seeded_rng;
use num_bigint::{BigUint, RandomBits};
use num_traits::Pow;
use rand::Rng;
use std::ops::{Add, Mul};
use wasm_bindgen_test::*;
use web_sys::console;

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
    console::log_1(
        &format!(
            "{}: ({}, {}, 0x{:013X})",
            label, sign, unbiased_exponent, mantissa
        )
        .into(),
    );
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

    let mut hi = a.mul_add(b, c1);
    print_f64("hi = a.mul_add(b, c1)", hi);

    let sub = c2 - hi;
    print_f64("sub = c2 - h1", sub);

    let mut lo = a.mul_add(b, sub);
    print_f64("lo = a.mul_add(b, sub)", lo);

    console::log_1(&format!("lo: {:013X}", lo.to_bits()).into());

    let mut hi = hi.to_bits() - c1.to_bits();
    console::log_1(&format!("hi = hi - c1: {:013X}", hi).into());

    let mut lo = lo.to_bits() - tt.to_bits();
    //console::log_1(&format!("lo = lo - tt: {:013X}", lo).into());
    console::log_1(&format!("lo: {:64b}", tt.to_bits()).into());
    console::log_1(&format!("tt: {:64b}", tt.to_bits()).into());

    // If there is an overflow, subtract 1 from the high term.
    if lo & 0x8000000000000000u64 > 0 {
        console::log_1(&format!("").into());
        hi -= 1;
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
    //let a = 1596695558896492f64; let b = 1049164860932151f64;

    // With these values, a conditional reduction is not performed
    let a = 1574331031398118f64; let b = 135495742621820f64;

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
    let c1: f64 = 2f64.powi(103i32);
    let c2: f64 = c1.add(2f64.powi(51i32).mul(3f64));
    let tt: f64 = 2f64.powi(51i32).mul(3f64);
    // Underflow:
    let a = 1979581330507385f64; let b = 1237434108915901f64;

    // Non-underflow:
    //let a = 1574331031398118f64; let b = 135495742621820f64;

    console::log_1(&format!("a: {}", a as u64).into());
    console::log_1(&format!("b: {}", b as u64).into());

    print_f64("c1 ", c1);
    console::log_1(&format!("c1: {:64b}", c1.to_bits()).into());
    print_f64("c2 ", c2);
    console::log_1(&format!("c2: {:64b}", c2.to_bits()).into());

    let mut hi = a.mul_add(b, c1);
    // hi: (0, 103, 0x3DD62DAC05B2B)
    let sub = c2 - hi;
    // lo : (0, 52, 0x769521FAF3755)
    let mut lo = a.mul_add(b, sub);

    print_f64("hi ", hi);
    console::log_1(&format!("hi: {:64b}", hi.to_bits()).into());
    print_f64("lo ", lo);
    console::log_1(&format!("lo: {:64b}", lo.to_bits()).into());
    print_f64("sub ", sub);
    console::log_1(&format!("sub: {:64b}", sub.to_bits()).into());
    print_f64("tt ", tt);
    console::log_1(&format!("tt: {:64b}", tt.to_bits()).into());
    let mut lo = lo.to_bits() - tt.to_bits();
    console::log_1(&format!("lo (after): {:64b}", lo).into());

    let mut hi = hi.to_bits() - c1.to_bits();
    // If there is an overflow, subtract 1 from the high term.
    if lo & 0x8000000000000000u64 > 0 {
        console::log_1(&format!("Overflow occured; subtraction done").into());
        hi -= 1;
    }

    let mask = 2u64.pow(51) - 1;
    lo = lo & mask;
    console::log_1(&format!("(high, lo): {:013X}, {:013X}", hi, lo).into());

    console::log_1(&format!("-------------------------------").into());
    ///////////////////////////////////////////////////////////////////
    // TEST 1
    // low 51 bits: 100000000000000000000000000000000000000000000000000
    // exponent: 103
    // will it round up?
    let test = 2f64.powi(50i32).add(c1);
    print_f64("test 1: 2^50 + 2^103", test);
    console::log_1(&format!("test 1: {:64b}", test.to_bits()).into());

    // Answer: it did NOT round up
    //  ╭─ e=103──╮
    // 0100011001100000000000000000000000000000000000000000000000000000
    console::log_1(&format!("-------------------------------").into());

    ///////////////////////////////////////////////////////////////////
    // TEST 2
    // low 51 bits: 110000000000000000000000000000000000000000000000000
    let test = 1688849860263936f64.add(c1);
    print_f64("test 2: 3 * 2^49 + 2^103", test);
    console::log_1(&format!("test 2: {:64b}", test.to_bits()).into());
    //  ╭─ e=103──╮
    // 0100011001100000000000000000000000000000000000000000000000000001
    console::log_1(&format!("-------------------------------").into());

    /*
    ///////////////////////////////////////////////////////////////////
    // TEST 3
    // (2^49 - 1) * (2^49-123) + 1125899906842747
    //                                                ╭─ lower 51 bits                                ──╮
    // 11111111111111111111111111111111111111111100001100000000000000000000000000000000000000000011110110
    */
    let a = 562949953421311f64;
    let b = 562949953421189f64;
    let c = 1125899906842747f64.add(c1);
    let hi = a.mul_add(b, c);
    console::log_1(&format!("test 3: {:64b}", hi.to_bits()).into());
    // In full binary:
    //                                                                 ╭─ lower 51 bits                                ──╮
    //            10000011111111111111111111111111111111111111111100001100000000000000000000000000000000000000000011110110
    // In floating-point:
    //  ╭─ e=103──╮
    // 0100011001100000011111111111111111111111111111111111111111100010
    //
    // Looks like it rounded up!

    //////////
    //TEST 4
    let a = 562949953421311f64;
    let b = 562949953421189f64;
    let c = 562949953421435f64.add(c1);
    let hi = a.mul_add(b, c);
    console::log_1(&format!("test 3: {:64b}", hi.to_bits()).into());
    // In full binary:
    //                                                                 ╭─ lower 51 bits                                ──╮
    //                  11111111111111111111111111111111111111111100001010000000000000000000000000000000000000000011110110
    // In floating-point:
    //  ╭─ e=103──╮
    // 0100011001100000011111111111111111111111111111111111111111100001
    //
    // Rounding didn't happen.



    ///////////////////////////////////////////////////////////////////////////////////////////////
    // # Illustration
    //
    // ## Underflow case
    //
    // let a = 1979581330507385f64; let b = 1237434108915901f64;
    // 
    // hi = a * b + c1
    //                                                      ╭─ The rounding bit (51)
    //  ╭─ The top 52 bits ────────────────────────────────╮╭── Lower 51 bits that will be discarded ─────────╮
    // 00011110111010110001011011010110000000101101100101010111011010010101001000011111101011110011011101010101              a * b (integer)
    // 10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000 c1 = 2^103
    // 10011110111010110001011011010110000000101101100101010111011010010101001000011111101011110011011101010101 hi = 2^103 + a * b
    // 10011110111010110001011011010110000000101101100101011000000000000000000000000000000000000000000000000000 hi after FMA rounding
    //
    // sub = c2 - hi
    // 10000000000000000000000000000000000000000000000000011000000000000000000000000000000000000000000000000000 c2 = 2^103 + 3 * 2^51
    // 10011110111010110001011011010110000000101101100101011000000000000000000000000000000000000000000000000000 hi after FMA rounding
    //   -11110111010110001011011010110000000101101100101000000000000000000000000000000000000000000000000000000 sub = c2 - (hi after rounding). see that the high bits are the same!
    //
    // lo = a * b + sub
    //                                                      ╭─ The rounding bit (51)
    //  ╭─ The top 52 bits ────────────────────────────────╮╭── Lower 51 bits will be kept           ─────────╮
    // 00011110111010110001011011010110000000101101100101010111011010010101001000011111101011110011011101010101              a * b (integer)
    //   -11110111010110001011011010110000000101101100101000000000000000000000000000000000000000000000000000000 sub = c2 - (hi after rounding). see that the high bits are the same!
    // 00000000000000000000000000000000000000000000000000010111011010010101001000011111101011110011011101010101 lo = (a * b) + sub
    //
    // tt = 3 * 2^51 as a floating point.
    // to_u64(lo) - to_u64(tt)
    //
    //  ╭─ e=52───╮ ╭─ The rounding bit (51)
    // 0100001100110111011010010101001000011111101011110011011101010101 lo = fma(a, b, sub) = (0, 52, 0x769521FAF3755)
    // 0100001100111000000000000000000000000000000000000000000000000000 tt = (0, 52, 0x8000000000000)
    // 1111111111111111011010010101001000011111101011110011011101010101 (lo - tt) & mask = 769521faf3755
    // ╰─ underflow ╰─ lo bits               ─────────────────────────╯
    // Since an underflow occurs, we know that hi was rounded up. But why?
    //
    // ## Non-underflow case
    //
    // let a = 1574331031398118f64; let b = 135495742621820f64;
    //
    // hi = a * b + c1
    //                                                      ╭─ The rounding bit (51)
    //  ╭─ The top 52 bits ────────────────────────────────╮╭── Lower 51 bits that will be discarded ─────────╮
    // 00000010101100010100001000101000100001011011000101110001110010100101011011011001101101010000011101101000              a * b (integer)
    // 10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000 c1 = 2^103
    // 10000010101100010100001000101000100001011011000101110001110010100101011011011001101101010000011101101000 hi = 2^103 + a * b
    // 10000010101100010100001000101000100001011011000101110000000000000000000000000000000000000000000000000000 hi after FMA rounding
    //
    // sub = c2 - hi
    // 10000000000000000000000000000000000000000000000000011000000000000000000000000000000000000000000000000000 c2 = 2^103 + 3 * 2^51
    // 10000010101100010100001000101000100001011011000101110000000000000000000000000000000000000000000000000000 hi after FMA rounding
    //     -10101100010100001000101000100001011011000101011000000000000000000000000000000000000000000000000000  sub = c2 - (hi after rounding) = (1, 97, 0x58A11442D8AC0)
    //
    // lo = a * b + sub
    //                                                      ╭─ The rounding bit (51)
    //  ╭─ The top 52 bits ────────────────────────────────╮╭── Lower 51 bits will be kept           ─────────╮
    // 00000010101100010100001000101000100001011011000101110001110010100101011011011001101101010000011101101000              a * b (integer)
    //      -10101100010100001000101000100001011011000101011000000000000000000000000000000000000000000000000000 sub = c2 - (hi after rounding) = (1, 97, 0x58A11442D8AC0)
    //                                                    11001110010100101011011011001101101010000011101101000 lo = (a * b) + sub
    //
    // tt = 3 * 2^51 as a floating point.
    // to_u64(lo) - to_u64(tt)
    //
    //  ╭─ e=52───╮ ╭─ The rounding bit (51)
    // 0100001100111001110010100101011011011001101101010000011101101000 lo = fma(a, b, sub) = (0, 52, 0x9CA56D9B50768)
    // 0100001100111000000000000000000000000000000000000000000000000000 tt = (0, 52, 0x8000000000000)
    // 0000000000000001110010100101011011011001101101010000011101101000
    //              ╰─ lo bits               ─────────────────────────╯
    // Since an underflow does not occur, we know that hi was not rounded up. But why?
    //
    //
    // ## Explanation
    //
    // Take these inputs as an example:
    //
    // ```rust
    // let a = 1979581330507385f64; let b = 1237434108915901f64;
    // ```
    //
    //  The full binary representation of a * b + 2^103 is:
    //
    // ```
    //  ╭╴ The implicit bit (at position 104, since 2^103 has 104 bits)
    //  10011110111010110001011011010110000000101101100101010111011010010101001000011111101011110011011101010101
    //   ╰─ The top 52 bits ────────────────────────────────╯╰── Lower 51 bits that will be discarded ─────────╯
    //                                                       ╰─ The rounding bit (51)?
    // ```
    //
    // As such we expect our results to be:
    //
    // ```
    // high: 0x3DD62DAC05B2A
    // lo:   0x769521FAF3755
    // ```
    //
    // ### Obtaining high bits (which may be rounded up)
    //
    // Compare the above with the mantissa of `hi = a.mul_add(b, c1)`. The binary representation of `hi` is:
    //
    // ```
    //  0100011001100011110111010110001011011010110000000101101100101011
    //  │╰─ e=103──╯╰── mantissa (rounded up) ─────────────────────────╯
    //  ╰╴ Sign (positive)                                          52 ╯
    // ```
    //
    // The expected top 52 bits are `0x3DD62DAC05B2A`, but the mantissa of `hi` is greater by 1. This is because the CPU rounds this floating point value up. We will discuss why this rounding occurs in a later section.
    //
    // ### Obtaining low bits and conditionally subtracting the high term
    //
    // Next, we need to obtain the lower bits. We perform:
    //
    // ```rust!
    // let sub = c2 - hi;
    // let mut lo = a.mul_add(b, sub);
    // lo = lo.to_bits() - tt.to_bits()
    // if lo & 0x8000000000000000u64 > 0 {
    //     hi -= 1;
    // }
    // lo = lo & mask;
    // ```
    //
    // The conditional subtraction only occurs if bit 52 of `lo` is 0. This is because the binary
    // representation of `tt` as a floating point is:
    //
    //             ╭── 52
    // 0100001100111000000000000000000000000000000000000000000000000000
    // │╰─ e=52───╯╰── mantissa = 2^51 ───────────────────────────────╯
    // ╰╴ Sign (positive)
    // Implicit bit: 53
    //
    //
    // The question is, why does `a.mul_add(b, sub)` cause bit 52 to be 0 if `hi` was rounded up?
    //
    // Let's work backwards. Since `sub = c2 - hi`, where `hi` was previously computed as the
    // floating point `a * b + 2^103`, and `hi` does not have any information about the low bits,
    // the CPU has no way to tell whether rounding up occured from `hi` alone. Moreover, `c2` is
    // constant, so the only source of this information is from `a.mul_add(b, sub)`, where `a * b`
    // is internally computed with infinite precision before `sub` is added and the floating-point
    // is restricted to its 52-bit mantissa.
    //
    // TBD...
    //
    //
    //
    // The fused-multiply-add operation first computes a * b to full precision (102 bits). Next,
    // consider that the subtraction of hi from c2 effectively subtracts the high bits and also
    // reduces the exponent from 103 to that of the lower bits, which is 52.
    //
    // Let us consider what the fused-multiply-add operation does behind the scenes:
    //
    // (a * b) + sub                     =
    // (a * b) + (c2              ) - hi =
    // (a * b) + (2^103 + 3 * 2^51) - hi
    //    │             │           ╰─ Subtracts the high bits and 2^103 from c2
    //    │             ╰─ Sets the exponent of the result to 52, and sets bit 52 to 1 if
    //    ╰─ Computes a * b with full precision (102 bits)
    //
    // The result should be a positive floating-point with exponent 52:
    // 0100001100110111011010010101001000011111101011110011011101010101
    // │╰─ e=52───╯╰── lower 52 bits ─────────────────────────────────╯
    // ╰╴ Sign (positive)
    //
    // Recall the full binary representation of the integer product:
    //
    // ╭╴ The implicit bit (at position 104, since 2^103 has 104 bits)
    // 10011110111010110001011011010110000000101101100101010111011010010101001000011111101011110011011101010101
    //  ╰─ The top 52 bits ────────────────────────────────╯╰── Lower 51 bits that will be discarded ─────────╯
    //                                                      ╰─ 51
    //
    // Recall that when we subtract hi from c2, the 52nd bit of sub will be 0 if the 52nd bit of hi
    // is 1.
    //
    // The next step is to subtract the 64-bit representation of tt from that of lo:
    //
    // let lo = lo.to_bits() - tt.to_bits();
    //
    // The binary representation of tt as a floating point is:
    //             ╭── 52
    //             │╭── The lower 51 bits ────────────────────────────╮
    // 0100001100111000000000000000000000000000000000000000000000000000
    // │╰─ e=52───╯╰── mantissa = 2^51 ───────────────────────────────╯
    // ╰╴ Sign (positive)
    // Implicit bit is set at position 53
    //
    //
    // An underflow will occur here:
    //             ╭── 52
    //             │╭── The lower 51 bits ────────────────────────────╮
    // 0100001100110111011010010101001000011111101011110011011101010101 - (lo)
    // 0100001100111000000000000000000000000000000000000000000000000000   (tt)
    //  ╰─ e=52───╯
    //
    //
    // An underflow will not occur here:
    //
    //             ╭── 52
    //             │╭── The lower 51 bits ────────────────────────────╮
    // 0100001100111001110010100101011011011001101101010000011101101000 - (lo)
    // 0100001100111000000000000000000000000000000000000000000000000000   (tt)
    //
    // This implies that if bit 52 is set, hi was rounded up.
    //
    //
    // - If hi was not rounded up, then bit 52 is 1, so an underflow does not occurs.
    //
    //
    // 0000000000000001110010100101011011011001101101010000011101101000
    // Note that the binary representation of c2 is:
    // 0100011001100000000000000000000000000000000000000000000000000011
    // │╰─ e=103──╯╰── mantissa = 3 ─── ──────────────────────────────╯
    //             ╰ 103                                           52 ╯
    // ╰╴ Sign (positive)
    //
    // We expect the lower term to be 0x769521FAF3755, which corresponds exactly to the lower 51
    // bits. We need to do two things:
    //
    // 1. Extract the lower 51 bits
    // 2. Determine if the rounding bit is 1 or 0.
    //
    // Note that the rounding bit is the leftmost bit of the lower 51 bits.
    //
    // Niall's technique is to force the implicit bit of a * b to 2^51.
    //
    // What remains is to show how we determine if a conditional subtraction is necessary. We
    // combine this step with masking lo via subtraction (since, as EZW18 points out, masking can
    // be wasteful).
    //
    // The binary representation of lo as a floating point is:
    //              ╭─ The rounding bit from the computation of hi
    //              ╭── The lower 51 bits ────────────────────────────╮
    // 0100001100110111011010010101001000011111101011110011011101010101
    // │╰─ e=52───╯╰── mantissa: 52 bits ─────────────────────────────╯
    // ╰╴ Sign (positive)
    //
    // The 52nd bit is the rounding bit from the computation of hi, followed by the 51 lower bits
    // we need.
    //
    // The binary representation of tt as a floating point is:
    //             ╭── 52
    // 0100001100111000000000000000000000000000000000000000000000000000
    // │╰─ e=52───╯╰── mantissa = 2^51 ───────────────────────────────╯
    // ╰╴ Sign (positive)
    //
    // The subtraction, which is of the 64-bit binary representations of lo and tt, can be
    // visualised as such:
    //
    // OVERFLOW:
    //             ╭── 52
    //             │╭── The lower 51 bits ────────────────────────────╮
    // 0100001100110111011010010101001000011111101011110011011101010101 - (lo)
    // 0100001100111000000000000000000000000000000000000000000000000000   (tt)
    //  ╰─ e=52───╯
    // 1111111111111111011010010101001000011111101011110011011101010101
    //
    // If there is an overflow, we need to subtract 1 from the high term.
    // This indicates that the high term was rounded up.
    //
    //
    // NO OVERFLOW (different inputs):
    // let a = 1574331031398118f64; let b = 135495742621820f64;
    // The full binary representation of a * b + 2^103 is:
    //
    // ╭╴ The implicit bit (at position 104)               ╭╴ 52
    // 10000010101100010100001000101000100001011011000101110001110010100101011011011001101101010000011101101000
    //  ╰─ The top 52 bits ────────────────────────────────╯╰── Lower 51 bits that will be discarded ─────────╯
    //
    //             ╭── 52
    //             │╭─ The rounding bit from the computation of hi
    //             │╭── The lower 51 bits ────────────────────────────╮
    // 0100001100111001110010100101011011011001101101010000011101101000 - (lo)
    // 0100001100111000000000000000000000000000000000000000000000000000   (tt)
    //  ╰─ e=52───╯
    // 0000000000000001110010100101011011011001101101010000011101101000
    // Note that the binary representation of c2 is:
    // 0100011001100000000000000000000000000000000000000000000000000011
    // │╰─ e=103──╯╰── mantissa = 3 ─── ──────────────────────────────╯
    //             ╰ 103                                           52 ╯
    // ╰╴ Sign (positive)
    //
    // The implicit bit is 104, so the rightmost bits are at the 53rd and 52nd bits.
    //
    //
    // The implicit bit is 53, and the leftmost mantissa bit is 1, which means that the in the
    // binary representation of tt as an integer, the 53nd and 53rd bits are 1.
}
