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

fn int_full_product(a: f64, b: f64) -> (u64, u64) {
    // This code uses slightly different values: https://github.com/z-prize/2023-entries/blob/a568b9eb1e73615f2cee6d0c641bb6a7bc5a428f/prize-2-msm-wasm/prize-2b-twisted-edwards/yrrid-snarkify/yrrid/SupportingFiles/FP51.java#L47
    //let c1 = 2f64.pow(103);
    //let c2 = 2f64.pow(103) + 3f64 * 2f64.pow(51);

    // The paper uses c1 = 2^104 and c2 = 2^104 + 2^52
    let c1 = 2f64.pow(104);
    let c2 = 2f64.pow(104) + 2f64.pow(52);

    let mask = 2u64.pow(52) - 1;

    let p_hi = fma_rz(a, b, c1);
    let sub = c2 - p_hi;
    let p_lo = fma_rz(a, b, sub);

    (p_hi.to_bits() & mask, p_lo.to_bits() & mask)
}

#[test]
#[wasm_bindgen_test]
fn test_fig_4() {
    let a = 211106232532993f64;
    let b = 211106232532993f64;

    let p_hi = int_full_product(a, b);

    assert_eq!(p_hi.0, 9895604649984u64);
    assert_eq!(p_hi.1, 422212465065985u64);

    let a = num_bigint::BigUint::from(a as u64);
    let b = num_bigint::BigUint::from(b as u64);
    let m = &a * &b;

    let x = num_bigint::BigUint::from(2u32).pow(52u32);
    let ph = num_bigint::BigUint::from(p_hi.0);
    let pl = num_bigint::BigUint::from(p_hi.1);

    assert_eq!(x * ph + pl, m)

    // It appears that f64::mul_add() does not have the same round-to-zero behaviour as CUDA's
    // __fma_rz. This causes an incorrect result for 52-bit limbs. Apparently, the relaxed SIMD
    // function f64x2_relaxed_madd() will work, but I'm having trouble getting it running in Rust.
    // The next step is to find out if the rounding mode really matters for slightly smaller limbs
    // (e.g. 48 bits).
}

