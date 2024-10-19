#![feature(wasm_target_feature)]
#![cfg(target_arch = "wasm32")]
mod utils;

use wasm_bindgen_test::*;
use num_bigint::{BigUint, RandomBits};
use rand::Rng;
use multiprecision_simd::bigint::{BigInt256, BigIntF, BigIntF255};
use multiprecision_simd::mont::{
    bm17_simd_mont_mul,
    bm17_non_simd_mont_mul,
    mont_mul_cios,
    mont_mul_cios_f64_no_simd,
    resolve_bigintf,
    reduce_bigintf,
};

use crate::utils::{
    compute_bm17_mu,
    get_timestamp_now,
    gen_seeded_rng,
    bigint_to_biguint,
    biguint_to_bigintf,
    biguint_to_bigint,
    bigintf_to_biguint,
};
use ark_bls12_377::fr::Fr;
use ark_ff::{PrimeField, BigInteger};

use web_sys::console;

wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

pub fn reference_function_num_bigint(
    a: &BigUint,
    b: &BigUint,
    p: &BigUint,
    cost: u32,
) -> BigUint {
    let mut x = a.clone();
    let mut y = b.clone();
    for _ in 0..cost {
        let z = &x * &y % p;
        x = y;
        y = z;
    }
    y.clone()
}

pub fn reference_function_bm17_simd(
    ar: &BigInt256,
    br: &BigInt256,
    p: &BigInt256,
    mu: u32,
    cost: u32,
) -> BigInt256 {
    let mut x = ar.clone();
    let mut y = br.clone();
    for _ in 0..cost {
        let z = unsafe {bm17_simd_mont_mul(&x, &y, &p, mu)};
        x = y;
        y = z;
    }
    y.clone()
}

pub fn reference_function_bm17_non_simd(
    ar: &BigInt256,
    br: &BigInt256,
    p: &BigInt256,
    mu: u32,
    cost: u32,
) -> BigInt256 {
    let mut x = ar.clone();
    let mut y = br.clone();
    for _ in 0..cost {
        let z = bm17_non_simd_mont_mul(&x, &y, &p, mu);
        x = y;
        y = z;
    }
    y.clone()
}

pub fn reference_function_cios(
    ar: &BigInt256,
    br: &BigInt256,
    p: &BigInt256,
    n0: u32,
    cost: u32,
) -> BigInt256 {
    let mut x = ar.clone();
    let mut y = br.clone();
    for _ in 0..cost {
        let z = unsafe {mont_mul_cios::<8, 9, 10, 32>(&x, &y, &p, n0)};
        x = y;
        y = z;
    }
    y.clone()
}

pub fn reference_function_ark_ff(
    a: &Fr,
    b: &Fr,
    cost: u32,
) -> Fr {
    let mut x = a.clone();
    let mut y = b.clone();
    for _ in 0..cost {
        let z = x * y;
        x = y;
        y = z;
    }
    y.clone()
}

const COST: u32 = 2u32.pow(8);

#[test]
#[wasm_bindgen_test]
pub fn benchmark_mont_mul() {
    console::log_1(&"Benchmarks for BLS12-377 scalar field (253 bits) multiplications".into());
    let num_limbs = 8; 
    let log_limb_size = 32; 

    let mut rng = gen_seeded_rng(0);
    let p_biguint = BigUint::parse_bytes(b"12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001", 16).unwrap();
    let r = BigUint::from(2u32).pow(num_limbs * log_limb_size);
    let mu = compute_bm17_mu(&p_biguint, &r, log_limb_size);

    let a: BigUint = rng.sample(RandomBits::new(256));
    let b: BigUint = rng.sample(RandomBits::new(256));

    let a = &a % &p_biguint;
    let b = &b % &p_biguint;

    // Naive num_bigint multiplication and modulo
    let start = get_timestamp_now();
    let expected_non_mont = reference_function_num_bigint(&a, &b, &p_biguint, COST);
    let end = get_timestamp_now();
    let expected_mont = (&expected_non_mont * &r) % &p_biguint;

    console::log_1(
        &format!(
            "{} full (non-Montgomery) multiplications and modulo with naive num_bigint took {} ms",
            COST,
            end - start,
        ).into()
    );

    let a_a = Fr::from_be_bytes_mod_order(&a.to_bytes_be());
    let a_b = Fr::from_be_bytes_mod_order(&b.to_bytes_be());
    let start = get_timestamp_now();
    let expected_ark_non_mont = reference_function_ark_ff(&a_a, &a_b, COST);
    let end = get_timestamp_now();


    console::log_1(
        &format!(
            "{} multiplications with ark-bls12-377 (using MontBackend) took {} ms",
            COST,
            end - start,
        ).into()
    );

    assert_eq!(&expected_ark_non_mont.into_bigint().to_bytes_be(), &expected_non_mont.to_bytes_be());

    // Convert the inputs into Montgomery form
    let ar = (&a * &r) % &p_biguint;
    let br = (&b * &r) % &p_biguint;

    let ar: BigInt256 = biguint_to_bigint(&ar);
    let br: BigInt256 = biguint_to_bigint(&br);
    let p: BigInt256 = biguint_to_bigint(&p_biguint);

    let start = get_timestamp_now();
    let result = reference_function_bm17_simd(&ar, &br, &p, mu, COST);
    let end = get_timestamp_now();

    assert_eq!(bigint_to_biguint(&result), expected_mont);

    console::log_1(
        &format!(
            "{} Montgomery multiplications with BM17 SIMD took {} ms",
            COST,
            end - start,
        ).into()
    );

    let start = get_timestamp_now();
    let result = reference_function_bm17_non_simd(&ar, &br, &p, mu, COST);
    let end = get_timestamp_now();

    assert_eq!(bigint_to_biguint(&result), expected_mont);

    console::log_1(
        &format!(
            "{} Montgomery multiplications with BM17 (non-SIMD) took {} ms",
            COST,
            end - start,
        ).into()
    );

    let n0 = 4294967295;
    let start = get_timestamp_now();
    let result = reference_function_cios(&ar, &br, &p, n0, COST);
    let end = get_timestamp_now();

    assert_eq!(bigint_to_biguint(&result), expected_mont);

    console::log_1(
        &format!(
            "{} Montgomery multiplications with CIOS (non-SIMD, without gnark optimisation) took {} ms",
            COST,
            end - start,
        ).into()
    );

    let r = BigUint::from(2u32).pow(5 * 51);
    let rinv = BigUint::parse_bytes(b"6964932758513947866663202575519623529371944708760353099416816606024780601348", 10).unwrap();
    let n0 = 422212465065983u64;
    let ar = (&a * &r) % &p_biguint;
    let br = (&b * &r) % &p_biguint;

    let ar: BigIntF255 = biguint_to_bigintf(&ar);
    let br: BigIntF255 = biguint_to_bigintf(&br);

    let p_bigintf: BigIntF255 = biguint_to_bigintf(&p_biguint);
    let p_for_redc = [
        f64::from_bits(0x1800000000001u64),
        f64::from_bits(0x7DA0000002142u64),
        f64::from_bits(0x0DEC00566A9DBu64),
        f64::from_bits(0x2AB305A268F2Eu64),
        f64::from_bits(0x12AB655E9A2CAu64),
    ];
    let p_for_redc = BigIntF::<5, 51>(p_for_redc);

    let start = get_timestamp_now();
    let result = reference_function_cios_f64_no_simd(&ar, &br, &p_bigintf, &p_for_redc, n0 as u64, COST);
    let end = get_timestamp_now();

    assert_eq!(&result * &rinv % &p_biguint, expected_non_mont);

    console::log_1(
        &format!(
            "{} Montgomery multiplications with f64s and CIOS (non-SIMD) took {} ms",
            COST,
            end - start,
        ).into()
    );

}

pub fn reference_function_cios_f64_no_simd(
    ar: &BigIntF<5, 51>,
    br: &BigIntF<5, 51>,
    p: &BigIntF<5, 51>,
    p_for_redc: &BigIntF<5, 51>,
    n0: u64,
    cost: u32,
) -> BigUint {
    let mut x = ar.clone();
    let mut y = br.clone();
    for _ in 0..cost {
        let z = unsafe { mont_mul_cios_f64_no_simd::<5, 6, 7, 11, 51>(&x, &y, p, n0) };
        let z = unsafe { reduce_bigintf::<5, 51>(&z, &p_for_redc) };
        let z = unsafe { resolve_bigintf::<5, 3, 51>(&z) };
        x = y;
        y = z;
    }
    bigintf_to_biguint::<5, 51>(&y.clone())
}

