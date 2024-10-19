#![feature(wasm_target_feature)]
#![cfg(target_arch = "wasm32")]
mod utils;

use wasm_bindgen_test::*;
use num_bigint::{BigUint, Sign, RandomBits};
use rand::Rng;
use multiprecision_simd::bigint::{BigInt64, BigInt256, BigIntF, BigIntF255};
use multiprecision_simd::mont::{
    bm17_simd_mont_mul,
    bm17_non_simd_mont_mul,
    mont_mul_cios,
    mont_mul_cios_f64_no_simd,
    resolve_bigintf,
    reduce_bigintf,
};

use crate::utils::{
    get_ps_and_n0s_8x32,
    compute_bm17_mu,
    get_timestamp_now,
    gen_seeded_rng,
    bigint_to_biguint,
    biguint_to_bigintf,
    biguint_to_bigint64,
    biguint_to_bigint,
    bigintf_to_biguint,
};
use ark_bls12_377::fr::Fr;
use ark_ff::{PrimeField, BigInteger};

use web_sys::console;

wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

const NUM_RUNS: u32 = 10000;

#[test]
#[wasm_bindgen_test]
fn test_mont_mul_cios_f64_no_simd() {
    let num_limbs = 5;
    let log_limb_size = 51;
    let r = BigUint::from(2u32).pow(num_limbs * log_limb_size);

    let p = BigUint::parse_bytes(b"12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001", 16).unwrap();
    let rinv = BigUint::parse_bytes(b"6964932758513947866663202575519623529371944708760353099416816606024780601348", 10);
    let n0 = 422212465065983u64;

    let mut rng = gen_seeded_rng(0);

    for _ in 0..NUM_RUNS {
        let a: BigUint = rng.sample(RandomBits::new(253));
        let b: BigUint = rng.sample(RandomBits::new(253));
        //console::log_1(&format!("a:  {:?}", a).into());
        //console::log_1(&format!("b:  {:?}", b).into());

        // Convert the inputs into Montgomery form
        let ar = (&a * &r) % &p;
        let br = (&b * &r) % &p;

        // The expected result
        let abr = (&ar * &b) % &p;

        let ar: BigIntF255 = biguint_to_bigintf(&ar);
        let br: BigIntF255 = biguint_to_bigintf(&br);
        let abr_bigint: BigInt64::<5, 51> = biguint_to_bigint64(&abr);
        let p_bigintf: BigIntF255 = biguint_to_bigintf(&p);

        let p_for_redc = [
            f64::from_bits(0x1800000000001u64),
            f64::from_bits(0x7DA0000002142u64),
            f64::from_bits(0x0DEC00566A9DBu64),
            f64::from_bits(0x2AB305A268F2Eu64),
            f64::from_bits(0x12AB655E9A2CAu64),
        ];
        let p_for_redc = BigIntF::<5, 51>(p_for_redc);

        let res = unsafe { mont_mul_cios_f64_no_simd::<5, 6, 7, 11, 51>(&ar, &br, &p_bigintf, n0) };
        let res = unsafe { reduce_bigintf::<5, 51>(&res, &p_for_redc) };
        let res = unsafe { resolve_bigintf::<5, 3, 51>(&res) };

        let res_biguint = bigintf_to_biguint::<5, 51>(&res);

        assert_eq!(res_biguint, abr);
    }
}

#[test]
#[wasm_bindgen_test]
fn test_mont_mul_cios() {
    let num_limbs = 8;
    let log_limb_size = 32;

    let r = BigUint::from(2u32).pow(num_limbs * log_limb_size);

    let mut rng = gen_seeded_rng(0);

    for (p, n0) in get_ps_and_n0s_8x32() {

        for _ in 0..NUM_RUNS {
            let a: BigUint = rng.sample(RandomBits::new(256));
            let b: BigUint = rng.sample(RandomBits::new(256));

            // Convert the inputs into Montgomery form
            let ar = (&a * &r) % &p;
            let br = (&b * &r) % &p;

            // The expected result
            let abr = (&a * &b * &r) % &p;

            let ar: BigInt256 = biguint_to_bigint(&ar);
            let br: BigInt256 = biguint_to_bigint(&br);
            let p: BigInt256 = biguint_to_bigint(&p);

            let result = unsafe {mont_mul_cios::<8, 9, 10, 32>(&ar, &br, &p, n0)};

            assert_eq!(bigint_to_biguint(&result), abr);
        }
    }
}

#[test]
#[wasm_bindgen_test]
fn test_bm17_simd_mont_mul() {
    let num_limbs = 8;
    let log_limb_size = 32;

    let mut rng = gen_seeded_rng(0);

    for (p, _) in get_ps_and_n0s_8x32() {
        let r = BigUint::from(2u32).pow(num_limbs * log_limb_size);
        let mu = compute_bm17_mu(&p, &r, log_limb_size);
        for _ in 0..NUM_RUNS {
            let a: BigUint = rng.sample(RandomBits::new(256));
            let b: BigUint = rng.sample(RandomBits::new(256));

            // Convert the inputs into Montgomery form
            let ar = (&a * &r) % &p;
            let br = (&b * &r) % &p;

            // The expected result
            let abr = (&a * &b * &r) % &p;

            let ar: BigInt256 = biguint_to_bigint(&ar);
            let br: BigInt256 = biguint_to_bigint(&br);
            let p: BigInt256 = biguint_to_bigint(&p);

            let result = unsafe {bm17_simd_mont_mul(&ar, &br, &p, mu)};

            assert_eq!(bigint_to_biguint(&result), abr);
        }
    }
}

#[test]
#[wasm_bindgen_test]
fn test_bm17_non_simd_mont_mul() {
    let num_limbs = 8;
    let log_limb_size = 32;

    let mut rng = gen_seeded_rng(0);

    for (p, _) in get_ps_and_n0s_8x32() {
        let r = BigUint::from(2u32).pow(num_limbs * log_limb_size);
        let mu = compute_bm17_mu(&p, &r, log_limb_size);
        for _ in 0..NUM_RUNS {
            let a: BigUint = rng.sample(RandomBits::new(256));
            let b: BigUint = rng.sample(RandomBits::new(256));

            // Convert the inputs into Montgomery form
            let ar = (&a * &r) % &p;
            let br = (&b * &r) % &p;

            // The expected result
            let abr = (&a * &b * &r) % &p;

            let ar: BigInt256 = biguint_to_bigint(&ar);
            let br: BigInt256 = biguint_to_bigint(&br);
            let p: BigInt256 = biguint_to_bigint(&p);

            let result = bm17_non_simd_mont_mul(&ar, &br, &p, mu);

            assert_eq!(bigint_to_biguint(&result), abr);
        }
    }
}
