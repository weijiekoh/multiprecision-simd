#![cfg(target_arch = "wasm32")]
mod utils;

use wasm_bindgen_test::*;
use num_bigint::{BigInt as nbBigInt, BigUint, Sign, RandomBits};
use rand::Rng;
use multiprecision_simd::bigint::BigInt256;
use multiprecision_simd::mont::bm17_simd_mont_mul;
use crate::utils::{get_timestamp_now, gen_seeded_rng, bigint_to_biguint, biguint_to_bigint};
use web_sys::console;

const NUM_RUNS: u32 = 100;

pub fn compute_bm17_mu(
    p: &BigUint,
    r: &BigUint,
    log_limb_size: u32,
) -> u32 {
    let pprime = calc_inv_and_pprime(p, r).1;
    let mu = &pprime % &BigUint::from(2u64.pow(log_limb_size));
    mu.to_u32_digits()[0]
}

/// Some large prime numbers
pub fn get_ps() -> Vec::<BigUint> {
    vec![
        BigUint::parse_bytes(b"12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001", 16).unwrap(),
        BigUint::parse_bytes(b"fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141", 16).unwrap(),
        BigUint::parse_bytes(b"73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001", 16).unwrap(),
    ]
}

#[test]
#[wasm_bindgen_test]
fn test_bm17_simd_mont_mul() {
    let num_limbs = 8;
    let log_limb_size = 32;

    let mut rng = gen_seeded_rng(0);

    for p in get_ps() {
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

            let result = bm17_simd_mont_mul(&ar, &br, &p, mu);

            assert_eq!(bigint_to_biguint(&result), abr);
        }
    }
}

fn egcd(a: &nbBigInt, b: &nbBigInt) -> (nbBigInt, nbBigInt, nbBigInt) {
    if *a == nbBigInt::from(0u32) {
        return (b.clone(), nbBigInt::from(0u32), nbBigInt::from(1u32));
    }
    let (g, x, y) = egcd(&(b % a), a);

    (g, y - (b / a) * x.clone(), x.clone())
}

pub fn calc_inv_and_pprime(
    p: &BigUint,
    r: &BigUint,
) -> (BigUint, BigUint) {
    assert!(*r != BigUint::from(0u32));

    let p_bigint = nbBigInt::from_biguint(Sign::Plus, p.clone());
    let r_bigint = nbBigInt::from_biguint(Sign::Plus, r.clone());
    let one = nbBigInt::from(1u32);
    let (_, mut rinv, mut pprime) = egcd(
        &nbBigInt::from_biguint(Sign::Plus, r.clone()),
        &nbBigInt::from_biguint(Sign::Plus, p.clone())
    );

    if rinv.sign() == Sign::Minus {
        rinv = nbBigInt::from_biguint(Sign::Plus, p.clone()) + rinv;
    }

    if pprime.sign() == Sign::Minus {
        pprime = nbBigInt::from_biguint(Sign::Plus, r.clone()) + pprime;
    }

    // r * rinv - p * pprime == 1
    assert!(
        (nbBigInt::from_biguint(Sign::Plus, r.clone()) * &rinv % &p_bigint) -
            (&p_bigint * &pprime % &p_bigint)
        == one
    );

    // r * rinv % p == 1
    assert!((nbBigInt::from_biguint(Sign::Plus, r.clone()) * &rinv % &p_bigint) == one);

    // p * pprime % r == 1
    assert!(
        (&p_bigint * &pprime % &r_bigint) == one
    );

    (
        rinv.to_biguint().unwrap(),
        pprime.to_biguint().unwrap(),
    )
}

const COST: u32 = 10000;

pub fn reference_function(
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

wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[test]
#[wasm_bindgen_test]
pub fn benchmark_bm17_simd_mont_mul() {
    console::log_1(&"Benchmarks for BLS12-377 scalar field (253 bits) multiplications".into());
    let num_limbs = 8; 
    let log_limb_size = 32; 

    let mut rng = gen_seeded_rng(0);
    let p = BigUint::parse_bytes(b"12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001", 16).unwrap();
    let r = BigUint::from(2u32).pow(num_limbs * log_limb_size);
    let mu = compute_bm17_mu(&p, &r, log_limb_size);

    let a: BigUint = rng.sample(RandomBits::new(256));
    let b: BigUint = rng.sample(RandomBits::new(256));

    let start = get_timestamp_now();
    let expected = reference_function(&a, &b, &p, COST);
    let end = get_timestamp_now();
    console::log_1(
        &format!(
            "{} full (non-Montgomery) multiplications with naive num_bigint took {} ms",
            COST,
            end - start,
        ).into()
    );

    let expected = &expected * &r % &p;

    // Convert the inputs into Montgomery form
    let ar = (&a * &r) % &p;
    let br = (&b * &r) % &p;

    let ar: BigInt256 = biguint_to_bigint(&ar);
    let br: BigInt256 = biguint_to_bigint(&br);
    let p: BigInt256 = biguint_to_bigint(&p);

    let mut x = ar.clone();
    let mut y = br.clone();

    let start = get_timestamp_now();

    for _ in 0..COST {
        let z = bm17_simd_mont_mul(&x, &y, &p, mu);
        x = y;
        y = z;
    }

    let end = get_timestamp_now();

    let result = y.clone();

    assert_eq!(bigint_to_biguint(&result), expected);

    console::log_1(
        &format!(
            "{} Montgomery multiplications with BM17 SIMD took {} ms",
            COST,
            end - start,
        ).into()
    );
}
