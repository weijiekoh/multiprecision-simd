#![feature(wasm_target_feature)]
#![cfg(target_arch = "wasm32")]
mod utils;

use wasm_bindgen_test::*;
use num_bigint::{BigInt as nbBigInt, BigUint, Sign, RandomBits};
use rand::Rng;
use multiprecision_simd::bigint::{BigInt64, BigInt256, BigIntF, BigIntF255, bigintf_sub};
use multiprecision_simd::mont::{
    bm17_simd_mont_mul,
    bm17_non_simd_mont_mul,
    mont_mul_cios,
    mont_mul_cios_f64_no_simd,
    resolve_bigintf,
    reduce_bigintf,
    msl_is_greater,
};

use crate::utils::{
    get_timestamp_now,
    gen_seeded_rng,
    bigint_to_biguint,
    biguint_to_bigintf,
    biguint_to_bigint64,
    biguint_to_bigint,
    bigint_to_hex,
    bigintf_to_biguint,
};
use ark_bls12_377::fr::Fr;
use ark_ff::{PrimeField, BigInteger};

use web_sys::console;

wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

const NUM_RUNS: u32 = 10000;

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

pub fn compute_bm17_mu(
    p: &BigUint,
    r: &BigUint,
    log_limb_size: u32,
) -> u32 {
    let pprime = calc_inv_and_pprime(p, r).1;
    let mu = &pprime % &BigUint::from(2u64.pow(log_limb_size));
    mu.to_u32_digits()[0]
}

/// Some large prime numbers and their corresponding n0 values assuming that we use 8x32-bit limbs.
pub fn get_ps_and_n0s_8x32() -> Vec::<(BigUint, u32)> {
    vec![
        (BigUint::parse_bytes(b"12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001", 16).unwrap(), 4294967295),
        (BigUint::parse_bytes(b"fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141", 16).unwrap(), 1435021631),
        (BigUint::parse_bytes(b"73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001", 16).unwrap(), 4294967295),
    ]
}

/*
use std::ops::Shl;

#[test]
#[wasm_bindgen_test]
fn test_make_initial() {
    // n specifies the total number of limbs in an instance
    // threads is the number of threads assigned to each instance
    // limbs is the number of 52-bit limbs in each thread

    /*
    let limbs = 10;

    for word in 0..limbs {
        let high_count = 2 * word;
        let low_count = 2 * word + 2;
        console::log_1(&format!("high_count: {}, low_count: {}", high_count, low_count).into());

        let value = 0x467u64 * high_count + 0x433u64 * low_count;
        let value = 0u64.wrapping_sub((value & 0xFFFu64).shl(52));
        console::log_1(&format!("{}: {:16x}", word, value).into());
    }
    */

    let high_count = 50;
    let low_count = 75;
    let value = 0x466u64 * high_count + 0x433u64 * low_count;
    let value = 0u64.wrapping_sub((value & 0xFFFu64)<<52);
    console::log_1(&format!("::::: {:16x}", value).into());

    let mut h = [0; 11];
    let mut l = [0; 11];

    for i in 0..5 {
        for j in 0..5 {
            h[j + 1] += 1;
        }
        for j in 0..5 {
            l[j] += 1;
        }
    }
    console::log_1(&format!("h: {:?}", h).into());
    console::log_1(&format!("l: {:?}", l).into());
}
*/

#[test]
#[wasm_bindgen_test]
fn test_simple_reduction() {
    //let p = BigUint::parse_bytes(b"12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001", 16).unwrap();
    //let r = BigUint::from(2u32).pow(256u32);
    //let res = calc_inv_and_pprime(&p, &r);
    //console::log_1(&format!("p:  {:064x}", p).into());
    //console::log_1(&format!("r:  {:064x}", r).into());
    //console::log_1(&format!("p': {:064x}", res.1).into());

    /*
    let p = BigUint::parse_bytes(b"328437590500042", 10).unwrap();
    let r = BigUint::from(2u32).pow(51);
    let res = calc_inv_and_pprime(&p, &r);
    console::log_1(&format!("p:  {:064x}", p).into());
    console::log_1(&format!("r:  {:064x}", r).into());
    console::log_1(&format!("p': {:064x}", res.1).into());
    */

    let num_limbs = 5;
    let log_limb_size = 51;
    let r = BigUint::from(2u32).pow(num_limbs * log_limb_size);
    let mut rng = gen_seeded_rng(2);
    let p = BigUint::parse_bytes(b"12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001", 16).unwrap();
    let p_bigintf: BigIntF255 = biguint_to_bigintf(&p);

    let a: BigUint = rng.sample(RandomBits::new(253));
    let ar = (&a * &r) % &p;
    let val = &ar + &p + &p;
    let val_bigintf: BigIntF255 = biguint_to_bigintf(&val);

    //console::log_1(&format!("p_bigintf:    {:?}", p_bigintf.0).into());
    //console::log_1(&format!("val_bigintf:  {:?}", val_bigintf.0).into());
}

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

        /*
        console::log_1(&format!("ar:  {:?}", ar.0).into());
        console::log_1(&format!("br:  {:?}", br.0).into());
        console::log_1(&format!("p:   {:064x}", p).into());
        console::log_1(&format!("abr: {:064x}", abr).into());
        console::log_1(&format!("res_biguint: {:064x}", res_biguint).into());

        for i in 0..num_limbs {
            console::log_1(&format!("abr[{}]: {:016x}", i, abr_bigint.0[i as usize]).into());
        }
        
        for i in 0..num_limbs {
            console::log_1(&format!("res[{}]: {:016x}", i, res.0[i as usize].to_bits()).into());
        }
        */

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
    let n0 = 422212465065983u64;
    let ar = (&a * &r) % &p_biguint;
    let br = (&b * &r) % &p_biguint;
    let ar: BigIntF255 = biguint_to_bigintf(&ar);
    let br: BigIntF255 = biguint_to_bigintf(&br);

    let abr = (&a * &b * &r) % &p_biguint;

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

    assert_eq!(abr, result);

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
        let z = unsafe { mont_mul_cios_f64_no_simd::<5, 6, 7, 11, 51>(ar, br, p, n0) };
        let z = unsafe { reduce_bigintf::<5, 51>(&z, &p_for_redc) };
        let z = unsafe { resolve_bigintf::<5, 3, 51>(&z) };
        x = y;
        y = z;
    }
    bigintf_to_biguint::<5, 51>(&y.clone())
}
