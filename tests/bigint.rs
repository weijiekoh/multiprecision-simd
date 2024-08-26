#![cfg(target_arch = "wasm32")]
mod utils;

use rand::Rng;
use wasm_bindgen_test::*;
use web_sys::console;
use num_bigint::{BigUint, RandomBits};
use multiprecision_simd::bigint::BigInt256;
use crate::utils::{gen_seeded_rng, biguint_to_bigint, bigint_to_hex};

/*
#![cfg(target_arch = "wasm32")]
mod utils;

extern crate wasm_bindgen_test;
//wasm_bindgen_test_configure!(run_in_browser);

use core::arch::wasm32::u64x2_extract_lane;
use wasm_bindgen_test::*;
use multiprecision_simd::bigint::{BigInt256, BigInt300};
use crate::utils::{gen_seeded_rng, biguint_to_bigint};
use num_bigint::{BigUint, RandomBits};
use rand::Rng;
//use web_sys::console;

#[test]
#[wasm_bindgen_test]
fn test_add_without_carry() {
    let a_data: [u32; 8] = [0, 1, 2, 3, 4, 5, 6, 0xffffffff];
    let b_data: [u32; 8] = [0, 1, 2, 3, 4, 5, 6, 1];

    let a: BigInt256 = BigInt256::new(a_data.clone());
    let b: BigInt256 = BigInt256::new(b_data.clone());

    let c = a.add_without_carry(&b);

    for i in 0..2 {
        let ab = u64x2_extract_lane::<0>(c.0[i * 2]);
        let cd = u64x2_extract_lane::<1>(c.0[i * 2]);
        let ef = u64x2_extract_lane::<0>(c.0[i * 2 + 1]);
        let gh = u64x2_extract_lane::<1>(c.0[i * 2 + 1]);

        let expected_ab = (a_data[i * 4    ] as u64) + (b_data[i * 4    ] as u64);
        let expected_cd = (a_data[i * 4 + 1] as u64) + (b_data[i * 4 + 1] as u64);
        let expected_ef = (a_data[i * 4 + 2] as u64) + (b_data[i * 4 + 2] as u64);
        let expected_gh = (a_data[i * 4 + 3] as u64) + (b_data[i * 4 + 3] as u64);

        assert_eq!(ab, expected_ab);
        assert_eq!(cd, expected_cd);
        assert_eq!(ef, expected_ef);
        assert_eq!(gh, expected_gh);
    }
}

#[test]
#[wasm_bindgen_test]
fn test_add_assign_without_carry() {
    let a_data: [u32; 8] = [0, 1, 2, 3, 4, 5, 6, 0xffffffff];
    let b_data: [u32; 8] = [0, 1, 2, 3, 4, 5, 6, 1];

    let mut a: BigInt256 = BigInt256::new(a_data.clone());
    let b: BigInt256 = BigInt256::new(b_data.clone());

    let c = a.add_without_carry(&b);
    a.add_assign_without_carry(&b);
    assert!(a == c);
}

#[test]
#[wasm_bindgen_test]
fn test_add_unsafe() {
    let a_data: [u32; 8] = [0xffffffff, 0xffffffff, 0, 0, 0, 0, 0, 0];
    let b_data: [u32; 8] = [1, 0, 0, 0, 0, 0, 0, 0];
    let c_data: [u32; 8] = [0, 0, 1, 0, 0, 0, 0, 0];

    let a: BigInt256 = BigInt256::new(a_data.clone());
    let b: BigInt256 = BigInt256::new(b_data.clone());
    let c: BigInt256 = BigInt256::new(c_data.clone());
    let result = a.add_unsafe(&b);

    assert!(result == c);
}

// TODO: write fuzz tests!
#[test]
#[wasm_bindgen_test]
fn test_add_unsafe_2() {
    let mut rng = gen_seeded_rng(1);
    let val_0_biguint: BigUint = rng.sample(RandomBits::new(256));
    let val_0_bigint: BigInt256 = biguint_to_bigint(&val_0_biguint);

    let val_1_biguint: BigUint = rng.sample(RandomBits::new(256));
    let val_1_bigint: BigInt256 = biguint_to_bigint(&val_1_biguint);

    let sum_biguint = &val_0_biguint + &val_1_biguint;
    let sum_bigint: BigInt256 = val_0_bigint.add_unsafe(&val_1_bigint);

    assert!(biguint_to_bigint(&sum_biguint) == sum_bigint);

    // SHA256(0) + SHA256(1)
    let expected_sum = "5f39e71f18cd55d10ec20ae3b506907cca7e940fe5f0aec3afa93f3e61bfd996";
    assert_eq!(hex::encode(&sum_biguint.to_bytes_be()), expected_sum);

    //console::log_1(&hex::encode(&val_0_biguint.to_bytes_be()).to_string().into());
    //console::log_1(&bigint_to_hex(&val_0_bigint).to_string().into());
    //console::log_1(&hex::encode(&val_1_biguint.to_bytes_be()).to_string().into());
    //console::log_1(&bigint_to_hex(&val_1_bigint).to_string().into());
    //console::log_1(&hex::encode(&sum_biguint.to_bytes_be()).to_string().into());
    //console::log_1(&bigint_to_hex(&sum_bigint).to_string().into());
}

#[test]
#[wasm_bindgen_test]
fn test_add_unsafe_bigint300() {
    let mut rng = gen_seeded_rng(1);
    let val_0_biguint: BigUint = rng.sample(RandomBits::new(256));
    let val_0_bigint: BigInt300 = biguint_to_bigint(&val_0_biguint);

    let val_1_biguint: BigUint = rng.sample(RandomBits::new(256));
    let val_1_bigint: BigInt300 = biguint_to_bigint(&val_1_biguint);

    let sum_biguint = val_0_biguint + val_1_biguint;
    let sum_bigint: BigInt300 = val_0_bigint.add_unsafe(&val_1_bigint);

    assert!(biguint_to_bigint(&sum_biguint) == sum_bigint);

    // SHA256(0) + SHA256(1)
    let expected_sum = "5f39e71f18cd55d10ec20ae3b506907cca7e940fe5f0aec3afa93f3e61bfd996";
    assert_eq!(hex::encode(&sum_biguint.to_bytes_be()), expected_sum);
}

#[test]
#[wasm_bindgen_test]
fn test_sub() {
    let a_data: [u32; 8] = [1, 2, 0, 0, 0, 0, 0, 0];
    let b_data: [u32; 8] = [0, 3, 0, 0, 0, 0, 0, 0];
    let c_data: [u32; 8] = [1, 0xffffffff, 0, 0, 0, 0, 0, 0];

    let a: BigInt256 = BigInt256::new(a_data.clone());
    let b: BigInt256 = BigInt256::new(b_data.clone());
    let c: BigInt256 = BigInt256::new(c_data.clone());
    let result = a.sub(&b);

    assert!(result == c);
}

#[test]
#[wasm_bindgen_test]
fn test_sub_2() {
    let a_data: [u32; 8] = [1, 0, 0, 0, 0, 0, 0, 1];
    let b_data: [u32; 8] = [1, 0, 0, 0, 0, 0, 0, 0];
    let c_data: [u32; 8] = [0, 0, 0, 0, 0, 0, 0, 1];

    let a: BigInt256 = BigInt256::new(a_data.clone());
    let b: BigInt256 = BigInt256::new(b_data.clone());
    let c: BigInt256 = BigInt256::new(c_data.clone());
    let result = a.sub(&b);

    assert!(result == c);
}
*/
