use num_bigint::{BigInt as nbBigInt, BigUint, Sign};
use multiprecision_simd::bigint::{BigInt, BigIntF, BigInt64};
use num_traits::identities::Zero;
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::SeedableRng;

//wasm_bindgen_test_configure!(run_in_browser);

pub fn egcd(a: &nbBigInt, b: &nbBigInt) -> (nbBigInt, nbBigInt, nbBigInt) {
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

pub fn get_timestamp_now() -> f64 {
    web_sys::window().expect("should have a Window")
        .performance()
        .expect("should have a Performance")
        .now()
}

pub fn gen_seeded_rng(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

pub fn bigint_to_biguint<const N: usize, const B: u32>(
    val: &BigInt<N, B>,
) -> BigUint {
    let mut res = BigUint::from(0u32);
    let max = 2u64.pow(B);

    for i in 0..N {
        assert!((val.0[i] as u64) < max);
        let idx = (N - 1 - i) as u32;
        let a = idx * B;
        let b = BigUint::from(2u32).pow(a) * BigUint::from(val.0[idx as usize]);

        res += BigUint::from(b);
    }

    res
}

/// Converts a num_bigint::BigUint into a BigIntF::<N, B>
pub fn biguint_to_bigintf<const N: usize, const B: u32>(
    val: &BigUint,
) -> BigIntF<N, B> {
    let mut res = [0f64; N];
    let mask: u64 = 2u64.pow(B as u32) - 1;
    let mask = BigUint::from(mask);

    for i in 0..N {
        let idx = N - 1 - i;
        let shift = (idx as u32) * B;
        let w = (val.clone() >> shift) & mask.clone();

        if !w.is_zero() {
            res[idx] = w.to_u64_digits()[0] as f64;
        }
    }

    BigIntF::<N, B>(res)
}

//use web_sys::console;
/// Converts a BigIntF::<N, B>into a num_bigint::BigUint 
pub fn bigintf_to_biguint<const N: usize, const B: u32>(
    val: &BigIntF<N, B>
) -> BigUint {
    let mut res = BigUint::from(0u32);
    let max = 2u64.pow(B);
    //let mask = (1u64 << B) - 1u64;

    for i in 0..N {
        let idx = (N - 1 - i) as u32;

        let masked = val.0[idx as usize] as u64;
        assert!(masked < max);

        let a = idx * B;
        let b = BigUint::from(2u64).pow(a) * BigUint::from(masked);

        res += BigUint::from(b);
    }

    res
}

/// Converts a num_bigint::BigUint into a BigInt::<N, B>
pub fn biguint_to_bigint<const N: usize, const B: u32>(
    val: &BigUint,
) -> BigInt<N, B> {
    let mut res = [0u32; N];
    let mask: u64 = 2u64.pow(B as u32) - 1;
    let mask = BigUint::from(mask);

    for i in 0..N {
        let idx = N - 1 - i;
        let shift = (idx as u32) * B;
        let w = (val.clone() >> shift) & mask.clone();

        if !w.is_zero() {
            res[idx] = w.to_u32_digits()[0];
        }
    }

    BigInt::<N, B>(res)
}

/// Converts a num_bigint::BigUint into a BigInt64::<N, B>
pub fn biguint_to_bigint64<const N: usize, const B: u32>(
    val: &BigUint,
) -> BigInt64<N, B> {
    let mut res = [0u64; N];
    let mask: u64 = 2u64.pow(B as u32) - 1;
    let mask = BigUint::from(mask);

    for i in 0..N {
        let idx = N - 1 - i;
        let shift = (idx as u32) * B;
        let w = (val.clone() >> shift) & mask.clone();

        if !w.is_zero() {
            res[idx] = w.to_u64_digits()[0];
        }
    }

    BigInt64::<N, B>(res)
}

pub fn bigint_to_hex<const N: usize, const B: u32>(
    v: &BigInt<N, B>
) -> String {
    let mut res = String::new();
    for i in 0..N {
        let limb = v.0[i];
        let limb_bytes = Vec::<u8>::from(&limb.to_be_bytes());
        let h = hex::encode(&limb_bytes);

        res = format!("{:0>8}{}", h, res);
    }
    String::from(res)
}
