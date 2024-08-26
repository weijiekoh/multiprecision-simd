use multiprecision_simd::bigint::BigInt;
use num_traits::identities::Zero;
use num_bigint::BigUint;
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::SeedableRng;

//wasm_bindgen_test_configure!(run_in_browser);

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

/// Converts a num_bigint::BigUint into a BigInt::<N, V, B>
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
