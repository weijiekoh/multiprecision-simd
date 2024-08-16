use multiprecision_simd::bigint::BigInt;
use num_traits::identities::Zero;
use core::arch::wasm32::u64x2_extract_lane;
use num_bigint::BigUint;

use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::SeedableRng;

pub fn gen_seeded_rng(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

/// Converts a num_bigint::BigUint into a BigInt::<N, V, B>
pub fn biguint_to_bigint<const N: usize, const V: usize, const B: u32>(
    val: &BigUint,
) -> BigInt<N, V, B> {
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

    BigInt::<N, V, B>::new(res)
}

pub fn bigint_to_hex<const N: usize, const V: usize, const B: u32>(
    v: &BigInt<N, V, B>
) -> String {
    let mut res = String::new();
    for i in 0..V {
        let a_limb = u64x2_extract_lane::<0>(v.0[i]);
        let b_limb = u64x2_extract_lane::<1>(v.0[i]);

        let a_limb_bytes = Vec::<u8>::from(&a_limb.to_be_bytes());
        let b_limb_bytes = Vec::<u8>::from(&b_limb.to_be_bytes());

        let a_limb_hex = hex::encode(&a_limb_bytes);
        let b_limb_hex = hex::encode(&b_limb_bytes);

        let r = if i < V - 1 {
            format!("{}, {}, ", a_limb_hex, b_limb_hex)
        } else {
            format!("{}, {}", a_limb_hex, b_limb_hex)
        };

        res = format!("{}{}", res, r);
    }

    String::from(res)
}
