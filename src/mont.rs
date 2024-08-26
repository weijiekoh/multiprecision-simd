use core::arch::wasm32;
use std::num::Wrapping;
use crate::bigint::{BigInt, gt, sub};
//use web_sys::console;

pub fn to_u64<const N: usize>(v: [u32; N]) -> [u64; N] {
    let mut result = [0u64; N];
    for i in 0..N {
        result[i] = v[i] as u64;
    }
    result
}

/// Algorithm 4 of "Montgomery Arithmetic from a Software Perspective" by Bos and Montgomery
/// Uses WASM SIMD opcodes.
/// Also see:
/// https://github.com/coreboot/vboot/blob/060efa0cf64d4b7ccbe3e88140c9da5f747355ee/firmware/2lib/2modpow_sse2.c#L113
pub fn bm17_simd_mont_mul<const N: usize, const B: u32>(
    ar_32: &BigInt<N, B>,
    br_32: &BigInt<N, B>,
    p_32: &BigInt<N, B>,
    mu: u32,
) -> BigInt<N, B> {
    let ar = to_u64(ar_32.0);
    let br = to_u64(br_32.0);
    let p = to_u64(p_32.0);

    let w_mu = Wrapping(mu as u64);
    let mask_64 = 2u64.pow(B) - 1;
    let mask = wasm32::u64x2(mask_64, mask_64);

    // The zero wasm32::v128
    let z = wasm32::u64x2(0, 0);

    // Initialise d and e
    let mut de = [z; N];

    // mu * br[0]
    let mu_b0 = w_mu * Wrapping(br[0] as u64);

    // Assign br and p into v128s
    let mut bp = [z; N];
    for i in 0..N {
        bp[i] = wasm32::u64x2(br[i], p[i]);
    }

    for j in 0..N {
        // Compute q
        let d0 = wasm32::u64x2_extract_lane::<0>(de[0]);
        let e0 = wasm32::u64x2_extract_lane::<1>(de[0]);
        let d0_minus_e0 = Wrapping(d0) - Wrapping(e0);

        // q = (mub0)aj + mu(d0 - e0) mod 2^32
        let q = (mu_b0 * Wrapping(ar[j] as u64) + w_mu * d0_minus_e0).0 & mask_64;

        // t0 = ajb0 + d0
        // t1 = qp0 + e0

        // aq = ar[j], q
        let aq = wasm32::u64x2(ar[j], q);

        let mut t01 = wasm32::u64x2_add(
            // ajb0, qp0
            wasm32::u64x2_mul(aq, bp[0]),
            de[0],
        );

        t01 = wasm32::u64x2_shr(t01, 32);

        for i in 1..N {
            // p0 = ajbi + t0 + di
            // p1 = qpi + t1 + ei
            let p01 = wasm32::u64x2_add(
                wasm32::u64x2_add(t01, de[i]),
                wasm32::u64x2_mul(aq, bp[i]),
            );

            // t0 = p0 / 2^32
            // t1 = p1 / 2^32
            t01 = wasm32::u64x2_shr(p01, 32);

            // d[i-1] = p0 mod 2^32
            // e[i-1] = p1 mod 2^32
            de[i - 1] = wasm32::v128_and(p01, mask);
        }
        de[N - 1] = t01;
    }

    let mut d = BigInt::<N, B>([0u32; N]);
    let mut e = BigInt::<N, B>([0u32; N]);

    for i in 0..N {
        d.0[i] = wasm32::u64x2_extract_lane::<0>(de[i]) as u32;
        e.0[i] = wasm32::u64x2_extract_lane::<1>(de[i]) as u32;
    }

    if gt(&e, &d) {
        sub(
            &p_32,
            &sub(&e, &d),
        )
    } else {
        sub(&d, &e)
    }
}

/// Algorithm 4, but without using SIMD instructions
pub fn bm17_non_simd_mont_mul<const N: usize, const B: u32>(
    ar_32: &BigInt<N, B>,
    br_32: &BigInt<N, B>,
    p_32: &BigInt<N, B>,
    mu: u32,
) -> BigInt<N, B> {
    let ar = to_u64(ar_32.0);
    let br = to_u64(br_32.0);
    let p = to_u64(p_32.0);

    let w_mu = Wrapping(mu as u64);
    let mask = 2u64.pow(B) - 1;
    let mut d = [0u64; N];
    let mut e = [0u64; N];
    let mu_b0 = w_mu * Wrapping(br[0]);

    for j in 0..N {
        let d0_minus_e0 = Wrapping(d[0]) - Wrapping(e[0]);

        // q = (mub0)aj + mu(d0 - e0) mod 2^32
        let q = (mu_b0 * Wrapping(ar[j]) + w_mu * d0_minus_e0).0 & mask;

        // t0 = (ajb0 + d0) / 2^32
        let mut t0 = (ar[j] * br[0] + d[0]) >> B;

        // t1 = (qp0 + e0) / 2^32
        let mut t1 = (q * p[0] + e[0]) >> B;

        for i in 1..N {
            // p0 = ajbi + t0 + di
            let p0 = ar[j] * br[i] + t0 + d[i];

            // t0 = p0 / 2^32
            t0 = p0 >> B;

            // d[i-1] = p0 mod 2^32
            d[i - 1] = p0 & mask;

            // p1 = qpi + t1 + ei
            let p1 = q * p[i] + t1 + e[i];

            // t1 = p1 / 2^32
            t1 = p1 >> B;

            // e[i-1] = p1 mod 2^32
            e[i - 1] = p1 & mask;
        }
        d[N - 1] = t0;
        e[N - 1] = t1;
    }

    let mut p_u32 = [0u32; N];
    let mut e_u32 = [0u32; N];
    let mut d_u32 = [0u32; N];
    for i in 0..N {
        p_u32[i] = p[i] as u32;
        e_u32[i] = e[i] as u32;
        d_u32[i] = d[i] as u32;
    }

    let d = BigInt::<N, B>(d_u32);
    let e = BigInt::<N, B>(e_u32);
    let p = BigInt::<N, B>(p_u32);

    if gt(&e, &d) {
        sub(
            &p,
            &sub(&e, &d),
        )
    } else {
        sub(&d, &e)
    }
}
