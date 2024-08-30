use core::arch::wasm32;
use crate::bigint::{BigInt, gt, sub};

pub fn to_u64<const N: usize>(v: [u32; N]) -> [u64; N] {
    let mut result = [0u64; N];
    for i in 0..N {
        result[i] = v[i] as u64;
    }
    result
}

/// Amine Mrabet, Nadia El-Mrabet, Ronan Lashermes, Jean-Baptiste Rigaud, Belgacem Bouallegue, et
/// al.. High-performance Elliptic Curve Cryptography by Using the CIOS Method for Modular
/// Multiplication. CRiSIS 2016, Sep 2016, Roscoff, France. hal-01383162
/// https://inria.hal.science/hal-01383162/document, page 4
/// Also see Acar 1996
/// Does not implement the gnark optimisation
pub unsafe fn mont_mul_cios<
    const N: usize,
    const N_PLUS_1: usize,
    const N_PLUS_2: usize,
    const B: u32
>(
    ar_32: &BigInt<N, B>,
    br_32: &BigInt<N, B>,
    p_32: &BigInt<N, B>,
    n0: u32,
) -> BigInt<N, B> {
    let ar = to_u64(ar_32.0);
    let br = to_u64(br_32.0);
    let p = to_u64(p_32.0);

    let n0 = n0 as u64;

    let mut t = [0u64; N_PLUS_2];
    let mask = (1u64 << B) - 1;

    for i in 0..N {
        let mut c = 0u64;
        for j in 0..N {
            let cs = t[j] + ar[i] * br[j] + c;
            c = hi(cs);
            t[j] = lo(cs);
        }

        let cs = t[N] + c;
        c = hi(cs);
        t[N] = lo(cs);
        t[N + 1] = c;

        let m = (t[0] * n0) & mask;
        let cs = t[0] + m * p[0];
        c = hi(cs);

        for j in 1..N {
            let cs = t[j] + m * p[j] + c;
            c = hi(cs);
            t[j - 1] = lo(cs);
        }

        let cs = t[N] + c;
        c = hi(cs);
        t[N - 1] = lo(cs);
        t[N] = t[N + 1] + c;
    }

    let mut t_gt_p = false;
    for idx in 0..N + 1 {
        let i = N - idx;
        let pi = if i < N {
            p[i]
        } else {
            0
        };
        if t[i] < pi {
            break;
        } else if t[i] > pi {
            t_gt_p = true;
            break;
        }
    }

    if !t_gt_p {
        let mut result = [0u32; N];
        for i in 0..N {
            result[i] = t[i] as u32;
        }
        return BigInt(result);
    }

    let mut t_wide = [0u32; N_PLUS_1];
    let mut p_wide = [0u32; N_PLUS_1];

    for i in 0.. N {
        p_wide[i] = p[i] as u32;
    }
    for i in 0.. N + 1 {
        t_wide[i] = t[i] as u32;
    }

    let mut result = [0; N_PLUS_1];
    let mut borrow = 0;
    let limb_mask = (1u64 << B) - 1;

    for i in 0..N_PLUS_1 {
        let lhs_limb = t_wide[i] as u64;
        let rhs_limb = p_wide[i] as u64;
        
        let diff = lhs_limb.wrapping_sub(rhs_limb).wrapping_sub(borrow);
        result[i] = (diff & limb_mask) as u32;
        
        borrow = (diff >> B) & 1;
    }

    let mut res = [0; N];
    for i in 0..N {
        res[i] = result[i];
    }

    BigInt::<N, B>(res)
}

pub fn hi(v: u64) -> u64 {
    v >> 32
}

pub fn lo(v: u64) -> u64 {
    v & 0xffffffff
}

/// Algorithm 4 of "Montgomery Arithmetic from a Software Perspective" by Bos and Montgomery
/// Uses WASM SIMD opcodes.
/// Also see:
/// https://github.com/coreboot/vboot/blob/060efa0cf64d4b7ccbe3e88140c9da5f747355ee/firmware/2lib/2modpow_sse2.c#L113
/// Note: overflow-checks = false should be set in Cargo.toml under [profile.dev], so the Wrapping
/// trait does not have to be used.
#[target_feature(enable = "simd128")]
pub unsafe fn bm17_simd_mont_mul<const N: usize, const B: u32>(
    ar_32: &BigInt<N, B>,
    br_32: &BigInt<N, B>,
    p_32: &BigInt<N, B>,
    mu: u32,
) -> BigInt<N, B> {
    let ar = to_u64(ar_32.0);
    let br = to_u64(br_32.0);
    let p = to_u64(p_32.0);

    let w_mu = mu as u64;
    let mask_64 = 2u64.pow(B) - 1;
    let mask = wasm32::u64x2(mask_64, mask_64);

    // The zero wasm32::v128
    let z = wasm32::u64x2(0, 0);

    // Initialise d and e
    let mut de = [z; N];

    // mu * br[0]
    let mu_b0 = w_mu * br[0];

    // Assign br and p into v128s
    let mut bp = [z; N];
    for i in 0..N {
        bp[i] = wasm32::u64x2(br[i], p[i]);
    }

    for j in 0..N {
        // Compute q
        let d0 = wasm32::u64x2_extract_lane::<0>(de[0]);
        let e0 = wasm32::u64x2_extract_lane::<1>(de[0]);
        let d0_minus_e0 = d0 - e0;

        // q = (mub0)aj + mu(d0 - e0) mod 2^32
        let q = (mu_b0 * ar[j] + w_mu * d0_minus_e0) & mask_64;

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

    let w_mu = mu as u64;
    let mask = 2u64.pow(B) - 1;
    let mut d = [0u64; N];
    let mut e = [0u64; N];
    let mu_b0 = w_mu * br[0];

    for j in 0..N {
        let d0_minus_e0 = d[0] - e[0];

        // q = (mub0)aj + mu(d0 - e0) mod 2^32
        let q = (mu_b0 * ar[j] + w_mu * d0_minus_e0) & mask;

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
