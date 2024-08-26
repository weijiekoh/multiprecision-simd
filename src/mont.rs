use core::arch::wasm32;
use std::num::Wrapping;
use crate::bigint::{BigInt, gt, sub};
//use web_sys::console;

/// Algorithm 4 of "Montgomery Arithmetic from a Software Perspective" by Bos and Montgomery
/// Uses WASM SIMD opcodes.
/// Also see:
/// https://github.com/coreboot/vboot/blob/060efa0cf64d4b7ccbe3e88140c9da5f747355ee/firmware/2lib/2modpow_sse2.c#L113
pub fn bm17_simd_mont_mul<const N: usize, const B: u32>(
    ar: &BigInt<N, B>,
    br: &BigInt<N, B>,
    p: &BigInt<N, B>,
    mu: u32,
) -> BigInt<N, B> {
    let mask_64 = 2u64.pow(B) - 1;
    let mask = wasm32::u64x2(mask_64, mask_64);

    // The zero wasm32::v128
    let z = wasm32::u64x2(0, 0);

    // Initialise d and e
    let mut de = [z; N];

    // mu * br[0]
    let mu_b0 = Wrapping(mu as u64) * Wrapping(br.0[0] as u64);

    // Assign br and p into v128s
    let mut bp = [z; N];
    for i in 0..N {
        bp[i] = wasm32::u32x4(br.0[i], 0, p.0[i], 0);
    }

    for j in 0..N {
        // Compute q
        let d0 = wasm32::u64x2_extract_lane::<0>(de[0]);
        let e0 = wasm32::u64x2_extract_lane::<1>(de[0]);
        let d0_minus_e0 = Wrapping(d0) - Wrapping(e0);

        // q = (mub0)aj + mu(d0 - e0) mod 2^32
        let q = (mu_b0 * Wrapping(ar.0[j] as u64) + Wrapping(mu as u64) * d0_minus_e0).0 & mask_64;

        // t0 = ajb0 + d0
        // t1 = qp0 + e0

        // aq = ar[j], q
        let aq = wasm32::u32x4(ar.0[j], 0, q as u32, 0 );

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
            &p,
            &sub(&e, &d),
        )
    } else {
        sub(&d, &e)
    }
}
