use core::arch::wasm32;
use crate::bigint::{BigInt, BigIntF, gt, sub, bigintf_sub};

//use web_sys::console;

pub unsafe fn msl_is_greater<
    const N: usize,
    const B: u32
>(
    val: &BigIntF<N, B>,
    p: &BigIntF<N, B>,
) -> bool {
    val.0[N - 1] > p.0[N - 1]
}

// In practice, we should look up from an array of multiples of P at index q, and subtract that
// from val.
// This function may fail for around 1% of inputs - in such cases, an explicit conditional
// subtraction step is needed.
pub unsafe fn reduce_bigintf<
    const N: usize,
    const B: u32
>(
    val: &BigIntF<N, B>,
    p: &BigIntF<N, B>,
) -> BigIntF<N, B> {
    if msl_is_greater(val, p) {
        return bigintf_sub(val, p);
    }
    *val
    /*
    let highest = val.0[N - 1];

    let q = (((highest.to_bits() as i64) * 0x1b6c as i64) >> 61usize) as u64;

    if q > 0 {
        return bigintf_sub(val, p);
    }
    return *val;
    */
}

pub unsafe fn resolve_bigintf<
    const N: usize,
    const N_MINUS_2: usize,
    const B: u32
>(
    val: &BigIntF<N, B>
) -> BigIntF<N, B> {
    let mask = 0x7ffffffffffffu64;
    let mut local = [0u64; N_MINUS_2];
    let mut res = [0f64; N];

    local[0] = val.0[1].to_bits() + ((val.0[0].to_bits() as i64) >> B) as u64;
    local[1] = val.0[2].to_bits() + ((local[0] as i64) >> B) as u64;
    local[2] = val.0[3].to_bits() + ((local[1] as i64) >> B) as u64;
    res[4]   = (val.0[4].to_bits() + ((local[2] as i64) >> B) as u64) as f64;

    res[0] = (val.0[0].to_bits() & mask) as f64;
    res[1] = (local[0]           & mask) as f64;
    res[2] = (local[1]           & mask) as f64;
    res[3] = (local[2]           & mask) as f64;

    BigIntF::<N, B>(res)
}

pub unsafe fn mont_mul_cios_f64_no_simd<
    const N: usize,
    const N_PLUS_1: usize,
    const N_PLUS_2: usize,
    const N_TIMES_2_PLUS_1: usize,
    const B: u32
>(
    ar_51: &BigIntF<N, B>,
    br_51: &BigIntF<N, B>,
    p_51: &BigIntF<N, B>,
    n0: u64,
) -> BigIntF<N, B> {
    let mut sum_h = [0u64; N_TIMES_2_PLUS_1];
    sum_h[0]=0x7990000000000000u64;
    sum_h[1]=0x6660000000000000u64;
    sum_h[2]=0x5330000000000000u64;
    sum_h[3]=0x4000000000000000u64;
    sum_h[4]=0x2CD0000000000000u64;
    sum_h[5]=0x26800000000000u64;
    sum_h[6]=0x39B0000000000000u64;
    sum_h[7]=0x4CE0000000000000u64;
    sum_h[8]=0x6010000000000000u64;
    sum_h[9]=0x7340000000000000u64;
    let mut sum_l = [0u64; N_TIMES_2_PLUS_1];
    sum_l[0]=0x7990000000000000u64;
    sum_l[1]=0x6660000000000000u64;
    sum_l[2]=0x5330000000000000u64;
    sum_l[3]=0x4000000000000000u64;
    sum_l[4]=0x2CD0000000000000u64;
    sum_l[5]=0x2680000000000000u64;
    sum_l[6]=0x39B0000000000000u64;
    sum_l[7]=0x4CE0000000000000u64;
    sum_l[8]=0x6010000000000000u64;
    sum_l[9]=0x7340000000000000u64;

    let c0 =                   0x7ffffffffffffu64;
    let c1 = f64::from_bits(0x4330000000000000u64);
    let c2 = c1;
    let c3 = f64::from_bits(0x4660000000000000u64);
    let c4 = f64::from_bits(0x4660000000000003u64);

    let mut p = [0f64; N];
    for i in 0..N {
        p[i] = p_51.0[i] as f64;
    }

    let mut l = [0f64; N];
    let mut h = [0f64; N];
    for i in 0..N {
        for j in 0..N {
            l[j] = ar_51.0[i].mul_add(br_51.0[j], c3);
            h[j] = c3;
            //console::log_1(&format!("lh: {:016x} {:016x}", l[j].to_bits(), h[j].to_bits()).into());
        }

        for j in 0..N {
            /*
            console::log_1(
                &format!(
                    "sum_l[{}] + l[{}] = {:016x} + {:016x} = {:016x}", 
                    j + 1,
                    j,
                    sum_l[j + 1] as i64,
                    l[j].to_bits() as i64,
                    (sum_l[j + 1] as i64 + l[j].to_bits() as i64) as u64,
                ).into()
            );
            */

            sum_l[j + 1] = (sum_l[j + 1] as i64 + l[j].to_bits() as i64) as u64;
            sum_h[j + 1] = (sum_h[j + 1] as i64 + h[j].to_bits() as i64) as u64;

            //console::log_1(&format!("lh[{j}]: {:016x} {:016x}", l[j].to_bits(), h[j].to_bits()).into());
            //console::log_1(&format!("sum[{j}]: {:016x} {:016x}", sum_l[j], sum_h[j]).into());
        }

        for j in 0..N {
            l[j] = c4 - l[j];
            h[j] = c4 - h[j];
        }

        for j in 0..N {
            l[j] = ar_51.0[i].mul_add(br_51.0[j], l[j]);
        }

        for j in 0..N {
            sum_l[j] = (sum_l[j] as i64 + l[j].to_bits() as i64) as u64;
            sum_h[j] = (sum_h[j] as i64 + h[j].to_bits() as i64) as u64;
            //console::log_1(&format!("sum: {:016x} {:016x}", sum_l[j], sum_h[j]).into());
        }

        let q_l = sum_l[0] * n0;
        let q_h = sum_h[0] * n0;

        //console::log_1(&format!("q_l: {:016x}; q_h: {:016x}", q_l, q_h).into());
        let term_l = f64::from_bits((((q_l & c0) as i64) + (c1.to_bits() as i64)) as u64) - c2;
        let term_h = f64::from_bits((((q_h & c0) as i64) + (c1.to_bits() as i64)) as u64) - c2;

        //console::log_1(&format!("term_l: {:016x}; term_h: {:016x}", term_l.to_bits(), term_h.to_bits()).into());
        for j in 0..N {
            l[j] = term_l.mul_add(p[j], c3);
            h[j] = term_h.mul_add(p[j], c3);
        }

        for j in 0..N {
            sum_l[j + 1] = (sum_l[j + 1] as i64 + l[j].to_bits() as i64) as u64;
            sum_h[j + 1] = (sum_h[j + 1] as i64 + h[j].to_bits() as i64) as u64;
            //console::log_1(&format!("sum: {:016x} {:016x}", sum_l[j], sum_h[j]).into());
        }

        for j in 0..N {
            l[j] = c4 - l[j];
            h[j] = c4 - h[j];
        }

        for j in 0..N {
            l[j] = term_l.mul_add(p[j], l[j]);
            h[j] = term_h.mul_add(p[j], h[j]);
            //console::log_1(&format!("lh: {:016x} {:016x}", l[j].to_bits(), h[j].to_bits()).into());
        }

        sum_l[0] = ((sum_l[0] as i64) + (l[0].to_bits() as i64)) as u64;
        sum_h[0] = ((sum_h[0] as i64) + (h[0].to_bits() as i64)) as u64;

        sum_l[1] = ((sum_l[1] as i64) + (l[1].to_bits() as i64)) as u64;
        sum_h[1] = ((sum_h[1] as i64) + (h[1].to_bits() as i64)) as u64;

        sum_l[0] = ((sum_l[1] as i64) + ((sum_l[0] as i64) >> 51) as i64) as u64;
        sum_h[0] = ((sum_h[1] as i64) + ((sum_h[0] as i64) >> 51) as i64) as u64;

        for j in 1..N - 1 {
            sum_l[j] = ((sum_l[j + 1] as i64) + (l[j + 1].to_bits() as i64)) as u64;
            sum_h[j] = ((sum_h[j + 1] as i64) + (h[j + 1].to_bits() as i64)) as u64;
        }

        sum_l[4] = sum_l[5];
        sum_h[4] = sum_h[5];

        sum_l[5] = sum_l[i + 6];
        sum_h[5] = sum_h[i + 6];

        //for j in 0..N {
            //console::log_1(&format!("sum: {:016x} {:016x}", sum_l[j], sum_h[j]).into());
        //}
        /*
        */
    }

    let mut res = [0f64; N];
    for i in 0..N {
        res[i] = f64::from_bits(sum_l[i]);
    }
    BigIntF::<N, B>(res)
}

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
/// Also see Acar, 1996.
/// This is the "classic" CIOS algorithm.
/// Does not implement the gnark optimisation (https://hackmd.io/@gnark/modular_multiplication),
/// but that should be useful.
/// Does not use SIMD instructions.
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

    // Conditional subtraction
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

#[inline(always)]
pub fn hi(v: u64) -> u64 {
    v >> 32
}

#[inline(always)]
pub fn lo(v: u64) -> u64 {
    v & 0xffffffff
}

/// Algorithm 4 of "Montgomery Arithmetic from a Software Perspective" by Bos and Montgomery
/// Uses WASM SIMD opcodes.
/// Counterintuitively, in browsers, this runs slower than the non-SIMD version, likely because the
/// SIMD opcodes are emulated rather than executed using the native processor's SIMD instructions. 
/// The performance difference can be seen in benchmarks.
/// See https://emscripten.org/docs/porting/simd.html#optimization-considerations for a list of
/// *some* WASM SIMD instructions which do not have equivalent x86 semantics; those which this
/// function uses probably suffer from the same issue.
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

/// Algorithm 4, but without using SIMD instructions. This is useful just for debugging.
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

/*
// Unfortunately I can't get this to run because there is a bug in wasm-pack related to
// relaxed-simd opcodes.
#[cfg(all(feature = "simd", target_feature = "simd128"))]
#[target_feature(enable = "relaxed-simd")]
pub unsafe fn mont_mul_cios_f64_simd<
    const N: usize,
    const N_PLUS_1: usize,
    const N_PLUS_2: usize,
    const N_TIMES_2: usize,
    const B: u32
>(
    ar_51: &BigIntF<N, B>,
    br_51: &BigIntF<N, B>,
    p_51: &BigIntF<N, B>,
    n0: u64,
) -> [wasm32::v128; N] {
    let zero = wasm32::u64x2_splat(0);
    let mut bd = [zero; N];
    let mut p = [zero; N];

    // Assign bd and p
    for i in 0..N {
        bd[i] = wasm32::f64x2_splat(br_51.0[i]);
        p[i] = wasm32::f64x2_splat(p_51.0[i]);
    }

    let mut sum = [zero; N_TIMES_2];
    // Niall's magic numbers
    sum[0]=wasm32::u64x2_splat(0x7990000000000000u64);
    sum[1]=wasm32::u64x2_splat(0x6660000000000000u64);
    sum[2]=wasm32::u64x2_splat(0x5330000000000000u64);
    sum[3]=wasm32::u64x2_splat(0x4000000000000000u64);
    sum[4]=wasm32::u64x2_splat(0x2CD0000000000000u64);
    sum[5]=wasm32::u64x2_splat(0x2680000000000000u64);
    sum[6]=wasm32::u64x2_splat(0x39B0000000000000u64);
    sum[7]=wasm32::u64x2_splat(0x4CE0000000000000u64);
    sum[8]=wasm32::u64x2_splat(0x6010000000000000u64);
    sum[9]=wasm32::u64x2_splat(0x7340000000000000u64);
    //sum[10]=wasm32::u64x2_splat(0);

    let c0 = wasm32::u64x2_splat(0x7fffffffffffffu64);
    let c1 = wasm32::u64x2_splat(0x4330000000000000u64);
    let c2 = c1;
    let c3 = wasm32::u64x2_splat(0x4660000000000000u64);
    let c4 = wasm32::u64x2_splat(0x4660000000000003u64);

    let mut term = wasm32::u64x2_splat(0);
    let mut lh = [zero; N];
    for i in 0..N {
        term = wasm32::f64x2(ar_51.0[i], 0f64);

        for j in 0..N {
            lh[j] = wasm32::f64x2_relaxed_madd(term, bd[j], c3);
        }

        for j in 0..N {
            sum[j + 1] = wasm32::f64x2_add(sum[j + 1], lh[j]);
        }

        for j in 0..N {
            lh[j] = wasm32::f64x2_sub(c4, lh[j]);
        }

        for j in 0..N {
            lh[j] = wasm32::f64x2_relaxed_madd(term, bd[j], lh[j]);
        }

        for j in 0..N {
            sum[j] = wasm32::f64x2_add(sum[j], lh[j]);
        }

        let q0: u64 = wasm32::u64x2_extract_lane::<0>(sum[0]) * n0;
        let q1: u64 = wasm32::u64x2_extract_lane::<1>(sum[0]) * n0;

        term = wasm32::f64x2_sub(
            wasm32::u64x2_add(
                wasm32::v128_and(
                    wasm32::u64x2(q0, q1),
                    c0
                ),
                c1
            ),
            c2
        );

        for j in 0..N {
            lh[j] = wasm32::f64x2_relaxed_madd(term, p[j], c3);
        }

        for j in 0..N {
            sum[j + 1] = wasm32::u64x2_add(sum[j + 1], lh[j]);
        }

        for j in 0..N {
            lh[j] = wasm32::f64x2_sub(c4, lh[j]);
        }

        for j in 0..N {
            lh[j] = wasm32::f64x2_relaxed_madd(term, p[j], lh[j]);
        }

        sum[0] = wasm32::u64x2_add(sum[0], lh[0]);
        sum[1] = wasm32::u64x2_add(sum[1], lh[1]);
        sum[0] = wasm32::u64x2_add(sum[1], wasm32::u64x2_shr(sum[0], 51));

        for j in 1..N - 1 {
            sum[j] = wasm32::u64x2_add(sum[j + 1], lh[j + 1]);
        }

        sum[4] = sum[5];
        sum[5] = sum[i + 6];
    }

    let mut res = [zero; N];

    for i in 0..N {
        res[i] = sum[i];
    }

    res
}
*/
