use std::num::Wrapping;

#[derive(Copy, Clone)]
pub struct BigInt<const N: usize, const B: u32>(
    #[doc(hidden)]
    pub [u32; N],
);

// 8 * 32 = 256
pub type BigInt256 = BigInt<8, 32>;

/// Returns lhs > rhs.
pub fn gt<const N: usize, const B: u32>(
    lhs: &BigInt<N, B>,
    rhs: &BigInt<N, B>,
) -> bool {
    for idx in 0..N {
        let i = N - 1 - idx;
        if lhs.0[i] < rhs.0[i] {
            return false;
        } else if lhs.0[i] > rhs.0[i] {
            return true;
        }
    }
    false
}

/// Returns the subtraction of rhs from lhs. lhs and rhs must have the same length. This function
/// assumes that rhs is smaller than lhs. If this is not the case, the
/// result will be incorrect.
///
/// # Arguments
/// 
/// * `lhs` - the left term (minuend)
/// * `rhs` - the right term (subtrahend)
pub fn sub<const N: usize, const B: u32>(
    lhs: &BigInt<N, B>,
    rhs: &BigInt<N, B>,
) -> BigInt<N, B> {
    let num_limbs = N;

    let mut w_borrow = Wrapping(0u32);
    let mut res = [0u32; N];

    let two_pow_word_size = 2u64.pow(B);

    for i in 0..num_limbs {
        let w_lhs = Wrapping(lhs.0[i]);
        let w_rhs = Wrapping(rhs.0[i]);

        res[i] = (w_lhs - w_rhs - w_borrow).0;

        if lhs.0[i] < (w_rhs + w_borrow).0 {
            res[i] = (((Wrapping(res[i] as u64) + Wrapping(two_pow_word_size)).0) % 2u64.pow(32)) as u32;
            w_borrow = Wrapping(1u32);
        } else {
            w_borrow = Wrapping(0u32);
        }
    }

    BigInt::<N, B>(res)
}


/*
use core::arch::wasm32::{v128, u32x4, u64x2, u64x2_add, u64x2_sub, u64x2_extract_lane, u64x2_replace_lane};
use std::cmp::{PartialEq, Eq};
use std::default::Default;

/// N is the number of B-bit limbs
/// V is the number of v128 variables which represent the BigInt.
/// B is the limb size in bits. The maximum is 32.
/// e.g. if M = 30, and there is no carry, the lower 32 bits are used
/// to store the limb. If there is a carry, the higher 32 bits are used.
pub struct BigInt<const N: usize, const V: usize, const B: u32>(
    #[doc(hidden)]
    pub [v128; V],
);

// A 256-bit BigInt represented by 4 v128 variables, each of which contains 2 limbs of 32 bits
// each.
pub type BigInt256 = BigInt<8, 4, 32>;

// A 256-bit BigInt represented by 5 v128 variables, each of which contains 2 limbs of 30 bits
// each.
pub type BigInt300 = BigInt<10, 5, 30>;

// The default value is 0.
impl<const N: usize, const V: usize, const B: u32> Default for BigInt<N, V, B> {
    fn default() -> BigInt<N, V, B> {
        // Create a zero-value v128
        let zero: v128 = u64x2(0, 0);

        // Return the default 0-value BigInt represented by an array of V 0-value v128s
        BigInt([zero; V])
    }
}

// Equality is strictly limb-wise, not equality of the whole BigInt. This means that a BigInt of a
// certain value, but with some unapplied carries in the upper bits of one ore more limbs, may
// equal another BigInt with normalised limbs.
impl<const N: usize, const V: usize, const B: u32> PartialEq for BigInt<N, V, B> {
    /// Compare the limbs of `self` against that of another.
    fn eq(&self, other: &Self) -> bool {
        for i in 0..V {
            let self_lo = u64x2_extract_lane::<0>(self.0[i]);
            let other_lo = u64x2_extract_lane::<0>(other.0[i]);

            if self_lo != other_lo {
                return false;
            }

            let self_hi = u64x2_extract_lane::<1>(self.0[i]);
            let other_hi = u64x2_extract_lane::<1>(other.0[i]);

            if self_hi != other_hi {
                return false;
            }
        }

        true
    }
}

impl<const N: usize, const V: usize, const B: u32> Eq for BigInt<N, V, B> {}

impl<const N: usize, const V: usize, const B: u32> BigInt<N, V, B> {
    // A compile-time function to calculate the mask
    const fn compute_mask() -> u64 {
        (1u64 << B) - 1
    }

    pub fn new(data: [u32; N]) -> BigInt<N, V, B> {
        Self::do_new(data, true)
    }

    pub fn new_unchecked(data: [u32; N]) -> BigInt<N, V, B> {
        Self::do_new(data, false)
    }

    pub fn do_new(data: [u32; N], check: bool) -> BigInt<N, V, B> {
        let max = 2u64.pow(B as u32);
        let mut res = Self::default();

        for i in 0..V {
            if check {
                assert!((data[i] as u64) < max, "limb is too large");
            }

            let w: v128 = u32x4(
                data[i * 2],
                0,
                data[i * 2 + 1],
                0,
            );
            res.0[i] = w;
        }

        res
    }

    pub fn add_assign_without_carry(&mut self, other: &Self) {
        let res = Self::add_without_carry(self, other);
        *self = res;
    }

    pub fn add_without_carry(&self, other: &Self) -> Self {
        let mut res = Self::default();

        for i in 0..V {
            res.0[i] = u64x2_add(self.0[i], other.0[i]);
        }

        res
    }

    /// Add self to other and perform carries. It is "unsafe" because it assumes that the result
    /// fits within M bits.
    pub fn add_unsafe(&self, other: &Self) -> Self {
        let mut res = self.add_without_carry(other);

        let mut carry_0;
        let mut carry_1 = 0;

        let mask = Self::compute_mask();

        for i in 0..V {
            let limb = u64x2_extract_lane::<0>(res.0[i]);
            let sum = limb + carry_1;
            let new_limb = sum & mask;
            res.0[i] = u64x2_replace_lane::<0>(res.0[i], new_limb);
            carry_0 = sum >> B;

            let limb = u64x2_extract_lane::<1>(res.0[i]);
            let sum = limb + carry_0;
            let new_limb = sum & mask;
            res.0[i] = u64x2_replace_lane::<1>(res.0[i], new_limb);

            carry_1 = sum >> B;
        }

        res
    }

    pub fn sub_without_carry(&self, other: &Self) -> Self {
        let mut res = Self::default();

        for i in 0..V {
            res.0[i] = u64x2_sub(self.0[i], other.0[i]);
        }

        res
    }

    /// Subtract self to other and perform carries. Assumes that self is greater than other. If
    /// not, the result will be incorrect.
    pub fn sub(&self, other: &Self) -> Self {
        let mut res = self.sub_without_carry(other);

        let mut carry_0;
        let mut carry_1 = 0u64;

        let mask = Self::compute_mask();

        for i in 0..V {
            // Subtract and handle carry for the lower lane (lane 0)
            let limb = u64x2_extract_lane::<0>(res.0[i]);
            let diff = limb.wrapping_sub(carry_1);
            let new_limb = diff & mask;
            res.0[i] = u64x2_replace_lane::<0>(res.0[i], new_limb);
            carry_0 = (limb < carry_1) as u64;

            // Subtract and handle carry for the upper lane (lane 1)
            let limb = u64x2_extract_lane::<1>(res.0[i]);
            let diff = limb.wrapping_sub(carry_0);
            let new_limb = diff & mask;
            res.0[i] = u64x2_replace_lane::<1>(res.0[i], new_limb);

            carry_1 = (limb < carry_0) as u64;
        }

        res
    }
}
*/
