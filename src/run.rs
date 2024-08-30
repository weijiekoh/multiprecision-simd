use wasm_bindgen::prelude::*;
use crate::bigint::{BigInt256, BigInt};
use crate::mont::{
    bm17_simd_mont_mul,
};
use web_sys::console;

#[wasm_bindgen]
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
pub fn do_run() {
    let ar: BigInt256 = BigInt::<8, 32>([3986754793, 2317702667, 2335837475, 3550774817, 3562346964, 124972835, 580341712, 89468053]);
    let br: BigInt256 = BigInt::<8, 32>([1587016091, 446184688, 2211600711, 4097266781, 2693467370, 3427571791, 3312884795, 155979341]);
    let p: BigInt256 = BigInt::<8, 32>([1, 168919040, 3489660929, 1504343806, 1547153409, 1622428958, 2586617174, 313222494]);
    let mu = 1;
    let result = unsafe {bm17_simd_mont_mul(&ar, &br, &p, mu)};
    console::log_1(&format!("result: {:?}", result.0).to_string().into());
}

