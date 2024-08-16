use wasm_bindgen::prelude::*;
use web_sys::console;
use core::arch::wasm32::*;

#[wasm_bindgen]
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
pub fn do_run() {
    let a: u32 = 0xabcd;
    let b: u32 = 2;
    let c: u32 = 3;
    let d: u32 = 4;

    let val = u32x4(a, b, c, d);

    let e = u32x4_extract_lane::<0>(val);
    console::log_1(&e.to_string().into());
}

