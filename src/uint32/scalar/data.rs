#![allow(clippy::needless_range_loop)]

use super::polyfill::*;
use crate::X64;

/// Load 8, 256 bit registers holding 64 32-bit elements.
pub(crate) fn load_u32x64(block: &[u32; X64]) -> [u32x8; 8] {
    let ptr: *const u32 = block.as_ptr();
    let mut data = [u32x8::ZERO; 8];
    for i in 0..8 {
        data[i] = unsafe { _scalar_load_u32x8(ptr.add(i)) };
    }
    data
}

/// Store 8, 256 bit registers holding 64 32-bit elements.
pub(crate) fn store_u32x64(block: &mut [u32; X64], data: [u32x8; 8]) {
    let ptr: *mut u32 = block.as_mut_ptr();
    for i in 0..8 {
        unsafe { _scalar_store_u32x8(ptr.add(i), data[i]) };
    }
}
