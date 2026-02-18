#![allow(clippy::needless_range_loop)]

use super::polyfill::*;
use crate::X64;

#[target_feature(enable = "neon")]
/// Load 8, 256 bit registers holding 64 32-bit elements.
pub(super) fn load_u32x64(block: &[u32; X64]) -> [u32x8; 8] {
    let ptr: *const u32 = block.as_ptr();
    let mut data = [_neon_set1_u32(0); 8];
    for i in 0..8 {
        data[i] = unsafe { _neon_load_u32x8(ptr.add(i * 8)) };
    }
    data
}

#[target_feature(enable = "neon")]
/// Store 8, 256 bit registers holding 64 32-bit elements.
pub(super) fn store_u32x64(block: &mut [u32; X64], data: [u32x8; 8]) {
    let ptr: *mut u32 = block.as_mut_ptr();
    for i in 0..8 {
        unsafe { _neon_store_u32x8(ptr.add(i * 8), data[i]) };
    }
}

#[target_feature(enable = "neon")]
/// Store 2, 256 bit registers holding 64 8-bit elements.
pub(super) unsafe fn store_u8x32x2(out: *mut u8, data: [u8x32; 2]) {
    unsafe { _neon_store_u8x32(out.add(0), data[0]) };
    unsafe { _neon_store_u8x32(out.add(32), data[1]) };
}

#[target_feature(enable = "neon")]
/// Store 4, 256 bit registers holding 64 16-bit elements.
pub(super) unsafe fn store_u16x16x4(out: *mut u8, data: [u16x16; 4]) {
    let out: *mut u16 = out.cast();
    unsafe { _neon_store_u16x16(out.add(0), data[0]) };
    unsafe { _neon_store_u16x16(out.add(16), data[1]) };
    unsafe { _neon_store_u16x16(out.add(32), data[2]) };
    unsafe { _neon_store_u16x16(out.add(48), data[3]) };
}

#[target_feature(enable = "neon")]
/// Store 8, 256 bit registers holding 64 32-bit elements.
pub(super) unsafe fn store_u32x8x8(out: *mut u8, data: [u32x8; 8]) {
    let out: *mut u32 = out.cast();
    unsafe { _neon_store_u32x8(out.add(0), data[0]) };
    unsafe { _neon_store_u32x8(out.add(8), data[1]) };
    unsafe { _neon_store_u32x8(out.add(16), data[2]) };
    unsafe { _neon_store_u32x8(out.add(24), data[3]) };
    unsafe { _neon_store_u32x8(out.add(32), data[4]) };
    unsafe { _neon_store_u32x8(out.add(40), data[5]) };
    unsafe { _neon_store_u32x8(out.add(48), data[6]) };
    unsafe { _neon_store_u32x8(out.add(56), data[7]) };
}

#[target_feature(enable = "neon")]
/// Load 2, 256 bit registers holding 64 8-bit elements.
pub(super) unsafe fn load_u8x32x2(ptr: *const u8) -> [u8x32; 2] {
    unsafe { [_neon_load_u8x32(ptr.add(0)), _neon_load_u8x32(ptr.add(32))] }
}

#[target_feature(enable = "neon")]
/// Load 4, 256 bit registers holding 64 16-bit elements.
pub(super) unsafe fn load_u16x16x4(ptr: *const u8) -> [u16x16; 4] {
    let ptr: *const u16 = ptr.cast();
    unsafe {
        [
            _neon_load_u16x16(ptr.add(0)),
            _neon_load_u16x16(ptr.add(16)),
            _neon_load_u16x16(ptr.add(32)),
            _neon_load_u16x16(ptr.add(48)),
        ]
    }
}

#[target_feature(enable = "neon")]
/// Load 8, 256 bit registers holding 64 32-bit elements.
pub(super) unsafe fn load_u32x8x8(ptr: *const u8) -> [u32x8; 8] {
    let ptr: *const u32 = ptr.cast();
    unsafe {
        [
            _neon_load_u32x8(ptr.add(0)),
            _neon_load_u32x8(ptr.add(8)),
            _neon_load_u32x8(ptr.add(16)),
            _neon_load_u32x8(ptr.add(24)),
            _neon_load_u32x8(ptr.add(32)),
            _neon_load_u32x8(ptr.add(40)),
            _neon_load_u32x8(ptr.add(48)),
            _neon_load_u32x8(ptr.add(56)),
        ]
    }
}
