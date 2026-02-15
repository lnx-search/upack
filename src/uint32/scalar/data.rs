#![allow(clippy::needless_range_loop)]

use super::polyfill::*;
use crate::X64;

/// Load 8, 256 bit registers holding 64 32-bit elements.
pub(crate) fn load_u32x64(block: &[u32; X64]) -> [u32x8; 8] {
    let ptr: *const u32 = block.as_ptr();
    let mut data = [u32x8::ZERO; 8];
    for i in 0..8 {
        data[i] = unsafe { _scalar_load_u32x8(ptr.add(i * 8)) };
    }
    data
}

/// Store 8, 256 bit registers holding 64 32-bit elements.
pub(crate) fn store_u32x64(block: &mut [u32; X64], data: [u32x8; 8]) {
    let ptr: *mut u32 = block.as_mut_ptr();
    for i in 0..8 {
        unsafe { _scalar_store_u32x8(ptr.add(i * 8), data[i]) };
    }
}

/// Store 2, 256 bit registers holding 64 8-bit elements.
pub(crate) unsafe fn store_u8x32x2(out: *mut u8, data: [u8x32; 2]) {
    unsafe { _scalar_store_u8x32(out.add(0), data[0].into()) };
    unsafe { _scalar_store_u8x32(out.add(32), data[1].into()) };
}

/// Store 4, 256 bit registers holding 64 16-bit elements.
pub(crate) unsafe fn store_u16x16x4(out: *mut u8, data: [u16x16; 4]) {
    unsafe { _scalar_store_u8x32(out.add(0), data[0].into()) };
    unsafe { _scalar_store_u8x32(out.add(32), data[1].into()) };
    unsafe { _scalar_store_u8x32(out.add(64), data[2].into()) };
    unsafe { _scalar_store_u8x32(out.add(96), data[3].into()) };
}

/// Store 8, 256 bit registers holding 64 32-bit elements.
pub(crate) unsafe fn store_u32x8x8(out: *mut u8, data: [u32x8; 8]) {
    unsafe { _scalar_store_u8x32(out.add(0), data[0].into()) };
    unsafe { _scalar_store_u8x32(out.add(32), data[1].into()) };
    unsafe { _scalar_store_u8x32(out.add(64), data[2].into()) };
    unsafe { _scalar_store_u8x32(out.add(96), data[3].into()) };
    unsafe { _scalar_store_u8x32(out.add(128), data[4].into()) };
    unsafe { _scalar_store_u8x32(out.add(160), data[5].into()) };
    unsafe { _scalar_store_u8x32(out.add(192), data[6].into()) };
    unsafe { _scalar_store_u8x32(out.add(224), data[7].into()) };
}

#[inline(never)]
/// Load 2, 256 bit registers holding 64 8-bit elements.
pub(crate) unsafe fn load_u8x32x2(out: *const u8) -> [u8x32; 2] {
    unsafe {
        [
            _scalar_load_u8x32(out.add(0)),
            _scalar_load_u8x32(out.add(32)),
        ]
    }
}

/// Load 4, 256 bit registers holding 64 16-bit elements.
pub(crate) unsafe fn load_u16x16x4(out: *const u8) -> [u16x16; 4] {
    unsafe {
        [
            _scalar_load_u8x32(out.add(0)).into(),
            _scalar_load_u8x32(out.add(32)).into(),
            _scalar_load_u8x32(out.add(64)).into(),
            _scalar_load_u8x32(out.add(96)).into(),
        ]
    }
}

/// Load 8, 256 bit registers holding 64 32-bit elements.
pub(crate) unsafe fn load_u32x8x8(out: *const u8) -> [u32x8; 8] {
    unsafe {
        [
            _scalar_load_u8x32(out.add(0)).into(),
            _scalar_load_u8x32(out.add(32)).into(),
            _scalar_load_u8x32(out.add(64)).into(),
            _scalar_load_u8x32(out.add(96)).into(),
            _scalar_load_u8x32(out.add(128)).into(),
            _scalar_load_u8x32(out.add(160)).into(),
            _scalar_load_u8x32(out.add(192)).into(),
            _scalar_load_u8x32(out.add(224)).into(),
        ]
    }
}
