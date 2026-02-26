#![allow(clippy::needless_range_loop)]

use super::polyfill::*;
use crate::X64;

#[inline]
/// Load 4, 256 bit registers holding 64 16-bit elements.
pub(crate) fn load_u16x64(block: &[u16; X64]) -> [u16x16; 4] {
    let ptr: *const u16 = block.as_ptr();
    let mut data = [u16x16::ZERO; 4];
    for i in 0..4 {
        data[i] = unsafe { _scalar_load_u16x16(ptr.add(i * 16)) };
    }
    data
}

#[inline]
/// Store 4, 256 bit registers holding 64 16-bit elements.
pub(crate) fn store_u16x64(block: &mut [u16; X64], data: [u16x16; 4]) {
    let ptr: *mut u16 = block.as_mut_ptr();
    for i in 0..4 {
        unsafe { _scalar_store_u16x16(ptr.add(i * 16), data[i]) };
    }
}

#[inline]
/// Store 2, 256 bit registers holding 64 8-bit elements.
pub(crate) unsafe fn store_u8x32x2(out: *mut u8, data: [u8x32; 2]) {
    unsafe { _scalar_store_u8x32(out.add(0), data[0]) };
    unsafe { _scalar_store_u8x32(out.add(32), data[1]) };
}

#[inline]
/// Store 4, 256 bit registers holding 64 16-bit elements.
pub(crate) unsafe fn store_u16x16x4(out: *mut u8, data: [u16x16; 4]) {
    unsafe { _scalar_store_u8x32(out.add(0), data[0].into()) };
    unsafe { _scalar_store_u8x32(out.add(32), data[1].into()) };
    unsafe { _scalar_store_u8x32(out.add(64), data[2].into()) };
    unsafe { _scalar_store_u8x32(out.add(96), data[3].into()) };
}

#[inline]
/// Load 2, 256 bit registers holding 64 8-bit elements.
pub(crate) unsafe fn load_u8x32x2(out: *const u8) -> [u8x32; 2] {
    unsafe {
        [
            _scalar_load_u8x32(out.add(0)),
            _scalar_load_u8x32(out.add(32)),
        ]
    }
}

#[inline]
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
