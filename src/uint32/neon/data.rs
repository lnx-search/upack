#![allow(clippy::needless_range_loop)]

use std::arch::aarch64::*;
use std::mem::MaybeUninit;

use super::polyfill::*;
use crate::X64;

#[target_feature(enable = "neon")]
/// Load 16, 128 bit registers holding 64 32-bit elements.
pub(super) fn load_u32x64(block: &[u32; X64]) -> [uint32x4_t; 16] {
    let ptr: *const u32 = block.as_ptr();
    let mut data: [MaybeUninit<uint32x4_t>; 16] = [const { MaybeUninit::uninit() }; 16];
    for i in 0..16 {
        data[i].write(unsafe { _neon_load_u32(ptr.add(i * 4)) });
    }
    unsafe { std::mem::transmute::<[MaybeUninit<uint32x4_t>; 16], [uint32x4_t; 16]>(data) }
}

#[target_feature(enable = "neon")]
/// Store 8, 256 bit registers holding 64 32-bit elements.
pub(super) fn store_u32x64(block: &mut [u32; X64], data: [uint32x4_t; 16]) {
    let ptr: *mut u32 = block.as_mut_ptr();
    for i in 0..16 {
        unsafe { _neon_store_u32(ptr.add(i * 4), data[i]) };
    }
}

#[target_feature(enable = "neon")]
/// Store 4, 128 bit registers holding 64 8-bit elements.
pub(super) unsafe fn store_u8x16x4(out: *mut u8, data: [uint8x16_t; 4]) {
    unsafe { _neon_store_u8(out.add(0), data[0]) };
    unsafe { _neon_store_u8(out.add(32), data[1]) };
}

#[target_feature(enable = "neon")]
/// Store 8, 128 bit registers holding 64 16-bit elements.
pub(super) unsafe fn store_u16x8x8(out: *mut u8, data: [uint16x8_t; 8]) {
    let out: *mut u16 = out.cast();
    for i in 0..8 {
        unsafe { _neon_store_u16(out.add(i * 8), data[i]) };
    }
}

#[target_feature(enable = "neon")]
/// Store 16, 128 bit registers holding 64 32-bit elements.
pub(super) unsafe fn store_u32x4x16(out: *mut u8, data: [uint32x4_t; 16]) {
    let out: *mut u32 = out.cast();
    for i in 0..16 {
        unsafe { _neon_store_u32(out.add(i * 4), data[i]) };
    }
}

#[target_feature(enable = "neon")]
/// Load 4, 128 bit registers holding 64 8-bit elements.
pub(super) unsafe fn load_u8x16x4(ptr: *const u8) -> [uint8x16_t; 4] {
    let d1 = unsafe { _neon_load_u8(ptr.add(0)) };
    let d2 = unsafe { _neon_load_u8(ptr.add(16)) };
    let d3 = unsafe { _neon_load_u8(ptr.add(32)) };
    let d4 = unsafe { _neon_load_u8(ptr.add(48)) };
    [d1, d2, d3, d4]
}

#[target_feature(enable = "neon")]
/// Load 8, 128 bit registers holding 64 16-bit elements.
pub(super) unsafe fn load_u16x8x8(ptr: *const u8) -> [uint16x8_t; 8] {
    let ptr: *const u16 = ptr.cast();
    let mut data: [MaybeUninit<uint16x8_t>; 8] = [const { MaybeUninit::uninit() }; 8];
    for i in 0..8 {
        data[i].write(unsafe { _neon_load_u16(ptr.add(i * 8)) });
    }
    unsafe { std::mem::transmute::<[MaybeUninit<uint16x8_t>; 8], [uint16x8_t; 8]>(data) }
}

#[target_feature(enable = "neon")]
/// Load 16, 128 bit registers holding 64 32-bit elements.
pub(super) unsafe fn load_u32x4x16(ptr: *const u8) -> [uint32x4_t; 16] {
    let ptr: *const u32 = ptr.cast();
    let mut data: [MaybeUninit<uint32x4_t>; 16] = [const { MaybeUninit::uninit() }; 16];
    for i in 0..16 {
        data[i].write(unsafe { _neon_load_u32(ptr.add(i * 4)) });
    }
    unsafe { std::mem::transmute::<[MaybeUninit<uint32x4_t>; 16], [uint32x4_t; 16]>(data) }
}
