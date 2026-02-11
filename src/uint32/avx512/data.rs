use std::arch::x86_64::*;

use crate::X64;

#[target_feature(enable = "avx512f")]
/// Load 4, 512 bit registers holding 64 32-bit elements.
pub(crate) fn load_u32x64(block: &[u32; X64]) -> [__m512i; 4] {
    let ptr: *const __m512i = block.as_ptr().cast();

    let d1 = unsafe { _mm512_loadu_si512(ptr.add(0)) };
    let d2 = unsafe { _mm512_loadu_si512(ptr.add(1)) };
    let d3 = unsafe { _mm512_loadu_si512(ptr.add(2)) };
    let d4 = unsafe { _mm512_loadu_si512(ptr.add(3)) };

    [d1, d2, d3, d4]
}

#[target_feature(enable = "avx512f")]
/// Store 4, 512 bit registers holding 64 32-bit elements.
pub(crate) fn store_u32x64(block: &mut [u32; X64], data: [__m512i; 4]) {
    let ptr: *mut __m512i = block.as_mut_ptr().cast();

    unsafe { _mm512_storeu_si512(ptr.add(0), data[0]) };
    unsafe { _mm512_storeu_si512(ptr.add(1), data[1]) };
    unsafe { _mm512_storeu_si512(ptr.add(2), data[2]) };
    unsafe { _mm512_storeu_si512(ptr.add(3), data[3]) };
}

#[target_feature(enable = "avx512f")]
/// Load 2, 512 bit registers.
///
/// # Safety
/// The provided `out` pointer must be safe to read 128 bytes from.
pub(crate) unsafe fn load_si512x2(input: *const u8) -> [__m512i; 2] {
    let ptr: *const __m512i = input.cast();
    let d1 = unsafe { _mm512_loadu_si512(ptr.add(0)) };
    let d2 = unsafe { _mm512_loadu_si512(ptr.add(1)) };
    [d1, d2]
}

#[target_feature(enable = "avx512f")]
/// Store 2, 512 bit registers.
///
/// # Safety
/// The provided `out` pointer must be safe to write 128 bytes to.
pub(crate) unsafe fn store_si512x2(out: *mut u8, data: [__m512i; 2]) {
    let ptr: *mut __m512i = out.cast();
    unsafe { _mm512_storeu_si512(ptr.add(0), data[0]) };
    unsafe { _mm512_storeu_si512(ptr.add(1), data[1]) };
}

#[target_feature(enable = "avx512f")]
/// Load 4, 512 bit registers.
///
/// # Safety
/// The provided `out` pointer must be safe to read 256 bytes from.
pub(crate) unsafe fn load_si512x4(input: *const u8) -> [__m512i; 4] {
    let ptr: *const __m512i = input.cast();
    let d1 = unsafe { _mm512_loadu_si512(ptr.add(0)) };
    let d2 = unsafe { _mm512_loadu_si512(ptr.add(1)) };
    let d3 = unsafe { _mm512_loadu_si512(ptr.add(2)) };
    let d4 = unsafe { _mm512_loadu_si512(ptr.add(3)) };
    [d1, d2, d3, d4]
}

#[target_feature(enable = "avx512f")]
/// Store 4, 512 bit registers.
///
/// # Safety
/// The provided `out` pointer must be safe to write 256 bytes to.
pub(crate) unsafe fn store_si512x4(out: *mut u8, data: [__m512i; 4]) {
    let ptr: *mut __m512i = out.cast();
    unsafe { _mm512_storeu_si512(ptr.add(0), data[0]) };
    unsafe { _mm512_storeu_si512(ptr.add(1), data[1]) };
    unsafe { _mm512_storeu_si512(ptr.add(2), data[2]) };
    unsafe { _mm512_storeu_si512(ptr.add(3), data[3]) };
}
