use std::arch::x86_64::*;

use crate::X64;

#[target_feature(enable = "avx2")]
/// Load 8, 256 bit registers holding 64 32-bit elements.
pub(crate) fn load_u32x64(block: &[u32; X64]) -> [__m256i; 8] {
    let ptr: *const __m256i = block.as_ptr().cast();
    let mut data = [_mm256_setzero_si256(); 8];
    for i in 0..8 {
        data[i] = unsafe { _mm256_loadu_si256(ptr.add(i)) };
    }
    data
}

#[target_feature(enable = "avx2")]
/// Store 8, 256 bit registers holding 64 32-bit elements.
pub(super) fn store_u32x64(block: &mut [u32; X64], data: [__m256i; 8]) {
    let ptr: *mut __m256i = block.as_mut_ptr().cast();
    for i in 0..8 {
        unsafe { _mm256_storeu_si256(ptr.add(i), data[i]) };
    }
}

#[target_feature(enable = "avx2")]
/// Store 2, 256 bit registers.
///
/// # Safety
/// The provided `out` pointer must be safe to write 64 bytes to.
pub(super) unsafe fn store_si256x2(out: *mut u8, data: [__m256i; 2]) {
    let ptr: *mut __m256i = out.cast();
    unsafe { _mm256_storeu_si256(ptr.add(0), data[0]) };
    unsafe { _mm256_storeu_si256(ptr.add(1), data[1]) };
}

#[target_feature(enable = "avx2")]
/// Store 4, 256 bit registers.
///
/// # Safety
/// The provided `out` pointer must be safe to write 128 bytes to.
pub(super) unsafe fn store_si256x4(out: *mut u8, data: [__m256i; 4]) {
    let ptr: *mut __m256i = out.cast();
    unsafe { _mm256_storeu_si256(ptr.add(0), data[0]) };
    unsafe { _mm256_storeu_si256(ptr.add(1), data[1]) };
    unsafe { _mm256_storeu_si256(ptr.add(2), data[2]) };
    unsafe { _mm256_storeu_si256(ptr.add(3), data[3]) };
}

#[target_feature(enable = "avx2")]
/// Store 8, 256 bit registers.
///
/// # Safety
/// The provided `out` pointer must be safe to write 256 bytes to.
pub(super) unsafe fn store_si256x8(out: *mut u8, data: [__m256i; 8]) {
    let ptr: *mut __m256i = out.cast();
    unsafe { _mm256_storeu_si256(ptr.add(0), data[0]) };
    unsafe { _mm256_storeu_si256(ptr.add(1), data[1]) };
    unsafe { _mm256_storeu_si256(ptr.add(2), data[2]) };
    unsafe { _mm256_storeu_si256(ptr.add(3), data[3]) };
    unsafe { _mm256_storeu_si256(ptr.add(4), data[4]) };
    unsafe { _mm256_storeu_si256(ptr.add(5), data[5]) };
    unsafe { _mm256_storeu_si256(ptr.add(6), data[6]) };
    unsafe { _mm256_storeu_si256(ptr.add(7), data[7]) };
}

#[target_feature(enable = "avx2")]
/// Load 2, 256 bit registers.
///
/// # Safety
/// The provided `out` pointer must be safe to read 64 bytes to.
pub(super) unsafe fn load_si256x2(out: *const u8) -> [__m256i; 2] {
    let ptr: *const __m256i = out.cast();
    let d1 = unsafe { _mm256_loadu_si256(ptr.add(0)) };
    let d2 = unsafe { _mm256_loadu_si256(ptr.add(1)) };
    [d1, d2]
}

#[target_feature(enable = "avx2")]
/// Load 4, 256 bit registers.
///
/// # Safety
/// The provided `out` pointer must be safe to read 128 bytes to.
pub(super) unsafe fn load_si256x4(out: *const u8) -> [__m256i; 4] {
    let ptr: *const __m256i = out.cast();
    let d1 = unsafe { _mm256_loadu_si256(ptr.add(0)) };
    let d2 = unsafe { _mm256_loadu_si256(ptr.add(1)) };
    let d3 = unsafe { _mm256_loadu_si256(ptr.add(2)) };
    let d4 = unsafe { _mm256_loadu_si256(ptr.add(3)) };
    [d1, d2, d3, d4]
}

#[target_feature(enable = "avx2")]
/// Load 8, 256 bit registers.
///
/// # Safety
/// The provided `out` pointer must be safe to read 256 bytes to.
pub(super) unsafe fn load_si256x8(out: *const u8) -> [__m256i; 8] {
    let ptr: *const __m256i = out.cast();
    let d1 = unsafe { _mm256_loadu_si256(ptr.add(0)) };
    let d2 = unsafe { _mm256_loadu_si256(ptr.add(1)) };
    let d3 = unsafe { _mm256_loadu_si256(ptr.add(2)) };
    let d4 = unsafe { _mm256_loadu_si256(ptr.add(3)) };
    let d5 = unsafe { _mm256_loadu_si256(ptr.add(4)) };
    let d6 = unsafe { _mm256_loadu_si256(ptr.add(5)) };
    let d7 = unsafe { _mm256_loadu_si256(ptr.add(6)) };
    let d8 = unsafe { _mm256_loadu_si256(ptr.add(7)) };
    [d1, d2, d3, d4, d5, d6, d7, d8]
}
