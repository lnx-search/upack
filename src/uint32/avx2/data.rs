use std::arch::x86_64::*;

use crate::{X64, X128};

#[target_feature(enable = "avx2")]
/// Load 8, 256 bit registers holding 128 32-bit elements.
pub(super) fn load_u32x64(block: &[u32; X64]) -> [__m256i; 8] {
    let ptr: *const __m256i = block.as_ptr().cast();
    let mut data = [_mm256_setzero_si256(); 8];
    for i in 0..8 {
        data[i] = unsafe { _mm256_loadu_si256(ptr.add(i)) };
    }
    data
}

#[inline]
pub(super) fn split_block(block: &[u32; X128]) -> [&[u32; X64]; 2] {
    let left: &[u32; X64] = (&block[..X64]).try_into().unwrap();
    let right: &[u32; X64] = (&block[X64..]).try_into().unwrap();
    [left, right]
}
