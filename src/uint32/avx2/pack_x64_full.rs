use std::arch::x86_64::*;
use std::cmp;

use super::data::*;
use super::utils::*;
use crate::X64;

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 1-bit elements.
pub unsafe fn to_u1(out: *mut u8, block: &[u32; X64]) {
    let partially_packed = pack_block_to_u8_unordered(block);
    unsafe { pack_u1_registers(out, partially_packed) }
}

#[target_feature(enable = "avx2")]
/// Pack four registers containing 32 8-bit elements each into a 1-bit
/// bitmap and write to `out`.
///
/// Any non-zero value will be treated as a set bit.
unsafe fn pack_u1_registers(out: *mut u8, data: [__m256i; 2]) {
    let [d1, d2] = data;

    let zeroes = _mm256_setzero_si256();
    let cmp1 = _mm256_cmpeq_epi8(d1, zeroes);
    let cmp2 = _mm256_cmpeq_epi8(d2, zeroes);

    let mask1 = !_mm256_movemask_epi8(cmp1);
    let mask2 = !_mm256_movemask_epi8(cmp2);

    // We assume LE endianness, so we know `mask2`, etc... can only ever be non-zero
    // when we have more than 32 elements in `pack_n`.
    unsafe { std::ptr::write_unaligned(out.add(0).cast(), mask1) };
    unsafe { std::ptr::write_unaligned(out.add(4).cast(), mask2) };
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 2-bit elements.
pub unsafe fn to_u2(out: *mut u8, block: &[u32; X64]) {
    let partially_packed = pack_block_to_u8_unordered(block);
    unsafe { pack_u2_registers(out, partially_packed) }
}

#[target_feature(enable = "avx2")]
/// Pack four registers containing 32 8-bit elements each into a 2-bit
/// bitmap and write to `out`.
unsafe fn pack_u2_registers(out: *mut u8, data: [__m256i; 2]) {
    let [d1, d2] = data;

    let lo_cmp1 = _mm256_slli_epi16::<7>(d1);
    let hi_cmp1 = _mm256_slli_epi16::<6>(d1);
    let lo_mask1 = _mm256_movemask_epi8(lo_cmp1) as u32;
    let hi_mask1 = _mm256_movemask_epi8(hi_cmp1) as u32;

    let lo_cmp2 = _mm256_slli_epi16::<7>(d2);
    let hi_cmp2 = _mm256_slli_epi16::<6>(d2);
    let lo_mask2 = _mm256_movemask_epi8(lo_cmp2) as u32;
    let hi_mask2 = _mm256_movemask_epi8(hi_cmp2) as u32;

    let lo_merged_mask1 = ((lo_mask2 as u64) << 32) | lo_mask1 as u64;
    let hi_merged_mask1 = ((hi_mask2 as u64) << 32) | hi_mask1 as u64;

    unsafe { std::ptr::write_unaligned(out.add(0).cast(), lo_merged_mask1) };
    unsafe { std::ptr::write_unaligned(out.add(8).cast(), hi_merged_mask1) };
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 3-bit elements.
pub unsafe fn to_u3(out: *mut u8, block: &[u32; X64]) {
    let partially_packed = pack_block_to_u8_unordered(block);
    unsafe { pack_u3_registers(out, partially_packed) }
}

#[target_feature(enable = "avx2")]
/// Pack four registers containing 32 8-bit elements each into a 3-bit
/// bitmap and write to `out`.
unsafe fn pack_u3_registers(out: *mut u8, data: [__m256i; 2]) {
    let [d1, d2] = data;

    let b0_cmp1 = _mm256_slli_epi16::<7>(d1);
    let b1_cmp1 = _mm256_slli_epi16::<6>(d1);
    let b2_cmp1 = _mm256_slli_epi16::<5>(d1);
    let b0_mask1 = _mm256_movemask_epi8(b0_cmp1) as u32;
    let b1_mask1 = _mm256_movemask_epi8(b1_cmp1) as u32;
    let b2_mask1 = _mm256_movemask_epi8(b2_cmp1) as u32;

    let b0_cmp2 = _mm256_slli_epi16::<7>(d2);
    let b1_cmp2 = _mm256_slli_epi16::<6>(d2);
    let b2_cmp2 = _mm256_slli_epi16::<5>(d2);
    let b0_mask2 = _mm256_movemask_epi8(b0_cmp2) as u32;
    let b1_mask2 = _mm256_movemask_epi8(b1_cmp2) as u32;
    let b2_mask2 = _mm256_movemask_epi8(b2_cmp2) as u32;

    let b0_merged_mask = ((b0_mask2 as u64) << 32) | b0_mask1 as u64;
    let b1_merged_mask = ((b1_mask2 as u64) << 32) | b1_mask1 as u64;
    let b2_merged_mask = ((b2_mask2 as u64) << 32) | b2_mask1 as u64;

    unsafe { std::ptr::write_unaligned(out.add(0).cast(), b0_merged_mask) };
    unsafe { std::ptr::write_unaligned(out.add(8).cast(), b1_merged_mask) };
    unsafe { std::ptr::write_unaligned(out.add(16).cast(), b2_merged_mask) };
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 4-bit elements.
pub unsafe fn to_u4(out: *mut u8, block: &[u32; X64]) {
    let partially_packed = pack_block_to_u8_unordered(block);
    unsafe { pack_u4_registers(out, partially_packed) }
}

#[target_feature(enable = "avx2")]
/// Pack four registers containing 32 8-bit elements each into a 4-bit
/// bitmap and write to `out`.
unsafe fn pack_u4_registers(out: *mut u8, data: [__m256i; 2]) {
    let [d1, d2] = data;

    let madd_multiplier = _mm256_set1_epi16(0x1001);

    let nibbles1 = _mm256_maddubs_epi16(d1, madd_multiplier);
    let nibbles2 = _mm256_maddubs_epi16(d2, madd_multiplier);
    let interleaved = _mm256_packus_epi16(nibbles1, nibbles2);

    unsafe { _mm256_storeu_si256(out.add(0).cast(), interleaved) };
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 5-bit elements.
pub unsafe fn to_u5(out: *mut u8, block: &[u32; X64]) {
    let partially_packed = pack_block_to_u8_unordered(block);
    unsafe { pack_u5_registers(out, partially_packed) }
}

#[target_feature(enable = "avx2")]
/// Pack four registers containing 32 8-bit elements each into a 5-bit
/// bitmap and write to `out`.
unsafe fn pack_u5_registers(out: *mut u8, data: [__m256i; 2]) {
    let mask = _mm256_set1_epi8(0b1111);
    let masked = and_si256(data, mask);
    unsafe { pack_u4_registers(out, masked) };

    // 4bit * 64 / 8-bits per byte.
    let remaining = andnot_si256(data, mask);
    unsafe { pack_u1_registers(out.add(32), remaining) };
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 6-bit elements.
pub unsafe fn to_u6(out: *mut u8, block: &[u32; X64]) {
    let partially_packed = pack_block_to_u8_unordered(block);
    unsafe { pack_u6_registers(out, partially_packed) }
}

#[target_feature(enable = "avx2")]
/// Pack four registers containing 32 8-bit elements each into a 6-bit
/// bitmap and write to `out`.
unsafe fn pack_u6_registers(out: *mut u8, data: [__m256i; 2]) {
    let mask = _mm256_set1_epi8(0b1111);
    let masked = and_si256(data, mask);
    unsafe { pack_u4_registers(out, masked) };

    // 4bit * 64 / 8-bits per byte.
    let remaining = andnot_si256(data, mask);
    unsafe { pack_u2_registers(out.add(32), remaining) };
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 7-bit elements.
pub unsafe fn to_u7(out: *mut u8, block: &[u32; X64]) {
    let partially_packed = pack_block_to_u8_unordered(block);
    unsafe { pack_u7_registers(out, partially_packed) }
}

#[target_feature(enable = "avx2")]
/// Pack four registers containing 32 8-bit elements each into a 7-bit
/// bitmap and write to `out`.
unsafe fn pack_u7_registers(out: *mut u8, data: [__m256i; 2]) {
    let mask = _mm256_set1_epi8(0b1111);
    let masked = and_si256(data, mask);
    unsafe { pack_u4_registers(out, masked) };

    // 4bit * 64 / 8-bits per byte.
    let remaining = andnot_si256(data, mask);
    unsafe { pack_u3_registers(out.add(32), remaining) };
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 8-bit elements.
pub unsafe fn to_u8(out: *mut u8, block: &[u32; X64]) {
    let partially_packed = pack_block_to_u8_unordered(block);
    unsafe { store_si256x2(out, partially_packed) }
}

#[target_feature(enable = "avx2")]
fn pack_block_to_u8_unordered(block: &[u32; X64]) -> [__m256i; 2] {
    let block = load_u32x64(block);
    let packed = pack_u32_to_u8_unordered(block);
    [packed[0], packed[1]]
}
