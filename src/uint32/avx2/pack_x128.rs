use std::arch::x86_64::*;

use super::data::*;
use super::utils::*;
use crate::X128;
use crate::uint32::X128_MAX_OUTPUT_LEN;

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to `nbits` sized elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128` along with `nbits` being between `1..=32`.
pub unsafe fn to_nbits(
    out: &mut [u8; X128_MAX_OUTPUT_LEN],
    nbits: u8,
    block: &[u32; X128],
    pack_n: usize,
) {
    debug_assert!(nbits <= 32, "BUG: invalid nbits provided: {nbits}");
    debug_assert!(pack_n <= X128, "BUG: invalid pack_n provided: {pack_n}");
    const LUT: [unsafe fn(&mut [u8; X128_MAX_OUTPUT_LEN], &[u32; X128], usize); 33] = [
        to_u0, to_u1, to_u2, to_u3, to_u4, to_u5, to_u6, to_u7, to_u8, to_u9, to_u10, to_u11,
        to_u12, to_u13, to_u14, to_u15, to_u16, to_u17, to_u18, to_u19, to_u20, to_u21, to_u22,
        to_u23, to_u24, to_u25, to_u26, to_u27, to_u28, to_u29, to_u30, to_u31, to_u32,
    ];
    let func = unsafe { LUT.get_unchecked(nbits as usize) };
    unsafe { func(out, block, pack_n) };
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 0-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u0(_out: &mut [u8; X128_MAX_OUTPUT_LEN], _block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 1-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u1(out: &mut [u8; X128_MAX_OUTPUT_LEN], block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
    let [left, right] = split_block(block);
    let left = load_u32x64(left);
    let left_packed = pack_u32_u8_x8(left);
    let right = load_u32x64(right);
    let right_packed = pack_u32_u8_x8(right);
    let partially_packed = [
        left_packed[0],
        left_packed[1],
        right_packed[0],
        right_packed[1],
    ];
    unsafe { pack_u1_registers(out.as_mut_ptr(), partially_packed) }
}

#[target_feature(enable = "avx2")]
/// Pack four registers containing 32 8-bit elements each into a 1-bit
/// bitmap and write to `out`.
///
/// Any non-zero value will be treated as a set bit.
unsafe fn pack_u1_registers(out: *mut u8, data: [__m256i; 4]) {
    let [d1, d2, d3, d4] = data;

    let zeroes = _mm256_setzero_si256();
    let cmp1 = _mm256_cmpeq_epi8(d1, zeroes);
    let cmp2 = _mm256_cmpeq_epi8(d2, zeroes);
    let cmp3 = _mm256_cmpeq_epi8(d3, zeroes);
    let cmp4 = _mm256_cmpeq_epi8(d4, zeroes);

    let mask1 = !_mm256_movemask_epi8(cmp1);
    let mask2 = !_mm256_movemask_epi8(cmp2);
    let mask3 = !_mm256_movemask_epi8(cmp3);
    let mask4 = !_mm256_movemask_epi8(cmp4);

    // We assume LE endianness, so we know `mask2`, etc... can only ever be non-zero
    // when we have more than 32 elements in `pack_n`.
    unsafe { std::ptr::write_unaligned(out.add(0).cast(), mask1) };
    unsafe { std::ptr::write_unaligned(out.add(4).cast(), mask2) };
    unsafe { std::ptr::write_unaligned(out.add(8).cast(), mask3) };
    unsafe { std::ptr::write_unaligned(out.add(12).cast(), mask4) };
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 2-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u2(_out: &mut [u8; X128_MAX_OUTPUT_LEN], _block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 3-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u3(_out: &mut [u8; X128_MAX_OUTPUT_LEN], _block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 4-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u4(_out: &mut [u8; X128_MAX_OUTPUT_LEN], _block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 5-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u5(_out: &mut [u8; X128_MAX_OUTPUT_LEN], _block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 6-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u6(_out: &mut [u8; X128_MAX_OUTPUT_LEN], _block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 7-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u7(_out: &mut [u8; X128_MAX_OUTPUT_LEN], _block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 8-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u8(_out: &mut [u8; X128_MAX_OUTPUT_LEN], _block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 9-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u9(_out: &mut [u8; X128_MAX_OUTPUT_LEN], _block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 10-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u10(_out: &mut [u8; X128_MAX_OUTPUT_LEN], _block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 11-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u11(_out: &mut [u8; X128_MAX_OUTPUT_LEN], _block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 12-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u12(_out: &mut [u8; X128_MAX_OUTPUT_LEN], _block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 13-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u13(_out: &mut [u8; X128_MAX_OUTPUT_LEN], _block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 14-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u14(_out: &mut [u8; X128_MAX_OUTPUT_LEN], _block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 15-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u15(_out: &mut [u8; X128_MAX_OUTPUT_LEN], _block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 16-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u16(_out: &mut [u8; X128_MAX_OUTPUT_LEN], _block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 17-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u17(_out: &mut [u8; X128_MAX_OUTPUT_LEN], _block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 18-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u18(_out: &mut [u8; X128_MAX_OUTPUT_LEN], _block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 19-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u19(_out: &mut [u8; X128_MAX_OUTPUT_LEN], _block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 20-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u20(_out: &mut [u8; X128_MAX_OUTPUT_LEN], _block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 21-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u21(_out: &mut [u8; X128_MAX_OUTPUT_LEN], _block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 22-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u22(_out: &mut [u8; X128_MAX_OUTPUT_LEN], _block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 23-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u23(_out: &mut [u8; X128_MAX_OUTPUT_LEN], _block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 24-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u24(_out: &mut [u8; X128_MAX_OUTPUT_LEN], _block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 25-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u25(_out: &mut [u8; X128_MAX_OUTPUT_LEN], _block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 26-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u26(_out: &mut [u8; X128_MAX_OUTPUT_LEN], _block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 27-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u27(_out: &mut [u8; X128_MAX_OUTPUT_LEN], _block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 28-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u28(_out: &mut [u8; X128_MAX_OUTPUT_LEN], _block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 29-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u29(_out: &mut [u8; X128_MAX_OUTPUT_LEN], _block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 30-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u30(_out: &mut [u8; X128_MAX_OUTPUT_LEN], _block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 31-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u31(_out: &mut [u8; X128_MAX_OUTPUT_LEN], _block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 32-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u32(_out: &mut [u8; X128_MAX_OUTPUT_LEN], _block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
}
