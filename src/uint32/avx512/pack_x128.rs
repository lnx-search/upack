use super::data::*;
use super::utils::*;
use crate::X128;
use crate::uint32::X128_MAX_OUTPUT_LEN;
use std::arch::x86_64::*;
use std::cmp;

#[target_feature(enable = "avx512f", enable = "avx512bw")]
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

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 1-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128` along with `nbits` being between `1..=32`.
pub unsafe fn to_u0(_out: &mut [u8; X128_MAX_OUTPUT_LEN], _block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 1-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u1(out: &mut [u8; X128_MAX_OUTPUT_LEN], block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
    let data = load_u32x128(block);
    let mask = _mm512_set1_epi32(0b1);
    let masked = and_si512(data, mask);
    let partially_packed = pack_u32_u8_x8(masked);
    unsafe { pack_u1_registers(out.as_mut_ptr(), partially_packed) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Pack two registers containing 64 8-bit elements each into a 1-bit
/// bitmap and write to `out`.
///
/// Any non-zero value will be treated as a set bit.
unsafe fn pack_u1_registers(out: *mut u8, data: [__m512i; 2]) {
    let [d1, d2] = data;

    let zeroes = _mm512_setzero_si512();
    let mask1 = _mm512_cmpneq_epi8_mask(d1, zeroes);
    let mask2 = _mm512_cmpneq_epi8_mask(d2, zeroes);

    // We assume LE endianness, so we know `mask2` can only ever be non-zero
    // when we have more than 64 elements in `pack_n`.
    unsafe { std::ptr::write_unaligned(out.add(0).cast(), mask1) };
    unsafe { std::ptr::write_unaligned(out.add(8).cast(), mask2) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 2-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u2(out: &mut [u8; X128_MAX_OUTPUT_LEN], block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
    let data = load_u32x128(block);
    let mask = _mm512_set1_epi32(0b11);
    let masked = and_si512(data, mask);
    let partially_packed = pack_u32_u8_x8(masked);
    unsafe { pack_u2_registers(out.as_mut_ptr(), partially_packed, pack_n) }
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Pack two registers containing 64 8-bit elements each into a 2-bit
/// bitmap and write to `out`.
unsafe fn pack_u2_registers(out: *mut u8, data: [__m512i; 2], pack_n: usize) {
    let [d1, d2] = data;

    let mask = _mm512_set1_epi8(0b01);
    let lo_mask1 = _mm512_test_epi8_mask(d1, mask);
    let lo_mask2 = _mm512_test_epi8_mask(d2, mask);

    let mask = _mm512_set1_epi8(0b10);
    let hi_mask1 = _mm512_test_epi8_mask(d1, mask);
    let hi_mask2 = _mm512_test_epi8_mask(d2, mask);

    let hi_offset = cmp::min(64, pack_n).div_ceil(8);
    unsafe { std::ptr::write_unaligned(out.add(0).cast(), lo_mask1) };
    unsafe { std::ptr::write_unaligned(out.add(hi_offset).cast(), hi_mask1) };

    let hi_offset = pack_n.saturating_sub(64).div_ceil(8);
    unsafe { std::ptr::write_unaligned(out.add(16).cast(), lo_mask2) };
    unsafe { std::ptr::write_unaligned(out.add(16 + hi_offset).cast(), hi_mask2) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 3-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u3(out: &mut [u8; X128_MAX_OUTPUT_LEN], block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
    let data = load_u32x128(block);
    let mask = _mm512_set1_epi32(0b111);
    let masked = and_si512(data, mask);
    let partially_packed = pack_u32_u8_x8(masked);
    unsafe { pack_u3_registers(out.as_mut_ptr(), partially_packed, pack_n) }
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Pack two registers containing 64 8-bit elements each into a 3-bit
/// bitmap and write to `out`.
unsafe fn pack_u3_registers(out: *mut u8, data: [__m512i; 2], pack_n: usize) {
    let [d1, d2] = data;

    let mask = _mm512_set1_epi8(0b001);
    let b0_mask1 = _mm512_test_epi8_mask(d1, mask);
    let b0_mask2 = _mm512_test_epi8_mask(d2, mask);

    let mask = _mm512_set1_epi8(0b010);
    let b1_mask1 = _mm512_test_epi8_mask(d1, mask);
    let b1_mask2 = _mm512_test_epi8_mask(d2, mask);

    let mask = _mm512_set1_epi8(0b100);
    let b2_mask1 = _mm512_test_epi8_mask(d1, mask);
    let b2_mask2 = _mm512_test_epi8_mask(d2, mask);

    let step = cmp::min(64, pack_n).div_ceil(8);
    unsafe { std::ptr::write_unaligned(out.add(0).cast(), b0_mask1) };
    unsafe { std::ptr::write_unaligned(out.add(step).cast(), b1_mask1) };
    unsafe { std::ptr::write_unaligned(out.add(step * 2).cast(), b2_mask1) };

    let step = pack_n.saturating_sub(64).div_ceil(8);
    unsafe { std::ptr::write_unaligned(out.add(24).cast(), b0_mask2) };
    unsafe { std::ptr::write_unaligned(out.add(24 + step).cast(), b1_mask2) };
    unsafe { std::ptr::write_unaligned(out.add(24 + step * 2).cast(), b2_mask2) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 4-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u4(out: &mut [u8; X128_MAX_OUTPUT_LEN], block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
    let out = out.as_mut_ptr();
    let data = load_u32x128(block);
    let mask = _mm512_set1_epi32(0xF);
    let masked = and_si512(data, mask);
    let partially_packed = pack_u32_u8_x8(masked);
    unsafe { pack_u4_registers(out, partially_packed) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Pack two registers containing 64 8-bit elements each into a 4-bit
/// bitmap and write to `out`.
unsafe fn pack_u4_registers(out: *mut u8, data: [__m512i; 2]) {
    let [d1, d2] = data;

    // Multipliers for the nibble merge in the form of [1, 16, 1, 16...]
    // This is because _mm512_maddubs_epi16 computes (byte[i]*mul[i] + byte[i+1]*mul[i+1]).
    // We want: (LowVal * 1) + (HighVal * 16)
    let madd_multiplier = _mm512_set1_epi16(0x1001);

    let nibbles1 = _mm512_maddubs_epi16(d1, madd_multiplier);
    let nibbles2 = _mm512_maddubs_epi16(d2, madd_multiplier);

    let permute_mask = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);

    let interleaved = _mm512_packus_epi16(nibbles1, nibbles2);
    let ordered = _mm512_permutexvar_epi64(permute_mask, interleaved);

    unsafe { _mm512_storeu_epi8(out.cast(), ordered) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 5-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u5(out: &mut [u8; X128_MAX_OUTPUT_LEN], block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
    let out = out.as_mut_ptr();
    let data = load_u32x128(block);
    let mask = _mm512_set1_epi32(0b1_1111);
    let masked = and_si512(data, mask);
    let partially_packed = pack_u32_u8_x8(masked);
    unsafe { pack_u5_registers(out, partially_packed, pack_n) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Pack two registers containing 64 8-bit elements each into a 5-bit
/// bitmap and write to `out`.
unsafe fn pack_u5_registers(out: *mut u8, data: [__m512i; 2], pack_n: usize) {
    let mask = _mm512_set1_epi8(0b1111);
    let masked = and_si512(data, mask);
    unsafe { pack_u4_registers(out.add(0), masked) };

    // 4bit * pack_n / 8-bits per byte.
    let offset = pack_n.div_ceil(2);
    let remaining = andnot_si512(data, mask);
    unsafe { pack_u1_registers(out.add(offset), remaining) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 6-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u6(out: &mut [u8; X128_MAX_OUTPUT_LEN], block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
    let out = out.as_mut_ptr();
    let data = load_u32x128(block);
    let mask = _mm512_set1_epi32(0b11_1111);
    let masked = and_si512(data, mask);
    let partially_packed = pack_u32_u8_x8(masked);
    unsafe { pack_u6_registers(out, partially_packed, pack_n) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Pack two registers containing 64 8-bit elements each into a 6-bit
/// bitmap and write to `out`.
unsafe fn pack_u6_registers(out: *mut u8, data: [__m512i; 2], pack_n: usize) {
    let mask = _mm512_set1_epi8(0b1111);
    let masked = and_si512(data, mask);
    unsafe { pack_u4_registers(out.add(0), masked) };

    // 4bit * pack_n / 8-bits per byte.
    let offset = pack_n.div_ceil(2);
    let remaining = srli_epi16::<4, 2>(data);
    unsafe { pack_u2_registers(out.add(offset), remaining, pack_n) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 7-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u7(out: &mut [u8; X128_MAX_OUTPUT_LEN], block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
    let out = out.as_mut_ptr();
    let data = load_u32x128(block);
    let mask = _mm512_set1_epi32(0b111_1111);
    let masked = and_si512(data, mask);
    let partially_packed = pack_u32_u8_x8(masked);
    unsafe { pack_u7_registers(out, partially_packed, pack_n) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Pack two registers containing 64 8-bit elements each into a 7-bit
/// bitmap and write to `out`.
unsafe fn pack_u7_registers(out: *mut u8, data: [__m512i; 2], pack_n: usize) {
    let mask = _mm512_set1_epi8(0b1111);
    let masked = and_si512(data, mask);
    unsafe { pack_u4_registers(out.add(0), masked) };

    // 4bit * pack_n / 8-bits per byte.
    let offset = pack_n.div_ceil(2);
    let remaining = srli_epi16::<4, 2>(data);
    unsafe { pack_u3_registers(out.add(offset), remaining, pack_n) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 8-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u8(out: &mut [u8; X128_MAX_OUTPUT_LEN], block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
    let out = out.as_mut_ptr();
    let data = load_u32x128(block);
    let mask = _mm512_set1_epi32(0xFF);
    let masked = and_si512(data, mask);
    let packed = pack_u32_u8_x8(masked);
    unsafe { store_si512x2(out, packed) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 9-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u9(out: &mut [u8; X128_MAX_OUTPUT_LEN], block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
    let out = out.as_mut_ptr();
    let data = load_u32x128(block);
    let mask = _mm512_set1_epi32(0x1FF);
    let masked = and_si512(data, mask);

    let (hi, lo) = pack_u32_u16_split_x8(masked);
    unsafe { store_si512x2(out, lo) };

    let offset = pack_n;
    unsafe { pack_u1_registers(out.add(offset), hi) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 10-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u10(out: &mut [u8; X128_MAX_OUTPUT_LEN], block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
    let out = out.as_mut_ptr();
    let data = load_u32x128(block);
    let mask = _mm512_set1_epi32(0x3FF);
    let masked = and_si512(data, mask);

    let (hi, lo) = pack_u32_u16_split_x8(masked);
    unsafe { store_si512x2(out, lo) };

    let offset = pack_n;
    unsafe { pack_u2_registers(out.add(offset), hi, pack_n) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 11-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u11(out: &mut [u8; X128_MAX_OUTPUT_LEN], block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
    let out = out.as_mut_ptr();
    let data = load_u32x128(block);
    let mask = _mm512_set1_epi32(0x7FF);
    let masked = and_si512(data, mask);

    let (hi, lo) = pack_u32_u16_split_x8(masked);
    unsafe { store_si512x2(out, lo) };

    let offset = pack_n;
    unsafe { pack_u3_registers(out.add(offset), hi, pack_n) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 12-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u12(out: &mut [u8; X128_MAX_OUTPUT_LEN], block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
    let out = out.as_mut_ptr();
    let data = load_u32x128(block);
    let mask = _mm512_set1_epi32(0xFFF);
    let masked = and_si512(data, mask);

    let (hi, lo) = pack_u32_u16_split_x8(masked);
    unsafe { store_si512x2(out, lo) };

    let offset = pack_n;
    unsafe { pack_u4_registers(out.add(offset), hi) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 13-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u13(out: &mut [u8; X128_MAX_OUTPUT_LEN], block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
    let out = out.as_mut_ptr();
    let data = load_u32x128(block);
    let mask = _mm512_set1_epi32(0x1FFF);
    let masked = and_si512(data, mask);

    let (hi, lo) = pack_u32_u16_split_x8(masked);
    unsafe { store_si512x2(out, lo) };

    let offset = pack_n;
    unsafe { pack_u5_registers(out.add(offset), hi, pack_n) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 14-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u14(out: &mut [u8; X128_MAX_OUTPUT_LEN], block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
    let out = out.as_mut_ptr();
    let data = load_u32x128(block);
    let mask = _mm512_set1_epi32(0x3FFF);
    let masked = and_si512(data, mask);

    let (hi, lo) = pack_u32_u16_split_x8(masked);
    unsafe { store_si512x2(out, lo) };

    let offset = pack_n;
    unsafe { pack_u6_registers(out.add(offset), hi, pack_n) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 15-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u15(out: &mut [u8; X128_MAX_OUTPUT_LEN], block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
    let out = out.as_mut_ptr();
    let data = load_u32x128(block);
    let mask = _mm512_set1_epi32(0x7FFF);
    let masked = and_si512(data, mask);

    let (hi, lo) = pack_u32_u16_split_x8(masked);
    unsafe { store_si512x2(out, lo) };

    let offset = pack_n;
    unsafe { pack_u7_registers(out.add(offset), hi, pack_n) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 16-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u16(out: &mut [u8; X128_MAX_OUTPUT_LEN], block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
    let out = out.as_mut_ptr();
    let data = load_u32x128(block);
    let mask = _mm512_set1_epi32(0xFFFF);
    let masked = and_si512(data, mask);
    let packed = pack_u32_u16_x8(masked);
    unsafe { store_si512x4(out, packed) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
unsafe fn store_lo_u16_registers(out: *mut u8, data: [__m512i; 8]) {
    let mask = _mm512_set1_epi32(0xFFFF);
    let shifted = and_si512(data, mask);
    let packed = pack_u32_u16_x8(shifted);
    unsafe { store_si512x4(out, packed) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 17-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u17(out: &mut [u8; X128_MAX_OUTPUT_LEN], block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
    let out = out.as_mut_ptr();

    let data = load_u32x128(block);
    let mask = _mm512_set1_epi32(0x1FFFF);
    let masked = and_si512(data, mask);

    unsafe { store_lo_u16_registers(out.add(0), masked) };

    let offset = pack_n * 2;
    let hi_bits = srli_epi32::<16, 8>(masked);
    let hi = pack_u32_u8_x8(hi_bits);
    unsafe { pack_u1_registers(out.add(offset), hi) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 18-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u18(out: &mut [u8; X128_MAX_OUTPUT_LEN], block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
    let out = out.as_mut_ptr();

    let data = load_u32x128(block);
    let mask = _mm512_set1_epi32(0x3FFFF);
    let masked = and_si512(data, mask);

    unsafe { store_lo_u16_registers(out.add(0), masked) };

    let offset = pack_n * 2;
    let hi_bits = srli_epi32::<16, 8>(masked);
    let hi = pack_u32_u8_x8(hi_bits);
    unsafe { pack_u2_registers(out.add(offset), hi, pack_n) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 19-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u19(out: &mut [u8; X128_MAX_OUTPUT_LEN], block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
    let out = out.as_mut_ptr();

    let data = load_u32x128(block);
    let mask = _mm512_set1_epi32(0x7FFFF);
    let masked = and_si512(data, mask);

    unsafe { store_lo_u16_registers(out.add(0), masked) };

    let offset = pack_n * 2;
    let hi_bits = srli_epi32::<16, 8>(masked);
    let hi = pack_u32_u8_x8(hi_bits);
    unsafe { pack_u3_registers(out.add(offset), hi, pack_n) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 20-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u20(out: &mut [u8; X128_MAX_OUTPUT_LEN], block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
    let out = out.as_mut_ptr();

    let data = load_u32x128(block);
    let mask = _mm512_set1_epi32(0xFFFFF);
    let masked = and_si512(data, mask);

    unsafe { store_lo_u16_registers(out.add(0), masked) };

    let offset = pack_n * 2;
    let hi_bits = srli_epi32::<16, 8>(masked);
    let hi = pack_u32_u8_x8(hi_bits);
    unsafe { pack_u4_registers(out.add(offset), hi) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 21-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u21(out: &mut [u8; X128_MAX_OUTPUT_LEN], block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
    let out = out.as_mut_ptr();

    let data = load_u32x128(block);
    let mask = _mm512_set1_epi32(0x1FFFFF);
    let masked = and_si512(data, mask);

    unsafe { store_lo_u16_registers(out.add(0), masked) };

    let offset = pack_n * 2;
    let hi_bits = srli_epi32::<16, 8>(masked);
    let hi = pack_u32_u8_x8(hi_bits);
    unsafe { pack_u5_registers(out.add(offset), hi, pack_n) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 22-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u22(out: &mut [u8; X128_MAX_OUTPUT_LEN], block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
    let out = out.as_mut_ptr();

    let data = load_u32x128(block);
    let mask = _mm512_set1_epi32(0x3FFFFF);
    let masked = and_si512(data, mask);

    unsafe { store_lo_u16_registers(out.add(0), masked) };

    let offset = pack_n * 2;
    let hi_bits = srli_epi32::<16, 8>(masked);
    let hi = pack_u32_u8_x8(hi_bits);
    unsafe { pack_u6_registers(out.add(offset), hi, pack_n) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 23-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u23(out: &mut [u8; X128_MAX_OUTPUT_LEN], block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
    let out = out.as_mut_ptr();

    let data = load_u32x128(block);
    let mask = _mm512_set1_epi32(0x7FFFFF);
    let masked = and_si512(data, mask);

    unsafe { store_lo_u16_registers(out.add(0), masked) };

    let offset = pack_n * 2;
    let hi_bits = srli_epi32::<16, 8>(masked);
    let hi = pack_u32_u8_x8(hi_bits);
    unsafe { pack_u7_registers(out.add(offset), hi, pack_n) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 24-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u24(out: &mut [u8; X128_MAX_OUTPUT_LEN], block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
    let out = out.as_mut_ptr();
    let data = load_u32x128(block);
    let mask = _mm512_set1_epi32(0xFFFFFF);
    let masked = and_si512(data, mask);
    unsafe { pack_u24_registers(out, masked, pack_n) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Pack eight registers containing 16 32-bit elements each into a 24-bit
/// compressed block and write to `out`.
unsafe fn pack_u24_registers(out: *mut u8, data: [__m512i; 8], pack_n: usize) {
    unsafe { store_lo_u16_registers(out.add(0), data) };

    let offset = pack_n * 2;
    let hi_bits = srli_epi32::<16, 8>(data);
    let packed_hi_8bits = pack_u32_u8_x8(hi_bits);
    unsafe { store_si512x2(out.add(offset), packed_hi_8bits) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 25-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u25(out: &mut [u8; X128_MAX_OUTPUT_LEN], block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
    let out = out.as_mut_ptr();

    let data = load_u32x128(block);
    let mask = _mm512_set1_epi32(0x1FFFFFF);
    let masked = and_si512(data, mask);

    let mut offset = 0;

    let lo_16bits = and_si512(masked, _mm512_set1_epi32(0xFFFF));
    let packed_lo_16bits = pack_u32_u16_x8(lo_16bits);
    unsafe { store_si512x4(out.add(offset), packed_lo_16bits) };

    let remaining_bits = srli_epi32::<16, 8>(masked);
    let (hi, lo) = pack_u32_u16_split_x8(remaining_bits);

    offset += pack_n * 2;
    unsafe { store_si512x2(out.add(offset), lo) };

    offset += pack_n;
    unsafe { pack_u1_registers(out.add(offset), hi) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 26-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u26(out: &mut [u8; X128_MAX_OUTPUT_LEN], block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
    let out = out.as_mut_ptr();

    let data = load_u32x128(block);
    let mask = _mm512_set1_epi32(0x3FFFFFF);
    let masked = and_si512(data, mask);

    let mut offset = 0;

    let lo_16bits = and_si512(masked, _mm512_set1_epi32(0xFFFF));
    let packed_lo_16bits = pack_u32_u16_x8(lo_16bits);
    unsafe { store_si512x4(out.add(offset), packed_lo_16bits) };

    let remaining_bits = srli_epi32::<16, 8>(masked);
    let (hi, lo) = pack_u32_u16_split_x8(remaining_bits);

    offset += pack_n * 2;
    unsafe { store_si512x2(out.add(offset), lo) };

    offset += pack_n;
    unsafe { pack_u2_registers(out.add(offset), hi, pack_n) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 27-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u27(out: &mut [u8; X128_MAX_OUTPUT_LEN], block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
    let out = out.as_mut_ptr();

    let data = load_u32x128(block);
    let mask = _mm512_set1_epi32(0x7FFFFFF);
    let masked = and_si512(data, mask);

    let mut offset = 0;

    let lo_16bits = and_si512(masked, _mm512_set1_epi32(0xFFFF));
    let packed_lo_16bits = pack_u32_u16_x8(lo_16bits);
    unsafe { store_si512x4(out.add(offset), packed_lo_16bits) };

    let remaining_bits = srli_epi32::<16, 8>(masked);
    let (hi, lo) = pack_u32_u16_split_x8(remaining_bits);

    offset += pack_n * 2;
    unsafe { store_si512x2(out.add(offset), lo) };

    offset += pack_n;
    unsafe { pack_u3_registers(out.add(offset), hi, pack_n) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 28-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u28(out: &mut [u8; X128_MAX_OUTPUT_LEN], block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
    let out = out.as_mut_ptr();

    let data = load_u32x128(block);
    let mask = _mm512_set1_epi32(0xFFFFFFF);
    let masked = and_si512(data, mask);

    let mut offset = 0;

    let lo_16bits = and_si512(masked, _mm512_set1_epi32(0xFFFF));
    let packed_lo_16bits = pack_u32_u16_x8(lo_16bits);
    unsafe { store_si512x4(out.add(offset), packed_lo_16bits) };

    let remaining_bits = srli_epi32::<16, 8>(masked);
    let (hi, lo) = pack_u32_u16_split_x8(remaining_bits);

    offset += pack_n * 2;
    unsafe { store_si512x2(out.add(offset), lo) };

    offset += pack_n;
    unsafe { pack_u4_registers(out.add(offset), hi) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 29-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u29(out: &mut [u8; X128_MAX_OUTPUT_LEN], block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
    let out = out.as_mut_ptr();

    let data = load_u32x128(block);
    let mask = _mm512_set1_epi32(0x1FFFFFFF);
    let masked = and_si512(data, mask);

    let mut offset = 0;

    let lo_16bits = and_si512(masked, _mm512_set1_epi32(0xFFFF));
    let packed_lo_16bits = pack_u32_u16_x8(lo_16bits);
    unsafe { store_si512x4(out.add(offset), packed_lo_16bits) };

    let remaining_bits = srli_epi32::<16, 8>(masked);
    let (hi, lo) = pack_u32_u16_split_x8(remaining_bits);

    offset += pack_n * 2;
    unsafe { store_si512x2(out.add(offset), lo) };

    offset += pack_n;
    unsafe { pack_u5_registers(out.add(offset), hi, pack_n) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 30-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u30(out: &mut [u8; X128_MAX_OUTPUT_LEN], block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
    let out = out.as_mut_ptr();

    let data = load_u32x128(block);
    let mask = _mm512_set1_epi32(0x3FFFFFFF);
    let masked = and_si512(data, mask);

    let mut offset = 0;

    let lo_16bits = and_si512(masked, _mm512_set1_epi32(0xFFFF));
    let packed_lo_16bits = pack_u32_u16_x8(lo_16bits);
    unsafe { store_si512x4(out.add(offset), packed_lo_16bits) };

    let remaining_bits = srli_epi32::<16, 8>(masked);
    let (hi, lo) = pack_u32_u16_split_x8(remaining_bits);

    offset += pack_n * 2;
    unsafe { store_si512x2(out.add(offset), lo) };

    offset += pack_n;
    unsafe { pack_u6_registers(out.add(offset), hi, pack_n) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 31-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u31(out: &mut [u8; X128_MAX_OUTPUT_LEN], block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
    let out = out.as_mut_ptr();

    let data = load_u32x128(block);
    let mask = _mm512_set1_epi32(0x7FFFFFFF);
    let masked = and_si512(data, mask);

    let mut offset = 0;

    let lo_16bits = and_si512(masked, _mm512_set1_epi32(0xFFFF));
    let packed_lo_16bits = pack_u32_u16_x8(lo_16bits);
    unsafe { store_si512x4(out.add(offset), packed_lo_16bits) };

    let remaining_bits = srli_epi32::<16, 8>(masked);
    let (hi, lo) = pack_u32_u16_split_x8(remaining_bits);

    offset += pack_n * 2;
    unsafe { store_si512x2(out.add(offset), lo) };

    offset += pack_n;
    unsafe { pack_u7_registers(out.add(offset), hi, pack_n) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 32-bit elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The provided `pack_n` must also be between `0..=128`.
pub unsafe fn to_u32(out: &mut [u8; X128_MAX_OUTPUT_LEN], block: &[u32; X128], pack_n: usize) {
    debug_assert!(pack_n <= 128, "pack_n must be less than or equal to 128");
    let data = load_u32x128(block);
    unsafe { store_si512x8(out.as_mut_ptr(), data) };
}

#[cfg(test)]
mod tests {
    use crate::uint32::max_compressed_size;
    use super::*;

    #[test]
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_to_u1() {
        let mut data = [0; X128];
        #[allow(clippy::needless_range_loop)]
        for i in 0..X128 {
            data[i] = (i % 2) as u32;
        }

        let mut out = [0; X128_MAX_OUTPUT_LEN];
        unsafe { to_u1(&mut out, &data, 128) };
        assert_eq!(out[..16], [170; 16]);

        let mut data = [0; X128];
        data[0] = 1;
        data[1] = 1;
        data[2] = 1;
        data[3] = 1;
        data[4] = 1;
        data[8] = 1;
        data[9] = 1;

        let mut out = [0; X128_MAX_OUTPUT_LEN];
        unsafe { to_u1(&mut out, &data, 10) };
        assert_eq!(out[..2], [31, 3]);
        assert_eq!(out[2..][..14], [0; 14]);
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_to_u2() {
        let mut data = [0; X128];
        #[allow(clippy::needless_range_loop)]
        for i in 0..X128 {
            data[i] = (i % 3) as u32;
        }

        let mut out = [0; X128_MAX_OUTPUT_LEN];
        unsafe { to_u2(&mut out, &data, 128) };
        assert_eq!(
            out[..32],
            [
                146, 36, 73, 146, 36, 73, 146, 36, 36, 73, 146, 36, 73, 146, 36, 73, 73, 146, 36,
                73, 146, 36, 73, 146, 146, 36, 73, 146, 36, 73, 146, 36
            ]
        );

        let mut data = [0; X128];
        data[0] = 1;
        data[1] = 1;
        data[2] = 1;
        data[3] = 1;
        data[4] = 1;
        data[8] = 1;
        data[9] = 2;

        let mut out = [0; X128_MAX_OUTPUT_LEN];
        unsafe { to_u2(&mut out, &data, 10) };

        // Imagine the data is in bits:
        //
        // lo_bits: 0b00000001_00011111 -> [31, 1, 0, 0] LE
        // hi_bits: 0b00000010_00000000 -> [0, 2, 0, 0] LE
        assert_eq!(out[..4], [31, 1, 0, 2]);
        assert_eq!(out[4..][..28], [0; 28]);
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_to_u3() {
        let mut data = [0; X128];
        #[allow(clippy::needless_range_loop)]
        for i in 0..X128 {
            data[i] = (i % 4) as u32;
        }

        let mut out = [0; X128_MAX_OUTPUT_LEN];
        unsafe { to_u3(&mut out, &data, 128) };
        assert_eq!(
            out[..24],
            [
                170, 170, 170, 170, 170, 170, 170, 170, 204, 204, 204, 204, 204, 204, 204, 204, 0,
                0, 0, 0, 0, 0, 0, 0
            ]
        );
        assert_eq!(
            out[24..][..24],
            [
                170, 170, 170, 170, 170, 170, 170, 170, 204, 204, 204, 204, 204, 204, 204, 204, 0,
                0, 0, 0, 0, 0, 0, 0
            ]
        );

        let mut data = [0; X128];
        data[0] = 1;
        data[1] = 1;
        data[2] = 1;
        data[3] = 1;
        data[4] = 1;
        data[8] = 5;
        data[9] = 2;

        let mut out = [0; X128_MAX_OUTPUT_LEN];
        unsafe { to_u3(&mut out, &data, 10) };

        // Imagine the data is in bits:
        //
        // b0_bits: 0b00000001_00011111 -> [31, 1, 0, 0] LE
        // b1_bits: 0b00000010_00000000 -> [0, 2, 0, 0] LE
        // b2_bits: 0b00000001_00000000 -> [0, 1, 0, 0] LE
        assert_eq!(out[..6], [31, 1, 0, 2, 0, 1]);
        assert_eq!(out[6..][..42], [0; 42]);
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_to_u4() {
        let mut data = [0; X128];
        #[allow(clippy::needless_range_loop)]
        for i in 0..X128 {
            data[i] = (i % 16) as u32;
        }

        let mut out = [0; X128_MAX_OUTPUT_LEN];
        unsafe { to_u4(&mut out, &data, 128) };
        assert_eq!(out[..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[8..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[16..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[24..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[32..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[40..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[48..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[56..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);

        let mut data = [0; X128];
        data[0] = 1;
        data[1] = 15;
        data[2] = 2;
        data[3] = 15;
        data[4] = 1;
        data[8] = 5;
        data[9] = 2;
        data[14] = 15;

        let mut out = [0; X128_MAX_OUTPUT_LEN];
        unsafe { to_u4(&mut out, &data, 15) };
        assert_eq!(out[0], 0b1111_0001); // [1, 15]
        assert_eq!(out[1], 0b1111_0010); // [2, 15]
        assert_eq!(out[2], 0b0000_0001); // [1, 0]
        assert_eq!(out[4], 0b0010_0101); // [5, 2]
        assert_eq!(out[7], 0b0000_1111); // [15, 0]
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_to_u5() {
        let mut data = [0; X128];
        #[allow(clippy::needless_range_loop)]
        for i in 0..X128 {
            data[i] = (i % 32) as u32;
        }

        let mut out = [0; X128_MAX_OUTPUT_LEN];
        unsafe { to_u5(&mut out, &data, 128) };
        assert_eq!(out[..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[8..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[16..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[24..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[32..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[40..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[48..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[56..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[64..][..8], [0, 0, 255, 255, 0, 0, 255, 255]);
        assert_eq!(out[72..][..8], [0, 0, 255, 255, 0, 0, 255, 255]);

        let mut data = [0; X128];
        data[0] = 1;
        data[1] = 15;
        data[2] = 2;
        data[3] = 15;
        data[4] = 1;
        data[8] = 5;
        data[9] = 2;
        data[14] = 17;

        let mut out = [0; X128_MAX_OUTPUT_LEN];
        unsafe { to_u5(&mut out, &data, 15) };
        assert_eq!(out[0], 0b1111_0001); // [1, 15]
        assert_eq!(out[1], 0b1111_0010); // [2, 15]
        assert_eq!(out[2], 0b0000_0001); // [1, 0]
        assert_eq!(out[4], 0b0010_0101); // [5, 2]
        assert_eq!(out[7], 0b0000_0001); // [17, 0]  -- well, the lower 4 bits
        assert_eq!(out[8], 0b0000_0000);
        assert_eq!(out[9], 0b0100_0000); // upper bits from that 17
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_to_u6() {
        let mut data = [0; X128];
        #[allow(clippy::needless_range_loop)]
        for i in 0..X128 {
            data[i] = (i % 64) as u32;
        }

        let mut out = [0; X128_MAX_OUTPUT_LEN];
        unsafe { to_u6(&mut out, &data, 128) };
        assert_eq!(out[..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[8..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[16..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[24..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[32..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[40..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[48..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[56..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[64..][..8], [0, 0, 255, 255, 0, 0, 255, 255]);
        assert_eq!(out[72..][..8], [0, 0, 0, 0, 255, 255, 255, 255]);
        assert_eq!(out[80..][..8], [0, 0, 255, 255, 0, 0, 255, 255]);
        assert_eq!(out[88..][..8], [0, 0, 0, 0, 255, 255, 255, 255]);

        let mut data = [0; X128];
        data[0] = 1;
        data[1] = 15;
        data[2] = 2;
        data[3] = 15;
        data[4] = 1;
        data[8] = 5;
        data[9] = 2;
        data[14] = 59;

        let mut out = [0; X128_MAX_OUTPUT_LEN];
        unsafe { to_u6(&mut out, &data, 15) };
        assert_eq!(out[0], 0b1111_0001); // [1, 15]
        assert_eq!(out[1], 0b1111_0010); // [2, 15]
        assert_eq!(out[2], 0b0000_0001); // [1, 0]
        assert_eq!(out[4], 0b0010_0101); // [5, 2]
        assert_eq!(out[7], 0b0000_1011); // [59, 0]  -- well, the lower 4 bits
        assert_eq!(out[8], 0b0000_0000);
        assert_eq!(out[9], 0b0100_0000); // upper 1 bit from that 59
        assert_eq!(out[10], 0b0000_0000);
        assert_eq!(out[11], 0b0100_0000); // upper 1 bit from that 59
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_to_u7() {
        let mut data = [0; X128];
        #[allow(clippy::needless_range_loop)]
        for i in 0..X128 {
            data[i] = (i % 64) as u32;
        }

        let mut out = [0; X128_MAX_OUTPUT_LEN];
        unsafe { to_u7(&mut out, &data, 128) };
        assert_eq!(out[..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[8..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[16..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[24..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[32..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[40..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[48..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[56..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[64..][..8], [0, 0, 255, 255, 0, 0, 255, 255]);
        assert_eq!(out[72..][..8], [0, 0, 0, 0, 255, 255, 255, 255]);
        assert_eq!(out[80..][..8], [0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(out[88..][..8], [0, 0, 255, 255, 0, 0, 255, 255]);
        assert_eq!(out[96..][..8], [0, 0, 0, 0, 255, 255, 255, 255]);
        assert_eq!(out[104..][..8], [0, 0, 0, 0, 0, 0, 0, 0]);

        let mut data = [0; X128];
        data[0] = 1;
        data[1] = 15;
        data[2] = 2;
        data[3] = 15;
        data[4] = 1;
        data[8] = 5;
        data[9] = 2;
        data[14] = 127;

        let mut out = [0; X128_MAX_OUTPUT_LEN];
        unsafe { to_u7(&mut out, &data, 15) };
        assert_eq!(out[0], 0b1111_0001); // [1, 15]
        assert_eq!(out[1], 0b1111_0010); // [2, 15]
        assert_eq!(out[2], 0b0000_0001); // [1, 0]
        assert_eq!(out[4], 0b0010_0101); // [5, 2]
        assert_eq!(out[7], 0b0000_1111); // [127, 0]  -- well, the lower 4 bits
        assert_eq!(out[8], 0b0000_0000);
        assert_eq!(out[9], 0b0100_0000); // upper 1 bit from that 127
        assert_eq!(out[10], 0b0000_0000);
        assert_eq!(out[11], 0b0100_0000); // upper 1 bit from that 127
        assert_eq!(out[12], 0b0000_0000);
        assert_eq!(out[13], 0b0100_0000); // upper 1 bit from that 127
    }

    #[rstest::rstest]
    #[case(1, to_u1)]
    #[case(2, to_u2)]
    #[case(3, to_u3)]
    #[case(4, to_u4)]
    #[case(5, to_u5)]
    #[case(6, to_u6)]
    #[case(7, to_u7)]
    #[case(8, to_u8)]
    #[case(9, to_u9)]
    #[case(10, to_u10)]
    #[case(11, to_u11)]
    #[case(12, to_u12)]
    #[case(13, to_u13)]
    #[case(14, to_u14)]
    #[case(15, to_u15)]
    #[case(16, to_u16)]
    #[case(17, to_u17)]
    #[case(18, to_u18)]
    #[case(19, to_u19)]
    #[case(20, to_u20)]
    #[case(21, to_u21)]
    #[case(22, to_u22)]
    #[case(23, to_u23)]
    #[case(24, to_u24)]
    #[case(25, to_u25)]
    #[case(26, to_u26)]
    #[case(27, to_u27)]
    #[case(28, to_u28)]
    #[case(29, to_u29)]
    #[case(30, to_u30)]
    #[case(31, to_u31)]
    #[case(32, to_u32)]
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_saturation(
        #[case] bit_len: u8,
        #[case] packer: unsafe fn(&mut [u8; X128_MAX_OUTPUT_LEN], &[u32; X128], usize),
    ) {
        let pack_value = (2u64.pow(bit_len as u32) - 1) as u32;

        let values = [pack_value; X128];
        let mut output = [0; X128_MAX_OUTPUT_LEN];
        unsafe { packer(&mut output, &values, X128) };
        assert!(output[..max_compressed_size::<X128>(bit_len as usize)].iter().all(|b| *b == u8::MAX));
    }
}
