use std::arch::x86_64::*;
use std::cmp;

use super::data::*;
use super::utils::*;
use crate::X128;

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the provided block of integers from `nbits` sized elements.
///
/// # Safety
/// - The CPU features required must be met.
/// - The `input` buffer must be at least [max_compressed_size(nbits)](crate::uint32::max_compressed_size)
///   in size to safely read.
/// - `read_n` must also be between `0..=128`.
/// - `nbits` being between `1..=32`.
pub unsafe fn from_nbits(input: *const u8, nbits: u8, block: &mut [u32; X128], read_n: usize) {
    const LUT: [unsafe fn(*const u8, &mut [u32; X128], usize); 33] = [
        from_u0, from_u1, from_u2, from_u3, from_u4, from_u5, from_u6, from_u7, from_u8, from_u9,
        from_u10, from_u11, from_u12, from_u13, from_u14, from_u15, from_u16, from_u17, from_u18,
        from_u19, from_u20, from_u21, from_u22, from_u23, from_u24, from_u25, from_u26, from_u27,
        from_u28, from_u29, from_u30, from_u31, from_u32,
    ];
    let func = unsafe { LUT.get_unchecked(nbits as usize) };
    unsafe { func(input, block, read_n) };
}

#[target_feature(enable = "avx512f")]
/// Fill the provided block with `0` values.
///
/// # Safety
/// - The CPU features required must be met.
/// - `read_n` must also be between `0..=128`.
pub unsafe fn from_u0(_input: *const u8, block: &mut [u32; X128], _read_n: usize) {
    let data = [_mm512_setzero_si512(); 8];
    store_u32x128(block, data);
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the packed block of 1-bit elements and write the resulting elements
/// to the `block`.
///
/// # Safety
/// - The CPU features required must be met.
/// - `read_n` must also be between `0..=128`.
pub unsafe fn from_u1(input: *const u8, block: &mut [u32; X128], read_n: usize) {
    debug_assert!(read_n <= 128, "read_n must be less than or equal to 128");

    let data = unsafe { unpack_u1_registers::<1>(input) };
    store_u32x128(block, data);
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack eight registers containing 16 32-bit elements from a 1-bit bitmap provided
/// by `input`.
unsafe fn unpack_u1_registers<const FILL_VALUE: u32>(input: *const u8) -> [__m512i; 8] {
    let mask1: u64 = unsafe { std::ptr::read_unaligned(input.add(0).cast()) };
    let mask2: u64 = unsafe { std::ptr::read_unaligned(input.add(8).cast()) };

    let ones = _mm512_set1_epi32(FILL_VALUE as i32);

    let d1 = _mm512_maskz_mov_epi32(mask1 as __mmask16, ones);
    let d2 = _mm512_maskz_mov_epi32((mask1 >> 16) as __mmask16, ones);
    let d3 = _mm512_maskz_mov_epi32((mask1 >> 32) as __mmask16, ones);
    let d4 = _mm512_maskz_mov_epi32((mask1 >> 48) as __mmask16, ones);
    let d5 = _mm512_maskz_mov_epi32(mask2 as __mmask16, ones);
    let d6 = _mm512_maskz_mov_epi32((mask2 >> 16) as __mmask16, ones);
    let d7 = _mm512_maskz_mov_epi32((mask2 >> 32) as __mmask16, ones);
    let d8 = _mm512_maskz_mov_epi32((mask2 >> 48) as __mmask16, ones);

    [d1, d2, d3, d4, d5, d6, d7, d8]
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the packed block of 2-bit elements and write the resulting elements
/// to the `block`.
///
/// # Safety
/// - The CPU features required must be met.
/// - `read_n` must also be between `0..=128`.
pub unsafe fn from_u2(input: *const u8, block: &mut [u32; X128], read_n: usize) {
    debug_assert!(read_n <= 128, "read_n must be less than or equal to 128");

    let data = unsafe { unpack_u2_registers(input, read_n) };
    store_u32x128(block, data);
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack eight registers containing 16 32-bit elements from a 2-bit bitmap provided
/// by `input`.
unsafe fn unpack_u2_registers(input: *const u8, read_n: usize) -> [__m512i; 8] {
    let hi_offset = cmp::min(64, read_n).div_ceil(8);
    let lo_mask1: u64 = unsafe { std::ptr::read_unaligned(input.add(0).cast()) };
    let hi_mask1: u64 = unsafe { std::ptr::read_unaligned(input.add(hi_offset).cast()) };

    let hi_offset = read_n.saturating_sub(64).div_ceil(8);
    let lo_mask2: u64 = unsafe { std::ptr::read_unaligned(input.add(16).cast()) };
    let hi_mask2: u64 = unsafe { std::ptr::read_unaligned(input.add(16 + hi_offset).cast()) };

    let lo_bits = _mm512_set1_epi8(0b01);
    let hi_bits = _mm512_set1_epi8(0b10);

    let lo_8bit1 = _mm512_maskz_mov_epi8(lo_mask1, lo_bits);
    let hi_8bit1 = _mm512_maskz_mov_epi8(hi_mask1, hi_bits);

    let lo_8bit2 = _mm512_maskz_mov_epi8(lo_mask2, lo_bits);
    let hi_8bit2 = _mm512_maskz_mov_epi8(hi_mask2, hi_bits);

    let packed1 = _mm512_or_si512(hi_8bit1, lo_8bit1);
    let packed2 = _mm512_or_si512(hi_8bit2, lo_8bit2);

    unpack_u8_u32_x8([packed1, packed2])
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the packed block of 3-bit elements and write the resulting elements
/// to the `block`.
///
/// # Safety
/// - The CPU features required must be met.
/// - `read_n` must also be between `0..=128`.
pub unsafe fn from_u3(input: *const u8, block: &mut [u32; X128], read_n: usize) {
    debug_assert!(read_n <= 128, "read_n must be less than or equal to 128");

    let packed = unsafe { unpack_u3_registers::<0>(input, read_n) };
    let data = unpack_u8_u32_x8(packed);
    store_u32x128(block, data);
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack two registers containing 64 8-bit elements from a 3-bit bitmap provided
/// by `input`.
unsafe fn unpack_u3_registers<const SHIFT_BITS_LEFT: u32>(
    input: *const u8,
    read_n: usize,
) -> [__m512i; 2] {
    let step = cmp::min(64, read_n).div_ceil(8);
    let b0_mask1: u64 = unsafe { std::ptr::read_unaligned(input.add(0).cast()) };
    let b1_mask1: u64 = unsafe { std::ptr::read_unaligned(input.add(step).cast()) };
    let b2_mask1: u64 = unsafe { std::ptr::read_unaligned(input.add(step * 2).cast()) };

    let step = read_n.saturating_sub(64).div_ceil(8);
    let b0_mask2: u64 = unsafe { std::ptr::read_unaligned(input.add(24).cast()) };
    let b1_mask2: u64 = unsafe { std::ptr::read_unaligned(input.add(24 + step).cast()) };
    let b2_mask2: u64 = unsafe { std::ptr::read_unaligned(input.add(24 + step * 2).cast()) };

    let b0_bits = _mm512_set1_epi8(0b001 << SHIFT_BITS_LEFT);
    let b1_bits = _mm512_set1_epi8(0b010 << SHIFT_BITS_LEFT);
    let b2_bits = _mm512_set1_epi8(0b100 << SHIFT_BITS_LEFT);

    let b0_8bit1 = _mm512_maskz_mov_epi8(b0_mask1, b0_bits);
    let b1_8bit1 = _mm512_maskz_mov_epi8(b1_mask1, b1_bits);
    let b2_8bit1 = _mm512_maskz_mov_epi8(b2_mask1, b2_bits);

    let b0_8bit2 = _mm512_maskz_mov_epi8(b0_mask2, b0_bits);
    let b1_8bit2 = _mm512_maskz_mov_epi8(b1_mask2, b1_bits);
    let b2_8bit2 = _mm512_maskz_mov_epi8(b2_mask2, b2_bits);

    const IMM8: i32 = _MM_TERNLOG_A | _MM_TERNLOG_B | _MM_TERNLOG_C;
    let packed1 = _mm512_ternarylogic_epi32::<IMM8>(b2_8bit1, b1_8bit1, b0_8bit1);
    let packed2 = _mm512_ternarylogic_epi32::<IMM8>(b2_8bit2, b1_8bit2, b0_8bit2);

    [packed1, packed2]
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the packed block of 4-bit elements and write the resulting elements
/// to the `block`.
///
/// # Safety
/// - The CPU features required must be met.
/// - `read_n` must also be between `0..=128`.
pub unsafe fn from_u4(input: *const u8, block: &mut [u32; X128], read_n: usize) {
    debug_assert!(read_n <= 128, "read_n must be less than or equal to 128");

    let packed = unsafe { unpack_u4_registers(input) };
    let data = unpack_u8_u32_x8(packed);
    store_u32x128(block, data);
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack 2 registers containing 64 8-bit elements from a 4-bit packed block provided
/// by `input`.
unsafe fn unpack_u4_registers(input: *const u8) -> [__m512i; 2] {
    // packed: AAAA BBBB ...
    let nibbles1 = unsafe { _mm256_loadu_epi8(input.add(0).cast()) };
    let nibbles2 = unsafe { _mm256_loadu_epi8(input.add(32).cast()) };

    // expanded: 0000 0000 AAAA BBBB ...
    let expanded1 = _mm512_cvtepu8_epi16(nibbles1);
    let expanded2 = _mm512_cvtepu8_epi16(nibbles2);

    // shifted: 0000 AAAA BBBB 0000 ...
    let shifted1 = _mm512_slli_epi16::<4>(expanded1);
    let shifted2 = _mm512_slli_epi16::<4>(expanded2);

    // Select just the low 4 bits of each byte
    let mask = _mm512_set1_epi32(0x0F0F0F0F);

    // do mask & (expanded | shifted)
    const IMM8: i32 = _MM_TERNLOG_A & (_MM_TERNLOG_B | _MM_TERNLOG_C);
    let packed1 = _mm512_ternarylogic_epi32::<IMM8>(mask, expanded1, shifted1);
    let packed2 = _mm512_ternarylogic_epi32::<IMM8>(mask, expanded2, shifted2);

    [packed1, packed2]
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the packed block of 5-bit elements and write the resulting elements
/// to the `block`.
///
/// # Safety
/// - The CPU features required must be met.
/// - `read_n` must also be between `0..=128`.
pub unsafe fn from_u5(input: *const u8, block: &mut [u32; X128], read_n: usize) {
    debug_assert!(read_n <= 128, "read_n must be less than or equal to 128");

    let data = unsafe { unpack_u5_registers(input, read_n) };
    store_u32x128(block, data);
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack eight registers containing 16 32-bit elements from a 5-bit packed block provided
/// by `input`.
unsafe fn unpack_u5_registers(input: *const u8, read_n: usize) -> [__m512i; 8] {
    let packed_lo_4bit = unsafe { unpack_u4_registers(input.add(0)) };

    let offset = read_n.div_ceil(2);
    let mask1: u64 = unsafe { std::ptr::read_unaligned(input.add(offset).cast()) };
    let mask2: u64 = unsafe { std::ptr::read_unaligned(input.add(offset + 8).cast()) };

    let set_bit = _mm512_set1_epi8(0b0001_0000);
    let packed_hi_1bit = [
        _mm512_maskz_mov_epi8(mask1, set_bit),
        _mm512_maskz_mov_epi8(mask2, set_bit),
    ];

    let packed1 = _mm512_or_si512(packed_hi_1bit[0], packed_lo_4bit[0]);
    let packed2 = _mm512_or_si512(packed_hi_1bit[1], packed_lo_4bit[1]);

    unpack_u8_u32_x8([packed1, packed2])
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the packed block of 6-bit elements and write the resulting elements
/// to the `block`.
///
/// # Safety
/// - The CPU features required must be met.
/// - `read_n` must also be between `0..=128`.
pub unsafe fn from_u6(input: *const u8, block: &mut [u32; X128], read_n: usize) {
    debug_assert!(read_n <= 128, "read_n must be less than or equal to 128");

    let data = unsafe { unpack_u6_registers(input, read_n) };
    store_u32x128(block, data);
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack eight registers containing 16 32-bit elements from a 6-bit packed block provided
/// by `input`.
unsafe fn unpack_u6_registers(input: *const u8, read_n: usize) -> [__m512i; 8] {
    let packed_lo_4bit = unsafe { unpack_u4_registers(input.add(0)) };

    let offset = read_n.div_ceil(2);
    let hi_offset = cmp::min(64, read_n).div_ceil(8);
    let lo_mask1: u64 = unsafe { std::ptr::read_unaligned(input.add(offset).cast()) };
    let hi_mask1: u64 = unsafe { std::ptr::read_unaligned(input.add(offset + hi_offset).cast()) };

    let hi_offset = read_n.saturating_sub(64).div_ceil(8);
    let lo_mask2: u64 = unsafe { std::ptr::read_unaligned(input.add(offset + 16).cast()) };
    let hi_mask2: u64 =
        unsafe { std::ptr::read_unaligned(input.add(offset + 16 + hi_offset).cast()) };

    let lo_bits = _mm512_set1_epi8(0b01_0000);
    let hi_bits = _mm512_set1_epi8(0b10_0000);

    let lo_bit1 = _mm512_maskz_mov_epi8(lo_mask1, lo_bits);
    let hi_bit1 = _mm512_maskz_mov_epi8(hi_mask1, hi_bits);

    let lo_bit2 = _mm512_maskz_mov_epi8(lo_mask2, lo_bits);
    let hi_bit2 = _mm512_maskz_mov_epi8(hi_mask2, hi_bits);

    const IMM8: i32 = _MM_TERNLOG_A | _MM_TERNLOG_B | _MM_TERNLOG_C;
    let packed1 = _mm512_ternarylogic_epi32::<IMM8>(hi_bit1, lo_bit1, packed_lo_4bit[0]);
    let packed2 = _mm512_ternarylogic_epi32::<IMM8>(hi_bit2, lo_bit2, packed_lo_4bit[1]);

    unpack_u8_u32_x8([packed1, packed2])
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the packed block of 7-bit elements and write the resulting elements
/// to the `block`.
///
/// # Safety
/// - The CPU features required must be met.
/// - `read_n` must also be between `0..=128`.
pub unsafe fn from_u7(input: *const u8, block: &mut [u32; X128], read_n: usize) {
    debug_assert!(read_n <= 128, "read_n must be less than or equal to 128");

    let data = unsafe { unpack_u7_registers(input, read_n) };
    store_u32x128(block, data);
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack eight registers containing 16 32-bit elements from a 7-bit packed block provided
/// by `input`.
unsafe fn unpack_u7_registers(input: *const u8, read_n: usize) -> [__m512i; 8] {
    let packed_lo_4bit = unsafe { unpack_u4_registers(input.add(0)) };

    let offset = read_n.div_ceil(2);
    let packed_hi_3bit = unsafe { unpack_u3_registers::<4>(input.add(offset), read_n) };

    let packed1 = _mm512_or_si512(packed_hi_3bit[0], packed_lo_4bit[0]);
    let packed2 = _mm512_or_si512(packed_hi_3bit[1], packed_lo_4bit[1]);

    unpack_u8_u32_x8([packed1, packed2])
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the packed block of 8-bit elements and write the resulting elements
/// to the `block`.
///
/// # Safety
/// - The CPU features required must be met.
/// - `read_n` must also be between `0..=128`.
pub unsafe fn from_u8(input: *const u8, block: &mut [u32; X128], read_n: usize) {
    debug_assert!(read_n <= 128, "read_n must be less than or equal to 128");

    let packed = unsafe { load_si512x2(input) };
    let data = unpack_u8_u32_x8(packed);
    store_u32x128(block, data);
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the packed block of 9-bit elements and write the resulting elements
/// to the `block`.
///
/// # Safety
/// - The CPU features required must be met.
/// - `read_n` must also be between `0..=128`.
pub unsafe fn from_u9(input: *const u8, block: &mut [u32; X128], read_n: usize) {
    debug_assert!(read_n <= 128, "read_n must be less than or equal to 128");

    let packed = unsafe { load_si512x2(input) };
    let data_lo_bits = unpack_u8_u32_x8(packed);

    let offset = read_n;
    const FILL_VALUE: u32 = 0b0001_0000_0000; // Removes the need to shift separately.
    let data_hi_bits = unsafe { unpack_u1_registers::<FILL_VALUE>(input.add(offset)) };

    let data = or_si512_all(data_hi_bits, data_lo_bits);

    store_u32x128(block, data);
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the packed block of 10-bit elements and write the resulting elements
/// to the `block`.
///
/// # Safety
/// - The CPU features required must be met.
/// - `read_n` must also be between `0..=128`.
pub unsafe fn from_u10(input: *const u8, block: &mut [u32; X128], read_n: usize) {
    debug_assert!(read_n <= 128, "read_n must be less than or equal to 128");

    let packed = unsafe { load_si512x2(input) };
    let data_lo_bits = unpack_u8_u32_x8(packed);

    let offset = read_n;
    let data_hi_bits = unsafe { unpack_u2_registers(input.add(offset), read_n) };

    let shifted_hi_bits = slli_epi32::<8, 8>(data_hi_bits);
    let data = or_si512_all(shifted_hi_bits, data_lo_bits);

    store_u32x128(block, data);
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the packed block of 11-bit elements and write the resulting elements
/// to the `block`.
///
/// # Safety
/// - The CPU features required must be met.
/// - `read_n` must also be between `0..=128`.
pub unsafe fn from_u11(input: *const u8, block: &mut [u32; X128], read_n: usize) {
    debug_assert!(read_n <= 128, "read_n must be less than or equal to 128");

    let packed = unsafe { load_si512x2(input) };
    let data_lo_bits = unpack_u8_u32_x8(packed);

    let offset = read_n;
    let packed = unsafe { unpack_u3_registers::<0>(input.add(offset), read_n) };
    let data_hi_bits = unpack_u8_u32_x8(packed);

    let shifted_hi_bits = slli_epi32::<8, 8>(data_hi_bits);
    let data = or_si512_all(shifted_hi_bits, data_lo_bits);

    store_u32x128(block, data);
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the packed block of 12-bit elements and write the resulting elements
/// to the `block`.
///
/// # Safety
/// - The CPU features required must be met.
/// - `read_n` must also be between `0..=128`.
pub unsafe fn from_u12(input: *const u8, block: &mut [u32; X128], read_n: usize) {
    debug_assert!(read_n <= 128, "read_n must be less than or equal to 128");

    let packed = unsafe { load_si512x2(input) };
    let data_lo_bits = unpack_u8_u32_x8(packed);

    let offset = read_n;
    let packed = unsafe { unpack_u4_registers(input.add(offset)) };
    let data_hi_bits = unpack_u8_u32_x8(packed);

    let shifted_hi_bits = slli_epi32::<8, 8>(data_hi_bits);
    let data = or_si512_all(shifted_hi_bits, data_lo_bits);

    store_u32x128(block, data);
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the packed block of 13-bit elements and write the resulting elements
/// to the `block`.
///
/// # Safety
/// - The CPU features required must be met.
/// - `read_n` must also be between `0..=128`.
pub unsafe fn from_u13(input: *const u8, block: &mut [u32; X128], read_n: usize) {
    debug_assert!(read_n <= 128, "read_n must be less than or equal to 128");

    let packed = unsafe { load_si512x2(input) };
    let data_lo_bits = unpack_u8_u32_x8(packed);

    let offset = read_n;
    let data_hi_bits = unsafe { unpack_u5_registers(input.add(offset), read_n) };

    let shifted_hi_bits = slli_epi32::<8, 8>(data_hi_bits);
    let data = or_si512_all(shifted_hi_bits, data_lo_bits);

    store_u32x128(block, data);
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the packed block of 14-bit elements and write the resulting elements
/// to the `block`.
///
/// # Safety
/// - The CPU features required must be met.
/// - `read_n` must also be between `0..=128`.
pub unsafe fn from_u14(input: *const u8, block: &mut [u32; X128], read_n: usize) {
    debug_assert!(read_n <= 128, "read_n must be less than or equal to 128");

    let packed = unsafe { load_si512x2(input) };
    let data_lo_bits = unpack_u8_u32_x8(packed);

    let offset = read_n;
    let data_hi_bits = unsafe { unpack_u6_registers(input.add(offset), read_n) };

    let shifted_hi_bits = slli_epi32::<8, 8>(data_hi_bits);
    let data = or_si512_all(shifted_hi_bits, data_lo_bits);

    store_u32x128(block, data);
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the packed block of 15-bit elements and write the resulting elements
/// to the `block`.
///
/// # Safety
/// - The CPU features required must be met.
/// - `read_n` must also be between `0..=128`.
pub unsafe fn from_u15(input: *const u8, block: &mut [u32; X128], read_n: usize) {
    debug_assert!(read_n <= 128, "read_n must be less than or equal to 128");

    let packed = unsafe { load_si512x2(input) };
    let data_lo_bits = unpack_u8_u32_x8(packed);

    let offset = read_n;
    let data_hi_bits = unsafe { unpack_u7_registers(input.add(offset), read_n) };

    let shifted_hi_bits = slli_epi32::<8, 8>(data_hi_bits);
    let data = or_si512_all(shifted_hi_bits, data_lo_bits);

    store_u32x128(block, data);
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the packed block of 16-bit elements and write the resulting elements
/// to the `block`.
///
/// # Safety
/// - The CPU features required must be met.
/// - `read_n` must also be between `0..=128`.
pub unsafe fn from_u16(input: *const u8, block: &mut [u32; X128], read_n: usize) {
    debug_assert!(read_n <= 128, "read_n must be less than or equal to 128");

    let packed = unsafe { load_si512x4(input) };
    let data = unpack_u16_u32_x8(packed);
    store_u32x128(block, data);
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the packed block of 17-bit elements and write the resulting elements
/// to the `block`.
///
/// # Safety
/// - The CPU features required must be met.
/// - `read_n` must also be between `0..=128`.
pub unsafe fn from_u17(input: *const u8, block: &mut [u32; X128], read_n: usize) {
    debug_assert!(read_n <= 128, "read_n must be less than or equal to 128");

    let packed = unsafe { load_si512x4(input) };
    let data_lo_bits = unpack_u16_u32_x8(packed);

    let offset = read_n * 2;
    const FILL_VALUE: u32 = 0b00000001_00000000_00000000; // Removes the need to shift separately.
    let data_hi_bits = unsafe { unpack_u1_registers::<FILL_VALUE>(input.add(offset)) };

    let data = or_si512_all(data_hi_bits, data_lo_bits);

    store_u32x128(block, data);
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the packed block of 18-bit elements and write the resulting elements
/// to the `block`.
///
/// # Safety
/// - The CPU features required must be met.
/// - `read_n` must also be between `0..=128`.
pub unsafe fn from_u18(input: *const u8, block: &mut [u32; X128], read_n: usize) {
    debug_assert!(read_n <= 128, "read_n must be less than or equal to 128");

    let packed = unsafe { load_si512x4(input) };
    let data_lo_bits = unpack_u16_u32_x8(packed);

    let offset = read_n * 2;
    let data_hi_bits = unsafe { unpack_u2_registers(input.add(offset), read_n) };

    let shifted_hi_bits = slli_epi32::<16, 8>(data_hi_bits);
    let data = or_si512_all(shifted_hi_bits, data_lo_bits);

    store_u32x128(block, data);
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the packed block of 19-bit elements and write the resulting elements
/// to the `block`.
///
/// # Safety
/// - The CPU features required must be met.
/// - `read_n` must also be between `0..=128`.
pub unsafe fn from_u19(input: *const u8, block: &mut [u32; X128], read_n: usize) {
    debug_assert!(read_n <= 128, "read_n must be less than or equal to 128");

    let packed = unsafe { load_si512x4(input) };
    let data_lo_bits = unpack_u16_u32_x8(packed);

    let offset = read_n * 2;
    let packed = unsafe { unpack_u3_registers::<0>(input.add(offset), read_n) };
    let data_hi_bits = unpack_u8_u32_x8(packed);

    let shifted_hi_bits = slli_epi32::<16, 8>(data_hi_bits);
    let data = or_si512_all(shifted_hi_bits, data_lo_bits);

    store_u32x128(block, data);
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the packed block of 20-bit elements and write the resulting elements
/// to the `block`.
///
/// # Safety
/// - The CPU features required must be met.
/// - `read_n` must also be between `0..=128`.
pub unsafe fn from_u20(input: *const u8, block: &mut [u32; X128], read_n: usize) {
    debug_assert!(read_n <= 128, "read_n must be less than or equal to 128");

    let packed = unsafe { load_si512x4(input) };
    let data_lo_bits = unpack_u16_u32_x8(packed);

    let offset = read_n * 2;
    let packed = unsafe { unpack_u4_registers(input.add(offset)) };
    let data_hi_bits = unpack_u8_u32_x8(packed);

    let shifted_hi_bits = slli_epi32::<16, 8>(data_hi_bits);
    let data = or_si512_all(shifted_hi_bits, data_lo_bits);

    store_u32x128(block, data);
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the packed block of 21-bit elements and write the resulting elements
/// to the `block`.
///
/// # Safety
/// - The CPU features required must be met.
/// - `read_n` must also be between `0..=128`.
pub unsafe fn from_u21(input: *const u8, block: &mut [u32; X128], read_n: usize) {
    debug_assert!(read_n <= 128, "read_n must be less than or equal to 128");

    let packed = unsafe { load_si512x4(input) };
    let data_lo_bits = unpack_u16_u32_x8(packed);

    let offset = read_n * 2;
    let data_hi_bits = unsafe { unpack_u5_registers(input.add(offset), read_n) };

    let shifted_hi_bits = slli_epi32::<16, 8>(data_hi_bits);
    let data = or_si512_all(shifted_hi_bits, data_lo_bits);

    store_u32x128(block, data);
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the packed block of 22-bit elements and write the resulting elements
/// to the `block`.
///
/// # Safety
/// - The CPU features required must be met.
/// - `read_n` must also be between `0..=128`.
pub unsafe fn from_u22(input: *const u8, block: &mut [u32; X128], read_n: usize) {
    debug_assert!(read_n <= 128, "read_n must be less than or equal to 128");

    let packed = unsafe { load_si512x4(input) };
    let data_lo_bits = unpack_u16_u32_x8(packed);

    let offset = read_n * 2;
    let data_hi_bits = unsafe { unpack_u6_registers(input.add(offset), read_n) };

    let shifted_hi_bits = slli_epi32::<16, 8>(data_hi_bits);
    let data = or_si512_all(shifted_hi_bits, data_lo_bits);

    store_u32x128(block, data);
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the packed block of 23-bit elements and write the resulting elements
/// to the `block`.
///
/// # Safety
/// - The CPU features required must be met.
/// - `read_n` must also be between `0..=128`.
pub unsafe fn from_u23(input: *const u8, block: &mut [u32; X128], read_n: usize) {
    debug_assert!(read_n <= 128, "read_n must be less than or equal to 128");

    let packed = unsafe { load_si512x4(input) };
    let data_lo_bits = unpack_u16_u32_x8(packed);

    let offset = read_n * 2;
    let data_hi_bits = unsafe { unpack_u7_registers(input.add(offset), read_n) };

    let shifted_hi_bits = slli_epi32::<16, 8>(data_hi_bits);
    let data = or_si512_all(shifted_hi_bits, data_lo_bits);

    store_u32x128(block, data);
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the packed block of 24-bit elements and write the resulting elements
/// to the `block`.
///
/// # Safety
/// - The CPU features required must be met.
/// - `read_n` must also be between `0..=128`.
pub unsafe fn from_u24(input: *const u8, block: &mut [u32; X128], read_n: usize) {
    debug_assert!(read_n <= 128, "read_n must be less than or equal to 128");

    let packed = unsafe { load_si512x4(input) };
    let data_lo_bits = unpack_u16_u32_x8(packed);

    let offset = read_n * 2;
    let packed = unsafe { load_si512x2(input.add(offset)) };
    let data_hi_bits = unpack_u8_u32_x8(packed);

    let shifted_hi_bits = slli_epi32::<16, 8>(data_hi_bits);
    let data = or_si512_all(shifted_hi_bits, data_lo_bits);

    store_u32x128(block, data);
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the packed block of 25-bit elements and write the resulting elements
/// to the `block`.
///
/// # Safety
/// - The CPU features required must be met.
/// - `read_n` must also be between `0..=128`.
pub unsafe fn from_u25(input: *const u8, block: &mut [u32; X128], read_n: usize) {
    debug_assert!(read_n <= 128, "read_n must be less than or equal to 128");

    let mut offset = 0;

    let packed = unsafe { load_si512x4(input.add(offset)) };
    let b0_bits = unpack_u16_u32_x8(packed);

    offset += read_n * 2;
    let packed = unsafe { load_si512x2(input.add(offset)) };
    let mut b1_bits = unpack_u8_u32_x8(packed);
    b1_bits = slli_epi32::<16, 8>(b1_bits);

    const FILL_VALUE: u32 = 1 << 24; // Removes the need to shift separately.
    offset += read_n;
    let b2_bits = unsafe { unpack_u1_registers::<FILL_VALUE>(input.add(offset)) };

    const IMM8: i32 = _MM_TERNLOG_A | _MM_TERNLOG_B | _MM_TERNLOG_C;
    let data = ternary_epi32_all::<IMM8, 8>(b2_bits, b1_bits, b0_bits);

    store_u32x128(block, data);
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the packed block of 26-bit elements and write the resulting elements
/// to the `block`.
///
/// # Safety
/// - The CPU features required must be met.
/// - `read_n` must also be between `0..=128`.
pub unsafe fn from_u26(input: *const u8, block: &mut [u32; X128], read_n: usize) {
    debug_assert!(read_n <= 128, "read_n must be less than or equal to 128");

    let mut offset = 0;

    let packed = unsafe { load_si512x4(input.add(offset)) };
    let b0_bits = unpack_u16_u32_x8(packed);

    offset += read_n * 2;
    let packed = unsafe { load_si512x2(input.add(offset)) };
    let mut b1_bits = unpack_u8_u32_x8(packed);
    b1_bits = slli_epi32::<16, 8>(b1_bits);

    offset += read_n;
    let mut b2_bits = unsafe { unpack_u2_registers(input.add(offset), read_n) };
    b2_bits = slli_epi32::<24, 8>(b2_bits);

    const IMM8: i32 = _MM_TERNLOG_A | _MM_TERNLOG_B | _MM_TERNLOG_C;
    let data = ternary_epi32_all::<IMM8, 8>(b2_bits, b1_bits, b0_bits);

    store_u32x128(block, data);
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the packed block of 27-bit elements and write the resulting elements
/// to the `block`.
///
/// # Safety
/// - The CPU features required must be met.
/// - `read_n` must also be between `0..=128`.
pub unsafe fn from_u27(input: *const u8, block: &mut [u32; X128], read_n: usize) {
    debug_assert!(read_n <= 128, "read_n must be less than or equal to 128");

    let mut offset = 0;

    let packed = unsafe { load_si512x4(input.add(offset)) };
    let b0_bits = unpack_u16_u32_x8(packed);

    offset += read_n * 2;
    let packed = unsafe { load_si512x2(input.add(offset)) };
    let mut b1_bits = unpack_u8_u32_x8(packed);
    b1_bits = slli_epi32::<16, 8>(b1_bits);

    offset += read_n;
    let packed = unsafe { unpack_u3_registers::<0>(input.add(offset), read_n) };
    let mut b2_bits = unpack_u8_u32_x8(packed);
    b2_bits = slli_epi32::<24, 8>(b2_bits);

    const IMM8: i32 = _MM_TERNLOG_A | _MM_TERNLOG_B | _MM_TERNLOG_C;
    let data = ternary_epi32_all::<IMM8, 8>(b2_bits, b1_bits, b0_bits);

    store_u32x128(block, data);
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the packed block of 28-bit elements and write the resulting elements
/// to the `block`.
///
/// # Safety
/// - The CPU features required must be met.
/// - `read_n` must also be between `0..=128`.
pub unsafe fn from_u28(input: *const u8, block: &mut [u32; X128], read_n: usize) {
    debug_assert!(read_n <= 128, "read_n must be less than or equal to 128");

    let mut offset = 0;

    let packed = unsafe { load_si512x4(input.add(offset)) };
    let b0_bits = unpack_u16_u32_x8(packed);

    offset += read_n * 2;
    let packed = unsafe { load_si512x2(input.add(offset)) };
    let mut b1_bits = unpack_u8_u32_x8(packed);
    b1_bits = slli_epi32::<16, 8>(b1_bits);

    offset += read_n;
    let packed = unsafe { unpack_u4_registers(input.add(offset)) };
    let mut b2_bits = unpack_u8_u32_x8(packed);
    b2_bits = slli_epi32::<24, 8>(b2_bits);

    const IMM8: i32 = _MM_TERNLOG_A | _MM_TERNLOG_B | _MM_TERNLOG_C;
    let data = ternary_epi32_all::<IMM8, 8>(b2_bits, b1_bits, b0_bits);

    store_u32x128(block, data);
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the packed block of 29-bit elements and write the resulting elements
/// to the `block`.
///
/// # Safety
/// - The CPU features required must be met.
/// - `read_n` must also be between `0..=128`.
pub unsafe fn from_u29(input: *const u8, block: &mut [u32; X128], read_n: usize) {
    debug_assert!(read_n <= 128, "read_n must be less than or equal to 128");

    let mut offset = 0;

    let packed = unsafe { load_si512x4(input.add(offset)) };
    let b0_bits = unpack_u16_u32_x8(packed);

    offset += read_n * 2;
    let packed = unsafe { load_si512x2(input.add(offset)) };
    let mut b1_bits = unpack_u8_u32_x8(packed);
    b1_bits = slli_epi32::<16, 8>(b1_bits);

    offset += read_n;
    let mut b2_bits = unsafe { unpack_u5_registers(input.add(offset), read_n) };
    b2_bits = slli_epi32::<24, 8>(b2_bits);

    const IMM8: i32 = _MM_TERNLOG_A | _MM_TERNLOG_B | _MM_TERNLOG_C;
    let data = ternary_epi32_all::<IMM8, 8>(b2_bits, b1_bits, b0_bits);

    store_u32x128(block, data);
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the packed block of 30-bit elements and write the resulting elements
/// to the `block`.
///
/// # Safety
/// - The CPU features required must be met.
/// - `read_n` must also be between `0..=128`.
pub unsafe fn from_u30(input: *const u8, block: &mut [u32; X128], read_n: usize) {
    debug_assert!(read_n <= 128, "read_n must be less than or equal to 128");

    let mut offset = 0;

    let packed = unsafe { load_si512x4(input.add(offset)) };
    let b0_bits = unpack_u16_u32_x8(packed);

    offset += read_n * 2;
    let packed = unsafe { load_si512x2(input.add(offset)) };
    let mut b1_bits = unpack_u8_u32_x8(packed);
    b1_bits = slli_epi32::<16, 8>(b1_bits);

    offset += read_n;
    let mut b2_bits = unsafe { unpack_u6_registers(input.add(offset), read_n) };
    b2_bits = slli_epi32::<24, 8>(b2_bits);

    const IMM8: i32 = _MM_TERNLOG_A | _MM_TERNLOG_B | _MM_TERNLOG_C;
    let data = ternary_epi32_all::<IMM8, 8>(b2_bits, b1_bits, b0_bits);

    store_u32x128(block, data);
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the packed block of 31-bit elements and write the resulting elements
/// to the `block`.
///
/// # Safety
/// - The CPU features required must be met.
/// - `read_n` must also be between `0..=128`.
pub unsafe fn from_u31(input: *const u8, block: &mut [u32; X128], read_n: usize) {
    debug_assert!(read_n <= 128, "read_n must be less than or equal to 128");

    let mut offset = 0;

    let packed = unsafe { load_si512x4(input.add(offset)) };
    let b0_bits = unpack_u16_u32_x8(packed);

    offset += read_n * 2;
    let packed = unsafe { load_si512x2(input.add(offset)) };
    let mut b1_bits = unpack_u8_u32_x8(packed);
    b1_bits = slli_epi32::<16, 8>(b1_bits);

    offset += read_n;
    let mut b2_bits = unsafe { unpack_u7_registers(input.add(offset), read_n) };
    b2_bits = slli_epi32::<24, 8>(b2_bits);

    const IMM8: i32 = _MM_TERNLOG_A | _MM_TERNLOG_B | _MM_TERNLOG_C;
    let data = ternary_epi32_all::<IMM8, 8>(b2_bits, b1_bits, b0_bits);

    store_u32x128(block, data);
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the packed block of 32-bit elements and write the resulting elements
/// to the `block`.
///
/// # Safety
/// - The CPU features required must be met.
/// - `read_n` must also be between `0..=128`.
pub unsafe fn from_u32(input: *const u8, block: &mut [u32; X128], read_n: usize) {
    debug_assert!(read_n <= 128, "read_n must be less than or equal to 128");

    let data = unsafe { load_si512x8(input) };
    store_u32x128(block, data);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uint32::X128_MAX_OUTPUT_LEN;
    use crate::uint32::avx512::pack_x128::*;

    #[rstest::rstest]
    #[case(0, from_u0)]
    #[case(1, from_u1)]
    #[case(2, from_u2)]
    #[case(3, from_u3)]
    #[case(4, from_u4)]
    #[case(5, from_u5)]
    #[case(6, from_u6)]
    #[case(7, from_u7)]
    #[case(8, from_u8)]
    #[case(9, from_u9)]
    #[case(10, from_u10)]
    #[case(11, from_u11)]
    #[case(12, from_u12)]
    #[case(13, from_u13)]
    #[case(14, from_u14)]
    #[case(15, from_u15)]
    #[case(16, from_u16)]
    #[case(17, from_u17)]
    #[case(18, from_u18)]
    #[case(19, from_u19)]
    #[case(20, from_u20)]
    #[case(21, from_u21)]
    #[case(22, from_u22)]
    #[case(23, from_u23)]
    #[case(24, from_u24)]
    #[case(25, from_u25)]
    #[case(26, from_u26)]
    #[case(27, from_u27)]
    #[case(28, from_u28)]
    #[case(29, from_u29)]
    #[case(30, from_u30)]
    #[case(31, from_u31)]
    #[case(32, from_u32)]
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_saturated_unpack(
        #[case] bit_len: u8,
        #[case] unpacker: unsafe fn(*const u8, &mut [u32; X128], usize),
    ) {
        let saturated_bytes = [u8::MAX; X128_MAX_OUTPUT_LEN];

        let mut unpacked = [0; X128];
        unsafe { unpacker(saturated_bytes.as_ptr(), &mut unpacked, X128) };

        let expected_value = (2u64.pow(bit_len as u32) - 1) as u32;
        assert_eq!(unpacked, [expected_value; X128]);
    }

    #[rstest::rstest]
    #[case(1, to_u1, from_u1)]
    #[case(2, to_u2, from_u2)]
    #[case(3, to_u3, from_u3)]
    #[case(4, to_u4, from_u4)]
    #[case(5, to_u5, from_u5)]
    #[case(6, to_u6, from_u6)]
    #[case(7, to_u7, from_u7)]
    #[case(8, to_u8, from_u8)]
    #[case(9, to_u9, from_u9)]
    #[case(10, to_u10, from_u10)]
    #[case(11, to_u11, from_u11)]
    #[case(12, to_u12, from_u12)]
    #[case(13, to_u13, from_u13)]
    #[case(14, to_u14, from_u14)]
    #[case(15, to_u15, from_u15)]
    #[case(16, to_u16, from_u16)]
    #[case(17, to_u17, from_u17)]
    #[case(18, to_u18, from_u18)]
    #[case(19, to_u19, from_u19)]
    #[case(20, to_u20, from_u20)]
    #[case(21, to_u21, from_u21)]
    #[case(22, to_u22, from_u22)]
    #[case(23, to_u23, from_u23)]
    #[case(24, to_u24, from_u24)]
    #[case(25, to_u25, from_u25)]
    #[case(26, to_u26, from_u26)]
    #[case(27, to_u27, from_u27)]
    #[case(28, to_u28, from_u28)]
    #[case(29, to_u29, from_u29)]
    #[case(30, to_u30, from_u30)]
    #[case(31, to_u31, from_u31)]
    #[case(32, to_u32, from_u32)]
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_pack_unpack(
        #[case] bit_len: u8,
        #[case] packer: unsafe fn(&mut [u8; X128_MAX_OUTPUT_LEN], &[u32; X128], usize),
        #[case] unpacker: unsafe fn(*const u8, &mut [u32; X128], usize),
    ) {
        fastrand::seed(5876358762523525);

        let max_value = (2u64.pow(bit_len as u32) - 1) as u32;

        let mut values = [0; X128];
        for value in values.iter_mut() {
            *value = fastrand::u32(0..max_value);
        }

        let mut packed = [0; X128_MAX_OUTPUT_LEN];
        unsafe { packer(&mut packed, &values, X128) };

        let mut unpacked = [0; X128];
        unsafe { unpacker(packed.as_ptr(), &mut unpacked, X128) };
        assert_eq!(unpacked, values);
    }

    #[rstest::rstest]
    #[case(1, to_u1, from_u1)]
    #[case(2, to_u2, from_u2)]
    #[case(3, to_u3, from_u3)]
    #[case(4, to_u4, from_u4)]
    #[case(5, to_u5, from_u5)]
    #[case(6, to_u6, from_u6)]
    #[case(7, to_u7, from_u7)]
    #[case(8, to_u8, from_u8)]
    #[case(9, to_u9, from_u9)]
    #[case(10, to_u10, from_u10)]
    #[case(11, to_u11, from_u11)]
    #[case(12, to_u12, from_u12)]
    #[case(13, to_u13, from_u13)]
    #[case(14, to_u14, from_u14)]
    #[case(15, to_u15, from_u15)]
    #[case(16, to_u16, from_u16)]
    #[case(17, to_u17, from_u17)]
    #[case(18, to_u18, from_u18)]
    #[case(19, to_u19, from_u19)]
    #[case(20, to_u20, from_u20)]
    #[case(21, to_u21, from_u21)]
    #[case(22, to_u22, from_u22)]
    #[case(23, to_u23, from_u23)]
    #[case(24, to_u24, from_u24)]
    #[case(25, to_u25, from_u25)]
    #[case(26, to_u26, from_u26)]
    #[case(27, to_u27, from_u27)]
    #[case(28, to_u28, from_u28)]
    #[case(29, to_u29, from_u29)]
    #[case(30, to_u30, from_u30)]
    #[case(31, to_u31, from_u31)]
    #[case(32, to_u32, from_u32)]
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_pack_unpack_partial(
        #[case] bit_len: u8,
        #[case] packer: unsafe fn(&mut [u8; X128_MAX_OUTPUT_LEN], &[u32; X128], usize),
        #[case] unpacker: unsafe fn(*const u8, &mut [u32; X128], usize),
        #[values(1, 32, 64, 65, 97)] num_values: usize,
    ) {
        fastrand::seed(5876358762523525);

        let max_value = (2u64.pow(bit_len as u32) - 1) as u32;

        let mut values = [0; X128];
        for value in values.iter_mut() {
            *value = fastrand::u32(0..max_value);
        }

        let mut packed = [0; X128_MAX_OUTPUT_LEN];
        unsafe { packer(&mut packed, &values, num_values) };

        let expected_bytes_required = crate::uint32::compressed_size(bit_len as usize, num_values);
        packed[expected_bytes_required..].fill(0);

        let mut unpacked = [0; X128];
        unsafe { unpacker(packed.as_ptr(), &mut unpacked, num_values) };
        assert_eq!(unpacked[..num_values], values[..num_values]);
    }

    // Checks that we can correctly decompress data based on the number of bytes we report
    // to the user for bitpacking.
    #[rstest::rstest]
    #[case(1, from_u1)]
    #[case(2, from_u2)]
    #[case(3, from_u3)]
    #[case(4, from_u4)]
    #[case(5, from_u5)]
    #[case(6, from_u6)]
    #[case(7, from_u7)]
    #[case(8, from_u8)]
    #[case(9, from_u9)]
    #[case(10, from_u10)]
    #[case(11, from_u11)]
    #[case(12, from_u12)]
    #[case(13, from_u13)]
    #[case(14, from_u14)]
    #[case(15, from_u15)]
    #[case(16, from_u16)]
    #[case(17, from_u17)]
    #[case(18, from_u18)]
    #[case(19, from_u19)]
    #[case(20, from_u20)]
    #[case(21, from_u21)]
    #[case(22, from_u22)]
    #[case(23, from_u23)]
    #[case(24, from_u24)]
    #[case(25, from_u25)]
    #[case(26, from_u26)]
    #[case(27, from_u27)]
    #[case(28, from_u28)]
    #[case(29, from_u29)]
    #[case(30, from_u30)]
    #[case(31, from_u31)]
    #[case(32, from_u32)]
    #[should_panic]
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_pack_unpack_partial_check_length_valid(
        #[case] bit_len: u8,
        #[case] unpacker: unsafe fn(*const u8, &mut [u32; X128], usize),
        #[values(1, 32, 64, 65, 97)] num_values: usize,
    ) {
        let mut saturated_bytes = [u8::MAX; X128_MAX_OUTPUT_LEN];

        let expected_bytes_required = crate::uint32::compressed_size(bit_len as usize, num_values);
        saturated_bytes[expected_bytes_required - 1..].fill(0);

        let mut unpacked = [0; X128];
        unsafe { unpacker(saturated_bytes.as_ptr(), &mut unpacked, num_values) };

        let max_value = (2u64.pow(bit_len as u32) - 1) as u32;
        assert_eq!(unpacked[..num_values], vec![max_value; num_values]);
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_length_calculation_permutations_saturated() {
        let block = [u32::MAX; X128];
        let mut decompressed = [0; X128];
        for n in 0..X128 {
            for bit_length in 0..32 {
                let mut out = [0; X128_MAX_OUTPUT_LEN];

                let expected = [((1u64 << bit_length) - 1) as u32; X128];
                let expected_length = crate::uint32::compressed_size(bit_length, n);
                unsafe { to_nbits(&mut out, bit_length as u8, &block, n) };

                let mut zeroed_output = out.clone();
                zeroed_output[expected_length..].fill(0);
                unsafe {
                    from_nbits(
                        zeroed_output.as_ptr(),
                        bit_length as u8,
                        &mut decompressed,
                        n,
                    )
                };

                assert_eq!(
                    decompressed[..n],
                    expected[..n],
                    "invalid size: n={n} bit_length={bit_length}, expected={expected_length}\n\
                    output:{out:?}",
                );
            }
        }
    }

    #[rstest::rstest]
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_length_calculation_permutations_fuzz(
        #[values(8762358237652, 25627532423, 828235235)] seed: u64,
    ) {
        fastrand::seed(seed);

        let mut decompressed = [0; X128];
        for n in 1..X128 {
            for bit_length in 1..32 {
                let expected = [fastrand::u32(0..((1u64 << bit_length) - 1) as u32); X128];

                let mut out = [0; X128_MAX_OUTPUT_LEN];
                let expected_length = crate::uint32::compressed_size(bit_length, n);
                unsafe { to_nbits(&mut out, bit_length as u8, &expected, n) };

                let mut zeroed_output = out.clone();
                zeroed_output[expected_length..].fill(0);
                unsafe {
                    from_nbits(
                        zeroed_output.as_ptr(),
                        bit_length as u8,
                        &mut decompressed,
                        n,
                    )
                };

                assert_eq!(
                    decompressed[..n],
                    expected[..n],
                    "invalid size: n={n} bit_length={bit_length}, expected={expected_length}\n\
                    output:{out:?}",
                );
            }
        }
    }
}
