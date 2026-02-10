use std::arch::x86_64::*;

use super::data::*;
use super::utils::*;

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the 1-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(1)` bytes from.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(super) unsafe fn from_u1(input: *const u8) -> [__m512i; 4] {
    let packed = unsafe { unpack_u1_registers(input) };
    unpack_u8_to_u32_unordered(packed)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack eight registers containing 64 8-bit elements from a 1-bit bitmap provided
/// by `input`.
unsafe fn unpack_u1_registers(input: *const u8) -> __m512i {
    let mask: u64 = unsafe { std::ptr::read_unaligned(input.add(0).cast()) };
    let ones = _mm512_set1_epi8(1);
    _mm512_maskz_mov_epi8(mask, ones)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the 2-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(2)` bytes from.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(super) unsafe fn from_u2(input: *const u8) -> [__m512i; 4] {
    let packed = unsafe { unpack_u2_registers(input) };
    unpack_u8_to_u32_unordered(packed)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack eight registers containing 64 8-bit elements from a 2-bit bitmap provided
/// by `input`.
unsafe fn unpack_u2_registers(input: *const u8) -> __m512i {
    let mask1: u64 = unsafe { std::ptr::read_unaligned(input.add(0).cast()) };
    let mask2: u64 = unsafe { std::ptr::read_unaligned(input.add(8).cast()) };

    let bit = _mm512_set1_epi8(0b01);
    let lo_bits_packed = _mm512_maskz_mov_epi8(mask1, bit);

    let bit = _mm512_set1_epi8(0b10);
    let hi_bits_packed = _mm512_maskz_mov_epi8(mask2, bit);

    _mm512_or_si512(hi_bits_packed, lo_bits_packed)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the 3-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(3)` bytes from.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(super) unsafe fn from_u3(input: *const u8) -> [__m512i; 4] {
    let packed = unsafe { unpack_u3_registers(input) };
    unpack_u8_to_u32_unordered(packed)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack eight registers containing 64 8-bit elements from a 3-bit bitmap provided
/// by `input`.
unsafe fn unpack_u3_registers(input: *const u8) -> __m512i {
    let mask1: u64 = unsafe { std::ptr::read_unaligned(input.add(0).cast()) };
    let mask2: u64 = unsafe { std::ptr::read_unaligned(input.add(8).cast()) };
    let mask3: u64 = unsafe { std::ptr::read_unaligned(input.add(16).cast()) };

    let bit = _mm512_set1_epi8(0b001);
    let b0_bits_packed = _mm512_maskz_mov_epi8(mask1, bit);

    let bit = _mm512_set1_epi8(0b010);
    let b1_bits_packed = _mm512_maskz_mov_epi8(mask2, bit);

    let bit = _mm512_set1_epi8(0b100);
    let b2_bits_packed = _mm512_maskz_mov_epi8(mask3, bit);

    const OP_MASK: i32 = _MM_TERNLOG_C | _MM_TERNLOG_B | _MM_TERNLOG_A;
    _mm512_ternarylogic_epi32::<OP_MASK>(b2_bits_packed, b1_bits_packed, b0_bits_packed)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the 4-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(4)` bytes from.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(super) unsafe fn from_u4(input: *const u8) -> [__m512i; 4] {
    let packed = unsafe { unpack_u4_registers(input) };
    unpack_u8_to_u32_unordered(packed)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack eight registers containing 64 8-bit elements from a 4-bit nibbles provided
/// by `input`.
unsafe fn unpack_u4_registers(input: *const u8) -> __m512i {
    unsafe { super::unpack_x64_partial::unpack_u4_registers(input) }
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the 5-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(5)` bytes from.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(super) unsafe fn from_u5(input: *const u8) -> [__m512i; 4] {
    let packed = unsafe { unpack_u5_registers(input) };
    unpack_u8_to_u32_unordered(packed)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack eight registers containing 64 8-bit elements from a 5-bit block provided
/// by `input`.
unsafe fn unpack_u5_registers(input: *const u8) -> __m512i {
    let lo_bits = unsafe { unpack_u4_registers(input.add(0)) };
    let mut hi_bits = unsafe { unpack_u1_registers(input.add(32)) };
    hi_bits = _mm512_slli_epi16::<4>(hi_bits);
    _mm512_or_si512(hi_bits, lo_bits)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the 6-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(6)` bytes from.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(super) unsafe fn from_u6(input: *const u8) -> [__m512i; 4] {
    let packed = unsafe { unpack_u6_registers(input) };
    unpack_u8_to_u32_unordered(packed)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack eight registers containing 64 8-bit elements from a 6-bit block provided
/// by `input`.
unsafe fn unpack_u6_registers(input: *const u8) -> __m512i {
    let lo_bits = unsafe { unpack_u4_registers(input.add(0)) };
    let mut hi_bits = unsafe { unpack_u2_registers(input.add(32)) };
    hi_bits = _mm512_slli_epi16::<4>(hi_bits);
    _mm512_or_si512(hi_bits, lo_bits)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the 7-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(7)` bytes from.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(super) unsafe fn from_u7(input: *const u8) -> [__m512i; 4] {
    let packed = unsafe { unpack_u7_registers(input) };
    unpack_u8_to_u32_unordered(packed)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack eight registers containing 64 8-bit elements from a 6-bit block provided
/// by `input`.
unsafe fn unpack_u7_registers(input: *const u8) -> __m512i {
    let lo_bits = unsafe { unpack_u4_registers(input.add(0)) };
    let mut hi_bits = unsafe { unpack_u3_registers(input.add(32)) };
    hi_bits = _mm512_slli_epi16::<4>(hi_bits);
    _mm512_or_si512(hi_bits, lo_bits)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the 8-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(8)` bytes from.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(super) unsafe fn from_u8(input: *const u8) -> [__m512i; 4] {
    let packed = unsafe { _mm512_loadu_epi8(input.cast()) };
    unpack_u8_to_u32_unordered(packed)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the 9-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(9)` bytes from.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(super) unsafe fn from_u9(input: *const u8) -> [__m512i; 4] {
    let lo_bits = unsafe { _mm512_loadu_epi8(input.add(0).cast()) };
    let lo_bits = unpack_u8_to_u16_unordered(lo_bits);

    let hi_bits = unsafe { unpack_u1_registers(input.add(64)) };
    let mut hi_bits = unpack_u8_to_u16_unordered(hi_bits);
    hi_bits = slli_epi16::<8, 2>(hi_bits);

    let packed = or_si512_all(hi_bits, lo_bits);
    unpack_u16_to_u32_unordered(packed)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the 10-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(10)` bytes from.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(super) unsafe fn from_u10(input: *const u8) -> [__m512i; 4] {
    let lo_bits = unsafe { _mm512_loadu_epi8(input.add(0).cast()) };
    let lo_bits = unpack_u8_to_u16_unordered(lo_bits);

    let hi_bits = unsafe { unpack_u2_registers(input.add(64)) };
    let mut hi_bits = unpack_u8_to_u16_unordered(hi_bits);
    hi_bits = slli_epi16::<8, 2>(hi_bits);

    let packed = or_si512_all(hi_bits, lo_bits);
    unpack_u16_to_u32_unordered(packed)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the 11-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(11)` bytes from.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(super) unsafe fn from_u11(input: *const u8) -> [__m512i; 4] {
    let lo_bits = unsafe { _mm512_loadu_epi8(input.add(0).cast()) };
    let lo_bits = unpack_u8_to_u16_unordered(lo_bits);

    let hi_bits = unsafe { unpack_u3_registers(input.add(64)) };
    let mut hi_bits = unpack_u8_to_u16_unordered(hi_bits);
    hi_bits = slli_epi16::<8, 2>(hi_bits);

    let packed = or_si512_all(hi_bits, lo_bits);
    unpack_u16_to_u32_unordered(packed)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the 12-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(12)` bytes from.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(super) unsafe fn from_u12(input: *const u8) -> [__m512i; 4] {
    let lo_bits = unsafe { _mm512_loadu_epi8(input.add(0).cast()) };
    let lo_bits = unpack_u8_to_u16_unordered(lo_bits);

    let hi_bits = unsafe { unpack_u4_registers(input.add(64)) };
    let mut hi_bits = unpack_u8_to_u16_unordered(hi_bits);
    hi_bits = slli_epi16::<8, 2>(hi_bits);

    let packed = or_si512_all(hi_bits, lo_bits);
    unpack_u16_to_u32_unordered(packed)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the 13-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(13)` bytes from.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(super) unsafe fn from_u13(input: *const u8) -> [__m512i; 4] {
    let lo_bits = unsafe { _mm512_loadu_epi8(input.add(0).cast()) };
    let lo_bits = unpack_u8_to_u16_unordered(lo_bits);

    let hi_bits = unsafe { unpack_u5_registers(input.add(64)) };
    let mut hi_bits = unpack_u8_to_u16_unordered(hi_bits);
    hi_bits = slli_epi16::<8, 2>(hi_bits);

    let packed = or_si512_all(hi_bits, lo_bits);
    unpack_u16_to_u32_unordered(packed)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the 14-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(14)` bytes from.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(super) unsafe fn from_u14(input: *const u8) -> [__m512i; 4] {
    let lo_bits = unsafe { _mm512_loadu_epi8(input.add(0).cast()) };
    let lo_bits = unpack_u8_to_u16_unordered(lo_bits);

    let hi_bits = unsafe { unpack_u6_registers(input.add(64)) };
    let mut hi_bits = unpack_u8_to_u16_unordered(hi_bits);
    hi_bits = slli_epi16::<8, 2>(hi_bits);

    let packed = or_si512_all(hi_bits, lo_bits);
    unpack_u16_to_u32_unordered(packed)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the 15-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(15)` bytes from.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(super) unsafe fn from_u15(input: *const u8) -> [__m512i; 4] {
    let lo_bits = unsafe { _mm512_loadu_epi8(input.add(0).cast()) };
    let lo_bits = unpack_u8_to_u16_unordered(lo_bits);

    let hi_bits = unsafe { unpack_u7_registers(input.add(64)) };
    let mut hi_bits = unpack_u8_to_u16_unordered(hi_bits);
    hi_bits = slli_epi16::<8, 2>(hi_bits);

    let packed = or_si512_all(hi_bits, lo_bits);
    unpack_u16_to_u32_unordered(packed)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the 16-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(16)` bytes from.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(super) unsafe fn from_u16(input: *const u8) -> [__m512i; 4] {
    let packed = unsafe { load_si512x2(input) };
    unpack_u16_to_u32_unordered(packed)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the 17-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(17)` bytes from.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(super) unsafe fn from_u17(input: *const u8) -> [__m512i; 4] {
    let lo_bits = unsafe { load_si512x2(input.add(0)) };
    let lo_bits = unpack_u16_to_u32_unordered(lo_bits);

    let hi_bits = unsafe { unpack_u1_registers(input.add(128)) };
    let mut hi_bits = unpack_u8_to_u32_unordered(hi_bits);
    hi_bits = slli_epi32::<16, 4>(hi_bits);

    or_si512_all(hi_bits, lo_bits)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the 18-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(18)` bytes from.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(super) unsafe fn from_u18(input: *const u8) -> [__m512i; 4] {
    let lo_bits = unsafe { load_si512x2(input.add(0)) };
    let lo_bits = unpack_u16_to_u32_unordered(lo_bits);

    let hi_bits = unsafe { unpack_u2_registers(input.add(128)) };
    let mut hi_bits = unpack_u8_to_u32_unordered(hi_bits);
    hi_bits = slli_epi32::<16, 4>(hi_bits);

    or_si512_all(hi_bits, lo_bits)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the 19-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(19)` bytes from.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(super) unsafe fn from_u19(input: *const u8) -> [__m512i; 4] {
    let lo_bits = unsafe { load_si512x2(input.add(0)) };
    let lo_bits = unpack_u16_to_u32_unordered(lo_bits);

    let hi_bits = unsafe { unpack_u3_registers(input.add(128)) };
    let mut hi_bits = unpack_u8_to_u32_unordered(hi_bits);
    hi_bits = slli_epi32::<16, 4>(hi_bits);

    or_si512_all(hi_bits, lo_bits)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the 20-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(20)` bytes from.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(super) unsafe fn from_u20(input: *const u8) -> [__m512i; 4] {
    let lo_bits = unsafe { load_si512x2(input.add(0)) };
    let lo_bits = unpack_u16_to_u32_unordered(lo_bits);

    let hi_bits = unsafe { unpack_u4_registers(input.add(128)) };
    let mut hi_bits = unpack_u8_to_u32_unordered(hi_bits);
    hi_bits = slli_epi32::<16, 4>(hi_bits);

    or_si512_all(hi_bits, lo_bits)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the 21-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(21)` bytes from.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(super) unsafe fn from_u21(input: *const u8) -> [__m512i; 4] {
    let lo_bits = unsafe { load_si512x2(input.add(0)) };
    let lo_bits = unpack_u16_to_u32_unordered(lo_bits);

    let hi_bits = unsafe { unpack_u5_registers(input.add(128)) };
    let mut hi_bits = unpack_u8_to_u32_unordered(hi_bits);
    hi_bits = slli_epi32::<16, 4>(hi_bits);

    or_si512_all(hi_bits, lo_bits)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the 22-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(22)` bytes from.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(super) unsafe fn from_u22(input: *const u8) -> [__m512i; 4] {
    let lo_bits = unsafe { load_si512x2(input.add(0)) };
    let lo_bits = unpack_u16_to_u32_unordered(lo_bits);

    let hi_bits = unsafe { unpack_u6_registers(input.add(128)) };
    let mut hi_bits = unpack_u8_to_u32_unordered(hi_bits);
    hi_bits = slli_epi32::<16, 4>(hi_bits);

    or_si512_all(hi_bits, lo_bits)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the 23-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(23)` bytes from.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(super) unsafe fn from_u23(input: *const u8) -> [__m512i; 4] {
    let lo_bits = unsafe { load_si512x2(input.add(0)) };
    let lo_bits = unpack_u16_to_u32_unordered(lo_bits);

    let hi_bits = unsafe { unpack_u7_registers(input.add(128)) };
    let mut hi_bits = unpack_u8_to_u32_unordered(hi_bits);
    hi_bits = slli_epi32::<16, 4>(hi_bits);

    or_si512_all(hi_bits, lo_bits)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the 24-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(24)` bytes from.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(super) unsafe fn from_u24(input: *const u8) -> [__m512i; 4] {
    let lo_bits = unsafe { load_si512x2(input.add(0)) };
    let lo_bits = unpack_u16_to_u32_unordered(lo_bits);

    let hi_bits = unsafe { _mm512_loadu_epi8(input.add(128).cast()) };
    let mut hi_bits = unpack_u8_to_u32_unordered(hi_bits);
    hi_bits = slli_epi32::<16, 4>(hi_bits);

    or_si512_all(hi_bits, lo_bits)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the 25-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(25)` bytes from.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(super) unsafe fn from_u25(input: *const u8) -> [__m512i; 4] {
    let lo_bits = unsafe { load_si512x2(input.add(0)) };
    let lo_bits = unpack_u16_to_u32_unordered(lo_bits);

    let hi_8bits = unsafe { _mm512_loadu_epi8(input.add(128).cast()) };
    let hi_8bits = unpack_u8_to_u16_unordered(hi_8bits);

    let hi_1bits = unsafe { unpack_u1_registers(input.add(192)) };
    let mut hi_1bits = unpack_u8_to_u16_unordered(hi_1bits);
    hi_1bits = slli_epi16::<8, 2>(hi_1bits);

    let hi_bits = or_si512_all(hi_1bits, hi_8bits);
    let mut hi_bits = unpack_u16_to_u32_unordered(hi_bits);
    hi_bits = slli_epi32::<16, 4>(hi_bits);

    or_si512_all(hi_bits, lo_bits)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the 26-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(26)` bytes from.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(super) unsafe fn from_u26(input: *const u8) -> [__m512i; 4] {
    let lo_bits = unsafe { load_si512x2(input.add(0)) };
    let lo_bits = unpack_u16_to_u32_unordered(lo_bits);

    let hi_8bits = unsafe { _mm512_loadu_epi8(input.add(128).cast()) };
    let hi_8bits = unpack_u8_to_u16_unordered(hi_8bits);

    let hi_1bits = unsafe { unpack_u2_registers(input.add(192)) };
    let mut hi_1bits = unpack_u8_to_u16_unordered(hi_1bits);
    hi_1bits = slli_epi16::<8, 2>(hi_1bits);

    let hi_bits = or_si512_all(hi_1bits, hi_8bits);
    let mut hi_bits = unpack_u16_to_u32_unordered(hi_bits);
    hi_bits = slli_epi32::<16, 4>(hi_bits);

    or_si512_all(hi_bits, lo_bits)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the 27-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(27)` bytes from.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(super) unsafe fn from_u27(input: *const u8) -> [__m512i; 4] {
    let lo_bits = unsafe { load_si512x2(input.add(0)) };
    let lo_bits = unpack_u16_to_u32_unordered(lo_bits);

    let hi_8bits = unsafe { _mm512_loadu_epi8(input.add(128).cast()) };
    let hi_8bits = unpack_u8_to_u16_unordered(hi_8bits);

    let hi_1bits = unsafe { unpack_u3_registers(input.add(192)) };
    let mut hi_1bits = unpack_u8_to_u16_unordered(hi_1bits);
    hi_1bits = slli_epi16::<8, 2>(hi_1bits);

    let hi_bits = or_si512_all(hi_1bits, hi_8bits);
    let mut hi_bits = unpack_u16_to_u32_unordered(hi_bits);
    hi_bits = slli_epi32::<16, 4>(hi_bits);

    or_si512_all(hi_bits, lo_bits)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the 28-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(28)` bytes from.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(super) unsafe fn from_u28(input: *const u8) -> [__m512i; 4] {
    let lo_bits = unsafe { load_si512x2(input.add(0)) };
    let lo_bits = unpack_u16_to_u32_unordered(lo_bits);

    let hi_8bits = unsafe { _mm512_loadu_epi8(input.add(128).cast()) };
    let hi_8bits = unpack_u8_to_u16_unordered(hi_8bits);

    let hi_1bits = unsafe { unpack_u4_registers(input.add(192)) };
    let mut hi_1bits = unpack_u8_to_u16_unordered(hi_1bits);
    hi_1bits = slli_epi16::<8, 2>(hi_1bits);

    let hi_bits = or_si512_all(hi_1bits, hi_8bits);
    let mut hi_bits = unpack_u16_to_u32_unordered(hi_bits);
    hi_bits = slli_epi32::<16, 4>(hi_bits);

    or_si512_all(hi_bits, lo_bits)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the 29-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(29)` bytes from.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(super) unsafe fn from_u29(input: *const u8) -> [__m512i; 4] {
    let lo_bits = unsafe { load_si512x2(input.add(0)) };
    let lo_bits = unpack_u16_to_u32_unordered(lo_bits);

    let hi_8bits = unsafe { _mm512_loadu_epi8(input.add(128).cast()) };
    let hi_8bits = unpack_u8_to_u16_unordered(hi_8bits);

    let hi_1bits = unsafe { unpack_u5_registers(input.add(192)) };
    let mut hi_1bits = unpack_u8_to_u16_unordered(hi_1bits);
    hi_1bits = slli_epi16::<8, 2>(hi_1bits);

    let hi_bits = or_si512_all(hi_1bits, hi_8bits);
    let mut hi_bits = unpack_u16_to_u32_unordered(hi_bits);
    hi_bits = slli_epi32::<16, 4>(hi_bits);

    or_si512_all(hi_bits, lo_bits)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the 30-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(30)` bytes from.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(super) unsafe fn from_u30(input: *const u8) -> [__m512i; 4] {
    let lo_bits = unsafe { load_si512x2(input.add(0)) };
    let lo_bits = unpack_u16_to_u32_unordered(lo_bits);

    let hi_8bits = unsafe { _mm512_loadu_epi8(input.add(128).cast()) };
    let hi_8bits = unpack_u8_to_u16_unordered(hi_8bits);

    let hi_1bits = unsafe { unpack_u6_registers(input.add(192)) };
    let mut hi_1bits = unpack_u8_to_u16_unordered(hi_1bits);
    hi_1bits = slli_epi16::<8, 2>(hi_1bits);

    let hi_bits = or_si512_all(hi_1bits, hi_8bits);
    let mut hi_bits = unpack_u16_to_u32_unordered(hi_bits);
    hi_bits = slli_epi32::<16, 4>(hi_bits);

    or_si512_all(hi_bits, lo_bits)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the 31-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(31)` bytes from.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(super) unsafe fn from_u31(input: *const u8) -> [__m512i; 4] {
    let lo_bits = unsafe { load_si512x2(input.add(0)) };
    let lo_bits = unpack_u16_to_u32_unordered(lo_bits);

    let hi_8bits = unsafe { _mm512_loadu_epi8(input.add(128).cast()) };
    let hi_8bits = unpack_u8_to_u16_unordered(hi_8bits);

    let hi_1bits = unsafe { unpack_u7_registers(input.add(192)) };
    let mut hi_1bits = unpack_u8_to_u16_unordered(hi_1bits);
    hi_1bits = slli_epi16::<8, 2>(hi_1bits);

    let hi_bits = or_si512_all(hi_1bits, hi_8bits);
    let mut hi_bits = unpack_u16_to_u32_unordered(hi_bits);
    hi_bits = slli_epi32::<16, 4>(hi_bits);

    or_si512_all(hi_bits, lo_bits)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the 32-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(32)` bytes from.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(super) unsafe fn from_u32(input: *const u8) -> [__m512i; 4] {
    unsafe { load_si512x4(input) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::X64;
    use crate::uint32::avx512::pack_x64_full::*;
    use crate::uint32::{X128_MAX_OUTPUT_LEN, avx2};

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
    #[cfg_attr(
        not(all(target_feature = "avx512f", target_feature = "avx512bw")),
        ignore
    )]
    fn test_saturated_unpack(
        #[case] bit_len: u8,
        #[case] unpacker: unsafe fn(*const u8) -> [__m512i; 4],
    ) {
        let saturated_bytes = [u8::MAX; X128_MAX_OUTPUT_LEN / 2];

        let unpacked = unsafe { unpacker(saturated_bytes.as_ptr()) };
        let unpacked = unsafe { std::mem::transmute::<[__m512i; 4], [u32; X64]>(unpacked) };

        let expected_value = (2u64.pow(bit_len as u32) - 1) as u32;
        assert_eq!(unpacked, [expected_value; X64]);
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
    #[cfg_attr(
        not(all(target_feature = "avx512f", target_feature = "avx512bw")),
        ignore
    )]
    fn test_pack_unpack(
        #[case] bit_len: u8,
        #[case] packer: unsafe fn(*mut u8, [__m512i; 4]),
        #[case] unpacker: unsafe fn(*const u8) -> [__m512i; 4],
    ) {
        fastrand::seed(5876358762523525);

        let max_value = (2u64.pow(bit_len as u32) - 1) as u32;

        let mut values = [0; X64];
        for value in values.iter_mut() {
            *value = fastrand::u32(0..max_value);
        }
        let data = unsafe { load_u32x64(&values) };

        let mut packed = [0; X128_MAX_OUTPUT_LEN / 2];
        unsafe { packer(packed.as_mut_ptr(), data) };

        let unpacked = unsafe { unpacker(packed.as_ptr()) };
        let unpacked = unsafe { std::mem::transmute::<[__m512i; 4], [u32; X64]>(unpacked) };
        assert_eq!(unpacked, values);
    }

    #[rstest::rstest]
    #[case(1, avx2::pack_x64_full::to_u1, from_u1)]
    #[case(2, avx2::pack_x64_full::to_u2, from_u2)]
    #[case(3, avx2::pack_x64_full::to_u3, from_u3)]
    #[case(4, avx2::pack_x64_full::to_u4, from_u4)]
    #[case(5, avx2::pack_x64_full::to_u5, from_u5)]
    #[case(6, avx2::pack_x64_full::to_u6, from_u6)]
    #[case(7, avx2::pack_x64_full::to_u7, from_u7)]
    #[case(8, avx2::pack_x64_full::to_u8, from_u8)]
    #[case(9, avx2::pack_x64_full::to_u9, from_u9)]
    #[case(10, avx2::pack_x64_full::to_u10, from_u10)]
    #[case(11, avx2::pack_x64_full::to_u11, from_u11)]
    #[case(12, avx2::pack_x64_full::to_u12, from_u12)]
    #[case(13, avx2::pack_x64_full::to_u13, from_u13)]
    #[case(14, avx2::pack_x64_full::to_u14, from_u14)]
    #[case(15, avx2::pack_x64_full::to_u15, from_u15)]
    #[case(16, avx2::pack_x64_full::to_u16, from_u16)]
    #[case(17, avx2::pack_x64_full::to_u17, from_u17)]
    #[case(18, avx2::pack_x64_full::to_u18, from_u18)]
    #[case(19, avx2::pack_x64_full::to_u19, from_u19)]
    #[case(20, avx2::pack_x64_full::to_u20, from_u20)]
    #[case(21, avx2::pack_x64_full::to_u21, from_u21)]
    #[case(22, avx2::pack_x64_full::to_u22, from_u22)]
    #[case(23, avx2::pack_x64_full::to_u23, from_u23)]
    #[case(24, avx2::pack_x64_full::to_u24, from_u24)]
    #[case(25, avx2::pack_x64_full::to_u25, from_u25)]
    #[case(26, avx2::pack_x64_full::to_u26, from_u26)]
    #[case(27, avx2::pack_x64_full::to_u27, from_u27)]
    #[case(28, avx2::pack_x64_full::to_u28, from_u28)]
    #[case(29, avx2::pack_x64_full::to_u29, from_u29)]
    #[case(30, avx2::pack_x64_full::to_u30, from_u30)]
    #[case(31, avx2::pack_x64_full::to_u31, from_u31)]
    #[case(32, avx2::pack_x64_full::to_u32, from_u32)]
    #[cfg_attr(
        not(all(target_feature = "avx512f", target_feature = "avx512bw")),
        ignore
    )]
    fn test_unpack_avx2_packed(
        #[case] bit_len: u8,
        #[case] packer: unsafe fn(*mut u8, [__m256i; 8]),
        #[case] unpacker: unsafe fn(*const u8) -> [__m512i; 4],
    ) {
        fastrand::seed(5876358762523525);

        let max_value = (2u64.pow(bit_len as u32) - 1) as u32;

        let mut values = [0; X64];
        for value in values.iter_mut() {
            *value = fastrand::u32(0..max_value);
        }
        let data = unsafe { avx2::data::load_u32x64(&values) };

        let mut packed = [0; X128_MAX_OUTPUT_LEN / 2];
        unsafe { packer(packed.as_mut_ptr(), data) };

        let unpacked = unsafe { unpacker(packed.as_ptr()) };
        let unpacked = unsafe { std::mem::transmute::<[__m512i; 4], [u32; X64]>(unpacked) };
        assert_eq!(unpacked, values);
    }
}
