use std::arch::x86_64::*;

use super::data::*;
use super::utils::*;

#[target_feature(enable = "avx2")]
/// Unpack the 1-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(1)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
/// - `read_n` must be between no larger than `64`.
pub unsafe fn from_u1(input: *const u8, read_n: usize) -> [__m256i; 8] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let packed = unsafe { unpack_u1_registers(input) };
    unpack_u8_to_u32_ordered(packed)
}

#[target_feature(enable = "avx2")]
/// Unpack eight registers containing 8 32-bit elements from a 1-bit bitmap provided
/// by `input`.
unsafe fn unpack_u1_registers(input: *const u8) -> [__m256i; 2] {
    let mask: u64 = unsafe { std::ptr::read_unaligned(input.cast()) };

    let ones = _mm256_set1_epi8(0b1);
    let mut packed1 = expand_mask_epi8(mask as __mmask32);
    let mut packed2 = expand_mask_epi8((mask >> 32) as __mmask32);
    packed1 = _mm256_and_si256(packed1, ones);
    packed2 = _mm256_and_si256(packed2, ones);

    [packed1, packed2]
}

#[target_feature(enable = "avx2")]
/// Unpack the 2-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(2)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
/// - `read_n` must be between no larger than `64`.
pub unsafe fn from_u2(input: *const u8, read_n: usize) -> [__m256i; 8] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let packed = unsafe { unpack_u2_registers(input, read_n) };
    unpack_u8_to_u32_ordered(packed)
}

#[target_feature(enable = "avx2")]
/// Unpack eight registers containing 8 32-bit elements from a 2-bit bitmap provided
/// by `input`.
unsafe fn unpack_u2_registers(input: *const u8, read_n: usize) -> [__m256i; 2] {
    let hi_offset = read_n.div_ceil(8);
    let mask1: u64 = unsafe { std::ptr::read_unaligned(input.add(0).cast()) };
    let mask2: u64 = unsafe { std::ptr::read_unaligned(input.add(hi_offset).cast()) };

    let bit = _mm256_set1_epi8(0b01);
    let mut lo_bits_packed1 = expand_mask_epi8(mask1 as __mmask32);
    let mut lo_bits_packed2 = expand_mask_epi8((mask1 >> 32) as __mmask32);
    lo_bits_packed1 = _mm256_and_si256(lo_bits_packed1, bit);
    lo_bits_packed2 = _mm256_and_si256(lo_bits_packed2, bit);

    let bit = _mm256_set1_epi8(0b10);
    let mut hi_bits_packed1 = expand_mask_epi8(mask2 as __mmask32);
    let mut hi_bits_packed2 = expand_mask_epi8((mask2 >> 32) as __mmask32);
    hi_bits_packed1 = _mm256_and_si256(hi_bits_packed1, bit);
    hi_bits_packed2 = _mm256_and_si256(hi_bits_packed2, bit);

    let two_bits1 = _mm256_or_si256(hi_bits_packed1, lo_bits_packed1);
    let two_bits2 = _mm256_or_si256(hi_bits_packed2, lo_bits_packed2);

    [two_bits1, two_bits2]
}

#[target_feature(enable = "avx2")]
/// Unpack the 3-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(3)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
/// - `read_n` must be between no larger than `64`.
pub unsafe fn from_u3(input: *const u8, read_n: usize) -> [__m256i; 8] {
    let packed = unsafe { unpack_u3_registers(input, read_n) };
    unpack_u8_to_u32_ordered(packed)
}

#[target_feature(enable = "avx2")]
/// Unpack eight registers containing 8 32-bit elements from a 3-bit bitmap provided
/// by `input`.
unsafe fn unpack_u3_registers(input: *const u8, read_n: usize) -> [__m256i; 2] {
    let step = read_n.div_ceil(8);

    let mask1: u64 = unsafe { std::ptr::read_unaligned(input.add(0).cast()) };
    let mask2: u64 = unsafe { std::ptr::read_unaligned(input.add(step).cast()) };
    let mask3: u64 = unsafe { std::ptr::read_unaligned(input.add(step * 2).cast()) };

    let bit = _mm256_set1_epi8(0b001);
    let mut b0_bits_packed1 = expand_mask_epi8(mask1 as __mmask32);
    let mut b0_bits_packed2 = expand_mask_epi8((mask1 >> 32) as __mmask32);
    b0_bits_packed1 = _mm256_and_si256(b0_bits_packed1, bit);
    b0_bits_packed2 = _mm256_and_si256(b0_bits_packed2, bit);

    let bit = _mm256_set1_epi8(0b010);
    let mut b1_bits_packed1 = expand_mask_epi8(mask2 as __mmask32);
    let mut b1_bits_packed2 = expand_mask_epi8((mask2 >> 32) as __mmask32);
    b1_bits_packed1 = _mm256_and_si256(b1_bits_packed1, bit);
    b1_bits_packed2 = _mm256_and_si256(b1_bits_packed2, bit);

    let bit = _mm256_set1_epi8(0b100);
    let mut b2_bits_packed1 = expand_mask_epi8(mask3 as __mmask32);
    let mut b2_bits_packed2 = expand_mask_epi8((mask3 >> 32) as __mmask32);
    b2_bits_packed1 = _mm256_and_si256(b2_bits_packed1, bit);
    b2_bits_packed2 = _mm256_and_si256(b2_bits_packed2, bit);

    let mut three_bits1 = _mm256_or_si256(b1_bits_packed1, b0_bits_packed1);
    let mut three_bits2 = _mm256_or_si256(b1_bits_packed2, b0_bits_packed2);
    three_bits1 = _mm256_or_si256(three_bits1, b2_bits_packed1);
    three_bits2 = _mm256_or_si256(three_bits2, b2_bits_packed2);

    [three_bits1, three_bits2]
}

#[target_feature(enable = "avx2")]
/// Unpack the 4-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(4)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
/// - `read_n` must be between no larger than `64`.
pub unsafe fn from_u4(input: *const u8, read_n: usize) -> [__m256i; 8] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let packed = unsafe { unpack_u4_registers(input) };
    unpack_u8_to_u32_ordered(packed)
}

#[target_feature(enable = "avx2")]
/// Unpack eight registers containing 8 32-bit elements from a 4-bit nibbles provided
/// by `input`.
pub(super) unsafe fn unpack_u4_registers(input: *const u8) -> [__m256i; 2] {
    let ordered = unsafe { _mm256_loadu_si256(input.cast()) };
    let interleaved = _mm256_permute4x64_epi64::<0xD8>(ordered);

    let low_mask = _mm256_set1_epi8(0x0F);
    let low_nibbles = _mm256_and_si256(interleaved, low_mask);
    let high_nibbles = _mm256_and_si256(_mm256_srli_epi16(interleaved, 4), low_mask);

    let d1 = _mm256_unpacklo_epi8(low_nibbles, high_nibbles);
    let d2 = _mm256_unpackhi_epi8(low_nibbles, high_nibbles);

    [d1, d2]
}

#[target_feature(enable = "avx2")]
/// Unpack the 5-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(5)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
/// - `read_n` must be between no larger than `64`.
pub unsafe fn from_u5(input: *const u8, read_n: usize) -> [__m256i; 8] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let packed = unsafe { unpack_u5_registers(input, read_n) };
    unpack_u8_to_u32_ordered(packed)
}

#[target_feature(enable = "avx2")]
/// Unpack eight registers containing 8 32-bit elements from a 5-bit integers provided
/// by `input`.
unsafe fn unpack_u5_registers(input: *const u8, read_n: usize) -> [__m256i; 2] {
    let offset = read_n.div_ceil(2);
    let lo_bits = unsafe { unpack_u4_registers(input.add(0)) };
    let mut hi_bits = unsafe { unpack_u1_registers(input.add(offset)) };
    hi_bits = slli_epi16::<4, 2>(hi_bits);
    or_si256_all(hi_bits, lo_bits)
}

#[target_feature(enable = "avx2")]
/// Unpack the 6-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(6)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
/// - `read_n` must be between no larger than `64`.
pub unsafe fn from_u6(input: *const u8, read_n: usize) -> [__m256i; 8] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let packed = unsafe { unpack_u6_registers(input, read_n) };
    unpack_u8_to_u32_ordered(packed)
}

#[target_feature(enable = "avx2")]
/// Unpack eight registers containing 8 32-bit elements from a 6-bit integers provided
/// by `input`.
unsafe fn unpack_u6_registers(input: *const u8, read_n: usize) -> [__m256i; 2] {
    let offset = read_n.div_ceil(2);
    let lo_bits = unsafe { unpack_u4_registers(input.add(0)) };
    let mut hi_bits = unsafe { unpack_u2_registers(input.add(offset), read_n) };
    hi_bits = slli_epi16::<4, 2>(hi_bits);
    or_si256_all(hi_bits, lo_bits)
}

#[target_feature(enable = "avx2")]
/// Unpack the 7-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(7)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
/// - `read_n` must be between no larger than `64`.
pub unsafe fn from_u7(input: *const u8, read_n: usize) -> [__m256i; 8] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let packed = unsafe { unpack_u7_registers(input, read_n) };
    unpack_u8_to_u32_ordered(packed)
}

#[target_feature(enable = "avx2")]
/// Unpack eight registers containing 8 32-bit elements from a 7-bit integers provided
/// by `input`.
unsafe fn unpack_u7_registers(input: *const u8, read_n: usize) -> [__m256i; 2] {
    let offset = read_n.div_ceil(2);
    let lo_bits = unsafe { unpack_u4_registers(input.add(0)) };
    let mut hi_bits = unsafe { unpack_u3_registers(input.add(offset), read_n) };
    hi_bits = slli_epi16::<4, 2>(hi_bits);
    or_si256_all(hi_bits, lo_bits)
}

#[target_feature(enable = "avx2")]
/// Unpack the 8-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(8)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
/// - `read_n` must be between no larger than `64`.
pub unsafe fn from_u8(input: *const u8, read_n: usize) -> [__m256i; 8] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let packed = unsafe { load_si256x2(input) };
    unpack_u8_to_u32_ordered(packed)
}

#[target_feature(enable = "avx2")]
/// Unpack the 9-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(9)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
/// - `read_n` must be between no larger than `64`.
pub unsafe fn from_u9(input: *const u8, read_n: usize) -> [__m256i; 8] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let lo_bits = unsafe { load_si256x2(input.add(0)) };
    let lo_bits = unpack_u8_to_u16_ordered(lo_bits);

    let offset = read_n;
    let hi_bits = unsafe { unpack_u1_registers(input.add(offset)) };
    let mut hi_bits = unpack_u8_to_u16_ordered(hi_bits);
    hi_bits = slli_epi16::<8, 4>(hi_bits);

    let packed = or_si256_all(hi_bits, lo_bits);
    unpack_u16_to_u32_ordered(packed)
}

#[target_feature(enable = "avx2")]
/// Unpack the 10-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(10)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
/// - `read_n` must be between no larger than `64`.
pub unsafe fn from_u10(input: *const u8, read_n: usize) -> [__m256i; 8] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let lo_bits = unsafe { load_si256x2(input.add(0)) };
    let lo_bits = unpack_u8_to_u16_ordered(lo_bits);

    let offset = read_n;
    let hi_bits = unsafe { unpack_u2_registers(input.add(offset), read_n) };
    let mut hi_bits = unpack_u8_to_u16_ordered(hi_bits);
    hi_bits = slli_epi16::<8, 4>(hi_bits);

    let packed = or_si256_all(hi_bits, lo_bits);
    unpack_u16_to_u32_ordered(packed)
}

#[target_feature(enable = "avx2")]
/// Unpack the 11-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(11)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
/// - `read_n` must be between no larger than `64`.
pub unsafe fn from_u11(input: *const u8, read_n: usize) -> [__m256i; 8] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let lo_bits = unsafe { load_si256x2(input.add(0)) };
    let lo_bits = unpack_u8_to_u16_ordered(lo_bits);

    let offset = read_n;
    let hi_bits = unsafe { unpack_u3_registers(input.add(offset), read_n) };
    let mut hi_bits = unpack_u8_to_u16_ordered(hi_bits);
    hi_bits = slli_epi16::<8, 4>(hi_bits);

    let packed = or_si256_all(hi_bits, lo_bits);
    unpack_u16_to_u32_ordered(packed)
}

#[target_feature(enable = "avx2")]
/// Unpack the 12-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(12)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
/// - `read_n` must be between no larger than `64`.
pub unsafe fn from_u12(input: *const u8, read_n: usize) -> [__m256i; 8] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let lo_bits = unsafe { load_si256x2(input.add(0)) };
    let lo_bits = unpack_u8_to_u16_ordered(lo_bits);

    let offset = read_n;
    let hi_bits = unsafe { unpack_u4_registers(input.add(offset)) };
    let mut hi_bits = unpack_u8_to_u16_ordered(hi_bits);
    hi_bits = slli_epi16::<8, 4>(hi_bits);

    let packed = or_si256_all(hi_bits, lo_bits);
    unpack_u16_to_u32_ordered(packed)
}

#[target_feature(enable = "avx2")]
/// Unpack the 13-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(13)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
/// - `read_n` must be between no larger than `64`.
pub unsafe fn from_u13(input: *const u8, read_n: usize) -> [__m256i; 8] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let lo_bits = unsafe { load_si256x2(input.add(0)) };
    let lo_bits = unpack_u8_to_u16_ordered(lo_bits);

    let offset = read_n;
    let hi_bits = unsafe { unpack_u5_registers(input.add(offset), read_n) };
    let mut hi_bits = unpack_u8_to_u16_ordered(hi_bits);
    hi_bits = slli_epi16::<8, 4>(hi_bits);

    let packed = or_si256_all(hi_bits, lo_bits);
    unpack_u16_to_u32_ordered(packed)
}

#[target_feature(enable = "avx2")]
/// Unpack the 14-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(14)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
/// - `read_n` must be between no larger than `64`.
pub unsafe fn from_u14(input: *const u8, read_n: usize) -> [__m256i; 8] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let lo_bits = unsafe { load_si256x2(input.add(0)) };
    let lo_bits = unpack_u8_to_u16_ordered(lo_bits);

    let offset = read_n;
    let hi_bits = unsafe { unpack_u6_registers(input.add(offset), read_n) };
    let mut hi_bits = unpack_u8_to_u16_ordered(hi_bits);
    hi_bits = slli_epi16::<8, 4>(hi_bits);

    let packed = or_si256_all(hi_bits, lo_bits);
    unpack_u16_to_u32_ordered(packed)
}

#[target_feature(enable = "avx2")]
/// Unpack the 15-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(15)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
/// - `read_n` must be between no larger than `64`.
pub unsafe fn from_u15(input: *const u8, read_n: usize) -> [__m256i; 8] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let lo_bits = unsafe { load_si256x2(input.add(0)) };
    let lo_bits = unpack_u8_to_u16_ordered(lo_bits);

    let offset = read_n;
    let hi_bits = unsafe { unpack_u7_registers(input.add(offset), read_n) };
    let mut hi_bits = unpack_u8_to_u16_ordered(hi_bits);
    hi_bits = slli_epi16::<8, 4>(hi_bits);

    let packed = or_si256_all(hi_bits, lo_bits);
    unpack_u16_to_u32_ordered(packed)
}

#[target_feature(enable = "avx2")]
/// Unpack the 16-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(16)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
/// - `read_n` must be between no larger than `64`.
pub unsafe fn from_u16(input: *const u8, read_n: usize) -> [__m256i; 8] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let packed = unsafe { load_si256x4(input.add(0)) };
    unpack_u16_to_u32_ordered(packed)
}

#[target_feature(enable = "avx2")]
/// Unpack the 17-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(17)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
/// - `read_n` must be between no larger than `64`.
pub unsafe fn from_u17(input: *const u8, read_n: usize) -> [__m256i; 8] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let lo_bits = unsafe { load_si256x4(input.add(0)) };
    let lo_bits = unpack_u16_to_u32_ordered(lo_bits);

    let offset = read_n * 2;
    let hi_bits = unsafe { unpack_u1_registers(input.add(offset)) };
    let mut hi_bits = unpack_u8_to_u32_ordered(hi_bits);
    hi_bits = slli_epi32::<16, 8>(hi_bits);

    or_si256_all(hi_bits, lo_bits)
}

#[target_feature(enable = "avx2")]
/// Unpack the 18-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(18)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
/// - `read_n` must be between no larger than `64`.
pub unsafe fn from_u18(input: *const u8, read_n: usize) -> [__m256i; 8] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let lo_bits = unsafe { load_si256x4(input.add(0)) };
    let lo_bits = unpack_u16_to_u32_ordered(lo_bits);

    let offset = read_n * 2;
    let hi_bits = unsafe { unpack_u2_registers(input.add(offset), read_n) };
    let mut hi_bits = unpack_u8_to_u32_ordered(hi_bits);
    hi_bits = slli_epi32::<16, 8>(hi_bits);

    or_si256_all(hi_bits, lo_bits)
}

#[target_feature(enable = "avx2")]
/// Unpack the 19-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(19)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
/// - `read_n` must be between no larger than `64`.
pub unsafe fn from_u19(input: *const u8, read_n: usize) -> [__m256i; 8] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let lo_bits = unsafe { load_si256x4(input.add(0)) };
    let lo_bits = unpack_u16_to_u32_ordered(lo_bits);

    let offset = read_n * 2;
    let hi_bits = unsafe { unpack_u3_registers(input.add(offset), read_n) };
    let mut hi_bits = unpack_u8_to_u32_ordered(hi_bits);
    hi_bits = slli_epi32::<16, 8>(hi_bits);

    or_si256_all(hi_bits, lo_bits)
}

#[target_feature(enable = "avx2")]
/// Unpack the 20-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(20)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
/// - `read_n` must be between no larger than `64`.
pub unsafe fn from_u20(input: *const u8, read_n: usize) -> [__m256i; 8] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let lo_bits = unsafe { load_si256x4(input.add(0)) };
    let lo_bits = unpack_u16_to_u32_ordered(lo_bits);

    let offset = read_n * 2;
    let hi_bits = unsafe { unpack_u4_registers(input.add(offset)) };
    let mut hi_bits = unpack_u8_to_u32_ordered(hi_bits);
    hi_bits = slli_epi32::<16, 8>(hi_bits);

    or_si256_all(hi_bits, lo_bits)
}

#[target_feature(enable = "avx2")]
/// Unpack the 21-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(21)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
/// - `read_n` must be between no larger than `64`.
pub unsafe fn from_u21(input: *const u8, read_n: usize) -> [__m256i; 8] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let lo_bits = unsafe { load_si256x4(input.add(0)) };
    let lo_bits = unpack_u16_to_u32_ordered(lo_bits);

    let offset = read_n * 2;
    let hi_bits = unsafe { unpack_u5_registers(input.add(offset), read_n) };
    let mut hi_bits = unpack_u8_to_u32_ordered(hi_bits);
    hi_bits = slli_epi32::<16, 8>(hi_bits);

    or_si256_all(hi_bits, lo_bits)
}

#[target_feature(enable = "avx2")]
/// Unpack the 22-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(22)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
/// - `read_n` must be between no larger than `64`.
pub unsafe fn from_u22(input: *const u8, read_n: usize) -> [__m256i; 8] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let lo_bits = unsafe { load_si256x4(input.add(0)) };
    let lo_bits = unpack_u16_to_u32_ordered(lo_bits);

    let offset = read_n * 2;
    let hi_bits = unsafe { unpack_u6_registers(input.add(offset), read_n) };
    let mut hi_bits = unpack_u8_to_u32_ordered(hi_bits);
    hi_bits = slli_epi32::<16, 8>(hi_bits);

    or_si256_all(hi_bits, lo_bits)
}

#[target_feature(enable = "avx2")]
/// Unpack the 23-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(23)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
/// - `read_n` must be between no larger than `64`.
pub unsafe fn from_u23(input: *const u8, read_n: usize) -> [__m256i; 8] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let lo_bits = unsafe { load_si256x4(input.add(0)) };
    let lo_bits = unpack_u16_to_u32_ordered(lo_bits);

    let offset = read_n * 2;
    let hi_bits = unsafe { unpack_u7_registers(input.add(offset), read_n) };
    let mut hi_bits = unpack_u8_to_u32_ordered(hi_bits);
    hi_bits = slli_epi32::<16, 8>(hi_bits);

    or_si256_all(hi_bits, lo_bits)
}

#[target_feature(enable = "avx2")]
/// Unpack the 24-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(24)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
/// - `read_n` must be between no larger than `64`.
pub unsafe fn from_u24(input: *const u8, read_n: usize) -> [__m256i; 8] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let lo_bits = unsafe { load_si256x4(input.add(0)) };
    let lo_bits = unpack_u16_to_u32_ordered(lo_bits);

    let offset = read_n * 2;
    let hi_bits = unsafe { load_si256x2(input.add(offset)) };
    let mut hi_bits = unpack_u8_to_u32_ordered(hi_bits);
    hi_bits = slli_epi32::<16, 8>(hi_bits);

    or_si256_all(hi_bits, lo_bits)
}

#[target_feature(enable = "avx2")]
/// Unpack the 25-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(25)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
/// - `read_n` must be between no larger than `64`.
pub unsafe fn from_u25(input: *const u8, read_n: usize) -> [__m256i; 8] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let lo_bits = unsafe { load_si256x4(input.add(0)) };
    let lo_bits = unpack_u16_to_u32_ordered(lo_bits);

    let mut offset = read_n * 2;
    let hi_8bits = unsafe { load_si256x2(input.add(offset)) };
    let hi_8bits = unpack_u8_to_u16_ordered(hi_8bits);

    offset += read_n;
    let hi_1bits = unsafe { unpack_u1_registers(input.add(offset)) };
    let mut hi_1bits = unpack_u8_to_u16_ordered(hi_1bits);
    hi_1bits = slli_epi16::<8, 4>(hi_1bits);

    let hi_bits = or_si256_all(hi_1bits, hi_8bits);
    let mut hi_bits = unpack_u16_to_u32_ordered(hi_bits);
    hi_bits = slli_epi32::<16, 8>(hi_bits);

    or_si256_all(hi_bits, lo_bits)
}

#[target_feature(enable = "avx2")]
/// Unpack the 26-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(26)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
/// - `read_n` must be between no larger than `64`.
pub unsafe fn from_u26(input: *const u8, read_n: usize) -> [__m256i; 8] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let lo_bits = unsafe { load_si256x4(input.add(0)) };
    let lo_bits = unpack_u16_to_u32_ordered(lo_bits);

    let mut offset = read_n * 2;
    let hi_8bits = unsafe { load_si256x2(input.add(offset)) };
    let hi_8bits = unpack_u8_to_u16_ordered(hi_8bits);

    offset += read_n;
    let hi_1bits = unsafe { unpack_u2_registers(input.add(offset), read_n) };
    let mut hi_1bits = unpack_u8_to_u16_ordered(hi_1bits);
    hi_1bits = slli_epi16::<8, 4>(hi_1bits);

    let hi_bits = or_si256_all(hi_1bits, hi_8bits);
    let mut hi_bits = unpack_u16_to_u32_ordered(hi_bits);
    hi_bits = slli_epi32::<16, 8>(hi_bits);

    or_si256_all(hi_bits, lo_bits)
}

#[target_feature(enable = "avx2")]
/// Unpack the 27-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(27)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
/// - `read_n` must be between no larger than `64`.
pub unsafe fn from_u27(input: *const u8, read_n: usize) -> [__m256i; 8] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let lo_bits = unsafe { load_si256x4(input.add(0)) };
    let lo_bits = unpack_u16_to_u32_ordered(lo_bits);

    let mut offset = read_n * 2;
    let hi_8bits = unsafe { load_si256x2(input.add(offset)) };
    let hi_8bits = unpack_u8_to_u16_ordered(hi_8bits);

    offset += read_n;
    let hi_1bits = unsafe { unpack_u3_registers(input.add(offset), read_n) };
    let mut hi_1bits = unpack_u8_to_u16_ordered(hi_1bits);
    hi_1bits = slli_epi16::<8, 4>(hi_1bits);

    let hi_bits = or_si256_all(hi_1bits, hi_8bits);
    let mut hi_bits = unpack_u16_to_u32_ordered(hi_bits);
    hi_bits = slli_epi32::<16, 8>(hi_bits);

    or_si256_all(hi_bits, lo_bits)
}

#[target_feature(enable = "avx2")]
/// Unpack the 28-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(28)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
/// - `read_n` must be between no larger than `64`.
pub unsafe fn from_u28(input: *const u8, read_n: usize) -> [__m256i; 8] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let lo_bits = unsafe { load_si256x4(input.add(0)) };
    let lo_bits = unpack_u16_to_u32_ordered(lo_bits);

    let mut offset = read_n * 2;
    let hi_8bits = unsafe { load_si256x2(input.add(offset)) };
    let hi_8bits = unpack_u8_to_u16_ordered(hi_8bits);

    offset += read_n;
    let hi_1bits = unsafe { unpack_u4_registers(input.add(offset)) };
    let mut hi_1bits = unpack_u8_to_u16_ordered(hi_1bits);
    hi_1bits = slli_epi16::<8, 4>(hi_1bits);

    let hi_bits = or_si256_all(hi_1bits, hi_8bits);
    let mut hi_bits = unpack_u16_to_u32_ordered(hi_bits);
    hi_bits = slli_epi32::<16, 8>(hi_bits);

    or_si256_all(hi_bits, lo_bits)
}

#[target_feature(enable = "avx2")]
/// Unpack the 29-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(29)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
/// - `read_n` must be between no larger than `64`.
pub unsafe fn from_u29(input: *const u8, read_n: usize) -> [__m256i; 8] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let lo_bits = unsafe { load_si256x4(input.add(0)) };
    let lo_bits = unpack_u16_to_u32_ordered(lo_bits);

    let mut offset = read_n * 2;
    let hi_8bits = unsafe { load_si256x2(input.add(offset)) };
    let hi_8bits = unpack_u8_to_u16_ordered(hi_8bits);

    offset += read_n;
    let hi_1bits = unsafe { unpack_u5_registers(input.add(offset), read_n) };
    let mut hi_1bits = unpack_u8_to_u16_ordered(hi_1bits);
    hi_1bits = slli_epi16::<8, 4>(hi_1bits);

    let hi_bits = or_si256_all(hi_1bits, hi_8bits);
    let mut hi_bits = unpack_u16_to_u32_ordered(hi_bits);
    hi_bits = slli_epi32::<16, 8>(hi_bits);

    or_si256_all(hi_bits, lo_bits)
}

#[target_feature(enable = "avx2")]
/// Unpack the 30-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(30)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
/// - `read_n` must be between no larger than `64`.
pub unsafe fn from_u30(input: *const u8, read_n: usize) -> [__m256i; 8] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let lo_bits = unsafe { load_si256x4(input.add(0)) };
    let lo_bits = unpack_u16_to_u32_ordered(lo_bits);

    let mut offset = read_n * 2;
    let hi_8bits = unsafe { load_si256x2(input.add(offset)) };
    let hi_8bits = unpack_u8_to_u16_ordered(hi_8bits);

    offset += read_n;
    let hi_1bits = unsafe { unpack_u6_registers(input.add(offset), read_n) };
    let mut hi_1bits = unpack_u8_to_u16_ordered(hi_1bits);
    hi_1bits = slli_epi16::<8, 4>(hi_1bits);

    let hi_bits = or_si256_all(hi_1bits, hi_8bits);
    let mut hi_bits = unpack_u16_to_u32_ordered(hi_bits);
    hi_bits = slli_epi32::<16, 8>(hi_bits);

    or_si256_all(hi_bits, lo_bits)
}

#[target_feature(enable = "avx2")]
/// Unpack the 31-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(31)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
pub unsafe fn from_u31(input: *const u8, read_n: usize) -> [__m256i; 8] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let lo_bits = unsafe { load_si256x4(input.add(0)) };
    let lo_bits = unpack_u16_to_u32_ordered(lo_bits);

    let mut offset = read_n * 2;
    let hi_8bits = unsafe { load_si256x2(input.add(offset)) };
    let hi_8bits = unpack_u8_to_u16_ordered(hi_8bits);

    offset += read_n;
    let hi_1bits = unsafe { unpack_u7_registers(input.add(offset), read_n) };
    let mut hi_1bits = unpack_u8_to_u16_ordered(hi_1bits);
    hi_1bits = slli_epi16::<8, 4>(hi_1bits);

    let hi_bits = or_si256_all(hi_1bits, hi_8bits);
    let mut hi_bits = unpack_u16_to_u32_ordered(hi_bits);
    hi_bits = slli_epi32::<16, 8>(hi_bits);

    or_si256_all(hi_bits, lo_bits)
}

#[target_feature(enable = "avx2")]
/// Unpack the 32-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(32)`
///   bytes from.
/// - The runtime CPU must support the `avx2` instructions.
/// - `read_n` must be between no larger than `64`.
pub unsafe fn from_u32(input: *const u8, read_n: usize) -> [__m256i; 8] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    unsafe { load_si256x8(input) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::X64;
    use crate::uint32::avx2::pack_x64_partial::*;
    use crate::uint32::{X128_MAX_OUTPUT_LEN, max_compressed_size};

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
    #[cfg_attr(not(target_feature = "avx2"), ignore)]
    fn test_saturated_unpack(
        #[case] bit_len: u8,
        #[case] unpacker: unsafe fn(*const u8, usize) -> [__m256i; 8],
    ) {
        let saturated_bytes = [u8::MAX; X128_MAX_OUTPUT_LEN / 2];

        let unpacked = unsafe { unpacker(saturated_bytes.as_ptr(), X64) };
        let unpacked = unsafe { std::mem::transmute::<[__m256i; 8], [u32; X64]>(unpacked) };

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
    #[cfg_attr(not(target_feature = "avx2"), ignore)]
    fn test_pack_unpack(
        #[case] bit_len: u8,
        #[case] packer: unsafe fn(*mut u8, [__m256i; 8], usize),
        #[case] unpacker: unsafe fn(*const u8, usize) -> [__m256i; 8],
    ) {
        fastrand::seed(5876358762523525);

        let max_value = (2u64.pow(bit_len as u32) - 1) as u32;

        let mut values = [0; X64];
        for value in values.iter_mut() {
            *value = fastrand::u32(0..max_value);
        }
        let data = unsafe { load_u32x64(&values) };

        let mut packed = [0; X128_MAX_OUTPUT_LEN / 2];
        unsafe { packer(packed.as_mut_ptr(), data, X64) };

        let unpacked = unsafe { unpacker(packed.as_ptr(), X64) };
        let unpacked = unsafe { std::mem::transmute::<[__m256i; 8], [u32; X64]>(unpacked) };
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
    #[cfg_attr(not(target_feature = "avx2"), ignore)]
    fn test_pack_unpack_length_permutations(
        #[case] bit_len: u8,
        #[case] packer: unsafe fn(*mut u8, [__m256i; 8], usize),
        #[case] unpacker: unsafe fn(*const u8, usize) -> [__m256i; 8],
    ) {
        fastrand::seed(5876358762523525);

        let max_value = (2u64.pow(bit_len as u32) - 1) as u32;

        for length in 0..X64 {
            let mut values = [0; X64];
            for value in values.iter_mut() {
                *value = fastrand::u32(0..max_value);
            }
            let data = unsafe { load_u32x64(&values) };

            let mut packed = [0; X128_MAX_OUTPUT_LEN / 2];
            unsafe { packer(packed.as_mut_ptr(), data, length) };
            packed[max_compressed_size::<X64>(bit_len as usize)..].fill(0);

            let unpacked = unsafe { unpacker(packed.as_ptr(), length) };
            let unpacked = unsafe { std::mem::transmute::<[__m256i; 8], [u32; X64]>(unpacked) };
            assert_eq!(unpacked[..length], values[..length], "length:{length}");
        }
    }
}
