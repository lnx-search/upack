use std::arch::x86_64::*;

use super::data::*;
use super::util::*;

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 1-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(1)` bytes to.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(crate) unsafe fn to_u1(out: *mut u8, block: [__m512i; 4], _pack_n: usize) {
    let partially_packed = pack_u32_to_u8_ordered(block);
    unsafe { pack_u1_registers(out, partially_packed) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Pack a register containing 32 8-bit elements each into a 1-bit
/// bitmap and write to `out`.
///
/// Any non-zero value will be treated as a set bit.
unsafe fn pack_u1_registers(out: *mut u8, data: __m512i) {
    let bits = _mm512_slli_epi16::<7>(data);
    let mask = _mm512_movepi8_mask(bits);
    unsafe { std::ptr::write_unaligned(out.add(0).cast(), mask) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 2-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(2)` bytes to.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(crate) unsafe fn to_u2(out: *mut u8, block: [__m512i; 4], pack_n: usize) {
    let partially_packed = pack_u32_to_u8_ordered(block);
    unsafe { pack_u2_registers(out, partially_packed, pack_n) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Pack a register containing 32 8-bit elements each into a 2-bit
/// bitmap and write to `out`.
///
/// Any non-zero value will be treated as a set bit.
unsafe fn pack_u2_registers(out: *mut u8, data: __m512i, pack_n: usize) {
    let lo_bits = _mm512_slli_epi16::<7>(data);
    let hi_bits = _mm512_slli_epi16::<6>(data);

    let lo_mask = _mm512_movepi8_mask(lo_bits);
    let hi_mask = _mm512_movepi8_mask(hi_bits);

    let hi_offset = pack_n.div_ceil(8);
    unsafe { std::ptr::write_unaligned(out.add(0).cast(), lo_mask) };
    unsafe { std::ptr::write_unaligned(out.add(hi_offset).cast(), hi_mask) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 3-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(3)` bytes to.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(crate) unsafe fn to_u3(out: *mut u8, block: [__m512i; 4], pack_n: usize) {
    let partially_packed = pack_u32_to_u8_ordered(block);
    unsafe { pack_u3_registers(out, partially_packed, pack_n) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Pack a register containing 32 8-bit elements each into a 3-bit
/// bitmap and write to `out`.
///
/// Any non-zero value will be treated as a set bit.
unsafe fn pack_u3_registers(out: *mut u8, data: __m512i, pack_n: usize) {
    let b0_bits = _mm512_slli_epi16::<7>(data);
    let b1_bits = _mm512_slli_epi16::<6>(data);
    let b2_bits = _mm512_slli_epi16::<5>(data);

    let b0_mask = _mm512_movepi8_mask(b0_bits);
    let b1_mask = _mm512_movepi8_mask(b1_bits);
    let b2_mask = _mm512_movepi8_mask(b2_bits);

    let step = pack_n.div_ceil(8);
    unsafe { std::ptr::write_unaligned(out.add(0).cast(), b0_mask) };
    unsafe { std::ptr::write_unaligned(out.add(step).cast(), b1_mask) };
    unsafe { std::ptr::write_unaligned(out.add(step * 2).cast(), b2_mask) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 4-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(4)` bytes to.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(crate) unsafe fn to_u4(out: *mut u8, block: [__m512i; 4], _pack_n: usize) {
    let partially_packed = pack_u32_to_u8_ordered(block);
    unsafe { pack_u4_registers(out, partially_packed) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Pack a register containing 32 8-bit elements each into a 4-bit
/// bitmap and write to `out`.
///
/// Any non-zero value will be treated as a set bit.
pub(super) unsafe fn pack_u4_registers(out: *mut u8, data: __m512i) {
    let madd_multiplier = _mm512_set1_epi16(0x1001);
    let nibbles = _mm512_maddubs_epi16(data, madd_multiplier);
    let ordered = _mm512_cvtepi16_epi8(nibbles);
    unsafe { _mm256_storeu_si256(out.cast(), ordered) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 5-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(5)` bytes to.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(crate) unsafe fn to_u5(out: *mut u8, block: [__m512i; 4], pack_n: usize) {
    let partially_packed = pack_u32_to_u8_ordered(block);
    unsafe { pack_u5_registers(out, partially_packed, pack_n) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Pack a register containing 32 8-bit elements each into a 5-bit
/// bitmap and write to `out`.
///
/// Any non-zero value will be treated as a set bit.
unsafe fn pack_u5_registers(out: *mut u8, data: __m512i, pack_n: usize) {
    let mask = _mm512_set1_epi8(0b1111);
    let masked = _mm512_and_si512(data, mask);
    unsafe { pack_u4_registers(out.add(0), masked) };

    // 4bit * 64 / 8-bits per byte.
    let offset = pack_n.div_ceil(2);
    let remaining = _mm512_srli_epi16::<4>(data);
    unsafe { pack_u1_registers(out.add(offset), remaining) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 6-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(6)` bytes to.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(crate) unsafe fn to_u6(out: *mut u8, block: [__m512i; 4], pack_n: usize) {
    let partially_packed = pack_u32_to_u8_ordered(block);
    unsafe { pack_u6_registers(out, partially_packed, pack_n) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Pack a register containing 32 8-bit elements each into a 6-bit
/// bitmap and write to `out`.
///
/// Any non-zero value will be treated as a set bit.
unsafe fn pack_u6_registers(out: *mut u8, data: __m512i, pack_n: usize) {
    let mask = _mm512_set1_epi8(0b1111);
    let masked = _mm512_and_si512(data, mask);
    unsafe { pack_u4_registers(out, masked) };

    // 4bit * 64 / 8-bits per byte.
    let offset = pack_n.div_ceil(2);
    let remaining = _mm512_srli_epi16::<4>(data);
    unsafe { pack_u2_registers(out.add(offset), remaining, pack_n) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 7-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(7)` bytes to.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(crate) unsafe fn to_u7(out: *mut u8, block: [__m512i; 4], pack_n: usize) {
    let partially_packed = pack_u32_to_u8_ordered(block);
    unsafe { pack_u7_registers(out, partially_packed, pack_n) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Pack a register containing 32 8-bit elements each into a 7-bit
/// bitmap and write to `out`.
///
/// Any non-zero value will be treated as a set bit.
unsafe fn pack_u7_registers(out: *mut u8, data: __m512i, pack_n: usize) {
    let mask = _mm512_set1_epi8(0b1111);
    let masked = _mm512_and_si512(data, mask);
    unsafe { pack_u4_registers(out, masked) };

    // 4bit * 64 / 8-bits per byte.
    let offset = pack_n.div_ceil(2);
    let remaining = _mm512_srli_epi16::<4>(data);
    unsafe { pack_u3_registers(out.add(offset), remaining, pack_n) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 8-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(8)` bytes to.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(crate) unsafe fn to_u8(out: *mut u8, block: [__m512i; 4], _pack_n: usize) {
    let partially_packed = pack_u32_to_u8_ordered(block);
    unsafe { _mm512_storeu_epi8(out.cast(), partially_packed) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 9-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(9)` bytes to.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(crate) unsafe fn to_u9(out: *mut u8, block: [__m512i; 4], pack_n: usize) {
    let (hi, lo) = pack_u32_to_u16_split_ordered(block);
    unsafe { _mm512_storeu_epi8(out.add(0).cast(), lo) };
    unsafe { pack_u1_registers(out.add(pack_n), hi) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 10-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(10)` bytes to.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(crate) unsafe fn to_u10(out: *mut u8, block: [__m512i; 4], pack_n: usize) {
    let (hi, lo) = pack_u32_to_u16_split_ordered(block);
    unsafe { _mm512_storeu_epi8(out.add(0).cast(), lo) };
    unsafe { pack_u2_registers(out.add(pack_n), hi, pack_n) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 11-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(11)` bytes to.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(crate) unsafe fn to_u11(out: *mut u8, block: [__m512i; 4], pack_n: usize) {
    let (hi, lo) = pack_u32_to_u16_split_ordered(block);
    unsafe { _mm512_storeu_epi8(out.add(0).cast(), lo) };
    unsafe { pack_u3_registers(out.add(pack_n), hi, pack_n) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 12-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(12)` bytes to.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(crate) unsafe fn to_u12(out: *mut u8, block: [__m512i; 4], pack_n: usize) {
    let (hi, lo) = pack_u32_to_u16_split_ordered(block);
    unsafe { _mm512_storeu_epi8(out.add(0).cast(), lo) };
    unsafe { pack_u4_registers(out.add(pack_n), hi) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 13-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(13)` bytes to.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(crate) unsafe fn to_u13(out: *mut u8, block: [__m512i; 4], pack_n: usize) {
    let (hi, lo) = pack_u32_to_u16_split_ordered(block);
    unsafe { _mm512_storeu_epi8(out.add(0).cast(), lo) };
    unsafe { pack_u5_registers(out.add(pack_n), hi, pack_n) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 14-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(14)` bytes to.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(crate) unsafe fn to_u14(out: *mut u8, block: [__m512i; 4], pack_n: usize) {
    let (hi, lo) = pack_u32_to_u16_split_ordered(block);
    unsafe { _mm512_storeu_epi8(out.add(0).cast(), lo) };
    unsafe { pack_u6_registers(out.add(pack_n), hi, pack_n) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 15-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(15)` bytes to.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(crate) unsafe fn to_u15(out: *mut u8, block: [__m512i; 4], pack_n: usize) {
    let (hi, lo) = pack_u32_to_u16_split_ordered(block);
    unsafe { _mm512_storeu_epi8(out.add(0).cast(), lo) };
    unsafe { pack_u7_registers(out.add(pack_n), hi, pack_n) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 16-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(16)` bytes to.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(crate) unsafe fn to_u16(out: *mut u8, block: [__m512i; 4], _pack_n: usize) {
    let packed = pack_u32_to_u16_ordered(block);
    unsafe { store_si512x2(out, packed) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
unsafe fn store_lo_u16_registers(out: *mut u8, data: [__m512i; 4]) {
    let mask = _mm512_set1_epi32(0xFFFF);
    let shifted = and_si512(data, mask);
    let packed = pack_u32_to_u16_ordered(shifted);
    unsafe { store_si512x2(out, packed) }
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 17-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(17)` bytes to.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(crate) unsafe fn to_u17(out: *mut u8, block: [__m512i; 4], pack_n: usize) {
    unsafe { store_lo_u16_registers(out.add(0), block) };

    let hi_bits = srli_epi32::<16, 4>(block);
    let hi_bits = pack_u32_to_u8_ordered(hi_bits);
    let offset = pack_n * 2;
    unsafe { pack_u1_registers(out.add(offset), hi_bits) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 18-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(18)` bytes to.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(crate) unsafe fn to_u18(out: *mut u8, block: [__m512i; 4], pack_n: usize) {
    unsafe { store_lo_u16_registers(out.add(0), block) };

    let hi_bits = srli_epi32::<16, 4>(block);
    let hi_bits = pack_u32_to_u8_ordered(hi_bits);
    let offset = pack_n * 2;
    unsafe { pack_u2_registers(out.add(offset), hi_bits, pack_n) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 19-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(19)` bytes to.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(crate) unsafe fn to_u19(out: *mut u8, block: [__m512i; 4], pack_n: usize) {
    unsafe { store_lo_u16_registers(out.add(0), block) };

    let hi_bits = srli_epi32::<16, 4>(block);
    let hi_bits = pack_u32_to_u8_ordered(hi_bits);
    let offset = pack_n * 2;
    unsafe { pack_u3_registers(out.add(offset), hi_bits, pack_n) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 20-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(20)` bytes to.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(crate) unsafe fn to_u20(out: *mut u8, block: [__m512i; 4], pack_n: usize) {
    unsafe { store_lo_u16_registers(out.add(0), block) };

    let hi_bits = srli_epi32::<16, 4>(block);
    let hi_bits = pack_u32_to_u8_ordered(hi_bits);
    let offset = pack_n * 2;
    unsafe { pack_u4_registers(out.add(offset), hi_bits) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 21-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(21)` bytes to.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(crate) unsafe fn to_u21(out: *mut u8, block: [__m512i; 4], pack_n: usize) {
    unsafe { store_lo_u16_registers(out.add(0), block) };

    let hi_bits = srli_epi32::<16, 4>(block);
    let hi_bits = pack_u32_to_u8_ordered(hi_bits);
    let offset = pack_n * 2;
    unsafe { pack_u5_registers(out.add(offset), hi_bits, pack_n) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 22-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(22)` bytes to.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(crate) unsafe fn to_u22(out: *mut u8, block: [__m512i; 4], pack_n: usize) {
    unsafe { store_lo_u16_registers(out.add(0), block) };

    let hi_bits = srli_epi32::<16, 4>(block);
    let hi_bits = pack_u32_to_u8_ordered(hi_bits);
    let offset = pack_n * 2;
    unsafe { pack_u6_registers(out.add(offset), hi_bits, pack_n) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 23-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(23)` bytes to.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(crate) unsafe fn to_u23(out: *mut u8, block: [__m512i; 4], pack_n: usize) {
    unsafe { store_lo_u16_registers(out.add(0), block) };

    let hi_bits = srli_epi32::<16, 4>(block);
    let hi_bits = pack_u32_to_u8_ordered(hi_bits);
    let offset = pack_n * 2;
    unsafe { pack_u7_registers(out.add(offset), hi_bits, pack_n) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 24-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(24)` bytes to.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(crate) unsafe fn to_u24(out: *mut u8, block: [__m512i; 4], pack_n: usize) {
    unsafe { store_lo_u16_registers(out.add(0), block) };

    let hi_bits = srli_epi32::<16, 4>(block);
    let hi_bits = pack_u32_to_u8_ordered(hi_bits);
    let offset = pack_n * 2;
    unsafe { _mm512_storeu_epi8(out.add(offset).cast(), hi_bits) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 25-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(25)` bytes to.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(crate) unsafe fn to_u25(out: *mut u8, block: [__m512i; 4], pack_n: usize) {
    unsafe { store_lo_u16_registers(out.add(0), block) };

    let hi_half = srli_epi32::<16, 4>(block);
    let (hi, lo) = pack_u32_to_u16_split_ordered(hi_half);

    let mut offset = pack_n * 2;
    unsafe { _mm512_storeu_epi8(out.add(offset).cast(), lo) };
    offset += pack_n;
    unsafe { pack_u1_registers(out.add(offset), hi) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 26-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(26)` bytes to.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(crate) unsafe fn to_u26(out: *mut u8, block: [__m512i; 4], pack_n: usize) {
    unsafe { store_lo_u16_registers(out.add(0), block) };

    let hi_half = srli_epi32::<16, 4>(block);
    let (hi, lo) = pack_u32_to_u16_split_ordered(hi_half);

    let mut offset = pack_n * 2;
    unsafe { _mm512_storeu_epi8(out.add(offset).cast(), lo) };
    offset += pack_n;
    unsafe { pack_u2_registers(out.add(offset), hi, pack_n) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 27-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(27)` bytes to.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(crate) unsafe fn to_u27(out: *mut u8, block: [__m512i; 4], pack_n: usize) {
    unsafe { store_lo_u16_registers(out.add(0), block) };

    let hi_half = srli_epi32::<16, 4>(block);
    let (hi, lo) = pack_u32_to_u16_split_ordered(hi_half);

    let mut offset = pack_n * 2;
    unsafe { _mm512_storeu_epi8(out.add(offset).cast(), lo) };
    offset += pack_n;
    unsafe { pack_u3_registers(out.add(offset), hi, pack_n) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 28-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(28)` bytes to.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(crate) unsafe fn to_u28(out: *mut u8, block: [__m512i; 4], pack_n: usize) {
    unsafe { store_lo_u16_registers(out.add(0), block) };

    let hi_half = srli_epi32::<16, 4>(block);
    let (hi, lo) = pack_u32_to_u16_split_ordered(hi_half);

    let mut offset = pack_n * 2;
    unsafe { _mm512_storeu_epi8(out.add(offset).cast(), lo) };
    offset += pack_n;
    unsafe { pack_u4_registers(out.add(offset), hi) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 29-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(29)` bytes to.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(crate) unsafe fn to_u29(out: *mut u8, block: [__m512i; 4], pack_n: usize) {
    unsafe { store_lo_u16_registers(out.add(0), block) };

    let hi_half = srli_epi32::<16, 4>(block);
    let (hi, lo) = pack_u32_to_u16_split_ordered(hi_half);

    let mut offset = pack_n * 2;
    unsafe { _mm512_storeu_epi8(out.add(offset).cast(), lo) };
    offset += pack_n;
    unsafe { pack_u5_registers(out.add(offset), hi, pack_n) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 30-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(30)` bytes to.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(crate) unsafe fn to_u30(out: *mut u8, block: [__m512i; 4], pack_n: usize) {
    unsafe { store_lo_u16_registers(out.add(0), block) };

    let hi_half = srli_epi32::<16, 4>(block);
    let (hi, lo) = pack_u32_to_u16_split_ordered(hi_half);

    let mut offset = pack_n * 2;
    unsafe { _mm512_storeu_epi8(out.add(offset).cast(), lo) };
    offset += pack_n;
    unsafe { pack_u6_registers(out.add(offset), hi, pack_n) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 31-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(31)` bytes to.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(crate) unsafe fn to_u31(out: *mut u8, block: [__m512i; 4], pack_n: usize) {
    unsafe { store_lo_u16_registers(out.add(0), block) };

    let hi_half = srli_epi32::<16, 4>(block);
    let (hi, lo) = pack_u32_to_u16_split_ordered(hi_half);

    let mut offset = pack_n * 2;
    unsafe { _mm512_storeu_epi8(out.add(offset).cast(), lo) };
    offset += pack_n;
    unsafe { pack_u7_registers(out.add(offset), hi, pack_n) };
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Bitpack the provided block of integers to 32-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(32)` bytes to.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(crate) unsafe fn to_u32(out: *mut u8, block: [__m512i; 4], _pack_n: usize) {
    unsafe { store_si512x4(out, block) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::X64;
    use crate::uint32::{X128_MAX_OUTPUT_LEN, max_compressed_size};

    #[test]
    #[cfg_attr(
        not(all(target_feature = "avx512f", target_feature = "avx512bw")),
        ignore
    )]
    fn test_to_u1() {
        let mut data = [0; X64];
        #[allow(clippy::needless_range_loop)]
        for i in 0..X64 {
            data[i] = (i % 2) as u32;
        }
        let data = unsafe { load_u32x64(&data) };

        let mut out = [0; X128_MAX_OUTPUT_LEN / 2];
        unsafe { to_u1(out.as_mut_ptr(), data, 64) };
        assert_eq!(out[..8], [170; 8]);

        let mut data = [0; X64];
        data[0] = 1;
        data[1] = 1;
        data[2] = 1;
        data[3] = 1;
        data[4] = 1;
        data[8] = 1;
        data[9] = 1;
        let data = unsafe { load_u32x64(&data) };

        let mut out = [0; X128_MAX_OUTPUT_LEN / 2];
        unsafe { to_u1(out.as_mut_ptr(), data, 10) };
        assert_eq!(out[..2], [31, 3]);
        assert_eq!(out[2..][..14], [0; 14]);
    }

    #[test]
    #[cfg_attr(
        not(all(target_feature = "avx512f", target_feature = "avx512bw")),
        ignore
    )]
    fn test_to_u2() {
        let mut data = [0; X64];
        #[allow(clippy::needless_range_loop)]
        for i in 0..X64 {
            data[i] = (i % 3) as u32;
        }
        let data = unsafe { load_u32x64(&data) };

        let mut out = [0; X128_MAX_OUTPUT_LEN / 2];
        unsafe { to_u2(out.as_mut_ptr(), data, 64) };
        assert_eq!(
            out[..16],
            [
                146, 36, 73, 146, 36, 73, 146, 36, 36, 73, 146, 36, 73, 146, 36, 73
            ]
        );

        let mut data = [0; X64];
        data[0] = 1;
        data[1] = 1;
        data[2] = 1;
        data[3] = 1;
        data[4] = 1;
        data[8] = 1;
        data[9] = 2;
        let data = unsafe { load_u32x64(&data) };

        let mut out = [0; X128_MAX_OUTPUT_LEN / 2];
        unsafe { to_u2(out.as_mut_ptr(), data, 10) };

        // Imagine the data is in bits:
        //
        // lo_bits: 0b00000001_00011111 -> [31, 1, 0, 0] LE
        // hi_bits: 0b00000010_00000000 -> [0, 2, 0, 0] LE
        assert_eq!(out[..4], [31, 1, 0, 2]);
        assert_eq!(out[4..][..28], [0; 28]);
    }

    #[test]
    #[cfg_attr(
        not(all(target_feature = "avx512f", target_feature = "avx512bw")),
        ignore
    )]
    fn test_to_u3() {
        let mut data = [0; X64];
        #[allow(clippy::needless_range_loop)]
        for i in 0..X64 {
            data[i] = (i % 4) as u32;
        }
        let data = unsafe { load_u32x64(&data) };

        let mut out = [0; X128_MAX_OUTPUT_LEN / 2];
        unsafe { to_u3(out.as_mut_ptr(), data, 64) };
        assert_eq!(
            out[..24],
            [
                170, 170, 170, 170, 170, 170, 170, 170, 204, 204, 204, 204, 204, 204, 204, 204, 0,
                0, 0, 0, 0, 0, 0, 0
            ]
        );
        assert_eq!(out[24..][..24], [0; 24]);

        let mut data = [0; X64];
        data[0] = 1;
        data[1] = 1;
        data[2] = 1;
        data[3] = 1;
        data[4] = 1;
        data[8] = 5;
        data[9] = 2;
        let data = unsafe { load_u32x64(&data) };

        let mut out = [0; X128_MAX_OUTPUT_LEN / 2];
        unsafe { to_u3(out.as_mut_ptr(), data, 10) };

        // Imagine the data is in bits:
        //
        // b0_bits: 0b00000001_00011111 -> [31, 1, 0, 0] LE
        // b1_bits: 0b00000010_00000000 -> [0, 2, 0, 0] LE
        // b2_bits: 0b00000001_00000000 -> [0, 1, 0, 0] LE
        assert_eq!(out[..6], [31, 1, 0, 2, 0, 1]);
        assert_eq!(out[6..][..42], [0; 42]);
    }

    #[test]
    #[cfg_attr(
        not(all(target_feature = "avx512f", target_feature = "avx512bw")),
        ignore
    )]
    fn test_to_u4() {
        let mut data = [0; X64];
        #[allow(clippy::needless_range_loop)]
        for i in 0..X64 {
            data[i] = (i % 16) as u32;
        }
        let data = unsafe { load_u32x64(&data) };

        let mut out = [0; X128_MAX_OUTPUT_LEN / 2];
        unsafe { to_u4(out.as_mut_ptr(), data, 64) };
        assert_eq!(out[..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[8..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[16..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[24..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);

        let mut data = [0; X64];
        data[0] = 1;
        data[1] = 15;
        data[2] = 2;
        data[3] = 15;
        data[4] = 1;
        data[8] = 5;
        data[9] = 2;
        data[14] = 15;
        let data = unsafe { load_u32x64(&data) };

        let mut out = [0; X128_MAX_OUTPUT_LEN / 2];
        unsafe { to_u4(out.as_mut_ptr(), data, 15) };
        assert_eq!(out[0], 0b1111_0001); // [1, 15]
        assert_eq!(out[1], 0b1111_0010); // [2, 15]
        assert_eq!(out[2], 0b0000_0001); // [1, 0]
        assert_eq!(out[4], 0b0010_0101); // [5, 2]
        assert_eq!(out[7], 0b0000_1111); // [15, 0]
    }

    #[test]
    #[cfg_attr(
        not(all(target_feature = "avx512f", target_feature = "avx512bw")),
        ignore
    )]
    fn test_to_u5() {
        let mut data = [0; X64];
        #[allow(clippy::needless_range_loop)]
        for i in 0..X64 {
            data[i] = (i % 32) as u32;
        }
        let data = unsafe { load_u32x64(&data) };

        let mut out = [0; X128_MAX_OUTPUT_LEN / 2];
        unsafe { to_u5(out.as_mut_ptr(), data, 64) };
        assert_eq!(out[..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[8..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[16..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[24..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[32..][..8], [0, 0, 255, 255, 0, 0, 255, 255]);

        let mut data = [0; X64];
        data[0] = 1;
        data[1] = 15;
        data[2] = 2;
        data[3] = 15;
        data[4] = 1;
        data[8] = 5;
        data[9] = 2;
        data[14] = 17;
        let data = unsafe { load_u32x64(&data) };

        let mut out = [0; X128_MAX_OUTPUT_LEN / 2];
        unsafe { to_u5(out.as_mut_ptr(), data, 15) };
        assert_eq!(out[0], 0b1111_0001); // [1, 15]
        assert_eq!(out[1], 0b1111_0010); // [2, 15]
        assert_eq!(out[2], 0b0000_0001); // [1, 0]
        assert_eq!(out[4], 0b0010_0101); // [5, 2]
        assert_eq!(out[7], 0b0000_0001); // [17, 0]  -- well, the lower 4 bits
        assert_eq!(out[8], 0b0000_0000);
        assert_eq!(out[9], 0b0100_0000); // upper bits from that 17
    }

    #[test]
    #[cfg_attr(
        not(all(target_feature = "avx512f", target_feature = "avx512bw")),
        ignore
    )]
    fn test_to_u6() {
        let mut data = [0; X64];
        #[allow(clippy::needless_range_loop)]
        for i in 0..X64 {
            data[i] = (i % 64) as u32;
        }
        let data = unsafe { load_u32x64(&data) };

        let mut out = [0; X128_MAX_OUTPUT_LEN / 2];
        unsafe { to_u6(out.as_mut_ptr(), data, 64) };
        assert_eq!(out[..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[8..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[16..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[24..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[32..][..8], [0, 0, 255, 255, 0, 0, 255, 255]);
        assert_eq!(out[40..][..8], [0, 0, 0, 0, 255, 255, 255, 255]);

        let mut data = [0; X64];
        data[0] = 1;
        data[1] = 15;
        data[2] = 2;
        data[3] = 15;
        data[4] = 1;
        data[8] = 5;
        data[9] = 2;
        data[14] = 59;
        let data = unsafe { load_u32x64(&data) };

        let mut out = [0; X128_MAX_OUTPUT_LEN / 2];
        unsafe { to_u6(out.as_mut_ptr(), data, 15) };
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
    #[cfg_attr(
        not(all(target_feature = "avx512f", target_feature = "avx512bw")),
        ignore
    )]
    fn test_to_u7() {
        let mut data = [0; X64];
        #[allow(clippy::needless_range_loop)]
        for i in 0..X64 {
            data[i] = (i % 64) as u32;
        }
        let data = unsafe { load_u32x64(&data) };

        let mut out = [0; X128_MAX_OUTPUT_LEN / 2];
        unsafe { to_u7(out.as_mut_ptr(), data, 64) };
        assert_eq!(out[..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[8..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[16..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[24..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[32..][..8], [0, 0, 255, 255, 0, 0, 255, 255]);
        assert_eq!(out[40..][..8], [0, 0, 0, 0, 255, 255, 255, 255]);
        assert_eq!(out[48..][..8], [0, 0, 0, 0, 0, 0, 0, 0]);

        let mut data = [0; X64];
        data[0] = 1;
        data[1] = 15;
        data[2] = 2;
        data[3] = 15;
        data[4] = 1;
        data[8] = 5;
        data[9] = 2;
        data[14] = 127;
        let data = unsafe { load_u32x64(&data) };

        let mut out = [0; X128_MAX_OUTPUT_LEN / 2];
        unsafe { to_u7(out.as_mut_ptr(), data, 15) };
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
    #[cfg_attr(
        not(all(target_feature = "avx512f", target_feature = "avx512bw")),
        ignore
    )]
    fn test_saturation(
        #[case] bit_len: u8,
        #[case] packer: unsafe fn(*mut u8, [__m512i; 4], usize),
    ) {
        let pack_value = (2u64.pow(bit_len as u32) - 1) as u32;

        let values = [pack_value; X64];
        let data = unsafe { load_u32x64(&values) };

        let mut output = [0; X128_MAX_OUTPUT_LEN / 2];
        unsafe { packer(output.as_mut_ptr(), data, X64) };
        assert!(
            output[..max_compressed_size::<X64>(bit_len as usize)]
                .iter()
                .all(|b| *b == u8::MAX)
        );
    }
}
