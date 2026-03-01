use std::arch::x86_64::*;

use super::data::*;
use super::util::*;

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the 1-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(1)` bytes from.
/// - `read_n` must be between `0` and `64`.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(crate) unsafe fn from_u1(input: *const u8, read_n: usize) -> [__m512i; 2] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let packed = unsafe { unpack_u1_registers(input) };
    unpack_u8_to_u16_ordered(packed)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
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
/// - `read_n` must be between `0` and `64`.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(crate) unsafe fn from_u2(input: *const u8, read_n: usize) -> [__m512i; 2] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let packed = unsafe { unpack_u2_registers(input, read_n) };
    unpack_u8_to_u16_ordered(packed)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
unsafe fn unpack_u2_registers(input: *const u8, read_n: usize) -> __m512i {
    let offset = read_n.div_ceil(8);
    let mask1: u64 = unsafe { std::ptr::read_unaligned(input.add(0).cast()) };
    let mask2: u64 = unsafe { std::ptr::read_unaligned(input.add(offset).cast()) };

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
/// - `read_n` must be between `0` and `64`.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(crate) unsafe fn from_u3(input: *const u8, read_n: usize) -> [__m512i; 2] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let packed = unsafe { unpack_u3_registers(input, read_n) };
    unpack_u8_to_u16_ordered(packed)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
unsafe fn unpack_u3_registers(input: *const u8, read_n: usize) -> __m512i {
    let step = read_n.div_ceil(8);

    let mask1: u64 = unsafe { std::ptr::read_unaligned(input.add(0).cast()) };
    let mask2: u64 = unsafe { std::ptr::read_unaligned(input.add(step).cast()) };
    let mask3: u64 = unsafe { std::ptr::read_unaligned(input.add(step * 2).cast()) };

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
/// - `read_n` must be between `0` and `64`.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(crate) unsafe fn from_u4(input: *const u8, read_n: usize) -> [__m512i; 2] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let packed = unsafe { unpack_u4_registers(input) };
    unpack_u8_to_u16_ordered(packed)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
pub(super) unsafe fn unpack_u4_registers(input: *const u8) -> __m512i {
    let packed = unsafe { _mm256_loadu_si256(input.cast()) };
    let wide = _mm512_cvtepu8_epi16(packed);
    let lo = _mm512_and_si512(wide, _mm512_set1_epi16(0x000F));
    let hi = _mm512_slli_epi16(_mm512_srli_epi16(wide, 4), 8);
    _mm512_or_si512(lo, hi)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the 5-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(5)` bytes from.
/// - `read_n` must be between `0` and `64`.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(crate) unsafe fn from_u5(input: *const u8, read_n: usize) -> [__m512i; 2] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let packed = unsafe { unpack_u5_registers(input, read_n) };
    unpack_u8_to_u16_ordered(packed)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
unsafe fn unpack_u5_registers(input: *const u8, read_n: usize) -> __m512i {
    let offset = read_n.div_ceil(2);
    let lo_bits = unsafe { unpack_u4_registers(input.add(0)) };
    let mut hi_bits = unsafe { unpack_u1_registers(input.add(offset)) };
    hi_bits = _mm512_slli_epi16::<4>(hi_bits);
    _mm512_or_si512(hi_bits, lo_bits)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the 6-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(6)` bytes from.
/// - `read_n` must be between `0` and `64`.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(crate) unsafe fn from_u6(input: *const u8, read_n: usize) -> [__m512i; 2] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let packed = unsafe { unpack_u6_registers(input, read_n) };
    unpack_u8_to_u16_ordered(packed)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
unsafe fn unpack_u6_registers(input: *const u8, read_n: usize) -> __m512i {
    let offset = read_n.div_ceil(2);
    let lo_bits = unsafe { unpack_u4_registers(input.add(0)) };
    let mut hi_bits = unsafe { unpack_u2_registers(input.add(offset), read_n) };
    hi_bits = _mm512_slli_epi16::<4>(hi_bits);
    _mm512_or_si512(hi_bits, lo_bits)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the 7-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(7)` bytes from.
/// - `read_n` must be between `0` and `64`.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(crate) unsafe fn from_u7(input: *const u8, read_n: usize) -> [__m512i; 2] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let packed = unsafe { unpack_u7_registers(input, read_n) };
    unpack_u8_to_u16_ordered(packed)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
unsafe fn unpack_u7_registers(input: *const u8, read_n: usize) -> __m512i {
    let offset = read_n.div_ceil(2);
    let lo_bits = unsafe { unpack_u4_registers(input.add(0)) };
    let mut hi_bits = unsafe { unpack_u3_registers(input.add(offset), read_n) };
    hi_bits = _mm512_slli_epi16::<4>(hi_bits);
    _mm512_or_si512(hi_bits, lo_bits)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the 8-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(8)` bytes from.
/// - `read_n` must be between `0` and `64`.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(crate) unsafe fn from_u8(input: *const u8, read_n: usize) -> [__m512i; 2] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let packed = unsafe { _mm512_loadu_epi8(input.cast()) };
    unpack_u8_to_u16_ordered(packed)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the 9-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(9)` bytes from.
/// - `read_n` must be between `0` and `64`.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(crate) unsafe fn from_u9(input: *const u8, read_n: usize) -> [__m512i; 2] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let lo_bits = unsafe { _mm512_loadu_epi8(input.add(0).cast()) };
    let lo_bits = unpack_u8_to_u16_ordered(lo_bits);

    let offset = read_n;
    let hi_bits = unsafe { unpack_u1_registers(input.add(offset)) };
    let mut hi_bits = unpack_u8_to_u16_ordered(hi_bits);
    hi_bits = slli_epi16::<8, 2>(hi_bits);

    or_si512_all(hi_bits, lo_bits)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the 10-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(10)` bytes from.
/// - `read_n` must be between `0` and `64`.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(crate) unsafe fn from_u10(input: *const u8, read_n: usize) -> [__m512i; 2] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let lo_bits = unsafe { _mm512_loadu_epi8(input.add(0).cast()) };
    let lo_bits = unpack_u8_to_u16_ordered(lo_bits);

    let offset = read_n;
    let hi_bits = unsafe { unpack_u2_registers(input.add(offset), read_n) };
    let mut hi_bits = unpack_u8_to_u16_ordered(hi_bits);
    hi_bits = slli_epi16::<8, 2>(hi_bits);

    or_si512_all(hi_bits, lo_bits)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the 11-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(11)` bytes from.
/// - `read_n` must be between `0` and `64`.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(crate) unsafe fn from_u11(input: *const u8, read_n: usize) -> [__m512i; 2] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let lo_bits = unsafe { _mm512_loadu_epi8(input.add(0).cast()) };
    let lo_bits = unpack_u8_to_u16_ordered(lo_bits);

    let offset = read_n;
    let hi_bits = unsafe { unpack_u3_registers(input.add(offset), read_n) };
    let mut hi_bits = unpack_u8_to_u16_ordered(hi_bits);
    hi_bits = slli_epi16::<8, 2>(hi_bits);

    or_si512_all(hi_bits, lo_bits)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the 12-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(12)` bytes from.
/// - `read_n` must be between `0` and `64`.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(crate) unsafe fn from_u12(input: *const u8, read_n: usize) -> [__m512i; 2] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let lo_bits = unsafe { _mm512_loadu_epi8(input.add(0).cast()) };
    let lo_bits = unpack_u8_to_u16_ordered(lo_bits);

    let offset = read_n;
    let hi_bits = unsafe { unpack_u4_registers(input.add(offset)) };
    let mut hi_bits = unpack_u8_to_u16_ordered(hi_bits);
    hi_bits = slli_epi16::<8, 2>(hi_bits);

    or_si512_all(hi_bits, lo_bits)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the 13-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(13)` bytes from.
/// - `read_n` must be between `0` and `64`.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(crate) unsafe fn from_u13(input: *const u8, read_n: usize) -> [__m512i; 2] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let lo_bits = unsafe { _mm512_loadu_epi8(input.add(0).cast()) };
    let lo_bits = unpack_u8_to_u16_ordered(lo_bits);

    let offset = read_n;
    let hi_bits = unsafe { unpack_u5_registers(input.add(offset), read_n) };
    let mut hi_bits = unpack_u8_to_u16_ordered(hi_bits);
    hi_bits = slli_epi16::<8, 2>(hi_bits);

    or_si512_all(hi_bits, lo_bits)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the 14-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(14)` bytes from.
/// - `read_n` must be between `0` and `64`.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(crate) unsafe fn from_u14(input: *const u8, read_n: usize) -> [__m512i; 2] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let lo_bits = unsafe { _mm512_loadu_epi8(input.add(0).cast()) };
    let lo_bits = unpack_u8_to_u16_ordered(lo_bits);

    let offset = read_n;
    let hi_bits = unsafe { unpack_u6_registers(input.add(offset), read_n) };
    let mut hi_bits = unpack_u8_to_u16_ordered(hi_bits);
    hi_bits = slli_epi16::<8, 2>(hi_bits);

    or_si512_all(hi_bits, lo_bits)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the 15-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(15)` bytes from.
/// - `read_n` must be between `0` and `64`.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(crate) unsafe fn from_u15(input: *const u8, read_n: usize) -> [__m512i; 2] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let lo_bits = unsafe { _mm512_loadu_epi8(input.add(0).cast()) };
    let lo_bits = unpack_u8_to_u16_ordered(lo_bits);

    let offset = read_n;
    let hi_bits = unsafe { unpack_u7_registers(input.add(offset), read_n) };
    let mut hi_bits = unpack_u8_to_u16_ordered(hi_bits);
    hi_bits = slli_epi16::<8, 2>(hi_bits);

    or_si512_all(hi_bits, lo_bits)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack the 16-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(16)` bytes from.
/// - `read_n` must be between `0` and `64`.
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
pub(crate) unsafe fn from_u16(input: *const u8, read_n: usize) -> [__m512i; 2] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    unsafe { load_si512x2(input) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::X64;
    #[cfg(feature = "avx2")]
    use crate::uint16::avx2;
    use crate::uint16::avx512::pack_x64_partial::*;
    use crate::uint16::{X128_MAX_OUTPUT_LEN, max_compressed_size};

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
    #[cfg_attr(
        not(all(target_feature = "avx512f", target_feature = "avx512bw",)),
        ignore
    )]
    fn test_saturated_unpack(
        #[case] bit_len: u8,
        #[case] unpacker: unsafe fn(*const u8, usize) -> [__m512i; 2],
    ) {
        let saturated_bytes = [u8::MAX; X128_MAX_OUTPUT_LEN / 2];

        let unpacked = unsafe { unpacker(saturated_bytes.as_ptr(), X64) };
        let unpacked = unsafe { std::mem::transmute::<[__m512i; 2], [u16; X64]>(unpacked) };

        let expected_value = (2u64.pow(bit_len as u32) - 1) as u16;
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
    #[cfg_attr(
        not(all(target_feature = "avx512f", target_feature = "avx512bw",)),
        ignore
    )]
    fn test_pack_unpack(
        #[case] bit_len: u8,
        #[case] packer: unsafe fn(*mut u8, [__m512i; 2], usize),
        #[case] unpacker: unsafe fn(*const u8, usize) -> [__m512i; 2],
    ) {
        fastrand::seed(5876358762523525);

        let max_value = (2u64.pow(bit_len as u32) - 1) as u16;

        let mut values = [0; X64];
        for value in values.iter_mut() {
            *value = fastrand::u16(0..max_value);
        }
        let data = unsafe { load_u16x64(&values) };

        let mut packed = [0; X128_MAX_OUTPUT_LEN / 2];
        unsafe { packer(packed.as_mut_ptr(), data, X64) };

        let unpacked = unsafe { unpacker(packed.as_ptr(), X64) };
        let unpacked = unsafe { std::mem::transmute::<[__m512i; 2], [u16; X64]>(unpacked) };
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
    #[cfg_attr(
        not(all(target_feature = "avx512f", target_feature = "avx512bw",)),
        ignore
    )]
    fn test_pack_unpack_length_permutations(
        #[case] bit_len: u8,
        #[case] packer: unsafe fn(*mut u8, [__m512i; 2], usize),
        #[case] unpacker: unsafe fn(*const u8, usize) -> [__m512i; 2],
    ) {
        fastrand::seed(5876358762523525);

        let max_value = (2u64.pow(bit_len as u32) - 1) as u16;

        for length in 0..X64 {
            let mut values = [0; X64];
            for value in values.iter_mut() {
                *value = fastrand::u16(0..max_value);
            }
            let data = unsafe { load_u16x64(&values) };

            let mut packed = [0; X128_MAX_OUTPUT_LEN / 2];
            unsafe { packer(packed.as_mut_ptr(), data, length) };
            packed[max_compressed_size::<X64>(bit_len as usize)..].fill(0);

            let unpacked = unsafe { unpacker(packed.as_ptr(), length) };
            let unpacked = unsafe { std::mem::transmute::<[__m512i; 2], [u16; X64]>(unpacked) };
            assert_eq!(unpacked[..length], values[..length], "length:{length}");
        }
    }

    #[cfg(feature = "avx2")]
    #[rstest::rstest]
    #[case(1, avx2::pack_x64_partial::to_u1, from_u1)]
    #[case(2, avx2::pack_x64_partial::to_u2, from_u2)]
    #[case(3, avx2::pack_x64_partial::to_u3, from_u3)]
    #[case(4, avx2::pack_x64_partial::to_u4, from_u4)]
    #[case(5, avx2::pack_x64_partial::to_u5, from_u5)]
    #[case(6, avx2::pack_x64_partial::to_u6, from_u6)]
    #[case(7, avx2::pack_x64_partial::to_u7, from_u7)]
    #[case(8, avx2::pack_x64_partial::to_u8, from_u8)]
    #[case(9, avx2::pack_x64_partial::to_u9, from_u9)]
    #[case(10, avx2::pack_x64_partial::to_u10, from_u10)]
    #[case(11, avx2::pack_x64_partial::to_u11, from_u11)]
    #[case(12, avx2::pack_x64_partial::to_u12, from_u12)]
    #[case(13, avx2::pack_x64_partial::to_u13, from_u13)]
    #[case(14, avx2::pack_x64_partial::to_u14, from_u14)]
    #[case(15, avx2::pack_x64_partial::to_u15, from_u15)]
    #[case(16, avx2::pack_x64_partial::to_u16, from_u16)]
    #[cfg_attr(
        not(all(target_feature = "avx512f", target_feature = "avx512bw")),
        ignore
    )]
    fn test_unpack_avx2_packed(
        #[case] bit_len: u8,
        #[case] packer: unsafe fn(*mut u8, [__m256i; 4], usize),
        #[case] unpacker: unsafe fn(*const u8, usize) -> [__m512i; 2],
    ) {
        fastrand::seed(5876358762523525);

        let max_value = (2u64.pow(bit_len as u32) - 1) as u16;

        for length in 0..X64 {
            let mut values = [0; X64];
            for value in values.iter_mut() {
                *value = fastrand::u16(0..max_value);
            }
            let data = unsafe { avx2::data::load_u16x64(&values) };

            let mut packed = [0; X128_MAX_OUTPUT_LEN / 2];
            unsafe { packer(packed.as_mut_ptr(), data, length) };
            packed[max_compressed_size::<X64>(bit_len as usize)..].fill(0);

            let unpacked = unsafe { unpacker(packed.as_ptr(), length) };
            let unpacked = unsafe { std::mem::transmute::<[__m512i; 2], [u16; X64]>(unpacked) };
            assert_eq!(unpacked[..length], values[..length], "length:{length}");
        }
    }
}
