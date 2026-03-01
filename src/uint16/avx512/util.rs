use std::arch::x86_64::*;

pub const _MM_TERNLOG_A: i32 = 0xF0; // 11110000
pub const _MM_TERNLOG_B: i32 = 0xCC; // 11001100
pub const _MM_TERNLOG_C: i32 = 0xAA; // 10101010

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Pack 2 sets of registers containing 16-bit elements and produce 1 register holding
/// 8-bit elements.
///
/// The order of elements is **not** maintained.
pub(super) fn pack_u16_to_u8_unordered(data: [__m512i; 2]) -> __m512i {
    let lo = data[0];
    let hi = _mm512_slli_epi16::<8>(data[1]);
    _mm512_or_si512(hi, lo)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Split the 16-bit values in the provided registers producing two 8-bit
/// halves, packing the results while maintaining the order.
pub(super) fn split_u16_ordered(data: [__m512i; 2]) -> (__m512i, __m512i) {
    let mask = _mm512_set1_epi16(0x00FF);

    let lo_bits = and_si512(data, mask);
    let hi_bits = srli_epi16::<8, 2>(data);

    let lo_packed = pack_u16_to_u8_ordered(lo_bits);
    let hi_packed = pack_u16_to_u8_ordered(hi_bits);

    (hi_packed, lo_packed)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Split the 16-bit values in the provided registers producing two 8-bit
/// halves, packing the results while maintaining the order.
pub(super) fn split_u16_unordered(data: [__m512i; 2]) -> (__m512i, __m512i) {
    let mask = _mm512_set1_epi16(0x00FF);

    let lo_bits = and_si512(data, mask);
    let hi_bits = srli_epi16::<8, 2>(data);

    let lo_packed = pack_u16_to_u8_unordered(lo_bits);
    let hi_packed = pack_u16_to_u8_unordered(hi_bits);

    (hi_packed, lo_packed)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Pack 2 sets of registers containing 16-bit elements and produce 1 register holding
/// 8-bit elements.
///
/// The order of elements is  maintained.
///
/// # Important note on saturation
/// This routine produces garbage results if the input elements are beyond the maximum value
/// that can be represented in a [u16] value, meaning any value over [u16::MAX] produces
/// invalid data.
pub(super) fn pack_u16_to_u8_ordered(data: [__m512i; 2]) -> __m512i {
    let permute_idx = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    let interleaved = _mm512_packus_epi16(data[0], data[1]);
    _mm512_permutexvar_epi64(permute_idx, interleaved)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack 1 register containing 8-bit elements and produce 2 registers holding
/// 32-bit elements.
pub(super) fn unpack_u8_to_u16_ordered(data: __m512i) -> [__m512i; 2] {
    let lo_256 = _mm512_castsi512_si256(data);
    let hi_256 = _mm512_extracti64x4_epi64(data, 1);

    let lo = _mm512_cvtepu8_epi16(lo_256);
    let hi = _mm512_cvtepu8_epi16(hi_256);

    [lo, hi]
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack 1 register containing 8-bit elements and produce 2 registers holding
/// 32-bit elements.
pub(super) fn unpack_u8_to_u16_unordered(data: __m512i) -> [__m512i; 2] {
    let zeroes = _mm512_setzero_si512();
    [
        _mm512_mask_blend_epi8(0xAAAAAAAAAAAAAAAA, data, zeroes),
        _mm512_srli_epi16::<8>(data),
    ]
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
pub(super) fn pack_u8_to_u4_unordered(data: __m512i) -> __m256i {
    let lo = _mm512_castsi512_si256(data);
    let hi = _mm512_extracti64x4_epi64::<1>(data);
    _mm256_or_si256(_mm256_slli_epi16::<4>(hi), lo)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
pub(super) fn unpack_u4_to_u8_unordered(data: __m256i) -> __m512i {
    let mask = _mm256_set1_epi8(0x0F);
    let lo_nibbles = _mm256_and_si256(data, mask);
    let hi_nibbles = _mm256_and_si256(_mm256_srli_epi16::<4>(data), mask);
    let extended = _mm512_castsi256_si512(lo_nibbles);
    _mm512_inserti64x4::<1>(extended, hi_nibbles)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
pub(super) fn pack_u8_to_u2_unordered(data: __m512i) -> __m128i {
    let nibbles = pack_u8_to_u4_unordered(data);
    let lo = _mm256_castsi256_si128(nibbles);
    let hi = _mm256_extracti128_si256::<1>(nibbles);
    _mm_or_si128(lo, _mm_slli_epi16::<2>(hi))
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
pub(super) fn unpack_u2_to_u8_unordered(data: __m128i) -> __m512i {
    let mask = _mm_set1_epi8(0b0011_0011);
    let lo = _mm_and_si128(data, mask);
    let hi = _mm_and_si128(_mm_srli_epi16::<2>(data), mask);
    let nibbles = _mm256_set_m128i(hi, lo);
    unpack_u4_to_u8_unordered(nibbles)
}

#[target_feature(enable = "avx512f")]
/// Perform a bitwise AND on all provided registers with another broadcast register.
pub(super) fn and_si512<const N: usize>(mut data: [__m512i; N], mask: __m512i) -> [__m512i; N] {
    let mut i = 0;
    while i < N {
        data[i] = _mm512_and_si512(data[i], mask);
        i += 1;
    }
    data
}

#[target_feature(enable = "avx512f")]
/// Perform a bitwise OR on the two sets of registers.
pub(super) fn or_si512_all<const N: usize>(mut d1: [__m512i; N], d2: [__m512i; N]) -> [__m512i; N] {
    let mut i = 0;
    while i < N {
        d1[i] = _mm512_or_si512(d1[i], d2[i]);
        i += 1;
    }
    d1
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Shift all registers right by [IMM8] in 16-bit lanes.
pub(super) fn srli_epi16<const IMM8: u32, const N: usize>(mut data: [__m512i; N]) -> [__m512i; N] {
    let mut i = 0;
    while i < N {
        data[i] = _mm512_srli_epi16::<IMM8>(data[i]);
        i += 1;
    }
    data
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Shift all registers left by [IMM8] in 32-bit lanes.
pub(super) fn slli_epi16<const IMM8: u32, const N: usize>(mut data: [__m512i; N]) -> [__m512i; N] {
    let mut i = 0;
    while i < N {
        data[i] = _mm512_slli_epi16::<IMM8>(data[i]);
        i += 1;
    }
    data
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::X64;
    use crate::uint16::avx512::data::load_si512x2;
    use crate::uint16::test_util::*;

    #[test]
    #[cfg_attr(
        not(all(target_feature = "avx512f", target_feature = "avx512bw")),
        ignore
    )]
    fn test_unpack_u8_u16_ordered() {
        let mut input = [0; X64];
        #[allow(clippy::needless_range_loop)]
        for i in 0..X64 {
            input[i] = i as u16;
        }

        let data = unsafe { load_si512x2(input.as_ptr().cast()) };
        let packed = unsafe { pack_u16_to_u8_ordered(data) };
        let unpacked = unsafe { unpack_u8_to_u16_ordered(packed) };

        let view = unsafe { std::mem::transmute::<[__m512i; 2], [u16; X64]>(unpacked) };
        assert_eq!(view, input);
    }

    #[test]
    #[cfg_attr(
        not(all(target_feature = "avx512f", target_feature = "avx512bw")),
        ignore
    )]
    fn test_unpack_u8_to_u16_unordered_layout() {
        let expected: [u16; X64] = std::array::from_fn(|i| i as u16);

        let data =
            unsafe { _mm512_loadu_epi8(PACK_U16_TO_U8_EXPECTED_UNORDERED_LAYOUT.as_ptr().cast()) };
        let unpacked = unsafe { unpack_u8_to_u16_unordered(data) };

        let view = unsafe { std::mem::transmute::<[__m512i; 2], [u16; X64]>(unpacked) };
        assert_eq!(view, expected);
    }

    #[test]
    #[cfg_attr(
        not(all(target_feature = "avx512f", target_feature = "avx512bw")),
        ignore
    )]
    fn test_unpack_u4_to_u8_unordered() {
        let expected: [u8; X64] = std::array::from_fn(|i| i as u8 % 16);

        let data = unsafe { _mm512_loadu_epi8(expected.as_ptr().cast()) };
        let packed = unsafe { pack_u8_to_u4_unordered(data) };
        let unpacked = unsafe { unpack_u4_to_u8_unordered(packed) };

        let view = unsafe { std::mem::transmute::<__m512i, [u8; X64]>(unpacked) };
        assert_eq!(view, expected);
    }

    #[test]
    #[cfg_attr(
        not(all(target_feature = "avx512f", target_feature = "avx512bw")),
        ignore
    )]
    fn test_unpack_u2_to_u8_unordered() {
        let expected: [u8; X64] = std::array::from_fn(|i| i as u8 % 4);

        let data = unsafe { _mm512_loadu_epi8(expected.as_ptr().cast()) };
        let packed = unsafe { pack_u8_to_u2_unordered(data) };
        let unpacked = unsafe { unpack_u2_to_u8_unordered(packed) };

        let view = unsafe { std::mem::transmute::<__m512i, [u8; X64]>(unpacked) };
        assert_eq!(view, expected);
    }
}
