use std::arch::x86_64::*;

#[inline]
#[allow(non_snake_case)]
pub const fn _mm_shuffle(z: u32, y: u32, x: u32, w: u32) -> i32 {
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}

#[target_feature(enable = "avx2")]
/// Expand the provided bitmask to a 256-bit register.
pub(super) fn expand_mask_epi8(mask: __mmask32) -> __m256i {
    #[rustfmt::skip]
    let powers = _mm256_setr_epi8(
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80u8 as i8,
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80u8 as i8,
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80u8 as i8,
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80u8 as i8,
    );

    let v_mask = _mm256_set1_epi32(mask as i32);
    #[rustfmt::skip]
    let shuffle_mask = _mm256_setr_epi8(
        0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,
        2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3
    );
    let bytes_broadcast = _mm256_shuffle_epi8(v_mask, shuffle_mask);
    let isolated = _mm256_and_si256(bytes_broadcast, powers);

    _mm256_cmpeq_epi8(isolated, powers)
}

#[target_feature(enable = "avx2")]
/// Pack 16-bit integers into 8-bit integers.
///
/// Ordering of elements is not maintained.
pub(super) fn pack_u16_to_u8_unordered(data: [__m256i; 4]) -> [__m256i; 2] {
    let lo = [data[0], data[1]];
    let hi = [
        _mm256_slli_epi16::<8>(data[2]),
        _mm256_slli_epi16::<8>(data[3]),
    ];
    or_si256_all(hi, lo)
}

#[target_feature(enable = "avx2")]
/// Split the 16-bit values in the provided registers producing two 8-bit
/// halves, packing the results while maintaining the order.
pub(super) fn split_u16_ordered(data: [__m256i; 4]) -> ([__m256i; 2], [__m256i; 2]) {
    let mask = _mm256_set1_epi16(0x00FF);

    let lo_bits = and_si256(data, mask);
    let hi_bits = srli_epi16::<8, 4>(data);

    let lo_packed = pack_u16_to_u8_ordered(lo_bits);
    let hi_packed = pack_u16_to_u8_ordered(hi_bits);

    (hi_packed, lo_packed)
}

#[target_feature(enable = "avx2")]
/// Split the 16-bit values in the provided registers producing two 8-bit
/// halves, packing the results while maintaining the order.
pub(super) fn split_u16_unordered(data: [__m256i; 4]) -> ([__m256i; 2], [__m256i; 2]) {
    let mask = _mm256_set1_epi16(0x00FF);

    let lo_bits = and_si256(data, mask);
    let hi_bits = srli_epi16::<8, 4>(data);

    let lo_packed = pack_u16_to_u8_unordered(lo_bits);
    let hi_packed = pack_u16_to_u8_unordered(hi_bits);

    (hi_packed, lo_packed)
}

#[target_feature(enable = "avx2")]
/// Unpack 2 sets of registers containing 8-bit elements and produce 4 registers holding
/// 16-bit elements.
pub(super) fn unpack_u8_to_u16_ordered(data: [__m256i; 2]) -> [__m256i; 4] {
    let zero = _mm256_setzero_si256();

    let unpack_block = |packed: __m256i| -> [__m256i; 2] {
        let unpermuted = _mm256_permute4x64_epi64::<0xD8>(packed);

        [
            _mm256_unpacklo_epi8(unpermuted, zero),
            _mm256_unpackhi_epi8(unpermuted, zero),
        ]
    };

    let block0 = unpack_block(data[0]);
    let block1 = unpack_block(data[1]);

    [block0[0], block0[1], block1[0], block1[1]]
}

// TODO: Blendv is not an optimisation here
#[target_feature(enable = "avx2")]
/// Unpack 2 sets of registers containing 16-bit elements and produce 4 registers holding
/// 16-bit elements.
pub(super) fn unpack_u8_to_u16_unordered(data: [__m256i; 2]) -> [__m256i; 4] {
    let zeroes = _mm256_setzero_si256();
    // Note: We're still operating on u8s, this is just a way to use a broadcast rather than array load.
    let mask = _mm256_set1_epi16(0xFF_00u16 as i16);

    [
        _mm256_blendv_epi8(data[0], zeroes, mask),
        _mm256_blendv_epi8(data[1], zeroes, mask),
        _mm256_srli_epi16::<8>(data[0]),
        _mm256_srli_epi16::<8>(data[1]),
    ]
}

#[target_feature(enable = "avx2")]
/// Pack 4 sets of registers containing 16-bit elements and produce 2 registers holding
/// 8-bit elements.
///
/// The order of elements is maintained.
///
/// # Important note on saturation
/// This routine produces garbage results if the input elements are beyond the maximum value
/// that can be represented in a [u8] value, meaning any value over [u8::MAX] produces
/// invalid data.
pub(super) fn pack_u16_to_u8_ordered(data: [__m256i; 4]) -> [__m256i; 2] {
    let pack_block = |a, b| {
        let packed = _mm256_packus_epi16(a, b);
        _mm256_permute4x64_epi64::<0xD8>(packed)
    };

    [pack_block(data[0], data[1]), pack_block(data[2], data[3])]
}

#[target_feature(enable = "avx2")]
pub(super) fn pack_u8_to_u4_unordered(data: [__m256i; 2]) -> __m256i {
    let shifted = _mm256_slli_epi16::<4>(data[1]);
    _mm256_or_si256(data[0], shifted)
}

#[target_feature(enable = "avx2")]
pub(super) fn unpack_u4_to_u8_unordered(data: __m256i) -> [__m256i; 2] {
    let mask = _mm256_set1_epi8(0x0F);
    let lo_nibbles = _mm256_and_si256(data, mask);
    let hi_nibbles = _mm256_and_si256(_mm256_srli_epi16::<4>(data), mask);
    [lo_nibbles, hi_nibbles]
}

#[target_feature(enable = "avx2")]
pub(super) fn pack_u8_to_u2_unordered(data: [__m256i; 2]) -> __m128i {
    let nibbles = pack_u8_to_u4_unordered(data);
    let lo = _mm256_castsi256_si128(nibbles);
    let hi = _mm256_extracti128_si256::<1>(nibbles);
    _mm_or_si128(lo, _mm_slli_epi16::<2>(hi))
}

#[target_feature(enable = "avx2")]
pub(super) fn unpack_u2_to_u8_unordered(data: __m128i) -> [__m256i; 2] {
    let mask = _mm_set1_epi8(0b0011_0011);
    let lo = _mm_and_si128(data, mask);
    let hi = _mm_and_si128(_mm_srli_epi16::<2>(data), mask);
    let nibbles = _mm256_set_m128i(hi, lo);
    unpack_u4_to_u8_unordered(nibbles)
}

#[target_feature(enable = "avx2")]
/// Perform a bitwise AND on all provided registers with another broadcast register.
pub(super) fn and_si256<const N: usize>(mut data: [__m256i; N], mask: __m256i) -> [__m256i; N] {
    let mut i = 0;
    while i < N {
        data[i] = _mm256_and_si256(data[i], mask);
        i += 1;
    }
    data
}

#[target_feature(enable = "avx2")]
/// Perform a bitwise OR on the two sets of registers.
pub(super) fn or_si256_all<const N: usize>(mut d1: [__m256i; N], d2: [__m256i; N]) -> [__m256i; N] {
    let mut i = 0;
    while i < N {
        d1[i] = _mm256_or_si256(d1[i], d2[i]);
        i += 1;
    }
    d1
}

#[target_feature(enable = "avx2")]
/// Shift all registers right by [IMM8] in 16-bit lanes.
pub(super) fn srli_epi16<const IMM8: i32, const N: usize>(mut data: [__m256i; N]) -> [__m256i; N] {
    let mut i = 0;
    while i < N {
        data[i] = _mm256_srli_epi16::<IMM8>(data[i]);
        i += 1;
    }
    data
}

#[target_feature(enable = "avx2")]
/// Shift all registers right by [IMM8] in 16-bit lanes.
pub(super) fn slli_epi16<const IMM8: i32, const N: usize>(mut data: [__m256i; N]) -> [__m256i; N] {
    let mut i = 0;
    while i < N {
        data[i] = _mm256_slli_epi16::<IMM8>(data[i]);
        i += 1;
    }
    data
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::X64;
    use crate::uint16::avx2::data::{load_si256x2, load_si256x4};
    use crate::uint16::test_util::*;

    #[test]
    #[cfg_attr(not(target_feature = "avx2"), ignore)]
    fn test_unpack_u8_u16_ordered() {
        let mut input = [0; X64];
        #[allow(clippy::needless_range_loop)]
        for i in 0..X64 {
            input[i] = i as u16;
        }

        let data = unsafe { load_si256x4(input.as_ptr().cast()) };
        let packed = unsafe { pack_u16_to_u8_ordered(data) };
        let unpacked = unsafe { unpack_u8_to_u16_ordered(packed) };

        let view = unsafe { std::mem::transmute::<[__m256i; 4], [u16; X64]>(unpacked) };
        assert_eq!(view, input);
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx2"), ignore)]
    fn test_unpack_u8_to_u16_unordered_layout() {
        let expected: [u16; X64] = std::array::from_fn(|i| i as u16);

        let data = unsafe { load_si256x2(PACK_U16_TO_U8_EXPECTED_UNORDERED_LAYOUT.as_ptr()) };
        let unpacked = unsafe { unpack_u8_to_u16_unordered(data) };

        let view = unsafe { std::mem::transmute::<[__m256i; 4], [u16; X64]>(unpacked) };
        assert_eq!(view, expected);
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx2"), ignore)]
    fn test_unpack_u4_to_u8_unordered() {
        let expected: [u8; X64] = std::array::from_fn(|i| i as u8 % 16);

        let data = unsafe { load_si256x2(expected.as_ptr()) };
        let packed = unsafe { pack_u8_to_u4_unordered(data) };
        let unpacked = unsafe { unpack_u4_to_u8_unordered(packed) };

        let view = unsafe { std::mem::transmute::<[__m256i; 2], [u8; X64]>(unpacked) };
        assert_eq!(view, expected);
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx2"), ignore)]
    fn test_unpack_u2_to_u8_unordered() {
        let expected: [u8; X64] = std::array::from_fn(|i| i as u8 % 4);

        let data = unsafe { load_si256x2(expected.as_ptr()) };
        let packed = unsafe { pack_u8_to_u2_unordered(data) };
        let unpacked = unsafe { unpack_u2_to_u8_unordered(packed) };

        let view = unsafe { std::mem::transmute::<[__m256i; 2], [u8; X64]>(unpacked) };
        assert_eq!(view, expected);
    }
}
