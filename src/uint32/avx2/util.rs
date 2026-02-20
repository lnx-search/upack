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

// #[target_feature(enable = "avx2")]
// /// Expand the provided bitmask to a 256-bit register.
// pub(super) fn expand_mask_epi16(mask: __mmask16) -> __m256i {
//     #[rustfmt::skip]
//     let powers = _mm256_setr_epi16(
//         0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
//         0x100, 0x200, 0x400, 0x800, 0x1000, 0x2000, 0x4000, 0x8000u16 as i16,
//     );
//
//     let v_mask = _mm256_set1_epi16(mask as i16);
//     let isolated = _mm256_and_si256(v_mask, powers);
//
//     _mm256_cmpeq_epi16(isolated, powers)
// }
//
// #[target_feature(enable = "avx2")]
// /// Expand the provided bitmask to a 256-bit register.
// pub(super) fn expand_mask_epi32(mask: __mmask8) -> __m256i {
//     let powers = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
//     let v_mask = _mm256_set1_epi32(mask as i32);
//     let isolated = _mm256_and_si256(v_mask, powers);
//
//     _mm256_cmpeq_epi32(isolated, powers)
// }

#[target_feature(enable = "avx2")]
/// Pack 8 sets of registers containing 32-bit elements and produce 2 registers holding
/// 8-bit elements.
///
/// The order of elements is maintained.
///
/// # Important note on saturation
/// This routine produces garbage results if the input elements are beyond the maximum value
/// that can be represented in a [u8] value, meaning any value over [u8::MAX] produces
/// invalid data.
pub(super) fn pack_u32_to_u8_ordered(data: [__m256i; 8]) -> [__m256i; 2] {
    let permute_idx = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

    let pack_block = |a, b, c, d| {
        let p1 = _mm256_packus_epi32(a, b);
        let p2 = _mm256_packus_epi32(c, d);

        let packed = _mm256_packus_epi16(p1, p2);

        _mm256_permutevar8x32_epi32(packed, permute_idx)
    };

    [
        pack_block(data[0], data[1], data[2], data[3]),
        pack_block(data[4], data[5], data[6], data[7]),
    ]
}

#[target_feature(enable = "avx2")]
/// Unpack 2 sets of registers containing 8-bit elements and produce 8 registers holding
/// 32-bit elements.
pub(super) fn unpack_u8_to_u32_ordered(data: [__m256i; 2]) -> [__m256i; 8] {
    let permute_idx = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
    let zeroes = _mm256_setzero_si256();

    let unpack_block = |packed: __m256i| -> [__m256i; 4] {
        let unpermuted = _mm256_permutevar8x32_epi32(packed, permute_idx);

        let lo_16s = _mm256_unpacklo_epi8(unpermuted, zeroes);
        let hi_16s = _mm256_unpackhi_epi8(unpermuted, zeroes);

        [
            _mm256_unpacklo_epi16(lo_16s, zeroes),
            _mm256_unpackhi_epi16(lo_16s, zeroes),
            _mm256_unpacklo_epi16(hi_16s, zeroes),
            _mm256_unpackhi_epi16(hi_16s, zeroes),
        ]
    };

    let block0 = unpack_block(data[0]);
    let block1 = unpack_block(data[1]);

    [
        block0[0], block0[1], block0[2], block0[3], block1[0], block1[1], block1[2], block1[3],
    ]
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

#[target_feature(enable = "avx2")]
/// Pack 8 sets of registers containing 32-bit elements and produce 2 registers holding
/// 8-bit elements.
///
/// The order of elements are _not_ maintained.
///
/// # Important note on saturation
/// This routine produces garbage results if the input elements are beyond the maximum value
/// that can be represented in a [u8] value, meaning any value over [u8::MAX] produces
/// invalid data.
pub(super) fn pack_u32_to_u8_unordered(data: [__m256i; 8]) -> [__m256i; 2] {
    let shift_1 = [data[0], data[1]];
    let shift_2 = [
        _mm256_slli_epi32::<8>(data[4]),
        _mm256_slli_epi32::<8>(data[5]),
    ];
    let shift_3 = [
        _mm256_slli_epi32::<16>(data[2]),
        _mm256_slli_epi32::<16>(data[3]),
    ];
    let shift_4 = [
        _mm256_slli_epi32::<24>(data[6]),
        _mm256_slli_epi32::<24>(data[7]),
    ];

    let mut packed = shift_1;
    packed = or_si256_all(packed, shift_2);
    packed = or_si256_all(packed, shift_3);
    packed = or_si256_all(packed, shift_4);

    packed
}

#[target_feature(enable = "avx2")]
/// Unpack 2 sets of registers containing 8-bit elements and produce 8 registers holding
/// 32-bit elements.
pub(super) fn unpack_u8_to_u32_unordered(data: [__m256i; 2]) -> [__m256i; 8] {
    let partially_unpacked = unpack_u8_to_u16_unordered(data);
    unpack_u16_to_u32_unordered(partially_unpacked)
}

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
/// Pack 8 sets of registers containing 32-bit elements and produce 4 registers holding
/// 16-bit elements.
///
/// The order of elements is maintained.
///
/// # Important note on saturation
/// This routine produces garbage results if the input elements are beyond the maximum value
/// that can be represented in a [u16] value, meaning any value over [u16::MAX] produces
/// invalid data.
pub(super) fn pack_u32_to_u16_ordered(data: [__m256i; 8]) -> [__m256i; 4] {
    let pack_block = |a, b| {
        let packed = _mm256_packus_epi32(a, b);
        _mm256_permute4x64_epi64::<0xD8>(packed)
    };

    [
        pack_block(data[0], data[1]),
        pack_block(data[2], data[3]),
        pack_block(data[4], data[5]),
        pack_block(data[6], data[7]),
    ]
}

#[target_feature(enable = "avx2")]
/// Unpack 4 sets of registers containing 16-bit elements and produce 8 registers holding
/// 32-bit elements.
pub(super) fn unpack_u16_to_u32_ordered(data: [__m256i; 4]) -> [__m256i; 8] {
    let zero = _mm256_setzero_si256();

    let unpack_block = |packed| {
        let unpermuted = _mm256_permute4x64_epi64::<0xD8>(packed);
        [
            _mm256_unpacklo_epi16(unpermuted, zero),
            _mm256_unpackhi_epi16(unpermuted, zero),
        ]
    };

    let block0 = unpack_block(data[0]);
    let block1 = unpack_block(data[1]);
    let block2 = unpack_block(data[2]);
    let block3 = unpack_block(data[3]);

    [
        block0[0], block0[1], block1[0], block1[1], block2[0], block2[1], block3[0], block3[1],
    ]
}

#[target_feature(enable = "avx2")]
/// Pack 8 sets of registers containing 32-bit elements and produce 4 registers holding
/// 16-bit elements.
///
/// The order of elements are _not_ maintained.
///
/// # Important note on saturation
/// This routine produces garbage results if the input elements are beyond the maximum value
/// that can be represented in a [u16] value, meaning any value over [u16::MAX] produces
/// invalid data.
pub(super) fn pack_u32_to_u16_unordered(data: [__m256i; 8]) -> [__m256i; 4] {
    let lo_1 = _mm256_or_si256(data[0], _mm256_slli_epi32::<16>(data[2]));
    let lo_2 = _mm256_or_si256(data[1], _mm256_slli_epi32::<16>(data[3]));
    let hi_1 = _mm256_or_si256(data[4], _mm256_slli_epi32::<16>(data[6]));
    let hi_2 = _mm256_or_si256(data[5], _mm256_slli_epi32::<16>(data[7]));

    [lo_1, lo_2, hi_1, hi_2]
}

#[target_feature(enable = "avx2")]
/// Unpack 4 sets of registers containing 16-bit elements and produce 8 registers holding
/// 32-bit elements.
pub(super) fn unpack_u16_to_u32_unordered(data: [__m256i; 4]) -> [__m256i; 8] {
    let zeroes = _mm256_setzero_si256();

    [
        _mm256_blend_epi16::<0b10101010>(data[0], zeroes),
        _mm256_blend_epi16::<0b10101010>(data[1], zeroes),
        _mm256_srli_epi32::<16>(data[0]),
        _mm256_srli_epi32::<16>(data[1]),
        _mm256_blend_epi16::<0b10101010>(data[2], zeroes),
        _mm256_blend_epi16::<0b10101010>(data[3], zeroes),
        _mm256_srli_epi32::<16>(data[2]),
        _mm256_srli_epi32::<16>(data[3]),
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
/// Given 8 sets of 32-bit registers, pack them into 4 sets of 16-bit registers, then
/// split each 16-bit element into a high and low halves, returning the 8-bit halves in separate
/// registers in their high and low forms respectively.
///
/// The order of elements is maintained.
///
/// # Important note on saturation
/// This routine produces garbage results if the input elements are beyond the maximum value
/// that can be represented in a [u16] value, meaning any value over [u16::MAX] produces
/// invalid data.
pub(super) fn pack_u32_to_u16_split_ordered(data: [__m256i; 8]) -> ([__m256i; 2], [__m256i; 2]) {
    let packed_u16 = pack_u32_to_u16_ordered(data);

    let mask = _mm256_set1_epi16(0x00FF);
    let lo_8bits = and_si256(packed_u16, mask);
    let hi_8bits = srli_epi16::<8, 4>(packed_u16);

    let packed_lo_8bits = pack_u16_to_u8_ordered(lo_8bits);
    let packed_hi_8bits = pack_u16_to_u8_ordered(hi_8bits);

    (packed_hi_8bits, packed_lo_8bits)
}

#[target_feature(enable = "avx2")]
/// Given 8 sets of 32-bit registers, pack them into 4 sets of 16-bit registers, then
/// split each 16-bit element into a high and low halves, returning the 8-bit halves in separate
/// registers in their high and low forms respectively.
///
/// The order of elements are not maintained.
///
/// # Important note on saturation
/// This routine produces garbage results if the input elements are beyond the maximum value
/// that can be represented in a [u16] value, meaning any value over [u16::MAX] produces
/// invalid data.
pub(super) fn pack_u32_to_u16_split_unordered(data: [__m256i; 8]) -> ([__m256i; 2], [__m256i; 2]) {
    let packed = pack_u32_to_u16_unordered(data);

    let lo_mask = _mm256_set1_epi16(0x00FFu16 as i16);
    let hi_mask = _mm256_set1_epi16(0xFF00u16 as i16);

    let lo_8bits_1a = _mm256_slli_epi16::<8>(packed[2]);
    let lo_8bits_1b = _mm256_slli_epi16::<8>(packed[3]);
    let lo_8bits_2a = _mm256_and_si256(packed[0], lo_mask);
    let lo_8bits_2b = _mm256_and_si256(packed[1], lo_mask);

    let lo_8bits_a = _mm256_or_si256(lo_8bits_1a, lo_8bits_2a);
    let lo_8bits_b = _mm256_or_si256(lo_8bits_1b, lo_8bits_2b);

    let hi_8bits_1a = _mm256_srli_epi16::<8>(packed[0]);
    let hi_8bits_1b = _mm256_srli_epi16::<8>(packed[1]);
    let hi_8bits_2a = _mm256_and_si256(packed[2], hi_mask);
    let hi_8bits_2b = _mm256_and_si256(packed[3], hi_mask);

    let hi_8bits_a = _mm256_or_si256(hi_8bits_1a, hi_8bits_2a);
    let hi_8bits_b = _mm256_or_si256(hi_8bits_1b, hi_8bits_2b);

    ([hi_8bits_a, hi_8bits_b], [lo_8bits_a, lo_8bits_b])
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
/// Shift all registers right by [IMM8] in 32-bit lanes.
pub(super) fn srli_epi32<const IMM8: i32, const N: usize>(mut data: [__m256i; N]) -> [__m256i; N] {
    let mut i = 0;
    while i < N {
        data[i] = _mm256_srli_epi32::<IMM8>(data[i]);
        i += 1;
    }
    data
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

#[target_feature(enable = "avx2")]
/// Shift all registers right by [IMM8] in 32-bit lanes.
pub(super) fn slli_epi32<const IMM8: i32, const N: usize>(mut data: [__m256i; N]) -> [__m256i; N] {
    let mut i = 0;
    while i < N {
        data[i] = _mm256_slli_epi32::<IMM8>(data[i]);
        i += 1;
    }
    data
}

#[cfg(test)]
mod tests {
    use std::cmp;

    use super::*;
    use crate::X64;
    use crate::uint32::avx2::data::{load_si256x2, load_si256x4, load_u32x64};
    use crate::uint32::test_util::*;

    #[test]
    #[cfg_attr(not(target_feature = "avx2"), ignore)]
    fn test_pack_u32_u16_ordered() {
        let mut input = [0; X64];
        #[allow(clippy::needless_range_loop)]
        for i in 0..X64 {
            input[i] = i as u32;
        }

        let data = unsafe { load_u32x64(&input) };
        let packed = unsafe { pack_u32_to_u16_ordered(data) };

        let mut expected = [0; X64];
        #[allow(clippy::needless_range_loop)]
        for i in 0..X64 {
            expected[i] = i as u16;
        }

        let view = unsafe { std::mem::transmute::<[__m256i; 4], [u16; X64]>(packed) };
        assert_eq!(view, expected);
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx2"), ignore)]
    fn test_pack_u32_u16_unordered() {
        let mut input = [0; X64];
        #[allow(clippy::needless_range_loop)]
        for i in 0..X64 {
            input[i] = i as u32;
        }

        let data = unsafe { load_u32x64(&input) };
        let packed = unsafe { pack_u32_to_u16_unordered(data) };

        let view = unsafe { std::mem::transmute::<[__m256i; 4], [u16; X64]>(packed) };
        assert_eq!(view[..8], [0, 16, 1, 17, 2, 18, 3, 19]);
        assert_eq!(view[8..][..8], [4, 20, 5, 21, 6, 22, 7, 23]);
        assert_eq!(view[16..][..8], [8, 24, 9, 25, 10, 26, 11, 27]);
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx2"), ignore)]
    fn test_unpack_u16_u32_ordered() {
        let mut input = [0; X64];
        #[allow(clippy::needless_range_loop)]
        for i in 0..X64 {
            input[i] = i as u32;
        }

        let data = unsafe { load_u32x64(&input) };
        let packed = unsafe { pack_u32_to_u16_ordered(data) };
        let unpacked = unsafe { unpack_u16_to_u32_ordered(packed) };

        let view = unsafe { std::mem::transmute::<[__m256i; 8], [u32; X64]>(unpacked) };
        assert_eq!(view, input);
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx2"), ignore)]
    fn test_unpack_u16_u32_unordered() {
        let mut input = [0; X64];
        #[allow(clippy::needless_range_loop)]
        for i in 0..X64 {
            input[i] = i as u32;
        }

        let data = unsafe { load_u32x64(&input) };
        let packed = unsafe { pack_u32_to_u16_unordered(data) };
        let unpacked = unsafe { unpack_u16_to_u32_unordered(packed) };

        let view = unsafe { std::mem::transmute::<[__m256i; 8], [u32; X64]>(unpacked) };
        assert_eq!(view, input);
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx2"), ignore)]
    fn test_pack_u32_u8_ordered() {
        let mut input = [0; X64];
        #[allow(clippy::needless_range_loop)]
        for i in 0..X64 {
            input[i] = i as u32;
        }

        let data = unsafe { load_u32x64(&input) };
        let packed = unsafe { pack_u32_to_u8_ordered(data) };

        let mut expected = [0; X64];
        #[allow(clippy::needless_range_loop)]
        for i in 0..X64 {
            expected[i] = cmp::min(i as u8, u8::MAX);
        }

        let view = unsafe { std::mem::transmute::<[__m256i; 2], [u8; X64]>(packed) };
        assert_eq!(view, expected);
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx2"), ignore)]
    fn test_pack_u32_u8_unordered() {
        let mut input = [0; X64];
        #[allow(clippy::needless_range_loop)]
        for i in 0..X64 {
            input[i] = i as u32;
        }

        let data = unsafe { load_u32x64(&input) };
        let packed = unsafe { pack_u32_to_u8_unordered(data) };

        let view = unsafe { std::mem::transmute::<[__m256i; 2], [u8; X64]>(packed) };
        assert_eq!(view[..8], [0, 32, 16, 48, 1, 33, 17, 49]);
        assert_eq!(view[8..][..8], [2, 34, 18, 50, 3, 35, 19, 51]);
        assert_eq!(view[16..][..8], [4, 36, 20, 52, 5, 37, 21, 53]);
        assert_eq!(view[24..][..8], [6, 38, 22, 54, 7, 39, 23, 55]);
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx2"), ignore)]
    fn test_unpack_u8_u32_ordered() {
        let mut input = [0; X64];
        #[allow(clippy::needless_range_loop)]
        for i in 0..X64 {
            input[i] = i as u32;
        }

        let data = unsafe { load_u32x64(&input) };
        let packed = unsafe { pack_u32_to_u8_ordered(data) };
        let unpacked = unsafe { unpack_u8_to_u32_ordered(packed) };

        let view = unsafe { std::mem::transmute::<[__m256i; 8], [u32; X64]>(unpacked) };
        assert_eq!(view, input);
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx2"), ignore)]
    fn test_unpack_u8_u32_unordered() {
        let mut input = [0; X64];
        #[allow(clippy::needless_range_loop)]
        for i in 0..X64 {
            input[i] = i as u32;
        }

        let data = unsafe { load_u32x64(&input) };
        let packed = unsafe { pack_u32_to_u8_unordered(data) };
        let unpacked = unsafe { unpack_u8_to_u32_unordered(packed) };

        let view = unsafe { std::mem::transmute::<[__m256i; 8], [u32; X64]>(unpacked) };
        assert_eq!(view, input);
    }

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
    fn test_pack_u32_to_u16_unordered_layout() {
        let input: [u32; X64] = std::array::from_fn(|i| i as u32);

        let data = unsafe { load_u32x64(&input) };
        let packed = unsafe { pack_u32_to_u16_unordered(data) };

        let view = unsafe { std::mem::transmute::<[__m256i; 4], [u16; X64]>(packed) };
        assert_eq!(view, PACK_U32_TO_U16_EXPECTED_UNORDERED_LAYOUT,);
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx2"), ignore)]
    fn test_pack_u32_to_u8_unordered_layout() {
        let input: [u32; X64] = std::array::from_fn(|i| i as u32);

        let data = unsafe { load_u32x64(&input) };
        let packed = unsafe { pack_u32_to_u8_unordered(data) };

        let view = unsafe { std::mem::transmute::<[__m256i; 2], [u8; X64]>(packed) };
        assert_eq!(view, PACK_U32_TO_U8_EXPECTED_UNORDERED_LAYOUT,);
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
