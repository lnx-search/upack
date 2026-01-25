use std::arch::x86_64::*;

#[inline]
#[allow(non_snake_case)]
pub const fn _mm_shuffle(z: u32, y: u32, x: u32, w: u32) -> i32 {
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}

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
    let permute_mask = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

    let pack_block = |a, b, c, d| {
        let p1 = _mm256_packus_epi32(a, b);
        let p2 = _mm256_packus_epi32(c, d);

        let packed = _mm256_packus_epi16(p1, p2);

        _mm256_permutevar8x32_epi32(packed, permute_mask)
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
    let inv_permute_mask = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
    let zero = _mm256_setzero_si256();

    let unpack_block = |packed: __m256i| -> [__m256i; 4] {
        let unpermuted = _mm256_permutevar8x32_epi32(packed, inv_permute_mask);

        let lo_16s = _mm256_unpacklo_epi8(unpermuted, zero);
        let hi_16s = _mm256_unpackhi_epi8(unpermuted, zero);

        [
            _mm256_unpacklo_epi16(lo_16s, zero),
            _mm256_unpackhi_epi16(lo_16s, zero),
            _mm256_unpacklo_epi16(hi_16s, zero),
            _mm256_unpackhi_epi16(hi_16s, zero),
        ]
    };

    let block0 = unpack_block(data[0]);
    let block1 = unpack_block(data[1]);

    [
        block0[0], block0[1], block0[2], block0[3], block1[0], block1[1], block1[2], block1[3],
    ]
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
    let pack_block = |a, b, c, d| {
        let p1 = _mm256_packus_epi32(a, b);
        let p2 = _mm256_packus_epi32(c, d);
        _mm256_packus_epi16(p1, p2)
    };

    [
        pack_block(data[0], data[1], data[2], data[3]),
        pack_block(data[4], data[5], data[6], data[7]),
    ]
}

#[target_feature(enable = "avx2")]
/// Unpack 2 sets of registers containing 8-bit elements and produce 8 registers holding
/// 32-bit elements.
pub(super) fn unpack_u8_to_u32_unordered(data: [__m256i; 2]) -> [__m256i; 8] {
    let zero = _mm256_setzero_si256();

    let unpack_block = |packed: __m256i| -> [__m256i; 4] {
        let lo_16s = _mm256_unpacklo_epi8(packed, zero);
        let hi_16s = _mm256_unpackhi_epi8(packed, zero);

        [
            _mm256_unpacklo_epi16(lo_16s, zero),
            _mm256_unpackhi_epi16(lo_16s, zero),
            _mm256_unpacklo_epi16(hi_16s, zero),
            _mm256_unpackhi_epi16(hi_16s, zero),
        ]
    };

    let block0 = unpack_block(data[0]);
    let block1 = unpack_block(data[1]);

    [
        block0[0], block0[1], block0[2], block0[3], block1[0], block1[1], block1[2], block1[3],
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
    [
        _mm256_packus_epi32(data[0], data[1]),
        _mm256_packus_epi32(data[2], data[3]),
        _mm256_packus_epi32(data[4], data[5]),
        _mm256_packus_epi32(data[6], data[7]),
    ]
}

#[target_feature(enable = "avx2")]
/// Unpack 4 sets of registers containing 16-bit elements and produce 8 registers holding
/// 32-bit elements.
pub(super) fn unpack_u16_to_u32_unordered(data: [__m256i; 4]) -> [__m256i; 8] {
    let zero = _mm256_setzero_si256();

    let unpack_block = |packed| {
        [
            _mm256_unpacklo_epi16(packed, zero),
            _mm256_unpackhi_epi16(packed, zero),
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
/// Pack 4 sets of registers containing 16-bit elements and produce 2 registers holding
/// 8-bit elements.
///
/// The order of elements is maintained.
///
/// # Important note on saturation
/// This routine produces garbage results if the input elements are beyond the maximum value
/// that can be represented in a [u8] value, meaning any value over [u8::MAX] produces
/// invalid data.
pub(super) fn pack_u16_to_u8_unordered(data: [__m256i; 4]) -> [__m256i; 2] {
    [
        _mm256_packus_epi16(data[0], data[1]),
        _mm256_packus_epi16(data[2], data[3]),
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
    let packed_u16 = pack_u32_to_u16_unordered(data);

    let mask = _mm256_set1_epi16(0x00FF);
    let lo_8bits = and_si256(packed_u16, mask);
    let hi_8bits = srli_epi16::<8, 4>(packed_u16);

    let packed_lo_8bits = pack_u16_to_u8_unordered(lo_8bits);
    let packed_hi_8bits = pack_u16_to_u8_unordered(hi_8bits);

    (packed_hi_8bits, packed_lo_8bits)
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
/// Perform a bitwise AND on all provided registers with another broadcast register _after_
/// performing a bitwise NOT of the broadcast register.
pub(super) fn andnot_si256<const N: usize>(mut data: [__m256i; N], mask: __m256i) -> [__m256i; N] {
    let mut i = 0;
    while i < N {
        data[i] = _mm256_andnot_si256(mask, data[i]);
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

#[cfg(test)]
mod tests {
    use std::cmp;

    use super::*;
    use crate::X64;
    use crate::uint32::avx2::data::load_u32x64;

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
        assert_eq!(view[..8], [0, 1, 2, 3, 8, 9, 10, 11]);
        assert_eq!(view[8..][..8], [4, 5, 6, 7, 12, 13, 14, 15]);
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx2"), ignore)]
    fn test_unpack_u32_u16_ordered() {
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
    fn test_unpack_u32_u16_unordered() {
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
        assert_eq!(view[..8], [0, 1, 2, 3, 8, 9, 10, 11]);
        assert_eq!(view[8..][..8], [16, 17, 18, 19, 24, 25, 26, 27]);
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx2"), ignore)]
    fn test_unpack_u32_u8_ordered() {
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
    fn test_unpack_u32_u8_unordered() {
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
}
