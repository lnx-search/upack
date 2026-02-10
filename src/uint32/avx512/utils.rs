use std::arch::x86_64::*;

pub const _MM_TERNLOG_A: i32 = 0xF0; // 11110000
pub const _MM_TERNLOG_B: i32 = 0xCC; // 11001100
pub const _MM_TERNLOG_C: i32 = 0xAA; // 10101010

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Pack 4 sets of registers containing 32-bit elements and produce 1 register holding
/// 8-bit elements.
///
/// The order of elements is maintained.
///
/// # Important note on saturation
/// This routine produces garbage results if the input elements are beyond the maximum value
/// that can be represented in a [u8] value, meaning any value over [u8::MAX] produces
/// invalid data.
pub(super) fn pack_u32_to_u8_ordered(data: [__m512i; 4]) -> __m512i {
    let p1 = _mm512_packus_epi32(data[0], data[1]);
    let p2 = _mm512_packus_epi32(data[2], data[3]);
    let packed = _mm512_packus_epi16(p1, p2);

    let permute_mask = _mm512_setr_epi32(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);

    _mm512_permutexvar_epi32(permute_mask, packed)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack 1 register containing 8-bit elements and produce 4 registers holding
/// 32-bit elements.
pub(super) fn unpack_u8_to_u32_ordered(data: __m512i) -> [__m512i; 4] {
    let permute_mask = _mm512_setr_epi32(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    let zeroes = _mm512_setzero_si512();

    let unpermuted = _mm512_permutexvar_epi32(permute_mask, data);
    let lo_16s = _mm512_unpacklo_epi8(unpermuted, zeroes);
    let hi_16s = _mm512_unpackhi_epi8(unpermuted, zeroes);

    [
        _mm512_unpacklo_epi16(lo_16s, zeroes),
        _mm512_unpackhi_epi16(lo_16s, zeroes),
        _mm512_unpacklo_epi16(hi_16s, zeroes),
        _mm512_unpackhi_epi16(hi_16s, zeroes),
    ]
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Pack 4 sets of registers containing 32-bit elements and produce 2 registers holding
/// 16-bit elements.
///
/// The order of elements is maintained.
///
/// # Important note on saturation
/// This routine produces garbage results if the input elements are beyond the maximum value
/// that can be represented in a [u16] value, meaning any value over [u16::MAX] produces
/// invalid data.
pub(super) fn pack_u32_to_u16_ordered(data: [__m512i; 4]) -> [__m512i; 2] {
    let p1 = _mm512_packus_epi32(data[0], data[1]);
    let p2 = _mm512_packus_epi32(data[2], data[3]);

    let permute_idx = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);

    let lo = _mm512_permutexvar_epi64(permute_idx, p1);
    let hi = _mm512_permutexvar_epi64(permute_idx, p2);

    [lo, hi]
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack 2 registers containing 16-bit elements and produce 4 registers holding
/// 32-bit elements.
pub(super) fn unpack_u16_to_u32_ordered(data: [__m512i; 2]) -> [__m512i; 4] {
    let lo_p1 = _mm512_castsi512_si256(data[0]);
    let lo_p2 = _mm512_extracti64x4_epi64(data[0], 1);

    let hi_p1 = _mm512_castsi512_si256(data[1]);
    let hi_p2 = _mm512_extracti64x4_epi64(data[1], 1);

    [
        _mm512_cvtepu16_epi32(lo_p1),
        _mm512_cvtepu16_epi32(lo_p2),
        _mm512_cvtepu16_epi32(hi_p1),
        _mm512_cvtepu16_epi32(hi_p2),
    ]
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Pack 2 sets of registers containing 16-bit elements and produce 1 register holding
/// 8-bit elements.
///
/// The order of elements is **not** maintained.
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
/// Pack 4 sets of registers containing 32-bit elements and produce 1 register holding
/// 8-bit elements.
///
/// The order of elements is **not** maintained.
///
/// # Important note on saturation
/// This routine produces garbage results if the input elements are beyond the maximum value
/// that can be represented in a [u8] value, meaning any value over [u8::MAX] produces
/// invalid data.
pub(super) fn pack_u32_to_u8_unordered(data: [__m512i; 4]) -> __m512i {
    let packed = pack_u32_to_u16_unordered(data);
    pack_u16_to_u8_unordered(packed)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack 1 register containing 8-bit elements and produce 4 registers holding
/// 32-bit elements.
pub(super) fn unpack_u8_to_u32_unordered(packed: __m512i) -> [__m512i; 4] {
    let zeroes = _mm512_setzero_si512();
    let lo_16s = _mm512_unpacklo_epi8(packed, zeroes);
    let hi_16s = _mm512_unpackhi_epi8(packed, zeroes);

    [
        _mm512_unpacklo_epi16(lo_16s, zeroes),
        _mm512_unpackhi_epi16(lo_16s, zeroes),
        _mm512_unpacklo_epi16(hi_16s, zeroes),
        _mm512_unpackhi_epi16(hi_16s, zeroes),
    ]
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Pack 4 sets of registers containing 32-bit elements and produce 2 registers holding
/// 16-bit elements.
///
/// The order of elements is **not** maintained.
///
/// # Important note on saturation
/// This routine produces garbage results if the input elements are beyond the maximum value
/// that can be represented in a [u16] value, meaning any value over [u16::MAX] produces
/// invalid data.
pub(super) fn pack_u32_to_u16_unordered(data: [__m512i; 4]) -> [__m512i; 2] {
    let lo = _mm512_packus_epi32(data[0], data[1]);
    let hi = _mm512_packus_epi32(data[2], data[3]);

    [lo, hi]
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack 2 registers containing 16-bit elements and produce 4 registers holding
/// 32-bit elements.
pub(super) fn unpack_u16_to_u32_unordered(data: [__m512i; 2]) -> [__m512i; 4] {
    let zeroes = _mm512_setzero_si512();
    [
        _mm512_unpacklo_epi16(data[0], zeroes),
        _mm512_unpackhi_epi16(data[0], zeroes),
        _mm512_unpacklo_epi16(data[1], zeroes),
        _mm512_unpackhi_epi16(data[1], zeroes),
    ]
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Pack 2 sets of registers containing 16-bit elements and produce 1 register holding
/// 8-bit elements.
///
/// The order of elements is **not** maintained.
///
/// # Important note on saturation
/// This routine produces garbage results if the input elements are beyond the maximum value
/// that can be represented in a [u16] value, meaning any value over [u16::MAX] produces
/// invalid data.
pub(super) fn pack_u16_to_u8_unordered(data: [__m512i; 2]) -> __m512i {
    _mm512_packus_epi16(data[0], data[1])
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Given 8 sets of 32-bit registers, pack them into 2 sets of 16-bit registers, then
/// split each 16-bit element into a high and low halves, returning the 8-bit halves in separate
/// registers in their high and low forms respectively.
///
/// The order of elements are maintained.
///
/// # Important note on saturation
/// This routine produces garbage results if the input elements are beyond the maximum value
/// that can be represented in a [u16] value, meaning any value over [u16::MAX] produces
/// invalid data.
pub(super) fn pack_u32_to_u16_split_ordered(data: [__m512i; 4]) -> (__m512i, __m512i) {
    let packed_u16 = pack_u32_to_u16_ordered(data);

    let mask = _mm512_set1_epi16(0x00FF);
    let lo_8bits = and_si512(packed_u16, mask);
    let hi_8bits = srli_epi16::<8, 2>(packed_u16);

    let packed_lo_8bits = pack_u16_to_u8_ordered(lo_8bits);
    let packed_hi_8bits = pack_u16_to_u8_ordered(hi_8bits);

    (packed_hi_8bits, packed_lo_8bits)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Given 8 sets of 32-bit registers, pack them into 2 sets of 16-bit registers, then
/// split each 16-bit element into a high and low halves, returning the 8-bit halves in separate
/// registers in their high and low forms respectively.
///
/// The order of elements are not maintained.
///
/// # Important note on saturation
/// This routine produces garbage results if the input elements are beyond the maximum value
/// that can be represented in a [u16] value, meaning any value over [u16::MAX] produces
/// invalid data.
pub(super) fn pack_u32_to_u16_split_unordered(data: [__m512i; 4]) -> (__m512i, __m512i) {
    let packed_u16 = pack_u32_to_u16_unordered(data);

    let mask = _mm512_set1_epi16(0x00FF);
    let lo_8bits = and_si512(packed_u16, mask);
    let hi_8bits = srli_epi16::<8, 2>(packed_u16);

    let packed_lo_8bits = pack_u16_to_u8_unordered(lo_8bits);
    let packed_hi_8bits = pack_u16_to_u8_unordered(hi_8bits);

    (packed_hi_8bits, packed_lo_8bits)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack 1 register containing 8-bit elements and produce 2 registers holding
/// 32-bit elements.
pub(super) fn unpack_u8_to_u16_unordered(data: __m512i) -> [__m512i; 2] {
    let zeroes = _mm512_setzero_si512();
    let lo = _mm512_unpacklo_epi8(data, zeroes);
    let hi = _mm512_unpackhi_epi8(data, zeroes);
    [lo, hi]
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
/// Perform a bitwise AND on all provided registers with another broadcast register _after_
/// performing a bitwise NOT of the broadcast register.
pub(super) fn andnot_si512<const N: usize>(mut data: [__m512i; N], mask: __m512i) -> [__m512i; N] {
    let mut i = 0;
    while i < N {
        data[i] = _mm512_andnot_si512(mask, data[i]);
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

#[target_feature(enable = "avx512f")]
/// Perform a bitwise OR on the two sets of registers.
pub(super) fn ternary_epi32_all<const IMM8: i32, const N: usize>(
    mut d1: [__m512i; N],
    d2: [__m512i; N],
    d3: [__m512i; N],
) -> [__m512i; N] {
    let mut i = 0;
    while i < N {
        d1[i] = _mm512_ternarylogic_epi32::<IMM8>(d1[i], d2[i], d3[i]);
        i += 1;
    }
    d1
}

#[target_feature(enable = "avx512f")]
/// Shift all registers right by [IMM8] in 32-bit lanes.
pub(super) fn srli_epi32<const IMM8: u32, const N: usize>(mut data: [__m512i; N]) -> [__m512i; N] {
    let mut i = 0;
    while i < N {
        data[i] = _mm512_srli_epi32::<IMM8>(data[i]);
        i += 1;
    }
    data
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

#[target_feature(enable = "avx512f")]
/// Shift all registers left by [IMM8] in 32-bit lanes.
pub(super) fn slli_epi32<const IMM8: u32, const N: usize>(mut data: [__m512i; N]) -> [__m512i; N] {
    let mut i = 0;
    while i < N {
        data[i] = _mm512_slli_epi32::<IMM8>(data[i]);
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
    use std::cmp;

    use super::*;
    use crate::X64;
    use crate::uint32::avx512::data::{load_si512x2, load_u32x64};
    use crate::uint32::test_util::*;

    #[test]
    #[cfg_attr(
        not(all(target_feature = "avx512f", target_feature = "avx512bw")),
        ignore
    )]
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

        let view = unsafe { std::mem::transmute::<[__m512i; 2], [u16; X64]>(packed) };
        assert_eq!(view, expected);
    }

    #[test]
    #[cfg_attr(
        not(all(target_feature = "avx512f", target_feature = "avx512bw")),
        ignore
    )]
    fn test_pack_u32_u16_unordered() {
        let mut input = [0; X64];
        #[allow(clippy::needless_range_loop)]
        for i in 0..X64 {
            input[i] = i as u32;
        }

        let data = unsafe { load_u32x64(&input) };
        let packed = unsafe { pack_u32_to_u16_unordered(data) };

        let view = unsafe { std::mem::transmute::<[__m512i; 2], [u16; X64]>(packed) };
        assert_eq!(
            view[..16],
            [0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23]
        );
        assert_eq!(
            view[16..][..16],
            [8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31]
        );
    }

    #[test]
    #[cfg_attr(
        not(all(target_feature = "avx512f", target_feature = "avx512bw")),
        ignore
    )]
    fn test_unpack_u32_u16_ordered() {
        let mut input = [0; X64];
        #[allow(clippy::needless_range_loop)]
        for i in 0..X64 {
            input[i] = i as u32;
        }

        let data = unsafe { load_u32x64(&input) };
        let packed = unsafe { pack_u32_to_u16_ordered(data) };
        let unpacked = unsafe { unpack_u16_to_u32_ordered(packed) };

        let view = unsafe { std::mem::transmute::<[__m512i; 4], [u32; X64]>(unpacked) };
        assert_eq!(view, input);
    }

    #[test]
    #[cfg_attr(
        not(all(target_feature = "avx512f", target_feature = "avx512bw")),
        ignore
    )]
    fn test_unpack_u32_u16_unordered() {
        let mut input = [0; X64];
        #[allow(clippy::needless_range_loop)]
        for i in 0..X64 {
            input[i] = i as u32;
        }

        let data = unsafe { load_u32x64(&input) };
        let packed = unsafe { pack_u32_to_u16_unordered(data) };
        let unpacked = unsafe { unpack_u16_to_u32_unordered(packed) };

        let view = unsafe { std::mem::transmute::<[__m512i; 4], [u32; X64]>(unpacked) };
        assert_eq!(view, input);
    }

    #[test]
    #[cfg_attr(
        not(all(target_feature = "avx512f", target_feature = "avx512bw")),
        ignore
    )]
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

        let view = unsafe { std::mem::transmute::<__m512i, [u8; X64]>(packed) };
        assert_eq!(view, expected);
    }

    #[test]
    #[cfg_attr(
        not(all(target_feature = "avx512f", target_feature = "avx512bw")),
        ignore
    )]
    fn test_pack_u32_u8_unordered() {
        let mut input = [0; X64];
        #[allow(clippy::needless_range_loop)]
        for i in 0..X64 {
            input[i] = i as u32;
        }

        let data = unsafe { load_u32x64(&input) };
        let packed = unsafe { pack_u32_to_u8_unordered(data) };

        let view = unsafe { std::mem::transmute::<__m512i, [u8; X64]>(packed) };
        assert_eq!(
            view[..16],
            [0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51]
        );
        assert_eq!(
            view[16..][..16],
            [4, 5, 6, 7, 20, 21, 22, 23, 36, 37, 38, 39, 52, 53, 54, 55]
        );
    }

    #[test]
    #[cfg_attr(
        not(all(target_feature = "avx512f", target_feature = "avx512bw")),
        ignore
    )]
    fn test_unpack_u32_u8_ordered() {
        let mut input = [0; X64];
        #[allow(clippy::needless_range_loop)]
        for i in 0..X64 {
            input[i] = i as u32;
        }

        let data = unsafe { load_u32x64(&input) };
        let packed = unsafe { pack_u32_to_u8_ordered(data) };
        let unpacked = unsafe { unpack_u8_to_u32_ordered(packed) };

        let view = unsafe { std::mem::transmute::<[__m512i; 4], [u32; X64]>(unpacked) };
        assert_eq!(view, input);
    }

    #[test]
    #[cfg_attr(
        not(all(target_feature = "avx512f", target_feature = "avx512bw")),
        ignore
    )]
    fn test_unpack_u32_u8_unordered() {
        let mut input = [0; X64];
        #[allow(clippy::needless_range_loop)]
        for i in 0..X64 {
            input[i] = i as u32;
        }

        let data = unsafe { load_u32x64(&input) };
        let packed = unsafe { pack_u32_to_u8_unordered(data) };
        let unpacked = unsafe { unpack_u8_to_u32_unordered(packed) };

        let view = unsafe { std::mem::transmute::<[__m512i; 4], [u32; X64]>(unpacked) };
        assert_eq!(view, input);
    }

    #[test]
    #[cfg_attr(
        not(all(target_feature = "avx512f", target_feature = "avx512bw")),
        ignore
    )]
    fn test_unpack_u8_u16_unordered() {
        let mut input = [0; X64];
        #[allow(clippy::needless_range_loop)]
        for i in 0..X64 {
            input[i] = i as u16;
        }

        let data = unsafe { load_si512x2(input.as_ptr().cast()) };
        let packed = unsafe { pack_u16_to_u8_unordered(data) };
        let unpacked = unsafe { unpack_u8_to_u16_unordered(packed) };

        let view = unsafe { std::mem::transmute::<[__m512i; 2], [u16; X64]>(unpacked) };
        assert_eq!(view, input);
    }

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
    fn test_pack_u16_to_u8_unordered_layout() {
        let input: [u16; X64] = std::array::from_fn(|i| i as u16);

        let data = unsafe { load_si512x2(input.as_ptr().cast()) };
        let packed = unsafe { pack_u16_to_u8_unordered(data) };

        let view = unsafe { std::mem::transmute::<__m512i, [u8; X64]>(packed) };
        assert_eq!(view, PACK_U16_TO_U8_EXPECTED_UNORDERED_LAYOUT,);
    }

    #[test]
    #[cfg_attr(
        not(all(target_feature = "avx512f", target_feature = "avx512bw")),
        ignore
    )]
    fn test_pack_u32_to_u16_unordered_layout() {
        let input: [u32; X64] = std::array::from_fn(|i| i as u32);

        let data = unsafe { load_u32x64(&input) };
        let packed = unsafe { pack_u32_to_u16_unordered(data) };

        let view = unsafe { std::mem::transmute::<[__m512i; 2], [u16; X64]>(packed) };
        assert_eq!(view, PACK_U32_TO_U16_EXPECTED_UNORDERED_LAYOUT,);
    }

    #[test]
    #[cfg_attr(
        not(all(target_feature = "avx512f", target_feature = "avx512bw")),
        ignore
    )]
    fn test_pack_u32_to_u8_unordered_layout() {
        let input: [u32; X64] = std::array::from_fn(|i| i as u32);

        let data = unsafe { load_u32x64(&input) };
        let packed = unsafe { pack_u32_to_u8_unordered(data) };

        let view = unsafe { std::mem::transmute::<__m512i, [u8; X64]>(packed) };
        assert_eq!(view, PACK_U32_TO_U8_EXPECTED_UNORDERED_LAYOUT,);
    }
}
