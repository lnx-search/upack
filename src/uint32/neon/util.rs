#![allow(clippy::needless_range_loop)]

use std::arch::aarch64::*;

use super::polyfill::*;

#[target_feature(enable = "neon")]
/// Pack 8 sets of registers containing 32-bit elements and produce 2 registers holding
/// 8-bit elements.
///
/// The order of elements is maintained.
pub(super) fn pack_u32_to_u8_ordered(data: [u32x8; 8]) -> [u8x32; 2] {
    let partially_packed = pack_u32_to_u16_ordered(data);
    pack_u16_to_u8_ordered(partially_packed)
}

#[target_feature(enable = "neon")]
/// Pack 8 sets of registers containing 32-bit elements and produce 4 registers holding
/// 16-bit elements.
///
/// The order of elements is maintained.
pub(super) fn pack_u32_to_u16_ordered(data: [u32x8; 8]) -> [u16x16; 4] {
    [
        _neon_combine_u16(_neon_cvteu32_u16(data[0]), _neon_cvteu32_u16(data[1])),
        _neon_combine_u16(_neon_cvteu32_u16(data[2]), _neon_cvteu32_u16(data[3])),
        _neon_combine_u16(_neon_cvteu32_u16(data[4]), _neon_cvteu32_u16(data[5])),
        _neon_combine_u16(_neon_cvteu32_u16(data[6]), _neon_cvteu32_u16(data[7])),
    ]
}

#[target_feature(enable = "neon")]
/// Pack 4 sets of registers containing 16-bit elements and produce 2 registers holding
/// 9-bit elements.
///
/// The order of elements is maintained.
pub(super) fn pack_u16_to_u8_ordered(data: [u16x16; 4]) -> [u8x32; 2] {
    [
        _neon_combine_u8(_neon_cvteu16_u8(data[0]), _neon_cvteu16_u8(data[1])),
        _neon_combine_u8(_neon_cvteu16_u8(data[2]), _neon_cvteu16_u8(data[3])),
    ]
}

#[target_feature(enable = "neon")]
/// Unpack 2 sets of registers containing 8-bit elements and produce 8 registers holding
/// 32-bit elements.
pub(super) fn unpack_u8_to_u32_ordered(data: [u8x32; 2]) -> [u32x8; 8] {
    let partially_unpacked = unpack_u8_to_u16_ordered(data);
    unpack_u16_to_u32_ordered(partially_unpacked)
}

#[target_feature(enable = "neon")]
/// Unpack 2 sets of registers containing 8-bit elements and produce 4 registers holding
/// 16-bit elements.
pub(super) fn unpack_u8_to_u16_ordered(data: [u8x32; 2]) -> [u16x16; 4] {
    let parts = [
        _neon_extract_u8x32::<0>(data[0]),
        _neon_extract_u8x32::<1>(data[0]),
        _neon_extract_u8x32::<0>(data[1]),
        _neon_extract_u8x32::<1>(data[1]),
    ];

    [
        _neon_cvteu8_u16(parts[0]),
        _neon_cvteu8_u16(parts[1]),
        _neon_cvteu8_u16(parts[2]),
        _neon_cvteu8_u16(parts[3]),
    ]
}

#[target_feature(enable = "neon")]
/// Unpack 4 sets of registers containing 16-bit elements and produce 8 registers holding
/// 32-bit elements.
pub(super) fn unpack_u16_to_u32_ordered(data: [u16x16; 4]) -> [u32x8; 8] {
    let split_u16s = [
        _neon_extract_u16x16::<0>(data[0]),
        _neon_extract_u16x16::<1>(data[0]),
        _neon_extract_u16x16::<0>(data[1]),
        _neon_extract_u16x16::<1>(data[1]),
        _neon_extract_u16x16::<0>(data[2]),
        _neon_extract_u16x16::<1>(data[2]),
        _neon_extract_u16x16::<0>(data[3]),
        _neon_extract_u16x16::<1>(data[3]),
    ];

    [
        _neon_cvteu16_u8(split_u16s[0]),
        _neon_cvteu16_u8(split_u16s[1]),
        _neon_cvteu16_u8(split_u16s[2]),
        _neon_cvteu16_u8(split_u16s[3]),
        _neon_cvteu16_u8(split_u16s[4]),
        _neon_cvteu16_u8(split_u16s[5]),
        _neon_cvteu16_u8(split_u16s[6]),
        _neon_cvteu16_u8(split_u16s[7]),
    ]
}

#[target_feature(enable = "neon")]
/// Pack 8 sets of registers containing 32-bit elements and produce 2 registers holding
/// 8-bit elements.
///
/// The order of elements are _not_ maintained.
pub(super) fn pack_u32_to_u8_unordered(data: [u32x8; 8]) -> [u8x32; 2] {
    let shift_1 = [data[0], data[1]];
    let shift_2 = [_neon_slli_u32::<8>(data[4]), _neon_slli_u32::<8>(data[5])];
    let shift_3 = [_neon_slli_u32::<16>(data[2]), _neon_slli_u32::<16>(data[3])];
    let shift_4 = [_neon_slli_u32::<24>(data[6]), _neon_slli_u32::<24>(data[7])];

    let mut packed = shift_1;
    packed = or_u32x8_all(packed, shift_2);
    packed = or_u32x8_all(packed, shift_3);
    packed = or_u32x8_all(packed, shift_4);

    [packed[0].into(), packed[1].into()]
}

#[target_feature(enable = "neon")]
/// Pack 8 sets of registers containing 32-bit elements and produce 4 registers holding
/// 16-bit elements.
///
/// The order of elements are _not_ maintained.
pub(super) fn pack_u32_to_u16_unordered(data: [u32x8; 8]) -> [u16x16; 4] {
    let lo_1 = _neon_or_u32(data[0], _neon_slli_u32::<16>(data[2]));
    let lo_2 = _neon_or_u32(data[1], _neon_slli_u32::<16>(data[3]));
    let hi_1 = _neon_or_u32(data[4], _neon_slli_u32::<16>(data[6]));
    let hi_2 = _neon_or_u32(data[5], _neon_slli_u32::<16>(data[7]));

    [lo_1.into(), lo_2.into(), hi_1.into(), hi_2.into()]
}

#[target_feature(enable = "neon")]
/// Unpack 2 sets of registers containing 8-bit elements and produce 8 registers holding
/// 32-bit elements.
pub(super) fn unpack_u8_to_u32_unordered(data: [u8x32; 2]) -> [u32x8; 8] {
    let partially_unpacked = unpack_u8_to_u16_unordered(data);
    unpack_u16_to_u32_unordered(partially_unpacked)
}

#[inline]
#[target_feature(enable = "neon")]
/// Unpack 2 sets of registers containing 8-bit elements and produce 4 registers holding
/// 16-bit elements.
pub(super) fn unpack_u8_to_u16_unordered(data: [u8x32; 2]) -> [u16x16; 4] {
    [
        _neon_blend_every_other_u8(data[0], _neon_set1_u8(0)).into(),
        _neon_blend_every_other_u8(data[1], _neon_set1_u8(0)).into(),
        _neon_srli_u16::<8>(data[0].into()),
        _neon_srli_u16::<8>(data[1].into()),
    ]
}

#[inline]
#[target_feature(enable = "neon")]
/// Unpack 4 sets of registers containing 16-bit elements and produce 8 registers holding
/// 32-bit elements.
pub(super) fn unpack_u16_to_u32_unordered(data: [u16x16; 4]) -> [u32x8; 8] {
    [
        _neon_blend_every_other_u16(data[0], _neon_set1_u16(0)).into(),
        _neon_blend_every_other_u16(data[1], _neon_set1_u16(0)).into(),
        _neon_srli_u32::<16>(data[0].into()),
        _neon_srli_u32::<16>(data[1].into()),
        _neon_blend_every_other_u16(data[2], _neon_set1_u16(0)).into(),
        _neon_blend_every_other_u16(data[3], _neon_set1_u16(0)).into(),
        _neon_srli_u32::<16>(data[2].into()),
        _neon_srli_u32::<16>(data[3].into()),
    ]
}

#[target_feature(enable = "neon")]
/// Given 8 sets of 32-bit registers, pack them into 4 sets of 16-bit registers, then
/// split each 16-bit element into a high and low halves, returning the 8-bit halves in separate
/// registers in their high and low forms respectively.
///
/// The order of elements is maintained.
pub(super) fn pack_u32_to_u16_split_ordered(data: [u32x8; 8]) -> ([u8x32; 2], [u8x32; 2]) {
    let packed_u16 = pack_u32_to_u16_ordered(data);

    let mask = _neon_set1_u16(0x00FF);
    let lo_8bits = and_u16x16(packed_u16, mask);
    let hi_8bits = srli_u16x16::<8, 4>(packed_u16);

    let packed_lo_8bits = pack_u16_to_u8_ordered(lo_8bits);
    let packed_hi_8bits = pack_u16_to_u8_ordered(hi_8bits);

    (packed_hi_8bits, packed_lo_8bits)
}

#[target_feature(enable = "neon")]
/// Given 8 sets of 32-bit registers, pack them into 4 sets of 16-bit registers, then
/// split each 16-bit element into a high and low halves, returning the 8-bit halves in separate
/// registers in their high and low forms respectively.
///
/// The order of elements are not maintained.
pub(super) fn pack_u32_to_u16_split_unordered(data: [u32x8; 8]) -> ([u8x32; 2], [u8x32; 2]) {
    let packed = pack_u32_to_u16_unordered(data);

    let lo_mask = _neon_set1_u16(0x00FF);
    let hi_mask = _neon_set1_u16(0xFF00);

    let lo_8bits_1a = _neon_slli_u16::<8>(packed[2]);
    let lo_8bits_1b = _neon_slli_u16::<8>(packed[3]);
    let lo_8bits_2a = _neon_and_u16(packed[0], lo_mask);
    let lo_8bits_2b = _neon_and_u16(packed[1], lo_mask);

    let lo_8bits_a = _neon_or_u16(lo_8bits_1a, lo_8bits_2a);
    let lo_8bits_b = _neon_or_u16(lo_8bits_1b, lo_8bits_2b);

    let hi_8bits_1a = _neon_srli_u16::<8>(packed[0]);
    let hi_8bits_1b = _neon_srli_u16::<8>(packed[1]);
    let hi_8bits_2a = _neon_and_u16(packed[2], hi_mask);
    let hi_8bits_2b = _neon_and_u16(packed[3], hi_mask);

    let hi_8bits_a = _neon_or_u16(hi_8bits_1a, hi_8bits_2a);
    let hi_8bits_b = _neon_or_u16(hi_8bits_1b, hi_8bits_2b);

    (
        [hi_8bits_a.into(), hi_8bits_b.into()],
        [lo_8bits_a.into(), lo_8bits_b.into()],
    )
}

#[target_feature(enable = "neon")]
pub(super) fn pack_u8_to_u4_unordered(data: [u8x32; 2]) -> u8x32 {
    let shifted = _neon_slli_u8::<4>(data[1]);
    _neon_or_u8(data[0], shifted)
}

#[target_feature(enable = "neon")]
pub(super) fn unpack_u4_to_u8_unordered(data: u8x32) -> [u8x32; 2] {
    let mask = _neon_set1_u8(0x0F);
    let lo_4bits = _neon_and_u8(data, mask);
    let hi_4bits = _neon_srli_u8::<4>(data);
    [lo_4bits, hi_4bits]
}

#[target_feature(enable = "neon")]
pub(super) fn pack_u8_to_u2_unordered(data: [u8x32; 2]) -> u8x16 {
    let nibbles = pack_u8_to_u4_unordered(data);
    let lo = nibbles[0];
    let hi = nibbles[1];
    let reg = vorrq_u8(lo, vshlq_n_u8::<2>(hi));
    u8x16(reg)
}

#[target_feature(enable = "neon")]
pub(super) fn unpack_u2_to_u8_unordered(data: u8x16) -> [u8x32; 2] {
    let mask = vdupq_n_u8(0b0011_0011);
    let lo = vandq_u8(data.0, mask);
    let hi = vandq_u8(vshrq_n_u8::<2>(data.0), mask);
    let nibbles = _neon_combine_u8(u8x16(lo), u8x16(hi));
    unpack_u4_to_u8_unordered(nibbles)
}

#[inline]
#[target_feature(enable = "neon")]
/// Perform a bitwise AND on all provided registers with another broadcast register.
pub(super) fn and_u32x8<const N: usize>(mut data: [u32x8; N], mask: u32x8) -> [u32x8; N] {
    for i in 0..N {
        data[i] = _neon_and_u32(data[i], mask);
    }
    data
}

#[inline]
#[target_feature(enable = "neon")]
/// Perform a bitwise AND on all provided registers with another broadcast register.
pub(super) fn and_u16x16<const N: usize>(mut data: [u16x16; N], mask: u16x16) -> [u16x16; N] {
    for i in 0..N {
        data[i] = _neon_and_u16(data[i], mask);
    }
    data
}

#[inline]
#[target_feature(enable = "neon")]
/// Perform a bitwise AND on all provided registers with another broadcast register.
pub(super) fn and_u8x32<const N: usize>(mut data: [u8x32; N], mask: u8x32) -> [u8x32; N] {
    for i in 0..N {
        data[i] = _neon_and_u8(data[i], mask);
    }
    data
}

#[inline]
#[target_feature(enable = "neon")]
/// Perform a bitwise OR on all provided registers with another broadcast register.
pub(super) fn or_u32x8_all<const N: usize>(mut a: [u32x8; N], b: [u32x8; N]) -> [u32x8; N] {
    for i in 0..N {
        a[i] = _neon_or_u32(a[i], b[i]);
    }
    a
}

#[inline]
#[target_feature(enable = "neon")]
/// Perform a bitwise OR on all provided registers with another broadcast register.
pub(super) fn or_u16x16_all<const N: usize>(mut a: [u16x16; N], b: [u16x16; N]) -> [u16x16; N] {
    for i in 0..N {
        a[i] = _neon_or_u16(a[i], b[i]);
    }
    a
}

#[inline]
#[target_feature(enable = "neon")]
/// Perform a bitwise OR on all provided registers with another broadcast register.
pub(super) fn or_u8x32_all<const N: usize>(mut a: [u8x32; N], b: [u8x32; N]) -> [u8x32; N] {
    for i in 0..N {
        a[i] = _neon_or_u8(a[i], b[i]);
    }
    a
}

#[inline]
#[target_feature(enable = "neon")]
/// Shift all registers right by [IMM8] in 8-bit lanes.
pub(super) fn srli_u8x32<const IMM8: i32, const N: usize>(mut data: [u8x32; N]) -> [u8x32; N] {
    for i in 0..N {
        data[i] = _neon_srli_u8::<IMM8>(data[i]);
    }
    data
}

#[inline]
#[target_feature(enable = "neon")]
/// Shift all registers right by [IMM8] in 16-bit lanes.
pub(super) fn srli_u16x16<const IMM8: i32, const N: usize>(mut data: [u16x16; N]) -> [u16x16; N] {
    for i in 0..N {
        data[i] = _neon_srli_u16::<IMM8>(data[i]);
    }
    data
}

#[inline]
#[target_feature(enable = "neon")]
/// Shift all registers right by [IMM8] in 32-bit lanes.
pub(super) fn srli_u32x8<const IMM8: i32, const N: usize>(mut data: [u32x8; N]) -> [u32x8; N] {
    for i in 0..N {
        data[i] = _neon_srli_u32::<IMM8>(data[i]);
    }
    data
}

#[inline]
#[target_feature(enable = "neon")]
/// Shift all registers left by [IMM8] in 8-bit lanes.
pub(super) fn slli_u8x32<const IMM8: i32, const N: usize>(mut data: [u8x32; N]) -> [u8x32; N] {
    for i in 0..N {
        data[i] = _neon_slli_u8::<IMM8>(data[i]);
    }
    data
}

#[inline]
#[target_feature(enable = "neon")]
/// Shift all registers left by [IMM8] in 16-bit lanes.
pub(super) fn slli_u16x16<const IMM8: i32, const N: usize>(mut data: [u16x16; N]) -> [u16x16; N] {
    for i in 0..N {
        data[i] = _neon_slli_u16::<IMM8>(data[i]);
    }
    data
}

#[inline]
#[target_feature(enable = "neon")]
/// Shift all registers left by [IMM8] in 32-bit lanes.
pub(super) fn slli_u32x8<const IMM8: i32, const N: usize>(mut data: [u32x8; N]) -> [u32x8; N] {
    for i in 0..N {
        data[i] = _neon_slli_u32::<IMM8>(data[i]);
    }
    data
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::X64;
    use crate::uint32::neon::data::{load_u8x32x2, load_u32x64};
    use crate::uint32::test_util::{
        PACK_U16_TO_U8_EXPECTED_UNORDERED_LAYOUT,
        PACK_U32_TO_U8_EXPECTED_UNORDERED_LAYOUT,
        PACK_U32_TO_U16_EXPECTED_UNORDERED_LAYOUT,
    };

    #[test]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_pack_u32_to_u16_ordered() {
        let input = std::array::from_fn(|i| i as u32);

        let data = unsafe { load_u32x64(&input) };
        let packed = unsafe { pack_u32_to_u16_ordered(data) };

        let expected = std::array::from_fn(|i| i as u16);
        let view = unsafe { std::mem::transmute::<[u16x16; 4], [u16; X64]>(packed) };
        assert_eq!(view, expected);
    }

    #[test]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_pack_u32_to_u8_ordered() {
        let input = std::array::from_fn(|i| i as u32);

        let data = unsafe { load_u32x64(&input) };
        let packed = unsafe { pack_u32_to_u8_ordered(data) };

        let expected = std::array::from_fn(|i| i as u8);
        let view = unsafe { std::mem::transmute::<[u8x32; 2], [u8; X64]>(packed) };
        assert_eq!(view, expected);
    }

    #[test]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_pack_u16_to_u8_ordered() {
        let input = std::array::from_fn(|i| i as u16);

        let data = unsafe { std::mem::transmute::<[u16; X64], [u16x16; 4]>(input) };
        let packed = unsafe { pack_u16_to_u8_ordered(data) };

        let expected = std::array::from_fn(|i| i as u8);
        let view = unsafe { std::mem::transmute::<[u8x32; 2], [u8; X64]>(packed) };
        assert_eq!(view, expected);
    }

    #[test]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_unpack_u8_to_u16_ordered() {
        let input = std::array::from_fn(|i| i as u8);

        let data = unsafe { std::mem::transmute::<[u8; X64], [u8x32; 2]>(input) };
        let unpacked = unsafe { unpack_u8_to_u16_ordered(data) };

        let expected = std::array::from_fn(|i| i as u16);
        let view = unsafe { std::mem::transmute::<[u16x16; 4], [u16; X64]>(unpacked) };
        assert_eq!(view, expected);
    }

    #[test]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_unpack_u16_to_u32_ordered() {
        let input = std::array::from_fn(|i| i as u16);

        let data = unsafe { std::mem::transmute::<[u16; X64], [u16x16; 4]>(input) };
        let unpacked = unsafe { unpack_u16_to_u32_ordered(data) };

        let expected = std::array::from_fn(|i| i as u32);
        let view = unsafe { std::mem::transmute::<[u32x8; 8], [u32; X64]>(unpacked) };
        assert_eq!(view, expected);
    }

    #[test]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_unpack_u8_to_u32_ordered() {
        let input = std::array::from_fn(|i| i as u8);

        let data = unsafe { std::mem::transmute::<[u8; X64], [u8x32; 2]>(input) };
        let unpacked = unsafe { unpack_u8_to_u32_ordered(data) };

        let expected = std::array::from_fn(|i| i as u32);
        let view = unsafe { std::mem::transmute::<[u32x8; 8], [u32; X64]>(unpacked) };
        assert_eq!(view, expected);
    }

    #[test]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_pack_u32_to_u16_unordered() {
        let input = std::array::from_fn(|i| i as u32);

        let data = unsafe { load_u32x64(&input) };
        let packed = unsafe { pack_u32_to_u16_unordered(data) };

        let view = unsafe { std::mem::transmute::<[u16x16; 4], [u16; X64]>(packed) };
        assert_eq!(view[..8], [0, 16, 1, 17, 2, 18, 3, 19]);
        assert_eq!(view[8..][..8], [4, 20, 5, 21, 6, 22, 7, 23]);
        assert_eq!(view[16..][..8], [8, 24, 9, 25, 10, 26, 11, 27]);
    }

    #[test]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_pack_u32_to_u8_unordered() {
        let input = std::array::from_fn(|i| i as u32);

        let data = unsafe { load_u32x64(&input) };
        let packed = unsafe { pack_u32_to_u8_unordered(data) };

        let view = unsafe { std::mem::transmute::<[u8x32; 2], [u8; X64]>(packed) };
        assert_eq!(view[..8], [0, 32, 16, 48, 1, 33, 17, 49]);
        assert_eq!(view[8..][..8], [2, 34, 18, 50, 3, 35, 19, 51]);
        assert_eq!(view[16..][..8], [4, 36, 20, 52, 5, 37, 21, 53]);
        assert_eq!(view[24..][..8], [6, 38, 22, 54, 7, 39, 23, 55]);
    }

    #[test]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_unpack_u16_to_u32_unordered() {
        let input = std::array::from_fn(|i| i as u32);

        let data = unsafe { std::mem::transmute::<[u32; X64], [u32x8; 8]>(input) };
        let packed = unsafe { pack_u32_to_u16_unordered(data) };
        let unpacked = unsafe { unpack_u16_to_u32_unordered(packed) };

        let view = unsafe { std::mem::transmute::<[u32x8; 8], [u32; X64]>(unpacked) };
        assert_eq!(view, input);
    }

    #[test]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_pack_u32_to_u16_unordered_layout() {
        let input: [u32; X64] = std::array::from_fn(|i| i as u32);

        let data = unsafe { load_u32x64(&input) };
        let packed = unsafe { pack_u32_to_u16_unordered(data) };

        let view = unsafe { std::mem::transmute::<[u16x16; 4], [u16; X64]>(packed) };
        assert_eq!(view, PACK_U32_TO_U16_EXPECTED_UNORDERED_LAYOUT);
    }

    #[test]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_pack_u32_to_u8_unordered_layout() {
        let input: [u32; X64] = std::array::from_fn(|i| i as u32);

        let data = unsafe { load_u32x64(&input) };
        let packed = unsafe { pack_u32_to_u8_unordered(data) };

        let view = unsafe { std::mem::transmute::<[u8x32; 2], [u8; X64]>(packed) };
        assert_eq!(view, PACK_U32_TO_U8_EXPECTED_UNORDERED_LAYOUT);
    }

    #[test]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_unpack_u8_to_u16_unordered_layout() {
        let expected: [u16; X64] = std::array::from_fn(|i| i as u16);

        let data = unsafe { load_u8x32x2(PACK_U16_TO_U8_EXPECTED_UNORDERED_LAYOUT.as_ptr()) };
        let unpacked = unsafe { unpack_u8_to_u16_unordered(data) };

        let view = unsafe { std::mem::transmute::<[u16x16; 4], [u16; X64]>(unpacked) };
        assert_eq!(view, expected);
    }

    #[test]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_unpack_u4_to_u8_unordered() {
        let expected: [u8; X64] = std::array::from_fn(|i| i as u8 % 16);

        let data = unsafe { load_u8x32x2(expected.as_ptr()) };
        let packed = unsafe { pack_u8_to_u4_unordered(data) };
        let unpacked = unsafe { unpack_u4_to_u8_unordered(packed) };

        let view = unsafe { std::mem::transmute::<[u8x32; 2], [u8; X64]>(unpacked) };
        assert_eq!(view, expected);
    }

    #[test]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_unpack_u2_to_u8_unordered() {
        let expected: [u8; X64] = std::array::from_fn(|i| i as u8 % 4);

        let data = unsafe { load_u8x32x2(expected.as_ptr()) };
        let packed = unsafe { pack_u8_to_u2_unordered(data) };
        let unpacked = unsafe { unpack_u2_to_u8_unordered(packed) };

        let view = unsafe { std::mem::transmute::<[u8x32; 2], [u8; X64]>(unpacked) };
        assert_eq!(view, expected);
    }
}
