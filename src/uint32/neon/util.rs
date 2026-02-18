#![allow(clippy::needless_range_loop)]

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
        _neon_combine_u16x8(_neon_cvteu16_u32x8(data[0]), _neon_cvteu16_u32x8(data[1])),
        _neon_combine_u16x8(_neon_cvteu16_u32x8(data[2]), _neon_cvteu16_u32x8(data[3])),
        _neon_combine_u16x8(_neon_cvteu16_u32x8(data[4]), _neon_cvteu16_u32x8(data[5])),
        _neon_combine_u16x8(_neon_cvteu16_u32x8(data[6]), _neon_cvteu16_u32x8(data[7])),
    ]
}

#[target_feature(enable = "neon")]
/// Pack 4 sets of registers containing 16-bit elements and produce 2 registers holding
/// 9-bit elements.
///
/// The order of elements is maintained.
pub(super) fn pack_u16_to_u8_ordered(data: [u16x16; 4]) -> [u8x32; 2] {
    [
        _neon_combine_u8x16(_neon_cvteu8_u16x16(data[0]), _neon_cvteu8_u16x16(data[1])),
        _neon_combine_u8x16(_neon_cvteu8_u16x16(data[2]), _neon_cvteu8_u16x16(data[3])),
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
        _neon_cvteu16_u8x16(parts[0]),
        _neon_cvteu16_u8x16(parts[1]),
        _neon_cvteu16_u8x16(parts[2]),
        _neon_cvteu16_u8x16(parts[3]),
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
        _neon_cvteu32_u16x8(split_u16s[0]),
        _neon_cvteu32_u16x8(split_u16s[1]),
        _neon_cvteu32_u16x8(split_u16s[2]),
        _neon_cvteu32_u16x8(split_u16s[3]),
        _neon_cvteu32_u16x8(split_u16s[4]),
        _neon_cvteu32_u16x8(split_u16s[5]),
        _neon_cvteu32_u16x8(split_u16s[6]),
        _neon_cvteu32_u16x8(split_u16s[7]),
    ]
}

#[target_feature(enable = "neon")]
/// Pack 8 sets of registers containing 32-bit elements and produce 2 registers holding
/// 8-bit elements.
///
/// The order of elements are _not_ maintained.
pub(super) fn pack_u32_to_u8_unordered(data: [u32x8; 8]) -> [u8x32; 2] {
    let partially_packed = pack_u32_to_u16_unordered(data);
    pack_u16_to_u8_unordered(partially_packed)
}

#[target_feature(enable = "neon")]
/// Pack 8 sets of registers containing 32-bit elements and produce 4 registers holding
/// 16-bit elements.
///
/// The order of elements are _not_ maintained.
pub(super) fn pack_u32_to_u16_unordered(data: [u32x8; 8]) -> [u16x16; 4] {
    [
        _neon_pack_u32x8(data[0], data[2]),
        _neon_pack_u32x8(data[1], data[3]),
        _neon_pack_u32x8(data[4], data[6]),
        _neon_pack_u32x8(data[5], data[7]),
    ]
}

#[target_feature(enable = "neon")]
/// Pack 4 sets of registers containing 16-bit elements and produce 2 registers holding
/// 8-bit elements.
///
/// The order of elements are _not_ maintained.
pub(super) fn pack_u16_to_u8_unordered(data: [u16x16; 4]) -> [u8x32; 2] {
    [
        _neon_pack_u16x16(data[0], data[2]),
        _neon_pack_u16x16(data[1], data[3]),
    ]
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
    let splits = [
        _neon_extract_u8x32::<0>(data[0]),
        _neon_extract_u8x32::<0>(data[1]),
        _neon_extract_u8x32::<1>(data[0]),
        _neon_extract_u8x32::<1>(data[1]),
    ];

    let interleaved = [
        _neon_cvteu16_u8x16(splits[0]),
        _neon_cvteu16_u8x16(splits[2]),
        _neon_cvteu16_u8x16(splits[1]),
        _neon_cvteu16_u8x16(splits[3]),
    ];

    let split_halves = [
        _neon_extract_u16x16::<0>(interleaved[0]),
        _neon_extract_u16x16::<1>(interleaved[0]),
        _neon_extract_u16x16::<0>(interleaved[1]),
        _neon_extract_u16x16::<1>(interleaved[1]),
        _neon_extract_u16x16::<0>(interleaved[2]),
        _neon_extract_u16x16::<1>(interleaved[2]),
        _neon_extract_u16x16::<0>(interleaved[3]),
        _neon_extract_u16x16::<1>(interleaved[3]),
    ];

    [
        _neon_combine_u16x8(split_halves[0], split_halves[2]),
        _neon_combine_u16x8(split_halves[4], split_halves[6]),
        _neon_combine_u16x8(split_halves[1], split_halves[3]),
        _neon_combine_u16x8(split_halves[5], split_halves[7]),
    ]
}

#[inline]
#[target_feature(enable = "neon")]
/// Unpack 4 sets of registers containing 16-bit elements and produce 8 registers holding
/// 32-bit elements.
pub(super) fn unpack_u16_to_u32_unordered(data: [u16x16; 4]) -> [u32x8; 8] {
    let splits = [
        _neon_extract_u16x16::<0>(data[0]),
        _neon_extract_u16x16::<1>(data[0]),
        _neon_extract_u16x16::<0>(data[1]),
        _neon_extract_u16x16::<1>(data[1]),
        _neon_extract_u16x16::<0>(data[2]),
        _neon_extract_u16x16::<1>(data[2]),
        _neon_extract_u16x16::<0>(data[3]),
        _neon_extract_u16x16::<1>(data[3]),
    ];

    let interleaved = [
        _neon_cvteu32_u16x8(splits[0]),
        _neon_cvteu32_u16x8(splits[2]),
        _neon_cvteu32_u16x8(splits[1]),
        _neon_cvteu32_u16x8(splits[3]),
        _neon_cvteu32_u16x8(splits[4]),
        _neon_cvteu32_u16x8(splits[6]),
        _neon_cvteu32_u16x8(splits[5]),
        _neon_cvteu32_u16x8(splits[7]),
    ];

    let split_halves = [
        _neon_extract_u32x8::<0>(interleaved[0]),
        _neon_extract_u32x8::<1>(interleaved[0]),
        _neon_extract_u32x8::<0>(interleaved[1]),
        _neon_extract_u32x8::<1>(interleaved[1]),
        _neon_extract_u32x8::<0>(interleaved[2]),
        _neon_extract_u32x8::<1>(interleaved[2]),
        _neon_extract_u32x8::<0>(interleaved[3]),
        _neon_extract_u32x8::<1>(interleaved[3]),
        _neon_extract_u32x8::<0>(interleaved[4]),
        _neon_extract_u32x8::<1>(interleaved[4]),
        _neon_extract_u32x8::<0>(interleaved[5]),
        _neon_extract_u32x8::<1>(interleaved[5]),
        _neon_extract_u32x8::<0>(interleaved[6]),
        _neon_extract_u32x8::<1>(interleaved[6]),
        _neon_extract_u32x8::<0>(interleaved[7]),
        _neon_extract_u32x8::<1>(interleaved[7]),
    ];

    [
        _neon_combine_u32x4(split_halves[0], split_halves[4]),
        _neon_combine_u32x4(split_halves[2], split_halves[6]),
        _neon_combine_u32x4(split_halves[1], split_halves[5]),
        _neon_combine_u32x4(split_halves[3], split_halves[7]),
        _neon_combine_u32x4(split_halves[8], split_halves[12]),
        _neon_combine_u32x4(split_halves[10], split_halves[14]),
        _neon_combine_u32x4(split_halves[9], split_halves[13]),
        _neon_combine_u32x4(split_halves[11], split_halves[15]),
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
    let packed_u16 = pack_u32_to_u16_unordered(data);

    let mask = _neon_set1_u16(0x00FF);
    let lo_8bits = and_u16x16(packed_u16, mask);
    let hi_8bits = srli_u16x16::<8, 4>(packed_u16);

    let packed_lo_8bits = pack_u16_to_u8_unordered(lo_8bits);
    let packed_hi_8bits = pack_u16_to_u8_unordered(hi_8bits);

    (packed_hi_8bits, packed_lo_8bits)
}

#[inline]
#[target_feature(enable = "neon")]
/// Perform a bitwise AND on all provided registers with another broadcast register.
pub(super) fn and_u32x8<const N: usize>(mut data: [u32x8; N], mask: u32x8) -> [u32x8; N] {
    for i in 0..N {
        data[i] = _neon_and_u32x8(data[i], mask);
    }
    data
}

#[inline]
#[target_feature(enable = "neon")]
/// Perform a bitwise AND on all provided registers with another broadcast register.
pub(super) fn and_u16x16<const N: usize>(mut data: [u16x16; N], mask: u16x16) -> [u16x16; N] {
    for i in 0..N {
        data[i] = _neon_and_u16x16(data[i], mask);
    }
    data
}

#[inline]
#[target_feature(enable = "neon")]
/// Perform a bitwise AND on all provided registers with another broadcast register.
pub(super) fn and_u8x32<const N: usize>(mut data: [u8x32; N], mask: u8x32) -> [u8x32; N] {
    for i in 0..N {
        data[i] = _neon_and_u8x32(data[i], mask);
    }
    data
}

#[inline]
#[target_feature(enable = "neon")]
/// Perform a bitwise OR on all provided registers with another broadcast register.
pub(super) fn or_u32x8_all<const N: usize>(mut a: [u32x8; N], b: [u32x8; N]) -> [u32x8; N] {
    for i in 0..N {
        a[i] = _neon_or_u32x8(a[i], b[i]);
    }
    a
}

#[inline]
#[target_feature(enable = "neon")]
/// Perform a bitwise OR on all provided registers with another broadcast register.
pub(super) fn or_u16x16_all<const N: usize>(mut a: [u16x16; N], b: [u16x16; N]) -> [u16x16; N] {
    for i in 0..N {
        a[i] = _neon_or_u16x16(a[i], b[i]);
    }
    a
}

#[inline]
#[target_feature(enable = "neon")]
/// Perform a bitwise OR on all provided registers with another broadcast register.
pub(super) fn or_u8x32_all<const N: usize>(mut a: [u8x32; N], b: [u8x32; N]) -> [u8x32; N] {
    for i in 0..N {
        a[i] = _neon_or_u8x32(a[i], b[i]);
    }
    a
}

#[inline]
#[target_feature(enable = "neon")]
/// Shift all registers right by [IMM8] in 8-bit lanes.
pub(super) fn srli_u8x32<const IMM8: i32, const N: usize>(mut data: [u8x32; N]) -> [u8x32; N] {
    for i in 0..N {
        data[i] = _neon_srli_u8x32::<IMM8>(data[i]);
    }
    data
}

#[inline]
#[target_feature(enable = "neon")]
/// Shift all registers right by [IMM8] in 16-bit lanes.
pub(super) fn srli_u16x16<const IMM8: i32, const N: usize>(mut data: [u16x16; N]) -> [u16x16; N] {
    for i in 0..N {
        data[i] = _neon_srli_u16x16::<IMM8>(data[i]);
    }
    data
}

#[inline]
#[target_feature(enable = "neon")]
/// Shift all registers right by [IMM8] in 32-bit lanes.
pub(super) fn srli_u32x8<const IMM8: i32, const N: usize>(mut data: [u32x8; N]) -> [u32x8; N] {
    for i in 0..N {
        data[i] = _neon_srli_u32x8::<IMM8>(data[i]);
    }
    data
}

#[inline]
#[target_feature(enable = "neon")]
/// Shift all registers left by [IMM8] in 8-bit lanes.
pub(super) fn slli_u8x32<const IMM8: i32, const N: usize>(mut data: [u8x32; N]) -> [u8x32; N] {
    for i in 0..N {
        data[i] = _neon_slli_u8x32::<IMM8>(data[i]);
    }
    data
}

#[inline]
#[target_feature(enable = "neon")]
/// Shift all registers left by [IMM8] in 16-bit lanes.
pub(super) fn slli_u16x16<const IMM8: i32, const N: usize>(mut data: [u16x16; N]) -> [u16x16; N] {
    for i in 0..N {
        data[i] = _neon_slli_u16x16::<IMM8>(data[i]);
    }
    data
}

#[inline]
#[target_feature(enable = "neon")]
/// Shift all registers left by [IMM8] in 32-bit lanes.
pub(super) fn slli_u32x8<const IMM8: i32, const N: usize>(mut data: [u32x8; N]) -> [u32x8; N] {
    for i in 0..N {
        data[i] = _neon_slli_u32x8::<IMM8>(data[i]);
    }
    data
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::X64;
    use crate::uint32::neon::data::load_u32x64;
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
        assert_eq!(view[..8], [0, 1, 2, 3, 16, 17, 18, 19]);
        assert_eq!(view[8..][..8], [4, 5, 6, 7, 20, 21, 22, 23]);
        assert_eq!(view[16..][..8], [8, 9, 10, 11, 24, 25, 26, 27]);
    }

    #[test]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_pack_u32_to_u8_unordered() {
        let input = std::array::from_fn(|i| i as u32);

        let data = unsafe { load_u32x64(&input) };
        let packed = unsafe { pack_u32_to_u8_unordered(data) };

        let view = unsafe { std::mem::transmute::<[u8x32; 2], [u8; X64]>(packed) };
        assert_eq!(view[..8], [0, 1, 2, 3, 16, 17, 18, 19]);
        assert_eq!(view[8..][..8], [32, 33, 34, 35, 48, 49, 50, 51]);
        assert_eq!(view[16..][..8], [4, 5, 6, 7, 20, 21, 22, 23]);
        assert_eq!(view[24..][..8], [36, 37, 38, 39, 52, 53, 54, 55]);
    }

    #[test]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_unpack_u8_to_u16_unordered() {
        let input = std::array::from_fn(|i| i as u16);

        let data = unsafe { std::mem::transmute::<[u16; X64], [u16x16; 4]>(input) };
        let packed = unsafe { pack_u16_to_u8_unordered(data) };
        let unpacked = unsafe { unpack_u8_to_u16_unordered(packed) };

        let view = unsafe { std::mem::transmute::<[u16x16; 4], [u16; X64]>(unpacked) };
        assert_eq!(view, input);
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
    fn test_pack_u16_to_u8_unordered_layout() {
        let input: [u16; X64] = std::array::from_fn(|i| i as u16);

        let data = unsafe { std::mem::transmute::<[u16; X64], [u16x16; 4]>(input) };
        let packed = unsafe { pack_u16_to_u8_unordered(data) };

        let view = unsafe { std::mem::transmute::<[u8x32; 2], [u8; X64]>(packed) };
        assert_eq!(view, PACK_U16_TO_U8_EXPECTED_UNORDERED_LAYOUT);
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
}
