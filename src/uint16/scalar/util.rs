#![allow(clippy::needless_range_loop)]

use super::polyfill::*;

/// Pack 4 sets of registers containing 16-bit elements and produce 2 registers holding
/// 8-bit elements.
///
/// The order of elements is not maintained.
pub(super) fn pack_u16_to_u8_unordered(data: [u16x16; 4]) -> [u8x32; 2] {
    let lo = [data[0].into(), data[1].into()];
    let hi = [
        _scalar_slli_u16x16::<8>(data[2]).into(),
        _scalar_slli_u16x16::<8>(data[3]).into(),
    ];
    or_u8x32_all(hi, lo)
}

/// Split the 16-bit values in the provided registers producing two 8-bit
/// halves, packing the results while maintaining the order.
pub(super) fn split_u16_ordered(data: [u16x16; 4]) -> ([u8x32; 2], [u8x32; 2]) {
    let mask = _scalar_set1_u16(0x00FF);

    let lo_bits = and_u16x16(data, mask);
    let hi_bits = srli_u16x16::<8, 4>(data);

    let lo_packed = pack_u16_to_u8_ordered(lo_bits);
    let hi_packed = pack_u16_to_u8_ordered(hi_bits);

    (hi_packed, lo_packed)
}

/// Split the 16-bit values in the provided registers producing two 8-bit
/// halves, packing the results but does not maintain the order.
pub(super) fn split_u16_unordered(data: [u16x16; 4]) -> ([u8x32; 2], [u8x32; 2]) {
    let mask = _scalar_set1_u16(0x00FF);

    let lo_bits = and_u16x16(data, mask);
    let hi_bits = srli_u16x16::<8, 4>(data);

    let lo_packed = pack_u16_to_u8_unordered(lo_bits);
    let hi_packed = pack_u16_to_u8_unordered(hi_bits);

    (hi_packed, lo_packed)
}

/// Pack 4 sets of registers containing 16-bit elements and produce 2 registers holding
/// 8-bit elements.
///
/// The order of elements is maintained.
pub(super) fn pack_u16_to_u8_ordered(data: [u16x16; 4]) -> [u8x32; 2] {
    [
        _scalar_combine_u8x16(
            _scalar_cvteu8_u16x16(data[0]),
            _scalar_cvteu8_u16x16(data[1]),
        ),
        _scalar_combine_u8x16(
            _scalar_cvteu8_u16x16(data[2]),
            _scalar_cvteu8_u16x16(data[3]),
        ),
    ]
}

/// Unpack 2 sets of registers containing 8-bit elements and produce 4 registers holding
/// 16-bit elements.
pub(super) fn unpack_u8_to_u16_ordered(data: [u8x32; 2]) -> [u16x16; 4] {
    let parts = [
        _scalar_extract_u8x32::<0>(data[0]),
        _scalar_extract_u8x32::<1>(data[0]),
        _scalar_extract_u8x32::<0>(data[1]),
        _scalar_extract_u8x32::<1>(data[1]),
    ];

    [
        _scalar_cvteu16_u8x16(parts[0]),
        _scalar_cvteu16_u8x16(parts[1]),
        _scalar_cvteu16_u8x16(parts[2]),
        _scalar_cvteu16_u8x16(parts[3]),
    ]
}

#[inline]
/// Unpack 2 sets of registers containing 8-bit elements and produce 4 registers holding
/// 16-bit elements.
pub(super) fn unpack_u8_to_u16_unordered(data: [u8x32; 2]) -> [u16x16; 4] {
    [
        _scalar_blend_every_other_u8(data[0], u8x32::ZERO).into(),
        _scalar_blend_every_other_u8(data[1], u8x32::ZERO).into(),
        _scalar_srli_u16x16::<8>(data[0].into()),
        _scalar_srli_u16x16::<8>(data[1].into()),
    ]
}

pub(super) fn pack_u8_to_u4_unordered(data: [u8x32; 2]) -> u8x32 {
    let shifted = _scalar_slli_u8x32::<4>(data[1]);
    _scalar_or_u8x32(data[0], shifted)
}

pub(super) fn unpack_u4_to_u8_unordered(data: u8x32) -> [u8x32; 2] {
    let mask = _scalar_set1_u8(0x0F);
    let lo_nibbles = _scalar_and_u8x32(data, mask);
    let hi_nibbles = _scalar_and_u8x32(_scalar_srli_u8x32::<4>(data), mask);
    [lo_nibbles, hi_nibbles]
}

pub(super) fn pack_u8_to_u2_unordered(data: [u8x32; 2]) -> u8x16 {
    let nibbles = pack_u8_to_u4_unordered(data);
    let nibbles: u16x16 = nibbles.into();

    let lo = _scalar_extract_u16x16::<0>(nibbles);
    let hi = _scalar_extract_u16x16::<1>(nibbles);

    let mut packed = u16x8::ZERO;
    for i in 0..8 {
        packed[i] = lo[i] | (hi[i] << 2);
    }

    packed.into()
}

pub(super) fn unpack_u2_to_u8_unordered(data: u8x16) -> [u8x32; 2] {
    const MASK: u16 = 0b0011_0011_0011_0011;

    let data: u16x8 = data.into();

    let mut lo = u16x8::ZERO;
    for i in 0..8 {
        lo[i] = data[i] & MASK;
    }

    let mut hi = u16x8::ZERO;
    for i in 0..8 {
        hi[i] = (data[i] >> 2) & MASK;
    }

    unpack_u4_to_u8_unordered(_scalar_combine_u16x8(lo, hi).into())
}

#[inline]
/// Perform a bitwise AND on all provided registers with another broadcast register.
pub(super) fn and_u16x16<const N: usize>(mut data: [u16x16; N], mask: u16x16) -> [u16x16; N] {
    for i in 0..N {
        data[i] = _scalar_and_u16x16(data[i], mask);
    }
    data
}

#[inline]
/// Perform a bitwise AND on all provided registers with another broadcast register.
pub(super) fn and_u8x32<const N: usize>(mut data: [u8x32; N], mask: u8x32) -> [u8x32; N] {
    for i in 0..N {
        data[i] = _scalar_and_u8x32(data[i], mask);
    }
    data
}

#[inline]
/// Perform a bitwise OR on all provided registers with another broadcast register.
pub(super) fn or_u16x16_all<const N: usize>(mut a: [u16x16; N], b: [u16x16; N]) -> [u16x16; N] {
    for i in 0..N {
        a[i] = _scalar_or_u16x16(a[i], b[i]);
    }
    a
}

#[inline]
/// Perform a bitwise OR on all provided registers with another broadcast register.
pub(super) fn or_u8x32_all<const N: usize>(mut a: [u8x32; N], b: [u8x32; N]) -> [u8x32; N] {
    for i in 0..N {
        a[i] = _scalar_or_u8x32(a[i], b[i]);
    }
    a
}

#[inline]
/// Shift all registers right by [IMM8] in 8-bit lanes.
pub(super) fn srli_u8x32<const IMM8: u32, const N: usize>(mut data: [u8x32; N]) -> [u8x32; N] {
    for i in 0..N {
        data[i] = _scalar_srli_u8x32::<IMM8>(data[i]);
    }
    data
}

#[inline]
/// Shift all registers right by [IMM8] in 16-bit lanes.
pub(super) fn srli_u16x16<const IMM8: u32, const N: usize>(mut data: [u16x16; N]) -> [u16x16; N] {
    for i in 0..N {
        data[i] = _scalar_srli_u16x16::<IMM8>(data[i]);
    }
    data
}

#[inline]
/// Shift all registers left by [IMM8] in 8-bit lanes.
pub(super) fn slli_u8x32<const IMM8: u32, const N: usize>(mut data: [u8x32; N]) -> [u8x32; N] {
    for i in 0..N {
        data[i] = _scalar_slli_u8x32::<IMM8>(data[i]);
    }
    data
}

#[inline]
/// Shift all registers left by [IMM8] in 16-bit lanes.
pub(super) fn slli_u16x16<const IMM8: u32, const N: usize>(mut data: [u16x16; N]) -> [u16x16; N] {
    for i in 0..N {
        data[i] = _scalar_slli_u16x16::<IMM8>(data[i]);
    }
    data
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::X64;
    use crate::uint16::scalar::data::{load_u8x32x2, load_u16x64};
    use crate::uint16::test_util::{
        PACK_U16_TO_U8_EXPECTED_UNORDERED_LAYOUT,
        PACK_U32_TO_U8_EXPECTED_UNORDERED_LAYOUT,
        PACK_U32_TO_U16_EXPECTED_UNORDERED_LAYOUT,
    };

    #[test]
    fn test_pack_u16_to_u8_ordered() {
        let input = std::array::from_fn(|i| i as u16);

        let data = unsafe { std::mem::transmute::<[u16; X64], [u16x16; 4]>(input) };
        let packed = pack_u16_to_u8_ordered(data);

        let expected = std::array::from_fn(|i| i as u8);
        let view = unsafe { std::mem::transmute::<[u8x32; 2], [u8; X64]>(packed) };
        assert_eq!(view, expected);
    }

    #[test]
    fn test_unpack_u8_to_u16_ordered() {
        let input = std::array::from_fn(|i| i as u8);

        let data = unsafe { std::mem::transmute::<[u8; X64], [u8x32; 2]>(input) };
        let unpacked = unpack_u8_to_u16_ordered(data);

        let expected = std::array::from_fn(|i| i as u16);
        let view = unsafe { std::mem::transmute::<[u16x16; 4], [u16; X64]>(unpacked) };
        assert_eq!(view, expected);
    }

    #[test]
    fn test_unpack_u8_to_u16_unordered_layout() {
        let expected: [u16; X64] = std::array::from_fn(|i| i as u16);

        let data = unsafe { load_u8x32x2(PACK_U16_TO_U8_EXPECTED_UNORDERED_LAYOUT.as_ptr()) };
        let unpacked = unpack_u8_to_u16_unordered(data);

        let view = unsafe { std::mem::transmute::<[u16x16; 4], [u16; X64]>(unpacked) };
        assert_eq!(view, expected);
    }

    #[test]
    fn test_unpack_u4_to_u8_unordered() {
        let expected: [u8; X64] = std::array::from_fn(|i| i as u8 % 16);

        let data = unsafe { load_u8x32x2(expected.as_ptr()) };
        let packed = pack_u8_to_u4_unordered(data);
        let unpacked = unpack_u4_to_u8_unordered(packed);

        let view = unsafe { std::mem::transmute::<[u8x32; 2], [u8; X64]>(unpacked) };
        assert_eq!(view, expected);
    }

    #[test]
    fn test_unpack_u2_to_u8_unordered() {
        let expected: [u8; X64] = std::array::from_fn(|i| i as u8 % 4);

        let data = unsafe { load_u8x32x2(expected.as_ptr()) };
        let packed = pack_u8_to_u2_unordered(data);
        let unpacked = unpack_u2_to_u8_unordered(packed);

        let view = unsafe { std::mem::transmute::<[u8x32; 2], [u8; X64]>(unpacked) };
        assert_eq!(view, expected);
    }
}
