#![allow(clippy::needless_range_loop)]

use std::arch::aarch64::*;

use super::polyfill::*;

#[target_feature(enable = "neon")]
/// Pack 4 sets of registers containing 16-bit elements and produce 4 registers holding
/// 8-bit elements.
///
/// The order of elements is **not** maintained.
pub(super) fn pack_u16_to_u8_unordered(data: [uint16x8_t; 8]) -> [uint8x16_t; 4] {
    let mask = _neon_set1_u16(0x00FF);
    let lo = [
        _neon_and_u16(data[0], mask),
        _neon_and_u16(data[1], mask),
        _neon_and_u16(data[2], mask),
        _neon_and_u16(data[3], mask),
    ];
    let hi = [
        _neon_slli_u16::<8>(data[4]),
        _neon_slli_u16::<8>(data[5]),
        _neon_slli_u16::<8>(data[6]),
        _neon_slli_u16::<8>(data[7]),
    ];

    [
        vreinterpretq_u8_u16(_neon_or_u16(hi[0], lo[0])),
        vreinterpretq_u8_u16(_neon_or_u16(hi[1], lo[1])),
        vreinterpretq_u8_u16(_neon_or_u16(hi[2], lo[2])),
        vreinterpretq_u8_u16(_neon_or_u16(hi[3], lo[3])),
    ]
}

#[target_feature(enable = "neon")]
/// Split the 16-bit values in the provided registers producing two 8-bit
/// halves, packing the results while **not** maintaining the order.
pub(super) fn split_u16_unordered(data: [uint16x8_t; 8]) -> ([uint8x16_t; 4], [uint8x16_t; 4]) {
    let mask = _neon_set1_u16(0x00FF);

    let lo_bits = and_u16(data, mask);
    let hi_bits = srli_u16::<8, 8>(data);

    let lo_packed = pack_u16_to_u8_unordered(lo_bits);
    let hi_packed = pack_u16_to_u8_unordered(hi_bits);

    (hi_packed, lo_packed)
}

#[target_feature(enable = "neon")]
/// Split the 16-bit values in the provided registers producing two 8-bit
/// halves, packing the results while maintaining the order.
pub(super) fn split_u16_ordered(data: [uint16x8_t; 8]) -> ([uint8x16_t; 4], [uint8x16_t; 4]) {
    let mask = _neon_set1_u16(0x00FF);

    let lo_bits = and_u16(data, mask);
    let hi_bits = srli_u16::<8, 8>(data);

    let lo_packed = pack_u16_to_u8_ordered(lo_bits);
    let hi_packed = pack_u16_to_u8_ordered(hi_bits);

    (hi_packed, lo_packed)
}

#[target_feature(enable = "neon")]
/// Pack 4 sets of registers containing 16-bit elements and produce 4 registers holding
/// 8-bit elements.
///
/// The order of elements is maintained.
pub(super) fn pack_u16_to_u8_ordered(data: [uint16x8_t; 8]) -> [uint8x16_t; 4] {
    [
        _neon_cvteu16_u8(data[0], data[1]),
        _neon_cvteu16_u8(data[2], data[3]),
        _neon_cvteu16_u8(data[4], data[5]),
        _neon_cvteu16_u8(data[6], data[7]),
    ]
}

#[target_feature(enable = "neon")]
/// Unpack 2 sets of registers containing 8-bit elements and produce 4 registers holding
/// 16-bit elements.
pub(super) fn unpack_u8_to_u16_ordered(data: [uint8x16_t; 4]) -> [uint16x8_t; 8] {
    let [d1, d2] = _neon_cvteu8_u16(data[0]);
    let [d3, d4] = _neon_cvteu8_u16(data[1]);
    let [d5, d6] = _neon_cvteu8_u16(data[2]);
    let [d7, d8] = _neon_cvteu8_u16(data[3]);

    [d1, d2, d3, d4, d5, d6, d7, d8]
}

#[inline]
#[target_feature(enable = "neon")]
/// Unpack 2 sets of registers containing 8-bit elements and produce 4 registers holding
/// 16-bit elements.
pub(super) fn unpack_u8_to_u16_unordered(data: [uint8x16_t; 4]) -> [uint16x8_t; 8] {
    let zeroes = _neon_set1_u8(0);
    let d1 = _neon_blend_every_other_u8(data[0], zeroes);
    let d2 = _neon_blend_every_other_u8(data[1], zeroes);
    let d3 = _neon_blend_every_other_u8(data[2], zeroes);
    let d4 = _neon_blend_every_other_u8(data[3], zeroes);

    let d5 = _neon_srli_u16::<8>(vreinterpretq_u16_u8(data[0]));
    let d6 = _neon_srli_u16::<8>(vreinterpretq_u16_u8(data[1]));
    let d7 = _neon_srli_u16::<8>(vreinterpretq_u16_u8(data[2]));
    let d8 = _neon_srli_u16::<8>(vreinterpretq_u16_u8(data[3]));

    [
        vreinterpretq_u16_u8(d1),
        vreinterpretq_u16_u8(d2),
        vreinterpretq_u16_u8(d3),
        vreinterpretq_u16_u8(d4),
        d5,
        d6,
        d7,
        d8,
    ]
}

#[inline]
#[target_feature(enable = "neon")]
pub(super) fn pack_u8_to_u4_unordered(data: [uint8x16_t; 4]) -> [uint8x16_t; 2] {
    let shifted_1 = _neon_slli_u8::<4>(data[2]);
    let shifted_2 = _neon_slli_u8::<4>(data[3]);
    let d1 = _neon_or_u8(data[0], shifted_1);
    let d2 = _neon_or_u8(data[1], shifted_2);
    [d1, d2]
}

#[inline]
#[target_feature(enable = "neon")]
pub(super) fn unpack_u4_to_u8_unordered(data: [uint8x16_t; 2]) -> [uint8x16_t; 4] {
    let mask = _neon_set1_u8(0x0F);
    let lo_4bits_1 = _neon_and_u8(data[0], mask);
    let lo_4bits_2 = _neon_and_u8(data[1], mask);
    let hi_4bits_1 = _neon_srli_u8::<4>(data[0]);
    let hi_4bits_2 = _neon_srli_u8::<4>(data[1]);
    [lo_4bits_1, lo_4bits_2, hi_4bits_1, hi_4bits_2]
}

#[inline]
#[target_feature(enable = "neon")]
pub(super) fn pack_u8_to_u2_unordered(data: [uint8x16_t; 4]) -> uint8x16_t {
    let [lo, hi] = pack_u8_to_u4_unordered(data);
    vorrq_u8(lo, vshlq_n_u8::<2>(hi))
}

#[inline]
#[target_feature(enable = "neon")]
pub(super) fn unpack_u2_to_u8_unordered(data: uint8x16_t) -> [uint8x16_t; 4] {
    let mask = vdupq_n_u8(0b0011_0011);
    let lo = vandq_u8(data, mask);
    let hi = vandq_u8(vshrq_n_u8::<2>(data), mask);
    unpack_u4_to_u8_unordered([lo, hi])
}

#[inline]
#[target_feature(enable = "neon")]
/// Perform a bitwise AND on all provided registers with another broadcast register.
pub(super) fn and_u16<const N: usize>(
    mut data: [uint16x8_t; N],
    mask: uint16x8_t,
) -> [uint16x8_t; N] {
    for i in 0..N {
        data[i] = _neon_and_u16(data[i], mask);
    }
    data
}

#[inline]
#[target_feature(enable = "neon")]
/// Perform a bitwise AND on all provided registers with another broadcast register.
pub(super) fn and_u8<const N: usize>(
    mut data: [uint8x16_t; N],
    mask: uint8x16_t,
) -> [uint8x16_t; N] {
    for i in 0..N {
        data[i] = _neon_and_u8(data[i], mask);
    }
    data
}

#[inline]
#[target_feature(enable = "neon")]
/// Perform a bitwise OR on all provided registers with another broadcast register.
pub(super) fn or_u16_all<const N: usize>(
    mut a: [uint16x8_t; N],
    b: [uint16x8_t; N],
) -> [uint16x8_t; N] {
    for i in 0..N {
        a[i] = _neon_or_u16(a[i], b[i]);
    }
    a
}

#[inline]
#[target_feature(enable = "neon")]
/// Perform a bitwise OR on all provided registers with another broadcast register.
pub(super) fn or_u8_all<const N: usize>(
    mut a: [uint8x16_t; N],
    b: [uint8x16_t; N],
) -> [uint8x16_t; N] {
    for i in 0..N {
        a[i] = _neon_or_u8(a[i], b[i]);
    }
    a
}

#[inline]
#[target_feature(enable = "neon")]
/// Shift all registers right by [IMM8] in 8-bit lanes.
pub(super) fn srli_u8<const IMM8: i32, const N: usize>(
    mut data: [uint8x16_t; N],
) -> [uint8x16_t; N] {
    for i in 0..N {
        data[i] = _neon_srli_u8::<IMM8>(data[i]);
    }
    data
}

#[inline]
#[target_feature(enable = "neon")]
/// Shift all registers right by [IMM8] in 16-bit lanes.
pub(super) fn srli_u16<const IMM8: i32, const N: usize>(
    mut data: [uint16x8_t; N],
) -> [uint16x8_t; N] {
    for i in 0..N {
        data[i] = _neon_srli_u16::<IMM8>(data[i]);
    }
    data
}

#[inline]
#[target_feature(enable = "neon")]
/// Shift all registers left by [IMM8] in 8-bit lanes.
pub(super) fn slli_u8<const IMM8: i32, const N: usize>(
    mut data: [uint8x16_t; N],
) -> [uint8x16_t; N] {
    for i in 0..N {
        data[i] = _neon_slli_u8::<IMM8>(data[i]);
    }
    data
}

#[inline]
#[target_feature(enable = "neon")]
/// Shift all registers left by [IMM8] in 16-bit lanes.
pub(super) fn slli_u16<const IMM8: i32, const N: usize>(
    mut data: [uint16x8_t; N],
) -> [uint16x8_t; N] {
    for i in 0..N {
        data[i] = _neon_slli_u16::<IMM8>(data[i]);
    }
    data
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::X64;
    use crate::uint16::neon::data::load_u8x16x4;
    use crate::uint16::test_util::PACK_U16_TO_U8_EXPECTED_UNORDERED_LAYOUT;

    #[test]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_pack_u16_to_u8_ordered() {
        let input = std::array::from_fn(|i| i as u16);

        let data = unsafe { std::mem::transmute::<[u16; X64], [uint16x8_t; 8]>(input) };
        let packed = unsafe { pack_u16_to_u8_ordered(data) };

        let expected = std::array::from_fn(|i| i as u8);
        let view = unsafe { std::mem::transmute::<[uint8x16_t; 4], [u8; X64]>(packed) };
        assert_eq!(view, expected);
    }

    #[test]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_unpack_u8_to_u16_ordered() {
        let input = std::array::from_fn(|i| i as u8);

        let data = unsafe { std::mem::transmute::<[u8; X64], [uint8x16_t; 4]>(input) };
        let unpacked = unsafe { unpack_u8_to_u16_ordered(data) };

        let expected = std::array::from_fn(|i| i as u16);
        let view = unsafe { std::mem::transmute::<[uint16x8_t; 8], [u16; X64]>(unpacked) };
        assert_eq!(view, expected);
    }

    #[test]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_unpack_u8_to_u16_unordered_layout() {
        let expected: [u16; X64] = std::array::from_fn(|i| i as u16);

        let data = unsafe { load_u8x16x4(PACK_U16_TO_U8_EXPECTED_UNORDERED_LAYOUT.as_ptr()) };
        let unpacked = unsafe { unpack_u8_to_u16_unordered(data) };

        let view = unsafe { std::mem::transmute::<[uint16x8_t; 8], [u16; X64]>(unpacked) };
        assert_eq!(view, expected);
    }

    #[test]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_unpack_u4_to_u8_unordered() {
        let expected: [u8; X64] = std::array::from_fn(|i| i as u8 % 16);

        let data = unsafe { load_u8x16x4(expected.as_ptr()) };
        let packed = unsafe { pack_u8_to_u4_unordered(data) };
        let unpacked = unsafe { unpack_u4_to_u8_unordered(packed) };

        let view = unsafe { std::mem::transmute::<[uint8x16_t; 4], [u8; X64]>(unpacked) };
        assert_eq!(view, expected);
    }

    #[test]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_unpack_u2_to_u8_unordered() {
        let expected: [u8; X64] = std::array::from_fn(|i| i as u8 % 4);

        let data = unsafe { load_u8x16x4(expected.as_ptr()) };
        let packed = unsafe { pack_u8_to_u2_unordered(data) };
        let unpacked = unsafe { unpack_u2_to_u8_unordered(packed) };

        let view = unsafe { std::mem::transmute::<[uint8x16_t; 4], [u8; X64]>(unpacked) };
        assert_eq!(view, expected);
    }
}
