#![allow(clippy::needless_range_loop)]

use std::arch::aarch64::*;

use super::polyfill::*;

#[target_feature(enable = "neon")]
/// Pack 8 sets of registers containing 32-bit elements and produce 2 registers holding
/// 8-bit elements.
///
/// The order of elements is maintained.
pub(super) fn pack_u32_to_u8_ordered(data: [uint32x4_t; 16]) -> [uint8x16_t; 4] {
    let partially_packed = pack_u32_to_u16_ordered(data);
    pack_u16_to_u8_ordered(partially_packed)
}

#[target_feature(enable = "neon")]
/// Pack 8 sets of registers containing 32-bit elements and produce 4 registers holding
/// 16-bit elements.
///
/// The order of elements is maintained.
pub(super) fn pack_u32_to_u16_ordered(data: [uint32x4_t; 16]) -> [uint16x8_t; 8] {
    [
        _neon_cvteu32_u16(data[0], data[1]),
        _neon_cvteu32_u16(data[2], data[3]),
        _neon_cvteu32_u16(data[4], data[5]),
        _neon_cvteu32_u16(data[6], data[7]),
        _neon_cvteu32_u16(data[8], data[9]),
        _neon_cvteu32_u16(data[10], data[11]),
        _neon_cvteu32_u16(data[12], data[13]),
        _neon_cvteu32_u16(data[14], data[15]),
    ]
}

#[target_feature(enable = "neon")]
/// Pack 4 sets of registers containing 16-bit elements and produce 2 registers holding
/// 9-bit elements.
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
/// Unpack 2 sets of registers containing 8-bit elements and produce 8 registers holding
/// 32-bit elements.
pub(super) fn unpack_u8_to_u32_ordered(data: [uint8x16_t; 4]) -> [uint32x4_t; 16] {
    let partially_unpacked = unpack_u8_to_u16_ordered(data);
    unpack_u16_to_u32_ordered(partially_unpacked)
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

#[target_feature(enable = "neon")]
/// Unpack 4 sets of registers containing 16-bit elements and produce 8 registers holding
/// 32-bit elements.
pub(super) fn unpack_u16_to_u32_ordered(data: [uint16x8_t; 8]) -> [uint32x4_t; 16] {
    let [d1, d2] = _neon_cvteu16_u32(data[0]);
    let [d3, d4] = _neon_cvteu16_u32(data[1]);
    let [d5, d6] = _neon_cvteu16_u32(data[2]);
    let [d7, d8] = _neon_cvteu16_u32(data[3]);
    let [d9, d10] = _neon_cvteu16_u32(data[4]);
    let [d11, d12] = _neon_cvteu16_u32(data[5]);
    let [d13, d14] = _neon_cvteu16_u32(data[6]);
    let [d15, d16] = _neon_cvteu16_u32(data[7]);

    [
        d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16,
    ]
}

#[target_feature(enable = "neon")]
/// Pack 8 sets of registers containing 32-bit elements and produce 2 registers holding
/// 8-bit elements.
///
/// The order of elements are _not_ maintained.
pub(super) fn pack_u32_to_u8_unordered(data: [uint32x4_t; 16]) -> [uint8x16_t; 4] {
    let d1 = pack_u32_to_u8_unordered_mini_block(data[0], data[8], data[4], data[12]);
    let d2 = pack_u32_to_u8_unordered_mini_block(data[1], data[9], data[5], data[13]);
    let d3 = pack_u32_to_u8_unordered_mini_block(data[2], data[10], data[6], data[14]);
    let d4 = pack_u32_to_u8_unordered_mini_block(data[3], data[11], data[7], data[15]);

    [d1, d2, d3, d4]
}

#[target_feature(enable = "neon")]
fn pack_u32_to_u8_unordered_mini_block(
    a: uint32x4_t,
    b: uint32x4_t,
    c: uint32x4_t,
    d: uint32x4_t,
) -> uint8x16_t {
    let shift_1 = a;
    let shift_2 = _neon_slli_u32::<8>(b);
    let shift_3 = _neon_slli_u32::<16>(c);
    let shift_4 = _neon_slli_u32::<24>(d);

    let mut accumulator = shift_1;
    accumulator = _neon_or_u32(accumulator, shift_2);
    accumulator = _neon_or_u32(accumulator, shift_3);
    accumulator = _neon_or_u32(accumulator, shift_4);

    vreinterpretq_u8_u32(accumulator)
}

#[target_feature(enable = "neon")]
/// Pack 8 sets of registers containing 32-bit elements and produce 4 registers holding
/// 16-bit elements.
///
/// The order of elements are _not_ maintained.
pub(super) fn pack_u32_to_u16_unordered(data: [uint32x4_t; 16]) -> [uint16x8_t; 8] {
    let d1 = _neon_or_u32(data[0], _neon_slli_u32::<16>(data[4]));
    let d2 = _neon_or_u32(data[1], _neon_slli_u32::<16>(data[5]));
    let d3 = _neon_or_u32(data[2], _neon_slli_u32::<16>(data[6]));
    let d4 = _neon_or_u32(data[3], _neon_slli_u32::<16>(data[7]));

    let d5 = _neon_or_u32(data[8], _neon_slli_u32::<16>(data[12]));
    let d6 = _neon_or_u32(data[9], _neon_slli_u32::<16>(data[13]));
    let d7 = _neon_or_u32(data[10], _neon_slli_u32::<16>(data[14]));
    let d8 = _neon_or_u32(data[11], _neon_slli_u32::<16>(data[15]));

    [
        vreinterpretq_u16_u32(d1),
        vreinterpretq_u16_u32(d2),
        vreinterpretq_u16_u32(d3),
        vreinterpretq_u16_u32(d4),
        vreinterpretq_u16_u32(d5),
        vreinterpretq_u16_u32(d6),
        vreinterpretq_u16_u32(d7),
        vreinterpretq_u16_u32(d8),
    ]
}

#[inline]
#[target_feature(enable = "neon")]
/// Unpack 2 sets of registers containing 8-bit elements and produce 8 registers holding
/// 32-bit elements.
pub(super) fn unpack_u8_to_u32_unordered(data: [uint8x16_t; 4]) -> [uint32x4_t; 16] {
    let partially_unpacked = unpack_u8_to_u16_unordered(data);
    unpack_u16_to_u32_unordered(partially_unpacked)
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
/// Unpack 4 sets of registers containing 16-bit elements and produce 8 registers holding
/// 32-bit elements.
pub(super) fn unpack_u16_to_u32_unordered(data: [uint16x8_t; 8]) -> [uint32x4_t; 16] {
    let zeroes = _neon_set1_u16(0);
    let d1 = _neon_blend_every_other_u16(data[0], zeroes);
    let d2 = _neon_blend_every_other_u16(data[1], zeroes);
    let d3 = _neon_blend_every_other_u16(data[2], zeroes);
    let d4 = _neon_blend_every_other_u16(data[3], zeroes);

    let d5 = _neon_srli_u32::<16>(vreinterpretq_u32_u16(data[0]));
    let d6 = _neon_srli_u32::<16>(vreinterpretq_u32_u16(data[1]));
    let d7 = _neon_srli_u32::<16>(vreinterpretq_u32_u16(data[2]));
    let d8 = _neon_srli_u32::<16>(vreinterpretq_u32_u16(data[3]));

    let d9 = _neon_blend_every_other_u16(data[4], zeroes);
    let d10 = _neon_blend_every_other_u16(data[5], zeroes);
    let d11 = _neon_blend_every_other_u16(data[6], zeroes);
    let d12 = _neon_blend_every_other_u16(data[7], zeroes);

    let d13 = _neon_srli_u32::<16>(vreinterpretq_u32_u16(data[4]));
    let d14 = _neon_srli_u32::<16>(vreinterpretq_u32_u16(data[5]));
    let d15 = _neon_srli_u32::<16>(vreinterpretq_u32_u16(data[6]));
    let d16 = _neon_srli_u32::<16>(vreinterpretq_u32_u16(data[7]));

    [
        vreinterpretq_u32_u16(d1),
        vreinterpretq_u32_u16(d2),
        vreinterpretq_u32_u16(d3),
        vreinterpretq_u32_u16(d4),
        d5,
        d6,
        d7,
        d8,
        vreinterpretq_u32_u16(d9),
        vreinterpretq_u32_u16(d10),
        vreinterpretq_u32_u16(d11),
        vreinterpretq_u32_u16(d12),
        d13,
        d14,
        d15,
        d16,
    ]
}

#[inline]
#[target_feature(enable = "neon")]
/// Given 8 sets of 32-bit registers, pack them into 4 sets of 16-bit registers, then
/// split each 16-bit element into a high and low halves, returning the 8-bit halves in separate
/// registers in their high and low forms respectively.
///
/// The order of elements is maintained.
pub(super) fn pack_u32_to_u16_split_ordered(
    data: [uint32x4_t; 16],
) -> ([uint8x16_t; 4], [uint8x16_t; 4]) {
    let packed_u16 = pack_u32_to_u16_ordered(data);

    let mask = _neon_set1_u16(0x00FF);
    let lo_8bits = and_u16(packed_u16, mask);
    let hi_8bits = srli_u16::<8, 8>(packed_u16);

    let packed_lo_8bits = pack_u16_to_u8_ordered(lo_8bits);
    let packed_hi_8bits = pack_u16_to_u8_ordered(hi_8bits);

    (packed_hi_8bits, packed_lo_8bits)
}

#[inline]
#[target_feature(enable = "neon")]
/// Given 8 sets of 32-bit registers, pack them into 4 sets of 16-bit registers, then
/// split each 16-bit element into a high and low halves, returning the 8-bit halves in separate
/// registers in their high and low forms respectively.
///
/// The order of elements are not maintained.
pub(super) fn pack_u32_to_u16_split_unordered(
    data: [uint32x4_t; 16],
) -> ([uint8x16_t; 4], [uint8x16_t; 4]) {
    let packed = pack_u32_to_u16_unordered(data);

    let lo_mask = _neon_set1_u16(0x00FF);
    let hi_mask = _neon_set1_u16(0xFF00);

    let lo_8bits_1a = _neon_slli_u16::<8>(packed[4]);
    let lo_8bits_1b = _neon_slli_u16::<8>(packed[5]);
    let lo_8bits_1c = _neon_slli_u16::<8>(packed[6]);
    let lo_8bits_1d = _neon_slli_u16::<8>(packed[7]);

    let lo_8bits_2a = _neon_and_u16(packed[0], lo_mask);
    let lo_8bits_2b = _neon_and_u16(packed[1], lo_mask);
    let lo_8bits_2c = _neon_and_u16(packed[2], lo_mask);
    let lo_8bits_2d = _neon_and_u16(packed[3], lo_mask);

    let lo_8bits_a = _neon_or_u16(lo_8bits_1a, lo_8bits_2a);
    let lo_8bits_b = _neon_or_u16(lo_8bits_1b, lo_8bits_2b);
    let lo_8bits_c = _neon_or_u16(lo_8bits_1c, lo_8bits_2c);
    let lo_8bits_d = _neon_or_u16(lo_8bits_1d, lo_8bits_2d);

    let lo = [
        vreinterpretq_u8_u16(lo_8bits_a),
        vreinterpretq_u8_u16(lo_8bits_b),
        vreinterpretq_u8_u16(lo_8bits_c),
        vreinterpretq_u8_u16(lo_8bits_d),
    ];

    let hi_8bits_1a = _neon_srli_u16::<8>(packed[0]);
    let hi_8bits_1b = _neon_srli_u16::<8>(packed[1]);
    let hi_8bits_1c = _neon_srli_u16::<8>(packed[2]);
    let hi_8bits_1d = _neon_srli_u16::<8>(packed[3]);
    let hi_8bits_2a = _neon_and_u16(packed[4], hi_mask);
    let hi_8bits_2b = _neon_and_u16(packed[5], hi_mask);
    let hi_8bits_2c = _neon_and_u16(packed[6], hi_mask);
    let hi_8bits_2d = _neon_and_u16(packed[7], hi_mask);

    let hi_8bits_a = _neon_or_u16(hi_8bits_1a, hi_8bits_2a);
    let hi_8bits_b = _neon_or_u16(hi_8bits_1b, hi_8bits_2b);
    let hi_8bits_c = _neon_or_u16(hi_8bits_1c, hi_8bits_2c);
    let hi_8bits_d = _neon_or_u16(hi_8bits_1d, hi_8bits_2d);

    let hi = [
        vreinterpretq_u8_u16(hi_8bits_a),
        vreinterpretq_u8_u16(hi_8bits_b),
        vreinterpretq_u8_u16(hi_8bits_c),
        vreinterpretq_u8_u16(hi_8bits_d),
    ];

    (hi, lo)
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
pub(super) fn and_u32<const N: usize>(
    mut data: [uint32x4_t; N],
    mask: uint32x4_t,
) -> [uint32x4_t; N] {
    for i in 0..N {
        data[i] = _neon_and_u32(data[i], mask);
    }
    data
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
pub(super) fn or_u32_all<const N: usize>(
    mut a: [uint32x4_t; N],
    b: [uint32x4_t; N],
) -> [uint32x4_t; N] {
    for i in 0..N {
        a[i] = _neon_or_u32(a[i], b[i]);
    }
    a
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
/// Shift all registers right by [IMM8] in 32-bit lanes.
pub(super) fn srli_u32<const IMM8: i32, const N: usize>(
    mut data: [uint32x4_t; N],
) -> [uint32x4_t; N] {
    for i in 0..N {
        data[i] = _neon_srli_u32::<IMM8>(data[i]);
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

#[inline]
#[target_feature(enable = "neon")]
/// Shift all registers left by [IMM8] in 32-bit lanes.
pub(super) fn slli_u32<const IMM8: i32, const N: usize>(
    mut data: [uint32x4_t; N],
) -> [uint32x4_t; N] {
    for i in 0..N {
        data[i] = _neon_slli_u32::<IMM8>(data[i]);
    }
    data
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::X64;
    use crate::uint32::neon::data::{load_u8x16x4, load_u32x64};
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
        let view = unsafe { std::mem::transmute::<[uint16x8_t; 8], [u16; X64]>(packed) };
        assert_eq!(view, expected);
    }

    #[test]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_pack_u32_to_u8_ordered() {
        let input = std::array::from_fn(|i| i as u32);

        let data = unsafe { load_u32x64(&input) };
        let packed = unsafe { pack_u32_to_u8_ordered(data) };

        let expected = std::array::from_fn(|i| i as u8);
        let view = unsafe { std::mem::transmute::<[uint8x16_t; 4], [u8; X64]>(packed) };
        assert_eq!(view, expected);
    }

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
    fn test_unpack_u16_to_u32_ordered() {
        let input = std::array::from_fn(|i| i as u16);

        let data = unsafe { std::mem::transmute::<[u16; X64], [uint16x8_t; 8]>(input) };
        let unpacked = unsafe { unpack_u16_to_u32_ordered(data) };

        let expected = std::array::from_fn(|i| i as u32);
        let view = unsafe { std::mem::transmute::<[uint32x4_t; 16], [u32; X64]>(unpacked) };
        assert_eq!(view, expected);
    }

    #[test]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_unpack_u8_to_u32_ordered() {
        let input = std::array::from_fn(|i| i as u8);

        let data = unsafe { std::mem::transmute::<[u8; X64], [uint8x16_t; 4]>(input) };
        let unpacked = unsafe { unpack_u8_to_u32_ordered(data) };

        let expected = std::array::from_fn(|i| i as u32);
        let view = unsafe { std::mem::transmute::<[uint32x4_t; 16], [u32; X64]>(unpacked) };
        assert_eq!(view, expected);
    }

    #[test]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_pack_u32_to_u16_unordered() {
        let input = std::array::from_fn(|i| i as u32);

        let data = unsafe { load_u32x64(&input) };
        let packed = unsafe { pack_u32_to_u16_unordered(data) };

        let view = unsafe { std::mem::transmute::<[uint16x8_t; 8], [u16; X64]>(packed) };
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

        let view = unsafe { std::mem::transmute::<[uint8x16_t; 4], [u8; X64]>(packed) };
        assert_eq!(view[..8], [0, 32, 16, 48, 1, 33, 17, 49]);
        assert_eq!(view[8..][..8], [2, 34, 18, 50, 3, 35, 19, 51]);
        assert_eq!(view[16..][..8], [4, 36, 20, 52, 5, 37, 21, 53]);
        assert_eq!(view[24..][..8], [6, 38, 22, 54, 7, 39, 23, 55]);
    }

    #[test]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_unpack_u16_to_u32_unordered() {
        let input = std::array::from_fn(|i| i as u32);

        let data = unsafe { std::mem::transmute::<[u32; X64], [uint32x4_t; 16]>(input) };
        let packed = unsafe { pack_u32_to_u16_unordered(data) };
        let unpacked = unsafe { unpack_u16_to_u32_unordered(packed) };

        let view = unsafe { std::mem::transmute::<[uint32x4_t; 16], [u32; X64]>(unpacked) };
        assert_eq!(view, input);
    }

    #[test]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_pack_u32_to_u16_unordered_layout() {
        let input: [u32; X64] = std::array::from_fn(|i| i as u32);

        let data = unsafe { load_u32x64(&input) };
        let packed = unsafe { pack_u32_to_u16_unordered(data) };

        let view = unsafe { std::mem::transmute::<[uint16x8_t; 8], [u16; X64]>(packed) };
        assert_eq!(view, PACK_U32_TO_U16_EXPECTED_UNORDERED_LAYOUT);
    }

    #[test]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_pack_u32_to_u8_unordered_layout() {
        let input: [u32; X64] = std::array::from_fn(|i| i as u32);

        let data = unsafe { load_u32x64(&input) };
        let packed = unsafe { pack_u32_to_u8_unordered(data) };

        let view = unsafe { std::mem::transmute::<[uint8x16_t; 4], [u8; X64]>(packed) };
        assert_eq!(view, PACK_U32_TO_U8_EXPECTED_UNORDERED_LAYOUT);
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
