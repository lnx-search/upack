use std::arch::x86_64::*;

pub const _MM_TERNLOG_A: i32 = 0xF0; // 11110000
pub const _MM_TERNLOG_B: i32 = 0xCC; // 11001100
pub const _MM_TERNLOG_C: i32 = 0xAA; // 10101010

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Pack 8 sets of registers containing 32-bit elements and produce 4 registers holding
/// 16-bit elements.
///
/// The order of elements is maintained.
///
/// # Important note on saturation
/// This routine produces garbage results if the input elements are beyond the maximum value
/// that can be represented in a [u16] value, meaning any value over [u16::MAX] produces
/// invalid data.
pub(super) fn pack_u32_u16_x8(data: [__m512i; 8]) -> [__m512i; 4] {
    let permute_mask = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);

    let unordered1 = _mm512_packus_epi32(data[0], data[1]);
    let unordered2 = _mm512_packus_epi32(data[2], data[3]);
    let unordered3 = _mm512_packus_epi32(data[4], data[5]);
    let unordered4 = _mm512_packus_epi32(data[6], data[7]);

    let ordered1 = _mm512_permutexvar_epi64(permute_mask, unordered1);
    let ordered2 = _mm512_permutexvar_epi64(permute_mask, unordered2);
    let ordered3 = _mm512_permutexvar_epi64(permute_mask, unordered3);
    let ordered4 = _mm512_permutexvar_epi64(permute_mask, unordered4);

    [ordered1, ordered2, ordered3, ordered4]
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack 4 sets of registers containing 16-bit elements and produce 8 registers holding
/// 32-bit elements.
///
/// This is the inverse of [pack_u32_u16_x8].
pub(super) fn unpack_u16_u32_x8(data: [__m512i; 4]) -> [__m512i; 8] {
    let [packed1, packed2, packed3, packed4] = data;

    let tmp = _mm512_alignr_epi32::<8>(packed1, packed1);
    let d1 = convert_lower_x16_u16_epi32(packed1);
    let d2 = convert_lower_x16_u16_epi32(tmp);

    let tmp = _mm512_alignr_epi32::<8>(packed2, packed2);
    let d3 = convert_lower_x16_u16_epi32(packed2);
    let d4 = convert_lower_x16_u16_epi32(tmp);

    let tmp = _mm512_alignr_epi32::<8>(packed3, packed3);
    let d5 = convert_lower_x16_u16_epi32(packed3);
    let d6 = convert_lower_x16_u16_epi32(tmp);

    let tmp = _mm512_alignr_epi32::<8>(packed4, packed4);
    let d7 = convert_lower_x16_u16_epi32(packed4);
    let d8 = convert_lower_x16_u16_epi32(tmp);

    [d1, d2, d3, d4, d5, d6, d7, d8]
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Pack 8 sets of registers containing 32-bit elements and produce 2 registers holding
/// 8-bit elements.
///
/// The order of elements is maintained.
///
/// # Important note on saturation
/// This routine produces garbage results if the input elements are beyond the maximum value
/// that can be represented in a [u8] value, meaning any value over [u8::MAX] produces
/// invalid data.
pub(super) fn pack_u32_u8_x8(data: [__m512i; 8]) -> [__m512i; 2] {
    let permute_mask = _mm512_set_epi32(
        15, 11, 7, 3, // D chunks
        14, 10, 6, 2, // C chunks
        13, 9, 5, 1, // B chunks
        12, 8, 4, 0, // A chunks
    );

    let packed1 = _mm512_packus_epi32(data[0], data[1]);
    let packed2 = _mm512_packus_epi32(data[2], data[3]);
    let packed3 = _mm512_packus_epi32(data[4], data[5]);
    let packed4 = _mm512_packus_epi32(data[6], data[7]);

    let unordered1 = _mm512_packus_epi16(packed1, packed2);
    let unordered2 = _mm512_packus_epi16(packed3, packed4);

    let ordered1 = _mm512_permutexvar_epi32(permute_mask, unordered1);
    let ordered2 = _mm512_permutexvar_epi32(permute_mask, unordered2);

    [ordered1, ordered2]
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack 2 sets of registers containing 8-bit elements and produce 8 registers holding
/// 32-bit elements.
///
/// This is the inverse of [pack_u32_u8_x8].
pub(super) fn unpack_u8_u32_x8(data: [__m512i; 2]) -> [__m512i; 8] {
    let [packed1, packed2] = data;

    let tmp1 = _mm512_alignr_epi32::<4>(packed1, packed1);
    let tmp2 = _mm512_alignr_epi32::<8>(packed1, packed1);
    let tmp3 = _mm512_alignr_epi32::<12>(packed1, packed1);

    let d1 = convert_lower_x16_u8_epi32(packed1);
    let d2 = convert_lower_x16_u8_epi32(tmp1);
    let d3 = convert_lower_x16_u8_epi32(tmp2);
    let d4 = convert_lower_x16_u8_epi32(tmp3);

    let tmp4 = _mm512_alignr_epi32::<4>(packed2, packed2);
    let tmp5 = _mm512_alignr_epi32::<8>(packed2, packed2);
    let tmp6 = _mm512_alignr_epi32::<12>(packed2, packed2);

    let d5 = convert_lower_x16_u8_epi32(packed2);
    let d6 = convert_lower_x16_u8_epi32(tmp4);
    let d7 = convert_lower_x16_u8_epi32(tmp5);
    let d8 = convert_lower_x16_u8_epi32(tmp6);

    [d1, d2, d3, d4, d5, d6, d7, d8]
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
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
pub(super) fn pack_u32_u16_split_x8(data: [__m512i; 8]) -> ([__m512i; 2], [__m512i; 2]) {
    let permute_mask = _mm512_set_epi32(
        15, 11, 7, 3, // D chunks
        14, 10, 6, 2, // C chunks
        13, 9, 5, 1, // B chunks
        12, 8, 4, 0, // A chunks
    );

    let packed1 = _mm512_packus_epi32(data[0], data[1]);
    let packed2 = _mm512_packus_epi32(data[2], data[3]);
    let packed3 = _mm512_packus_epi32(data[4], data[5]);
    let packed4 = _mm512_packus_epi32(data[6], data[7]);

    let lo_mask = _mm512_set1_epi16(0b1111_1111);
    let lo_bits = and_si512([packed1, packed2, packed3, packed4], lo_mask);
    let lo_unordered1 = _mm512_packus_epi16(lo_bits[0], lo_bits[1]);
    let lo_unordered2 = _mm512_packus_epi16(lo_bits[2], lo_bits[3]);

    let hi_bits = srli_epi16::<8, 4>([packed1, packed2, packed3, packed4]);
    let hi_unordered1 = _mm512_packus_epi16(hi_bits[0], hi_bits[1]);
    let hi_unordered2 = _mm512_packus_epi16(hi_bits[2], hi_bits[3]);

    let lo_ordered1 = _mm512_permutexvar_epi32(permute_mask, lo_unordered1);
    let lo_ordered2 = _mm512_permutexvar_epi32(permute_mask, lo_unordered2);

    let hi_ordered1 = _mm512_permutexvar_epi32(permute_mask, hi_unordered1);
    let hi_ordered2 = _mm512_permutexvar_epi32(permute_mask, hi_unordered2);

    ([hi_ordered1, hi_ordered2], [lo_ordered1, lo_ordered2])
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
fn convert_lower_x16_u8_epi32(reg: __m512i) -> __m512i {
    _mm512_cvtepu8_epi32(_mm512_castsi512_si128(reg))
}

#[target_feature(enable = "avx512f")]
fn convert_lower_x16_u16_epi32(reg: __m512i) -> __m512i {
    _mm512_cvtepu16_epi32(_mm512_castsi512_si256(reg))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::X128;
    use crate::uint32::avx512::data::load_u32x128;

    #[test]
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_pack_u32_u16() {
        let mut data = [0; X128];
        #[allow(clippy::needless_range_loop)]
        for i in 0..X128 {
            data[i] = i as u32;
        }

        let data = unsafe { load_u32x128(&data) };
        let packed = unsafe { pack_u32_u16_x8(data) };

        let mut expected = [0; X128];
        #[allow(clippy::needless_range_loop)]
        for i in 0..X128 {
            expected[i] = i as u16;
        }

        let view = unsafe { std::mem::transmute::<[__m512i; 4], [u16; X128]>(packed) };
        assert_eq!(view, expected);
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_pack_u32_u16_saturation() {
        let data = [u16::MAX as u32 + 1; X128];
        let data = unsafe { load_u32x128(&data) };
        let packed = unsafe { pack_u32_u16_x8(data) };
        let view = unsafe { std::mem::transmute::<[__m512i; 4], [u16; X128]>(packed) };
        assert_eq!(view, [u16::MAX; X128]);

        let data = [u16::MAX as u32; X128];
        let data = unsafe { load_u32x128(&data) };
        let packed = unsafe { pack_u32_u16_x8(data) };
        let view = unsafe { std::mem::transmute::<[__m512i; 4], [u16; X128]>(packed) };
        assert_eq!(view, [u16::MAX; X128]);
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_pack_u32_u8() {
        let mut data = [0; X128];
        #[allow(clippy::needless_range_loop)]
        for i in 0..X128 {
            data[i] = i as u32;
        }

        let data = unsafe { load_u32x128(&data) };
        let packed = unsafe { pack_u32_u8_x8(data) };

        let mut expected = [0; X128];
        #[allow(clippy::needless_range_loop)]
        for i in 0..X128 {
            expected[i] = i as u8;
        }

        let view = unsafe { std::mem::transmute::<[__m512i; 2], [u8; X128]>(packed) };
        assert_eq!(view, expected);
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_pack_u32_u8_saturation() {
        let data = [u8::MAX as u32 + 1; X128];
        let data = unsafe { load_u32x128(&data) };
        let packed = unsafe { pack_u32_u8_x8(data) };
        let view = unsafe { std::mem::transmute::<[__m512i; 2], [u8; X128]>(packed) };
        assert_eq!(view, [255; X128]); // Because of how the packs instruction saturates.

        let data = [u8::MAX as u32; X128];
        let data = unsafe { load_u32x128(&data) };
        let packed = unsafe { pack_u32_u8_x8(data) };
        let view = unsafe { std::mem::transmute::<[__m512i; 2], [u8; X128]>(packed) };
        assert_eq!(view, [255; X128]);
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_unpack_u8_u32() {
        let mut data: [u8; X128] = [0; X128];
        #[allow(clippy::needless_range_loop)]
        for i in 0..X128 {
            data[i] = i as u8;
        }

        let d1 = unsafe { _mm512_loadu_epi8(data.as_ptr().add(0).cast()) };
        let d2 = unsafe { _mm512_loadu_epi8(data.as_ptr().add(64).cast()) };
        let unpacked = unsafe { unpack_u8_u32_x8([d1, d2]) };

        let mut expected = [0; X128];
        #[allow(clippy::needless_range_loop)]
        for i in 0..X128 {
            expected[i] = i as u32;
        }

        let view = unsafe { std::mem::transmute::<[__m512i; 8], [u32; X128]>(unpacked) };
        assert_eq!(view, expected);
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_pack_u32_u16_split_x8() {
        let mut data = [0; X128];
        #[allow(clippy::needless_range_loop)]
        for i in 0..X128 {
            data[i] = i as u32 * 8;
        }

        let data = unsafe { load_u32x128(&data) };
        let (hi, lo) = unsafe { pack_u32_u16_split_x8(data) };

        let lo = unsafe { std::mem::transmute::<[__m512i; 2], [u8; X128]>(lo) };
        let hi = unsafe { std::mem::transmute::<[__m512i; 2], [u8; X128]>(hi) };
        assert_eq!(lo[0..4], [0, 8, 16, 24]);
        assert_eq!(hi[0..4], [0, 0, 0, 0]);

        assert_eq!(lo[64], 0b0000_0000);
        assert_eq!(hi[64], 0b0000_0010);

        assert_eq!(lo[65], 0b0000_1000);
        assert_eq!(hi[65], 0b0000_0010);

        assert_eq!(lo[73], 0b0100_1000);
        assert_eq!(hi[73], 0b0000_0010);
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_pack_u32_u16_split_x8_saturation() {
        let data = [(u16::MAX - 1) as u32; X128];
        let data = unsafe { load_u32x128(&data) };
        let (hi, lo) = unsafe { pack_u32_u16_split_x8(data) };

        let lo = unsafe { std::mem::transmute::<[__m512i; 2], [u8; X128]>(lo) };
        let hi = unsafe { std::mem::transmute::<[__m512i; 2], [u8; X128]>(hi) };
        assert_eq!(lo[64], 0b1111_1110);
        assert_eq!(hi[64], 0b1111_1111);

        assert_eq!(lo[65], 0b1111_1110);
        assert_eq!(hi[65], 0b1111_1111);

        assert_eq!(lo[73], 0b1111_1110);
        assert_eq!(hi[73], 0b1111_1111);

        let data = [u16::MAX as u32; X128];
        let data = unsafe { load_u32x128(&data) };
        let (hi, lo) = unsafe { pack_u32_u16_split_x8(data) };

        let lo = unsafe { std::mem::transmute::<[__m512i; 2], [u8; X128]>(lo) };
        let hi = unsafe { std::mem::transmute::<[__m512i; 2], [u8; X128]>(hi) };
        assert_eq!(lo[64], 0b1111_1111);
        assert_eq!(hi[64], 0b1111_1111);

        assert_eq!(lo[65], 0b1111_1111);
        assert_eq!(hi[65], 0b1111_1111);

        assert_eq!(lo[73], 0b1111_1111);
        assert_eq!(hi[73], 0b1111_1111);
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_unpack_u16_u32() {
        let mut data: [u16; X128] = [0; X128];
        #[allow(clippy::needless_range_loop)]
        for i in 0..X128 {
            data[i] = i as u16;
        }

        let d1 = unsafe { _mm512_loadu_epi16(data.as_ptr().add(0).cast()) };
        let d2 = unsafe { _mm512_loadu_epi16(data.as_ptr().add(32).cast()) };
        let d3 = unsafe { _mm512_loadu_epi16(data.as_ptr().add(64).cast()) };
        let d4 = unsafe { _mm512_loadu_epi16(data.as_ptr().add(96).cast()) };
        let unpacked = unsafe { unpack_u16_u32_x8([d1, d2, d3, d4]) };

        let mut expected = [0; X128];
        #[allow(clippy::needless_range_loop)]
        for i in 0..X128 {
            expected[i] = i as u32;
        }

        let view = unsafe { std::mem::transmute::<[__m512i; 8], [u32; X128]>(unpacked) };
        assert_eq!(view, expected);
    }
}
