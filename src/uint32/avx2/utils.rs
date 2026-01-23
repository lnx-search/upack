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
/// Pack 4 sets of registers containing 16-bit elements and produce 2 registers holding
/// 8-bit elements.
///
/// The order of elements is maintained.
///
/// # Important note on saturation
/// This routine produces garbage results if the input elements are beyond the maximum value
/// that can be represented in a [u8] value, meaning any value over [u8::MAX] produces
/// invalid data.
pub(super) fn pack_u16_u8_x8(data: [__m256i; 4]) -> [__m256i; 2] {
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
pub(super) fn pack_u32_u16_split_x8(data: [__m256i; 8]) -> ([__m256i; 2], [__m256i; 2]) {
    let packed_u16 = pack_u32_to_u16_ordered(data);

    let mask = _mm256_set1_epi16(0x00FF);
    let lo_8bits = and_si256(packed_u16, mask);
    let hi_8bits = srli_epi16::<8, 4>(packed_u16);

    let packed_lo_8bits = pack_u16_u8_x8(lo_8bits);
    let packed_hi_8bits = pack_u16_u8_x8(hi_8bits);

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
