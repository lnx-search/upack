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
pub(super) fn pack_u32_u8_x8(data: [__m256i; 8]) -> [__m256i; 2] {
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
