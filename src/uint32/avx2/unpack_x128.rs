use std::arch::x86_64::*;

use super::data::*;
use super::{unpack_x64_full, unpack_x64_partial};
use crate::uint32::{max_compressed_size, split_block_mut};
use crate::{X64, X128};

#[inline]
#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to `nbits` bit length  elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X128>(nbits)` bytes to.
/// - The runtime CPU must support the `avx2` instructions.
/// - `nbits` must be between 0 and 32.
/// - `read_n` must be no greater than 128.
pub unsafe fn from_nbits(nbits: usize, input: *const u8, out: &mut [u32; X128], read_n: usize) {
    debug_assert!(read_n <= 32, "BUG: invalid nbits provided: {nbits}");
    debug_assert!(read_n <= X128, "BUG: invalid read_n provided: {read_n}");
    #[allow(clippy::type_complexity)]
    const LUT: [unsafe fn(out: *const u8, &mut [u32; X128], usize); 33] = [
        from_u0, from_u1, from_u2, from_u3, from_u4, from_u5, from_u6, from_u7, from_u8, from_u9,
        from_u10, from_u11, from_u12, from_u13, from_u14, from_u15, from_u16, from_u17, from_u18,
        from_u19, from_u20, from_u21, from_u22, from_u23, from_u24, from_u25, from_u26, from_u27,
        from_u28, from_u29, from_u30, from_u31, from_u32,
    ];
    let func = unsafe { LUT.get_unchecked(nbits) };
    unsafe { func(input, out, read_n) };
}

#[inline]
#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to `nbits` bit length elements which have
/// been delta-encoded.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X128>(nbits)` bytes to.
/// - The runtime CPU must support the `avx2` instructions.
/// - `nbits` must be between 0 and 32.
/// - `read_n` must be no greater than 128.
pub unsafe fn from_nbits_delta(
    nbits: usize,
    last_value: u32,
    input: *const u8,
    out: &mut [u32; X128],
    read_n: usize,
) {
    debug_assert!(read_n <= 32, "BUG: invalid nbits provided: {nbits}");
    debug_assert!(read_n <= X128, "BUG: invalid read_n provided: {read_n}");
    #[allow(clippy::type_complexity)]
    const LUT: [unsafe fn(u32, out: *const u8, &mut [u32; X128], usize); 33] = [
        from_u0_delta,
        from_u1_delta,
        from_u2_delta,
        from_u3_delta,
        from_u4_delta,
        from_u5_delta,
        from_u6_delta,
        from_u7_delta,
        from_u8_delta,
        from_u9_delta,
        from_u10_delta,
        from_u11_delta,
        from_u12_delta,
        from_u13_delta,
        from_u14_delta,
        from_u15_delta,
        from_u16_delta,
        from_u17_delta,
        from_u18_delta,
        from_u19_delta,
        from_u20_delta,
        from_u21_delta,
        from_u22_delta,
        from_u23_delta,
        from_u24_delta,
        from_u25_delta,
        from_u26_delta,
        from_u27_delta,
        from_u28_delta,
        from_u29_delta,
        from_u30_delta,
        from_u31_delta,
        from_u32_delta,
    ];
    let func = unsafe { LUT.get_unchecked(nbits) };
    unsafe { func(last_value, input, out, read_n) };
}

#[inline]
#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to `nbits` bit length elements which have
/// been delta-1-encoded.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X128>(nbits)` bytes to.
/// - The runtime CPU must support the `avx2` instructions.
/// - `nbits` must be between 0 and 32.
/// - `read_n` must be no greater than 128.
pub unsafe fn from_nbits_delta1(
    nbits: usize,
    last_value: u32,
    input: *const u8,
    out: &mut [u32; X128],
    read_n: usize,
) {
    debug_assert!(read_n <= 32, "BUG: invalid nbits provided: {nbits}");
    debug_assert!(read_n <= X128, "BUG: invalid read_n provided: {read_n}");
    #[allow(clippy::type_complexity)]
    const LUT: [unsafe fn(u32, out: *const u8, &mut [u32; X128], usize); 33] = [
        from_u0_delta,
        from_u1_delta1,
        from_u2_delta1,
        from_u3_delta1,
        from_u4_delta1,
        from_u5_delta1,
        from_u6_delta1,
        from_u7_delta1,
        from_u8_delta1,
        from_u9_delta1,
        from_u10_delta1,
        from_u11_delta1,
        from_u12_delta1,
        from_u13_delta1,
        from_u14_delta1,
        from_u15_delta1,
        from_u16_delta1,
        from_u17_delta1,
        from_u18_delta1,
        from_u19_delta1,
        from_u20_delta1,
        from_u21_delta1,
        from_u22_delta1,
        from_u23_delta1,
        from_u24_delta1,
        from_u25_delta1,
        from_u26_delta1,
        from_u27_delta1,
        from_u28_delta1,
        from_u29_delta1,
        from_u30_delta1,
        from_u31_delta1,
        from_u32_delta1,
    ];
    let func = unsafe { LUT.get_unchecked(nbits) };
    unsafe { func(last_value, input, out, read_n) };
}

#[target_feature(enable = "avx2")]
unsafe fn from_u0(_input: *const u8, out: &mut [u32; X128], _read_n: usize) {
    out.fill(0);
}

#[target_feature(enable = "avx2")]
unsafe fn from_u0_delta(
    _last_value: u32,
    _input: *const u8,
    out: &mut [u32; X128],
    _read_n: usize,
) {
    out.fill(0);
}

macro_rules! define_x128_unpacker {
    ($func_name:ident, $bit_length:expr) => {
        #[target_feature(enable = "avx2")]
        unsafe fn $func_name(input: *const u8, out: &mut [u32; X128], read_n: usize) {
            let [left, right] = split_block_mut(out);

            if read_n <= 64 {
                let unpacked = unsafe { unpack_x64_partial::$func_name(input.add(0), read_n) };
                store_u32x64(left, unpacked);
            } else if read_n < 128 {
                let unpacked = unsafe { unpack_x64_full::$func_name(input.add(0)) };
                store_u32x64(left, unpacked);
                let unpacked = unsafe {
                    unpack_x64_partial::$func_name(
                        input.add(max_compressed_size::<X64>($bit_length)),
                        read_n - X64,
                    )
                };
                store_u32x64(right, unpacked);
            } else {
                let unpacked = unsafe { unpack_x64_full::$func_name(input.add(0)) };
                store_u32x64(left, unpacked);
                let unpacked = unsafe {
                    unpack_x64_full::$func_name(input.add(max_compressed_size::<X64>($bit_length)))
                };
                store_u32x64(right, unpacked);
            }
        }
    };
}

macro_rules! define_x128_unpacker_delta {
    ($func_name:ident, $unpack_func_name:ident, $bit_length:expr, $delta_func_name:ident) => {
        #[target_feature(enable = "avx2")]
        unsafe fn $func_name(
            last_value: u32,
            input: *const u8,
            out: &mut [u32; X128],
            read_n: usize,
        ) {
            let [left, right] = split_block_mut(out);

            let mut last_value = _mm256_set1_epi32(last_value as i32);

            if read_n <= 64 {
                let mut unpacked =
                    unsafe { unpack_x64_partial::$unpack_func_name(input.add(0), read_n) };
                $delta_func_name(last_value, &mut unpacked);
                store_u32x64(left, unpacked);
            } else if read_n < 128 {
                let mut unpacked = unsafe { unpack_x64_full::$unpack_func_name(input.add(0)) };
                last_value = $delta_func_name(last_value, &mut unpacked);
                store_u32x64(left, unpacked);

                unpacked = unsafe {
                    unpack_x64_partial::$unpack_func_name(
                        input.add(max_compressed_size::<X64>($bit_length)),
                        read_n - X64,
                    )
                };
                $delta_func_name(last_value, &mut unpacked);
                store_u32x64(right, unpacked);
            } else {
                let mut unpacked = unsafe { unpack_x64_full::$unpack_func_name(input.add(0)) };
                last_value = $delta_func_name(last_value, &mut unpacked);
                store_u32x64(left, unpacked);

                unpacked = unsafe {
                    unpack_x64_full::$unpack_func_name(
                        input.add(max_compressed_size::<X64>($bit_length)),
                    )
                };
                $delta_func_name(last_value, &mut unpacked);
                store_u32x64(right, unpacked);
            }
        }
    };
}

define_x128_unpacker!(from_u1, 1);
define_x128_unpacker!(from_u2, 2);
define_x128_unpacker!(from_u3, 3);
define_x128_unpacker!(from_u4, 4);
define_x128_unpacker!(from_u5, 5);
define_x128_unpacker!(from_u6, 6);
define_x128_unpacker!(from_u7, 7);
define_x128_unpacker!(from_u8, 8);
define_x128_unpacker!(from_u9, 9);
define_x128_unpacker!(from_u10, 10);
define_x128_unpacker!(from_u11, 11);
define_x128_unpacker!(from_u12, 12);
define_x128_unpacker!(from_u13, 13);
define_x128_unpacker!(from_u14, 14);
define_x128_unpacker!(from_u15, 15);
define_x128_unpacker!(from_u16, 16);
define_x128_unpacker!(from_u17, 17);
define_x128_unpacker!(from_u18, 18);
define_x128_unpacker!(from_u19, 19);
define_x128_unpacker!(from_u20, 20);
define_x128_unpacker!(from_u21, 21);
define_x128_unpacker!(from_u22, 22);
define_x128_unpacker!(from_u23, 23);
define_x128_unpacker!(from_u24, 24);
define_x128_unpacker!(from_u25, 25);
define_x128_unpacker!(from_u26, 26);
define_x128_unpacker!(from_u27, 27);
define_x128_unpacker!(from_u28, 28);
define_x128_unpacker!(from_u29, 29);
define_x128_unpacker!(from_u30, 30);
define_x128_unpacker!(from_u31, 31);
define_x128_unpacker!(from_u32, 32);

// Delta encoding
define_x128_unpacker_delta!(from_u1_delta, from_u1, 1, decode_delta);
define_x128_unpacker_delta!(from_u2_delta, from_u2, 2, decode_delta);
define_x128_unpacker_delta!(from_u3_delta, from_u3, 3, decode_delta);
define_x128_unpacker_delta!(from_u4_delta, from_u4, 4, decode_delta);
define_x128_unpacker_delta!(from_u5_delta, from_u5, 5, decode_delta);
define_x128_unpacker_delta!(from_u6_delta, from_u6, 6, decode_delta);
define_x128_unpacker_delta!(from_u7_delta, from_u7, 7, decode_delta);
define_x128_unpacker_delta!(from_u8_delta, from_u8, 8, decode_delta);
define_x128_unpacker_delta!(from_u9_delta, from_u9, 9, decode_delta);
define_x128_unpacker_delta!(from_u10_delta, from_u10, 10, decode_delta);
define_x128_unpacker_delta!(from_u11_delta, from_u11, 11, decode_delta);
define_x128_unpacker_delta!(from_u12_delta, from_u12, 12, decode_delta);
define_x128_unpacker_delta!(from_u13_delta, from_u13, 13, decode_delta);
define_x128_unpacker_delta!(from_u14_delta, from_u14, 14, decode_delta);
define_x128_unpacker_delta!(from_u15_delta, from_u15, 15, decode_delta);
define_x128_unpacker_delta!(from_u16_delta, from_u16, 16, decode_delta);
define_x128_unpacker_delta!(from_u17_delta, from_u17, 17, decode_delta);
define_x128_unpacker_delta!(from_u18_delta, from_u18, 18, decode_delta);
define_x128_unpacker_delta!(from_u19_delta, from_u19, 19, decode_delta);
define_x128_unpacker_delta!(from_u20_delta, from_u20, 20, decode_delta);
define_x128_unpacker_delta!(from_u21_delta, from_u21, 21, decode_delta);
define_x128_unpacker_delta!(from_u22_delta, from_u22, 22, decode_delta);
define_x128_unpacker_delta!(from_u23_delta, from_u23, 23, decode_delta);
define_x128_unpacker_delta!(from_u24_delta, from_u24, 24, decode_delta);
define_x128_unpacker_delta!(from_u25_delta, from_u25, 25, decode_delta);
define_x128_unpacker_delta!(from_u26_delta, from_u26, 26, decode_delta);
define_x128_unpacker_delta!(from_u27_delta, from_u27, 27, decode_delta);
define_x128_unpacker_delta!(from_u28_delta, from_u28, 28, decode_delta);
define_x128_unpacker_delta!(from_u29_delta, from_u29, 29, decode_delta);
define_x128_unpacker_delta!(from_u30_delta, from_u30, 30, decode_delta);
define_x128_unpacker_delta!(from_u31_delta, from_u31, 31, decode_delta);
define_x128_unpacker_delta!(from_u32_delta, from_u32, 32, decode_delta);

// Delta-1 encoding
define_x128_unpacker_delta!(from_u1_delta1, from_u1, 1, decode_delta1);
define_x128_unpacker_delta!(from_u2_delta1, from_u2, 2, decode_delta1);
define_x128_unpacker_delta!(from_u3_delta1, from_u3, 3, decode_delta1);
define_x128_unpacker_delta!(from_u4_delta1, from_u4, 4, decode_delta1);
define_x128_unpacker_delta!(from_u5_delta1, from_u5, 5, decode_delta1);
define_x128_unpacker_delta!(from_u6_delta1, from_u6, 6, decode_delta1);
define_x128_unpacker_delta!(from_u7_delta1, from_u7, 7, decode_delta1);
define_x128_unpacker_delta!(from_u8_delta1, from_u8, 8, decode_delta1);
define_x128_unpacker_delta!(from_u9_delta1, from_u9, 9, decode_delta1);
define_x128_unpacker_delta!(from_u10_delta1, from_u10, 10, decode_delta1);
define_x128_unpacker_delta!(from_u11_delta1, from_u11, 11, decode_delta1);
define_x128_unpacker_delta!(from_u12_delta1, from_u12, 12, decode_delta1);
define_x128_unpacker_delta!(from_u13_delta1, from_u13, 13, decode_delta1);
define_x128_unpacker_delta!(from_u14_delta1, from_u14, 14, decode_delta1);
define_x128_unpacker_delta!(from_u15_delta1, from_u15, 15, decode_delta1);
define_x128_unpacker_delta!(from_u16_delta1, from_u16, 16, decode_delta1);
define_x128_unpacker_delta!(from_u17_delta1, from_u17, 17, decode_delta1);
define_x128_unpacker_delta!(from_u18_delta1, from_u18, 18, decode_delta1);
define_x128_unpacker_delta!(from_u19_delta1, from_u19, 19, decode_delta1);
define_x128_unpacker_delta!(from_u20_delta1, from_u20, 20, decode_delta1);
define_x128_unpacker_delta!(from_u21_delta1, from_u21, 21, decode_delta1);
define_x128_unpacker_delta!(from_u22_delta1, from_u22, 22, decode_delta1);
define_x128_unpacker_delta!(from_u23_delta1, from_u23, 23, decode_delta1);
define_x128_unpacker_delta!(from_u24_delta1, from_u24, 24, decode_delta1);
define_x128_unpacker_delta!(from_u25_delta1, from_u25, 25, decode_delta1);
define_x128_unpacker_delta!(from_u26_delta1, from_u26, 26, decode_delta1);
define_x128_unpacker_delta!(from_u27_delta1, from_u27, 27, decode_delta1);
define_x128_unpacker_delta!(from_u28_delta1, from_u28, 28, decode_delta1);
define_x128_unpacker_delta!(from_u29_delta1, from_u29, 29, decode_delta1);
define_x128_unpacker_delta!(from_u30_delta1, from_u30, 30, decode_delta1);
define_x128_unpacker_delta!(from_u31_delta1, from_u31, 31, decode_delta1);
define_x128_unpacker_delta!(from_u32_delta1, from_u32, 32, decode_delta1);

#[target_feature(enable = "avx2")]
fn decode_delta(mut last_value: __m256i, block: &mut [__m256i; 8]) -> __m256i {
    #[allow(clippy::needless_range_loop)]
    for i in 0..8 {
        let deltas = block[i];

        let shift1 = _mm256_slli_si256(deltas, 4);
        let sum1 = _mm256_add_epi32(deltas, shift1);

        let shift2 = _mm256_slli_si256(sum1, 8);
        let sum2 = _mm256_add_epi32(sum1, shift2);

        let sum_low_lane = _mm256_shuffle_epi32(sum2, 0xFF);
        let low_lane_broadcast = _mm256_permute2x128_si256(sum_low_lane, sum_low_lane, 0x00);
        let cross_lane_add = _mm256_blend_epi32(_mm256_setzero_si256(), low_lane_broadcast, 0xF0);
        let sum3 = _mm256_add_epi32(sum2, cross_lane_add);

        let result = _mm256_add_epi32(sum3, last_value);

        block[i] = result;

        last_value = _mm256_permutevar8x32_epi32(result, _mm256_set1_epi32(7));
    }

    last_value
}

#[target_feature(enable = "avx2")]
fn decode_delta1(mut last_value: __m256i, block: &mut [__m256i; 8]) -> __m256i {
    let ones = _mm256_set1_epi32(1);

    #[allow(clippy::needless_range_loop)]
    for i in 0..8 {
        let deltas = block[i];

        let deltas_plus_one = _mm256_add_epi32(deltas, ones);
        let shift1 = _mm256_slli_si256(deltas_plus_one, 4);
        let sum1 = _mm256_add_epi32(deltas_plus_one, shift1);

        let shift2 = _mm256_slli_si256(sum1, 8);
        let sum2 = _mm256_add_epi32(sum1, shift2);

        let sum_low_lane = _mm256_shuffle_epi32(sum2, 0xFF);
        let low_lane_broadcast = _mm256_permute2x128_si256(sum_low_lane, sum_low_lane, 0x00);

        let cross_lane_add = _mm256_blend_epi32(_mm256_setzero_si256(), low_lane_broadcast, 0xF0);

        let sum3 = _mm256_add_epi32(sum2, cross_lane_add);
        let result = _mm256_add_epi32(sum3, last_value);

        block[i] = result;

        last_value = _mm256_permutevar8x32_epi32(result, _mm256_set1_epi32(7));
    }

    last_value
}
