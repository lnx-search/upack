use std::arch::x86_64::*;

use super::data::*;
use super::{unpack_x64_full, unpack_x64_partial};
use crate::uint16::{max_compressed_size, split_block_mut};
use crate::{X64, X128};

#[inline]
#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to `nbits` bit length  elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X128>(nbits)` bytes to.
/// - The runtime CPU must support the `avx2` instructions.
/// - `nbits` must be between 0 and 16.
/// - `read_n` must be no greater than 128.
pub unsafe fn from_nbits(nbits: usize, input: *const u8, out: &mut [u16; X128], read_n: usize) {
    debug_assert!(nbits <= 16, "BUG: invalid nbits provided: {nbits}");
    debug_assert!(read_n <= X128, "BUG: invalid read_n provided: {read_n}");
    #[allow(clippy::type_complexity)]
    const LUT: [unsafe fn(out: *const u8, &mut [u16; X128], usize); 17] = [
        from_u0, from_u1, from_u2, from_u3, from_u4, from_u5, from_u6, from_u7, from_u8, from_u9,
        from_u10, from_u11, from_u12, from_u13, from_u14, from_u15, from_u16,
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
/// - `nbits` must be between 0 and 16.
/// - `read_n` must be no greater than 128.
pub unsafe fn from_nbits_delta(
    nbits: usize,
    last_value: u16,
    input: *const u8,
    out: &mut [u16; X128],
    read_n: usize,
) {
    debug_assert!(nbits <= 16, "BUG: invalid nbits provided: {nbits}");
    debug_assert!(read_n <= X128, "BUG: invalid read_n provided: {read_n}");
    #[allow(clippy::type_complexity)]
    const LUT: [unsafe fn(u16, out: *const u8, &mut [u16; X128], usize); 17] = [
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
/// - `nbits` must be between 0 and 16.
/// - `read_n` must be no greater than 128.
pub unsafe fn from_nbits_delta1(
    nbits: usize,
    last_value: u16,
    input: *const u8,
    out: &mut [u16; X128],
    read_n: usize,
) {
    debug_assert!(nbits <= 16, "BUG: invalid nbits provided: {nbits}");
    debug_assert!(read_n <= X128, "BUG: invalid read_n provided: {read_n}");
    #[allow(clippy::type_complexity)]
    const LUT: [unsafe fn(u16, out: *const u8, &mut [u16; X128], usize); 17] = [
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
    ];
    let func = unsafe { LUT.get_unchecked(nbits) };
    unsafe { func(last_value, input, out, read_n) };
}

#[target_feature(enable = "avx2")]
unsafe fn from_u0(_input: *const u8, out: &mut [u16; X128], _read_n: usize) {
    out.fill(0);
}

#[target_feature(enable = "avx2")]
unsafe fn from_u0_delta(
    _last_value: u16,
    _input: *const u8,
    out: &mut [u16; X128],
    _read_n: usize,
) {
    out.fill(0);
}

macro_rules! define_x128_unpacker {
    ($func_name:ident, $bit_length:expr) => {
        #[target_feature(enable = "avx2")]
        unsafe fn $func_name(input: *const u8, out: &mut [u16; X128], read_n: usize) {
            let [left, right] = split_block_mut(out);

            if read_n <= 64 {
                let unpacked = unsafe { unpack_x64_partial::$func_name(input.add(0), read_n) };
                store_u16x64(left, unpacked);
            } else if read_n < 128 {
                let unpacked = unsafe { unpack_x64_full::$func_name(input.add(0)) };
                store_u16x64(left, unpacked);
                let unpacked = unsafe {
                    unpack_x64_partial::$func_name(
                        input.add(max_compressed_size::<X64>($bit_length)),
                        read_n - X64,
                    )
                };
                store_u16x64(right, unpacked);
            } else {
                let unpacked = unsafe { unpack_x64_full::$func_name(input.add(0)) };
                store_u16x64(left, unpacked);
                let unpacked = unsafe {
                    unpack_x64_full::$func_name(input.add(max_compressed_size::<X64>($bit_length)))
                };
                store_u16x64(right, unpacked);
            }
        }
    };
}

macro_rules! define_x128_unpacker_delta {
    ($func_name:ident, $unpack_func_name:ident, $bit_length:expr, $delta_func_name:ident) => {
        #[target_feature(enable = "avx2")]
        unsafe fn $func_name(
            last_value: u16,
            input: *const u8,
            out: &mut [u16; X128],
            read_n: usize,
        ) {
            let [left, right] = split_block_mut(out);

            let mut last_value = _mm256_set1_epi32(last_value as i32);

            if read_n <= 64 {
                let mut unpacked =
                    unsafe { unpack_x64_partial::$unpack_func_name(input.add(0), read_n) };
                $delta_func_name(last_value, &mut unpacked);
                store_u16x64(left, unpacked);
            } else if read_n < 128 {
                let mut unpacked = unsafe { unpack_x64_full::$unpack_func_name(input.add(0)) };
                last_value = $delta_func_name(last_value, &mut unpacked);
                store_u16x64(left, unpacked);

                unpacked = unsafe {
                    unpack_x64_partial::$unpack_func_name(
                        input.add(max_compressed_size::<X64>($bit_length)),
                        read_n - X64,
                    )
                };
                $delta_func_name(last_value, &mut unpacked);
                store_u16x64(right, unpacked);
            } else {
                let mut unpacked = unsafe { unpack_x64_full::$unpack_func_name(input.add(0)) };
                last_value = $delta_func_name(last_value, &mut unpacked);
                store_u16x64(left, unpacked);

                unpacked = unsafe {
                    unpack_x64_full::$unpack_func_name(
                        input.add(max_compressed_size::<X64>($bit_length)),
                    )
                };
                $delta_func_name(last_value, &mut unpacked);
                store_u16x64(right, unpacked);
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

#[target_feature(enable = "avx2")]
fn decode_delta(mut last_value: __m256i, block: &mut [__m256i; 4]) -> __m256i {
    #[allow(clippy::needless_range_loop)]
    for i in 0..4 {
        let deltas = block[i];

        let shift1 = _mm256_slli_si256::<2>(deltas);
        let sum1 = _mm256_add_epi16(deltas, shift1);

        let shift2 = _mm256_slli_si256::<4>(sum1);
        let sum2 = _mm256_add_epi16(sum1, shift2);

        let shift3 = _mm256_slli_si256::<8>(sum2);
        let sum3 = _mm256_add_epi16(sum2, shift3);

        let last32 = _mm256_shuffle_epi32::<0xFF>(sum3);
        let tmp = _mm256_shufflelo_epi16::<0xFF>(last32);
        let lane_sum = _mm256_shufflehi_epi16::<0xFF>(tmp);

        let low_lane_broadcast = _mm256_permute2x128_si256::<0x00>(lane_sum, lane_sum);
        let cross_lane_add = _mm256_blend_epi32::<0xF0>(_mm256_setzero_si256(), low_lane_broadcast);
        let sum4 = _mm256_add_epi16(sum3, cross_lane_add);

        let result = _mm256_add_epi16(sum4, last_value);
        block[i] = result;

        let high_lane = _mm256_permute2x128_si256::<0x11>(result, result);
        let l32 = _mm256_shuffle_epi32::<0xFF>(high_lane);
        let ltmp = _mm256_shufflelo_epi16::<0xFF>(l32);
        last_value = _mm256_shufflehi_epi16::<0xFF>(ltmp);
    }

    last_value
}

#[target_feature(enable = "avx2")]
fn decode_delta1(mut last_value: __m256i, block: &mut [__m256i; 4]) -> __m256i {
    let ones = _mm256_set1_epi16(1);

    #[allow(clippy::needless_range_loop)]
    for i in 0..4 {
        let deltas = block[i];

        let deltas_plus_one = _mm256_add_epi16(deltas, ones);

        let shift1 = _mm256_slli_si256::<2>(deltas_plus_one);
        let sum1 = _mm256_add_epi16(deltas_plus_one, shift1);

        let shift2 = _mm256_slli_si256::<4>(sum1);
        let sum2 = _mm256_add_epi16(sum1, shift2);

        let shift3 = _mm256_slli_si256::<8>(sum2);
        let sum3 = _mm256_add_epi16(sum2, shift3);

        let last32 = _mm256_shuffle_epi32::<0xFF>(sum3);
        let tmp = _mm256_shufflelo_epi16::<0xFF>(last32);
        let lane_sum = _mm256_shufflehi_epi16::<0xFF>(tmp);

        let low_lane_broadcast = _mm256_permute2x128_si256::<0x00>(lane_sum, lane_sum);
        let cross_lane_add = _mm256_blend_epi32::<0xF0>(_mm256_setzero_si256(), low_lane_broadcast);
        let sum4 = _mm256_add_epi16(sum3, cross_lane_add);

        let result = _mm256_add_epi16(sum4, last_value);
        block[i] = result;

        let high_lane = _mm256_permute2x128_si256::<0x11>(result, result);
        let l32 = _mm256_shuffle_epi32::<0xFF>(high_lane);
        let ltmp = _mm256_shufflelo_epi16::<0xFF>(l32);
        last_value = _mm256_shufflehi_epi16::<0xFF>(ltmp);
    }

    last_value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg_attr(not(target_feature = "avx2"), ignore)]
    fn test_decode_delta() {
        let expected_values: [u16; X64] = std::array::from_fn(|i| i as u16);
        let mut values = [1; X64];
        values[0] = 0;

        let initial_value = unsafe { _mm256_set1_epi32(0) };
        let mut block = unsafe { load_u16x64(&values) };
        unsafe { decode_delta(initial_value, &mut block) };

        let result = unsafe { std::mem::transmute::<[__m256i; 4], [u16; X64]>(block) };
        assert_eq!(result, expected_values);
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx2"), ignore)]
    fn test_decode_delta1() {
        let expected_values: [u16; X64] = std::array::from_fn(|i| i as u16 + 1);
        let values = [0; X64];

        let initial_value = unsafe { _mm256_set1_epi32(0) };
        let mut block = unsafe { load_u16x64(&values) };
        unsafe { decode_delta1(initial_value, &mut block) };

        let result = unsafe { std::mem::transmute::<[__m256i; 4], [u16; X64]>(block) };
        assert_eq!(result, expected_values);
    }

    // #[test]
    // #[cfg_attr(not(target_feature = "avx2"), ignore)]
    // fn test_v1_layout_regression() {
    //     let tester = crate::uint16::test_util::load_uint32_regression_layout();
    //
    //     let mut output_buffer = [0u16; X128];
    //     for (len, bit_len, expected_output, input) in tester.iter_tests() {
    //         unsafe { from_nbits(bit_len as usize, input.as_ptr(), &mut output_buffer, len) };
    //
    //         let produced_buffer = &output_buffer[..len];
    //         assert_eq!(
    //             produced_buffer,
    //             &expected_output[..len],
    //             "regression test failed, outputs do not match, length:{len} bit_len:{bit_len}"
    //         )
    //     }
    // }
}
