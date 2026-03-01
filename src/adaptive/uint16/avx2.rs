use std::arch::x86_64::*;

use crate::adaptive::uint16::{DELTA_OVERHEAD, X128_MAX_OUTPUT_LEN, compressed_size};
use crate::uint16::avx2::{data, unpack_x64_full, unpack_x64_partial};
use crate::{CompressionDetails, X64, X128};

#[inline]
/// Returns `true` if the runtime CPU can safely execute the AVX2 backed implementation.
pub fn can_use() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
}

#[target_feature(enable = "avx2")]
/// Pack a block of 128 16-bit integers after applying the adaptive delta algorithm
/// and write the compressed block to `out`.
///
/// # Safety
/// - The runtime CPU must support the `avx2` instructions.
/// - `pack_n` must be less than or equal to 128.
pub unsafe fn pack_adaptive_delta_x128(
    mut last_value: u16,
    out: &mut [u8; X128_MAX_OUTPUT_LEN],
    block: &mut [u16; X128],
    pack_n: usize,
) -> CompressionDetails {
    let adaptive_delta = super::util::adaptive_delta_encode(&mut last_value, block, pack_n);

    unsafe { std::ptr::write_unaligned(out.as_mut_ptr().cast(), adaptive_delta) };
    let out = super::select_compression_buffer(out);
    let details = unsafe { crate::uint16::avx2::pack_x128(out, block, pack_n) };

    CompressionDetails {
        compressed_bit_length: details.compressed_bit_length,
        bytes_written: compressed_size(details.compressed_bit_length as usize, pack_n),
    }
}

#[target_feature(enable = "avx2")]
/// Unpack a block of 128 16-bit integers from the compressed input after reversing
/// the Adaptive Delta encoding.
///
/// # Safety
/// - The runtime CPU must support the `avx2` instructions.
/// - `read_n` must be less than or equal to 128.
/// - `input` buffer must be able to hold the _maximum_ possible length of the packed values for
///   a given bit length.
/// - `nbits` must be no greater than `16`.
pub unsafe fn unpack_adaptive_delta_x128(
    nbits: u8,
    last_value: u16,
    input: &[u8],
    block: &mut [u16; X128],
    read_n: usize,
) -> usize {
    let input_ptr = input.as_ptr();

    let adaptive_delta: u16 = unsafe { std::ptr::read_unaligned(input_ptr.add(0).cast()) };

    unsafe {
        from_nbits(
            nbits as usize,
            last_value,
            adaptive_delta,
            input_ptr.add(DELTA_OVERHEAD),
            block,
            read_n,
        )
    };

    compressed_size(nbits as usize, read_n)
}

#[target_feature(enable = "avx2")]
unsafe fn from_nbits(
    nbits: usize,
    last_value: u16,
    adaptive_delta: u16,
    input: *const u8,
    out: &mut [u16; X128],
    read_n: usize,
) {
    debug_assert!(nbits <= 16, "BUG: invalid nbits provided: {nbits}");
    debug_assert!(read_n <= X128, "BUG: invalid read_n provided: {read_n}");
    #[allow(clippy::type_complexity)]
    const LUT: [unsafe fn(u16, u16, out: *const u8, &mut [u16; X128], usize); 17] = [
        from_u0, from_u1, from_u2, from_u3, from_u4, from_u5, from_u6, from_u7, from_u8, from_u9,
        from_u10, from_u11, from_u12, from_u13, from_u14, from_u15, from_u16,
    ];
    let func = unsafe { LUT.get_unchecked(nbits) };
    unsafe { func(last_value, adaptive_delta, input, out, read_n) };
}

#[target_feature(enable = "avx2")]
unsafe fn from_u0(
    last_value: u16,
    adaptive_delta: u16,
    _input: *const u8,
    out: &mut [u16; X128],
    _read_n: usize,
) {
    #[allow(clippy::needless_range_loop)]
    for i in 0..X128 {
        out[i] = (i as u16)
            .wrapping_add(last_value)
            .wrapping_add(adaptive_delta);
    }
}

macro_rules! define_x128_unpacker_adaptive_delta {
    ($func_name:ident, $unpack_func_name:ident, $bit_length:expr, $delta_func_name:ident) => {
        #[target_feature(enable = "avx2")]
        unsafe fn $func_name(
            last_value: u16,
            adaptive_delta: u16,
            input: *const u8,
            out: &mut [u16; X128],
            read_n: usize,
        ) {
            let [left, right] = crate::util::split_slice_mut(out);

            let mut last_value = _mm256_set1_epi16(last_value as i16);
            let adaptive_delta = _mm256_set1_epi16(adaptive_delta as i16);

            if read_n <= 64 {
                let mut unpacked =
                    unsafe { unpack_x64_partial::$unpack_func_name(input.add(0), read_n) };
                $delta_func_name(last_value, adaptive_delta, &mut unpacked);
                data::store_u16x64(left, unpacked);
            } else if read_n < 128 {
                let mut unpacked = unsafe { unpack_x64_full::$unpack_func_name(input.add(0)) };
                last_value = $delta_func_name(last_value, adaptive_delta, &mut unpacked);
                data::store_u16x64(left, unpacked);

                unpacked = unsafe {
                    unpack_x64_partial::$unpack_func_name(
                        input.add(crate::uint16::max_compressed_size::<X64>($bit_length)),
                        read_n - X64,
                    )
                };
                $delta_func_name(last_value, adaptive_delta, &mut unpacked);
                data::store_u16x64(right, unpacked);
            } else {
                let mut unpacked = unsafe { unpack_x64_full::$unpack_func_name(input.add(0)) };
                last_value = $delta_func_name(last_value, adaptive_delta, &mut unpacked);
                data::store_u16x64(left, unpacked);

                unpacked = unsafe {
                    unpack_x64_full::$unpack_func_name(
                        input.add(crate::uint16::max_compressed_size::<X64>($bit_length)),
                    )
                };
                $delta_func_name(last_value, adaptive_delta, &mut unpacked);
                data::store_u16x64(right, unpacked);
            }
        }
    };
}

define_x128_unpacker_adaptive_delta!(from_u1, from_u1, 1, decode_adaptive_delta);
define_x128_unpacker_adaptive_delta!(from_u2, from_u2, 2, decode_adaptive_delta);
define_x128_unpacker_adaptive_delta!(from_u3, from_u3, 3, decode_adaptive_delta);
define_x128_unpacker_adaptive_delta!(from_u4, from_u4, 4, decode_adaptive_delta);
define_x128_unpacker_adaptive_delta!(from_u5, from_u5, 5, decode_adaptive_delta);
define_x128_unpacker_adaptive_delta!(from_u6, from_u6, 6, decode_adaptive_delta);
define_x128_unpacker_adaptive_delta!(from_u7, from_u7, 7, decode_adaptive_delta);
define_x128_unpacker_adaptive_delta!(from_u8, from_u8, 8, decode_adaptive_delta);
define_x128_unpacker_adaptive_delta!(from_u9, from_u9, 9, decode_adaptive_delta);
define_x128_unpacker_adaptive_delta!(from_u10, from_u10, 10, decode_adaptive_delta);
define_x128_unpacker_adaptive_delta!(from_u11, from_u11, 11, decode_adaptive_delta);
define_x128_unpacker_adaptive_delta!(from_u12, from_u12, 12, decode_adaptive_delta);
define_x128_unpacker_adaptive_delta!(from_u13, from_u13, 13, decode_adaptive_delta);
define_x128_unpacker_adaptive_delta!(from_u14, from_u14, 14, decode_adaptive_delta);
define_x128_unpacker_adaptive_delta!(from_u15, from_u15, 15, decode_adaptive_delta);
define_x128_unpacker_adaptive_delta!(from_u16, from_u16, 16, decode_adaptive_delta);

#[target_feature(enable = "avx2")]
fn decode_adaptive_delta(
    mut last_value: __m256i,
    adaptive_delta: __m256i,
    block: &mut [__m256i; 4],
) -> __m256i {
    #[allow(clippy::needless_range_loop)]
    for i in 0..4 {
        let deltas = block[i];

        let deltas_plus_one = _mm256_add_epi16(deltas, adaptive_delta);

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
