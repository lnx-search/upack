use super::AdaptiveCompressibleArray;
use crate::{CompressionDetails, X64, X128};

#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
pub mod avx2;
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub mod avx512;
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
pub mod neon;
pub mod scalar;
mod util;

/// The maximum output size of a compressed buffer for a [X128] block, assuming worst case compression
/// using the **Adaptive Delta** algorithm.
pub const X128_MAX_OUTPUT_LEN: usize = <[u32; X128] as AdaptiveCompressibleArray>::MAX_OUTPUT_SIZE;
const DELTA_OVERHEAD: usize = size_of::<u32>();

#[track_caller]
/// Returns the number of bytes that will be been written for a given bit length and number
/// of elements that were packed when using the **Adaptive Delta** algorithm.
pub const fn max_compressed_size<const BLOCK_SIZE: usize>(bit_length: usize) -> usize {
    const {
        assert!(
            BLOCK_SIZE == X128 || BLOCK_SIZE == X64,
            "BLOCK_SIZE must be either X128 or X256"
        )
    };
    compressed_size(bit_length, BLOCK_SIZE)
}

#[inline]
#[track_caller]
/// Returns the number of bytes that will be been written for a given bit length and number
/// of elements that were packed when using the **Adaptive Delta** algorithm.
pub const fn compressed_size(bit_length: usize, num_elements: usize) -> usize {
    crate::uint32::compressed_size(bit_length, num_elements) + DELTA_OVERHEAD
}

impl AdaptiveCompressibleArray for [u32; X128] {
    type CompressedBuffer = [u8; <Self as AdaptiveCompressibleArray>::MAX_OUTPUT_SIZE];
    type InitialValue = u32;
    const MAX_OUTPUT_SIZE: usize = (X128 * size_of::<u32>()) + DELTA_OVERHEAD;

    fn compress_adaptive_delta(
        initial_value: Self::InitialValue,
        n: usize,
        input: &mut Self,
        output: &mut Self::CompressedBuffer,
    ) -> CompressionDetails {
        assert!(n <= X128, "provided n is is greater than 128");

        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        if avx512::can_use() {
            return unsafe { avx512::pack_adaptive_delta_x128(initial_value, output, input, n) };
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
        if avx2::can_use() {
            return unsafe { avx2::pack_adaptive_delta_x128(initial_value, output, input, n) };
        }

        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        if neon::can_use() {
            return unsafe { neon::pack_adaptive_delta_x128(initial_value, output, input, n) };
        }

        unsafe { scalar::pack_adaptive_delta_x128(initial_value, output, input, n) }
    }

    fn decompress_adaptive_delta(
        initial_value: Self::InitialValue,
        n: usize,
        compressed_bit_length: u8,
        input: &[u8],
        output: &mut Self,
    ) -> usize {
        assert!(
            compressed_bit_length <= 32,
            "compressed bitlength must be no more than 32"
        );
        assert!(
            input.len() >= max_compressed_size::<X128>(compressed_bit_length as usize),
            "input buffer is too small/incorrectly padded to safely decompress len={} expected={}",
            input.len(),
            max_compressed_size::<X128>(compressed_bit_length as usize),
        );
        assert!(n <= X128, "provided n is is greater than 128");

        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        if avx512::can_use() {
            return unsafe {
                avx512::unpack_adaptive_delta_x128(
                    compressed_bit_length,
                    initial_value,
                    input,
                    output,
                    n,
                )
            };
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
        if avx2::can_use() {
            return unsafe {
                avx2::unpack_adaptive_delta_x128(
                    compressed_bit_length,
                    initial_value,
                    input,
                    output,
                    n,
                )
            };
        }

        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        if neon::can_use() {
            return unsafe {
                neon::unpack_adaptive_delta_x128(
                    compressed_bit_length,
                    initial_value,
                    input,
                    output,
                    n,
                )
            };
        }

        unsafe {
            scalar::unpack_adaptive_delta_x128(
                compressed_bit_length,
                initial_value,
                input,
                output,
                n,
            )
        }
    }
}

/// Selects the compression output buffer from the provided buffer that has the additional
/// delta overhead padding.
fn select_compression_buffer(
    input: &mut [u8; X128_MAX_OUTPUT_LEN],
) -> &mut [u8; crate::uint32::X128_MAX_OUTPUT_LEN] {
    const { assert!(X128_MAX_OUTPUT_LEN == crate::uint32::X128_MAX_OUTPUT_LEN + DELTA_OVERHEAD) };
    unsafe { (&mut input[DELTA_OVERHEAD..]).try_into().unwrap_unchecked() }
}
