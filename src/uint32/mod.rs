//! 32-bit integer bitpacking routines
use crate::core::{CompressibleArray, CompressionDetails};
use crate::{X128, X256};

#[cfg(target_endian = "big")]
compile_error!("big endian machines are not supported");

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[doc(hidden)]
pub mod avx512;

/// The maximum output size of a compressed buffer for a [X128] block, assuming worst case compression.
pub const X128_MAX_OUTPUT_LEN: usize = <[u32; X128] as CompressibleArray>::MAX_OUTPUT_SIZE;
// /// The maximum output size of a compressed buffer for a [X256] block, assuming worst case compression.
// pub const X256_MAX_OUTPUT_LEN: usize = <[u32; X256] as CompressibleArray>::MAX_OUTPUT_SIZE;

#[track_caller]
/// Returns the number of bytes that will be been written for a given bit length and number
/// of elements that were packed when using the bitpacking functions.
pub const fn max_compressed_size<const BLOCK_SIZE: usize>(bit_length: usize) -> usize {
    const {
        assert!(
            BLOCK_SIZE == X128 || BLOCK_SIZE == X256,
            "BLOCK_SIZE must be either X128 or X256"
        )
    };
    compressed_size(bit_length, BLOCK_SIZE)
}

#[track_caller]
/// Returns the number of bytes that will be been written for a given bit length and number
/// of elements that were packed when using the bitpacking functions.
const fn compressed_size(bit_length: usize, num_elements: usize) -> usize {
    assert!(bit_length <= 32, "bit length must be between 0 and 32");
    let quotient = bit_length / 4;
    let remainder = bit_length % 4;
    let remainder_bytes = num_elements.div_ceil(8) * remainder;
    (quotient * num_elements).div_ceil(2) + remainder_bytes
}

/// Split the provided block of 256 elements into two 128 bit elements.
pub(super) fn split_x256(block: &mut [u32; X256]) -> [&mut [u32; X128]; 2] {
    // SAFETY:
    // We know the exact length of the input block and can guarantee
    // the ranges do not overlap.
    let [half1, half2] = unsafe { block.get_disjoint_unchecked_mut([0..X128, X128..X256]) };
    let half1: &mut [u32; X128] = unsafe { half1.try_into().unwrap_unchecked() };
    let half2: &mut [u32; X128] = unsafe { half2.try_into().unwrap_unchecked() };
    [half1, half2]
}

impl CompressibleArray for [u32; X128] {
    type CompressedBuffer = [u8; Self::MAX_OUTPUT_SIZE];
    type InitialValue = u32;
    const MAX_OUTPUT_SIZE: usize = X128 * size_of::<u32>();

    fn compress(n: usize, input: &Self, output: &mut Self::CompressedBuffer) -> CompressionDetails {
        let max_value = input.iter().take(n).fold(0, |a, b| a.max(*b));
        let nbits = (32 - max_value.leading_zeros()) as u8;

        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        if avx512::can_use() {
            // SAFETY: The runtime CPU supports the required features.
            unsafe { avx512::pack_x128::to_nbits(output, nbits, input, n) };
        }

        CompressionDetails {
            bytes_written: compressed_size(nbits as usize, input.len()),
            compressed_bit_length: nbits,
        }
    }

    fn compress_delta(
        initial_value: u32,
        n: usize,
        input: &mut Self,
        output: &mut Self::CompressedBuffer,
    ) -> CompressionDetails {
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        if avx512::can_use() {
            // SAFETY: The runtime CPU supports the required features.
            unsafe { avx512::modifiers::delta_encode_x128(initial_value, input) };
        }

        Self::compress(n, input, output)
    }

    fn compress_delta1(
        initial_value: u32,
        n: usize,
        input: &mut Self,
        output: &mut Self::CompressedBuffer,
    ) -> CompressionDetails {
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        if avx512::can_use() {
            // SAFETY: The runtime CPU supports the required features.
            unsafe { avx512::modifiers::delta1_encode_x128(initial_value, input) };
        }

        Self::compress(n, input, output)
    }

    fn decompress(
        n: usize,
        compressed_bit_length: u8,
        input: &Self::CompressedBuffer,
        output: &mut Self,
    ) -> usize {
        assert!(
            compressed_bit_length <= 32,
            "compressed bitlength must be no more than 32"
        );

        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        if avx512::can_use() {
            // SAFETY: The runtime CPU supports the required features.
            unsafe { avx512::unpack_x128::from_nbits(input, compressed_bit_length, output, n) };
        }

        compressed_size(compressed_bit_length as usize, input.len())
    }

    fn decompress_delta(
        initial_value: u32,
        n: usize,
        compressed_bit_length: u8,
        input: &Self::CompressedBuffer,
        output: &mut Self,
    ) -> usize {
        let bytes_read = Self::decompress(n, compressed_bit_length, input, output);

        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        if avx512::can_use() {
            // SAFETY: The runtime CPU supports the required features.
            unsafe { avx512::modifiers::delta_decode_x128(initial_value, output) };
        }

        bytes_read
    }

    fn decompress_delta1(
        initial_value: u32,
        n: usize,
        compressed_bit_length: u8,
        input: &Self::CompressedBuffer,
        output: &mut Self,
    ) -> usize {
        let bytes_read = Self::decompress(n, compressed_bit_length, input, output);

        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        if avx512::can_use() {
            // SAFETY: The runtime CPU supports the required features.
            unsafe { avx512::modifiers::delta_decode_x128(initial_value, output) };
        }

        bytes_read
    }
}
