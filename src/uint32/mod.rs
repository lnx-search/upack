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
pub const fn compressed_size(bit_length: usize, num_elements: usize) -> usize {
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
            bytes_written: compressed_size(nbits as usize, n),
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

        compressed_size(compressed_bit_length as usize, n)
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
            unsafe { avx512::modifiers::delta1_decode_x128(initial_value, output) };
        }

        bytes_read
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rstest::rstest]
    #[case(32, 128)]
    #[case(0, 0)]
    #[should_panic(expected = "bit length must be between 0 and 32")]
    #[case(33, 128)]
    fn test_bitpacking_output_length_validation(
        #[case] bit_length: usize,
        #[case] num_elements: usize,
    ) {
        compressed_size(bit_length, num_elements);
    }

    #[test]
    fn test_compress_and_decompress_one_exception() {
        let mut values = [8; 128];
        values[77] = 90876324;
        let original_values = values;

        let mut compressed = [0; X128_MAX_OUTPUT_LEN];
        let details = crate::compress(X128, &mut values, &mut compressed);

        let mut decompressed = [0; X128];
        let bytes_read = crate::decompress(
            X128,
            details.compressed_bit_length,
            &compressed,
            &mut decompressed,
        );

        assert_eq!(details.bytes_written, bytes_read);
        assert_eq!(decompressed, original_values);
    }

    #[test]
    fn test_compress_and_decompress_two_exception() {
        let mut values = [8; 128];
        values[69] = 3252352;
        values[125] = 874632124;
        let original_values = values;

        let mut compressed = [0; X128_MAX_OUTPUT_LEN];
        let details = crate::compress(X128, &mut values, &mut compressed);

        let mut decompressed = [0; X128];
        let bytes_read = crate::decompress(
            X128,
            details.compressed_bit_length,
            &compressed,
            &mut decompressed,
        );

        assert_eq!(details.bytes_written, bytes_read);
        assert_eq!(decompressed, original_values);
    }

    #[test]
    fn test_compress_and_decompress_many_exception() {
        let mut values = [8; 128];
        values[0] = 874632124;
        values[1] = 252151;
        values[7] = 555;
        values[69] = 3234;
        values[100] = 522332525;
        values[125] = 874632124;
        let original_values = values;

        let mut compressed = [0; X128_MAX_OUTPUT_LEN];
        let details = crate::compress(X128, &mut values, &mut compressed);

        let mut decompressed = [0; X128];
        let bytes_read = crate::decompress(
            X128,
            details.compressed_bit_length,
            &compressed,
            &mut decompressed,
        );

        assert_eq!(details.bytes_written, bytes_read);
        assert_eq!(decompressed, original_values);
    }

    #[rstest::rstest]
    #[case::all_zeroes(0)]
    #[case::all_max(u32::MAX)]
    fn test_compress_and_decompress_edge_cases(#[case] value: u32) {
        let mut values = [value; 128];
        let original_values = values;

        let mut compressed = [0; X128_MAX_OUTPUT_LEN];
        let details = crate::compress(X128, &mut values, &mut compressed);

        let mut decompressed = [0; X128];
        let bytes_read = crate::decompress(
            X128,
            details.compressed_bit_length,
            &compressed,
            &mut decompressed,
        );

        assert_eq!(details.bytes_written, bytes_read);
        assert_eq!(decompressed, original_values);
    }

    #[test]
    fn test_compress_decompress_real_data_sample() {
        let sample_data = crate::test_utils::load_sample_u32_doc_id_data_x128();

        for select_n in 1..X128 {
            let mut data = sample_data.clone();
            let mut out = [0; X128_MAX_OUTPUT_LEN];
            let mut decompressed = [0; X128];
            for _ in 0..10 {
                for sample in data.iter_mut() {
                    let original = *sample;
                    let details = crate::compress(select_n, sample, &mut out);
                    let read = crate::decompress(
                        select_n,
                        details.compressed_bit_length,
                        &out,
                        &mut decompressed,
                    );
                    assert_eq!(details.bytes_written, read);
                    assert_eq!(
                        original[..select_n],
                        decompressed[..select_n],
                        "select_n:{select_n}, details:{details:?} read:{read}",
                    );
                }
            }
        }
    }

    #[test]
    fn test_compress_decompress_base_fuzz() {
        let mut data = crate::test_utils::load_sample_u32_doc_id_data_x128();
        let mut out = [0; X128_MAX_OUTPUT_LEN];
        for _ in 0..10 {
            for sample in data.iter_mut() {
                crate::compress(X128, sample, &mut out);
            }
        }
    }

    #[test]
    fn test_compress_decompress_delta_fuzz() {
        let mut data = crate::test_utils::load_sample_u32_doc_id_data_x128();
        let mut out = [0; X128_MAX_OUTPUT_LEN];
        for _ in 0..10 {
            for sample in data.iter_mut() {
                crate::compress_delta(0, X128, sample, &mut out);
            }
        }
    }

    #[test]
    fn test_compress_decompress_delta1_fuzz() {
        let mut data = crate::test_utils::load_sample_u32_doc_id_data_x128();
        let mut out = [0; X128_MAX_OUTPUT_LEN];
        for _ in 0..10 {
            for sample in data.iter_mut() {
                crate::compress_delta1(0, X128, sample, &mut out);
            }
        }
    }
}
