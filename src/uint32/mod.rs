//! 32-bit integer bitpacking routines
use crate::X128;
use crate::core::{CompressibleArray, CompressionDetails};

#[cfg(target_endian = "big")]
compile_error!("big endian machines are not supported");

#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
mod avx2;
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
mod avx512;
#[cfg(all(
    any(target_arch = "x86_64", target_arch = "aarch64"),
    any(feature = "sse", feature = "neon")
))]
mod dual_128;
#[cfg(all(
    any(target_arch = "x86_64", target_arch = "aarch64"),
    any(feature = "sse", feature = "neon")
))]
mod polyfill;
mod scalar;

/// The maximum output size of a compressed buffer for a [X128] block, assuming worst case compression.
pub const X128_MAX_OUTPUT_LEN: usize = <[u32; X128] as CompressibleArray>::MAX_OUTPUT_SIZE;

#[track_caller]
/// Returns the number of bytes that will be been written for a given bit length and number
/// of elements that were packed when using the bitpacking functions.
pub const fn max_compressed_size<const BLOCK_SIZE: usize>(bit_length: usize) -> usize {
    const { assert!(BLOCK_SIZE == X128, "BLOCK_SIZE must be either X128 or X256") };
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

    fn decompress(n: usize, compressed_bit_length: u8, input: &[u8], output: &mut Self) -> usize {
        assert!(
            compressed_bit_length <= 32,
            "compressed bitlength must be no more than 32"
        );
        assert!(
            input.len() >= max_compressed_size::<X128>(compressed_bit_length as usize),
            "input buffer is too small/incorrectly padded to safely decompress",
        );

        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        if avx512::can_use() {
            // SAFETY:
            // - The runtime CPU supports the required features.
            // - We have ensured the input buffer is correctly padded so the SIMD reads do
            //   not go out of bounds.
            unsafe {
                avx512::unpack_x128::from_nbits(input.as_ptr(), compressed_bit_length, output, n)
            };
        }

        compressed_size(compressed_bit_length as usize, n)
    }

    fn decompress_delta(
        initial_value: u32,
        n: usize,
        compressed_bit_length: u8,
        input: &[u8],
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
        input: &[u8],
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
    fn test_compress_decompress_diff() {
        let mut data = crate::test_utils::load_sample_u32_doc_id_data_x128();
        let mut out = [0; X128_MAX_OUTPUT_LEN];
        for sample in data.iter_mut() {
            crate::compress_delta1(0, X128, sample, &mut out);
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

    #[test]
    fn test_sequentially_increasing_delta() {
        let mut compressed = [0; X128_MAX_OUTPUT_LEN];
        let mut decompressed = [0; X128];

        let mut values: Vec<u32> = (0..896).collect();
        let mut last_value = 0;
        for chunk in values.chunks_exact_mut(X128) {
            let original = chunk.to_vec();
            let input: &mut [u32; X128] = chunk.try_into().unwrap();

            let details = crate::compress_delta(last_value, X128, input, &mut compressed);
            assert_eq!(details.compressed_bit_length, 1);

            let read = crate::decompress_delta(
                last_value,
                X128,
                details.compressed_bit_length,
                &compressed,
                &mut decompressed,
            );
            assert_eq!(read, details.bytes_written);
            assert_eq!(decompressed.as_slice(), original.as_slice());

            last_value = *original.last().unwrap();
        }
    }

    #[test]
    fn test_sequentially_increasing_delta1() {
        let mut compressed = [0; X128_MAX_OUTPUT_LEN];
        let mut decompressed = [0; X128];

        let mut values: Vec<u32> = (1..897).collect();
        let mut last_value = 0;
        for chunk in values.chunks_exact_mut(X128) {
            let original = chunk.to_vec();
            let input: &mut [u32; X128] = chunk.try_into().unwrap();

            let details = crate::compress_delta1(last_value, X128, input, &mut compressed);
            assert_eq!(details.compressed_bit_length, 0);

            let read = crate::decompress_delta1(
                last_value,
                X128,
                details.compressed_bit_length,
                &compressed,
                &mut decompressed,
            );
            assert_eq!(read, details.bytes_written);
            assert_eq!(decompressed.as_slice(), original.as_slice());

            last_value = *original.last().unwrap();
        }
    }
}
