//! 32-bit integer bitpacking routines
use crate::core::{CompressibleArray, CompressionDetails};
use crate::{X64, X128};

#[cfg(target_endian = "big")]
compile_error!("big endian machines are not supported");

#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
pub mod avx2;
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub mod avx512;
mod scalar;
#[cfg(test)]
mod test_util;

/// The maximum output size of a compressed buffer for a [X128] block, assuming worst case compression.
pub const X128_MAX_OUTPUT_LEN: usize = <[u32; X128] as CompressibleArray>::MAX_OUTPUT_SIZE;

#[track_caller]
/// Returns the number of bytes that will be been written for a given bit length and number
/// of elements that were packed when using the bitpacking functions.
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
/// of elements that were packed when using the bitpacking functions.
pub const fn compressed_size(bit_length: usize, num_elements: usize) -> usize {
    assert!(bit_length <= 32, "bit length must be between 0 and 32");
    if num_elements > 64 {
        block_bytes(bit_length, 64) + block_bytes(bit_length, num_elements - 64)
    } else {
        block_bytes(bit_length, num_elements)
    }
}

const fn block_bytes(bit_length: usize, num_elements: usize) -> usize {
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
        assert!(n <= X128, "provided n is is greater than 128",);

        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        if avx512::can_use() {
            return unsafe { avx512::pack_x128(output, input, n) };
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
        if avx2::can_use() {
            return unsafe { avx2::pack_x128(output, input, n) };
        }

        todo!()
    }

    fn compress_delta(
        initial_value: u32,
        n: usize,
        input: &mut Self,
        output: &mut Self::CompressedBuffer,
    ) -> CompressionDetails {
        assert!(n <= X128, "provided n is is greater than 128",);

        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        if avx512::can_use() {
            return unsafe { avx512::pack_delta_x128(initial_value, output, input, n) };
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
        if avx2::can_use() {
            return unsafe { avx2::pack_delta_x128(initial_value, output, input, n) };
        }

        todo!()
    }

    fn compress_delta1(
        initial_value: u32,
        n: usize,
        input: &mut Self,
        output: &mut Self::CompressedBuffer,
    ) -> CompressionDetails {
        assert!(n <= X128, "provided n is is greater than 128",);

        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        if avx512::can_use() {
            return unsafe { avx512::pack_delta1_x128(initial_value, output, input, n) };
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
        if avx2::can_use() {
            return unsafe { avx2::pack_delta1_x128(initial_value, output, input, n) };
        }

        todo!()
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
        assert!(n <= X128, "provided n is is greater than 128",);

        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        if avx512::can_use() {
            return unsafe { avx512::unpack_x128(compressed_bit_length, input, output, n) };
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
        if avx2::can_use() {
            return unsafe { avx2::unpack_x128(compressed_bit_length, input, output, n) };
        }

        todo!()
    }

    fn decompress_delta(
        initial_value: u32,
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
            "input buffer is too small/incorrectly padded to safely decompress",
        );
        assert!(n <= X128, "provided n is is greater than 128",);

        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        if avx512::can_use() {
            return unsafe {
                avx512::unpack_delta_x128(compressed_bit_length, initial_value, input, output, n)
            };
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
        if avx2::can_use() {
            return unsafe {
                avx2::unpack_delta_x128(compressed_bit_length, initial_value, input, output, n)
            };
        }

        todo!()
    }

    fn decompress_delta1(
        initial_value: u32,
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
            "input buffer is too small/incorrectly padded to safely decompress",
        );
        assert!(n <= X128, "provided n is is greater than 128",);

        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        if avx512::can_use() {
            return unsafe {
                avx512::unpack_delta1_x128(compressed_bit_length, initial_value, input, output, n)
            };
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
        if avx2::can_use() {
            return unsafe {
                avx2::unpack_delta1_x128(compressed_bit_length, initial_value, input, output, n)
            };
        }

        todo!()
    }
}

#[inline]
pub(super) fn split_block(block: &[u32; X128]) -> [&[u32; X64]; 2] {
    crate::util::split_slice::<_, X128, X64>(block)
}

#[inline]
pub(super) fn split_block_mut(block: &mut [u32; X128]) -> [&mut [u32; X64]; 2] {
    crate::util::split_slice_mut::<_, X128, X64>(block)
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
