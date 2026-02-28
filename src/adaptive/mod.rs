//! Adaptive Delta Encoding
//!
//! Delta encode blocks of sequentially increasing integers and take out the common value from
//! the resulting deltas.
//!
//! This algorithm behaves exactly the same as more standard `delta-1` encoding but instead of
//! subtracting `1` from the deltas, we subtract the minimum delta value.
//!
//! Depending on the data this can lead to significantly smaller output blocks for relatively
//! minimal compression and decompression overhead.
//!
//! However, unlike the standard `compress`, `compress_delta` and `compress_delta1` functions,
//! the adaptive delta encoding requires adding an integer to be stored at the
//! head of the output block.
//!
//! As a result, you must use the `adaptive_max_compressed_size()` and `adaptive_compressed_size`
//! functions from _this_ module rather than the standard variants.

use crate::CompressionDetails;

pub mod uint16;
pub mod uint32;

pub trait AdaptiveCompressibleArray {
    /// The output array to have the compressed output written to.
    type CompressedBuffer;
    /// The type of the initial value.
    type InitialValue;

    /// The maximum number of bytes that can be written to the output
    const MAX_OUTPUT_SIZE: usize;

    /// Apply bitpacking compression to the provided input after first
    /// applying Adaptive Delta encoding to the array.
    ///
    /// This requires that the input values are sequentially increasing.
    fn compress_adaptive_delta(
        initial_value: Self::InitialValue,
        n: usize,
        input: &mut Self,
        out: &mut Self::CompressedBuffer,
    ) -> CompressionDetails;

    /// Decompress the input block containing the packed values, reverse the Adaptive Delta encoding
    /// and then write the decompressed values to `out`.
    ///
    /// This requires that the values contained were originally compressed
    /// with [compress_adaptive_delta](AdaptiveCompressibleArray::compress_adaptive_delta).
    ///
    /// - `n` should be the number of elements that the compressed buffer holds.
    /// - `compressed_bit_length` should be the bit length of the compressed block values
    ///   as reported by the [CompressionDetails] after compressing the block.
    ///
    /// Returns the number of bytes read from the `input`.
    fn decompress_adaptive_delta(
        initial_value: Self::InitialValue,
        n: usize,
        compressed_bit_length: u8,
        input: &[u8],
        out: &mut Self,
    ) -> usize;
}
