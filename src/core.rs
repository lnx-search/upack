#[derive(Copy, Clone, Debug)]
/// Information about the compressed block.
pub struct CompressionDetails {
    /// The bit length of the compressed values.
    pub compressed_bit_length: u8,
    /// The number of bytes written to the `output`.
    pub bytes_written: usize,
}

/// An array of values that can be compressed.
pub trait CompressibleArray {
    /// The output array to have the compressed output written to.
    type CompressedBuffer;
    /// The type of the initial value.
    type InitialValue;

    /// The maximum number of bytes that can be written to the output
    const MAX_OUTPUT_SIZE: usize;

    /// Compress the input and write the compressed data to output.
    ///
    /// `n` should be the number of elements to select from the input
    /// to compress.
    fn compress(n: usize, input: &Self, output: &mut Self::CompressedBuffer) -> CompressionDetails;

    /// Compress the input after applying standard Delta encoding
    /// and write the compressed data to output.
    ///
    /// `n` should be the number of elements to select from the input
    /// to compress.
    fn compress_delta(
        initial_value: Self::InitialValue,
        n: usize,
        input: &mut Self,
        output: &mut Self::CompressedBuffer,
    ) -> CompressionDetails;

    /// Compress the input after applying standard Delta-1 encoding
    /// and write the compressed data to output.
    ///
    /// `n` should be the number of elements to select from the input
    /// to compress.
    fn compress_delta1(
        initial_value: Self::InitialValue,
        n: usize,
        input: &mut Self,
        output: &mut Self::CompressedBuffer,
    ) -> CompressionDetails;

    /// Decompress the input and write the recovered values to the output.
    ///
    /// - `n` should be the number of elements that the compressed buffer holds.
    /// - `compressed_bit_length` should be the bit length of the compressed block values
    ///   as reported by the [CompressionDetails] after compressing the block.
    ///
    /// Returns the number of bytes read from the input.
    fn decompress(n: usize, compressed_bit_length: u8, input: &[u8], output: &mut Self) -> usize;

    /// Decompress the input and write the recovered values, reverse the Delta encoding
    /// and write the output.
    ///
    /// - `n` should be the number of elements that the compressed buffer holds.
    /// - `compressed_bit_length` should be the bit length of the compressed block values
    ///   as reported by the [CompressionDetails] after compressing the block.
    ///
    /// Returns the number of bytes read from the input.
    fn decompress_delta(
        initial_value: Self::InitialValue,
        n: usize,
        compressed_bit_length: u8,
        input: &[u8],
        output: &mut Self,
    ) -> usize;

    /// Decompress the input and write the recovered values, reverse the Delta-1 encoding
    /// and write the output.
    ///
    /// - `n` should be the number of elements that the compressed buffer holds.
    /// - `compressed_bit_length` should be the bit length of the compressed block values
    ///   as reported by the [CompressionDetails] after compressing the block.
    ///
    /// Returns the number of bytes read from the input.
    fn decompress_delta1(
        initial_value: Self::InitialValue,
        n: usize,
        compressed_bit_length: u8,
        input: &[u8],
        output: &mut Self,
    ) -> usize;
}
