#[derive(Copy, Clone)]
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
        n: usize,
        input: &Self,
        output: &mut Self::CompressedBuffer,
    ) -> CompressionDetails;

    /// Compress the input after applying standard Delta-1 encoding
    /// and write the compressed data to output.
    ///
    /// `n` should be the number of elements to select from the input
    /// to compress.
    fn compress_delta1(
        n: usize,
        input: &Self,
        output: &mut Self::CompressedBuffer,
    ) -> CompressionDetails;

    /// Decompress the input and write the recovered values to the output.
    ///
    /// - `n` should be the number of elements that the compressed buffer holds.
    /// - `compressed_bit_length` should be the bit length of the compressed block values
    ///   as reported by the [CompressionDetails] after compressing the block.
    fn decompress(
        n: usize,
        compressed_bit_length: u8,
        input: &Self::CompressedBuffer,
        out: &mut Self,
    );

    /// Decompress the input and write the recovered values, reverse the Delta encoding
    /// and write the output.
    ///
    /// - `n` should be the number of elements that the compressed buffer holds.
    /// - `compressed_bit_length` should be the bit length of the compressed block values
    ///   as reported by the [CompressionDetails] after compressing the block.
    fn decompress_delta(
        n: usize,
        compressed_bit_length: u8,
        input: &Self::CompressedBuffer,
        out: &mut Self,
    );

    /// Decompress the input and write the recovered values, reverse the Delta-1 encoding
    /// and write the output.
    ///
    /// - `n` should be the number of elements that the compressed buffer holds.
    /// - `compressed_bit_length` should be the bit length of the compressed block values
    ///   as reported by the [CompressionDetails] after compressing the block.
    fn decompress_delta1(
        n: usize,
        compressed_bit_length: u8,
        input: &Self::CompressedBuffer,
        out: &mut Self,
    );
}
