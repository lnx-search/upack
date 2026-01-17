mod core;
pub mod uint32;

pub use self::core::{CompressibleArray, CompressionDetails};

/// 128 elements
pub const X128: usize = 128;
/// 256 elements
pub const X256: usize = 256;

/// Apply bitpacking compression to the provided input.
pub fn compress<A>(n: usize, input: &A, out: &mut A::CompressedBuffer) -> CompressionDetails
where
    A: CompressibleArray,
{
    A::compress(n, input, out)
}

/// Apply bitpacking compression to the provided input after first
/// applying Delta encoding to the array.
///
/// This requires that the input values are sorted from smallest to largest.
pub fn compress_delta<A>(
    initial_value: A::InitialValue,
    n: usize,
    input: &mut A,
    out: &mut A::CompressedBuffer,
) -> CompressionDetails
where
    A: CompressibleArray,
{
    A::compress_delta(initial_value, n, input, out)
}

/// Apply bitpacking compression to the provided input after first
/// applying Delta-1 encoding to the array.
///
/// This requires that the input values are monotonic, meaning there must be _at least_
/// a gap of `1` between each succeeding value.
pub fn compress_delta1<A>(
    initial_value: A::InitialValue,
    n: usize,
    input: &mut A,
    out: &mut A::CompressedBuffer,
) -> CompressionDetails
where
    A: CompressibleArray,
{
    A::compress_delta1(initial_value, n, input, out)
}

/// Decompress the input block containing the packed values, writing the decompressed
/// values to `out`.
///
/// This requires that the values contained were originally compressed with [compress].
///
/// - `n` should be the number of elements that the compressed buffer holds.
/// - `compressed_bit_length` should be the bit length of the compressed block values
///   as reported by the [CompressionDetails] after compressing the block.
pub fn decompress<A>(n: usize, compressed_bit_length: u8, input: &[u8], out: &mut A) -> usize
where
    A: CompressibleArray,
{
    A::decompress(n, compressed_bit_length, input, out)
}

/// Decompress the input block containing the packed values, reverse the Delta encoding and then
/// writing the decompressed values to `out`.
///
/// This requires that the values contained were originally compressed with [compress_delta].
///
/// - `n` should be the number of elements that the compressed buffer holds.
/// - `compressed_bit_length` should be the bit length of the compressed block values
///   as reported by the [CompressionDetails] after compressing the block.
pub fn decompress_delta<A>(
    initial_value: A::InitialValue,
    n: usize,
    compressed_bit_length: u8,
    input: &[u8],
    out: &mut A,
) -> usize
where
    A: CompressibleArray,
{
    A::decompress_delta(initial_value, n, compressed_bit_length, input, out)
}

/// Decompress the input block containing the packed values, reverse the Delta-1 encoding and then
/// writing the decompressed values to `out`.
///
/// This requires that the values contained were originally compressed with [compress_delta1].
///
/// - `n` should be the number of elements that the compressed buffer holds.
/// - `compressed_bit_length` should be the bit length of the compressed block values
///   as reported by the [CompressionDetails] after compressing the block.
pub fn decompress_delta1<A>(
    initial_value: A::InitialValue,
    n: usize,
    compressed_bit_length: u8,
    input: &[u8],
    out: &mut A,
) -> usize
where
    A: CompressibleArray,
{
    A::decompress_delta1(initial_value, n, compressed_bit_length, input, out)
}

#[cfg(test)]
mod test_utils {
    use crate::X128;

    pub fn load_sample_u32_doc_id_data_x128() -> Vec<[u32; X128]> {
        let raw_data_le = std::fs::read("data/wikipedia-sample-docids.bin")
            .expect("unable to read wikipedia-sample-docids.bin");
        let slice: &[[u32; X128]] = bytemuck::cast_slice(&raw_data_le);
        slice.to_vec()
    }
}
