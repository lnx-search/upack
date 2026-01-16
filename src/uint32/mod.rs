//! 32-bit integer bitpacking routines
use crate::{X128, X256};

#[doc(hidden)]
pub mod avx512;

/// The maximum output size of a compressed buffer for a [X128] block, assuming worst case compression.
pub const X128_MAX_OUTPUT_LEN: usize = X128 * size_of::<u32>();
/// The maximum output size of a compressed buffer for a [X256] block, assuming worst case compression.
pub const X256_MAX_OUTPUT_LEN: usize = X256 * size_of::<u32>();

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
