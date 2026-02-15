use crate::uint32::{X128_MAX_OUTPUT_LEN, compressed_size};
use crate::{CompressionDetails, X128};

pub(super) mod data;
mod pack_x128;
pub mod pack_x64_full;
pub mod pack_x64_partial;
mod polyfill;
mod unpack_x128;
mod unpack_x64_full;
mod unpack_x64_partial;
mod util;

/// Pack a block of 128 32-bit integers and write the compressed block to `out`.
///
/// # Safety
/// - `pack_n` must be less than or equal to 128.
pub unsafe fn pack_x128(
    out: &mut [u8; X128_MAX_OUTPUT_LEN],
    block: &[u32; X128],
    pack_n: usize,
) -> CompressionDetails {
    let max = block.iter().fold(0, |a, b| a.max(*b));
    let nbits = 32 - max.leading_zeros();

    unsafe { pack_x128::to_nbits(nbits as usize, out.as_mut_ptr(), block, pack_n) };

    CompressionDetails {
        compressed_bit_length: nbits as u8,
        bytes_written: compressed_size(nbits as usize, pack_n),
    }
}

/// Pack a block of 128 32-bit integers and write the compressed block to `out` after
/// applying Delta encoding.
///
/// # Safety
/// - `pack_n` must be less than or equal to 128.
pub unsafe fn pack_delta_x128(
    mut last_value: u32,
    out: &mut [u8; X128_MAX_OUTPUT_LEN],
    block: &mut [u32; X128],
    pack_n: usize,
) -> CompressionDetails {
    for v in block.iter_mut() {
        let value = *v;
        *v = value.wrapping_sub(last_value);
        last_value = value;
    }

    unsafe { pack_x128(out, block, pack_n) }
}

/// Pack a block of 128 32-bit integers and write the compressed block to `out` after
/// applying Delta-1 encoding.
///
/// # Safety
/// - `pack_n` must be less than or equal to 128.
pub unsafe fn pack_delta1_x128(
    mut last_value: u32,
    out: &mut [u8; X128_MAX_OUTPUT_LEN],
    block: &mut [u32; X128],
    pack_n: usize,
) -> CompressionDetails {
    for v in block.iter_mut() {
        let value = *v;
        *v = value.wrapping_sub(last_value).wrapping_sub(1);
        last_value = value;
    }

    unsafe { pack_x128(out, block, pack_n) }
}

/// Pack a block of 128 32-bit integers and write the compressed block to `out`.
///
/// # Safety
/// - `read_n` must be less than or equal to 128.
/// - `input` buffer must be able to hold the _maximum_ possible length of the packed values for
///   a given bit length.
/// - `nbits` must be no greater than `32`.
pub unsafe fn unpack_x128(
    nbits: u8,
    input: &[u8],
    block: &mut [u32; X128],
    read_n: usize,
) -> usize {
    unsafe { unpack_x128::from_nbits(nbits as usize, input.as_ptr(), block, read_n) };
    compressed_size(nbits as usize, read_n)
}

/// Pack a block of 128 32-bit integers and write the compressed block to `out` after
/// applying Delta encoding.
///
/// # Safety
/// - `read_n` must be less than or equal to 128.
/// - `input` buffer must be able to hold the _maximum_ possible length of the packed values for
///   a given bit length.
/// - `nbits` must be no greater than `32`.
pub unsafe fn unpack_delta_x128(
    nbits: u8,
    last_value: u32,
    input: &[u8],
    block: &mut [u32; X128],
    read_n: usize,
) -> usize {
    unsafe {
        unpack_x128::from_nbits_delta(nbits as usize, last_value, input.as_ptr(), block, read_n)
    };
    compressed_size(nbits as usize, read_n)
}

/// Pack a block of 128 32-bit integers and write the compressed block to `out` after
/// applying Delta-1 encoding.
///
/// # Safety
/// - `read_n` must be less than or equal to 128.
/// - `input` buffer must be able to hold the _maximum_ possible length of the packed values for
///   a given bit length.
/// - `nbits` must be no greater than `32`.
pub unsafe fn unpack_delta1_x128(
    nbits: u8,
    last_value: u32,
    input: &[u8],
    block: &mut [u32; X128],
    read_n: usize,
) -> usize {
    unsafe {
        unpack_x128::from_nbits_delta1(nbits as usize, last_value, input.as_ptr(), block, read_n)
    };
    compressed_size(nbits as usize, read_n)
}
