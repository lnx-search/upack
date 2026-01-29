use crate::uint32::{X128_MAX_OUTPUT_LEN, compressed_size};
use crate::{CompressionDetails, X64, X128};

mod data;
pub mod modifiers;
mod pack_x128;
pub mod pack_x64_full;
pub mod pack_x64_partial;
pub mod unpack_x64_full;
pub mod unpack_x64_partial;
mod utils;

#[inline]
/// Returns `true` if the runtime CPU can safely execute the AVX2 backed implementation.
pub fn can_use() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
}

#[target_feature(enable = "avx2", enable = "popcnt")]
/// Pack a block of 128 32-bit integers and write the compressed block to `out`.
///
/// # Safety
/// - The runtime CPU must support the `avx2` instructions.
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

#[target_feature(enable = "avx2")]
/// Pack a block of 128 32-bit integers and write the compressed block to `out` after
/// applying Delta encoding.
///
/// # Safety
/// - The runtime CPU must support the `avx2` instructions.
/// - `n` must be less than or equal to 128.
pub unsafe fn pack_delta_x128(
    out: &mut [u8; X128_MAX_OUTPUT_LEN],
    block: &mut [u32; X128],
    pack_n: usize,
    mut last_value: u32,
) -> CompressionDetails {
    for v in block.iter_mut() {
        let value = *v;
        *v = value - last_value;
        last_value = value;
    }

    unsafe { pack_x128(out, block, pack_n) }
}

#[target_feature(enable = "avx2")]
/// Pack a block of 128 32-bit integers and write the compressed block to `out` after
/// applying Delta-1 encoding.
///
/// # Safety
/// - The runtime CPU must support the `avx2` instructions.
/// - `n` must be less than or equal to 128.
pub unsafe fn pack_delta1_x128(
    out: &mut [u8; X128_MAX_OUTPUT_LEN],
    block: &mut [u32; X128],
    pack_n: usize,
    mut last_value: u32,
) -> CompressionDetails {
    for v in block.iter_mut() {
        let value = *v;
        *v = (value - last_value) - 1;
        last_value = value;
    }

    unsafe { pack_x128(out, block, pack_n) }
}
