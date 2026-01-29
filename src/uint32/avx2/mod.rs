use std::hint::black_box;
use crate::uint32::{X128_MAX_OUTPUT_LEN, compressed_size, max_compressed_size, split_block};
use crate::{CompressionDetails, X64, X128};

mod data;
pub mod modifiers;
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
    let mut max = 0;
    let mut last_value = 0;
    for i in 0..X128 {
        max = max.max(block[i] - last_value);
        last_value = block[i];
    }
    let nbits = 32 - max.leading_zeros();

    let [left, right] = split_block(block);
    let offset = max_compressed_size::<X64>(nbits as usize);

    let out = out.as_mut_ptr();
    let block = data::load_u32x64(left);
    unsafe { pack_x64_partial::to_nbits(nbits as usize, out.add(0), block, std::cmp::min(pack_n, X128)) };
    black_box(right);

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
    block: [u32; X128],
    pack_n: usize,
) -> CompressionDetails {
    todo!()
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
    block: [u32; X128],
    pack_n: usize,
) -> CompressionDetails {
    todo!()
}
