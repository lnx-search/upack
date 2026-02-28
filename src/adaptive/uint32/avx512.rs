use crate::adaptive::uint32::X128_MAX_OUTPUT_LEN;
use crate::{CompressionDetails, X128};

#[inline]
/// Returns `true` if the runtime CPU can safely execute the AVX512 backed implementation.
pub fn can_use() -> bool {
    std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Pack a block of 128 32-bit integers after applying the adaptive delta algorithm
/// and write the compressed block to `out`.
///
/// # Safety
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
/// - `pack_n` must be less than or equal to 128.
pub unsafe fn pack_adaptive_delta_x128(
    mut last_value: u32,
    out: &mut [u8; X128_MAX_OUTPUT_LEN],
    block: &mut [u32; X128],
    pack_n: usize,
) -> CompressionDetails {
    let mut min_delta = u32::MAX;
    for v in block.iter_mut().take(pack_n) {
        let value = *v;
        *v = value.wrapping_sub(last_value);
        min_delta = min_delta.min(*v);
        last_value = value;
    }

    for delta in block.iter_mut() {
        *delta -= min_delta;
    }

    unsafe { std::ptr::write_unaligned(out.as_mut_ptr().cast(), min_delta) };
    let out = super::select_compression_buffer(out);
    unsafe { crate::uint32::avx512::pack_x128(out, block, pack_n) }
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
/// Unpack a block of 128 32-bit integers from the compressed input after reversing
/// the Adaptive Delta encoding.
///
/// # Safety
/// - The runtime CPU must support the `avx512f` and `avx512bw` instructions.
/// - `read_n` must be less than or equal to 128.
/// - `input` buffer must be able to hold the _maximum_ possible length of the packed values for
///   a given bit length.
/// - `nbits` must be no greater than `32`.
pub unsafe fn unpack_adaptive_delta_x128(
    _nbits: u8,
    _last_value: u32,
    _input: &[u8],
    _block: &mut [u32; X128],
    _read_n: usize,
) -> usize {
    todo!()
}
