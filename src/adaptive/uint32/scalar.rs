use crate::adaptive::uint32::{DELTA_OVERHEAD, X128_MAX_OUTPUT_LEN, compressed_size};
use crate::{CompressionDetails, X128};

/// Pack a block of 128 32-bit integers after applying the adaptive delta algorithm
/// and write the compressed block to `out`.
///
/// # Safety
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
    let details = unsafe { crate::uint32::scalar::pack_x128(out, block, pack_n) };

    CompressionDetails {
        compressed_bit_length: details.compressed_bit_length,
        bytes_written: compressed_size(details.compressed_bit_length as usize, pack_n),
    }
}

/// Unpack a block of 128 32-bit integers from the compressed input after reversing
/// the Adaptive Delta encoding.
///
/// # Safety
/// - `read_n` must be less than or equal to 128.
/// - `input` buffer must be able to hold the _maximum_ possible length of the packed values for
///   a given bit length.
/// - `nbits` must be no greater than `32`.
pub unsafe fn unpack_adaptive_delta_x128(
    nbits: u8,
    last_value: u32,
    input: &[u8],
    block: &mut [u32; X128],
    read_n: usize,
) -> usize {
    let adaptive_delta: u32 = unsafe { std::ptr::read_unaligned(input.as_ptr().cast()) };
    unsafe { crate::uint32::scalar::unpack_x128(nbits, &input[DELTA_OVERHEAD..], block, read_n) };
    decode_adaptive_delta(last_value, adaptive_delta, block);
    compressed_size(nbits as usize, read_n)
}

fn decode_adaptive_delta(mut last_value: u32, adaptive_delta: u32, block: &mut [u32; X128]) -> u32 {
    #[allow(clippy::needless_range_loop)]
    for i in 0..128 {
        last_value = last_value
            .wrapping_add(block[i])
            .wrapping_add(adaptive_delta);
        block[i] = last_value;
    }
    last_value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_adaptive_delta() {
        let mut input = [0; X128];
        decode_adaptive_delta(0, 4, &mut input);
        let expected: [u32; X128] = std::array::from_fn(|i| (i as u32 + 1) * 4);
        assert_eq!(input, expected);

        let mut input = [0; X128];
        decode_adaptive_delta(4, 4, &mut input);
        let expected: [u32; X128] = std::array::from_fn(|i| 4 + (i as u32 + 1) * 4);
        assert_eq!(input, expected);
    }
}
