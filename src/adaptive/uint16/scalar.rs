use crate::adaptive::uint16::{DELTA_OVERHEAD, X128_MAX_OUTPUT_LEN, compressed_size};
use crate::{CompressionDetails, X128};

/// Pack a block of 128 16-bit integers after applying the adaptive delta algorithm
/// and write the compressed block to `out`.
///
/// # Safety
/// - `pack_n` must be less than or equal to 128.
pub unsafe fn pack_adaptive_delta_x128(
    mut last_value: u16,
    out: &mut [u8; X128_MAX_OUTPUT_LEN],
    block: &mut [u16; X128],
    pack_n: usize,
) -> CompressionDetails {
    let adaptive_delta = super::util::adaptive_delta_encode(&mut last_value, block, pack_n);

    unsafe { std::ptr::write_unaligned(out.as_mut_ptr().cast(), adaptive_delta) };
    let out = super::select_compression_buffer(out);
    let details = unsafe { crate::uint16::scalar::pack_x128(out, block, pack_n) };

    CompressionDetails {
        compressed_bit_length: details.compressed_bit_length,
        bytes_written: compressed_size(details.compressed_bit_length as usize, pack_n),
    }
}

/// Unpack a block of 128 16-bit integers from the compressed input after reversing
/// the Adaptive Delta encoding.
///
/// # Safety
/// - `read_n` must be less than or equal to 128.
/// - `input` buffer must be able to hold the _maximum_ possible length of the packed values for
///   a given bit length.
/// - `nbits` must be no greater than `32`.
pub unsafe fn unpack_adaptive_delta_x128(
    nbits: u8,
    last_value: u16,
    input: &[u8],
    block: &mut [u16; X128],
    read_n: usize,
) -> usize {
    let adaptive_delta: u16 = unsafe { std::ptr::read_unaligned(input.as_ptr().cast()) };
    unsafe { crate::uint16::scalar::unpack_x128(nbits, &input[DELTA_OVERHEAD..], block, read_n) };
    decode_adaptive_delta(last_value, adaptive_delta, block);
    compressed_size(nbits as usize, read_n)
}

fn decode_adaptive_delta(mut last_value: u16, adaptive_delta: u16, block: &mut [u16; X128]) -> u16 {
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
        let expected: [u16; X128] = std::array::from_fn(|i| (i as u16 + 1) * 4);
        assert_eq!(input, expected);

        let mut input = [0; X128];
        decode_adaptive_delta(4, 4, &mut input);
        let expected: [u16; X128] = std::array::from_fn(|i| 4 + (i as u16 + 1) * 4);
        assert_eq!(input, expected);
    }
}
