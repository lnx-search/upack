use super::data::*;
use super::{unpack_x64_full, unpack_x64_partial};
use crate::uint32::{max_compressed_size, split_block_mut};
use crate::{X64, X128};

#[inline]
/// Bitpack the provided block of integers to `nbits` bit length  elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X128>(nbits)` bytes to.
/// - `nbits` must be between 0 and 32.
/// - `read_n` must be no greater than 128.
pub unsafe fn from_nbits(nbits: usize, input: *const u8, out: &mut [u32; X128], read_n: usize) {
    debug_assert!(nbits <= 32, "BUG: invalid nbits provided: {nbits}");
    debug_assert!(read_n <= X128, "BUG: invalid read_n provided: {read_n}");
    #[allow(clippy::type_complexity)]
    const LUT: [unsafe fn(out: *const u8, &mut [u32; X128], usize); 33] = [
        from_u0, from_u1, from_u2, from_u3, from_u4, from_u5, from_u6, from_u7, from_u8, from_u9,
        from_u10, from_u11, from_u12, from_u13, from_u14, from_u15, from_u16, from_u17, from_u18,
        from_u19, from_u20, from_u21, from_u22, from_u23, from_u24, from_u25, from_u26, from_u27,
        from_u28, from_u29, from_u30, from_u31, from_u32,
    ];
    let func = unsafe { LUT.get_unchecked(nbits) };
    unsafe { func(input, out, read_n) };
}

#[inline]
/// Bitpack the provided block of integers to `nbits` bit length elements which have
/// been delta-encoded.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X128>(nbits)` bytes to.
/// - `nbits` must be between 0 and 32.
/// - `read_n` must be no greater than 128.
pub unsafe fn from_nbits_delta(
    nbits: usize,
    last_value: u32,
    input: *const u8,
    out: &mut [u32; X128],
    read_n: usize,
) {
    unsafe { from_nbits(nbits, input, out, read_n) };
    decode_delta(last_value, out);
}

#[inline]
/// Bitpack the provided block of integers to `nbits` bit length elements which have
/// been delta-1-encoded.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X128>(nbits)` bytes to.
/// - `nbits` must be between 0 and 32.
/// - `read_n` must be no greater than 128.
pub unsafe fn from_nbits_delta1(
    nbits: usize,
    last_value: u32,
    input: *const u8,
    out: &mut [u32; X128],
    read_n: usize,
) {
    unsafe { from_nbits(nbits, input, out, read_n) };
    decode_delta1(last_value, out);
}

unsafe fn from_u0(_input: *const u8, out: &mut [u32; X128], _read_n: usize) {
    out.fill(0);
}

macro_rules! define_x128_unpacker {
    ($func_name:ident, $bit_length:expr) => {
        unsafe fn $func_name(input: *const u8, out: &mut [u32; X128], read_n: usize) {
            let [left, right] = split_block_mut(out);

            if read_n <= 64 {
                let unpacked = unsafe { unpack_x64_partial::$func_name(input.add(0), read_n) };
                store_u32x64(left, unpacked);
            } else if read_n < 128 {
                let unpacked = unsafe { unpack_x64_full::$func_name(input.add(0)) };
                store_u32x64(left, unpacked);
                let unpacked = unsafe {
                    unpack_x64_partial::$func_name(
                        input.add(max_compressed_size::<X64>($bit_length)),
                        read_n - X64,
                    )
                };
                store_u32x64(right, unpacked);
            } else {
                let unpacked = unsafe { unpack_x64_full::$func_name(input.add(0)) };
                store_u32x64(left, unpacked);
                let unpacked = unsafe {
                    unpack_x64_full::$func_name(input.add(max_compressed_size::<X64>($bit_length)))
                };
                store_u32x64(right, unpacked);
            }
        }
    };
}

define_x128_unpacker!(from_u1, 1);
define_x128_unpacker!(from_u2, 2);
define_x128_unpacker!(from_u3, 3);
define_x128_unpacker!(from_u4, 4);
define_x128_unpacker!(from_u5, 5);
define_x128_unpacker!(from_u6, 6);
define_x128_unpacker!(from_u7, 7);
define_x128_unpacker!(from_u8, 8);
define_x128_unpacker!(from_u9, 9);
define_x128_unpacker!(from_u10, 10);
define_x128_unpacker!(from_u11, 11);
define_x128_unpacker!(from_u12, 12);
define_x128_unpacker!(from_u13, 13);
define_x128_unpacker!(from_u14, 14);
define_x128_unpacker!(from_u15, 15);
define_x128_unpacker!(from_u16, 16);
define_x128_unpacker!(from_u17, 17);
define_x128_unpacker!(from_u18, 18);
define_x128_unpacker!(from_u19, 19);
define_x128_unpacker!(from_u20, 20);
define_x128_unpacker!(from_u21, 21);
define_x128_unpacker!(from_u22, 22);
define_x128_unpacker!(from_u23, 23);
define_x128_unpacker!(from_u24, 24);
define_x128_unpacker!(from_u25, 25);
define_x128_unpacker!(from_u26, 26);
define_x128_unpacker!(from_u27, 27);
define_x128_unpacker!(from_u28, 28);
define_x128_unpacker!(from_u29, 29);
define_x128_unpacker!(from_u30, 30);
define_x128_unpacker!(from_u31, 31);
define_x128_unpacker!(from_u32, 32);

fn decode_delta(mut last_value: u32, block: &mut [u32; X128]) -> u32 {
    #[allow(clippy::needless_range_loop)]
    for i in 0..128 {
        last_value = last_value.wrapping_add(block[i]);
        block[i] = last_value;
    }
    last_value
}

fn decode_delta1(mut last_value: u32, block: &mut [u32; X128]) -> u32 {
    #[allow(clippy::needless_range_loop)]
    for i in 0..128 {
        last_value = last_value.wrapping_add(block[i]).wrapping_add(1);
        block[i] = last_value;
    }
    last_value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_delta_zero_starting_value() {
        let expected_values: [u32; X128] = std::array::from_fn(|i| i as u32);
        let mut block = [1; X128];
        block[0] = 0;
        decode_delta(0, &mut block);
        assert_eq!(block, expected_values);
    }

    #[test]
    fn test_decode_delta_starting_value() {
        let expected_values: [u32; X128] = std::array::from_fn(|i| 4 + i as u32);
        let mut block = [1; X128];
        block[0] = 0;
        decode_delta(4, &mut block);
        assert_eq!(block, expected_values);
    }

    #[test]
    fn test_decode_delta1_zero_starting_value() {
        let expected_values: [u32; X128] = std::array::from_fn(|i| i as u32 + 1);
        let mut block = [0; X128];
        decode_delta1(0, &mut block);
        assert_eq!(block, expected_values);
    }

    #[test]
    fn test_decode_delta1_starting_value() {
        let expected_values: [u32; X128] = std::array::from_fn(|i| i as u32 + 5);
        let mut block = [0; X128];
        decode_delta1(4, &mut block);
        assert_eq!(block, expected_values);
    }

    #[test]
    fn test_v1_layout_regression() {
        let tester = crate::uint32::test_util::load_uint32_regression_layout();

        let mut output_buffer = [0u32; X128];
        for (len, bit_len, expected_output, input) in tester.iter_tests() {
            unsafe { from_nbits(bit_len as usize, input.as_ptr(), &mut output_buffer, len) };

            let produced_buffer = &output_buffer[..len];
            assert_eq!(
                produced_buffer,
                &expected_output[..len],
                "regression test failed, outputs do not match, length:{len} bit_len:{bit_len}"
            )
        }
    }
}
