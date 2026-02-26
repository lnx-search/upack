use super::data::load_u16x64;
use super::{pack_x64_full, pack_x64_partial};
use crate::uint16::{max_compressed_size, split_block};
use crate::{X64, X128};

#[inline]
#[target_feature(enable = "neon")]
/// Bitpack the provided block of integers to `nbits` bit length  elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X128>(nbits)` bytes to.
/// - `nbits` must be between 0 and 16.
/// - `pack_n` must be no greater than 128.
pub unsafe fn to_nbits(nbits: usize, out: *mut u8, block: &[u16; X128], pack_n: usize) {
    debug_assert!(nbits <= 16, "BUG: invalid nbits provided: {nbits}");
    debug_assert!(pack_n <= X128, "BUG: invalid pack_n provided: {pack_n}");
    #[allow(clippy::type_complexity)]
    const LUT: [unsafe fn(out: *mut u8, &[u16; X128], usize); 17] = [
        to_u0, to_u1, to_u2, to_u3, to_u4, to_u5, to_u6, to_u7, to_u8, to_u9, to_u10, to_u11,
        to_u12, to_u13, to_u14, to_u15, to_u16,
    ];
    let func = unsafe { LUT.get_unchecked(nbits) };
    unsafe { func(out, block, pack_n) };
}

#[target_feature(enable = "neon")]
unsafe fn to_u0(_out: *mut u8, _block: &[u16; X128], _pack_n: usize) {}

macro_rules! define_x128_packer {
    ($func_name:ident, $bit_length:expr) => {
        #[target_feature(enable = "neon")]
        unsafe fn $func_name(out: *mut u8, block: &[u16; X128], pack_n: usize) {
            let [left, right] = split_block(block);

            let left = load_u16x64(left);
            if pack_n <= 64 {
                unsafe { pack_x64_partial::$func_name(out.add(0), left, pack_n) };
            } else if pack_n < 128 {
                unsafe { pack_x64_full::$func_name(out.add(0), left) };
                let right = load_u16x64(right);
                unsafe {
                    pack_x64_partial::$func_name(
                        out.add(max_compressed_size::<X64>($bit_length)),
                        right,
                        pack_n - X64,
                    )
                }
            } else {
                unsafe { pack_x64_full::$func_name(out.add(0), left) };
                let right = load_u16x64(right);
                unsafe {
                    pack_x64_full::$func_name(
                        out.add(max_compressed_size::<X64>($bit_length)),
                        right,
                    )
                }
            }
        }
    };
}

define_x128_packer!(to_u1, 1);
define_x128_packer!(to_u2, 2);
define_x128_packer!(to_u3, 3);
define_x128_packer!(to_u4, 4);
define_x128_packer!(to_u5, 5);
define_x128_packer!(to_u6, 6);
define_x128_packer!(to_u7, 7);
define_x128_packer!(to_u8, 8);
define_x128_packer!(to_u9, 9);
define_x128_packer!(to_u10, 10);
define_x128_packer!(to_u11, 11);
define_x128_packer!(to_u12, 12);
define_x128_packer!(to_u13, 13);
define_x128_packer!(to_u14, 14);
define_x128_packer!(to_u15, 15);
define_x128_packer!(to_u16, 16);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uint16::X128_MAX_OUTPUT_LEN;

    #[test]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_v1_layout_regression() {
        let tester = crate::uint16::test_util::load_uint16_regression_layout();

        let mut output_buffer = [0; X128_MAX_OUTPUT_LEN];
        for (len, bit_len, input, expected_output) in tester.iter_tests() {
            unsafe { to_nbits(bit_len as usize, output_buffer.as_mut_ptr(), input, len) };

            let produced_buffer = &output_buffer[..expected_output.len()];
            assert_eq!(
                produced_buffer, expected_output,
                "regression test failed, outputs do not match, length:{len} bit_len:{bit_len}"
            )
        }
    }
}
