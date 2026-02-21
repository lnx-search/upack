use super::data::load_u32x64;
use super::{pack_x64_full, pack_x64_partial};
use crate::uint32::{max_compressed_size, split_block};
use crate::{X64, X128};

#[inline]
/// Bitpack the provided block of integers to `nbits` bit length  elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X128>(nbits)` bytes to.
/// - `nbits` must be between 0 and 32.
/// - `pack_n` must be no greater than 128.
pub unsafe fn to_nbits(nbits: usize, out: *mut u8, block: &[u32; X128], pack_n: usize) {
    debug_assert!(nbits <= 32, "BUG: invalid nbits provided: {nbits}");
    debug_assert!(pack_n <= X128, "BUG: invalid pack_n provided: {pack_n}");
    #[allow(clippy::type_complexity)]
    const LUT: [unsafe fn(out: *mut u8, &[u32; X128], usize); 33] = [
        to_u0, to_u1, to_u2, to_u3, to_u4, to_u5, to_u6, to_u7, to_u8, to_u9, to_u10, to_u11,
        to_u12, to_u13, to_u14, to_u15, to_u16, to_u17, to_u18, to_u19, to_u20, to_u21, to_u22,
        to_u23, to_u24, to_u25, to_u26, to_u27, to_u28, to_u29, to_u30, to_u31, to_u32,
    ];
    let func = unsafe { LUT.get_unchecked(nbits) };
    unsafe { func(out, block, pack_n) };
}

unsafe fn to_u0(_out: *mut u8, _block: &[u32; X128], _pack_n: usize) {}

macro_rules! define_x128_packer {
    ($func_name:ident, $bit_length:expr) => {
        unsafe fn $func_name(out: *mut u8, block: &[u32; X128], pack_n: usize) {
            let [left, right] = split_block(block);

            let left = load_u32x64(left);
            if pack_n <= 64 {
                unsafe { pack_x64_partial::$func_name(out.add(0), left, pack_n) };
            } else if pack_n < 128 {
                unsafe { pack_x64_full::$func_name(out.add(0), left) };
                let right = load_u32x64(right);
                unsafe {
                    pack_x64_partial::$func_name(
                        out.add(max_compressed_size::<X64>($bit_length)),
                        right,
                        pack_n - X64,
                    )
                }
            } else {
                unsafe { pack_x64_full::$func_name(out.add(0), left) };
                let right = load_u32x64(right);
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
define_x128_packer!(to_u17, 17);
define_x128_packer!(to_u18, 18);
define_x128_packer!(to_u19, 19);
define_x128_packer!(to_u20, 20);
define_x128_packer!(to_u21, 21);
define_x128_packer!(to_u22, 22);
define_x128_packer!(to_u23, 23);
define_x128_packer!(to_u24, 24);
define_x128_packer!(to_u25, 25);
define_x128_packer!(to_u26, 26);
define_x128_packer!(to_u27, 27);
define_x128_packer!(to_u28, 28);
define_x128_packer!(to_u29, 29);
define_x128_packer!(to_u30, 30);
define_x128_packer!(to_u31, 31);
define_x128_packer!(to_u32, 32);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uint32::X128_MAX_OUTPUT_LEN;

    #[test]
    fn test_v1_layout_regression() {
        let tester = crate::uint32::test_util::load_uint32_regression_layout();

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
