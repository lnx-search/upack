use std::arch::x86_64::*;

use super::{pack_x64_full, pack_x64_partial};
use crate::uint32::avx2::data::load_u32x64;
use crate::uint32::{max_compressed_size, split_block};
use crate::{X64, X128};

#[inline]
#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to `nbits` bit length  elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X128>(nbits)` bytes to.
/// - The runtime CPU must support the `avx2` instructions.
/// - `nbits` must be between 0 and 32.
/// - `pack_n` must be no greater than 128.
pub unsafe fn to_nbits(nbits: usize, out: *mut u8, block: &[u32; X128], pack_n: usize) {
    debug_assert!(nbits <= 32, "BUG: invalid nbits provided: {nbits}");
    debug_assert!(pack_n <= X128, "BUG: invalid pack_n provided: {pack_n}");
    #[allow(clippy::type_complexity)]
    const LUT: [unsafe fn(out: *mut u8, &[u32; X128], usize, u32); 33] = [
        to_u0::<0>,
        to_u1::<0>,
        to_u2::<0>,
        to_u3::<0>,
        to_u4::<0>,
        to_u5::<0>,
        to_u6::<0>,
        to_u7::<0>,
        to_u8::<0>,
        to_u9::<0>,
        to_u10::<0>,
        to_u11::<0>,
        to_u12::<0>,
        to_u13::<0>,
        to_u14::<0>,
        to_u15::<0>,
        to_u16::<0>,
        to_u17::<0>,
        to_u18::<0>,
        to_u19::<0>,
        to_u20::<0>,
        to_u21::<0>,
        to_u22::<0>,
        to_u23::<0>,
        to_u24::<0>,
        to_u25::<0>,
        to_u26::<0>,
        to_u27::<0>,
        to_u28::<0>,
        to_u29::<0>,
        to_u30::<0>,
        to_u31::<0>,
        to_u32::<0>,
    ];
    let func = unsafe { LUT.get_unchecked(nbits) };
    unsafe { func(out, block, pack_n, 0) };
}

#[inline]
#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to `nbits` bit length  elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X128>(nbits)` bytes to.
/// - The runtime CPU must support the `avx2` instructions.
/// - `nbits` must be between 0 and 32.
/// - `pack_n` must be no greater than 128.
pub unsafe fn to_nbits_delta(
    nbits: usize,
    out: *mut u8,
    block: &[u32; X128],
    pack_n: usize,
    initial_value: u32,
) {
    debug_assert!(nbits <= 32, "BUG: invalid nbits provided: {nbits}");
    debug_assert!(pack_n <= X128, "BUG: invalid pack_n provided: {pack_n}");
    #[allow(clippy::type_complexity)]
    const LUT: [unsafe fn(out: *mut u8, &[u32; X128], usize, u32); 33] = [
        to_u0::<1>,
        to_u1::<1>,
        to_u2::<1>,
        to_u3::<1>,
        to_u4::<1>,
        to_u5::<1>,
        to_u6::<1>,
        to_u7::<1>,
        to_u8::<1>,
        to_u9::<1>,
        to_u10::<1>,
        to_u11::<1>,
        to_u12::<1>,
        to_u13::<1>,
        to_u14::<1>,
        to_u15::<1>,
        to_u16::<1>,
        to_u17::<1>,
        to_u18::<1>,
        to_u19::<1>,
        to_u20::<1>,
        to_u21::<1>,
        to_u22::<1>,
        to_u23::<1>,
        to_u24::<1>,
        to_u25::<1>,
        to_u26::<1>,
        to_u27::<1>,
        to_u28::<1>,
        to_u29::<1>,
        to_u30::<1>,
        to_u31::<1>,
        to_u32::<1>,
    ];
    let func = unsafe { LUT.get_unchecked(nbits) };
    unsafe { func(out, block, pack_n, initial_value) };
}

#[inline]
#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to `nbits` bit length  elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X128>(nbits)` bytes to.
/// - The runtime CPU must support the `avx2` instructions.
/// - `nbits` must be between 0 and 32.
/// - `pack_n` must be no greater than 128.
pub unsafe fn to_nbits_delta1(
    nbits: usize,
    out: *mut u8,
    block: &[u32; X128],
    pack_n: usize,
    initial_value: u32,
) {
    debug_assert!(nbits <= 32, "BUG: invalid nbits provided: {nbits}");
    debug_assert!(pack_n <= X128, "BUG: invalid pack_n provided: {pack_n}");
    #[allow(clippy::type_complexity)]
    const LUT: [unsafe fn(out: *mut u8, &[u32; X128], usize, u32); 33] = [
        to_u0::<2>,
        to_u1::<2>,
        to_u2::<2>,
        to_u3::<2>,
        to_u4::<2>,
        to_u5::<2>,
        to_u6::<2>,
        to_u7::<2>,
        to_u8::<2>,
        to_u9::<2>,
        to_u10::<2>,
        to_u11::<2>,
        to_u12::<2>,
        to_u13::<2>,
        to_u14::<2>,
        to_u15::<2>,
        to_u16::<2>,
        to_u17::<2>,
        to_u18::<2>,
        to_u19::<2>,
        to_u20::<2>,
        to_u21::<2>,
        to_u22::<2>,
        to_u23::<2>,
        to_u24::<2>,
        to_u25::<2>,
        to_u26::<2>,
        to_u27::<2>,
        to_u28::<2>,
        to_u29::<2>,
        to_u30::<2>,
        to_u31::<2>,
        to_u32::<2>,
    ];
    let func = unsafe { LUT.get_unchecked(nbits) };
    unsafe { func(out, block, pack_n, initial_value) };
}

#[target_feature(enable = "avx2")]
fn apply_encoding<const MODE: usize>(
    initial_value: __m256i,
    mut block: [__m256i; 8],
) -> [__m256i; 8] {
    if MODE == 1 {
        block = super::modifiers::delta_encode_x64(initial_value, block);
    } else if MODE == 2 {
        block = super::modifiers::delta1_encode_x64(initial_value, block);
    }
    block
}
#[target_feature(enable = "avx2")]
unsafe fn to_u0<const MODE: usize>(
    _out: *mut u8,
    _block: &[u32; X128],
    _pack_n: usize,
    _initial_value: u32,
) {
}

macro_rules! define_x128_packer {
    ($func_name:ident, $bit_length:expr) => {
        #[target_feature(enable = "avx2")]
        unsafe fn $func_name<const MODE: usize>(
            out: *mut u8,
            block: &[u32; X128],
            pack_n: usize,
            initial_value: u32,
        ) {
            let [left, right] = split_block(block);

            let left = load_u32x64(left);

            if pack_n <= 64 {
                if MODE == 0 {
                    unsafe { pack_x64_partial::$func_name(out.add(0), left, pack_n) };
                } else {
                    let initial_value = _mm256_set1_epi32(initial_value as i32);
                    let deltas = apply_encoding::<MODE>(initial_value, left);
                    unsafe { pack_x64_partial::$func_name(out.add(0), deltas, pack_n) };
                }
            } else if MODE == 0 {
                // Greater than X64, no deltas
                unsafe { pack_x64_full::$func_name(out.add(0), left) };
                let right = load_u32x64(right);
                unsafe {
                    pack_x64_partial::$func_name(
                        out.add(max_compressed_size::<X64>($bit_length)),
                        right,
                        pack_n - X64,
                    )
                };
            } else {
                // Greater than X64, deltas
                let initial_value = _mm256_set1_epi32(initial_value as i32);
                let deltas = apply_encoding::<MODE>(initial_value, left);
                unsafe { pack_x64_full::$func_name(out.add(0), deltas) };
                let right = load_u32x64(right);
                let deltas = apply_encoding::<MODE>(left[7], right);
                unsafe {
                    pack_x64_partial::$func_name(
                        out.add(max_compressed_size::<X64>($bit_length)),
                        deltas,
                        pack_n - X64,
                    )
                };
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
