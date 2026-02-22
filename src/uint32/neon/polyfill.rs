use std::arch::aarch64::*;
use std::ops::Index;

#[allow(non_camel_case_types)]
#[repr(transparent)]
#[derive(Copy, Clone)]
/// A block of 8, 32-bit elements.
pub(super) struct u32x8([uint32x4_t; 2]);

impl From<u16x16> for u32x8 {
    fn from(value: u16x16) -> Self {
        unsafe { std::mem::transmute(value) }
    }
}

impl From<u8x32> for u32x8 {
    fn from(value: u8x32) -> Self {
        unsafe { std::mem::transmute(value) }
    }
}

impl Index<usize> for u32x8 {
    type Output = uint32x4_t;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

#[allow(non_camel_case_types)]
#[repr(transparent)]
#[derive(Copy, Clone)]
/// A block of 16, 16-bit elements.
pub(super) struct u16x16([uint16x8_t; 2]);

impl From<u32x8> for u16x16 {
    fn from(value: u32x8) -> Self {
        unsafe { std::mem::transmute(value) }
    }
}

impl From<u8x32> for u16x16 {
    fn from(value: u8x32) -> Self {
        unsafe { std::mem::transmute(value) }
    }
}

impl Index<usize> for u16x16 {
    type Output = uint16x8_t;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

#[allow(non_camel_case_types)]
#[repr(transparent)]
#[derive(Copy, Clone)]
/// A block of 32, 8-bit elements.
pub(super) struct u8x32([uint8x16_t; 2]);

impl From<u32x8> for u8x32 {
    fn from(value: u32x8) -> Self {
        unsafe { std::mem::transmute(value) }
    }
}

impl From<u16x16> for u8x32 {
    fn from(value: u16x16) -> Self {
        unsafe { std::mem::transmute(value) }
    }
}

impl Index<usize> for u8x32 {
    type Output = uint8x16_t;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

#[allow(non_camel_case_types)]
#[repr(transparent)]
#[derive(Copy, Clone)]
/// A block of 8, 16-bit elements.
pub(super) struct u16x8(uint16x8_t);

#[allow(non_camel_case_types)]
#[repr(transparent)]
#[derive(Copy, Clone)]
/// A block of 16, 8-bit elements.
pub(super) struct u8x16(pub(super) uint8x16_t);

#[target_feature(enable = "neon")]
/// Broadcast the single 32-bit element across all lanes in the register.
pub(super) fn _neon_set1_u32(value: u32) -> u32x8 {
    u32x8([vdupq_n_u32(value), vdupq_n_u32(value)])
}

#[target_feature(enable = "neon")]
/// Broadcast the single 16-bit element across all lanes in the register.
pub(super) fn _neon_set1_u16(value: u16) -> u16x16 {
    u16x16([vdupq_n_u16(value), vdupq_n_u16(value)])
}

#[target_feature(enable = "neon")]
/// Broadcast the single 8-bit element across all lanes in the register.
pub(super) fn _neon_set1_u8(value: u8) -> u8x32 {
    u8x32([vdupq_n_u8(value), vdupq_n_u8(value)])
}

#[target_feature(enable = "neon")]
/// Load 8, 32-bit elements from the provided `ptr`.
///
/// # Safety
/// The provided `ptr` must be safe to read `8` elements.
pub(super) unsafe fn _neon_load_u32x8(ptr: *const u32) -> u32x8 {
    let a = unsafe { vld1q_u32(ptr.add(0)) };
    let b = unsafe { vld1q_u32(ptr.add(4)) };
    u32x8([a, b])
}

#[target_feature(enable = "neon")]
/// Load 16, 16-bit elements from the provided `ptr`.
///
/// # Safety
/// The provided `ptr` must be safe to read `16` elements.
pub(super) unsafe fn _neon_load_u16x16(ptr: *const u16) -> u16x16 {
    let a = unsafe { vld1q_u16(ptr.add(0)) };
    let b = unsafe { vld1q_u16(ptr.add(8)) };
    u16x16([a, b])
}

#[target_feature(enable = "neon")]
/// Load 32, 8-bit elements from the provided `ptr`.
///
/// # Safety
/// The provided `ptr` must be safe to read `32` elements.
pub(super) unsafe fn _neon_load_u8x32(ptr: *const u8) -> u8x32 {
    let a = unsafe { vld1q_u8(ptr.add(0)) };
    let b = unsafe { vld1q_u8(ptr.add(16)) };
    u8x32([a, b])
}

#[target_feature(enable = "neon")]
/// Load 16, 8-bit elements from the provided `ptr`.
///
/// # Safety
/// The provided `ptr` must be safe to read `16` elements.
pub(super) unsafe fn _neon_load_u8x16(ptr: *const u8) -> u8x16 {
    let a = unsafe { vld1q_u8(ptr.add(0)) };
    u8x16(a)
}

#[target_feature(enable = "neon")]
/// Store 8, 32-bit elements in the provided `ptr`.
///
/// # Safety
/// The provided `ptr` must be safe to write `8` elements.
pub(super) unsafe fn _neon_store_u32x8(ptr: *mut u32, reg: u32x8) {
    unsafe { vst1q_u32(ptr.add(0), reg[0]) };
    unsafe { vst1q_u32(ptr.add(4), reg[1]) };
}

#[target_feature(enable = "neon")]
/// Store 16, 16-bit elements in the provided `ptr`.
///
/// # Safety
/// The provided `ptr` must be safe to write `16` elements.
pub(super) unsafe fn _neon_store_u16x16(ptr: *mut u16, reg: u16x16) {
    unsafe { vst1q_u16(ptr.add(0), reg[0]) };
    unsafe { vst1q_u16(ptr.add(8), reg[1]) };
}

#[target_feature(enable = "neon")]
/// Store 32, 8-bit elements in the provided `ptr`.
///
/// # Safety
/// The provided `ptr` must be safe to write `32` elements.
pub(super) unsafe fn _neon_store_u8x32(ptr: *mut u8, reg: u8x32) {
    unsafe { vst1q_u8(ptr.add(0), reg[0]) };
    unsafe { vst1q_u8(ptr.add(16), reg[1]) };
}

#[target_feature(enable = "neon")]
/// Store 16, 8-bit elements in the provided `ptr`.
///
/// # Safety
/// The provided `ptr` must be safe to write `16` elements.
pub(super) unsafe fn _neon_store_u8x16(ptr: *mut u8, reg: u8x16) {
    unsafe { vst1q_u8(ptr, reg.0) };
}

#[target_feature(enable = "neon")]
/// Pack the two provide [u32x8] registers to unsigned 16-bit integers in blocks
/// of 128-bits.
pub(super) fn _neon_pack_u32x8(a: u32x8, b: u32x8) -> u16x16 {
    let a_lo = vmovn_u32(a[0]);
    let a_hi = vmovn_u32(a[1]);
    let b_lo = vmovn_u32(b[0]);
    let b_hi = vmovn_u32(b[1]);
    let r1 = vcombine_u16(a_lo, b_lo);
    let r2 = vcombine_u16(a_hi, b_hi);
    u16x16([r1, r2])
}

#[target_feature(enable = "neon")]
/// Pack the two provide [u16x16] registers to unsigned 8-bit integers in blocks
/// of 128-bits.
pub(super) fn _neon_pack_u16x16(a: u16x16, b: u16x16) -> u8x32 {
    let a_lo = vmovn_u16(a[0]);
    let a_hi = vmovn_u16(a[1]);
    let b_lo = vmovn_u16(b[0]);
    let b_hi = vmovn_u16(b[1]);
    let r1 = vcombine_u8(a_lo, b_lo);
    let r2 = vcombine_u8(a_hi, b_hi);
    u8x32([r1, r2])
}

#[target_feature(enable = "neon")]
/// Convert the provided 8-bit elements in register `a` to 16-bit integers via extension.
pub(super) fn _neon_cvteu16_u8x16(a: u8x16) -> u16x16 {
    let lo = vmovl_u8(vget_low_u8(a.0));
    let hi = vmovl_u8(vget_high_u8(a.0));
    u16x16([lo, hi])
}

#[target_feature(enable = "neon")]
/// Convert the provided 16-bit elements in register `a` to 32-bit integers via extension.
pub(super) fn _neon_cvteu32_u16x8(a: u16x8) -> u32x8 {
    let lo = vmovl_u16(vget_low_u16(a.0));
    let hi = vmovl_u16(vget_high_u16(a.0));
    u32x8([lo, hi])
}

#[target_feature(enable = "neon")]
/// Convert the provided [u32x8] register into a single [u16x8] register using
/// truncation.
pub(super) fn _neon_cvteu16_u32x8(a: u32x8) -> u16x8 {
    let lo = vmovn_u32(a[0]);
    let hi = vmovn_u32(a[1]);
    u16x8(vcombine_u16(lo, hi))
}

#[target_feature(enable = "neon")]
/// Convert the provided [u16x16] register into a single [u8x16] register using
/// truncation.
pub(super) fn _neon_cvteu8_u16x16(a: u16x16) -> u8x16 {
    let lo = vmovn_u16(a[0]);
    let hi = vmovn_u16(a[1]);
    u8x16(vcombine_u8(lo, hi))
}

#[target_feature(enable = "neon")]
/// Combine two [u16x8] registers via concatenation forming a [u16x16] register.
pub(super) fn _neon_combine_u16x8(a: u16x8, b: u16x8) -> u16x16 {
    u16x16([a.0, b.0])
}

#[target_feature(enable = "neon")]
/// Combine two [u8x16] registers via concatenation forming a [u8x32] register.
pub(super) fn _neon_combine_u8x16(a: u8x16, b: u8x16) -> u8x32 {
    u8x32([a.0, b.0])
}

#[target_feature(enable = "neon")]
/// Perform a bitwise AND on the provided registers `a` and `b`.
pub(super) fn _neon_and_u32x8(a: u32x8, b: u32x8) -> u32x8 {
    let r1 = vandq_u32(a[0], b[0]);
    let r2 = vandq_u32(a[1], b[1]);
    u32x8([r1, r2])
}

#[target_feature(enable = "neon")]
/// Perform a bitwise AND on the provided registers `a` and `b`.
pub(super) fn _neon_and_u16x16(a: u16x16, b: u16x16) -> u16x16 {
    let r1 = vandq_u16(a[0], b[0]);
    let r2 = vandq_u16(a[1], b[1]);
    u16x16([r1, r2])
}

#[target_feature(enable = "neon")]
/// Perform a bitwise AND on the provided registers `a` and `b`.
pub(super) fn _neon_and_u8x32(a: u8x32, b: u8x32) -> u8x32 {
    let r1 = vandq_u8(a[0], b[0]);
    let r2 = vandq_u8(a[1], b[1]);
    u8x32([r1, r2])
}

#[target_feature(enable = "neon")]
/// Perform a bitwise OR on the provided registers `a` and `b`.
pub(super) fn _neon_or_u32x8(a: u32x8, b: u32x8) -> u32x8 {
    let r1 = vorrq_u32(a[0], b[0]);
    let r2 = vorrq_u32(a[1], b[1]);
    u32x8([r1, r2])
}

#[target_feature(enable = "neon")]
/// Perform a bitwise OR on the provided registers `a` and `b`.
pub(super) fn _neon_or_u16x16(a: u16x16, b: u16x16) -> u16x16 {
    let r1 = vorrq_u16(a[0], b[0]);
    let r2 = vorrq_u16(a[1], b[1]);
    u16x16([r1, r2])
}

#[target_feature(enable = "neon")]
/// Perform a bitwise OR on the provided registers `a` and `b`.
pub(super) fn _neon_or_u8x32(a: u8x32, b: u8x32) -> u8x32 {
    let r1 = vorrq_u8(a[0], b[0]);
    let r2 = vorrq_u8(a[1], b[1]);
    u8x32([r1, r2])
}

#[target_feature(enable = "neon")]
/// Perform a bitwise shift right on the provided register `a`.
pub(super) fn _neon_srli_u32x8<const IMM8: i32>(a: u32x8) -> u32x8 {
    let r1 = vshrq_n_u32::<IMM8>(a[0]);
    let r2 = vshrq_n_u32::<IMM8>(a[1]);
    u32x8([r1, r2])
}

#[target_feature(enable = "neon")]
/// Perform a bitwise shift right on the provided register `a`.
pub(super) fn _neon_srli_u16x16<const IMM8: i32>(a: u16x16) -> u16x16 {
    let r1 = vshrq_n_u16::<IMM8>(a[0]);
    let r2 = vshrq_n_u16::<IMM8>(a[1]);
    u16x16([r1, r2])
}

#[target_feature(enable = "neon")]
/// Perform a bitwise shift right on the provided register `a`.
pub(super) fn _neon_srli_u8x32<const IMM8: i32>(a: u8x32) -> u8x32 {
    let r1 = vshrq_n_u8::<IMM8>(a[0]);
    let r2 = vshrq_n_u8::<IMM8>(a[1]);
    u8x32([r1, r2])
}

#[target_feature(enable = "neon")]
/// Perform a bitwise shift left on the provided register `a`.
pub(super) fn _neon_slli_u32x8<const IMM8: i32>(a: u32x8) -> u32x8 {
    let r1 = vshlq_n_u32::<IMM8>(a[0]);
    let r2 = vshlq_n_u32::<IMM8>(a[1]);
    u32x8([r1, r2])
}

#[target_feature(enable = "neon")]
/// Perform a bitwise shift left on the provided register `a`.
pub(super) fn _neon_slli_u16x16<const IMM8: i32>(a: u16x16) -> u16x16 {
    let r1 = vshlq_n_u16::<IMM8>(a[0]);
    let r2 = vshlq_n_u16::<IMM8>(a[1]);
    u16x16([r1, r2])
}

#[target_feature(enable = "neon")]
/// Perform a bitwise shift left on the provided register `a`.
pub(super) fn _neon_slli_u8x32<const IMM8: i32>(a: u8x32) -> u8x32 {
    let r1 = vshlq_n_u8::<IMM8>(a[0]);
    let r2 = vshlq_n_u8::<IMM8>(a[1]);
    u8x32([r1, r2])
}

#[target_feature(enable = "neon")]
/// Extract one 128-bit half from the input register.
///
/// `HALF` can be either `0` or `1` representing the index of the halves respectively.
pub(super) fn _neon_extract_u16x16<const HALF: usize>(a: u16x16) -> u16x8 {
    const { assert!(HALF <= 1, "selector must be either 0 or 1") };
    if HALF == 0 { u16x8(a[0]) } else { u16x8(a[1]) }
}

#[target_feature(enable = "neon")]
/// Extract one 128-bit half from the input register.
///
/// `HALF` can be either `0` or `1` representing the index of the halves respectively.
pub(super) fn _neon_extract_u8x32<const HALF: usize>(a: u8x32) -> u8x16 {
    const { assert!(HALF <= 1, "selector must be either 0 or 1") };
    if HALF == 0 { u8x16(a[0]) } else { u8x16(a[1]) }
}

#[target_feature(enable = "neon")]
/// Return a bitmask with a set bit indicating the element at the same index is non-zero.
pub(super) fn _neon_nonzero_mask_u8x32(a: u8x32) -> u32 {
    let lo_mask = _neon_nonzero_mask_u8x16(u8x16(a[0]));
    let hi_mask = _neon_nonzero_mask_u8x16(u8x16(a[1]));
    ((hi_mask as u32) << 16) | (lo_mask as u32)
}

#[target_feature(enable = "neon")]
fn _neon_nonzero_mask_u8x16(a: u8x16) -> u16 {
    const MASK: [u8; 16] = [
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40,
        0x80,
    ];

    let cmp = vmvnq_u8(vceqq_u8(a.0, vdupq_n_u8(0)));
    let bitmask = unsafe { vld1q_u8(MASK.as_ptr()) };

    let masked = vandq_u8(cmp, bitmask);
    let mut paired = vpadd_u8(vget_low_u8(masked), vget_high_u8(masked));
    paired = vpadd_u8(paired, paired);
    paired = vpadd_u8(paired, paired);

    vget_lane_u16::<0>(vreinterpret_u16_u8(paired))
}

#[target_feature(enable = "neon")]
/// Selectively move elements from `a` into the output register based on the respective
/// bit flag in `mask`.
pub(super) fn _neon_mov_maskz_u8x32(mask: u32, a: u8x32) -> u8x32 {
    let r1 = _neon_mov_maskz_u8x16(mask as u16, u8x16(a[0]));
    let r2 = _neon_mov_maskz_u8x16((mask >> 16) as u16, u8x16(a[1]));
    u8x32([r1.0, r2.0])
}

#[target_feature(enable = "neon")]
fn _neon_mov_maskz_u8x16(mask: u16, a: u8x16) -> u8x16 {
    const BIT_POS: [u8; 8] = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80];

    let k_lo = vdup_n_u8(mask as u8);
    let k_hi = vdup_n_u8((mask >> 8) as u8);

    let bit_pos = unsafe { vld1_u8(BIT_POS.as_ptr()) };

    let test_lo = vtst_u8(k_lo, bit_pos);
    let test_hi = vtst_u8(k_hi, bit_pos);

    let mask = vcombine_u8(test_lo, test_hi);
    let r = vandq_u8(a.0, mask);
    u8x16(r)
}

#[target_feature(enable = "neon")]
pub(super) fn _neon_pack_nibbles(a: u8x32, b: u8x32) -> u8x32 {
    let a_even = vuzp1q_u8(a[0], a[1]); // [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]
    let a_odd = vuzp2q_u8(a[0], a[1]); // [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31]

    let mask = vdupq_n_u8(0x0F);
    let a_masked = vandq_u8(a_even, mask);
    let a_shifted = vshlq_n_u8::<4>(a_odd);
    let packed1 = vorrq_u8(a_masked, a_shifted);

    // Process d2: pack 32 bytes -> 16 bytes
    let b_even = vuzp1q_u8(b[0], b[1]);
    let b_odd = vuzp2q_u8(b[0], b[1]);

    let b_masked = vandq_u8(b_even, mask);
    let b_shifted = vshlq_n_u8::<4>(b_odd);
    let packed2 = vorrq_u8(b_masked, b_shifted);

    u8x32([packed1, packed2])
}

#[target_feature(enable = "neon")]
pub(super) fn _neon_unpack_nibbles(packed: u8x32) -> [u8x32; 2] {
    let mask = vdupq_n_u8(0x0F);

    let a_lo = vandq_u8(packed[0], mask); // low nibbles (even positions)
    let a_hi = vshrq_n_u8::<4>(packed[0]); // high nibbles (odd positions)
    let a_0 = vzip1q_u8(a_lo, a_hi); // interleave to form first 16 bytes
    let a_1 = vzip2q_u8(a_lo, a_hi); // interleave to form second 16 bytes

    let b_lo = vandq_u8(packed[1], mask);
    let b_hi = vshrq_n_u8::<4>(packed[1]);
    let b_0 = vzip1q_u8(b_lo, b_hi);
    let b_1 = vzip2q_u8(b_lo, b_hi);

    [u8x32([a_0, a_1]), u8x32([b_0, b_1])]
}

#[target_feature(enable = "neon")]
pub(super) fn _neon_blend_every_other_u8(a: u8x32, b: u8x32) -> u8x32 {
    let mask = vreinterpretq_u8_u16(vdupq_n_u16(0x00FF));
    u8x32([vbslq_u8(mask, a[0], b[0]), vbslq_u8(mask, a[1], b[1])])
}

#[target_feature(enable = "neon")]
pub(super) fn _neon_blend_every_other_u16(a: u16x16, b: u16x16) -> u16x16 {
    let mask = vreinterpretq_u16_u32(vdupq_n_u32(0x0000_FFFF));
    u16x16([vbslq_u16(mask, a[0], b[0]), vbslq_u16(mask, a[1], b[1])])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_pack_u32x8() {
        unsafe {
            let a = _neon_set1_u32(4);
            let b = _neon_set1_u32(2);
            let result = _neon_pack_u32x8(a, b);
            let view = std::mem::transmute::<u16x16, [u16; 16]>(result);
            assert_eq!(
                view,
                [
                    4, 4, 4, 4, // a_lo
                    2, 2, 2, 2, // b_lo
                    4, 4, 4, 4, // a_hi
                    2, 2, 2, 2, // b_hi
                ]
            );
        }
    }

    #[test]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_pack_u16x16() {
        unsafe {
            let a = _neon_set1_u16(4);
            let b = _neon_set1_u16(2);
            let result = _neon_pack_u16x16(a, b);
            let view = std::mem::transmute::<u8x32, [u8; 32]>(result);
            assert_eq!(
                view,
                [
                    4, 4, 4, 4, 4, 4, 4, 4, // a_lo
                    2, 2, 2, 2, 2, 2, 2, 2, // b_lo
                    4, 4, 4, 4, 4, 4, 4, 4, // a_hi
                    2, 2, 2, 2, 2, 2, 2, 2, // b_hi
                ]
            );
        }
    }

    #[test]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_cvteu16_u32x8() {
        unsafe {
            let a = _neon_set1_u32(4);
            let result = _neon_cvteu16_u32x8(a);
            let view = std::mem::transmute::<u16x8, [u16; 8]>(result);
            assert_eq!(view, [4; 8]);
        }
    }

    #[test]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_cvteu8_u16x16() {
        unsafe {
            let a = _neon_set1_u16(4);
            let result = _neon_cvteu8_u16x16(a);
            let view = std::mem::transmute::<u8x16, [u8; 16]>(result);
            assert_eq!(view, [4; 16]);
        }
    }

    #[test]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_cvteu16_u8x16() {
        unsafe {
            let a = _neon_extract_u8x32::<0>(_neon_set1_u8(4));
            let result = _neon_cvteu16_u8x16(a);
            let view = std::mem::transmute::<u16x16, [u16; 16]>(result);
            assert_eq!(view, [4; 16]);
        }
    }

    #[test]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_cvteu32_u16x8() {
        unsafe {
            let a = _neon_extract_u16x16::<0>(_neon_set1_u16(4));
            let result = _neon_cvteu32_u16x8(a);
            let view = std::mem::transmute::<u32x8, [u32; 8]>(result);
            assert_eq!(view, [4; 8]);
        }
    }

    #[test]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_combine_u16x8() {
        unsafe {
            let a = _neon_extract_u16x16::<0>(_neon_set1_u16(1));
            let b = _neon_extract_u16x16::<0>(_neon_set1_u16(2));
            let result = _neon_combine_u16x8(a, b);
            let view = std::mem::transmute::<u16x16, [u16; 16]>(result);
            assert_eq!(view[..8], [1; 8]);
            assert_eq!(view[8..], [2; 8]);
        }
    }

    #[test]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_combine_u8x16() {
        unsafe {
            let a = _neon_extract_u8x32::<0>(_neon_set1_u8(1));
            let b = _neon_extract_u8x32::<0>(_neon_set1_u8(2));
            let result = _neon_combine_u8x16(a, b);
            let view = std::mem::transmute::<u8x32, [u8; 32]>(result);
            assert_eq!(view[..16], [1; 16]);
            assert_eq!(view[16..], [2; 16]);
        }
    }

    #[test]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_extract_u16x16() {
        unsafe {
            let a = _neon_load_u16x16([4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2].as_ptr());

            let result = _neon_extract_u16x16::<0>(a);
            let view = std::mem::transmute::<u16x8, [u16; 8]>(result);
            assert_eq!(view, [4; 8]);

            let result = _neon_extract_u16x16::<1>(a);
            let view = std::mem::transmute::<u16x8, [u16; 8]>(result);
            assert_eq!(view, [2; 8]);
        }
    }

    #[test]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_extract_u8x32() {
        unsafe {
            let a = _neon_load_u8x32(
                [
                    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                    2, 2, 2, 2, 2, 2,
                ]
                .as_ptr(),
            );

            let result = _neon_extract_u8x32::<0>(a);
            let view = std::mem::transmute::<u8x16, [u8; 16]>(result);
            assert_eq!(view, [4; 16]);

            let result = _neon_extract_u8x32::<1>(a);
            let view = std::mem::transmute::<u8x16, [u8; 16]>(result);
            assert_eq!(view, [2; 16]);
        }
    }

    #[test]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_movmask_u8x32() {
        unsafe {
            let a = _neon_set1_u8(0);
            let result = _neon_nonzero_mask_u8x32(a);
            assert_eq!(result, 0);

            let a = _neon_set1_u8(u8::MAX);
            let result = _neon_nonzero_mask_u8x32(a);
            assert_eq!(result, u32::MAX);
        }
    }

    #[test]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_mov_maskz_u8x32() {
        unsafe {
            let a = _neon_set1_u8(2);
            let result = _neon_mov_maskz_u8x32(u32::MAX, a);
            let view = std::mem::transmute::<u8x32, [u8; 32]>(result);
            assert_eq!(view, [2; 32]);

            let a = _neon_set1_u8(2);
            let result = _neon_mov_maskz_u8x32(0, a);
            let view = std::mem::transmute::<u8x32, [u8; 32]>(result);
            assert_eq!(view, [0; 32]);
        }
    }
}
