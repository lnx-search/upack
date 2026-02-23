use std::arch::aarch64::*;

#[inline]
#[target_feature(enable = "neon")]
/// Broadcast the single 32-bit element across all lanes in the register.
pub(super) fn _neon_set1_u32(value: u32) -> uint32x4_t {
    vdupq_n_u32(value)
}

#[inline]
#[target_feature(enable = "neon")]
/// Broadcast the single 16-bit element across all lanes in the register.
pub(super) fn _neon_set1_u16(value: u16) -> uint16x8_t {
    vdupq_n_u16(value)
}

#[inline]
#[target_feature(enable = "neon")]
/// Broadcast the single 8-bit element across all lanes in the register.
pub(super) fn _neon_set1_u8(value: u8) -> uint8x16_t {
    vdupq_n_u8(value)
}

#[inline]
#[target_feature(enable = "neon")]
/// Load 32-bit elements from the provided `ptr`.
///
/// # Safety
/// The provided `ptr` must be safe to read `8` elements.
pub(super) unsafe fn _neon_load_u32(ptr: *const u32) -> uint32x4_t {
    unsafe { vld1q_u32(ptr) }
}

#[inline]
#[target_feature(enable = "neon")]
/// Load 16-bit elements from the provided `ptr`.
///
/// # Safety
/// The provided `ptr` must be safe to read `16` elements.
pub(super) unsafe fn _neon_load_u16(ptr: *const u16) -> uint16x8_t {
    unsafe { vld1q_u16(ptr) }
}

#[inline]
#[target_feature(enable = "neon")]
/// Load 8-bit elements from the provided `ptr`.
///
/// # Safety
/// The provided `ptr` must be safe to read `32` elements.
pub(super) unsafe fn _neon_load_u8(ptr: *const u8) -> uint8x16_t {
    unsafe { vld1q_u8(ptr) }
}

#[inline]
#[target_feature(enable = "neon")]
/// Store 8, 32-bit elements in the provided `ptr`.
///
/// # Safety
/// The provided `ptr` must be safe to write `8` elements.
pub(super) unsafe fn _neon_store_u32(ptr: *mut u32, reg: uint32x4_t) {
    unsafe { vst1q_u32(ptr, reg) };
}

#[inline]
#[target_feature(enable = "neon")]
/// Store 16, 16-bit elements in the provided `ptr`.
///
/// # Safety
/// The provided `ptr` must be safe to write `16` elements.
pub(super) unsafe fn _neon_store_u16(ptr: *mut u16, reg: uint16x8_t) {
    unsafe { vst1q_u16(ptr, reg) };
}

#[inline]
#[target_feature(enable = "neon")]
/// Store 32, 8-bit elements in the provided `ptr`.
///
/// # Safety
/// The provided `ptr` must be safe to write `32` elements.
pub(super) unsafe fn _neon_store_u8(ptr: *mut u8, reg: uint8x16_t) {
    unsafe { vst1q_u8(ptr, reg) };
}

#[inline]
#[target_feature(enable = "neon")]
/// Pack the two provide [u32x8] registers to unsigned 16-bit integers in blocks
/// of 128-bits.
pub(super) fn _neon_pack_u32(a: uint32x4_t, b: uint32x4_t) -> uint16x8_t {
    let lo = vmovn_u32(a);
    let hi = vmovn_u32(b);
    vcombine_u16(lo, hi)
}

#[inline]
#[target_feature(enable = "neon")]
/// Pack the two provide [u16x16] registers to unsigned 8-bit integers in blocks
/// of 128-bits.
pub(super) fn _neon_pack_u16x16(a: uint16x8_t, b: uint16x8_t) -> uint8x16_t {
    let lo = vmovn_u16(a);
    let hi = vmovn_u16(b);
    vcombine_u8(lo, hi)
}

#[inline]
#[target_feature(enable = "neon")]
/// Convert the provided 8-bit elements in register `a` to 16-bit integers via extension.
pub(super) fn _neon_cvteu8_u16(a: uint8x16_t) -> [uint16x8_t; 2] {
    let lo = vmovl_u8(vget_low_u8(a));
    let hi = vmovl_u8(vget_high_u8(a));
    [lo, hi]
}

#[inline]
#[target_feature(enable = "neon")]
/// Convert the provided 16-bit elements in register `a` to 32-bit integers via extension.
pub(super) fn _neon_cvteu16_u32(a: uint16x8_t) -> [uint32x4_t; 2] {
    let lo = vmovl_u16(vget_low_u16(a));
    let hi = vmovl_u16(vget_high_u16(a));
    [lo, hi]
}

#[inline]
#[target_feature(enable = "neon")]
/// Convert the provided [u32x8] register into a single [u16x8] register using
/// truncation.
pub(super) fn _neon_cvteu32_u16(a: uint32x4_t, b: uint32x4_t) -> uint16x8_t {
    let lo = vmovn_u32(a);
    let hi = vmovn_u32(b);
    vcombine_u16(lo, hi)
}

#[inline]
#[target_feature(enable = "neon")]
/// Convert the provided [u16x16] register into a single [u8x16] register using
/// truncation.
pub(super) fn _neon_cvteu16_u8(a: uint16x8_t, b: uint16x8_t) -> uint8x16_t {
    let lo = vmovn_u16(a);
    let hi = vmovn_u16(b);
    vcombine_u8(lo, hi)
}

#[inline]
#[target_feature(enable = "neon")]
/// Perform a bitwise AND on the provided registers `a` and `b`.
pub(super) fn _neon_and_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    vandq_u32(a, b)
}

#[inline]
#[target_feature(enable = "neon")]
/// Perform a bitwise AND on the provided registers `a` and `b`.
pub(super) fn _neon_and_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    vandq_u16(a, b)
}

#[inline]
#[target_feature(enable = "neon")]
/// Perform a bitwise AND on the provided registers `a` and `b`.
pub(super) fn _neon_and_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    vandq_u8(a, b)
}

#[inline]
#[target_feature(enable = "neon")]
/// Perform a bitwise OR on the provided registers `a` and `b`.
pub(super) fn _neon_or_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    vorrq_u32(a, b)
}

#[inline]
#[target_feature(enable = "neon")]
/// Perform a bitwise OR on the provided registers `a` and `b`.
pub(super) fn _neon_or_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    vorrq_u16(a, b)
}

#[inline]
#[target_feature(enable = "neon")]
/// Perform a bitwise OR on the provided registers `a` and `b`.
pub(super) fn _neon_or_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    vorrq_u8(a, b)
}

#[inline]
#[target_feature(enable = "neon")]
/// Perform a bitwise shift right on the provided register `a`.
pub(super) fn _neon_srli_u32<const IMM8: i32>(a: uint32x4_t) -> uint32x4_t {
    vshrq_n_u32::<IMM8>(a)
}

#[inline]
#[target_feature(enable = "neon")]
/// Perform a bitwise shift right on the provided register `a`.
pub(super) fn _neon_srli_u16<const IMM8: i32>(a: uint16x8_t) -> uint16x8_t {
    vshrq_n_u16::<IMM8>(a)
}

#[inline]
#[target_feature(enable = "neon")]
/// Perform a bitwise shift right on the provided register `a`.
pub(super) fn _neon_srli_u8<const IMM8: i32>(a: uint8x16_t) -> uint8x16_t {
    vshrq_n_u8::<IMM8>(a)
}

#[inline]
#[target_feature(enable = "neon")]
/// Perform a bitwise shift left on the provided register `a`.
pub(super) fn _neon_slli_u32<const IMM8: i32>(a: uint32x4_t) -> uint32x4_t {
    vshlq_n_u32::<IMM8>(a)
}

#[inline]
#[target_feature(enable = "neon")]
/// Perform a bitwise shift left on the provided register `a`.
pub(super) fn _neon_slli_u16<const IMM8: i32>(a: uint16x8_t) -> uint16x8_t {
    vshlq_n_u16::<IMM8>(a)
}

#[inline]
#[target_feature(enable = "neon")]
/// Perform a bitwise shift left on the provided register `a`.
pub(super) fn _neon_slli_u8<const IMM8: i32>(a: uint8x16_t) -> uint8x16_t {
    vshlq_n_u8::<IMM8>(a)
}

#[inline]
#[target_feature(enable = "neon")]
/// Return a bitmask with a set bit indicating the element at the same index is non-zero.
pub fn _neon_nonzero_mask_u8(a: uint8x16_t) -> u16 {
    const MASK: [u8; 16] = [
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40,
        0x80,
    ];

    let cmp = vmvnq_u8(vceqq_u8(a, vdupq_n_u8(0)));
    let bitmask = unsafe { vld1q_u8(MASK.as_ptr()) };

    let masked = vandq_u8(cmp, bitmask);
    let mut paired = vpadd_u8(vget_low_u8(masked), vget_high_u8(masked));
    paired = vpadd_u8(paired, paired);
    paired = vpadd_u8(paired, paired);

    vget_lane_u16::<0>(vreinterpret_u16_u8(paired))
}

#[inline]
#[target_feature(enable = "neon")]
/// Selectively move elements from `a` into the output register based on the respective
/// bit flag in `mask`.
pub(super) fn _neon_mov_maskz_u8(mask: u16, a: uint8x16_t) -> uint8x16_t {
    const BIT_POS: [u8; 8] = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80];

    let k_lo = vdup_n_u8(mask as u8);
    let k_hi = vdup_n_u8((mask >> 8) as u8);

    let bit_pos = unsafe { vld1_u8(BIT_POS.as_ptr()) };

    let test_lo = vtst_u8(k_lo, bit_pos);
    let test_hi = vtst_u8(k_hi, bit_pos);

    let mask = vcombine_u8(test_lo, test_hi);
    vandq_u8(a, mask)
}

#[inline]
#[target_feature(enable = "neon")]
pub(super) fn _neon_pack_nibbles(a: [uint8x16_t; 2], b: [uint8x16_t; 2]) -> [uint8x16_t; 2] {
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

    [packed1, packed2]
}

#[inline]
#[target_feature(enable = "neon")]
pub(super) fn _neon_unpack_nibbles(packed: uint8x16_t) -> [uint8x16_t; 2] {
    let mask = vdupq_n_u8(0x0F);

    let lo = vandq_u8(packed, mask); // low nibbles (even positions)
    let hi = vshrq_n_u8::<4>(packed); // high nibbles (odd positions)
    let d1 = vzip1q_u8(lo, hi); // interleave to form first 16 bytes
    let d2 = vzip2q_u8(lo, hi); // interleave to form second 16 bytes

    [d1, d2]
}

#[inline]
#[target_feature(enable = "neon")]
pub(super) fn _neon_blend_every_other_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    let mask = vreinterpretq_u8_u16(vdupq_n_u16(0x00FF));
    vbslq_u8(mask, a, b)
}

#[inline]
#[target_feature(enable = "neon")]
pub(super) fn _neon_blend_every_other_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    let mask = vreinterpretq_u16_u32(vdupq_n_u32(0x0000_FFFF));
    vbslq_u16(mask, a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_pack_u32() {
        unsafe {
            let a = _neon_set1_u32(4);
            let b = _neon_set1_u32(2);
            let result = _neon_pack_u32(a, b);
            let view = std::mem::transmute::<uint16x8_t, [u16; 8]>(result);
            assert_eq!(
                view,
                [
                    4, 4, 4, 4, // a_lo
                    2, 2, 2, 2, // b_lo
                ]
            );
        }
    }

    #[test]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_pack_u16() {
        unsafe {
            let a = _neon_set1_u16(4);
            let b = _neon_set1_u16(2);
            let result = _neon_pack_u16x16(a, b);
            let view = std::mem::transmute::<uint8x16_t, [u8; 16]>(result);
            assert_eq!(
                view,
                [
                    4, 4, 4, 4, 4, 4, 4, 4, // a_lo
                    2, 2, 2, 2, 2, 2, 2, 2, // b_lo
                ]
            );
        }
    }

    #[test]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_cvteu32_u16() {
        unsafe {
            let a = _neon_set1_u32(4);
            let b = _neon_set1_u32(4);
            let result = _neon_cvteu32_u16(a, b);
            let view = std::mem::transmute::<uint16x8_t, [u16; 8]>(result);
            assert_eq!(view, [4; 8]);
        }
    }

    #[test]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_cvteu16_u8() {
        unsafe {
            let a = _neon_set1_u16(4);
            let b = _neon_set1_u16(4);
            let result = _neon_cvteu16_u8(a, b);
            let view = std::mem::transmute::<uint8x16_t, [u8; 16]>(result);
            assert_eq!(view, [4; 16]);
        }
    }

    #[test]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_cvteu8_u16() {
        unsafe {
            let a = _neon_set1_u8(4);
            let result = _neon_cvteu8_u16(a);
            let view = std::mem::transmute::<[uint16x8_t; 2], [u16; 16]>(result);
            assert_eq!(view, [4; 16]);
        }
    }

    #[test]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_cvteu16_u32() {
        unsafe {
            let a = _neon_set1_u16(4);
            let result = _neon_cvteu16_u32(a);
            let view = std::mem::transmute::<[uint32x4_t; 2], [u32; 8]>(result);
            assert_eq!(view, [4; 8]);
        }
    }

    #[test]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_movmask_u8() {
        unsafe {
            let a = _neon_set1_u8(0);
            let result = _neon_nonzero_mask_u8(a);
            assert_eq!(result, 0);

            let a = _neon_set1_u8(u8::MAX);
            let result = _neon_nonzero_mask_u8(a);
            assert_eq!(result, u16::MAX);
        }
    }

    #[test]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_mov_maskz_u8() {
        unsafe {
            let a = _neon_set1_u8(2);
            let result = _neon_mov_maskz_u8(u16::MAX, a);
            let view = std::mem::transmute::<uint8x16_t, [u8; 16]>(result);
            assert_eq!(view, [2; 16]);

            let a = _neon_set1_u8(2);
            let result = _neon_mov_maskz_u8(0, a);
            let view = std::mem::transmute::<uint8x16_t, [u8; 16]>(result);
            assert_eq!(view, [0; 16]);
        }
    }
}
