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
pub fn _neon_nonzero_mask_u8(regs: [uint8x16_t; 4]) -> u64 {
    let view = unsafe { std::mem::transmute::<[uint8x16_t; 4], [u8; 64]>(regs) };
    let interleaved = unsafe { vld4q_u8(view.as_ptr()) };

    let zeroes = vdupq_n_u8(0);
    let chunks = [
        vceqq_u8(interleaved.0, zeroes),
        vceqq_u8(interleaved.1, zeroes),
        vceqq_u8(interleaved.2, zeroes),
        vceqq_u8(interleaved.3, zeroes),
    ];

    !vmovmaskq_u8(chunks)
}

#[inline]
#[target_feature(enable = "neon")]
fn vmovmaskq_u8(chunks: [uint8x16_t; 4]) -> u64 {
    let t0 = vsriq_n_u8::<1>(chunks[1], chunks[0]);
    let t1 = vsriq_n_u8::<1>(chunks[3], chunks[2]);
    let t2 = vsriq_n_u8::<2>(t1, t0);
    let t3 = vsriq_n_u8::<4>(t2, t2);
    let t4 = vshrn_n_u16::<4>(vreinterpretq_u16_u8(t3));
    unsafe { std::mem::transmute::<uint8x8_t, u64>(t4) }
}

#[inline]
#[target_feature(enable = "neon")]
/// Broadcast a u64 bitmask to 64 8-bit elements, where each element is set
/// to the corresponding bit in the input mask.
pub(super) fn _neon_mov_maskz_u8(mask: u64) -> [uint8x16_t; 4] {
    const MASK_1: [u8; 16] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
    const MASK_2: [u8; 16] = [2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3];
    const MASK_3: [u8; 16] = [4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5];
    const MASK_4: [u8; 16] = [6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7];
    const TESTS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];

    let mask = vreinterpretq_u8_u64(vdupq_n_u64(mask));

    let idx0 = unsafe { vld1q_u8(MASK_1.as_ptr()) };
    let idx1 = unsafe { vld1q_u8(MASK_2.as_ptr()) };
    let idx2 = unsafe { vld1q_u8(MASK_3.as_ptr()) };
    let idx3 = unsafe { vld1q_u8(MASK_4.as_ptr()) };

    let bits = unsafe { vld1q_u8(TESTS.as_ptr()) };
    let ones = vdupq_n_u8(1);

    let s0 = vqtbl1q_u8(mask, idx0);
    let s1 = vqtbl1q_u8(mask, idx1);
    let s2 = vqtbl1q_u8(mask, idx2);
    let s3 = vqtbl1q_u8(mask, idx3);

    let d1 = vandq_u8(vtstq_u8(s0, bits), ones);
    let d2 = vandq_u8(vtstq_u8(s1, bits), ones);
    let d3 = vandq_u8(vtstq_u8(s2, bits), ones);
    let d4 = vandq_u8(vtstq_u8(s3, bits), ones);

    [d1, d2, d3, d4]
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
            let result = _neon_nonzero_mask_u8([a, a, a, a]);
            assert_eq!(result, 0);

            let a = _neon_set1_u8(u8::MAX);
            let result = _neon_nonzero_mask_u8([a, a, a, a]);
            assert_eq!(result, u64::MAX);

            const BLOCKS_1: [u32; 4] = [0xFF_00_00_FF, 0xFF_FF_00_FF, 0xFF_00_FF_FF, 0xFF_FF_FF_00];
            const BLOCKS_2: [u32; 4] = [0xFF_FF_00_FF, 0xFF_FF_00_FF, 0xFF_00_FF_FF, 0xFF_FF_FF_00];

            let a = vreinterpretq_u8_u32(_neon_load_u32(BLOCKS_1.as_ptr()));
            let b = vreinterpretq_u8_u32(_neon_load_u32(BLOCKS_2.as_ptr()));

            let result = _neon_nonzero_mask_u8([a, b, a, a]);
            assert_eq!(format!("{:016b}", result as u16), "1110101111011001");
            assert_eq!(
                format!("{:016b}", (result >> 16) as u16),
                "1110101111011101"
            );
            assert_eq!(
                format!("{:016b}", (result >> 32) as u16),
                "1110101111011001"
            );
            assert_eq!(
                format!("{:016b}", (result >> 48) as u16),
                "1110101111011001"
            );

            let result = _neon_nonzero_mask_u8([b, b, a, b]);
            assert_eq!(format!("{:016b}", result as u16), "1110101111011101");
            assert_eq!(
                format!("{:016b}", (result >> 16) as u16),
                "1110101111011101"
            );
            assert_eq!(
                format!("{:016b}", (result >> 32) as u16),
                "1110101111011001"
            );
            assert_eq!(
                format!("{:016b}", (result >> 48) as u16),
                "1110101111011101"
            );
        }
    }

    #[test]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_mov_maskz_u8() {
        unsafe {
            let result = _neon_mov_maskz_u8(u64::MAX);
            let view = std::mem::transmute::<[uint8x16_t; 4], [u8; 64]>(result);
            assert_eq!(view, [1; 64]);

            let result = _neon_mov_maskz_u8(0);
            let view = std::mem::transmute::<[uint8x16_t; 4], [u8; 64]>(result);
            assert_eq!(view, [0; 64]);
        }
    }
}
