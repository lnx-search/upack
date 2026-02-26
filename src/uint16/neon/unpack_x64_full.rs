use std::arch::aarch64::*;

use super::data::*;
use super::polyfill::*;
use super::util::*;

#[inline]
#[target_feature(enable = "neon")]
/// Unpack the 1-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(1)` bytes from.
pub unsafe fn from_u1(input: *const u8) -> [uint16x8_t; 8] {
    let packed = unsafe { unpack_u1_registers(input) };
    unpack_u8_to_u16_unordered(packed)
}

#[inline]
#[target_feature(enable = "neon")]
/// Unpack eight registers containing 64 8-bit elements from a 1-bit bitmap provided
/// by `input`.
unsafe fn unpack_u1_registers(input: *const u8) -> [uint8x16_t; 4] {
    let mask: u64 = unsafe { std::ptr::read_unaligned(input.add(0).cast()) };
    _neon_mov_maskz_u8(mask)
}

#[inline]
#[target_feature(enable = "neon")]
/// Unpack the 2-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(2)` bytes from.
pub unsafe fn from_u2(input: *const u8) -> [uint16x8_t; 8] {
    let packed = unsafe { unpack_u2_registers(input) };
    unpack_u8_to_u16_unordered(packed)
}

#[inline]
#[target_feature(enable = "neon")]
/// Unpack eight registers containing 64 8-bit elements from a 2-bit bitmap provided
/// by `input`.
unsafe fn unpack_u2_registers(input: *const u8) -> [uint8x16_t; 4] {
    let packed = unsafe { _neon_load_u8(input) };
    unpack_u2_to_u8_unordered(packed)
}

#[inline]
#[target_feature(enable = "neon")]
/// Unpack the 3-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(3)` bytes from.
pub unsafe fn from_u3(input: *const u8) -> [uint16x8_t; 8] {
    let packed = unsafe { unpack_u3_registers(input) };
    unpack_u8_to_u16_unordered(packed)
}

#[inline]
#[target_feature(enable = "neon")]
/// Unpack eight registers containing 64 8-bit elements from a 3-bit bitmap provided
/// by `input`.
unsafe fn unpack_u3_registers(input: *const u8) -> [uint8x16_t; 4] {
    let packed_2bit = unsafe { _neon_load_u8(input.add(0)) };
    let lo_2bits = unpack_u2_to_u8_unordered(packed_2bit);

    let hi_bitmask: u64 = unsafe { std::ptr::read_unaligned(input.add(16).cast()) };

    let mut hi_1bits = _neon_mov_maskz_u8(hi_bitmask);
    hi_1bits = slli_u8::<2, 4>(hi_1bits);

    or_u8_all(hi_1bits, lo_2bits)
}

#[target_feature(enable = "neon")]
/// Unpack the 4-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(4)` bytes from.
pub unsafe fn from_u4(input: *const u8) -> [uint16x8_t; 8] {
    let packed = unsafe { unpack_u4_registers(input) };
    unpack_u8_to_u16_unordered(packed)
}

#[inline]
#[target_feature(enable = "neon")]
/// Unpack eight registers containing 8 32-bit elements from a 4-bit nibbles provided
/// by `input`.
unsafe fn unpack_u4_registers(input: *const u8) -> [uint8x16_t; 4] {
    let packed1 = unsafe { _neon_load_u8(input.add(0)) };
    let packed2 = unsafe { _neon_load_u8(input.add(16)) };
    unpack_u4_to_u8_unordered([packed1, packed2])
}

#[inline]
#[target_feature(enable = "neon")]
/// Unpack the 5-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(5)` bytes from.
pub unsafe fn from_u5(input: *const u8) -> [uint16x8_t; 8] {
    let packed = unsafe { unpack_u5_registers(input) };
    unpack_u8_to_u16_unordered(packed)
}

#[inline]
#[target_feature(enable = "neon")]
/// Unpack eight registers containing 8 32-bit elements from a 5-bit integers provided
/// by `input`.
unsafe fn unpack_u5_registers(input: *const u8) -> [uint8x16_t; 4] {
    let lo_bits = unsafe { unpack_u4_registers(input.add(0)) };
    let mut hi_bits = unsafe { unpack_u1_registers(input.add(32)) };
    hi_bits = slli_u8::<4, 4>(hi_bits);
    or_u8_all(hi_bits, lo_bits)
}

#[inline]
#[target_feature(enable = "neon")]
/// Unpack the 6-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(6)` bytes from.
pub unsafe fn from_u6(input: *const u8) -> [uint16x8_t; 8] {
    let packed = unsafe { unpack_u6_registers(input) };
    unpack_u8_to_u16_unordered(packed)
}

#[inline]
#[target_feature(enable = "neon")]
/// Unpack eight registers containing 8 32-bit elements from a 6-bit integers provided
/// by `input`.
unsafe fn unpack_u6_registers(input: *const u8) -> [uint8x16_t; 4] {
    let lo_bits = unsafe { unpack_u4_registers(input.add(0)) };
    let mut hi_bits = unsafe { unpack_u2_registers(input.add(32)) };
    hi_bits = slli_u8::<4, 4>(hi_bits);
    or_u8_all(hi_bits, lo_bits)
}

#[inline]
#[target_feature(enable = "neon")]
/// Unpack the 7-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(7)` bytes from.
pub unsafe fn from_u7(input: *const u8) -> [uint16x8_t; 8] {
    let packed = unsafe { unpack_u7_registers(input) };
    unpack_u8_to_u16_unordered(packed)
}

#[inline]
#[target_feature(enable = "neon")]
/// Unpack eight registers containing 8 32-bit elements from a 7-bit integers provided
/// by `input`.
unsafe fn unpack_u7_registers(input: *const u8) -> [uint8x16_t; 4] {
    let lo_bits = unsafe { unpack_u4_registers(input.add(0)) };
    let mut hi_bits = unsafe { unpack_u3_registers(input.add(32)) };
    hi_bits = slli_u8::<4, 4>(hi_bits);
    or_u8_all(hi_bits, lo_bits)
}

#[inline]
#[target_feature(enable = "neon")]
/// Unpack the 8-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(8)` bytes from.
pub unsafe fn from_u8(input: *const u8) -> [uint16x8_t; 8] {
    let packed = unsafe { load_u8x16x4(input) };
    unpack_u8_to_u16_unordered(packed)
}

#[inline]
#[target_feature(enable = "neon")]
/// Unpack the 9-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(9)` bytes from.
pub unsafe fn from_u9(input: *const u8) -> [uint16x8_t; 8] {
    let lo_bits = unsafe { load_u8x16x4(input.add(0)) };
    let lo_bits = unpack_u8_to_u16_unordered(lo_bits);

    let hi_bits = unsafe { unpack_u1_registers(input.add(64)) };
    let mut hi_bits = unpack_u8_to_u16_unordered(hi_bits);
    hi_bits = slli_u16::<8, 8>(hi_bits);

    or_u16_all(hi_bits, lo_bits)
}

#[inline]
#[target_feature(enable = "neon")]
/// Unpack the 10-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(10)` bytes from.
pub unsafe fn from_u10(input: *const u8) -> [uint16x8_t; 8] {
    let lo_bits = unsafe { load_u8x16x4(input.add(0)) };
    let lo_bits = unpack_u8_to_u16_unordered(lo_bits);

    let hi_bits = unsafe { unpack_u2_registers(input.add(64)) };
    let mut hi_bits = unpack_u8_to_u16_unordered(hi_bits);
    hi_bits = slli_u16::<8, 8>(hi_bits);

    or_u16_all(hi_bits, lo_bits)
}

#[inline]
#[target_feature(enable = "neon")]
/// Unpack the 11-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(11)` bytes from.
pub unsafe fn from_u11(input: *const u8) -> [uint16x8_t; 8] {
    let lo_bits = unsafe { load_u8x16x4(input.add(0)) };
    let lo_bits = unpack_u8_to_u16_unordered(lo_bits);

    let hi_bits = unsafe { unpack_u3_registers(input.add(64)) };
    let mut hi_bits = unpack_u8_to_u16_unordered(hi_bits);
    hi_bits = slli_u16::<8, 8>(hi_bits);

    or_u16_all(hi_bits, lo_bits)
}

#[inline]
#[target_feature(enable = "neon")]
/// Unpack the 12-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(12)` bytes from.
pub unsafe fn from_u12(input: *const u8) -> [uint16x8_t; 8] {
    let lo_bits = unsafe { load_u8x16x4(input.add(0)) };
    let lo_bits = unpack_u8_to_u16_unordered(lo_bits);

    let hi_bits = unsafe { unpack_u4_registers(input.add(64)) };
    let mut hi_bits = unpack_u8_to_u16_unordered(hi_bits);
    hi_bits = slli_u16::<8, 8>(hi_bits);

    or_u16_all(hi_bits, lo_bits)
}

#[inline]
#[target_feature(enable = "neon")]
/// Unpack the 13-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(13)` bytes from.
pub unsafe fn from_u13(input: *const u8) -> [uint16x8_t; 8] {
    let lo_bits = unsafe { load_u8x16x4(input.add(0)) };
    let lo_bits = unpack_u8_to_u16_unordered(lo_bits);

    let hi_bits = unsafe { unpack_u5_registers(input.add(64)) };
    let mut hi_bits = unpack_u8_to_u16_unordered(hi_bits);
    hi_bits = slli_u16::<8, 8>(hi_bits);

    or_u16_all(hi_bits, lo_bits)
}

#[inline]
#[target_feature(enable = "neon")]
/// Unpack the 14-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(14)` bytes from.
pub unsafe fn from_u14(input: *const u8) -> [uint16x8_t; 8] {
    let lo_bits = unsafe { load_u8x16x4(input.add(0)) };
    let lo_bits = unpack_u8_to_u16_unordered(lo_bits);

    let hi_bits = unsafe { unpack_u6_registers(input.add(64)) };
    let mut hi_bits = unpack_u8_to_u16_unordered(hi_bits);
    hi_bits = slli_u16::<8, 8>(hi_bits);

    or_u16_all(hi_bits, lo_bits)
}

#[inline]
#[target_feature(enable = "neon")]
/// Unpack the 15-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(15)` bytes from.
pub unsafe fn from_u15(input: *const u8) -> [uint16x8_t; 8] {
    let lo_bits = unsafe { load_u8x16x4(input.add(0)) };
    let lo_bits = unpack_u8_to_u16_unordered(lo_bits);

    let hi_bits = unsafe { unpack_u7_registers(input.add(64)) };
    let mut hi_bits = unpack_u8_to_u16_unordered(hi_bits);
    hi_bits = slli_u16::<8, 8>(hi_bits);

    or_u16_all(hi_bits, lo_bits)
}

#[inline]
#[target_feature(enable = "neon")]
/// Unpack the 16-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(16)` bytes from.
pub unsafe fn from_u16(input: *const u8) -> [uint16x8_t; 8] {
    unsafe { load_u16x8x8(input.add(0)) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::X64;
    use crate::uint16::X128_MAX_OUTPUT_LEN;
    use crate::uint16::neon::pack_x64_full::*;

    #[rstest::rstest]
    #[case(1, from_u1)]
    #[case(2, from_u2)]
    #[case(3, from_u3)]
    #[case(4, from_u4)]
    #[case(5, from_u5)]
    #[case(6, from_u6)]
    #[case(7, from_u7)]
    #[case(8, from_u8)]
    #[case(9, from_u9)]
    #[case(10, from_u10)]
    #[case(11, from_u11)]
    #[case(12, from_u12)]
    #[case(13, from_u13)]
    #[case(14, from_u14)]
    #[case(15, from_u15)]
    #[case(16, from_u16)]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_saturated_unpack(
        #[case] bit_len: u8,
        #[case] unpacker: unsafe fn(*const u8) -> [uint16x8_t; 8],
    ) {
        let saturated_bytes = [u8::MAX; X128_MAX_OUTPUT_LEN / 2];

        let unpacked = unsafe { unpacker(saturated_bytes.as_ptr()) };
        let unpacked = unsafe { std::mem::transmute::<[uint16x8_t; 8], [u16; X64]>(unpacked) };

        let expected_value = (2u64.pow(bit_len as u32) - 1) as u16;
        assert_eq!(unpacked, [expected_value; X64]);
    }

    #[rstest::rstest]
    #[case(1, to_u1, from_u1)]
    #[case(2, to_u2, from_u2)]
    #[case(3, to_u3, from_u3)]
    #[case(4, to_u4, from_u4)]
    #[case(5, to_u5, from_u5)]
    #[case(6, to_u6, from_u6)]
    #[case(7, to_u7, from_u7)]
    #[case(8, to_u8, from_u8)]
    #[case(9, to_u9, from_u9)]
    #[case(10, to_u10, from_u10)]
    #[case(11, to_u11, from_u11)]
    #[case(12, to_u12, from_u12)]
    #[case(13, to_u13, from_u13)]
    #[case(14, to_u14, from_u14)]
    #[case(15, to_u15, from_u15)]
    #[case(16, to_u16, from_u16)]
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_pack_unpack(
        #[case] bit_len: u8,
        #[case] packer: unsafe fn(*mut u8, [uint16x8_t; 8]),
        #[case] unpacker: unsafe fn(*const u8) -> [uint16x8_t; 8],
    ) {
        fastrand::seed(5876358762523525);

        let max_value = (2u64.pow(bit_len as u32) - 1) as u16;

        let mut values = [0; X64];
        for value in values.iter_mut() {
            *value = fastrand::u16(0..max_value);
        }
        let data = unsafe { load_u16x64(&values) };

        let mut packed = [0; X128_MAX_OUTPUT_LEN / 2];
        unsafe { packer(packed.as_mut_ptr(), data) };

        let unpacked = unsafe { unpacker(packed.as_ptr()) };
        let unpacked = unsafe { std::mem::transmute::<[uint16x8_t; 8], [u16; X64]>(unpacked) };
        assert_eq!(unpacked, values);
    }
}
