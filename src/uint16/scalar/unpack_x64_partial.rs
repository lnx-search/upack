use super::data::*;
use super::polyfill::*;
use super::util::*;

/// Unpack the 1-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(1)` bytes from.
/// - `read_n` must be between no larger than `64`.
pub(crate) unsafe fn from_u1(input: *const u8, read_n: usize) -> [u16x16; 4] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let packed = unsafe { unpack_u1_registers(input) };
    unpack_u8_to_u16_ordered(packed)
}

#[inline]
unsafe fn unpack_u1_registers(input: *const u8) -> [u8x32; 2] {
    let mask: u64 = unsafe { std::ptr::read_unaligned(input.cast()) };

    let ones = _scalar_set1_u8(1);
    let packed1 = _scalar_mov_maskz_u8x32(mask as u32, ones);
    let packed2 = _scalar_mov_maskz_u8x32((mask >> 32) as u32, ones);

    [packed1, packed2]
}

/// Unpack the 2-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(2)` bytes from.
/// - `read_n` must be between no larger than `64`.
pub(crate) unsafe fn from_u2(input: *const u8, read_n: usize) -> [u16x16; 4] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let packed = unsafe { unpack_u2_registers(input, read_n) };
    unpack_u8_to_u16_ordered(packed)
}

#[inline]
unsafe fn unpack_u2_registers(input: *const u8, read_n: usize) -> [u8x32; 2] {
    let hi_offset = read_n.div_ceil(8);
    let mask1: u64 = unsafe { std::ptr::read_unaligned(input.add(0).cast()) };
    let mask2: u64 = unsafe { std::ptr::read_unaligned(input.add(hi_offset).cast()) };

    let bit = _scalar_set1_u8(0b01);
    let lo_bits_packed1 = _scalar_mov_maskz_u8x32(mask1 as u32, bit);
    let lo_bits_packed2 = _scalar_mov_maskz_u8x32((mask1 >> 32) as u32, bit);

    let bit = _scalar_set1_u8(0b10);
    let hi_bits_packed1 = _scalar_mov_maskz_u8x32(mask2 as u32, bit);
    let hi_bits_packed2 = _scalar_mov_maskz_u8x32((mask2 >> 32) as u32, bit);

    let two_bits1 = _scalar_or_u8x32(hi_bits_packed1, lo_bits_packed1);
    let two_bits2 = _scalar_or_u8x32(hi_bits_packed2, lo_bits_packed2);
    [two_bits1, two_bits2]
}

/// Unpack the 3-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(3)` bytes from.
/// - `read_n` must be between no larger than `64`.
pub(crate) unsafe fn from_u3(input: *const u8, read_n: usize) -> [u16x16; 4] {
    let packed = unsafe { unpack_u3_registers(input, read_n) };
    unpack_u8_to_u16_ordered(packed)
}

#[inline]
unsafe fn unpack_u3_registers(input: *const u8, read_n: usize) -> [u8x32; 2] {
    let step = read_n.div_ceil(8);

    let mask1: u64 = unsafe { std::ptr::read_unaligned(input.add(0).cast()) };
    let mask2: u64 = unsafe { std::ptr::read_unaligned(input.add(step).cast()) };
    let mask3: u64 = unsafe { std::ptr::read_unaligned(input.add(step * 2).cast()) };

    let bit = _scalar_set1_u8(0b001);
    let b0_bits_packed1 = _scalar_mov_maskz_u8x32(mask1 as u32, bit);
    let b0_bits_packed2 = _scalar_mov_maskz_u8x32((mask1 >> 32) as u32, bit);

    let bit = _scalar_set1_u8(0b010);
    let b1_bits_packed1 = _scalar_mov_maskz_u8x32(mask2 as u32, bit);
    let b1_bits_packed2 = _scalar_mov_maskz_u8x32((mask2 >> 32) as u32, bit);

    let bit = _scalar_set1_u8(0b100);
    let b2_bits_packed1 = _scalar_mov_maskz_u8x32(mask3 as u32, bit);
    let b2_bits_packed2 = _scalar_mov_maskz_u8x32((mask3 >> 32) as u32, bit);

    let mut three_bits1 = _scalar_or_u8x32(b1_bits_packed1, b0_bits_packed1);
    let mut three_bits2 = _scalar_or_u8x32(b1_bits_packed2, b0_bits_packed2);
    three_bits1 = _scalar_or_u8x32(three_bits1, b2_bits_packed1);
    three_bits2 = _scalar_or_u8x32(three_bits2, b2_bits_packed2);

    [three_bits1, three_bits2]
}

/// Unpack the 4-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(4)` bytes from.
/// - `read_n` must be between no larger than `64`.
pub(crate) unsafe fn from_u4(input: *const u8, read_n: usize) -> [u16x16; 4] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let packed = unsafe { unpack_u4_registers(input) };
    unpack_u8_to_u16_ordered(packed)
}

#[inline]
pub(super) unsafe fn unpack_u4_registers(input: *const u8) -> [u8x32; 2] {
    let packed = unsafe { _scalar_load_u8x32(input.cast()) };

    let mut d1 = u8x32::ZERO;
    let mut d2 = u8x32::ZERO;

    for i in 0..16 {
        d1[i * 2] = packed[i] & 0x0F;
        d1[i * 2 + 1] = (packed[i] >> 4) & 0x0F;
    }

    for i in 0..16 {
        d2[i * 2] = packed[i + 16] & 0x0F;
        d2[i * 2 + 1] = (packed[i + 16] >> 4) & 0x0F;
    }

    [d1, d2]
}

/// Unpack the 5-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(5)` bytes from.
/// - `read_n` must be between no larger than `64`.
pub(crate) unsafe fn from_u5(input: *const u8, read_n: usize) -> [u16x16; 4] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let packed = unsafe { unpack_u5_registers(input, read_n) };
    unpack_u8_to_u16_ordered(packed)
}

#[inline]
unsafe fn unpack_u5_registers(input: *const u8, read_n: usize) -> [u8x32; 2] {
    let offset = read_n.div_ceil(2);
    let lo_bits = unsafe { unpack_u4_registers(input.add(0)) };
    let mut hi_bits = unsafe { unpack_u1_registers(input.add(offset)) };
    hi_bits = slli_u8x32::<4, 2>(hi_bits);
    or_u8x32_all(hi_bits, lo_bits)
}

/// Unpack the 6-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(6)` bytes from.
/// - `read_n` must be between no larger than `64`.
pub(crate) unsafe fn from_u6(input: *const u8, read_n: usize) -> [u16x16; 4] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let packed = unsafe { unpack_u6_registers(input, read_n) };
    unpack_u8_to_u16_ordered(packed)
}

#[inline]
unsafe fn unpack_u6_registers(input: *const u8, read_n: usize) -> [u8x32; 2] {
    let offset = read_n.div_ceil(2);
    let lo_bits = unsafe { unpack_u4_registers(input.add(0)) };
    let mut hi_bits = unsafe { unpack_u2_registers(input.add(offset), read_n) };
    hi_bits = slli_u8x32::<4, 2>(hi_bits);
    or_u8x32_all(hi_bits, lo_bits)
}

/// Unpack the 7-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(7)` bytes from.
/// - `read_n` must be between no larger than `64`.
pub(crate) unsafe fn from_u7(input: *const u8, read_n: usize) -> [u16x16; 4] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let packed = unsafe { unpack_u7_registers(input, read_n) };
    unpack_u8_to_u16_ordered(packed)
}

#[inline]
unsafe fn unpack_u7_registers(input: *const u8, read_n: usize) -> [u8x32; 2] {
    let offset = read_n.div_ceil(2);
    let lo_bits = unsafe { unpack_u4_registers(input.add(0)) };
    let mut hi_bits = unsafe { unpack_u3_registers(input.add(offset), read_n) };
    hi_bits = slli_u8x32::<4, 2>(hi_bits);
    or_u8x32_all(hi_bits, lo_bits)
}

/// Unpack the 8-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(8)` bytes from.
/// - `read_n` must be between no larger than `64`.
pub(crate) unsafe fn from_u8(input: *const u8, read_n: usize) -> [u16x16; 4] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let packed = unsafe { load_u8x32x2(input) };
    unpack_u8_to_u16_ordered(packed)
}

/// Unpack the 9-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(9)` bytes from.
/// - `read_n` must be between no larger than `64`.
pub(crate) unsafe fn from_u9(input: *const u8, read_n: usize) -> [u16x16; 4] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let lo_bits = unsafe { load_u8x32x2(input.add(0)) };
    let lo_bits = unpack_u8_to_u16_ordered(lo_bits);

    let offset = read_n;
    let hi_bits = unsafe { unpack_u1_registers(input.add(offset)) };
    let mut hi_bits = unpack_u8_to_u16_ordered(hi_bits);
    hi_bits = slli_u16x16::<8, 4>(hi_bits);

    or_u16x16_all(hi_bits, lo_bits)
}

/// Unpack the 10-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(10)` bytes from.
/// - `read_n` must be between no larger than `64`.
pub(crate) unsafe fn from_u10(input: *const u8, read_n: usize) -> [u16x16; 4] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let lo_bits = unsafe { load_u8x32x2(input.add(0)) };
    let lo_bits = unpack_u8_to_u16_ordered(lo_bits);

    let offset = read_n;
    let hi_bits = unsafe { unpack_u2_registers(input.add(offset), read_n) };
    let mut hi_bits = unpack_u8_to_u16_ordered(hi_bits);
    hi_bits = slli_u16x16::<8, 4>(hi_bits);

    or_u16x16_all(hi_bits, lo_bits)
}

/// Unpack the 11-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(11)` bytes from.
/// - `read_n` must be between no larger than `64`.
pub(crate) unsafe fn from_u11(input: *const u8, read_n: usize) -> [u16x16; 4] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let lo_bits = unsafe { load_u8x32x2(input.add(0)) };
    let lo_bits = unpack_u8_to_u16_ordered(lo_bits);

    let offset = read_n;
    let hi_bits = unsafe { unpack_u3_registers(input.add(offset), read_n) };
    let mut hi_bits = unpack_u8_to_u16_ordered(hi_bits);
    hi_bits = slli_u16x16::<8, 4>(hi_bits);

    or_u16x16_all(hi_bits, lo_bits)
}

/// Unpack the 12-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(12)` bytes from.
/// - `read_n` must be between no larger than `64`.
pub(crate) unsafe fn from_u12(input: *const u8, read_n: usize) -> [u16x16; 4] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let lo_bits = unsafe { load_u8x32x2(input.add(0)) };
    let lo_bits = unpack_u8_to_u16_ordered(lo_bits);

    let offset = read_n;
    let hi_bits = unsafe { unpack_u4_registers(input.add(offset)) };
    let mut hi_bits = unpack_u8_to_u16_ordered(hi_bits);
    hi_bits = slli_u16x16::<8, 4>(hi_bits);

    or_u16x16_all(hi_bits, lo_bits)
}

/// Unpack the 13-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(13)` bytes from.
/// - `read_n` must be between no larger than `64`.
pub(crate) unsafe fn from_u13(input: *const u8, read_n: usize) -> [u16x16; 4] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let lo_bits = unsafe { load_u8x32x2(input.add(0)) };
    let lo_bits = unpack_u8_to_u16_ordered(lo_bits);

    let offset = read_n;
    let hi_bits = unsafe { unpack_u5_registers(input.add(offset), read_n) };
    let mut hi_bits = unpack_u8_to_u16_ordered(hi_bits);
    hi_bits = slli_u16x16::<8, 4>(hi_bits);

    or_u16x16_all(hi_bits, lo_bits)
}

/// Unpack the 14-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(14)` bytes from.
/// - `read_n` must be between no larger than `64`.
pub(crate) unsafe fn from_u14(input: *const u8, read_n: usize) -> [u16x16; 4] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let lo_bits = unsafe { load_u8x32x2(input.add(0)) };
    let lo_bits = unpack_u8_to_u16_ordered(lo_bits);

    let offset = read_n;
    let hi_bits = unsafe { unpack_u6_registers(input.add(offset), read_n) };
    let mut hi_bits = unpack_u8_to_u16_ordered(hi_bits);
    hi_bits = slli_u16x16::<8, 4>(hi_bits);

    or_u16x16_all(hi_bits, lo_bits)
}

/// Unpack the 15-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(15)` bytes from.
/// - `read_n` must be between no larger than `64`.
pub(crate) unsafe fn from_u15(input: *const u8, read_n: usize) -> [u16x16; 4] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    let lo_bits = unsafe { load_u8x32x2(input.add(0)) };
    let lo_bits = unpack_u8_to_u16_ordered(lo_bits);

    let offset = read_n;
    let hi_bits = unsafe { unpack_u7_registers(input.add(offset), read_n) };
    let mut hi_bits = unpack_u8_to_u16_ordered(hi_bits);
    hi_bits = slli_u16x16::<8, 4>(hi_bits);

    or_u16x16_all(hi_bits, lo_bits)
}

/// Unpack the 16-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(16)` bytes from.
/// - `read_n` must be between no larger than `64`.
pub(crate) unsafe fn from_u16(input: *const u8, read_n: usize) -> [u16x16; 4] {
    debug_assert!(read_n <= 64, "read_n must be less than or equal to 64.");
    unsafe { load_u16x16x4(input.add(0)) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::X64;
    use crate::uint16::scalar::pack_x64_partial::*;
    use crate::uint16::{X128_MAX_OUTPUT_LEN, max_compressed_size};

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
    fn test_saturated_unpack(
        #[case] bit_len: u8,
        #[case] unpacker: unsafe fn(*const u8, usize) -> [u16x16; 4],
    ) {
        let saturated_bytes = [u8::MAX; X128_MAX_OUTPUT_LEN / 2];

        let unpacked = unsafe { unpacker(saturated_bytes.as_ptr(), X64) };
        let unpacked = unsafe { std::mem::transmute::<[u16x16; 4], [u16; X64]>(unpacked) };

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
    fn test_pack_unpack(
        #[case] bit_len: u8,
        #[case] packer: unsafe fn(*mut u8, [u16x16; 4], usize),
        #[case] unpacker: unsafe fn(*const u8, usize) -> [u16x16; 4],
    ) {
        fastrand::seed(5876358762523525);

        let max_value = (2u64.pow(bit_len as u32) - 1) as u16;

        let mut values = [0; X64];
        for value in values.iter_mut() {
            *value = fastrand::u16(0..max_value);
        }
        let data = load_u16x64(&values);

        let mut packed = [0; X128_MAX_OUTPUT_LEN / 2];
        unsafe { packer(packed.as_mut_ptr(), data, X64) };

        let unpacked = unsafe { unpacker(packed.as_ptr(), X64) };
        let unpacked = unsafe { std::mem::transmute::<[u16x16; 4], [u16; X64]>(unpacked) };
        assert_eq!(unpacked, values);
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
    fn test_pack_unpack_length_permutations(
        #[case] bit_len: u8,
        #[case] packer: unsafe fn(*mut u8, [u16x16; 4], usize),
        #[case] unpacker: unsafe fn(*const u8, usize) -> [u16x16; 4],
    ) {
        fastrand::seed(5876358762523525);

        let max_value = (2u64.pow(bit_len as u32) - 1) as u16;

        for length in 0..X64 {
            let mut values = [0; X64];
            for value in values.iter_mut() {
                *value = fastrand::u16(0..max_value);
            }
            let data = load_u16x64(&values);

            let mut packed = [0; X128_MAX_OUTPUT_LEN / 2];
            unsafe { packer(packed.as_mut_ptr(), data, length) };
            packed[max_compressed_size::<X64>(bit_len as usize)..].fill(0);

            let unpacked = unsafe { unpacker(packed.as_ptr(), length) };
            let unpacked = unsafe { std::mem::transmute::<[u16x16; 4], [u16; X64]>(unpacked) };
            assert_eq!(unpacked[..length], values[..length], "length:{length}");
        }
    }
}
