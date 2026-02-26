use super::data::*;
use super::polyfill::*;
use super::util::*;

/// Bitpack the provided block of integers to 1-bit elements.
pub(crate) unsafe fn to_u1(out: *mut u8, block: [u16x16; 4], _pack_n: usize) {
    let partially_packed = pack_u16_to_u8_ordered(block);
    unsafe { pack_u1_registers(out, partially_packed) }
}

#[inline]
/// Pack four registers containing 32 8-bit elements each into a 1-bit
/// bitmap and write to `out`.
///
/// Any non-zero value will be treated as a set bit.
unsafe fn pack_u1_registers(out: *mut u8, data: [u8x32; 2]) {
    let [d1, d2] = data;

    let cmp1 = _scalar_slli_u8x32::<7>(d1);
    let cmp2 = _scalar_slli_u8x32::<7>(d2);

    let mask1 = _scalar_mask_u8x32(cmp1);
    let mask2 = _scalar_mask_u8x32(cmp2);

    let merged_mask = ((mask2 as u64) << 32) | mask1 as u64;
    // We assume LE endianness
    unsafe { std::ptr::write_unaligned(out.cast(), merged_mask) };
}

/// Bitpack the provided block of integers to 2-bit elements.
pub(crate) unsafe fn to_u2(out: *mut u8, block: [u16x16; 4], pack_n: usize) {
    let partially_packed = pack_u16_to_u8_ordered(block);
    unsafe { pack_u2_registers(out, partially_packed, pack_n) }
}

#[inline]
/// Pack four registers containing 32 8-bit elements each into a 2-bit
/// bitmap and write to `out`.
unsafe fn pack_u2_registers(out: *mut u8, data: [u8x32; 2], pack_n: usize) {
    let [d1, d2] = data;

    let lo_cmp1 = _scalar_slli_u8x32::<7>(d1);
    let hi_cmp1 = _scalar_slli_u8x32::<6>(d1);
    let lo_mask1 = _scalar_mask_u8x32(lo_cmp1);
    let hi_mask1 = _scalar_mask_u8x32(hi_cmp1);

    let lo_cmp2 = _scalar_slli_u8x32::<7>(d2);
    let hi_cmp2 = _scalar_slli_u8x32::<6>(d2);
    let lo_mask2 = _scalar_mask_u8x32(lo_cmp2);
    let hi_mask2 = _scalar_mask_u8x32(hi_cmp2);

    let lo_merged_mask = ((lo_mask2 as u64) << 32) | lo_mask1 as u64;
    let hi_merged_mask = ((hi_mask2 as u64) << 32) | hi_mask1 as u64;

    let hi_offset = pack_n.div_ceil(8);
    unsafe { std::ptr::write_unaligned(out.add(0).cast(), lo_merged_mask) };
    unsafe { std::ptr::write_unaligned(out.add(hi_offset).cast(), hi_merged_mask) };
}

/// Bitpack the provided block of integers to 3-bit elements.
pub(crate) unsafe fn to_u3(out: *mut u8, block: [u16x16; 4], pack_n: usize) {
    let partially_packed = pack_u16_to_u8_ordered(block);
    unsafe { pack_u3_registers(out, partially_packed, pack_n) }
}

#[inline]
/// Pack four registers containing 32 8-bit elements each into a 3-bit
/// bitmap and write to `out`.
unsafe fn pack_u3_registers(out: *mut u8, data: [u8x32; 2], pack_n: usize) {
    let [d1, d2] = data;

    let b0_cmp1 = _scalar_slli_u8x32::<7>(d1);
    let b1_cmp1 = _scalar_slli_u8x32::<6>(d1);
    let b2_cmp1 = _scalar_slli_u8x32::<5>(d1);
    let b0_mask1 = _scalar_mask_u8x32(b0_cmp1);
    let b1_mask1 = _scalar_mask_u8x32(b1_cmp1);
    let b2_mask1 = _scalar_mask_u8x32(b2_cmp1);

    let b0_cmp2 = _scalar_slli_u8x32::<7>(d2);
    let b1_cmp2 = _scalar_slli_u8x32::<6>(d2);
    let b2_cmp2 = _scalar_slli_u8x32::<5>(d2);
    let b0_mask2 = _scalar_mask_u8x32(b0_cmp2);
    let b1_mask2 = _scalar_mask_u8x32(b1_cmp2);
    let b2_mask2 = _scalar_mask_u8x32(b2_cmp2);

    let b0_merged_mask = ((b0_mask2 as u64) << 32) | b0_mask1 as u64;
    let b1_merged_mask = ((b1_mask2 as u64) << 32) | b1_mask1 as u64;
    let b2_merged_mask = ((b2_mask2 as u64) << 32) | b2_mask1 as u64;

    let step = pack_n.div_ceil(8);
    unsafe { std::ptr::write_unaligned(out.add(0).cast(), b0_merged_mask) };
    unsafe { std::ptr::write_unaligned(out.add(step).cast(), b1_merged_mask) };
    unsafe { std::ptr::write_unaligned(out.add(step * 2).cast(), b2_merged_mask) };
}

/// Bitpack the provided block of integers to 4-bit elements.
pub(crate) unsafe fn to_u4(out: *mut u8, block: [u16x16; 4], _pack_n: usize) {
    let partially_packed = pack_u16_to_u8_ordered(block);
    unsafe { pack_u4_registers(out, partially_packed) }
}

#[inline]
/// Pack four registers containing 32 8-bit elements each into a 4-bit
/// bitmap and write to `out`.
pub(super) unsafe fn pack_u4_registers(out: *mut u8, data: [u8x32; 2]) {
    let [d1, d2] = data;

    let mut packed = u8x32::ZERO;
    for i in 0..16 {
        packed[i] = (d1[i * 2] & 0x0F) | (d1[i * 2 + 1] << 4);
    }

    for i in 0..16 {
        packed[i + 16] = (d2[i * 2] & 0x0F) | (d2[i * 2 + 1] << 4);
    }

    unsafe { _scalar_store_u8x32(out, packed) };
}

/// Bitpack the provided block of integers to 5-bit elements.
pub(crate) unsafe fn to_u5(out: *mut u8, block: [u16x16; 4], pack_n: usize) {
    let partially_packed = pack_u16_to_u8_ordered(block);
    unsafe { pack_u5_registers(out, partially_packed, pack_n) }
}

#[inline]
/// Pack four registers containing 32 8-bit elements each into a 5-bit
/// bitmap and write to `out`.
unsafe fn pack_u5_registers(out: *mut u8, data: [u8x32; 2], pack_n: usize) {
    let mask = _scalar_set1_u8(0b1111);
    let masked = and_u8x32(data, mask);
    unsafe { pack_u4_registers(out, masked) };

    // 4bit * 64 / 8-bits per byte.
    let offset = pack_n.div_ceil(2);
    let remaining = srli_u8x32::<4, 2>(data);
    unsafe { pack_u1_registers(out.add(offset), remaining) };
}

/// Bitpack the provided block of integers to 6-bit elements.
pub(crate) unsafe fn to_u6(out: *mut u8, block: [u16x16; 4], pack_n: usize) {
    let partially_packed = pack_u16_to_u8_ordered(block);
    unsafe { pack_u6_registers(out, partially_packed, pack_n) }
}

#[inline]
/// Pack four registers containing 32 8-bit elements each into a 6-bit
/// bitmap and write to `out`.
unsafe fn pack_u6_registers(out: *mut u8, data: [u8x32; 2], pack_n: usize) {
    let mask = _scalar_set1_u8(0b1111);
    let masked = and_u8x32(data, mask);
    unsafe { pack_u4_registers(out, masked) };

    // 4bit * 64 / 8-bits per byte.
    let offset = pack_n.div_ceil(2);
    let remaining = srli_u8x32::<4, 2>(data);
    unsafe { pack_u2_registers(out.add(offset), remaining, pack_n) };
}

/// Bitpack the provided block of integers to 7-bit elements.
pub(crate) unsafe fn to_u7(out: *mut u8, block: [u16x16; 4], pack_n: usize) {
    let partially_packed = pack_u16_to_u8_ordered(block);
    unsafe { pack_u7_registers(out, partially_packed, pack_n) }
}

#[inline]
/// Pack four registers containing 32 8-bit elements each into a 7-bit
/// bitmap and write to `out`.
unsafe fn pack_u7_registers(out: *mut u8, data: [u8x32; 2], pack_n: usize) {
    let mask = _scalar_set1_u8(0b1111);
    let masked = and_u8x32(data, mask);
    unsafe { pack_u4_registers(out, masked) };

    // 4bit * 64 / 8-bits per byte.
    let offset = pack_n.div_ceil(2);
    let remaining = srli_u8x32::<4, 2>(data);
    unsafe { pack_u3_registers(out.add(offset), remaining, pack_n) };
}

/// Bitpack the provided block of integers to 8-bit elements.
pub(crate) unsafe fn to_u8(out: *mut u8, block: [u16x16; 4], _pack_n: usize) {
    let partially_packed = pack_u16_to_u8_ordered(block);
    unsafe { store_u8x32x2(out, partially_packed) }
}

/// Bitpack the provided block of integers to 9-bit elements.
pub(crate) unsafe fn to_u9(out: *mut u8, block: [u16x16; 4], pack_n: usize) {
    let (hi, lo) = split_u16_ordered(block);
    unsafe { store_u8x32x2(out.add(0), lo) };
    unsafe { pack_u1_registers(out.add(pack_n), hi) }
}

/// Bitpack the provided block of integers to 10-bit elements.
pub(crate) unsafe fn to_u10(out: *mut u8, block: [u16x16; 4], pack_n: usize) {
    let (hi, lo) = split_u16_ordered(block);
    unsafe { store_u8x32x2(out.add(0), lo) };
    unsafe { pack_u2_registers(out.add(pack_n), hi, pack_n) }
}

/// Bitpack the provided block of integers to 11-bit elements.
pub(crate) unsafe fn to_u11(out: *mut u8, block: [u16x16; 4], pack_n: usize) {
    let (hi, lo) = split_u16_ordered(block);
    unsafe { store_u8x32x2(out.add(0), lo) };
    unsafe { pack_u3_registers(out.add(pack_n), hi, pack_n) }
}

/// Bitpack the provided block of integers to 12-bit elements.
pub(crate) unsafe fn to_u12(out: *mut u8, block: [u16x16; 4], pack_n: usize) {
    let (hi, lo) = split_u16_ordered(block);
    unsafe { store_u8x32x2(out.add(0), lo) };
    unsafe { pack_u4_registers(out.add(pack_n), hi) }
}

/// Bitpack the provided block of integers to 13-bit elements.
pub(crate) unsafe fn to_u13(out: *mut u8, block: [u16x16; 4], pack_n: usize) {
    let (hi, lo) = split_u16_ordered(block);
    unsafe { store_u8x32x2(out.add(0), lo) };
    unsafe { pack_u5_registers(out.add(pack_n), hi, pack_n) }
}

/// Bitpack the provided block of integers to 14-bit elements.
pub(crate) unsafe fn to_u14(out: *mut u8, block: [u16x16; 4], pack_n: usize) {
    let (hi, lo) = split_u16_ordered(block);
    unsafe { store_u8x32x2(out.add(0), lo) };
    unsafe { pack_u6_registers(out.add(pack_n), hi, pack_n) }
}

/// Bitpack the provided block of integers to 15-bit elements.
pub(crate) unsafe fn to_u15(out: *mut u8, block: [u16x16; 4], pack_n: usize) {
    let (hi, lo) = split_u16_ordered(block);
    unsafe { store_u8x32x2(out.add(0), lo) };
    unsafe { pack_u7_registers(out.add(pack_n), hi, pack_n) }
}

/// Bitpack the provided block of integers to 16-bit elements.
pub(crate) unsafe fn to_u16(out: *mut u8, block: [u16x16; 4], _pack_n: usize) {
    unsafe { store_u16x16x4(out, block) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::X64;
    use crate::uint16::{X128_MAX_OUTPUT_LEN, max_compressed_size};

    #[test]
    fn test_to_u1() {
        let data: [u16; X64] = std::array::from_fn(|i| (i % 2) as u16);
        let data = load_u16x64(&data);

        let mut out = [0; X128_MAX_OUTPUT_LEN / 2];
        unsafe { to_u1(out.as_mut_ptr(), data, 64) };
        assert_eq!(out[..8], [170; 8]);

        let mut data = [0; X64];
        data[0] = 1;
        data[1] = 1;
        data[2] = 1;
        data[3] = 1;
        data[4] = 1;
        data[8] = 1;
        data[9] = 1;
        let data = load_u16x64(&data);

        let mut out = [0; X128_MAX_OUTPUT_LEN / 2];
        unsafe { to_u1(out.as_mut_ptr(), data, 10) };
        assert_eq!(out[..2], [31, 3]);
        assert_eq!(out[2..][..14], [0; 14]);
    }

    #[test]
    fn test_to_u2() {
        let data: [u16; X64] = std::array::from_fn(|i| (i % 3) as u16);
        let data = load_u16x64(&data);

        let mut out = [0; X128_MAX_OUTPUT_LEN / 2];
        unsafe { to_u2(out.as_mut_ptr(), data, 64) };
        assert_eq!(
            out[..16],
            [
                146, 36, 73, 146, 36, 73, 146, 36, 36, 73, 146, 36, 73, 146, 36, 73
            ]
        );

        let mut data = [0; X64];
        data[0] = 1;
        data[1] = 1;
        data[2] = 1;
        data[3] = 1;
        data[4] = 1;
        data[8] = 1;
        data[9] = 2;
        let data = load_u16x64(&data);

        let mut out = [0; X128_MAX_OUTPUT_LEN / 2];
        unsafe { to_u2(out.as_mut_ptr(), data, 10) };

        // Imagine the data is in bits:
        //
        // lo_bits: 0b00000001_00011111 -> [31, 1, 0, 0] LE
        // hi_bits: 0b00000010_00000000 -> [0, 2, 0, 0] LE
        assert_eq!(out[..4], [31, 1, 0, 2]);
        assert_eq!(out[4..][..28], [0; 28]);
    }

    #[test]
    fn test_to_u3() {
        let data: [u16; X64] = std::array::from_fn(|i| (i % 4) as u16);
        let data = load_u16x64(&data);

        let mut out = [0; X128_MAX_OUTPUT_LEN / 2];
        unsafe { to_u3(out.as_mut_ptr(), data, 64) };
        assert_eq!(
            out[..24],
            [
                170, 170, 170, 170, 170, 170, 170, 170, 204, 204, 204, 204, 204, 204, 204, 204, 0,
                0, 0, 0, 0, 0, 0, 0
            ]
        );
        assert_eq!(out[24..][..24], [0; 24]);

        let mut data = [0; X64];
        data[0] = 1;
        data[1] = 1;
        data[2] = 1;
        data[3] = 1;
        data[4] = 1;
        data[8] = 5;
        data[9] = 2;
        let data = load_u16x64(&data);

        let mut out = [0; X128_MAX_OUTPUT_LEN / 2];
        unsafe { to_u3(out.as_mut_ptr(), data, 10) };

        // Imagine the data is in bits:
        //
        // b0_bits: 0b00000001_00011111 -> [31, 1, 0, 0] LE
        // b1_bits: 0b00000010_00000000 -> [0, 2, 0, 0] LE
        // b2_bits: 0b00000001_00000000 -> [0, 1, 0, 0] LE
        assert_eq!(out[..6], [31, 1, 0, 2, 0, 1]);
        assert_eq!(out[6..][..42], [0; 42]);
    }

    #[test]
    fn test_to_u4() {
        let data: [u16; X64] = std::array::from_fn(|i| (i % 16) as u16);
        let data = load_u16x64(&data);

        let mut out = [0; X128_MAX_OUTPUT_LEN / 2];
        unsafe { to_u4(out.as_mut_ptr(), data, 64) };
        assert_eq!(out[..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[8..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[16..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[24..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);

        let mut data = [0; X64];
        data[0] = 1;
        data[1] = 15;
        data[2] = 2;
        data[3] = 15;
        data[4] = 1;
        data[8] = 5;
        data[9] = 2;
        data[14] = 15;
        let data = load_u16x64(&data);

        let mut out = [0; X128_MAX_OUTPUT_LEN / 2];
        unsafe { to_u4(out.as_mut_ptr(), data, 15) };
        assert_eq!(out[0], 0b1111_0001); // [1, 15]
        assert_eq!(out[1], 0b1111_0010); // [2, 15]
        assert_eq!(out[2], 0b0000_0001); // [1, 0]
        assert_eq!(out[4], 0b0010_0101); // [5, 2]
        assert_eq!(out[7], 0b0000_1111); // [15, 0]
    }

    #[test]
    fn test_to_u5() {
        let data: [u16; X64] = std::array::from_fn(|i| (i % 32) as u16);
        let data = load_u16x64(&data);

        let mut out = [0; X128_MAX_OUTPUT_LEN / 2];
        unsafe { to_u5(out.as_mut_ptr(), data, 64) };
        assert_eq!(out[..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[8..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[16..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[24..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[32..][..8], [0, 0, 255, 255, 0, 0, 255, 255]);

        let mut data = [0; X64];
        data[0] = 1;
        data[1] = 15;
        data[2] = 2;
        data[3] = 15;
        data[4] = 1;
        data[8] = 5;
        data[9] = 2;
        data[14] = 17;
        let data = load_u16x64(&data);

        let mut out = [0; X128_MAX_OUTPUT_LEN / 2];
        unsafe { to_u5(out.as_mut_ptr(), data, 15) };
        assert_eq!(out[0], 0b1111_0001); // [1, 15]
        assert_eq!(out[1], 0b1111_0010); // [2, 15]
        assert_eq!(out[2], 0b0000_0001); // [1, 0]
        assert_eq!(out[4], 0b0010_0101); // [5, 2]
        assert_eq!(out[7], 0b0000_0001); // [17, 0]  -- well, the lower 4 bits
        assert_eq!(out[8], 0b0000_0000);
        assert_eq!(out[9], 0b0100_0000); // upper bits from that 17
    }

    #[test]
    fn test_to_u6() {
        let data: [u16; X64] = std::array::from_fn(|i| (i % 64) as u16);
        let data = load_u16x64(&data);

        let mut out = [0; X128_MAX_OUTPUT_LEN / 2];
        unsafe { to_u6(out.as_mut_ptr(), data, 64) };
        assert_eq!(out[..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[8..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[16..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[24..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[32..][..8], [0, 0, 255, 255, 0, 0, 255, 255]);
        assert_eq!(out[40..][..8], [0, 0, 0, 0, 255, 255, 255, 255]);

        let mut data = [0; X64];
        data[0] = 1;
        data[1] = 15;
        data[2] = 2;
        data[3] = 15;
        data[4] = 1;
        data[8] = 5;
        data[9] = 2;
        data[14] = 59;
        let data = load_u16x64(&data);

        let mut out = [0; X128_MAX_OUTPUT_LEN / 2];
        unsafe { to_u6(out.as_mut_ptr(), data, 15) };
        assert_eq!(out[0], 0b1111_0001); // [1, 15]
        assert_eq!(out[1], 0b1111_0010); // [2, 15]
        assert_eq!(out[2], 0b0000_0001); // [1, 0]
        assert_eq!(out[4], 0b0010_0101); // [5, 2]
        assert_eq!(out[7], 0b0000_1011); // [59, 0]  -- well, the lower 4 bits
        assert_eq!(out[8], 0b0000_0000);
        assert_eq!(out[9], 0b0100_0000); // upper 1 bit from that 59
        assert_eq!(out[10], 0b0000_0000);
        assert_eq!(out[11], 0b0100_0000); // upper 1 bit from that 59
    }

    #[test]
    fn test_to_u7() {
        let data: [u16; X64] = std::array::from_fn(|i| (i % 64) as u16);
        let data = load_u16x64(&data);

        let mut out = [0; X128_MAX_OUTPUT_LEN / 2];
        unsafe { to_u7(out.as_mut_ptr(), data, 64) };
        assert_eq!(out[..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[8..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[16..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[24..][..8], [16, 50, 84, 118, 152, 186, 220, 254]);
        assert_eq!(out[32..][..8], [0, 0, 255, 255, 0, 0, 255, 255]);
        assert_eq!(out[40..][..8], [0, 0, 0, 0, 255, 255, 255, 255]);
        assert_eq!(out[48..][..8], [0, 0, 0, 0, 0, 0, 0, 0]);

        let mut data = [0; X64];
        data[0] = 1;
        data[1] = 15;
        data[2] = 2;
        data[3] = 15;
        data[4] = 1;
        data[8] = 5;
        data[9] = 2;
        data[14] = 127;
        let data = load_u16x64(&data);

        let mut out = [0; X128_MAX_OUTPUT_LEN / 2];
        unsafe { to_u7(out.as_mut_ptr(), data, 15) };
        assert_eq!(out[0], 0b1111_0001); // [1, 15]
        assert_eq!(out[1], 0b1111_0010); // [2, 15]
        assert_eq!(out[2], 0b0000_0001); // [1, 0]
        assert_eq!(out[4], 0b0010_0101); // [5, 2]
        assert_eq!(out[7], 0b0000_1111); // [127, 0]  -- well, the lower 4 bits
        assert_eq!(out[8], 0b0000_0000);
        assert_eq!(out[9], 0b0100_0000); // upper 1 bit from that 127
        assert_eq!(out[10], 0b0000_0000);
        assert_eq!(out[11], 0b0100_0000); // upper 1 bit from that 127
        assert_eq!(out[12], 0b0000_0000);
        assert_eq!(out[13], 0b0100_0000); // upper 1 bit from that 127
    }

    #[rstest::rstest]
    #[case(1, to_u1)]
    #[case(2, to_u2)]
    #[case(3, to_u3)]
    #[case(4, to_u4)]
    #[case(5, to_u5)]
    #[case(6, to_u6)]
    #[case(7, to_u7)]
    #[case(8, to_u8)]
    #[case(9, to_u9)]
    #[case(10, to_u10)]
    #[case(11, to_u11)]
    #[case(12, to_u12)]
    #[case(13, to_u13)]
    #[case(14, to_u14)]
    #[case(15, to_u15)]
    #[case(16, to_u16)]
    fn test_saturation(
        #[case] bit_len: u8,
        #[case] packer: unsafe fn(*mut u8, [u16x16; 4], usize),
    ) {
        let pack_value = (2u64.pow(bit_len as u32) - 1) as u16;

        let values = [pack_value; X64];
        let data = load_u16x64(&values);

        let mut output = [0; X128_MAX_OUTPUT_LEN / 2];
        unsafe { packer(output.as_mut_ptr(), data, X64) };
        assert!(
            output[..max_compressed_size::<X64>(bit_len as usize)]
                .iter()
                .all(|b| *b == u8::MAX)
        );
    }
}
