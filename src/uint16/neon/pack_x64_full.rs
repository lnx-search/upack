use std::arch::aarch64::*;

use super::data::*;
use super::polyfill::*;
use super::util::*;

#[target_feature(enable = "neon")]
/// Bitpack the provided block of integers to 1-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(1)` bytes to.
pub(crate) unsafe fn to_u1(out: *mut u8, block: [uint16x8_t; 8]) {
    let partially_packed = pack_u16_to_u8_unordered(block);
    unsafe { pack_u1_registers(out, partially_packed) }
}

#[inline]
#[target_feature(enable = "neon")]
/// Pack two registers containing 32 8-bit elements each into a 1-bit
/// bitmap and write to `out`.
///
/// Any non-zero value will be treated as a set bit.
unsafe fn pack_u1_registers(out: *mut u8, data: [uint8x16_t; 4]) {
    let [d1, d2, d3, d4] = data;

    let select_mask = _neon_set1_u8(0b1);
    let cmp1 = _neon_and_u8(d1, select_mask);
    let cmp2 = _neon_and_u8(d2, select_mask);
    let cmp3 = _neon_and_u8(d3, select_mask);
    let cmp4 = _neon_and_u8(d4, select_mask);

    let mask = _neon_nonzero_mask_u8([cmp1, cmp2, cmp3, cmp4]);
    // We assume LE endianness
    unsafe { std::ptr::write_unaligned(out.add(0).cast(), mask) };
}

#[target_feature(enable = "neon")]
/// Bitpack the provided block of integers to 2-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(2)` bytes to.
pub(crate) unsafe fn to_u2(out: *mut u8, block: [uint16x8_t; 8]) {
    let partially_packed = pack_u16_to_u8_unordered(block);
    unsafe { pack_u2_registers(out, partially_packed) }
}

#[inline]
#[target_feature(enable = "neon")]
/// Pack two registers containing 32 8-bit elements each into a 2-bit
/// bitmap and write to `out`.
unsafe fn pack_u2_registers(out: *mut u8, data: [uint8x16_t; 4]) {
    let packed = pack_u8_to_u2_unordered(data);
    unsafe { _neon_store_u8(out.cast(), packed) };
}

#[target_feature(enable = "neon")]
/// Bitpack the provided block of integers to 3-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(3)` bytes to.
pub(crate) unsafe fn to_u3(out: *mut u8, block: [uint16x8_t; 8]) {
    let partially_packed = pack_u16_to_u8_unordered(block);
    unsafe { pack_u3_registers(out, partially_packed) }
}

#[inline]
#[target_feature(enable = "neon")]
/// Pack two registers containing 32 8-bit elements each into a 3-bit
/// bitmap and write to `out`.
unsafe fn pack_u3_registers(out: *mut u8, data: [uint8x16_t; 4]) {
    let mask = _neon_set1_u8(0b11);

    let lo_2bit = and_u8(data, mask);
    let packed = pack_u8_to_u2_unordered(lo_2bit);
    unsafe { _neon_store_u8(out.add(0), packed) };

    let hi_1bit1 = _neon_srli_u8::<2>(data[0]);
    let hi_1bit2 = _neon_srli_u8::<2>(data[1]);
    let hi_1bit3 = _neon_srli_u8::<2>(data[2]);
    let hi_1bit4 = _neon_srli_u8::<2>(data[3]);

    let mask = _neon_nonzero_mask_u8([hi_1bit1, hi_1bit2, hi_1bit3, hi_1bit4]);
    unsafe { std::ptr::write_unaligned(out.add(16).cast(), mask) };
}

#[target_feature(enable = "neon")]
/// Bitpack the provided block of integers to 4-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(4)` bytes to.
pub(crate) unsafe fn to_u4(out: *mut u8, block: [uint16x8_t; 8]) {
    let partially_packed = pack_u16_to_u8_unordered(block);
    unsafe { pack_u4_registers(out, partially_packed) }
}

#[inline]
#[target_feature(enable = "neon")]
/// Pack two registers containing 32 8-bit elements each into a 4-bit
/// bitmap and write to `out`.
unsafe fn pack_u4_registers(out: *mut u8, data: [uint8x16_t; 4]) {
    let packed = pack_u8_to_u4_unordered(data);
    unsafe { _neon_store_u8(out.add(0), packed[0]) };
    unsafe { _neon_store_u8(out.add(16), packed[1]) };
}

#[target_feature(enable = "neon")]
/// Bitpack the provided block of integers to 5-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(5)` bytes to.
pub(crate) unsafe fn to_u5(out: *mut u8, block: [uint16x8_t; 8]) {
    let partially_packed = pack_u16_to_u8_unordered(block);
    unsafe { pack_u5_registers(out, partially_packed) }
}

#[inline]
#[target_feature(enable = "neon")]
/// Pack two registers containing 32 8-bit elements each into a 5-bit
/// bitmap and write to `out`.
unsafe fn pack_u5_registers(out: *mut u8, data: [uint8x16_t; 4]) {
    let mask = _neon_set1_u8(0b1111);
    let masked = and_u8(data, mask);
    unsafe { pack_u4_registers(out, masked) };

    // 4bit * 64 / 8-bits per byte.
    let remaining = srli_u8::<4, 4>(data);
    unsafe { pack_u1_registers(out.add(32), remaining) };
}

#[target_feature(enable = "neon")]
/// Bitpack the provided block of integers to 6-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(6)` bytes to.
pub(crate) unsafe fn to_u6(out: *mut u8, block: [uint16x8_t; 8]) {
    let partially_packed = pack_u16_to_u8_unordered(block);
    unsafe { pack_u6_registers(out, partially_packed) }
}

#[inline]
#[target_feature(enable = "neon")]
/// Pack two registers containing 32 8-bit elements each into a 6-bit
/// bitmap and write to `out`.
unsafe fn pack_u6_registers(out: *mut u8, data: [uint8x16_t; 4]) {
    let mask = _neon_set1_u8(0b1111);
    let masked = and_u8(data, mask);
    unsafe { pack_u4_registers(out, masked) };

    // 4bit * 64 / 8-bits per byte.
    let remaining = srli_u8::<4, 4>(data);
    unsafe { pack_u2_registers(out.add(32), remaining) };
}

#[target_feature(enable = "neon")]
/// Bitpack the provided block of integers to 7-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(7)` bytes to.
pub(crate) unsafe fn to_u7(out: *mut u8, block: [uint16x8_t; 8]) {
    let partially_packed = pack_u16_to_u8_unordered(block);
    unsafe { pack_u7_registers(out, partially_packed) }
}

#[inline]
#[target_feature(enable = "neon")]
/// Pack two registers containing 32 8-bit elements each into a 7-bit
/// bitmap and write to `out`.
unsafe fn pack_u7_registers(out: *mut u8, data: [uint8x16_t; 4]) {
    let mask = _neon_set1_u8(0b1111);
    let masked = and_u8(data, mask);
    unsafe { pack_u4_registers(out, masked) };

    // 4bit * 64 / 8-bits per byte.
    let remaining = srli_u8::<4, 4>(data);
    unsafe { pack_u3_registers(out.add(32), remaining) };
}

#[target_feature(enable = "neon")]
/// Bitpack the provided block of integers to 8-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(8)` bytes to.
pub(crate) unsafe fn to_u8(out: *mut u8, block: [uint16x8_t; 8]) {
    let partially_packed = pack_u16_to_u8_unordered(block);
    unsafe { store_u8x16x4(out, partially_packed) }
}

#[target_feature(enable = "neon")]
/// Bitpack the provided block of integers to 9-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(9)` bytes to.
pub(crate) unsafe fn to_u9(out: *mut u8, block: [uint16x8_t; 8]) {
    let (hi, lo) = split_u16_unordered(block);
    unsafe { store_u8x16x4(out.add(0), lo) };
    unsafe { pack_u1_registers(out.add(64), hi) }
}

#[target_feature(enable = "neon")]
/// Bitpack the provided block of integers to 10-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(10)` bytes to.
pub(crate) unsafe fn to_u10(out: *mut u8, block: [uint16x8_t; 8]) {
    let (hi, lo) = split_u16_unordered(block);
    unsafe { store_u8x16x4(out.add(0), lo) };
    unsafe { pack_u2_registers(out.add(64), hi) }
}

#[target_feature(enable = "neon")]
/// Bitpack the provided block of integers to 11-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(11)` bytes to.
pub(crate) unsafe fn to_u11(out: *mut u8, block: [uint16x8_t; 8]) {
    let (hi, lo) = split_u16_unordered(block);
    unsafe { store_u8x16x4(out.add(0), lo) };
    unsafe { pack_u3_registers(out.add(64), hi) }
}

#[target_feature(enable = "neon")]
/// Bitpack the provided block of integers to 12-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(12)` bytes to.
pub(crate) unsafe fn to_u12(out: *mut u8, block: [uint16x8_t; 8]) {
    let (hi, lo) = split_u16_unordered(block);
    unsafe { store_u8x16x4(out.add(0), lo) };
    unsafe { pack_u4_registers(out.add(64), hi) }
}

#[target_feature(enable = "neon")]
/// Bitpack the provided block of integers to 13-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(13)` bytes to.
pub(crate) unsafe fn to_u13(out: *mut u8, block: [uint16x8_t; 8]) {
    let (hi, lo) = split_u16_unordered(block);
    unsafe { store_u8x16x4(out.add(0), lo) };
    unsafe { pack_u5_registers(out.add(64), hi) }
}

#[target_feature(enable = "neon")]
/// Bitpack the provided block of integers to 14-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(14)` bytes to.
pub(crate) unsafe fn to_u14(out: *mut u8, block: [uint16x8_t; 8]) {
    let (hi, lo) = split_u16_unordered(block);
    unsafe { store_u8x16x4(out.add(0), lo) };
    unsafe { pack_u6_registers(out.add(64), hi) }
}

#[target_feature(enable = "neon")]
/// Bitpack the provided block of integers to 15-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(15)` bytes to.
pub(crate) unsafe fn to_u15(out: *mut u8, block: [uint16x8_t; 8]) {
    let (hi, lo) = split_u16_unordered(block);
    unsafe { store_u8x16x4(out.add(0), lo) };
    unsafe { pack_u7_registers(out.add(64), hi) }
}

#[target_feature(enable = "neon")]
/// Bitpack the provided block of integers to 16-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(16)` bytes to.
pub(crate) unsafe fn to_u16(out: *mut u8, block: [uint16x8_t; 8]) {
    unsafe { store_u16x8x8(out, block) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::X64;
    use crate::uint16::{X128_MAX_OUTPUT_LEN, max_compressed_size};

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
    #[cfg_attr(not(target_feature = "neon"), ignore)]
    fn test_saturation(#[case] bit_len: u8, #[case] packer: unsafe fn(*mut u8, [uint16x8_t; 8])) {
        let pack_value = (2u64.pow(bit_len as u32) - 1) as u16;

        let values = [pack_value; X64];
        let data = unsafe { load_u16x64(&values) };

        let mut output = [0; X128_MAX_OUTPUT_LEN / 2];
        unsafe { packer(output.as_mut_ptr(), data) };
        assert!(
            output[..max_compressed_size::<X64>(bit_len as usize)]
                .iter()
                .all(|b| *b == u8::MAX)
        );
    }
}
