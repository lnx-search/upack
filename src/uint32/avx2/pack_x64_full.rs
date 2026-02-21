use std::arch::x86_64::*;

use super::data::*;
use super::util::*;

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 1-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(1)` bytes to.
/// - The runtime CPU must support the `avx2` instructions.
pub(crate) unsafe fn to_u1(out: *mut u8, block: [__m256i; 8]) {
    let partially_packed = pack_u32_to_u8_unordered(block);
    unsafe { pack_u1_registers(out, partially_packed) }
}

#[target_feature(enable = "avx2")]
/// Pack two registers containing 32 8-bit elements each into a 1-bit
/// bitmap and write to `out`.
///
/// Any non-zero value will be treated as a set bit.
unsafe fn pack_u1_registers(out: *mut u8, data: [__m256i; 2]) {
    let [d1, d2] = data;

    let cmp1 = _mm256_slli_epi16::<7>(d1);
    let cmp2 = _mm256_slli_epi16::<7>(d2);

    let mask1 = _mm256_movemask_epi8(cmp1) as u32;
    let mask2 = _mm256_movemask_epi8(cmp2) as u32;

    let merged_mask = ((mask2 as u64) << 32) | mask1 as u64;
    // We assume LE endianness
    unsafe { std::ptr::write_unaligned(out.add(0).cast(), merged_mask) };
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 2-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(2)` bytes to.
/// - The runtime CPU must support the `avx2` instructions.
pub(crate) unsafe fn to_u2(out: *mut u8, block: [__m256i; 8]) {
    let partially_packed = pack_u32_to_u8_unordered(block);
    unsafe { pack_u2_registers(out, partially_packed) }
}

#[target_feature(enable = "avx2")]
/// Pack two registers containing 32 8-bit elements each into a 2-bit
/// bitmap and write to `out`.
unsafe fn pack_u2_registers(out: *mut u8, data: [__m256i; 2]) {
    let packed = pack_u8_to_u2_unordered(data);
    unsafe { _mm_storeu_si128(out.cast(), packed) };
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 3-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(3)` bytes to.
/// - The runtime CPU must support the `avx2` instructions.
pub(crate) unsafe fn to_u3(out: *mut u8, block: [__m256i; 8]) {
    let partially_packed = pack_u32_to_u8_unordered(block);
    unsafe { pack_u3_registers(out, partially_packed) }
}

#[target_feature(enable = "avx2")]
/// Pack two registers containing 32 8-bit elements each into a 3-bit
/// bitmap and write to `out`.
unsafe fn pack_u3_registers(out: *mut u8, data: [__m256i; 2]) {
    let mask = _mm256_set1_epi8(0b11);

    let lo_2bit = and_si256(data, mask);
    let packed = pack_u8_to_u2_unordered(lo_2bit);
    unsafe { _mm_storeu_si128(out.add(0).cast(), packed) };

    let [d1, d2] = data;

    let hi_1bit1 = _mm256_slli_epi16::<5>(d1);
    let hi_1bitmask1 = _mm256_movemask_epi8(hi_1bit1) as u32;

    let hi_1bit2 = _mm256_slli_epi16::<5>(d2);
    let hi_1bitmask2 = _mm256_movemask_epi8(hi_1bit2) as u32;

    let hi_merged_mask = ((hi_1bitmask2 as u64) << 32) | hi_1bitmask1 as u64;
    unsafe { std::ptr::write_unaligned(out.add(16).cast(), hi_merged_mask) };
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 4-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(4)` bytes to.
/// - The runtime CPU must support the `avx2` instructions.
pub(crate) unsafe fn to_u4(out: *mut u8, block: [__m256i; 8]) {
    let partially_packed = pack_u32_to_u8_unordered(block);
    unsafe { pack_u4_registers(out, partially_packed) }
}

#[target_feature(enable = "avx2")]
/// Pack two registers containing 32 8-bit elements each into a 4-bit
/// bitmap and write to `out`.
unsafe fn pack_u4_registers(out: *mut u8, data: [__m256i; 2]) {
    let packed = pack_u8_to_u4_unordered(data);
    unsafe { _mm256_storeu_si256(out.cast(), packed) }
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 5-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(5)` bytes to.
/// - The runtime CPU must support the `avx2` instructions.
pub(crate) unsafe fn to_u5(out: *mut u8, block: [__m256i; 8]) {
    let partially_packed = pack_u32_to_u8_unordered(block);
    unsafe { pack_u5_registers(out, partially_packed) }
}

#[target_feature(enable = "avx2")]
/// Pack two registers containing 32 8-bit elements each into a 5-bit
/// bitmap and write to `out`.
unsafe fn pack_u5_registers(out: *mut u8, data: [__m256i; 2]) {
    let mask = _mm256_set1_epi8(0b1111);
    let masked = and_si256(data, mask);
    unsafe { pack_u4_registers(out, masked) };

    // 4bit * 64 / 8-bits per byte.
    let remaining = srli_epi16::<4, 2>(data);
    unsafe { pack_u1_registers(out.add(32), remaining) };
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 6-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(6)` bytes to.
/// - The runtime CPU must support the `avx2` instructions.
pub(crate) unsafe fn to_u6(out: *mut u8, block: [__m256i; 8]) {
    let partially_packed = pack_u32_to_u8_unordered(block);
    unsafe { pack_u6_registers(out, partially_packed) }
}

#[target_feature(enable = "avx2")]
/// Pack two registers containing 32 8-bit elements each into a 6-bit
/// bitmap and write to `out`.
unsafe fn pack_u6_registers(out: *mut u8, data: [__m256i; 2]) {
    let mask = _mm256_set1_epi8(0b1111);
    let masked = and_si256(data, mask);
    unsafe { pack_u4_registers(out, masked) };

    // 4bit * 64 / 8-bits per byte.
    let remaining = srli_epi16::<4, 2>(data);
    let mask = _mm256_set1_epi8(0b11);
    let masked = and_si256(remaining, mask);
    unsafe { pack_u2_registers(out.add(32), masked) };
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 7-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(7)` bytes to.
/// - The runtime CPU must support the `avx2` instructions.
pub(crate) unsafe fn to_u7(out: *mut u8, block: [__m256i; 8]) {
    let partially_packed = pack_u32_to_u8_unordered(block);
    unsafe { pack_u7_registers(out, partially_packed) }
}

#[target_feature(enable = "avx2")]
/// Pack two registers containing 32 8-bit elements each into a 7-bit
/// bitmap and write to `out`.
unsafe fn pack_u7_registers(out: *mut u8, data: [__m256i; 2]) {
    let mask = _mm256_set1_epi8(0b1111);
    let masked = and_si256(data, mask);
    unsafe { pack_u4_registers(out, masked) };

    // 4bit * 64 / 8-bits per byte.
    let remaining = srli_epi16::<4, 2>(data);
    unsafe { pack_u3_registers(out.add(32), remaining) };
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 8-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(8)` bytes to.
/// - The runtime CPU must support the `avx2` instructions.
pub(crate) unsafe fn to_u8(out: *mut u8, block: [__m256i; 8]) {
    let partially_packed = pack_u32_to_u8_unordered(block);
    unsafe { store_si256x2(out, partially_packed) }
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 9-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(9)` bytes to.
/// - The runtime CPU must support the `avx2` instructions.
pub(crate) unsafe fn to_u9(out: *mut u8, block: [__m256i; 8]) {
    let (hi, lo) = pack_u32_to_u16_split_unordered(block);
    unsafe { store_si256x2(out.add(0), lo) };
    unsafe { pack_u1_registers(out.add(64), hi) }
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 10-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(10)` bytes to.
/// - The runtime CPU must support the `avx2` instructions.
pub(crate) unsafe fn to_u10(out: *mut u8, block: [__m256i; 8]) {
    let (hi, lo) = pack_u32_to_u16_split_unordered(block);
    unsafe { store_si256x2(out.add(0), lo) };
    unsafe { pack_u2_registers(out.add(64), hi) }
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 11-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(11)` bytes to.
/// - The runtime CPU must support the `avx2` instructions.
pub(crate) unsafe fn to_u11(out: *mut u8, block: [__m256i; 8]) {
    let (hi, lo) = pack_u32_to_u16_split_unordered(block);
    unsafe { store_si256x2(out.add(0), lo) };
    unsafe { pack_u3_registers(out.add(64), hi) }
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 12-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(12)` bytes to.
/// - The runtime CPU must support the `avx2` instructions.
pub(crate) unsafe fn to_u12(out: *mut u8, block: [__m256i; 8]) {
    let (hi, lo) = pack_u32_to_u16_split_unordered(block);
    unsafe { store_si256x2(out.add(0), lo) };
    unsafe { pack_u4_registers(out.add(64), hi) }
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 13-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(13)` bytes to.
/// - The runtime CPU must support the `avx2` instructions.
pub(crate) unsafe fn to_u13(out: *mut u8, block: [__m256i; 8]) {
    let (hi, lo) = pack_u32_to_u16_split_unordered(block);
    unsafe { store_si256x2(out.add(0), lo) };
    unsafe { pack_u5_registers(out.add(64), hi) }
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 14-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(14)` bytes to.
/// - The runtime CPU must support the `avx2` instructions.
pub(crate) unsafe fn to_u14(out: *mut u8, block: [__m256i; 8]) {
    let (hi, lo) = pack_u32_to_u16_split_unordered(block);
    unsafe { store_si256x2(out.add(0), lo) };
    unsafe { pack_u6_registers(out.add(64), hi) }
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 15-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(15)` bytes to.
/// - The runtime CPU must support the `avx2` instructions.
pub(crate) unsafe fn to_u15(out: *mut u8, block: [__m256i; 8]) {
    let (hi, lo) = pack_u32_to_u16_split_unordered(block);
    unsafe { store_si256x2(out.add(0), lo) };
    unsafe { pack_u7_registers(out.add(64), hi) }
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 16-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(16)` bytes to.
/// - The runtime CPU must support the `avx2` instructions.
pub(crate) unsafe fn to_u16(out: *mut u8, block: [__m256i; 8]) {
    let packed = pack_u32_to_u16_unordered(block);
    unsafe { store_si256x4(out, packed) };
}

#[target_feature(enable = "avx2")]
unsafe fn store_lo_u16_registers(out: *mut u8, data: [__m256i; 8]) {
    let mask = _mm256_set1_epi32(0xFFFF);
    let shifted = and_si256(data, mask);
    let packed = pack_u32_to_u16_unordered(shifted);
    unsafe { store_si256x4(out, packed) };
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 17-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(17)` bytes to.
/// - The runtime CPU must support the `avx2` instructions.
pub(crate) unsafe fn to_u17(out: *mut u8, block: [__m256i; 8]) {
    unsafe { store_lo_u16_registers(out.add(0), block) };

    let hi_bits = srli_epi32::<16, 8>(block);
    let hi_bits = pack_u32_to_u8_unordered(hi_bits);
    unsafe { pack_u1_registers(out.add(128), hi_bits) }
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 18-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(18)` bytes to.
/// - The runtime CPU must support the `avx2` instructions.
pub(crate) unsafe fn to_u18(out: *mut u8, block: [__m256i; 8]) {
    unsafe { store_lo_u16_registers(out.add(0), block) };

    let hi_bits = srli_epi32::<16, 8>(block);
    let hi_bits = pack_u32_to_u8_unordered(hi_bits);
    unsafe { pack_u2_registers(out.add(128), hi_bits) }
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 19-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(19)` bytes to.
/// - The runtime CPU must support the `avx2` instructions.
pub(crate) unsafe fn to_u19(out: *mut u8, block: [__m256i; 8]) {
    unsafe { store_lo_u16_registers(out.add(0), block) };

    let hi_bits = srli_epi32::<16, 8>(block);
    let hi_bits = pack_u32_to_u8_unordered(hi_bits);
    unsafe { pack_u3_registers(out.add(128), hi_bits) }
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 20-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(20)` bytes to.
/// - The runtime CPU must support the `avx2` instructions.
pub(crate) unsafe fn to_u20(out: *mut u8, block: [__m256i; 8]) {
    unsafe { store_lo_u16_registers(out.add(0), block) };

    let hi_bits = srli_epi32::<16, 8>(block);
    let hi_bits = pack_u32_to_u8_unordered(hi_bits);
    unsafe { pack_u4_registers(out.add(128), hi_bits) }
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 21-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(21)` bytes to.
/// - The runtime CPU must support the `avx2` instructions.
pub(crate) unsafe fn to_u21(out: *mut u8, block: [__m256i; 8]) {
    unsafe { store_lo_u16_registers(out.add(0), block) };

    let hi_bits = srli_epi32::<16, 8>(block);
    let hi_bits = pack_u32_to_u8_unordered(hi_bits);
    unsafe { pack_u5_registers(out.add(128), hi_bits) }
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 22-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(22)` bytes to.
/// - The runtime CPU must support the `avx2` instructions.
pub(crate) unsafe fn to_u22(out: *mut u8, block: [__m256i; 8]) {
    unsafe { store_lo_u16_registers(out.add(0), block) };

    let hi_bits = srli_epi32::<16, 8>(block);
    let hi_bits = pack_u32_to_u8_unordered(hi_bits);
    unsafe { pack_u6_registers(out.add(128), hi_bits) }
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 23-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(23)` bytes to.
/// - The runtime CPU must support the `avx2` instructions.
pub(crate) unsafe fn to_u23(out: *mut u8, block: [__m256i; 8]) {
    unsafe { store_lo_u16_registers(out.add(0), block) };

    let hi_bits = srli_epi32::<16, 8>(block);
    let hi_bits = pack_u32_to_u8_unordered(hi_bits);
    unsafe { pack_u7_registers(out.add(128), hi_bits) }
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 24-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(24)` bytes to.
/// - The runtime CPU must support the `avx2` instructions.
pub(crate) unsafe fn to_u24(out: *mut u8, block: [__m256i; 8]) {
    unsafe { store_lo_u16_registers(out.add(0), block) };

    let hi_bits = srli_epi32::<16, 8>(block);
    let hi_bits = pack_u32_to_u8_unordered(hi_bits);
    unsafe { store_si256x2(out.add(128), hi_bits) };
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 25-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(25)` bytes to.
/// - The runtime CPU must support the `avx2` instructions.
pub(crate) unsafe fn to_u25(out: *mut u8, block: [__m256i; 8]) {
    unsafe { store_lo_u16_registers(out.add(0), block) };

    let hi_half = srli_epi32::<16, 8>(block);
    let (hi, lo) = pack_u32_to_u16_split_unordered(hi_half);
    unsafe { store_si256x2(out.add(128), lo) };
    unsafe { pack_u1_registers(out.add(192), hi) };
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 26-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(26)` bytes to.
/// - The runtime CPU must support the `avx2` instructions.
pub(crate) unsafe fn to_u26(out: *mut u8, block: [__m256i; 8]) {
    unsafe { store_lo_u16_registers(out.add(0), block) };

    let hi_half = srli_epi32::<16, 8>(block);
    let (hi, lo) = pack_u32_to_u16_split_unordered(hi_half);
    unsafe { store_si256x2(out.add(128), lo) };
    unsafe { pack_u2_registers(out.add(192), hi) };
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 27-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(27)` bytes to.
/// - The runtime CPU must support the `avx2` instructions.
pub(crate) unsafe fn to_u27(out: *mut u8, block: [__m256i; 8]) {
    unsafe { store_lo_u16_registers(out.add(0), block) };

    let hi_half = srli_epi32::<16, 8>(block);
    let (hi, lo) = pack_u32_to_u16_split_unordered(hi_half);
    unsafe { store_si256x2(out.add(128), lo) };
    unsafe { pack_u3_registers(out.add(192), hi) };
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 28-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(28)` bytes to.
/// - The runtime CPU must support the `avx2` instructions.
pub(crate) unsafe fn to_u28(out: *mut u8, block: [__m256i; 8]) {
    unsafe { store_lo_u16_registers(out.add(0), block) };

    let hi_half = srli_epi32::<16, 8>(block);
    let (hi, lo) = pack_u32_to_u16_split_unordered(hi_half);
    unsafe { store_si256x2(out.add(128), lo) };
    unsafe { pack_u4_registers(out.add(192), hi) };
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 29-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(29)` bytes to.
/// - The runtime CPU must support the `avx2` instructions.
pub(crate) unsafe fn to_u29(out: *mut u8, block: [__m256i; 8]) {
    unsafe { store_lo_u16_registers(out.add(0), block) };

    let hi_half = srli_epi32::<16, 8>(block);
    let (hi, lo) = pack_u32_to_u16_split_unordered(hi_half);
    unsafe { store_si256x2(out.add(128), lo) };
    unsafe { pack_u5_registers(out.add(192), hi) };
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 30-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(30)` bytes to.
/// - The runtime CPU must support the `avx2` instructions.
pub(crate) unsafe fn to_u30(out: *mut u8, block: [__m256i; 8]) {
    unsafe { store_lo_u16_registers(out.add(0), block) };

    let hi_half = srli_epi32::<16, 8>(block);
    let (hi, lo) = pack_u32_to_u16_split_unordered(hi_half);
    unsafe { store_si256x2(out.add(128), lo) };
    unsafe { pack_u6_registers(out.add(192), hi) };
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 31-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(31)` bytes to.
/// - The runtime CPU must support the `avx2` instructions.
pub(crate) unsafe fn to_u31(out: *mut u8, block: [__m256i; 8]) {
    unsafe { store_lo_u16_registers(out.add(0), block) };

    let hi_half = srli_epi32::<16, 8>(block);
    let (hi, lo) = pack_u32_to_u16_split_unordered(hi_half);
    unsafe { store_si256x2(out.add(128), lo) };
    unsafe { pack_u7_registers(out.add(192), hi) };
}

#[target_feature(enable = "avx2")]
/// Bitpack the provided block of integers to 32-bit elements.
///
/// # Safety
/// - `out` must be safe to write `max_compressed_size::<X64>(32)` bytes to.
/// - The runtime CPU must support the `avx2` instructions.
pub(crate) unsafe fn to_u32(out: *mut u8, block: [__m256i; 8]) {
    unsafe { store_si256x8(out, block) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::X64;
    use crate::uint32::{X128_MAX_OUTPUT_LEN, max_compressed_size};

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
    #[case(17, to_u17)]
    #[case(18, to_u18)]
    #[case(19, to_u19)]
    #[case(20, to_u20)]
    #[case(21, to_u21)]
    #[case(22, to_u22)]
    #[case(23, to_u23)]
    #[case(24, to_u24)]
    #[case(25, to_u25)]
    #[case(26, to_u26)]
    #[case(27, to_u27)]
    #[case(28, to_u28)]
    #[case(29, to_u29)]
    #[case(30, to_u30)]
    #[case(31, to_u31)]
    #[case(32, to_u32)]
    #[cfg_attr(not(target_feature = "avx2"), ignore)]
    fn test_saturation(#[case] bit_len: u8, #[case] packer: unsafe fn(*mut u8, [__m256i; 8])) {
        let pack_value = (2u64.pow(bit_len as u32) - 1) as u32;

        let values = [pack_value; X64];
        let data = unsafe { load_u32x64(&values) };

        let mut output = [0; X128_MAX_OUTPUT_LEN / 2];
        unsafe { packer(output.as_mut_ptr(), data) };
        assert!(
            output[..max_compressed_size::<X64>(bit_len as usize)]
                .iter()
                .all(|b| *b == u8::MAX)
        );
    }
}
