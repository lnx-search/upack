use std::arch::x86_64::*;

use super::data::*;
use super::utils::*;
use crate::X64;

#[target_feature(enable = "avx2")]
/// Unpack the 1-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(1)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
pub unsafe fn from_u1(input: *const u8, read_n: usize) -> [__m256i; 8] {
    todo!()
}

#[target_feature(enable = "avx2")]
/// Unpack the 2-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(2)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
pub unsafe fn from_u2(input: *const u8, read_n: usize) -> [__m256i; 8] {
    todo!()
}

#[target_feature(enable = "avx2")]
/// Unpack the 3-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(3)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
pub unsafe fn from_u3(input: *const u8, read_n: usize) -> [__m256i; 8] {
    todo!()
}

#[target_feature(enable = "avx2")]
/// Unpack the 4-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(4)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
pub unsafe fn from_u4(input: *const u8, read_n: usize) -> [__m256i; 8] {
    todo!()
}

#[target_feature(enable = "avx2")]
/// Unpack the 5-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(5)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
pub unsafe fn from_u5(input: *const u8, read_n: usize) -> [__m256i; 8] {
    todo!()
}

#[target_feature(enable = "avx2")]
/// Unpack the 6-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(6)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
pub unsafe fn from_u6(input: *const u8, read_n: usize) -> [__m256i; 8] {
    todo!()
}

#[target_feature(enable = "avx2")]
/// Unpack the 7-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(7)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
pub unsafe fn from_u7(input: *const u8, read_n: usize) -> [__m256i; 8] {
    todo!()
}

#[target_feature(enable = "avx2")]
/// Unpack the 8-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(8)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
pub unsafe fn from_u8(input: *const u8, read_n: usize) -> [__m256i; 8] {
    todo!()
}

#[target_feature(enable = "avx2")]
/// Unpack the 9-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(9)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
pub unsafe fn from_u9(input: *const u8, read_n: usize) -> [__m256i; 8] {
    todo!()
}

#[target_feature(enable = "avx2")]
/// Unpack the 10-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(10)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
pub unsafe fn from_u10(input: *const u8, read_n: usize) -> [__m256i; 8] {
    todo!()
}

#[target_feature(enable = "avx2")]
/// Unpack the 11-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(11)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
pub unsafe fn from_u11(input: *const u8, read_n: usize) -> [__m256i; 8] {
    todo!()
}

#[target_feature(enable = "avx2")]
/// Unpack the 12-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(12)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
pub unsafe fn from_u12(input: *const u8, read_n: usize) -> [__m256i; 8] {
    todo!()
}

#[target_feature(enable = "avx2")]
/// Unpack the 13-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(13)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
pub unsafe fn from_u13(input: *const u8, read_n: usize) -> [__m256i; 8] {
    todo!()
}

#[target_feature(enable = "avx2")]
/// Unpack the 14-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(14)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
pub unsafe fn from_u14(input: *const u8, read_n: usize) -> [__m256i; 8] {
    todo!()
}

#[target_feature(enable = "avx2")]
/// Unpack the 15-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(15)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
pub unsafe fn from_u15(input: *const u8, read_n: usize) -> [__m256i; 8] {
    todo!()
}

#[target_feature(enable = "avx2")]
/// Unpack the 16-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(16)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
pub unsafe fn from_u16(input: *const u8, read_n: usize) -> [__m256i; 8] {
    todo!()
}

#[target_feature(enable = "avx2")]
/// Unpack the 17-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(17)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
pub unsafe fn from_u17(input: *const u8, read_n: usize) -> [__m256i; 8] {
    todo!()
}

#[target_feature(enable = "avx2")]
/// Unpack the 18-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(18)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
pub unsafe fn from_u18(input: *const u8, read_n: usize) -> [__m256i; 8] {
    todo!()
}

#[target_feature(enable = "avx2")]
/// Unpack the 19-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(19)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
pub unsafe fn from_u19(input: *const u8, read_n: usize) -> [__m256i; 8] {
    todo!()
}

#[target_feature(enable = "avx2")]
/// Unpack the 20-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(20)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
pub unsafe fn from_u20(input: *const u8, read_n: usize) -> [__m256i; 8] {
    todo!()
}

#[target_feature(enable = "avx2")]
/// Unpack the 21-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(21)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
pub unsafe fn from_u21(input: *const u8, read_n: usize) -> [__m256i; 8] {
    todo!()
}

#[target_feature(enable = "avx2")]
/// Unpack the 22-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(22)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
pub unsafe fn from_u22(input: *const u8, read_n: usize) -> [__m256i; 8] {
    todo!()
}

#[target_feature(enable = "avx2")]
/// Unpack the 23-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(23)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
pub unsafe fn from_u23(input: *const u8, read_n: usize) -> [__m256i; 8] {
    todo!()
}

#[target_feature(enable = "avx2")]
/// Unpack the 24-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(24)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
pub unsafe fn from_u24(input: *const u8, read_n: usize) -> [__m256i; 8] {
    todo!()
}

#[target_feature(enable = "avx2")]
/// Unpack the 25-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(25)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
pub unsafe fn from_u25(input: *const u8, read_n: usize) -> [__m256i; 8] {
    todo!()
}

#[target_feature(enable = "avx2")]
/// Unpack the 26-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(26)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
pub unsafe fn from_u26(input: *const u8, read_n: usize) -> [__m256i; 8] {
    todo!()
}

#[target_feature(enable = "avx2")]
/// Unpack the 27-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(27)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
pub unsafe fn from_u27(input: *const u8, read_n: usize) -> [__m256i; 8] {
    todo!()
}

#[target_feature(enable = "avx2")]
/// Unpack the 28-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(28)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
pub unsafe fn from_u28(input: *const u8, read_n: usize) -> [__m256i; 8] {
    todo!()
}

#[target_feature(enable = "avx2")]
/// Unpack the 29-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(29)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
pub unsafe fn from_u29(input: *const u8, read_n: usize) -> [__m256i; 8] {
    todo!()
}

#[target_feature(enable = "avx2")]
/// Unpack the 30-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(30)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
pub unsafe fn from_u30(input: *const u8, read_n: usize) -> [__m256i; 8] {
    todo!()
}

#[target_feature(enable = "avx2")]
/// Unpack the 31-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(31)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
pub unsafe fn from_u31(input: *const u8, read_n: usize) -> [__m256i; 8] {
    todo!()
}

#[target_feature(enable = "avx2")]
/// Unpack the 32-bit integers from the input pointer and return the result registers.
///
/// # Safety
/// - `input` must be safe to read `max_compressed_size::<X64>(32)` bytes from.
/// - The runtime CPU must support the `avx2` instructions.
pub unsafe fn from_u32(input: *const u8, read_n: usize) -> [__m256i; 8] {
    todo!()
}
