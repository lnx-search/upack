use std::ops::{Index, IndexMut};

#[allow(non_camel_case_types)]
#[repr(align(32))]
#[derive(Copy, Clone)]
/// A block of 8, 32-bit elements.
pub(super) struct u32x8([u32; 8]);

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
    type Output = u32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<usize> for u32x8 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

#[allow(non_camel_case_types)]
#[repr(align(32))]
#[derive(Copy, Clone)]
/// A block of 16, 16-bit elements.
pub(super) struct u16x16([u16; 16]);

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
    type Output = u16;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<usize> for u16x16 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

#[allow(non_camel_case_types)]
#[repr(align(32))]
#[derive(Copy, Clone)]
/// A block of 32, 8-bit elements.
pub(super) struct u8x32([u8; 32]);

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
    type Output = u8;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<usize> for u8x32 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

#[allow(non_camel_case_types)]
#[repr(align(16))]
#[derive(Copy, Clone)]
/// A block of 4, 32-bit elements.
pub(super) struct u32x4([u32; 4]);

impl From<u16x8> for u32x4 {
    fn from(value: u16x8) -> Self {
        unsafe { std::mem::transmute(value) }
    }
}

impl From<u8x16> for u32x4 {
    fn from(value: u8x16) -> Self {
        unsafe { std::mem::transmute(value) }
    }
}

impl Index<usize> for u32x4 {
    type Output = u32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<usize> for u32x4 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

#[allow(non_camel_case_types)]
#[repr(align(16))]
#[derive(Copy, Clone)]
/// A block of 8, 16-bit elements.
pub(super) struct u16x8([u16; 8]);

impl From<u32x4> for u16x8 {
    fn from(value: u32x4) -> Self {
        unsafe { std::mem::transmute(value) }
    }
}

impl From<u8x16> for u16x8 {
    fn from(value: u8x16) -> Self {
        unsafe { std::mem::transmute(value) }
    }
}

impl Index<usize> for u16x8 {
    type Output = u16;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<usize> for u16x8 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

#[allow(non_camel_case_types)]
#[repr(align(16))]
#[derive(Copy, Clone)]
/// A block of 16, 8-bit elements.
pub(super) struct u8x16([u8; 16]);

impl From<u32x4> for u8x16 {
    fn from(value: u32x4) -> Self {
        unsafe { std::mem::transmute(value) }
    }
}

impl From<u16x8> for u8x16 {
    fn from(value: u16x8) -> Self {
        unsafe { std::mem::transmute(value) }
    }
}

impl Index<usize> for u8x16 {
    type Output = u8;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<usize> for u8x16 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

/// Broadcast the single 32-bit element across all lanes in the register.
pub(crate) fn _scalar_set1_u32(value: u32) -> u32x8 {
    u32x8([value; 8])
}

/// Broadcast the single 16-bit element across all lanes in the register.
pub(crate) fn _scalar_set1_u16(value: u16) -> u16x16 {
    u16x16([value; 16])
}

/// Broadcast the single 8-bit element across all lanes in the register.
pub(crate) fn _scalar_set1_u8(value: u8) -> u8x32 {
    u8x32([value; 32])
}

/// Load 8, 32-bit elements from the provided `ptr`.
///
/// # Safety
/// The provided `ptr` must be safe to read `8` elements.
pub(crate) unsafe fn _scalar_load_u32x8(ptr: *const u32) -> u32x8 {
    unsafe { _scalar_load_u8x32(ptr.cast()).into() }
}

/// Load 16, 16-bit elements from the provided `ptr`.
///
/// # Safety
/// The provided `ptr` must be safe to read `16` elements.
pub(crate) unsafe fn _scalar_load_u16x16(ptr: *const u16) -> u16x16 {
    unsafe { _scalar_load_u8x32(ptr.cast()).into() }
}

/// Load 32, 8-bit elements from the provided `ptr`.
///
/// # Safety
/// The provided `ptr` must be safe to read `32` elements.
pub(crate) unsafe fn _scalar_load_u8x32(ptr: *const u8) -> u8x32 {
    let mut block = u8x32([0; 32]);
    unsafe { std::ptr::copy_nonoverlapping(ptr, block.0.as_mut_ptr(), 32) };
    block
}

/// Store 8, 32-bit elements in the provided `ptr`.
///
/// # Safety
/// The provided `ptr` must be safe to write `8` elements.
pub(crate) unsafe fn _scalar_store_u32x8(ptr: *mut u32, reg: u32x8) {
    unsafe { std::ptr::copy_nonoverlapping(reg.0.as_ptr(), ptr, 8) };
}

/// Store 16, 16-bit elements in the provided `ptr`.
///
/// # Safety
/// The provided `ptr` must be safe to write `16` elements.
pub(crate) unsafe fn _scalar_store_u16x16(ptr: *mut u16, reg: u16x16) {
    unsafe { std::ptr::copy_nonoverlapping(reg.0.as_ptr(), ptr, 16) };
}

/// Store 32, 8-bit elements in the provided `ptr`.
///
/// # Safety
/// The provided `ptr` must be safe to write `32` elements.
pub(crate) unsafe fn _scalar_store_u8x32(ptr: *mut u8, reg: u8x32) {
    unsafe { std::ptr::copy_nonoverlapping(reg.0.as_ptr(), ptr, 32) };
}

/// Pack the two provide [u32x8] registers to unsigned 16-bit integers in blocks
/// of 128-bits.
pub(crate) fn _scalar_pack_u32x8(a: u32x8, b: u32x8) -> u16x16 {
    let mut out = u16x16([0; 16]);

    // Note:
    // Mirrors the `packusdw` intel instruction in terms of layout, but the saturation
    // behaviour is slightly different, that being said, we don't depend on that behaviour
    // at all, technically the codegen could be a bit better, however, it isn't worth the effort.

    // low 128-bits in a
    for i in 0..4 {
        out[i] = a[i] as u16;
    }

    // low 128-bits in b
    for i in 0..4 {
        out[i + 4] = b[i] as u16;
    }

    // high 128-bits in a
    for i in 4..8 {
        out[i + 4] = a[i] as u16;
    }

    // high 128-bits in b
    for i in 4..8 {
        out[i + 8] = b[i] as u16;
    }

    out
}

/// Pack the two provide [u16x16] registers to unsigned 8-bit integers in blocks
/// of 128-bits.
pub(crate) fn _scalar_pack_u16x16(a: u16x16, b: u16x16) -> u8x32 {
    let mut out = u8x32([0; 32]);

    // Note: See _scalar_pack_u32x8 on differences in saturation behaviour.

    // low 128-bits in a
    for i in 0..8 {
        out[i] = a[i] as u8;
    }

    // low 128-bits in b
    for i in 0..8 {
        out[i + 8] = b[i] as u8;
    }

    // high 128-bits in a
    for i in 8..16 {
        out[i + 8] = a[i] as u8;
    }

    // high 128-bits in b
    for i in 8..16 {
        out[i + 16] = b[i] as u8;
    }

    out
}

/// Convert the provided 8-bit elements in register `a` to 16-bit integers via extension.
pub(crate) fn _scalar_cvteu16_u8x16(a: u8x16) -> u16x16 {
    let mut out = u16x16([0; 16]);
    for i in 0..16 {
        out[i] = a[i] as u16;
    }
    out
}

/// Convert the provided 16-bit elements in register `a` to 32-bit integers via extension.
pub(crate) fn _scalar_cvteu32_u16x8(a: u16x8) -> u32x8 {
    let mut out = u32x8([0; 8]);
    for i in 0..8 {
        out[i] = a[i] as u32;
    }
    out
}

/// Convert the provided [u32x8] register into a single [u16x8] register using
/// truncation.
pub(crate) fn _scalar_cvteu16_u32x8(a: u32x8) -> u16x8 {
    let mut out = u16x8([0; 8]);
    for i in 0..8 {
        out[i] = a[i] as u16;
    }
    out
}

/// Convert the provided [u16x16] register into a single [u8x16] register using
/// truncation.
pub(crate) fn _scalar_cvteu8_u16x16(a: u16x16) -> u8x16 {
    let mut out = u8x16([0; 16]);
    for i in 0..16 {
        out[i] = a[i] as u8;
    }
    out
}

/// Combine two [u32x4] registers via concatenation forming a [u32x8] register.
pub(crate) fn _scalar_combine_u32x4(a: u32x4, b: u32x4) -> u32x8 {
    let mut block = u32x8([0; 8]);
    let ptr = block.0.as_mut_ptr();
    unsafe { std::ptr::copy_nonoverlapping(a.0.as_ptr(), ptr.add(0), 4) };
    unsafe { std::ptr::copy_nonoverlapping(b.0.as_ptr(), ptr.add(4), 4) };
    block
}

/// Combine two [u16x8] registers via concatenation forming a [u16x16] register.
pub(crate) fn _scalar_combine_u16x8(a: u16x8, b: u16x8) -> u16x16 {
    let mut block = u16x16([0; 16]);
    let ptr = block.0.as_mut_ptr();
    unsafe { std::ptr::copy_nonoverlapping(a.0.as_ptr(), ptr.add(0), 8) };
    unsafe { std::ptr::copy_nonoverlapping(b.0.as_ptr(), ptr.add(8), 8) };
    block
}

/// Combine two [u8x16] registers via concatenation forming a [u8x32] register.
pub(crate) fn _scalar_combine_u8x16(a: u8x16, b: u8x16) -> u8x32 {
    let mut block = u8x32([0; 32]);
    let ptr = block.0.as_mut_ptr();
    unsafe { std::ptr::copy_nonoverlapping(a.0.as_ptr(), ptr.add(0), 16) };
    unsafe { std::ptr::copy_nonoverlapping(b.0.as_ptr(), ptr.add(16), 16) };
    block
}

/// Perform a bitwise AND on the provided registers `a` and `b`.
pub(crate) fn _scalar_and_u32x8(mut a: u32x8, b: u32x8) -> u32x8 {
    for i in 0..8 {
        a[i] &= b[i];
    }
    a
}

/// Perform a bitwise AND on the provided registers `a` and `b`.
pub(crate) fn _scalar_and_u16x16(a: u16x16, b: u16x16) -> u16x16 {
    _scalar_and_u32x8(a.into(), b.into()).into()
}

/// Perform a bitwise AND on the provided registers `a` and `b`.
pub(crate) fn _scalar_and_u8x32(a: u8x32, b: u8x32) -> u8x32 {
    _scalar_and_u32x8(a.into(), b.into()).into()
}

/// Perform a bitwise OR on the provided registers `a` and `b`.
pub(crate) fn _scalar_or_u32x8(mut a: u32x8, b: u32x8) -> u32x8 {
    for i in 0..8 {
        a[i] |= b[i];
    }
    a
}

/// Perform a bitwise OR on the provided registers `a` and `b`.
pub(crate) fn _scalar_or_u16x16(a: u16x16, b: u16x16) -> u16x16 {
    _scalar_or_u32x8(a.into(), b.into()).into()
}

/// Perform a bitwise OR on the provided registers `a` and `b`.
pub(crate) fn _scalar_or_u8x32(a: u8x32, b: u8x32) -> u8x32 {
    _scalar_or_u32x8(a.into(), b.into()).into()
}

/// Perform a bitwise shift right on the provided register `a`.
pub(crate) fn _scalar_srli_u32x8<const IMM8: u32>(mut a: u32x8) -> u32x8 {
    for i in 0..8 {
        a[i] >>= IMM8;
    }
    a
}

/// Perform a bitwise shift right on the provided register `a`.
pub(crate) fn _scalar_srli_u16x16<const IMM8: u32>(mut a: u16x16) -> u16x16 {
    for i in 0..16 {
        a[i] >>= IMM8;
    }
    a
}

/// Perform a bitwise shift right on the provided register `a`.
pub(crate) fn _scalar_srli_u8x32<const IMM8: u32>(mut a: u8x32) -> u8x32 {
    for i in 0..32 {
        a[i] >>= IMM8;
    }
    a
}

/// Perform a bitwise shift left on the provided register `a`.
pub(crate) fn _scalar_slli_u32x8<const IMM8: u32>(mut a: u32x8) -> u32x8 {
    for i in 0..8 {
        a[i] <<= IMM8;
    }
    a
}

/// Perform a bitwise shift left on the provided register `a`.
pub(crate) fn _scalar_slli_u16x16<const IMM8: u32>(mut a: u16x16) -> u16x16 {
    for i in 0..16 {
        a[i] <<= IMM8;
    }
    a
}

/// Perform a bitwise shift left on the provided register `a`.
pub(crate) fn _scalar_slli_u8x32<const IMM8: u32>(mut a: u8x32) -> u8x32 {
    for i in 0..32 {
        a[i] <<= IMM8;
    }
    a
}

/// Extract one 128-bit half from the input register.
///
/// `HALF` can be either `0` or `1` representing the index of the halves respectively.
pub(crate) fn _scalar_extract_u32x8<const HALF: usize>(a: u32x8) -> u32x4 {
    const { assert!(HALF <= 1, "selector must be either 0 or 1") };
    let offset = HALF * 4;
    let ptr = unsafe { a.0.as_ptr().add(offset) };
    let mut out = u32x4([0; 4]);
    unsafe { std::ptr::copy_nonoverlapping(ptr, out.0.as_mut_ptr(), 4) };
    out
}

/// Extract one 128-bit half from the input register.
///
/// `HALF` can be either `0` or `1` representing the index of the halves respectively.
pub(crate) fn _scalar_extract_u16x16<const HALF: usize>(a: u16x16) -> u16x8 {
    const { assert!(HALF <= 1, "selector must be either 0 or 1") };
    let offset = HALF * 8;
    let ptr = unsafe { a.0.as_ptr().add(offset) };
    let mut out = u16x8([0; 8]);
    unsafe { std::ptr::copy_nonoverlapping(ptr, out.0.as_mut_ptr(), 8) };
    out
}

/// Extract one 128-bit half from the input register.
///
/// `HALF` can be either `0` or `1` representing the index of the halves respectively.
pub(crate) fn _scalar_extract_u8x32<const HALF: usize>(a: u8x32) -> u8x16 {
    const { assert!(HALF <= 1, "selector must be either 0 or 1") };
    let offset = HALF * 16;
    let ptr = unsafe { a.0.as_ptr().add(offset) };
    let mut out = u8x16([0; 16]);
    unsafe { std::ptr::copy_nonoverlapping(ptr, out.0.as_mut_ptr(), 16) };
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_u32x8() {
        let a = _scalar_set1_u32(4);
        let b = _scalar_set1_u32(2);
        let result = _scalar_pack_u32x8(a, b);
        assert_eq!(
            result.0,
            [
                4, 4, 4, 4, // a_lo
                2, 2, 2, 2, // b_lo
                4, 4, 4, 4, // a_hi
                2, 2, 2, 2, // b_hi
            ]
        );
    }

    #[test]
    fn test_pack_u16x16() {
        let a = _scalar_set1_u16(4);
        let b = _scalar_set1_u16(2);
        let result = _scalar_pack_u16x16(a, b);
        assert_eq!(
            result.0,
            [
                4, 4, 4, 4, 4, 4, 4, 4, // a_lo
                2, 2, 2, 2, 2, 2, 2, 2, // b_lo
                4, 4, 4, 4, 4, 4, 4, 4, // a_hi
                2, 2, 2, 2, 2, 2, 2, 2, // b_hi
            ]
        );
    }

    #[test]
    fn test_cvteu16_u32x8() {
        let a = _scalar_set1_u32(4);
        let result = _scalar_cvteu16_u32x8(a);
        assert_eq!(result.0, [4; 8]);
    }

    #[test]
    fn test_cvteu8_u16x16() {
        let a = _scalar_set1_u16(4);
        let result = _scalar_cvteu8_u16x16(a);
        assert_eq!(result.0, [4; 16]);
    }

    #[test]
    fn test_cvteu16_u8x16() {
        let a = u8x16([4; 16]);
        let result = _scalar_cvteu16_u8x16(a);
        assert_eq!(result.0, [4; 16]);
    }

    #[test]
    fn test_cvteu32_u16x8() {
        let a = u16x8([4; 8]);
        let result = _scalar_cvteu32_u16x8(a);
        assert_eq!(result.0, [4; 8]);
    }

    #[test]
    fn test_combine_u32x4() {
        let a = u32x4([1; 4]);
        let b = u32x4([2; 4]);
        let result = _scalar_combine_u32x4(a, b);
        assert_eq!(result.0[..4], [1; 4]);
        assert_eq!(result.0[4..], [2; 4]);
    }

    #[test]
    fn test_combine_u16x8() {
        let a = u16x8([1; 8]);
        let b = u16x8([2; 8]);
        let result = _scalar_combine_u16x8(a, b);
        assert_eq!(result.0[..8], [1; 8]);
        assert_eq!(result.0[8..], [2; 8]);
    }

    #[test]
    fn test_combine_u8x16() {
        let a = u8x16([1; 16]);
        let b = u8x16([2; 16]);
        let result = _scalar_combine_u8x16(a, b);
        assert_eq!(result.0[..16], [1; 16]);
        assert_eq!(result.0[16..], [2; 16]);
    }

    #[test]
    fn test_extract_u32x8() {
        let a = u32x8([4, 4, 4, 4, 2, 2, 2, 2]);

        let result = _scalar_extract_u32x8::<0>(a);
        assert_eq!(result.0, [4; 4]);

        let result = _scalar_extract_u32x8::<1>(a);
        assert_eq!(result.0, [2; 4]);
    }

    #[test]
    fn test_extract_u16x16() {
        let a = u16x16([4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2]);

        let result = _scalar_extract_u16x16::<0>(a);
        assert_eq!(result.0, [4; 8]);

        let result = _scalar_extract_u16x16::<1>(a);
        assert_eq!(result.0, [2; 8]);
    }

    #[test]
    fn test_extract_u8x32() {
        let a = u8x32([
            4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2,
        ]);

        let result = _scalar_extract_u8x32::<0>(a);
        assert_eq!(result.0, [4; 16]);

        let result = _scalar_extract_u8x32::<1>(a);
        assert_eq!(result.0, [2; 16]);
    }
}
