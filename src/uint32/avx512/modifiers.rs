#![allow(clippy::needless_range_loop)]
//! Modifiers alter the input block data into a format that is more compressible providing
//! the block values meet a certain criteria.

use std::arch::x86_64::*;

use super::data::*;
use crate::{X128, X256};

#[target_feature(enable = "avx512f")]
/// Apply Delta-1 encoding to the input block, using the `initial_value` as
/// the starting point for the delta.
///
/// _WARNING!_ The provided block of integers must be **strictly monotonic**, meaning they
/// must be consecutive i.e. `1, 2, 3` otherwise this routine will mangle the input data
/// and be useless.
///
/// # Safety
/// The caller must ensure that the runtime CPU must support the required CPU features.
pub fn delta1_encode_x128(initial_value: u32, block: &mut [u32; X128]) {
    let data = load_u32x128(block);
    let previous_value = _mm512_set1_epi32(initial_value as i32);
    delta1_encode_x128_registers(block, previous_value, data);
}

#[target_feature(enable = "avx512f")]
/// Reverse the Delta-1 encoding on the input block.
///
/// # Safety
/// The caller must ensure that the runtime CPU must support the required CPU features.
pub fn delta1_decode_x128(initial_value: u32, block: &mut [u32; X128]) {
    let data = load_u32x128(block);
    let previous_value = _mm512_set1_epi32(initial_value as i32);
    delta1_decode_x128_registers(block, previous_value, data);
}

#[target_feature(enable = "avx512f")]
/// Apply Delta-1 encoding to the input block, using the `initial_value` as
/// the starting point for the delta.
///
/// _WARNING!_ The provided block of integers must be **strictly monotonic**, meaning they
/// must be consecutive i.e. `1, 2, 3` otherwise this routine will mangle the input data
/// and be useless.
///
/// # Safety
/// The caller must ensure that the runtime CPU must support the required CPU features.
pub fn delta1_encode_x256(initial_value: u32, block: &mut [u32; X256]) {
    let [half1, half2] = crate::uint32::split_x256(block);

    let data = load_u32x128(half1);
    let mut previous_value = _mm512_set1_epi32(initial_value as i32);

    delta1_encode_x128_registers(half1, previous_value, data);
    previous_value = data[7];

    let data = load_u32x128(half2);
    delta1_encode_x128_registers(half2, previous_value, data);
}

#[target_feature(enable = "avx512f")]
/// Apply Delta encoding to the input block, using the `initial_value` as
/// the starting point for the delta.
///
/// _WARNING!_ The provided block of integers must be **monotonic or constant**, meaning
/// this function is less strict than the delta-1 routines but still require the _difference_
/// between each integer to be at least `0`.
///
/// # Safety
/// The caller must ensure that the runtime CPU must support the required CPU features.
pub fn delta_encode_x128(initial_value: u32, block: &mut [u32; X128]) {
    let data = load_u32x128(block);
    let previous_value = _mm512_set1_epi32(initial_value as i32);
    delta_encode_x128_registers(block, previous_value, data);
}

#[target_feature(enable = "avx512f")]
/// Reverse the Delta encoding on the input block.
///
/// # Safety
/// The caller must ensure that the runtime CPU must support the required CPU features.
pub fn delta_decode_x128(initial_value: u32, block: &mut [u32; X128]) {
    let data = load_u32x128(block);
    let previous_value = _mm512_set1_epi32(initial_value as i32);
    delta_decode_x128_registers(block, previous_value, data);
}

#[target_feature(enable = "avx512f")]
/// Apply Delta encoding to the input block, using the `initial_value` as
/// the starting point for the delta.
///
/// _WARNING!_ The provided block of integers must be **monotonic or constant**, meaning
/// this function is less strict than the delta-1 routines but still require the _difference_
/// between each integer to be at least `0`.
///
/// # Safety
/// The caller must ensure that the runtime CPU must support the required CPU features.
pub fn delta_encode_x256(initial_value: u32, block: &mut [u32; X256]) {
    let [half1, half2] = crate::uint32::split_x256(block);

    let data = load_u32x128(half1);
    let mut previous_value = _mm512_set1_epi32(initial_value as i32);

    delta_encode_x128_registers(half1, previous_value, data);
    previous_value = data[7];

    let data = load_u32x128(half2);
    delta_encode_x128_registers(half2, previous_value, data);
}

#[target_feature(enable = "avx512f")]
fn delta_encode_x128_registers(
    block: &mut [u32; X128],
    mut previous_value: __m512i,
    data: [__m512i; 8],
) {
    let block_ptr: *mut __m512i = block.as_mut_ptr().cast();

    for (offset, reg) in data.into_iter().enumerate() {
        let shifted = _mm512_alignr_epi32::<15>(reg, previous_value);
        let delta = _mm512_sub_epi32(reg, shifted);
        unsafe { _mm512_storeu_si512(block_ptr.add(offset), delta) };
        previous_value = reg;
    }
}

#[target_feature(enable = "avx512f")]
fn delta_decode_x128_registers(
    block: &mut [u32; X128],
    previous_value: __m512i,
    mut data: [__m512i; 8],
) {
    let block_ptr: *mut __m512i = block.as_mut_ptr().cast();
    let zero = _mm512_setzero_si512();

    for i in 0..8 {
        let t = _mm512_alignr_epi32::<15>(data[i], zero);
        data[i] = _mm512_add_epi32(data[i], t);
    }

    for i in 0..8 {
        let t = _mm512_alignr_epi32::<14>(data[i], zero);
        data[i] = _mm512_add_epi32(data[i], t);
    }

    for i in 0..8 {
        let t = _mm512_alignr_epi32::<12>(data[i], zero);
        data[i] = _mm512_add_epi32(data[i], t);
    }

    for i in 0..8 {
        let t = _mm512_alignr_epi32::<8>(data[i], zero);
        data[i] = _mm512_add_epi32(data[i], t);
    }

    let idx_broadcast = _mm512_set1_epi32(15);
    let mut accumulator = _mm512_permutexvar_epi32(idx_broadcast, previous_value);
    for i in 0..8 {
        let result = _mm512_add_epi32(data[i], accumulator);
        unsafe { _mm512_storeu_si512(block_ptr.add(i), result) };
        accumulator = _mm512_permutexvar_epi32(idx_broadcast, result);
    }
}

#[target_feature(enable = "avx512f")]
fn delta1_encode_x128_registers(
    block: &mut [u32; X128],
    mut previous_value: __m512i,
    data: [__m512i; 8],
) {
    let block_ptr: *mut __m512i = block.as_mut_ptr().cast();

    let ones = _mm512_set1_epi32(1);
    for (offset, reg) in data.into_iter().enumerate() {
        let shifted = _mm512_alignr_epi32::<15>(reg, previous_value);
        let delta = _mm512_sub_epi32(reg, shifted);
        let delta1 = _mm512_sub_epi32(delta, ones);
        unsafe { _mm512_storeu_si512(block_ptr.add(offset), delta1) };
        previous_value = reg;
    }
}

#[target_feature(enable = "avx512f")]
fn delta1_decode_x128_registers(
    block: &mut [u32; X128],
    previous_value: __m512i,
    mut data: [__m512i; 8],
) {
    let ones = _mm512_set1_epi32(1);
    for i in 0..8 {
        data[i] = _mm512_add_epi32(data[i], ones);
    }
    delta_decode_x128_registers(block, previous_value, data);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_delta_x128() {
        let mut values = [0u32; X128];
        for (i, value) in values.iter_mut().enumerate() {
            *value = i as u32 + 1;
        }
        let original_values = values;

        unsafe { delta_encode_x128(0, &mut values) };
        assert_eq!(values, [1; X128]);

        unsafe { delta_decode_x128(0, &mut values) };
        assert_eq!(values, original_values);
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_delta_x256() {
        let mut values = [0u32; X256];
        for (i, value) in values.iter_mut().enumerate() {
            *value = i as u32 + 1;
        }
        unsafe { delta_encode_x256(0, &mut values) };
        assert_eq!(values, [1; X256]);
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_delta1_x128() {
        let mut values = [0u32; X128];
        for (i, value) in values.iter_mut().enumerate() {
            *value = i as u32 + 1;
        }
        let original_values = values;

        unsafe { delta1_encode_x128(0, &mut values) };
        assert_eq!(values, [0; X128]);

        unsafe { delta1_decode_x128(0, &mut values) };
        assert_eq!(values, original_values);
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_delta1_x256() {
        let mut values = [0u32; X256];
        for (i, value) in values.iter_mut().enumerate() {
            *value = i as u32 + 1;
        }
        unsafe { delta1_encode_x256(0, &mut values) };
        assert_eq!(values, [0; X256]);
    }
}
