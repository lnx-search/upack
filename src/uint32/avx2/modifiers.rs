use std::arch::x86_64::*;

use super::data::*;
use crate::X128;

#[target_feature(enable = "avx2")]
/// Apply Delta-1 encoding to the input block, using the `initial_value` as
/// the starting point for the delta.
///
/// _WARNING!_ The provided block of integers must be **strictly monotonic**, meaning they
/// must be consecutive i.e. `1, 2, 3` otherwise this routine will mangle the input data
/// and be useless.
///
/// # Safety
/// The caller must ensure that the runtime CPU must support the required CPU features.
pub(super) fn delta1_encode_x64(mut previous_value: __m256i, block: [__m256i; 8]) -> [__m256i; 8] {
    let mut output = [_mm256_setzero_si256(); 8];

    let rotate_right_perm = _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 7);
    let permute = _mm256_set1_epi32(7);
    let ones = _mm256_set1_epi32(1);

    for i in 0..8 {
        let reg_rotated = _mm256_permutevar8x32_epi32(block[i], rotate_right_perm);
        let prev_last = _mm256_permutevar8x32_epi32(previous_value, permute);
        let shifted = _mm256_blend_epi32::<0b00000001>(reg_rotated, prev_last);
        let delta = _mm256_sub_epi32(block[i], shifted);
        output[i] = _mm256_sub_epi32(delta, ones);
        previous_value = block[i];
    }

    output
}

#[target_feature(enable = "avx2")]
/// Reverse the Delta-1 encoding on the input block.
///
/// # Safety
/// The caller must ensure that the runtime CPU must support the required CPU features.
pub(super) fn delta1_decode_x64(mut previous_value: __m256i, block: [__m256i; 8]) -> [__m256i; 8] {
    todo!()
}

#[target_feature(enable = "avx2")]
/// Apply Delta encoding to the input block, using the `initial_value` as
/// the starting point for the delta.
///
/// _WARNING!_ The provided block of integers must be **monotonic or constant**, meaning
/// this function is less strict than the delta-1 routines but still require the _difference_
/// between each integer to be at least `0`.
///
/// # Safety
/// The caller must ensure that the runtime CPU must support the required CPU features.
pub(super) fn delta_encode_x64(mut previous_value: __m256i, block: [__m256i; 8]) -> [__m256i; 8] {
    let rotate_right_perm = _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 7);
    let permute = _mm256_set1_epi32(7);

    let mut output = [_mm256_setzero_si256(); 8];
    for i in 0..8 {
        let reg_rotated = _mm256_permutevar8x32_epi32(block[i], rotate_right_perm);
        let prev_last = _mm256_permutevar8x32_epi32(previous_value, permute);
        let shifted = _mm256_blend_epi32::<0b00000001>(reg_rotated, prev_last);
        output[i] = _mm256_sub_epi32(block[i], shifted);
        previous_value = block[i];
    }

    output
}

#[target_feature(enable = "avx2")]
/// Reverse the Delta encoding on the input block.
///
/// # Safety
/// The caller must ensure that the runtime CPU must support the required CPU features.
fn delta_decode_x64(mut previous_value: __m256i, block: [__m256i; 8]) -> [__m256i; 8] {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::X64;

    #[test]
    #[cfg_attr(not(target_feature = "avx2"), ignore)]
    fn test_delta_x64() {
        let mut values = [0u32; X64];
        for (i, value) in values.iter_mut().enumerate() {
            *value = i as u32 + 1;
        }

        let initial_value = unsafe { _mm256_set1_epi32(0) };
        let block = unsafe { load_u32x64(&values) };
        let encoded = unsafe { delta_encode_x64(initial_value, block) };
        let view = unsafe { std::mem::transmute::<[__m256i; 8], [u32; 64]>(encoded) };
        assert_eq!(view, [1; X64]);
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx2"), ignore)]
    fn test_delta1_x64() {
        let mut values = [0u32; X64];
        for (i, value) in values.iter_mut().enumerate() {
            *value = i as u32 + 1;
        }

        let initial_value = unsafe { _mm256_set1_epi32(0) };
        let block = unsafe { load_u32x64(&values) };
        let encoded = unsafe { delta1_encode_x64(initial_value, block) };
        let view = unsafe { std::mem::transmute::<[__m256i; 8], [u32; 64]>(encoded) };
        assert_eq!(view, [0; X64]);
    }
}
