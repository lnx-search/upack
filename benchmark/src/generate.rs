use std::cmp;

use upack::X128;

pub struct GeneratedSamples {
    pub samples: Vec<[u32; X128]>,
    pub last_values: Vec<u32>,
}

/// Generates N samples of data which is randomly generated but correct
/// to test against the bitpacking compression routines.
///
/// This should produce sample with varying bit lengths and varying
/// gap sizes.
pub fn generate_sample_input(
    num_samples: usize,
    seed: u64,
    min_gap: u32,
    max_gap: u32,
    max_bits: u8,
    corpus_size: u32,
) -> GeneratedSamples {
    fastrand::seed(seed);

    let mut samples = Vec::with_capacity(num_samples * X128);
    let mut last_values = Vec::with_capacity(num_samples);

    for _ in 0..num_samples {
        let gap_bit_length = fastrand::u8(1..max_bits + 1);
        let low_bound = cmp::max((1u32 << (gap_bit_length - 1)) - 1, min_gap);
        let high_bound = cmp::min(1u32 << gap_bit_length, max_gap);

        let mut previous_value = fastrand::u32(0..corpus_size);
        last_values.push(previous_value);

        let mut block = [0; X128];
        #[allow(clippy::needless_range_loop)]
        for i in 0..X128 {
            previous_value += fastrand::u32(low_bound..high_bound);
            block[i] = previous_value;
        }

        samples.push(block);
    }

    GeneratedSamples {
        samples,
        last_values,
    }
}
