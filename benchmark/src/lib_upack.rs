use upack::uint32::X128_MAX_OUTPUT_LEN;

use crate::generate::GeneratedSamples;
use crate::routine::Routine;

/// Execute the base upack compressor.
pub struct UpackCompressBase {
    output: Box<[u8; X128_MAX_OUTPUT_LEN]>,
}

impl Default for UpackCompressBase {
    fn default() -> Self {
        Self {
            output: Box::new([0; X128_MAX_OUTPUT_LEN]),
        }
    }
}

impl Routine for UpackCompressBase {
    type PreparedInput = GeneratedSamples;

    fn name() -> &'static str {
        "compress/upack-base/x128"
    }

    fn prep(&mut self, samples: GeneratedSamples) -> Self::PreparedInput {
        samples
    }

    fn execute(&mut self, input: &mut Self::PreparedInput) {
        let GeneratedSamples {
            samples,
            last_values,
        } = input;
        for (sample, _last_value) in std::iter::zip(samples.iter(), last_values) {
            let output = upack::compress(128, sample, &mut self.output);
            std::hint::black_box(output);
        }
    }
}

/// Execute the delta upack compressor.
pub struct UpackCompressDelta {
    output: Box<[u8; X128_MAX_OUTPUT_LEN]>,
}

impl Default for UpackCompressDelta {
    fn default() -> Self {
        Self {
            output: Box::new([0; X128_MAX_OUTPUT_LEN]),
        }
    }
}

impl Routine for UpackCompressDelta {
    type PreparedInput = GeneratedSamples;

    fn name() -> &'static str {
        "compress/upack-delta/x128"
    }

    fn prep(&mut self, samples: GeneratedSamples) -> Self::PreparedInput {
        samples
    }

    fn execute(&mut self, input: &mut Self::PreparedInput) {
        let GeneratedSamples {
            samples,
            last_values,
        } = input;
        for (sample, last_value) in std::iter::zip(samples.iter_mut(), last_values) {
            let output = upack::compress_delta(*last_value, 128, sample, &mut self.output);
            std::hint::black_box(output);
        }
    }
}

/// Execute the delta-1 upack compressor.
pub struct UpackCompressDelta1 {
    output: Box<[u8; X128_MAX_OUTPUT_LEN]>,
}

impl Default for UpackCompressDelta1 {
    fn default() -> Self {
        Self {
            output: Box::new([0; X128_MAX_OUTPUT_LEN]),
        }
    }
}

impl Routine for UpackCompressDelta1 {
    type PreparedInput = GeneratedSamples;

    fn name() -> &'static str {
        "compress/upack-delta1/x128"
    }

    fn prep(&mut self, samples: GeneratedSamples) -> Self::PreparedInput {
        samples
    }

    fn execute(&mut self, input: &mut Self::PreparedInput) {
        let GeneratedSamples {
            samples,
            last_values,
        } = input;
        for (sample, last_value) in std::iter::zip(samples.iter_mut(), last_values) {
            let output = upack::compress_delta1(*last_value, 128, sample, &mut self.output);
            std::hint::black_box(output);
        }
    }
}
