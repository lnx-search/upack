use bitpacking::BitPacker;
use upack::uint32::X128_MAX_OUTPUT_LEN;

use crate::generate::GeneratedSamples;
use crate::routine::Routine;

/// Execute the base bitpacking compressor.
pub struct BitpackingCompressBase {
    packer: bitpacking::BitPacker4x,
    output: Box<[u8; X128_MAX_OUTPUT_LEN]>,
}

impl Default for BitpackingCompressBase {
    fn default() -> Self {
        Self {
            packer: bitpacking::BitPacker4x::new(),
            output: Box::new([0; X128_MAX_OUTPUT_LEN]),
        }
    }
}

impl Routine for BitpackingCompressBase {
    type PreparedInput = GeneratedSamples;

    fn name() -> &'static str {
        "compress/bitpacking-base/x128"
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
            let bits = self.packer.num_bits(sample);
            let output = self
                .packer
                .compress(sample, self.output.as_mut_slice(), bits);
            std::hint::black_box(output);
        }
    }
}

/// Execute the delta bitpacking compressor.
pub struct BitpackingCompressDelta {
    packer: bitpacking::BitPacker4x,
    output: Box<[u8; X128_MAX_OUTPUT_LEN]>,
}

impl Default for BitpackingCompressDelta {
    fn default() -> Self {
        Self {
            packer: bitpacking::BitPacker4x::new(),
            output: Box::new([0; X128_MAX_OUTPUT_LEN]),
        }
    }
}

impl Routine for BitpackingCompressDelta {
    type PreparedInput = GeneratedSamples;

    fn name() -> &'static str {
        "compress/bitpacking-delta/x128"
    }

    fn prep(&mut self, samples: GeneratedSamples) -> Self::PreparedInput {
        samples
    }

    fn execute(&mut self, input: &mut Self::PreparedInput) {
        let GeneratedSamples {
            samples,
            last_values,
        } = input;
        for (sample, last_value) in std::iter::zip(samples.iter(), last_values) {
            let bits = self.packer.num_bits_sorted(*last_value, sample);
            let output =
                self.packer
                    .compress_sorted(*last_value, sample, self.output.as_mut_slice(), bits);
            std::hint::black_box(output);
        }
    }
}

/// Execute the delta-1 bitpacking compressor.
pub struct BitpackingCompressDelta1 {
    packer: bitpacking::BitPacker4x,
    output: Box<[u8; X128_MAX_OUTPUT_LEN]>,
}

impl Default for BitpackingCompressDelta1 {
    fn default() -> Self {
        Self {
            packer: bitpacking::BitPacker4x::new(),
            output: Box::new([0; X128_MAX_OUTPUT_LEN]),
        }
    }
}

impl Routine for BitpackingCompressDelta1 {
    type PreparedInput = GeneratedSamples;

    fn name() -> &'static str {
        "compress/bitpacking-delta1/x128"
    }

    fn prep(&mut self, samples: GeneratedSamples) -> Self::PreparedInput {
        samples
    }

    fn execute(&mut self, input: &mut Self::PreparedInput) {
        let GeneratedSamples {
            samples,
            last_values,
        } = input;
        for (sample, last_value) in std::iter::zip(samples.iter(), last_values) {
            let bits = self
                .packer
                .num_bits_strictly_sorted(Some(*last_value), sample);
            let output = self.packer.compress_strictly_sorted(
                Some(*last_value),
                sample,
                self.output.as_mut_slice(),
                bits,
            );
            std::hint::black_box(output);
        }
    }
}
