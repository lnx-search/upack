use bitpacking::BitPacker;
use upack::X128;
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

pub struct PreCompressed {
    compressed: Vec<u8>,
    metadata: Vec<(u32, u8)>,
}

/// Execute the base bitpacking decompressor.
pub struct BitpackingDecompressBase {
    packer: bitpacking::BitPacker4x,
    output: Box<[u32; X128]>,
}

impl Default for BitpackingDecompressBase {
    fn default() -> Self {
        Self {
            packer: bitpacking::BitPacker4x::new(),
            output: Box::new([0; X128]),
        }
    }
}

impl Routine for BitpackingDecompressBase {
    type PreparedInput = PreCompressed;

    fn name() -> &'static str {
        "decompress/bitpacking-base/x128"
    }

    fn prep(&mut self, samples: GeneratedSamples) -> Self::PreparedInput {
        let GeneratedSamples {
            mut samples,
            last_values,
        } = samples;

        let mut temp_buffer = Box::new([0; X128_MAX_OUTPUT_LEN]);

        let mut compressed = Vec::new();
        let mut metadata = Vec::new();

        for (sample, last_value) in std::iter::zip(samples.iter_mut(), last_values) {
            let nbits = self.packer.num_bits(sample);
            let n = self
                .packer
                .compress(sample, temp_buffer.as_mut_slice(), nbits);
            compressed.extend_from_slice(&temp_buffer[..n]);
            metadata.push((last_value, nbits));
        }

        PreCompressed {
            compressed,
            metadata,
        }
    }

    fn execute(&mut self, input: &mut Self::PreparedInput) {
        let PreCompressed {
            compressed,
            metadata,
        } = input;

        let mut offset = 0;
        for (_last_value, nbits) in metadata.iter().copied() {
            offset += self
                .packer
                .decompress(&compressed[offset..], &mut *self.output, nbits);
        }
    }
}

/// Execute the delta bitpacking decompressor.
pub struct BitpackingDecompressDelta {
    packer: bitpacking::BitPacker4x,
    output: Box<[u32; X128]>,
}

impl Default for BitpackingDecompressDelta {
    fn default() -> Self {
        Self {
            packer: bitpacking::BitPacker4x::new(),
            output: Box::new([0; X128]),
        }
    }
}

impl Routine for BitpackingDecompressDelta {
    type PreparedInput = PreCompressed;

    fn name() -> &'static str {
        "decompress/bitpacking-delta/x128"
    }

    fn prep(&mut self, samples: GeneratedSamples) -> Self::PreparedInput {
        let GeneratedSamples {
            mut samples,
            last_values,
        } = samples;

        let mut temp_buffer = Box::new([0; X128_MAX_OUTPUT_LEN]);

        let mut compressed = Vec::new();
        let mut metadata = Vec::new();

        for (sample, last_value) in std::iter::zip(samples.iter_mut(), last_values) {
            let nbits = self.packer.num_bits_sorted(last_value, sample);
            let n =
                self.packer
                    .compress_sorted(last_value, sample, temp_buffer.as_mut_slice(), nbits);
            compressed.extend_from_slice(&temp_buffer[..n]);
            metadata.push((last_value, nbits));
        }

        PreCompressed {
            compressed,
            metadata,
        }
    }

    fn execute(&mut self, input: &mut Self::PreparedInput) {
        let PreCompressed {
            compressed,
            metadata,
        } = input;

        let mut offset = 0;
        for (last_value, nbits) in metadata.iter().copied() {
            offset += self.packer.decompress_sorted(
                last_value,
                &compressed[offset..],
                &mut *self.output,
                nbits,
            );
        }
    }
}

/// Execute the delta-1 bitpacking decompressor.
pub struct BitpackingDecompressDelta1 {
    packer: bitpacking::BitPacker4x,
    output: Box<[u32; X128]>,
}

impl Default for BitpackingDecompressDelta1 {
    fn default() -> Self {
        Self {
            packer: bitpacking::BitPacker4x::new(),
            output: Box::new([0; X128]),
        }
    }
}

impl Routine for BitpackingDecompressDelta1 {
    type PreparedInput = PreCompressed;

    fn name() -> &'static str {
        "decompress/bitpacking-delta1/x128"
    }

    fn prep(&mut self, samples: GeneratedSamples) -> Self::PreparedInput {
        let GeneratedSamples {
            mut samples,
            last_values,
        } = samples;

        let mut temp_buffer = Box::new([0; X128_MAX_OUTPUT_LEN]);

        let mut compressed = Vec::new();
        let mut metadata = Vec::new();

        for (sample, last_value) in std::iter::zip(samples.iter_mut(), last_values) {
            let nbits = self
                .packer
                .num_bits_strictly_sorted(Some(last_value), sample);
            let n = self.packer.compress_strictly_sorted(
                Some(last_value),
                sample,
                temp_buffer.as_mut_slice(),
                nbits,
            );
            compressed.extend_from_slice(&temp_buffer[..n]);
            metadata.push((last_value, nbits));
        }

        PreCompressed {
            compressed,
            metadata,
        }
    }

    fn execute(&mut self, input: &mut Self::PreparedInput) {
        let PreCompressed {
            compressed,
            metadata,
        } = input;

        let mut offset = 0;
        for (last_value, nbits) in metadata.iter().copied() {
            offset += self.packer.decompress_strictly_sorted(
                Some(last_value),
                &compressed[offset..],
                &mut *self.output,
                nbits,
            );
        }
    }
}
