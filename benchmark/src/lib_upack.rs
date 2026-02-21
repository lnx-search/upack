use upack::X128;
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

pub struct PreCompressed {
    compressed: Vec<u8>,
    metadata: Vec<(u32, u8)>,
}

/// Execute the base upack decompressor.
pub struct UpackDecompressBase {
    output: Box<[u32; X128]>,
}

impl Default for UpackDecompressBase {
    fn default() -> Self {
        Self {
            output: Box::new([0; X128]),
        }
    }
}

impl Routine for UpackDecompressBase {
    type PreparedInput = PreCompressed;

    fn name() -> &'static str {
        "decompress/upack-base/x128"
    }

    fn prep(&mut self, samples: GeneratedSamples) -> Self::PreparedInput {
        let GeneratedSamples {
            samples,
            last_values,
        } = samples;

        let mut temp_buffer = Box::new([0; X128_MAX_OUTPUT_LEN]);

        let mut compressed = Vec::new();
        let mut metadata = Vec::new();

        for (sample, last_value) in std::iter::zip(samples.iter(), last_values) {
            let output = upack::compress(128, sample, &mut temp_buffer);
            compressed.extend_from_slice(&temp_buffer[..output.bytes_written]);
            metadata.push((last_value, output.compressed_bit_length));
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
            offset += upack::decompress(128, nbits, &compressed[offset..], &mut *self.output);
        }
    }
}

/// Execute the delta upack decompressor.
pub struct UpackDecompressDelta {
    output: Box<[u32; X128]>,
}

impl Default for UpackDecompressDelta {
    fn default() -> Self {
        Self {
            output: Box::new([0; X128]),
        }
    }
}

impl Routine for UpackDecompressDelta {
    type PreparedInput = PreCompressed;

    fn name() -> &'static str {
        "decompress/upack-delta/x128"
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
            let output = upack::compress_delta(last_value, 128, sample, &mut temp_buffer);
            compressed.extend_from_slice(&temp_buffer[..output.bytes_written]);
            metadata.push((last_value, output.compressed_bit_length));
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
            offset += upack::decompress_delta(
                last_value,
                128,
                nbits,
                &compressed[offset..],
                &mut *self.output,
            );
        }
    }
}

/// Execute the delta-1 upack decompressor.
pub struct UpackDecompressDelta1 {
    output: Box<[u32; X128]>,
}

impl Default for UpackDecompressDelta1 {
    fn default() -> Self {
        Self {
            output: Box::new([0; X128]),
        }
    }
}

impl Routine for UpackDecompressDelta1 {
    type PreparedInput = PreCompressed;

    fn name() -> &'static str {
        "decompress/upack-delta1/x128"
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
            let output = upack::compress_delta1(last_value, 128, sample, &mut temp_buffer);
            compressed.extend_from_slice(&temp_buffer[..output.bytes_written]);
            metadata.push((last_value, output.compressed_bit_length));
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
            offset += upack::decompress_delta1(
                last_value,
                128,
                nbits,
                &compressed[offset..],
                &mut *self.output,
            );
        }
    }
}
