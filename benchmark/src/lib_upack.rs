use upack::X128;
use upack::uint32::X128_MAX_OUTPUT_LEN;

use crate::generate::GeneratedSamples;
use crate::routine::Routine;

/// Execute the base upack compressor.
pub struct UpackCompressBase<const BLOCK_SIZE: usize = X128> {
    output: Box<[u8; X128_MAX_OUTPUT_LEN]>,
}

impl<const BLOCK_SIZE: usize> Default for UpackCompressBase<BLOCK_SIZE> {
    fn default() -> Self {
        Self {
            output: Box::new([0; X128_MAX_OUTPUT_LEN]),
        }
    }
}

impl<const BLOCK_SIZE: usize> Routine for UpackCompressBase<BLOCK_SIZE> {
    type PreparedInput = GeneratedSamples;

    fn name() -> String {
        format!("compress/upack-base/x{BLOCK_SIZE}")
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
pub struct UpackCompressDelta<const BLOCK_SIZE: usize = X128> {
    output: Box<[u8; X128_MAX_OUTPUT_LEN]>,
}

impl<const BLOCK_SIZE: usize> Default for UpackCompressDelta<BLOCK_SIZE> {
    fn default() -> Self {
        Self {
            output: Box::new([0; X128_MAX_OUTPUT_LEN]),
        }
    }
}

impl<const BLOCK_SIZE: usize> Routine for UpackCompressDelta<BLOCK_SIZE> {
    type PreparedInput = GeneratedSamples;

    fn name() -> String {
        format!("compress/upack-delta/x{BLOCK_SIZE}")
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
pub struct UpackCompressDelta1<const BLOCK_SIZE: usize = X128> {
    output: Box<[u8; X128_MAX_OUTPUT_LEN]>,
}

impl<const BLOCK_SIZE: usize> Default for UpackCompressDelta1<BLOCK_SIZE> {
    fn default() -> Self {
        Self {
            output: Box::new([0; X128_MAX_OUTPUT_LEN]),
        }
    }
}

impl<const BLOCK_SIZE: usize> Routine for UpackCompressDelta1<BLOCK_SIZE> {
    type PreparedInput = GeneratedSamples;

    fn name() -> String {
        format!("compress/upack-delta1/x{BLOCK_SIZE}")
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
pub struct UpackDecompressBase<const BLOCK_SIZE: usize = X128> {
    output: Box<[u32; X128]>,
}

impl<const BLOCK_SIZE: usize> Default for UpackDecompressBase<BLOCK_SIZE> {
    fn default() -> Self {
        Self {
            output: Box::new([0; X128]),
        }
    }
}

impl<const BLOCK_SIZE: usize> Routine for UpackDecompressBase<BLOCK_SIZE> {
    type PreparedInput = PreCompressed;

    fn name() -> String {
        format!("decompress/upack-base/x{BLOCK_SIZE}")
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
pub struct UpackDecompressDelta<const BLOCK_SIZE: usize = X128> {
    output: Box<[u32; X128]>,
}

impl<const BLOCK_SIZE: usize> Default for UpackDecompressDelta<BLOCK_SIZE> {
    fn default() -> Self {
        Self {
            output: Box::new([0; X128]),
        }
    }
}

impl<const BLOCK_SIZE: usize> Routine for UpackDecompressDelta<BLOCK_SIZE> {
    type PreparedInput = PreCompressed;

    fn name() -> String {
        format!("decompress/upack-delta/x{BLOCK_SIZE}")
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
pub struct UpackDecompressDelta1<const BLOCK_SIZE: usize = X128> {
    output: Box<[u32; X128]>,
}

impl<const BLOCK_SIZE: usize> Default for UpackDecompressDelta1<BLOCK_SIZE> {
    fn default() -> Self {
        Self {
            output: Box::new([0; X128]),
        }
    }
}

impl<const BLOCK_SIZE: usize> Routine for UpackDecompressDelta1<BLOCK_SIZE> {
    type PreparedInput = PreCompressed;

    fn name() -> String {
        format!("decompress/upack-delta1/x{BLOCK_SIZE}")
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
