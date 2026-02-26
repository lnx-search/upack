use upack::X128;
use upack::uint32::X128_MAX_OUTPUT_LEN;

use super::generate::GeneratedSamples;
use super::routine::Routine;

/// Execute the base upack compressor.
pub struct UpackRandomLenCompressBase {
    output: Box<[u8; X128_MAX_OUTPUT_LEN]>,
}

impl Default for UpackRandomLenCompressBase {
    fn default() -> Self {
        Self {
            output: Box::new([0; X128_MAX_OUTPUT_LEN]),
        }
    }
}

impl Routine for UpackRandomLenCompressBase {
    type PreparedInput = GeneratedSamples;

    fn name() -> String {
        "compress/upack-base/random-len".to_string()
    }

    fn prep(&mut self, samples: GeneratedSamples) -> Self::PreparedInput {
        samples
    }

    fn execute(&mut self, input: &mut Self::PreparedInput) -> usize {
        let GeneratedSamples {
            samples,
            last_values,
        } = input;

        let mut total_samples = 0;
        for (sample, _last_value) in std::iter::zip(samples.iter(), last_values) {
            let n = fastrand::usize(0..X128);
            total_samples += n;
            let output = upack::compress(n, sample, &mut self.output);
            std::hint::black_box(output);
        }

        total_samples
    }
}

/// Execute the delta upack compressor.
pub struct UpackRandomLenCompressDelta {
    output: Box<[u8; X128_MAX_OUTPUT_LEN]>,
}

impl Default for UpackRandomLenCompressDelta {
    fn default() -> Self {
        Self {
            output: Box::new([0; X128_MAX_OUTPUT_LEN]),
        }
    }
}

impl Routine for UpackRandomLenCompressDelta {
    type PreparedInput = GeneratedSamples;

    fn name() -> String {
        "compress/upack-delta/random-len".to_string()
    }

    fn prep(&mut self, samples: GeneratedSamples) -> Self::PreparedInput {
        samples
    }

    fn execute(&mut self, input: &mut Self::PreparedInput) -> usize {
        let GeneratedSamples {
            samples,
            last_values,
        } = input;

        let mut total_samples = 0;
        for (sample, last_value) in std::iter::zip(samples.iter_mut(), last_values) {
            let n = fastrand::usize(0..X128);
            total_samples += n;
            let output = upack::compress_delta(*last_value, n, sample, &mut self.output);
            std::hint::black_box(output);
        }

        total_samples
    }
}

/// Execute the delta-1 upack compressor.
pub struct UpackRandomLenCompressDelta1 {
    output: Box<[u8; X128_MAX_OUTPUT_LEN]>,
}

impl Default for UpackRandomLenCompressDelta1 {
    fn default() -> Self {
        Self {
            output: Box::new([0; X128_MAX_OUTPUT_LEN]),
        }
    }
}

impl Routine for UpackRandomLenCompressDelta1 {
    type PreparedInput = GeneratedSamples;

    fn name() -> String {
        "compress/upack-delta1/random-len".to_string()
    }

    fn prep(&mut self, samples: GeneratedSamples) -> Self::PreparedInput {
        samples
    }

    fn execute(&mut self, input: &mut Self::PreparedInput) -> usize {
        let GeneratedSamples {
            samples,
            last_values,
        } = input;

        let mut total_samples = 0;
        for (sample, last_value) in std::iter::zip(samples.iter_mut(), last_values) {
            let n = fastrand::usize(0..X128);
            total_samples += n;
            let output = upack::compress_delta1(*last_value, n, sample, &mut self.output);
            std::hint::black_box(output);
        }

        total_samples
    }
}

pub struct PreCompressed {
    compressed: Vec<u8>,
    metadata: Vec<(u32, usize, u8)>,
}

/// Execute the base upack decompressor.
pub struct UpackRandomLenDecompressBase {
    output: Box<[u32; X128]>,
}

impl Default for UpackRandomLenDecompressBase {
    fn default() -> Self {
        Self {
            output: Box::new([0; X128]),
        }
    }
}

impl Routine for UpackRandomLenDecompressBase {
    type PreparedInput = PreCompressed;

    fn name() -> String {
        "decompress/upack-base/random-len".to_string()
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
            let n = fastrand::usize(0..X128);
            let output = upack::compress(n, sample, &mut temp_buffer);
            compressed.extend_from_slice(&temp_buffer[..output.bytes_written]);
            metadata.push((last_value, n, output.compressed_bit_length));
        }
        compressed.resize(compressed.len() + X128_MAX_OUTPUT_LEN, 0);

        PreCompressed {
            compressed,
            metadata,
        }
    }

    fn execute(&mut self, input: &mut Self::PreparedInput) -> usize {
        let PreCompressed {
            compressed,
            metadata,
        } = input;

        let mut total_samples = 0;
        let mut offset = 0;
        for (_last_value, n, nbits) in metadata.iter().copied() {
            total_samples += n;
            offset += upack::decompress(n, nbits, &compressed[offset..], &mut *self.output);
        }

        total_samples
    }
}

/// Execute the delta upack decompressor.
pub struct UpackRandomLenDecompressDelta {
    output: Box<[u32; X128]>,
}

impl Default for UpackRandomLenDecompressDelta {
    fn default() -> Self {
        Self {
            output: Box::new([0; X128]),
        }
    }
}

impl Routine for UpackRandomLenDecompressDelta {
    type PreparedInput = PreCompressed;

    fn name() -> String {
        "decompress/upack-delta/random-len".to_string()
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
            let n = fastrand::usize(0..X128);
            let output = upack::compress_delta(last_value, n, sample, &mut temp_buffer);
            compressed.extend_from_slice(&temp_buffer[..output.bytes_written]);
            metadata.push((last_value, n, output.compressed_bit_length));
        }
        compressed.resize(compressed.len() + X128_MAX_OUTPUT_LEN, 0);

        PreCompressed {
            compressed,
            metadata,
        }
    }

    fn execute(&mut self, input: &mut Self::PreparedInput) -> usize {
        let PreCompressed {
            compressed,
            metadata,
        } = input;

        let mut total_samples = 0;
        let mut offset = 0;
        for (last_value, n, nbits) in metadata.iter().copied() {
            total_samples += n;
            offset += upack::decompress_delta(
                last_value,
                n,
                nbits,
                &compressed[offset..],
                &mut *self.output,
            );
        }

        total_samples
    }
}

/// Execute the delta-1 upack decompressor.
pub struct UpackRandomLenDecompressDelta1 {
    output: Box<[u32; X128]>,
}

impl Default for UpackRandomLenDecompressDelta1 {
    fn default() -> Self {
        Self {
            output: Box::new([0; X128]),
        }
    }
}

impl Routine for UpackRandomLenDecompressDelta1 {
    type PreparedInput = PreCompressed;

    fn name() -> String {
        "decompress/upack-delta1/random-len".to_string()
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
            let n = fastrand::usize(0..X128);
            let output = upack::compress_delta1(last_value, n, sample, &mut temp_buffer);
            compressed.extend_from_slice(&temp_buffer[..output.bytes_written]);
            metadata.push((last_value, n, output.compressed_bit_length));
        }
        compressed.resize(compressed.len() + X128_MAX_OUTPUT_LEN, 0);

        PreCompressed {
            compressed,
            metadata,
        }
    }

    fn execute(&mut self, input: &mut Self::PreparedInput) -> usize {
        let PreCompressed {
            compressed,
            metadata,
        } = input;
        let mut total_samples = 0;
        let mut offset = 0;
        for (last_value, n, nbits) in metadata.iter().copied() {
            total_samples += n;
            offset += upack::decompress_delta1(
                last_value,
                n,
                nbits,
                &compressed[offset..],
                &mut *self.output,
            );
        }

        total_samples
    }
}
