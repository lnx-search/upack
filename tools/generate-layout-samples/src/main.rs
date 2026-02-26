use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::Context;
use clap::Parser;
use serde::Serialize;
use upack::{X128, uint16, uint32};

const SEEDS: [u64; 3] = [287365827356235, 93847569834756, 52673285690852452];

#[derive(Debug, Parser)]
struct Args {
    #[arg(short, long)]
    output: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    tracing_subscriber::fmt::init();

    tracing::info!("generating layout data output={}", args.output.display());

    process_uint32(&args.output).context("generate uint32")?;
    process_uint16(&args.output).context("generate uint16")?;

    tracing::info!("done!");

    Ok(())
}

fn process_uint32(output: &Path) -> anyhow::Result<()> {
    let uint32_path = output.join("uint32");
    let _ = std::fs::create_dir_all(&uint32_path);

    let mut metadata = Metadata {
        seeds: SEEDS.to_vec(),
        ..Default::default()
    };

    for seed in SEEDS {
        generate_permutations_uint32(&uint32_path, &mut metadata, seed)
            .context("generate permutations")?;
    }

    let dumped = serde_json::to_vec(&metadata).context("serialize metadata")?;
    std::fs::write(uint32_path.join("metadata.json"), dumped).context("write metadata")?;

    Ok(())
}

fn process_uint16(output: &Path) -> anyhow::Result<()> {
    let uint16_path = output.join("uint16");
    let _ = std::fs::create_dir_all(&uint16_path);

    let mut metadata = Metadata {
        seeds: SEEDS.to_vec(),
        ..Default::default()
    };

    for seed in SEEDS {
        generate_permutations_uint16(&uint16_path, &mut metadata, seed)
            .context("generate permutations")?;
    }

    let dumped = serde_json::to_vec(&metadata).context("serialize metadata")?;
    std::fs::write(uint16_path.join("metadata.json"), dumped).context("write metadata")?;

    Ok(())
}

#[derive(Default, Serialize)]
struct Metadata {
    seeds: Vec<u64>,
    offsets: BTreeMap<String, Vec<BlockMetadata>>,
}

#[derive(Serialize)]
struct BlockMetadata {
    seed: u64,
    compressed_start: usize,
    compressed_length: usize,
    input_start: usize,
}

fn generate_permutations_uint32(
    output: &Path,
    metadata: &mut Metadata,
    seed: u64,
) -> anyhow::Result<()> {
    fastrand::seed(seed);

    let mut temp_output = [0; uint32::X128_MAX_OUTPUT_LEN];

    let mut output_buffer = Vec::new();
    let mut input_buffer = Vec::new();

    for len in 1..=X128 {
        for bit_len in 1..=32 {
            let max_value = 2u64.pow(bit_len);
            let min_value = 2u64.pow(bit_len - 1);

            let mut sample = [0; X128];
            for v in sample.iter_mut() {
                *v = fastrand::u64(min_value..max_value) as u32;
            }

            let output = upack::compress(len, &sample, &mut temp_output);
            assert_eq!(
                output.compressed_bit_length, bit_len as u8,
                "bit len doesn't match expected"
            );

            let output_start = output_buffer.len();
            output_buffer.extend_from_slice(&temp_output[..output.bytes_written]);

            let input_start = input_buffer.len();
            input_buffer.extend_from_slice(&sample);

            let block = BlockMetadata {
                seed,
                compressed_start: output_start,
                compressed_length: output.bytes_written,
                input_start,
            };

            metadata
                .offsets
                .entry(format!("len:{len},bit:{bit_len}"))
                .or_default()
                .push(block);
        }
    }

    let raw_buffer: &[u8] = bytemuck::cast_slice(&input_buffer);
    std::fs::write(output.join(format!("seeded-{seed}.raw")), raw_buffer)
        .context("write raw input buffer")?;

    std::fs::write(output.join(format!("seeded-{seed}.upack")), output_buffer)
        .context("write compressed buffer")?;

    Ok(())
}

fn generate_permutations_uint16(
    output: &Path,
    metadata: &mut Metadata,
    seed: u64,
) -> anyhow::Result<()> {
    fastrand::seed(seed);

    let mut temp_output = [0; uint16::X128_MAX_OUTPUT_LEN];

    let mut output_buffer = Vec::new();
    let mut input_buffer = Vec::new();

    for len in 1..=X128 {
        for bit_len in 1..=16 {
            let max_value = 2u64.pow(bit_len);
            let min_value = 2u64.pow(bit_len - 1);

            let mut sample = [0; X128];
            for v in sample.iter_mut() {
                *v = fastrand::u64(min_value..max_value) as u16;
            }

            let output = upack::compress(len, &sample, &mut temp_output);
            assert_eq!(
                output.compressed_bit_length, bit_len as u8,
                "bit len doesn't match expected"
            );

            let output_start = output_buffer.len();
            output_buffer.extend_from_slice(&temp_output[..output.bytes_written]);

            let input_start = input_buffer.len();
            input_buffer.extend_from_slice(&sample);

            let block = BlockMetadata {
                seed,
                compressed_start: output_start,
                compressed_length: output.bytes_written,
                input_start,
            };

            metadata
                .offsets
                .entry(format!("len:{len},bit:{bit_len}"))
                .or_default()
                .push(block);
        }
    }

    let raw_buffer: &[u8] = bytemuck::cast_slice(&input_buffer);
    std::fs::write(output.join(format!("seeded-{seed}.raw")), raw_buffer)
        .context("write raw input buffer")?;

    std::fs::write(output.join(format!("seeded-{seed}.upack")), output_buffer)
        .context("write compressed buffer")?;

    Ok(())
}
