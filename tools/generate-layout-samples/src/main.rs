use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::Context;
use clap::Parser;
use serde::Serialize;
use upack::X128;
use upack::uint32::X128_MAX_OUTPUT_LEN;

const SEEDS: [u64; 3] = [287365827356235, 93847569834756, 52673285690852452];

#[derive(Debug, Parser)]
struct Args {
    #[arg(short, long)]
    output: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    tracing_subscriber::fmt::init();

    let _ = std::fs::create_dir_all(&args.output);

    tracing::info!("generating layout data output={}", args.output.display());

    let mut metadata = Metadata::default();
    for seed in SEEDS {
        generate_permutations(&args.output, &mut metadata, seed)
            .context("generate permutations")?;
    }

    let dumped = serde_json::to_vec(&metadata).context("serialize metadata")?;
    std::fs::write(args.output.join("metadata.json"), dumped).context("write metadata")?;

    tracing::info!("done!");

    Ok(())
}

#[derive(Default, Serialize)]
struct Metadata {
    offsets: BTreeMap<String, Vec<BlockMetadata>>,
}

#[derive(Serialize)]
struct BlockMetadata {
    seed: u64,
    start: usize,
    length: usize,
}

fn generate_permutations(output: &Path, metadata: &mut Metadata, seed: u64) -> anyhow::Result<()> {
    fastrand::seed(seed);

    let mut temp_output = [0; X128_MAX_OUTPUT_LEN];

    let mut output_buffer = Vec::new();

    for len in 1..=X128 {
        for bit_len in 1..=32 {
            let max_value = 2u64.pow(bit_len) as u32;
            let min_value = 2u64.pow(bit_len - 1) as u32;

            let mut sample = [0; X128];
            for v in sample.iter_mut() {
                *v = fastrand::u32(min_value..max_value);
            }

            let output = upack::compress(len, &sample, &mut temp_output);
            assert_eq!(
                output.compressed_bit_length, bit_len as u8,
                "bit len doesn't match expected"
            );

            let start = output_buffer.len();
            output_buffer.extend_from_slice(&temp_output[..output.bytes_written]);

            let block = BlockMetadata {
                seed,
                start,
                length: output.bytes_written,
            };

            metadata
                .offsets
                .entry(format!("len:{len},bit:{bit_len}"))
                .or_default()
                .push(block);
        }
    }

    std::fs::write(output.join(format!("seeded-{seed}.upack")), output_buffer)
        .context("write compressed buffer")?;

    Ok(())
}
