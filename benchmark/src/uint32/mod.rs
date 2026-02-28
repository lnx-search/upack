use clap::Parser;

mod generate;
pub mod lib_bitpacking;
pub mod lib_upack;
pub mod lib_upack_rng;
mod routine;
pub mod runner;

#[derive(Debug, Parser)]
pub struct Args {
    #[arg(short = 'd', long = "duration", default_value = "15s")]
    /// The time to run for each routine in some human-readable format, i.e. "10m".
    run_duration: humantime::Duration,
    #[arg(short = 's', long, default_value = "1000000")]
    /// The number of sample blocks to generate per run.
    sample_size: usize,
    #[arg(long, default_value = "5346436346346")]
    /// The rng seed to use for generating samples.
    seed: u64,
    #[arg(long, default_value_t = u32::MAX)]
    /// The maximum gap to use between the sequential integers in the block.
    ///
    /// Useful for delta based comparisons, but probably shouldn't be touched
    /// unless you know what you're doing.
    max_gap: u32,
    #[arg(long, default_value = "26", value_parser = clap::value_parser!(u8).range(1..=32))]
    /// The maximum bit length of produced integers for the sample inputs.
    max_bits: u8,
    #[arg(long, default_value_t = 40_000_000u32)]
    /// The theoretical corpus size assuming this was in a search engine or similar.
    ///
    /// You probably shouldn't touch this unless you know what you're doing.
    corpus_size: u32,
    #[arg(short, long, default_value = "")]
    /// A prefix filter to only run routines that start with the given input.
    filter: String,
}

pub fn run_benchmark(args: Args) {
    tracing::info!("starting benchmark args={args:?}");

    let config = runner::Config {
        sample_size: args.sample_size,
        seed: args.seed,
        max_gap: args.max_gap,
        max_bits: args.max_bits,
        corpus_size: args.corpus_size,
        run_duration: args.run_duration.into(),
        filter: args.filter,
    };

    let mut runner = runner::RunContext::from(config);

    // Compressors
    runner.run::<lib_upack::UpackCompressBase>();
    runner.run::<lib_upack::UpackCompressDelta>();
    runner.run::<lib_upack::UpackCompressDelta1>();
    runner.run::<lib_upack::UpackCompressAdaptiveDelta>();
    runner.run::<lib_bitpacking::BitpackingCompressBase>();
    runner.run::<lib_bitpacking::BitpackingCompressDelta>();
    runner.run::<lib_bitpacking::BitpackingCompressDelta1>();

    // Decompressors
    runner.run::<lib_upack::UpackDecompressBase>();
    runner.run::<lib_upack::UpackDecompressDelta>();
    runner.run::<lib_upack::UpackDecompressDelta1>();
    runner.run::<lib_bitpacking::BitpackingDecompressBase>();
    runner.run::<lib_bitpacking::BitpackingDecompressDelta>();
    runner.run::<lib_bitpacking::BitpackingDecompressDelta1>();

    // // Decompressors - random block len
    // runner.run::<lib_upack_rng::UpackRandomLenDecompressBase>();
    // runner.run::<lib_upack_rng::UpackRandomLenDecompressDelta>();
    // runner.run::<lib_upack_rng::UpackRandomLenDecompressDelta1>();
    //
    // // Compressors - random block len
    // runner.run::<lib_upack_rng::UpackRandomLenCompressBase>();
    // runner.run::<lib_upack_rng::UpackRandomLenCompressDelta>();
    // runner.run::<lib_upack_rng::UpackRandomLenCompressDelta1>();
    //
    // // Compressors - block step len
    // runner.run::<lib_upack::UpackCompressBase<32>>();
    // runner.run::<lib_upack::UpackCompressBase<64>>();
    // runner.run::<lib_upack::UpackCompressBase<96>>();
    // runner.run::<lib_upack::UpackCompressDelta<32>>();
    // runner.run::<lib_upack::UpackCompressDelta<64>>();
    // runner.run::<lib_upack::UpackCompressDelta<96>>();
    // runner.run::<lib_upack::UpackCompressDelta1<32>>();
    // runner.run::<lib_upack::UpackCompressDelta1<64>>();
    // runner.run::<lib_upack::UpackCompressDelta1<96>>();
    //
    // // Deompressors - block step len
    // runner.run::<lib_upack::UpackDecompressBase<32>>();
    // runner.run::<lib_upack::UpackDecompressBase<64>>();
    // runner.run::<lib_upack::UpackDecompressBase<96>>();
    // runner.run::<lib_upack::UpackDecompressDelta<32>>();
    // runner.run::<lib_upack::UpackDecompressDelta<64>>();
    // runner.run::<lib_upack::UpackDecompressDelta<96>>();
    // runner.run::<lib_upack::UpackDecompressDelta1<32>>();
    // runner.run::<lib_upack::UpackDecompressDelta1<64>>();
    // runner.run::<lib_upack::UpackDecompressDelta1<96>>();

    tracing::info!("benchmark complete, results:");

    runner.display()
}
