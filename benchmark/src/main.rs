use clap::Parser;

use crate::runner::Config;

mod generate;
mod lib_bitpacking;
mod lib_upack;
mod routine;
mod runner;

#[derive(Debug, Parser)]
struct Args {
    #[arg(short = 'd', long = "duration", default_value = "30s")]
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
    #[arg(long, default_value = "24", value_parser = clap::value_parser!(u8).range(1..=32))]
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

fn main() {
    let args = Args::parse();

    tracing_subscriber::fmt::init();

    tracing::info!("starting benchmark args={args:?}");

    let config = Config {
        sample_size: args.sample_size,
        seed: args.seed,
        max_gap: args.max_gap,
        max_bits: args.max_bits,
        corpus_size: args.corpus_size,
        run_duration: args.run_duration.into(),
        filter: args.filter,
    };

    let mut runner = runner::RunContext::from(config);
    runner.run::<lib_upack::UpackCompressBase>();
    runner.run::<lib_upack::UpackCompressDelta>();
    runner.run::<lib_upack::UpackCompressDelta1>();
    runner.run::<lib_bitpacking::BitpackingCompressBase>();
    runner.run::<lib_bitpacking::BitpackingCompressDelta>();
    runner.run::<lib_bitpacking::BitpackingCompressDelta1>();

    tracing::info!("benchmark complete, results:");

    runner.display()
}
