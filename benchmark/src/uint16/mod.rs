use clap::Parser;

mod generate;
mod lib_upack;
mod routine;
mod runner;

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
    #[arg(long, default_value_t = u16::MAX)]
    /// The maximum gap to use between the sequential integers in the block.
    ///
    /// Useful for delta based comparisons, but probably shouldn't be touched
    /// unless you know what you're doing.
    max_gap: u16,
    #[arg(long, default_value = "16", value_parser = clap::value_parser!(u8).range(1..=16))]
    /// The maximum bit length of produced integers for the sample inputs.
    max_bits: u8,
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
        run_duration: args.run_duration.into(),
        filter: args.filter,
    };

    let mut runner = runner::RunContext::from(config);

    // Compressors
    runner.run::<lib_upack::UpackCompressBase>();
    runner.run::<lib_upack::UpackCompressDelta>();
    runner.run::<lib_upack::UpackCompressDelta1>();

    // Decompressors
    runner.run::<lib_upack::UpackDecompressBase>();
    runner.run::<lib_upack::UpackDecompressDelta>();
    runner.run::<lib_upack::UpackDecompressDelta1>();

    tracing::info!("benchmark complete, results:");

    runner.display()
}
