use clap::Parser;

mod generate;
mod lib_bitpacking;
mod lib_upack;
mod routine;

#[derive(Debug, Parser)]
struct Args {
    #[arg(short = 'd', long = "duration", default_value = "60s")]
    /// The time to run for each routine in some human-readable format, i.e. "10m".
    run_duration: humantime::Duration,
    #[arg(short = 's', long, default_value = "40000")]
    /// The number of sample blocks to generate per run.
    sample_size: usize,
    #[arg(short, long, default_value = "")]
    /// A prefix filter to only run routines that start with the given input.
    filter: String,
}

fn main() {
    let args = Args::parse();

    tracing::info!("starting benchmark args={args:?}");

    let config = Config {
        sample_size: args.sample_size,
        run_duration: args.run_duration.into(),
    };

    tracing::info!("benchmark complete");
}

#[derive(Debug, Copy, Clone)]
pub struct Config {
    pub sample_size: usize,
    pub run_duration: std::time::Duration,
}
