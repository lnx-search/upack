use clap::{Parser, Subcommand};

mod uint16;
mod uint32;

#[derive(Debug, Subcommand)]
enum Commands {
    Uint32(uint32::Args),
    Uint16(uint16::Args),
}

#[derive(Debug, Parser)]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

fn main() {
    let args = Args::parse();

    tracing_subscriber::fmt::init();

    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        tracing::info!("cpu feature enabled = neon");
    }

    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f") {
        tracing::info!("cpu feature enabled = avx512");
    }

    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        tracing::info!("cpu feature enabled = avx2");
    }

    match args.command {
        Commands::Uint32(args) => uint32::run_benchmark(args),
        Commands::Uint16(args) => uint16::run_benchmark(args),
    }
}
