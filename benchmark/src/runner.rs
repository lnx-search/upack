use std::time::{Duration, Instant};

use comfy_table::Table;
use comfy_table::modifiers::UTF8_ROUND_CORNERS;
use comfy_table::presets::UTF8_FULL;
use upack::X128;

use crate::routine::Routine;

const WARMUP_TIME: Duration = Duration::from_secs(10);

#[derive(Debug, Clone)]
/// The runner config determines the size of samples and time spent on the bench for a particular
/// routine. Typically, the longer the run and largish the sample, the more accurate the results.
pub struct Config {
    pub sample_size: usize,
    pub seed: u64,
    pub max_gap: u32,
    pub max_bits: u8,
    pub corpus_size: u32,
    pub run_duration: Duration,
    pub filter: String,
}

/// The benchmark runner runs multiple routines then produces a result table.
pub struct RunContext {
    config: Config,
    table: Table,
}

impl From<Config> for RunContext {
    fn from(config: Config) -> Self {
        let mut table = Table::new();
        table
            .load_preset(UTF8_FULL)
            .apply_modifier(UTF8_ROUND_CORNERS)
            .set_header([
                "Routine Name",
                "Min Avg Sample Time",
                "Max Avg Sample Time",
                "True Avg Sample Time",
                "Integer Throughput",
                "Bytes Throughput",
            ]);

        Self { config, table }
    }
}

impl RunContext {
    /// Run the target routine.
    pub fn run<R: Routine>(&mut self) {
        let name = R::name();
        if !name.starts_with(&self.config.filter) {
            tracing::info!("skipping {name}");
            return;
        }

        tracing::info!("running {name}");

        let mut routine = R::default();

        tracing::info!("starting warmup");
        let start = Instant::now();
        while start.elapsed() < WARMUP_TIME {
            let samples = crate::generate::sample_input(
                self.config.sample_size,
                self.config.seed,
                1,
                self.config.max_gap,
                self.config.max_bits,
                self.config.corpus_size,
            );
            let mut prepared = std::hint::black_box(routine.prep(samples));
            routine.execute(&mut prepared);
        }

        tracing::info!("starting measurements");

        let mut timing_samples = Vec::new();
        let mut total_execution_time = Duration::default();
        let mut total_samples_processed = 0;

        let measurements_start = Instant::now();
        loop {
            let samples = crate::generate::sample_input(
                self.config.sample_size,
                self.config.seed,
                1,
                self.config.max_gap,
                self.config.max_bits,
                self.config.corpus_size,
            );

            let mut prepared = std::hint::black_box(routine.prep(samples));

            let start = Instant::now();
            routine.execute(&mut prepared);
            let elapsed = start.elapsed();

            // Drop after timing.
            drop(prepared);

            timing_samples.push(elapsed / self.config.sample_size as u32);
            total_execution_time += elapsed;
            total_samples_processed += self.config.sample_size;

            if measurements_start.elapsed() >= self.config.run_duration {
                break;
            }
        }

        tracing::info!(
            "run complete! processed {total_samples_processed} samples in {}",
            humantime::Duration::from(measurements_start.elapsed()),
        );

        let total_ints_processed = total_samples_processed * X128;
        let total_bytes_processed = total_ints_processed * size_of::<u32>();

        let ints_per_sec = total_ints_processed as f32 / total_execution_time.as_secs_f32();
        let bytes_per_sec = total_bytes_processed as f32 / total_execution_time.as_secs_f32();

        self.push_sample(
            name,
            min_sample(&timing_samples),
            max_sample(&timing_samples),
            total_execution_time / total_samples_processed as u32,
            ints_per_sec,
            bytes_per_sec,
        )
    }

    /// Display the results table.
    pub fn display(&self) {
        println!("\n{}", self.table)
    }

    fn push_sample(
        &mut self,
        name: &str,
        min_avg_sample_time: Duration,
        max_avg_sample_time: Duration,
        true_avg_sample_time: Duration,
        ints_per_sec: f32,
        bytes_per_sec: f32,
    ) {
        self.table.add_row(vec![
            name.to_string(),
            humantime::format_duration(min_avg_sample_time).to_string(),
            humantime::format_duration(max_avg_sample_time).to_string(),
            humantime::format_duration(true_avg_sample_time).to_string(),
            pretty_print_ints(ints_per_sec),
            pretty_print_bytes(bytes_per_sec),
        ]);
    }
}

fn min_sample(samples: &[Duration]) -> Duration {
    samples
        .iter()
        .min()
        .copied()
        .unwrap_or_else(Duration::default)
}

fn max_sample(samples: &[Duration]) -> Duration {
    samples
        .iter()
        .max()
        .copied()
        .unwrap_or_else(Duration::default)
}

fn pretty_print_bytes(rate: f32) -> String {
    let opt = humansize::DECIMAL.suffix("/sec");
    humansize::format_size(rate as u64, opt)
}

fn pretty_print_ints(rate: f32) -> String {
    if rate >= 1_000_000_000.0 {
        let reduced = rate / 1_000_000_000.0;
        format!("{reduced:.2}B int/sec")
    } else if rate >= 1_000_000.0 {
        let reduced = rate / 1_000_000.0;
        format!("{reduced:.2}M int/sec")
    } else if rate >= 1_000.0 {
        let reduced = rate / 1_000.0;
        format!("{reduced:.2}K int/sec")
    } else {
        format!("{rate:.2} int/sec")
    }
}
