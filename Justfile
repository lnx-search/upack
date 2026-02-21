#!/usr/bin/env just --justfile

release:
    cargo build --release

lint:
    cargo +nightly clippy

format:
    cargo +nightly fmt --all

[arg("features", long="features", help="The upack features to enable")]
[arg("duration", long="duration", help="The duration to bench each routine")]
bench features="avx512,avx2,neon" duration="15s":
    cargo run -p benchmark --release --no-default-features {{ if features == "" { "" } else { "--features " + features } }} -- --duration {{duration}}

test:
    RUSTFLAGS="-Ctarget-cpu=native" cargo nextest run --workspace --no-fail-fast