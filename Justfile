#!/usr/bin/env just --justfile

release:
    cargo build --release

lint:
    cargo +nightly clippy

format:
    cargo +nightly fmt --all

bench features="avx512,avx2,neon":
    cargo run -p benchmark --release --no-default-features --features {{features}}

test:
    RUSTFLAGS="-Ctarget-cpu=native" cargo nextest run --workspace --no-fail-fast