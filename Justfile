#!/usr/bin/env just --justfile

release:
    cargo build --release

lint:
    cargo +nightly clippy

format:
    cargo +nightly fmt --all

bench:
    cargo bench

test:
    RUSTFLAGS="-Ctarget-cpu=native" cargo nextest run --workspace