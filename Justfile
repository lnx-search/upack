#!/usr/bin/env just --justfile

release:
    cargo build --release

clippy:
    cargo +nightly clippy

format:
    cargo +nightly fmt --all

build:
    cargo build --all-features

[arg("features", long="features", help="The upack features to enable")]
[arg("duration", long="duration", help="The duration to bench each routine")]
[arg("kind", long="kind", help="The datatype kind to benchmark")]
bench kind="uint32" features="avx512,avx2,neon" duration="15s":
    cargo run -p benchmark --release --no-default-features {{ if features == "" { "" } else { "--features " + features } }} -- {{kind}} --duration {{duration}}

test:
    RUSTFLAGS="-Ctarget-cpu=native" cargo nextest run --workspace --no-fail-fast --all-features

asm target="":
    cargo asm --all-features -p upack --lib --simplify {{target}}

mca target:
    cargo asm --all-features -p upack --lib --simplify {{target}} --mca