use std::hint::black_box;

use bitpacking::BitPacker;
use divan::Bencher;
use upack::X128;
use upack::uint32::X128_MAX_OUTPUT_LEN;

mod utils;

fn main() {
    divan::main();
}

#[divan::bench(sample_size = 1000, sample_count = 5000)]
fn bench_upack_decompress_x128(bencher: Bencher) {
    let sample_data = utils::load_sample_u32_doc_id_data_x128();
    let total_entries = sample_data.len() * X128;

    let mut buffer = Vec::new();
    let mut bit_lengths = Vec::new();
    let mut compressed = [0; X128_MAX_OUTPUT_LEN];
    for sample in sample_data {
        let details = upack::compress(X128, &sample, &mut compressed);
        buffer.extend_from_slice(&compressed[..details.bytes_written]);
        bit_lengths.push(details.compressed_bit_length);
    }

    let mut out = [0; X128];
    bencher
        .counter(divan::counter::ItemsCount::new(total_entries))
        .bench_local(|| {
            let mut offset = 0;
            for bit_length in bit_lengths.iter().copied() {
                let data = &buffer[offset..];
                offset += unsafe {
                    upack::uint32::avx2::unpack_x128(
                        bit_length,
                        black_box(data),
                        black_box(&mut out),
                        X128,
                    )
                };
            }
        });
}

#[divan::bench(sample_size = 1000, sample_count = 5000)]
fn bench_upack_decompress_delta_x128(bencher: Bencher) {
    let sample_data = utils::load_sample_u32_doc_id_data_x128();
    let total_entries = sample_data.len() * X128;

    let mut buffer = Vec::new();
    let mut bit_lengths = Vec::new();
    let mut compressed = [0; X128_MAX_OUTPUT_LEN];
    for mut sample in sample_data {
        let details = upack::compress_delta(0, X128, &mut sample, &mut compressed);
        buffer.extend_from_slice(&compressed[..details.bytes_written]);
        bit_lengths.push(details.compressed_bit_length);
    }

    let mut out = [0; X128];
    bencher
        .counter(divan::counter::ItemsCount::new(total_entries))
        .bench_local(|| {
            let mut offset = 0;
            for bit_length in bit_lengths.iter().copied() {
                let data = &buffer[offset..];
                offset += unsafe {
                    upack::uint32::avx2::unpack_delta_x128(
                        bit_length,
                        0,
                        black_box(data),
                        black_box(&mut out),
                        X128,
                    )
                };
            }
        });
}

#[divan::bench(sample_size = 1000, sample_count = 5000)]
fn bench_upack_decompress_delta1_x128(bencher: Bencher) {
    let sample_data = utils::load_sample_u32_doc_id_data_x128();
    let total_entries = sample_data.len() * X128;

    let mut buffer = Vec::new();
    let mut bit_lengths = Vec::new();
    let mut compressed = [0; X128_MAX_OUTPUT_LEN];
    for mut sample in sample_data {
        let details = upack::compress_delta1(0, X128, &mut sample, &mut compressed);
        buffer.extend_from_slice(&compressed[..details.bytes_written]);
        bit_lengths.push(details.compressed_bit_length);
    }

    let mut out = [0; X128];
    bencher
        .counter(divan::counter::ItemsCount::new(total_entries))
        .bench_local(|| {
            let mut offset = 0;
            for bit_length in bit_lengths.iter().copied() {
                let data = &buffer[offset..];
                offset += unsafe {
                    upack::uint32::avx2::unpack_delta1_x128(
                        bit_length,
                        0,
                        black_box(data),
                        black_box(&mut out),
                        X128,
                    )
                };
            }
        });
}

#[divan::bench(sample_size = 1000, sample_count = 5000)]
fn bench_bitpacking_decompress_x128(bencher: Bencher) {
    let sample_data = utils::load_sample_u32_doc_id_data_x128();
    let total_entries = sample_data.len() * X128;

    let packer = bitpacking::BitPacker4x::new();

    let mut buffer = Vec::new();
    let mut bit_lengths = Vec::new();
    let mut compressed = [0; X128_MAX_OUTPUT_LEN];
    for sample in sample_data {
        let bits = packer.num_bits(&sample);
        let bytes_written = packer.compress(&sample, &mut compressed, bits);
        buffer.extend_from_slice(&compressed[..bytes_written]);
        bit_lengths.push(bits);
    }

    let mut out = [0; X128];
    bencher
        .counter(divan::counter::ItemsCount::new(total_entries))
        .bench_local(|| {
            let mut offset = 0;
            for bit_length in bit_lengths.iter().copied() {
                let data = &buffer[offset..];
                offset += packer.decompress(black_box(data), black_box(&mut out), bit_length);
            }
        });
}

#[divan::bench(sample_size = 1000, sample_count = 5000)]
fn bench_bitpacking_decompress_delta_x128(bencher: Bencher) {
    let sample_data = utils::load_sample_u32_doc_id_data_x128();
    let total_entries = sample_data.len() * X128;

    let packer = bitpacking::BitPacker4x::new();

    let mut buffer = Vec::new();
    let mut bit_lengths = Vec::new();
    let mut compressed = [0; X128_MAX_OUTPUT_LEN];
    for sample in sample_data {
        let bits = packer.num_bits_sorted(0, &sample);
        let bytes_written = packer.compress_sorted(0, &sample, &mut compressed, bits);
        buffer.extend_from_slice(&compressed[..bytes_written]);
        bit_lengths.push(bits);
    }

    let mut out = [0; X128];
    bencher
        .counter(divan::counter::ItemsCount::new(total_entries))
        .bench_local(|| {
            let mut offset = 0;
            for bit_length in bit_lengths.iter().copied() {
                let data = &buffer[offset..];
                offset +=
                    packer.decompress_sorted(0, black_box(data), black_box(&mut out), bit_length);
            }
        });
}

#[divan::bench(sample_size = 1000, sample_count = 5000)]
fn bench_bitpacking_decompress_delta1_x128(bencher: Bencher) {
    let sample_data = utils::load_sample_u32_doc_id_data_x128();
    let total_entries = sample_data.len() * X128;

    let packer = bitpacking::BitPacker4x::new();

    let mut buffer = Vec::new();
    let mut bit_lengths = Vec::new();
    let mut compressed = [0; X128_MAX_OUTPUT_LEN];
    for sample in sample_data {
        let bits = packer.num_bits_strictly_sorted(None, &sample);
        let bytes_written = packer.compress_strictly_sorted(None, &sample, &mut compressed, bits);
        buffer.extend_from_slice(&compressed[..bytes_written]);
        bit_lengths.push(bits);
    }

    let mut out = [0; X128];
    bencher
        .counter(divan::counter::ItemsCount::new(total_entries))
        .bench_local(|| {
            let mut offset = 0;
            for bit_length in bit_lengths.iter().copied() {
                let data = &buffer[offset..];
                offset += packer.decompress_strictly_sorted(
                    None,
                    black_box(data),
                    black_box(&mut out),
                    bit_length,
                );
            }
        });
}
