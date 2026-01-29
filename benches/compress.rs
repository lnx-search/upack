use std::hint::black_box;
use bitpacking::BitPacker;
use divan::Bencher;
use upack::uint32::X128_MAX_OUTPUT_LEN;
use upack::X128;

mod utils;

fn main() {
    divan::main();
}

#[divan::bench(sample_size = 1000, sample_count = 5000)]
fn bench_upack_compress_x128(bencher: Bencher) {
    let sample_data = utils::load_sample_u32_doc_id_data_x128();
    let total_entries = sample_data.len() * X128;

    let mut out = [0; X128_MAX_OUTPUT_LEN];
    bencher
        .counter(divan::counter::ItemsCount::new(total_entries))
        .bench_local(|| {
            for block in sample_data.iter() {
                unsafe { upack::uint32::avx2::pack_x128(black_box(&mut out), black_box(block), X128) };
            }
        });
}

#[divan::bench(sample_size = 1000, sample_count = 5000)]
fn bench_packer_compress_x128(bencher: Bencher) {
    let sample_data = utils::load_sample_u32_doc_id_data_x128();
    let total_entries = sample_data.len() * X128;

    let packer = bitpacking::BitPacker4x::new();

    let mut out = [0; X128_MAX_OUTPUT_LEN];
    bencher
        .counter(divan::counter::ItemsCount::new(total_entries))
        .bench_local(|| {
            for block in sample_data.iter() {
                let nbits = packer.num_bits(block);
                let n = packer.compress(block, &mut out, nbits);
                black_box(n);
            }
        });
}