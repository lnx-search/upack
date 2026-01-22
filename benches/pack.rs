use std::hint::black_box;
use divan::Bencher;
use upack::uint32::X128_MAX_OUTPUT_LEN;
use upack::X128;

mod utils;

fn main() {
    divan::main();
}

#[divan::bench(sample_size = 1000, sample_count = 5000)]
#[cfg_attr(not(target_feature = "avx2"), ignore)]
fn bench_pack_avx2_u9_x128(bencher: Bencher) {
    let sample_data = utils::load_sample_u32_doc_id_data_x128();

    let mut out = [0; X128_MAX_OUTPUT_LEN];
    bencher
        .counter(divan::counter::ItemsCount::new(sample_data.len() * X128))
        .bench_local(|| {
            for sample in sample_data.iter() {
                unsafe { upack::uint32::avx2::pack_x128::to_u9(black_box(&mut out), black_box(sample), X128) };
            }
        });
}

#[divan::bench(sample_size = 1000, sample_count = 5000)]
#[cfg_attr(not(target_feature = "avx512f"), ignore)]
fn bench_pack_avx512_u9_x128(bencher: Bencher) {
    let sample_data = utils::load_sample_u32_doc_id_data_x128();

    let mut out = [0; X128_MAX_OUTPUT_LEN];
    bencher
        .counter(divan::counter::ItemsCount::new(sample_data.len() * X128))
        .bench_local(|| {
            for sample in sample_data.iter() {
                unsafe { upack::uint32::avx512::pack_x128::to_u9(black_box(&mut out), black_box(sample), X128) };
            }
        });
}

#[divan::bench(sample_size = 1000, sample_count = 5000)]
#[cfg_attr(not(target_feature = "avx2"), ignore)]
fn bench_pack_avx2_u13_x128(bencher: Bencher) {
    let sample_data = utils::load_sample_u32_doc_id_data_x128();

    let mut out = [0; X128_MAX_OUTPUT_LEN];
    bencher
        .counter(divan::counter::ItemsCount::new(sample_data.len() * X128))
        .bench_local(|| {
            for sample in sample_data.iter() {
                unsafe { upack::uint32::avx2::pack_x128::to_u13(black_box(&mut out), black_box(sample), X128) };
            }
        });
}

#[divan::bench(sample_size = 1000, sample_count = 5000)]
#[cfg_attr(not(target_feature = "avx512f"), ignore)]
fn bench_pack_avx512_u13_x128(bencher: Bencher) {
    let sample_data = utils::load_sample_u32_doc_id_data_x128();

    let mut out = [0; X128_MAX_OUTPUT_LEN];
    bencher
        .counter(divan::counter::ItemsCount::new(sample_data.len() * X128))
        .bench_local(|| {
            for sample in sample_data.iter() {
                unsafe { upack::uint32::avx512::pack_x128::to_u13(black_box(&mut out), black_box(sample), X128) };
            }
        });
}

#[divan::bench(sample_size = 1000, sample_count = 5000)]
#[cfg_attr(not(target_feature = "avx2"), ignore)]
fn bench_pack_avx2_u17_x128(bencher: Bencher) {
    let sample_data = utils::load_sample_u32_doc_id_data_x128();

    let mut out = [0; X128_MAX_OUTPUT_LEN];
    bencher
        .counter(divan::counter::ItemsCount::new(sample_data.len() * X128))
        .bench_local(|| {
            for sample in sample_data.iter() {
                unsafe { upack::uint32::avx2::pack_x128::to_u17(black_box(&mut out), black_box(sample), X128) };
            }
        });
}

#[divan::bench(sample_size = 1000, sample_count = 5000)]
#[cfg_attr(not(target_feature = "avx512f"), ignore)]
fn bench_pack_avx512_u17_x128(bencher: Bencher) {
    let sample_data = utils::load_sample_u32_doc_id_data_x128();

    let mut out = [0; X128_MAX_OUTPUT_LEN];
    bencher
        .counter(divan::counter::ItemsCount::new(sample_data.len() * X128))
        .bench_local(|| {
            for sample in sample_data.iter() {
                unsafe { upack::uint32::avx512::pack_x128::to_u17(black_box(&mut out), black_box(sample), X128) };
            }
        });
}

#[divan::bench(sample_size = 1000, sample_count = 5000)]
#[cfg_attr(not(target_feature = "avx2"), ignore)]
fn bench_pack_avx2_u23_x128(bencher: Bencher) {
    let sample_data = utils::load_sample_u32_doc_id_data_x128();

    let mut out = [0; X128_MAX_OUTPUT_LEN];
    bencher
        .counter(divan::counter::ItemsCount::new(sample_data.len() * X128))
        .bench_local(|| {
            for sample in sample_data.iter() {
                unsafe { upack::uint32::avx2::pack_x128::to_u17(black_box(&mut out), black_box(sample), X128) };
            }
        });
}

#[divan::bench(sample_size = 1000, sample_count = 5000)]
#[cfg_attr(not(target_feature = "avx512f"), ignore)]
fn bench_pack_avx512_u23_x128(bencher: Bencher) {
    let sample_data = utils::load_sample_u32_doc_id_data_x128();

    let mut out = [0; X128_MAX_OUTPUT_LEN];
    bencher
        .counter(divan::counter::ItemsCount::new(sample_data.len() * X128))
        .bench_local(|| {
            for sample in sample_data.iter() {
                unsafe { upack::uint32::avx512::pack_x128::to_u17(black_box(&mut out), black_box(sample), X128) };
            }
        });
}