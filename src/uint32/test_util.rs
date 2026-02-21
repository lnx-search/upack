use std::collections::BTreeMap;

use serde::Deserialize;

use crate::{X64, X128};

#[rustfmt::skip]
/// The expected layout for unordered packing functions.
pub const PACK_U16_TO_U8_EXPECTED_UNORDERED_LAYOUT: [u8; X64] = [
    0, 32, 1, 33, 2, 34, 3, 35,
    4, 36, 5, 37, 6, 38, 7, 39,
    8, 40, 9, 41, 10, 42, 11, 43,
    12, 44, 13, 45, 14, 46, 15, 47,
    16, 48, 17, 49, 18, 50, 19, 51,
    20, 52, 21, 53, 22, 54, 23, 55,
    24, 56, 25, 57, 26, 58, 27, 59,
    28, 60, 29, 61, 30, 62, 31, 63,
];
#[rustfmt::skip]
/// The expected layout for unordered packing functions.
pub const PACK_U32_TO_U16_EXPECTED_UNORDERED_LAYOUT: [u16; X64] = [
    0, 16, 1, 17,
    2, 18, 3, 19,
    4, 20, 5, 21,
    6, 22, 7, 23,
    8, 24, 9, 25,
    10, 26, 11, 27,
    12, 28, 13, 29,
    14, 30, 15, 31,
    32, 48, 33, 49,
    34, 50, 35, 51,
    36, 52, 37, 53,
    38, 54, 39, 55,
    40, 56, 41, 57,
    42, 58, 43, 59,
    44, 60, 45, 61,
    46, 62, 47, 63,
];
#[rustfmt::skip]
/// The expected layout for unordered packing functions.
pub const PACK_U32_TO_U8_EXPECTED_UNORDERED_LAYOUT: [u8; X64] = [
    0, 32, 16, 48,
    1, 33, 17, 49,
    2, 34, 18, 50,
    3, 35, 19, 51,
    4, 36, 20, 52,
    5, 37, 21, 53,
    6, 38, 22, 54,
    7, 39, 23, 55,
    8, 40, 24, 56,
    9, 41, 25, 57,
    10, 42, 26, 58,
    11, 43, 27, 59,
    12, 44, 28, 60,
    13, 45, 29, 61,
    14, 46, 30, 62,
    15, 47, 31, 63,
];

#[derive(Deserialize)]
struct LayoutMetadata {
    seeds: Vec<u64>,
    offsets: BTreeMap<String, Vec<BlockMetadata>>,
}

#[derive(Deserialize, Clone)]
struct BlockMetadata {
    seed: u64,
    compressed_start: usize,
    compressed_length: usize,
    input_start: usize,
}

/// The regression layout allows a test to lookup the expected
/// compressed format for a given
pub struct RegressionLayout {
    metadata: LayoutMetadata,
    inputs_data: BTreeMap<u64, Vec<u8>>,
    compressed_data: BTreeMap<u64, Vec<u8>>,
}

impl RegressionLayout {
    /// Iterate over tests to see if the (de)compressed output of the various routines, are as expected.
    pub fn iter_tests(&self) -> impl Iterator<Item = (usize, u8, &[u32; X128], &[u8])> {
        (1..=X128)
            .flat_map(|len| (1..=32).map(move |bit_len: u8| (len, bit_len)))
            .flat_map(move |(len, bit_len)| {
                let key = format!("len:{len},bit:{bit_len}");
                let blocks = self.metadata.offsets.get(&key).expect("unknown key");

                blocks.iter().cloned().map(move |meta| {
                    let data = self.inputs_data.get(&meta.seed).expect("unknown seed");
                    let input_data: &[u32] = bytemuck::cast_slice(data);

                    let compressed = self.compressed_data.get(&meta.seed).expect("unknown seed");

                    let input: &[u32; X128] =
                        (input_data[meta.input_start..][..X128]).try_into().unwrap();
                    let output = &compressed[meta.compressed_start..][..meta.compressed_length];
                    (len, bit_len, input, output)
                })
            })
    }
}

/// Load the v1 layout samples from the data folder for regression testing.
pub fn load_uint32_regression_layout() -> RegressionLayout {
    let layout_path = std::path::Path::new("data/v1-layout-uint32/");

    let metadata_bytes =
        std::fs::read(layout_path.join("metadata.json")).expect("read v1 layout metadata");
    let metadata: LayoutMetadata =
        serde_json::from_slice(&metadata_bytes).expect("deserialize metadata");

    let mut inputs_data = BTreeMap::new();
    let mut compressed_data = BTreeMap::new();

    for seed in metadata.seeds.iter().copied() {
        let inputs =
            std::fs::read(layout_path.join(format!("seeded-{seed}.raw"))).expect("read raw inputs");
        let compressed = std::fs::read(layout_path.join(format!("seeded-{seed}.upack")))
            .expect("read compressed outputs");
        inputs_data.insert(seed, inputs);
        compressed_data.insert(seed, compressed);
    }

    RegressionLayout {
        metadata,
        inputs_data,
        compressed_data,
    }
}
