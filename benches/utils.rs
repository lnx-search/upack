use upack::{X128, X256};

/// Load the sample DocID data for 128 block size.
pub fn load_sample_u32_doc_id_data_x128() -> Vec<[u32; X128]> {
    let raw_data_le = std::fs::read("data/wikipedia-sample-docids.bin")
        .expect("unable to read wikipedia-sample-docids.bin");
    let slice: &[[u32; X128]] = bytemuck::cast_slice(&raw_data_le);
    slice.to_vec()
}

/// Load the sample DocID data for 256 block size.
pub fn load_sample_u32_doc_id_data_x256() -> Vec<[u32; X256]> {
    let raw_data_le = std::fs::read("data/wikipedia-sample-docids.bin")
        .expect("unable to read wikipedia-sample-docids.bin");
    let slice: &[[u32; X256]] = bytemuck::cast_slice(&raw_data_le);
    slice.to_vec()
}
