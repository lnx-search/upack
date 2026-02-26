use upack::X128;

fn load_sample_u32_doc_id_data_x128() -> Vec<[u32; X128]> {
    let raw_data_le = std::fs::read("data/wikipedia-sample-docids.bin")
        .expect("unable to read wikipedia-sample-docids.bin");
    let slice: &[[u32; X128]] = bytemuck::cast_slice(&raw_data_le);
    slice.to_vec()
}

#[test]
fn test_uint32_compress_decompress() {
    let samples = load_sample_u32_doc_id_data_x128();

    let mut compressed = [0; upack::uint32::X128_MAX_OUTPUT_LEN];
    let mut decompressed: [u32; X128] = [0; X128];
    for sample in samples.iter() {
        let details = upack::compress(X128, sample, &mut compressed);
        let read_n = upack::decompress(
            X128,
            details.compressed_bit_length,
            &compressed,
            &mut decompressed,
        );
        assert_eq!(read_n, details.bytes_written);
    }
}

#[test]
fn test_uint32_compress_decompress_delta() {
    let mut samples = load_sample_u32_doc_id_data_x128();

    let mut compressed = [0; upack::uint32::X128_MAX_OUTPUT_LEN];
    let mut decompressed: [u32; X128] = [0; X128];
    for sample in samples.iter_mut() {
        let details = upack::compress_delta(0, X128, sample, &mut compressed);
        let read_n = upack::decompress_delta(
            0,
            X128,
            details.compressed_bit_length,
            &compressed,
            &mut decompressed,
        );
        assert_eq!(read_n, details.bytes_written);
    }
}

#[test]
fn test_uint32_compress_decompress_delta1() {
    let mut samples = load_sample_u32_doc_id_data_x128();

    let mut compressed = [0; upack::uint32::X128_MAX_OUTPUT_LEN];
    let mut decompressed: [u32; X128] = [0; X128];
    for sample in samples.iter_mut() {
        let details = upack::compress_delta1(0, X128, sample, &mut compressed);
        let read_n = upack::decompress_delta1(
            0,
            X128,
            details.compressed_bit_length,
            &compressed,
            &mut decompressed,
        );
        assert_eq!(read_n, details.bytes_written);
    }
}

#[test]
fn test_uint16_compress_decompress() {
    let sample: [u16; X128] = std::array::from_fn(|i| i as u16);

    let mut compressed = [0; upack::uint16::X128_MAX_OUTPUT_LEN];
    let mut decompressed: [u32; X128] = [0; X128];
    let details = upack::compress(X128, &sample, &mut compressed);
    let read_n = upack::decompress(
        X128,
        details.compressed_bit_length,
        &compressed,
        &mut decompressed,
    );
    assert_eq!(read_n, details.bytes_written);
}

#[test]
fn test_uint16_compress_decompress_delta() {
    let mut sample: [u16; X128] = std::array::from_fn(|i| i as u16);

    let mut compressed = [0; upack::uint16::X128_MAX_OUTPUT_LEN];
    let mut decompressed: [u32; X128] = [0; X128];
    let details = upack::compress_delta(0, X128, &mut sample, &mut compressed);
    let read_n = upack::decompress_delta1(
        0,
        X128,
        details.compressed_bit_length,
        &compressed,
        &mut decompressed,
    );
    assert_eq!(read_n, details.bytes_written);
}

#[test]
fn test_uint16_compress_decompress_delta1() {
    let mut sample: [u16; X128] = std::array::from_fn(|i| (i + 1) as u16);

    let mut compressed = [0; upack::uint16::X128_MAX_OUTPUT_LEN];
    let mut decompressed: [u32; X128] = [0; X128];
    let details = upack::compress_delta1(0, X128, &mut sample, &mut compressed);
    let read_n = upack::decompress_delta1(
        0,
        X128,
        details.compressed_bit_length,
        &compressed,
        &mut decompressed,
    );
    assert_eq!(read_n, details.bytes_written);
}
