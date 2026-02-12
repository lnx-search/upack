# μpack

A small, highly optimised bitpacking SIMD library with zero dependencies.

Bitpacking blocks of integers with SIMD is not a new idea, with the [simdcomp](https://github.com/fast-pack/simdcomp)
library by Daniel Lemire being one of the most well known implementations. However, this library has one
novel difference; unlike simdcomp and other bitpacking algorithms, μpack supports _variable size output blocks._

This means if you have a block that is not some fixed sized (in μpack's case, 128 elements.) You can safely truncate
the compressed output without data loss.

The reason why this is possible is because blocks that are smaller than 128 elements are packed in a way that maintains
their original ordering. 

## Features

- Optimised AVX512, AVX2 and NEON implementations of the compression routines.
- Optimised scalar fallback which can optimise well for SSE3, LoongArch, etc...
- Variable size output blocks offering better compression ratios than StreamVByte and other algorithms that
  are typically used when there is not enough data to compress a full block.
- Zero dependencies, Zero allocations.
- Supports `u32` and `u16` integers, `u64` is possible, but is not currently on my radar for now.
- Delta and Delta-1 encoding variants available for sorted sequences offering better compression ratios.

## Example

```rust
const EXPECTED_BITLENGTH: usize = 3;

fn main() {
    let mut values = [4; 128];
  
    // Create a buffer to hold our compressed data, must be able to hold the data 
    // assuming worst case compression (aka, no compression.)  
    let mut compressed = [0; upack::uint32::X128_MAX_OUTPUT_LEN];
    
    // Compress 128 integers (which is the max per block.)
    let details = upack::compress(128, &values, &mut compressed);
    assert_eq!(details.compressed_bit_length, EXPECTED_BITLENGTH as u8);
  
    // You can calculate the output size of the compressed block using 
    // the `compressed_size` functions.
    assert_eq!(details.bytes_written, upack::uint32::compressed_size(EXPECTED_BITLENGTH, 128));
    
    let mut decompressed = [0; upack::X128];  // Helpful alias for max block size.
  
    // Get our original values back, the decompressor returns the number of bytes 
    // read in case we're doing some streaming workload. Note that the bit length 
    // of the compressed integers must be known upfront.
    let bytes_read = upack::decompress(
        128,
        details.compressed_bit_length,
        &compressed,
        &mut decompressed,
    );  
    assert_eq!(bytes_read, details.bytes_written);
    
    // If we only have 19 integers we want to pack into the output block, we can set that.
    let details = upack::compress(19, &values, &mut compressed);
    assert_eq!(details.compressed_bit_length, EXPECTED_BITLENGTH as u8);
    assert_eq!(details.bytes_written, upack::uint32::compressed_size(EXPECTED_BITLENGTH, 19));
  
    // Packing 19 integers is a lot smaller than packing 128 of them! - Not perfect though 
    assert_eq!(upack::uint32::compressed_size(EXPECTED_BITLENGTH, 19), 10);
    assert_eq!(upack::uint32::compressed_size(EXPECTED_BITLENGTH, 128), 64);
  
    // Even though the memory still needs to be padded when compressing and decompressing, it's not 
    // something you need to keep when storing on disk, etc...
    compressed[details.bytes_written..].fill(0);
    
    // Works fine!
    let bytes_read = upack::decompress(
      19,
      details.compressed_bit_length,
      &compressed,
      &mut decompressed,
    );
    assert_eq!(bytes_read, details.bytes_written);
}
```