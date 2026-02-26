# Data samples for testing

- `wikipedia-sample-docids.bin` is a dump of LE encoded u32 values which are DocIDs produced by tantivy
  after indexing the entire wikipedia text. It contains `12_800` raw values making up `1000` sample blocks.
- `v1-layout-uint32` full permutations of every possible compressed block layout for any given length and bit length
  with 3 variations derived from an RNG seed. Used for regression tests.
- `v1-layout-uint16` full permutations of every possible compressed block layout for any given length and bit length
  with 3 variations derived from an RNG seed. Used for regression tests.