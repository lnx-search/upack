use crate::X64;

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
