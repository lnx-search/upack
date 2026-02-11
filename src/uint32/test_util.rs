use crate::X64;

#[rustfmt::skip]
/// The expected layout for unordered packing functions.
pub const PACK_U16_TO_U8_EXPECTED_UNORDERED_LAYOUT: [u8; X64] = [
    0, 1, 2, 3, 4, 5, 6, 7,
    32, 33, 34, 35, 36, 37, 38, 39,
    8, 9, 10, 11, 12, 13, 14, 15,
    40, 41, 42, 43, 44, 45, 46, 47,
    16, 17, 18, 19, 20, 21, 22, 23,
    48, 49, 50, 51, 52, 53, 54, 55,
    24, 25, 26, 27, 28, 29, 30, 31,
    56, 57, 58, 59, 60, 61, 62, 63,
];
#[rustfmt::skip]
/// The expected layout for unordered packing functions.
pub const PACK_U32_TO_U16_EXPECTED_UNORDERED_LAYOUT: [u16; X64] = [
    0, 1, 2, 3,
    16, 17, 18, 19,
    4, 5, 6, 7,
    20, 21, 22, 23,
    8, 9, 10, 11,
    24, 25, 26, 27,
    12, 13, 14, 15,
    28, 29, 30, 31,
    32, 33, 34, 35,
    48, 49, 50, 51,
    36, 37, 38, 39,
    52, 53, 54, 55,
    40, 41, 42, 43,
    56, 57, 58, 59,
    44, 45, 46, 47,
    60, 61, 62, 63,
];
#[rustfmt::skip]
/// The expected layout for unordered packing functions.
pub const PACK_U32_TO_U8_EXPECTED_UNORDERED_LAYOUT: [u8; X64] = [
    0, 1, 2, 3,
    16, 17, 18, 19,
    32, 33, 34, 35,
    48, 49, 50, 51,
    4, 5, 6, 7,
    20, 21, 22, 23,
    36, 37, 38, 39,
    52, 53, 54, 55,
    8, 9, 10, 11,
    24, 25, 26, 27,
    40, 41, 42, 43,
    56, 57, 58, 59,
    12, 13, 14, 15,
    28, 29, 30, 31,
    44, 45, 46, 47,
    60, 61, 62, 63,
];
