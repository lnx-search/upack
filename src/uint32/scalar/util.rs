use super::polyfill::*;

/// Pack 8 sets of registers containing 32-bit elements and produce 2 registers holding
/// 8-bit elements.
///
/// The order of elements is maintained.
pub(super) fn pack_u32_to_u8_ordered(data: [u32x8; 8]) -> [u8x32; 2] {
    let partially_packed = pack_u32_to_u16_ordered(data);
    pack_u16_to_u8_ordered(partially_packed)
}

/// Pack 8 sets of registers containing 32-bit elements and produce 4 registers holding
/// 16-bit elements.
///
/// The order of elements is maintained.
pub(crate) fn pack_u32_to_u16_ordered(data: [u32x8; 8]) -> [u16x16; 4] {
    [
        _scalar_combine_u16x8(
            _scalar_cvteu16_u32x8(data[0]),
            _scalar_cvteu16_u32x8(data[1]),
        ),
        _scalar_combine_u16x8(
            _scalar_cvteu16_u32x8(data[2]),
            _scalar_cvteu16_u32x8(data[3]),
        ),
        _scalar_combine_u16x8(
            _scalar_cvteu16_u32x8(data[4]),
            _scalar_cvteu16_u32x8(data[5]),
        ),
        _scalar_combine_u16x8(
            _scalar_cvteu16_u32x8(data[6]),
            _scalar_cvteu16_u32x8(data[7]),
        ),
    ]
}

/// Pack 4 sets of registers containing 16-bit elements and produce 2 registers holding
/// 9-bit elements.
///
/// The order of elements is maintained.
pub(crate) fn pack_u16_to_u8_ordered(data: [u16x16; 4]) -> [u8x32; 2] {
    [
        _scalar_combine_u8x16(
            _scalar_cvteu8_u16x16(data[0]),
            _scalar_cvteu8_u16x16(data[1]),
        ),
        _scalar_combine_u8x16(
            _scalar_cvteu8_u16x16(data[2]),
            _scalar_cvteu8_u16x16(data[3]),
        ),
    ]
}

/// Unpack 2 sets of registers containing 8-bit elements and produce 8 registers holding
/// 32-bit elements.
pub(crate) fn unpack_u8_to_u32_ordered(data: [u8x32; 2]) -> [u32x8; 8] {
    let partially_unpacked = unpack_u8_to_u16_ordered(data);
    unpack_u16_to_u32_ordered(partially_unpacked)
}

/// Unpack 2 sets of registers containing 8-bit elements and produce 4 registers holding
/// 16-bit elements.
pub(crate) fn unpack_u8_to_u16_ordered(data: [u8x32; 2]) -> [u16x16; 4] {
    let parts = [
        _scalar_extract_u8x32::<0>(data[0]),
        _scalar_extract_u8x32::<1>(data[0]),
        _scalar_extract_u8x32::<0>(data[1]),
        _scalar_extract_u8x32::<1>(data[1]),
    ];

    [
        _scalar_cvteu16_u8x16(parts[0]),
        _scalar_cvteu16_u8x16(parts[1]),
        _scalar_cvteu16_u8x16(parts[2]),
        _scalar_cvteu16_u8x16(parts[3]),
    ]
}

/// Unpack 4 sets of registers containing 16-bit elements and produce 8 registers holding
/// 32-bit elements.
pub(crate) fn unpack_u16_to_u32_ordered(data: [u16x16; 4]) -> [u32x8; 8] {
    let split_u16s = [
        _scalar_extract_u16x16::<0>(data[0]),
        _scalar_extract_u16x16::<1>(data[0]),
        _scalar_extract_u16x16::<0>(data[1]),
        _scalar_extract_u16x16::<1>(data[1]),
        _scalar_extract_u16x16::<0>(data[2]),
        _scalar_extract_u16x16::<1>(data[2]),
        _scalar_extract_u16x16::<0>(data[3]),
        _scalar_extract_u16x16::<1>(data[3]),
    ];

    [
        _scalar_cvteu32_u16x8(split_u16s[0]),
        _scalar_cvteu32_u16x8(split_u16s[1]),
        _scalar_cvteu32_u16x8(split_u16s[2]),
        _scalar_cvteu32_u16x8(split_u16s[3]),
        _scalar_cvteu32_u16x8(split_u16s[4]),
        _scalar_cvteu32_u16x8(split_u16s[5]),
        _scalar_cvteu32_u16x8(split_u16s[6]),
        _scalar_cvteu32_u16x8(split_u16s[7]),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::X64;
    use crate::uint32::scalar::data::load_u32x64;

    #[test]
    fn test_pack_u32_to_u16_ordered() {
        let input = std::array::from_fn(|i| i as u32);

        let data = load_u32x64(&input);
        let packed = pack_u32_to_u16_ordered(data);

        let expected = std::array::from_fn(|i| i as u16);
        let view = unsafe { std::mem::transmute::<[u16x16; 4], [u16; X64]>(packed) };
        assert_eq!(view, expected);
    }

    #[test]
    fn test_pack_u32_to_u8_ordered() {
        let input = std::array::from_fn(|i| i as u32);

        let data = load_u32x64(&input);
        let packed = pack_u32_to_u8_ordered(data);

        let expected = std::array::from_fn(|i| i as u8);
        let view = unsafe { std::mem::transmute::<[u8x32; 2], [u8; X64]>(packed) };
        assert_eq!(view, expected);
    }

    #[test]
    fn test_pack_u16_to_u8_ordered() {
        let input = std::array::from_fn(|i| i as u16);

        let data = unsafe { std::mem::transmute::<[u16; X64], [u16x16; 4]>(input) };
        let packed = pack_u16_to_u8_ordered(data);

        let expected = std::array::from_fn(|i| i as u8);
        let view = unsafe { std::mem::transmute::<[u8x32; 2], [u8; X64]>(packed) };
        assert_eq!(view, expected);
    }

    #[test]
    fn test_unpack_u8_to_u16_ordered() {
        let input = std::array::from_fn(|i| i as u8);

        let data = unsafe { std::mem::transmute::<[u8; X64], [u8x32; 2]>(input) };
        let unpacked = unpack_u8_to_u16_ordered(data);

        let expected = std::array::from_fn(|i| i as u16);
        let view = unsafe { std::mem::transmute::<[u16x16; 4], [u16; X64]>(unpacked) };
        assert_eq!(view, expected);
    }

    #[test]
    fn test_unpack_u16_to_u32_ordered() {
        let input = std::array::from_fn(|i| i as u16);

        let data = unsafe { std::mem::transmute::<[u16; X64], [u16x16; 4]>(input) };
        let unpacked = unpack_u16_to_u32_ordered(data);

        let expected = std::array::from_fn(|i| i as u32);
        let view = unsafe { std::mem::transmute::<[u32x8; 8], [u32; X64]>(unpacked) };
        assert_eq!(view, expected);
    }

    #[test]
    fn test_unpack_u8_to_u32_ordered() {
        let input = std::array::from_fn(|i| i as u8);

        let data = unsafe { std::mem::transmute::<[u8; X64], [u8x32; 2]>(input) };
        let unpacked = unpack_u8_to_u32_ordered(data);

        let expected = std::array::from_fn(|i| i as u32);
        let view = unsafe { std::mem::transmute::<[u32x8; 8], [u32; X64]>(unpacked) };
        assert_eq!(view, expected);
    }
}
