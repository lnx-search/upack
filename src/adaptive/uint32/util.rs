use crate::X128;

#[inline(always)]
/// Delta encodes the provided block of integers and produces an adaptive delta value
/// used for recovering the original data of the block.
pub(super) fn adaptive_delta_encode(
    last_value: &mut u32,
    block: &mut [u32; X128],
    pack_n: usize,
) -> u32 {
    let mut min_delta = u32::MAX;
    for v in block.iter_mut().take(pack_n) {
        let value = *v;
        *v = value.wrapping_sub(*last_value);
        min_delta = min_delta.min(*v);
        *last_value = value;
    }

    for delta in block.iter_mut() {
        *delta -= min_delta;
    }

    min_delta
}
