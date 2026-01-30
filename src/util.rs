#[inline]
pub(super) fn split_slice<T, const N1: usize, const N2: usize>(data: &[T; N1]) -> [&[T; N2]; 2] {
    let left: &[T; N2] = (&data[..N2]).try_into().unwrap();
    let right: &[T; N2] = (&data[N2..]).try_into().unwrap();
    [left, right]
}

#[inline]
pub(super) fn split_slice_mut<T, const N1: usize, const N2: usize>(
    data: &mut [T; N1],
) -> [&mut [T; N2]; 2] {
    let [left, right] = data.get_disjoint_mut([0..N2, N2..N1]).unwrap();
    let left: &mut [T; N2] = left.try_into().unwrap();
    let right: &mut [T; N2] = right.try_into().unwrap();
    [left, right]
}
