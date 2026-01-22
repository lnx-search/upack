
#[inline]
pub(super) fn split_slice<T, const N1: usize, const N2: usize>(data: &[T; N1]) -> [&[T; N2]; 2] {
    let left: &[T; N2] = (&data[..N2]).try_into().unwrap();
    let right: &[T; N2] = (&data[N2..]).try_into().unwrap();
    [left, right]
}

