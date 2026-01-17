#[inline]
/// Returns `true` if the runtime CPU can safely execute the AVX2 backed implementation.
pub fn can_use() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
}
