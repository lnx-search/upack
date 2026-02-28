#[inline]
/// Returns `true` if the runtime CPU can safely execute the NEON backed implementation.
pub fn can_use() -> bool {
    std::arch::is_aarch64_feature_detected!("neon")
}
