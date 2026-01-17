//! 128-bit register based acceleration, this includes both NEON and SSE based implementations.

#[cfg(all(target_arch = "x86_64", feature = "sse"))]
#[inline]
/// Returns `true` if the runtime CPU can safely execute the SSE4.1 backed implementation.
pub fn can_use() -> bool {
    std::arch::is_x86_feature_detected!("sse4.1")
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
#[inline]
/// Returns `true` if the runtime CPU can safely execute the NEON backed implementation.
pub fn can_use() -> bool {
    std::arch::is_aarch64_feature_detected!("neon")
}
