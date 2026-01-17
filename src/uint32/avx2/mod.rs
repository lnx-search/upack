mod data;
pub mod modifiers;
pub mod pack_x128;
pub mod unpack_x128;
mod utils;

#[inline]
/// Returns `true` if the runtime CPU can safely execute the AVX2 backed implementation.
pub fn can_use() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
}
