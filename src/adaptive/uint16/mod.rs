#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
mod avx2;
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
mod avx512;
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
mod neon;
mod scalar;
