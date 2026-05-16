#[cfg(not(target_family = "wasm"))]
pub mod catalog;
pub mod gguf;
#[cfg(not(target_family = "wasm"))]
pub mod metal;
#[cfg(not(target_family = "wasm"))]
pub mod mmap;
pub mod model;
pub mod runtime;
pub mod sampling;
#[cfg(all(not(target_family = "wasm"), feature = "server"))]
pub mod server;
pub mod simd;
pub mod tokenizer;

pub use runtime::{
    ChatMessage, ChatRole, EmbeddingResult, GenerationOptions, GenerationResult, GenerationStats,
    LoadInfo, Runner,
};
