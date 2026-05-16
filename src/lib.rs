#[cfg(not(target_family = "wasm"))]
pub mod catalog;
pub mod gguf;
#[cfg(not(target_family = "wasm"))]
pub mod mmap;
pub mod model;
pub mod runtime;
pub mod sampling;
#[cfg(not(target_family = "wasm"))]
pub mod server;
pub mod simd;
pub mod tokenizer;

pub use runtime::{
    ChatMessage, ChatRole, GenerationOptions, GenerationResult, GenerationStats, LoadInfo, Runner,
};
