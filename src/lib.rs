//! RustyLLM is a lightweight, educational GGUF inference library.
//!
//! The crate exposes the same core runner used by the command-line binary and
//! optional HTTP server. It is intentionally small enough to read module by
//! module while still showing the complete path from GGUF metadata parsing and
//! tokenization to matrix-vector kernels, transformer decoding, sampling,
//! embeddings, and OpenAI-compatible serving.
//!
//! Start with [`runtime::Runner`] for the high-level API. For learning the
//! internals, read the modules in this order:
//!
//! 1. [`gguf`] for the model-file container format.
//! 2. [`tokenizer`] for metadata-driven text/token conversion.
//! 3. [`simd`] for scalar and optimized tensor kernels.
//! 4. [`model`] for weight loading and transformer forward passes.
//! 5. [`runtime`] for generation, chat rendering, embeddings, and sessions.

#![allow(
    clippy::collapsible_if,
    clippy::collapsible_match,
    clippy::explicit_counter_loop,
    clippy::manual_checked_ops,
    clippy::manual_is_multiple_of,
    clippy::unnecessary_unwrap,
    clippy::useless_conversion
)]

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
#[cfg(all(not(target_family = "wasm"), feature = "server"))]
pub mod session;
pub mod simd;
pub mod tokenizer;

pub use runtime::{
    ChatMessage, ChatRole, EmbeddingResult, GenerationOptions, GenerationResult, GenerationStats,
    LoadInfo, Runner, cosine_similarity,
};
