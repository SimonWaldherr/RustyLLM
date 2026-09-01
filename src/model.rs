// model.rs — LLaMA-architecture model with zero-copy mmap'd weights
//
// Key design: quantized weights stay as raw byte slices pointing into the mmap.
// The SIMD kernels do fused dequant+dot, avoiding intermediate f32 buffers.
// Only normalization weights and embeddings are stored as f32.
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

use crate::gguf::{GGMLType, GGUFFile};
use crate::simd;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, Default)]
pub struct Qwen35Profile {
    pub tokens: usize,
    pub recurrent: Duration,
    pub attention: Duration,
    pub ffn: Duration,
    pub output: Duration,
}

pub fn qwen35_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("RUSTY_LLM_QWEN_PROFILE").as_deref(),
            Ok("1") | Ok("true") | Ok("yes") | Ok("on")
        )
    })
}

fn qwen35_profile_store() -> &'static Mutex<Qwen35Profile> {
    static PROFILE: OnceLock<Mutex<Qwen35Profile>> = OnceLock::new();
    PROFILE.get_or_init(|| Mutex::new(Qwen35Profile::default()))
}

pub fn qwen35_profile_reset() {
    if qwen35_profile_enabled()
        && let Ok(mut profile) = qwen35_profile_store().lock()
    {
        *profile = Qwen35Profile::default();
    }
}

pub fn qwen35_profile_snapshot() -> Qwen35Profile {
    qwen35_profile_store()
        .lock()
        .map(|profile| *profile)
        .unwrap_or_default()
}

// ─── Config ──────────────────────────────────────────────────────────────────

// ─── Config ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Config {
    pub arch: String,
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub head_dim: usize,
    pub kv_dim: usize,
    pub kv_mul: usize,
    pub value_dim: usize,
    pub sliding_window: usize,
    pub expert_count: usize,
    pub expert_used_count: usize,
    pub rope_scaling_factor: f32,
    pub rope_original_context_length: usize,
}

impl Config {
    /// Builds the runtime model configuration from GGUF metadata.
    pub fn from_gguf(gguf: &GGUFFile) -> Self {
        let arch = gguf.get_str("general.architecture").unwrap_or("llama");
        let p = arch.to_string();

        let dim = gguf.get_u32(&format!("{}.embedding_length", p), 0) as usize;
        let n_heads = gguf.get_u32(&format!("{}.attention.head_count", p), 0).max(
            gguf.metadata
                .get(&format!("{}.attention.head_count", p))
                .and_then(crate::gguf::MetaValue::as_u32_array)
                .and_then(|heads| heads.into_iter().max())
                .unwrap_or(0),
        ) as usize;
        let n_kv_heads =
            gguf.get_u32(&format!("{}.attention.head_count_kv", p), n_heads as u32) as usize;
        let rope_dim = gguf.get_u32(&format!("{}.rope.dimension_count", p), 0) as usize;
        let default_head_dim = if dim > 0 && n_heads > 0 {
            dim / n_heads
        } else {
            rope_dim
        };
        let head_dim = gguf.get_u32(
            &format!("{}.attention.key_length", p),
            default_head_dim as u32,
        ) as usize;
        let value_dim =
            gguf.get_u32(&format!("{}.attention.value_length", p), head_dim as u32) as usize;
        let kv_dim = value_dim.saturating_mul(n_kv_heads);
        let kv_mul = if n_kv_heads > 0 {
            n_heads / n_kv_heads
        } else {
            0
        };

        let vocab_size = gguf.get_u32(&format!("{}.vocab_size", p), 0).max(
            gguf.metadata
                .get("tokenizer.ggml.tokens")
                .and_then(|v| v.as_string_array())
                .map(|v| v.len() as u32)
                .unwrap_or(0),
        ) as usize;
        let hidden_dim = match gguf.metadata.get(&format!("{}.feed_forward_length", p)) {
            Some(value) => value
                .as_u32()
                .or_else(|| {
                    if let crate::gguf::MetaValue::Array(values) = value {
                        values.iter().filter_map(|v| v.as_u32()).max()
                    } else {
                        None
                    }
                })
                .unwrap_or(0),
            None => 0,
        } as usize;

        Config {
            arch: p.clone(),
            dim,
            hidden_dim,
            n_layers: gguf.get_u32(&format!("{}.block_count", p), 0) as usize,
            n_heads,
            n_kv_heads,
            vocab_size,
            max_seq_len: gguf.get_u32(&format!("{}.context_length", p), 2048) as usize,
            rope_theta: gguf.get_f32(&format!("{}.rope.freq_base", p), 10000.0),
            rms_norm_eps: gguf.get_f32(&format!("{}.attention.layer_norm_rms_epsilon", p), 1e-5),
            head_dim,
            kv_dim,
            kv_mul,
            value_dim,
            sliding_window: gguf.get_u32(&format!("{}.attention.sliding_window", p), 0) as usize,
            expert_count: gguf.get_u32(&format!("{}.expert_count", p), 0) as usize,
            expert_used_count: gguf.get_u32(&format!("{}.expert_used_count", p), 0) as usize,
            rope_scaling_factor: gguf.get_f32(&format!("{}.rope.scaling.factor", p), 1.0),
            rope_original_context_length: gguf
                .get_u32(&format!("{}.rope.scaling.original_context_length", p), 0)
                as usize,
        }
    }
}

// ─── Weight storage: either f32 Vec or raw quantized bytes (zero-copy) ───────

// ─── Weight storage: either f32 Vec or raw quantized bytes (zero-copy) ───────

pub enum RawTensorData {
    Owned(Vec<u8>),
    View { ptr: *const u8, len: usize },
}

impl Clone for RawTensorData {
    /// Creates an independent handle to the same raw tensor storage.
    fn clone(&self) -> Self {
        match self {
            Self::Owned(data) => Self::Owned(data.clone()),
            Self::View { ptr, len } => Self::View {
                ptr: *ptr,
                len: *len,
            },
        }
    }
}

// SAFETY: Raw tensor data is immutable after model load. `View` points into an
// mmap kept alive by the owning `Runner`, so cross-thread reads are safe.
unsafe impl Send for RawTensorData {}
unsafe impl Sync for RawTensorData {}

impl RawTensorData {
    /// Copies tensor bytes into owned storage for in-memory model loading.
    fn owned(data: &[u8]) -> Self {
        Self::Owned(data.to_vec())
    }

    /// Borrows tensor bytes directly from the mapped GGUF file.
    fn view(data: &[u8]) -> Self {
        Self::View {
            ptr: data.as_ptr(),
            len: data.len(),
        }
    }

    /// Returns the tensor bytes regardless of whether they are owned or borrowed.
    pub(crate) fn as_slice(&self) -> &[u8] {
        match self {
            Self::Owned(data) => data,
            Self::View { ptr, len } => unsafe { std::slice::from_raw_parts(*ptr, *len) },
        }
    }
}

#[derive(Clone)]
pub enum Weight {
    F32(Vec<f32>),
    Quantized {
        data: RawTensorData,
        dtype: GGMLType,
        rows: usize,
        cols: usize,
    },
}

impl Weight {
    /// Matrix-vector multiply: `self[rows x cols] * x[cols] -> out[rows]`.
    pub fn matvec(&self, x: &[f32]) -> Vec<f32> {
        match self {
            Weight::F32(data) => {
                let cols = x.len();
                let rows = data.len() / cols;
                simd::matvec_f32(data, x, rows, cols)
            }
            Weight::Quantized {
                data,
                dtype,
                rows,
                cols,
            } => {
                let data = data.as_slice();
                match dtype {
                    GGMLType::Q8_0 => simd::matvec_q8_0(data, x, *rows, *cols),
                    GGMLType::Q8_1 => simd::matvec_q8_1(data, x, *rows, *cols),
                    GGMLType::Q4_0 => simd::matvec_q4_0(data, x, *rows, *cols),
                    GGMLType::Q4_1 => simd::matvec_q4_1(data, x, *rows, *cols),
                    GGMLType::Q5_0 => simd::matvec_q5_0(data, x, *rows, *cols),
                    GGMLType::Q5_1 => simd::matvec_q5_1(data, x, *rows, *cols),
                    GGMLType::Q4_K => simd::matvec_q4_k(data, x, *rows, *cols),
                    GGMLType::Q5_K => simd::matvec_q5_k(data, x, *rows, *cols),
                    GGMLType::Q6_K => simd::matvec_q6_k(data, x, *rows, *cols),
                    GGMLType::MXFP4 => simd::matvec_mxfp4(data, x, *rows, *cols),
                    _ => panic!("Unsupported quantized matvec: {:?}", dtype),
                }
            }
        }
    }

    /// Matrix-vector multiply, writing into a pre-allocated output buffer.
    pub fn matvec_into(&self, x: &[f32], out: &mut Vec<f32>) {
        match self {
            Weight::F32(data) => {
                let cols = x.len();
                let rows = data.len() / cols;
                out.resize(rows, 0.0);
                simd::matvec_f32_into(data, x, rows, cols, out);
            }
            Weight::Quantized {
                data,
                dtype,
                rows,
                cols,
            } => {
                let data = data.as_slice();
                out.resize(*rows, 0.0);
                match dtype {
                    GGMLType::Q8_0 => {
                        if !crate::metal::q8_0_matvec_into(data, x, *rows, *cols, out) {
                            simd::matvec_q8_0_into(data, x, *rows, *cols, out);
                        }
                    }
                    GGMLType::Q8_1 => simd::matvec_q8_1_into(data, x, *rows, *cols, out),
                    GGMLType::Q4_0 => {
                        if !crate::metal::q4_0_matvec_into(data, x, *rows, *cols, out) {
                            simd::matvec_q4_0_into(data, x, *rows, *cols, out);
                        }
                    }
                    GGMLType::Q4_1 => simd::matvec_q4_1_into(data, x, *rows, *cols, out),
                    GGMLType::Q5_0 => simd::matvec_q5_0_into(data, x, *rows, *cols, out),
                    GGMLType::Q5_1 => simd::matvec_q5_1_into(data, x, *rows, *cols, out),
                    GGMLType::Q4_K => simd::matvec_q4_k_into(data, x, *rows, *cols, out),
                    GGMLType::Q5_K => simd::matvec_q5_k_into(data, x, *rows, *cols, out),
                    GGMLType::Q6_K => simd::matvec_q6_k_into(data, x, *rows, *cols, out),
                    GGMLType::MXFP4 => simd::matvec_mxfp4_into(data, x, *rows, *cols, out),
                    _ => panic!("Unsupported quantized matvec: {:?}", dtype),
                }
            }
        }
    }

    /// Extract one row as f32 values.
    pub fn row(&self, row: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0; cols];
        self.row_into(row, cols, &mut out);
        out
    }

    /// Extract one row as f32 values into caller-owned storage.
    pub fn row_into(&self, row: usize, cols: usize, out: &mut Vec<f32>) {
        out.resize(cols, 0.0);
        match self {
            Weight::F32(data) => {
                let start = row * cols;
                out.copy_from_slice(&data[start..start + cols]);
            }
            Weight::Quantized {
                data,
                dtype,
                rows,
                cols: qcols,
            } => {
                let data = data.as_slice();
                assert_eq!(*qcols, cols, "row(): column mismatch");
                assert!(row < *rows, "row(): row out of bounds");
                let row_bytes = quantized_row_bytes(*dtype, cols)
                    .unwrap_or_else(|| panic!("Unsupported quantized row extraction: {:?}", dtype));
                let start = row * row_bytes;
                dequantize_row_into(*dtype, &data[start..start + row_bytes], out);
            }
        }
    }

    /// Returns a borrowed row from an unquantized float weight.
    pub fn row_f32(&self, row: usize, cols: usize) -> &[f32] {
        match self {
            Weight::F32(data) => {
                let start = row * cols;
                &data[start..start + cols]
            }
            _ => panic!("Expected f32 row storage"),
        }
    }
}

/// Reports whether the forward pass may fuse several projections that share
/// one activation into a single dispatch (default on; set
/// `RUSTY_LLM_FUSED_PROJ=0` to force one dispatch per matrix, e.g. for A/B
/// measurement). Fusion trades a barrier for a wider row range per dispatch,
/// and which side wins is machine dependent, so this stays measurable at
/// runtime rather than being baked in.
#[cfg(not(target_family = "wasm"))]
fn fused_projections_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("RUSTY_LLM_FUSED_PROJ").as_deref(),
            Ok("0") | Ok("false") | Ok("off")
        )
    })
}

#[cfg(not(target_family = "wasm"))]
/// Attempts fused K-quant triple-projection fast paths and reports whether one ran.
fn try_quant_matvec3_into(
    wq: &Weight,
    wk: &Weight,
    wv: &Weight,
    x: &[f32],
    q: &mut Vec<f32>,
    k: &mut Vec<f32>,
    v: &mut Vec<f32>,
) -> bool {
    if !fused_projections_enabled() {
        return false;
    }
    match (wq, wk, wv) {
        (
            Weight::Quantized {
                data: q_data,
                dtype: GGMLType::Q4_0,
                rows: q_rows,
                cols: q_cols,
            },
            Weight::Quantized {
                data: k_data,
                dtype: GGMLType::Q4_0,
                rows: k_rows,
                cols: k_cols,
            },
            Weight::Quantized {
                data: v_data,
                dtype: GGMLType::Q4_0,
                rows: v_rows,
                cols: v_cols,
            },
        ) if *q_cols == *k_cols && *q_cols == *v_cols && *q_cols == x.len() => {
            if crate::metal::q4_0_matvec3_into(
                (q_data.as_slice(), *q_rows, *q_cols),
                (k_data.as_slice(), *k_rows, *k_cols),
                (v_data.as_slice(), *v_rows, *v_cols),
                x,
                q,
                k,
                v,
            ) {
                true
            } else {
                crate::simd::matvec_quant3_into(
                    (
                        crate::simd::QuantMatvecKind::Q4_0,
                        q_data.as_slice(),
                        *q_rows,
                        *q_cols,
                    ),
                    (
                        crate::simd::QuantMatvecKind::Q4_0,
                        k_data.as_slice(),
                        *k_rows,
                        *k_cols,
                    ),
                    (
                        crate::simd::QuantMatvecKind::Q4_0,
                        v_data.as_slice(),
                        *v_rows,
                        *v_cols,
                    ),
                    x,
                    q,
                    k,
                    v,
                )
            }
        }
        (
            Weight::Quantized {
                data: q_data,
                dtype: GGMLType::Q4_K,
                rows: q_rows,
                cols: q_cols,
            },
            Weight::Quantized {
                data: k_data,
                dtype: GGMLType::Q4_K,
                rows: k_rows,
                cols: k_cols,
            },
            Weight::Quantized {
                data: v_data,
                dtype: GGMLType::Q4_K,
                rows: v_rows,
                cols: v_cols,
            },
        ) if *q_cols == *k_cols && *q_cols == *v_cols && *q_cols == x.len() => {
            crate::simd::matvec_q4_k3_into(
                (q_data.as_slice(), *q_rows, *q_cols),
                (k_data.as_slice(), *k_rows, *k_cols),
                (v_data.as_slice(), *v_rows, *v_cols),
                x,
                q,
                k,
                v,
            )
        }
        (
            Weight::Quantized {
                data: q_data,
                dtype: GGMLType::Q5_K,
                rows: q_rows,
                cols: q_cols,
            },
            Weight::Quantized {
                data: k_data,
                dtype: GGMLType::Q5_K,
                rows: k_rows,
                cols: k_cols,
            },
            Weight::Quantized {
                data: v_data,
                dtype: GGMLType::Q5_K,
                rows: v_rows,
                cols: v_cols,
            },
        ) if *q_cols == *k_cols && *q_cols == *v_cols && *q_cols == x.len() => {
            crate::simd::matvec_q5_k3_into(
                (q_data.as_slice(), *q_rows, *q_cols),
                (k_data.as_slice(), *k_rows, *k_cols),
                (v_data.as_slice(), *v_rows, *v_cols),
                x,
                q,
                k,
                v,
            )
        }
        (
            Weight::Quantized {
                data: q_data,
                dtype: GGMLType::Q6_K,
                rows: q_rows,
                cols: q_cols,
            },
            Weight::Quantized {
                data: k_data,
                dtype: GGMLType::Q6_K,
                rows: k_rows,
                cols: k_cols,
            },
            Weight::Quantized {
                data: v_data,
                dtype: GGMLType::Q6_K,
                rows: v_rows,
                cols: v_cols,
            },
        ) if *q_cols == *k_cols && *q_cols == *v_cols && *q_cols == x.len() => {
            crate::simd::matvec_q6_k3_into(
                (q_data.as_slice(), *q_rows, *q_cols),
                (k_data.as_slice(), *k_rows, *k_cols),
                (v_data.as_slice(), *v_rows, *v_cols),
                x,
                q,
                k,
                v,
            )
        }
        (
            Weight::Quantized {
                data: q_data,
                dtype: q_dtype,
                rows: q_rows,
                cols: q_cols,
            },
            Weight::Quantized {
                data: k_data,
                dtype: k_dtype,
                rows: k_rows,
                cols: k_cols,
            },
            Weight::Quantized {
                data: v_data,
                dtype: v_dtype,
                rows: v_rows,
                cols: v_cols,
            },
        ) if *q_cols == *k_cols && *q_cols == *v_cols && *q_cols == x.len() => {
            let Some(q_kind) = quant_matvec_kind(*q_dtype) else {
                return false;
            };
            let Some(k_kind) = quant_matvec_kind(*k_dtype) else {
                return false;
            };
            let Some(v_kind) = quant_matvec_kind(*v_dtype) else {
                return false;
            };
            if crate::metal::dispatch_enabled()
                && (quant_kind_prefers_single_metal(q_kind)
                    || quant_kind_prefers_single_metal(k_kind)
                    || quant_kind_prefers_single_metal(v_kind))
            {
                return false;
            }
            if let (Some(q_kind), Some(k_kind), Some(v_kind)) = (
                kquant_matvec_kind(*q_dtype),
                kquant_matvec_kind(*k_dtype),
                kquant_matvec_kind(*v_dtype),
            ) {
                return crate::simd::matvec_kquant3_into(
                    (q_kind, q_data.as_slice(), *q_rows, *q_cols),
                    (k_kind, k_data.as_slice(), *k_rows, *k_cols),
                    (v_kind, v_data.as_slice(), *v_rows, *v_cols),
                    x,
                    q,
                    k,
                    v,
                );
            }
            crate::simd::matvec_quant3_into(
                (q_kind, q_data.as_slice(), *q_rows, *q_cols),
                (k_kind, k_data.as_slice(), *k_rows, *k_cols),
                (v_kind, v_data.as_slice(), *v_rows, *v_cols),
                x,
                q,
                k,
                v,
            )
        }
        _ => false,
    }
}

#[cfg(target_family = "wasm")]
/// Attempts fused K-quant triple-projection fast paths and reports whether one ran.
fn try_quant_matvec3_into(
    _wq: &Weight,
    _wk: &Weight,
    _wv: &Weight,
    _x: &[f32],
    _q: &mut Vec<f32>,
    _k: &mut Vec<f32>,
    _v: &mut Vec<f32>,
) -> bool {
    false
}

#[cfg(not(target_family = "wasm"))]
/// Attempts fused K-quant double-projection fast paths and reports whether one ran.
fn try_quant_matvec2_into(
    a: &Weight,
    b: &Weight,
    x: &[f32],
    out_a: &mut Vec<f32>,
    out_b: &mut Vec<f32>,
) -> bool {
    if !fused_projections_enabled() {
        return false;
    }
    match (a, b) {
        (
            Weight::Quantized {
                data: a_data,
                dtype: GGMLType::Q4_0,
                rows: a_rows,
                cols: a_cols,
            },
            Weight::Quantized {
                data: b_data,
                dtype: GGMLType::Q4_0,
                rows: b_rows,
                cols: b_cols,
            },
        ) if *a_cols == *b_cols && *a_cols == x.len() => {
            if crate::metal::q4_0_matvec2_into(
                (a_data.as_slice(), *a_rows, *a_cols),
                (b_data.as_slice(), *b_rows, *b_cols),
                x,
                out_a,
                out_b,
            ) {
                true
            } else {
                crate::simd::matvec_quant2_into(
                    (
                        crate::simd::QuantMatvecKind::Q4_0,
                        a_data.as_slice(),
                        *a_rows,
                        *a_cols,
                    ),
                    (
                        crate::simd::QuantMatvecKind::Q4_0,
                        b_data.as_slice(),
                        *b_rows,
                        *b_cols,
                    ),
                    x,
                    out_a,
                    out_b,
                )
            }
        }
        (
            Weight::Quantized {
                data: a_data,
                dtype: GGMLType::Q4_K,
                rows: a_rows,
                cols: a_cols,
            },
            Weight::Quantized {
                data: b_data,
                dtype: GGMLType::Q4_K,
                rows: b_rows,
                cols: b_cols,
            },
        ) if *a_cols == *b_cols && *a_cols == x.len() => crate::simd::matvec_q4_k2_into(
            (a_data.as_slice(), *a_rows, *a_cols),
            (b_data.as_slice(), *b_rows, *b_cols),
            x,
            out_a,
            out_b,
        ),
        (
            Weight::Quantized {
                data: a_data,
                dtype: GGMLType::Q5_K,
                rows: a_rows,
                cols: a_cols,
            },
            Weight::Quantized {
                data: b_data,
                dtype: GGMLType::Q5_K,
                rows: b_rows,
                cols: b_cols,
            },
        ) if *a_cols == *b_cols && *a_cols == x.len() => crate::simd::matvec_q5_k2_into(
            (a_data.as_slice(), *a_rows, *a_cols),
            (b_data.as_slice(), *b_rows, *b_cols),
            x,
            out_a,
            out_b,
        ),
        (
            Weight::Quantized {
                data: a_data,
                dtype: GGMLType::Q6_K,
                rows: a_rows,
                cols: a_cols,
            },
            Weight::Quantized {
                data: b_data,
                dtype: GGMLType::Q6_K,
                rows: b_rows,
                cols: b_cols,
            },
        ) if *a_cols == *b_cols && *a_cols == x.len() => crate::simd::matvec_q6_k2_into(
            (a_data.as_slice(), *a_rows, *a_cols),
            (b_data.as_slice(), *b_rows, *b_cols),
            x,
            out_a,
            out_b,
        ),
        (
            Weight::Quantized {
                data: a_data,
                dtype: a_dtype,
                rows: a_rows,
                cols: a_cols,
            },
            Weight::Quantized {
                data: b_data,
                dtype: b_dtype,
                rows: b_rows,
                cols: b_cols,
            },
        ) if *a_cols == *b_cols && *a_cols == x.len() => {
            let Some(a_kind) = quant_matvec_kind(*a_dtype) else {
                return false;
            };
            let Some(b_kind) = quant_matvec_kind(*b_dtype) else {
                return false;
            };
            if crate::metal::dispatch_enabled()
                && (quant_kind_prefers_single_metal(a_kind)
                    || quant_kind_prefers_single_metal(b_kind))
            {
                return false;
            }
            crate::simd::matvec_quant2_into(
                (a_kind, a_data.as_slice(), *a_rows, *a_cols),
                (b_kind, b_data.as_slice(), *b_rows, *b_cols),
                x,
                out_a,
                out_b,
            )
        }
        _ => false,
    }
}

#[cfg(not(target_family = "wasm"))]
fn quant_matvec_kind(dtype: GGMLType) -> Option<crate::simd::QuantMatvecKind> {
    match dtype {
        GGMLType::Q8_0 => Some(crate::simd::QuantMatvecKind::Q8_0),
        GGMLType::Q8_1 => Some(crate::simd::QuantMatvecKind::Q8_1),
        GGMLType::Q4_0 => Some(crate::simd::QuantMatvecKind::Q4_0),
        GGMLType::Q4_1 => Some(crate::simd::QuantMatvecKind::Q4_1),
        GGMLType::Q5_0 => Some(crate::simd::QuantMatvecKind::Q5_0),
        GGMLType::Q5_1 => Some(crate::simd::QuantMatvecKind::Q5_1),
        GGMLType::Q4_K => Some(crate::simd::QuantMatvecKind::Q4K),
        GGMLType::Q5_K => Some(crate::simd::QuantMatvecKind::Q5K),
        GGMLType::Q6_K => Some(crate::simd::QuantMatvecKind::Q6K),
        GGMLType::MXFP4 => Some(crate::simd::QuantMatvecKind::Mxfp4),
        _ => None,
    }
}

/// Maps the K-quants that can share one activation quantization in a fused
/// projection. Q/K/V commonly mix Q4_K and Q6_K in embedding GGUFs.
#[cfg(not(target_family = "wasm"))]
fn kquant_matvec_kind(dtype: GGMLType) -> Option<crate::simd::KQuantMatvecKind> {
    match dtype {
        GGMLType::Q4_K => Some(crate::simd::KQuantMatvecKind::Q4K),
        GGMLType::Q5_K => Some(crate::simd::KQuantMatvecKind::Q5K),
        GGMLType::Q6_K => Some(crate::simd::KQuantMatvecKind::Q6K),
        _ => None,
    }
}

/// Returns the raw layout for K-quant weights that can share one Q8_K
/// activation quantization. The encoder batch path keeps these bytes borrowed
/// from the GGUF mapping, just like the single-token matvec path.
#[cfg(not(target_family = "wasm"))]
fn kquant_weight_parts(
    weight: &Weight,
) -> Option<(crate::simd::KQuantMatvecKind, &[u8], usize, usize)> {
    let Weight::Quantized {
        data,
        dtype,
        rows,
        cols,
    } = weight
    else {
        return None;
    };
    if *cols == 0 || *cols % 256 != 0 {
        return None;
    }
    let kind = kquant_matvec_kind(*dtype)?;
    let bytes = quantized_row_bytes(*dtype, *cols)?.checked_mul(*rows)?;
    let data = data.as_slice();
    if data.len() < bytes {
        return None;
    }
    Some((kind, data, *rows, *cols))
}

/// Returns the borrowed row layout for every quantized matrix format handled
/// by the CPU batch worker. This covers models whose dimensions force a
/// 32-value block format instead of a K-quant layout.
#[cfg(not(target_family = "wasm"))]
fn quant_weight_parts(
    weight: &Weight,
) -> Option<(crate::simd::QuantMatvecKind, &[u8], usize, usize)> {
    let Weight::Quantized {
        data,
        dtype,
        rows,
        cols,
    } = weight
    else {
        return None;
    };
    let kind = quant_matvec_kind(*dtype)?;
    let bytes = quantized_row_bytes(*dtype, *cols)?.checked_mul(*rows)?;
    let data = data.as_slice();
    if data.len() < bytes {
        return None;
    }
    Some((kind, data, *rows, *cols))
}

#[cfg(not(target_family = "wasm"))]
fn try_kquant_matvec_batch_into(weight: &Weight, inputs: &[f32], out: &mut Vec<f32>) -> bool {
    let Some(parts) = quant_weight_parts(weight) else {
        return false;
    };
    crate::simd::matvec_quant_batch_into(parts, inputs, out)
}

#[cfg(not(target_family = "wasm"))]
fn try_kquant_matvec2_batch_into(
    a: &Weight,
    b: &Weight,
    inputs: &[f32],
    out_a: &mut Vec<f32>,
    out_b: &mut Vec<f32>,
) -> bool {
    let (Some(a), Some(b)) = (quant_weight_parts(a), quant_weight_parts(b)) else {
        return false;
    };
    crate::simd::matvec_quant2_batch_into(a, b, inputs, out_a, out_b)
}

#[cfg(not(target_family = "wasm"))]
fn try_kquant_matvec3_batch_into(
    a: &Weight,
    b: &Weight,
    c: &Weight,
    inputs: &[f32],
    out_a: &mut Vec<f32>,
    out_b: &mut Vec<f32>,
    out_c: &mut Vec<f32>,
) -> bool {
    let (Some(a), Some(b), Some(c)) = (
        quant_weight_parts(a),
        quant_weight_parts(b),
        quant_weight_parts(c),
    ) else {
        return false;
    };
    crate::simd::matvec_quant3_batch_into(a, b, c, inputs, out_a, out_b, out_c)
}

#[cfg(not(target_family = "wasm"))]
fn quant_kind_prefers_single_metal(kind: crate::simd::QuantMatvecKind) -> bool {
    matches!(
        kind,
        crate::simd::QuantMatvecKind::Q4_0 | crate::simd::QuantMatvecKind::Q8_0
    )
}

#[cfg(target_family = "wasm")]
/// Attempts fused K-quant double-projection fast paths and reports whether one ran.
fn try_quant_matvec2_into(
    _a: &Weight,
    _b: &Weight,
    _x: &[f32],
    _out_a: &mut Vec<f32>,
    _out_b: &mut Vec<f32>,
) -> bool {
    false
}

#[cfg(not(target_family = "wasm"))]
/// Attempts a Gemma-style Q4_0 GELU feed-forward block as one Metal command buffer.
fn try_metal_gemma4_ffn_into(
    gate: &Weight,
    up: &Weight,
    down: &Weight,
    x: &[f32],
    out: &mut Vec<f32>,
) -> bool {
    let (
        Weight::Quantized {
            data: gate_data,
            dtype: GGMLType::Q4_0,
            rows: gate_rows,
            cols: gate_cols,
        },
        Weight::Quantized {
            data: up_data,
            dtype: GGMLType::Q4_0,
            rows: up_rows,
            cols: up_cols,
        },
        Weight::Quantized {
            data: down_data,
            dtype: GGMLType::Q4_0,
            rows: down_rows,
            cols: down_cols,
        },
    ) = (gate, up, down)
    else {
        return false;
    };
    crate::metal::q4_0_gelu_ffn_into(
        (gate_data.as_slice(), *gate_rows, *gate_cols),
        (up_data.as_slice(), *up_rows, *up_cols),
        (down_data.as_slice(), *down_rows, *down_cols),
        x,
        out,
    )
}

#[cfg(target_family = "wasm")]
fn try_metal_gemma4_ffn_into(
    _gate: &Weight,
    _up: &Weight,
    _down: &Weight,
    _x: &[f32],
    _out: &mut Vec<f32>,
) -> bool {
    false
}

#[cfg(not(target_family = "wasm"))]
/// Attempts to run a Mistral-style Q4_K/Q4_K/Q6_K FFN block as one Metal command buffer.
fn try_metal_mistral_ffn_into(
    gate: &Weight,
    up: &Weight,
    down: &Weight,
    x: &[f32],
    out: &mut Vec<f32>,
) -> bool {
    let (
        Weight::Quantized {
            data: gate_data,
            dtype: GGMLType::Q4_K,
            rows: gate_rows,
            cols: gate_cols,
        },
        Weight::Quantized {
            data: up_data,
            dtype: GGMLType::Q4_K,
            rows: up_rows,
            cols: up_cols,
        },
        Weight::Quantized {
            data: down_data,
            dtype: GGMLType::Q6_K,
            rows: down_rows,
            cols: down_cols,
        },
    ) = (gate, up, down)
    else {
        return false;
    };
    if *gate_cols != *up_cols
        || *gate_cols != x.len()
        || *gate_rows != *up_rows
        || *gate_rows != *down_cols
    {
        return false;
    }
    crate::metal::q4k_q4k_q6k_ffn_into(
        (gate_data.as_slice(), *gate_rows, *gate_cols),
        (up_data.as_slice(), *up_rows, *up_cols),
        (down_data.as_slice(), *down_rows, *down_cols),
        x,
        out,
    )
}

#[cfg(target_family = "wasm")]
fn try_metal_mistral_ffn_into(
    _gate: &Weight,
    _up: &Weight,
    _down: &Weight,
    _x: &[f32],
    _out: &mut Vec<f32>,
) -> bool {
    false
}

#[cfg(not(target_family = "wasm"))]
/// Attempts to run Mistral post-attention output projection, residual norm, and FFN in one Metal command buffer.
fn try_metal_mistral_post_attention_ffn_into(
    wo: &Weight,
    gate: &Weight,
    up: &Weight,
    down: &Weight,
    x: &mut [f32],
    attn_out: &[f32],
    ffn_norm: &[f32],
    rms_eps: f32,
) -> bool {
    if !crate::metal::post_attention_ffn_enabled() {
        return false;
    }
    let (
        Weight::Quantized {
            data: wo_data,
            dtype: GGMLType::Q4_K,
            rows: wo_rows,
            cols: wo_cols,
        },
        Weight::Quantized {
            data: gate_data,
            dtype: GGMLType::Q4_K,
            rows: gate_rows,
            cols: gate_cols,
        },
        Weight::Quantized {
            data: up_data,
            dtype: GGMLType::Q4_K,
            rows: up_rows,
            cols: up_cols,
        },
        Weight::Quantized {
            data: down_data,
            dtype: GGMLType::Q6_K,
            rows: down_rows,
            cols: down_cols,
        },
    ) = (wo, gate, up, down)
    else {
        return false;
    };
    if *wo_rows != x.len()
        || *wo_cols != attn_out.len()
        || *gate_cols != x.len()
        || *up_cols != x.len()
        || *gate_rows != *up_rows
        || *gate_rows != *down_cols
        || *down_rows != x.len()
        || ffn_norm.len() != x.len()
    {
        return false;
    }
    crate::metal::mistral_post_attention_ffn_into(
        (wo_data.as_slice(), *wo_rows, *wo_cols),
        (gate_data.as_slice(), *gate_rows, *gate_cols),
        (up_data.as_slice(), *up_rows, *up_cols),
        (down_data.as_slice(), *down_rows, *down_cols),
        x,
        attn_out,
        ffn_norm,
        rms_eps,
    )
}

#[cfg(target_family = "wasm")]
fn try_metal_mistral_post_attention_ffn_into(
    _wo: &Weight,
    _gate: &Weight,
    _up: &Weight,
    _down: &Weight,
    _x: &mut Vec<f32>,
    _attn_out: &[f32],
    _ffn_norm: &[f32],
    _rms_eps: f32,
) -> bool {
    false
}

// ─── Layer + Model weights ───────────────────────────────────────────────────

pub struct LayerWeights {
    pub attn_norm: Vec<f32>,
    pub wq: Weight,
    pub bq: Vec<f32>,
    pub wk: Weight,
    pub bk: Vec<f32>,
    pub wv: Weight,
    pub bv: Vec<f32>,
    /// Optional per-head RMSNorm weights applied to queries and keys before
    /// RoPE. Qwen3 GGUFs expose these as `attn_q_norm` / `attn_k_norm`.
    /// Keeping them optional preserves the regular LLaMA/Qwen2 path.
    pub attn_q_norm: Vec<f32>,
    pub attn_k_norm: Vec<f32>,
    pub wo: Weight,
    pub ffn_norm: Vec<f32>,
    pub w1: Weight, // gate
    pub w2: Weight, // down
    pub w3: Weight, // up
    /// Routed sparse experts, when the GGUF carries a Mixtral-style MoE block
    /// instead of a dense feed-forward network. `w1`/`w2`/`w3` are unused
    /// placeholders in that case; keeping this an `Option` rather than swapping
    /// the three fields for an enum leaves every dense call site untouched.
    pub moe: Option<Box<RoutedMoeWeights>>,
}

/// Routed feed-forward experts, as used by Mixtral and other Mistral MoE GGUFs.
///
/// Mistral's MoE blocks are SwiGLU experts selected by a softmax router, which
/// is the same shape as the dense path repeated per expert — unlike the Laguna
/// layout, there is no shared expert and no router bias.
pub struct RoutedMoeWeights {
    /// Router projection: `expert_count` rows of `dim` columns.
    pub router: Weight,
    pub gate_experts: ExpertWeight,
    pub up_experts: ExpertWeight,
    pub down_experts: ExpertWeight,
}

pub struct ModelWeights {
    pub token_embd: Weight,
    pub output_norm: Vec<f32>,
    pub output: Weight,
    pub layers: Vec<LayerWeights>,
}

pub struct ExpertWeight {
    pub data: RawTensorData,
    pub dtype: GGMLType,
    pub experts: usize,
    pub rows: usize,
    pub cols: usize,
}

impl ExpertWeight {
    /// Runs one expert matrix from a mixture-of-experts tensor and returns its output.
    pub fn matvec_expert(&self, expert: usize, x: &[f32]) -> Vec<f32> {
        assert!(expert < self.experts, "expert index out of bounds");
        let mut out = Vec::new();
        self.matvec_expert_into(expert, x, &mut out);
        out
    }

    /// Runs one expert matrix from a mixture-of-experts tensor into a reusable buffer.
    pub fn matvec_expert_into(&self, expert: usize, x: &[f32], out: &mut Vec<f32>) {
        assert!(expert < self.experts, "expert index out of bounds");
        let data = self.data.as_slice();
        let row_bytes = quantized_row_bytes(self.dtype, self.cols)
            .unwrap_or_else(|| panic!("Unsupported expert weight dtype: {:?}", self.dtype));
        let expert_bytes = self.rows * row_bytes;
        let start = expert * expert_bytes;
        let weights = &data[start..start + expert_bytes];
        match self.dtype {
            GGMLType::Q4_K => simd::matvec_q4_k_into(weights, x, self.rows, self.cols, out),
            GGMLType::Q5_K => simd::matvec_q5_k_into(weights, x, self.rows, self.cols, out),
            GGMLType::Q6_K => simd::matvec_q6_k_into(weights, x, self.rows, self.cols, out),
            GGMLType::MXFP4 => simd::matvec_mxfp4_into(weights, x, self.rows, self.cols, out),
            _ => panic!("Unsupported expert weight dtype: {:?}", self.dtype),
        }
    }

    /// Attempts to run two expert matrices against the same activation in one
    /// worker-pool job. GPT-OSS evaluates gate and up projections together for
    /// every selected expert, so combining them avoids a second rendezvous with
    /// the matrix-vector workers without changing the per-row calculation.
    /// Returns `false` when the two tensors cannot safely share this path.
    pub fn try_matvec_expert_pair_into(
        &self,
        other: &Self,
        expert: usize,
        x: &[f32],
        out_self: &mut Vec<f32>,
        out_other: &mut Vec<f32>,
    ) -> bool {
        if self.dtype != GGMLType::MXFP4
            || other.dtype != GGMLType::MXFP4
            || expert >= self.experts
            || expert >= other.experts
            || self.experts != other.experts
            || self.rows != other.rows
            || self.cols != other.cols
            || self.cols == 0
            || self.cols % 32 != 0
            || self.cols != x.len()
        {
            return false;
        }

        let row_bytes = (self.cols / 32) * 17;
        let Some(expert_bytes) = self.rows.checked_mul(row_bytes) else {
            return false;
        };
        let Some(start) = expert.checked_mul(expert_bytes) else {
            return false;
        };
        let Some(end) = start.checked_add(expert_bytes) else {
            return false;
        };
        let self_data = self.data.as_slice();
        let other_data = other.data.as_slice();
        if end > self_data.len() || end > other_data.len() {
            return false;
        }

        simd::matvec_quant2_into(
            (
                simd::QuantMatvecKind::Mxfp4,
                &self_data[start..end],
                self.rows,
                self.cols,
            ),
            (
                simd::QuantMatvecKind::Mxfp4,
                &other_data[start..end],
                other.rows,
                other.cols,
            ),
            x,
            out_self,
            out_other,
        )
    }
}

pub struct GptOssLayerWeights {
    pub attn_norm: Vec<f32>,
    pub wq: Weight,
    pub bq: Vec<f32>,
    pub wk: Weight,
    pub bk: Vec<f32>,
    pub wv: Weight,
    pub bv: Vec<f32>,
    pub wo: Weight,
    pub bo: Vec<f32>,
    pub sinks: Vec<f32>,
    pub post_attn_norm: Vec<f32>,
    pub gate_inp: Weight,
    pub gate_inp_bias: Vec<f32>,
    pub gate_exps: ExpertWeight,
    pub gate_exps_bias: Weight,
    pub up_exps: ExpertWeight,
    pub up_exps_bias: Weight,
    pub down_exps: ExpertWeight,
    pub down_exps_bias: Weight,
}

pub struct GptOssWeights {
    pub token_embd: Weight,
    pub output_norm: Vec<f32>,
    pub output: Weight,
    pub layers: Vec<GptOssLayerWeights>,
}

/// Dense and sparse feed-forward layouts used by Poolside's Laguna models.
pub struct LagunaSparseMlpWeights {
    pub router: Weight,
    pub router_bias: Vec<f32>,
    pub gate_experts: ExpertWeight,
    pub up_experts: ExpertWeight,
    pub down_experts: ExpertWeight,
    pub shared_gate: Weight,
    pub shared_up: Weight,
    pub shared_down: Weight,
}

pub enum LagunaMlpWeights {
    Dense {
        gate: Weight,
        up: Weight,
        down: Weight,
    },
    Sparse(Box<LagunaSparseMlpWeights>),
}

/// One Laguna decoder block. Attention dimensions vary by layer (48 or 64
/// heads in Laguna-XS), while keys and values keep eight heads throughout.
pub struct LagunaLayerWeights {
    pub attn_norm: Vec<f32>,
    pub wq: Weight,
    pub wk: Weight,
    pub wv: Weight,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub attn_gate: Weight,
    pub wo: Weight,
    pub ffn_norm: Vec<f32>,
    pub mlp: LagunaMlpWeights,
    pub n_heads: usize,
    pub rotary_dim: usize,
    pub rope_inv_freq: Vec<f32>,
    pub sliding_window: bool,
}

/// Fixed dimensions of a Mamba-2 state-space mixer, derived from the
/// `{arch}.ssm.*` GGUF metadata.
#[derive(Clone, Copy, Debug)]
pub struct SsmDims {
    /// Depthwise convolution width (`ssm.conv_kernel`).
    pub d_conv: usize,
    /// Total SSM channel count (`ssm.inner_size`).
    pub d_inner: usize,
    /// Recurrent state width per channel (`ssm.state_size`).
    pub d_state: usize,
    /// Number of SSM heads. Mamba-2 reuses the Mamba-1 `ssm.time_step_rank`
    /// key for this, which is why the name does not match the meaning.
    pub n_head: usize,
    /// Number of B/C groups shared across heads (`ssm.group_count`).
    pub n_group: usize,
}

impl SsmDims {
    /// Channels carried through the depthwise convolution: the SSM input plus
    /// the B and C projections, which Mamba-2 convolves together.
    pub fn conv_dim(&self) -> usize {
        self.d_inner + 2 * self.n_group * self.d_state
    }

    /// Width of the fused input projection: gate, convolved channels, and one
    /// timestep scalar per head.
    pub fn d_in_proj(&self) -> usize {
        self.d_inner + self.conv_dim() + self.n_head
    }

    /// Channels handled by one SSM head.
    pub fn head_dim(&self) -> usize {
        self.d_inner / self.n_head
    }
}

/// One Mamba-2 mixer block.
pub struct Mamba2LayerWeights {
    /// Fused gate/x/B/C/dt projection: `d_in_proj` rows of `dim` columns.
    pub in_proj: Weight,
    /// Depthwise filters, channel-major: `conv_dim * d_conv`.
    pub conv_w: Vec<f32>,
    /// Optional convolution bias of `conv_dim` entries.
    pub conv_b: Vec<f32>,
    /// Per-head timestep bias.
    pub dt_bias: Vec<f32>,
    /// Per-head state decay. The GGUF already stores `-exp(A_log)`, so these
    /// are negative and must not be negated again.
    pub a: Vec<f32>,
    /// Per-head skip-connection scale, broadcast across the head's channels.
    pub d: Vec<f32>,
    /// Grouped RMSNorm weights over `d_inner`, laid out as `n_group` rows.
    pub norm: Vec<f32>,
    /// Output projection back to the residual stream.
    pub out_proj: Weight,
}

/// One position-free (NoPE) attention block of a Nemotron-H model.
pub struct NemotronAttnWeights {
    pub wq: Weight,
    pub wk: Weight,
    pub wv: Weight,
    pub wo: Weight,
    pub bo: Vec<f32>,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    /// Index of this layer's slice in the compacted KV cache, which holds one
    /// entry per attention layer rather than one per block.
    pub kv_slot: usize,
}

/// Routed experts of a Nemotron-H MoE block. Unlike Mixtral these have no gate
/// projection and use a squared-ReLU activation.
pub struct NemotronMoeWeights {
    pub router: Weight,
    pub router_bias: Vec<f32>,
    pub up_experts: ExpertWeight,
    pub down_experts: ExpertWeight,
    pub shared_up: Weight,
    pub shared_down: Weight,
}

/// A dense squared-ReLU feed-forward block, used by non-MoE Nemotron-H files.
pub struct NemotronDenseFfnWeights {
    pub up: Weight,
    pub up_bias: Vec<f32>,
    pub down: Weight,
    pub down_bias: Vec<f32>,
}

/// The single mixer a Nemotron-H block applies to its normalised residual.
/// Unlike a LLaMA block, each block has exactly one of these, not attention
/// *and* a feed-forward network.
pub enum NemotronMixer {
    Mamba2(Box<Mamba2LayerWeights>),
    Attention(Box<NemotronAttnWeights>),
    Moe(Box<NemotronMoeWeights>),
    DenseFfn(Box<NemotronDenseFfnWeights>),
}

pub struct NemotronHLayerWeights {
    pub attn_norm: Vec<f32>,
    pub mixer: NemotronMixer,
}

/// Hybrid Mamba-2 / attention / MoE decoder, as used by NVIDIA's Nemotron-H
/// family and by Soofi S Isar.
pub struct NemotronHWeights {
    pub token_embd: Weight,
    pub output_norm: Vec<f32>,
    pub output: Weight,
    pub layers: Vec<NemotronHLayerWeights>,
    pub ssm: SsmDims,
    /// Number of attention blocks, i.e. the KV cache's layer count.
    pub attn_layer_count: usize,
    pub router_normalize_weights: bool,
    pub routed_scaling_factor: f32,
}

/// Gated DeltaNet mixer used by Qwen3.5/Qwen3.8 recurrent blocks.
///
/// These tensors deliberately do not reuse [`Mamba2LayerWeights`]: Qwen's
/// recurrence is a delta-rule associative memory rather than a Mamba scan.
pub struct Qwen35LinearWeights {
    /// Concatenated Q/K/V projection: `[Q | K | V]`.
    pub qkv: Weight,
    /// Per-value-channel output gate (`z`).
    pub gate: Weight,
    /// Channel-major depthwise causal convolution, `conv_dim * d_conv`.
    pub conv_w: Vec<f32>,
    /// Per-value-head decay bias.
    pub dt_bias: Vec<f32>,
    /// Stored as `-exp(A_log)` by the GGUF converter.
    pub a: Vec<f32>,
    /// Per-value-head beta projection.
    pub beta: Weight,
    /// Per-value-head alpha projection.
    pub alpha: Weight,
    /// Shared RMSNorm vector for every value head.
    pub norm: Vec<f32>,
    /// Output projection back to the residual stream.
    pub out: Weight,
}

/// Full-attention mixer used every fourth Qwen3.5/Qwen3.8 block.
pub struct Qwen35AttentionWeights {
    /// Joint Q + sigmoid-gate projection. The two vectors are interleaved per
    /// head as `[Q_h | gate_h]`, not split into two global halves.
    pub q_gate: Weight,
    pub k: Weight,
    pub v: Weight,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub out: Weight,
    /// Slot in the compact cache containing only the full-attention blocks.
    pub kv_slot: usize,
}

pub enum Qwen35Mixer {
    Linear(Box<Qwen35LinearWeights>),
    Attention(Box<Qwen35AttentionWeights>),
}

/// One Qwen3.5/Qwen3.8 trunk block. Both mixer kinds are followed by the
/// same post-attention RMSNorm and dense SwiGLU FFN.
pub struct Qwen35LayerWeights {
    pub attn_norm: Vec<f32>,
    pub post_attn_norm: Vec<f32>,
    pub mixer: Qwen35Mixer,
    pub ffn_gate: Weight,
    pub ffn_up: Weight,
    pub ffn_down: Weight,
}

/// Single-token draft head embedded after a Qwen hybrid trunk. It combines
/// the trunk residual with the embedding of the just-selected token, then
/// predicts one additional token through its own gated-attention block.
pub struct Qwen35MtpWeights {
    pub eh_proj: Weight,
    pub embedding_norm: Vec<f32>,
    pub hidden_norm: Vec<f32>,
    pub attn_norm: Vec<f32>,
    pub post_attn_norm: Vec<f32>,
    pub attention: Qwen35AttentionWeights,
    pub ffn_gate: Weight,
    pub ffn_up: Weight,
    pub ffn_down: Weight,
    pub head_norm: Vec<f32>,
    pub token_embd: Option<Weight>,
    pub output: Option<Weight>,
}

/// Text decoder portion of a Qwen3.5/Qwen3.8 GGUF.
///
pub struct Qwen35Weights {
    pub token_embd: Weight,
    pub output_norm: Vec<f32>,
    pub output: Weight,
    pub layers: Vec<Qwen35LayerWeights>,
    /// Gated DeltaNet dimensions, sharing the physical cache storage type used
    /// by the existing hybrid implementation.
    pub ssm: SsmDims,
    pub recurrent_layer_count: usize,
    pub attn_layer_count: usize,
    /// Only this prefix of the 256-wide full-attention heads is rotary.
    pub rotary_dim: usize,
    /// Text uses the same scalar position on all MRoPE axes.
    pub rope_inv_freq: Vec<f32>,
    /// Optional embedded one-step draft head. It is kept separate from the
    /// trunk layer list and therefore never consumes trunk cache slots.
    pub mtp: Option<Box<Qwen35MtpWeights>>,
}

/// Per-layer recurrent state for Mamba-2 blocks — the state-space equivalent of
/// a KV cache. Unlike keys and values this does not grow with position: each
/// token overwrites it in place, so a rewind requires a replay from the start.
#[derive(Clone)]
pub struct SsmState {
    /// Per recurrent layer, a `conv_dim * (d_conv - 1)` shift register holding
    /// the previous convolution inputs, oldest first within each channel.
    pub conv: Vec<Vec<f32>>,
    /// Per recurrent layer, a `d_inner * d_state` state matrix indexed as
    /// `state_index + channel * d_state`.
    pub ssm: Vec<Vec<f32>>,
    pub dims: SsmDims,
}

impl SsmState {
    /// Allocates zeroed recurrent state for `layers` Mamba-2 blocks.
    pub fn new(layers: usize, dims: SsmDims) -> Self {
        Self {
            conv: vec![vec![0.0; dims.conv_dim() * dims.d_conv.saturating_sub(1)]; layers],
            ssm: vec![vec![0.0; dims.d_inner * dims.d_state]; layers],
            dims,
        }
    }

    /// Clears every recurrent slot, returning the model to its pre-prompt state.
    pub fn reset(&mut self) {
        for layer in &mut self.conv {
            layer.fill(0.0);
        }
        for layer in &mut self.ssm {
            layer.fill(0.0);
        }
    }
}

pub struct LagunaWeights {
    pub token_embd: Weight,
    pub output_norm: Vec<f32>,
    pub output: Weight,
    pub layers: Vec<LagunaLayerWeights>,
    pub router_normalize_weights: bool,
    pub routed_scaling_factor: f32,
}

// ─── KV Cache ────────────────────────────────────────────────────────────────

pub struct KVCache {
    pub k: Vec<Vec<f32>>, // [layer][slot * per_pos_k_dim ..]
    pub v: Vec<Vec<f32>>,
    /// Populated instead of `k`/`v` once `enable_bf16` is called. Halves the
    /// bytes read from RAM per scanned KV position during attention, which is
    /// the dominant remaining lever once decode is DRAM-bandwidth-bound (see
    /// BENCHMARK.md / perf notes) — weights already saturate the bus, so at
    /// long context the KV read itself becomes a comparable or larger share
    /// of per-token traffic.
    pub k_bf16: Vec<Vec<u16>>,
    pub v_bf16: Vec<Vec<u16>>,
    pub bf16: bool,
    pub per_pos_k_dim: usize,
    pub per_pos_v_dim: usize,
    pub max_len: usize,
    pub storage_len: usize,
    pub sliding_window: Option<usize>,
    /// Once a Qwen stream has entered the GPU-resident graph, falling back to
    /// the CPU mid-stream would use an empty CPU recurrent/KV prefix. Track
    /// that invariant so a backend failure cannot silently corrupt output.
    qwen_resident_active: bool,
    /// Recurrent state for hybrid Mamba-2 architectures. `None` for every
    /// attention-only model. It lives here rather than on `Session` so that the
    /// stateless one-shot generate path carries it too.
    pub ssm: Option<SsmState>,
}

impl KVCache {
    /// Marks the next token as the start of a fresh resident Qwen stream. The
    /// Metal backend clears its recurrent state when that token is encoded at
    /// position zero.
    pub(crate) fn reset_resident_stream(&mut self) {
        self.qwen_resident_active = false;
    }

    /// Allocates per-layer key and value cache buffers for autoregressive decode reuse.
    pub fn new(
        n_layers: usize,
        per_pos_k_dim: usize,
        per_pos_v_dim: usize,
        max_len: usize,
    ) -> Self {
        Self::with_sliding_window(n_layers, per_pos_k_dim, per_pos_v_dim, max_len, None)
    }

    /// Allocates a KV cache, using a ring buffer when sliding-window attention is active.
    pub fn with_sliding_window(
        n_layers: usize,
        per_pos_k_dim: usize,
        per_pos_v_dim: usize,
        max_len: usize,
        sliding_window: Option<usize>,
    ) -> Self {
        let max_len = max_len.max(1);
        let storage_len = Self::storage_len_for(max_len, sliding_window);
        Self {
            k: vec![vec![0.0; storage_len * per_pos_k_dim]; n_layers],
            v: vec![vec![0.0; storage_len * per_pos_v_dim]; n_layers],
            k_bf16: Vec::new(),
            v_bf16: Vec::new(),
            bf16: false,
            per_pos_k_dim,
            per_pos_v_dim,
            max_len,
            storage_len,
            sliding_window,
            qwen_resident_active: false,
            ssm: None,
        }
    }

    /// Attaches zeroed Mamba-2 recurrent state for a hybrid model.
    ///
    /// `attn_layers` sizes the key/value storage, which only covers attention
    /// blocks, while `recurrent_layers` sizes the state-space storage — the two
    /// counts differ because a hybrid block is one or the other, never both.
    pub fn with_recurrent_state(
        attn_layers: usize,
        per_pos_k_dim: usize,
        per_pos_v_dim: usize,
        max_len: usize,
        recurrent_layers: usize,
        dims: SsmDims,
    ) -> Self {
        let mut cache = Self::new(attn_layers, per_pos_k_dim, per_pos_v_dim, max_len);
        cache.ssm = Some(SsmState::new(recurrent_layers, dims));
        cache
    }

    /// Switches this (freshly constructed, not-yet-written) cache to store
    /// keys/values as bf16 instead of f32. The Standard LLaMA-style path and
    /// the CPU Qwen3.5 hybrid path both read/write this storage; other model
    /// families must keep f32 cache buffers.
    pub fn enable_bf16(&mut self) {
        if self.bf16 {
            return;
        }
        self.k_bf16 = self.k.iter().map(|layer| vec![0u16; layer.len()]).collect();
        self.v_bf16 = self.v.iter().map(|layer| vec![0u16; layer.len()]).collect();
        // Drop the f32 backing storage rather than leaving it allocated and
        // unused; keep the outer Vec length so `cache.k.len() == n_layers`
        // still holds for any caller that uses it that way.
        self.k = vec![Vec::new(); self.k.len()];
        self.v = vec![Vec::new(); self.v.len()];
        self.bf16 = true;
    }

    /// Updates the active sliding window and resizes storage if the ring size changed.
    pub fn set_sliding_window(&mut self, sliding_window: Option<usize>) -> bool {
        let storage_len = Self::storage_len_for(self.max_len, sliding_window);
        let changed = self.sliding_window != sliding_window || self.storage_len != storage_len;
        self.sliding_window = sliding_window;
        if storage_len != self.storage_len {
            self.storage_len = storage_len;
            if self.bf16 {
                for layer in &mut self.k_bf16 {
                    layer.resize(storage_len * self.per_pos_k_dim, 0);
                }
                for layer in &mut self.v_bf16 {
                    layer.resize(storage_len * self.per_pos_v_dim, 0);
                }
            } else {
                for layer in &mut self.k {
                    layer.resize(storage_len * self.per_pos_k_dim, 0.0);
                }
                for layer in &mut self.v {
                    layer.resize(storage_len * self.per_pos_v_dim, 0.0);
                }
            }
        }
        changed
    }

    #[inline]
    fn storage_len_for(max_len: usize, sliding_window: Option<usize>) -> usize {
        sliding_window
            .filter(|window| *window > 0)
            .map(|window| window.min(max_len.max(1)))
            .unwrap_or(max_len.max(1))
    }

    #[inline]
    fn slot_for_pos(&self, pos: usize) -> usize {
        if self.sliding_window.filter(|window| *window > 0).is_some() {
            pos % self.storage_len
        } else {
            pos
        }
    }

    #[inline]
    pub fn k_offset(&self, pos: usize) -> usize {
        self.slot_for_pos(pos) * self.per_pos_k_dim
    }

    #[inline]
    pub fn v_offset(&self, pos: usize) -> usize {
        self.slot_for_pos(pos) * self.per_pos_v_dim
    }

    /// Writes one position's key row, narrowing to bf16 if that mode is active.
    #[inline]
    pub fn write_k(&mut self, layer: usize, pos: usize, values: &[f32]) {
        let off = self.k_offset(pos);
        if self.bf16 {
            let dst = &mut self.k_bf16[layer][off..off + values.len()];
            for (d, &v) in dst.iter_mut().zip(values.iter()) {
                *d = crate::simd::f32_to_bf16(v);
            }
        } else {
            self.k[layer][off..off + values.len()].copy_from_slice(values);
        }
    }

    /// Writes one position's value row, narrowing to bf16 if that mode is active.
    #[inline]
    pub fn write_v(&mut self, layer: usize, pos: usize, values: &[f32]) {
        let off = self.v_offset(pos);
        if self.bf16 {
            let dst = &mut self.v_bf16[layer][off..off + values.len()];
            for (d, &v) in dst.iter_mut().zip(values.iter()) {
                *d = crate::simd::f32_to_bf16(v);
            }
        } else {
            self.v[layer][off..off + values.len()].copy_from_slice(values);
        }
    }
}

#[inline]
fn active_sliding_window(config: &Config, cache: &KVCache) -> usize {
    cache.sliding_window.unwrap_or(config.sliding_window)
}

#[inline]
fn attention_start_pos(pos: usize, sliding_window: usize) -> usize {
    if sliding_window > 0 {
        // Match the Mistral/Hugging Face sliding causal mask: the lower bound
        // is exclusive, so the current token plus visible history totals
        // exactly `sliding_window` positions.
        pos.saturating_add(1).saturating_sub(sliding_window)
    } else {
        0
    }
}

#[inline]
fn attention_uses_linear_slots(start_t: usize, end_t: usize, slot_count: usize) -> bool {
    start_t <= end_t && end_t < slot_count
}

#[cfg(test)]
mod tests {
    use super::{
        ExpertWeight, GGMLType, KVCache, RawTensorData, Weight, apply_rope_qk_neox,
        attention_start_pos, attention_uses_linear_slots, build_rope_inv_freq_with_factors,
    };

    #[test]
    fn sliding_attention_start_keeps_exact_window_width() {
        assert_eq!(attention_start_pos(0, 2), 0);
        assert_eq!(attention_start_pos(1, 2), 0);
        assert_eq!(attention_start_pos(2, 2), 1);
        assert_eq!(attention_start_pos(3, 2), 2);
    }

    #[test]
    fn sliding_attention_start_zero_disables_windowing() {
        assert_eq!(attention_start_pos(0, 0), 0);
        assert_eq!(attention_start_pos(128, 0), 0);
    }

    #[test]
    fn attention_linear_slots_detects_non_wrapping_cache_ranges() {
        assert!(attention_uses_linear_slots(0, 7, 8));
        assert!(attention_uses_linear_slots(3, 7, 8));
        assert!(!attention_uses_linear_slots(3, 8, 8));
        assert!(!attention_uses_linear_slots(4, 3, 8));
    }

    #[test]
    fn grouped_six_head_attention_matches_individual_heads() {
        const HEAD_DIM: usize = 8;
        const VALUE_DIM: usize = 6;
        const TOKENS: usize = 11;
        let queries: Vec<f32> = (0..6 * HEAD_DIM).map(|i| i as f32 * 0.013 - 0.2).collect();
        let keys: Vec<f32> = (0..TOKENS * HEAD_DIM)
            .map(|i| i as f32 * 0.021 - 0.3)
            .collect();
        let values: Vec<f32> = (0..TOKENS * VALUE_DIM)
            .map(|i| i as f32 * 0.009 - 0.4)
            .collect();
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let mut grouped = vec![0.0; 6 * VALUE_DIM];
        super::online_attention_grouped(
            &queries,
            &keys,
            &values,
            HEAD_DIM,
            VALUE_DIM,
            TOKENS,
            HEAD_DIM,
            VALUE_DIM,
            6,
            0,
            TOKENS - 1,
            scale,
            &mut grouped,
        );
        for head in 0..6 {
            let mut expected = vec![0.0; VALUE_DIM];
            super::online_attention(
                &queries[head * HEAD_DIM..(head + 1) * HEAD_DIM],
                &keys,
                &values,
                HEAD_DIM,
                VALUE_DIM,
                TOKENS,
                HEAD_DIM,
                VALUE_DIM,
                0,
                TOKENS - 1,
                scale,
                &mut expected,
            );
            for i in 0..VALUE_DIM {
                assert!(
                    (grouped[head * VALUE_DIM + i] - expected[i]).abs() < 1e-5,
                    "head {head}, value {i}: {} vs {}",
                    grouped[head * VALUE_DIM + i],
                    expected[i]
                );
            }
        }
    }

    #[test]
    fn grouped_attention_ring_slots_match_linearized_window() {
        const HEAD_DIM: usize = 8;
        const VALUE_DIM: usize = 6;
        const SLOT_COUNT: usize = 5;
        const START: usize = 7;
        const END: usize = 11;
        let queries: Vec<f32> = (0..4 * HEAD_DIM).map(|i| i as f32 * 0.017 - 0.15).collect();
        let ring_keys: Vec<f32> = (0..SLOT_COUNT * HEAD_DIM)
            .map(|i| i as f32 * 0.023 - 0.31)
            .collect();
        let ring_values: Vec<f32> = (0..SLOT_COUNT * VALUE_DIM)
            .map(|i| i as f32 * 0.011 - 0.27)
            .collect();
        let order = [2usize, 3, 4, 0, 1];
        let mut linear_keys = Vec::with_capacity(ring_keys.len());
        let mut linear_values = Vec::with_capacity(ring_values.len());
        for slot in order {
            linear_keys.extend_from_slice(&ring_keys[slot * HEAD_DIM..(slot + 1) * HEAD_DIM]);
            linear_values.extend_from_slice(&ring_values[slot * VALUE_DIM..(slot + 1) * VALUE_DIM]);
        }

        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let mut wrapped = vec![0.0; 4 * VALUE_DIM];
        super::online_attention_grouped(
            &queries,
            &ring_keys,
            &ring_values,
            HEAD_DIM,
            VALUE_DIM,
            SLOT_COUNT,
            HEAD_DIM,
            VALUE_DIM,
            4,
            START,
            END,
            scale,
            &mut wrapped,
        );
        let mut linear = vec![0.0; 4 * VALUE_DIM];
        super::online_attention_grouped(
            &queries,
            &linear_keys,
            &linear_values,
            HEAD_DIM,
            VALUE_DIM,
            SLOT_COUNT,
            HEAD_DIM,
            VALUE_DIM,
            4,
            0,
            SLOT_COUNT - 1,
            scale,
            &mut linear,
        );
        for (wrapped, linear) in wrapped.iter().zip(linear) {
            assert!((wrapped - linear).abs() < 1e-5, "{wrapped} vs {linear}");
        }

        let mut grouped_mha = vec![0.0; VALUE_DIM];
        let mut single_head = vec![0.0; VALUE_DIM];
        super::online_attention_grouped(
            &queries[..HEAD_DIM],
            &ring_keys,
            &ring_values,
            HEAD_DIM,
            VALUE_DIM,
            SLOT_COUNT,
            HEAD_DIM,
            VALUE_DIM,
            1,
            START,
            END,
            scale,
            &mut grouped_mha,
        );
        super::online_attention(
            &queries[..HEAD_DIM],
            &ring_keys,
            &ring_values,
            HEAD_DIM,
            VALUE_DIM,
            SLOT_COUNT,
            HEAD_DIM,
            VALUE_DIM,
            START,
            END,
            scale,
            &mut single_head,
        );
        assert_eq!(grouped_mha, single_head);
    }

    #[test]
    /// The core invariant KV-block-tiled prefill attention depends on:
    /// several `online_attention_grouped_scan` calls over consecutive,
    /// gap-free sub-ranges with carried-over (max_score, denom, out) state
    /// must be bit-identical to one `online_attention_grouped` call over the
    /// concatenated range. Uses kv_mul=6 to exercise both the x4-fused fast
    /// path (first 4 groups) and the scalar tail (groups 4-5) inside the
    /// scan core, and deliberately uneven block sizes (7 positions per
    /// block, 50 total, so the last block is a 1-position remainder) to
    /// catch any off-by-one at a chunk boundary.
    fn online_attention_grouped_scan_chunks_match_single_call() {
        const HEAD_DIM: usize = 8;
        const VALUE_DIM: usize = 6;
        const KV_MUL: usize = 6;
        const N: usize = 50;
        const BLOCK: usize = 7;
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();

        let queries: Vec<f32> = (0..KV_MUL * HEAD_DIM)
            .map(|i| (i as f32 * 0.031).sin())
            .collect();
        let keys: Vec<f32> = (0..N * HEAD_DIM)
            .map(|i| (i as f32 * 0.017).cos())
            .collect();
        let values: Vec<f32> = (0..N * VALUE_DIM)
            .map(|i| (i as f32 * 0.023).sin() * 0.4)
            .collect();

        let mut reference = vec![0.0f32; KV_MUL * VALUE_DIM];
        super::online_attention_grouped(
            &queries,
            &keys,
            &values,
            HEAD_DIM,
            VALUE_DIM,
            N,
            HEAD_DIM,
            VALUE_DIM,
            KV_MUL,
            0,
            N - 1,
            scale,
            &mut reference,
        );

        let mut chunked = vec![0.0f32; KV_MUL * VALUE_DIM];
        let mut max_score = [f32::NEG_INFINITY; KV_MUL];
        let mut denom = [0.0f32; KV_MUL];
        let mut start = 0usize;
        while start < N {
            let end = (start + BLOCK - 1).min(N - 1);
            super::online_attention_grouped_scan(
                &queries,
                &keys,
                &values,
                HEAD_DIM,
                VALUE_DIM,
                N,
                HEAD_DIM,
                VALUE_DIM,
                KV_MUL,
                start,
                end,
                scale,
                &mut max_score,
                &mut denom,
                &mut chunked,
            );
            start = end + 1;
        }
        super::online_attention_grouped_finalize(KV_MUL, VALUE_DIM, &denom, &mut chunked);

        assert_eq!(reference, chunked);
    }

    #[test]
    /// `attention_over_heads_with_sink` must agree bit-for-bit with a plain
    /// serial per-head reference loop (the code it replaced at both gpt-oss
    /// call sites), whichever internal path (worker-pool or serial fallback)
    /// actually runs for this process's thread count. Shape mirrors
    /// gpt-oss-20b (64 heads, 8 KV heads, kv_mul 8) with enough scanned
    /// positions (96 * 64 = 6144 > the 4096 work threshold) to exercise the
    /// parallel path when more than one worker thread is available.
    fn attention_over_heads_with_sink_matches_serial_reference() {
        const HEAD_DIM: usize = 8;
        const VALUE_DIM: usize = 6;
        const N_KV_HEADS: usize = 8;
        const KV_MUL: usize = 8;
        const N_HEADS: usize = N_KV_HEADS * KV_MUL;
        const SLOT_COUNT: usize = 96;
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();

        let queries: Vec<f32> = (0..N_HEADS * HEAD_DIM)
            .map(|i| (i as f32 * 0.013).sin())
            .collect();
        let keys: Vec<f32> = (0..SLOT_COUNT * N_KV_HEADS * HEAD_DIM)
            .map(|i| (i as f32 * 0.007).cos())
            .collect();
        let values: Vec<f32> = (0..SLOT_COUNT * N_KV_HEADS * VALUE_DIM)
            .map(|i| (i as f32 * 0.011).sin() * 0.5)
            .collect();
        let sinks: Vec<f32> = (0..N_HEADS).map(|h| h as f32 * 0.02 - 0.3).collect();

        let key_stride = N_KV_HEADS * HEAD_DIM;
        let value_stride = N_KV_HEADS * VALUE_DIM;
        let start_t = 0;
        let end_t = SLOT_COUNT - 1;

        let mut reference = vec![0.0f32; N_HEADS * VALUE_DIM];
        for h in 0..N_HEADS {
            let kv_h = h / KV_MUL;
            let q_off = h * HEAD_DIM;
            let out_off = h * VALUE_DIM;
            super::online_attention_with_sink(
                &queries[q_off..q_off + HEAD_DIM],
                &keys[kv_h * HEAD_DIM..],
                &values[kv_h * VALUE_DIM..],
                key_stride,
                value_stride,
                SLOT_COUNT,
                HEAD_DIM,
                VALUE_DIM,
                start_t,
                end_t,
                scale,
                sinks[h],
                &mut reference[out_off..out_off + VALUE_DIM],
            );
        }

        let mut actual = vec![0.0f32; N_HEADS * VALUE_DIM];
        super::attention_over_heads_with_sink(
            &queries,
            &keys,
            &values,
            &sinks,
            key_stride,
            value_stride,
            SLOT_COUNT,
            HEAD_DIM,
            VALUE_DIM,
            N_HEADS,
            KV_MUL,
            start_t,
            end_t,
            scale,
            &mut actual,
        );

        assert_eq!(reference, actual);
    }

    #[test]
    #[ignore = "manual microbenchmark, not a correctness check"]
    /// Measures the wall-clock win from parallelizing gpt-oss's sink
    /// attention, at a shape/context length representative of gpt-oss-20b
    /// (64 heads, 8 KV heads, kv_mul 8, ctx 4096) but with no GGUF involved —
    /// no gpt-oss model is available on this box (11.5 GB, not locally
    /// cached), so this isolates just the mechanism this change actually
    /// touches instead of leaving it completely unverified end-to-end.
    /// Interleaves both arms across several rounds to average out scheduler
    /// noise; run manually with `--ignored --nocapture`.
    fn attention_over_heads_with_sink_parallel_speedup() {
        use std::hint::black_box;
        use std::time::Instant;

        const HEAD_DIM: usize = 64;
        const VALUE_DIM: usize = 64;
        const N_KV_HEADS: usize = 8;
        const KV_MUL: usize = 8;
        const N_HEADS: usize = N_KV_HEADS * KV_MUL;
        const SLOT_COUNT: usize = 4096;
        const ROUNDS: usize = 5;
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();

        let queries: Vec<f32> = (0..N_HEADS * HEAD_DIM)
            .map(|i| (i as f32 * 0.013).sin())
            .collect();
        let keys: Vec<f32> = (0..SLOT_COUNT * N_KV_HEADS * HEAD_DIM)
            .map(|i| (i as f32 * 0.007).cos())
            .collect();
        let values: Vec<f32> = (0..SLOT_COUNT * N_KV_HEADS * VALUE_DIM)
            .map(|i| (i as f32 * 0.011).sin() * 0.5)
            .collect();
        let sinks: Vec<f32> = (0..N_HEADS).map(|h| h as f32 * 0.02 - 0.3).collect();
        let key_stride = N_KV_HEADS * HEAD_DIM;
        let value_stride = N_KV_HEADS * VALUE_DIM;
        let start_t = 0;
        let end_t = SLOT_COUNT - 1;

        let mut serial_out = vec![0.0f32; N_HEADS * VALUE_DIM];
        let mut parallel_out = vec![0.0f32; N_HEADS * VALUE_DIM];
        let mut serial_times = Vec::with_capacity(ROUNDS);
        let mut parallel_times = Vec::with_capacity(ROUNDS);

        for _ in 0..ROUNDS {
            let start = Instant::now();
            // online_attention_with_sink seeds its running max from the
            // (finite) sink score rather than NEG_INFINITY, so it can take
            // the additive branch on the very first real token and must
            // start from a zeroed buffer each round, same as the production
            // function does internally for `parallel_out`. Zeroing inside
            // the timed section keeps this symmetric with the parallel arm,
            // which times its own internal zero-fill too.
            for value in serial_out.iter_mut() {
                *value = 0.0;
            }
            for h in 0..N_HEADS {
                let kv_h = h / KV_MUL;
                let q_off = h * HEAD_DIM;
                let out_off = h * VALUE_DIM;
                super::online_attention_with_sink(
                    black_box(&queries[q_off..q_off + HEAD_DIM]),
                    &keys[kv_h * HEAD_DIM..],
                    &values[kv_h * VALUE_DIM..],
                    key_stride,
                    value_stride,
                    SLOT_COUNT,
                    HEAD_DIM,
                    VALUE_DIM,
                    start_t,
                    end_t,
                    scale,
                    sinks[h],
                    &mut serial_out[out_off..out_off + VALUE_DIM],
                );
            }
            serial_times.push(start.elapsed());

            let start = Instant::now();
            super::attention_over_heads_with_sink(
                black_box(&queries),
                &keys,
                &values,
                &sinks,
                key_stride,
                value_stride,
                SLOT_COUNT,
                HEAD_DIM,
                VALUE_DIM,
                N_HEADS,
                KV_MUL,
                start_t,
                end_t,
                scale,
                &mut parallel_out,
            );
            parallel_times.push(start.elapsed());
        }

        assert_eq!(serial_out, parallel_out);

        serial_times.sort();
        parallel_times.sort();
        let median = |v: &[std::time::Duration]| v[v.len() / 2];
        let (serial_med, parallel_med) = (median(&serial_times), median(&parallel_times));
        println!(
            "serial: {serial_times:?}\nparallel: {parallel_times:?}\nmedian serial={serial_med:?} parallel={parallel_med:?} speedup={:.2}x (threads={})",
            serial_med.as_secs_f64() / parallel_med.as_secs_f64().max(1e-12),
            crate::simd::num_threads(),
        );
    }

    #[test]
    #[cfg(not(target_family = "wasm"))]
    /// `attention_over_kv_heads_prefill_batch` must agree bit-for-bit with a
    /// per-token reference built from the exact single-token call it
    /// replaces (`online_attention_grouped` per (t, kv_h), same as the old
    /// `forward_prefill_batch` inner loop). Sized so the parallel dispatch
    /// path actually runs (work = sum_t(pos_t+1) * n_kv_heads = 8320, above
    /// the 4096 threshold) rather than only exercising the serial fallback —
    /// the `prefill_batch_parity_case` integration tests use a tiny model
    /// too small to ever cross that threshold, so they alone wouldn't catch
    /// a bug specific to the parallel branch.
    fn attention_over_kv_heads_prefill_batch_matches_per_token_reference() {
        const HEAD_DIM: usize = 8;
        const VALUE_DIM: usize = 6;
        const N_KV_HEADS: usize = 4;
        const KV_MUL: usize = 4;
        const N_HEADS: usize = N_KV_HEADS * KV_MUL;
        const B: usize = 64;
        const SLOT_COUNT: usize = B;
        const START_POS: usize = 0;
        const SLIDING_WINDOW: usize = 0;
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();

        let q_rows = N_HEADS * HEAD_DIM;
        let attn_dim = N_HEADS * VALUE_DIM;
        let key_stride = N_KV_HEADS * HEAD_DIM;
        let value_stride = N_KV_HEADS * VALUE_DIM;

        let queries: Vec<f32> = (0..B * q_rows).map(|i| (i as f32 * 0.0091).sin()).collect();
        let keys: Vec<f32> = (0..SLOT_COUNT * key_stride)
            .map(|i| (i as f32 * 0.0037).cos())
            .collect();
        let values: Vec<f32> = (0..SLOT_COUNT * value_stride)
            .map(|i| (i as f32 * 0.0059).sin() * 0.3)
            .collect();

        let mut reference = vec![0.0f32; B * attn_dim];
        for t in 0..B {
            let pos = START_POS + t;
            let attn_window = super::attention_start_pos(pos, SLIDING_WINDOW);
            for kv_h in 0..N_KV_HEADS {
                let q_off = t * q_rows + kv_h * KV_MUL * HEAD_DIM;
                let out_off = t * attn_dim + kv_h * KV_MUL * VALUE_DIM;
                super::online_attention_grouped(
                    &queries[q_off..q_off + KV_MUL * HEAD_DIM],
                    &keys[kv_h * HEAD_DIM..],
                    &values[kv_h * VALUE_DIM..],
                    key_stride,
                    value_stride,
                    SLOT_COUNT,
                    HEAD_DIM,
                    VALUE_DIM,
                    KV_MUL,
                    attn_window,
                    pos,
                    scale,
                    &mut reference[out_off..out_off + KV_MUL * VALUE_DIM],
                );
            }
        }

        let mut actual = vec![0.0f32; B * attn_dim];
        super::attention_over_kv_heads_prefill_batch(
            &queries,
            &keys,
            &values,
            key_stride,
            value_stride,
            SLOT_COUNT,
            HEAD_DIM,
            VALUE_DIM,
            N_KV_HEADS,
            KV_MUL,
            B,
            START_POS,
            SLIDING_WINDOW,
            scale,
            &mut actual,
        );

        assert_eq!(reference, actual);
    }

    #[test]
    #[cfg(not(target_family = "wasm"))]
    /// `attention_over_kv_heads_prefill_batch_tiled` must agree bit-for-bit
    /// with a per-token reference built the same way as the untiled
    /// dispatcher's test above. B=140 with `KV_TILE_BLOCK`=128 and
    /// `PREFILL_TOKEN_TILE`=64 means this spans 3 token tiles and, for the
    /// later tiles, 2 KV blocks — deliberately not a multiple of either
    /// constant, to catch an off-by-one at a tile or block boundary (the
    /// kind of bug the chunked-scan invariant test alone wouldn't catch,
    /// since that test doesn't exercise the tile/suffix bookkeeping this
    /// dispatcher adds on top of the scan core).
    fn attention_over_kv_heads_prefill_batch_tiled_matches_per_token_reference() {
        const HEAD_DIM: usize = 8;
        const VALUE_DIM: usize = 6;
        const N_KV_HEADS: usize = 2;
        const KV_MUL: usize = 4;
        const N_HEADS: usize = N_KV_HEADS * KV_MUL;
        const B: usize = 140;
        const SLOT_COUNT: usize = B;
        const START_POS: usize = 0;
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();

        let q_rows = N_HEADS * HEAD_DIM;
        let attn_dim = N_HEADS * VALUE_DIM;
        let key_stride = N_KV_HEADS * HEAD_DIM;
        let value_stride = N_KV_HEADS * VALUE_DIM;

        let queries: Vec<f32> = (0..B * q_rows).map(|i| (i as f32 * 0.0091).sin()).collect();
        let keys: Vec<f32> = (0..SLOT_COUNT * key_stride)
            .map(|i| (i as f32 * 0.0037).cos())
            .collect();
        let values: Vec<f32> = (0..SLOT_COUNT * value_stride)
            .map(|i| (i as f32 * 0.0059).sin() * 0.3)
            .collect();

        let mut reference = vec![0.0f32; B * attn_dim];
        for t in 0..B {
            let pos = START_POS + t;
            for kv_h in 0..N_KV_HEADS {
                let q_off = t * q_rows + kv_h * KV_MUL * HEAD_DIM;
                let out_off = t * attn_dim + kv_h * KV_MUL * VALUE_DIM;
                super::online_attention_grouped(
                    &queries[q_off..q_off + KV_MUL * HEAD_DIM],
                    &keys[kv_h * HEAD_DIM..],
                    &values[kv_h * VALUE_DIM..],
                    key_stride,
                    value_stride,
                    SLOT_COUNT,
                    HEAD_DIM,
                    VALUE_DIM,
                    KV_MUL,
                    0,
                    pos,
                    scale,
                    &mut reference[out_off..out_off + KV_MUL * VALUE_DIM],
                );
            }
        }

        let mut actual = vec![0.0f32; B * attn_dim];
        super::attention_over_kv_heads_prefill_batch_tiled(
            &queries,
            &keys,
            &values,
            key_stride,
            value_stride,
            SLOT_COUNT,
            HEAD_DIM,
            VALUE_DIM,
            N_KV_HEADS,
            KV_MUL,
            B,
            START_POS,
            scale,
            &mut actual,
        );

        assert_eq!(reference, actual);
    }

    #[test]
    fn sliding_kv_cache_uses_ring_storage_without_lowering_context_limit() {
        let mut cache = KVCache::with_sliding_window(2, 4, 6, 128, Some(8));
        assert_eq!(cache.max_len, 128);
        assert_eq!(cache.storage_len, 8);
        assert_eq!(cache.k[0].len(), 32);
        assert_eq!(cache.v[0].len(), 48);
        assert_eq!(cache.k_offset(9), 4);
        assert_eq!(cache.v_offset(9), 6);

        assert!(cache.set_sliding_window(None));
        assert_eq!(cache.max_len, 128);
        assert_eq!(cache.storage_len, 128);
        assert_eq!(cache.k_offset(9), 36);
        assert_eq!(cache.v_offset(9), 54);
    }

    #[test]
    fn enable_bf16_moves_storage_and_write_k_v_narrows_losslessly_within_precision() {
        let mut cache = KVCache::with_sliding_window(2, 4, 6, 16, None);
        cache.enable_bf16();
        assert!(cache.bf16);
        // f32 backing storage is dropped, but the outer per-layer Vec length
        // (n_layers) is preserved.
        assert_eq!(cache.k.len(), 2);
        assert_eq!(cache.k[0].len(), 0);
        assert_eq!(cache.k_bf16[0].len(), 4 * 16);
        assert_eq!(cache.v_bf16[0].len(), 6 * 16);

        let k_row = [0.5f32, -1.25, 3.0, 0.125];
        let v_row = [1.0f32, -2.0, 0.0, 4.5, -0.75, 2.25];
        cache.write_k(0, 3, &k_row);
        cache.write_v(0, 3, &v_row);
        let off_k = cache.k_offset(3);
        let off_v = cache.v_offset(3);
        for (i, &expected) in k_row.iter().enumerate() {
            assert_eq!(
                crate::simd::bf16_to_f32(cache.k_bf16[0][off_k + i]),
                expected,
                "bf16 exactly represents these round values"
            );
        }
        for (i, &expected) in v_row.iter().enumerate() {
            assert_eq!(
                crate::simd::bf16_to_f32(cache.v_bf16[0][off_v + i]),
                expected
            );
        }

        // set_sliding_window must resize the bf16 storage, not the (empty) f32 one.
        assert!(cache.set_sliding_window(Some(4)));
        assert_eq!(cache.k_bf16[0].len(), 4 * 4);
        assert_eq!(cache.k[0].len(), 0);
    }

    fn assert_grouped_attention_bf16_matches_f32_within_precision(kv_mul: usize) {
        const HEAD_DIM: usize = 8;
        const VALUE_DIM: usize = 6;
        const TOKENS: usize = 5;
        let queries: Vec<f32> = (0..kv_mul * HEAD_DIM)
            .map(|i| i as f32 * 0.017 - 0.15)
            .collect();
        let keys: Vec<f32> = (0..TOKENS * HEAD_DIM)
            .map(|i| i as f32 * 0.023 - 0.31)
            .collect();
        let values: Vec<f32> = (0..TOKENS * VALUE_DIM)
            .map(|i| i as f32 * 0.011 - 0.27)
            .collect();
        let keys_bf16: Vec<u16> = keys.iter().map(|&v| crate::simd::f32_to_bf16(v)).collect();
        let values_bf16: Vec<u16> = values
            .iter()
            .map(|&v| crate::simd::f32_to_bf16(v))
            .collect();

        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let mut f32_out = vec![0.0; kv_mul * VALUE_DIM];
        super::online_attention_grouped(
            &queries,
            &keys,
            &values,
            HEAD_DIM,
            VALUE_DIM,
            TOKENS,
            HEAD_DIM,
            VALUE_DIM,
            kv_mul,
            0,
            TOKENS - 1,
            scale,
            &mut f32_out,
        );
        let mut bf16_out = vec![0.0; kv_mul * VALUE_DIM];
        super::online_attention_grouped_bf16(
            &queries,
            &keys_bf16,
            &values_bf16,
            HEAD_DIM,
            VALUE_DIM,
            TOKENS,
            HEAD_DIM,
            VALUE_DIM,
            kv_mul,
            0,
            TOKENS - 1,
            scale,
            &mut bf16_out,
        );
        for (got, want) in bf16_out.iter().zip(f32_out.iter()) {
            assert!(
                (got - want).abs() <= 0.01 * want.abs().max(1.0),
                "got {got}, want {want}"
            );
        }
    }

    #[test]
    fn grouped_four_head_attention_bf16_matches_f32_within_bf16_precision() {
        assert_grouped_attention_bf16_matches_f32_within_precision(4);
    }

    #[test]
    fn grouped_six_head_attention_bf16_matches_f32_within_bf16_precision() {
        assert_grouped_attention_bf16_matches_f32_within_precision(6);
    }

    #[test]
    #[ignore = "manual release benchmark; run cargo test --release --lib attention_bf16_speedup_at_long_context -- --ignored --nocapture"]
    fn attention_bf16_speedup_at_long_context() {
        // Isolates the exact kernel changed by KVCache::enable_bf16 — no
        // model load, no weight matvecs, no tokenizer — so it measures the
        // bytes-moved argument directly instead of being swamped by the
        // DRAM-bandwidth-bound weight reads that dominate a real decode step.
        // Dims match Ministral-3B (head_dim=128, value_dim=128, n_kv_heads=8,
        // kv_mul=4, 26 layers); CTX matches the "long context" regime where
        // the plan (perf-work-status memory) expects the win to show up.
        //
        // Each (layer, kv_head) gets its OWN buffer rather than reusing one:
        // a single kv_head's K+V at ctx=8192 is only ~8 MiB, well inside this
        // CPU's L3 — repeatedly scanning the same 8 MiB would benchmark cache
        // bandwidth, not the DRAM bandwidth the whole optimization targets.
        // The full N_LAYERS x N_KV_HEADS working set (~1.6 GiB f32 / 0.8 GiB
        // bf16) cannot be cache-resident, matching a real decode step where
        // every layer's KV is read fresh and evicted by the next layer's.
        const HEAD_DIM: usize = 128;
        const VALUE_DIM: usize = 128;
        const N_KV_HEADS: usize = 8;
        const KV_MUL: usize = 4;
        const N_LAYERS: usize = 26;
        const CTX: usize = 8192;
        const RUNS: usize = 2;
        const N_BUFFERS: usize = N_LAYERS * N_KV_HEADS;

        let queries: Vec<f32> = (0..KV_MUL * HEAD_DIM)
            .map(|i| (i as f32 * 0.013).sin())
            .collect();
        let keys: Vec<f32> = (0..N_BUFFERS * CTX * HEAD_DIM)
            .map(|i| (i as f32 * 0.0000007).cos() * 0.5)
            .collect();
        let values: Vec<f32> = (0..N_BUFFERS * CTX * VALUE_DIM)
            .map(|i| (i as f32 * 0.0000011).sin() * 0.5)
            .collect();
        let keys_bf16: Vec<u16> = keys.iter().map(|&v| crate::simd::f32_to_bf16(v)).collect();
        let values_bf16: Vec<u16> = values
            .iter()
            .map(|&v| crate::simd::f32_to_bf16(v))
            .collect();
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();

        let mut f32_out = vec![0.0f32; KV_MUL * VALUE_DIM];
        let f32_time = {
            let start = std::time::Instant::now();
            for _ in 0..RUNS {
                for buf in 0..N_BUFFERS {
                    let k = &keys[buf * CTX * HEAD_DIM..(buf + 1) * CTX * HEAD_DIM];
                    let v = &values[buf * CTX * VALUE_DIM..(buf + 1) * CTX * VALUE_DIM];
                    super::online_attention_grouped(
                        &queries,
                        k,
                        v,
                        HEAD_DIM,
                        VALUE_DIM,
                        CTX,
                        HEAD_DIM,
                        VALUE_DIM,
                        KV_MUL,
                        0,
                        CTX - 1,
                        scale,
                        &mut f32_out,
                    );
                }
            }
            start.elapsed()
        };

        let mut bf16_out = vec![0.0f32; KV_MUL * VALUE_DIM];
        let bf16_time = {
            let start = std::time::Instant::now();
            for _ in 0..RUNS {
                for buf in 0..N_BUFFERS {
                    let k = &keys_bf16[buf * CTX * HEAD_DIM..(buf + 1) * CTX * HEAD_DIM];
                    let v = &values_bf16[buf * CTX * VALUE_DIM..(buf + 1) * CTX * VALUE_DIM];
                    super::online_attention_grouped_bf16(
                        &queries,
                        k,
                        v,
                        HEAD_DIM,
                        VALUE_DIM,
                        CTX,
                        HEAD_DIM,
                        VALUE_DIM,
                        KV_MUL,
                        0,
                        CTX - 1,
                        scale,
                        &mut bf16_out,
                    );
                }
            }
            start.elapsed()
        };

        for (got, want) in bf16_out.iter().zip(f32_out.iter()) {
            assert!(
                (got - want).abs() <= 0.02 * want.abs().max(1.0),
                "got {got}, want {want}"
            );
        }
        let speedup = f32_time.as_secs_f64() / bf16_time.as_secs_f64();
        let total_f32_mib = (N_BUFFERS * CTX * (HEAD_DIM + VALUE_DIM) * 4) / (1024 * 1024);
        let total_bf16_mib = (N_BUFFERS * CTX * (HEAD_DIM + VALUE_DIM) * 2) / (1024 * 1024);
        eprintln!(
            "Full-model-shaped attention scan at ctx={CTX}, {N_LAYERS} layers x {N_KV_HEADS} kv-heads: f32={:.1} ms ({total_f32_mib} MiB), bf16={:.1} ms ({total_bf16_mib} MiB), speedup={:.2}x",
            f32_time.as_secs_f64() * 1000.0 / RUNS as f64,
            bf16_time.as_secs_f64() * 1000.0 / RUNS as f64,
            speedup,
        );
    }

    #[test]
    fn rope_freq_factors_can_disable_rotation_pairs() {
        let inv = build_rope_inv_freq_with_factors(10_000.0, 4, 1.0, Some(&[1.0, 1e30]));
        assert!((inv[0] - 1.0).abs() < 1e-6);
        assert!(inv[1] < 1e-30);
    }

    #[test]
    fn neox_rope_rotates_across_head_halves() {
        let mut q = vec![1.0, 2.0, 3.0, 4.0];
        let mut k = vec![5.0, 6.0, 7.0, 8.0];
        let inv = vec![std::f32::consts::FRAC_PI_2, 0.0];

        apply_rope_qk_neox(&mut q, &mut k, 1, 4, 1, 1, &inv);

        assert!((q[0] + 3.0).abs() < 1e-5);
        assert!((q[1] - 2.0).abs() < 1e-5);
        assert!((q[2] - 1.0).abs() < 1e-5);
        assert!((q[3] - 4.0).abs() < 1e-5);
        assert!((k[0] + 7.0).abs() < 1e-5);
        assert!((k[2] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn prepared_rope_matches_per_layer_angle_calculation() {
        let inv = vec![0.73, 0.19, 0.05, 0.01];
        let mut sin = vec![0.0; inv.len()];
        let mut cos = vec![0.0; inv.len()];
        super::prepare_rope_sin_cos_into(17, &inv, &mut sin, &mut cos);

        let q: Vec<f32> = (0..16).map(|i| i as f32 * 0.07 - 0.3).collect();
        let k: Vec<f32> = (0..8).map(|i| i as f32 * -0.11 + 0.2).collect();

        let mut expected_q = q.clone();
        let mut expected_k = k.clone();
        super::apply_rope_qk(&mut expected_q, &mut expected_k, 17, 8, 2, 1, &inv);
        let mut prepared_q = q.clone();
        let mut prepared_k = k.clone();
        super::apply_rope_qk_prepared(&mut prepared_q, &mut prepared_k, 8, 2, 1, &sin, &cos);
        assert_eq!(prepared_q, expected_q);
        assert_eq!(prepared_k, expected_k);

        let mut expected_q = q.clone();
        let mut expected_k = k.clone();
        super::apply_rope_qk_neox(&mut expected_q, &mut expected_k, 17, 8, 2, 1, &inv);
        let mut prepared_q = q;
        let mut prepared_k = k;
        super::apply_rope_qk_neox_prepared(&mut prepared_q, &mut prepared_k, 8, 2, 1, &sin, &cos);
        assert_eq!(prepared_q, expected_q);
        assert_eq!(prepared_k, expected_k);
    }

    #[test]
    fn qwen_rope_uses_rotate_half_layout() {
        let config = super::Config {
            arch: "qwen3".to_string(),
            dim: 4,
            hidden_dim: 8,
            n_layers: 1,
            n_heads: 1,
            n_kv_heads: 1,
            vocab_size: 8,
            max_seq_len: 8,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
            head_dim: 4,
            kv_dim: 4,
            kv_mul: 1,
            value_dim: 4,
            sliding_window: 0,
            expert_count: 0,
            expert_used_count: 0,
            rope_scaling_factor: 1.0,
            rope_original_context_length: 0,
        };
        let mut q = vec![1.0, 2.0, 3.0, 4.0];
        let mut k = vec![5.0, 6.0, 7.0, 8.0];
        let inv = vec![std::f32::consts::FRAC_PI_2, 0.0];

        super::apply_model_rope(&config, &mut q, &mut k, 1, &inv);

        assert!((q[0] + 3.0).abs() < 1e-5);
        assert!((q[1] - 2.0).abs() < 1e-5);
        assert!((q[2] - 1.0).abs() < 1e-5);
        assert!((q[3] - 4.0).abs() < 1e-5);
        assert!((k[0] + 7.0).abs() < 1e-5);
        assert!((k[2] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn qk_norm_normalizes_each_head_before_rope() {
        let mut q = vec![3.0, 4.0, 0.0, 5.0];
        let mut k = vec![6.0, 8.0];
        super::apply_qk_norm_if_present(&mut q, &mut k, 2, 2, 1, &[2.0, 1.0], &[1.0, 3.0], 0.0);

        assert!((q[0] - 1.697_056_3).abs() < 1e-5);
        assert!((q[1] - 1.131_370_9).abs() < 1e-5);
        assert!((q[2] - 0.0).abs() < 1e-5);
        assert!((q[3] - std::f32::consts::SQRT_2).abs() < 1e-5);
        assert!((k[0] - 0.848_528_15).abs() < 1e-5);
        assert!((k[1] - 3.394_112_6).abs() < 1e-5);
    }

    #[test]
    fn layer_norm_matches_manual_mean_variance() {
        // x = [1,2,3,4]: mean 2.5, var 1.25; weight 2, bias 1.
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
        let w = vec![2.0f32; 4];
        let b = vec![1.0f32; 4];
        super::layer_norm_in_place(&mut x, &w, &b, 0.0);
        let inv = 1.0 / 1.25f32.sqrt();
        let expect = [
            (1.0 - 2.5) * inv * 2.0 + 1.0,
            (2.0 - 2.5) * inv * 2.0 + 1.0,
            (3.0 - 2.5) * inv * 2.0 + 1.0,
            (4.0 - 2.5) * inv * 2.0 + 1.0,
        ];
        for (got, want) in x.iter().zip(expect.iter()) {
            assert!((got - want).abs() < 1e-5, "layer norm {got} vs {want}");
        }
    }

    #[test]
    fn paired_mxfp4_expert_matvec_matches_separate_matvecs() {
        const EXPERTS: usize = 2;
        const ROWS: usize = 3;
        const COLS: usize = 32;
        const ROW_BYTES: usize = 17;

        let make_weight = |seed: u8| {
            let mut data = vec![0u8; EXPERTS * ROWS * ROW_BYTES];
            for expert in 0..EXPERTS {
                for row in 0..ROWS {
                    let base = (expert * ROWS + row) * ROW_BYTES;
                    for i in 0..16 {
                        let lo = seed.wrapping_add((expert * 7 + row * 5 + i) as u8) & 0x0f;
                        let hi = seed.wrapping_add((expert * 3 + row * 11 + i * 2) as u8) & 0x0f;
                        data[base + i] = lo | (hi << 4);
                    }
                    // MXFP4 exponent byte: 127 encodes a scale of 1.0.
                    data[base + 16] = 127;
                }
            }
            ExpertWeight {
                data: RawTensorData::Owned(data),
                dtype: GGMLType::MXFP4,
                experts: EXPERTS,
                rows: ROWS,
                cols: COLS,
            }
        };

        let gate = make_weight(1);
        let up = make_weight(9);
        let x: Vec<f32> = (0..COLS).map(|i| i as f32 * 0.125 - 1.75).collect();
        let expert = 1;

        let mut separate_gate = Vec::new();
        let mut separate_up = Vec::new();
        gate.matvec_expert_into(expert, &x, &mut separate_gate);
        up.matvec_expert_into(expert, &x, &mut separate_up);

        let mut paired_gate = Vec::new();
        let mut paired_up = Vec::new();
        assert!(gate.try_matvec_expert_pair_into(
            &up,
            expert,
            &x,
            &mut paired_gate,
            &mut paired_up,
        ));
        assert_eq!(paired_gate.len(), ROWS);
        assert_eq!(paired_up.len(), ROWS);
        for (got, expected) in paired_gate.iter().zip(&separate_gate) {
            assert!((got - expected).abs() < 1e-6, "gate {got} vs {expected}");
        }
        for (got, expected) in paired_up.iter().zip(&separate_up) {
            assert!((got - expected).abs() < 1e-6, "up {got} vs {expected}");
        }
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    /// Mixed Q4_K/Q4_K/Q6_K Q/K/V weights share the activation quantization
    /// and worker-pool rendezvous without changing the three projections.
    fn mixed_kquant_qkv_fusion_matches_individual_matvecs() {
        const COLS: usize = 256;
        const ROWS: usize = 3;
        const Q4_ROW_BYTES: usize = 144;
        const Q6_ROW_BYTES: usize = 210;

        let q4_weight = |salt: u8| {
            let mut data = vec![0u8; ROWS * Q4_ROW_BYTES];
            for (row, bytes) in data.chunks_exact_mut(Q4_ROW_BYTES).enumerate() {
                // Q4_K block: d=1, dmin=0, followed by 12 scale bytes.
                // and 128 packed quants. Values are deterministic but nonzero.
                bytes[0] = 0;
                bytes[1] = 0x3c;
                for (i, value) in bytes[4..16].iter_mut().enumerate() {
                    *value = salt.wrapping_add((row * 13 + i * 7) as u8);
                }
                for (i, value) in bytes[16..].iter_mut().enumerate() {
                    *value = salt.wrapping_add((row * 17 + i * 11) as u8);
                }
            }
            Weight::Quantized {
                data: RawTensorData::Owned(data),
                dtype: GGMLType::Q4_K,
                rows: ROWS,
                cols: COLS,
            }
        };
        let q6_weight = |salt: u8| {
            let mut data = vec![0u8; ROWS * Q6_ROW_BYTES];
            for (row, bytes) in data.chunks_exact_mut(Q6_ROW_BYTES).enumerate() {
                // Q6_K block: 128 low bits, 64 high bits, 16 scales, then d.
                for (i, value) in bytes[..192].iter_mut().enumerate() {
                    *value = salt.wrapping_add((row * 19 + i * 5) as u8);
                }
                for (i, value) in bytes[192..208].iter_mut().enumerate() {
                    *value = (i as i8 - 8) as u8;
                }
                bytes[208] = 0;
                bytes[209] = 0x3c;
            }
            Weight::Quantized {
                data: RawTensorData::Owned(data),
                dtype: GGMLType::Q6_K,
                rows: ROWS,
                cols: COLS,
            }
        };

        let q = q4_weight(3);
        let k = q4_weight(29);
        let v = q6_weight(71);
        let x: Vec<f32> = (0..COLS)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.0625)
            .collect();

        let mut expected_q = Vec::new();
        let mut expected_k = Vec::new();
        let mut expected_v = Vec::new();
        q.matvec_into(&x, &mut expected_q);
        k.matvec_into(&x, &mut expected_k);
        v.matvec_into(&x, &mut expected_v);
        assert!(
            expected_q
                .iter()
                .chain(&expected_k)
                .chain(&expected_v)
                .all(|v| v.is_finite())
        );

        let mut fused_q = Vec::new();
        let mut fused_k = Vec::new();
        let mut fused_v = Vec::new();
        assert!(super::try_quant_matvec3_into(
            &q,
            &k,
            &v,
            &x,
            &mut fused_q,
            &mut fused_k,
            &mut fused_v,
        ));

        for (got, expected) in fused_q.iter().zip(&expected_q) {
            assert!((got - expected).abs() <= expected.abs().max(1.0) * 1e-5);
        }
        for (got, expected) in fused_k.iter().zip(&expected_k) {
            assert!((got - expected).abs() <= expected.abs().max(1.0) * 1e-5);
        }
        for (got, expected) in fused_v.iter().zip(&expected_v) {
            assert!((got - expected).abs() <= expected.abs().max(1.0) * 1e-5);
        }
    }

    #[test]
    #[ignore = "manual release benchmark; run cargo test --release --lib mxfp4_expert_pair_speedup -- --ignored --nocapture"]
    /// Measures the worker-pool rendezvous saved by GPT-OSS gate/up fusion on
    /// the actual 2,880 x 2,880 expert projection shape.
    fn mxfp4_expert_pair_speedup() {
        const ROWS: usize = 2880;
        const COLS: usize = 2880;
        const ROW_BYTES: usize = (COLS / 32) * 17;
        const RUNS: usize = 100;

        let make_weight = |seed: u8| {
            let mut data = vec![0u8; ROWS * ROW_BYTES];
            for row in 0..ROWS {
                for block in 0..(COLS / 32) {
                    let base = row * ROW_BYTES + block * 17;
                    for i in 0..16 {
                        let lo = ((row * 3 + block * 5 + i * 7 + seed as usize) & 0x0f) as u8;
                        let hi = ((row * 11 + block * 13 + i * 2 + seed as usize) & 0x0f) as u8;
                        data[base + i] = lo | (hi << 4);
                    }
                    data[base + 16] = 123 + ((row + block) as u8 % 9);
                }
            }
            ExpertWeight {
                data: RawTensorData::Owned(data),
                dtype: GGMLType::MXFP4,
                experts: 1,
                rows: ROWS,
                cols: COLS,
            }
        };

        let gate = make_weight(1);
        let up = make_weight(9);
        let x: Vec<f32> = (0..COLS)
            .map(|i| (i as f32 * 0.017).cos() * 0.75 + (i % 19) as f32 * 0.01)
            .collect();
        let mut gate_out = Vec::new();
        let mut up_out = Vec::new();

        // Start the persistent worker pool before timing either implementation.
        gate.matvec_expert_into(0, &x, &mut gate_out);
        up.matvec_expert_into(0, &x, &mut up_out);
        assert!(gate.try_matvec_expert_pair_into(&up, 0, &x, &mut gate_out, &mut up_out));

        let separate_start = std::time::Instant::now();
        let mut separate_checksum = 0.0f32;
        for _ in 0..RUNS {
            gate.matvec_expert_into(0, &x, &mut gate_out);
            up.matvec_expert_into(0, &x, &mut up_out);
            separate_checksum += gate_out[0] + up_out[0];
        }
        let separate_time = separate_start.elapsed();

        let fused_start = std::time::Instant::now();
        let mut fused_checksum = 0.0f32;
        for _ in 0..RUNS {
            assert!(gate.try_matvec_expert_pair_into(&up, 0, &x, &mut gate_out, &mut up_out));
            fused_checksum += gate_out[0] + up_out[0];
        }
        let fused_time = fused_start.elapsed();

        assert!((fused_checksum - separate_checksum).abs() < 1e-4);
        let speedup = separate_time.as_secs_f64() / fused_time.as_secs_f64();
        eprintln!(
            "GPT-OSS expert gate/up: separate={:.3} ms, fused={:.3} ms, speedup={:.2}x",
            separate_time.as_secs_f64() * 1000.0,
            fused_time.as_secs_f64() * 1000.0,
            speedup,
        );
    }

    /// Builds a tiny synthetic nomic-bert model (dim 8, 2 heads, head_dim 4,
    /// 1 layer, SwiGLU FFN) with deterministic F32 weights for forward tests.
    #[cfg(test)]
    fn tiny_nomic_model() -> (super::Config, super::NomicBertWeights) {
        use super::{Config, NomicBertLayerWeights, NomicBertWeights, Weight};
        let dim = 8usize;
        let hidden = 16usize;
        let vocab = 12usize;
        let head_dim = 4usize;
        let n_heads = 2usize;
        let config = Config {
            arch: "nomic-bert".to_string(),
            dim,
            hidden_dim: hidden,
            n_layers: 1,
            n_heads,
            n_kv_heads: n_heads,
            vocab_size: vocab,
            max_seq_len: 64,
            rope_theta: 1000.0,
            rms_norm_eps: 1e-12,
            head_dim,
            kv_dim: head_dim * n_heads,
            kv_mul: 1,
            value_dim: head_dim,
            sliding_window: 0,
            expert_count: 0,
            expert_used_count: 0,
            rope_scaling_factor: 1.0,
            rope_original_context_length: 0,
        };
        // Deterministic pseudo-random f32 fill in [-0.5, 0.5).
        let fill = |n: usize, seed: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((i * 2654435761 + seed * 40503) % 1000) as f32 / 1000.0) - 0.5)
                .collect()
        };
        let sq = dim * dim;
        let layer = NomicBertLayerWeights {
            wq: Weight::F32(fill(sq, 1)),
            bq: vec![0.0; dim],
            wk: Weight::F32(fill(sq, 2)),
            bk: vec![0.0; dim],
            wv: Weight::F32(fill(sq, 3)),
            bv: vec![0.0; dim],
            wo: Weight::F32(fill(sq, 4)),
            bo: vec![0.0; dim],
            attn_out_norm: vec![1.0; dim],
            attn_out_norm_b: vec![0.0; dim],
            ffn_gate: Some(Weight::F32(fill(hidden * dim, 5))),
            ffn_up: Weight::F32(fill(hidden * dim, 6)),
            ffn_up_b: vec![0.0; hidden],
            ffn_down: Weight::F32(fill(dim * hidden, 7)),
            ffn_down_b: vec![0.0; dim],
            layer_out_norm: vec![1.0; dim],
            layer_out_norm_b: vec![0.0; dim],
        };
        let weights = NomicBertWeights {
            token_embd: Weight::F32(fill(vocab * dim, 8)),
            token_type0: Vec::new(),
            tok_norm: vec![1.0; dim],
            tok_norm_b: vec![0.0; dim],
            layers: vec![layer],
            ln_eps: 1e-12,
        };
        (config, weights)
    }

    /// Creates compact, non-zero Q5_K rows for exercising the quantized Nomic
    /// batch path. Q5_K intentionally has no Metal kernel, which makes the
    /// serial/batched parity check independent of the caller's Metal setting.
    fn tiny_q5k_weight(rows: usize, cols: usize, seed: u8) -> Weight {
        assert_eq!(cols % 256, 0);
        let row_bytes = (cols / 256) * 176;
        let mut data = vec![0u8; rows * row_bytes];
        for row in 0..rows {
            for block in 0..cols / 256 {
                let base = row * row_bytes + block * 176;
                // f16 0.0078125: small enough that the synthetic LayerNorm
                // path stays well-conditioned while preserving non-zero dots.
                data[base..base + 2].copy_from_slice(&0x2000u16.to_le_bytes());
                // dmin remains zero. Pack scale=1/min=0 for all eight groups.
                for i in 0..4 {
                    data[base + 4 + i] = 1;
                    data[base + 12 + i] = 1;
                }
                // qh is zero, so the remaining bytes encode 4-bit values.
                for i in 0..128 {
                    data[base + 48 + i] =
                        seed.wrapping_add((row * 17 + block * 29 + i * 3) as u8) & 0x77;
                }
            }
        }
        Weight::Quantized {
            data: RawTensorData::Owned(data),
            dtype: GGMLType::Q5_K,
            rows,
            cols,
        }
    }

    /// Builds one full-width Q5_K Nomic layer. Eight tokens select the
    /// row-balanced batch scheduler, while the test can still force the
    /// original per-token execution through `forward_nomic_bert_hidden_impl`.
    fn quantized_nomic_model() -> (super::Config, super::NomicBertWeights) {
        use super::{Config, NomicBertLayerWeights, NomicBertWeights};

        let dim = 256usize;
        let hidden = 256usize;
        let vocab = 16usize;
        let config = Config {
            arch: "nomic-bert".to_string(),
            dim,
            hidden_dim: hidden,
            n_layers: 1,
            n_heads: 1,
            n_kv_heads: 1,
            vocab_size: vocab,
            max_seq_len: 64,
            rope_theta: 1000.0,
            rms_norm_eps: 1e-12,
            head_dim: dim,
            kv_dim: dim,
            kv_mul: 1,
            value_dim: dim,
            sliding_window: 0,
            expert_count: 0,
            expert_used_count: 0,
            rope_scaling_factor: 1.0,
            rope_original_context_length: 0,
        };
        let fill = |n: usize, seed: usize| -> Vec<f32> {
            (0..n)
                .map(|i| ((i * 17 + seed * 13) % 31) as f32 * 0.001 - 0.015)
                .collect()
        };
        let bias = |n: usize, seed: usize| -> Vec<f32> {
            (0..n)
                .map(|i| ((i + seed) % 5) as f32 * 0.001 - 0.002)
                .collect()
        };
        let layer = NomicBertLayerWeights {
            wq: tiny_q5k_weight(dim, dim, 3),
            bq: bias(dim, 1),
            wk: tiny_q5k_weight(dim, dim, 17),
            bk: bias(dim, 2),
            wv: tiny_q5k_weight(dim, dim, 31),
            bv: bias(dim, 3),
            wo: tiny_q5k_weight(dim, dim, 47),
            bo: bias(dim, 4),
            attn_out_norm: vec![1.0; dim],
            attn_out_norm_b: bias(dim, 5),
            ffn_gate: Some(tiny_q5k_weight(hidden, dim, 61)),
            ffn_up: tiny_q5k_weight(hidden, dim, 79),
            ffn_up_b: bias(hidden, 6),
            ffn_down: tiny_q5k_weight(dim, hidden, 97),
            ffn_down_b: bias(dim, 7),
            layer_out_norm: vec![1.0; dim],
            layer_out_norm_b: bias(dim, 8),
        };
        let weights = NomicBertWeights {
            token_embd: Weight::F32(fill(vocab * dim, 9)),
            token_type0: Vec::new(),
            tok_norm: vec![1.0; dim],
            tok_norm_b: bias(dim, 10),
            layers: vec![layer],
            ln_eps: 1e-12,
        };
        (config, weights)
    }

    #[test]
    fn nomic_bert_forward_produces_finite_hidden_states() {
        let (config, weights) = tiny_nomic_model();
        let tokens = [2u32, 5, 7, 3];
        let hs = super::forward_nomic_bert_hidden(&config, &weights, &tokens);
        assert_eq!(hs.len(), tokens.len() * config.dim);
        assert!(hs.iter().all(|v| v.is_finite()), "non-finite hidden state");
        // Post-norm output should not be all-zero.
        assert!(hs.iter().any(|&v| v.abs() > 1e-6));
    }

    /// Builds a synthetic Q4_K weight (d=1.0, dmin=0.25, patterned scales and
    /// nibbles) so the batched-prefill parity tests exercise the real K-quant
    /// batch kernels rather than the F32 fallback.
    #[cfg(not(target_family = "wasm"))]
    fn tiny_q4k_weight(rows: usize, cols: usize, seed: u8) -> super::Weight {
        assert_eq!(cols % 256, 0);
        let blocks_per_row = cols / 256;
        let mut data = vec![0u8; rows * blocks_per_row * 144];
        for (block_idx, block) in data.chunks_exact_mut(144).enumerate() {
            block[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0
            block[2..4].copy_from_slice(&0x3400u16.to_le_bytes()); // dmin = 0.25
            for i in 0..12 {
                block[4 + i] = seed.wrapping_add((block_idx * 7 + i * 5) as u8) & 0x3F;
            }
            for i in 0..128 {
                block[16 + i] = seed.wrapping_add((block_idx * 13 + i * 3) as u8);
            }
        }
        super::Weight::Quantized {
            data: super::RawTensorData::Owned(data),
            dtype: crate::gguf::GGMLType::Q4_K,
            rows,
            cols,
        }
    }

    /// Builds a Mamba-2 mixer small enough to reason about exactly.
    ///
    /// `in_proj` is the identity, so the caller drives the gate/x/B/C/dt splits
    /// directly from the hidden vector, and `out_proj` copies the first
    /// `d_inner` channels straight out. That makes the block's internals
    /// observable without exposing any implementation detail.
    #[cfg(not(target_family = "wasm"))]
    fn tiny_mamba2_layer(dims: super::SsmDims) -> super::Mamba2LayerWeights {
        let d_in_proj = dims.d_in_proj();
        let conv_dim = dims.conv_dim();
        let mut identity = vec![0.0f32; d_in_proj * d_in_proj];
        for i in 0..d_in_proj {
            identity[i * d_in_proj + i] = 1.0;
        }
        let mut out = vec![0.0f32; d_in_proj * dims.d_inner];
        for i in 0..dims.d_inner {
            out[i * dims.d_inner + i] = 1.0;
        }
        super::Mamba2LayerWeights {
            in_proj: super::Weight::F32(identity),
            // One-hot on the OLDEST tap turns the convolution into a pure
            // (d_conv - 1)-step delay line, which is checkable by hand.
            conv_w: (0..conv_dim)
                .flat_map(|_| {
                    let mut taps = vec![0.0f32; dims.d_conv];
                    taps[0] = 1.0;
                    taps
                })
                .collect(),
            conv_b: Vec::new(),
            dt_bias: vec![0.0; dims.n_head],
            a: vec![-0.5; dims.n_head],
            d: vec![0.0; dims.n_head],
            norm: Vec::new(),
            out_proj: super::Weight::F32(out),
        }
    }

    #[cfg(not(target_family = "wasm"))]
    fn mamba2_test_dims() -> super::SsmDims {
        super::SsmDims {
            d_conv: 4,
            d_inner: 4,
            d_state: 2,
            n_head: 1,
            n_group: 1,
        }
    }

    /// The depthwise convolution must behave as a causal shift register: after
    /// N tokens its state holds the last `d_conv - 1` inputs, oldest first.
    #[test]
    #[cfg(not(target_family = "wasm"))]
    fn mamba2_conv_state_is_a_causal_shift_register() {
        let dims = mamba2_test_dims();
        let layer = tiny_mamba2_layer(dims);
        let window = dims.d_conv - 1;
        let mut conv = vec![0.0f32; dims.conv_dim() * window];
        let mut ssm = vec![0.0f32; dims.d_inner * dims.d_state];
        let mut scratch = super::Mamba2Scratch::default();
        let mut out = Vec::new();

        // Token t writes value `t + 1` into every convolved channel.
        for step in 1..=5u32 {
            let mut hidden = vec![0.0f32; dims.d_in_proj()];
            for channel in 0..dims.conv_dim() {
                hidden[dims.d_inner + channel] = step as f32;
            }
            super::nemotron_mamba2_step(
                &layer,
                &dims,
                &mut conv,
                &mut ssm,
                &hidden,
                1e-5,
                &mut scratch,
                &mut out,
            );
        }

        // After five tokens the register holds inputs 3, 4 and 5.
        for channel in 0..dims.conv_dim() {
            let slot = &conv[channel * window..channel * window + window];
            assert_eq!(
                slot,
                &[3.0, 4.0, 5.0],
                "conv shift register wrong for channel {}",
                channel
            );
        }
    }

    /// Re-running the same tokens from a cleared state must reproduce the same
    /// outputs — i.e. every mutable thing the mixer touches lives in the state
    /// buffers and nothing leaks between sequences.
    #[test]
    #[cfg(not(target_family = "wasm"))]
    fn mamba2_reset_reproduces_identical_outputs() {
        let dims = mamba2_test_dims();
        let layer = tiny_mamba2_layer(dims);
        let mut state = super::SsmState::new(1, dims);
        let mut scratch = super::Mamba2Scratch::default();

        let inputs: Vec<Vec<f32>> = (1..=4)
            .map(|step| {
                (0..dims.d_in_proj())
                    .map(|i| ((i + step) % 5) as f32 * 0.25 - 0.5)
                    .collect()
            })
            .collect();

        let run = |state: &mut super::SsmState, scratch: &mut super::Mamba2Scratch| {
            let mut collected = Vec::new();
            let mut out = Vec::new();
            for hidden in &inputs {
                super::nemotron_mamba2_step(
                    &layer,
                    &dims,
                    &mut state.conv[0],
                    &mut state.ssm[0],
                    hidden,
                    1e-5,
                    scratch,
                    &mut out,
                );
                collected.push(out.clone());
            }
            collected
        };

        let first = run(&mut state, &mut scratch);
        state.reset();
        let second = run(&mut state, &mut scratch);
        assert_eq!(first, second, "reset did not restore the initial state");

        // The recurrence must actually carry information forward: identical
        // inputs at different positions should not produce identical outputs.
        assert!(
            first.windows(2).any(|pair| pair[0] != pair[1]),
            "outputs are position-independent, so the state is not being used"
        );
        assert!(
            first.iter().flatten().all(|v| v.is_finite()),
            "non-finite Mamba-2 output"
        );
    }

    #[test]
    fn qwen35_delta_step_updates_then_reads_transposed_memory() {
        // State rows are V channels and columns are K channels. This fixture
        // catches both the transpose and the load-bearing update-before-read
        // ordering of the gated delta rule.
        let mut state = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut out = vec![0.0f32; 2];
        super::qwen35_delta_head_step(
            &[3.0, 4.0],
            &[1.0, 2.0],
            &[5.0, 6.0],
            0.5,
            0.25,
            &mut state,
            &mut out,
        );
        let expected_state = [1.125f32, 2.25, 1.625, 2.25];
        let expected_out = [8.750_446f32, 9.811_107];
        for (got, expected) in state.iter().zip(expected_state) {
            assert!((got - expected).abs() < 1e-6, "state {got} != {expected}");
        }
        for (got, expected) in out.iter().zip(expected_out) {
            assert!((got - expected).abs() < 2e-6, "output {got} != {expected}");
        }
    }

    #[test]
    fn qwen35_delta_x4_matches_reference_recurrence() {
        let width = 128usize;
        let q: Vec<f32> = (0..width)
            .map(|i| ((i * 17 % 61) as f32 - 30.0) * 0.009)
            .collect();
        let k: Vec<f32> = (0..width)
            .map(|i| ((i * 29 % 67) as f32 - 33.0) * 0.008)
            .collect();
        let mut state: Vec<f32> = (0..width * width)
            .map(|i| ((i * 13 % 71) as f32 - 35.0) * 0.0007)
            .collect();
        let mut reference = state.clone();
        let mut got = vec![0.0f32; width];
        let mut expected = vec![0.0f32; width];
        let decay = 0.93f32;
        let beta = 0.37f32;

        for step in 0..4 {
            let v: Vec<f32> = (0..width)
                .map(|i| ((i * 7 + step * 11) % 53) as f32 * 0.006 - 0.15)
                .collect();
            super::qwen35_delta_head_step(&q, &k, &v, decay, beta, &mut state, &mut got);

            let q_scale = 1.0 / (width as f32).sqrt();
            for value_row in 0..width {
                let row = &mut reference[value_row * width..(value_row + 1) * width];
                for entry in row.iter_mut() {
                    *entry *= decay;
                }
                let predicted = crate::simd::dot_f32(row, &k);
                let delta = (v[value_row] - predicted) * beta;
                crate::simd::axpy_f32(row, delta, &k);
                expected[value_row] = crate::simd::dot_f32(row, &q) * q_scale;
            }
        }

        for (index, (&actual, &want)) in state.iter().zip(&reference).enumerate() {
            let tolerance = 2e-5 * (1.0 + want.abs());
            assert!(
                (actual - want).abs() <= tolerance,
                "state[{index}]={actual}, reference={want}"
            );
        }
        for (index, (&actual, &want)) in got.iter().zip(&expected).enumerate() {
            let tolerance = 2e-5 * (1.0 + want.abs());
            assert!(
                (actual - want).abs() <= tolerance,
                "out[{index}]={actual}, reference={want}"
            );
        }
    }

    #[test]
    #[ignore = "manual release benchmark; run cargo test --release --lib qwen35_delta_x4_speedup -- --ignored --nocapture"]
    fn qwen35_delta_x4_speedup() {
        use std::hint::black_box;
        use std::time::Instant;

        let width = 128usize;
        let heads = 64usize;
        let rounds = 16usize;
        let q: Vec<f32> = (0..width)
            .map(|i| ((i * 17 % 61) as f32 - 30.0) * 0.009)
            .collect();
        let k: Vec<f32> = (0..width)
            .map(|i| ((i * 29 % 67) as f32 - 33.0) * 0.008)
            .collect();
        let v: Vec<f32> = (0..width)
            .map(|i| ((i * 7 % 53) as f32) * 0.006 - 0.15)
            .collect();
        let initial: Vec<f32> = (0..heads * width * width)
            .map(|i| ((i * 13 % 71) as f32 - 35.0) * 0.0007)
            .collect();
        let mut reference_state = initial.clone();
        let mut split_simd_state = initial.clone();
        let mut optimized_state = initial;
        let mut out = vec![0.0f32; width];
        let decay = 0.93f32;
        let beta = 0.37f32;
        let q_scale = 1.0 / (width as f32).sqrt();

        let started = Instant::now();
        for _ in 0..rounds {
            for head in 0..heads {
                let state = &mut reference_state[head * width * width..(head + 1) * width * width];
                for value_row in 0..width {
                    let row = &mut state[value_row * width..(value_row + 1) * width];
                    for entry in row.iter_mut() {
                        *entry *= decay;
                    }
                    let predicted = crate::simd::dot_f32(row, &k);
                    let delta = (v[value_row] - predicted) * beta;
                    crate::simd::axpy_f32(row, delta, &k);
                    out[value_row] = crate::simd::dot_f32(row, &q) * q_scale;
                }
                black_box(&out);
            }
        }
        let reference_elapsed = started.elapsed();

        let started = Instant::now();
        for _ in 0..rounds {
            for head in 0..heads {
                let state = &mut split_simd_state[head * width * width..(head + 1) * width * width];
                for value_row in (0..width).step_by(4) {
                    let rows = &mut state[value_row * width..(value_row + 4) * width];
                    let (row0, rows) = rows.split_at_mut(width);
                    let (row1, rows) = rows.split_at_mut(width);
                    let (row2, row3) = rows.split_at_mut(width);
                    let mut predicted = crate::simd::dot_f32x4(row0, row1, row2, row3, &k);
                    for score in &mut predicted {
                        *score *= decay;
                    }
                    let delta = [
                        (v[value_row] - predicted[0]) * beta,
                        (v[value_row + 1] - predicted[1]) * beta,
                        (v[value_row + 2] - predicted[2]) * beta,
                        (v[value_row + 3] - predicted[3]) * beta,
                    ];
                    crate::simd::affine_add_f32x4(row0, row1, row2, row3, [decay; 4], delta, &k);
                    let projected = crate::simd::dot_f32x4(row0, row1, row2, row3, &q);
                    for lane in 0..4 {
                        out[value_row + lane] = projected[lane] * q_scale;
                    }
                }
                black_box(&out);
            }
        }
        let split_simd_elapsed = started.elapsed();

        let started = Instant::now();
        for _ in 0..rounds {
            for head in 0..heads {
                let state = &mut optimized_state[head * width * width..(head + 1) * width * width];
                super::qwen35_delta_head_step(&q, &k, &v, decay, beta, state, &mut out);
                black_box(&out);
            }
        }
        let optimized_elapsed = started.elapsed();
        black_box((&reference_state, &split_simd_state, &optimized_state));

        let speedup = reference_elapsed.as_secs_f64() / optimized_elapsed.as_secs_f64();
        let fusion_speedup = split_simd_elapsed.as_secs_f64() / optimized_elapsed.as_secs_f64();
        eprintln!(
            "Qwen35 DeltaNet x4: scalar={:.3} ms, split={:.3} ms, fused={:.3} ms, scalar_speedup={speedup:.2}x, fusion_speedup={fusion_speedup:.2}x",
            reference_elapsed.as_secs_f64() * 1e3,
            split_simd_elapsed.as_secs_f64() * 1e3,
            optimized_elapsed.as_secs_f64() * 1e3,
        );
        assert!(speedup > 1.05, "x4 recurrence unexpectedly regressed");
        assert!(
            fusion_speedup > 1.05,
            "fused recurrence unexpectedly regressed"
        );
    }

    #[test]
    fn qwen35_delta_heads_use_tiled_key_mapping() {
        // GGUF conversion tiles Qwen's value-head-indexed tensors, so h=2
        // maps back to key head 0 rather than to key head 1 (h / 2).
        let q = [2.0f32, 3.0];
        let k = [5.0f32, 7.0];
        let v = [10.0f32, 20.0, 30.0, 40.0];
        let mut state = vec![0.0f32; 4];
        let mut out = vec![0.0f32; 4];
        for value_head in 0..4 {
            let key_head = super::qwen35_key_head_for_value_head(value_head, 2);
            super::qwen35_delta_head_step(
                &q[key_head..key_head + 1],
                &k[key_head..key_head + 1],
                &v[value_head..value_head + 1],
                1.0,
                1.0,
                &mut state[value_head..value_head + 1],
                &mut out[value_head..value_head + 1],
            );
        }
        assert_eq!(out, vec![100.0, 420.0, 300.0, 840.0]);
        assert_eq!(state, vec![50.0, 140.0, 150.0, 280.0]);
    }

    #[test]
    fn qwen35_l2_norm_clamps_the_denominator_not_its_square() {
        let mut values = vec![3e-7f32, 4e-7];
        super::qwen35_l2_normalize_heads(&mut values, 2, 1, 1e-6);
        assert!((values[0] - 0.3).abs() < 1e-6);
        assert!((values[1] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn qwen35_full_attention_q_and_gate_are_interleaved_per_head() {
        let raw = vec![1.0f32, 2.0, 10.0, 20.0, 3.0, 4.0, 30.0, 40.0];
        let mut q = Vec::new();
        let mut gate = Vec::new();
        super::qwen35_split_q_gate(&raw, 2, 2, &mut q, &mut gate);
        assert_eq!(q, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(gate, vec![10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn qwen35_gates_after_per_head_rms_norm() {
        let mut y = vec![3.0f32, 4.0];
        super::rms_norm_heads_in_place(&mut y, 2, 1, Some(&[2.0, 0.5]), 0.0);
        let z = [0.0f32, 1.0];
        for (value, gate) in y.iter_mut().zip(z) {
            *value *= super::silu(gate);
        }
        assert!(y[0].abs() < 1e-6);
        assert!((y[1] - 0.413_549_18).abs() < 1e-6);
    }

    /// Builds a synthetic Q4_K expert stack laid out exactly as GGUF stores
    /// `ffn_*_exps` tensors: expert-major, each expert a full `rows x cols`
    /// matrix. Every expert gets distinct nibbles so routing differences are
    /// observable in the output.
    #[cfg(not(target_family = "wasm"))]
    fn tiny_q4k_expert_weight(
        experts: usize,
        rows: usize,
        cols: usize,
        seed: u8,
    ) -> super::ExpertWeight {
        assert_eq!(cols % 256, 0);
        let blocks_per_row = cols / 256;
        let mut data = vec![0u8; experts * rows * blocks_per_row * 144];
        for (block_idx, block) in data.chunks_exact_mut(144).enumerate() {
            block[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0
            block[2..4].copy_from_slice(&0x3400u16.to_le_bytes()); // dmin = 0.25
            for i in 0..12 {
                block[4 + i] = seed.wrapping_add((block_idx * 7 + i * 5) as u8) & 0x3F;
            }
            for i in 0..128 {
                block[16 + i] = seed.wrapping_add((block_idx * 13 + i * 3) as u8);
            }
        }
        super::ExpertWeight {
            data: super::RawTensorData::Owned(data),
            dtype: crate::gguf::GGMLType::Q4_K,
            experts,
            rows,
            cols,
        }
    }

    /// Verifies the routed feed-forward block against an independently written
    /// reference: softmax over *every* expert logit, top-k, then renormalise.
    ///
    /// `routed_moe_ffn_into` instead takes the top-k raw logits and softmaxes
    /// only those. The two are equivalent, and this test is what holds that
    /// claim honest — a regression to un-normalised weights would show up here.
    #[test]
    #[cfg(not(target_family = "wasm"))]
    fn routed_moe_matches_full_softmax_reference() {
        let dim = 256usize;
        let hidden = 512usize;
        let experts = 4usize;
        let used = 2usize;

        let moe = super::RoutedMoeWeights {
            router: tiny_q4k_weight(experts, dim, 5),
            gate_experts: tiny_q4k_expert_weight(experts, hidden, dim, 17),
            up_experts: tiny_q4k_expert_weight(experts, hidden, dim, 29),
            down_experts: tiny_q4k_expert_weight(experts, dim, hidden, 43),
        };

        let mut config = tiny_standard_model(0).0;
        config.expert_count = experts;
        config.expert_used_count = used;
        let mut buf = super::DecodeBuffer::new(&config, 128, 1, 128);
        for (i, cell) in buf.xn2.iter_mut().enumerate() {
            *cell = ((i % 7) as f32 - 3.0) * 0.05;
        }

        super::routed_moe_ffn_into(&moe, used, &mut buf);
        let got = buf.proj.clone();

        // ── Reference, written independently of the implementation ──
        let mut logits = Vec::new();
        moe.router.matvec_into(&buf.xn2, &mut logits);
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
        let total: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|e| e / total).collect();

        let mut order: Vec<usize> = (0..experts).collect();
        order.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]));
        let chosen = &order[..used];
        let chosen_total: f32 = chosen.iter().map(|&e| probs[e]).sum();

        let mut expected = vec![0.0f32; dim];
        for &expert in chosen {
            let gate = moe.gate_experts.matvec_expert(expert, &buf.xn2);
            let up = moe.up_experts.matvec_expert(expert, &buf.xn2);
            let act: Vec<f32> = gate
                .iter()
                .zip(&up)
                .map(|(&g, &u)| super::silu(g) * u)
                .collect();
            let down = moe.down_experts.matvec_expert(expert, &act);
            let weight = probs[expert] / chosen_total;
            for (slot, value) in expected.iter_mut().zip(&down) {
                *slot += value * weight;
            }
        }

        assert_eq!(got.len(), expected.len());
        let scale = expected.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1.0);
        for (i, (&a, &b)) in got.iter().zip(&expected).enumerate() {
            assert!(
                (a - b).abs() <= 1e-3 * scale,
                "routed MoE output diverged at {}: {} vs {}",
                i,
                a,
                b
            );
        }
        // A routed block that collapsed to zero would pass the comparison above
        // while being useless, so require a real signal.
        assert!(
            got.iter().any(|v| v.abs() > 1e-6),
            "routed MoE output is zero"
        );
    }

    /// Tiny standard-path (LLaMA-style) model with all-Q4_K projections:
    /// dim 256, 2 heads (GQA 2:1), hidden 512, 2 layers, vocab 32.
    #[cfg(not(target_family = "wasm"))]
    fn tiny_standard_model(sliding_window: usize) -> (super::Config, super::ModelWeights) {
        let dim = 256usize;
        let hidden = 512usize;
        let head_dim = 128usize;
        let config = super::Config {
            arch: "mistral3".to_string(),
            dim,
            hidden_dim: hidden,
            n_layers: 2,
            n_heads: 2,
            n_kv_heads: 1,
            vocab_size: 32,
            max_seq_len: 64,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            head_dim,
            kv_dim: head_dim,
            kv_mul: 2,
            value_dim: head_dim,
            sliding_window,
            expert_count: 0,
            expert_used_count: 0,
            rope_scaling_factor: 1.0,
            rope_original_context_length: 0,
        };
        let token_embd = tiny_q4k_weight(config.vocab_size, dim, 3);
        let layers = (0..config.n_layers)
            .map(|l| {
                let s = (l * 31) as u8;
                super::LayerWeights {
                    attn_norm: vec![1.0; dim],
                    wq: tiny_q4k_weight(config.n_heads * head_dim, dim, s.wrapping_add(11)),
                    bq: Vec::new(),
                    wk: tiny_q4k_weight(config.n_kv_heads * head_dim, dim, s.wrapping_add(23)),
                    bk: Vec::new(),
                    wv: tiny_q4k_weight(config.n_kv_heads * head_dim, dim, s.wrapping_add(37)),
                    bv: Vec::new(),
                    attn_q_norm: Vec::new(),
                    attn_k_norm: Vec::new(),
                    wo: tiny_q4k_weight(dim, config.n_heads * head_dim, s.wrapping_add(41)),
                    ffn_norm: vec![1.0; dim],
                    w1: tiny_q4k_weight(hidden, dim, s.wrapping_add(53)),
                    w2: tiny_q4k_weight(dim, hidden, s.wrapping_add(67)),
                    w3: tiny_q4k_weight(hidden, dim, s.wrapping_add(79)),
                    moe: None,
                }
            })
            .collect();
        let weights = super::ModelWeights {
            token_embd: token_embd.clone(),
            output_norm: vec![1.0; dim],
            output: token_embd,
            layers,
        };
        (config, weights)
    }

    /// Small all-Q4_K hybrid model that exercises one DeltaNet and one full
    /// attention block through the real Qwen batched-prefill implementation.
    #[cfg(not(target_family = "wasm"))]
    fn tiny_qwen35_model() -> (super::Config, super::Qwen35Weights) {
        let dim = 256usize;
        let head_dim = 256usize;
        let hidden = 256usize;
        let ssm = super::SsmDims {
            d_conv: 2,
            d_inner: 256,
            d_state: 256,
            n_head: 1,
            n_group: 1,
        };
        let config = super::Config {
            arch: "qwen35".to_string(),
            dim,
            hidden_dim: hidden,
            n_layers: 2,
            n_heads: 1,
            n_kv_heads: 1,
            vocab_size: 32,
            max_seq_len: 16,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            head_dim,
            kv_dim: head_dim,
            kv_mul: 1,
            value_dim: head_dim,
            sliding_window: 0,
            expert_count: 0,
            expert_used_count: 0,
            rope_scaling_factor: 1.0,
            rope_original_context_length: 0,
        };
        let token_embd = tiny_q4k_weight(config.vocab_size, dim, 7);
        let ffn = |seed: u8| {
            (
                tiny_q4k_weight(hidden, dim, seed),
                tiny_q4k_weight(hidden, dim, seed.wrapping_add(11)),
                tiny_q4k_weight(dim, hidden, seed.wrapping_add(23)),
            )
        };
        let (linear_gate, linear_up, linear_down) = ffn(31);
        let linear = super::Qwen35LayerWeights {
            attn_norm: vec![1.0; dim],
            post_attn_norm: vec![1.0; dim],
            mixer: super::Qwen35Mixer::Linear(Box::new(super::Qwen35LinearWeights {
                qkv: tiny_q4k_weight(ssm.conv_dim(), dim, 41),
                gate: tiny_q4k_weight(ssm.d_inner, dim, 43),
                conv_w: (0..ssm.conv_dim())
                    .flat_map(|i| [0.02 + (i % 3) as f32 * 0.005, 0.08])
                    .collect(),
                dt_bias: vec![0.0; ssm.n_head],
                a: vec![-0.01; ssm.n_head],
                beta: tiny_q4k_weight(ssm.n_head, dim, 47),
                alpha: tiny_q4k_weight(ssm.n_head, dim, 53),
                norm: vec![1.0; ssm.head_dim()],
                out: tiny_q4k_weight(dim, ssm.d_inner, 59),
            })),
            ffn_gate: linear_gate,
            ffn_up: linear_up,
            ffn_down: linear_down,
        };
        let (attn_gate, attn_up, attn_down) = ffn(67);
        let attention = super::Qwen35LayerWeights {
            attn_norm: vec![1.0; dim],
            post_attn_norm: vec![1.0; dim],
            mixer: super::Qwen35Mixer::Attention(Box::new(super::Qwen35AttentionWeights {
                q_gate: tiny_q4k_weight(2 * dim, dim, 71),
                k: tiny_q4k_weight(dim, dim, 73),
                v: tiny_q4k_weight(dim, dim, 79),
                q_norm: vec![1.0; head_dim],
                k_norm: vec![1.0; head_dim],
                out: tiny_q4k_weight(dim, dim, 83),
                kv_slot: 0,
            })),
            ffn_gate: attn_gate,
            ffn_up: attn_up,
            ffn_down: attn_down,
        };
        let weights = super::Qwen35Weights {
            token_embd: token_embd.clone(),
            output_norm: vec![1.0; dim],
            output: token_embd,
            layers: vec![linear, attention],
            ssm,
            recurrent_layer_count: 1,
            attn_layer_count: 1,
            rotary_dim: head_dim,
            rope_inv_freq: super::build_rope_inv_freq(10_000.0, head_dim, 1.0),
            mtp: None,
        };
        (config, weights)
    }

    #[cfg(not(target_family = "wasm"))]
    fn tiny_nemotron_h_model() -> (super::Config, super::NemotronHWeights) {
        let dim = 256usize;
        let hidden = 256usize;
        let ssm = super::SsmDims {
            d_conv: 2,
            d_inner: 256,
            d_state: 64,
            n_head: 1,
            n_group: 1,
        };
        let config = super::Config {
            arch: "nemotron_h_moe".to_string(),
            dim,
            hidden_dim: hidden,
            n_layers: 4,
            n_heads: 1,
            n_kv_heads: 1,
            vocab_size: 32,
            max_seq_len: 16,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            head_dim: dim,
            kv_dim: dim,
            kv_mul: 1,
            value_dim: dim,
            sliding_window: 0,
            expert_count: 2,
            expert_used_count: 1,
            rope_scaling_factor: 1.0,
            rope_original_context_length: 0,
        };
        let token_embd = tiny_q4k_weight(config.vocab_size, dim, 9);
        let mamba = super::NemotronHLayerWeights {
            attn_norm: vec![1.0; dim],
            mixer: super::NemotronMixer::Mamba2(Box::new(super::Mamba2LayerWeights {
                in_proj: tiny_q4k_weight(ssm.d_in_proj(), dim, 17),
                conv_w: (0..ssm.conv_dim())
                    .flat_map(|index| [0.01 + (index % 5) as f32 * 0.002, 0.07])
                    .collect(),
                conv_b: vec![0.0; ssm.conv_dim()],
                dt_bias: vec![0.0; ssm.n_head],
                a: vec![-0.02; ssm.n_head],
                d: vec![0.1; ssm.n_head],
                norm: vec![1.0; ssm.d_inner],
                out_proj: tiny_q4k_weight(dim, ssm.d_inner, 23),
            })),
        };
        let attention = super::NemotronHLayerWeights {
            attn_norm: vec![1.0; dim],
            mixer: super::NemotronMixer::Attention(Box::new(super::NemotronAttnWeights {
                wq: tiny_q4k_weight(dim, dim, 31),
                wk: tiny_q4k_weight(dim, dim, 37),
                wv: tiny_q4k_weight(dim, dim, 41),
                wo: tiny_q4k_weight(dim, dim, 43),
                bo: vec![0.0; dim],
                n_heads: 1,
                n_kv_heads: 1,
                kv_slot: 0,
            })),
        };
        let moe = super::NemotronHLayerWeights {
            attn_norm: vec![1.0; dim],
            mixer: super::NemotronMixer::Moe(Box::new(super::NemotronMoeWeights {
                router: tiny_q4k_weight(config.expert_count, dim, 47),
                router_bias: vec![0.0; config.expert_count],
                up_experts: tiny_q4k_expert_weight(config.expert_count, hidden, dim, 53),
                down_experts: tiny_q4k_expert_weight(config.expert_count, dim, hidden, 59),
                shared_up: tiny_q4k_weight(hidden, dim, 61),
                shared_down: tiny_q4k_weight(dim, hidden, 67),
            })),
        };
        let dense = super::NemotronHLayerWeights {
            attn_norm: vec![1.0; dim],
            mixer: super::NemotronMixer::DenseFfn(Box::new(super::NemotronDenseFfnWeights {
                up: tiny_q4k_weight(hidden, dim, 71),
                up_bias: vec![0.0; hidden],
                down: tiny_q4k_weight(dim, hidden, 73),
                down_bias: vec![0.0; dim],
            })),
        };
        let weights = super::NemotronHWeights {
            token_embd: token_embd.clone(),
            output_norm: vec![1.0; dim],
            output: token_embd,
            layers: vec![mamba, attention, moe, dense],
            ssm,
            attn_layer_count: 1,
            router_normalize_weights: true,
            routed_scaling_factor: 1.0,
        };
        (config, weights)
    }

    #[test]
    #[cfg(not(target_family = "wasm"))]
    fn qwen35_batched_prefill_matches_sequential_next_token() {
        let (config, weights) = tiny_qwen35_model();
        assert!(super::qwen35_prefill_batchable(&weights));
        let prompt = [3u32, 8, 13, 21];

        let mut sequential = super::KVCache::with_recurrent_state(
            1,
            config.kv_dim,
            config.kv_dim,
            config.max_seq_len,
            1,
            weights.ssm,
        );
        let mut seq_buf = super::DecodeBuffer::new(&config, config.head_dim, 1, config.value_dim);
        for (pos, &token) in prompt.iter().enumerate() {
            super::forward_prefill_qwen35(
                &config,
                &weights,
                &mut sequential,
                &mut seq_buf,
                token,
                pos,
            );
        }

        let mut batched = super::KVCache::with_recurrent_state(
            1,
            config.kv_dim,
            config.kv_dim,
            config.max_seq_len,
            1,
            weights.ssm,
        );
        let mut batch_buf = super::PrefillBatchBuffer::new(&config);
        assert!(super::forward_prefill_batch_qwen35(
            &config,
            &weights,
            &mut batched,
            &mut batch_buf,
            &prompt,
            0,
        ));

        let mut seq_logits = Vec::new();
        let mut batch_logits = Vec::new();
        let mut next_seq_buf =
            super::DecodeBuffer::new(&config, config.head_dim, 1, config.value_dim);
        let mut next_batch_buf =
            super::DecodeBuffer::new(&config, config.head_dim, 1, config.value_dim);
        super::forward_qwen35_into(
            &config,
            &weights,
            &mut sequential,
            &mut next_seq_buf,
            5,
            prompt.len(),
            &mut seq_logits,
        );
        super::forward_qwen35_into(
            &config,
            &weights,
            &mut batched,
            &mut next_batch_buf,
            5,
            prompt.len(),
            &mut batch_logits,
        );
        assert_eq!(seq_logits.len(), batch_logits.len());
        for (index, (&seq, &batch)) in seq_logits.iter().zip(&batch_logits).enumerate() {
            let tolerance = 2e-3 * seq.abs().max(1.0);
            assert!(
                (seq - batch).abs() <= tolerance,
                "logits[{index}] sequential={seq} batched={batch}"
            );
        }
    }

    #[test]
    #[cfg(not(target_family = "wasm"))]
    fn qwen35_batched_verification_matches_sequential_rows() {
        let (config, weights) = tiny_qwen35_model();
        let draft = [4u32, 9, 17];
        let mut sequential = super::KVCache::with_recurrent_state(
            1,
            config.kv_dim,
            config.kv_dim,
            config.max_seq_len,
            1,
            weights.ssm,
        );
        let mut seq_buf = super::DecodeBuffer::new(&config, config.head_dim, 1, config.value_dim);
        let mut expected_logits = Vec::new();
        let mut row_logits = Vec::new();
        let mut expected_hidden = Vec::new();
        for (pos, &token) in draft.iter().enumerate() {
            super::forward_qwen35_into(
                &config,
                &weights,
                &mut sequential,
                &mut seq_buf,
                token,
                pos,
                &mut row_logits,
            );
            expected_logits.extend_from_slice(&row_logits);
            expected_hidden.extend_from_slice(&seq_buf.x);
        }

        let mut batched = super::KVCache::with_recurrent_state(
            1,
            config.kv_dim,
            config.kv_dim,
            config.max_seq_len,
            1,
            weights.ssm,
        );
        let mut batch_buf = super::PrefillBatchBuffer::new(&config);
        let mut hidden = Vec::new();
        let mut logits = Vec::new();
        assert!(super::forward_verify_batch_qwen35(
            &config,
            &weights,
            &mut batched,
            &mut batch_buf,
            &draft,
            0,
            &mut hidden,
            &mut logits,
        ));
        for (index, (&expected, &actual)) in expected_logits.iter().zip(&logits).enumerate() {
            let tolerance = 2e-3 * expected.abs().max(1.0);
            assert!(
                (expected - actual).abs() <= tolerance,
                "qwen verification logits[{index}] sequential={expected} batched={actual}"
            );
        }
        for (index, (&expected, &actual)) in expected_hidden.iter().zip(&hidden).enumerate() {
            let tolerance = 2e-3 * expected.abs().max(1.0);
            assert!(
                (expected - actual).abs() <= tolerance,
                "qwen verification hidden[{index}] sequential={expected} batched={actual}"
            );
        }
    }

    #[test]
    #[cfg(not(target_family = "wasm"))]
    fn standard_batched_verification_matches_sequential_rows() {
        let (config, weights) = tiny_standard_model(0);
        let draft = [2u32, 11, 19];
        let mut sequential = super::KVCache::new(
            config.n_layers,
            config.kv_dim,
            config.kv_dim,
            config.max_seq_len,
        );
        let mut seq_buf = super::DecodeBuffer::new(
            &config,
            config.head_dim,
            config.n_kv_heads,
            config.value_dim,
        );
        let mut expected_logits = Vec::new();
        let mut expected_hidden = Vec::new();
        let mut row_logits = Vec::new();
        for (pos, &token) in draft.iter().enumerate() {
            super::forward_into(
                &config,
                &weights,
                &mut sequential,
                &mut seq_buf,
                token,
                pos,
                &mut row_logits,
            );
            expected_logits.extend_from_slice(&row_logits);
            expected_hidden.extend_from_slice(&seq_buf.x);
        }

        let mut batched = super::KVCache::new(
            config.n_layers,
            config.kv_dim,
            config.kv_dim,
            config.max_seq_len,
        );
        let mut batch_buf = super::PrefillBatchBuffer::new(&config);
        let mut hidden = Vec::new();
        let mut logits = Vec::new();
        assert!(super::forward_verify_batch(
            &config,
            &weights,
            &mut batched,
            &mut batch_buf,
            &draft,
            0,
            &mut hidden,
            &mut logits,
        ));
        for (index, (&expected, &actual)) in expected_logits.iter().zip(&logits).enumerate() {
            let tolerance = 2e-3 * expected.abs().max(1.0);
            assert!(
                (expected - actual).abs() <= tolerance,
                "standard verification logits[{index}] sequential={expected} batched={actual}"
            );
        }
        for (index, (&expected, &actual)) in expected_hidden.iter().zip(&hidden).enumerate() {
            let tolerance = 2e-3 * expected.abs().max(1.0);
            assert!(
                (expected - actual).abs() <= tolerance,
                "standard verification hidden[{index}] sequential={expected} batched={actual}"
            );
        }
    }

    #[test]
    #[cfg(not(target_family = "wasm"))]
    fn nemotron_h_batched_verification_matches_sequential_rows() {
        let (config, weights) = tiny_nemotron_h_model();
        let draft = [5u32, 12, 27];
        assert!(super::nemotron_h_verify_batchable(&weights));
        let mut sequential = super::KVCache::with_recurrent_state(
            1,
            config.kv_dim,
            config.kv_dim,
            config.max_seq_len,
            1,
            weights.ssm,
        );
        let mut seq_buf = super::DecodeBuffer::new(&config, config.head_dim, 1, config.value_dim);
        let mut expected_logits = Vec::new();
        let mut expected_hidden = Vec::new();
        let mut row_logits = Vec::new();
        for (pos, &token) in draft.iter().enumerate() {
            super::forward_nemotron_h_into(
                &config,
                &weights,
                &mut sequential,
                &mut seq_buf,
                token,
                pos,
                &mut row_logits,
            );
            expected_logits.extend_from_slice(&row_logits);
            expected_hidden.extend_from_slice(&seq_buf.x);
        }

        let mut batched = super::KVCache::with_recurrent_state(
            1,
            config.kv_dim,
            config.kv_dim,
            config.max_seq_len,
            1,
            weights.ssm,
        );
        let mut batch_buf = super::PrefillBatchBuffer::new(&config);
        let mut hidden = Vec::new();
        let mut logits = Vec::new();
        assert!(super::forward_verify_batch_nemotron_h(
            &config,
            &weights,
            &mut batched,
            &mut batch_buf,
            &draft,
            0,
            &mut hidden,
            &mut logits,
        ));
        for (index, (&expected, &actual)) in expected_logits.iter().zip(&logits).enumerate() {
            let tolerance = 3e-3 * expected.abs().max(1.0);
            assert!(
                (expected - actual).abs() <= tolerance,
                "hybrid verification logits[{index}] sequential={expected} batched={actual}"
            );
        }
        for (index, (&expected, &actual)) in expected_hidden.iter().zip(&hidden).enumerate() {
            let tolerance = 3e-3 * expected.abs().max(1.0);
            assert!(
                (expected - actual).abs() <= tolerance,
                "hybrid verification hidden[{index}] sequential={expected} batched={actual}"
            );
        }
    }

    /// The batched prefill must fill the KV cache identically to the
    /// sequential per-token path (same kernels, same order per token).
    #[cfg(not(target_family = "wasm"))]
    fn prefill_batch_parity_case(sliding_window: usize) {
        let (config, weights) = tiny_standard_model(sliding_window);
        assert!(super::standard_prefill_batchable(&weights));
        let tokens: Vec<u32> = (0..17u32).map(|i| (i * 7) % 32).collect();
        let window = (sliding_window > 0).then_some(sliding_window);

        let mut cache_seq = super::KVCache::with_sliding_window(
            config.n_layers,
            config.kv_dim,
            config.kv_dim,
            config.max_seq_len,
            window,
        );
        let mut buf = super::DecodeBuffer::new(&config, config.head_dim, 1, config.value_dim);
        for (pos, &token) in tokens.iter().enumerate() {
            // Sequential CPU reference: forward_hidden_impl directly, so the
            // comparison never routes through the GPU-resident path on
            // Metal-capable machines.
            let _ = super::forward_hidden_impl(
                &config,
                &weights,
                &mut cache_seq,
                &mut buf,
                token,
                pos,
                false,
            );
        }

        let mut cache_batch = super::KVCache::with_sliding_window(
            config.n_layers,
            config.kv_dim,
            config.kv_dim,
            config.max_seq_len,
            window,
        );
        let mut batch_buf = super::PrefillBatchBuffer::new(&config);
        assert!(super::forward_prefill_batch(
            &config,
            &weights,
            &mut cache_batch,
            &mut batch_buf,
            &tokens,
            0,
        ));

        for l in 0..config.n_layers {
            for (i, (a, b)) in cache_seq.k[l]
                .iter()
                .zip(cache_batch.k[l].iter())
                .enumerate()
            {
                let tol = 1e-3f32.max(a.abs() * 1e-3);
                assert!((a - b).abs() <= tol, "k[{l}][{i}] seq {a} batch {b}");
            }
            for (i, (a, b)) in cache_seq.v[l]
                .iter()
                .zip(cache_batch.v[l].iter())
                .enumerate()
            {
                let tol = 1e-3f32.max(a.abs() * 1e-3);
                assert!((a - b).abs() <= tol, "v[{l}][{i}] seq {a} batch {b}");
            }
        }
    }

    #[test]
    #[cfg(not(target_family = "wasm"))]
    fn batched_prefill_matches_sequential() {
        prefill_batch_parity_case(0);
    }

    #[test]
    #[cfg(not(target_family = "wasm"))]
    fn batched_prefill_matches_sequential_with_ring_window() {
        // Window 8 with 17 tokens wraps the ring cache more than once, so
        // this covers the store→attend interleaving the ring layout requires.
        prefill_batch_parity_case(8);
    }

    #[test]
    #[cfg(not(target_family = "wasm"))]
    /// `prefill_batch_parity_case` above only compares KV *cache* contents,
    /// which tiling cannot affect (pass 1, the cache write, is identical
    /// either way — only pass 2, attention, differs) and which are too
    /// small (17 tokens) to cross the `attention_parallel_min_work`
    /// threshold that engages `forward_prefill_batch`'s new KV-block-tiled
    /// path anyway. This forces that path (n_kv_heads=1 here, so work =
    /// 140*141/2 = 9870 >= the 4096 threshold) and checks the actual
    /// per-token *hidden state* output against sequential
    /// `forward_hidden_impl` calls, at 140 tokens so the run also crosses
    /// both a `PREFILL_TOKEN_TILE`=64 tile boundary (3 tiles: 64/64/12) and
    /// a `KV_TILE_BLOCK`=128 KV-block boundary within the later tiles.
    fn batched_prefill_tiled_path_matches_sequential_hidden_states() {
        let (mut config, weights) = tiny_standard_model(0);
        assert!(super::standard_prefill_batchable(&weights));
        config.max_seq_len = 256;
        let tokens: Vec<u32> = (0..140u32).map(|i| (i * 7) % 32).collect();

        let mut cache_seq = super::KVCache::new(
            config.n_layers,
            config.kv_dim,
            config.kv_dim,
            config.max_seq_len,
        );
        let mut seq_buf = super::DecodeBuffer::new(&config, config.head_dim, 1, config.value_dim);
        let mut expected_hidden = Vec::new();
        for (pos, &token) in tokens.iter().enumerate() {
            let _ = super::forward_hidden_impl(
                &config,
                &weights,
                &mut cache_seq,
                &mut seq_buf,
                token,
                pos,
                false,
            );
            expected_hidden.extend_from_slice(&seq_buf.x);
        }

        let mut cache_batch = super::KVCache::new(
            config.n_layers,
            config.kv_dim,
            config.kv_dim,
            config.max_seq_len,
        );
        let mut batch_buf = super::PrefillBatchBuffer::new(&config);
        assert!(super::forward_prefill_batch(
            &config,
            &weights,
            &mut cache_batch,
            &mut batch_buf,
            &tokens,
            0,
        ));

        assert_eq!(expected_hidden.len(), batch_buf.x.len());
        for (index, (&expected, &actual)) in expected_hidden.iter().zip(&batch_buf.x).enumerate() {
            let tolerance = 1e-3 * expected.abs().max(1.0);
            assert!(
                (expected - actual).abs() <= tolerance,
                "hidden[{index}] sequential={expected} tiled_batch={actual}"
            );
        }
    }

    #[test]
    fn nomic_bert_attention_is_bidirectional() {
        // Changing a LATER token must affect an EARLIER token's hidden state —
        // only possible with non-causal (bidirectional) attention.
        let (config, weights) = tiny_nomic_model();
        let a = super::forward_nomic_bert_hidden(&config, &weights, &[2u32, 5, 7, 3]);
        let b = super::forward_nomic_bert_hidden(&config, &weights, &[2u32, 5, 9, 3]);
        let dim = config.dim;
        // Token 0 ([CLS]) hidden state differs because token 2 changed.
        let delta: f32 = (0..dim).map(|j| (a[j] - b[j]).abs()).sum();
        assert!(
            delta > 1e-5,
            "token 0 unchanged ⇒ attention is not bidirectional"
        );
    }

    #[test]
    fn nomic_batched_embeddings_keep_inputs_attention_isolated() {
        let (config, weights) = quantized_nomic_model();
        // The flattened total is at least eight tokens, so this exercises the
        // batch path while ensuring each sequence keeps local RoPE positions
        // and cannot attend to the other sequence.
        let first = [1u32, 2, 3, 4];
        let second = [5u32, 6, 7, 8];
        let pooled = super::forward_nomic_bert_pooled_batch(
            &config,
            &weights,
            &[first.as_slice(), second.as_slice()],
        );
        assert_eq!(pooled.len(), 2);

        for (tokens, embedding) in [first.as_slice(), second.as_slice()]
            .into_iter()
            .zip(&pooled)
        {
            let hidden = super::forward_nomic_bert_hidden(&config, &weights, tokens);
            let mut expected = vec![0.0f32; config.dim];
            for row in hidden.chunks_exact(config.dim) {
                for (value, hidden) in expected.iter_mut().zip(row) {
                    *value += hidden;
                }
            }
            for value in &mut expected {
                *value /= tokens.len() as f32;
            }
            for (index, (actual, expected)) in embedding.iter().zip(&expected).enumerate() {
                assert!(
                    (actual - expected).abs() <= expected.abs().max(1.0) * 2e-5,
                    "embedding[{index}] batched={actual} expected={expected}"
                );
            }
        }
    }

    #[test]
    fn quantized_nomic_batched_forward_matches_per_token_path() {
        let (config, weights) = quantized_nomic_model();
        let tokens = [1u32, 2, 3, 4, 5, 6, 7, 8];

        let serial = super::forward_nomic_bert_hidden_impl(&config, &weights, &tokens, false);
        let batched = super::forward_nomic_bert_hidden(&config, &weights, &tokens);
        assert_eq!(batched.len(), tokens.len() * config.dim);
        assert!(batched.iter().all(|value| value.is_finite()));
        for (index, (batched, serial)) in batched.iter().zip(&serial).enumerate() {
            let tolerance = serial.abs().max(1.0) * 2e-5;
            assert!(
                (batched - serial).abs() <= tolerance,
                "hidden[{index}] batched={batched} serial={serial} tolerance={tolerance}",
            );
        }
    }
}

// ─── Per-token decode scratch buffers (reused across tokens) ─────────────────

/// Pre-allocated working memory for a single forward pass.
/// Eliminates per-token heap allocations in the hot decode loop.
pub struct DecodeBuffer {
    pub x: Vec<f32>,        // residual stream (dim)
    pub xn: Vec<f32>,       // rms-normed residual (dim)
    pub xn2: Vec<f32>,      // second rms norm (dim)
    pub q: Vec<f32>,        // query (n_heads * head_dim)
    pub k: Vec<f32>,        // key   (n_kv_heads * head_dim)
    pub v: Vec<f32>,        // value (n_kv_heads * value_dim)
    pub attn_out: Vec<f32>, // attention output (n_heads * value_dim)
    pub proj: Vec<f32>,     // projection output (dim)
    pub gate: Vec<f32>,     // FFN gate projection (hidden_dim)
    pub up: Vec<f32>,       // FFN up projection (hidden_dim)
    pub hidden: Vec<f32>,   // FFN hidden (hidden_dim)
    pub moe: Vec<f32>,      // MoE residual contribution (dim)
    pub ple_inputs: Vec<f32>,
    pub ple_proj: Vec<f32>,
    pub ple_gate: Vec<f32>,
    pub router_logits: Vec<f32>,
    pub top_experts: Vec<(usize, f32)>,
    pub expert_probs: Vec<f32>,
    pub sampler_candidates: Vec<(usize, f32)>,
    pub rope_inv_freq: Vec<f32>,
    /// Per-position RoPE angles, prepared once before all transformer layers.
    /// Standard decoder blocks share the same RoPE frequencies, so recalculating
    /// `sin_cos` in every layer is pure duplicate work (notably 26 times for
    /// Ministral 3).
    pub rope_sin: Vec<f32>,
    pub rope_cos: Vec<f32>,
    pub rope_gpt_oss_inv_freq: Vec<f32>,
    pub rope_gpt_oss_concentration: f32,
    /// Qwen3.5 Gated DeltaNet `[Q | K | V]` projection / convolution buffer.
    pub qwen35_qkv: Vec<f32>,
    /// Qwen3.5 Gated DeltaNet z gate or full-attention gate.
    pub qwen35_gate: Vec<f32>,
    /// Joint full-attention Q/gate projection before per-head deinterleaving.
    pub qwen35_q_gate: Vec<f32>,
    /// Per-value-head recurrent parameters.
    pub qwen35_alpha: Vec<f32>,
    pub qwen35_beta: Vec<f32>,
}

/// Precomputes inverse frequencies for rotary positional embeddings.
fn build_rope_inv_freq(theta: f32, head_dim: usize, scaling: f32) -> Vec<f32> {
    build_rope_inv_freq_with_factors(theta, head_dim, scaling, None)
}

fn build_rope_inv_freq_with_factors(
    theta: f32,
    head_dim: usize,
    scaling: f32,
    freq_factors: Option<&[f32]>,
) -> Vec<f32> {
    let pair_count = head_dim / 2;
    let mut inv = vec![0.0f32; pair_count];
    for (pair, slot) in inv.iter_mut().enumerate() {
        let i = (pair * 2) as f32;
        let base_freq = theta.powf(i / head_dim as f32);
        let factor = freq_factors
            .and_then(|factors| factors.get(pair))
            .copied()
            .unwrap_or(1.0);
        *slot = if factor == 0.0 {
            0.0
        } else {
            1.0 / (scaling * base_freq * factor)
        };
    }
    inv
}

/// Precomputes GPT-OSS rotary frequencies and attention scaling.
fn build_rope_inv_freq_gpt_oss(config: &Config) -> (Vec<f32>, f32) {
    let d_half = config.head_dim as f32 / 2.0;
    let mut low = 0.0f32;
    let mut high = 0.0f32;
    if config.rope_scaling_factor > 1.0 {
        low = d_half
            * ((config.rope_original_context_length as f32 / (32.0 * 2.0 * std::f32::consts::PI))
                .ln()
                / config.rope_theta.ln());
        high = d_half
            * ((config.rope_original_context_length as f32 / (1.0 * 2.0 * std::f32::consts::PI))
                .ln()
                / config.rope_theta.ln());
    }

    let concentration = if config.rope_scaling_factor > 1.0 {
        0.1 * config.rope_scaling_factor.ln() + 1.0
    } else {
        1.0
    };

    let pair_count = config.head_dim / 2;
    let mut inv = vec![0.0f32; pair_count];
    for (pair, slot) in inv.iter_mut().enumerate() {
        let i = (pair * 2) as f32;
        let base_freq = config.rope_theta.powf(i / config.head_dim as f32);
        *slot = if config.rope_scaling_factor > 1.0 {
            let idx = pair as f32;
            let ramp = ((idx - low) / (high - low)).clamp(0.0, 1.0);
            let mask = 1.0 - ramp;
            let interpolation = 1.0 / (config.rope_scaling_factor * base_freq);
            let extrapolation = 1.0 / base_freq;
            interpolation * (1.0 - mask) + extrapolation * mask
        } else {
            1.0 / base_freq
        };
    }
    (inv, concentration)
}

impl DecodeBuffer {
    /// Allocates all scratch vectors reused by one-token transformer forward passes.
    pub fn new(
        config: &Config,
        max_head_dim: usize,
        max_n_kv_heads: usize,
        max_value_dim: usize,
    ) -> Self {
        let rope_inv_freq = build_rope_inv_freq(config.rope_theta, max_head_dim, 1.0);
        let (rope_gpt_oss_inv_freq, rope_gpt_oss_concentration) =
            build_rope_inv_freq_gpt_oss(config);
        Self {
            x: vec![0.0; config.dim],
            xn: vec![0.0; config.dim],
            xn2: vec![0.0; config.dim],
            q: vec![0.0; config.n_heads * max_head_dim],
            k: vec![0.0; max_n_kv_heads * max_head_dim],
            v: vec![0.0; max_n_kv_heads * max_value_dim],
            attn_out: vec![0.0; config.n_heads * max_value_dim],
            proj: vec![0.0; config.dim],
            gate: vec![0.0; config.hidden_dim],
            up: vec![0.0; config.hidden_dim],
            hidden: vec![0.0; config.hidden_dim],
            moe: vec![0.0; config.dim],
            ple_inputs: Vec::new(),
            ple_proj: Vec::new(),
            ple_gate: Vec::new(),
            router_logits: vec![0.0; config.expert_count],
            top_experts: Vec::with_capacity(config.expert_count.max(config.expert_used_count)),
            expert_probs: Vec::with_capacity(config.expert_used_count),
            sampler_candidates: Vec::with_capacity(64),
            rope_inv_freq,
            rope_sin: vec![0.0; max_head_dim / 2],
            rope_cos: vec![0.0; max_head_dim / 2],
            rope_gpt_oss_inv_freq,
            rope_gpt_oss_concentration,
            qwen35_qkv: Vec::new(),
            qwen35_gate: Vec::new(),
            qwen35_q_gate: Vec::new(),
            qwen35_alpha: Vec::new(),
            qwen35_beta: Vec::new(),
        }
    }
}

// ─── Loading ─────────────────────────────────────────────────────────────────

fn quantized_row_bytes(dtype: GGMLType, cols: usize) -> Option<usize> {
    match dtype {
        GGMLType::Q4_0 => Some(cols.div_ceil(32) * 18),
        GGMLType::Q4_1 => Some(cols.div_ceil(32) * 20),
        GGMLType::Q5_0 => Some(cols.div_ceil(32) * 22),
        GGMLType::Q5_1 => Some(cols.div_ceil(32) * 24),
        GGMLType::Q8_0 => Some(cols.div_ceil(32) * 34),
        GGMLType::Q8_1 => Some(cols.div_ceil(32) * 36),
        GGMLType::Q4_K => Some(cols.div_ceil(256) * 144),
        GGMLType::Q5_K => Some(cols.div_ceil(256) * 176),
        GGMLType::Q6_K => Some(cols.div_ceil(256) * 210),
        GGMLType::MXFP4 => Some(cols.div_ceil(32) * 17),
        _ => None,
    }
}

fn dequantize_row_into(dtype: GGMLType, raw: &[u8], out: &mut [f32]) {
    match dtype {
        GGMLType::Q4_0 => simd::dequant_row_q4_0_into(raw, out),
        GGMLType::Q4_1 => simd::dequant_row_q4_1_into(raw, out),
        GGMLType::Q5_0 => simd::dequant_row_q5_0_into(raw, out),
        GGMLType::Q5_1 => simd::dequant_row_q5_1_into(raw, out),
        GGMLType::Q8_0 => simd::dequant_row_q8_0_into(raw, out),
        GGMLType::Q8_1 => simd::dequant_row_q8_1_into(raw, out),
        GGMLType::Q4_K => simd::dequant_row_q4_k_into(raw, out),
        GGMLType::Q5_K => simd::dequant_row_q5_k_into(raw, out),
        GGMLType::Q6_K => simd::dequant_row_q6_k_into(raw, out),
        GGMLType::MXFP4 => simd::dequant_row_mxfp4_into(raw, out),
        _ => panic!("Unsupported quantized dequantization: {:?}", dtype),
    }
}

fn dequantize_tensor_rows(dtype: GGMLType, raw: &[u8], rows: usize, cols: usize) -> Vec<f32> {
    let row_bytes = quantized_row_bytes(dtype, cols)
        .unwrap_or_else(|| panic!("Unsupported quantized dequantization: {:?}", dtype));
    let mut out = vec![0.0; rows * cols];
    for row in 0..rows {
        let start = row * row_bytes;
        let end = start + row_bytes;
        dequantize_row_into(
            dtype,
            &raw[start..end],
            &mut out[row * cols..(row + 1) * cols],
        );
    }
    out
}

/// Load a tensor as either f32 or quantized raw bytes. If the naive
/// byte-size (based on dtype × numel) would overflow the mmap, we fall back
/// to an inferred size provided in `inferred_sizes` which is computed from
/// neighboring tensor offsets.
fn load_weight(
    mmap_data: &[u8],
    data_offset: usize,
    name: &str,
    tensors: &HashMap<String, &crate::gguf::TensorInfo>,
    inferred_sizes: &HashMap<String, usize>,
    force_f32: bool,
    borrow_quantized: bool,
) -> Weight {
    let info = tensors
        .get(name)
        .unwrap_or_else(|| panic!("Missing tensor: {}", name));
    let numel = info.numel();
    let mut byte_size = info
        .dtype
        .data_size(numel)
        .or_else(|| inferred_sizes.get(name).copied())
        .unwrap_or_else(|| {
            panic!(
                "Unsupported tensor type/size for {}: {:?}",
                name, info.dtype
            )
        });
    let offset = data_offset + info.offset as usize;

    if !offset
        .checked_add(byte_size)
        .map(|end| end <= mmap_data.len())
        .unwrap_or(false)
    {
        if let Some(&inferred) = inferred_sizes.get(name) {
            byte_size = inferred;
        } else {
            panic!(
                "Tensor {}: offset {} + byte_size {} exceeds mmap length {}",
                name,
                offset,
                byte_size,
                mmap_data.len()
            );
        }
    }

    let raw_end = std::cmp::min(offset + byte_size, mmap_data.len());
    let raw_slice = &mmap_data[offset..raw_end];
    // If the available bytes are smaller than our determined byte_size,
    // allow padding for quantized formats (safer than panicking mid-matvec).
    let available = raw_end.saturating_sub(offset);
    let padded;
    let raw_view: &[u8] = if available < byte_size {
        match info.dtype {
            GGMLType::F32 | GGMLType::F16 | GGMLType::BF16 => {
                panic!(
                    "Tensor {}: offset {} + byte_size {} exceeds mmap length {}",
                    name,
                    offset,
                    byte_size,
                    mmap_data.len()
                );
            }
            _ => {
                padded = {
                    let mut v = raw_slice.to_vec();
                    v.resize(byte_size, 0);
                    v
                };
                &padded
            }
        }
    } else {
        raw_slice
    };

    let effective_force_f32 = force_f32;

    match info.dtype {
        GGMLType::F32 => {
            let mut data = vec![0.0f32; numel];
            for i in 0..numel {
                data[i] = f32::from_le_bytes([
                    raw_view[i * 4],
                    raw_view[i * 4 + 1],
                    raw_view[i * 4 + 2],
                    raw_view[i * 4 + 3],
                ]);
            }
            Weight::F32(data)
        }
        GGMLType::F16 if effective_force_f32 => {
            let mut data = vec![0.0f32; numel];
            for i in 0..numel {
                data[i] =
                    simd::f16_to_f32(u16::from_le_bytes([raw_view[i * 2], raw_view[i * 2 + 1]]));
            }
            Weight::F32(data)
        }
        GGMLType::F16 => {
            let mut data = vec![0.0f32; numel];
            for i in 0..numel {
                data[i] =
                    simd::f16_to_f32(u16::from_le_bytes([raw_view[i * 2], raw_view[i * 2 + 1]]));
            }
            Weight::F32(data)
        }
        GGMLType::BF16 => {
            let mut data = vec![0.0f32; numel];
            for i in 0..numel {
                let bits = u16::from_le_bytes([raw_view[i * 2], raw_view[i * 2 + 1]]);
                data[i] = f32::from_bits((bits as u32) << 16);
            }
            Weight::F32(data)
        }
        GGMLType::Q8_0
        | GGMLType::Q4_0
        | GGMLType::Q4_K
        | GGMLType::Q5_K
        | GGMLType::Q6_K
        | GGMLType::MXFP4
        | GGMLType::Q8_1
        | GGMLType::Q4_1
        | GGMLType::Q5_0
        | GGMLType::Q5_1 => {
            if effective_force_f32 {
                let rows = if info.dims.len() >= 2 {
                    info.dims[1..].iter().map(|d| *d as usize).product()
                } else {
                    1
                };
                let cols = info.dims[0] as usize;
                let data_f = dequantize_tensor_rows(info.dtype, raw_view, rows, cols);
                Weight::F32(data_f)
            } else {
                // Keep quantized — use fused SIMD dot products
                let rows = if info.dims.len() >= 2 {
                    info.dims[1] as usize
                } else {
                    1
                };
                let cols = info.dims[0] as usize;
                Weight::Quantized {
                    data: if borrow_quantized && available >= byte_size {
                        RawTensorData::view(raw_slice)
                    } else {
                        RawTensorData::owned(raw_view)
                    },
                    dtype: info.dtype,
                    rows,
                    cols,
                }
            }
        }
        _ => panic!("Unsupported tensor type for {}: {:?}", name, info.dtype),
    }
}

fn load_weight_rows(
    mmap_data: &[u8],
    data_offset: usize,
    name: &str,
    tensors: &HashMap<String, &crate::gguf::TensorInfo>,
    inferred_sizes: &HashMap<String, usize>,
    start_row: usize,
    rows: usize,
    cols: usize,
    borrow_quantized: bool,
) -> Weight {
    let info = tensors
        .get(name)
        .unwrap_or_else(|| panic!("Missing tensor: {}", name));
    if info.dims.len() < 2 || info.dims[0] as usize != cols {
        panic!(
            "Tensor {} cannot be row-split as {} columns; dims={:?}",
            name, cols, info.dims
        );
    }
    let total_rows: usize = info.dims[1..].iter().map(|d| *d as usize).product();
    let end_row = start_row
        .checked_add(rows)
        .unwrap_or_else(|| panic!("Tensor {} row slice overflows usize", name));
    if end_row > total_rows {
        panic!(
            "Tensor {} row slice {}..{} exceeds {} rows",
            name, start_row, end_row, total_rows
        );
    }

    match info.dtype {
        GGMLType::F32 => {
            let offset = data_offset + info.offset as usize + start_row * cols * 4;
            let byte_size = rows * cols * 4;
            let raw = &mmap_data[offset..offset + byte_size];
            let mut data = vec![0.0f32; rows * cols];
            for i in 0..data.len() {
                data[i] = f32::from_le_bytes([
                    raw[i * 4],
                    raw[i * 4 + 1],
                    raw[i * 4 + 2],
                    raw[i * 4 + 3],
                ]);
            }
            Weight::F32(data)
        }
        GGMLType::F16 => {
            let offset = data_offset + info.offset as usize + start_row * cols * 2;
            let byte_size = rows * cols * 2;
            let raw = &mmap_data[offset..offset + byte_size];
            let mut data = vec![0.0f32; rows * cols];
            for i in 0..data.len() {
                data[i] = simd::f16_to_f32(u16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]]));
            }
            Weight::F32(data)
        }
        GGMLType::BF16 => {
            let offset = data_offset + info.offset as usize + start_row * cols * 2;
            let byte_size = rows * cols * 2;
            let raw = &mmap_data[offset..offset + byte_size];
            let mut data = vec![0.0f32; rows * cols];
            for i in 0..data.len() {
                let bits = u16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]]);
                data[i] = f32::from_bits((bits as u32) << 16);
            }
            Weight::F32(data)
        }
        dtype => {
            let row_bytes = quantized_row_bytes(dtype, cols)
                .unwrap_or_else(|| panic!("Unsupported tensor type for {}: {:?}", name, dtype));
            let offset = data_offset + info.offset as usize + start_row * row_bytes;
            let byte_size = rows * row_bytes;
            let raw_end = offset + byte_size;
            if raw_end > mmap_data.len() {
                let inferred = inferred_sizes.get(name).copied().unwrap_or(0);
                panic!(
                    "Tensor {} row slice exceeds mmap length (offset {}, byte_size {}, inferred full {})",
                    name, offset, byte_size, inferred
                );
            }
            let raw = &mmap_data[offset..raw_end];
            Weight::Quantized {
                data: if borrow_quantized {
                    RawTensorData::view(raw)
                } else {
                    RawTensorData::owned(raw)
                },
                dtype,
                rows,
                cols,
            }
        }
    }
}

/// Load norm weight always as f32 (small, needs exact values)
fn load_f32_vec(
    mmap_data: &[u8],
    data_offset: usize,
    name: &str,
    tensors: &HashMap<String, &crate::gguf::TensorInfo>,
    inferred_sizes: &HashMap<String, usize>,
) -> Vec<f32> {
    match load_weight(
        mmap_data,
        data_offset,
        name,
        tensors,
        inferred_sizes,
        true,
        false,
    ) {
        Weight::F32(v) => v,
        _ => panic!("Expected f32 for {}", name),
    }
}

/// Loads an optional one-dimensional float tensor when present.
fn load_optional_f32_vec(
    mmap_data: &[u8],
    data_offset: usize,
    name: &str,
    tensors: &HashMap<String, &crate::gguf::TensorInfo>,
    inferred_sizes: &HashMap<String, usize>,
    _len: usize,
) -> Vec<f32> {
    if tensors.contains_key(name) {
        load_f32_vec(mmap_data, data_offset, name, tensors, inferred_sizes)
    } else {
        Vec::new()
    }
}

fn load_optional_f32_slice(
    mmap_data: &[u8],
    data_offset: usize,
    name: &str,
    tensors: &HashMap<String, &crate::gguf::TensorInfo>,
    inferred_sizes: &HashMap<String, usize>,
    start: usize,
    len: usize,
) -> Vec<f32> {
    if tensors.contains_key(name) {
        let values = load_f32_vec(mmap_data, data_offset, name, tensors, inferred_sizes);
        values[start..start + len].to_vec()
    } else {
        Vec::new()
    }
}

fn validate_global_shape(name: &str, w: &Weight, exp_rows: usize, exp_cols: usize) {
    match w {
        Weight::F32(v) => {
            let expected = exp_rows.checked_mul(exp_cols).unwrap_or(0);
            if v.len() != expected {
                panic!(
                    "Shape mismatch for {}: f32 elements {} != expected {} ({}x{})",
                    name,
                    v.len(),
                    expected,
                    exp_rows,
                    exp_cols
                );
            }
        }
        Weight::Quantized { rows, cols, .. } => {
            if *rows != exp_rows || *cols != exp_cols {
                panic!(
                    "Shape mismatch for {}: quantized shape {}x{} != expected {}x{}",
                    name, rows, cols, exp_rows, exp_cols
                );
            }
        }
    }
}

/// Loads a mixture-of-experts tensor using the naming variants used by GGUF models.
fn load_expert_weight(
    mmap_data: &[u8],
    data_offset: usize,
    name: &str,
    tensors: &HashMap<String, &crate::gguf::TensorInfo>,
    inferred_sizes: &HashMap<String, usize>,
    borrow_quantized: bool,
) -> ExpertWeight {
    let info = tensors
        .get(name)
        .unwrap_or_else(|| panic!("Missing tensor: {}", name));
    assert!(
        info.dims.len() == 3,
        "Expected 3D expert tensor for {}",
        name
    );
    let numel = info.numel();
    let byte_size = info
        .dtype
        .data_size(numel)
        .or_else(|| inferred_sizes.get(name).copied())
        .unwrap_or_else(|| {
            panic!(
                "Unsupported expert tensor type/size for {}: {:?}",
                name, info.dtype
            )
        });
    let offset = data_offset + info.offset as usize;
    let raw = &mmap_data[offset..offset + byte_size];
    ExpertWeight {
        data: if borrow_quantized {
            RawTensorData::view(raw)
        } else {
            RawTensorData::owned(raw)
        },
        dtype: info.dtype,
        experts: info.dims[2] as usize,
        rows: info.dims[1] as usize,
        cols: info.dims[0] as usize,
    }
}

/// Loads standard transformer weights from a parsed GGUF file.
pub fn load_model(
    mmap_data: &[u8],
    gguf: &GGUFFile,
    borrow_quantized: bool,
) -> (Config, ModelWeights) {
    let mut config = Config::from_gguf(gguf);
    eprintln!(
        "Config: dim={}, layers={}, heads={}/{}, hidden={}, vocab={}, ctx={}",
        config.dim,
        config.n_layers,
        config.n_heads,
        config.n_kv_heads,
        config.hidden_dim,
        config.vocab_size,
        config.max_seq_len
    );

    // Index tensors by name
    let tensor_idx: HashMap<String, &crate::gguf::TensorInfo> =
        gguf.tensors.iter().map(|t| (t.name.clone(), t)).collect();

    let data_offset = gguf.data_offset;

    // Calculate expected end of tensor data by inferring each tensor's byte
    // size from the distance to the next tensor offset. This is robust for
    // block-packed or custom quant layouts where a simple bytes-per-element
    // formula may be incorrect. Offsets in GGUF are relative to `data_offset`.
    let mut max_required_end: usize = 0;
    let mut inferred_sizes: HashMap<String, usize> = HashMap::new();
    if !gguf.tensors.is_empty() {
        let mmap_len = mmap_data.len();
        // Build sorted list of (offset, idx)
        let mut offs: Vec<(u64, usize)> = gguf
            .tensors
            .iter()
            .enumerate()
            .map(|(i, t)| (t.offset, i))
            .collect();
        offs.sort_unstable_by_key(|o| o.0);

        for w in 0..offs.len() {
            let (off, idx) = offs[w];
            let next_off = if w + 1 < offs.len() {
                offs[w + 1].0
            } else {
                (mmap_len as u64).saturating_sub(data_offset as u64)
            };
            let byte_size = if next_off > off {
                (next_off - off) as usize
            } else {
                0
            };
            // Some quantized layouts do not match a simple dtype*numel formula,
            // so neighboring offsets are the most reliable fallback.
            let name = &gguf.tensors[idx].name;
            inferred_sizes.insert(name.clone(), byte_size);
            let end = data_offset + off as usize + byte_size;
            if end > max_required_end {
                max_required_end = end;
            }
        }
    }
    // Embeddings can be quantized; keep native format and dequantize selected rows on demand.
    let token_embd = load_weight(
        mmap_data,
        data_offset,
        "token_embd.weight",
        &tensor_idx,
        &inferred_sizes,
        false,
        borrow_quantized,
    );
    let output_norm = load_f32_vec(
        mmap_data,
        data_offset,
        "output_norm.weight",
        &tensor_idx,
        &inferred_sizes,
    );

    // Output projection (may be tied)
    let output = if tensor_idx.contains_key("output.weight") {
        load_weight(
            mmap_data,
            data_offset,
            "output.weight",
            &tensor_idx,
            &inferred_sizes,
            false,
            borrow_quantized,
        )
    } else {
        eprintln!("Note: output tied to embeddings");
        token_embd.clone()
    };
    // Infer attention head/value dimensions from tensor shapes when GGUF
    // metadata appears inconsistent. We examine available blk.* attn_q/attn_v
    // tensors and prefer derived shapes over possibly-misleading metadata.
    {
        let mut head_dim_cand: Option<usize> = None;
        let mut value_dim_cand: Option<usize> = None;
        for l in 0..config.n_layers {
            let qn = format!("blk.{}.attn_q.weight", l);
            if let Some(info) = tensor_idx.get(&qn) {
                if info.dims.len() >= 2 {
                    let rows = info.dims[1] as usize;
                    let cols = info.dims[0] as usize;
                    if cols == config.dim && rows % config.n_heads == 0 {
                        head_dim_cand = Some(rows / config.n_heads);
                    }
                }
            }
            let vn = format!("blk.{}.attn_v.weight", l);
            if let Some(info) = tensor_idx.get(&vn) {
                if info.dims.len() >= 2 {
                    let rows = info.dims[1] as usize;
                    let cols = info.dims[0] as usize;
                    if cols == config.dim && rows % config.n_kv_heads == 0 {
                        value_dim_cand = Some(rows / config.n_kv_heads);
                    }
                }
            }
            if head_dim_cand.is_some() && value_dim_cand.is_some() {
                break;
            }
        }
        if let Some(hd) = head_dim_cand {
            if hd != config.head_dim {
                eprintln!(
                    "[INFO] Overriding config.head_dim {} -> {} based on attn_q tensor shapes",
                    config.head_dim, hd
                );
                config.head_dim = hd;
            }
        }
        if let Some(vd) = value_dim_cand {
            if vd != config.value_dim {
                eprintln!(
                    "[INFO] Overriding config.value_dim {} -> {} based on attn_v tensor shapes",
                    config.value_dim, vd
                );
                config.value_dim = vd;
            }
        }
        // Recompute derived kv sizes
        config.kv_dim = config.value_dim * config.n_kv_heads;
        config.kv_mul = config.n_heads / config.n_kv_heads;
        eprintln!(
            "Adjusted config: head_dim={}, value_dim={}, kv_dim={}, kv_mul={}",
            config.head_dim, config.value_dim, config.kv_dim, config.kv_mul
        );
    }

    // Layers
    let mut layers = Vec::with_capacity(config.n_layers);
    let q_rows = config.n_heads * config.head_dim;
    let k_rows = config.n_kv_heads * config.head_dim;
    let v_rows = config.n_kv_heads * config.value_dim;
    for l in 0..config.n_layers {
        let q_name = format!("blk.{}.attn_q.weight", l);
        let k_name = format!("blk.{}.attn_k.weight", l);
        let v_name = format!("blk.{}.attn_v.weight", l);
        let qkv_name = format!("blk.{}.attn_qkv.weight", l);
        let q_bias_name = format!("blk.{}.attn_q.bias", l);
        let k_bias_name = format!("blk.{}.attn_k.bias", l);
        let v_bias_name = format!("blk.{}.attn_v.bias", l);
        let qkv_bias_name = format!("blk.{}.attn_qkv.bias", l);

        let (wq, bq, wk, bk, wv, bv) = if tensor_idx.contains_key(&q_name) {
            (
                load_weight(
                    mmap_data,
                    data_offset,
                    &q_name,
                    &tensor_idx,
                    &inferred_sizes,
                    false,
                    borrow_quantized,
                ),
                load_optional_f32_vec(
                    mmap_data,
                    data_offset,
                    &q_bias_name,
                    &tensor_idx,
                    &inferred_sizes,
                    q_rows,
                ),
                load_weight(
                    mmap_data,
                    data_offset,
                    &k_name,
                    &tensor_idx,
                    &inferred_sizes,
                    false,
                    borrow_quantized,
                ),
                load_optional_f32_vec(
                    mmap_data,
                    data_offset,
                    &k_bias_name,
                    &tensor_idx,
                    &inferred_sizes,
                    k_rows,
                ),
                load_weight(
                    mmap_data,
                    data_offset,
                    &v_name,
                    &tensor_idx,
                    &inferred_sizes,
                    false,
                    borrow_quantized,
                ),
                load_optional_f32_vec(
                    mmap_data,
                    data_offset,
                    &v_bias_name,
                    &tensor_idx,
                    &inferred_sizes,
                    v_rows,
                ),
            )
        } else if tensor_idx.contains_key(&qkv_name) {
            (
                load_weight_rows(
                    mmap_data,
                    data_offset,
                    &qkv_name,
                    &tensor_idx,
                    &inferred_sizes,
                    0,
                    q_rows,
                    config.dim,
                    borrow_quantized,
                ),
                load_optional_f32_slice(
                    mmap_data,
                    data_offset,
                    &qkv_bias_name,
                    &tensor_idx,
                    &inferred_sizes,
                    0,
                    q_rows,
                ),
                load_weight_rows(
                    mmap_data,
                    data_offset,
                    &qkv_name,
                    &tensor_idx,
                    &inferred_sizes,
                    q_rows,
                    k_rows,
                    config.dim,
                    borrow_quantized,
                ),
                load_optional_f32_slice(
                    mmap_data,
                    data_offset,
                    &qkv_bias_name,
                    &tensor_idx,
                    &inferred_sizes,
                    q_rows,
                    k_rows,
                ),
                load_weight_rows(
                    mmap_data,
                    data_offset,
                    &qkv_name,
                    &tensor_idx,
                    &inferred_sizes,
                    q_rows + k_rows,
                    v_rows,
                    config.dim,
                    borrow_quantized,
                ),
                load_optional_f32_slice(
                    mmap_data,
                    data_offset,
                    &qkv_bias_name,
                    &tensor_idx,
                    &inferred_sizes,
                    q_rows + k_rows,
                    v_rows,
                ),
            )
        } else {
            panic!("Missing tensor: {} (or {})", q_name, qkv_name);
        };

        // Mixtral-style routed experts replace the dense gate/up/down trio.
        // Detect them before the dense path so a MoE GGUF does not trip the
        // "missing ffn_gate" panic below.
        let router_name = format!("blk.{}.ffn_gate_inp.weight", l);
        let moe = if tensor_idx.contains_key(&router_name) {
            Some(Box::new(RoutedMoeWeights {
                router: load_weight(
                    mmap_data,
                    data_offset,
                    &router_name,
                    &tensor_idx,
                    &inferred_sizes,
                    false,
                    borrow_quantized,
                ),
                gate_experts: load_expert_weight(
                    mmap_data,
                    data_offset,
                    &format!("blk.{}.ffn_gate_exps.weight", l),
                    &tensor_idx,
                    &inferred_sizes,
                    borrow_quantized,
                ),
                up_experts: load_expert_weight(
                    mmap_data,
                    data_offset,
                    &format!("blk.{}.ffn_up_exps.weight", l),
                    &tensor_idx,
                    &inferred_sizes,
                    borrow_quantized,
                ),
                down_experts: load_expert_weight(
                    mmap_data,
                    data_offset,
                    &format!("blk.{}.ffn_down_exps.weight", l),
                    &tensor_idx,
                    &inferred_sizes,
                    borrow_quantized,
                ),
            }))
        } else {
            None
        };

        let gate_name = format!("blk.{}.ffn_gate.weight", l);
        let up_name = format!("blk.{}.ffn_up.weight", l);
        let (w1, w3) = if moe.is_some() {
            // Routed layers have no dense projections; the placeholders are
            // never read because every FFN site checks `moe` first.
            (Weight::F32(Vec::new()), Weight::F32(Vec::new()))
        } else if tensor_idx.contains_key(&gate_name) {
            (
                load_weight(
                    mmap_data,
                    data_offset,
                    &gate_name,
                    &tensor_idx,
                    &inferred_sizes,
                    false,
                    borrow_quantized,
                ),
                load_weight(
                    mmap_data,
                    data_offset,
                    &up_name,
                    &tensor_idx,
                    &inferred_sizes,
                    false,
                    borrow_quantized,
                ),
            )
        } else {
            let info = tensor_idx
                .get(&up_name)
                .unwrap_or_else(|| panic!("Missing tensor: {} (or {})", gate_name, up_name));
            let up_rows = info.dims.get(1).copied().unwrap_or(0) as usize;
            if up_rows < config.hidden_dim * 2 {
                panic!(
                    "Missing tensor: {} and {} is not a fused gate/up projection",
                    gate_name, up_name
                );
            }
            (
                load_weight_rows(
                    mmap_data,
                    data_offset,
                    &up_name,
                    &tensor_idx,
                    &inferred_sizes,
                    0,
                    config.hidden_dim,
                    config.dim,
                    borrow_quantized,
                ),
                load_weight_rows(
                    mmap_data,
                    data_offset,
                    &up_name,
                    &tensor_idx,
                    &inferred_sizes,
                    config.hidden_dim,
                    config.hidden_dim,
                    config.dim,
                    borrow_quantized,
                ),
            )
        };

        let w2 = if moe.is_some() {
            Weight::F32(Vec::new())
        } else {
            load_weight(
                mmap_data,
                data_offset,
                &format!("blk.{}.ffn_down.weight", l),
                &tensor_idx,
                &inferred_sizes,
                false,
                borrow_quantized,
            )
        };

        let layer = LayerWeights {
            attn_norm: load_f32_vec(
                mmap_data,
                data_offset,
                &format!("blk.{}.attn_norm.weight", l),
                &tensor_idx,
                &inferred_sizes,
            ),
            wq,
            bq,
            wk,
            bk,
            wv,
            bv,
            attn_q_norm: load_optional_f32_vec(
                mmap_data,
                data_offset,
                &format!("blk.{}.attn_q_norm.weight", l),
                &tensor_idx,
                &inferred_sizes,
                config.head_dim,
            ),
            attn_k_norm: load_optional_f32_vec(
                mmap_data,
                data_offset,
                &format!("blk.{}.attn_k_norm.weight", l),
                &tensor_idx,
                &inferred_sizes,
                config.head_dim,
            ),
            wo: load_weight(
                mmap_data,
                data_offset,
                &format!("blk.{}.attn_output.weight", l),
                &tensor_idx,
                &inferred_sizes,
                false,
                borrow_quantized,
            ),
            ffn_norm: load_f32_vec(
                mmap_data,
                data_offset,
                &format!("blk.{}.ffn_norm.weight", l),
                &tensor_idx,
                &inferred_sizes,
            ),
            w1,
            w2,
            w3,
            moe,
        };
        layers.push(layer);
        if l == 0 || (l + 1) % 8 == 0 || l + 1 == config.n_layers {
            eprintln!("  Loaded layer {}/{}", l + 1, config.n_layers);
        }
    }

    let weights = ModelWeights {
        token_embd,
        output_norm,
        output,
        layers,
    };
    (config, weights)
}

/// Loads Poolside Laguna's alternating dense/sparse-MoE decoder layout.
pub fn load_laguna_model(
    mmap_data: &[u8],
    gguf: &GGUFFile,
    borrow_quantized: bool,
) -> (Config, LagunaWeights) {
    let config = Config::from_gguf(gguf);
    eprintln!(
        "Config: dim={}, layers={}, max-heads={}/{}, hidden={}, experts={}/{}, vocab={}, ctx={}",
        config.dim,
        config.n_layers,
        config.n_heads,
        config.n_kv_heads,
        config.hidden_dim,
        config.expert_used_count,
        config.expert_count,
        config.vocab_size,
        config.max_seq_len
    );
    let tensor_idx: HashMap<String, &crate::gguf::TensorInfo> = gguf
        .tensors
        .iter()
        .map(|tensor| (tensor.name.clone(), tensor))
        .collect();
    let inferred_sizes = HashMap::new();
    let data_offset = gguf.data_offset;
    let token_embd = load_weight(
        mmap_data,
        data_offset,
        "token_embd.weight",
        &tensor_idx,
        &inferred_sizes,
        false,
        borrow_quantized,
    );
    let output_norm = load_f32_vec(
        mmap_data,
        data_offset,
        "output_norm.weight",
        &tensor_idx,
        &inferred_sizes,
    );
    let output = load_weight(
        mmap_data,
        data_offset,
        "output.weight",
        &tensor_idx,
        &inferred_sizes,
        false,
        borrow_quantized,
    );
    let rope_dim = gguf.get_u32("laguna.rope.dimension_count", config.head_dim as u32) as usize;
    let swa_rope_dim =
        gguf.get_u32("laguna.rope.dimension_count_swa", config.head_dim as u32) as usize;
    let rope_theta = gguf.get_f32("laguna.rope.freq_base", config.rope_theta);
    let swa_rope_theta = gguf.get_f32("laguna.rope.freq_base_swa", rope_theta);
    // Laguna's base head count is used by full-attention layers. The larger
    // per-layer count identifies SWA layers (and therefore selects Laguna's
    // separate 128-dim, theta=10_000 RoPE configuration).
    let full_attention_heads = gguf
        .metadata
        .get("laguna.attention.head_count")
        .and_then(crate::gguf::MetaValue::as_u32_array)
        .and_then(|counts| counts.into_iter().min())
        .map(|count| count as usize)
        .unwrap_or(config.n_heads);
    let router_normalize_weights = match gguf.metadata.get("laguna.expert_weights_norm") {
        Some(crate::gguf::MetaValue::Bool(value)) => *value,
        Some(value) => value.as_u32().map(|value| value != 0).unwrap_or(true),
        None => true,
    };
    let routed_scaling_factor = gguf.get_f32("laguna.expert_weights_scale", 1.0);

    let mut layers = Vec::with_capacity(config.n_layers);
    for l in 0..config.n_layers {
        let q_name = format!("blk.{l}.attn_q.weight");
        let q_info = tensor_idx
            .get(&q_name)
            .unwrap_or_else(|| panic!("Missing tensor: {q_name}"));
        let n_heads = q_info.dims[1] as usize / config.head_dim;
        let sliding_window = n_heads > full_attention_heads;
        let rotary_dim = if sliding_window {
            swa_rope_dim
        } else {
            rope_dim
        };
        let layer_rope_theta = if sliding_window {
            swa_rope_theta
        } else {
            rope_theta
        };

        let mlp = if tensor_idx.contains_key(&format!("blk.{l}.ffn_gate.weight")) {
            LagunaMlpWeights::Dense {
                gate: load_weight(
                    mmap_data,
                    data_offset,
                    &format!("blk.{l}.ffn_gate.weight"),
                    &tensor_idx,
                    &inferred_sizes,
                    false,
                    borrow_quantized,
                ),
                up: load_weight(
                    mmap_data,
                    data_offset,
                    &format!("blk.{l}.ffn_up.weight"),
                    &tensor_idx,
                    &inferred_sizes,
                    false,
                    borrow_quantized,
                ),
                down: load_weight(
                    mmap_data,
                    data_offset,
                    &format!("blk.{l}.ffn_down.weight"),
                    &tensor_idx,
                    &inferred_sizes,
                    false,
                    borrow_quantized,
                ),
            }
        } else {
            LagunaMlpWeights::Sparse(Box::new(LagunaSparseMlpWeights {
                router: load_weight(
                    mmap_data,
                    data_offset,
                    &format!("blk.{l}.ffn_gate_inp.weight"),
                    &tensor_idx,
                    &inferred_sizes,
                    false,
                    borrow_quantized,
                ),
                router_bias: load_f32_vec(
                    mmap_data,
                    data_offset,
                    &format!("blk.{l}.exp_probs_b.bias"),
                    &tensor_idx,
                    &inferred_sizes,
                ),
                gate_experts: load_expert_weight(
                    mmap_data,
                    data_offset,
                    &format!("blk.{l}.ffn_gate_exps.weight"),
                    &tensor_idx,
                    &inferred_sizes,
                    borrow_quantized,
                ),
                up_experts: load_expert_weight(
                    mmap_data,
                    data_offset,
                    &format!("blk.{l}.ffn_up_exps.weight"),
                    &tensor_idx,
                    &inferred_sizes,
                    borrow_quantized,
                ),
                down_experts: load_expert_weight(
                    mmap_data,
                    data_offset,
                    &format!("blk.{l}.ffn_down_exps.weight"),
                    &tensor_idx,
                    &inferred_sizes,
                    borrow_quantized,
                ),
                shared_gate: load_weight(
                    mmap_data,
                    data_offset,
                    &format!("blk.{l}.ffn_gate_shexp.weight"),
                    &tensor_idx,
                    &inferred_sizes,
                    false,
                    borrow_quantized,
                ),
                shared_up: load_weight(
                    mmap_data,
                    data_offset,
                    &format!("blk.{l}.ffn_up_shexp.weight"),
                    &tensor_idx,
                    &inferred_sizes,
                    false,
                    borrow_quantized,
                ),
                shared_down: load_weight(
                    mmap_data,
                    data_offset,
                    &format!("blk.{l}.ffn_down_shexp.weight"),
                    &tensor_idx,
                    &inferred_sizes,
                    false,
                    borrow_quantized,
                ),
            }))
        };
        layers.push(LagunaLayerWeights {
            attn_norm: load_f32_vec(
                mmap_data,
                data_offset,
                &format!("blk.{l}.attn_norm.weight"),
                &tensor_idx,
                &inferred_sizes,
            ),
            wq: load_weight(
                mmap_data,
                data_offset,
                &q_name,
                &tensor_idx,
                &inferred_sizes,
                false,
                borrow_quantized,
            ),
            wk: load_weight(
                mmap_data,
                data_offset,
                &format!("blk.{l}.attn_k.weight"),
                &tensor_idx,
                &inferred_sizes,
                false,
                borrow_quantized,
            ),
            wv: load_weight(
                mmap_data,
                data_offset,
                &format!("blk.{l}.attn_v.weight"),
                &tensor_idx,
                &inferred_sizes,
                false,
                borrow_quantized,
            ),
            q_norm: load_f32_vec(
                mmap_data,
                data_offset,
                &format!("blk.{l}.attn_q_norm.weight"),
                &tensor_idx,
                &inferred_sizes,
            ),
            k_norm: load_f32_vec(
                mmap_data,
                data_offset,
                &format!("blk.{l}.attn_k_norm.weight"),
                &tensor_idx,
                &inferred_sizes,
            ),
            attn_gate: load_weight(
                mmap_data,
                data_offset,
                &format!("blk.{l}.attn_gate.weight"),
                &tensor_idx,
                &inferred_sizes,
                false,
                borrow_quantized,
            ),
            wo: load_weight(
                mmap_data,
                data_offset,
                &format!("blk.{l}.attn_output.weight"),
                &tensor_idx,
                &inferred_sizes,
                false,
                borrow_quantized,
            ),
            ffn_norm: load_f32_vec(
                mmap_data,
                data_offset,
                &format!("blk.{l}.ffn_norm.weight"),
                &tensor_idx,
                &inferred_sizes,
            ),
            mlp,
            n_heads,
            rotary_dim,
            rope_inv_freq: build_rope_inv_freq(layer_rope_theta, rotary_dim, 1.0),
            sliding_window,
        });
        if l == 0 || (l + 1) % 8 == 0 || l + 1 == config.n_layers {
            eprintln!("  Loaded layer {}/{}", l + 1, config.n_layers);
        }
    }
    (
        config,
        LagunaWeights {
            token_embd,
            output_norm,
            output,
            layers,
            router_normalize_weights,
            routed_scaling_factor,
        },
    )
}

/// Loads GPT-OSS-specific weights from a parsed GGUF file.
/// Reads a metadata value that may be either a scalar or a per-layer array,
/// expanding it to one entry per block.
///
/// Nemotron-H encodes its hybrid layout this way: `attention.head_count_kv` and
/// `feed_forward_length` are arrays with zeros marking the blocks that are not
/// attention or feed-forward, which is what drives layer classification.
/// Exposes [`per_layer_metadata`] to the compatibility validator so it
/// classifies blocks exactly as the loader will.
pub fn per_layer_metadata_for_validation(gguf: &GGUFFile, key: &str, layers: usize) -> Vec<usize> {
    per_layer_metadata(gguf, key, layers)
}

fn per_layer_metadata(gguf: &GGUFFile, key: &str, layers: usize) -> Vec<usize> {
    if let Some(values) = gguf
        .metadata
        .get(key)
        .and_then(crate::gguf::MetaValue::as_u32_array)
        && values.len() >= layers
    {
        return values
            .into_iter()
            .take(layers)
            .map(|v| v as usize)
            .collect();
    }
    let scalar = gguf.get_u32(key, 0) as usize;
    vec![scalar; layers]
}

/// Loads a Nemotron-H hybrid model (Mamba-2 + NoPE attention + optional routed
/// MoE), the architecture behind Soofi S Isar.
pub fn load_nemotron_h_model(
    mmap_data: &[u8],
    gguf: &GGUFFile,
    borrow_quantized: bool,
) -> (Config, NemotronHWeights) {
    let mut config = Config::from_gguf(gguf);
    let p = config.arch.clone();

    let ssm = SsmDims {
        d_conv: gguf.get_u32(&format!("{}.ssm.conv_kernel", p), 0) as usize,
        d_inner: gguf.get_u32(&format!("{}.ssm.inner_size", p), 0) as usize,
        d_state: gguf.get_u32(&format!("{}.ssm.state_size", p), 0) as usize,
        n_head: gguf.get_u32(&format!("{}.ssm.time_step_rank", p), 0) as usize,
        n_group: gguf.get_u32(&format!("{}.ssm.group_count", p), 0) as usize,
    };
    assert!(
        ssm.d_conv > 1 && ssm.d_inner > 0 && ssm.d_state > 0 && ssm.n_head > 0 && ssm.n_group > 0,
        "Incomplete {}.ssm.* metadata: {:?}",
        p,
        ssm
    );
    assert_eq!(
        ssm.d_inner % ssm.n_head,
        0,
        "ssm.inner_size {} is not divisible by ssm head count {}",
        ssm.d_inner,
        ssm.n_head
    );
    assert_eq!(
        ssm.n_head % ssm.n_group,
        0,
        "ssm head count {} is not divisible by ssm.group_count {}",
        ssm.n_head,
        ssm.n_group
    );

    // A trailing multi-token-prediction head is a separate draft model, not
    // part of the trunk; including it would shift every later block index.
    let nextn = gguf.get_u32(&format!("{}.nextn_predict_layers", p), 0) as usize;
    let trunk_layers = config.n_layers.saturating_sub(nextn);
    assert!(
        trunk_layers > 0,
        "No trunk layers left after removing MTP head"
    );
    config.n_layers = trunk_layers;

    let head_counts =
        per_layer_metadata(gguf, &format!("{}.attention.head_count", p), trunk_layers);
    let kv_counts = per_layer_metadata(
        gguf,
        &format!("{}.attention.head_count_kv", p),
        trunk_layers,
    );
    let ff_lengths = per_layer_metadata(gguf, &format!("{}.feed_forward_length", p), trunk_layers);

    let router_normalize_weights = matches!(
        gguf.metadata.get(&format!("{}.expert_weights_norm", p)),
        Some(crate::gguf::MetaValue::Bool(true))
    ) || gguf.get_u32(&format!("{}.expert_weights_norm", p), 0) == 1;
    let routed_scaling_factor = gguf.get_f32(&format!("{}.expert_weights_scale", p), 1.0);

    eprintln!(
        "Config: dim={}, layers={} (+{} MTP), ssm={}x{}h/{}g state={}, experts={}/{}, vocab={}",
        config.dim,
        trunk_layers,
        nextn,
        ssm.d_inner,
        ssm.n_head,
        ssm.n_group,
        ssm.d_state,
        config.expert_used_count,
        config.expert_count,
        config.vocab_size
    );

    let tensor_idx: HashMap<String, &crate::gguf::TensorInfo> = gguf
        .tensors
        .iter()
        .map(|tensor| (tensor.name.clone(), tensor))
        .collect();
    let inferred_sizes = HashMap::new();
    let data_offset = gguf.data_offset;

    let token_embd = load_weight(
        mmap_data,
        data_offset,
        "token_embd.weight",
        &tensor_idx,
        &inferred_sizes,
        false,
        borrow_quantized,
    );
    let output = if tensor_idx.contains_key("output.weight") {
        load_weight(
            mmap_data,
            data_offset,
            "output.weight",
            &tensor_idx,
            &inferred_sizes,
            false,
            borrow_quantized,
        )
    } else {
        token_embd.clone()
    };

    let mut layers = Vec::with_capacity(trunk_layers);
    let mut attn_layer_count = 0usize;
    let mut max_heads = 0usize;
    let mut max_kv_heads = 0usize;

    for l in 0..trunk_layers {
        let kv_heads = kv_counts.get(l).copied().unwrap_or(0);
        let ff_len = ff_lengths.get(l).copied().unwrap_or(0);
        let heads = head_counts.get(l).copied().unwrap_or(0);

        // A block is recurrent when it declares neither attention heads nor a
        // feed-forward width; attention when it declares no feed-forward width.
        let mixer = if kv_heads == 0 && ff_len == 0 {
            let load_vec = |name: &str| {
                load_f32_vec(mmap_data, data_offset, name, &tensor_idx, &inferred_sizes)
            };
            NemotronMixer::Mamba2(Box::new(Mamba2LayerWeights {
                in_proj: load_weight(
                    mmap_data,
                    data_offset,
                    &format!("blk.{}.ssm_in.weight", l),
                    &tensor_idx,
                    &inferred_sizes,
                    false,
                    borrow_quantized,
                ),
                conv_w: load_vec(&format!("blk.{}.ssm_conv1d.weight", l)),
                conv_b: load_optional_f32_vec(
                    mmap_data,
                    data_offset,
                    &format!("blk.{}.ssm_conv1d.bias", l),
                    &tensor_idx,
                    &inferred_sizes,
                    ssm.conv_dim(),
                ),
                dt_bias: load_vec(&format!("blk.{}.ssm_dt.bias", l)),
                // Stored without a `.weight` suffix, unlike every other tensor.
                a: load_vec(&format!("blk.{}.ssm_a", l)),
                d: load_vec(&format!("blk.{}.ssm_d", l)),
                norm: load_optional_f32_vec(
                    mmap_data,
                    data_offset,
                    &format!("blk.{}.ssm_norm.weight", l),
                    &tensor_idx,
                    &inferred_sizes,
                    ssm.d_inner,
                ),
                out_proj: load_weight(
                    mmap_data,
                    data_offset,
                    &format!("blk.{}.ssm_out.weight", l),
                    &tensor_idx,
                    &inferred_sizes,
                    false,
                    borrow_quantized,
                ),
            }))
        } else if ff_len == 0 {
            let attn = NemotronAttnWeights {
                wq: load_weight(
                    mmap_data,
                    data_offset,
                    &format!("blk.{}.attn_q.weight", l),
                    &tensor_idx,
                    &inferred_sizes,
                    false,
                    borrow_quantized,
                ),
                wk: load_weight(
                    mmap_data,
                    data_offset,
                    &format!("blk.{}.attn_k.weight", l),
                    &tensor_idx,
                    &inferred_sizes,
                    false,
                    borrow_quantized,
                ),
                wv: load_weight(
                    mmap_data,
                    data_offset,
                    &format!("blk.{}.attn_v.weight", l),
                    &tensor_idx,
                    &inferred_sizes,
                    false,
                    borrow_quantized,
                ),
                wo: load_weight(
                    mmap_data,
                    data_offset,
                    &format!("blk.{}.attn_output.weight", l),
                    &tensor_idx,
                    &inferred_sizes,
                    false,
                    borrow_quantized,
                ),
                bo: load_optional_f32_vec(
                    mmap_data,
                    data_offset,
                    &format!("blk.{}.attn_output.bias", l),
                    &tensor_idx,
                    &inferred_sizes,
                    config.dim,
                ),
                n_heads: heads.max(1),
                n_kv_heads: kv_heads.max(1),
                kv_slot: attn_layer_count,
            };
            max_heads = max_heads.max(attn.n_heads);
            max_kv_heads = max_kv_heads.max(attn.n_kv_heads);
            attn_layer_count += 1;
            NemotronMixer::Attention(Box::new(attn))
        } else if config.expert_count > 0 {
            NemotronMixer::Moe(Box::new(NemotronMoeWeights {
                router: load_weight(
                    mmap_data,
                    data_offset,
                    &format!("blk.{}.ffn_gate_inp.weight", l),
                    &tensor_idx,
                    &inferred_sizes,
                    false,
                    borrow_quantized,
                ),
                router_bias: load_optional_f32_vec(
                    mmap_data,
                    data_offset,
                    &format!("blk.{}.exp_probs_b.bias", l),
                    &tensor_idx,
                    &inferred_sizes,
                    config.expert_count,
                ),
                // Nemotron-H experts have no gate projection.
                up_experts: load_expert_weight(
                    mmap_data,
                    data_offset,
                    &format!("blk.{}.ffn_up_exps.weight", l),
                    &tensor_idx,
                    &inferred_sizes,
                    borrow_quantized,
                ),
                down_experts: load_expert_weight(
                    mmap_data,
                    data_offset,
                    &format!("blk.{}.ffn_down_exps.weight", l),
                    &tensor_idx,
                    &inferred_sizes,
                    borrow_quantized,
                ),
                shared_up: load_weight(
                    mmap_data,
                    data_offset,
                    &format!("blk.{}.ffn_up_shexp.weight", l),
                    &tensor_idx,
                    &inferred_sizes,
                    false,
                    borrow_quantized,
                ),
                shared_down: load_weight(
                    mmap_data,
                    data_offset,
                    &format!("blk.{}.ffn_down_shexp.weight", l),
                    &tensor_idx,
                    &inferred_sizes,
                    false,
                    borrow_quantized,
                ),
            }))
        } else {
            NemotronMixer::DenseFfn(Box::new(NemotronDenseFfnWeights {
                up: load_weight(
                    mmap_data,
                    data_offset,
                    &format!("blk.{}.ffn_up.weight", l),
                    &tensor_idx,
                    &inferred_sizes,
                    false,
                    borrow_quantized,
                ),
                up_bias: load_optional_f32_vec(
                    mmap_data,
                    data_offset,
                    &format!("blk.{}.ffn_up.bias", l),
                    &tensor_idx,
                    &inferred_sizes,
                    ff_len,
                ),
                down: load_weight(
                    mmap_data,
                    data_offset,
                    &format!("blk.{}.ffn_down.weight", l),
                    &tensor_idx,
                    &inferred_sizes,
                    false,
                    borrow_quantized,
                ),
                down_bias: load_optional_f32_vec(
                    mmap_data,
                    data_offset,
                    &format!("blk.{}.ffn_down.bias", l),
                    &tensor_idx,
                    &inferred_sizes,
                    config.dim,
                ),
            }))
        };

        layers.push(NemotronHLayerWeights {
            attn_norm: load_f32_vec(
                mmap_data,
                data_offset,
                &format!("blk.{}.attn_norm.weight", l),
                &tensor_idx,
                &inferred_sizes,
            ),
            mixer,
        });
        if l == 0 || (l + 1) % 8 == 0 || l + 1 == trunk_layers {
            eprintln!("  Loaded layer {}/{}", l + 1, trunk_layers);
        }
    }

    // Buffer sizing follows the widest attention block, since blocks differ.
    config.n_heads = max_heads.max(1);
    config.n_kv_heads = max_kv_heads.max(1);
    config.kv_mul = config.n_heads / config.n_kv_heads;
    config.kv_dim = config.n_kv_heads * config.head_dim;
    config.value_dim = config.head_dim;
    // The feed-forward scratch must hold the widest expert or dense width.
    config.hidden_dim = config
        .hidden_dim
        .max(gguf.get_u32(&format!("{}.expert_feed_forward_length", p), 0) as usize)
        .max(gguf.get_u32(&format!("{}.expert_shared_feed_forward_length", p), 0) as usize)
        .max(ff_lengths.iter().copied().max().unwrap_or(0));

    let weights = NemotronHWeights {
        token_embd,
        output_norm: load_f32_vec(
            mmap_data,
            data_offset,
            "output_norm.weight",
            &tensor_idx,
            &inferred_sizes,
        ),
        output,
        layers,
        ssm,
        attn_layer_count,
        router_normalize_weights,
        routed_scaling_factor,
    };
    (config, weights)
}

/// Loads the text-decoder trunk of a Qwen3.5/Qwen3.8 GGUF.
///
/// Qwen3.8 uses `general.architecture = qwen35`: a hybrid decoder with three
/// Gated DeltaNet blocks followed by one gated full-attention block. It cannot
/// use the generic Qwen3/LLaMA loader because its recurrent tensors, joint
/// Q+gate projection, post-attention norm, and trailing MTP block all have a
/// different layout.
pub fn load_qwen35_model(
    mmap_data: &[u8],
    gguf: &GGUFFile,
    borrow_quantized: bool,
) -> (Config, Qwen35Weights) {
    let mut config = Config::from_gguf(gguf);
    let p = config.arch.clone();

    let ssm = SsmDims {
        d_conv: gguf.get_u32(&format!("{}.ssm.conv_kernel", p), 0) as usize,
        d_inner: gguf.get_u32(&format!("{}.ssm.inner_size", p), 0) as usize,
        d_state: gguf.get_u32(&format!("{}.ssm.state_size", p), 0) as usize,
        // Qwen calls this the number of value heads, despite using the shared
        // GGUF `time_step_rank` key.
        n_head: gguf.get_u32(&format!("{}.ssm.time_step_rank", p), 0) as usize,
        // Qwen calls this the number of key heads/groups.
        n_group: gguf.get_u32(&format!("{}.ssm.group_count", p), 0) as usize,
    };
    assert!(
        ssm.d_conv > 1 && ssm.d_inner > 0 && ssm.d_state > 0 && ssm.n_head > 0 && ssm.n_group > 0,
        "Incomplete {}.ssm.* metadata: {:?}",
        p,
        ssm
    );
    assert_eq!(
        ssm.d_inner % ssm.n_head,
        0,
        "qwen35 ssm.inner_size must be divisible by its value-head count"
    );
    assert_eq!(
        ssm.n_head % ssm.n_group,
        0,
        "qwen35 value-head count must be divisible by its key-head count"
    );
    // Every per-value-head memory is stored as a square
    // `[value_dim, key_dim]` matrix. Current qwen35 dense models keep both
    // dimensions equal; reject a future incompatible variant explicitly.
    assert_eq!(
        ssm.head_dim(),
        ssm.d_state,
        "qwen35 Gated DeltaNet requires equal key and value head widths"
    );
    assert_eq!(
        config.value_dim, config.head_dim,
        "qwen35 full-attention key/value widths must match"
    );
    assert!(
        config.n_heads > 0 && config.n_kv_heads > 0 && config.n_heads % config.n_kv_heads == 0,
        "Invalid qwen35 attention-head metadata"
    );

    // The final NextN/MTP block is a draft head. It is not a 65th ordinary
    // decoder layer and must never consume a KV-cache or recurrent-state slot.
    let nextn = gguf.get_u32(&format!("{}.nextn_predict_layers", p), 0) as usize;
    let trunk_layers = config.n_layers.saturating_sub(nextn);
    assert!(
        trunk_layers > 0,
        "No qwen35 trunk layers remain after excluding MTP/NextN blocks"
    );
    config.n_layers = trunk_layers;

    let rotary_dim = gguf.get_u32(&format!("{}.rope.dimension_count", p), 0) as usize;
    assert!(
        rotary_dim > 0 && rotary_dim <= config.head_dim && rotary_dim % 2 == 0,
        "Invalid qwen35 rotary dimension {} for head width {}",
        rotary_dim,
        config.head_dim
    );
    let rope_sections = gguf
        .metadata
        .get(&format!("{}.rope.dimension_sections", p))
        .and_then(crate::gguf::MetaValue::as_u32_array)
        .unwrap_or_default();
    assert!(
        rope_sections.len() >= 3 && rope_sections[..3].iter().all(|section| *section > 0),
        "qwen35 requires non-empty MRoPE dimension sections"
    );

    let conv_dim = ssm.conv_dim();
    let key_dim = ssm.n_group * ssm.d_state;
    let value_dim = ssm.d_inner;
    let value_head_dim = ssm.head_dim();
    eprintln!(
        "Config: qwen35 dim={}, trunk_layers={} (+{} MTP), full_attn_heads={}/{}, GDN={}v/{}k x {}, hidden={}, vocab={}, ctx={}",
        config.dim,
        trunk_layers,
        nextn,
        config.n_heads,
        config.n_kv_heads,
        ssm.n_head,
        ssm.n_group,
        ssm.d_state,
        config.hidden_dim,
        config.vocab_size,
        config.max_seq_len
    );

    let tensor_idx: HashMap<String, &crate::gguf::TensorInfo> = gguf
        .tensors
        .iter()
        .map(|tensor| (tensor.name.clone(), tensor))
        .collect();
    // All qwen35 tensors use standard f32/K-quant layouts, so their sizes are
    // known from the dtype and no adjacent-offset inference is needed.
    let inferred_sizes = HashMap::new();
    let data_offset = gguf.data_offset;

    let token_embd = load_weight(
        mmap_data,
        data_offset,
        "token_embd.weight",
        &tensor_idx,
        &inferred_sizes,
        false,
        borrow_quantized,
    );
    let output = if tensor_idx.contains_key("output.weight") {
        load_weight(
            mmap_data,
            data_offset,
            "output.weight",
            &tensor_idx,
            &inferred_sizes,
            false,
            borrow_quantized,
        )
    } else {
        token_embd.clone()
    };

    let mut layers = Vec::with_capacity(trunk_layers);
    let mut recurrent_layer_count = 0usize;
    let mut attn_layer_count = 0usize;
    for layer_index in 0..trunk_layers {
        let prefix = format!("blk.{}", layer_index);
        let qkv_name = format!("{}.attn_qkv.weight", prefix);
        let mixer = if tensor_idx.contains_key(&qkv_name) {
            let qkv = load_weight(
                mmap_data,
                data_offset,
                &qkv_name,
                &tensor_idx,
                &inferred_sizes,
                false,
                borrow_quantized,
            );
            validate_global_shape(&qkv_name, &qkv, 2 * key_dim + value_dim, config.dim);

            let gate_name = format!("{}.attn_gate.weight", prefix);
            let gate = load_weight(
                mmap_data,
                data_offset,
                &gate_name,
                &tensor_idx,
                &inferred_sizes,
                false,
                borrow_quantized,
            );
            validate_global_shape(&gate_name, &gate, value_dim, config.dim);

            let alpha_name = format!("{}.ssm_alpha.weight", prefix);
            let alpha = load_weight(
                mmap_data,
                data_offset,
                &alpha_name,
                &tensor_idx,
                &inferred_sizes,
                false,
                borrow_quantized,
            );
            validate_global_shape(&alpha_name, &alpha, ssm.n_head, config.dim);

            let beta_name = format!("{}.ssm_beta.weight", prefix);
            let beta = load_weight(
                mmap_data,
                data_offset,
                &beta_name,
                &tensor_idx,
                &inferred_sizes,
                false,
                borrow_quantized,
            );
            validate_global_shape(&beta_name, &beta, ssm.n_head, config.dim);

            let conv_name = format!("{}.ssm_conv1d.weight", prefix);
            let conv_w = load_f32_vec(
                mmap_data,
                data_offset,
                &conv_name,
                &tensor_idx,
                &inferred_sizes,
            );
            assert_eq!(
                conv_w.len(),
                conv_dim * ssm.d_conv,
                "Shape mismatch for {}",
                conv_name
            );
            let dt_name = format!("{}.ssm_dt.bias", prefix);
            let dt_bias = load_f32_vec(
                mmap_data,
                data_offset,
                &dt_name,
                &tensor_idx,
                &inferred_sizes,
            );
            assert_eq!(dt_bias.len(), ssm.n_head, "Shape mismatch for {}", dt_name);
            let a_name = format!("{}.ssm_a", prefix);
            let a = load_f32_vec(
                mmap_data,
                data_offset,
                &a_name,
                &tensor_idx,
                &inferred_sizes,
            );
            assert_eq!(a.len(), ssm.n_head, "Shape mismatch for {}", a_name);
            let norm_name = format!("{}.ssm_norm.weight", prefix);
            let norm = load_f32_vec(
                mmap_data,
                data_offset,
                &norm_name,
                &tensor_idx,
                &inferred_sizes,
            );
            assert_eq!(
                norm.len(),
                value_head_dim,
                "Shape mismatch for {}",
                norm_name
            );
            let out_name = format!("{}.ssm_out.weight", prefix);
            let out = load_weight(
                mmap_data,
                data_offset,
                &out_name,
                &tensor_idx,
                &inferred_sizes,
                false,
                borrow_quantized,
            );
            validate_global_shape(&out_name, &out, config.dim, value_dim);

            recurrent_layer_count += 1;
            Qwen35Mixer::Linear(Box::new(Qwen35LinearWeights {
                qkv,
                gate,
                conv_w,
                dt_bias,
                a,
                beta,
                alpha,
                norm,
                out,
            }))
        } else {
            let q_gate_name = format!("{}.attn_q.weight", prefix);
            let q_gate = load_weight(
                mmap_data,
                data_offset,
                &q_gate_name,
                &tensor_idx,
                &inferred_sizes,
                false,
                borrow_quantized,
            );
            validate_global_shape(
                &q_gate_name,
                &q_gate,
                2 * config.n_heads * config.head_dim,
                config.dim,
            );
            let k_name = format!("{}.attn_k.weight", prefix);
            let k = load_weight(
                mmap_data,
                data_offset,
                &k_name,
                &tensor_idx,
                &inferred_sizes,
                false,
                borrow_quantized,
            );
            validate_global_shape(&k_name, &k, config.n_kv_heads * config.head_dim, config.dim);
            let v_name = format!("{}.attn_v.weight", prefix);
            let v = load_weight(
                mmap_data,
                data_offset,
                &v_name,
                &tensor_idx,
                &inferred_sizes,
                false,
                borrow_quantized,
            );
            validate_global_shape(
                &v_name,
                &v,
                config.n_kv_heads * config.value_dim,
                config.dim,
            );
            let q_norm = load_f32_vec(
                mmap_data,
                data_offset,
                &format!("{}.attn_q_norm.weight", prefix),
                &tensor_idx,
                &inferred_sizes,
            );
            let k_norm = load_f32_vec(
                mmap_data,
                data_offset,
                &format!("{}.attn_k_norm.weight", prefix),
                &tensor_idx,
                &inferred_sizes,
            );
            assert_eq!(
                q_norm.len(),
                config.head_dim,
                "qwen35 Q-norm width mismatch"
            );
            assert_eq!(
                k_norm.len(),
                config.head_dim,
                "qwen35 K-norm width mismatch"
            );
            let out_name = format!("{}.attn_output.weight", prefix);
            let out = load_weight(
                mmap_data,
                data_offset,
                &out_name,
                &tensor_idx,
                &inferred_sizes,
                false,
                borrow_quantized,
            );
            validate_global_shape(
                &out_name,
                &out,
                config.dim,
                config.n_heads * config.value_dim,
            );

            let kv_slot = attn_layer_count;
            attn_layer_count += 1;
            Qwen35Mixer::Attention(Box::new(Qwen35AttentionWeights {
                q_gate,
                k,
                v,
                q_norm,
                k_norm,
                out,
                kv_slot,
            }))
        };

        let ffn_gate_name = format!("{}.ffn_gate.weight", prefix);
        let ffn_gate = load_weight(
            mmap_data,
            data_offset,
            &ffn_gate_name,
            &tensor_idx,
            &inferred_sizes,
            false,
            borrow_quantized,
        );
        validate_global_shape(&ffn_gate_name, &ffn_gate, config.hidden_dim, config.dim);
        let ffn_up_name = format!("{}.ffn_up.weight", prefix);
        let ffn_up = load_weight(
            mmap_data,
            data_offset,
            &ffn_up_name,
            &tensor_idx,
            &inferred_sizes,
            false,
            borrow_quantized,
        );
        validate_global_shape(&ffn_up_name, &ffn_up, config.hidden_dim, config.dim);
        let ffn_down_name = format!("{}.ffn_down.weight", prefix);
        let ffn_down = load_weight(
            mmap_data,
            data_offset,
            &ffn_down_name,
            &tensor_idx,
            &inferred_sizes,
            false,
            borrow_quantized,
        );
        validate_global_shape(&ffn_down_name, &ffn_down, config.dim, config.hidden_dim);

        layers.push(Qwen35LayerWeights {
            attn_norm: load_f32_vec(
                mmap_data,
                data_offset,
                &format!("{}.attn_norm.weight", prefix),
                &tensor_idx,
                &inferred_sizes,
            ),
            post_attn_norm: load_f32_vec(
                mmap_data,
                data_offset,
                &format!("{}.post_attention_norm.weight", prefix),
                &tensor_idx,
                &inferred_sizes,
            ),
            mixer,
            ffn_gate,
            ffn_up,
            ffn_down,
        });
        if layer_index == 0 || (layer_index + 1) % 8 == 0 || layer_index + 1 == trunk_layers {
            eprintln!(
                "  Loaded qwen35 trunk layer {}/{}",
                layer_index + 1,
                trunk_layers
            );
        }
    }
    assert!(
        recurrent_layer_count > 0 && attn_layer_count > 0,
        "qwen35 trunk must contain both Gated DeltaNet and full-attention blocks"
    );

    let mtp = if nextn == 1 {
        let mtp_prefix = format!("blk.{}.nextn", trunk_layers);
        let plain_prefix = format!("blk.{}", trunk_layers);
        let find_name = |suffix: &str, global: Option<&str>| -> Option<String> {
            let scoped = format!("{}.{}", mtp_prefix, suffix);
            if tensor_idx.contains_key(&scoped) {
                return Some(scoped);
            }
            let plain = format!("{}.{}", plain_prefix, suffix);
            if tensor_idx.contains_key(&plain) {
                return Some(plain);
            }
            global
                .filter(|name| tensor_idx.contains_key(*name))
                .map(str::to_string)
        };
        let loaded = (|| -> Option<Qwen35MtpWeights> {
            let load_required_weight = |suffix: &str| -> Option<(String, Weight)> {
                let name = find_name(suffix, None)?;
                let weight = load_weight(
                    mmap_data,
                    data_offset,
                    &name,
                    &tensor_idx,
                    &inferred_sizes,
                    false,
                    borrow_quantized,
                );
                Some((name, weight))
            };
            let load_required_vec = |suffix: &str, global: Option<&str>| -> Option<Vec<f32>> {
                let name = find_name(suffix, global)?;
                Some(load_f32_vec(
                    mmap_data,
                    data_offset,
                    &name,
                    &tensor_idx,
                    &inferred_sizes,
                ))
            };

            let (eh_name, eh_proj) = load_required_weight("eh_proj.weight")?;
            validate_global_shape(&eh_name, &eh_proj, config.dim, 2 * config.dim);

            let (q_name, q_gate) = load_required_weight("attn_q.weight")?;
            validate_global_shape(
                &q_name,
                &q_gate,
                2 * config.n_heads * config.head_dim,
                config.dim,
            );
            let (k_name, k) = load_required_weight("attn_k.weight")?;
            validate_global_shape(&k_name, &k, config.n_kv_heads * config.head_dim, config.dim);
            let (v_name, v) = load_required_weight("attn_v.weight")?;
            validate_global_shape(
                &v_name,
                &v,
                config.n_kv_heads * config.value_dim,
                config.dim,
            );
            let (out_name, attn_out) = load_required_weight("attn_output.weight")?;
            validate_global_shape(
                &out_name,
                &attn_out,
                config.dim,
                config.n_heads * config.value_dim,
            );

            let (gate_name, ffn_gate) = load_required_weight("ffn_gate.weight")?;
            validate_global_shape(&gate_name, &ffn_gate, config.hidden_dim, config.dim);
            let (up_name, ffn_up) = load_required_weight("ffn_up.weight")?;
            validate_global_shape(&up_name, &ffn_up, config.hidden_dim, config.dim);
            let (down_name, ffn_down) = load_required_weight("ffn_down.weight")?;
            validate_global_shape(&down_name, &ffn_down, config.dim, config.hidden_dim);

            let optional_weight = |suffix: &str, global: Option<&str>| -> Option<Weight> {
                let name = find_name(suffix, global)?;
                Some(load_weight(
                    mmap_data,
                    data_offset,
                    &name,
                    &tensor_idx,
                    &inferred_sizes,
                    false,
                    borrow_quantized,
                ))
            };

            Some(Qwen35MtpWeights {
                eh_proj,
                embedding_norm: load_required_vec("enorm.weight", Some("nextn.enorm.weight"))?,
                hidden_norm: load_required_vec("hnorm.weight", Some("nextn.hnorm.weight"))?,
                attn_norm: load_required_vec("attn_norm.weight", None)?,
                post_attn_norm: load_required_vec("ffn_norm.weight", None)
                    .or_else(|| load_required_vec("post_attention_norm.weight", None))?,
                attention: Qwen35AttentionWeights {
                    q_gate,
                    k,
                    v,
                    q_norm: load_required_vec("attn_q_norm.weight", None)?,
                    k_norm: load_required_vec("attn_k_norm.weight", None)?,
                    out: attn_out,
                    kv_slot: 0,
                },
                ffn_gate,
                ffn_up,
                ffn_down,
                head_norm: load_required_vec(
                    "shared_head_norm.weight",
                    Some("nextn.shared_head_norm.weight"),
                )?,
                token_embd: optional_weight(
                    "embed_tokens.weight",
                    Some("nextn.embed_tokens.weight"),
                ),
                output: optional_weight(
                    "shared_head_head.weight",
                    Some("nextn.shared_head_head.weight"),
                ),
            })
        })();
        if loaded.is_none() {
            eprintln!(
                "  Embedded Qwen draft head metadata is present, but one or more required tensors are missing; native drafting is disabled."
            );
        } else {
            eprintln!("  Loaded embedded Qwen one-step draft head");
        }
        loaded.map(Box::new)
    } else {
        None
    };

    let weights = Qwen35Weights {
        token_embd,
        output_norm: load_f32_vec(
            mmap_data,
            data_offset,
            "output_norm.weight",
            &tensor_idx,
            &inferred_sizes,
        ),
        output,
        layers,
        ssm,
        recurrent_layer_count,
        attn_layer_count,
        rotary_dim,
        rope_inv_freq: build_rope_inv_freq(
            config.rope_theta,
            rotary_dim,
            config.rope_scaling_factor,
        ),
        mtp,
    };
    (config, weights)
}

pub fn load_gpt_oss_model(
    mmap_data: &[u8],
    gguf: &GGUFFile,
    borrow_quantized: bool,
) -> (Config, GptOssWeights) {
    let config = Config::from_gguf(gguf);
    eprintln!(
        "Config: dim={}, layers={}, heads={}/{}, hidden={}, vocab={}, ctx={}",
        config.dim,
        config.n_layers,
        config.n_heads,
        config.n_kv_heads,
        config.hidden_dim,
        config.vocab_size,
        config.max_seq_len
    );

    let tensor_idx: HashMap<String, &crate::gguf::TensorInfo> =
        gguf.tensors.iter().map(|t| (t.name.clone(), t)).collect();
    let data_offset = gguf.data_offset;

    let mut inferred_sizes: HashMap<String, usize> = HashMap::new();
    if !gguf.tensors.is_empty() {
        let mmap_len = mmap_data.len();
        let mut offs: Vec<(u64, usize)> = gguf
            .tensors
            .iter()
            .enumerate()
            .map(|(i, t)| (t.offset, i))
            .collect();
        offs.sort_unstable_by_key(|o| o.0);
        for w in 0..offs.len() {
            let (off, idx) = offs[w];
            let next_off = if w + 1 < offs.len() {
                offs[w + 1].0
            } else {
                (mmap_len as u64).saturating_sub(data_offset as u64)
            };
            let byte_size = if next_off > off {
                (next_off - off) as usize
            } else {
                0
            };
            inferred_sizes.insert(gguf.tensors[idx].name.clone(), byte_size);
        }
    }

    let token_embd = load_weight(
        mmap_data,
        data_offset,
        "token_embd.weight",
        &tensor_idx,
        &inferred_sizes,
        false,
        borrow_quantized,
    );
    let output_norm = load_f32_vec(
        mmap_data,
        data_offset,
        "output_norm.weight",
        &tensor_idx,
        &inferred_sizes,
    );
    let output = load_weight(
        mmap_data,
        data_offset,
        "output.weight",
        &tensor_idx,
        &inferred_sizes,
        false,
        borrow_quantized,
    );

    let mut layers = Vec::with_capacity(config.n_layers);
    for l in 0..config.n_layers {
        let layer = GptOssLayerWeights {
            attn_norm: load_f32_vec(
                mmap_data,
                data_offset,
                &format!("blk.{}.attn_norm.weight", l),
                &tensor_idx,
                &inferred_sizes,
            ),
            wq: load_weight(
                mmap_data,
                data_offset,
                &format!("blk.{}.attn_q.weight", l),
                &tensor_idx,
                &inferred_sizes,
                false,
                borrow_quantized,
            ),
            bq: load_f32_vec(
                mmap_data,
                data_offset,
                &format!("blk.{}.attn_q.bias", l),
                &tensor_idx,
                &inferred_sizes,
            ),
            wk: load_weight(
                mmap_data,
                data_offset,
                &format!("blk.{}.attn_k.weight", l),
                &tensor_idx,
                &inferred_sizes,
                false,
                borrow_quantized,
            ),
            bk: load_f32_vec(
                mmap_data,
                data_offset,
                &format!("blk.{}.attn_k.bias", l),
                &tensor_idx,
                &inferred_sizes,
            ),
            wv: load_weight(
                mmap_data,
                data_offset,
                &format!("blk.{}.attn_v.weight", l),
                &tensor_idx,
                &inferred_sizes,
                false,
                borrow_quantized,
            ),
            bv: load_f32_vec(
                mmap_data,
                data_offset,
                &format!("blk.{}.attn_v.bias", l),
                &tensor_idx,
                &inferred_sizes,
            ),
            wo: load_weight(
                mmap_data,
                data_offset,
                &format!("blk.{}.attn_output.weight", l),
                &tensor_idx,
                &inferred_sizes,
                false,
                borrow_quantized,
            ),
            bo: load_f32_vec(
                mmap_data,
                data_offset,
                &format!("blk.{}.attn_output.bias", l),
                &tensor_idx,
                &inferred_sizes,
            ),
            sinks: load_f32_vec(
                mmap_data,
                data_offset,
                &format!("blk.{}.attn_sinks.weight", l),
                &tensor_idx,
                &inferred_sizes,
            ),
            post_attn_norm: load_f32_vec(
                mmap_data,
                data_offset,
                &format!("blk.{}.post_attention_norm.weight", l),
                &tensor_idx,
                &inferred_sizes,
            ),
            gate_inp: load_weight(
                mmap_data,
                data_offset,
                &format!("blk.{}.ffn_gate_inp.weight", l),
                &tensor_idx,
                &inferred_sizes,
                false,
                borrow_quantized,
            ),
            gate_inp_bias: load_f32_vec(
                mmap_data,
                data_offset,
                &format!("blk.{}.ffn_gate_inp.bias", l),
                &tensor_idx,
                &inferred_sizes,
            ),
            gate_exps: load_expert_weight(
                mmap_data,
                data_offset,
                &format!("blk.{}.ffn_gate_exps.weight", l),
                &tensor_idx,
                &inferred_sizes,
                borrow_quantized,
            ),
            gate_exps_bias: load_weight(
                mmap_data,
                data_offset,
                &format!("blk.{}.ffn_gate_exps.bias", l),
                &tensor_idx,
                &inferred_sizes,
                true,
                false,
            ),
            up_exps: load_expert_weight(
                mmap_data,
                data_offset,
                &format!("blk.{}.ffn_up_exps.weight", l),
                &tensor_idx,
                &inferred_sizes,
                borrow_quantized,
            ),
            up_exps_bias: load_weight(
                mmap_data,
                data_offset,
                &format!("blk.{}.ffn_up_exps.bias", l),
                &tensor_idx,
                &inferred_sizes,
                true,
                false,
            ),
            down_exps: load_expert_weight(
                mmap_data,
                data_offset,
                &format!("blk.{}.ffn_down_exps.weight", l),
                &tensor_idx,
                &inferred_sizes,
                borrow_quantized,
            ),
            down_exps_bias: load_weight(
                mmap_data,
                data_offset,
                &format!("blk.{}.ffn_down_exps.bias", l),
                &tensor_idx,
                &inferred_sizes,
                true,
                false,
            ),
        };
        layers.push(layer);
        if l == 0 || (l + 1) % 8 == 0 || l + 1 == config.n_layers {
            eprintln!("  Loaded layer {}/{}", l + 1, config.n_layers);
        }
    }

    let weights = GptOssWeights {
        token_embd,
        output_norm,
        output,
        layers,
    };
    (config, weights)
}

// ─── Forward Pass ────────────────────────────────────────────────────────────

/// RMS Normalization writing into a pre-allocated output buffer.
#[inline]
/// Applies RMSNorm to an activation vector into an output buffer.
pub(crate) fn rms_norm_into(x: &[f32], weight: &[f32], eps: f32, out: &mut Vec<f32>) {
    let n = x.len();
    let ss = simd::dot_f32(x, x) / n as f32;
    let scale = 1.0 / (ss + eps).sqrt();
    out.resize(n, 0.0);
    for i in 0..n {
        out[i] = x[i] * scale * weight[i];
    }
}

/// True LayerNorm (mean-subtract, unit variance, affine weight + bias) in
/// place. Used by BERT-style encoders (nomic-bert); RMSNorm omits the mean.
#[inline]
pub(crate) fn layer_norm_in_place(x: &mut [f32], weight: &[f32], bias: &[f32], eps: f32) {
    let n = x.len();
    if n == 0 {
        return;
    }
    let mean = x.iter().sum::<f32>() / n as f32;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let inv = 1.0 / (var + eps).sqrt();
    for i in 0..n {
        x[i] = (x[i] - mean) * inv * weight[i] + bias[i];
    }
}

#[inline]
/// Applies per-head RMSNorm in place, using the same weight vector for each head.
fn rms_norm_heads_in_place(
    x: &mut [f32],
    head_dim: usize,
    heads: usize,
    weight: Option<&[f32]>,
    eps: f32,
) {
    if head_dim == 0 || heads == 0 {
        return;
    }
    debug_assert!(x.len() >= head_dim * heads);
    if let Some(weight) = weight {
        debug_assert_eq!(weight.len(), head_dim);
    }
    for h in 0..heads {
        let start = h * head_dim;
        let end = start + head_dim;
        let head = &mut x[start..end];
        let ss = simd::dot_f32(head, head) / head_dim as f32;
        let scale = 1.0 / (ss + eps).sqrt();
        if let Some(weight) = weight {
            for i in 0..head_dim {
                head[i] *= scale * weight[i];
            }
        } else {
            for value in head {
                *value *= scale;
            }
        }
    }
}

/// Applies optional Q/K per-head RMSNorm used by Qwen3 and related models.
/// The GGUF tensors are one head wide and are shared by all query/key heads.
#[inline]
fn apply_qk_norm_if_present(
    q: &mut [f32],
    k: &mut [f32],
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    q_weight: &[f32],
    k_weight: &[f32],
    eps: f32,
) {
    if !q_weight.is_empty() {
        assert_eq!(
            q_weight.len(),
            head_dim,
            "attn_q_norm width must match attention key length"
        );
        rms_norm_heads_in_place(q, head_dim, n_heads, Some(q_weight), eps);
    }
    if !k_weight.is_empty() {
        assert_eq!(
            k_weight.len(),
            head_dim,
            "attn_k_norm width must match attention key length"
        );
        rms_norm_heads_in_place(k, head_dim, n_kv_heads, Some(k_weight), eps);
    }
}

/// Applies the RoPE layout expected by the model architecture. Qwen2 and
/// Qwen3 use Hugging Face's `rotate_half` convention: each element in the
/// first half of a head rotates with its counterpart in the second half.
/// LLaMA-family GGUFs use adjacent pairs instead.
#[cfg(test)]
#[inline]
fn apply_model_rope(config: &Config, q: &mut [f32], k: &mut [f32], pos: usize, inv_freq: &[f32]) {
    if matches!(config.arch.as_str(), "qwen2" | "qwen3") {
        apply_rope_qk_neox(
            q,
            k,
            pos,
            config.head_dim,
            config.n_heads,
            config.n_kv_heads,
            inv_freq,
        );
    } else {
        apply_rope_qk(
            q,
            k,
            pos,
            config.head_dim,
            config.n_heads,
            config.n_kv_heads,
            inv_freq,
        );
    }
}

/// Fills reusable RoPE angle scratch for one position. The same angles apply
/// to every standard decoder layer at that position, so callers prepare them
/// once and reuse them through the complete layer stack.
#[inline]
fn prepare_rope_sin_cos_into(pos: usize, inv_freq: &[f32], sin: &mut [f32], cos: &mut [f32]) {
    debug_assert_eq!(sin.len(), inv_freq.len());
    debug_assert_eq!(cos.len(), inv_freq.len());
    for ((&freq, sin), cos) in inv_freq.iter().zip(sin.iter_mut()).zip(cos.iter_mut()) {
        (*sin, *cos) = (pos as f32 * freq).sin_cos();
    }
}

/// Applies already prepared RoPE angles using the layout expected by a
/// standard decoder architecture.
#[inline]
fn apply_model_rope_prepared(
    config: &Config,
    q: &mut [f32],
    k: &mut [f32],
    sin: &[f32],
    cos: &[f32],
) {
    if matches!(config.arch.as_str(), "qwen2" | "qwen3") {
        apply_rope_qk_neox_prepared(
            q,
            k,
            config.head_dim,
            config.n_heads,
            config.n_kv_heads,
            sin,
            cos,
        );
    } else {
        apply_rope_qk_prepared(
            q,
            k,
            config.head_dim,
            config.n_heads,
            config.n_kv_heads,
            sin,
            cos,
        );
    }
}

#[inline]
/// Adds an optional projection bias when the model stores one.
fn add_bias_if_present(out: &mut [f32], bias: &[f32]) {
    if bias.is_empty() {
        return;
    }
    debug_assert_eq!(out.len(), bias.len());
    for i in 0..out.len() {
        out[i] += bias[i];
    }
}

/// Applies the same rotary angles to query and key vectors in one pass.
pub(crate) fn apply_rope_qk(
    q: &mut [f32],
    k: &mut [f32],
    pos: usize,
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    inv_freq: &[f32],
) {
    debug_assert!(inv_freq.len() >= head_dim / 2);
    let last = head_dim - (head_dim % 2);
    for i in (0..last).step_by(2) {
        let angle = pos as f32 * inv_freq[i / 2];
        let (sin_a, cos_a) = angle.sin_cos();

        for h in 0..n_heads {
            let off = h * head_dim;
            let idx0 = off + i;
            let idx1 = off + i + 1;
            if idx1 >= q.len() {
                break;
            }
            let v0 = q[idx0];
            let v1 = q[idx1];
            q[idx0] = v0 * cos_a - v1 * sin_a;
            q[idx1] = v0 * sin_a + v1 * cos_a;
        }

        for h in 0..n_kv_heads {
            let off = h * head_dim;
            let idx0 = off + i;
            let idx1 = off + i + 1;
            if idx1 >= k.len() {
                break;
            }
            let v0 = k[idx0];
            let v1 = k[idx1];
            k[idx0] = v0 * cos_a - v1 * sin_a;
            k[idx1] = v0 * sin_a + v1 * cos_a;
        }
    }
}

/// Applies adjacent-pair RoPE using caller-prepared sine/cosine angles.
#[inline]
fn apply_rope_qk_prepared(
    q: &mut [f32],
    k: &mut [f32],
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    sin: &[f32],
    cos: &[f32],
) {
    let last = head_dim - (head_dim % 2);
    let pairs = last / 2;
    debug_assert!(sin.len() >= pairs && cos.len() >= pairs);
    for pair in 0..pairs {
        let i = pair * 2;
        let sin_a = sin[pair];
        let cos_a = cos[pair];

        for h in 0..n_heads {
            let off = h * head_dim;
            let idx0 = off + i;
            let idx1 = idx0 + 1;
            if idx1 >= q.len() {
                break;
            }
            let v0 = q[idx0];
            let v1 = q[idx1];
            q[idx0] = v0 * cos_a - v1 * sin_a;
            q[idx1] = v0 * sin_a + v1 * cos_a;
        }

        for h in 0..n_kv_heads {
            let off = h * head_dim;
            let idx0 = off + i;
            let idx1 = idx0 + 1;
            if idx1 >= k.len() {
                break;
            }
            let v0 = k[idx0];
            let v1 = k[idx1];
            k[idx0] = v0 * cos_a - v1 * sin_a;
            k[idx1] = v0 * sin_a + v1 * cos_a;
        }
    }
}

/// Applies NeoX-style RoPE where each pair spans the first and second half of a head.
pub(crate) fn apply_rope_qk_neox(
    q: &mut [f32],
    k: &mut [f32],
    pos: usize,
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    inv_freq: &[f32],
) {
    let half = head_dim / 2;
    debug_assert!(inv_freq.len() >= half);
    for i in 0..half {
        let angle = pos as f32 * inv_freq[i];
        let (sin_a, cos_a) = angle.sin_cos();

        for h in 0..n_heads {
            let off = h * head_dim;
            let idx0 = off + i;
            let idx1 = off + i + half;
            if idx1 >= q.len() {
                break;
            }
            let v0 = q[idx0];
            let v1 = q[idx1];
            q[idx0] = v0 * cos_a - v1 * sin_a;
            q[idx1] = v0 * sin_a + v1 * cos_a;
        }

        for h in 0..n_kv_heads {
            let off = h * head_dim;
            let idx0 = off + i;
            let idx1 = off + i + half;
            if idx1 >= k.len() {
                break;
            }
            let v0 = k[idx0];
            let v1 = k[idx1];
            k[idx0] = v0 * cos_a - v1 * sin_a;
            k[idx1] = v0 * sin_a + v1 * cos_a;
        }
    }
}

/// Applies NeoX rotate-half RoPE using caller-prepared sine/cosine angles.
#[inline]
fn apply_rope_qk_neox_prepared(
    q: &mut [f32],
    k: &mut [f32],
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    sin: &[f32],
    cos: &[f32],
) {
    let half = head_dim / 2;
    debug_assert!(sin.len() >= half && cos.len() >= half);
    for i in 0..half {
        let sin_a = sin[i];
        let cos_a = cos[i];

        for h in 0..n_heads {
            let off = h * head_dim;
            let idx0 = off + i;
            let idx1 = off + i + half;
            if idx1 >= q.len() {
                break;
            }
            let v0 = q[idx0];
            let v1 = q[idx1];
            q[idx0] = v0 * cos_a - v1 * sin_a;
            q[idx1] = v0 * sin_a + v1 * cos_a;
        }

        for h in 0..n_kv_heads {
            let off = h * head_dim;
            let idx0 = off + i;
            let idx1 = off + i + half;
            if idx1 >= k.len() {
                break;
            }
            let v0 = k[idx0];
            let v1 = k[idx1];
            k[idx0] = v0 * cos_a - v1 * sin_a;
            k[idx1] = v0 * sin_a + v1 * cos_a;
        }
    }
}

/// Applies Qwen/GLM-style `rotate_half` RoPE to a prefix of each head. Laguna
/// uses 64 rotary dimensions for full-attention layers and 128 for its SWA
/// layers, while every head remains 128 elements wide.
fn apply_rope_qk_neox_partial(
    q: &mut [f32],
    k: &mut [f32],
    pos: usize,
    head_dim: usize,
    rotary_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    inv_freq: &[f32],
) {
    let rotary_dim = rotary_dim.min(head_dim) & !1;
    let half = rotary_dim / 2;
    debug_assert!(inv_freq.len() >= half);
    for i in 0..half {
        let (sin, cos) = (pos as f32 * inv_freq[i]).sin_cos();
        for (values, heads) in [(&mut *q, n_heads), (&mut *k, n_kv_heads)] {
            for head in 0..heads {
                let start = head * head_dim;
                if start + rotary_dim > values.len() {
                    break;
                }
                let a = values[start + i];
                let b = values[start + half + i];
                values[start + i] = a * cos - b * sin;
                values[start + half + i] = b * cos + a * sin;
            }
        }
    }
}

/// Applies NeoX-style RoPE to one query/key tensor.
pub(crate) fn apply_rope_neox(
    x: &mut [f32],
    pos: usize,
    head_dim: usize,
    heads: usize,
    inv_freq: &[f32],
) {
    let half = head_dim / 2;
    debug_assert!(inv_freq.len() >= half);
    for i in 0..half {
        let angle = pos as f32 * inv_freq[i];
        let (sin_a, cos_a) = angle.sin_cos();

        for h in 0..heads {
            let off = h * head_dim;
            let idx0 = off + i;
            let idx1 = off + i + half;
            if idx1 >= x.len() {
                break;
            }
            let v0 = x[idx0];
            let v1 = x[idx1];
            x[idx0] = v0 * cos_a - v1 * sin_a;
            x[idx1] = v0 * sin_a + v1 * cos_a;
        }
    }
}

#[inline]
/// Checks whether the optional approximate attention exponent path is enabled.
fn fast_attn_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("RUSTY_LLM_FAST_ATTN").is_some())
}

#[inline(always)]
/// Computes the exponential used by attention, selecting exact or approximate behavior.
fn exp_attn(x: f32) -> f32 {
    if fast_attn_enabled() {
        fast_exp_approx(x)
    } else {
        x.exp()
    }
}

#[inline(always)]
/// Computes a fast approximate exponential for optional attention speed experiments.
fn fast_exp_approx(x: f32) -> f32 {
    // Schraudolph-style approximation; enable only for aggressive throughput mode.
    let xc = x.clamp(-80.0, 80.0);
    let bits = (12102203.0f32 * xc + 1064866805.0f32) as i32;
    f32::from_bits(bits as u32)
}

#[inline]
/// Runs numerically stable online attention with an additional attention-sink score.
pub(crate) fn online_attention_with_sink(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    key_stride: usize,
    value_stride: usize,
    slot_count: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    start_t: usize,
    end_t: usize,
    scale: f32,
    sink_score: f32,
    out: &mut [f32],
) {
    if slot_count == 0 || start_t > end_t {
        return;
    }
    let mut max_score = sink_score;
    let mut denom = 1.0f32;
    let linear_slots = attention_uses_linear_slots(start_t, end_t, slot_count);
    // Advance the ring slot incrementally. Decode normally follows the
    // linear path, where this removes a per-token modulo and multiplication;
    // wrapped sliding-window ranges still get one predictable wrap check.
    let mut slot = if linear_slots {
        start_t
    } else {
        start_t % slot_count
    };

    for _ in start_t..=end_t {
        let k_off = slot * key_stride;
        let keys_sub = unsafe { keys.get_unchecked(k_off..k_off + key_head_dim) };
        let score = simd::dot_f32(query, keys_sub) * scale;
        let v_off = slot * value_stride;
        let value_row = unsafe { values.get_unchecked(v_off..v_off + value_head_dim) };

        let out_sub = unsafe { out.get_unchecked_mut(..value_head_dim) };
        if score > max_score {
            let old_scale = if max_score.is_finite() {
                exp_attn(max_score - score)
            } else {
                0.0
            };
            simd::scale_add_f32(out_sub, old_scale, value_row);
            denom = denom * old_scale + 1.0;
            max_score = score;
        } else {
            let weight = exp_attn(score - max_score);
            simd::axpy_f32(out_sub, weight, value_row);
            denom += weight;
        }

        slot += 1;
        if slot == slot_count {
            slot = 0;
        }
    }

    if denom > 0.0 {
        let inv = 1.0 / denom;
        let out_sub = unsafe { out.get_unchecked_mut(..value_head_dim) };
        simd::scale_f32(out_sub, inv);
    }
}

/// Raw context for the parallel attention-sink trampoline, one shard per
/// individual query head (not grouped by KV head like [`AttnHeadsCtx`]):
/// [`online_attention_with_sink`] takes one sink score per head, and
/// gpt-oss's head count is large enough (64 for gpt-oss-20b) that plain
/// contiguous chunking already keeps most `kv_mul`-sized KV-sharing groups
/// inside one shard without needing the grouped kernel's extra complexity.
#[cfg(not(target_family = "wasm"))]
struct AttnHeadsSinkCtx {
    q: *const f32,
    k: *const f32,
    v: *const f32,
    out: *mut f32,
    sinks: *const f32,
    k_len: usize,
    v_len: usize,
    key_stride: usize,
    value_stride: usize,
    slot_count: usize,
    head_dim: usize,
    value_dim: usize,
    kv_mul: usize,
    start_t: usize,
    end_t: usize,
    scale: f32,
}

#[cfg(not(target_family = "wasm"))]
unsafe fn attn_heads_sink_trampoline(ctx: *const (), start: usize, end: usize) {
    // SAFETY: `ctx` is a live &AttnHeadsSinkCtx for the blocking call; each
    // head index reads its own (possibly KV-shared, but read-only) K/V band
    // and writes a disjoint `out` slice, and `out` was fully zeroed by the
    // caller before dispatch.
    unsafe {
        let c = &*(ctx as *const AttnHeadsSinkCtx);
        for h in start..end {
            let kv_h = h / c.kv_mul;
            let q_off = h * c.head_dim;
            let out_off = h * c.value_dim;
            let k_start = kv_h * c.head_dim;
            let v_start = kv_h * c.value_dim;
            let q = std::slice::from_raw_parts(c.q.add(q_off), c.head_dim);
            let keys = std::slice::from_raw_parts(c.k.add(k_start), c.k_len - k_start);
            let values = std::slice::from_raw_parts(c.v.add(v_start), c.v_len - v_start);
            let out = std::slice::from_raw_parts_mut(c.out.add(out_off), c.value_dim);
            online_attention_with_sink(
                q,
                keys,
                values,
                c.key_stride,
                c.value_stride,
                c.slot_count,
                c.head_dim,
                c.value_dim,
                c.start_t,
                c.end_t,
                c.scale,
                *c.sinks.add(h),
                out,
            );
        }
    }
}

/// Parallel (worker-pool) counterpart of the gpt-oss sink-attention loop.
/// `out` is zeroed unconditionally before any head runs: unlike
/// [`online_attention`], [`online_attention_with_sink`] seeds its running max
/// from the (finite) sink score rather than `NEG_INFINITY`, so its first
/// real-token branch can take the additive path and would accumulate onto
/// whatever `out` already held instead of overwriting it.
pub(crate) fn attention_over_heads_with_sink(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    sinks: &[f32],
    key_stride: usize,
    value_stride: usize,
    slot_count: usize,
    head_dim: usize,
    value_dim: usize,
    n_heads: usize,
    kv_mul: usize,
    start_t: usize,
    end_t: usize,
    scale: f32,
    out: &mut [f32],
) {
    for value in out.iter_mut() {
        *value = 0.0;
    }

    #[cfg(not(target_family = "wasm"))]
    let scanned = end_t.saturating_sub(start_t) + 1;
    #[cfg(not(target_family = "wasm"))]
    let work = scanned.saturating_mul(n_heads);
    #[cfg(not(target_family = "wasm"))]
    let threads = crate::simd::num_threads();

    #[cfg(not(target_family = "wasm"))]
    if n_heads > 1
        && threads > 1
        && work >= attention_parallel_min_work(n_heads, kv_mul, head_dim, value_dim, threads)
    {
        let ctx = AttnHeadsSinkCtx {
            q: queries.as_ptr(),
            k: keys.as_ptr(),
            v: values.as_ptr(),
            out: out.as_mut_ptr(),
            sinks: sinks.as_ptr(),
            k_len: keys.len(),
            v_len: values.len(),
            key_stride,
            value_stride,
            slot_count,
            head_dim,
            value_dim,
            kv_mul,
            start_t,
            end_t,
            scale,
        };
        // SAFETY: `ctx` outlives the blocking call; each head writes a
        // disjoint `out` band and only reads (never writes) K/V state.
        unsafe {
            crate::simd::parallel_range(
                n_heads,
                attn_heads_sink_trampoline,
                &ctx as *const AttnHeadsSinkCtx as *const (),
            );
        }
        return;
    }

    // Serial fallback (short contexts, single-threaded, or wasm).
    for h in 0..n_heads {
        let kv_h = h / kv_mul;
        let q_off = h * head_dim;
        let out_off = h * value_dim;
        online_attention_with_sink(
            &queries[q_off..q_off + head_dim],
            &keys[kv_h * head_dim..],
            &values[kv_h * value_dim..],
            key_stride,
            value_stride,
            slot_count,
            head_dim,
            value_dim,
            start_t,
            end_t,
            scale,
            sinks[h],
            &mut out[out_off..out_off + value_dim],
        );
    }
}

#[inline]
/// Runs numerically stable online attention over cached keys and values.
pub(crate) fn online_attention(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    key_stride: usize,
    value_stride: usize,
    slot_count: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    start_t: usize,
    end_t: usize,
    scale: f32,
    out: &mut [f32],
) {
    if slot_count == 0 || start_t > end_t {
        return;
    }
    let mut max_score = f32::NEG_INFINITY;
    let mut denom = 0.0f32;
    let linear_slots = attention_uses_linear_slots(start_t, end_t, slot_count);
    // Keep slot traversal sequential for the common non-wrapping decode
    // range and use a single conditional reset for ring-buffer attention.
    let mut slot = if linear_slots {
        start_t
    } else {
        start_t % slot_count
    };

    for _ in start_t..=end_t {
        let k_off = slot * key_stride;
        let keys_sub = unsafe { keys.get_unchecked(k_off..k_off + key_head_dim) };
        let score = simd::dot_f32(query, keys_sub) * scale;
        let v_off = slot * value_stride;
        let value_row = unsafe { values.get_unchecked(v_off..v_off + value_head_dim) };

        let out_sub = unsafe { out.get_unchecked_mut(..value_head_dim) };
        if score > max_score {
            let old_scale = if max_score.is_finite() {
                exp_attn(max_score - score)
            } else {
                0.0
            };
            simd::scale_add_f32(out_sub, old_scale, value_row);
            denom = denom * old_scale + 1.0;
            max_score = score;
        } else {
            let weight = exp_attn(score - max_score);
            simd::axpy_f32(out_sub, weight, value_row);
            denom += weight;
        }

        slot += 1;
        if slot == slot_count {
            slot = 0;
        }
    }

    if denom > 0.0 {
        let inv = 1.0 / denom;
        let out_sub = unsafe { out.get_unchecked_mut(..value_head_dim) };
        simd::scale_f32(out_sub, inv);
    }
}

/// Minimum attention work (`scanned positions × kv heads`) below which the
/// per-token worker-pool rendezvous costs more than the serial scan saves.
#[cfg(not(target_family = "wasm"))]
const ATTENTION_PARALLEL_MIN_WORK: usize = 4096;

/// Returns the attention-parallelization work threshold, allowing an override
/// via `RUSTY_LLM_ATTN_PARALLEL_MIN_WORK` (set very high to force the serial
/// scan, e.g. for A/B measurement or tuning).
///
/// Ministral 3's exact 32Q/8KV, 128-wide GQA layout has eight independent,
/// expensive grouped scans per layer. On Apple Silicon, distributing those
/// scans is already profitable at short contexts: 26 layers amortize the
/// worker rendezvous within one token. Keep the conservative threshold for
/// every other shape and platform, and honour an explicit environment override
/// before applying the targeted fast path.
#[cfg(not(target_family = "wasm"))]
fn attention_parallel_min_work(
    n_kv_heads: usize,
    kv_mul: usize,
    head_dim: usize,
    value_dim: usize,
    threads: usize,
) -> usize {
    use std::sync::OnceLock;
    static USER_MIN_WORK: OnceLock<Option<usize>> = OnceLock::new();
    if let Some(min_work) = *USER_MIN_WORK.get_or_init(|| {
        std::env::var("RUSTY_LLM_ATTN_PARALLEL_MIN_WORK")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
    }) {
        return min_work;
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if n_kv_heads == 8 && kv_mul == 4 && head_dim == 128 && value_dim == 128 && threads >= 4 {
        return 1;
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    let _ = (n_kv_heads, kv_mul, head_dim, value_dim, threads);

    ATTENTION_PARALLEL_MIN_WORK
}

/// Raw context for the parallel attention-over-heads trampoline. All pointers
/// reference buffers owned by the caller for the (blocking) duration of the
/// `parallel_range` call; each KV head writes a disjoint `out` band.
#[cfg(not(target_family = "wasm"))]
struct AttnHeadsCtx {
    q: *const f32,
    k: *const f32,
    v: *const f32,
    out: *mut f32,
    k_len: usize,
    v_len: usize,
    key_stride: usize,
    value_stride: usize,
    slot_count: usize,
    head_dim: usize,
    value_dim: usize,
    kv_mul: usize,
    start_t: usize,
    end_t: usize,
    scale: f32,
}

#[cfg(not(target_family = "wasm"))]
unsafe fn attn_heads_trampoline(ctx: *const (), start: usize, end: usize) {
    // SAFETY: `ctx` is a live `&AttnHeadsCtx` for the blocking call; each kv_h
    // reads disjoint K/V bands and writes a disjoint `out` slice.
    unsafe {
        let c = &*(ctx as *const AttnHeadsCtx);
        for kv_h in start..end {
            let q_off = kv_h * c.kv_mul * c.head_dim;
            let out_off = kv_h * c.kv_mul * c.value_dim;
            let q = std::slice::from_raw_parts(c.q.add(q_off), c.kv_mul * c.head_dim);
            let k_start = kv_h * c.head_dim;
            let v_start = kv_h * c.value_dim;
            let keys = std::slice::from_raw_parts(c.k.add(k_start), c.k_len - k_start);
            let values = std::slice::from_raw_parts(c.v.add(v_start), c.v_len - v_start);
            let out = std::slice::from_raw_parts_mut(c.out.add(out_off), c.kv_mul * c.value_dim);
            online_attention_grouped(
                q,
                keys,
                values,
                c.key_stride,
                c.value_stride,
                c.slot_count,
                c.head_dim,
                c.value_dim,
                c.kv_mul,
                c.start_t,
                c.end_t,
                c.scale,
                out,
            );
        }
    }
}

/// Runs grouped attention for every KV head, fanning the heads across the
/// worker pool when the scan is large enough to amortize the rendezvous and
/// running serially otherwise. Each head reads a disjoint K/V band and writes a
/// disjoint slice of `out`, so the parallel and serial paths are equivalent.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_over_kv_heads(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    key_stride: usize,
    value_stride: usize,
    slot_count: usize,
    head_dim: usize,
    value_dim: usize,
    n_kv_heads: usize,
    kv_mul: usize,
    start_t: usize,
    end_t: usize,
    scale: f32,
    out: &mut [f32],
) {
    #[cfg(not(target_family = "wasm"))]
    let scanned = end_t.saturating_sub(start_t) + 1;
    #[cfg(not(target_family = "wasm"))]
    let work = scanned.saturating_mul(n_kv_heads);
    #[cfg(not(target_family = "wasm"))]
    let threads = crate::simd::num_threads();

    #[cfg(not(target_family = "wasm"))]
    if n_kv_heads > 1
        && threads > 1
        && work >= attention_parallel_min_work(n_kv_heads, kv_mul, head_dim, value_dim, threads)
    {
        let ctx = AttnHeadsCtx {
            q: queries.as_ptr(),
            k: keys.as_ptr(),
            v: values.as_ptr(),
            out: out.as_mut_ptr(),
            k_len: keys.len(),
            v_len: values.len(),
            key_stride,
            value_stride,
            slot_count,
            head_dim,
            value_dim,
            kv_mul,
            start_t,
            end_t,
            scale,
        };
        // SAFETY: `ctx` outlives the blocking call; each KV head writes a
        // disjoint `out` band and reads disjoint K/V state.
        unsafe {
            crate::simd::parallel_range(
                n_kv_heads,
                attn_heads_trampoline,
                &ctx as *const AttnHeadsCtx as *const (),
            );
        }
        return;
    }

    // Serial fallback (short contexts, single-threaded, or wasm).
    for kv_h in 0..n_kv_heads {
        let q_off = kv_h * kv_mul * head_dim;
        let out_off = kv_h * kv_mul * value_dim;
        online_attention_grouped(
            &queries[q_off..q_off + kv_mul * head_dim],
            &keys[kv_h * head_dim..],
            &values[kv_h * value_dim..],
            key_stride,
            value_stride,
            slot_count,
            head_dim,
            value_dim,
            kv_mul,
            start_t,
            end_t,
            scale,
            &mut out[out_off..out_off + kv_mul * value_dim],
        );
    }
}

#[inline]
/// Runs online attention for all `kv_mul` query heads that share one KV head
/// at once, reading each cached key/value row exactly once instead of once
/// per query head. Under GQA (`kv_mul` > 1) this avoids re-streaming the same
/// K/V cache rows `kv_mul` times, which otherwise evicts them from L1/L2
/// between repeated per-head passes over long contexts.
pub(crate) fn online_attention_grouped(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    key_stride: usize,
    value_stride: usize,
    slot_count: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    kv_mul: usize,
    start_t: usize,
    end_t: usize,
    scale: f32,
    out: &mut [f32],
) {
    debug_assert_eq!(queries.len(), kv_mul * key_head_dim);
    debug_assert_eq!(out.len(), kv_mul * value_head_dim);
    if slot_count == 0 || start_t > end_t {
        return;
    }

    // MHA (one query head per KV head) does not need the grouped scratch
    // arrays or per-token group loop. Reuse the lean scalar-head path.
    if kv_mul == 1 {
        return online_attention(
            queries,
            keys,
            values,
            key_stride,
            value_stride,
            slot_count,
            key_head_dim,
            value_head_dim,
            start_t,
            end_t,
            scale,
            out,
        );
    }

    let mut max_score = [f32::NEG_INFINITY; MAX_KV_MUL];
    let mut denom = [0.0f32; MAX_KV_MUL];
    let kv_mul = kv_mul.min(MAX_KV_MUL);
    online_attention_grouped_scan(
        queries,
        keys,
        values,
        key_stride,
        value_stride,
        slot_count,
        key_head_dim,
        value_head_dim,
        kv_mul,
        start_t,
        end_t,
        scale,
        &mut max_score[..kv_mul],
        &mut denom[..kv_mul],
        out,
    );
    online_attention_grouped_finalize(kv_mul, value_head_dim, &denom[..kv_mul], out);
}

#[inline]
#[allow(clippy::too_many_arguments)]
/// Core of [`online_attention_grouped`], factored out so KV-block-tiled
/// prefill (below) can call it repeatedly over consecutive sub-ranges with
/// state (`max_score`/`denom`/`out`) that persists across calls, instead of
/// being freshly initialized every time. The online-softmax recurrence only
/// depends on visiting positions in increasing order — it does not care
/// whether that happens in one call over `[a, z]` or several calls over
/// `[a, b], [b+1, c], ..., [y+1, z]` with the running state carried between
/// them, so this produces bit-identical results either way. Does **not**
/// normalize `out` (divide by `denom`) — callers finalize once after the
/// position range for a query group is fully scanned, via
/// [`online_attention_grouped_finalize`].
///
/// `kv_mul` here must already be `<= MAX_KV_MUL` (callers slice their
/// scratch arrays to it) and `max_score`/`denom` must have exactly `kv_mul`
/// elements — this only exists to be called from the two trusted call sites
/// above and below, so it skips the public wrapper's own clamping/dispatch.
fn online_attention_grouped_scan(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    key_stride: usize,
    value_stride: usize,
    slot_count: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    kv_mul: usize,
    start_t: usize,
    end_t: usize,
    scale: f32,
    max_score: &mut [f32],
    denom: &mut [f32],
    out: &mut [f32],
) {
    if slot_count == 0 || start_t > end_t {
        return;
    }
    debug_assert_eq!(max_score.len(), kv_mul);
    debug_assert_eq!(denom.len(), kv_mul);
    let linear_slots = attention_uses_linear_slots(start_t, end_t, slot_count);
    let mut slot = if linear_slots {
        start_t
    } else {
        start_t % slot_count
    };

    for _ in start_t..=end_t {
        let k_off = slot * key_stride;
        let keys_sub = unsafe { keys.get_unchecked(k_off..k_off + key_head_dim) };
        let v_off = slot * value_stride;
        let value_row = unsafe { values.get_unchecked(v_off..v_off + value_head_dim) };

        // Qwen3.8-27B is 24Q/4KV GQA, i.e. six query heads share every
        // cached K/V row. Reuse the x4 SIMD path for its first four heads and
        // finish the remaining pair below; this keeps the row hot and avoids
        // falling back to six independent scalar dot/AXPY streams.
        if kv_mul == 4 || kv_mul == 6 {
            let q0 = unsafe { queries.get_unchecked(0..key_head_dim) };
            let q1 = unsafe { queries.get_unchecked(key_head_dim..2 * key_head_dim) };
            let q2 = unsafe { queries.get_unchecked(2 * key_head_dim..3 * key_head_dim) };
            let q3 = unsafe { queries.get_unchecked(3 * key_head_dim..4 * key_head_dim) };
            let scores = simd::dot_f32x4(q0, q1, q2, q3, keys_sub);
            let mut multiplier = [1.0; 4];
            let mut alpha = [0.0; 4];
            for g in 0..4 {
                let score = scores[g] * scale;
                if score > max_score[g] {
                    let old_scale = if max_score[g].is_finite() {
                        exp_attn(max_score[g] - score)
                    } else {
                        0.0
                    };
                    multiplier[g] = old_scale;
                    alpha[g] = 1.0;
                    denom[g] = denom[g] * old_scale + 1.0;
                    max_score[g] = score;
                } else {
                    alpha[g] = exp_attn(score - max_score[g]);
                    denom[g] += alpha[g];
                }
            }
            {
                let (out0, rest) = out.split_at_mut(value_head_dim);
                let (out1, rest) = rest.split_at_mut(value_head_dim);
                let (out2, rest) = rest.split_at_mut(value_head_dim);
                let (out3, _) = rest.split_at_mut(value_head_dim);
                simd::affine_add_f32x4(out0, out1, out2, out3, multiplier, alpha, value_row);
            }
            if kv_mul == 6 {
                for g in 4..6 {
                    let q_sub = unsafe {
                        queries.get_unchecked(g * key_head_dim..g * key_head_dim + key_head_dim)
                    };
                    let score = simd::dot_f32(q_sub, keys_sub) * scale;
                    let out_sub = unsafe {
                        out.get_unchecked_mut(
                            g * value_head_dim..g * value_head_dim + value_head_dim,
                        )
                    };
                    if score > max_score[g] {
                        let old_scale = if max_score[g].is_finite() {
                            exp_attn(max_score[g] - score)
                        } else {
                            0.0
                        };
                        simd::scale_add_f32(out_sub, old_scale, value_row);
                        denom[g] = denom[g] * old_scale + 1.0;
                        max_score[g] = score;
                    } else {
                        let weight = exp_attn(score - max_score[g]);
                        simd::axpy_f32(out_sub, weight, value_row);
                        denom[g] += weight;
                    }
                }
            }
            slot += 1;
            if slot == slot_count {
                slot = 0;
            }
            continue;
        }

        for g in 0..kv_mul {
            let q_sub =
                unsafe { queries.get_unchecked(g * key_head_dim..g * key_head_dim + key_head_dim) };
            let score = simd::dot_f32(q_sub, keys_sub) * scale;
            let out_sub = unsafe {
                out.get_unchecked_mut(g * value_head_dim..g * value_head_dim + value_head_dim)
            };
            if score > max_score[g] {
                let old_scale = if max_score[g].is_finite() {
                    exp_attn(max_score[g] - score)
                } else {
                    0.0
                };
                simd::scale_add_f32(out_sub, old_scale, value_row);
                denom[g] = denom[g] * old_scale + 1.0;
                max_score[g] = score;
            } else {
                let weight = exp_attn(score - max_score[g]);
                simd::axpy_f32(out_sub, weight, value_row);
                denom[g] += weight;
            }
        }

        slot += 1;
        if slot == slot_count {
            slot = 0;
        }
    }
}

#[inline]
/// Normalizes each query group's unnormalized weighted-sum accumulator by
/// its running softmax denominator. Called once after
/// [`online_attention_grouped_scan`] has processed a query group's entire
/// position range (whether via one call or several consecutive ones).
fn online_attention_grouped_finalize(
    kv_mul: usize,
    value_head_dim: usize,
    denom: &[f32],
    out: &mut [f32],
) {
    for g in 0..kv_mul {
        if denom[g] > 0.0 {
            let inv = 1.0 / denom[g];
            let out_sub = unsafe {
                out.get_unchecked_mut(g * value_head_dim..g * value_head_dim + value_head_dim)
            };
            simd::scale_f32(out_sub, inv);
        }
    }
}

/// Upper bound on GQA group size (`n_heads / n_kv_heads`) across supported
/// architectures; backs the fixed-size scratch arrays in
/// `online_attention_grouped` so it stays allocation-free.
const MAX_KV_MUL: usize = 16;

// ─── bf16 KV-cache attention (Standard architecture only) ────────────────────
// Mirrors online_attention / online_attention_grouped / attention_over_kv_heads
// exactly, but reads bf16-stored keys/values instead of f32. The query and the
// running softmax accumulator (out/max_score/denom) stay f32 — only the
// per-position K/V row read from the cache is bf16, widened inline by the
// simd::dot_bf16_f32/axpy_bf16_f32/scale_add_bf16_f32 kernels (and, for the
// common GQA kv_mul == 4 case, simd::dot_bf16x4_f32/affine_add_bf16x4_f32,
// which widen each bf16 element once and reuse it across all four query
// heads). That fused path is not optional: an earlier version of this
// function used only the generic per-head loop, which re-widened the same
// key/value row 4 times per position and measured as a real end-to-end
// decode *regression* versus f32 (paired A/B on Ministral, ~10-25% slower)
// instead of the expected bandwidth win — the redundant widening cost more
// than the halved KV bytes saved. Used only when `KVCache::bf16` is set.

#[inline]
pub(crate) fn online_attention_bf16(
    query: &[f32],
    keys: &[u16],
    values: &[u16],
    key_stride: usize,
    value_stride: usize,
    slot_count: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    start_t: usize,
    end_t: usize,
    scale: f32,
    out: &mut [f32],
) {
    if slot_count == 0 || start_t > end_t {
        return;
    }
    let mut max_score = f32::NEG_INFINITY;
    let mut denom = 0.0f32;
    let linear_slots = attention_uses_linear_slots(start_t, end_t, slot_count);
    let mut slot = if linear_slots {
        start_t
    } else {
        start_t % slot_count
    };

    for _ in start_t..=end_t {
        let k_off = slot * key_stride;
        let keys_sub = unsafe { keys.get_unchecked(k_off..k_off + key_head_dim) };
        let score = simd::dot_bf16_f32(query, keys_sub) * scale;
        let v_off = slot * value_stride;
        let value_row = unsafe { values.get_unchecked(v_off..v_off + value_head_dim) };

        let out_sub = unsafe { out.get_unchecked_mut(..value_head_dim) };
        if score > max_score {
            let old_scale = if max_score.is_finite() {
                exp_attn(max_score - score)
            } else {
                0.0
            };
            simd::scale_add_bf16_f32(out_sub, old_scale, value_row);
            denom = denom * old_scale + 1.0;
            max_score = score;
        } else {
            let weight = exp_attn(score - max_score);
            simd::axpy_bf16_f32(out_sub, weight, value_row);
            denom += weight;
        }

        slot += 1;
        if slot == slot_count {
            slot = 0;
        }
    }

    if denom > 0.0 {
        let inv = 1.0 / denom;
        let out_sub = unsafe { out.get_unchecked_mut(..value_head_dim) };
        simd::scale_f32(out_sub, inv);
    }
}

#[inline]
pub(crate) fn online_attention_grouped_bf16(
    queries: &[f32],
    keys: &[u16],
    values: &[u16],
    key_stride: usize,
    value_stride: usize,
    slot_count: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    kv_mul: usize,
    start_t: usize,
    end_t: usize,
    scale: f32,
    out: &mut [f32],
) {
    debug_assert_eq!(queries.len(), kv_mul * key_head_dim);
    debug_assert_eq!(out.len(), kv_mul * value_head_dim);
    if slot_count == 0 || start_t > end_t {
        return;
    }

    if kv_mul == 1 {
        return online_attention_bf16(
            queries,
            keys,
            values,
            key_stride,
            value_stride,
            slot_count,
            key_head_dim,
            value_head_dim,
            start_t,
            end_t,
            scale,
            out,
        );
    }

    let mut max_score = [f32::NEG_INFINITY; MAX_KV_MUL];
    let mut denom = [0.0f32; MAX_KV_MUL];
    let kv_mul = kv_mul.min(MAX_KV_MUL);
    let linear_slots = attention_uses_linear_slots(start_t, end_t, slot_count);
    let mut slot = if linear_slots {
        start_t
    } else {
        start_t % slot_count
    };

    for _ in start_t..=end_t {
        let k_off = slot * key_stride;
        let keys_sub = unsafe { keys.get_unchecked(k_off..k_off + key_head_dim) };
        let v_off = slot * value_stride;
        let value_row = unsafe { values.get_unchecked(v_off..v_off + value_head_dim) };

        // Mirrors the f32 x4 path. Qwen3.8's six-query-head GQA layout uses
        // it for the first four heads and evaluates the remaining pair below,
        // so each bf16 K/V element is widened once for the SIMD quartet.
        if kv_mul == 4 || kv_mul == 6 {
            let q0 = unsafe { queries.get_unchecked(0..key_head_dim) };
            let q1 = unsafe { queries.get_unchecked(key_head_dim..2 * key_head_dim) };
            let q2 = unsafe { queries.get_unchecked(2 * key_head_dim..3 * key_head_dim) };
            let q3 = unsafe { queries.get_unchecked(3 * key_head_dim..4 * key_head_dim) };
            let scores = simd::dot_bf16x4_f32(q0, q1, q2, q3, keys_sub);
            let mut multiplier = [1.0; 4];
            let mut alpha = [0.0; 4];
            for g in 0..4 {
                let score = scores[g] * scale;
                if score > max_score[g] {
                    let old_scale = if max_score[g].is_finite() {
                        exp_attn(max_score[g] - score)
                    } else {
                        0.0
                    };
                    multiplier[g] = old_scale;
                    alpha[g] = 1.0;
                    denom[g] = denom[g] * old_scale + 1.0;
                    max_score[g] = score;
                } else {
                    alpha[g] = exp_attn(score - max_score[g]);
                    denom[g] += alpha[g];
                }
            }
            {
                let (out0, rest) = out.split_at_mut(value_head_dim);
                let (out1, rest) = rest.split_at_mut(value_head_dim);
                let (out2, rest) = rest.split_at_mut(value_head_dim);
                let (out3, _) = rest.split_at_mut(value_head_dim);
                simd::affine_add_bf16x4_f32(out0, out1, out2, out3, multiplier, alpha, value_row);
            }
            if kv_mul == 6 {
                for g in 4..6 {
                    let q_sub = unsafe {
                        queries.get_unchecked(g * key_head_dim..g * key_head_dim + key_head_dim)
                    };
                    let score = simd::dot_bf16_f32(q_sub, keys_sub) * scale;
                    let out_sub = unsafe {
                        out.get_unchecked_mut(
                            g * value_head_dim..g * value_head_dim + value_head_dim,
                        )
                    };
                    if score > max_score[g] {
                        let old_scale = if max_score[g].is_finite() {
                            exp_attn(max_score[g] - score)
                        } else {
                            0.0
                        };
                        simd::scale_add_bf16_f32(out_sub, old_scale, value_row);
                        denom[g] = denom[g] * old_scale + 1.0;
                        max_score[g] = score;
                    } else {
                        let weight = exp_attn(score - max_score[g]);
                        simd::axpy_bf16_f32(out_sub, weight, value_row);
                        denom[g] += weight;
                    }
                }
            }
            slot += 1;
            if slot == slot_count {
                slot = 0;
            }
            continue;
        }

        for g in 0..kv_mul {
            let q_sub =
                unsafe { queries.get_unchecked(g * key_head_dim..g * key_head_dim + key_head_dim) };
            let score = simd::dot_bf16_f32(q_sub, keys_sub) * scale;
            let out_sub = unsafe {
                out.get_unchecked_mut(g * value_head_dim..g * value_head_dim + value_head_dim)
            };
            if score > max_score[g] {
                let old_scale = if max_score[g].is_finite() {
                    exp_attn(max_score[g] - score)
                } else {
                    0.0
                };
                simd::scale_add_bf16_f32(out_sub, old_scale, value_row);
                denom[g] = denom[g] * old_scale + 1.0;
                max_score[g] = score;
            } else {
                let weight = exp_attn(score - max_score[g]);
                simd::axpy_bf16_f32(out_sub, weight, value_row);
                denom[g] += weight;
            }
        }

        slot += 1;
        if slot == slot_count {
            slot = 0;
        }
    }

    for g in 0..kv_mul {
        if denom[g] > 0.0 {
            let inv = 1.0 / denom[g];
            let out_sub = unsafe {
                out.get_unchecked_mut(g * value_head_dim..g * value_head_dim + value_head_dim)
            };
            simd::scale_f32(out_sub, inv);
        }
    }
}

#[cfg(not(target_family = "wasm"))]
struct AttnHeadsCtxBf16 {
    q: *const f32,
    k: *const u16,
    v: *const u16,
    out: *mut f32,
    k_len: usize,
    v_len: usize,
    key_stride: usize,
    value_stride: usize,
    slot_count: usize,
    head_dim: usize,
    value_dim: usize,
    kv_mul: usize,
    start_t: usize,
    end_t: usize,
    scale: f32,
}

#[cfg(not(target_family = "wasm"))]
unsafe fn attn_heads_trampoline_bf16(ctx: *const (), start: usize, end: usize) {
    // SAFETY: mirrors attn_heads_trampoline — `ctx` is a live &AttnHeadsCtxBf16
    // for the blocking call; each kv_h reads disjoint K/V bands and writes a
    // disjoint `out` slice.
    unsafe {
        let c = &*(ctx as *const AttnHeadsCtxBf16);
        for kv_h in start..end {
            let q_off = kv_h * c.kv_mul * c.head_dim;
            let out_off = kv_h * c.kv_mul * c.value_dim;
            let q = std::slice::from_raw_parts(c.q.add(q_off), c.kv_mul * c.head_dim);
            let k_start = kv_h * c.head_dim;
            let v_start = kv_h * c.value_dim;
            let keys = std::slice::from_raw_parts(c.k.add(k_start), c.k_len - k_start);
            let values = std::slice::from_raw_parts(c.v.add(v_start), c.v_len - v_start);
            let out = std::slice::from_raw_parts_mut(c.out.add(out_off), c.kv_mul * c.value_dim);
            online_attention_grouped_bf16(
                q,
                keys,
                values,
                c.key_stride,
                c.value_stride,
                c.slot_count,
                c.head_dim,
                c.value_dim,
                c.kv_mul,
                c.start_t,
                c.end_t,
                c.scale,
                out,
            );
        }
    }
}

/// bf16-KV-cache counterpart of `attention_over_kv_heads`; same fan-out/serial
/// split, same head-parallelism threshold.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_over_kv_heads_bf16(
    queries: &[f32],
    keys: &[u16],
    values: &[u16],
    key_stride: usize,
    value_stride: usize,
    slot_count: usize,
    head_dim: usize,
    value_dim: usize,
    n_kv_heads: usize,
    kv_mul: usize,
    start_t: usize,
    end_t: usize,
    scale: f32,
    out: &mut [f32],
) {
    #[cfg(not(target_family = "wasm"))]
    let scanned = end_t.saturating_sub(start_t) + 1;
    #[cfg(not(target_family = "wasm"))]
    let work = scanned.saturating_mul(n_kv_heads);
    #[cfg(not(target_family = "wasm"))]
    let threads = crate::simd::num_threads();

    #[cfg(not(target_family = "wasm"))]
    if n_kv_heads > 1
        && threads > 1
        && work >= attention_parallel_min_work(n_kv_heads, kv_mul, head_dim, value_dim, threads)
    {
        let ctx = AttnHeadsCtxBf16 {
            q: queries.as_ptr(),
            k: keys.as_ptr(),
            v: values.as_ptr(),
            out: out.as_mut_ptr(),
            k_len: keys.len(),
            v_len: values.len(),
            key_stride,
            value_stride,
            slot_count,
            head_dim,
            value_dim,
            kv_mul,
            start_t,
            end_t,
            scale,
        };
        // SAFETY: `ctx` outlives the blocking call; each KV head writes a
        // disjoint `out` band and reads disjoint K/V state.
        unsafe {
            crate::simd::parallel_range(
                n_kv_heads,
                attn_heads_trampoline_bf16,
                &ctx as *const AttnHeadsCtxBf16 as *const (),
            );
        }
        return;
    }

    // Serial fallback (short contexts, single-threaded, or wasm).
    for kv_h in 0..n_kv_heads {
        let q_off = kv_h * kv_mul * head_dim;
        let out_off = kv_h * kv_mul * value_dim;
        online_attention_grouped_bf16(
            &queries[q_off..q_off + kv_mul * head_dim],
            &keys[kv_h * head_dim..],
            &values[kv_h * value_dim..],
            key_stride,
            value_stride,
            slot_count,
            head_dim,
            value_dim,
            kv_mul,
            start_t,
            end_t,
            scale,
            &mut out[out_off..out_off + kv_mul * value_dim],
        );
    }
}

/// Raw context for the batched-prefill attention trampoline. Unlike
/// [`AttnHeadsCtx`] (one shard per KV head, called once per token from the
/// prefill batch loop), this shards over the flattened `(token, KV head)`
/// space for the *whole* microbatch in one dispatch: every token's K/V is
/// already written to `cache` by the time this runs (the prefill batch loop
/// splits cache-write and attention into two passes precisely so this can
/// read the whole batch's cache state), so token t's causal window is
/// enforced purely by `attn_window(t)..=pos(t)`, not by ordering. This is
/// what lets a narrow-GQA model (e.g. Qwen2's 2 KV heads) actually use more
/// than `n_kv_heads` worker threads during prefill.
#[cfg(not(target_family = "wasm"))]
struct PrefillAttnBatchCtx {
    q: *const f32,
    k: *const f32,
    v: *const f32,
    out: *mut f32,
    k_len: usize,
    v_len: usize,
    key_stride: usize,
    value_stride: usize,
    slot_count: usize,
    head_dim: usize,
    value_dim: usize,
    n_kv_heads: usize,
    kv_mul: usize,
    start_pos: usize,
    sliding_window: usize,
    scale: f32,
}

#[cfg(not(target_family = "wasm"))]
unsafe fn prefill_attn_batch_range(ctx: *const (), start: usize, end: usize) {
    // SAFETY: `ctx` is a live &PrefillAttnBatchCtx for the blocking call;
    // each (t, kv_h) unit writes a disjoint `out` band and only reads
    // (never writes) K/V state, which the caller guarantees is already
    // fully populated for the whole microbatch.
    unsafe {
        let c = &*(ctx as *const PrefillAttnBatchCtx);
        let q_rows = c.n_kv_heads * c.kv_mul * c.head_dim;
        let attn_dim = c.n_kv_heads * c.kv_mul * c.value_dim;
        for u in start..end {
            let t = u / c.n_kv_heads;
            let kv_h = u % c.n_kv_heads;
            let pos = c.start_pos + t;
            let attn_window = attention_start_pos(pos, c.sliding_window);
            let q_off = t * q_rows + kv_h * c.kv_mul * c.head_dim;
            let out_off = t * attn_dim + kv_h * c.kv_mul * c.value_dim;
            let k_start = kv_h * c.head_dim;
            let v_start = kv_h * c.value_dim;
            let q = std::slice::from_raw_parts(c.q.add(q_off), c.kv_mul * c.head_dim);
            let keys = std::slice::from_raw_parts(c.k.add(k_start), c.k_len - k_start);
            let values = std::slice::from_raw_parts(c.v.add(v_start), c.v_len - v_start);
            let out = std::slice::from_raw_parts_mut(c.out.add(out_off), c.kv_mul * c.value_dim);
            online_attention_grouped(
                q,
                keys,
                values,
                c.key_stride,
                c.value_stride,
                c.slot_count,
                c.head_dim,
                c.value_dim,
                c.kv_mul,
                attn_window,
                pos,
                c.scale,
                out,
            );
        }
    }
}

#[cfg(not(target_family = "wasm"))]
#[allow(clippy::too_many_arguments)]
/// Computes attention for every token in a prefill microbatch, parallelizing
/// over `(token, KV head)` pairs rather than just KV heads. `keys`/`values`
/// must already hold every token's cache entry for this layer (the caller
/// writes the whole batch's K/V before calling this). `queries`/`out` are
/// laid out `[token][kv_head][kv_mul][dim]`, i.e. `b` copies of the same
/// per-token layout the single-token [`attention_over_kv_heads`] uses.
/// Only called from `forward_prefill_batch`, which is itself excluded from
/// wasm (no batch prefill there), hence the matching cfg gate.
pub(crate) fn attention_over_kv_heads_prefill_batch(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    key_stride: usize,
    value_stride: usize,
    slot_count: usize,
    head_dim: usize,
    value_dim: usize,
    n_kv_heads: usize,
    kv_mul: usize,
    b: usize,
    start_pos: usize,
    sliding_window: usize,
    scale: f32,
    out: &mut [f32],
) {
    let q_rows = n_kv_heads * kv_mul * head_dim;
    let attn_dim = n_kv_heads * kv_mul * value_dim;

    #[cfg(not(target_family = "wasm"))]
    let units = b.saturating_mul(n_kv_heads);
    #[cfg(not(target_family = "wasm"))]
    let work: usize = (0..b)
        .map(|t| {
            let pos = start_pos + t;
            (pos - attention_start_pos(pos, sliding_window) + 1) * n_kv_heads
        })
        .sum();
    #[cfg(not(target_family = "wasm"))]
    let threads = crate::simd::num_threads();

    #[cfg(not(target_family = "wasm"))]
    if units > 1
        && threads > 1
        && work >= attention_parallel_min_work(n_kv_heads, kv_mul, head_dim, value_dim, threads)
    {
        let ctx = PrefillAttnBatchCtx {
            q: queries.as_ptr(),
            k: keys.as_ptr(),
            v: values.as_ptr(),
            out: out.as_mut_ptr(),
            k_len: keys.len(),
            v_len: values.len(),
            key_stride,
            value_stride,
            slot_count,
            head_dim,
            value_dim,
            n_kv_heads,
            kv_mul,
            start_pos,
            sliding_window,
            scale,
        };
        // SAFETY: `ctx` outlives the blocking call; each (t, kv_h) unit
        // writes a disjoint `out` band and only reads K/V state.
        unsafe {
            crate::simd::parallel_range(
                units,
                prefill_attn_batch_range,
                &ctx as *const PrefillAttnBatchCtx as *const (),
            );
        }
        return;
    }

    // Serial fallback (short batches/contexts, single-threaded, or wasm).
    for t in 0..b {
        let pos = start_pos + t;
        let attn_window = attention_start_pos(pos, sliding_window);
        for kv_h in 0..n_kv_heads {
            let q_off = t * q_rows + kv_h * kv_mul * head_dim;
            let out_off = t * attn_dim + kv_h * kv_mul * value_dim;
            online_attention_grouped(
                &queries[q_off..q_off + kv_mul * head_dim],
                &keys[kv_h * head_dim..],
                &values[kv_h * value_dim..],
                key_stride,
                value_stride,
                slot_count,
                head_dim,
                value_dim,
                kv_mul,
                attn_window,
                pos,
                scale,
                &mut out[out_off..out_off + kv_mul * value_dim],
            );
        }
    }
}

/// bf16-KV-cache counterpart of [`PrefillAttnBatchCtx`]/
/// [`attention_over_kv_heads_prefill_batch`]; mirrors it exactly except for
/// reading bf16-stored keys/values.
#[cfg(not(target_family = "wasm"))]
struct PrefillAttnBatchCtxBf16 {
    q: *const f32,
    k: *const u16,
    v: *const u16,
    out: *mut f32,
    k_len: usize,
    v_len: usize,
    key_stride: usize,
    value_stride: usize,
    slot_count: usize,
    head_dim: usize,
    value_dim: usize,
    n_kv_heads: usize,
    kv_mul: usize,
    start_pos: usize,
    sliding_window: usize,
    scale: f32,
}

#[cfg(not(target_family = "wasm"))]
unsafe fn prefill_attn_batch_range_bf16(ctx: *const (), start: usize, end: usize) {
    // SAFETY: mirrors prefill_attn_batch_range — `ctx` is a live
    // &PrefillAttnBatchCtxBf16 for the blocking call; each (t, kv_h) unit
    // writes a disjoint `out` band and only reads K/V state.
    unsafe {
        let c = &*(ctx as *const PrefillAttnBatchCtxBf16);
        let q_rows = c.n_kv_heads * c.kv_mul * c.head_dim;
        let attn_dim = c.n_kv_heads * c.kv_mul * c.value_dim;
        for u in start..end {
            let t = u / c.n_kv_heads;
            let kv_h = u % c.n_kv_heads;
            let pos = c.start_pos + t;
            let attn_window = attention_start_pos(pos, c.sliding_window);
            let q_off = t * q_rows + kv_h * c.kv_mul * c.head_dim;
            let out_off = t * attn_dim + kv_h * c.kv_mul * c.value_dim;
            let k_start = kv_h * c.head_dim;
            let v_start = kv_h * c.value_dim;
            let q = std::slice::from_raw_parts(c.q.add(q_off), c.kv_mul * c.head_dim);
            let keys = std::slice::from_raw_parts(c.k.add(k_start), c.k_len - k_start);
            let values = std::slice::from_raw_parts(c.v.add(v_start), c.v_len - v_start);
            let out = std::slice::from_raw_parts_mut(c.out.add(out_off), c.kv_mul * c.value_dim);
            online_attention_grouped_bf16(
                q,
                keys,
                values,
                c.key_stride,
                c.value_stride,
                c.slot_count,
                c.head_dim,
                c.value_dim,
                c.kv_mul,
                attn_window,
                pos,
                c.scale,
                out,
            );
        }
    }
}

#[cfg(not(target_family = "wasm"))]
#[allow(clippy::too_many_arguments)]
/// bf16-KV-cache counterpart of [`attention_over_kv_heads_prefill_batch`].
/// Only called from `forward_prefill_batch`, hence the matching cfg gate.
pub(crate) fn attention_over_kv_heads_prefill_batch_bf16(
    queries: &[f32],
    keys: &[u16],
    values: &[u16],
    key_stride: usize,
    value_stride: usize,
    slot_count: usize,
    head_dim: usize,
    value_dim: usize,
    n_kv_heads: usize,
    kv_mul: usize,
    b: usize,
    start_pos: usize,
    sliding_window: usize,
    scale: f32,
    out: &mut [f32],
) {
    let q_rows = n_kv_heads * kv_mul * head_dim;
    let attn_dim = n_kv_heads * kv_mul * value_dim;

    #[cfg(not(target_family = "wasm"))]
    let units = b.saturating_mul(n_kv_heads);
    #[cfg(not(target_family = "wasm"))]
    let work: usize = (0..b)
        .map(|t| {
            let pos = start_pos + t;
            (pos - attention_start_pos(pos, sliding_window) + 1) * n_kv_heads
        })
        .sum();
    #[cfg(not(target_family = "wasm"))]
    let threads = crate::simd::num_threads();

    #[cfg(not(target_family = "wasm"))]
    if units > 1
        && threads > 1
        && work >= attention_parallel_min_work(n_kv_heads, kv_mul, head_dim, value_dim, threads)
    {
        let ctx = PrefillAttnBatchCtxBf16 {
            q: queries.as_ptr(),
            k: keys.as_ptr(),
            v: values.as_ptr(),
            out: out.as_mut_ptr(),
            k_len: keys.len(),
            v_len: values.len(),
            key_stride,
            value_stride,
            slot_count,
            head_dim,
            value_dim,
            n_kv_heads,
            kv_mul,
            start_pos,
            sliding_window,
            scale,
        };
        // SAFETY: `ctx` outlives the blocking call; each (t, kv_h) unit
        // writes a disjoint `out` band and only reads K/V state.
        unsafe {
            crate::simd::parallel_range(
                units,
                prefill_attn_batch_range_bf16,
                &ctx as *const PrefillAttnBatchCtxBf16 as *const (),
            );
        }
        return;
    }

    // Serial fallback (short batches/contexts, single-threaded, or wasm).
    for t in 0..b {
        let pos = start_pos + t;
        let attn_window = attention_start_pos(pos, sliding_window);
        for kv_h in 0..n_kv_heads {
            let q_off = t * q_rows + kv_h * kv_mul * head_dim;
            let out_off = t * attn_dim + kv_h * kv_mul * value_dim;
            online_attention_grouped_bf16(
                &queries[q_off..q_off + kv_mul * head_dim],
                &keys[kv_h * head_dim..],
                &values[kv_h * value_dim..],
                key_stride,
                value_stride,
                slot_count,
                head_dim,
                value_dim,
                kv_mul,
                attn_window,
                pos,
                scale,
                &mut out[out_off..out_off + kv_mul * value_dim],
            );
        }
    }
}

/// KV positions per block in the tiled prefill attention path below: large
/// enough to amortize the online-softmax bookkeeping per block, small
/// enough that one block's K+V for one KV head stays comfortably
/// L2-resident while every participating token in a tile scans it.
#[cfg(not(target_family = "wasm"))]
const KV_TILE_BLOCK: usize = 128;

/// Tokens per tile in the tiled prefill attention path below, and the
/// compile-time bound for the per-unit stack scratch (`PREFILL_TOKEN_TILE`
/// tokens x `MAX_KV_MUL` groups of `max_score`/`denom`). Chosen so
/// `n_kv_heads * ceil(b / PREFILL_TOKEN_TILE)` gives enough dispatch units
/// to use every thread even for a narrow-GQA model's short prompts, while
/// keeping the per-unit scratch (`PREFILL_TOKEN_TILE * MAX_KV_MUL * 4`
/// bytes x 2 arrays = 8 KiB at these values) a modest stack allocation.
#[cfg(not(target_family = "wasm"))]
const PREFILL_TOKEN_TILE: usize = 64;

/// Raw context for the KV-block-tiled prefill attention trampoline. Scoped
/// to `sliding_window == 0` (plain causal): every token's window starts at
/// position 0, so a KV block's participating tokens are always a *suffix*
/// of the tile (later positions keep needing later blocks; earlier
/// positions stop once the block moves past their own position). A sliding
/// window would additionally need an upper-exclusion check per token per
/// block (the window's lower bound also advances with position), which
/// this does not implement — see `attention_over_kv_heads_prefill_batch`
/// for the windowed case, still used for `sliding_window > 0`.
#[cfg(not(target_family = "wasm"))]
struct PrefillAttnTiledCtx {
    q: *const f32,
    k: *const f32,
    v: *const f32,
    out: *mut f32,
    k_len: usize,
    v_len: usize,
    key_stride: usize,
    value_stride: usize,
    slot_count: usize,
    head_dim: usize,
    value_dim: usize,
    n_kv_heads: usize,
    kv_mul: usize,
    b: usize,
    start_pos: usize,
    scale: f32,
    num_tiles: usize,
}

#[cfg(not(target_family = "wasm"))]
unsafe fn prefill_attn_tiled_range(ctx: *const (), start: usize, end: usize) {
    // SAFETY: `ctx` is a live &PrefillAttnTiledCtx for the blocking call;
    // each (kv_h, tile) unit writes a disjoint `out` band (every token in
    // the tile, for this kv_h's kv_mul query heads) and only reads K/V
    // state, which the caller guarantees is already fully populated for the
    // whole microbatch.
    unsafe {
        let c = &*(ctx as *const PrefillAttnTiledCtx);
        let kv_mul = c.kv_mul.min(MAX_KV_MUL);
        let q_rows = c.n_kv_heads * kv_mul * c.head_dim;
        let attn_dim = c.n_kv_heads * kv_mul * c.value_dim;

        for u in start..end {
            let kv_h = u / c.num_tiles;
            let tile_idx = u % c.num_tiles;
            let tile_start = tile_idx * PREFILL_TOKEN_TILE;
            let tile_end = (tile_start + PREFILL_TOKEN_TILE).min(c.b);
            if tile_start >= tile_end {
                continue;
            }
            let tile_len = tile_end - tile_start;

            // Per-token-in-tile running softmax state, indexed
            // [local_t * kv_mul + g]. Persists across every KV block below
            // for this tile — this is the whole point of the tiling: each
            // block's K/V rows are read once and applied to every token
            // that still needs them, instead of once per token.
            let mut max_score = [f32::NEG_INFINITY; PREFILL_TOKEN_TILE * MAX_KV_MUL];
            let mut denom = [0.0f32; PREFILL_TOKEN_TILE * MAX_KV_MUL];

            let pos_first = c.start_pos + tile_start;
            let pos_last = c.start_pos + tile_end - 1;
            let k_start = kv_h * c.head_dim;
            let v_start = kv_h * c.value_dim;
            let keys = std::slice::from_raw_parts(c.k.add(k_start), c.k_len - k_start);
            let values = std::slice::from_raw_parts(c.v.add(v_start), c.v_len - v_start);

            let mut block_start = 0usize;
            while block_start <= pos_last {
                let block_end = (block_start + KV_TILE_BLOCK - 1).min(pos_last);
                // Causal: token t needs this block iff pos_t >= block_start.
                // pos_t is increasing in local_t, so the participating set
                // is exactly the suffix [first_local_t, tile_len).
                let first_local_t = block_start.saturating_sub(pos_first);
                for local_t in first_local_t..tile_len {
                    let pos_t = pos_first + local_t;
                    let this_end = block_end.min(pos_t);
                    let t = tile_start + local_t;
                    let ms_off = local_t * kv_mul;
                    let q_off = t * q_rows + kv_h * kv_mul * c.head_dim;
                    let out_off = t * attn_dim + kv_h * kv_mul * c.value_dim;
                    let q = std::slice::from_raw_parts(c.q.add(q_off), kv_mul * c.head_dim);
                    let out =
                        std::slice::from_raw_parts_mut(c.out.add(out_off), kv_mul * c.value_dim);
                    online_attention_grouped_scan(
                        q,
                        keys,
                        values,
                        c.key_stride,
                        c.value_stride,
                        c.slot_count,
                        c.head_dim,
                        c.value_dim,
                        kv_mul,
                        block_start,
                        this_end,
                        c.scale,
                        &mut max_score[ms_off..ms_off + kv_mul],
                        &mut denom[ms_off..ms_off + kv_mul],
                        out,
                    );
                }
                block_start = block_end + 1;
            }

            for local_t in 0..tile_len {
                let t = tile_start + local_t;
                let ms_off = local_t * kv_mul;
                let out_off = t * attn_dim + kv_h * kv_mul * c.value_dim;
                let out = std::slice::from_raw_parts_mut(c.out.add(out_off), kv_mul * c.value_dim);
                online_attention_grouped_finalize(
                    kv_mul,
                    c.value_dim,
                    &denom[ms_off..ms_off + kv_mul],
                    out,
                );
            }
        }
    }
}

#[cfg(not(target_family = "wasm"))]
#[allow(clippy::too_many_arguments)]
/// KV-block-tiled counterpart of [`attention_over_kv_heads_prefill_batch`],
/// used when `sliding_window == 0` (plain causal prefill) and there is
/// enough work to be worth it (see `forward_prefill_batch`'s dispatch,
/// which decides that, not this function). Where the untiled version has
/// each (token, KV head) unit sweep the *whole* KV history from position 0
/// independently — O(b^2) total KV bytes read per layer across the
/// microbatch, since every token re-reads every earlier position from
/// scratch — this tiles the KV axis into `KV_TILE_BLOCK`-sized chunks, read
/// ONCE per (KV head, token tile) unit and shared across every token in
/// that tile that still needs them, cutting total KV bytes read to
/// O(max position) per layer instead of O(b x max position).
pub(crate) fn attention_over_kv_heads_prefill_batch_tiled(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    key_stride: usize,
    value_stride: usize,
    slot_count: usize,
    head_dim: usize,
    value_dim: usize,
    n_kv_heads: usize,
    kv_mul: usize,
    b: usize,
    start_pos: usize,
    scale: f32,
    out: &mut [f32],
) {
    if b == 0 || n_kv_heads == 0 {
        return;
    }
    let num_tiles = b.div_ceil(PREFILL_TOKEN_TILE).max(1);
    let ctx = PrefillAttnTiledCtx {
        q: queries.as_ptr(),
        k: keys.as_ptr(),
        v: values.as_ptr(),
        out: out.as_mut_ptr(),
        k_len: keys.len(),
        v_len: values.len(),
        key_stride,
        value_stride,
        slot_count,
        head_dim,
        value_dim,
        n_kv_heads,
        kv_mul,
        b,
        start_pos,
        scale,
        num_tiles,
    };
    // SAFETY: `ctx` outlives the blocking call; each (kv_h, tile) unit
    // writes a disjoint `out` band and only reads K/V state.
    unsafe {
        crate::simd::parallel_range(
            n_kv_heads * num_tiles,
            prefill_attn_tiled_range,
            &ctx as *const PrefillAttnTiledCtx as *const (),
        );
    }
}

/// SiLU activation
#[inline(always)]
/// Computes the SiLU activation.
pub(crate) fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline(always)]
/// Computes the tanh-approximate GELU activation used by Gemma feed-forward blocks.
pub(crate) fn gelu(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044_715 * x * x * x)).tanh())
}

#[inline(always)]
/// Computes the GPT-OSS SwiGLU activation variant.
fn swiglu_gpt_oss(g: f32, u: f32) -> f32 {
    let g = g.min(7.0);
    let u = u.clamp(-7.0, 7.0);
    g * (1.0 / (1.0 + (-1.702 * g).exp())) * (u + 1.0)
}

/// Applies the GPT-OSS rotary embedding layout to query/key vectors.
fn apply_rope_gpt_oss(
    q: &mut [f32],
    k: &mut [f32],
    pos: usize,
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    concentration: f32,
    inv_freq: &[f32],
) {
    debug_assert!(inv_freq.len() >= head_dim / 2);
    for i in (0..head_dim).step_by(2) {
        let angle = pos as f32 * inv_freq[i / 2];
        let (sin_a, cos_a) = angle.sin_cos();
        let cos_a = cos_a * concentration;
        let sin_a = sin_a * concentration;

        for h in 0..n_heads {
            let off = h * head_dim;
            let v0 = q[off + i];
            let v1 = q[off + i + 1];
            q[off + i] = v0 * cos_a - v1 * sin_a;
            q[off + i + 1] = v0 * sin_a + v1 * cos_a;
        }

        for h in 0..n_kv_heads {
            let off = h * head_dim;
            let v0 = k[off + i];
            let v1 = k[off + i + 1];
            k[off + i] = v0 * cos_a - v1 * sin_a;
            k[off + i + 1] = v0 * sin_a + v1 * cos_a;
        }
    }
}

/// Normalizes selected router logits into probabilities.
fn softmax_selected_into(values: &[(usize, f32)], out: &mut Vec<f32>) {
    let max = values
        .iter()
        .map(|(_, v)| *v)
        .fold(f32::NEG_INFINITY, f32::max);
    out.resize(values.len(), 0.0);
    let mut sum = 0.0f32;
    for (out_cell, (_, value)) in out.iter_mut().zip(values.iter()) {
        let exp = (*value - max).exp();
        *out_cell = exp;
        sum += exp;
    }
    if sum > 0.0 {
        for value in out.iter_mut() {
            *value /= sum;
        }
    }
}

/// Keeps only the highest router logits without sorting the full expert list.
fn select_top_logits_into(logits: &[f32], k: usize, out: &mut Vec<(usize, f32)>) {
    out.clear();
    if k == 0 {
        return;
    }
    if out.capacity() < k {
        out.reserve(k - out.capacity());
    }

    for (idx, &value) in logits.iter().enumerate() {
        if out.len() < k {
            out.push((idx, value));
            bubble_up_router_last(out);
        } else if value.total_cmp(&out[out.len() - 1].1).is_gt() {
            let last = out.len() - 1;
            out[last] = (idx, value);
            bubble_up_router_last(out);
        }
    }
}

/// Runs a Mixtral-style routed feed-forward block, leaving the result in
/// `buf.proj` so the caller adds it to the residual exactly as for a dense FFN.
///
/// The reference implementation softmaxes every expert logit, takes the top-k,
/// then renormalises the survivors. Taking the top-k of the raw logits and
/// softmaxing only those is equivalent — softmax is monotonic, so the same
/// experts win, and normalising over the selected subset yields the same
/// weights — while avoiding a full expert-wide softmax on every layer.
fn routed_moe_ffn_into(moe: &RoutedMoeWeights, expert_used_count: usize, buf: &mut DecodeBuffer) {
    moe.router.matvec_into(&buf.xn2, &mut buf.router_logits);
    select_top_logits_into(&buf.router_logits, expert_used_count, &mut buf.top_experts);
    softmax_selected_into(&buf.top_experts, &mut buf.expert_probs);

    buf.moe.fill(0.0);
    for (slot, &(expert, _)) in buf.top_experts.iter().enumerate() {
        moe.gate_experts
            .matvec_expert_into(expert, &buf.xn2, &mut buf.gate);
        moe.up_experts
            .matvec_expert_into(expert, &buf.xn2, &mut buf.up);
        crate::simd::silu_mul_into(&buf.gate, &buf.up, &mut buf.hidden);
        moe.down_experts
            .matvec_expert_into(expert, &buf.hidden, &mut buf.proj);
        let scale = buf.expert_probs[slot];
        for (sum, value) in buf.moe.iter_mut().zip(&buf.proj) {
            *sum += value * scale;
        }
    }
    buf.proj.clone_from(&buf.moe);
}

fn bubble_up_router_last(values: &mut [(usize, f32)]) {
    let mut i = values.len() - 1;
    while i > 0 && values[i].1.total_cmp(&values[i - 1].1).is_gt() {
        values.swap(i, i - 1);
        i -= 1;
    }
}

/// Runs one GPT-OSS decode step and returns logits.
pub fn forward_gpt_oss(
    config: &Config,
    weights: &GptOssWeights,
    cache: &mut KVCache,
    buf: &mut DecodeBuffer,
    token: u32,
    pos: usize,
) -> Vec<f32> {
    let mut logits = Vec::new();
    forward_gpt_oss_into(config, weights, cache, buf, token, pos, &mut logits);
    logits
}

/// Runs one GPT-OSS decode step into a reusable logits buffer.
pub fn forward_gpt_oss_into(
    config: &Config,
    weights: &GptOssWeights,
    cache: &mut KVCache,
    buf: &mut DecodeBuffer,
    token: u32,
    pos: usize,
    logits: &mut Vec<f32>,
) {
    weights
        .token_embd
        .row_into(token as usize, config.dim, &mut buf.x);

    for l in 0..config.n_layers {
        let layer = &weights.layers[l];

        rms_norm_into(&buf.x, &layer.attn_norm, config.rms_norm_eps, &mut buf.xn);
        if !try_quant_matvec3_into(
            &layer.wq, &layer.wk, &layer.wv, &buf.xn, &mut buf.q, &mut buf.k, &mut buf.v,
        ) {
            layer.wq.matvec_into(&buf.xn, &mut buf.q);
            layer.wk.matvec_into(&buf.xn, &mut buf.k);
            layer.wv.matvec_into(&buf.xn, &mut buf.v);
        }
        for i in 0..buf.q.len() {
            buf.q[i] += layer.bq[i];
        }
        for i in 0..buf.k.len() {
            buf.k[i] += layer.bk[i];
        }
        for i in 0..buf.v.len() {
            buf.v[i] += layer.bv[i];
        }

        apply_rope_gpt_oss(
            &mut buf.q,
            &mut buf.k,
            pos,
            config.head_dim,
            config.n_heads,
            config.n_kv_heads,
            buf.rope_gpt_oss_concentration,
            &buf.rope_gpt_oss_inv_freq,
        );

        let kv_k_dim = cache.per_pos_k_dim;
        let kv_v_dim = cache.per_pos_v_dim;
        let kv_k_start = cache.k_offset(pos);
        let kv_v_start = cache.v_offset(pos);
        cache.k[l][kv_k_start..kv_k_start + buf.k.len()].copy_from_slice(&buf.k);
        cache.v[l][kv_v_start..kv_v_start + buf.v.len()].copy_from_slice(&buf.v);

        let scale = 1.0 / (config.head_dim as f32).sqrt();
        let sliding_window = active_sliding_window(config, cache);
        let attn_window = if l % 2 == 0 {
            attention_start_pos(pos, sliding_window)
        } else {
            0
        };

        if !crate::metal::attention_with_sink_into(
            &buf.q,
            &cache.k[l],
            &cache.v[l],
            &layer.sinks,
            &mut buf.attn_out,
            config.n_heads,
            config.kv_mul,
            config.head_dim,
            config.value_dim,
            kv_k_dim,
            kv_v_dim,
            cache.storage_len,
            attn_window,
            pos,
            scale,
        ) {
            attention_over_heads_with_sink(
                &buf.q,
                &cache.k[l],
                &cache.v[l],
                &layer.sinks,
                kv_k_dim,
                kv_v_dim,
                cache.storage_len,
                config.head_dim,
                config.value_dim,
                config.n_heads,
                config.kv_mul,
                attn_window,
                pos,
                scale,
                &mut buf.attn_out,
            );
        }

        layer.wo.matvec_into(&buf.attn_out, &mut buf.proj);
        for i in 0..config.dim {
            buf.x[i] += buf.proj[i] + layer.bo[i];
        }

        rms_norm_into(
            &buf.x,
            &layer.post_attn_norm,
            config.rms_norm_eps,
            &mut buf.xn2,
        );
        layer.gate_inp.matvec_into(&buf.xn2, &mut buf.router_logits);
        for i in 0..buf.router_logits.len() {
            buf.router_logits[i] += layer.gate_inp_bias[i];
        }

        select_top_logits_into(
            &buf.router_logits,
            config.expert_used_count,
            &mut buf.top_experts,
        );
        softmax_selected_into(&buf.top_experts, &mut buf.expert_probs);

        // Evaluate only the routed experts, then accumulate their weighted
        // contributions back into the residual stream.
        for value in buf.moe.iter_mut() {
            *value = 0.0;
        }
        for expert_slot in 0..buf.top_experts.len() {
            let expert_idx = buf.top_experts[expert_slot].0;
            let expert_prob = buf.expert_probs[expert_slot];
            let gate_bias = layer.gate_exps_bias.row_f32(expert_idx, config.hidden_dim);
            let up_bias = layer.up_exps_bias.row_f32(expert_idx, config.hidden_dim);
            let down_bias = layer.down_exps_bias.row_f32(expert_idx, config.dim);

            if !layer.gate_exps.try_matvec_expert_pair_into(
                &layer.up_exps,
                expert_idx,
                &buf.xn2,
                &mut buf.gate,
                &mut buf.up,
            ) {
                layer
                    .gate_exps
                    .matvec_expert_into(expert_idx, &buf.xn2, &mut buf.gate);
                layer
                    .up_exps
                    .matvec_expert_into(expert_idx, &buf.xn2, &mut buf.up);
            }
            for i in 0..config.hidden_dim {
                buf.gate[i] = swiglu_gpt_oss(buf.gate[i] + gate_bias[i], buf.up[i] + up_bias[i]);
            }

            layer
                .down_exps
                .matvec_expert_into(expert_idx, &buf.gate, &mut buf.proj);
            for i in 0..config.dim {
                buf.moe[i] += (buf.proj[i] + down_bias[i]) * expert_prob;
            }
        }

        for i in 0..config.dim {
            buf.x[i] += buf.moe[i];
        }
    }

    rms_norm_into(
        &buf.x,
        &weights.output_norm,
        config.rms_norm_eps,
        &mut buf.xn,
    );
    weights.output.matvec_into(&buf.xn, logits);
}

/// Maps a quantized weight's dtype to the resident decoder's `w_dt` code, or
/// `None` if the resident kernels don't support it (only Q4_K/Q6_K today).
fn resident_dtype_code(dtype: GGMLType) -> Option<u32> {
    match dtype {
        GGMLType::Q4_K => Some(0),
        GGMLType::Q6_K => Some(1),
        _ => None,
    }
}

/// Fingerprints a model+cache combination so the (process-lifetime, one-shot)
/// resident-decoder setup below is never reused across a different model.
fn resident_fingerprint(config: &Config, weights: &ModelWeights, storage_len: usize) -> u64 {
    let ptr = match &weights.token_embd {
        Weight::Quantized { data, .. } => data.as_slice().as_ptr() as usize as u64,
        Weight::F32(v) => v.as_ptr() as usize as u64,
    };
    [
        ptr,
        config.n_layers as u64,
        config.dim as u64,
        config.hidden_dim as u64,
        config.n_heads as u64,
        config.n_kv_heads as u64,
        config.head_dim as u64,
        config.value_dim as u64,
        config.vocab_size as u64,
        storage_len as u64,
    ]
    .into_iter()
    .fold(0xcbf29ce484222325u64, |h, part| {
        (h ^ part).wrapping_mul(0x100000001b3)
    })
}

/// Registers every layer's weights (and the tied output projection) with the
/// experimental GPU-resident decoder. Runs once per process; returns whether
/// setup succeeded so the caller can fall back to the normal per-op path.
fn resident_configure_once(
    config: &Config,
    weights: &ModelWeights,
    cache: &KVCache,
    buf: &DecodeBuffer,
) -> bool {
    let attn_dim = config.n_heads * config.value_dim;
    let expected_cols = [
        config.dim,
        config.dim,
        config.dim,
        attn_dim,
        config.dim,
        config.dim,
        config.hidden_dim,
    ];
    let expected_rows = [
        config.n_heads * config.head_dim,
        config.n_kv_heads * config.head_dim,
        config.n_kv_heads * config.value_dim,
        config.dim,
        config.hidden_dim,
        config.hidden_dim,
        config.dim,
    ];

    if buf.rope_inv_freq.len() < config.head_dim / 2 {
        return false;
    }
    let (output_bytes, output_dt) = match &weights.output {
        Weight::Quantized {
            data,
            dtype,
            rows,
            cols,
        } if *rows == config.vocab_size && *cols == config.dim => {
            match resident_dtype_code(*dtype) {
                Some(dt) => (data.as_slice(), dt),
                None => return false,
            }
        }
        _ => return false,
    };

    if !crate::metal::resident_configure(
        config.n_layers,
        config.dim,
        config.n_heads,
        config.n_kv_heads,
        config.head_dim,
        config.value_dim,
        config.hidden_dim,
        config.vocab_size,
        cache.storage_len,
        config.rms_norm_eps,
        matches!(config.arch.as_str(), "mistral3" | "ministral"),
    ) {
        return false;
    }

    for (l, layer) in weights.layers.iter().enumerate() {
        // The resident Metal kernel implements the unnormalised LLaMA Q/K
        // path. Qwen3 needs the CPU/regular Metal operators below so Q/K
        // normalization is applied before RoPE.
        if !layer.attn_q_norm.is_empty() || !layer.attn_k_norm.is_empty() {
            return false;
        }
        // The resident decoder hard-codes a dense SwiGLU feed-forward block and
        // has no router or expert dispatch, so routed layers must stay on the
        // CPU/regular Metal path.
        if layer.moe.is_some() {
            return false;
        }
        let ws = [
            &layer.wq, &layer.wk, &layer.wv, &layer.wo, &layer.w1, &layer.w3, &layer.w2,
        ];
        let mut w_bytes: [&[u8]; 7] = [&[]; 7];
        let mut w_rows = [0u32; 7];
        let mut w_dt = [0u32; 7];
        for i in 0..7 {
            match ws[i] {
                Weight::Quantized {
                    data,
                    dtype,
                    rows,
                    cols,
                } if *cols == expected_cols[i] && *rows == expected_rows[i] => {
                    match resident_dtype_code(*dtype) {
                        Some(dt) => {
                            w_bytes[i] = data.as_slice();
                            w_rows[i] = *rows as u32;
                            w_dt[i] = dt;
                        }
                        None => return false,
                    }
                }
                _ => return false,
            }
        }
        if layer.attn_norm.len() != config.dim || layer.ffn_norm.len() != config.dim {
            return false;
        }
        let input = crate::metal::ResidentLayerInput {
            w: w_bytes,
            w_rows,
            w_dt,
            attn_norm: &layer.attn_norm,
            ffn_norm: &layer.ffn_norm,
            bq: &layer.bq,
            bk: &layer.bk,
            bv: &layer.bv,
        };
        if !crate::metal::resident_set_layer(l, &input) {
            return false;
        }
    }

    crate::metal::resident_set_output(
        &weights.output_norm,
        output_bytes,
        config.vocab_size,
        output_dt,
        &buf.rope_inv_freq,
    )
}

/// Prepares the resident decoder for this exact model, doing the (fairly
/// expensive) GPU buffer setup at most once per process. A mismatched
/// fingerprint (a different model loaded in the same process) safely
/// disables the fast path instead of reusing another model's GPU buffers.
fn resident_ready(
    config: &Config,
    weights: &ModelWeights,
    cache: &KVCache,
    buf: &DecodeBuffer,
) -> bool {
    if !crate::metal::resident_enabled() || !crate::metal::dispatch_enabled() {
        return false;
    }
    if config.sliding_window != 0
        || config.n_layers == 0
        || config.n_layers > 200
        || config.dim == 0
        || config.dim % 256 != 0
        || config.hidden_dim == 0
        || config.hidden_dim % 256 != 0
        || config.head_dim == 0
        || config.head_dim > 256
        || config.value_dim == 0
        || config.value_dim > 256
        || config.n_kv_heads == 0
        || config.n_heads % config.n_kv_heads != 0
    {
        return false;
    }
    static RESIDENT_READY: OnceLock<(u64, bool)> = OnceLock::new();
    let fingerprint = resident_fingerprint(config, weights, cache.storage_len);
    let (ready_fingerprint, ready) = *RESIDENT_READY.get_or_init(|| {
        (
            fingerprint,
            resident_configure_once(config, weights, cache, buf),
        )
    });
    ready_fingerprint == fingerprint && ready
}

/// Attempts one full token forward pass on the experimental GPU-resident
/// decoder. A global lock serializes calls: the decoder keeps its working
/// buffers and KV cache in static GPU memory, so two forward passes must
/// never run concurrently.
fn resident_forward_attempt(
    config: &Config,
    weights: &ModelWeights,
    cache: &KVCache,
    buf: &DecodeBuffer,
    pos: usize,
    logits: &mut Vec<f32>,
) -> bool {
    if pos >= cache.storage_len || !resident_ready(config, weights, cache, buf) {
        return false;
    }
    let _guard = resident_lock();
    crate::metal::resident_decode_into(&buf.x, pos, config.vocab_size, logits)
}

fn resident_greedy_attempt(
    config: &Config,
    weights: &ModelWeights,
    cache: &KVCache,
    buf: &DecodeBuffer,
    pos: usize,
    recent: &[u32],
    repeat_penalty: f32,
) -> Option<u32> {
    if pos >= cache.storage_len || !resident_ready(config, weights, cache, buf) {
        return None;
    }
    let _guard = resident_lock();
    crate::metal::resident_decode_greedy(&buf.x, pos, recent, repeat_penalty)
}

/// Runs one prompt token on the resident decoder for its KV-cache writes only.
///
/// Prefill discards every position's logits except the last, so this skips the
/// vocabulary projection entirely instead of computing it into a throwaway
/// buffer. On a tied-embedding model like Ministral that projection is the
/// largest single weight read in the graph, so this removes it — plus a
/// full-vocabulary GPU-to-CPU copy — from every prefilled position.
fn resident_prefill_attempt(
    config: &Config,
    weights: &ModelWeights,
    cache: &KVCache,
    buf: &DecodeBuffer,
    pos: usize,
) -> bool {
    if pos >= cache.storage_len || !resident_ready(config, weights, cache, buf) {
        return false;
    }
    let _guard = resident_lock();
    crate::metal::resident_prefill(&buf.x, pos)
}

/// Serializes resident-decoder calls. It keeps its working buffers and KV cache
/// in static GPU memory, so two forward passes must never run concurrently.
fn resident_lock() -> std::sync::MutexGuard<'static, ()> {
    static RESIDENT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    RESIDENT_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Single forward pass for one token at position `pos`
pub fn forward(
    config: &Config,
    weights: &ModelWeights,
    cache: &mut KVCache,
    buf: &mut DecodeBuffer,
    token: u32,
    pos: usize,
) -> Vec<f32> {
    let mut logits = Vec::new();
    forward_into(config, weights, cache, buf, token, pos, &mut logits);
    logits
}

/// Runs one standard transformer decode step into a reusable logits buffer.
pub fn forward_into(
    config: &Config,
    weights: &ModelWeights,
    cache: &mut KVCache,
    buf: &mut DecodeBuffer,
    token: u32,
    pos: usize,
    logits: &mut Vec<f32>,
) {
    let dim = config.dim;
    let head_dim = config.head_dim;
    let _kv_dim = config.kv_dim;
    let kv_mul = config.kv_mul;
    let fused_post_attention_ffn = crate::metal::post_attention_ffn_enabled();

    // Token embedding
    weights.token_embd.row_into(token as usize, dim, &mut buf.x);

    if active_sliding_window(config, cache) == 0
        && resident_forward_attempt(config, weights, cache, buf, pos, logits)
    {
        return;
    }

    buf.rope_sin.resize(buf.rope_inv_freq.len(), 0.0);
    buf.rope_cos.resize(buf.rope_inv_freq.len(), 0.0);
    prepare_rope_sin_cos_into(
        pos,
        &buf.rope_inv_freq,
        &mut buf.rope_sin,
        &mut buf.rope_cos,
    );

    for l in 0..config.n_layers {
        let layer = &weights.layers[l];

        // ── Attention ──
        rms_norm_into(&buf.x, &layer.attn_norm, config.rms_norm_eps, &mut buf.xn);

        if !try_quant_matvec3_into(
            &layer.wq, &layer.wk, &layer.wv, &buf.xn, &mut buf.q, &mut buf.k, &mut buf.v,
        ) {
            layer.wq.matvec_into(&buf.xn, &mut buf.q);
            layer.wk.matvec_into(&buf.xn, &mut buf.k);
            layer.wv.matvec_into(&buf.xn, &mut buf.v);
        }

        add_bias_if_present(&mut buf.q, &layer.bq);
        add_bias_if_present(&mut buf.k, &layer.bk);
        add_bias_if_present(&mut buf.v, &layer.bv);

        apply_qk_norm_if_present(
            &mut buf.q,
            &mut buf.k,
            head_dim,
            config.n_heads,
            config.n_kv_heads,
            &layer.attn_q_norm,
            &layer.attn_k_norm,
            config.rms_norm_eps,
        );

        apply_model_rope_prepared(config, &mut buf.q, &mut buf.k, &buf.rope_sin, &buf.rope_cos);

        // Store KV (keys and values may have different per-head dims)
        let kv_k_dim = cache.per_pos_k_dim;
        let kv_v_dim = cache.per_pos_v_dim;
        cache.write_k(l, pos, &buf.k);
        cache.write_v(l, pos, &buf.v);

        // Multi-head attention with GQA
        let scale = 1.0 / (head_dim as f32).sqrt();
        // Models with sliding-window attention should ignore cache entries that
        // fall outside the active local context.
        let sliding_window = active_sliding_window(config, cache);
        let attn_window = attention_start_pos(pos, sliding_window);

        if cache.bf16 {
            attention_over_kv_heads_bf16(
                &buf.q,
                &cache.k_bf16[l],
                &cache.v_bf16[l],
                kv_k_dim,
                kv_v_dim,
                cache.storage_len,
                head_dim,
                config.value_dim,
                config.n_kv_heads,
                kv_mul,
                attn_window,
                pos,
                scale,
                &mut buf.attn_out,
            );
        } else if !crate::metal::attention_into(
            &buf.q,
            &cache.k[l],
            &cache.v[l],
            &mut buf.attn_out,
            config.n_heads,
            kv_mul,
            head_dim,
            config.value_dim,
            kv_k_dim,
            kv_v_dim,
            cache.storage_len,
            attn_window,
            pos,
            scale,
        ) {
            attention_over_kv_heads(
                &buf.q,
                &cache.k[l],
                &cache.v[l],
                kv_k_dim,
                kv_v_dim,
                cache.storage_len,
                head_dim,
                config.value_dim,
                config.n_kv_heads,
                kv_mul,
                attn_window,
                pos,
                scale,
                &mut buf.attn_out,
            );
        }

        // The fused kernels below read the dense w1/w3/w2 trio, which a routed
        // layer does not have.
        if fused_post_attention_ffn
            && layer.moe.is_none()
            && try_metal_mistral_post_attention_ffn_into(
                &layer.wo,
                &layer.w1,
                &layer.w3,
                &layer.w2,
                &mut buf.x,
                &buf.attn_out,
                &layer.ffn_norm,
                config.rms_norm_eps,
            )
        {
            continue;
        }

        // Output projection + residual
        layer.wo.matvec_into(&buf.attn_out, &mut buf.proj);
        for i in 0..dim {
            buf.x[i] += buf.proj[i];
        }

        // ── FFN (SwiGLU, dense or routed) ──
        rms_norm_into(&buf.x, &layer.ffn_norm, config.rms_norm_eps, &mut buf.xn2);

        if let Some(moe) = &layer.moe {
            routed_moe_ffn_into(moe, config.expert_used_count, buf);
        } else if !try_metal_mistral_ffn_into(
            &layer.w1,
            &layer.w3,
            &layer.w2,
            &buf.xn2,
            &mut buf.proj,
        ) {
            if !try_quant_matvec2_into(&layer.w1, &layer.w3, &buf.xn2, &mut buf.gate, &mut buf.up) {
                layer.w1.matvec_into(&buf.xn2, &mut buf.gate);
                layer.w3.matvec_into(&buf.xn2, &mut buf.up);
            }

            crate::simd::silu_mul_into(
                &buf.gate[..config.hidden_dim],
                &buf.up[..config.hidden_dim],
                &mut buf.hidden,
            );

            layer.w2.matvec_into(&buf.hidden, &mut buf.proj);
        }
        for i in 0..dim {
            buf.x[i] += buf.proj[i];
        }
    }

    // Final norm → logits
    rms_norm_into(
        &buf.x,
        &weights.output_norm,
        config.rms_norm_eps,
        &mut buf.xn,
    );
    weights.output.matvec_into(&buf.xn, logits);
}

/// Runs a standard decoder step and, when the resident backend is active,
/// reduces greedy sampling to one token on the device. A `None` result means
/// the regular logits buffer was populated and should be sampled on the CPU.
pub fn forward_greedy_into(
    config: &Config,
    weights: &ModelWeights,
    cache: &mut KVCache,
    buf: &mut DecodeBuffer,
    token: u32,
    pos: usize,
    recent: &[u32],
    repeat_penalty: f32,
    logits: &mut Vec<f32>,
) -> Option<u32> {
    weights
        .token_embd
        .row_into(token as usize, config.dim, &mut buf.x);
    if active_sliding_window(config, cache) == 0
        && let Some(selected) =
            resident_greedy_attempt(config, weights, cache, buf, pos, recent, repeat_penalty)
    {
        return Some(selected);
    }
    forward_into(config, weights, cache, buf, token, pos, logits);
    None
}

/// Runs one Laguna decoder step. Laguna combines variable-width GQA attention,
/// a positive per-head attention gate, and sparse routed SwiGLU experts.
fn forward_laguna_impl(
    config: &Config,
    weights: &LagunaWeights,
    cache: &mut KVCache,
    buf: &mut DecodeBuffer,
    token: u32,
    pos: usize,
    logits: Option<&mut Vec<f32>>,
) {
    weights
        .token_embd
        .row_into(token as usize, config.dim, &mut buf.x);
    for (layer_index, layer) in weights.layers.iter().enumerate() {
        rms_norm_into(&buf.x, &layer.attn_norm, config.rms_norm_eps, &mut buf.xn);
        if !try_quant_matvec3_into(
            &layer.wq, &layer.wk, &layer.wv, &buf.xn, &mut buf.q, &mut buf.k, &mut buf.v,
        ) {
            layer.wq.matvec_into(&buf.xn, &mut buf.q);
            layer.wk.matvec_into(&buf.xn, &mut buf.k);
            layer.wv.matvec_into(&buf.xn, &mut buf.v);
        }
        rms_norm_heads_in_place(
            &mut buf.q,
            config.head_dim,
            layer.n_heads,
            Some(&layer.q_norm),
            config.rms_norm_eps,
        );
        rms_norm_heads_in_place(
            &mut buf.k,
            config.head_dim,
            config.n_kv_heads,
            Some(&layer.k_norm),
            config.rms_norm_eps,
        );
        apply_rope_qk_neox_partial(
            &mut buf.q,
            &mut buf.k,
            pos,
            config.head_dim,
            layer.rotary_dim,
            layer.n_heads,
            config.n_kv_heads,
            &layer.rope_inv_freq,
        );

        let k_start = cache.k_offset(pos);
        let v_start = cache.v_offset(pos);
        cache.k[layer_index][k_start..k_start + buf.k.len()].copy_from_slice(&buf.k);
        cache.v[layer_index][v_start..v_start + buf.v.len()].copy_from_slice(&buf.v);
        let attn_dim = layer.n_heads * config.value_dim;
        let attn_start = if layer.sliding_window {
            attention_start_pos(pos, config.sliding_window)
        } else {
            0
        };
        attention_over_kv_heads(
            &buf.q,
            &cache.k[layer_index],
            &cache.v[layer_index],
            cache.per_pos_k_dim,
            cache.per_pos_v_dim,
            cache.storage_len,
            config.head_dim,
            config.value_dim,
            config.n_kv_heads,
            layer.n_heads / config.n_kv_heads,
            attn_start,
            pos,
            1.0 / (config.head_dim as f32).sqrt(),
            &mut buf.attn_out[..attn_dim],
        );
        layer.attn_gate.matvec_into(&buf.xn, &mut buf.gate);
        for head in 0..layer.n_heads {
            let gate = softplus(buf.gate[head]);
            let start = head * config.value_dim;
            for value in &mut buf.attn_out[start..start + config.value_dim] {
                *value *= gate;
            }
        }
        layer
            .wo
            .matvec_into(&buf.attn_out[..attn_dim], &mut buf.proj);
        for (residual, projection) in buf.x.iter_mut().zip(&buf.proj) {
            *residual += projection;
        }

        rms_norm_into(&buf.x, &layer.ffn_norm, config.rms_norm_eps, &mut buf.xn2);
        match &layer.mlp {
            LagunaMlpWeights::Dense { gate, up, down } => {
                if !try_quant_matvec2_into(gate, up, &buf.xn2, &mut buf.gate, &mut buf.up) {
                    gate.matvec_into(&buf.xn2, &mut buf.gate);
                    up.matvec_into(&buf.xn2, &mut buf.up);
                }
                crate::simd::silu_mul_into(&buf.gate, &buf.up, &mut buf.hidden);
                down.matvec_into(&buf.hidden, &mut buf.proj);
            }
            LagunaMlpWeights::Sparse(sparse) => {
                let LagunaSparseMlpWeights {
                    router,
                    router_bias,
                    gate_experts,
                    up_experts,
                    down_experts,
                    shared_gate,
                    shared_up,
                    shared_down,
                } = sparse.as_ref();
                if !try_quant_matvec2_into(
                    shared_gate,
                    shared_up,
                    &buf.xn2,
                    &mut buf.gate,
                    &mut buf.up,
                ) {
                    shared_gate.matvec_into(&buf.xn2, &mut buf.gate);
                    shared_up.matvec_into(&buf.xn2, &mut buf.up);
                }
                crate::simd::silu_mul_into(&buf.gate, &buf.up, &mut buf.hidden);
                shared_down.matvec_into(&buf.hidden, &mut buf.moe);

                router.matvec_into(&buf.xn2, &mut buf.router_logits);
                select_laguna_experts(
                    &mut buf.router_logits,
                    router_bias,
                    config.expert_used_count,
                    weights.router_normalize_weights,
                    &mut buf.top_experts,
                    &mut buf.expert_probs,
                );
                for (slot, &(expert, _)) in buf.top_experts.iter().enumerate() {
                    gate_experts.matvec_expert_into(expert, &buf.xn2, &mut buf.gate);
                    up_experts.matvec_expert_into(expert, &buf.xn2, &mut buf.up);
                    crate::simd::silu_mul_into(&buf.gate, &buf.up, &mut buf.hidden);
                    down_experts.matvec_expert_into(expert, &buf.hidden, &mut buf.proj);
                    let scale = buf.expert_probs[slot] * weights.routed_scaling_factor;
                    for (sum, value) in buf.moe.iter_mut().zip(&buf.proj) {
                        *sum += value * scale;
                    }
                }
                buf.proj.clone_from(&buf.moe);
            }
        }
        for (residual, projection) in buf.x.iter_mut().zip(&buf.proj) {
            *residual += projection;
        }
    }
    rms_norm_into(
        &buf.x,
        &weights.output_norm,
        config.rms_norm_eps,
        &mut buf.xn,
    );
    if let Some(logits) = logits {
        weights.output.matvec_into(&buf.xn, logits);
    }
}

pub fn forward_laguna_into(
    config: &Config,
    weights: &LagunaWeights,
    cache: &mut KVCache,
    buf: &mut DecodeBuffer,
    token: u32,
    pos: usize,
    logits: &mut Vec<f32>,
) {
    forward_laguna_impl(config, weights, cache, buf, token, pos, Some(logits));
}

/// Returns Laguna's final normalized residual stream. This shares the decode
/// implementation so embedding callers remain compatible with the decoder.
pub fn forward_hidden_laguna<'a>(
    config: &Config,
    weights: &LagunaWeights,
    cache: &mut KVCache,
    buf: &'a mut DecodeBuffer,
    token: u32,
    pos: usize,
) -> &'a [f32] {
    forward_laguna_impl(config, weights, cache, buf, token, pos, None);
    &buf.xn
}

/// Advances a Laguna cache entry without calculating logits that cannot be
/// sampled until the final prompt token.
pub fn forward_prefill_laguna(
    config: &Config,
    weights: &LagunaWeights,
    cache: &mut KVCache,
    buf: &mut DecodeBuffer,
    token: u32,
    pos: usize,
) {
    forward_laguna_impl(config, weights, cache, buf, token, pos, None);
}

#[inline]
fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        value.exp().ln_1p()
    }
}

/// Qwen3.5's Gated DeltaNet uses true L2 normalisation, unlike RMSNorm.
///
/// The model definition clamps the norm itself (`max(sqrt(sum_sq), eps)`),
/// rather than adding epsilon under the square root. Keeping that distinction
/// matters for exact replay and for zero-valued test vectors.
fn qwen35_l2_normalize_heads(x: &mut [f32], head_dim: usize, heads: usize, eps: f32) {
    if head_dim == 0 || heads == 0 {
        return;
    }
    debug_assert!(x.len() >= head_dim * heads);
    for head in 0..heads {
        let values = &mut x[head * head_dim..(head + 1) * head_dim];
        let denom = simd::dot_f32(values, values).sqrt().max(eps);
        for value in values {
            *value /= denom;
        }
    }
}

#[inline]
fn qwen35_key_head_for_value_head(value_head: usize, key_heads: usize) -> usize {
    value_head % key_heads
}

/// Splits Qwen3.5's full-attention joint projection. Its layout alternates Q
/// and gate per head, unlike common fused projections that put all Q rows
/// before all gate rows.
fn qwen35_split_q_gate(
    q_gate: &[f32],
    head_dim: usize,
    heads: usize,
    q: &mut Vec<f32>,
    gate: &mut Vec<f32>,
) {
    debug_assert_eq!(q_gate.len(), 2 * heads * head_dim);
    q.resize(heads * head_dim, 0.0);
    gate.resize(heads * head_dim, 0.0);
    for head in 0..heads {
        let source = head * 2 * head_dim;
        let target = head * head_dim;
        q[target..target + head_dim].copy_from_slice(&q_gate[source..source + head_dim]);
        gate[target..target + head_dim]
            .copy_from_slice(&q_gate[source + head_dim..source + 2 * head_dim]);
    }
}

/// Advances one value-head of Qwen3.5's Gated DeltaNet associative memory.
///
/// `state` is laid out as `[value_row][key_column]`, the transpose of the
/// mathematical `S[key][value]` notation. It makes each value row contiguous
/// for the two dot products and matches the tensor layout stored in the model.
fn qwen35_delta_head_step(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    decay: f32,
    beta: f32,
    state: &mut [f32],
    out: &mut [f32],
) {
    let width = q.len();
    debug_assert_eq!(k.len(), width);
    debug_assert_eq!(v.len(), width);
    debug_assert_eq!(out.len(), width);
    debug_assert_eq!(state.len(), width * width);

    let q_scale = 1.0 / (width as f32).sqrt();
    let mut value_row = 0usize;

    // Four rows share the same K/Q vectors and recurrence coefficients. The
    // x4 kernels load those shared vectors once per SIMD block. Fusing the
    // update with its following projection cuts the dominant 128x128 state
    // walk from three reads to two while preserving update-before-read order.
    while value_row + 4 <= width {
        let rows = &mut state[value_row * width..(value_row + 4) * width];
        let (row0, rows) = rows.split_at_mut(width);
        let (row1, rows) = rows.split_at_mut(width);
        let (row2, row3) = rows.split_at_mut(width);

        // Scaling every state element before the dot product is algebraically
        // equivalent to scaling the four reductions and avoids an extra full
        // write/read pass over the recurrent memory.
        let mut predicted = simd::dot_f32x4(row0, row1, row2, row3, k);
        for score in &mut predicted {
            *score *= decay;
        }
        let delta = [
            (v[value_row] - predicted[0]) * beta,
            (v[value_row + 1] - predicted[1]) * beta,
            (v[value_row + 2] - predicted[2]) * beta,
            (v[value_row + 3] - predicted[3]) * beta,
        ];
        // The 1/sqrt(d) scale applies after the state update. The fused SIMD
        // primitive projects the register-resident updated rows before they
        // would otherwise need to be loaded from memory a third time.
        let projected = simd::affine_add_dot_f32x4(row0, row1, row2, row3, [decay; 4], delta, k, q);
        for lane in 0..4 {
            out[value_row + lane] = projected[lane] * q_scale;
        }
        value_row += 4;
    }

    // Future model variants may use a width that is not divisible by four.
    // Keep the original operation order for that small tail.
    for value_row in value_row..width {
        let row = &mut state[value_row * width..(value_row + 1) * width];
        for entry in row.iter_mut() {
            *entry *= decay;
        }
        let predicted = simd::dot_f32(row, k);
        let delta = (v[value_row] - predicted) * beta;
        simd::axpy_f32(row, delta, k);
        out[value_row] = simd::dot_f32(row, q) * q_scale;
    }
}

/// Raw context for the independent value-head updates in one Gated DeltaNet
/// layer. Each worker owns complete `[value_dim, key_dim]` state matrices, so
/// head-parallel execution has no synchronisation or reduction overhead.
#[cfg(not(target_family = "wasm"))]
struct Qwen35DeltaHeadsCtx {
    q: *const f32,
    k: *const f32,
    v: *const f32,
    alpha: *const f32,
    beta: *const f32,
    a: *const f32,
    dt_bias: *const f32,
    state: *mut f32,
    out: *mut f32,
    key_heads: usize,
    head_dim: usize,
}

#[cfg(not(target_family = "wasm"))]
unsafe fn qwen35_delta_heads_range(ctx: *const (), start: usize, end: usize) {
    // SAFETY: `parallel_range` blocks for the context lifetime. The caller
    // assigns disjoint head ranges, and each head writes a separate state
    // matrix/output row while Q/K/V and recurrence parameters are immutable.
    let ctx = unsafe { &*(ctx as *const Qwen35DeltaHeadsCtx) };
    for value_head in start..end {
        let key_head = qwen35_key_head_for_value_head(value_head, ctx.key_heads);
        let q =
            unsafe { std::slice::from_raw_parts(ctx.q.add(key_head * ctx.head_dim), ctx.head_dim) };
        let k =
            unsafe { std::slice::from_raw_parts(ctx.k.add(key_head * ctx.head_dim), ctx.head_dim) };
        let v = unsafe {
            std::slice::from_raw_parts(ctx.v.add(value_head * ctx.head_dim), ctx.head_dim)
        };
        let state = unsafe {
            std::slice::from_raw_parts_mut(
                ctx.state.add(value_head * ctx.head_dim * ctx.head_dim),
                ctx.head_dim * ctx.head_dim,
            )
        };
        let out = unsafe {
            std::slice::from_raw_parts_mut(ctx.out.add(value_head * ctx.head_dim), ctx.head_dim)
        };
        let alpha = softplus(unsafe { *ctx.alpha.add(value_head) + *ctx.dt_bias.add(value_head) });
        let decay = (unsafe { *ctx.a.add(value_head) } * alpha).exp();
        let beta = 1.0 / (1.0 + (-unsafe { *ctx.beta.add(value_head) }).exp());
        qwen35_delta_head_step(q, k, v, decay, beta, state, out);
    }
}

/// Evaluates all independent Qwen Gated DeltaNet value heads. This puts the
/// 48 x 128x128 recurrent update on the shared worker pool after the quantized
/// projections complete; it avoids allocating per-token work buffers.
fn qwen35_delta_heads(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    alpha: &[f32],
    beta: &[f32],
    a: &[f32],
    dt_bias: &[f32],
    state: &mut [f32],
    out: &mut [f32],
    key_heads: usize,
    value_heads: usize,
    head_dim: usize,
) {
    debug_assert_eq!(q.len(), key_heads * head_dim);
    debug_assert_eq!(k.len(), key_heads * head_dim);
    debug_assert_eq!(v.len(), value_heads * head_dim);
    debug_assert_eq!(alpha.len(), value_heads);
    debug_assert_eq!(beta.len(), value_heads);
    debug_assert_eq!(a.len(), value_heads);
    debug_assert_eq!(dt_bias.len(), value_heads);
    debug_assert_eq!(state.len(), value_heads * head_dim * head_dim);
    debug_assert_eq!(out.len(), value_heads * head_dim);

    #[cfg(not(target_family = "wasm"))]
    if value_heads >= 4 && crate::simd::num_threads() > 1 {
        let ctx = Qwen35DeltaHeadsCtx {
            q: q.as_ptr(),
            k: k.as_ptr(),
            v: v.as_ptr(),
            alpha: alpha.as_ptr(),
            beta: beta.as_ptr(),
            a: a.as_ptr(),
            dt_bias: dt_bias.as_ptr(),
            state: state.as_mut_ptr(),
            out: out.as_mut_ptr(),
            key_heads,
            head_dim,
        };
        // SAFETY: each callback range owns distinct value-head state/output
        // regions, described by the raw context above.
        unsafe {
            crate::simd::parallel_range(
                value_heads,
                qwen35_delta_heads_range,
                &ctx as *const Qwen35DeltaHeadsCtx as *const (),
            );
        }
        return;
    }

    for value_head in 0..value_heads {
        let key_head = qwen35_key_head_for_value_head(value_head, key_heads);
        let q = &q[key_head * head_dim..(key_head + 1) * head_dim];
        let k = &k[key_head * head_dim..(key_head + 1) * head_dim];
        let v = &v[value_head * head_dim..(value_head + 1) * head_dim];
        let state =
            &mut state[value_head * head_dim * head_dim..(value_head + 1) * head_dim * head_dim];
        let out = &mut out[value_head * head_dim..(value_head + 1) * head_dim];
        let alpha = softplus(alpha[value_head] + dt_bias[value_head]);
        let decay = (a[value_head] * alpha).exp();
        let beta = 1.0 / (1.0 + (-beta[value_head]).exp());
        qwen35_delta_head_step(q, k, v, decay, beta, state, out);
    }
}

/// Runs one causal Qwen3.5 Gated DeltaNet mixer and writes its residual-space
/// projection to `buf.proj`.
fn qwen35_linear_step(
    layer: &Qwen35LinearWeights,
    dims: &SsmDims,
    conv_state: &mut [f32],
    ssm_state: &mut [f32],
    eps: f32,
    buf: &mut DecodeBuffer,
) {
    let key_dim = dims.n_group * dims.d_state;
    let value_dim = dims.d_inner;
    let value_head_dim = dims.head_dim();
    let conv_dim = dims.conv_dim();
    let qkv_dim = 2 * key_dim + value_dim;
    debug_assert_eq!(qkv_dim, conv_dim);
    debug_assert_eq!(layer.conv_w.len(), conv_dim * dims.d_conv);
    debug_assert_eq!(layer.dt_bias.len(), dims.n_head);
    debug_assert_eq!(layer.a.len(), dims.n_head);
    debug_assert_eq!(layer.norm.len(), value_head_dim);
    debug_assert_eq!(conv_state.len(), conv_dim * (dims.d_conv - 1));
    debug_assert_eq!(ssm_state.len(), value_dim * dims.d_state);

    let hidden = &buf.xn;

    if !try_quant_matvec2_into(
        &layer.qkv,
        &layer.gate,
        hidden,
        &mut buf.qwen35_qkv,
        &mut buf.qwen35_gate,
    ) {
        layer.qkv.matvec_into(hidden, &mut buf.qwen35_qkv);
        layer.gate.matvec_into(hidden, &mut buf.qwen35_gate);
    }
    if !try_quant_matvec2_into(
        &layer.alpha,
        &layer.beta,
        hidden,
        &mut buf.qwen35_alpha,
        &mut buf.qwen35_beta,
    ) {
        layer.alpha.matvec_into(hidden, &mut buf.qwen35_alpha);
        layer.beta.matvec_into(hidden, &mut buf.qwen35_beta);
    }
    debug_assert_eq!(buf.qwen35_qkv.len(), qkv_dim);
    debug_assert_eq!(buf.qwen35_gate.len(), value_dim);
    debug_assert_eq!(buf.qwen35_alpha.len(), dims.n_head);
    debug_assert_eq!(buf.qwen35_beta.len(), dims.n_head);

    // Depthwise causal convolution. The shift register is oldest-to-newest,
    // with the current sample occupying the final tap; this mirrors Conv1d's
    // left padding convention used by the Qwen reference implementation.
    let history_len = dims.d_conv - 1;
    for channel in 0..conv_dim {
        let current = buf.qwen35_qkv[channel];
        let taps = &layer.conv_w[channel * dims.d_conv..(channel + 1) * dims.d_conv];
        let history = &mut conv_state[channel * history_len..(channel + 1) * history_len];
        let mut convolved = current * taps[history_len];
        for (past, tap) in history.iter().zip(&taps[..history_len]) {
            convolved += past * tap;
        }
        history.copy_within(1..history_len, 0);
        history[history_len - 1] = current;
        // Keep the convolution result in place until the complete vector is
        // available. Applying SiLU below through the shared SIMD SwiGLU
        // kernel avoids a scalar `expf` call for each of Q/K/V's 10,240
        // channels in every recurrent layer.
        buf.qwen35_qkv[channel] = convolved;
    }
    crate::simd::silu_mul_into(&buf.qwen35_qkv, &buf.qwen35_qkv, &mut buf.attn_out);
    std::mem::swap(&mut buf.qwen35_qkv, &mut buf.attn_out);

    let (q_all, rest) = buf.qwen35_qkv.split_at_mut(key_dim);
    let (k_all, v_all) = rest.split_at_mut(key_dim);
    qwen35_l2_normalize_heads(q_all, dims.d_state, dims.n_group, eps);
    qwen35_l2_normalize_heads(k_all, dims.d_state, dims.n_group, eps);

    buf.attn_out.resize(value_dim, 0.0);
    // Qwen GGUF conversion reorders all value-head-indexed tensors from HF's
    // grouped order into tiled order. Thus V head `h` uses Q/K head `h % Hk`.
    qwen35_delta_heads(
        q_all,
        k_all,
        v_all,
        &buf.qwen35_alpha,
        &buf.qwen35_beta,
        &layer.a,
        &layer.dt_bias,
        ssm_state,
        &mut buf.attn_out,
        dims.n_group,
        dims.n_head,
        value_head_dim,
    );

    // Qwen's ordering is RMSNorm first, then SiLU(z). Reversing the two
    // produces plausible-looking but incorrect generations.
    rms_norm_heads_in_place(
        &mut buf.attn_out,
        value_head_dim,
        dims.n_head,
        Some(&layer.norm),
        eps,
    );
    // Reuse the normal SwiGLU SIMD kernel for DeltaNet's output gate. `hidden`
    // is scratch at this point and will be overwritten by the following FFN.
    crate::simd::silu_mul_into(&buf.qwen35_gate, &buf.attn_out, &mut buf.hidden);
    layer.out.matvec_into(&buf.hidden, &mut buf.proj);
}

/// Runs one gated full-attention Qwen3.5 mixer and writes its residual-space
/// projection to `buf.proj`. Text positions use the same scalar on all three
/// MRoPE axes; vision/video position construction is deliberately separate.
fn qwen35_attention_step(
    config: &Config,
    weights: &Qwen35Weights,
    layer: &Qwen35AttentionWeights,
    cache: &mut KVCache,
    pos: usize,
    buf: &mut DecodeBuffer,
) {
    let query_dim = config.n_heads * config.head_dim;
    let key_dim = config.n_kv_heads * config.head_dim;
    let value_dim = config.n_kv_heads * config.value_dim;

    let hidden = &buf.xn;
    // Qwen3.8-27B-Q4_K_M stores these as Q4_K/Q4_K/Q6_K.  The generic
    // three-projection K-quant kernel shares activation preparation and one
    // worker-pool dispatch across the joint Q+gate, K, and V projections;
    // other quantizations retain their individually equivalent fallback.
    if !try_quant_matvec3_into(
        &layer.q_gate,
        &layer.k,
        &layer.v,
        hidden,
        &mut buf.qwen35_q_gate,
        &mut buf.k,
        &mut buf.v,
    ) {
        layer.q_gate.matvec_into(hidden, &mut buf.qwen35_q_gate);
        layer.k.matvec_into(hidden, &mut buf.k);
        layer.v.matvec_into(hidden, &mut buf.v);
    }
    debug_assert_eq!(buf.qwen35_q_gate.len(), 2 * query_dim);
    qwen35_split_q_gate(
        &buf.qwen35_q_gate,
        config.head_dim,
        config.n_heads,
        &mut buf.q,
        &mut buf.qwen35_gate,
    );
    debug_assert_eq!(buf.k.len(), key_dim);
    debug_assert_eq!(buf.v.len(), value_dim);

    apply_qk_norm_if_present(
        &mut buf.q,
        &mut buf.k,
        config.head_dim,
        config.n_heads,
        config.n_kv_heads,
        &layer.q_norm,
        &layer.k_norm,
        config.rms_norm_eps,
    );
    // Qwen3.5 uses interleaved MRoPE frequencies with the NeoX rotate-half
    // layout. For text all MRoPE axes equal `pos`, leaving this partial helper
    // numerically equivalent to the full multi-axis operation.
    apply_rope_qk_neox_partial(
        &mut buf.q,
        &mut buf.k,
        pos,
        config.head_dim,
        weights.rotary_dim,
        config.n_heads,
        config.n_kv_heads,
        &weights.rope_inv_freq,
    );

    cache.write_k(layer.kv_slot, pos, &buf.k[..key_dim]);
    cache.write_v(layer.kv_slot, pos, &buf.v[..value_dim]);
    buf.attn_out.resize(query_dim, 0.0);
    let scale = 1.0 / (config.head_dim as f32).sqrt();
    if cache.bf16 {
        attention_over_kv_heads_bf16(
            &buf.q,
            &cache.k_bf16[layer.kv_slot],
            &cache.v_bf16[layer.kv_slot],
            cache.per_pos_k_dim,
            cache.per_pos_v_dim,
            cache.storage_len,
            config.head_dim,
            config.value_dim,
            config.n_kv_heads,
            config.kv_mul,
            0,
            pos,
            scale,
            &mut buf.attn_out,
        );
    } else {
        attention_over_kv_heads(
            &buf.q,
            &cache.k[layer.kv_slot],
            &cache.v[layer.kv_slot],
            cache.per_pos_k_dim,
            cache.per_pos_v_dim,
            cache.storage_len,
            config.head_dim,
            config.value_dim,
            config.n_kv_heads,
            config.kv_mul,
            0,
            pos,
            scale,
            &mut buf.attn_out,
        );
    }
    crate::simd::sigmoid_mul_in_place(&mut buf.attn_out, &buf.qwen35_gate);
    layer.out.matvec_into(&buf.attn_out, &mut buf.proj);
}

fn qwen35_resident_quant_parts(weight: &Weight) -> Option<(&[u8], u32, u32, u32)> {
    match weight {
        Weight::Quantized {
            data,
            dtype,
            rows,
            cols,
        } => Some((
            data.as_slice(),
            u32::try_from(*rows).ok()?,
            u32::try_from(*cols).ok()?,
            resident_dtype_code(*dtype)?,
        )),
        Weight::F32(_) => None,
    }
}

// The resident backend wraps quantized bytes with `newBufferWithBytesNoCopy`.
// Only mmap views have a lifetime tied to the Runner independently of the
// individual Weight values; owned Vec-backed weights from `from_gguf_bytes`
// deliberately stay on the CPU path.
fn qwen35_resident_mmap_backed(weights: &Qwen35Weights) -> bool {
    let borrowed = |weight: &Weight| {
        matches!(
            weight,
            Weight::Quantized {
                data: RawTensorData::View { .. },
                dtype: GGMLType::Q4_K | GGMLType::Q6_K,
                ..
            }
        )
    };
    if !borrowed(&weights.output) {
        return false;
    }
    weights.layers.iter().all(|layer| {
        let mixer_ok = match &layer.mixer {
            Qwen35Mixer::Linear(linear) => [
                &linear.qkv,
                &linear.gate,
                &linear.alpha,
                &linear.beta,
                &linear.out,
            ]
            .into_iter()
            .all(&borrowed),
            Qwen35Mixer::Attention(attention) => [
                &attention.q_gate,
                &attention.k,
                &attention.v,
                &attention.out,
            ]
            .into_iter()
            .all(&borrowed),
        };
        mixer_ok
            && [&layer.ffn_gate, &layer.ffn_up, &layer.ffn_down]
                .into_iter()
                .all(&borrowed)
    })
}

/// Reports whether this loaded Qwen can use the no-copy resident Metal graph.
/// Runtime-dependent cache checks remain in `qwen35_resident_ready`.
pub(crate) fn qwen35_resident_eligible(config: &Config, weights: &Qwen35Weights) -> bool {
    crate::metal::resident_enabled()
        && qwen35_resident_mmap_backed(weights)
        && config.sliding_window == 0
        && config.dim != 0
        && config.dim % 256 == 0
        && config.hidden_dim != 0
        && config.hidden_dim % 256 == 0
        && config.head_dim != 0
        && config.head_dim <= 256
        && weights.ssm.d_state == 128
}

fn qwen35_resident_fingerprint(
    config: &Config,
    weights: &Qwen35Weights,
) -> u64 {
    let ptr = match &weights.token_embd {
        Weight::Quantized { data, .. } => data.as_slice().as_ptr() as usize as u64,
        Weight::F32(data) => data.as_ptr() as usize as u64,
    };
    [
        ptr,
        config.n_layers as u64,
        config.dim as u64,
        config.hidden_dim as u64,
        config.n_heads as u64,
        config.n_kv_heads as u64,
        config.head_dim as u64,
        weights.rotary_dim as u64,
        weights.ssm.n_head as u64,
        weights.ssm.n_group as u64,
        weights.ssm.d_state as u64,
    ]
    .into_iter()
    .fold(0xcbf29ce484222325u64, |hash, part| {
        (hash ^ part).wrapping_mul(0x100000001b3)
    })
}

fn qwen35_resident_configure_once(
    config: &Config,
    weights: &Qwen35Weights,
    cache: &KVCache,
) -> bool {
    if !crate::metal::qwen_resident_configure(
        weights.layers.len(),
        config.dim,
        config.hidden_dim,
        config.vocab_size,
        cache.storage_len,
        config.rms_norm_eps,
        config.n_heads,
        config.n_kv_heads,
        config.head_dim,
        weights.rotary_dim,
        weights.ssm.n_head,
        weights.ssm.n_group,
        weights.ssm.d_state,
        weights.ssm.d_conv,
    ) {
        return false;
    }

    let empty_bytes: &[u8] = &[];
    let empty_floats: &[f32] = &[];
    for (layer_index, layer) in weights.layers.iter().enumerate() {
        let mut w = [empty_bytes; 8];
        let mut w_rows = [0u32; 8];
        let mut w_cols = [0u32; 8];
        let mut w_dt = [0u32; 8];
        let (layer_type, slots, conv_w, a, dt_bias, norm, q_norm, k_norm): (
            u32,
            Vec<(usize, &Weight)>,
            &[f32],
            &[f32],
            &[f32],
            &[f32],
            &[f32],
            &[f32],
        ) = match &layer.mixer {
            Qwen35Mixer::Linear(linear) => (
                crate::metal::QWEN_RESIDENT_LAYER_LINEAR,
                vec![
                    (0, &linear.qkv),
                    (1, &linear.gate),
                    (2, &linear.alpha),
                    (3, &linear.beta),
                    (4, &linear.out),
                    (5, &layer.ffn_gate),
                    (6, &layer.ffn_up),
                    (7, &layer.ffn_down),
                ],
                &linear.conv_w,
                &linear.a,
                &linear.dt_bias,
                &linear.norm,
                empty_floats,
                empty_floats,
            ),
            Qwen35Mixer::Attention(attention) => (
                crate::metal::QWEN_RESIDENT_LAYER_ATTENTION,
                vec![
                    (0, &attention.q_gate),
                    (1, &attention.k),
                    (2, &attention.v),
                    (3, &attention.out),
                    (4, &layer.ffn_gate),
                    (5, &layer.ffn_up),
                    (6, &layer.ffn_down),
                ],
                empty_floats,
                empty_floats,
                empty_floats,
                empty_floats,
                &attention.q_norm,
                &attention.k_norm,
            ),
        };
        for (slot, weight) in slots {
            let Some((bytes, rows, cols, dtype)) = qwen35_resident_quant_parts(weight) else {
                return false;
            };
            w[slot] = bytes;
            w_rows[slot] = rows;
            w_cols[slot] = cols;
            w_dt[slot] = dtype;
        }
        let input = crate::metal::QwenResidentLayerInput {
            layer_type,
            w,
            w_rows,
            w_cols,
            w_dt,
            attn_norm: &layer.attn_norm,
            post_norm: &layer.post_attn_norm,
            conv_w,
            a,
            dt_bias,
            norm,
            q_norm,
            k_norm,
        };
        if !crate::metal::qwen_resident_set_layer(layer_index, &input) {
            return false;
        }
    }

    let Some((output, output_rows, output_cols, output_dt)) =
        qwen35_resident_quant_parts(&weights.output)
    else {
        return false;
    };
    output_rows as usize == config.vocab_size
        && output_cols as usize == config.dim
        && crate::metal::qwen_resident_set_output(
            &weights.output_norm,
            output,
            output_rows as usize,
            output_dt,
            &weights.rope_inv_freq,
        )
}

fn qwen35_resident_ready(
    config: &Config,
    weights: &Qwen35Weights,
    cache: &KVCache,
    allow_reconfigure: bool,
) -> bool {
    if !qwen35_resident_eligible(config, weights)
        // Once a cache has entered the resident graph its CPU recurrent/KV
        // state is intentionally unseeded.  Keep that stream on its resident
        // state even if a later session turn requests the CPU policy; falling
        // through at a non-zero position would silently resume from an empty
        // CPU prefix.  A fresh cache still obeys the current dispatch policy.
        || (!cache.qwen_resident_active && !crate::metal::dispatch_enabled())
        || cache.bf16
        // The resident attention cache uses linear position-indexed slots and
        // always scans from position zero.  A runtime sliding-window override
        // turns KVCache storage into a ring, which this graph cannot represent.
        || cache.sliding_window.is_some()
    {
        return false;
    }
    #[derive(Clone, Copy)]
    struct Registration {
        fingerprint: u64,
        capacity: usize,
        ready: bool,
    }
    static REGISTRATION: std::sync::Mutex<Option<Registration>> =
        std::sync::Mutex::new(None);
    let fingerprint = qwen35_resident_fingerprint(config, weights);
    let mut registration = REGISTRATION
        .lock()
        .expect("Qwen resident registration lock poisoned");
    if let Some(current) = *registration
        && current.fingerprint == fingerprint
        && current.capacity >= cache.storage_len
        && current.ready
    {
        return true;
    }
    if !allow_reconfigure {
        return false;
    }
    let ready = qwen35_resident_configure_once(config, weights, cache);
    *registration = Some(Registration {
        fingerprint,
        capacity: cache.storage_len,
        ready,
    });
    ready
}

fn qwen35_resident_decode_attempt(
    config: &Config,
    weights: &Qwen35Weights,
    cache: &mut KVCache,
    buf: &DecodeBuffer,
    pos: usize,
    logits: &mut Vec<f32>,
) -> bool {
    let _guard = resident_lock();
    if pos >= cache.storage_len
        || (!cache.qwen_resident_active && pos != 0)
        || !qwen35_resident_ready(config, weights, cache, pos == 0)
    {
        assert!(
            !cache.qwen_resident_active,
            "Qwen resident Metal backend became unavailable mid-stream"
        );
        return false;
    }
    let ok = crate::metal::qwen_resident_decode_into(&buf.x, pos, config.vocab_size, logits);
    assert!(
        ok || !cache.qwen_resident_active,
        "Qwen resident Metal decode failed after the stream entered GPU state"
    );
    cache.qwen_resident_active = ok;
    ok
}

fn qwen35_resident_prefill_attempt(
    config: &Config,
    weights: &Qwen35Weights,
    cache: &mut KVCache,
    buf: &DecodeBuffer,
    pos: usize,
) -> bool {
    let _guard = resident_lock();
    if pos >= cache.storage_len
        || (!cache.qwen_resident_active && pos != 0)
        || !qwen35_resident_ready(config, weights, cache, pos == 0)
    {
        assert!(
            !cache.qwen_resident_active,
            "Qwen resident Metal backend became unavailable mid-stream"
        );
        return false;
    }
    let ok = crate::metal::qwen_resident_prefill(&buf.x, pos);
    assert!(
        ok || !cache.qwen_resident_active,
        "Qwen resident Metal prefill failed after the stream entered GPU state"
    );
    cache.qwen_resident_active = ok;
    ok
}

fn qwen35_resident_greedy_attempt(
    config: &Config,
    weights: &Qwen35Weights,
    cache: &mut KVCache,
    buf: &DecodeBuffer,
    pos: usize,
    recent: &[u32],
    repeat_penalty: f32,
) -> Option<u32> {
    let _guard = resident_lock();
    if pos >= cache.storage_len
        || (!cache.qwen_resident_active && pos != 0)
        || !qwen35_resident_ready(config, weights, cache, pos == 0)
    {
        assert!(
            !cache.qwen_resident_active,
            "Qwen resident Metal backend became unavailable mid-stream"
        );
        return None;
    }
    let token = crate::metal::qwen_resident_greedy(&buf.x, pos, recent, repeat_penalty);
    assert!(
        token.is_some() || !cache.qwen_resident_active,
        "Qwen resident Metal greedy decode failed after the stream entered GPU state"
    );
    cache.qwen_resident_active = token.is_some();
    token
}

fn forward_qwen35_impl(
    config: &Config,
    weights: &Qwen35Weights,
    cache: &mut KVCache,
    buf: &mut DecodeBuffer,
    token: u32,
    pos: usize,
    logits: Option<&mut Vec<f32>>,
) {
    let profiling = qwen35_profile_enabled();
    let mut profile = Qwen35Profile::default();
    weights
        .token_embd
        .row_into(token as usize, config.dim, &mut buf.x);
    let mut recurrent_index = 0usize;

    for layer in &weights.layers {
        rms_norm_into(&buf.x, &layer.attn_norm, config.rms_norm_eps, &mut buf.xn);
        match &layer.mixer {
            Qwen35Mixer::Linear(linear) => {
                let started = profiling.then(Instant::now);
                let state = cache
                    .ssm
                    .as_mut()
                    .expect("qwen35 requires Gated DeltaNet recurrent state");
                qwen35_linear_step(
                    linear,
                    &weights.ssm,
                    &mut state.conv[recurrent_index],
                    &mut state.ssm[recurrent_index],
                    config.rms_norm_eps,
                    buf,
                );
                recurrent_index += 1;
                if let Some(started) = started {
                    profile.recurrent += started.elapsed();
                }
            }
            Qwen35Mixer::Attention(attn) => {
                let started = profiling.then(Instant::now);
                qwen35_attention_step(config, weights, attn, cache, pos, buf);
                if let Some(started) = started {
                    profile.attention += started.elapsed();
                }
            }
        }
        for (residual, projection) in buf.x.iter_mut().zip(&buf.proj) {
            *residual += projection;
        }

        let ffn_started = profiling.then(Instant::now);
        rms_norm_into(
            &buf.x,
            &layer.post_attn_norm,
            config.rms_norm_eps,
            &mut buf.xn2,
        );
        if !try_quant_matvec2_into(
            &layer.ffn_gate,
            &layer.ffn_up,
            &buf.xn2,
            &mut buf.gate,
            &mut buf.up,
        ) {
            layer.ffn_gate.matvec_into(&buf.xn2, &mut buf.gate);
            layer.ffn_up.matvec_into(&buf.xn2, &mut buf.up);
        }
        crate::simd::silu_mul_into(&buf.gate, &buf.up, &mut buf.hidden);
        layer.ffn_down.matvec_into(&buf.hidden, &mut buf.proj);
        for (residual, projection) in buf.x.iter_mut().zip(&buf.proj) {
            *residual += projection;
        }
        if let Some(started) = ffn_started {
            profile.ffn += started.elapsed();
        }
    }
    debug_assert_eq!(recurrent_index, weights.recurrent_layer_count);
    let output_started = profiling.then(Instant::now);
    rms_norm_into(
        &buf.x,
        &weights.output_norm,
        config.rms_norm_eps,
        &mut buf.xn,
    );
    if let Some(logits) = logits {
        weights.output.matvec_into(&buf.xn, logits);
    }
    if let Some(started) = output_started {
        profile.output += started.elapsed();
    }
    if profiling && let Ok(mut total) = qwen35_profile_store().lock() {
        total.tokens += 1;
        total.recurrent += profile.recurrent;
        total.attention += profile.attention;
        total.ffn += profile.ffn;
        total.output += profile.output;
    }
}

/// Runs one Qwen3.5/Qwen3.8 decode step and writes logits.
pub fn forward_qwen35_into(
    config: &Config,
    weights: &Qwen35Weights,
    cache: &mut KVCache,
    buf: &mut DecodeBuffer,
    token: u32,
    pos: usize,
    logits: &mut Vec<f32>,
) {
    weights
        .token_embd
        .row_into(token as usize, config.dim, &mut buf.x);
    if qwen35_resident_decode_attempt(config, weights, cache, buf, pos, logits) {
        return;
    }
    forward_qwen35_impl(config, weights, cache, buf, token, pos, Some(logits));
}

/// Runs a Qwen3.5/Qwen3.8 decode step with on-device greedy reduction when the
/// complete resident Metal graph is available. A `None` result leaves logits
/// populated through the regular CPU/per-operation path.
#[allow(clippy::too_many_arguments)]
pub fn forward_greedy_qwen35_into(
    config: &Config,
    weights: &Qwen35Weights,
    cache: &mut KVCache,
    buf: &mut DecodeBuffer,
    token: u32,
    pos: usize,
    recent: &[u32],
    repeat_penalty: f32,
    logits: &mut Vec<f32>,
) -> Option<u32> {
    weights
        .token_embd
        .row_into(token as usize, config.dim, &mut buf.x);
    if let Some(token) = qwen35_resident_greedy_attempt(
        config,
        weights,
        cache,
        buf,
        pos,
        recent,
        repeat_penalty,
    ) {
        return Some(token);
    }
    forward_qwen35_impl(config, weights, cache, buf, token, pos, Some(logits));
    None
}

/// Runs one Qwen3.5/Qwen3.8 step and returns the final normalised hidden state.
pub fn forward_hidden_qwen35<'a>(
    config: &Config,
    weights: &Qwen35Weights,
    cache: &mut KVCache,
    buf: &'a mut DecodeBuffer,
    token: u32,
    pos: usize,
) -> &'a [f32] {
    forward_qwen35_impl(config, weights, cache, buf, token, pos, None);
    &buf.xn
}

/// Advances a Qwen3.5/Qwen3.8 prompt token without producing logits.
pub fn forward_prefill_qwen35(
    config: &Config,
    weights: &Qwen35Weights,
    cache: &mut KVCache,
    buf: &mut DecodeBuffer,
    token: u32,
    pos: usize,
) {
    weights
        .token_embd
        .row_into(token as usize, config.dim, &mut buf.x);
    if qwen35_resident_prefill_attempt(config, weights, cache, buf, pos) {
        return;
    }
    forward_qwen35_impl(config, weights, cache, buf, token, pos, None);
}

/// Runs Qwen's embedded one-step draft head. The head receives the raw final
/// trunk residual for position `t` and the embedding of the already selected
/// token `t + 1`, then predicts `t + 2`.
///
/// A one-step head has a single attention item, so its softmax is exactly one:
/// each query head receives its grouped value row. Avoiding a synthetic cache
/// and unused Q/K dot products keeps this path both exact and inexpensive.
pub fn forward_qwen35_mtp_into(
    config: &Config,
    weights: &Qwen35Weights,
    mtp: &Qwen35MtpWeights,
    token: u32,
    trunk_hidden: &[f32],
    buf: &mut DecodeBuffer,
    next_hidden: &mut Vec<f32>,
    logits: &mut Vec<f32>,
) {
    let dim = config.dim;
    let token_embd = mtp.token_embd.as_ref().unwrap_or(&weights.token_embd);
    token_embd.row_into(token as usize, dim, &mut buf.x);
    rms_norm_into(
        &buf.x,
        &mtp.embedding_norm,
        config.rms_norm_eps,
        &mut buf.xn,
    );
    rms_norm_into(
        trunk_hidden,
        &mtp.hidden_norm,
        config.rms_norm_eps,
        &mut buf.xn2,
    );

    buf.ple_inputs.resize(2 * dim, 0.0);
    buf.ple_inputs[..dim].copy_from_slice(&buf.xn);
    buf.ple_inputs[dim..].copy_from_slice(&buf.xn2);
    mtp.eh_proj.matvec_into(&buf.ple_inputs, &mut buf.x);

    rms_norm_into(&buf.x, &mtp.attn_norm, config.rms_norm_eps, &mut buf.xn);
    if !try_quant_matvec2_into(
        &mtp.attention.q_gate,
        &mtp.attention.v,
        &buf.xn,
        &mut buf.qwen35_q_gate,
        &mut buf.v,
    ) {
        mtp.attention
            .q_gate
            .matvec_into(&buf.xn, &mut buf.qwen35_q_gate);
        mtp.attention.v.matvec_into(&buf.xn, &mut buf.v);
    }
    qwen35_split_q_gate(
        &buf.qwen35_q_gate,
        config.head_dim,
        config.n_heads,
        &mut buf.q,
        &mut buf.qwen35_gate,
    );
    let query_dim = config.n_heads * config.value_dim;
    let kv_mul = config.kv_mul;
    buf.attn_out.resize(query_dim, 0.0);
    for head in 0..config.n_heads {
        let kv_head = head / kv_mul;
        let source = kv_head * config.value_dim;
        let target = head * config.value_dim;
        buf.attn_out[target..target + config.value_dim]
            .copy_from_slice(&buf.v[source..source + config.value_dim]);
    }
    simd::sigmoid_mul_in_place(&mut buf.attn_out, &buf.qwen35_gate);
    mtp.attention.out.matvec_into(&buf.attn_out, &mut buf.proj);
    for index in 0..dim {
        buf.x[index] += buf.proj[index];
    }

    rms_norm_into(
        &buf.x,
        &mtp.post_attn_norm,
        config.rms_norm_eps,
        &mut buf.xn2,
    );
    if !try_quant_matvec2_into(
        &mtp.ffn_gate,
        &mtp.ffn_up,
        &buf.xn2,
        &mut buf.gate,
        &mut buf.up,
    ) {
        mtp.ffn_gate.matvec_into(&buf.xn2, &mut buf.gate);
        mtp.ffn_up.matvec_into(&buf.xn2, &mut buf.up);
    }
    simd::silu_mul_into(&buf.gate, &buf.up, &mut buf.hidden);
    mtp.ffn_down.matvec_into(&buf.hidden, &mut buf.proj);
    for index in 0..dim {
        buf.x[index] += buf.proj[index];
    }

    next_hidden.clear();
    next_hidden.extend_from_slice(&buf.x);
    rms_norm_into(&buf.x, &mtp.head_norm, config.rms_norm_eps, &mut buf.xn);
    mtp.output
        .as_ref()
        .unwrap_or(&weights.output)
        .matvec_into(&buf.xn, logits);
}

fn select_laguna_experts(
    logits: &mut [f32],
    correction_bias: &[f32],
    top_k: usize,
    normalize: bool,
    selected: &mut Vec<(usize, f32)>,
    probabilities: &mut Vec<f32>,
) {
    selected.clear();
    for (expert, logit) in logits.iter_mut().enumerate() {
        *logit = 1.0 / (1.0 + (-*logit).exp());
        let corrected = *logit + correction_bias.get(expert).copied().unwrap_or(0.0);
        if selected.len() < top_k {
            selected.push((expert, corrected));
            selected.sort_by(|a, b| b.1.total_cmp(&a.1));
        } else if corrected
            > selected
                .last()
                .map(|entry| entry.1)
                .unwrap_or(f32::INFINITY)
        {
            selected.pop();
            selected.push((expert, corrected));
            selected.sort_by(|a, b| b.1.total_cmp(&a.1));
        }
    }
    probabilities.clear();
    probabilities.extend(selected.iter().map(|(expert, _)| logits[*expert]));
    if normalize {
        let total: f32 = probabilities.iter().sum();
        if total > 0.0 {
            for probability in probabilities {
                *probability /= total;
            }
        }
    }
}

/// Forward pass for Gemma-4 models (initial implementation mirroring the
/// standard LLaMA-style forward). Bias terms are currently ignored when
/// missing; the loader warns about absent tensors.
/// Squared ReLU, the feed-forward activation used throughout Nemotron-H.
#[inline]
fn relu2(value: f32) -> f32 {
    let clamped = value.max(0.0);
    clamped * clamped
}

/// Advances one Mamba-2 mixer by a single token, updating its recurrent state
/// in place and writing the block output to `out`.
///
/// The fused projection splits into gate/x/B/C/dt, a depthwise causal convolution runs
/// over x, B and C together, the scan applies a per-head scalar decay, and the
/// result is gated by `silu(z)` *before* a per-group RMSNorm — that ordering is
/// load-bearing and the reverse produces plausible-looking garbage.
fn nemotron_mamba2_step(
    layer: &Mamba2LayerWeights,
    dims: &SsmDims,
    conv_state: &mut [f32],
    ssm_state: &mut [f32],
    hidden: &[f32],
    eps: f32,
    scratch: &mut Mamba2Scratch,
    out: &mut Vec<f32>,
) {
    layer.in_proj.matvec_into(hidden, &mut scratch.projected);
    nemotron_mamba2_core(
        layer,
        dims,
        conv_state,
        ssm_state,
        &scratch.projected,
        eps,
        &mut scratch.convolved,
        &mut scratch.y,
    );
    layer.out_proj.matvec_into(&scratch.y, out);
}

/// Applies the causal convolution and selective state update to one already
/// projected activation. Keeping the projection separate lets verification
/// reuse each quantized weight row across several consecutive tokens.
#[allow(clippy::too_many_arguments)]
fn nemotron_mamba2_core(
    layer: &Mamba2LayerWeights,
    dims: &SsmDims,
    conv_state: &mut [f32],
    ssm_state: &mut [f32],
    projected: &[f32],
    eps: f32,
    convolved: &mut Vec<f32>,
    y: &mut Vec<f32>,
) {
    let d_inner = dims.d_inner;
    let d_state = dims.d_state;
    let d_conv = dims.d_conv;
    let n_group = dims.n_group;
    let n_head = dims.n_head;
    let head_dim = dims.head_dim();
    let conv_dim = dims.conv_dim();

    let (z, rest) = projected.split_at(d_inner);
    let (xbc, dt) = rest.split_at(conv_dim);

    // ── Depthwise causal convolution over x, B and C ──
    convolved.resize(conv_dim, 0.0);
    let window = d_conv.saturating_sub(1);
    for channel in 0..conv_dim {
        let taps = &layer.conv_w[channel * d_conv..channel * d_conv + d_conv];
        let history = &mut conv_state[channel * window..channel * window + window];
        let mut acc = layer.conv_b.get(channel).copied().unwrap_or(0.0);
        for (past, tap) in history.iter().zip(taps.iter()) {
            acc += past * tap;
        }
        // The current sample occupies the newest tap position.
        acc += xbc[channel] * taps[window];
        // Roll the shift register forward, newest last.
        if window > 0 {
            history.copy_within(1..window, 0);
            history[window - 1] = xbc[channel];
        }
        convolved[channel] = silu(acc);
    }

    let (xs, bc) = convolved.split_at(d_inner);
    let (b_all, c_all) = bc.split_at(n_group * d_state);

    // ── Selective scan, one step ──
    y.resize(d_inner, 0.0);
    let heads_per_group = n_head / n_group.max(1);
    for head in 0..n_head {
        let dt_eff = softplus(dt[head] + layer.dt_bias[head]);
        // `a` is already stored as -exp(A_log), so no negation here.
        let decay = (dt_eff * layer.a[head]).exp();
        let group = head / heads_per_group.max(1);
        let b = &b_all[group * d_state..group * d_state + d_state];
        let c = &c_all[group * d_state..group * d_state + d_state];
        for channel in 0..head_dim {
            let ii = channel + head * head_dim;
            let x_dt = xs[ii] * dt_eff;
            let state = &mut ssm_state[ii * d_state..ii * d_state + d_state];
            let mut sum = 0.0f32;
            for i in 0..d_state {
                let updated = state[i] * decay + b[i] * x_dt;
                sum += updated * c[i];
                state[i] = updated;
            }
            // Skip connection, per head and broadcast across its channels.
            y[ii] = sum + layer.d[head] * xs[ii];
        }
    }

    // ── Gate, then grouped RMSNorm ──
    for (value, gate) in y.iter_mut().zip(z.iter()) {
        *value *= silu(*gate);
    }
    if !layer.norm.is_empty() {
        let group_size = d_inner / n_group.max(1);
        for group in 0..n_group.max(1) {
            let span = &mut y[group * group_size..group * group_size + group_size];
            let weights = &layer.norm[group * group_size..group * group_size + group_size];
            let mean_square = span.iter().map(|v| v * v).sum::<f32>() / group_size.max(1) as f32;
            let inv = 1.0 / (mean_square + eps).sqrt();
            for (value, weight) in span.iter_mut().zip(weights.iter()) {
                *value = *value * inv * weight;
            }
        }
    }
}

/// Reusable scratch buffers for the Mamba-2 mixer.
#[derive(Default)]
struct Mamba2Scratch {
    projected: Vec<f32>,
    convolved: Vec<f32>,
    y: Vec<f32>,
}

/// Runs a Nemotron-H routed feed-forward block into `out`.
///
/// Routing is sigmoid-scored with an additive bias used only for *selecting*
/// experts; blended weights use the unbiased probabilities. Experts have no
/// gate projection.
fn nemotron_moe_ffn_into(
    moe: &NemotronMoeWeights,
    weights: &NemotronHWeights,
    expert_used_count: usize,
    buf: &mut DecodeBuffer,
) {
    moe.router.matvec_into(&buf.xn, &mut buf.router_logits);
    select_laguna_experts(
        &mut buf.router_logits,
        &moe.router_bias,
        expert_used_count,
        weights.router_normalize_weights,
        &mut buf.top_experts,
        &mut buf.expert_probs,
    );

    // A zero scale means the GGUF omitted the key; treat that as unscaled
    // rather than silently annihilating every routed contribution.
    let scale = if weights.routed_scaling_factor > 0.0 {
        weights.routed_scaling_factor
    } else {
        1.0
    };

    buf.moe.fill(0.0);
    for (slot, &(expert, _)) in buf.top_experts.iter().enumerate() {
        moe.up_experts
            .matvec_expert_into(expert, &buf.xn, &mut buf.up);
        for value in buf.up.iter_mut() {
            *value = relu2(*value);
        }
        moe.down_experts
            .matvec_expert_into(expert, &buf.up, &mut buf.proj);
        let weight = buf.expert_probs[slot] * scale;
        for (sum, value) in buf.moe.iter_mut().zip(&buf.proj) {
            *sum += value * weight;
        }
    }

    // The always-on shared expert runs alongside the routed ones.
    moe.shared_up.matvec_into(&buf.xn, &mut buf.up);
    for value in buf.up.iter_mut() {
        *value = relu2(*value);
    }
    moe.shared_down.matvec_into(&buf.up, &mut buf.proj);
    for (value, routed) in buf.proj.iter_mut().zip(&buf.moe) {
        *value += routed;
    }
}

/// Runs one Nemotron-H decode step. Each block applies exactly one mixer —
/// Mamba-2, attention, or a feed-forward network — to its normalised residual.
fn forward_nemotron_h_impl(
    config: &Config,
    weights: &NemotronHWeights,
    cache: &mut KVCache,
    buf: &mut DecodeBuffer,
    token: u32,
    pos: usize,
    logits: Option<&mut Vec<f32>>,
) {
    let dim = config.dim;
    weights.token_embd.row_into(token as usize, dim, &mut buf.x);

    let mut scratch = Mamba2Scratch::default();
    let mut recurrent_index = 0usize;

    for layer in &weights.layers {
        rms_norm_into(&buf.x, &layer.attn_norm, config.rms_norm_eps, &mut buf.xn);

        match &layer.mixer {
            NemotronMixer::Mamba2(mamba) => {
                let state = cache
                    .ssm
                    .as_mut()
                    .expect("hybrid model requires recurrent state");
                let conv = &mut state.conv[recurrent_index];
                let ssm = &mut state.ssm[recurrent_index];
                nemotron_mamba2_step(
                    mamba,
                    &weights.ssm,
                    conv,
                    ssm,
                    &buf.xn,
                    config.rms_norm_eps,
                    &mut scratch,
                    &mut buf.proj,
                );
                recurrent_index += 1;
            }
            NemotronMixer::Attention(attn) => {
                attn.wq.matvec_into(&buf.xn, &mut buf.q);
                attn.wk.matvec_into(&buf.xn, &mut buf.k);
                attn.wv.matvec_into(&buf.xn, &mut buf.v);

                // Nemotron-H attention is position-free: no RoPE is applied.
                let head_dim = config.head_dim;
                let k_dim = attn.n_kv_heads * head_dim;
                let v_dim = attn.n_kv_heads * head_dim;
                cache.write_k(attn.kv_slot, pos, &buf.k[..k_dim]);
                cache.write_v(attn.kv_slot, pos, &buf.v[..v_dim]);

                let kv_mul = attn.n_heads / attn.n_kv_heads.max(1);
                let scale = 1.0 / (head_dim as f32).sqrt();
                attention_over_kv_heads(
                    &buf.q,
                    &cache.k[attn.kv_slot],
                    &cache.v[attn.kv_slot],
                    k_dim,
                    v_dim,
                    cache.storage_len,
                    head_dim,
                    head_dim,
                    attn.n_kv_heads,
                    kv_mul,
                    0,
                    pos,
                    scale,
                    &mut buf.attn_out,
                );
                let attn_dim = attn.n_heads * head_dim;
                attn.wo
                    .matvec_into(&buf.attn_out[..attn_dim], &mut buf.proj);
                add_bias_if_present(&mut buf.proj, &attn.bo);
            }
            NemotronMixer::Moe(moe) => {
                nemotron_moe_ffn_into(moe, weights, config.expert_used_count, buf);
            }
            NemotronMixer::DenseFfn(ffn) => {
                ffn.up.matvec_into(&buf.xn, &mut buf.up);
                add_bias_if_present(&mut buf.up, &ffn.up_bias);
                for value in buf.up.iter_mut() {
                    *value = relu2(*value);
                }
                ffn.down.matvec_into(&buf.up, &mut buf.proj);
                add_bias_if_present(&mut buf.proj, &ffn.down_bias);
            }
        }

        for (residual, projection) in buf.x.iter_mut().zip(&buf.proj) {
            *residual += projection;
        }
    }

    rms_norm_into(
        &buf.x,
        &weights.output_norm,
        config.rms_norm_eps,
        &mut buf.xn,
    );
    if let Some(logits) = logits {
        weights.output.matvec_into(&buf.xn, logits);
    }
}

/// Runs one Nemotron-H decode step and writes logits.
pub fn forward_nemotron_h_into(
    config: &Config,
    weights: &NemotronHWeights,
    cache: &mut KVCache,
    buf: &mut DecodeBuffer,
    token: u32,
    pos: usize,
    logits: &mut Vec<f32>,
) {
    forward_nemotron_h_impl(config, weights, cache, buf, token, pos, Some(logits));
}

/// Runs one Nemotron-H step and returns the final normalised hidden state.
pub fn forward_hidden_nemotron_h<'a>(
    config: &Config,
    weights: &NemotronHWeights,
    cache: &mut KVCache,
    buf: &'a mut DecodeBuffer,
    token: u32,
    pos: usize,
) -> &'a [f32] {
    forward_nemotron_h_impl(config, weights, cache, buf, token, pos, None);
    &buf.xn
}

/// Advances a Nemotron-H model by one prompt token without producing logits.
pub fn forward_prefill_nemotron_h(
    config: &Config,
    weights: &NemotronHWeights,
    cache: &mut KVCache,
    buf: &mut DecodeBuffer,
    token: u32,
    pos: usize,
) {
    forward_nemotron_h_impl(config, weights, cache, buf, token, pos, None);
}

pub fn forward_gemma4(
    config: &Config,
    weights: &Gemma4Weights,
    cache: &mut KVCache,
    buf: &mut DecodeBuffer,
    token: u32,
    pos: usize,
) -> Vec<f32> {
    let mut logits = Vec::new();
    forward_gemma4_into(config, weights, cache, buf, token, pos, &mut logits);
    logits
}

fn prepare_gemma4_per_layer_inputs(
    config: &Config,
    weights: &Gemma4Weights,
    buf: &mut DecodeBuffer,
    token: u32,
) -> bool {
    let per_layer_dim = weights.per_layer_dim;
    if per_layer_dim == 0 {
        return false;
    }
    let Some(per_layer_token_embd) = &weights.per_layer_token_embd else {
        return false;
    };
    let Some(per_layer_model_proj) = &weights.per_layer_model_proj else {
        return false;
    };
    if weights.per_layer_proj_norm.len() != per_layer_dim {
        return false;
    }

    let per_layer_len = per_layer_dim * config.n_layers;
    per_layer_token_embd.row_into(token as usize, per_layer_len, &mut buf.ple_inputs);
    let token_scale = (per_layer_dim as f32).sqrt();

    per_layer_model_proj.matvec_into(&buf.x, &mut buf.ple_proj);
    if buf.ple_proj.len() != per_layer_len {
        return false;
    }
    let proj_scale = 1.0 / (config.dim as f32).sqrt();
    for value in &mut buf.ple_proj {
        *value *= proj_scale;
    }

    let input_scale = 1.0 / 2.0f32.sqrt();
    for layer_idx in 0..config.n_layers {
        let start = layer_idx * per_layer_dim;
        let end = start + per_layer_dim;
        rms_norm_into(
            &buf.ple_proj[start..end],
            &weights.per_layer_proj_norm,
            config.rms_norm_eps,
            &mut buf.ple_gate,
        );
        for i in 0..per_layer_dim {
            buf.ple_inputs[start + i] =
                (buf.ple_inputs[start + i] * token_scale + buf.ple_gate[i]) * input_scale;
        }
    }

    true
}

fn apply_gemma4_per_layer_residual(
    config: &Config,
    layer: &Gemma4LayerWeights,
    buf: &mut DecodeBuffer,
    layer_idx: usize,
    per_layer_dim: usize,
) {
    if per_layer_dim == 0 {
        return;
    }
    let (Some(inp_gate), Some(proj)) = (&layer.per_layer_inp_gate, &layer.per_layer_proj) else {
        return;
    };
    if layer.per_layer_post_norm.len() != config.dim {
        return;
    }
    let start = layer_idx * per_layer_dim;
    let end = start + per_layer_dim;
    if end > buf.ple_inputs.len() {
        return;
    }

    inp_gate.matvec_into(&buf.x, &mut buf.ple_gate);
    if buf.ple_gate.len() < per_layer_dim {
        return;
    }
    for i in 0..per_layer_dim {
        buf.ple_gate[i] = gelu(buf.ple_gate[i]) * buf.ple_inputs[start + i];
    }

    proj.matvec_into(&buf.ple_gate[..per_layer_dim], &mut buf.proj);
    rms_norm_into(
        &buf.proj,
        &layer.per_layer_post_norm,
        config.rms_norm_eps,
        &mut buf.xn2,
    );
    for i in 0..config.dim {
        buf.x[i] += buf.xn2[i];
    }
}

/// Runs one Gemma 4 decode step into a reusable logits buffer.
pub fn forward_gemma4_into(
    config: &Config,
    weights: &Gemma4Weights,
    cache: &mut KVCache,
    buf: &mut DecodeBuffer,
    token: u32,
    pos: usize,
    logits: &mut Vec<f32>,
) {
    let dim = config.dim;
    // Per-layer head/value/k_v layout is stored in each Gemma4 layer.
    // `buf` and `cache` are sized using layer maxima; here we use the
    // per-layer descriptors to read/write the correct slices and strides.

    // Token embedding
    weights.token_embd.row_into(token as usize, dim, &mut buf.x);
    let emb_scale = (dim as f32).sqrt();
    for value in &mut buf.x {
        *value *= emb_scale;
    }
    let has_per_layer_inputs = prepare_gemma4_per_layer_inputs(config, weights, buf, token);

    for l in 0..config.n_layers {
        let layer = &weights.layers[l];

        // Standard attention path (or K=V reuse when attn_v is missing)
        rms_norm_into(&buf.x, &layer.attn_norm, config.rms_norm_eps, &mut buf.xn);

        let head_dim_l = layer.head_dim;
        let n_kv_heads_l = layer.n_kv_heads;
        let value_dim_l = layer.value_dim;
        let shared_kv_source_layer = layer.shared_kv_source_layer;
        let kv_cache_layer = shared_kv_source_layer.unwrap_or(l);

        if shared_kv_source_layer.is_some() {
            layer.attn_q.matvec_into(&buf.xn, &mut buf.q);
        } else if layer.has_attn_v {
            if !try_quant_matvec3_into(
                &layer.attn_q,
                &layer.attn_k,
                &layer.attn_v,
                &buf.xn,
                &mut buf.q,
                &mut buf.k,
                &mut buf.v,
            ) {
                layer.attn_q.matvec_into(&buf.xn, &mut buf.q);
                layer.attn_k.matvec_into(&buf.xn, &mut buf.k);
                layer.attn_v.matvec_into(&buf.xn, &mut buf.v);
            }
        } else {
            if !try_quant_matvec2_into(
                &layer.attn_q,
                &layer.attn_k,
                &buf.xn,
                &mut buf.q,
                &mut buf.k,
            ) {
                layer.attn_q.matvec_into(&buf.xn, &mut buf.q);
                layer.attn_k.matvec_into(&buf.xn, &mut buf.k);
            }
            let kv_size = n_kv_heads_l * head_dim_l;
            buf.v[..kv_size].copy_from_slice(&buf.k[..kv_size]);
        }

        let q_len = config.n_heads * head_dim_l;
        let kv_k_size = n_kv_heads_l * head_dim_l;
        let kv_v_size = n_kv_heads_l * value_dim_l;
        rms_norm_heads_in_place(
            &mut buf.q[..q_len],
            head_dim_l,
            config.n_heads,
            Some(&layer.attn_q_norm),
            config.rms_norm_eps,
        );
        if shared_kv_source_layer.is_some() {
            apply_rope_neox(
                &mut buf.q[..q_len],
                pos,
                head_dim_l,
                config.n_heads,
                &layer.rope_inv_freq,
            );
        } else {
            rms_norm_heads_in_place(
                &mut buf.k[..kv_k_size],
                head_dim_l,
                n_kv_heads_l,
                Some(&layer.attn_k_norm),
                config.rms_norm_eps,
            );
            rms_norm_heads_in_place(
                &mut buf.v[..kv_v_size],
                value_dim_l,
                n_kv_heads_l,
                None,
                config.rms_norm_eps,
            );

            // Gemma 4 uses the NeoX-style rotate-half layout.
            apply_rope_qk_neox(
                &mut buf.q,
                &mut buf.k,
                pos,
                head_dim_l,
                config.n_heads,
                n_kv_heads_l,
                &layer.rope_inv_freq,
            );

            // Store KV into per-pos slots (cache uses fixed per-pos stride)
            // Important: only write the relevant portion based on per-layer dims
            let kv_k_start = cache.k_offset(pos);
            let kv_v_start = cache.v_offset(pos);
            cache.k[l][kv_k_start..kv_k_start + kv_k_size].copy_from_slice(&buf.k[..kv_k_size]);
            cache.v[l][kv_v_start..kv_v_start + kv_v_size].copy_from_slice(&buf.v[..kv_v_size]);
        }

        // Multi-head attention with GQA
        // Gemma 4 applies Q/K normalization before attention and uses a raw
        // attention scale of 1.0 rather than the usual 1/sqrt(head_dim).
        let scale = 1.0;
        let sliding_window = active_sliding_window(config, cache);
        let attn_window = if layer.is_swa {
            attention_start_pos(pos, sliding_window)
        } else {
            0
        };

        let kv_mul_l = config.n_heads / n_kv_heads_l;
        let attn_out_len = config.n_heads * value_dim_l;
        if !crate::metal::attention_into(
            &buf.q[..config.n_heads * head_dim_l],
            &cache.k[kv_cache_layer],
            &cache.v[kv_cache_layer],
            &mut buf.attn_out[..attn_out_len],
            config.n_heads,
            kv_mul_l,
            head_dim_l,
            value_dim_l,
            cache.per_pos_k_dim,
            cache.per_pos_v_dim,
            cache.storage_len,
            attn_window,
            pos,
            scale,
        ) {
            attention_over_kv_heads(
                &buf.q[..config.n_heads * head_dim_l],
                &cache.k[kv_cache_layer],
                &cache.v[kv_cache_layer],
                cache.per_pos_k_dim,
                cache.per_pos_v_dim,
                cache.storage_len,
                head_dim_l,
                value_dim_l,
                n_kv_heads_l,
                kv_mul_l,
                attn_window,
                pos,
                scale,
                &mut buf.attn_out[..attn_out_len],
            );
        }

        // Output projection + residual
        layer
            .attn_output
            .matvec_into(&buf.attn_out[..attn_out_len], &mut buf.proj);
        rms_norm_into(
            &buf.proj,
            &layer.post_attn_norm,
            config.rms_norm_eps,
            &mut buf.xn2,
        );
        for i in 0..dim {
            buf.x[i] += buf.xn2[i];
        }

        rms_norm_into(&buf.x, &layer.ffn_norm, config.rms_norm_eps, &mut buf.xn2);
        if !try_metal_gemma4_ffn_into(
            &layer.ffn_gate,
            &layer.ffn_up,
            &layer.ffn_down,
            &buf.xn2,
            &mut buf.proj,
        ) {
            if !try_quant_matvec2_into(
                &layer.ffn_gate,
                &layer.ffn_up,
                &buf.xn2,
                &mut buf.gate,
                &mut buf.up,
            ) {
                layer.ffn_gate.matvec_into(&buf.xn2, &mut buf.gate);
                layer.ffn_up.matvec_into(&buf.xn2, &mut buf.up);
            }

            let ffn_hidden_dim = layer.ffn_hidden_dim;
            buf.hidden.resize(ffn_hidden_dim, 0.0);
            for i in 0..ffn_hidden_dim {
                buf.hidden[i] = gelu(buf.gate[i]) * buf.up[i];
            }

            layer.ffn_down.matvec_into(&buf.hidden, &mut buf.proj);
        }
        rms_norm_into(
            &buf.proj,
            &layer.post_ffw_norm,
            config.rms_norm_eps,
            &mut buf.xn2,
        );
        for i in 0..dim {
            buf.x[i] += buf.xn2[i];
        }
        if has_per_layer_inputs {
            apply_gemma4_per_layer_residual(config, layer, buf, l, weights.per_layer_dim);
        }
        if let Some(&scale) = layer.layer_output_scale.first() {
            for value in &mut buf.x {
                *value *= scale;
            }
        }
    }

    // Final norm → logits
    rms_norm_into(
        &buf.x,
        &weights.output_norm,
        config.rms_norm_eps,
        &mut buf.xn,
    );
    weights.output.matvec_into(&buf.xn, logits);
    if weights.final_logit_softcap.is_finite() && weights.final_logit_softcap > 0.0 {
        let cap = weights.final_logit_softcap;
        for logit in logits {
            *logit = (*logit / cap).tanh() * cap;
        }
    }
}

#[derive(Clone)]
pub struct Gemma4LayerWeights {
    pub attn_norm: Vec<f32>,
    pub attn_q: Weight,
    pub attn_k: Weight,
    pub attn_v: Weight,
    pub attn_q_norm: Vec<f32>,
    pub attn_k_norm: Vec<f32>,
    pub attn_output: Weight,
    pub post_attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub ffn_down: Weight,
    pub ffn_up: Weight,
    pub ffn_gate: Weight,
    pub post_ffw_norm: Vec<f32>,
    pub layer_output_scale: Vec<f32>,
    pub rope_inv_freq: Vec<f32>,
    pub per_layer_inp_gate: Option<Weight>,
    pub per_layer_proj: Option<Weight>,
    pub per_layer_post_norm: Vec<f32>,
    pub head_dim: usize,
    pub n_kv_heads: usize,
    pub value_dim: usize,
    pub ffn_hidden_dim: usize,
    pub is_swa: bool,
    pub shared_kv_source_layer: Option<usize>,
    pub has_attn_v: bool, // True if layer has separate V projection; false = use K as V
}

#[derive(Clone)]
pub struct Gemma4Weights {
    pub token_embd: Weight,
    pub output_norm: Vec<f32>,
    pub output: Weight,
    pub final_logit_softcap: f32,
    pub per_layer_token_embd: Option<Weight>,
    pub per_layer_model_proj: Option<Weight>,
    pub per_layer_proj_norm: Vec<f32>,
    pub per_layer_dim: usize,
    pub layers: Vec<Gemma4LayerWeights>,
}

/// Loads Gemma-family dense decoder weights, including Gemma 4 GGUF variants.
pub fn load_gemma4_model(
    mmap_data: &[u8],
    gguf: &GGUFFile,
    borrow_quantized: bool,
) -> (Config, Gemma4Weights) {
    let mut config = Config::from_gguf(gguf);
    eprintln!(
        "Config: dim={}, layers={}, heads={}/{}, hidden={}, vocab={}, ctx={}",
        config.dim,
        config.n_layers,
        config.n_heads,
        config.n_kv_heads,
        config.hidden_dim,
        config.vocab_size,
        config.max_seq_len
    );

    let tensor_idx: HashMap<String, &crate::gguf::TensorInfo> =
        gguf.tensors.iter().map(|t| (t.name.clone(), t)).collect();
    let data_offset = gguf.data_offset;
    let mut inferred_sizes: HashMap<String, usize> = HashMap::new();
    if !gguf.tensors.is_empty() {
        let mmap_len = mmap_data.len();
        let mut offs: Vec<(u64, usize)> = gguf
            .tensors
            .iter()
            .enumerate()
            .map(|(i, t)| (t.offset, i))
            .collect();
        offs.sort_unstable_by_key(|o| o.0);
        for w in 0..offs.len() {
            let (off, idx) = offs[w];
            let next_off = if w + 1 < offs.len() {
                offs[w + 1].0
            } else {
                (mmap_len as u64).saturating_sub(data_offset as u64)
            };
            let byte_size = if next_off > off {
                (next_off - off) as usize
            } else {
                0
            };
            let name = &gguf.tensors[idx].name;
            inferred_sizes.insert(name.clone(), byte_size);
        }
    }

    // Infer head/value dims from available tensors (some Gemma-4 GGUFs
    // have unreliable metadata). Prefer inferred shapes when possible.
    {
        let mut head_dim_cand: Option<usize> = None;
        let mut value_dim_cand: Option<usize> = None;
        let mut kv_heads_cand: Option<usize> = None;
        for l in 0..config.n_layers {
            let qn = format!("blk.{}.attn_q.weight", l);
            let vn = format!("blk.{}.attn_v.weight", l);
            if head_dim_cand.is_none() {
                if let Some(info) = tensor_idx.get(&qn) {
                    if info.dims.len() >= 2 {
                        let rows = info.dims[1] as usize;
                        let cols = info.dims[0] as usize;
                        if cols == config.dim && config.n_heads > 0 {
                            head_dim_cand = Some(rows / config.n_heads);
                        }
                    }
                }
            }
            if value_dim_cand.is_none() || kv_heads_cand.is_none() {
                if let Some(info) = tensor_idx.get(&vn) {
                    if info.dims.len() >= 2 {
                        let rows = info.dims[1] as usize;
                        let cols = info.dims[0] as usize;
                        if cols == config.dim && head_dim_cand.is_some() {
                            let hd = head_dim_cand.unwrap();
                            if rows % hd == 0 {
                                kv_heads_cand = Some(rows / hd);
                                value_dim_cand = Some(hd); // assume value_dim matches head_dim
                            }
                        }
                    }
                }
            }
            if head_dim_cand.is_some() && value_dim_cand.is_some() && kv_heads_cand.is_some() {
                break;
            }
        }
        if let Some(hd) = head_dim_cand {
            if hd != config.head_dim {
                eprintln!(
                    "[INFO] Overriding config.head_dim {} -> {} based on attn_q tensor shapes",
                    config.head_dim, hd
                );
                config.head_dim = hd;
            }
        }
        if let Some(vd) = value_dim_cand {
            if vd != config.value_dim {
                eprintln!(
                    "[INFO] Overriding config.value_dim {} -> {} based on attn_v tensor shapes",
                    config.value_dim, vd
                );
                config.value_dim = vd;
            }
        }
        if let Some(kvh) = kv_heads_cand {
            if kvh != config.n_kv_heads {
                eprintln!(
                    "[INFO] Overriding config.n_kv_heads {} -> {} based on attn_v tensor shapes",
                    config.n_kv_heads, kvh
                );
                config.n_kv_heads = kvh;
            }
        }
        config.kv_dim = config.value_dim * config.n_kv_heads;
        config.kv_mul = config.n_heads / config.n_kv_heads;
        eprintln!(
            "Adjusted Gemma4 config: head_dim={}, value_dim={}, kv_dim={}, kv_mul={}",
            config.head_dim, config.value_dim, config.kv_dim, config.kv_mul
        );
    }

    let token_embd = load_weight(
        mmap_data,
        data_offset,
        "token_embd.weight",
        &tensor_idx,
        &inferred_sizes,
        false,
        borrow_quantized,
    );
    let output_norm = load_f32_vec(
        mmap_data,
        data_offset,
        "output_norm.weight",
        &tensor_idx,
        &inferred_sizes,
    );
    let output = if tensor_idx.contains_key("output.weight") {
        load_weight(
            mmap_data,
            data_offset,
            "output.weight",
            &tensor_idx,
            &inferred_sizes,
            false,
            borrow_quantized,
        )
    } else {
        eprintln!("Note: output tied to embeddings");
        token_embd.clone()
    };
    // Only softcap when the checkpoint actually ships the key (Gemma-2 era);
    // Gemma-3/4 checkpoints dropped final softcapping, and a phantom default
    // costs a scalar tanh over the whole vocab every decoded token.
    let arch = gguf.get_str("general.architecture").unwrap_or("gemma4");
    let final_logit_softcap = gguf.get_f32(
        &format!("{}.final_logit_softcapping", arch),
        gguf.get_f32("gemma4.final_logit_softcapping", 0.0),
    );
    let rope_base = gguf.get_f32("gemma4.rope.freq_base", config.rope_theta);
    let rope_base_swa = gguf.get_f32("gemma4.rope.freq_base_swa", rope_base);
    let rope_freqs_full = load_optional_f32_vec(
        mmap_data,
        data_offset,
        "rope_freqs.weight",
        &tensor_idx,
        &inferred_sizes,
        config.head_dim / 2,
    );
    let sliding_window_pattern: Vec<bool> =
        match gguf.metadata.get("gemma4.attention.sliding_window_pattern") {
            Some(crate::gguf::MetaValue::Array(values)) => values
                .iter()
                .filter_map(|v| {
                    if let crate::gguf::MetaValue::Bool(value) = v {
                        Some(*value)
                    } else {
                        None
                    }
                })
                .collect(),
            _ => Vec::new(),
        };
    let layer_is_swa: Vec<bool> = (0..config.n_layers)
        .map(|l| {
            let v_name = format!("blk.{}.attn_v.weight", l);
            sliding_window_pattern
                .get(l)
                .copied()
                .unwrap_or_else(|| tensor_idx.contains_key(&v_name))
        })
        .collect();
    let shared_kv_layers = gguf.get_u32("gemma4.attention.shared_kv_layers", 0) as usize;
    let first_shared_kv_layer = (shared_kv_layers > 0 && shared_kv_layers < config.n_layers)
        .then_some(config.n_layers - shared_kv_layers);
    let per_layer_dim = gguf.get_u32("gemma4.embedding_length_per_layer_input", 0) as usize;
    let per_layer_len = per_layer_dim.saturating_mul(config.n_layers);
    let per_layer_token_embd =
        if per_layer_dim > 0 && tensor_idx.contains_key("per_layer_token_embd.weight") {
            let w = load_weight(
                mmap_data,
                data_offset,
                "per_layer_token_embd.weight",
                &tensor_idx,
                &inferred_sizes,
                false,
                borrow_quantized,
            );
            validate_global_shape(
                "per_layer_token_embd.weight",
                &w,
                config.vocab_size,
                per_layer_len,
            );
            Some(w)
        } else {
            None
        };
    let per_layer_model_proj =
        if per_layer_dim > 0 && tensor_idx.contains_key("per_layer_model_proj.weight") {
            let w = load_weight(
                mmap_data,
                data_offset,
                "per_layer_model_proj.weight",
                &tensor_idx,
                &inferred_sizes,
                false,
                borrow_quantized,
            );
            validate_global_shape("per_layer_model_proj.weight", &w, per_layer_len, config.dim);
            Some(w)
        } else {
            None
        };
    let per_layer_proj_norm =
        if per_layer_dim > 0 && tensor_idx.contains_key("per_layer_proj_norm.weight") {
            load_f32_vec(
                mmap_data,
                data_offset,
                "per_layer_proj_norm.weight",
                &tensor_idx,
                &inferred_sizes,
            )
        } else {
            Vec::new()
        };

    let mut layers = Vec::with_capacity(config.n_layers);
    for l in 0..config.n_layers {
        // Helper: find an alternative tensor for this block that matches
        // the provided substrings (simple substring match, not regex).
        fn find_alternative(
            tensor_idx: &HashMap<String, &crate::gguf::TensorInfo>,
            layer: usize,
            subs: &[&str],
        ) -> Option<String> {
            let prefix = format!("blk.{}.", layer);
            for k in tensor_idx.keys() {
                if !k.starts_with(&prefix) || !k.ends_with(".weight") {
                    continue;
                }
                let rest = &k[prefix.len()..];
                let mut ok = true;
                for s in subs.iter() {
                    if !rest.contains(s) {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    return Some(k.clone());
                }
            }
            None
        }

        // Helper: validate a loaded weight's shape and panic with a clear
        // message if it doesn't match the expectation.
        fn validate_shape(
            name: &str,
            layer: usize,
            w: &Weight,
            exp_rows: usize,
            exp_cols: usize,
            config: &Config,
        ) {
            match w {
                Weight::F32(v) => {
                    let actual = v.len();
                    let expected = exp_rows.checked_mul(exp_cols).unwrap_or(0);
                    if actual != expected {
                        eprintln!(
                            "[ERROR] {} (layer {}): f32 elements {} != expected {} ({}x{}). config: dim={}, head_dim={}, n_heads={}, n_kv_heads={}, value_dim={}, kv_dim={}",
                            name,
                            layer,
                            actual,
                            expected,
                            exp_rows,
                            exp_cols,
                            config.dim,
                            config.head_dim,
                            config.n_heads,
                            config.n_kv_heads,
                            config.value_dim,
                            config.kv_dim
                        );
                        panic!("Shape mismatch for {} (layer {})", name, layer);
                    }
                }
                Weight::Quantized { rows, cols, .. } => {
                    if *rows != exp_rows || *cols != exp_cols {
                        eprintln!(
                            "[ERROR] {} (layer {}): quantized shape {}x{} != expected {}x{}. config: dim={}, head_dim={}, n_heads={}, n_kv_heads={}, value_dim={}, kv_dim={}",
                            name,
                            layer,
                            rows,
                            cols,
                            exp_rows,
                            exp_cols,
                            config.dim,
                            config.head_dim,
                            config.n_heads,
                            config.n_kv_heads,
                            config.value_dim,
                            config.kv_dim
                        );
                        panic!("Shape mismatch for {} (layer {})", name, layer);
                    }
                }
            }
        }

        let dim = config.dim;

        // Determine per-layer head/value layout heuristically from available
        // tensors. Many Gemma-4 GGUFs interleave layers with different
        // head/value sizes, so compute per-layer values rather than relying
        // solely on the global `config`.
        let mut head_dim_l = config.head_dim;
        let mut n_kv_heads_l = config.n_kv_heads;
        let mut value_dim_l = config.value_dim;

        // Try Q tensor first (preferred source of head_dim)
        let q_name = format!("blk.{}.attn_q.weight", l);
        let k_name = format!("blk.{}.attn_k.weight", l);
        let v_name = format!("blk.{}.attn_v.weight", l);
        if let Some(info) = tensor_idx.get(&q_name) {
            if info.dims.len() >= 2 {
                let rows = info.dims[1] as usize;
                let cols = info.dims[0] as usize;
                if cols == dim && config.n_heads > 0 && rows % config.n_heads == 0 {
                    head_dim_l = rows / config.n_heads;
                }
            }
        }

        // K tensor can reveal n_kv_heads when its rows are n_kv_heads * head_dim
        if let Some(info) = tensor_idx.get(&k_name) {
            if info.dims.len() >= 2 {
                let rows = info.dims[1] as usize;
                let cols = info.dims[0] as usize;
                if cols == dim && head_dim_l > 0 && rows % head_dim_l == 0 {
                    n_kv_heads_l = rows / head_dim_l;
                }
            }
        }

        // V tensor reveals value_dim (rows = n_kv_heads * value_dim) — derive
        if let Some(info) = tensor_idx.get(&v_name) {
            if info.dims.len() >= 2 {
                let rows = info.dims[1] as usize;
                let cols = info.dims[0] as usize;
                if cols == dim {
                    if n_kv_heads_l > 0 && rows % n_kv_heads_l == 0 {
                        value_dim_l = rows / n_kv_heads_l;
                    } else if head_dim_l > 0 && rows % head_dim_l == 0 {
                        // some GGUFs use value_dim == head_dim
                        value_dim_l = head_dim_l;
                        n_kv_heads_l = rows / head_dim_l;
                    }
                }
            }
        } else {
            // V tensor is missing: use K=V reuse.
            // value_dim_l should match K's geometry: k_rows = n_kv_heads * head_dim
            // So value_dim_l = head_dim_l (since V will use the same projection as K)
            value_dim_l = head_dim_l;
            eprintln!(
                "[INFO] Layer {}: attn_v missing, using K=V reuse. value_dim set to head_dim = {}",
                l, head_dim_l
            );
        }

        let q_rows = config.n_heads * head_dim_l;
        let k_rows = n_kv_heads_l * head_dim_l;
        let v_rows = n_kv_heads_l * value_dim_l;
        let out_rows = config.dim;

        let out_name = format!("blk.{}.attn_output.weight", l);
        let ffn_gate_name = format!("blk.{}.ffn_gate.weight", l);
        let ffn_up_name = format!("blk.{}.ffn_up.weight", l);
        let ffn_down_name = format!("blk.{}.ffn_down.weight", l);
        let per_layer_inp_gate_name = format!("blk.{}.inp_gate.weight", l);
        let per_layer_proj_name = format!("blk.{}.proj.weight", l);
        let per_layer_post_norm_name = format!("blk.{}.post_norm.weight", l);
        let is_swa = layer_is_swa[l];
        let shared_kv_source_layer = first_shared_kv_layer.and_then(|first_shared| {
            if l < first_shared {
                return None;
            }
            (0..first_shared)
                .rev()
                .find(|&source| layer_is_swa[source] == is_swa)
        });
        let ffn_hidden_dim_l = tensor_idx
            .get(&ffn_gate_name)
            .and_then(|info| {
                if info.dims.len() >= 2 && info.dims[0] as usize == dim {
                    Some(info.dims[1] as usize)
                } else {
                    None
                }
            })
            .unwrap_or(config.hidden_dim);

        // Load or fallback Q
        let attn_q = if tensor_idx.contains_key(&q_name) {
            let w = load_weight(
                mmap_data,
                data_offset,
                &q_name,
                &tensor_idx,
                &inferred_sizes,
                false,
                borrow_quantized,
            );
            validate_shape(&q_name, l, &w, q_rows, dim, &config);
            w
        } else if let Some(alt) = find_alternative(&tensor_idx, l, &["attn", "q"]) {
            eprintln!(
                "[INFO] Using alternative tensor {} for {} (layer {})",
                alt, q_name, l
            );
            let w = load_weight(
                mmap_data,
                data_offset,
                &alt,
                &tensor_idx,
                &inferred_sizes,
                false,
                borrow_quantized,
            );
            validate_shape(&alt, l, &w, q_rows, dim, &config);
            w
        } else {
            panic!(
                "Missing tensor: {} (or alternative attention query tensor for layer {})",
                q_name, l
            );
        };

        // Load or fallback K
        let attn_k = if tensor_idx.contains_key(&k_name) {
            let w = load_weight(
                mmap_data,
                data_offset,
                &k_name,
                &tensor_idx,
                &inferred_sizes,
                false,
                borrow_quantized,
            );
            validate_shape(&k_name, l, &w, k_rows, dim, &config);
            w
        } else if let Some(alt) = find_alternative(&tensor_idx, l, &["attn", "k"]) {
            eprintln!(
                "[INFO] Using alternative tensor {} for {} (layer {})",
                alt, k_name, l
            );
            let w = load_weight(
                mmap_data,
                data_offset,
                &alt,
                &tensor_idx,
                &inferred_sizes,
                false,
                borrow_quantized,
            );
            validate_shape(&alt, l, &w, k_rows, dim, &config);
            w
        } else {
            panic!(
                "Missing tensor: {} (or alternative attention key tensor for layer {})",
                k_name, l
            );
        };

        // Load or fallback V
        // Special handling: if V tensor is missing, use K as V (K=V reuse for full-attention layers)
        let (attn_v, has_attn_v) = if tensor_idx.contains_key(&v_name) {
            let w = load_weight(
                mmap_data,
                data_offset,
                &v_name,
                &tensor_idx,
                &inferred_sizes,
                false,
                borrow_quantized,
            );
            validate_shape(&v_name, l, &w, v_rows, dim, &config);
            (w, true)
        } else if let Some(alt) = find_alternative(&tensor_idx, l, &["attn", "v"]) {
            eprintln!(
                "[INFO] Using alternative tensor {} for {} (layer {})",
                alt, v_name, l
            );
            let w = load_weight(
                mmap_data,
                data_offset,
                &alt,
                &tensor_idx,
                &inferred_sizes,
                false,
                borrow_quantized,
            );
            validate_shape(&alt, l, &w, v_rows, dim, &config);
            (w, true)
        } else {
            // K=V reuse: missing attn_v means use K tensor as V
            // This is common in full-attention/sliding-window layers
            eprintln!(
                "[INFO] Missing tensor: {} (layer {}) — using K as V (K=V reuse)",
                v_name, l
            );
            (attn_k.clone(), false)
        };
        let full_rope_factors = if is_swa {
            None
        } else if rope_freqs_full.len() >= head_dim_l / 2 {
            Some(&rope_freqs_full[..head_dim_l / 2])
        } else {
            eprintln!(
                "[WARN] Layer {}: missing rope_freqs.weight for full-attention Gemma4 layer; proportional RoPE may be inaccurate",
                l
            );
            None
        };
        let rope_inv_freq = build_rope_inv_freq_with_factors(
            if is_swa { rope_base_swa } else { rope_base },
            head_dim_l,
            1.0,
            full_rope_factors,
        );

        // Load or fallback output projection
        let attn_output = if tensor_idx.contains_key(&out_name) {
            let w = load_weight(
                mmap_data,
                data_offset,
                &out_name,
                &tensor_idx,
                &inferred_sizes,
                false,
                borrow_quantized,
            );
            // attn_output: rows = dim, cols = n_heads * value_dim
            validate_shape(
                &out_name,
                l,
                &w,
                out_rows,
                config.n_heads * value_dim_l,
                &config,
            );
            w
        } else if let Some(alt) = find_alternative(&tensor_idx, l, &["attn", "output"]) {
            eprintln!(
                "[INFO] Using alternative tensor {} for {} (layer {})",
                alt, out_name, l
            );
            let w = load_weight(
                mmap_data,
                data_offset,
                &alt,
                &tensor_idx,
                &inferred_sizes,
                false,
                borrow_quantized,
            );
            validate_shape(&alt, l, &w, out_rows, config.n_heads * value_dim_l, &config);
            w
        } else {
            panic!(
                "Missing tensor: {} (or alternative attention output tensor for layer {})",
                out_name, l
            );
        };

        // FFN weights: gate/up/down
        let ffn_gate = if tensor_idx.contains_key(&ffn_gate_name) {
            let w = load_weight(
                mmap_data,
                data_offset,
                &ffn_gate_name,
                &tensor_idx,
                &inferred_sizes,
                false,
                borrow_quantized,
            );
            validate_shape(&ffn_gate_name, l, &w, ffn_hidden_dim_l, dim, &config);
            w
        } else {
            panic!("Missing tensor: {} (layer {})", ffn_gate_name, l);
        };

        let ffn_up = if tensor_idx.contains_key(&ffn_up_name) {
            let w = load_weight(
                mmap_data,
                data_offset,
                &ffn_up_name,
                &tensor_idx,
                &inferred_sizes,
                false,
                borrow_quantized,
            );
            validate_shape(&ffn_up_name, l, &w, ffn_hidden_dim_l, dim, &config);
            w
        } else {
            panic!("Missing tensor: {} (layer {})", ffn_up_name, l);
        };

        let ffn_down = if tensor_idx.contains_key(&ffn_down_name) {
            let w = load_weight(
                mmap_data,
                data_offset,
                &ffn_down_name,
                &tensor_idx,
                &inferred_sizes,
                false,
                borrow_quantized,
            );
            validate_shape(&ffn_down_name, l, &w, dim, ffn_hidden_dim_l, &config);
            w
        } else {
            panic!("Missing tensor: {} (layer {})", ffn_down_name, l);
        };
        let per_layer_inp_gate =
            if per_layer_dim > 0 && tensor_idx.contains_key(&per_layer_inp_gate_name) {
                let w = load_weight(
                    mmap_data,
                    data_offset,
                    &per_layer_inp_gate_name,
                    &tensor_idx,
                    &inferred_sizes,
                    false,
                    borrow_quantized,
                );
                validate_shape(&per_layer_inp_gate_name, l, &w, per_layer_dim, dim, &config);
                Some(w)
            } else {
                None
            };
        let per_layer_proj = if per_layer_dim > 0 && tensor_idx.contains_key(&per_layer_proj_name) {
            let w = load_weight(
                mmap_data,
                data_offset,
                &per_layer_proj_name,
                &tensor_idx,
                &inferred_sizes,
                false,
                borrow_quantized,
            );
            validate_shape(&per_layer_proj_name, l, &w, dim, per_layer_dim, &config);
            Some(w)
        } else {
            None
        };
        let per_layer_post_norm =
            if per_layer_dim > 0 && tensor_idx.contains_key(&per_layer_post_norm_name) {
                load_f32_vec(
                    mmap_data,
                    data_offset,
                    &per_layer_post_norm_name,
                    &tensor_idx,
                    &inferred_sizes,
                )
            } else {
                Vec::new()
            };

        let layer = Gemma4LayerWeights {
            attn_norm: load_f32_vec(
                mmap_data,
                data_offset,
                &format!("blk.{}.attn_norm.weight", l),
                &tensor_idx,
                &inferred_sizes,
            ),
            attn_q,
            attn_k,
            attn_v,
            attn_q_norm: load_f32_vec(
                mmap_data,
                data_offset,
                &format!("blk.{}.attn_q_norm.weight", l),
                &tensor_idx,
                &inferred_sizes,
            ),
            attn_k_norm: load_f32_vec(
                mmap_data,
                data_offset,
                &format!("blk.{}.attn_k_norm.weight", l),
                &tensor_idx,
                &inferred_sizes,
            ),
            attn_output,
            post_attn_norm: load_f32_vec(
                mmap_data,
                data_offset,
                &format!("blk.{}.post_attention_norm.weight", l),
                &tensor_idx,
                &inferred_sizes,
            ),
            ffn_norm: load_f32_vec(
                mmap_data,
                data_offset,
                &format!("blk.{}.ffn_norm.weight", l),
                &tensor_idx,
                &inferred_sizes,
            ),
            ffn_down,
            ffn_up,
            ffn_gate,
            post_ffw_norm: load_f32_vec(
                mmap_data,
                data_offset,
                &format!("blk.{}.post_ffw_norm.weight", l),
                &tensor_idx,
                &inferred_sizes,
            ),
            layer_output_scale: load_f32_vec(
                mmap_data,
                data_offset,
                &format!("blk.{}.layer_output_scale.weight", l),
                &tensor_idx,
                &inferred_sizes,
            ),
            rope_inv_freq,
            per_layer_inp_gate,
            per_layer_proj,
            per_layer_post_norm,
            head_dim: head_dim_l,
            n_kv_heads: n_kv_heads_l,
            value_dim: value_dim_l,
            ffn_hidden_dim: ffn_hidden_dim_l,
            is_swa,
            shared_kv_source_layer,
            has_attn_v,
        };
        layers.push(layer);
        if l == 0 || (l + 1) % 8 == 0 || l + 1 == config.n_layers {
            eprintln!("  Loaded Gemma4 layer {}/{}", l + 1, config.n_layers);
        }
    }

    let weights = Gemma4Weights {
        token_embd,
        output_norm,
        output,
        final_logit_softcap,
        per_layer_token_embd,
        per_layer_model_proj,
        per_layer_proj_norm,
        per_layer_dim,
        layers,
    };
    (config, weights)
}

// ─── Embedding forward passes (return normalized hidden state, not logits) ───
//
// These are identical to the generation forwards but skip the final output
// projection so the caller gets the residual stream after the last RMSNorm.
// Used by Runner::embed for text embedding / RAG retrieval.

/// Forward for standard (LLaMA-style) models; returns the normalized hidden
/// state of dimension `config.dim` instead of vocabulary logits.
fn forward_hidden_impl<'a>(
    config: &Config,
    weights: &ModelWeights,
    cache: &mut KVCache,
    buf: &'a mut DecodeBuffer,
    token: u32,
    pos: usize,
    final_norm: bool,
) -> &'a [f32] {
    let dim = config.dim;
    let head_dim = config.head_dim;
    let kv_mul = config.kv_mul;

    weights.token_embd.row_into(token as usize, dim, &mut buf.x);

    buf.rope_sin.resize(buf.rope_inv_freq.len(), 0.0);
    buf.rope_cos.resize(buf.rope_inv_freq.len(), 0.0);
    prepare_rope_sin_cos_into(
        pos,
        &buf.rope_inv_freq,
        &mut buf.rope_sin,
        &mut buf.rope_cos,
    );

    for l in 0..config.n_layers {
        let layer = &weights.layers[l];

        rms_norm_into(&buf.x, &layer.attn_norm, config.rms_norm_eps, &mut buf.xn);

        if !try_quant_matvec3_into(
            &layer.wq, &layer.wk, &layer.wv, &buf.xn, &mut buf.q, &mut buf.k, &mut buf.v,
        ) {
            layer.wq.matvec_into(&buf.xn, &mut buf.q);
            layer.wk.matvec_into(&buf.xn, &mut buf.k);
            layer.wv.matvec_into(&buf.xn, &mut buf.v);
        }

        add_bias_if_present(&mut buf.q, &layer.bq);
        add_bias_if_present(&mut buf.k, &layer.bk);
        add_bias_if_present(&mut buf.v, &layer.bv);

        apply_qk_norm_if_present(
            &mut buf.q,
            &mut buf.k,
            head_dim,
            config.n_heads,
            config.n_kv_heads,
            &layer.attn_q_norm,
            &layer.attn_k_norm,
            config.rms_norm_eps,
        );

        apply_model_rope_prepared(config, &mut buf.q, &mut buf.k, &buf.rope_sin, &buf.rope_cos);

        let kv_k_dim = cache.per_pos_k_dim;
        let kv_v_dim = cache.per_pos_v_dim;
        cache.write_k(l, pos, &buf.k);
        cache.write_v(l, pos, &buf.v);

        let scale = 1.0 / (head_dim as f32).sqrt();
        let sliding_window = active_sliding_window(config, cache);
        let attn_window = attention_start_pos(pos, sliding_window);

        if cache.bf16 {
            attention_over_kv_heads_bf16(
                &buf.q,
                &cache.k_bf16[l],
                &cache.v_bf16[l],
                kv_k_dim,
                kv_v_dim,
                cache.storage_len,
                head_dim,
                config.value_dim,
                config.n_kv_heads,
                kv_mul,
                attn_window,
                pos,
                scale,
                &mut buf.attn_out,
            );
        } else if !crate::metal::attention_into(
            &buf.q,
            &cache.k[l],
            &cache.v[l],
            &mut buf.attn_out,
            config.n_heads,
            kv_mul,
            head_dim,
            config.value_dim,
            kv_k_dim,
            kv_v_dim,
            cache.storage_len,
            attn_window,
            pos,
            scale,
        ) {
            attention_over_kv_heads(
                &buf.q,
                &cache.k[l],
                &cache.v[l],
                kv_k_dim,
                kv_v_dim,
                cache.storage_len,
                head_dim,
                config.value_dim,
                config.n_kv_heads,
                kv_mul,
                attn_window,
                pos,
                scale,
                &mut buf.attn_out,
            );
        }

        layer.wo.matvec_into(&buf.attn_out, &mut buf.proj);
        for i in 0..dim {
            buf.x[i] += buf.proj[i];
        }

        rms_norm_into(&buf.x, &layer.ffn_norm, config.rms_norm_eps, &mut buf.xn2);

        // Keep prefill on the same fused Mistral FFN path as decode.  This is
        // particularly important for Ministral Q4_K_M GGUFs: every prompt
        // token except the final one reaches this function, so falling back to
        // three independent projections needlessly adds Metal command-buffer
        // and host-buffer traffic.
        if let Some(moe) = &layer.moe {
            routed_moe_ffn_into(moe, config.expert_used_count, buf);
        } else if !try_metal_mistral_ffn_into(
            &layer.w1,
            &layer.w3,
            &layer.w2,
            &buf.xn2,
            &mut buf.proj,
        ) {
            if !try_quant_matvec2_into(&layer.w1, &layer.w3, &buf.xn2, &mut buf.gate, &mut buf.up) {
                layer.w1.matvec_into(&buf.xn2, &mut buf.gate);
                layer.w3.matvec_into(&buf.xn2, &mut buf.up);
            }

            crate::simd::silu_mul_into(
                &buf.gate[..config.hidden_dim],
                &buf.up[..config.hidden_dim],
                &mut buf.hidden,
            );

            layer.w2.matvec_into(&buf.hidden, &mut buf.proj);
        }
        for i in 0..dim {
            buf.x[i] += buf.proj[i];
        }
    }

    if final_norm {
        // Embedding-style callers need the final normalized residual stream.
        rms_norm_into(
            &buf.x,
            &weights.output_norm,
            config.rms_norm_eps,
            &mut buf.xn,
        );
        &buf.xn
    } else {
        &buf.x
    }
}

/// Forward for standard (LLaMA-style) models; returns the normalized hidden
/// state of dimension `config.dim` instead of vocabulary logits.
pub fn forward_hidden<'a>(
    config: &Config,
    weights: &ModelWeights,
    cache: &mut KVCache,
    buf: &'a mut DecodeBuffer,
    token: u32,
    pos: usize,
) -> &'a [f32] {
    forward_hidden_impl(config, weights, cache, buf, token, pos, true)
}

/// Advances the KV cache for one standard-model prompt token without
/// computing a final normalized hidden state.
pub fn forward_prefill(
    config: &Config,
    weights: &ModelWeights,
    cache: &mut KVCache,
    buf: &mut DecodeBuffer,
    token: u32,
    pos: usize,
) {
    // The resident decoder keeps its own GPU-side KV cache instead of
    // `cache.k`/`cache.v`, so prompt tokens must also flow through it here —
    // otherwise decode would attend over never-written (garbage) GPU cache
    // slots for every prefilled position.
    if active_sliding_window(config, cache) == 0 {
        weights
            .token_embd
            .row_into(token as usize, config.dim, &mut buf.x);
        if resident_prefill_attempt(config, weights, cache, buf, pos) {
            return;
        }
    }
    let _ = forward_hidden_impl(config, weights, cache, buf, token, pos, false);
}

/// Token-major scratch matrices for the batched standard-path prefill.
/// Each field holds `batch` rows laid out row-major; capacity is retained
/// across chunks and layers, so steady-state prefill does not allocate.
#[cfg(not(target_family = "wasm"))]
pub struct PrefillBatchBuffer {
    x: Vec<f32>,
    xn: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn_out: Vec<f32>,
    proj: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    hidden: Vec<f32>,
    qwen35_qkv: Vec<f32>,
    qwen35_activated: Vec<f32>,
    qwen35_gate: Vec<f32>,
    qwen35_q_gate: Vec<f32>,
    qwen35_alpha: Vec<f32>,
    qwen35_beta: Vec<f32>,
    embd_row: Vec<f32>,
    rope_inv_freq: Vec<f32>,
    /// Token-major RoPE angles, prepared once per prompt chunk instead of once
    /// per layer. This removes 26 redundant `sin_cos` passes for Ministral 3.
    rope_sin: Vec<f32>,
    rope_cos: Vec<f32>,
    /// Token-local routed-expert scratch. Dense and recurrent projections use
    /// the matrices above; routed experts remain sparse and token-dependent.
    nemotron_token: Option<Box<DecodeBuffer>>,
}

#[cfg(not(target_family = "wasm"))]
impl PrefillBatchBuffer {
    /// Allocates the shared scratch; matrices grow lazily to the chunk size.
    pub fn new(config: &Config) -> Self {
        Self {
            x: Vec::new(),
            xn: Vec::new(),
            q: Vec::new(),
            k: Vec::new(),
            v: Vec::new(),
            attn_out: Vec::new(),
            proj: Vec::new(),
            gate: Vec::new(),
            up: Vec::new(),
            hidden: Vec::new(),
            qwen35_qkv: Vec::new(),
            qwen35_activated: Vec::new(),
            qwen35_gate: Vec::new(),
            qwen35_q_gate: Vec::new(),
            qwen35_alpha: Vec::new(),
            qwen35_beta: Vec::new(),
            embd_row: Vec::new(),
            // Matches DecodeBuffer::new for the standard path.
            rope_inv_freq: build_rope_inv_freq(config.rope_theta, config.head_dim, 1.0),
            rope_sin: Vec::new(),
            rope_cos: Vec::new(),
            nemotron_token: None,
        }
    }
}

/// Reports whether every per-layer projection of a standard-path model can go
/// through the K-quant batch kernels. Checked once up front so the batch path
/// never leaves a chunk half-written to the KV cache before falling back.
#[cfg(not(target_family = "wasm"))]
pub fn standard_prefill_batchable(weights: &ModelWeights) -> bool {
    weights.layers.iter().all(|layer| {
        // Routed layers have no dense w1/w3/w2 for the batch kernels to read.
        layer.moe.is_none()
            // The batched prefill kernel does not yet apply Q/K per-head RMSNorm.
            // Fall back to the sequential path for Qwen3-style layers rather than
            // filling the cache with unnormalised keys.
            && layer.attn_q_norm.is_empty()
            && layer.attn_k_norm.is_empty()
            && quant_weight_parts(&layer.wq).is_some()
            && quant_weight_parts(&layer.wk).is_some()
            && quant_weight_parts(&layer.wv).is_some()
            && quant_weight_parts(&layer.wo).is_some()
            && quant_weight_parts(&layer.w1).is_some()
            && quant_weight_parts(&layer.w3).is_some()
            && quant_weight_parts(&layer.w2).is_some()
    })
}

/// Reports whether a standard decoder can verify a complete draft with the
/// token-major CPU kernels, including the final vocabulary projection.
#[cfg(not(target_family = "wasm"))]
pub fn standard_verify_batchable(weights: &ModelWeights) -> bool {
    standard_prefill_batchable(weights) && quant_weight_parts(&weights.output).is_some()
}

/// Reports whether all Qwen3.5/Qwen3.8 trunk projections can use the
/// row-major K-quant batch kernels.
#[cfg(not(target_family = "wasm"))]
pub fn qwen35_prefill_batchable(weights: &Qwen35Weights) -> bool {
    weights.layers.iter().all(|layer| {
        let mixer_batchable = match &layer.mixer {
            Qwen35Mixer::Linear(linear) => {
                quant_weight_parts(&linear.qkv).is_some()
                    && quant_weight_parts(&linear.gate).is_some()
                    && quant_weight_parts(&linear.alpha).is_some()
                    && quant_weight_parts(&linear.beta).is_some()
                    && quant_weight_parts(&linear.out).is_some()
            }
            Qwen35Mixer::Attention(attn) => {
                quant_weight_parts(&attn.q_gate).is_some()
                    && quant_weight_parts(&attn.k).is_some()
                    && quant_weight_parts(&attn.v).is_some()
                    && quant_weight_parts(&attn.out).is_some()
            }
        };
        mixer_batchable
            && quant_weight_parts(&layer.ffn_gate).is_some()
            && quant_weight_parts(&layer.ffn_up).is_some()
            && quant_weight_parts(&layer.ffn_down).is_some()
    })
}

/// Reports whether a Qwen hybrid decoder can verify a complete draft with the
/// token-major CPU kernels, including the final vocabulary projection.
#[cfg(not(target_family = "wasm"))]
pub fn qwen35_verify_batchable(weights: &Qwen35Weights) -> bool {
    qwen35_prefill_batchable(weights) && quant_weight_parts(&weights.output).is_some()
}

/// Reports whether every dense projection in a Nemotron-H/Soofi trunk can use
/// token-major verification. Routed expert matrices remain sparse and are
/// deliberately evaluated only for the experts selected by each token.
#[cfg(not(target_family = "wasm"))]
pub fn nemotron_h_verify_batchable(weights: &NemotronHWeights) -> bool {
    quant_weight_parts(&weights.output).is_some()
        && weights.layers.iter().all(|layer| match &layer.mixer {
            NemotronMixer::Mamba2(mamba) => {
                quant_weight_parts(&mamba.in_proj).is_some()
                    && quant_weight_parts(&mamba.out_proj).is_some()
            }
            NemotronMixer::Attention(attn) => {
                quant_weight_parts(&attn.wq).is_some()
                    && quant_weight_parts(&attn.wk).is_some()
                    && quant_weight_parts(&attn.wv).is_some()
                    && quant_weight_parts(&attn.wo).is_some()
            }
            NemotronMixer::DenseFfn(ffn) => {
                quant_weight_parts(&ffn.up).is_some() && quant_weight_parts(&ffn.down).is_some()
            }
            NemotronMixer::Moe(_) => true,
        })
}

/// Applies RMSNorm from one matrix row into another (slice output variant of
/// `rms_norm_into` for the token-major batch buffers).
#[cfg(not(target_family = "wasm"))]
#[inline]
fn rms_norm_slice_into(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    let ss = simd::dot_f32(x, x) / n as f32;
    let scale = 1.0 / (ss + eps).sqrt();
    for i in 0..n {
        out[i] = x[i] * scale * weight[i];
    }
}

/// Batched prompt prefill for standard (LLaMA-style) models: advances the KV
/// cache for `tokens` starting at `start_pos`, running every weight matrix
/// once per chunk instead of once per token.
///
/// The projections (QKV, attention output, gate/up/down) run through the
/// K-quant batch kernels, which stream each weight row from memory once and
/// reuse it L1-hot across the whole chunk — prefill's dominant cost is weight
/// bandwidth, so this divides it by the chunk length. RoPE, the KV-cache
/// store, and the attention scan stay strictly per-token IN ORDER: with a
/// sliding-window ring cache, a later token's store may reuse the slot of a
/// position still inside an earlier token's window, so store→attend must
/// interleave exactly like the sequential path.
///
/// Returns `false` (without touching the cache) when any projection cannot
/// take the batch path; the caller then uses the per-token fallback.
#[cfg(not(target_family = "wasm"))]
pub fn forward_prefill_batch(
    config: &Config,
    weights: &ModelWeights,
    cache: &mut KVCache,
    buf: &mut PrefillBatchBuffer,
    tokens: &[u32],
    start_pos: usize,
) -> bool {
    if tokens.is_empty() {
        return true;
    }
    if !standard_prefill_batchable(weights) {
        return false;
    }

    let b = tokens.len();
    let dim = config.dim;
    let head_dim = config.head_dim;
    let kv_mul = config.kv_mul;
    let q_rows = config.n_heads * head_dim;
    let k_rows = config.n_kv_heads * head_dim;
    let v_rows = config.n_kv_heads * config.value_dim;
    let attn_dim = config.n_heads * config.value_dim;

    // Residual streams start as the token embeddings.
    buf.x.resize(b * dim, 0.0);
    buf.xn.resize(b * dim, 0.0);
    buf.attn_out.resize(b * attn_dim, 0.0);
    for (t, &token) in tokens.iter().enumerate() {
        weights
            .token_embd
            .row_into(token as usize, dim, &mut buf.embd_row);
        buf.x[t * dim..(t + 1) * dim].copy_from_slice(&buf.embd_row);
    }

    let scale = 1.0 / (head_dim as f32).sqrt();
    let sliding_window = active_sliding_window(config, cache);
    // Writing every token's K/V before any attention read (below) is only
    // safe if the ring can't wrap within this microbatch: `cache.storage_len`
    // is the physical ring capacity (== the sliding window, or the full
    // context for a non-sliding cache), and a batch larger than that would
    // let a later token's write evict an earlier token's still-unread slot.
    // `batched_prefill_matches_sequential_with_ring_window` pins this with a
    // window of 8 and 17 tokens.
    let batch_attention_safe = b <= cache.storage_len;
    let rope_pairs = buf.rope_inv_freq.len();
    buf.rope_sin.resize(b * rope_pairs, 0.0);
    buf.rope_cos.resize(b * rope_pairs, 0.0);
    for t in 0..b {
        let rope_start = t * rope_pairs;
        prepare_rope_sin_cos_into(
            start_pos + t,
            &buf.rope_inv_freq,
            &mut buf.rope_sin[rope_start..rope_start + rope_pairs],
            &mut buf.rope_cos[rope_start..rope_start + rope_pairs],
        );
    }

    for l in 0..config.n_layers {
        let layer = &weights.layers[l];

        for t in 0..b {
            let x_row = &buf.x[t * dim..(t + 1) * dim];
            rms_norm_slice_into(
                x_row,
                &layer.attn_norm,
                config.rms_norm_eps,
                &mut buf.xn[t * dim..(t + 1) * dim],
            );
        }

        if !try_kquant_matvec3_batch_into(
            &layer.wq,
            &layer.wk,
            &layer.wv,
            &buf.xn[..b * dim],
            &mut buf.q,
            &mut buf.k,
            &mut buf.v,
        ) {
            // standard_prefill_batchable() vetted every layer, so this should
            // be unreachable. Falling back mid-chunk is still safe: the
            // per-token path recomputes the whole chunk from the embeddings
            // and overwrites every cache slot it touches.
            debug_assert!(false, "batchable QKV rejected by batch kernel");
            return false;
        }

        if batch_attention_safe {
            // Pass 1: bias, RoPE, and cache writes for every token in the
            // microbatch. Kept separate from attention (pass 2 below) so
            // that by the time any token's attention scan runs, the *whole*
            // batch's K/V is already in cache — each token still only ever
            // reads up to its own position (attn_window(t)..=pos(t)
            // enforces that), but this lets attention be parallelized over
            // (token, KV head) instead of just KV head.
            for t in 0..b {
                let pos = start_pos + t;
                let q_row = &mut buf.q[t * q_rows..(t + 1) * q_rows];
                let k_row = &mut buf.k[t * k_rows..(t + 1) * k_rows];
                let v_row = &mut buf.v[t * v_rows..(t + 1) * v_rows];
                add_bias_if_present(q_row, &layer.bq);
                add_bias_if_present(k_row, &layer.bk);
                add_bias_if_present(v_row, &layer.bv);

                let rope_start = t * rope_pairs;
                apply_model_rope_prepared(
                    config,
                    q_row,
                    k_row,
                    &buf.rope_sin[rope_start..rope_start + rope_pairs],
                    &buf.rope_cos[rope_start..rope_start + rope_pairs],
                );

                cache.write_k(l, pos, k_row);
                cache.write_v(l, pos, v_row);
            }

            // Pass 2: attention for every token. Prefer the KV-block-tiled
            // path (reads each KV block once per tile instead of once per
            // token) whenever it applies — plain causal only (no sliding
            // window, no bf16 cache yet) and enough work that the tiling
            // and extra dispatch bookkeeping pay for themselves. Otherwise
            // fall back to the untiled (token, KV head)-parallel path.
            let use_tiled = sliding_window == 0 && !cache.bf16 && {
                let work: usize = (0..b)
                    .map(|t| (start_pos + t + 1) * config.n_kv_heads)
                    .sum();
                work >= attention_parallel_min_work(
                    config.n_kv_heads,
                    kv_mul,
                    head_dim,
                    config.value_dim,
                    crate::simd::num_threads(),
                )
            };
            if use_tiled {
                attention_over_kv_heads_prefill_batch_tiled(
                    &buf.q[..b * q_rows],
                    &cache.k[l],
                    &cache.v[l],
                    cache.per_pos_k_dim,
                    cache.per_pos_v_dim,
                    cache.storage_len,
                    head_dim,
                    config.value_dim,
                    config.n_kv_heads,
                    kv_mul,
                    b,
                    start_pos,
                    scale,
                    &mut buf.attn_out[..b * attn_dim],
                );
            } else if cache.bf16 {
                attention_over_kv_heads_prefill_batch_bf16(
                    &buf.q[..b * q_rows],
                    &cache.k_bf16[l],
                    &cache.v_bf16[l],
                    cache.per_pos_k_dim,
                    cache.per_pos_v_dim,
                    cache.storage_len,
                    head_dim,
                    config.value_dim,
                    config.n_kv_heads,
                    kv_mul,
                    b,
                    start_pos,
                    sliding_window,
                    scale,
                    &mut buf.attn_out[..b * attn_dim],
                );
            } else {
                attention_over_kv_heads_prefill_batch(
                    &buf.q[..b * q_rows],
                    &cache.k[l],
                    &cache.v[l],
                    cache.per_pos_k_dim,
                    cache.per_pos_v_dim,
                    cache.storage_len,
                    head_dim,
                    config.value_dim,
                    config.n_kv_heads,
                    kv_mul,
                    b,
                    start_pos,
                    sliding_window,
                    scale,
                    &mut buf.attn_out[..b * attn_dim],
                );
            }
        } else {
            // Ring-wrap fallback: a batch larger than the cache's physical
            // ring capacity must interleave each token's write with its own
            // attention read, exactly like the single-token decode path,
            // or a later token's write could evict an earlier token's
            // still-unread slot before pass 2 (above) ever reads it.
            for t in 0..b {
                let pos = start_pos + t;
                let q_row = &mut buf.q[t * q_rows..(t + 1) * q_rows];
                let k_row = &mut buf.k[t * k_rows..(t + 1) * k_rows];
                let v_row = &mut buf.v[t * v_rows..(t + 1) * v_rows];
                add_bias_if_present(q_row, &layer.bq);
                add_bias_if_present(k_row, &layer.bk);
                add_bias_if_present(v_row, &layer.bv);

                let rope_start = t * rope_pairs;
                apply_model_rope_prepared(
                    config,
                    q_row,
                    k_row,
                    &buf.rope_sin[rope_start..rope_start + rope_pairs],
                    &buf.rope_cos[rope_start..rope_start + rope_pairs],
                );

                cache.write_k(l, pos, k_row);
                cache.write_v(l, pos, v_row);

                let attn_window = attention_start_pos(pos, sliding_window);
                let out_row = &mut buf.attn_out[t * attn_dim..(t + 1) * attn_dim];
                if cache.bf16 {
                    attention_over_kv_heads_bf16(
                        &buf.q[t * q_rows..(t + 1) * q_rows],
                        &cache.k_bf16[l],
                        &cache.v_bf16[l],
                        cache.per_pos_k_dim,
                        cache.per_pos_v_dim,
                        cache.storage_len,
                        head_dim,
                        config.value_dim,
                        config.n_kv_heads,
                        kv_mul,
                        attn_window,
                        pos,
                        scale,
                        out_row,
                    );
                } else {
                    attention_over_kv_heads(
                        &buf.q[t * q_rows..(t + 1) * q_rows],
                        &cache.k[l],
                        &cache.v[l],
                        cache.per_pos_k_dim,
                        cache.per_pos_v_dim,
                        cache.storage_len,
                        head_dim,
                        config.value_dim,
                        config.n_kv_heads,
                        kv_mul,
                        attn_window,
                        pos,
                        scale,
                        out_row,
                    );
                }
            }
        }

        if !try_kquant_matvec_batch_into(&layer.wo, &buf.attn_out[..b * attn_dim], &mut buf.proj) {
            debug_assert!(false, "batchable wo rejected by batch kernel");
            return false;
        }
        for t in 0..b {
            let proj_row = &buf.proj[t * dim..(t + 1) * dim];
            let x_row = &mut buf.x[t * dim..(t + 1) * dim];
            for i in 0..dim {
                x_row[i] += proj_row[i];
            }
        }

        for t in 0..b {
            let x_row = &buf.x[t * dim..(t + 1) * dim];
            rms_norm_slice_into(
                x_row,
                &layer.ffn_norm,
                config.rms_norm_eps,
                &mut buf.xn[t * dim..(t + 1) * dim],
            );
        }

        if !try_kquant_matvec2_batch_into(
            &layer.w1,
            &layer.w3,
            &buf.xn[..b * dim],
            &mut buf.gate,
            &mut buf.up,
        ) {
            debug_assert!(false, "batchable w1/w3 rejected by batch kernel");
            return false;
        }
        let hidden_len = b * config.hidden_dim;
        simd::silu_mul_into(
            &buf.gate[..hidden_len],
            &buf.up[..hidden_len],
            &mut buf.hidden,
        );
        if !try_kquant_matvec_batch_into(&layer.w2, &buf.hidden[..hidden_len], &mut buf.proj) {
            debug_assert!(false, "batchable w2 rejected by batch kernel");
            return false;
        }
        for t in 0..b {
            let proj_row = &buf.proj[t * dim..(t + 1) * dim];
            let x_row = &mut buf.x[t * dim..(t + 1) * dim];
            for i in 0..dim {
                x_row[i] += proj_row[i];
            }
        }
    }

    true
}

/// Evaluates every draft token in one standard-decoder micro-batch and returns
/// one normalized hidden row and one vocabulary-logit row per token. Row `i`
/// predicts the token following `tokens[i]`.
#[cfg(not(target_family = "wasm"))]
pub fn forward_verify_batch(
    config: &Config,
    weights: &ModelWeights,
    cache: &mut KVCache,
    buf: &mut PrefillBatchBuffer,
    tokens: &[u32],
    start_pos: usize,
    hidden: &mut Vec<f32>,
    logits: &mut Vec<f32>,
) -> bool {
    if !standard_verify_batchable(weights) {
        return false;
    }
    if !forward_prefill_batch(config, weights, cache, buf, tokens, start_pos) {
        return false;
    }
    finish_verify_batch(
        config,
        &weights.output_norm,
        &weights.output,
        buf,
        tokens.len(),
        hidden,
        logits,
    )
}

/// Batched prompt prefill for Qwen3.5/Qwen3.8 hybrid decoders.
///
/// The recurrent DeltaNet update and causal attention remain token ordered,
/// while every quantized projection is evaluated once for the complete
/// micro-batch. This is particularly important for the 27B model: it reuses
/// each decoded K-quant weight row across all prompt activations before that
/// row leaves cache instead of streaming the model once per prompt token.
#[cfg(not(target_family = "wasm"))]
pub fn forward_prefill_batch_qwen35(
    config: &Config,
    weights: &Qwen35Weights,
    cache: &mut KVCache,
    buf: &mut PrefillBatchBuffer,
    tokens: &[u32],
    start_pos: usize,
) -> bool {
    if tokens.is_empty() {
        return true;
    }
    if !qwen35_prefill_batchable(weights) {
        return false;
    }

    let b = tokens.len();
    let dim = config.dim;
    let query_dim = config.n_heads * config.head_dim;
    let key_dim = config.n_kv_heads * config.head_dim;
    let value_dim = config.n_kv_heads * config.value_dim;
    let scale = 1.0 / (config.head_dim as f32).sqrt();

    buf.x.resize(b * dim, 0.0);
    buf.xn.resize(b * dim, 0.0);
    for (t, &token) in tokens.iter().enumerate() {
        weights
            .token_embd
            .row_into(token as usize, dim, &mut buf.embd_row);
        buf.x[t * dim..(t + 1) * dim].copy_from_slice(&buf.embd_row);
    }

    let mut recurrent_index = 0usize;
    for layer in &weights.layers {
        for t in 0..b {
            rms_norm_slice_into(
                &buf.x[t * dim..(t + 1) * dim],
                &layer.attn_norm,
                config.rms_norm_eps,
                &mut buf.xn[t * dim..(t + 1) * dim],
            );
        }

        match &layer.mixer {
            Qwen35Mixer::Linear(linear) => {
                let dims = &weights.ssm;
                let recurrent_key_dim = dims.n_group * dims.d_state;
                let recurrent_value_dim = dims.d_inner;
                let value_head_dim = dims.head_dim();
                let conv_dim = dims.conv_dim();
                let qkv_dim = 2 * recurrent_key_dim + recurrent_value_dim;
                debug_assert_eq!(qkv_dim, conv_dim);

                if !try_kquant_matvec2_batch_into(
                    &linear.qkv,
                    &linear.gate,
                    &buf.xn[..b * dim],
                    &mut buf.qwen35_qkv,
                    &mut buf.qwen35_gate,
                ) || !try_kquant_matvec2_batch_into(
                    &linear.alpha,
                    &linear.beta,
                    &buf.xn[..b * dim],
                    &mut buf.qwen35_alpha,
                    &mut buf.qwen35_beta,
                ) {
                    debug_assert!(false, "batchable Qwen DeltaNet projection rejected");
                    return false;
                }

                buf.qwen35_activated.resize(b * qkv_dim, 0.0);
                buf.attn_out.resize(b * recurrent_value_dim, 0.0);
                buf.hidden.resize(b * recurrent_value_dim, 0.0);
                let state = cache
                    .ssm
                    .as_mut()
                    .expect("qwen35 requires Gated DeltaNet recurrent state");
                let conv_state = &mut state.conv[recurrent_index];
                let ssm_state = &mut state.ssm[recurrent_index];
                let history_len = dims.d_conv - 1;

                // The convolution and associative-memory update are causal,
                // so only this comparatively small part stays token-major.
                for t in 0..b {
                    let qkv_row = &mut buf.qwen35_qkv[t * qkv_dim..(t + 1) * qkv_dim];
                    for channel in 0..conv_dim {
                        let current = qkv_row[channel];
                        let taps =
                            &linear.conv_w[channel * dims.d_conv..(channel + 1) * dims.d_conv];
                        let history =
                            &mut conv_state[channel * history_len..(channel + 1) * history_len];
                        let mut convolved = current * taps[history_len];
                        for (past, tap) in history.iter().zip(&taps[..history_len]) {
                            convolved += past * tap;
                        }
                        history.copy_within(1..history_len, 0);
                        history[history_len - 1] = current;
                        qkv_row[channel] = convolved;
                    }

                    let activated = &mut buf.qwen35_activated[t * qkv_dim..(t + 1) * qkv_dim];
                    simd::silu_mul_slice_into(qkv_row, qkv_row, activated);
                    let (q_all, rest) = activated.split_at_mut(recurrent_key_dim);
                    let (k_all, v_all) = rest.split_at_mut(recurrent_key_dim);
                    qwen35_l2_normalize_heads(
                        q_all,
                        dims.d_state,
                        dims.n_group,
                        config.rms_norm_eps,
                    );
                    qwen35_l2_normalize_heads(
                        k_all,
                        dims.d_state,
                        dims.n_group,
                        config.rms_norm_eps,
                    );

                    let out_row =
                        &mut buf.attn_out[t * recurrent_value_dim..(t + 1) * recurrent_value_dim];
                    qwen35_delta_heads(
                        q_all,
                        k_all,
                        v_all,
                        &buf.qwen35_alpha[t * dims.n_head..(t + 1) * dims.n_head],
                        &buf.qwen35_beta[t * dims.n_head..(t + 1) * dims.n_head],
                        &linear.a,
                        &linear.dt_bias,
                        ssm_state,
                        out_row,
                        dims.n_group,
                        dims.n_head,
                        value_head_dim,
                    );
                    rms_norm_heads_in_place(
                        out_row,
                        value_head_dim,
                        dims.n_head,
                        Some(&linear.norm),
                        config.rms_norm_eps,
                    );
                    simd::silu_mul_slice_into(
                        &buf.qwen35_gate[t * recurrent_value_dim..(t + 1) * recurrent_value_dim],
                        out_row,
                        &mut buf.hidden[t * recurrent_value_dim..(t + 1) * recurrent_value_dim],
                    );
                }
                recurrent_index += 1;

                if !try_kquant_matvec_batch_into(
                    &linear.out,
                    &buf.hidden[..b * recurrent_value_dim],
                    &mut buf.proj,
                ) {
                    debug_assert!(false, "batchable Qwen DeltaNet output rejected");
                    return false;
                }
            }
            Qwen35Mixer::Attention(attn) => {
                if !try_kquant_matvec3_batch_into(
                    &attn.q_gate,
                    &attn.k,
                    &attn.v,
                    &buf.xn[..b * dim],
                    &mut buf.qwen35_q_gate,
                    &mut buf.k,
                    &mut buf.v,
                ) {
                    debug_assert!(false, "batchable Qwen attention projection rejected");
                    return false;
                }

                buf.q.resize(b * query_dim, 0.0);
                buf.qwen35_gate.resize(b * query_dim, 0.0);
                buf.attn_out.resize(b * query_dim, 0.0);
                for t in 0..b {
                    let joint = &buf.qwen35_q_gate[t * 2 * query_dim..(t + 1) * 2 * query_dim];
                    let q_row = &mut buf.q[t * query_dim..(t + 1) * query_dim];
                    let gate_row = &mut buf.qwen35_gate[t * query_dim..(t + 1) * query_dim];
                    for head in 0..config.n_heads {
                        let source = head * 2 * config.head_dim;
                        let target = head * config.head_dim;
                        q_row[target..target + config.head_dim]
                            .copy_from_slice(&joint[source..source + config.head_dim]);
                        gate_row[target..target + config.head_dim].copy_from_slice(
                            &joint[source + config.head_dim..source + 2 * config.head_dim],
                        );
                    }

                    let k_row = &mut buf.k[t * key_dim..(t + 1) * key_dim];
                    apply_qk_norm_if_present(
                        q_row,
                        k_row,
                        config.head_dim,
                        config.n_heads,
                        config.n_kv_heads,
                        &attn.q_norm,
                        &attn.k_norm,
                        config.rms_norm_eps,
                    );
                    let pos = start_pos + t;
                    apply_rope_qk_neox_partial(
                        q_row,
                        k_row,
                        pos,
                        config.head_dim,
                        weights.rotary_dim,
                        config.n_heads,
                        config.n_kv_heads,
                        &weights.rope_inv_freq,
                    );

                    let v_row = &buf.v[t * value_dim..(t + 1) * value_dim];
                    cache.write_k(attn.kv_slot, pos, k_row);
                    cache.write_v(attn.kv_slot, pos, v_row);
                    let out_row = &mut buf.attn_out[t * query_dim..(t + 1) * query_dim];
                    if cache.bf16 {
                        attention_over_kv_heads_bf16(
                            q_row,
                            &cache.k_bf16[attn.kv_slot],
                            &cache.v_bf16[attn.kv_slot],
                            cache.per_pos_k_dim,
                            cache.per_pos_v_dim,
                            cache.storage_len,
                            config.head_dim,
                            config.value_dim,
                            config.n_kv_heads,
                            config.kv_mul,
                            0,
                            pos,
                            scale,
                            out_row,
                        );
                    } else {
                        attention_over_kv_heads(
                            q_row,
                            &cache.k[attn.kv_slot],
                            &cache.v[attn.kv_slot],
                            cache.per_pos_k_dim,
                            cache.per_pos_v_dim,
                            cache.storage_len,
                            config.head_dim,
                            config.value_dim,
                            config.n_kv_heads,
                            config.kv_mul,
                            0,
                            pos,
                            scale,
                            out_row,
                        );
                    }
                    simd::sigmoid_mul_in_place(out_row, gate_row);
                }

                if !try_kquant_matvec_batch_into(
                    &attn.out,
                    &buf.attn_out[..b * query_dim],
                    &mut buf.proj,
                ) {
                    debug_assert!(false, "batchable Qwen attention output rejected");
                    return false;
                }
            }
        }

        for t in 0..b {
            let projection = &buf.proj[t * dim..(t + 1) * dim];
            let residual = &mut buf.x[t * dim..(t + 1) * dim];
            for i in 0..dim {
                residual[i] += projection[i];
            }
            rms_norm_slice_into(
                residual,
                &layer.post_attn_norm,
                config.rms_norm_eps,
                &mut buf.xn[t * dim..(t + 1) * dim],
            );
        }

        if !try_kquant_matvec2_batch_into(
            &layer.ffn_gate,
            &layer.ffn_up,
            &buf.xn[..b * dim],
            &mut buf.gate,
            &mut buf.up,
        ) {
            debug_assert!(false, "batchable Qwen FFN gate/up rejected");
            return false;
        }
        let hidden_len = b * config.hidden_dim;
        simd::silu_mul_into(
            &buf.gate[..hidden_len],
            &buf.up[..hidden_len],
            &mut buf.hidden,
        );
        if !try_kquant_matvec_batch_into(&layer.ffn_down, &buf.hidden[..hidden_len], &mut buf.proj)
        {
            debug_assert!(false, "batchable Qwen FFN down rejected");
            return false;
        }
        for t in 0..b {
            let projection = &buf.proj[t * dim..(t + 1) * dim];
            let residual = &mut buf.x[t * dim..(t + 1) * dim];
            for i in 0..dim {
                residual[i] += projection[i];
            }
        }
    }

    debug_assert_eq!(recurrent_index, weights.recurrent_layer_count);
    true
}

/// Evaluates every draft token in one Qwen hybrid micro-batch and returns one
/// normalized hidden row and one vocabulary-logit row per token. Recurrent
/// state is advanced in token order while projections reuse each weight row.
#[cfg(not(target_family = "wasm"))]
pub fn forward_verify_batch_qwen35(
    config: &Config,
    weights: &Qwen35Weights,
    cache: &mut KVCache,
    buf: &mut PrefillBatchBuffer,
    tokens: &[u32],
    start_pos: usize,
    hidden: &mut Vec<f32>,
    logits: &mut Vec<f32>,
) -> bool {
    if !qwen35_verify_batchable(weights) {
        return false;
    }
    if !forward_prefill_batch_qwen35(config, weights, cache, buf, tokens, start_pos) {
        return false;
    }
    finish_verify_batch(
        config,
        &weights.output_norm,
        &weights.output,
        buf,
        tokens.len(),
        hidden,
        logits,
    )
}

/// Evaluates a complete draft through a Nemotron-H/Soofi hybrid trunk. Dense
/// projections are token-major; causal attention and recurrent state updates
/// stay in strict position order. Sparse expert work remains token-local so no
/// unselected expert weights are read.
#[cfg(not(target_family = "wasm"))]
pub fn forward_verify_batch_nemotron_h(
    config: &Config,
    weights: &NemotronHWeights,
    cache: &mut KVCache,
    buf: &mut PrefillBatchBuffer,
    tokens: &[u32],
    start_pos: usize,
    hidden: &mut Vec<f32>,
    logits: &mut Vec<f32>,
) -> bool {
    if !nemotron_h_verify_batchable(weights) {
        return false;
    }
    if tokens.is_empty() {
        hidden.clear();
        logits.clear();
        return true;
    }

    let batch = tokens.len();
    let dim = config.dim;
    buf.x.resize(batch * dim, 0.0);
    buf.xn.resize(batch * dim, 0.0);
    for (row, &token) in tokens.iter().enumerate() {
        weights
            .token_embd
            .row_into(token as usize, dim, &mut buf.embd_row);
        buf.x[row * dim..(row + 1) * dim].copy_from_slice(&buf.embd_row);
    }

    let mut recurrent_index = 0usize;
    for layer in &weights.layers {
        for row in 0..batch {
            rms_norm_slice_into(
                &buf.x[row * dim..(row + 1) * dim],
                &layer.attn_norm,
                config.rms_norm_eps,
                &mut buf.xn[row * dim..(row + 1) * dim],
            );
        }

        match &layer.mixer {
            NemotronMixer::Mamba2(mamba) => {
                if !try_kquant_matvec_batch_into(
                    &mamba.in_proj,
                    &buf.xn[..batch * dim],
                    &mut buf.qwen35_qkv,
                ) {
                    return false;
                }
                let projected_width =
                    weights.ssm.d_inner + weights.ssm.conv_dim() + weights.ssm.n_head;
                buf.hidden.resize(batch * weights.ssm.d_inner, 0.0);
                let state = cache
                    .ssm
                    .as_mut()
                    .expect("hybrid model requires recurrent state");
                let mut convolved = Vec::new();
                let mut y = Vec::new();
                for row in 0..batch {
                    nemotron_mamba2_core(
                        mamba,
                        &weights.ssm,
                        &mut state.conv[recurrent_index],
                        &mut state.ssm[recurrent_index],
                        &buf.qwen35_qkv[row * projected_width..(row + 1) * projected_width],
                        config.rms_norm_eps,
                        &mut convolved,
                        &mut y,
                    );
                    buf.hidden[row * weights.ssm.d_inner..(row + 1) * weights.ssm.d_inner]
                        .copy_from_slice(&y);
                }
                recurrent_index += 1;
                if !try_kquant_matvec_batch_into(
                    &mamba.out_proj,
                    &buf.hidden[..batch * weights.ssm.d_inner],
                    &mut buf.proj,
                ) {
                    return false;
                }
            }
            NemotronMixer::Attention(attn) => {
                if !try_kquant_matvec3_batch_into(
                    &attn.wq,
                    &attn.wk,
                    &attn.wv,
                    &buf.xn[..batch * dim],
                    &mut buf.q,
                    &mut buf.k,
                    &mut buf.v,
                ) {
                    return false;
                }
                let head_dim = config.head_dim;
                let q_dim = attn.n_heads * head_dim;
                let k_dim = attn.n_kv_heads * head_dim;
                let v_dim = k_dim;
                let kv_mul = attn.n_heads / attn.n_kv_heads.max(1);
                let scale = 1.0 / (head_dim as f32).sqrt();
                buf.attn_out.resize(batch * q_dim, 0.0);
                for row in 0..batch {
                    let pos = start_pos + row;
                    let k_row = &buf.k[row * k_dim..(row + 1) * k_dim];
                    let v_row = &buf.v[row * v_dim..(row + 1) * v_dim];
                    cache.write_k(attn.kv_slot, pos, k_row);
                    cache.write_v(attn.kv_slot, pos, v_row);
                    attention_over_kv_heads(
                        &buf.q[row * q_dim..(row + 1) * q_dim],
                        &cache.k[attn.kv_slot],
                        &cache.v[attn.kv_slot],
                        k_dim,
                        v_dim,
                        cache.storage_len,
                        head_dim,
                        head_dim,
                        attn.n_kv_heads,
                        kv_mul,
                        0,
                        pos,
                        scale,
                        &mut buf.attn_out[row * q_dim..(row + 1) * q_dim],
                    );
                }
                if !try_kquant_matvec_batch_into(
                    &attn.wo,
                    &buf.attn_out[..batch * q_dim],
                    &mut buf.proj,
                ) {
                    return false;
                }
                if !attn.bo.is_empty() {
                    for row in 0..batch {
                        add_bias_if_present(&mut buf.proj[row * dim..(row + 1) * dim], &attn.bo);
                    }
                }
            }
            NemotronMixer::DenseFfn(ffn) => {
                if !try_kquant_matvec_batch_into(&ffn.up, &buf.xn[..batch * dim], &mut buf.up) {
                    return false;
                }
                let up_width = quant_weight_parts(&ffn.up)
                    .map(|parts| parts.2)
                    .unwrap_or(0);
                for row in 0..batch {
                    let values = &mut buf.up[row * up_width..(row + 1) * up_width];
                    add_bias_if_present(values, &ffn.up_bias);
                    for value in values {
                        *value = relu2(*value);
                    }
                }
                if !try_kquant_matvec_batch_into(
                    &ffn.down,
                    &buf.up[..batch * up_width],
                    &mut buf.proj,
                ) {
                    return false;
                }
                if !ffn.down_bias.is_empty() {
                    for row in 0..batch {
                        add_bias_if_present(
                            &mut buf.proj[row * dim..(row + 1) * dim],
                            &ffn.down_bias,
                        );
                    }
                }
            }
            NemotronMixer::Moe(moe) => {
                let token_buf = buf.nemotron_token.get_or_insert_with(|| {
                    Box::new(DecodeBuffer::new(
                        config,
                        config.head_dim,
                        config.n_kv_heads,
                        config.value_dim,
                    ))
                });
                buf.proj.resize(batch * dim, 0.0);
                for row in 0..batch {
                    token_buf
                        .xn
                        .copy_from_slice(&buf.xn[row * dim..(row + 1) * dim]);
                    nemotron_moe_ffn_into(moe, weights, config.expert_used_count, token_buf);
                    buf.proj[row * dim..(row + 1) * dim].copy_from_slice(&token_buf.proj[..dim]);
                }
            }
        }

        for row in 0..batch {
            let projection = &buf.proj[row * dim..(row + 1) * dim];
            let residual = &mut buf.x[row * dim..(row + 1) * dim];
            for index in 0..dim {
                residual[index] += projection[index];
            }
        }
    }
    debug_assert_eq!(
        recurrent_index,
        weights
            .layers
            .iter()
            .filter(|layer| matches!(layer.mixer, NemotronMixer::Mamba2(_)))
            .count()
    );

    finish_verify_batch(
        config,
        &weights.output_norm,
        &weights.output,
        buf,
        batch,
        hidden,
        logits,
    )
}

#[cfg(not(target_family = "wasm"))]
fn finish_verify_batch(
    config: &Config,
    output_norm: &[f32],
    output: &Weight,
    buf: &mut PrefillBatchBuffer,
    batch: usize,
    hidden: &mut Vec<f32>,
    logits: &mut Vec<f32>,
) -> bool {
    let dim = config.dim;
    buf.xn.resize(batch * dim, 0.0);
    for row in 0..batch {
        rms_norm_slice_into(
            &buf.x[row * dim..(row + 1) * dim],
            output_norm,
            config.rms_norm_eps,
            &mut buf.xn[row * dim..(row + 1) * dim],
        );
    }
    hidden.clear();
    hidden.extend_from_slice(&buf.x[..batch * dim]);
    try_kquant_matvec_batch_into(output, &buf.xn[..batch * dim], logits)
}

/// Forward for GPT-OSS (MoE) models; returns the normalized hidden state.
fn forward_hidden_gpt_oss_impl<'a>(
    config: &Config,
    weights: &GptOssWeights,
    cache: &mut KVCache,
    buf: &'a mut DecodeBuffer,
    token: u32,
    pos: usize,
    final_norm: bool,
) -> &'a [f32] {
    weights
        .token_embd
        .row_into(token as usize, config.dim, &mut buf.x);

    for l in 0..config.n_layers {
        let layer = &weights.layers[l];

        rms_norm_into(&buf.x, &layer.attn_norm, config.rms_norm_eps, &mut buf.xn);
        if !try_quant_matvec3_into(
            &layer.wq, &layer.wk, &layer.wv, &buf.xn, &mut buf.q, &mut buf.k, &mut buf.v,
        ) {
            layer.wq.matvec_into(&buf.xn, &mut buf.q);
            layer.wk.matvec_into(&buf.xn, &mut buf.k);
            layer.wv.matvec_into(&buf.xn, &mut buf.v);
        }
        for i in 0..buf.q.len() {
            buf.q[i] += layer.bq[i];
        }
        for i in 0..buf.k.len() {
            buf.k[i] += layer.bk[i];
        }
        for i in 0..buf.v.len() {
            buf.v[i] += layer.bv[i];
        }

        apply_rope_gpt_oss(
            &mut buf.q,
            &mut buf.k,
            pos,
            config.head_dim,
            config.n_heads,
            config.n_kv_heads,
            buf.rope_gpt_oss_concentration,
            &buf.rope_gpt_oss_inv_freq,
        );

        let kv_k_dim = cache.per_pos_k_dim;
        let kv_v_dim = cache.per_pos_v_dim;
        let kv_k_start = cache.k_offset(pos);
        let kv_v_start = cache.v_offset(pos);
        cache.k[l][kv_k_start..kv_k_start + buf.k.len()].copy_from_slice(&buf.k);
        cache.v[l][kv_v_start..kv_v_start + buf.v.len()].copy_from_slice(&buf.v);

        let scale = 1.0 / (config.head_dim as f32).sqrt();
        let sliding_window = active_sliding_window(config, cache);
        let attn_window = if l % 2 == 0 {
            attention_start_pos(pos, sliding_window)
        } else {
            0
        };

        if !crate::metal::attention_with_sink_into(
            &buf.q,
            &cache.k[l],
            &cache.v[l],
            &layer.sinks,
            &mut buf.attn_out,
            config.n_heads,
            config.kv_mul,
            config.head_dim,
            config.value_dim,
            kv_k_dim,
            kv_v_dim,
            cache.storage_len,
            attn_window,
            pos,
            scale,
        ) {
            attention_over_heads_with_sink(
                &buf.q,
                &cache.k[l],
                &cache.v[l],
                &layer.sinks,
                kv_k_dim,
                kv_v_dim,
                cache.storage_len,
                config.head_dim,
                config.value_dim,
                config.n_heads,
                config.kv_mul,
                attn_window,
                pos,
                scale,
                &mut buf.attn_out,
            );
        }

        layer.wo.matvec_into(&buf.attn_out, &mut buf.proj);
        for i in 0..config.dim {
            buf.x[i] += buf.proj[i] + layer.bo[i];
        }

        rms_norm_into(
            &buf.x,
            &layer.post_attn_norm,
            config.rms_norm_eps,
            &mut buf.xn2,
        );
        layer.gate_inp.matvec_into(&buf.xn2, &mut buf.router_logits);
        for i in 0..buf.router_logits.len() {
            buf.router_logits[i] += layer.gate_inp_bias[i];
        }

        select_top_logits_into(
            &buf.router_logits,
            config.expert_used_count,
            &mut buf.top_experts,
        );
        softmax_selected_into(&buf.top_experts, &mut buf.expert_probs);

        for value in buf.moe.iter_mut() {
            *value = 0.0;
        }
        for expert_slot in 0..buf.top_experts.len() {
            let expert_idx = buf.top_experts[expert_slot].0;
            let expert_prob = buf.expert_probs[expert_slot];
            let gate_bias = layer.gate_exps_bias.row_f32(expert_idx, config.hidden_dim);
            let up_bias = layer.up_exps_bias.row_f32(expert_idx, config.hidden_dim);
            let down_bias = layer.down_exps_bias.row_f32(expert_idx, config.dim);

            if !layer.gate_exps.try_matvec_expert_pair_into(
                &layer.up_exps,
                expert_idx,
                &buf.xn2,
                &mut buf.gate,
                &mut buf.up,
            ) {
                layer
                    .gate_exps
                    .matvec_expert_into(expert_idx, &buf.xn2, &mut buf.gate);
                layer
                    .up_exps
                    .matvec_expert_into(expert_idx, &buf.xn2, &mut buf.up);
            }
            for i in 0..config.hidden_dim {
                buf.gate[i] = swiglu_gpt_oss(buf.gate[i] + gate_bias[i], buf.up[i] + up_bias[i]);
            }

            layer
                .down_exps
                .matvec_expert_into(expert_idx, &buf.gate, &mut buf.proj);
            for i in 0..config.dim {
                buf.moe[i] += (buf.proj[i] + down_bias[i]) * expert_prob;
            }
        }

        for i in 0..config.dim {
            buf.x[i] += buf.moe[i];
        }
    }

    if final_norm {
        rms_norm_into(
            &buf.x,
            &weights.output_norm,
            config.rms_norm_eps,
            &mut buf.xn,
        );
        &buf.xn
    } else {
        &buf.x
    }
}

/// Forward for GPT-OSS (MoE) models; returns the normalized hidden state.
pub fn forward_hidden_gpt_oss<'a>(
    config: &Config,
    weights: &GptOssWeights,
    cache: &mut KVCache,
    buf: &'a mut DecodeBuffer,
    token: u32,
    pos: usize,
) -> &'a [f32] {
    forward_hidden_gpt_oss_impl(config, weights, cache, buf, token, pos, true)
}

/// Advances the KV cache for one GPT-OSS prompt token without computing a
/// final normalized hidden state.
pub fn forward_prefill_gpt_oss(
    config: &Config,
    weights: &GptOssWeights,
    cache: &mut KVCache,
    buf: &mut DecodeBuffer,
    token: u32,
    pos: usize,
) {
    let _ = forward_hidden_gpt_oss_impl(config, weights, cache, buf, token, pos, false);
}

/// Forward for Gemma-4 models; returns the normalized hidden state.
fn forward_hidden_gemma4_impl<'a>(
    config: &Config,
    weights: &Gemma4Weights,
    cache: &mut KVCache,
    buf: &'a mut DecodeBuffer,
    token: u32,
    pos: usize,
    final_norm: bool,
) -> &'a [f32] {
    let dim = config.dim;

    weights.token_embd.row_into(token as usize, dim, &mut buf.x);
    let emb_scale = (dim as f32).sqrt();
    for value in &mut buf.x {
        *value *= emb_scale;
    }
    let has_per_layer_inputs = prepare_gemma4_per_layer_inputs(config, weights, buf, token);

    for l in 0..config.n_layers {
        let layer = &weights.layers[l];

        rms_norm_into(&buf.x, &layer.attn_norm, config.rms_norm_eps, &mut buf.xn);

        let head_dim_l = layer.head_dim;
        let n_kv_heads_l = layer.n_kv_heads;
        let value_dim_l = layer.value_dim;
        let shared_kv_source_layer = layer.shared_kv_source_layer;
        let kv_cache_layer = shared_kv_source_layer.unwrap_or(l);

        if shared_kv_source_layer.is_some() {
            layer.attn_q.matvec_into(&buf.xn, &mut buf.q);
        } else if layer.has_attn_v {
            if !try_quant_matvec3_into(
                &layer.attn_q,
                &layer.attn_k,
                &layer.attn_v,
                &buf.xn,
                &mut buf.q,
                &mut buf.k,
                &mut buf.v,
            ) {
                layer.attn_q.matvec_into(&buf.xn, &mut buf.q);
                layer.attn_k.matvec_into(&buf.xn, &mut buf.k);
                layer.attn_v.matvec_into(&buf.xn, &mut buf.v);
            }
        } else {
            if !try_quant_matvec2_into(
                &layer.attn_q,
                &layer.attn_k,
                &buf.xn,
                &mut buf.q,
                &mut buf.k,
            ) {
                layer.attn_q.matvec_into(&buf.xn, &mut buf.q);
                layer.attn_k.matvec_into(&buf.xn, &mut buf.k);
            }
            let kv_size = n_kv_heads_l * head_dim_l;
            buf.v[..kv_size].copy_from_slice(&buf.k[..kv_size]);
        }

        let q_len = config.n_heads * head_dim_l;
        let kv_k_size = n_kv_heads_l * head_dim_l;
        let kv_v_size = n_kv_heads_l * value_dim_l;
        rms_norm_heads_in_place(
            &mut buf.q[..q_len],
            head_dim_l,
            config.n_heads,
            Some(&layer.attn_q_norm),
            config.rms_norm_eps,
        );
        if shared_kv_source_layer.is_some() {
            apply_rope_neox(
                &mut buf.q[..q_len],
                pos,
                head_dim_l,
                config.n_heads,
                &layer.rope_inv_freq,
            );
        } else {
            rms_norm_heads_in_place(
                &mut buf.k[..kv_k_size],
                head_dim_l,
                n_kv_heads_l,
                Some(&layer.attn_k_norm),
                config.rms_norm_eps,
            );
            rms_norm_heads_in_place(
                &mut buf.v[..kv_v_size],
                value_dim_l,
                n_kv_heads_l,
                None,
                config.rms_norm_eps,
            );

            apply_rope_qk_neox(
                &mut buf.q,
                &mut buf.k,
                pos,
                head_dim_l,
                config.n_heads,
                n_kv_heads_l,
                &layer.rope_inv_freq,
            );

            let kv_k_start = cache.k_offset(pos);
            let kv_v_start = cache.v_offset(pos);
            cache.k[l][kv_k_start..kv_k_start + kv_k_size].copy_from_slice(&buf.k[..kv_k_size]);
            cache.v[l][kv_v_start..kv_v_start + kv_v_size].copy_from_slice(&buf.v[..kv_v_size]);
        }

        let scale = 1.0;
        let sliding_window = active_sliding_window(config, cache);
        let attn_window = if layer.is_swa {
            attention_start_pos(pos, sliding_window)
        } else {
            0
        };

        let kv_mul_l = config.n_heads / n_kv_heads_l;
        let attn_out_len = config.n_heads * value_dim_l;
        if !crate::metal::attention_into(
            &buf.q[..config.n_heads * head_dim_l],
            &cache.k[kv_cache_layer],
            &cache.v[kv_cache_layer],
            &mut buf.attn_out[..attn_out_len],
            config.n_heads,
            kv_mul_l,
            head_dim_l,
            value_dim_l,
            cache.per_pos_k_dim,
            cache.per_pos_v_dim,
            cache.storage_len,
            attn_window,
            pos,
            scale,
        ) {
            attention_over_kv_heads(
                &buf.q[..config.n_heads * head_dim_l],
                &cache.k[kv_cache_layer],
                &cache.v[kv_cache_layer],
                cache.per_pos_k_dim,
                cache.per_pos_v_dim,
                cache.storage_len,
                head_dim_l,
                value_dim_l,
                n_kv_heads_l,
                kv_mul_l,
                attn_window,
                pos,
                scale,
                &mut buf.attn_out[..attn_out_len],
            );
        }

        layer
            .attn_output
            .matvec_into(&buf.attn_out[..attn_out_len], &mut buf.proj);
        rms_norm_into(
            &buf.proj,
            &layer.post_attn_norm,
            config.rms_norm_eps,
            &mut buf.xn2,
        );
        for i in 0..dim {
            buf.x[i] += buf.xn2[i];
        }

        rms_norm_into(&buf.x, &layer.ffn_norm, config.rms_norm_eps, &mut buf.xn2);
        if !try_metal_gemma4_ffn_into(
            &layer.ffn_gate,
            &layer.ffn_up,
            &layer.ffn_down,
            &buf.xn2,
            &mut buf.proj,
        ) {
            if !try_quant_matvec2_into(
                &layer.ffn_gate,
                &layer.ffn_up,
                &buf.xn2,
                &mut buf.gate,
                &mut buf.up,
            ) {
                layer.ffn_gate.matvec_into(&buf.xn2, &mut buf.gate);
                layer.ffn_up.matvec_into(&buf.xn2, &mut buf.up);
            }

            let ffn_hidden_dim = layer.ffn_hidden_dim;
            buf.hidden.resize(ffn_hidden_dim, 0.0);
            for i in 0..ffn_hidden_dim {
                buf.hidden[i] = gelu(buf.gate[i]) * buf.up[i];
            }

            layer.ffn_down.matvec_into(&buf.hidden, &mut buf.proj);
        }
        rms_norm_into(
            &buf.proj,
            &layer.post_ffw_norm,
            config.rms_norm_eps,
            &mut buf.xn2,
        );
        for i in 0..dim {
            buf.x[i] += buf.xn2[i];
        }
        if has_per_layer_inputs {
            apply_gemma4_per_layer_residual(config, layer, buf, l, weights.per_layer_dim);
        }
        if let Some(&scale) = layer.layer_output_scale.first() {
            for value in &mut buf.x {
                *value *= scale;
            }
        }
    }

    if final_norm {
        rms_norm_into(
            &buf.x,
            &weights.output_norm,
            config.rms_norm_eps,
            &mut buf.xn,
        );
        &buf.xn
    } else {
        &buf.x
    }
}

/// Forward for Gemma-4 models; returns the normalized hidden state.
pub fn forward_hidden_gemma4<'a>(
    config: &Config,
    weights: &Gemma4Weights,
    cache: &mut KVCache,
    buf: &'a mut DecodeBuffer,
    token: u32,
    pos: usize,
) -> &'a [f32] {
    forward_hidden_gemma4_impl(config, weights, cache, buf, token, pos, true)
}

/// Advances the KV cache for one Gemma-4 prompt token without computing a
/// final normalized hidden state.
pub fn forward_prefill_gemma4(
    config: &Config,
    weights: &Gemma4Weights,
    cache: &mut KVCache,
    buf: &mut DecodeBuffer,
    token: u32,
    pos: usize,
) {
    let _ = forward_hidden_gemma4_impl(config, weights, cache, buf, token, pos, false);
}

// ─── nomic-bert (BERT-style encoder for embeddings) ─────────────────────────

/// Per-layer weights for a nomic-bert / BERT encoder block. FFN is SwiGLU when
/// `ffn_gate` is `Some` (nomic-bert) and GELU-sequential otherwise (plain BERT
/// or nomic-bert-moe dense layers). All norms are true LayerNorm (weight+bias).
pub struct NomicBertLayerWeights {
    pub wq: Weight,
    pub bq: Vec<f32>,
    pub wk: Weight,
    pub bk: Vec<f32>,
    pub wv: Weight,
    pub bv: Vec<f32>,
    pub wo: Weight,
    pub bo: Vec<f32>,
    pub attn_out_norm: Vec<f32>,
    pub attn_out_norm_b: Vec<f32>,
    pub ffn_gate: Option<Weight>,
    pub ffn_up: Weight,
    pub ffn_up_b: Vec<f32>,
    pub ffn_down: Weight,
    pub ffn_down_b: Vec<f32>,
    pub layer_out_norm: Vec<f32>,
    pub layer_out_norm_b: Vec<f32>,
}

/// Full weight set for a nomic-bert encoder. Output is the last hidden state;
/// pooling and L2 normalization happen in the runtime's `embed`.
pub struct NomicBertWeights {
    pub token_embd: Weight,
    /// `token_types.weight` row 0 (segment embedding); empty when absent.
    pub token_type0: Vec<f32>,
    pub tok_norm: Vec<f32>,
    pub tok_norm_b: Vec<f32>,
    pub layers: Vec<NomicBertLayerWeights>,
    /// LayerNorm epsilon (nomic uses 1e-12, distinct from the RMS eps).
    pub ln_eps: f32,
}

/// Loads a nomic-bert / BERT encoder model from mapped GGUF bytes.
pub fn load_nomic_bert_model(
    mmap_data: &[u8],
    gguf: &GGUFFile,
    borrow_quantized: bool,
) -> (Config, NomicBertWeights) {
    let mut config = Config::from_gguf(gguf);
    // head_dim defaults to dim / n_heads via Config::from_gguf; ensure kv heads
    // mirror query heads (BERT is full multi-head, no GQA).
    if config.n_kv_heads == 0 {
        config.n_kv_heads = config.n_heads;
    }
    let arch = gguf.get_str("general.architecture").unwrap_or("nomic-bert");
    // nomic stores the LayerNorm eps under attention.layer_norm_epsilon, NOT
    // the RMS key Config::from_gguf reads; default to the BERT-standard 1e-12.
    let ln_eps = gguf.get_f32(&format!("{}.attention.layer_norm_epsilon", arch), 1e-12);

    let tensor_idx: HashMap<String, &crate::gguf::TensorInfo> =
        gguf.tensors.iter().map(|t| (t.name.clone(), t)).collect();
    let data_offset = gguf.data_offset;
    let mut inferred_sizes: HashMap<String, usize> = HashMap::new();
    if !gguf.tensors.is_empty() {
        let mmap_len = mmap_data.len();
        let mut offs: Vec<(u64, usize)> = gguf
            .tensors
            .iter()
            .enumerate()
            .map(|(i, t)| (t.offset, i))
            .collect();
        offs.sort_unstable_by_key(|o| o.0);
        for w in 0..offs.len() {
            let (off, idx) = offs[w];
            let next_off = if w + 1 < offs.len() {
                offs[w + 1].0
            } else {
                (mmap_len as u64).saturating_sub(data_offset as u64)
            };
            let byte_size = if next_off > off {
                (next_off - off) as usize
            } else {
                0
            };
            inferred_sizes.insert(gguf.tensors[idx].name.clone(), byte_size);
        }
    }

    let dim = config.dim;
    let head_dim = config.head_dim;
    let q_rows = config.n_heads * head_dim;
    let kv_rows = config.n_kv_heads * head_dim;

    let token_embd = load_weight(
        mmap_data,
        data_offset,
        "token_embd.weight",
        &tensor_idx,
        &inferred_sizes,
        false,
        borrow_quantized,
    );
    let token_type0 = if tensor_idx.contains_key("token_types.weight") {
        // token_types.weight is [dim x n_types]; segment ids are always 0.
        load_optional_f32_slice(
            mmap_data,
            data_offset,
            "token_types.weight",
            &tensor_idx,
            &inferred_sizes,
            0,
            dim,
        )
    } else {
        Vec::new()
    };
    let tok_norm = load_f32_vec(
        mmap_data,
        data_offset,
        "token_embd_norm.weight",
        &tensor_idx,
        &inferred_sizes,
    );
    let tok_norm_b = load_f32_vec(
        mmap_data,
        data_offset,
        "token_embd_norm.bias",
        &tensor_idx,
        &inferred_sizes,
    );

    let mut layers = Vec::with_capacity(config.n_layers);
    for l in 0..config.n_layers {
        let qkv_name = format!("blk.{}.attn_qkv.weight", l);
        let (wq, wk, wv, bq, bk, bv);
        if tensor_idx.contains_key(&qkv_name) {
            // Fused QKV: rows [0..q) = Q, [q..q+kv) = K, [q+kv..q+2kv) = V.
            wq = load_weight_rows(
                mmap_data,
                data_offset,
                &qkv_name,
                &tensor_idx,
                &inferred_sizes,
                0,
                q_rows,
                dim,
                borrow_quantized,
            );
            wk = load_weight_rows(
                mmap_data,
                data_offset,
                &qkv_name,
                &tensor_idx,
                &inferred_sizes,
                q_rows,
                kv_rows,
                dim,
                borrow_quantized,
            );
            wv = load_weight_rows(
                mmap_data,
                data_offset,
                &qkv_name,
                &tensor_idx,
                &inferred_sizes,
                q_rows + kv_rows,
                kv_rows,
                dim,
                borrow_quantized,
            );
            let qkv_bias = format!("blk.{}.attn_qkv.bias", l);
            bq = load_optional_f32_slice(
                mmap_data,
                data_offset,
                &qkv_bias,
                &tensor_idx,
                &inferred_sizes,
                0,
                q_rows,
            );
            bk = load_optional_f32_slice(
                mmap_data,
                data_offset,
                &qkv_bias,
                &tensor_idx,
                &inferred_sizes,
                q_rows,
                kv_rows,
            );
            bv = load_optional_f32_slice(
                mmap_data,
                data_offset,
                &qkv_bias,
                &tensor_idx,
                &inferred_sizes,
                q_rows + kv_rows,
                kv_rows,
            );
        } else {
            // Separate q/k/v tensors (plain BERT layout).
            wq = load_weight(
                mmap_data,
                data_offset,
                &format!("blk.{}.attn_q.weight", l),
                &tensor_idx,
                &inferred_sizes,
                false,
                borrow_quantized,
            );
            wk = load_weight(
                mmap_data,
                data_offset,
                &format!("blk.{}.attn_k.weight", l),
                &tensor_idx,
                &inferred_sizes,
                false,
                borrow_quantized,
            );
            wv = load_weight(
                mmap_data,
                data_offset,
                &format!("blk.{}.attn_v.weight", l),
                &tensor_idx,
                &inferred_sizes,
                false,
                borrow_quantized,
            );
            bq = load_optional_f32_vec(
                mmap_data,
                data_offset,
                &format!("blk.{}.attn_q.bias", l),
                &tensor_idx,
                &inferred_sizes,
                q_rows,
            );
            bk = load_optional_f32_vec(
                mmap_data,
                data_offset,
                &format!("blk.{}.attn_k.bias", l),
                &tensor_idx,
                &inferred_sizes,
                kv_rows,
            );
            bv = load_optional_f32_vec(
                mmap_data,
                data_offset,
                &format!("blk.{}.attn_v.bias", l),
                &tensor_idx,
                &inferred_sizes,
                kv_rows,
            );
        }

        let wo = load_weight(
            mmap_data,
            data_offset,
            &format!("blk.{}.attn_output.weight", l),
            &tensor_idx,
            &inferred_sizes,
            false,
            borrow_quantized,
        );
        let bo = load_optional_f32_vec(
            mmap_data,
            data_offset,
            &format!("blk.{}.attn_output.bias", l),
            &tensor_idx,
            &inferred_sizes,
            dim,
        );
        let attn_out_norm = load_f32_vec(
            mmap_data,
            data_offset,
            &format!("blk.{}.attn_output_norm.weight", l),
            &tensor_idx,
            &inferred_sizes,
        );
        let attn_out_norm_b = load_f32_vec(
            mmap_data,
            data_offset,
            &format!("blk.{}.attn_output_norm.bias", l),
            &tensor_idx,
            &inferred_sizes,
        );

        let gate_name = format!("blk.{}.ffn_gate.weight", l);
        let ffn_gate = if tensor_idx.contains_key(&gate_name) {
            Some(load_weight(
                mmap_data,
                data_offset,
                &gate_name,
                &tensor_idx,
                &inferred_sizes,
                false,
                borrow_quantized,
            ))
        } else {
            None
        };
        let ffn_up = load_weight(
            mmap_data,
            data_offset,
            &format!("blk.{}.ffn_up.weight", l),
            &tensor_idx,
            &inferred_sizes,
            false,
            borrow_quantized,
        );
        let ffn_up_b = load_optional_f32_vec(
            mmap_data,
            data_offset,
            &format!("blk.{}.ffn_up.bias", l),
            &tensor_idx,
            &inferred_sizes,
            config.hidden_dim,
        );
        let ffn_down = load_weight(
            mmap_data,
            data_offset,
            &format!("blk.{}.ffn_down.weight", l),
            &tensor_idx,
            &inferred_sizes,
            false,
            borrow_quantized,
        );
        let ffn_down_b = load_optional_f32_vec(
            mmap_data,
            data_offset,
            &format!("blk.{}.ffn_down.bias", l),
            &tensor_idx,
            &inferred_sizes,
            dim,
        );
        let layer_out_norm = load_f32_vec(
            mmap_data,
            data_offset,
            &format!("blk.{}.layer_output_norm.weight", l),
            &tensor_idx,
            &inferred_sizes,
        );
        let layer_out_norm_b = load_f32_vec(
            mmap_data,
            data_offset,
            &format!("blk.{}.layer_output_norm.bias", l),
            &tensor_idx,
            &inferred_sizes,
        );

        layers.push(NomicBertLayerWeights {
            wq,
            bq,
            wk,
            bk,
            wv,
            bv,
            wo,
            bo,
            attn_out_norm,
            attn_out_norm_b,
            ffn_gate,
            ffn_up,
            ffn_up_b,
            ffn_down,
            ffn_down_b,
            layer_out_norm,
            layer_out_norm_b,
        });
    }

    let weights = NomicBertWeights {
        token_embd,
        token_type0,
        tok_norm,
        tok_norm_b,
        layers,
        ln_eps,
    };
    (config, weights)
}

#[cfg(not(target_family = "wasm"))]
/// Checks that a Nomic BERT layer can use the token-batched K-quant path
/// without changing its mathematical layout. The checks happen before the
/// first residual update, so a failed capability check always falls back to
/// the existing per-token implementation cleanly.
fn nomic_batched_layer_supported(layer: &NomicBertLayerWeights, dim: usize, kv_row: usize) -> bool {
    let Some(gate) = layer.ffn_gate.as_ref() else {
        return false;
    };
    let (Some(q), Some(k), Some(v), Some(wo), Some(gate), Some(up), Some(down)) = (
        kquant_weight_parts(&layer.wq),
        kquant_weight_parts(&layer.wk),
        kquant_weight_parts(&layer.wv),
        kquant_weight_parts(&layer.wo),
        kquant_weight_parts(gate),
        kquant_weight_parts(&layer.ffn_up),
        kquant_weight_parts(&layer.ffn_down),
    ) else {
        return false;
    };

    q.2 == kv_row
        && k.2 == kv_row
        && v.2 == kv_row
        && q.3 == dim
        && k.3 == dim
        && v.3 == dim
        && wo.2 == dim
        && wo.3 == kv_row
        && gate.3 == dim
        && up.3 == dim
        && gate.2 == up.2
        && down.2 == dim
        && down.3 == gate.2
}

/// Raw context for independent bidirectional-attention rows. `parallel_range`
/// blocks until every worker returns, and each callback writes a disjoint row
/// of `out`, so the borrowed vectors remain valid and data-race free.
#[cfg(not(target_family = "wasm"))]
struct NomicAttentionBatch {
    q: *const f32,
    k: *const f32,
    v: *const f32,
    out: *mut f32,
    n: usize,
    n_heads: usize,
    head_dim: usize,
    kv_row: usize,
    scale: f32,
    /// Inclusive/exclusive token ranges for each flattened input sequence.
    /// Attention is bidirectional within one range, never across ranges.
    sequence_starts: *const usize,
    sequence_ends: *const usize,
}

#[cfg(not(target_family = "wasm"))]
unsafe fn nomic_attention_batch_range(ctx: *const (), start: usize, end: usize) {
    // SAFETY: `parallel_range` blocks for the lifetime of the stack context;
    // q/k/v are immutable and each worker receives a disjoint output range.
    let ctx = unsafe { &*(ctx as *const NomicAttentionBatch) };
    let q_all = unsafe { std::slice::from_raw_parts(ctx.q, ctx.n * ctx.kv_row) };
    let k_all = unsafe { std::slice::from_raw_parts(ctx.k, ctx.n * ctx.kv_row) };
    let v_all = unsafe { std::slice::from_raw_parts(ctx.v, ctx.n * ctx.kv_row) };
    let sequence_starts = unsafe { std::slice::from_raw_parts(ctx.sequence_starts, ctx.n) };
    let sequence_ends = unsafe { std::slice::from_raw_parts(ctx.sequence_ends, ctx.n) };
    for i in start..end {
        let sequence_start = sequence_starts[i];
        let sequence_end = sequence_ends[i];
        let sequence_len = sequence_end - sequence_start;
        let attn_out =
            unsafe { std::slice::from_raw_parts_mut(ctx.out.add(i * ctx.kv_row), ctx.kv_row) };
        attn_out.fill(0.0);
        for h in 0..ctx.n_heads {
            let q_head =
                &q_all[i * ctx.kv_row + h * ctx.head_dim..i * ctx.kv_row + (h + 1) * ctx.head_dim];
            let keys = &k_all[sequence_start * ctx.kv_row + h * ctx.head_dim..];
            let values = &v_all[sequence_start * ctx.kv_row + h * ctx.head_dim..];
            let out_head = &mut attn_out[h * ctx.head_dim..(h + 1) * ctx.head_dim];
            online_attention_grouped(
                q_head,
                keys,
                values,
                ctx.kv_row,
                ctx.kv_row,
                sequence_len,
                ctx.head_dim,
                ctx.head_dim,
                1,
                0,
                sequence_len - 1,
                ctx.scale,
                out_head,
            );
        }
    }
}

/// Runs the nomic-bert encoder over `tokens` and returns the last-layer hidden
/// states as an `n_tokens * dim` row-major buffer. Attention is bidirectional
/// (every position attends every position), there is no KV cache, and the
/// architecture is post-norm (LayerNorm after each residual add).
pub fn forward_nomic_bert_hidden(
    config: &Config,
    weights: &NomicBertWeights,
    tokens: &[u32],
) -> Vec<f32> {
    forward_nomic_bert_hidden_impl(config, weights, tokens, true)
}

/// Runs multiple independent Nomic/BERT inputs as one flattened micro-batch
/// and mean-pools each sequence. Projection kernels can then reuse hot weight
/// rows across short texts, while per-token sequence ranges keep attention and
/// RoPE positions identical to separate encoder calls.
pub fn forward_nomic_bert_pooled_batch(
    config: &Config,
    weights: &NomicBertWeights,
    sequences: &[&[u32]],
) -> Vec<Vec<f32>> {
    if sequences.is_empty() {
        return Vec::new();
    }
    let total_tokens = sequences.iter().map(|sequence| sequence.len()).sum();
    let mut tokens = Vec::with_capacity(total_tokens);
    let mut boundaries = Vec::with_capacity(sequences.len() + 1);
    boundaries.push(0);
    for sequence in sequences {
        assert!(
            !sequence.is_empty(),
            "Nomic embedding batch contains an empty sequence"
        );
        tokens.extend_from_slice(sequence);
        boundaries.push(tokens.len());
    }
    let hidden =
        forward_nomic_bert_hidden_segmented_impl(config, weights, &tokens, &boundaries, true);
    let mut pooled = Vec::with_capacity(sequences.len());
    for boundary in boundaries.windows(2) {
        let start = boundary[0];
        let end = boundary[1];
        let mut sum = vec![0.0f32; config.dim];
        for row in hidden[start * config.dim..end * config.dim].chunks_exact(config.dim) {
            for (value, hidden) in sum.iter_mut().zip(row) {
                *value += hidden;
            }
        }
        let scale = 1.0 / (end - start) as f32;
        for value in &mut sum {
            *value *= scale;
        }
        pooled.push(sum);
    }
    pooled
}

/// Internal Nomic BERT forward implementation. The test-only serial switch
/// lets the quantized batched path be checked against the established
/// per-token execution without changing production behavior.
fn forward_nomic_bert_hidden_impl(
    config: &Config,
    weights: &NomicBertWeights,
    tokens: &[u32],
    _allow_batched: bool,
) -> Vec<f32> {
    forward_nomic_bert_hidden_segmented_impl(
        config,
        weights,
        tokens,
        &[0, tokens.len()],
        _allow_batched,
    )
}

fn forward_nomic_bert_hidden_segmented_impl(
    config: &Config,
    weights: &NomicBertWeights,
    tokens: &[u32],
    boundaries: &[usize],
    _allow_batched: bool,
) -> Vec<f32> {
    let n = tokens.len();
    assert!(n > 0, "Nomic encoder requires at least one token");
    assert_eq!(boundaries.first(), Some(&0));
    assert_eq!(boundaries.last(), Some(&n));
    assert!(boundaries.windows(2).all(|range| range[0] < range[1]));
    let dim = config.dim;
    let head_dim = config.head_dim;
    let n_heads = config.n_heads;
    let eps = weights.ln_eps;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let inv_freq = build_rope_inv_freq(config.rope_theta, head_dim, 1.0);
    let kv_row = n_heads * head_dim;

    // Store the local RoPE position and attention bounds for every flattened
    // token once. They are reused by every encoder layer.
    let mut token_positions = vec![0usize; n];
    let mut sequence_starts = vec![0usize; n];
    let mut sequence_ends = vec![n; n];
    for range in boundaries.windows(2) {
        let start = range[0];
        let end = range[1];
        for (position, index) in (start..end).enumerate() {
            token_positions[index] = position;
            sequence_starts[index] = start;
            sequence_ends[index] = end;
        }
    }

    // Embedding + token-type row 0, then embedding LayerNorm.
    let mut hs = vec![0.0f32; n * dim];
    let mut row = vec![0.0f32; dim];
    for (i, &tok) in tokens.iter().enumerate() {
        weights.token_embd.row_into(tok as usize, dim, &mut row);
        if weights.token_type0.len() == dim {
            for j in 0..dim {
                row[j] += weights.token_type0[j];
            }
        }
        let dst = &mut hs[i * dim..(i + 1) * dim];
        dst.copy_from_slice(&row);
        layer_norm_in_place(dst, &weights.tok_norm, &weights.tok_norm_b, eps);
    }

    let mut q_all = vec![0.0f32; n * kv_row];
    let mut k_all = vec![0.0f32; n * kv_row];
    let mut v_all = vec![0.0f32; n * kv_row];
    let mut q_buf = Vec::new();
    let mut k_buf = Vec::new();
    let mut v_buf = Vec::new();
    let mut proj = Vec::new();
    let mut gate_buf = Vec::new();
    let mut up_buf = Vec::new();
    let mut ffn_out = Vec::new();
    let mut attn_out = vec![0.0f32; kv_row];
    // Batched encoder phases reuse these buffers per layer. Very short inputs
    // retain the lower-latency per-token path below.
    #[cfg(not(target_family = "wasm"))]
    let mut attn_all = vec![0.0f32; n * kv_row];
    #[cfg(not(target_family = "wasm"))]
    let mut proj_all: Vec<f32> = Vec::new();
    #[cfg(not(target_family = "wasm"))]
    let mut gate_all: Vec<f32> = Vec::new();
    #[cfg(not(target_family = "wasm"))]
    let mut up_all: Vec<f32> = Vec::new();
    #[cfg(not(target_family = "wasm"))]
    let mut ffn_all: Vec<f32> = Vec::new();

    for layer in &weights.layers {
        #[cfg(not(target_family = "wasm"))]
        if _allow_batched && n >= 8 && nomic_batched_layer_supported(layer, dim, kv_row) {
            // Four batched projection phases plus one batched attention phase
            // replace the old per-token QKV/Wo/gate+up/down jobs. Each
            // projection quantizes every activation once for the batch, then
            // keeps weight rows hot while dynamic worker chunks consume it.
            assert!(try_kquant_matvec3_batch_into(
                &layer.wq, &layer.wk, &layer.wv, &hs, &mut q_all, &mut k_all, &mut v_all,
            ));
            for i in 0..n {
                let q = &mut q_all[i * kv_row..(i + 1) * kv_row];
                let k = &mut k_all[i * kv_row..(i + 1) * kv_row];
                let v = &mut v_all[i * kv_row..(i + 1) * kv_row];
                add_bias_if_present(q, &layer.bq);
                add_bias_if_present(k, &layer.bk);
                add_bias_if_present(v, &layer.bv);
                apply_rope_qk_neox(
                    q,
                    k,
                    token_positions[i],
                    head_dim,
                    n_heads,
                    config.n_kv_heads,
                    &inv_freq,
                );
            }

            let attention = NomicAttentionBatch {
                q: q_all.as_ptr(),
                k: k_all.as_ptr(),
                v: v_all.as_ptr(),
                out: attn_all.as_mut_ptr(),
                n,
                n_heads,
                head_dim,
                kv_row,
                scale,
                sequence_starts: sequence_starts.as_ptr(),
                sequence_ends: sequence_ends.as_ptr(),
            };
            unsafe {
                simd::parallel_range(
                    n,
                    nomic_attention_batch_range,
                    &attention as *const NomicAttentionBatch as *const (),
                );
            }

            assert!(try_kquant_matvec_batch_into(
                &layer.wo,
                &attn_all,
                &mut proj_all,
            ));
            for i in 0..n {
                let projection = &mut proj_all[i * dim..(i + 1) * dim];
                add_bias_if_present(projection, &layer.bo);
                let dst = &mut hs[i * dim..(i + 1) * dim];
                for j in 0..dim {
                    dst[j] += projection[j];
                }
                layer_norm_in_place(dst, &layer.attn_out_norm, &layer.attn_out_norm_b, eps);
            }

            let gate = layer
                .ffn_gate
                .as_ref()
                .expect("batched Nomic path requires a SwiGLU gate");
            assert!(try_kquant_matvec2_batch_into(
                gate,
                &layer.ffn_up,
                &hs,
                &mut gate_all,
                &mut up_all,
            ));
            let ffn_dim = up_all.len() / n;
            for i in 0..n {
                let gate = &gate_all[i * ffn_dim..(i + 1) * ffn_dim];
                let up = &mut up_all[i * ffn_dim..(i + 1) * ffn_dim];
                add_bias_if_present(up, &layer.ffn_up_b);
                for j in 0..ffn_dim {
                    up[j] *= silu(gate[j]);
                }
            }
            assert!(try_kquant_matvec_batch_into(
                &layer.ffn_down,
                &up_all,
                &mut ffn_all,
            ));
            for i in 0..n {
                let ffn = &mut ffn_all[i * dim..(i + 1) * dim];
                add_bias_if_present(ffn, &layer.ffn_down_b);
                let dst = &mut hs[i * dim..(i + 1) * dim];
                for j in 0..dim {
                    dst[j] += ffn[j];
                }
                layer_norm_in_place(dst, &layer.layer_out_norm, &layer.layer_out_norm_b, eps);
            }
            continue;
        }

        // Project Q/K/V for every position from the (already post-normed) hidden
        // state, add biases, then apply NeoX RoPE per position.
        for i in 0..n {
            let x = &hs[i * dim..(i + 1) * dim];
            if !try_quant_matvec3_into(
                &layer.wq, &layer.wk, &layer.wv, x, &mut q_buf, &mut k_buf, &mut v_buf,
            ) {
                layer.wq.matvec_into(x, &mut q_buf);
                layer.wk.matvec_into(x, &mut k_buf);
                layer.wv.matvec_into(x, &mut v_buf);
            }
            add_bias_if_present(&mut q_buf, &layer.bq);
            add_bias_if_present(&mut k_buf, &layer.bk);
            add_bias_if_present(&mut v_buf, &layer.bv);
            apply_rope_qk_neox(
                &mut q_buf,
                &mut k_buf,
                token_positions[i],
                head_dim,
                n_heads,
                config.n_kv_heads,
                &inv_freq,
            );
            q_all[i * kv_row..(i + 1) * kv_row].copy_from_slice(&q_buf);
            k_all[i * kv_row..(i + 1) * kv_row].copy_from_slice(&k_buf);
            v_all[i * kv_row..(i + 1) * kv_row].copy_from_slice(&v_buf);
        }

        // Bidirectional attention per position, then output projection +
        // residual + LayerNorm (post-norm).
        for i in 0..n {
            for value in attn_out.iter_mut() {
                *value = 0.0;
            }
            let sequence_start = sequence_starts[i];
            let sequence_end = sequence_ends[i];
            let sequence_len = sequence_end - sequence_start;
            for h in 0..n_heads {
                let q_head =
                    &q_all[i * kv_row + h * head_dim..i * kv_row + h * head_dim + head_dim];
                let keys = &k_all[sequence_start * kv_row + h * head_dim..];
                let values = &v_all[sequence_start * kv_row + h * head_dim..];
                let out_head = &mut attn_out[h * head_dim..(h + 1) * head_dim];
                online_attention_grouped(
                    q_head,
                    keys,
                    values,
                    kv_row,
                    kv_row,
                    sequence_len,
                    head_dim,
                    head_dim,
                    1,
                    0,
                    sequence_len - 1,
                    scale,
                    out_head,
                );
            }
            layer.wo.matvec_into(&attn_out, &mut proj);
            add_bias_if_present(&mut proj, &layer.bo);
            let dst = &mut hs[i * dim..(i + 1) * dim];
            for j in 0..dim {
                dst[j] += proj[j];
            }
            layer_norm_in_place(dst, &layer.attn_out_norm, &layer.attn_out_norm_b, eps);
        }

        // Feed-forward per position, then residual + LayerNorm.
        for i in 0..n {
            let x = &hs[i * dim..(i + 1) * dim];
            match &layer.ffn_gate {
                Some(gate) => {
                    // SwiGLU: silu(gate(x)) * up(x).
                    if !try_quant_matvec2_into(gate, &layer.ffn_up, x, &mut gate_buf, &mut up_buf) {
                        gate.matvec_into(x, &mut gate_buf);
                        layer.ffn_up.matvec_into(x, &mut up_buf);
                    }
                    add_bias_if_present(&mut up_buf, &layer.ffn_up_b);
                    for j in 0..up_buf.len() {
                        up_buf[j] *= silu(gate_buf[j]);
                    }
                }
                None => {
                    // GELU-sequential: gelu(up(x)).
                    layer.ffn_up.matvec_into(x, &mut up_buf);
                    add_bias_if_present(&mut up_buf, &layer.ffn_up_b);
                    for value in up_buf.iter_mut() {
                        *value = gelu(*value);
                    }
                }
            }
            layer.ffn_down.matvec_into(&up_buf, &mut ffn_out);
            add_bias_if_present(&mut ffn_out, &layer.ffn_down_b);
            let dst = &mut hs[i * dim..(i + 1) * dim];
            for j in 0..dim {
                dst[j] += ffn_out[j];
            }
            layer_norm_in_place(dst, &layer.layer_out_norm, &layer.layer_out_norm_b, eps);
        }
    }

    hs
}
