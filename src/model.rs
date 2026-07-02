// model.rs — LLaMA-architecture model with zero-copy mmap'd weights
//
// Key design: quantized weights stay as raw byte slices pointing into the mmap.
// The SIMD kernels do fused dequant+dot, avoiding intermediate f32 buffers.
// Only normalization weights and embeddings are stored as f32.
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

use crate::gguf::{GGMLType, GGUFFile};
use crate::simd;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::OnceLock;

thread_local! {
    // TEMP debug scaffolding for the resident-decoder correctness check;
    // remove once RUSTY_LLM_METAL_RESIDENT is validated.
    static RESIDENT_DEBUG_LOGITS: RefCell<Option<Vec<f32>>> = const { RefCell::new(None) };
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
        let n_heads = gguf.get_u32(&format!("{}.attention.head_count", p), 0) as usize;
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
    match (wq, wk, wv) {
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
            if crate::metal::enabled()
                && (quant_kind_prefers_single_metal(q_kind)
                    || quant_kind_prefers_single_metal(k_kind)
                    || quant_kind_prefers_single_metal(v_kind))
            {
                return false;
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
    match (a, b) {
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
            if crate::metal::enabled()
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
    pub wo: Weight,
    pub ffn_norm: Vec<f32>,
    pub w1: Weight, // gate
    pub w2: Weight, // down
    pub w3: Weight, // up
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
        let data = self.data.as_slice();
        match self.dtype {
            GGMLType::MXFP4 => {
                let row_bytes = (self.cols / 32) * 17;
                let expert_bytes = self.rows * row_bytes;
                let start = expert * expert_bytes;
                simd::matvec_mxfp4(&data[start..start + expert_bytes], x, self.rows, self.cols)
            }
            _ => panic!("Unsupported expert weight dtype: {:?}", self.dtype),
        }
    }

    /// Runs one expert matrix from a mixture-of-experts tensor into a reusable buffer.
    pub fn matvec_expert_into(&self, expert: usize, x: &[f32], out: &mut Vec<f32>) {
        assert!(expert < self.experts, "expert index out of bounds");
        let data = self.data.as_slice();
        match self.dtype {
            GGMLType::MXFP4 => {
                let row_bytes = (self.cols / 32) * 17;
                let expert_bytes = self.rows * row_bytes;
                let start = expert * expert_bytes;
                simd::matvec_mxfp4_into(
                    &data[start..start + expert_bytes],
                    x,
                    self.rows,
                    self.cols,
                    out,
                );
            }
            _ => panic!("Unsupported expert weight dtype: {:?}", self.dtype),
        }
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

// ─── KV Cache ────────────────────────────────────────────────────────────────

pub struct KVCache {
    pub k: Vec<Vec<f32>>, // [layer][slot * per_pos_k_dim ..]
    pub v: Vec<Vec<f32>>,
    pub per_pos_k_dim: usize,
    pub per_pos_v_dim: usize,
    pub max_len: usize,
    pub storage_len: usize,
    pub sliding_window: Option<usize>,
}

impl KVCache {
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
            per_pos_k_dim,
            per_pos_v_dim,
            max_len,
            storage_len,
            sliding_window,
        }
    }

    /// Updates the active sliding window and resizes storage if the ring size changed.
    pub fn set_sliding_window(&mut self, sliding_window: Option<usize>) -> bool {
        let storage_len = Self::storage_len_for(self.max_len, sliding_window);
        let changed = self.sliding_window != sliding_window || self.storage_len != storage_len;
        self.sliding_window = sliding_window;
        if storage_len != self.storage_len {
            self.storage_len = storage_len;
            for layer in &mut self.k {
                layer.resize(storage_len * self.per_pos_k_dim, 0.0);
            }
            for layer in &mut self.v {
                layer.resize(storage_len * self.per_pos_v_dim, 0.0);
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
        KVCache, apply_rope_qk_neox, attention_start_pos, attention_uses_linear_slots,
        build_rope_inv_freq_with_factors,
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
    pub rope_gpt_oss_inv_freq: Vec<f32>,
    pub rope_gpt_oss_concentration: f32,
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
            rope_gpt_oss_inv_freq,
            rope_gpt_oss_concentration,
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

        let gate_name = format!("blk.{}.ffn_gate.weight", l);
        let up_name = format!("blk.{}.ffn_up.weight", l);
        let (w1, w3) = if tensor_idx.contains_key(&gate_name) {
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
            w2: load_weight(
                mmap_data,
                data_offset,
                &format!("blk.{}.ffn_down.weight", l),
                &tensor_idx,
                &inferred_sizes,
                false,
                borrow_quantized,
            ),
            w3,
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

/// Loads GPT-OSS-specific weights from a parsed GGUF file.
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
    let mut max_score = sink_score;
    let mut denom = 1.0f32;
    let linear_slots = attention_uses_linear_slots(start_t, end_t, slot_count);

    for t in start_t..=end_t {
        let slot = if linear_slots { t } else { t % slot_count };
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
    }

    if denom > 0.0 {
        let inv = 1.0 / denom;
        let out_sub = unsafe { out.get_unchecked_mut(..value_head_dim) };
        simd::scale_f32(out_sub, inv);
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
    let mut max_score = f32::NEG_INFINITY;
    let mut denom = 0.0f32;
    let linear_slots = attention_uses_linear_slots(start_t, end_t, slot_count);

    for t in start_t..=end_t {
        let slot = if linear_slots { t } else { t % slot_count };
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
    }

    if denom > 0.0 {
        let inv = 1.0 / denom;
        let out_sub = unsafe { out.get_unchecked_mut(..value_head_dim) };
        simd::scale_f32(out_sub, inv);
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

    let mut max_score = [f32::NEG_INFINITY; MAX_KV_MUL];
    let mut denom = [0.0f32; MAX_KV_MUL];
    let kv_mul = kv_mul.min(MAX_KV_MUL);
    let linear_slots = attention_uses_linear_slots(start_t, end_t, slot_count);

    for t in start_t..=end_t {
        let slot = if linear_slots { t } else { t % slot_count };
        let k_off = slot * key_stride;
        let keys_sub = unsafe { keys.get_unchecked(k_off..k_off + key_head_dim) };
        let v_off = slot * value_stride;
        let value_row = unsafe { values.get_unchecked(v_off..v_off + value_head_dim) };

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

/// Upper bound on GQA group size (`n_heads / n_kv_heads`) across supported
/// architectures; backs the fixed-size scratch arrays in
/// `online_attention_grouped` so it stays allocation-free.
const MAX_KV_MUL: usize = 16;

/// SiLU activation
#[inline(always)]
/// Computes the SiLU activation.
pub(crate) fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline(always)]
/// Computes the tanh-approximate GELU activation used by Gemma feed-forward blocks.
fn gelu(x: f32) -> f32 {
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
            for value in buf.attn_out.iter_mut() {
                *value = 0.0;
            }
            for h in 0..config.n_heads {
                let kv_h = h / config.kv_mul;
                let q_off = h * config.head_dim;
                let out_off = h * config.value_dim;
                online_attention_with_sink(
                    &buf.q[q_off..q_off + config.head_dim],
                    &cache.k[l][kv_h * config.head_dim..],
                    &cache.v[l][kv_h * config.value_dim..],
                    kv_k_dim,
                    kv_v_dim,
                    cache.storage_len,
                    config.head_dim,
                    config.value_dim,
                    attn_window,
                    pos,
                    scale,
                    layer.sinks[h],
                    &mut buf.attn_out[out_off..out_off + config.value_dim],
                );
            }
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

            layer
                .gate_exps
                .matvec_expert_into(expert_idx, &buf.xn2, &mut buf.gate);
            layer
                .up_exps
                .matvec_expert_into(expert_idx, &buf.xn2, &mut buf.up);
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
    ) {
        return false;
    }

    for (l, layer) in weights.layers.iter().enumerate() {
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
    static RESIDENT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = RESIDENT_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    crate::metal::resident_decode_into(&buf.x, pos, config.vocab_size, logits)
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

    let resident_debug = std::env::var("RUSTY_LLM_METAL_RESIDENT_DEBUG").is_ok();
    if resident_debug {
        let mut resident_logits = Vec::new();
        let ok = resident_forward_attempt(config, weights, cache, buf, pos, &mut resident_logits);
        eprintln!(
            "[resident-debug] pos={} ok={} len={}",
            pos,
            ok,
            resident_logits.len()
        );
        RESIDENT_DEBUG_LOGITS
            .with(|cell| *cell.borrow_mut() = if ok { Some(resident_logits) } else { None });
    } else if active_sliding_window(config, cache) == 0
        && resident_forward_attempt(config, weights, cache, buf, pos, logits)
    {
        return;
    }

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

        apply_rope_qk(
            &mut buf.q,
            &mut buf.k,
            pos,
            head_dim,
            config.n_heads,
            config.n_kv_heads,
            &buf.rope_inv_freq,
        );

        // Store KV (keys and values may have different per-head dims)
        let kv_k_dim = cache.per_pos_k_dim;
        let kv_v_dim = cache.per_pos_v_dim;
        let kv_k_start = cache.k_offset(pos);
        let kv_v_start = cache.v_offset(pos);
        // debug log removed
        cache.k[l][kv_k_start..kv_k_start + buf.k.len()].copy_from_slice(&buf.k);
        cache.v[l][kv_v_start..kv_v_start + buf.v.len()].copy_from_slice(&buf.v);

        // Multi-head attention with GQA
        let scale = 1.0 / (head_dim as f32).sqrt();
        // Models with sliding-window attention should ignore cache entries that
        // fall outside the active local context.
        let sliding_window = active_sliding_window(config, cache);
        let attn_window = attention_start_pos(pos, sliding_window);

        if !crate::metal::attention_into(
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
            for kv_h in 0..config.n_kv_heads {
                let q_off = kv_h * kv_mul * head_dim;
                let out_off = kv_h * kv_mul * config.value_dim;
                online_attention_grouped(
                    &buf.q[q_off..q_off + kv_mul * head_dim],
                    &cache.k[l][kv_h * config.head_dim..],
                    &cache.v[l][kv_h * config.value_dim..],
                    kv_k_dim,
                    kv_v_dim,
                    cache.storage_len,
                    head_dim,
                    config.value_dim,
                    kv_mul,
                    attn_window,
                    pos,
                    scale,
                    &mut buf.attn_out[out_off..out_off + kv_mul * config.value_dim],
                );
            }
        }

        if fused_post_attention_ffn
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

        // ── FFN (SwiGLU) ──
        rms_norm_into(&buf.x, &layer.ffn_norm, config.rms_norm_eps, &mut buf.xn2);

        if !try_metal_mistral_ffn_into(&layer.w1, &layer.w3, &layer.w2, &buf.xn2, &mut buf.proj) {
            if !try_quant_matvec2_into(&layer.w1, &layer.w3, &buf.xn2, &mut buf.gate, &mut buf.up) {
                layer.w1.matvec_into(&buf.xn2, &mut buf.gate);
                layer.w3.matvec_into(&buf.xn2, &mut buf.up);
            }

            buf.hidden.resize(config.hidden_dim, 0.0);
            for i in 0..config.hidden_dim {
                buf.hidden[i] = silu(buf.gate[i]) * buf.up[i];
            }

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

    if resident_debug {
        RESIDENT_DEBUG_LOGITS.with(|cell| {
            if let Some(resident_logits) = cell.borrow_mut().take() {
                let n = resident_logits.len().min(logits.len());
                let mut max_abs_diff = 0.0f32;
                let mut sum_abs_diff = 0.0f64;
                for i in 0..n {
                    let d = (resident_logits[i] - logits[i]).abs();
                    max_abs_diff = max_abs_diff.max(d);
                    sum_abs_diff += d as f64;
                }
                let cpu_argmax = logits
                    .iter()
                    .enumerate()
                    .fold((0usize, f32::MIN), |acc, (i, &v)| if v > acc.1 { (i, v) } else { acc });
                let resident_argmax = resident_logits
                    .iter()
                    .enumerate()
                    .fold((0usize, f32::MIN), |acc, (i, &v)| if v > acc.1 { (i, v) } else { acc });
                eprintln!(
                    "[resident-debug] n={} max_abs_diff={:.4} mean_abs_diff={:.6} cpu_argmax={:?} resident_argmax={:?}",
                    n,
                    max_abs_diff,
                    sum_abs_diff / n as f64,
                    cpu_argmax,
                    resident_argmax
                );
            }
        });
    }
}

/// Forward pass for Gemma-4 models (initial implementation mirroring the
/// standard LLaMA-style forward). Bias terms are currently ignored when
/// missing; the loader warns about absent tensors.
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

            // Gemma 4 uses HF/GGML NeoX-style rotate_half layout.
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
            for kv_h in 0..n_kv_heads_l {
                let q_off = kv_h * kv_mul_l * head_dim_l;
                let out_off = kv_h * kv_mul_l * value_dim_l;
                online_attention_grouped(
                    &buf.q[q_off..q_off + kv_mul_l * head_dim_l],
                    &cache.k[kv_cache_layer][kv_h * head_dim_l..],
                    &cache.v[kv_cache_layer][kv_h * value_dim_l..],
                    cache.per_pos_k_dim,
                    cache.per_pos_v_dim,
                    cache.storage_len,
                    head_dim_l,
                    value_dim_l,
                    kv_mul_l,
                    attn_window,
                    pos,
                    scale,
                    &mut buf.attn_out[out_off..out_off + kv_mul_l * value_dim_l],
                );
            }
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

        // ── FFN (SwiGLU-like) ──
        rms_norm_into(&buf.x, &layer.ffn_norm, config.rms_norm_eps, &mut buf.xn2);

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
    let final_logit_softcap = gguf.get_f32("gemma4.final_logit_softcapping", 30.0);
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

        apply_rope_qk(
            &mut buf.q,
            &mut buf.k,
            pos,
            head_dim,
            config.n_heads,
            config.n_kv_heads,
            &buf.rope_inv_freq,
        );

        let kv_k_dim = cache.per_pos_k_dim;
        let kv_v_dim = cache.per_pos_v_dim;
        let kv_k_start = cache.k_offset(pos);
        let kv_v_start = cache.v_offset(pos);
        cache.k[l][kv_k_start..kv_k_start + buf.k.len()].copy_from_slice(&buf.k);
        cache.v[l][kv_v_start..kv_v_start + buf.v.len()].copy_from_slice(&buf.v);

        let scale = 1.0 / (head_dim as f32).sqrt();
        let sliding_window = active_sliding_window(config, cache);
        let attn_window = attention_start_pos(pos, sliding_window);

        if !crate::metal::attention_into(
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
            for kv_h in 0..config.n_kv_heads {
                let q_off = kv_h * kv_mul * head_dim;
                let out_off = kv_h * kv_mul * config.value_dim;
                online_attention_grouped(
                    &buf.q[q_off..q_off + kv_mul * head_dim],
                    &cache.k[l][kv_h * config.head_dim..],
                    &cache.v[l][kv_h * config.value_dim..],
                    kv_k_dim,
                    kv_v_dim,
                    cache.storage_len,
                    head_dim,
                    config.value_dim,
                    kv_mul,
                    attn_window,
                    pos,
                    scale,
                    &mut buf.attn_out[out_off..out_off + kv_mul * config.value_dim],
                );
            }
        }

        layer.wo.matvec_into(&buf.attn_out, &mut buf.proj);
        for i in 0..dim {
            buf.x[i] += buf.proj[i];
        }

        rms_norm_into(&buf.x, &layer.ffn_norm, config.rms_norm_eps, &mut buf.xn2);

        if !try_quant_matvec2_into(&layer.w1, &layer.w3, &buf.xn2, &mut buf.gate, &mut buf.up) {
            layer.w1.matvec_into(&buf.xn2, &mut buf.gate);
            layer.w3.matvec_into(&buf.xn2, &mut buf.up);
        }

        buf.hidden.resize(config.hidden_dim, 0.0);
        for i in 0..config.hidden_dim {
            buf.hidden[i] = silu(buf.gate[i]) * buf.up[i];
        }

        layer.w2.matvec_into(&buf.hidden, &mut buf.proj);
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
        let mut discard = Vec::new();
        if resident_forward_attempt(config, weights, cache, buf, pos, &mut discard) {
            return;
        }
    }
    let _ = forward_hidden_impl(config, weights, cache, buf, token, pos, false);
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
            for value in buf.attn_out.iter_mut() {
                *value = 0.0;
            }
            for h in 0..config.n_heads {
                let kv_h = h / config.kv_mul;
                let q_off = h * config.head_dim;
                let out_off = h * config.value_dim;
                online_attention_with_sink(
                    &buf.q[q_off..q_off + config.head_dim],
                    &cache.k[l][kv_h * config.head_dim..],
                    &cache.v[l][kv_h * config.value_dim..],
                    kv_k_dim,
                    kv_v_dim,
                    cache.storage_len,
                    config.head_dim,
                    config.value_dim,
                    attn_window,
                    pos,
                    scale,
                    layer.sinks[h],
                    &mut buf.attn_out[out_off..out_off + config.value_dim],
                );
            }
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

            layer
                .gate_exps
                .matvec_expert_into(expert_idx, &buf.xn2, &mut buf.gate);
            layer
                .up_exps
                .matvec_expert_into(expert_idx, &buf.xn2, &mut buf.up);
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
            for kv_h in 0..n_kv_heads_l {
                let q_off = kv_h * kv_mul_l * head_dim_l;
                let out_off = kv_h * kv_mul_l * value_dim_l;
                online_attention_grouped(
                    &buf.q[q_off..q_off + kv_mul_l * head_dim_l],
                    &cache.k[kv_cache_layer][kv_h * head_dim_l..],
                    &cache.v[kv_cache_layer][kv_h * value_dim_l..],
                    cache.per_pos_k_dim,
                    cache.per_pos_v_dim,
                    cache.storage_len,
                    head_dim_l,
                    value_dim_l,
                    kv_mul_l,
                    attn_window,
                    pos,
                    scale,
                    &mut buf.attn_out[out_off..out_off + kv_mul_l * value_dim_l],
                );
            }
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
