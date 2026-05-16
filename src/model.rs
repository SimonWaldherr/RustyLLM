// model.rs — LLaMA-architecture model with zero-copy mmap'd weights
//
// Key design: quantized weights stay as raw byte slices pointing into the mmap.
// The SIMD kernels do fused dequant+dot, avoiding intermediate f32 buffers.
// Only normalization weights and embeddings are stored as f32.

use crate::gguf::{GGMLType, GGUFFile};
use crate::simd;
use std::collections::HashMap;
use std::sync::OnceLock;

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
    pub fn from_gguf(gguf: &GGUFFile) -> Self {
        let arch = gguf.get_str("general.architecture").unwrap_or("llama");
        let p = arch.to_string();

        let dim = gguf.get_u32(&format!("{}.embedding_length", p), 0) as usize;
        let n_heads = gguf.get_u32(&format!("{}.attention.head_count", p), 0) as usize;
        let n_kv_heads =
            gguf.get_u32(&format!("{}.attention.head_count_kv", p), n_heads as u32) as usize;
        let head_dim = gguf
            .get_u32(&format!("{}.attention.key_length", p), 0)
            .max((dim / n_heads) as u32) as usize;
        let value_dim =
            gguf.get_u32(&format!("{}.attention.value_length", p), head_dim as u32) as usize;

        let vocab_size = gguf.get_u32(&format!("{}.vocab_size", p), 0).max(
            gguf.metadata
                .get("tokenizer.ggml.tokens")
                .and_then(|v| v.as_string_array())
                .map(|v| v.len() as u32)
                .unwrap_or(0),
        ) as usize;

        Config {
            arch: p.clone(),
            dim,
            hidden_dim: gguf.get_u32(&format!("{}.feed_forward_length", p), 0) as usize,
            n_layers: gguf.get_u32(&format!("{}.block_count", p), 0) as usize,
            n_heads,
            n_kv_heads,
            vocab_size,
            max_seq_len: gguf.get_u32(&format!("{}.context_length", p), 2048) as usize,
            rope_theta: gguf.get_f32(&format!("{}.rope.freq_base", p), 10000.0),
            rms_norm_eps: gguf.get_f32(&format!("{}.attention.layer_norm_rms_epsilon", p), 1e-5),
            head_dim,
            kv_dim: value_dim * n_kv_heads,
            kv_mul: n_heads / n_kv_heads,
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
    fn owned(data: &[u8]) -> Self {
        Self::Owned(data.to_vec())
    }

    fn view(data: &[u8]) -> Self {
        Self::View {
            ptr: data.as_ptr(),
            len: data.len(),
        }
    }

    fn as_slice(&self) -> &[u8] {
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
    /// Matrix-vector multiply: self[rows × cols] · x[cols] → out[rows]
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
                    GGMLType::Q4_0 => simd::matvec_q4_0(data, x, *rows, *cols),
                    GGMLType::Q4_K => simd::matvec_q4_k(data, x, *rows, *cols),
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
                    GGMLType::Q8_0 => simd::matvec_q8_0_into(data, x, *rows, *cols, out),
                    GGMLType::Q4_0 => simd::matvec_q4_0_into(data, x, *rows, *cols, out),
                    GGMLType::Q4_K => simd::matvec_q4_k_into(data, x, *rows, *cols, out),
                    GGMLType::Q6_K => simd::matvec_q6_k_into(data, x, *rows, *cols, out),
                    GGMLType::MXFP4 => simd::matvec_mxfp4_into(data, x, *rows, *cols, out),
                    _ => panic!("Unsupported quantized matvec: {:?}", dtype),
                }
            }
        }
    }

    /// Extract one row as f32 values.
    pub fn row(&self, row: usize, cols: usize) -> Vec<f32> {
        match self {
            Weight::F32(data) => {
                let start = row * cols;
                data[start..start + cols].to_vec()
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
                match dtype {
                    GGMLType::Q8_0 => {
                        let row_bytes = (cols / 32) * 34;
                        simd::dequant_row_q8_0(&data[row * row_bytes..(row + 1) * row_bytes], cols)
                    }
                    GGMLType::Q4_0 => {
                        let row_bytes = (cols / 32) * 18;
                        simd::dequant_row_q4_0(&data[row * row_bytes..(row + 1) * row_bytes], cols)
                    }
                    GGMLType::Q4_K => {
                        let row_bytes = (cols / 256) * 144;
                        simd::dequant_row_q4_k(&data[row * row_bytes..(row + 1) * row_bytes], cols)
                    }
                    GGMLType::Q6_K => {
                        let row_bytes = (cols / 256) * 210;
                        simd::dequant_row_q6_k(&data[row * row_bytes..(row + 1) * row_bytes], cols)
                    }
                    GGMLType::MXFP4 => {
                        let row_bytes = (cols / 32) * 17;
                        simd::dequant_row_mxfp4(&data[row * row_bytes..(row + 1) * row_bytes], cols)
                    }
                    _ => panic!("Unsupported quantized row extraction: {:?}", dtype),
                }
            }
        }
    }

    /// Extract one row as f32 values into caller-owned storage.
    pub fn row_into(&self, row: usize, cols: usize, out: &mut Vec<f32>) {
        out.resize(cols, 0.0);
        match self {
            Weight::F32(data) => {
                let start = row * cols;
                out.copy_from_slice(&data[start..start + cols]);
            }
            Weight::Quantized { .. } => {
                let row_data = self.row(row, cols);
                out.copy_from_slice(&row_data);
            }
        }
    }

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
    pub k: Vec<Vec<f32>>, // [layer][pos * per_pos_k_dim ..]
    pub v: Vec<Vec<f32>>,
    pub per_pos_k_dim: usize,
    pub per_pos_v_dim: usize,
    pub max_len: usize,
}

impl KVCache {
    pub fn new(
        n_layers: usize,
        per_pos_k_dim: usize,
        per_pos_v_dim: usize,
        max_len: usize,
    ) -> Self {
        Self {
            k: vec![vec![0.0; max_len * per_pos_k_dim]; n_layers],
            v: vec![vec![0.0; max_len * per_pos_v_dim]; n_layers],
            per_pos_k_dim,
            per_pos_v_dim,
            max_len,
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
    pub router_logits: Vec<f32>,
    pub top_experts: Vec<(usize, f32)>,
    pub expert_probs: Vec<f32>,
    pub rope_inv_freq: Vec<f32>,
    pub rope_gpt_oss_inv_freq: Vec<f32>,
    pub rope_gpt_oss_concentration: f32,
}

fn build_rope_inv_freq(theta: f32, head_dim: usize, scaling: f32) -> Vec<f32> {
    let pair_count = head_dim / 2;
    let mut inv = vec![0.0f32; pair_count];
    for (pair, slot) in inv.iter_mut().enumerate() {
        let i = (pair * 2) as f32;
        let base_freq = theta.powf(i / head_dim as f32);
        *slot = 1.0 / (scaling * base_freq);
    }
    inv
}

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
            router_logits: vec![0.0; config.expert_count],
            top_experts: Vec::with_capacity(config.expert_count.max(config.expert_used_count)),
            expert_probs: Vec::with_capacity(config.expert_used_count),
            rope_inv_freq,
            rope_gpt_oss_inv_freq,
            rope_gpt_oss_concentration,
        }
    }
}

// ─── Loading ─────────────────────────────────────────────────────────────────

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

    if offset
        .checked_add(byte_size)
        .map(|end| end <= mmap_data.len())
        .unwrap_or(false)
        == false
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
            GGMLType::F32 | GGMLType::F16 => {
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

    // Treat Q5 variants as needing dequantization into f32 for now
    // (SIMD kernels for Q5 aren't implemented yet).
    let effective_force_f32 = force_f32 || matches!(info.dtype, GGMLType::Q5_0 | GGMLType::Q5_1);

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
        GGMLType::Q8_0
        | GGMLType::Q4_0
        | GGMLType::Q4_K
        | GGMLType::Q6_K
        | GGMLType::MXFP4
        | GGMLType::Q8_1
        | GGMLType::Q4_1
        | GGMLType::Q5_0
        | GGMLType::Q5_1 => {
            if effective_force_f32 {
                if matches!(
                    info.dtype,
                    GGMLType::Q4_K | GGMLType::Q6_K | GGMLType::MXFP4
                ) {
                    panic!(
                        "{} force_f32 dequantization not implemented for {}",
                        format!("{:?}", info.dtype),
                        name
                    );
                }
                // Dequantize into f32 vector
                let mut data_f = vec![0.0f32; numel];
                if matches!(
                    info.dtype,
                    GGMLType::Q8_0 | GGMLType::Q8_1 | GGMLType::Q5_0 | GGMLType::Q5_1
                ) {
                    let block_size = 34; // 2 bytes scale + 32 i8
                    let n_blocks = numel / 32;
                    for b in 0..n_blocks {
                        let base = b * block_size;
                        let scale = simd::f16_to_f32(u16::from_le_bytes([
                            raw_view[base],
                            raw_view[base + 1],
                        ]));
                        for i in 0..32 {
                            data_f[b * 32 + i] = scale * (raw_view[base + 2 + i] as i8) as f32;
                        }
                    }
                } else {
                    // Q4_0 / Q4_1
                    let block_size = 18; // 2 bytes scale + 16 bytes (32 nibbles)
                    let n_blocks = numel / 32;
                    for b in 0..n_blocks {
                        let base = b * block_size;
                        let scale = simd::f16_to_f32(u16::from_le_bytes([
                            raw_view[base],
                            raw_view[base + 1],
                        ]));
                        for i in 0..16 {
                            let byte = raw_view[base + 2 + i];
                            let lo = ((byte & 0x0F) as i32 - 8) as f32;
                            let hi = (((byte >> 4) & 0x0F) as i32 - 8) as f32;
                            data_f[b * 32 + i * 2] = scale * lo;
                            data_f[b * 32 + i * 2 + 1] = scale * hi;
                        }
                    }
                }
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

fn load_optional_f32_vec(
    mmap_data: &[u8],
    data_offset: usize,
    name: &str,
    tensors: &HashMap<String, &crate::gguf::TensorInfo>,
    inferred_sizes: &HashMap<String, usize>,
    len: usize,
) -> Vec<f32> {
    if tensors.contains_key(name) {
        load_f32_vec(mmap_data, data_offset, name, tensors, inferred_sizes)
    } else {
        vec![0.0; len]
    }
}

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
            .map(|(i, t)| (t.offset as u64, i))
            .collect();
        offs.sort_unstable_by_key(|o| o.0);

        for w in 0..offs.len() {
            let (off, _idx) = offs[w];
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
            let idx = _idx as usize;
            let name = &gguf.tensors[idx].name;
            inferred_sizes.insert(name.clone(), byte_size);
            let end = data_offset as usize + off as usize + byte_size;
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
    for l in 0..config.n_layers {
        let layer = LayerWeights {
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
            bq: load_optional_f32_vec(
                mmap_data,
                data_offset,
                &format!("blk.{}.attn_q.bias", l),
                &tensor_idx,
                &inferred_sizes,
                config.dim,
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
            bk: load_optional_f32_vec(
                mmap_data,
                data_offset,
                &format!("blk.{}.attn_k.bias", l),
                &tensor_idx,
                &inferred_sizes,
                config.kv_dim,
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
            bv: load_optional_f32_vec(
                mmap_data,
                data_offset,
                &format!("blk.{}.attn_v.bias", l),
                &tensor_idx,
                &inferred_sizes,
                config.kv_dim,
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
            w1: load_weight(
                mmap_data,
                data_offset,
                &format!("blk.{}.ffn_gate.weight", l),
                &tensor_idx,
                &inferred_sizes,
                false,
                borrow_quantized,
            ),
            w2: load_weight(
                mmap_data,
                data_offset,
                &format!("blk.{}.ffn_down.weight", l),
                &tensor_idx,
                &inferred_sizes,
                false,
                borrow_quantized,
            ),
            w3: load_weight(
                mmap_data,
                data_offset,
                &format!("blk.{}.ffn_up.weight", l),
                &tensor_idx,
                &inferred_sizes,
                false,
                borrow_quantized,
            ),
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
            .map(|(i, t)| (t.offset as u64, i))
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

/// RMS Normalization
#[inline]
fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let scale = 1.0 / (ss + eps).sqrt();
    x.iter()
        .zip(weight)
        .map(|(xi, wi)| xi * scale * wi)
        .collect()
}

/// RMS Normalization writing into a pre-allocated output buffer.
#[inline]
fn rms_norm_into(x: &[f32], weight: &[f32], eps: f32, out: &mut Vec<f32>) {
    let n = x.len();
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let scale = 1.0 / (ss + eps).sqrt();
    out.resize(n, 0.0);
    for i in 0..n {
        out[i] = x[i] * scale * weight[i];
    }
}

/// Apply RoPE to q/k vectors
#[inline]
fn apply_rope(vec: &mut [f32], pos: usize, head_dim: usize, n_heads: usize, inv_freq: &[f32]) {
    debug_assert!(inv_freq.len() >= head_dim / 2);
    for h in 0..n_heads {
        let off = h * head_dim;
        let last = head_dim - (head_dim % 2);
        for i in (0..last).step_by(2) {
            let idx0 = off + i;
            let idx1 = off + i + 1;
            if idx1 >= vec.len() {
                break;
            }
            let angle = pos as f32 * inv_freq[i / 2];
            let (sin_a, cos_a) = angle.sin_cos();
            let v0 = vec[idx0];
            let v1 = vec[idx1];
            vec[idx0] = v0 * cos_a - v1 * sin_a;
            vec[idx1] = v0 * sin_a + v1 * cos_a;
        }
    }
}

#[inline]
fn fast_attn_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("RUSTY_LLM_FAST_ATTN").is_some())
}

#[inline(always)]
fn exp_attn(x: f32) -> f32 {
    if fast_attn_enabled() {
        fast_exp_approx(x)
    } else {
        x.exp()
    }
}

#[inline(always)]
fn fast_exp_approx(x: f32) -> f32 {
    // Schraudolph-style approximation; enable only for aggressive throughput mode.
    let xc = x.clamp(-80.0, 80.0);
    let bits = (12102203.0f32 * xc + 1064866805.0f32) as i32;
    f32::from_bits(bits as u32)
}

#[inline]
fn online_attention_with_sink(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    key_stride: usize,
    value_stride: usize,
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

    for t in start_t..=end_t {
        let k_off = t * key_stride;
        let score = simd::dot_f32(query, &keys[k_off..k_off + key_head_dim]) * scale;
        let v_off = t * value_stride;
        let value_row = &values[v_off..v_off + value_head_dim];

        if score > max_score {
            let old_scale = if max_score.is_finite() {
                exp_attn(max_score - score)
            } else {
                0.0
            };
            simd::scale_add_f32(&mut out[..value_head_dim], old_scale, value_row);
            denom = denom * old_scale + 1.0;
            max_score = score;
        } else {
            let weight = exp_attn(score - max_score);
            simd::axpy_f32(&mut out[..value_head_dim], weight, value_row);
            denom += weight;
        }
    }

    if denom > 0.0 {
        let inv = 1.0 / denom;
        simd::scale_f32(&mut out[..value_head_dim], inv);
    }
}

#[inline]
fn online_attention(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    key_stride: usize,
    value_stride: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    start_t: usize,
    end_t: usize,
    scale: f32,
    out: &mut [f32],
) {
    let mut max_score = f32::NEG_INFINITY;
    let mut denom = 0.0f32;

    for t in start_t..=end_t {
        let k_off = t * key_stride;
        let score = simd::dot_f32(query, &keys[k_off..k_off + key_head_dim]) * scale;
        let v_off = t * value_stride;
        let value_row = &values[v_off..v_off + value_head_dim];

        if score > max_score {
            let old_scale = if max_score.is_finite() {
                exp_attn(max_score - score)
            } else {
                0.0
            };
            simd::scale_add_f32(&mut out[..value_head_dim], old_scale, value_row);
            denom = denom * old_scale + 1.0;
            max_score = score;
        } else {
            let weight = exp_attn(score - max_score);
            simd::axpy_f32(&mut out[..value_head_dim], weight, value_row);
            denom += weight;
        }
    }

    if denom > 0.0 {
        let inv = 1.0 / denom;
        simd::scale_f32(&mut out[..value_head_dim], inv);
    }
}

/// SiLU activation
#[inline(always)]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline(always)]
fn swiglu_gpt_oss(g: f32, u: f32) -> f32 {
    let g = g.min(7.0);
    let u = u.clamp(-7.0, 7.0);
    g * (1.0 / (1.0 + (-1.702 * g).exp())) * (u + 1.0)
}

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
    let apply = |vec: &mut [f32], n_heads: usize| {
        for h in 0..n_heads {
            let off = h * head_dim;
            for i in (0..head_dim).step_by(2) {
                let angle = pos as f32 * inv_freq[i / 2];
                let (sin_a, cos_a) = angle.sin_cos();
                let cos_a = cos_a * concentration;
                let sin_a = sin_a * concentration;
                let v0 = vec[off + i];
                let v1 = vec[off + i + 1];
                vec[off + i] = v0 * cos_a - v1 * sin_a;
                vec[off + i + 1] = v0 * sin_a + v1 * cos_a;
            }
        }
    };

    apply(q, n_heads);
    apply(k, n_kv_heads);
}

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

pub fn forward_gpt_oss(
    config: &Config,
    weights: &GptOssWeights,
    cache: &mut KVCache,
    buf: &mut DecodeBuffer,
    token: u32,
    pos: usize,
) -> Vec<f32> {
    weights
        .token_embd
        .row_into(token as usize, config.dim, &mut buf.x);

    for l in 0..config.n_layers {
        let layer = &weights.layers[l];

        rms_norm_into(&buf.x, &layer.attn_norm, config.rms_norm_eps, &mut buf.xn);
        layer.wq.matvec_into(&buf.xn, &mut buf.q);
        layer.wk.matvec_into(&buf.xn, &mut buf.k);
        layer.wv.matvec_into(&buf.xn, &mut buf.v);
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
        let kv_k_start = pos * cache.per_pos_k_dim;
        let kv_v_start = pos * cache.per_pos_v_dim;
        cache.k[l][kv_k_start..kv_k_start + buf.k.len()].copy_from_slice(&buf.k);
        cache.v[l][kv_v_start..kv_v_start + buf.v.len()].copy_from_slice(&buf.v);

        for value in buf.attn_out.iter_mut() {
            *value = 0.0;
        }
        let scale = 1.0 / (config.head_dim as f32).sqrt();
        let attn_window = if l % 2 == 0 && config.sliding_window > 0 {
            pos.saturating_sub(config.sliding_window)
        } else {
            0
        };

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
                config.head_dim,
                config.value_dim,
                attn_window,
                pos,
                scale,
                layer.sinks[h],
                &mut buf.attn_out[out_off..out_off + config.value_dim],
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

        buf.top_experts.clear();
        buf.top_experts
            .extend(buf.router_logits.iter().copied().enumerate());
        buf.top_experts.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        buf.top_experts.truncate(config.expert_used_count);
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
    weights.output.matvec(&buf.xn)
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
    let dim = config.dim;
    let head_dim = config.head_dim;
    let _kv_dim = config.kv_dim;
    let kv_mul = config.kv_mul;

    // Token embedding
    let mut x = weights.token_embd.row(token as usize, dim);

    for l in 0..config.n_layers {
        let layer = &weights.layers[l];

        // ── Attention ──
        rms_norm_into(&x, &layer.attn_norm, config.rms_norm_eps, &mut buf.xn);

        layer.wq.matvec_into(&buf.xn, &mut buf.q);
        layer.wk.matvec_into(&buf.xn, &mut buf.k);
        layer.wv.matvec_into(&buf.xn, &mut buf.v);

        for i in 0..buf.q.len() {
            buf.q[i] += layer.bq[i];
        }
        for i in 0..buf.k.len() {
            buf.k[i] += layer.bk[i];
        }
        for i in 0..buf.v.len() {
            buf.v[i] += layer.bv[i];
        }

        apply_rope(
            &mut buf.q,
            pos,
            head_dim,
            config.n_heads,
            &buf.rope_inv_freq,
        );
        apply_rope(
            &mut buf.k,
            pos,
            head_dim,
            config.n_kv_heads,
            &buf.rope_inv_freq,
        );

        // Store KV (keys and values may have different per-head dims)
        let kv_k_dim = cache.per_pos_k_dim;
        let kv_v_dim = cache.per_pos_v_dim;
        let kv_k_start = pos * cache.per_pos_k_dim;
        let kv_v_start = pos * cache.per_pos_v_dim;
        // debug log removed
        cache.k[l][kv_k_start..kv_k_start + buf.k.len()].copy_from_slice(&buf.k);
        cache.v[l][kv_v_start..kv_v_start + buf.v.len()].copy_from_slice(&buf.v);

        // Multi-head attention with GQA
        let scale = 1.0 / (head_dim as f32).sqrt();
        // Models with sliding-window attention should ignore cache entries that
        // fall outside the active local context.
        let attn_window = if config.sliding_window > 0 {
            pos.saturating_sub(config.sliding_window)
        } else {
            0
        };

        // Zero output buffer before accumulating attention results.
        for v in buf.attn_out.iter_mut() {
            *v = 0.0;
        }

        for h in 0..config.n_heads {
            let kv_h = h / kv_mul;
            let q_off = h * head_dim;
            let out_off = h * config.value_dim;
            online_attention(
                &buf.q[q_off..q_off + head_dim],
                &cache.k[l][kv_h * config.head_dim..],
                &cache.v[l][kv_h * config.value_dim..],
                kv_k_dim,
                kv_v_dim,
                head_dim,
                config.value_dim,
                attn_window,
                pos,
                scale,
                &mut buf.attn_out[out_off..out_off + config.value_dim],
            );
        }

        // Output projection + residual
        layer.wo.matvec_into(&buf.attn_out, &mut buf.proj);
        for i in 0..dim {
            x[i] += buf.proj[i];
        }

        // ── FFN (SwiGLU) ──
        rms_norm_into(&x, &layer.ffn_norm, config.rms_norm_eps, &mut buf.xn2);

        layer.w1.matvec_into(&buf.xn2, &mut buf.gate);
        layer.w3.matvec_into(&buf.xn2, &mut buf.up);

        buf.hidden.resize(config.hidden_dim, 0.0);
        for i in 0..config.hidden_dim {
            buf.hidden[i] = silu(buf.gate[i]) * buf.up[i];
        }

        layer.w2.matvec_into(&buf.hidden, &mut buf.proj);
        for i in 0..dim {
            x[i] += buf.proj[i];
        }
    }

    // Final norm → logits
    rms_norm_into(&x, &weights.output_norm, config.rms_norm_eps, &mut buf.xn);
    weights.output.matvec(&buf.xn)
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
    let dim = config.dim;
    // Per-layer head/value/k_v layout is stored in each Gemma4 layer.
    // `buf` and `cache` are sized using layer maxima; here we use the
    // per-layer descriptors to read/write the correct slices and strides.

    // Token embedding
    let mut x = weights.token_embd.row(token as usize, dim);

    for l in 0..config.n_layers {
        let layer = &weights.layers[l];

        // Standard attention path (or K=V reuse when attn_v is missing)
        rms_norm_into(&x, &layer.attn_norm, config.rms_norm_eps, &mut buf.xn);

        layer.attn_q.matvec_into(&buf.xn, &mut buf.q);
        layer.attn_k.matvec_into(&buf.xn, &mut buf.k);

        // If has_attn_v is false, V = K (K=V reuse)
        let head_dim_l = layer.head_dim;
        let n_kv_heads_l = layer.n_kv_heads;
        let value_dim_l = layer.value_dim;

        if layer.has_attn_v {
            layer.attn_v.matvec_into(&buf.xn, &mut buf.v);
        } else {
            // K=V reuse: copy only the relevant portion of k into v
            let kv_size = n_kv_heads_l * head_dim_l;
            buf.v[..kv_size].copy_from_slice(&buf.k[..kv_size]);
        }

        // Apply RoPE using per-head dims
        apply_rope(
            &mut buf.q,
            pos,
            head_dim_l,
            config.n_heads,
            &buf.rope_inv_freq,
        );
        apply_rope(
            &mut buf.k,
            pos,
            head_dim_l,
            n_kv_heads_l,
            &buf.rope_inv_freq,
        );

        // Store KV into per-pos slots (cache uses fixed per-pos stride)
        // Important: only write the relevant portion based on per-layer dims
        let kv_k_size = n_kv_heads_l * head_dim_l;
        let kv_v_size = n_kv_heads_l * value_dim_l;

        let kv_k_start = pos * cache.per_pos_k_dim;
        let kv_v_start = pos * cache.per_pos_v_dim;
        cache.k[l][kv_k_start..kv_k_start + kv_k_size].copy_from_slice(&buf.k[..kv_k_size]);
        cache.v[l][kv_v_start..kv_v_start + kv_v_size].copy_from_slice(&buf.v[..kv_v_size]);

        // Multi-head attention with GQA
        let scale = 1.0 / (head_dim_l as f32).sqrt();
        let attn_window = if config.sliding_window > 0 {
            pos.saturating_sub(config.sliding_window)
        } else {
            0
        };

        for v in buf.attn_out.iter_mut() {
            *v = 0.0;
        }

        let kv_mul_l = config.n_heads / n_kv_heads_l;
        for h in 0..config.n_heads {
            let kv_h = h / kv_mul_l;
            let q_off = h * head_dim_l;
            let out_off = h * value_dim_l;
            online_attention(
                &buf.q[q_off..q_off + head_dim_l],
                &cache.k[l][kv_h * head_dim_l..],
                &cache.v[l][kv_h * value_dim_l..],
                cache.per_pos_k_dim,
                cache.per_pos_v_dim,
                head_dim_l,
                value_dim_l,
                attn_window,
                pos,
                scale,
                &mut buf.attn_out[out_off..out_off + value_dim_l],
            );
        }

        // Output projection + residual
        layer.attn_output.matvec_into(&buf.attn_out, &mut buf.proj);
        for i in 0..dim {
            x[i] += buf.proj[i];
        }

        // ── FFN (SwiGLU-like) ──
        rms_norm_into(&x, &layer.ffn_norm, config.rms_norm_eps, &mut buf.xn2);

        layer.ffn_gate.matvec_into(&buf.xn2, &mut buf.gate);
        layer.ffn_up.matvec_into(&buf.xn2, &mut buf.up);

        buf.hidden.resize(config.hidden_dim, 0.0);
        for i in 0..config.hidden_dim {
            buf.hidden[i] = silu(buf.gate[i]) * buf.up[i];
        }

        layer.ffn_down.matvec_into(&buf.hidden, &mut buf.proj);
        for i in 0..dim {
            x[i] += buf.proj[i];
        }
    }

    // Final norm → logits
    rms_norm_into(&x, &weights.output_norm, config.rms_norm_eps, &mut buf.xn);
    weights.output.matvec(&buf.xn)
}

#[derive(Clone)]
pub struct Gemma4LayerWeights {
    pub attn_norm: Vec<f32>,
    pub attn_q: Weight,
    pub attn_k: Weight,
    pub attn_v: Weight,
    pub attn_output: Weight,
    pub ffn_norm: Vec<f32>,
    pub ffn_down: Weight,
    pub ffn_up: Weight,
    pub ffn_gate: Weight,
    pub head_dim: usize,
    pub n_kv_heads: usize,
    pub value_dim: usize,
    pub has_attn_v: bool, // True if layer has separate V projection; false = use K as V
                          // TODO: Add biases, global attn, etc. falls benötigt
}

#[derive(Clone)]
pub struct Gemma4Weights {
    pub token_embd: Weight,
    pub output_norm: Vec<f32>,
    pub output: Weight,
    pub layers: Vec<Gemma4LayerWeights>,
}

/// Loader für Gemma-4 Modelle (Stub, lädt nur Grundstruktur)
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
            .map(|(i, t)| (t.offset as u64, i))
            .collect();
        offs.sort_unstable_by_key(|o| o.0);
        for w in 0..offs.len() {
            let (off, _idx) = offs[w];
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
            let idx = _idx as usize;
            let name = &gguf.tensors[idx].name;
            inferred_sizes.insert(name.clone(), byte_size);
        }
    }

    // Infer head/value dims from available tensors (some Gemma-4 GGUFs
    // have unreliable metadata). Prefer inferred shapes when possible.
    {
        for l in 0..config.n_layers {
            let q_name = format!("blk.{}.attn_q.weight", l);
            if let Some(info) = tensor_idx.get(&q_name) {
                eprintln!("Layer {} attn_q shape: {:?}", l, info.dims);
            }
        }
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

    // Infer head/value dims from available tensors
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
        config.kv_dim = config.value_dim * config.n_kv_heads;
        config.kv_mul = config.n_heads / config.n_kv_heads;
        eprintln!(
            "Adjusted Gemma4 config: head_dim={}, value_dim={}, kv_dim={}, kv_mul={}",
            config.head_dim, config.value_dim, config.kv_dim, config.kv_mul
        );
    }

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
                let mut ok = true;
                for s in subs.iter() {
                    if !k.contains(s) {
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
            eprintln!(
                "[WARN] Missing tensor: {} (layer {}) expected shape {}x{} — using zero fallback",
                q_name, l, q_rows, dim
            );
            Weight::F32(vec![0.0; q_rows * dim])
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
            eprintln!(
                "[WARN] Missing tensor: {} (layer {}) expected shape {}x{} — using zero fallback",
                k_name, l, k_rows, dim
            );
            Weight::F32(vec![0.0; k_rows * dim])
        };

        // Load or fallback V
        // Special handling: if V tensor is missing, use K as V (K=V reuse for full-attention layers)
        let has_attn_v = tensor_idx.contains_key(&v_name);
        let attn_v = if has_attn_v {
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
            w
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
            w
        } else {
            // K=V reuse: missing attn_v means use K tensor as V
            // This is common in full-attention/sliding-window layers
            eprintln!(
                "[INFO] Missing tensor: {} (layer {}) — using K as V (K=V reuse)",
                v_name, l
            );
            attn_k.clone()
        };

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
            eprintln!(
                "[WARN] Missing tensor: {} (layer {}) expected shape {}x{} — using zero fallback",
                out_name,
                l,
                out_rows,
                config.n_heads * value_dim_l
            );
            Weight::F32(vec![0.0; out_rows * config.n_heads * value_dim_l])
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
            validate_shape(&ffn_gate_name, l, &w, config.hidden_dim, dim, &config);
            w
        } else {
            eprintln!(
                "[WARN] Missing tensor: {} (layer {}) expected shape {}x{} — using zero fallback",
                ffn_gate_name, l, config.hidden_dim, dim
            );
            Weight::F32(vec![0.0; config.hidden_dim * dim])
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
            validate_shape(&ffn_up_name, l, &w, config.hidden_dim, dim, &config);
            w
        } else {
            eprintln!(
                "[WARN] Missing tensor: {} (layer {}) expected shape {}x{} — using zero fallback",
                ffn_up_name, l, config.hidden_dim, dim
            );
            Weight::F32(vec![0.0; config.hidden_dim * dim])
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
            validate_shape(&ffn_down_name, l, &w, dim, config.hidden_dim, &config);
            w
        } else {
            eprintln!(
                "[WARN] Missing tensor: {} (layer {}) expected shape {}x{} — using zero fallback",
                ffn_down_name, l, dim, config.hidden_dim
            );
            Weight::F32(vec![0.0; dim * config.hidden_dim])
        };

        let has_attn_v = tensor_idx.contains_key(&v_name);
        let layer = Gemma4LayerWeights {
            attn_norm: if tensor_idx.contains_key(&format!("blk.{}.attn_norm.weight", l)) {
                load_f32_vec(
                    mmap_data,
                    data_offset,
                    &format!("blk.{}.attn_norm.weight", l),
                    &tensor_idx,
                    &inferred_sizes,
                )
            } else {
                eprintln!(
                    "[WARN] Missing tensor: blk.{}.attn_norm.weight (layer {})",
                    l, l
                );
                vec![0.0; dim]
            },
            attn_q,
            attn_k,
            attn_v,
            attn_output,
            ffn_norm: if tensor_idx.contains_key(&format!("blk.{}.ffn_norm.weight", l)) {
                load_f32_vec(
                    mmap_data,
                    data_offset,
                    &format!("blk.{}.ffn_norm.weight", l),
                    &tensor_idx,
                    &inferred_sizes,
                )
            } else {
                eprintln!(
                    "[WARN] Missing tensor: blk.{}.ffn_norm.weight (layer {})",
                    l, l
                );
                vec![0.0; dim]
            },
            ffn_down,
            ffn_up,
            ffn_gate,
            head_dim: head_dim_l,
            n_kv_heads: n_kv_heads_l,
            value_dim: value_dim_l,
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
pub fn forward_hidden(
    config: &Config,
    weights: &ModelWeights,
    cache: &mut KVCache,
    buf: &mut DecodeBuffer,
    token: u32,
    pos: usize,
) -> Vec<f32> {
    let dim = config.dim;
    let head_dim = config.head_dim;
    let kv_mul = config.kv_mul;

    let mut x = weights.token_embd.row(token as usize, dim);

    for l in 0..config.n_layers {
        let layer = &weights.layers[l];

        rms_norm_into(&x, &layer.attn_norm, config.rms_norm_eps, &mut buf.xn);

        layer.wq.matvec_into(&buf.xn, &mut buf.q);
        layer.wk.matvec_into(&buf.xn, &mut buf.k);
        layer.wv.matvec_into(&buf.xn, &mut buf.v);

        for i in 0..buf.q.len() {
            buf.q[i] += layer.bq[i];
        }
        for i in 0..buf.k.len() {
            buf.k[i] += layer.bk[i];
        }
        for i in 0..buf.v.len() {
            buf.v[i] += layer.bv[i];
        }

        apply_rope(
            &mut buf.q,
            pos,
            head_dim,
            config.n_heads,
            &buf.rope_inv_freq,
        );
        apply_rope(
            &mut buf.k,
            pos,
            head_dim,
            config.n_kv_heads,
            &buf.rope_inv_freq,
        );

        let kv_k_dim = cache.per_pos_k_dim;
        let kv_v_dim = cache.per_pos_v_dim;
        let kv_k_start = pos * cache.per_pos_k_dim;
        let kv_v_start = pos * cache.per_pos_v_dim;
        cache.k[l][kv_k_start..kv_k_start + buf.k.len()].copy_from_slice(&buf.k);
        cache.v[l][kv_v_start..kv_v_start + buf.v.len()].copy_from_slice(&buf.v);

        let scale = 1.0 / (head_dim as f32).sqrt();
        let attn_window = if config.sliding_window > 0 {
            pos.saturating_sub(config.sliding_window)
        } else {
            0
        };

        for v in buf.attn_out.iter_mut() {
            *v = 0.0;
        }

        for h in 0..config.n_heads {
            let kv_h = h / kv_mul;
            let q_off = h * head_dim;
            let out_off = h * config.value_dim;
            online_attention(
                &buf.q[q_off..q_off + head_dim],
                &cache.k[l][kv_h * config.head_dim..],
                &cache.v[l][kv_h * config.value_dim..],
                kv_k_dim,
                kv_v_dim,
                head_dim,
                config.value_dim,
                attn_window,
                pos,
                scale,
                &mut buf.attn_out[out_off..out_off + config.value_dim],
            );
        }

        layer.wo.matvec_into(&buf.attn_out, &mut buf.proj);
        for i in 0..dim {
            x[i] += buf.proj[i];
        }

        rms_norm_into(&x, &layer.ffn_norm, config.rms_norm_eps, &mut buf.xn2);

        layer.w1.matvec_into(&buf.xn2, &mut buf.gate);
        layer.w3.matvec_into(&buf.xn2, &mut buf.up);

        buf.hidden.resize(config.hidden_dim, 0.0);
        for i in 0..config.hidden_dim {
            buf.hidden[i] = silu(buf.gate[i]) * buf.up[i];
        }

        layer.w2.matvec_into(&buf.hidden, &mut buf.proj);
        for i in 0..dim {
            x[i] += buf.proj[i];
        }
    }

    // Apply final norm but skip the output projection — return the hidden state.
    rms_norm(&x, &weights.output_norm, config.rms_norm_eps)
}

/// Forward for GPT-OSS (MoE) models; returns the normalized hidden state.
pub fn forward_hidden_gpt_oss(
    config: &Config,
    weights: &GptOssWeights,
    cache: &mut KVCache,
    buf: &mut DecodeBuffer,
    token: u32,
    pos: usize,
) -> Vec<f32> {
    weights
        .token_embd
        .row_into(token as usize, config.dim, &mut buf.x);

    for l in 0..config.n_layers {
        let layer = &weights.layers[l];

        rms_norm_into(&buf.x, &layer.attn_norm, config.rms_norm_eps, &mut buf.xn);
        layer.wq.matvec_into(&buf.xn, &mut buf.q);
        layer.wk.matvec_into(&buf.xn, &mut buf.k);
        layer.wv.matvec_into(&buf.xn, &mut buf.v);
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
        let kv_k_start = pos * cache.per_pos_k_dim;
        let kv_v_start = pos * cache.per_pos_v_dim;
        cache.k[l][kv_k_start..kv_k_start + buf.k.len()].copy_from_slice(&buf.k);
        cache.v[l][kv_v_start..kv_v_start + buf.v.len()].copy_from_slice(&buf.v);

        for value in buf.attn_out.iter_mut() {
            *value = 0.0;
        }
        let scale = 1.0 / (config.head_dim as f32).sqrt();
        let attn_window = if l % 2 == 0 && config.sliding_window > 0 {
            pos.saturating_sub(config.sliding_window)
        } else {
            0
        };

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
                config.head_dim,
                config.value_dim,
                attn_window,
                pos,
                scale,
                layer.sinks[h],
                &mut buf.attn_out[out_off..out_off + config.value_dim],
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

        buf.top_experts.clear();
        buf.top_experts
            .extend(buf.router_logits.iter().copied().enumerate());
        buf.top_experts.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        buf.top_experts.truncate(config.expert_used_count);
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

    rms_norm_into(
        &buf.x,
        &weights.output_norm,
        config.rms_norm_eps,
        &mut buf.xn,
    );
    buf.xn.clone()
}

/// Forward for Gemma-4 models; returns the normalized hidden state.
pub fn forward_hidden_gemma4(
    config: &Config,
    weights: &Gemma4Weights,
    cache: &mut KVCache,
    buf: &mut DecodeBuffer,
    token: u32,
    pos: usize,
) -> Vec<f32> {
    let dim = config.dim;

    let mut x = weights.token_embd.row(token as usize, dim);

    for l in 0..config.n_layers {
        let layer = &weights.layers[l];

        rms_norm_into(&x, &layer.attn_norm, config.rms_norm_eps, &mut buf.xn);

        layer.attn_q.matvec_into(&buf.xn, &mut buf.q);
        layer.attn_k.matvec_into(&buf.xn, &mut buf.k);

        let head_dim_l = layer.head_dim;
        let n_kv_heads_l = layer.n_kv_heads;
        let value_dim_l = layer.value_dim;

        if layer.has_attn_v {
            layer.attn_v.matvec_into(&buf.xn, &mut buf.v);
        } else {
            let kv_size = n_kv_heads_l * head_dim_l;
            buf.v[..kv_size].copy_from_slice(&buf.k[..kv_size]);
        }

        apply_rope(
            &mut buf.q,
            pos,
            head_dim_l,
            config.n_heads,
            &buf.rope_inv_freq,
        );
        apply_rope(
            &mut buf.k,
            pos,
            head_dim_l,
            n_kv_heads_l,
            &buf.rope_inv_freq,
        );

        let kv_k_size = n_kv_heads_l * head_dim_l;
        let kv_v_size = n_kv_heads_l * value_dim_l;

        let kv_k_start = pos * cache.per_pos_k_dim;
        let kv_v_start = pos * cache.per_pos_v_dim;
        cache.k[l][kv_k_start..kv_k_start + kv_k_size].copy_from_slice(&buf.k[..kv_k_size]);
        cache.v[l][kv_v_start..kv_v_start + kv_v_size].copy_from_slice(&buf.v[..kv_v_size]);

        let scale = 1.0 / (head_dim_l as f32).sqrt();
        let attn_window = if config.sliding_window > 0 {
            pos.saturating_sub(config.sliding_window)
        } else {
            0
        };

        for v in buf.attn_out.iter_mut() {
            *v = 0.0;
        }

        let kv_mul_l = config.n_heads / n_kv_heads_l;
        for h in 0..config.n_heads {
            let kv_h = h / kv_mul_l;
            let q_off = h * head_dim_l;
            let out_off = h * value_dim_l;
            online_attention(
                &buf.q[q_off..q_off + head_dim_l],
                &cache.k[l][kv_h * head_dim_l..],
                &cache.v[l][kv_h * value_dim_l..],
                cache.per_pos_k_dim,
                cache.per_pos_v_dim,
                head_dim_l,
                value_dim_l,
                attn_window,
                pos,
                scale,
                &mut buf.attn_out[out_off..out_off + value_dim_l],
            );
        }

        layer.attn_output.matvec_into(&buf.attn_out, &mut buf.proj);
        for i in 0..dim {
            x[i] += buf.proj[i];
        }

        rms_norm_into(&x, &layer.ffn_norm, config.rms_norm_eps, &mut buf.xn2);

        layer.ffn_gate.matvec_into(&buf.xn2, &mut buf.gate);
        layer.ffn_up.matvec_into(&buf.xn2, &mut buf.up);

        buf.hidden.resize(config.hidden_dim, 0.0);
        for i in 0..config.hidden_dim {
            buf.hidden[i] = silu(buf.gate[i]) * buf.up[i];
        }

        layer.ffn_down.matvec_into(&buf.hidden, &mut buf.proj);
        for i in 0..dim {
            x[i] += buf.proj[i];
        }
    }

    rms_norm(&x, &weights.output_norm, config.rms_norm_eps)
}
