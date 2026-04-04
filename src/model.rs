// model.rs — LLaMA-architecture model with zero-copy mmap'd weights
//
// Key design: quantized weights stay as raw byte slices pointing into the mmap.
// The SIMD kernels do fused dequant+dot, avoiding intermediate f32 buffers.
// Only normalization weights and embeddings are stored as f32.

use std::collections::HashMap;
use crate::gguf::{GGUFFile, GGMLType};
use crate::simd;

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
        let n_kv_heads = gguf.get_u32(&format!("{}.attention.head_count_kv", p), n_heads as u32) as usize;
        let head_dim = gguf.get_u32(&format!("{}.attention.key_length", p), 0)
            .max((dim / n_heads) as u32) as usize;
        let value_dim = gguf.get_u32(&format!("{}.attention.value_length", p), head_dim as u32) as usize;

        let vocab_size = gguf
            .get_u32(&format!("{}.vocab_size", p), 0)
            .max(
                gguf.metadata
                    .get("tokenizer.ggml.tokens")
                    .and_then(|v| v.as_string_array())
                    .map(|v| v.len() as u32)
                    .unwrap_or(0)
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
            rope_original_context_length: gguf.get_u32(&format!("{}.rope.scaling.original_context_length", p), 0) as usize,
        }
    }
}

// ─── Weight storage: either f32 Vec or raw quantized bytes (zero-copy) ───────

#[derive(Clone)]
pub enum Weight {
    F32(Vec<f32>),
    Quantized {
        data: Vec<u8>,   // owned copy (from mmap slice)
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
            Weight::Quantized { data, dtype, rows, cols } => {
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

    /// Extract one row as f32 values.
    pub fn row(&self, row: usize, cols: usize) -> Vec<f32> {
        match self {
            Weight::F32(data) => {
                let start = row * cols;
                data[start..start + cols].to_vec()
            }
            Weight::Quantized { data, dtype, rows, cols: qcols } => {
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
    pub w1: Weight,  // gate
    pub w2: Weight,  // down
    pub w3: Weight,  // up
}

pub struct ModelWeights {
    pub token_embd: Weight,
    pub output_norm: Vec<f32>,
    pub output: Weight,
    pub layers: Vec<LayerWeights>,
}

pub struct ExpertWeight {
    pub data: Vec<u8>,
    pub dtype: GGMLType,
    pub experts: usize,
    pub rows: usize,
    pub cols: usize,
}

impl ExpertWeight {
    pub fn matvec_expert(&self, expert: usize, x: &[f32]) -> Vec<f32> {
        assert!(expert < self.experts, "expert index out of bounds");
        match self.dtype {
            GGMLType::MXFP4 => {
                let row_bytes = (self.cols / 32) * 17;
                let expert_bytes = self.rows * row_bytes;
                let start = expert * expert_bytes;
                simd::matvec_mxfp4(&self.data[start..start + expert_bytes], x, self.rows, self.cols)
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
    pub k: Vec<Vec<f32>>,  // [layer][pos * kv_dim .. (pos+1) * kv_dim]
    pub v: Vec<Vec<f32>>,
}

impl KVCache {
    pub fn new(n_layers: usize, kv_dim: usize, max_len: usize) -> Self {
        Self {
            k: vec![vec![0.0; max_len * kv_dim]; n_layers],
            v: vec![vec![0.0; max_len * kv_dim]; n_layers],
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
) -> Weight {
    let info = tensors.get(name).unwrap_or_else(|| panic!("Missing tensor: {}", name));
    let numel = info.numel();
    let mut byte_size = info
        .dtype
        .data_size(numel)
        .or_else(|| inferred_sizes.get(name).copied())
        .unwrap_or_else(|| panic!("Unsupported tensor type/size for {}: {:?}", name, info.dtype));
    let offset = data_offset + info.offset as usize;

    if offset.checked_add(byte_size).map(|end| end <= mmap_data.len()).unwrap_or(false) == false {
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

    match info.dtype {
        GGMLType::F32 => {
            let mut data = vec![0.0f32; numel];
            for i in 0..numel {
                data[i] = f32::from_le_bytes([
                    raw_view[i * 4], raw_view[i * 4 + 1], raw_view[i * 4 + 2], raw_view[i * 4 + 3],
                ]);
            }
            Weight::F32(data)
        }
        GGMLType::F16 if force_f32 => {
            let mut data = vec![0.0f32; numel];
            for i in 0..numel {
                data[i] = simd::f16_to_f32(u16::from_le_bytes([raw_view[i * 2], raw_view[i * 2 + 1]]));
            }
            Weight::F32(data)
        }
        GGMLType::F16 => {
            let mut data = vec![0.0f32; numel];
            for i in 0..numel {
                data[i] = simd::f16_to_f32(u16::from_le_bytes([raw_view[i * 2], raw_view[i * 2 + 1]]));
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
            if force_f32 {
                if matches!(info.dtype, GGMLType::Q4_K | GGMLType::Q6_K | GGMLType::MXFP4) {
                    panic!("{} force_f32 dequantization not implemented for {}", format!("{:?}", info.dtype), name);
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
                        let scale = simd::f16_to_f32(u16::from_le_bytes([raw_view[base], raw_view[base + 1]]));
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
                        let scale = simd::f16_to_f32(u16::from_le_bytes([raw_view[base], raw_view[base + 1]]));
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
                let rows = if info.dims.len() >= 2 { info.dims[1] as usize } else { 1 };
                let cols = info.dims[0] as usize;
                Weight::Quantized {
                    data: raw_view.to_vec(),
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
    match load_weight(mmap_data, data_offset, name, tensors, inferred_sizes, true) {
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
) -> ExpertWeight {
    let info = tensors.get(name).unwrap_or_else(|| panic!("Missing tensor: {}", name));
    assert!(info.dims.len() == 3, "Expected 3D expert tensor for {}", name);
    let numel = info.numel();
    let byte_size = info
        .dtype
        .data_size(numel)
        .or_else(|| inferred_sizes.get(name).copied())
        .unwrap_or_else(|| panic!("Unsupported expert tensor type/size for {}: {:?}", name, info.dtype));
    let offset = data_offset + info.offset as usize;
    let raw = &mmap_data[offset..offset + byte_size];
    ExpertWeight {
        data: raw.to_vec(),
        dtype: info.dtype,
        experts: info.dims[2] as usize,
        rows: info.dims[1] as usize,
        cols: info.dims[0] as usize,
    }
}

pub fn load_model(mmap_data: &[u8], gguf: &GGUFFile) -> (Config, ModelWeights) {
    let config = Config::from_gguf(gguf);
    eprintln!("Config: dim={}, layers={}, heads={}/{}, hidden={}, vocab={}, ctx={}",
        config.dim, config.n_layers, config.n_heads, config.n_kv_heads,
        config.hidden_dim, config.vocab_size, config.max_seq_len);

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
        let mut offs: Vec<(u64, usize)> = gguf.tensors.iter().enumerate().map(|(i, t)| (t.offset as u64, i)).collect();
        offs.sort_unstable_by_key(|o| o.0);

        for w in 0..offs.len() {
            let (off, _idx) = offs[w];
            let next_off = if w + 1 < offs.len() { offs[w + 1].0 } else { (mmap_len as u64).saturating_sub(data_offset as u64) };
            let byte_size = if next_off > off { (next_off - off) as usize } else { 0 };
            // Map tensor name to inferred byte_size
            let idx = _idx as usize;
            let name = &gguf.tensors[idx].name;
            inferred_sizes.insert(name.clone(), byte_size);
            let end = data_offset as usize + off as usize + byte_size;
            if end > max_required_end { max_required_end = end; }
        }
    }
    // Embeddings can be quantized; keep native format and dequantize selected rows on demand.
    let token_embd = load_weight(mmap_data, data_offset, "token_embd.weight", &tensor_idx, &inferred_sizes, false);
    let output_norm = load_f32_vec(mmap_data, data_offset, "output_norm.weight", &tensor_idx, &inferred_sizes);

    // Output projection (may be tied)
    let output = if tensor_idx.contains_key("output.weight") {
        load_weight(mmap_data, data_offset, "output.weight", &tensor_idx, &inferred_sizes, false)
    } else {
        eprintln!("Note: output tied to embeddings");
        token_embd.clone()
    };

    // Layers
    let mut layers = Vec::with_capacity(config.n_layers);
    for l in 0..config.n_layers {
        let layer = LayerWeights {
            attn_norm: load_f32_vec(mmap_data, data_offset,
                &format!("blk.{}.attn_norm.weight", l), &tensor_idx, &inferred_sizes),
            wq: load_weight(mmap_data, data_offset,
                &format!("blk.{}.attn_q.weight", l), &tensor_idx, &inferred_sizes, false),
            bq: load_optional_f32_vec(mmap_data, data_offset,
                &format!("blk.{}.attn_q.bias", l), &tensor_idx, &inferred_sizes, config.dim),
            wk: load_weight(mmap_data, data_offset,
                &format!("blk.{}.attn_k.weight", l), &tensor_idx, &inferred_sizes, false),
            bk: load_optional_f32_vec(mmap_data, data_offset,
                &format!("blk.{}.attn_k.bias", l), &tensor_idx, &inferred_sizes, config.kv_dim),
            wv: load_weight(mmap_data, data_offset,
                &format!("blk.{}.attn_v.weight", l), &tensor_idx, &inferred_sizes, false),
            bv: load_optional_f32_vec(mmap_data, data_offset,
                &format!("blk.{}.attn_v.bias", l), &tensor_idx, &inferred_sizes, config.kv_dim),
            wo: load_weight(mmap_data, data_offset,
                &format!("blk.{}.attn_output.weight", l), &tensor_idx, &inferred_sizes, false),
            ffn_norm: load_f32_vec(mmap_data, data_offset,
                &format!("blk.{}.ffn_norm.weight", l), &tensor_idx, &inferred_sizes),
            w1: load_weight(mmap_data, data_offset,
                &format!("blk.{}.ffn_gate.weight", l), &tensor_idx, &inferred_sizes, false),
            w2: load_weight(mmap_data, data_offset,
                &format!("blk.{}.ffn_down.weight", l), &tensor_idx, &inferred_sizes, false),
            w3: load_weight(mmap_data, data_offset,
                &format!("blk.{}.ffn_up.weight", l), &tensor_idx, &inferred_sizes, false),
        };
        layers.push(layer);
        if l == 0 || (l + 1) % 8 == 0 || l + 1 == config.n_layers {
            eprintln!("  Loaded layer {}/{}", l + 1, config.n_layers);
        }
    }

    let weights = ModelWeights { token_embd, output_norm, output, layers };
    (config, weights)
}

pub fn load_gpt_oss_model(mmap_data: &[u8], gguf: &GGUFFile) -> (Config, GptOssWeights) {
    let config = Config::from_gguf(gguf);
    eprintln!("Config: dim={}, layers={}, heads={}/{}, hidden={}, vocab={}, ctx={}",
        config.dim, config.n_layers, config.n_heads, config.n_kv_heads,
        config.hidden_dim, config.vocab_size, config.max_seq_len);

    let tensor_idx: HashMap<String, &crate::gguf::TensorInfo> =
        gguf.tensors.iter().map(|t| (t.name.clone(), t)).collect();
    let data_offset = gguf.data_offset;

    let mut inferred_sizes: HashMap<String, usize> = HashMap::new();
    if !gguf.tensors.is_empty() {
        let mmap_len = mmap_data.len();
        let mut offs: Vec<(u64, usize)> = gguf.tensors.iter().enumerate().map(|(i, t)| (t.offset as u64, i)).collect();
        offs.sort_unstable_by_key(|o| o.0);
        for w in 0..offs.len() {
            let (off, idx) = offs[w];
            let next_off = if w + 1 < offs.len() { offs[w + 1].0 } else { (mmap_len as u64).saturating_sub(data_offset as u64) };
            let byte_size = if next_off > off { (next_off - off) as usize } else { 0 };
            inferred_sizes.insert(gguf.tensors[idx].name.clone(), byte_size);
        }
    }

    let token_embd = load_weight(mmap_data, data_offset, "token_embd.weight", &tensor_idx, &inferred_sizes, false);
    let output_norm = load_f32_vec(mmap_data, data_offset, "output_norm.weight", &tensor_idx, &inferred_sizes);
    let output = load_weight(mmap_data, data_offset, "output.weight", &tensor_idx, &inferred_sizes, false);

    let mut layers = Vec::with_capacity(config.n_layers);
    for l in 0..config.n_layers {
        let layer = GptOssLayerWeights {
            attn_norm: load_f32_vec(mmap_data, data_offset, &format!("blk.{}.attn_norm.weight", l), &tensor_idx, &inferred_sizes),
            wq: load_weight(mmap_data, data_offset, &format!("blk.{}.attn_q.weight", l), &tensor_idx, &inferred_sizes, false),
            bq: load_f32_vec(mmap_data, data_offset, &format!("blk.{}.attn_q.bias", l), &tensor_idx, &inferred_sizes),
            wk: load_weight(mmap_data, data_offset, &format!("blk.{}.attn_k.weight", l), &tensor_idx, &inferred_sizes, false),
            bk: load_f32_vec(mmap_data, data_offset, &format!("blk.{}.attn_k.bias", l), &tensor_idx, &inferred_sizes),
            wv: load_weight(mmap_data, data_offset, &format!("blk.{}.attn_v.weight", l), &tensor_idx, &inferred_sizes, false),
            bv: load_f32_vec(mmap_data, data_offset, &format!("blk.{}.attn_v.bias", l), &tensor_idx, &inferred_sizes),
            wo: load_weight(mmap_data, data_offset, &format!("blk.{}.attn_output.weight", l), &tensor_idx, &inferred_sizes, false),
            bo: load_f32_vec(mmap_data, data_offset, &format!("blk.{}.attn_output.bias", l), &tensor_idx, &inferred_sizes),
            sinks: load_f32_vec(mmap_data, data_offset, &format!("blk.{}.attn_sinks.weight", l), &tensor_idx, &inferred_sizes),
            post_attn_norm: load_f32_vec(mmap_data, data_offset, &format!("blk.{}.post_attention_norm.weight", l), &tensor_idx, &inferred_sizes),
            gate_inp: load_weight(mmap_data, data_offset, &format!("blk.{}.ffn_gate_inp.weight", l), &tensor_idx, &inferred_sizes, false),
            gate_inp_bias: load_f32_vec(mmap_data, data_offset, &format!("blk.{}.ffn_gate_inp.bias", l), &tensor_idx, &inferred_sizes),
            gate_exps: load_expert_weight(mmap_data, data_offset, &format!("blk.{}.ffn_gate_exps.weight", l), &tensor_idx, &inferred_sizes),
            gate_exps_bias: load_weight(mmap_data, data_offset, &format!("blk.{}.ffn_gate_exps.bias", l), &tensor_idx, &inferred_sizes, true),
            up_exps: load_expert_weight(mmap_data, data_offset, &format!("blk.{}.ffn_up_exps.weight", l), &tensor_idx, &inferred_sizes),
            up_exps_bias: load_weight(mmap_data, data_offset, &format!("blk.{}.ffn_up_exps.bias", l), &tensor_idx, &inferred_sizes, true),
            down_exps: load_expert_weight(mmap_data, data_offset, &format!("blk.{}.ffn_down_exps.weight", l), &tensor_idx, &inferred_sizes),
            down_exps_bias: load_weight(mmap_data, data_offset, &format!("blk.{}.ffn_down_exps.bias", l), &tensor_idx, &inferred_sizes, true),
        };
        layers.push(layer);
        if l == 0 || (l + 1) % 8 == 0 || l + 1 == config.n_layers {
            eprintln!("  Loaded layer {}/{}", l + 1, config.n_layers);
        }
    }

    let weights = GptOssWeights { token_embd, output_norm, output, layers };
    (config, weights)
}

// ─── Forward Pass ────────────────────────────────────────────────────────────

/// RMS Normalization
#[inline]
fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let scale = 1.0 / (ss + eps).sqrt();
    x.iter().zip(weight).map(|(xi, wi)| xi * scale * wi).collect()
}

/// Apply RoPE to q/k vectors
#[inline]
fn apply_rope(vec: &mut [f32], pos: usize, head_dim: usize, n_heads: usize, theta: f32) {
    for h in 0..n_heads {
        let off = h * head_dim;
        for i in (0..head_dim).step_by(2) {
            let freq = 1.0 / theta.powf(i as f32 / head_dim as f32);
            let angle = pos as f32 * freq;
            let (sin_a, cos_a) = angle.sin_cos();
            let v0 = vec[off + i];
            let v1 = vec[off + i + 1];
            vec[off + i] = v0 * cos_a - v1 * sin_a;
            vec[off + i + 1] = v0 * sin_a + v1 * cos_a;
        }
    }
}

/// Softmax in-place
#[inline]
fn softmax(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv_sum = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv_sum;
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

fn apply_rope_gpt_oss(q: &mut [f32], k: &mut [f32], pos: usize, config: &Config) {
    let d_half = config.head_dim as f32 / 2.0;
    let mut low = 0.0f32;
    let mut high = 0.0f32;
    if config.rope_scaling_factor > 1.0 {
        low = d_half
            * ((config.rope_original_context_length as f32 / (32.0 * 2.0 * std::f32::consts::PI)).ln()
                / config.rope_theta.ln());
        high = d_half
            * ((config.rope_original_context_length as f32 / (1.0 * 2.0 * std::f32::consts::PI)).ln()
                / config.rope_theta.ln());
    }

    let concentration = if config.rope_scaling_factor > 1.0 {
        0.1 * config.rope_scaling_factor.ln() + 1.0
    } else {
        1.0
    };

    let apply = |vec: &mut [f32], n_heads: usize| {
        for h in 0..n_heads {
            let off = h * config.head_dim;
            for i in (0..config.head_dim).step_by(2) {
                let base_freq = config.rope_theta.powf(i as f32 / config.head_dim as f32);
                let inv_freq = if config.rope_scaling_factor > 1.0 {
                    let idx = i as f32 / 2.0;
                    let ramp = ((idx - low) / (high - low)).clamp(0.0, 1.0);
                    let mask = 1.0 - ramp;
                    let interpolation = 1.0 / (config.rope_scaling_factor * base_freq);
                    let extrapolation = 1.0 / base_freq;
                    interpolation * (1.0 - mask) + extrapolation * mask
                } else {
                    1.0 / base_freq
                };
                let angle = pos as f32 * inv_freq;
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

    apply(q, config.n_heads);
    apply(k, config.n_kv_heads);
}

fn softmax_selected(values: &[(usize, f32)]) -> Vec<f32> {
    let max = values.iter().map(|(_, v)| *v).fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = values.iter().map(|(_, v)| (*v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|v| v / sum).collect()
}

pub fn forward_gpt_oss(
    config: &Config,
    weights: &GptOssWeights,
    cache: &mut KVCache,
    token: u32,
    pos: usize,
) -> Vec<f32> {
    let mut x = weights.token_embd.row(token as usize, config.dim);

    for l in 0..config.n_layers {
        let layer = &weights.layers[l];

        let xn = rms_norm(&x, &layer.attn_norm, config.rms_norm_eps);
        let mut q = layer.wq.matvec(&xn);
        let mut k = layer.wk.matvec(&xn);
        let v = layer.wv.matvec(&xn);
        for i in 0..q.len() { q[i] += layer.bq[i]; }
        for i in 0..k.len() { k[i] += layer.bk[i]; }
        let mut v = v;
        for i in 0..v.len() { v[i] += layer.bv[i]; }

        apply_rope_gpt_oss(&mut q, &mut k, pos, config);

        let kv_start = pos * config.kv_dim;
        cache.k[l][kv_start..kv_start + config.n_kv_heads * config.head_dim].copy_from_slice(&k);
        cache.v[l][kv_start..kv_start + config.n_kv_heads * config.value_dim].copy_from_slice(&v);

        let mut attn_out = vec![0.0f32; config.n_heads * config.value_dim];
        let scale = 1.0 / (config.head_dim as f32).sqrt();
        let attn_window = if l % 2 == 0 && config.sliding_window > 0 {
            pos.saturating_sub(config.sliding_window)
        } else {
            0
        };

        for h in 0..config.n_heads {
            let kv_h = h / config.kv_mul;
            let q_off = h * config.head_dim;

            let mut max_score = layer.sinks[h];
            let mut scores = Vec::with_capacity(pos - attn_window + 1);
            for t in attn_window..=pos {
                let k_off = t * config.kv_dim + kv_h * config.head_dim;
                let score = simd::dot_f32(
                    &q[q_off..q_off + config.head_dim],
                    &cache.k[l][k_off..k_off + config.head_dim],
                ) * scale;
                if score > max_score { max_score = score; }
                scores.push((t, score));
            }

            let sink_exp = (layer.sinks[h] - max_score).exp();
            let mut denom = sink_exp;
            let mut probs = Vec::with_capacity(scores.len());
            for &(t, s) in &scores {
                let p = (s - max_score).exp();
                denom += p;
                probs.push((t, p));
            }
            let inv = 1.0 / denom;
            let out_off = h * config.value_dim;
            for (t, p) in probs {
                let v_off = t * config.kv_dim + kv_h * config.value_dim;
                let p = p * inv;
                for d in 0..config.value_dim {
                    attn_out[out_off + d] += p * cache.v[l][v_off + d];
                }
            }
        }

        let mut attn_proj = layer.wo.matvec(&attn_out);
        for i in 0..attn_proj.len() { attn_proj[i] += layer.bo[i]; }
        for i in 0..config.dim { x[i] += attn_proj[i]; }

        let xn2 = rms_norm(&x, &layer.post_attn_norm, config.rms_norm_eps);
        let mut gate_logits = layer.gate_inp.matvec(&xn2);
        for i in 0..gate_logits.len() { gate_logits[i] += layer.gate_inp_bias[i]; }

        let mut top: Vec<(usize, f32)> = gate_logits.iter().copied().enumerate().collect();
        top.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        top.truncate(config.expert_used_count);
        let expert_probs = softmax_selected(&top);

        let mut moe_sum = vec![0.0f32; config.dim];
        for ((expert_idx, _), expert_prob) in top.into_iter().zip(expert_probs.into_iter()) {
            let gate_bias = layer.gate_exps_bias.row(expert_idx, config.hidden_dim);
            let up_bias = layer.up_exps_bias.row(expert_idx, config.hidden_dim);
            let down_bias = layer.down_exps_bias.row(expert_idx, config.dim);

            let mut gate = layer.gate_exps.matvec_expert(expert_idx, &xn2);
            let mut up = layer.up_exps.matvec_expert(expert_idx, &xn2);
            for i in 0..config.hidden_dim {
                gate[i] += gate_bias[i];
                up[i] += up_bias[i];
                gate[i] = swiglu_gpt_oss(gate[i], up[i]);
            }

            let mut down = layer.down_exps.matvec_expert(expert_idx, &gate);
            for i in 0..config.dim {
                down[i] = (down[i] + down_bias[i]) * expert_prob;
                moe_sum[i] += down[i];
            }
        }

        for i in 0..config.dim { x[i] += moe_sum[i]; }
    }

    let x = rms_norm(&x, &weights.output_norm, config.rms_norm_eps);
    weights.output.matvec(&x)
}

/// Single forward pass for one token at position `pos`
pub fn forward(
    config: &Config,
    weights: &ModelWeights,
    cache: &mut KVCache,
    token: u32,
    pos: usize,
) -> Vec<f32> {
    let dim = config.dim;
    let head_dim = config.head_dim;
    let kv_dim = config.kv_dim;
    let kv_mul = config.kv_mul;

    // Token embedding
    let mut x = weights.token_embd.row(token as usize, dim);

    for l in 0..config.n_layers {
        let layer = &weights.layers[l];

        // ── Attention ──
        let xn = rms_norm(&x, &layer.attn_norm, config.rms_norm_eps);

        let mut q = layer.wq.matvec(&xn);
        let mut k = layer.wk.matvec(&xn);
        let mut v = layer.wv.matvec(&xn);

        for i in 0..q.len() { q[i] += layer.bq[i]; }
        for i in 0..k.len() { k[i] += layer.bk[i]; }
        for i in 0..v.len() { v[i] += layer.bv[i]; }

        apply_rope(&mut q, pos, head_dim, config.n_heads, config.rope_theta);
        apply_rope(&mut k, pos, head_dim, config.n_kv_heads, config.rope_theta);

        // Store KV
        let kv_start = pos * kv_dim;
        cache.k[l][kv_start..kv_start + kv_dim].copy_from_slice(&k);
        cache.v[l][kv_start..kv_start + kv_dim].copy_from_slice(&v);

        // Multi-head attention with GQA
        let mut attn_out = vec![0.0f32; dim];
        let scale = 1.0 / (head_dim as f32).sqrt();

        for h in 0..config.n_heads {
            let kv_h = h / kv_mul;
            let q_off = h * head_dim;

            // Attention scores
            let mut scores = vec![0.0f32; pos + 1];
            for t in 0..=pos {
                let k_off = t * kv_dim + kv_h * head_dim;
                scores[t] = simd::dot_f32(
                    &q[q_off..q_off + head_dim],
                    &cache.k[l][k_off..k_off + head_dim],
                ) * scale;
            }

            softmax(&mut scores);

            // Weighted value sum
            let out_off = h * head_dim;
            for t in 0..=pos {
                let v_off = t * kv_dim + kv_h * head_dim;
                let s = scores[t];
                for d in 0..head_dim {
                    attn_out[out_off + d] += s * cache.v[l][v_off + d];
                }
            }
        }

        // Output projection + residual
        let attn_proj = layer.wo.matvec(&attn_out);
        for i in 0..dim { x[i] += attn_proj[i]; }

        // ── FFN (SwiGLU) ──
        let xn2 = rms_norm(&x, &layer.ffn_norm, config.rms_norm_eps);
        let gate = layer.w1.matvec(&xn2);
        let up = layer.w3.matvec(&xn2);

        let hidden: Vec<f32> = gate.iter().zip(up.iter())
            .map(|(g, u)| silu(*g) * u)
            .collect();

        let ffn_out = layer.w2.matvec(&hidden);
        for i in 0..dim { x[i] += ffn_out[i]; }
    }

    // Final norm → logits
    let x = rms_norm(&x, &weights.output_norm, config.rms_norm_eps);
    weights.output.matvec(&x)
}
