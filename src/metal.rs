use std::cell::Cell;
use std::sync::OnceLock;

pub const Q4K_MIN_METAL_ROWS: usize = 8_192;
pub const Q4K_MIN_METAL_COLS: usize = 4_096;
pub const Q6K_MIN_METAL_ROWS: usize = 2_048;
pub const FUSED_KQUANT_MIN_METAL_ROWS: usize = 512;
/// Minimum `rows * cols` for a fused K-quant projection to be worth a blocking
/// Metal command buffer. Below this, dispatch+sync latency exceeds the GPU
/// matvec: gemma-4-E2B QKV (~6.3M weights) loses to CPU while Ministral-3B
/// QKV (~15.7M) wins (see BENCHMARK.md). Env-tunable via
/// `RUSTY_LLM_METAL_FUSED_MIN_WORK`.
pub const FUSED_KQUANT_MIN_METAL_WORK: usize = 12_000_000;
static FUSED_KQUANT_MIN_METAL_WORK_RUNTIME: OnceLock<usize> = OnceLock::new();
pub const ATTENTION_MIN_METAL_TOKENS: usize = 8_192;
pub const ULTRA_ATTENTION_MIN_METAL_TOKENS: usize = 512;
pub const ULTRA_Q4K_MIN_METAL_ROWS: usize = 512;
pub const ULTRA_Q6K_MIN_METAL_ROWS: usize = 512;
static ATTENTION_MIN_METAL_TOKENS_RUNTIME: OnceLock<usize> = OnceLock::new();
static ULTRA_ATTENTION_MIN_METAL_TOKENS_RUNTIME: OnceLock<usize> = OnceLock::new();
static ULTRA_Q4K_MIN_METAL_ROWS_RUNTIME: OnceLock<usize> = OnceLock::new();
static ULTRA_Q6K_MIN_METAL_ROWS_RUNTIME: OnceLock<usize> = OnceLock::new();

thread_local! {
    static ULTRA_MODE: Cell<bool> = const { Cell::new(false) };
    static CPU_ONLY_MODE: Cell<bool> = const { Cell::new(false) };
}

/// Restores the previous per-thread Metal ultra-mode flag when dropped.
pub struct UltraModeGuard {
    previous: bool,
}

impl Drop for UltraModeGuard {
    fn drop(&mut self) {
        ULTRA_MODE.with(|flag| flag.set(self.previous));
    }
}

/// Restores the previous per-thread backend dispatch policy when dropped.
pub struct DispatchPolicyGuard {
    previous_ultra: bool,
    previous_cpu_only: bool,
}

impl Drop for DispatchPolicyGuard {
    fn drop(&mut self) {
        ULTRA_MODE.with(|flag| flag.set(self.previous_ultra));
        CPU_ONLY_MODE.with(|flag| flag.set(self.previous_cpu_only));
    }
}

fn parse_attention_min_metal_tokens(raw: Option<&str>) -> usize {
    raw.and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(ATTENTION_MIN_METAL_TOKENS)
}

fn parse_usize_or(raw: Option<&str>, default: usize) -> usize {
    raw.and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

/// Returns the attention window size threshold that enables Metal.
pub fn attention_min_metal_tokens() -> usize {
    *ATTENTION_MIN_METAL_TOKENS_RUNTIME.get_or_init(|| {
        let raw = std::env::var("RUSTY_LLM_METAL_ATTENTION_MIN_TOKENS").ok();
        parse_attention_min_metal_tokens(raw.as_deref())
    })
}

/// Returns the short-context threshold used by explicit Mistral ultra mode.
pub fn ultra_attention_min_metal_tokens() -> usize {
    *ULTRA_ATTENTION_MIN_METAL_TOKENS_RUNTIME.get_or_init(|| {
        let raw = std::env::var("RUSTY_LLM_METAL_ULTRA_ATTENTION_MIN_TOKENS").ok();
        parse_usize_or(raw.as_deref(), ULTRA_ATTENTION_MIN_METAL_TOKENS)
    })
}

/// Returns the Q4_K row threshold used by explicit Mistral ultra mode.
pub fn ultra_q4k_min_metal_rows() -> usize {
    *ULTRA_Q4K_MIN_METAL_ROWS_RUNTIME.get_or_init(|| {
        let raw = std::env::var("RUSTY_LLM_METAL_ULTRA_Q4K_MIN_ROWS").ok();
        parse_usize_or(raw.as_deref(), ULTRA_Q4K_MIN_METAL_ROWS)
    })
}

/// Returns the Q6_K row threshold used by explicit Mistral ultra mode.
pub fn ultra_q6k_min_metal_rows() -> usize {
    *ULTRA_Q6K_MIN_METAL_ROWS_RUNTIME.get_or_init(|| {
        let raw = std::env::var("RUSTY_LLM_METAL_ULTRA_Q6K_MIN_ROWS").ok();
        parse_usize_or(raw.as_deref(), ULTRA_Q6K_MIN_METAL_ROWS)
    })
}

/// Enables aggressive Metal routing for the current thread while the guard lives.
pub fn scoped_ultra_mode(enabled: bool) -> UltraModeGuard {
    ULTRA_MODE.with(|flag| {
        let previous = flag.replace(enabled);
        UltraModeGuard { previous }
    })
}

/// Sets the Metal dispatch policy for the current runtime call.
pub fn scoped_dispatch_policy(cpu_only: bool, ultra: bool) -> DispatchPolicyGuard {
    let previous_ultra = ULTRA_MODE.with(|flag| flag.replace(ultra));
    let previous_cpu_only = CPU_ONLY_MODE.with(|flag| flag.replace(cpu_only));
    DispatchPolicyGuard {
        previous_ultra,
        previous_cpu_only,
    }
}

/// Reports whether the current thread is using aggressive Metal routing.
pub fn ultra_mode_enabled() -> bool {
    ULTRA_MODE.with(Cell::get)
}

/// Reports whether Metal kernels may be dispatched on this thread.
pub fn dispatch_enabled() -> bool {
    enabled() && !CPU_ONLY_MODE.with(Cell::get)
}

#[cfg(all(target_os = "macos", rusty_metal))]
mod ffi {
    unsafe extern "C" {
        /// Returns whether the Objective-C Metal backend initialized successfully.
        pub fn rusty_metal_available() -> i32;
        /// Runs one Q4_K matrix-vector multiply on the Metal backend.
        pub fn rusty_metal_q4k_matvec(
            weights: *const u8,
            weights_len: usize,
            x: *const f32,
            rows: usize,
            cols: usize,
            out: *mut f32,
        ) -> i32;
        /// Runs one Q6_K matrix-vector multiply on the Metal backend.
        pub fn rusty_metal_q6k_matvec(
            weights: *const u8,
            weights_len: usize,
            x: *const f32,
            rows: usize,
            cols: usize,
            out: *mut f32,
        ) -> i32;
        /// Runs two Q6_K projections in one Metal dispatch.
        pub fn rusty_metal_q6k_matvec2(
            weights_a: *const u8,
            weights_a_len: usize,
            rows_a: usize,
            weights_b: *const u8,
            weights_b_len: usize,
            rows_b: usize,
            x: *const f32,
            cols: usize,
            out_a: *mut f32,
            out_b: *mut f32,
        ) -> i32;
        /// Runs three Q6_K projections in one Metal dispatch.
        pub fn rusty_metal_q6k_matvec3(
            weights_a: *const u8,
            weights_a_len: usize,
            rows_a: usize,
            weights_b: *const u8,
            weights_b_len: usize,
            rows_b: usize,
            weights_c: *const u8,
            weights_c_len: usize,
            rows_c: usize,
            x: *const f32,
            cols: usize,
            out_a: *mut f32,
            out_b: *mut f32,
            out_c: *mut f32,
        ) -> i32;
        /// Runs two Q4_K projections in one Metal dispatch.
        pub fn rusty_metal_q4k_matvec2(
            weights_a: *const u8,
            weights_a_len: usize,
            rows_a: usize,
            weights_b: *const u8,
            weights_b_len: usize,
            rows_b: usize,
            x: *const f32,
            cols: usize,
            out_a: *mut f32,
            out_b: *mut f32,
        ) -> i32;
        /// Runs three Q4_K projections in one Metal dispatch.
        pub fn rusty_metal_q4k_matvec3(
            weights_a: *const u8,
            weights_a_len: usize,
            rows_a: usize,
            weights_b: *const u8,
            weights_b_len: usize,
            rows_b: usize,
            weights_c: *const u8,
            weights_c_len: usize,
            rows_c: usize,
            x: *const f32,
            cols: usize,
            out_a: *mut f32,
            out_b: *mut f32,
            out_c: *mut f32,
        ) -> i32;
        /// Runs Q4_K, Q4_K, and Q6_K projections in one Metal dispatch.
        pub fn rusty_metal_q4k_q4k_q6k_matvec3(
            weights_a: *const u8,
            weights_a_len: usize,
            rows_a: usize,
            weights_b: *const u8,
            weights_b_len: usize,
            rows_b: usize,
            weights_c: *const u8,
            weights_c_len: usize,
            rows_c: usize,
            x: *const f32,
            cols: usize,
            out_a: *mut f32,
            out_b: *mut f32,
            out_c: *mut f32,
        ) -> i32;
        /// Runs a Mistral-style Q4_K/Q4_K/Q6_K FFN block in one Metal command buffer.
        pub fn rusty_metal_q4k_q4k_q6k_ffn(
            gate_weights: *const u8,
            gate_weights_len: usize,
            up_weights: *const u8,
            up_weights_len: usize,
            down_weights: *const u8,
            down_weights_len: usize,
            x: *const f32,
            input_cols: usize,
            hidden_rows: usize,
            down_rows: usize,
            down_cols: usize,
            out: *mut f32,
        ) -> i32;
        /// Runs Mistral post-attention output projection, residual norm, and FFN in one command buffer.
        pub fn rusty_metal_mistral_post_attention_ffn(
            wo_weights: *const u8,
            wo_weights_len: usize,
            gate_weights: *const u8,
            gate_weights_len: usize,
            up_weights: *const u8,
            up_weights_len: usize,
            down_weights: *const u8,
            down_weights_len: usize,
            x: *mut f32,
            dim: usize,
            attn_out: *const f32,
            attn_cols: usize,
            ffn_norm: *const f32,
            rms_eps: f32,
            hidden_rows: usize,
            down_rows: usize,
            down_cols: usize,
        ) -> i32;
        /// Runs one Q4_0 matrix-vector multiply on the Metal backend.
        pub fn rusty_metal_q4_0_matvec(
            weights: *const u8,
            weights_len: usize,
            x: *const f32,
            rows: usize,
            cols: usize,
            out: *mut f32,
        ) -> i32;
        /// Runs two Q4_0 projections sharing one activation and synchronization point.
        pub fn rusty_metal_q4_0_matvec2(
            weights_a: *const u8,
            weights_a_len: usize,
            rows_a: usize,
            weights_b: *const u8,
            weights_b_len: usize,
            rows_b: usize,
            x: *const f32,
            cols: usize,
            out_a: *mut f32,
            out_b: *mut f32,
        ) -> i32;
        /// Runs three Q4_0 projections sharing one activation and synchronization point.
        pub fn rusty_metal_q4_0_matvec3(
            weights_a: *const u8,
            weights_a_len: usize,
            rows_a: usize,
            weights_b: *const u8,
            weights_b_len: usize,
            rows_b: usize,
            weights_c: *const u8,
            weights_c_len: usize,
            rows_c: usize,
            x: *const f32,
            cols: usize,
            out_a: *mut f32,
            out_b: *mut f32,
            out_c: *mut f32,
        ) -> i32;
        /// Runs a Q4_0 GELU feed-forward block in one Metal command buffer.
        pub fn rusty_metal_q4_0_gelu_ffn(
            gate_weights: *const u8,
            gate_weights_len: usize,
            up_weights: *const u8,
            up_weights_len: usize,
            down_weights: *const u8,
            down_weights_len: usize,
            x: *const f32,
            input_cols: usize,
            hidden_rows: usize,
            down_rows: usize,
            down_cols: usize,
            out: *mut f32,
        ) -> i32;
        /// Runs one Q8_0 matrix-vector multiply on the Metal backend.
        pub fn rusty_metal_q8_0_matvec(
            weights: *const u8,
            weights_len: usize,
            x: *const f32,
            rows: usize,
            cols: usize,
            out: *mut f32,
        ) -> i32;
        /// Runs one attention scan over cached keys and values on the Metal backend.
        pub fn rusty_metal_attention(
            query: *const f32,
            query_len: usize,
            keys: *const f32,
            keys_len: usize,
            values: *const f32,
            values_len: usize,
            sinks: *const f32,
            sinks_len: usize,
            out: *mut f32,
            out_len: usize,
            heads: usize,
            kv_mul: usize,
            head_dim: usize,
            value_dim: usize,
            key_stride: usize,
            value_stride: usize,
            slot_count: usize,
            start_t: usize,
            end_t: usize,
            scale: f32,
            use_sink: i32,
        ) -> i32;
        #[cfg(test)]
        /// Test-only shim for the serial and temporally parallel resident attention kernels.
        pub fn rusty_metal_test_resident_attention(
            query: *const f32,
            keys: *const f32,
            values: *const f32,
            out: *mut f32,
            heads: u32,
            kv_mul: u32,
            head_dim: u32,
            value_dim: u32,
            key_stride: u32,
            value_stride: u32,
            slot_count: u32,
            start_t: u32,
            end_t: u32,
            scale: f32,
            parallel: i32,
        ) -> i32;
        #[cfg(test)]
        pub fn rusty_metal_test_greedy_argmax(
            logits: *const f32,
            vocab: u32,
            recent: *const u32,
            recent_len: u32,
            repeat_penalty: f32,
            token_out: *mut u32,
        ) -> i32;

        /// Configures the GPU-resident decoder (allocates resident buffers).
        pub fn rusty_metal_resident_configure(
            n_layers: u32,
            dim: u32,
            n_heads: u32,
            n_kv_heads: u32,
            head_dim: u32,
            value_dim: u32,
            hidden_dim: u32,
            vocab: u32,
            storage_len: u32,
            eps: f32,
            neox: u32,
            prefer_half_inputs: u32,
        ) -> i32;
        /// Registers one transformer layer's weights with the resident decoder.
        pub fn rusty_metal_resident_set_layer(l: u32, desc: *const super::ResidentLayerDesc)
        -> i32;
        /// Registers the output norm/projection and RoPE frequencies.
        pub fn rusty_metal_resident_set_output(
            output_norm: *const f32,
            output_w: *const u8,
            output_w_len: usize,
            output_rows: u32,
            output_dt: u32,
            inv_freq: *const f32,
            inv_freq_len: u32,
        ) -> i32;
        /// Runs one full token forward pass entirely on the GPU (one command
        /// buffer). `want_logits` == 0 stops after the layer stack, leaving the
        /// KV cache updated but producing no logits; `logits_out` may then be
        /// null.
        pub fn rusty_metal_resident_decode(
            x_embed: *const f32,
            pos: u32,
            start_t: u32,
            output_mode: i32,
            logits_out: *mut f32,
            recent: *const u32,
            recent_len: u32,
            repeat_penalty: f32,
            token_out: *mut u32,
        ) -> i32;

        /// Configures the Qwen3.5/Qwen3.8 GPU-resident hybrid decoder.
        pub fn rusty_metal_qwen_resident_configure(
            n_layers: u32,
            dim: u32,
            hidden_dim: u32,
            vocab: u32,
            storage_len: u32,
            eps: f32,
            n_heads: u32,
            n_kv_heads: u32,
            head_dim: u32,
            rotary_dim: u32,
            value_heads: u32,
            key_heads: u32,
            state_dim: u32,
            d_conv: u32,
        ) -> i32;
        /// Registers one recurrent or full-attention Qwen hybrid layer.
        pub fn rusty_metal_qwen_resident_set_layer(
            l: u32,
            desc: *const super::QwenResidentLayerDesc,
        ) -> i32;
        /// Registers Qwen's final norm, vocabulary projection, and RoPE table.
        pub fn rusty_metal_qwen_resident_set_output(
            output_norm: *const f32,
            output_w: *const u8,
            output_w_len: usize,
            output_rows: u32,
            output_dt: u32,
            inv_freq: *const f32,
            inv_freq_len: u32,
        ) -> i32;
        /// Runs one complete Qwen hybrid token in a single command buffer.
        pub fn rusty_metal_qwen_resident_decode(
            x_embed: *const f32,
            pos: u32,
            start_t: u32,
            output_mode: i32,
            logits_out: *mut f32,
            recent: *const u32,
            recent_len: u32,
            repeat_penalty: f32,
            token_out: *mut u32,
        ) -> i32;
    }
}

/// C-compatible descriptor for one transformer layer, passed to the resident
/// decoder. Field order and types must match `RustyResidentLayerDesc` in
/// `metal_backend.m`. Weight order: wq, wk, wv, wo, gate(w1), up(w3), down(w2).
/// Each `w_dt` is 0 for Q4_K, 1 for Q6_K.
#[repr(C)]
pub struct ResidentLayerDesc {
    pub w: [*const u8; 7],
    pub w_len: [usize; 7],
    pub w_rows: [u32; 7],
    pub w_dt: [u32; 7],
    pub attn_norm: *const f32,
    pub ffn_norm: *const f32,
    pub bq: *const f32,
    pub bq_len: u32,
    pub bk: *const f32,
    pub bk_len: u32,
    pub bv: *const f32,
    pub bv_len: u32,
}

/// Qwen resident-layer tag for a Gated DeltaNet recurrent block.
pub(crate) const QWEN_RESIDENT_LAYER_LINEAR: u32 = 0;
/// Qwen resident-layer tag for a gated full-attention block.
pub(crate) const QWEN_RESIDENT_LAYER_ATTENTION: u32 = 1;

/// C-compatible descriptor for one Qwen3.5/Qwen3.8 hybrid layer.
///
/// Field order and types must match `RustyQwenResidentLayerDesc` in
/// `metal_backend.m`. The eight weight slots are interpreted according to
/// `layer_type`: recurrent layers use qkv, gate, alpha, beta, mixer-output,
/// FFN-gate, FFN-up, FFN-down; attention layers use q+gate, K, V,
/// attention-output, FFN-gate, FFN-up, FFN-down, unused. An unused slot has a
/// null pointer and zero dimensions. Every `w_dt` is 0 for Q4_K or 1 for Q6_K.
#[repr(C)]
pub struct QwenResidentLayerDesc {
    pub layer_type: u32,
    pub w: [*const u8; 8],
    pub w_len: [usize; 8],
    pub w_rows: [u32; 8],
    pub w_cols: [u32; 8],
    pub w_dt: [u32; 8],
    pub attn_norm: *const f32,
    pub attn_norm_len: u32,
    pub post_norm: *const f32,
    pub post_norm_len: u32,
    pub conv_w: *const f32,
    pub conv_w_len: u32,
    pub a: *const f32,
    pub a_len: u32,
    pub dt_bias: *const f32,
    pub dt_bias_len: u32,
    pub norm: *const f32,
    pub norm_len: u32,
    pub q_norm: *const f32,
    pub q_norm_len: u32,
    pub k_norm: *const f32,
    pub k_norm_len: u32,
}

static AUTO_RESIDENT_ALLOWED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

/// Marks the process as serving multiple potentially-concurrent conversations
/// (the HTTP/HTTPS `--serve` API with its per-connection threads and
/// `SessionStore`), so auto-enabling the resident decoder is unsafe: it keeps
/// one global GPU-resident KV cache indexed only by position, so two
/// interleaved sessions would silently overwrite each other's slots even
/// though no single call ever races (the resident lock only prevents data
/// races, not cross-session slot collisions). Must be called before the
/// first generation request; an explicit `RUSTY_LLM_METAL_RESIDENT` still
/// overrides this. The `--mcp` stdio server is exempt: it processes one
/// request at a time in a single loop, so sequential reuse is safe.
pub fn disable_auto_resident_for_server() {
    AUTO_RESIDENT_ALLOWED.store(false, std::sync::atomic::Ordering::Relaxed);
}

/// Reports whether the experimental GPU-resident single-command-buffer
/// decoder is enabled. It keeps its own GPU-resident KV cache in static
/// buffers, so only one exclusive decode stream may safely reuse it at a
/// time. An explicit `RUSTY_LLM_METAL_RESIDENT=0|1` always wins; otherwise it
/// auto-enables unless the process called `disable_auto_resident_for_server`.
pub fn resident_enabled() -> bool {
    // Cache only the environment override. The server safety flag may be set
    // after startup diagnostics queried this function, so its Atomic value
    // must remain live until every automatic-policy decision.
    static EXPLICIT_RESIDENT: OnceLock<Option<bool>> = OnceLock::new();
    match *EXPLICIT_RESIDENT.get_or_init(|| env_flag("RUSTY_LLM_METAL_RESIDENT")) {
        Some(explicit) => explicit,
        None => AUTO_RESIDENT_ALLOWED.load(std::sync::atomic::Ordering::Relaxed),
    }
}

/// One transformer layer's weights, borrowed just long enough to register
/// them with the resident decoder. `w_dt` is 0 for Q4_K, 1 for Q6_K.
pub struct ResidentLayerInput<'a> {
    pub w: [&'a [u8]; 7],
    pub w_rows: [u32; 7],
    pub w_dt: [u32; 7],
    pub attn_norm: &'a [f32],
    pub ffn_norm: &'a [f32],
    pub bq: &'a [f32],
    pub bk: &'a [f32],
    pub bv: &'a [f32],
}

/// Borrowed inputs used to register one Qwen hybrid layer with the resident
/// decoder. Empty slices represent fields that do not apply to the selected
/// `layer_type` and are passed to the Objective-C backend as null pointers.
pub(crate) struct QwenResidentLayerInput<'a> {
    pub layer_type: u32,
    pub w: [&'a [u8]; 8],
    pub w_rows: [u32; 8],
    pub w_cols: [u32; 8],
    pub w_dt: [u32; 8],
    pub attn_norm: &'a [f32],
    pub post_norm: &'a [f32],
    pub conv_w: &'a [f32],
    pub a: &'a [f32],
    pub dt_bias: &'a [f32],
    pub norm: &'a [f32],
    pub q_norm: &'a [f32],
    pub k_norm: &'a [f32],
}

#[cfg(all(target_os = "macos", rusty_metal))]
#[allow(clippy::too_many_arguments)]
/// Allocates the resident decoder's per-layer GPU buffers for a new model.
pub fn resident_configure(
    n_layers: usize,
    dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    hidden_dim: usize,
    vocab: usize,
    storage_len: usize,
    eps: f32,
    prefer_half_inputs: bool,
) -> bool {
    unsafe {
        ffi::rusty_metal_resident_configure(
            n_layers as u32,
            dim as u32,
            n_heads as u32,
            n_kv_heads as u32,
            head_dim as u32,
            value_dim as u32,
            hidden_dim as u32,
            vocab as u32,
            storage_len as u32,
            eps,
            0,
            u32::from(prefer_half_inputs),
        ) != 0
    }
}

#[cfg(not(all(target_os = "macos", rusty_metal)))]
#[allow(clippy::too_many_arguments)]
pub fn resident_configure(
    _n_layers: usize,
    _dim: usize,
    _n_heads: usize,
    _n_kv_heads: usize,
    _head_dim: usize,
    _value_dim: usize,
    _hidden_dim: usize,
    _vocab: usize,
    _storage_len: usize,
    _eps: f32,
    _prefer_half_inputs: bool,
) -> bool {
    false
}

#[cfg(all(target_os = "macos", rusty_metal))]
/// Registers one transformer layer's weights with the resident decoder.
pub fn resident_set_layer(l: usize, input: &ResidentLayerInput) -> bool {
    let desc = ResidentLayerDesc {
        w: std::array::from_fn(|i| input.w[i].as_ptr()),
        w_len: std::array::from_fn(|i| input.w[i].len()),
        w_rows: input.w_rows,
        w_dt: input.w_dt,
        attn_norm: input.attn_norm.as_ptr(),
        ffn_norm: input.ffn_norm.as_ptr(),
        bq: if input.bq.is_empty() {
            std::ptr::null()
        } else {
            input.bq.as_ptr()
        },
        bq_len: input.bq.len() as u32,
        bk: if input.bk.is_empty() {
            std::ptr::null()
        } else {
            input.bk.as_ptr()
        },
        bk_len: input.bk.len() as u32,
        bv: if input.bv.is_empty() {
            std::ptr::null()
        } else {
            input.bv.as_ptr()
        },
        bv_len: input.bv.len() as u32,
    };
    unsafe { ffi::rusty_metal_resident_set_layer(l as u32, &desc) != 0 }
}

#[cfg(not(all(target_os = "macos", rusty_metal)))]
pub fn resident_set_layer(_l: usize, _input: &ResidentLayerInput) -> bool {
    false
}

#[cfg(all(target_os = "macos", rusty_metal))]
/// Registers the output norm/projection and RoPE frequency table.
pub fn resident_set_output(
    output_norm: &[f32],
    output_w: &[u8],
    output_rows: usize,
    output_dt: u32,
    inv_freq: &[f32],
) -> bool {
    unsafe {
        ffi::rusty_metal_resident_set_output(
            output_norm.as_ptr(),
            output_w.as_ptr(),
            output_w.len(),
            output_rows as u32,
            output_dt,
            inv_freq.as_ptr(),
            inv_freq.len() as u32,
        ) != 0
    }
}

#[cfg(not(all(target_os = "macos", rusty_metal)))]
pub fn resident_set_output(
    _output_norm: &[f32],
    _output_w: &[u8],
    _output_rows: usize,
    _output_dt: u32,
    _inv_freq: &[f32],
) -> bool {
    false
}

#[cfg(all(target_os = "macos", rusty_metal))]
/// Runs one full token forward pass on the GPU-resident decoder (embedding
/// already computed on the CPU side; everything else happens in one command
/// buffer). `start_t` is always 0 since resident mode requires a disabled
/// sliding window.
pub fn resident_decode_into(
    x_embed: &[f32],
    pos: usize,
    vocab: usize,
    logits: &mut Vec<f32>,
) -> bool {
    logits.resize(vocab, 0.0);
    unsafe {
        ffi::rusty_metal_resident_decode(
            x_embed.as_ptr(),
            pos as u32,
            0,
            1,
            logits.as_mut_ptr(),
            std::ptr::null(),
            0,
            1.0,
            std::ptr::null_mut(),
        ) != 0
    }
}

#[cfg(not(all(target_os = "macos", rusty_metal)))]
pub fn resident_decode_into(
    _x_embed: &[f32],
    _pos: usize,
    _vocab: usize,
    _logits: &mut Vec<f32>,
) -> bool {
    false
}

#[cfg(all(target_os = "macos", rusty_metal))]
/// Runs one resident forward pass and returns the exact greedy token after the
/// configured repetition penalty, without copying the vocabulary to the CPU.
pub fn resident_decode_greedy(
    x_embed: &[f32],
    pos: usize,
    recent: &[u32],
    repeat_penalty: f32,
) -> Option<u32> {
    if recent.len() > 64 {
        return None;
    }
    let mut token = 0u32;
    let ok = unsafe {
        ffi::rusty_metal_resident_decode(
            x_embed.as_ptr(),
            pos as u32,
            0,
            2,
            std::ptr::null_mut(),
            recent.as_ptr(),
            recent.len() as u32,
            repeat_penalty,
            &mut token,
        ) != 0
    };
    ok.then_some(token)
}

#[cfg(not(all(target_os = "macos", rusty_metal)))]
pub fn resident_decode_greedy(
    _x_embed: &[f32],
    _pos: usize,
    _recent: &[u32],
    _repeat_penalty: f32,
) -> Option<u32> {
    None
}

#[cfg(all(target_os = "macos", rusty_metal))]
/// Runs one prompt token through the GPU-resident decoder for its KV-cache
/// writes alone, skipping the vocabulary projection whose result prefill throws
/// away. Saves the largest weight read in the model plus a full-vocabulary
/// device-to-host copy on every prefilled position.
pub fn resident_prefill(x_embed: &[f32], pos: usize) -> bool {
    unsafe {
        ffi::rusty_metal_resident_decode(
            x_embed.as_ptr(),
            pos as u32,
            0,
            0,
            std::ptr::null_mut(),
            std::ptr::null(),
            0,
            1.0,
            std::ptr::null_mut(),
        ) != 0
    }
}

#[cfg(not(all(target_os = "macos", rusty_metal)))]
pub fn resident_prefill(_x_embed: &[f32], _pos: usize) -> bool {
    false
}

#[cfg(all(target_os = "macos", rusty_metal))]
#[allow(clippy::too_many_arguments)]
/// Allocates the persistent buffers and recurrent state for a Qwen3.5/Qwen3.8
/// hybrid decoder. Full-attention KV storage is allocated only for the tagged
/// attention layers registered after this call.
pub(crate) fn qwen_resident_configure(
    n_layers: usize,
    dim: usize,
    hidden_dim: usize,
    vocab: usize,
    storage_len: usize,
    eps: f32,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    value_heads: usize,
    key_heads: usize,
    state_dim: usize,
    d_conv: usize,
) -> bool {
    let (
        Ok(n_layers),
        Ok(dim),
        Ok(hidden_dim),
        Ok(vocab),
        Ok(storage_len),
        Ok(n_heads),
        Ok(n_kv_heads),
        Ok(head_dim),
        Ok(rotary_dim),
        Ok(value_heads),
        Ok(key_heads),
        Ok(state_dim),
        Ok(d_conv),
    ) = (
        u32::try_from(n_layers),
        u32::try_from(dim),
        u32::try_from(hidden_dim),
        u32::try_from(vocab),
        u32::try_from(storage_len),
        u32::try_from(n_heads),
        u32::try_from(n_kv_heads),
        u32::try_from(head_dim),
        u32::try_from(rotary_dim),
        u32::try_from(value_heads),
        u32::try_from(key_heads),
        u32::try_from(state_dim),
        u32::try_from(d_conv),
    )
    else {
        return false;
    };
    unsafe {
        ffi::rusty_metal_qwen_resident_configure(
            n_layers,
            dim,
            hidden_dim,
            vocab,
            storage_len,
            eps,
            n_heads,
            n_kv_heads,
            head_dim,
            rotary_dim,
            value_heads,
            key_heads,
            state_dim,
            d_conv,
        ) != 0
    }
}

#[cfg(not(all(target_os = "macos", rusty_metal)))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_resident_configure(
    _n_layers: usize,
    _dim: usize,
    _hidden_dim: usize,
    _vocab: usize,
    _storage_len: usize,
    _eps: f32,
    _n_heads: usize,
    _n_kv_heads: usize,
    _head_dim: usize,
    _rotary_dim: usize,
    _value_heads: usize,
    _key_heads: usize,
    _state_dim: usize,
    _d_conv: usize,
) -> bool {
    false
}

#[cfg(all(target_os = "macos", rusty_metal))]
/// Registers one tagged Qwen recurrent or full-attention layer. The backend
/// retains its own Metal buffer objects before the borrowed slices expire.
pub(crate) fn qwen_resident_set_layer(l: usize, input: &QwenResidentLayerInput<'_>) -> bool {
    if !matches!(
        input.layer_type,
        QWEN_RESIDENT_LAYER_LINEAR | QWEN_RESIDENT_LAYER_ATTENTION
    ) {
        return false;
    }
    if input
        .w
        .iter()
        .zip(input.w_dt)
        .any(|(weight, dtype)| !weight.is_empty() && dtype > 1)
    {
        return false;
    }
    let (
        Ok(l),
        Ok(attn_norm_len),
        Ok(post_norm_len),
        Ok(conv_w_len),
        Ok(a_len),
        Ok(dt_bias_len),
        Ok(norm_len),
        Ok(q_norm_len),
        Ok(k_norm_len),
    ) = (
        u32::try_from(l),
        u32::try_from(input.attn_norm.len()),
        u32::try_from(input.post_norm.len()),
        u32::try_from(input.conv_w.len()),
        u32::try_from(input.a.len()),
        u32::try_from(input.dt_bias.len()),
        u32::try_from(input.norm.len()),
        u32::try_from(input.q_norm.len()),
        u32::try_from(input.k_norm.len()),
    )
    else {
        return false;
    };
    let float_ptr = |values: &[f32]| {
        if values.is_empty() {
            std::ptr::null()
        } else {
            values.as_ptr()
        }
    };
    let desc = QwenResidentLayerDesc {
        layer_type: input.layer_type,
        w: std::array::from_fn(|i| {
            if input.w[i].is_empty() {
                std::ptr::null()
            } else {
                input.w[i].as_ptr()
            }
        }),
        w_len: std::array::from_fn(|i| input.w[i].len()),
        w_rows: input.w_rows,
        w_cols: input.w_cols,
        w_dt: input.w_dt,
        attn_norm: float_ptr(input.attn_norm),
        attn_norm_len,
        post_norm: float_ptr(input.post_norm),
        post_norm_len,
        conv_w: float_ptr(input.conv_w),
        conv_w_len,
        a: float_ptr(input.a),
        a_len,
        dt_bias: float_ptr(input.dt_bias),
        dt_bias_len,
        norm: float_ptr(input.norm),
        norm_len,
        q_norm: float_ptr(input.q_norm),
        q_norm_len,
        k_norm: float_ptr(input.k_norm),
        k_norm_len,
    };
    unsafe { ffi::rusty_metal_qwen_resident_set_layer(l, &desc) != 0 }
}

#[cfg(not(all(target_os = "macos", rusty_metal)))]
pub(crate) fn qwen_resident_set_layer(_l: usize, _input: &QwenResidentLayerInput<'_>) -> bool {
    false
}

#[cfg(all(target_os = "macos", rusty_metal))]
/// Registers Qwen's final RMSNorm, vocabulary projection, and partial-RoPE
/// frequency table with the resident decoder.
pub(crate) fn qwen_resident_set_output(
    output_norm: &[f32],
    output_w: &[u8],
    output_rows: usize,
    output_dt: u32,
    inv_freq: &[f32],
) -> bool {
    if output_norm.is_empty() || output_w.is_empty() || inv_freq.is_empty() || output_dt > 1 {
        return false;
    }
    let (Ok(output_rows), Ok(inv_freq_len)) = (
        u32::try_from(output_rows),
        u32::try_from(inv_freq.len()),
    ) else {
        return false;
    };
    unsafe {
        ffi::rusty_metal_qwen_resident_set_output(
            output_norm.as_ptr(),
            output_w.as_ptr(),
            output_w.len(),
            output_rows,
            output_dt,
            inv_freq.as_ptr(),
            inv_freq_len,
        ) != 0
    }
}

#[cfg(not(all(target_os = "macos", rusty_metal)))]
pub(crate) fn qwen_resident_set_output(
    _output_norm: &[f32],
    _output_w: &[u8],
    _output_rows: usize,
    _output_dt: u32,
    _inv_freq: &[f32],
) -> bool {
    false
}

#[cfg(all(target_os = "macos", rusty_metal))]
/// Runs one Qwen token through the resident decoder and copies vocabulary
/// logits back to the caller.
pub(crate) fn qwen_resident_decode_into(
    x_embed: &[f32],
    pos: usize,
    vocab: usize,
    logits: &mut Vec<f32>,
) -> bool {
    if x_embed.is_empty() || u32::try_from(vocab).is_err() {
        return false;
    }
    let Ok(pos) = u32::try_from(pos) else {
        return false;
    };
    logits.resize(vocab, 0.0);
    unsafe {
        ffi::rusty_metal_qwen_resident_decode(
            x_embed.as_ptr(),
            pos,
            0,
            1,
            logits.as_mut_ptr(),
            std::ptr::null(),
            0,
            1.0,
            std::ptr::null_mut(),
        ) != 0
    }
}

#[cfg(not(all(target_os = "macos", rusty_metal)))]
pub(crate) fn qwen_resident_decode_into(
    _x_embed: &[f32],
    _pos: usize,
    _vocab: usize,
    _logits: &mut Vec<f32>,
) -> bool {
    false
}

#[cfg(all(target_os = "macos", rusty_metal))]
/// Runs one resident Qwen forward pass and returns its exact greedy token
/// without copying the full vocabulary to the CPU.
pub(crate) fn qwen_resident_greedy(
    x_embed: &[f32],
    pos: usize,
    recent: &[u32],
    repeat_penalty: f32,
) -> Option<u32> {
    if x_embed.is_empty() || recent.len() > 64 {
        return None;
    }
    let (Ok(pos), Ok(recent_len)) = (u32::try_from(pos), u32::try_from(recent.len())) else {
        return None;
    };
    let recent_ptr = if recent.is_empty() {
        std::ptr::null()
    } else {
        recent.as_ptr()
    };
    let mut token = 0u32;
    let ok = unsafe {
        ffi::rusty_metal_qwen_resident_decode(
            x_embed.as_ptr(),
            pos,
            0,
            2,
            std::ptr::null_mut(),
            recent_ptr,
            recent_len,
            repeat_penalty,
            &mut token,
        ) != 0
    };
    ok.then_some(token)
}

#[cfg(not(all(target_os = "macos", rusty_metal)))]
pub(crate) fn qwen_resident_greedy(
    _x_embed: &[f32],
    _pos: usize,
    _recent: &[u32],
    _repeat_penalty: f32,
) -> Option<u32> {
    None
}

#[cfg(all(target_os = "macos", rusty_metal))]
/// Advances Qwen's resident recurrent and KV state for one prompt token while
/// skipping the final vocabulary projection.
pub(crate) fn qwen_resident_prefill(x_embed: &[f32], pos: usize) -> bool {
    if x_embed.is_empty() {
        return false;
    }
    let Ok(pos) = u32::try_from(pos) else {
        return false;
    };
    unsafe {
        ffi::rusty_metal_qwen_resident_decode(
            x_embed.as_ptr(),
            pos,
            0,
            0,
            std::ptr::null_mut(),
            std::ptr::null(),
            0,
            1.0,
            std::ptr::null_mut(),
        ) != 0
    }
}

#[cfg(not(all(target_os = "macos", rusty_metal)))]
pub(crate) fn qwen_resident_prefill(_x_embed: &[f32], _pos: usize) -> bool {
    false
}

#[cfg(all(target_os = "macos", rusty_metal))]
/// Reports whether the optional Metal backend is compiled and usable.
pub fn available() -> bool {
    unsafe { ffi::rusty_metal_available() != 0 }
}

#[cfg(not(all(target_os = "macos", rusty_metal)))]
/// Reports whether the optional Metal backend is compiled and usable.
pub fn available() -> bool {
    false
}

/// Reports whether Metal acceleration is active.
///
/// On macOS the GPU backend is enabled by default whenever it is available,
/// since it is a large decode-throughput win on unified-memory Apple Silicon.
/// Set `RUSTY_LLM_METAL=0` to force the CPU path.
pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(metal_enabled_default)
}

#[cfg(test)]
fn metal_enabled_default() -> bool {
    requested() == Some(true) && available()
}

#[cfg(not(test))]
fn metal_enabled_default() -> bool {
    requested() != Some(false) && available()
}

/// Reads the environment flag that requests Metal acceleration.
pub fn requested() -> Option<bool> {
    env_flag("RUSTY_LLM_METAL")
}

/// Reads the environment flag for experimental Q6_K Metal acceleration.
pub fn q6k_enabled() -> bool {
    dispatch_enabled()
}

/// Reports whether the Metal backend should prefer Shared/NoCopy host buffers.
pub fn nocopy_enabled() -> bool {
    static NOCOPY_ENABLED: OnceLock<bool> = OnceLock::new();
    *NOCOPY_ENABLED.get_or_init(|| env_flag("RUSTY_LLM_METAL_NOCOPY") == Some(true))
}

/// Reports whether Mistral-style fused Metal FFN blocks are enabled.
pub fn fused_ffn_enabled() -> bool {
    static FUSED_FFN_ENABLED: OnceLock<bool> = OnceLock::new();
    *FUSED_FFN_ENABLED.get_or_init(|| env_flag("RUSTY_LLM_METAL_FUSED_FFN") != Some(false))
}

/// Reports whether the experimental fused Mistral post-attention/FFN block is enabled.
///
/// Opt-in (like `RUSTY_LLM_METAL_NOCOPY`): BENCHMARK.md measured this fusion
/// as slower than the standard path on Ministral/M2 Max, so it must not be on
/// by default just because the model shape matches.
pub fn post_attention_ffn_enabled() -> bool {
    static POST_ATTENTION_FFN_ENABLED: OnceLock<bool> = OnceLock::new();
    *POST_ATTENTION_FFN_ENABLED.get_or_init(|| env_flag("RUSTY_LLM_METAL_POST_FFN") == Some(true))
}

/// Attempts a Metal attention scan across all query heads.
#[allow(clippy::too_many_arguments)]
pub fn attention_into(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    out: &mut [f32],
    heads: usize,
    kv_mul: usize,
    head_dim: usize,
    value_dim: usize,
    key_stride: usize,
    value_stride: usize,
    slot_count: usize,
    start_t: usize,
    end_t: usize,
    scale: f32,
) -> bool {
    attention_raw(
        query,
        keys,
        values,
        None,
        out,
        heads,
        kv_mul,
        head_dim,
        value_dim,
        key_stride,
        value_stride,
        slot_count,
        start_t,
        end_t,
        scale,
    )
}

/// Attempts a Metal attention scan with per-head sink scores.
#[allow(clippy::too_many_arguments)]
pub fn attention_with_sink_into(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    sinks: &[f32],
    out: &mut [f32],
    heads: usize,
    kv_mul: usize,
    head_dim: usize,
    value_dim: usize,
    key_stride: usize,
    value_stride: usize,
    slot_count: usize,
    start_t: usize,
    end_t: usize,
    scale: f32,
) -> bool {
    attention_raw(
        query,
        keys,
        values,
        Some(sinks),
        out,
        heads,
        kv_mul,
        head_dim,
        value_dim,
        key_stride,
        value_stride,
        slot_count,
        start_t,
        end_t,
        scale,
    )
}

/// Attempts a Metal Q4_K matrix-vector multiply into the output buffer.
pub fn q4k_matvec_into(
    weights: &[u8],
    x: &[f32],
    rows: usize,
    cols: usize,
    out: &mut Vec<f32>,
) -> bool {
    if !dispatch_enabled() || !q4k_single_should_use_metal(rows, cols) {
        return false;
    }
    out.resize(rows, 0.0);
    q4k_matvec_raw(weights, x, rows, cols, out)
}

/// Attempts a Metal Q6_K matrix-vector multiply into the output buffer.
pub fn q6k_matvec_into(
    weights: &[u8],
    x: &[f32],
    rows: usize,
    cols: usize,
    out: &mut Vec<f32>,
) -> bool {
    if !dispatch_enabled() || rows < q6k_min_metal_rows() {
        return false;
    }
    out.resize(rows, 0.0);
    q6k_matvec_raw(weights, x, rows, cols, out)
}

/// Attempts two fused Metal Q6_K matrix-vector projections.
pub fn q6k_matvec2_into(
    a: (&[u8], usize, usize),
    b: (&[u8], usize, usize),
    x: &[f32],
    out_a: &mut Vec<f32>,
    out_b: &mut Vec<f32>,
) -> bool {
    if !dispatch_enabled() {
        return false;
    }
    let (weights_a, rows_a, cols_a) = a;
    let (weights_b, rows_b, cols_b) = b;
    if cols_a != cols_b || cols_a != x.len() || rows_a + rows_b < q6k_min_metal_rows() {
        return false;
    }
    out_a.resize(rows_a, 0.0);
    out_b.resize(rows_b, 0.0);
    q6k_matvec2_raw(
        weights_a, rows_a, weights_b, rows_b, x, cols_a, out_a, out_b,
    )
}

/// Attempts three fused Metal Q6_K matrix-vector projections.
pub fn q6k_matvec3_into(
    a: (&[u8], usize, usize),
    b: (&[u8], usize, usize),
    c: (&[u8], usize, usize),
    x: &[f32],
    out_a: &mut Vec<f32>,
    out_b: &mut Vec<f32>,
    out_c: &mut Vec<f32>,
) -> bool {
    if !dispatch_enabled() {
        return false;
    }
    let (weights_a, rows_a, cols_a) = a;
    let (weights_b, rows_b, cols_b) = b;
    let (weights_c, rows_c, cols_c) = c;
    if cols_a != cols_b
        || cols_a != cols_c
        || cols_a != x.len()
        || rows_a + rows_b + rows_c < q6k_min_metal_rows()
    {
        return false;
    }
    out_a.resize(rows_a, 0.0);
    out_b.resize(rows_b, 0.0);
    out_c.resize(rows_c, 0.0);
    q6k_matvec3_raw(
        weights_a, rows_a, weights_b, rows_b, weights_c, rows_c, x, cols_a, out_a, out_b, out_c,
    )
}

/// Attempts two fused Metal Q4_K matrix-vector projections.
pub fn q4k_matvec2_into(
    a: (&[u8], usize, usize),
    b: (&[u8], usize, usize),
    x: &[f32],
    out_a: &mut Vec<f32>,
    out_b: &mut Vec<f32>,
) -> bool {
    if !dispatch_enabled() {
        return false;
    }
    let (weights_a, rows_a, cols_a) = a;
    let (weights_b, rows_b, cols_b) = b;
    if cols_a != cols_b
        || cols_a != x.len()
        || !fused_kquant_should_use_metal(rows_a + rows_b, cols_a)
    {
        return false;
    }
    out_a.resize(rows_a, 0.0);
    out_b.resize(rows_b, 0.0);
    q4k_matvec2_raw(
        weights_a, rows_a, weights_b, rows_b, x, cols_a, out_a, out_b,
    )
}

/// Attempts three fused Metal Q4_K matrix-vector projections.
pub fn q4k_matvec3_into(
    a: (&[u8], usize, usize),
    b: (&[u8], usize, usize),
    c: (&[u8], usize, usize),
    x: &[f32],
    out_a: &mut Vec<f32>,
    out_b: &mut Vec<f32>,
    out_c: &mut Vec<f32>,
) -> bool {
    if !dispatch_enabled() {
        return false;
    }
    let (weights_a, rows_a, cols_a) = a;
    let (weights_b, rows_b, cols_b) = b;
    let (weights_c, rows_c, cols_c) = c;
    if cols_a != cols_b
        || cols_a != cols_c
        || cols_a != x.len()
        || !fused_kquant_should_use_metal(rows_a + rows_b + rows_c, cols_a)
    {
        return false;
    }
    out_a.resize(rows_a, 0.0);
    out_b.resize(rows_b, 0.0);
    out_c.resize(rows_c, 0.0);
    q4k_matvec3_raw(
        weights_a, rows_a, weights_b, rows_b, weights_c, rows_c, x, cols_a, out_a, out_b, out_c,
    )
}

/// Attempts fused Q4_K, Q4_K, and Q6_K Metal projections.
pub fn q4k_q4k_q6k_matvec3_into(
    a: (&[u8], usize, usize),
    b: (&[u8], usize, usize),
    c: (&[u8], usize, usize),
    x: &[f32],
    out_a: &mut Vec<f32>,
    out_b: &mut Vec<f32>,
    out_c: &mut Vec<f32>,
) -> bool {
    if !dispatch_enabled() {
        return false;
    }
    let (weights_a, rows_a, cols_a) = a;
    let (weights_b, rows_b, cols_b) = b;
    let (weights_c, rows_c, cols_c) = c;
    if cols_a != cols_b
        || cols_a != cols_c
        || cols_a != x.len()
        || !fused_kquant_should_use_metal(rows_a + rows_b + rows_c, cols_a)
    {
        return false;
    }
    out_a.resize(rows_a, 0.0);
    out_b.resize(rows_b, 0.0);
    out_c.resize(rows_c, 0.0);
    q4k_q4k_q6k_matvec3_raw(
        weights_a, rows_a, weights_b, rows_b, weights_c, rows_c, x, cols_a, out_a, out_b, out_c,
    )
}

/// Attempts a fused Mistral-style Q4_K/Q4_K/Q6_K FFN block on Metal.
pub fn q4k_q4k_q6k_ffn_into(
    gate: (&[u8], usize, usize),
    up: (&[u8], usize, usize),
    down: (&[u8], usize, usize),
    x: &[f32],
    out: &mut Vec<f32>,
) -> bool {
    if !dispatch_enabled() || !fused_ffn_enabled() {
        return false;
    }
    let (gate_weights, gate_rows, gate_cols) = gate;
    let (up_weights, up_rows, up_cols) = up;
    let (down_weights, down_rows, down_cols) = down;
    if gate_cols != up_cols
        || gate_cols != x.len()
        || gate_rows != up_rows
        || gate_rows != down_cols
        || gate_rows < q6k_min_metal_rows()
        || gate_cols % 256 != 0
        || down_cols % 256 != 0
    {
        return false;
    }
    let gate_row_bytes = (gate_cols / 256) * 144;
    let down_row_bytes = (down_cols / 256) * 210;
    let Some(gate_needed) = gate_row_bytes.checked_mul(gate_rows) else {
        return false;
    };
    let Some(up_needed) = gate_row_bytes.checked_mul(up_rows) else {
        return false;
    };
    let Some(down_needed) = down_row_bytes.checked_mul(down_rows) else {
        return false;
    };
    if gate_weights.len() < gate_needed
        || up_weights.len() < up_needed
        || down_weights.len() < down_needed
    {
        return false;
    }
    out.resize(down_rows, 0.0);
    q4k_q4k_q6k_ffn_raw(
        gate_weights,
        up_weights,
        down_weights,
        x,
        gate_cols,
        gate_rows,
        down_rows,
        down_cols,
        out,
    )
}

/// Decides whether a single Q4_K projection is large enough for Metal dispatch.
fn q4k_single_should_use_metal(rows: usize, cols: usize) -> bool {
    if ultra_mode_enabled() {
        rows >= ultra_q4k_min_metal_rows() || cols >= Q4K_MIN_METAL_COLS
    } else {
        rows >= Q4K_MIN_METAL_ROWS || cols >= Q4K_MIN_METAL_COLS
    }
}

fn q6k_min_metal_rows() -> usize {
    if ultra_mode_enabled() {
        ultra_q6k_min_metal_rows()
    } else {
        Q6K_MIN_METAL_ROWS
    }
}

/// Returns the minimum fused-projection work (`rows * cols`) for Metal dispatch.
fn fused_kquant_min_metal_work() -> usize {
    *FUSED_KQUANT_MIN_METAL_WORK_RUNTIME.get_or_init(|| {
        std::env::var("RUSTY_LLM_METAL_FUSED_MIN_WORK")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(FUSED_KQUANT_MIN_METAL_WORK)
    })
}

/// Decides whether a fused K-quant projection is large enough to amortize Metal
/// dispatch. Requires both a row-count floor AND enough total work
/// (`rows * cols`) to cover the blocking command-buffer latency; a wide-cols
/// escape keeps genuinely large matvecs on the GPU. Ultra mode keeps the old
/// row-only gate for aggressive routing.
fn fused_kquant_should_use_metal(total_rows: usize, cols: usize) -> bool {
    if ultra_mode_enabled() {
        return total_rows >= ultra_q4k_min_metal_rows() || cols >= Q4K_MIN_METAL_COLS;
    }
    if cols >= Q4K_MIN_METAL_COLS {
        return true;
    }
    total_rows >= FUSED_KQUANT_MIN_METAL_ROWS
        && total_rows.saturating_mul(cols) >= fused_kquant_min_metal_work()
}

/// Decides whether a full attention scan is large enough for Metal dispatch.
#[cfg(any(all(target_os = "macos", rusty_metal), test))]
fn attention_scan_should_use_metal(start_t: usize, end_t: usize) -> bool {
    let threshold = if ultra_mode_enabled() {
        ultra_attention_min_metal_tokens()
    } else {
        attention_min_metal_tokens()
    };
    end_t
        .checked_sub(start_t)
        .map(|span| span + 1 >= threshold)
        .unwrap_or(false)
}

#[cfg(all(target_os = "macos", rusty_metal))]
/// Calls the raw Metal Q4_K projection shim or reports unsupported.
fn q4k_matvec_raw(weights: &[u8], x: &[f32], rows: usize, cols: usize, out: &mut [f32]) -> bool {
    unsafe {
        ffi::rusty_metal_q4k_matvec(
            weights.as_ptr(),
            weights.len(),
            x.as_ptr(),
            rows,
            cols,
            out.as_mut_ptr(),
        ) != 0
    }
}

#[cfg(all(target_os = "macos", rusty_metal))]
#[allow(clippy::too_many_arguments)]
/// Calls the raw Metal attention shim or reports unsupported.
fn attention_raw(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    sinks: Option<&[f32]>,
    out: &mut [f32],
    heads: usize,
    kv_mul: usize,
    head_dim: usize,
    value_dim: usize,
    key_stride: usize,
    value_stride: usize,
    slot_count: usize,
    start_t: usize,
    end_t: usize,
    scale: f32,
) -> bool {
    if !dispatch_enabled()
        || !attention_scan_should_use_metal(start_t, end_t)
        || heads == 0
        || kv_mul == 0
        || head_dim == 0
        || value_dim == 0
        || slot_count == 0
        || query.len() < heads.saturating_mul(head_dim)
        || keys.len() < slot_count.saturating_mul(key_stride)
        || values.len() < slot_count.saturating_mul(value_stride)
        || out.len() < heads.saturating_mul(value_dim)
    {
        return false;
    }
    if let Some(sinks) = sinks {
        if sinks.len() < heads {
            return false;
        }
    }
    let query_len = std::mem::size_of_val(query);
    let keys_len = std::mem::size_of_val(keys);
    let values_len = std::mem::size_of_val(values);
    let out_len = std::mem::size_of_val(out);
    let sinks_len = sinks.map(std::mem::size_of_val).unwrap_or(0);
    unsafe {
        ffi::rusty_metal_attention(
            query.as_ptr(),
            query_len,
            keys.as_ptr(),
            keys_len,
            values.as_ptr(),
            values_len,
            sinks.map(|s| s.as_ptr()).unwrap_or(std::ptr::null()),
            sinks_len,
            out.as_mut_ptr(),
            out_len,
            heads,
            kv_mul,
            head_dim,
            value_dim,
            key_stride,
            value_stride,
            slot_count,
            start_t,
            end_t,
            scale,
            sinks.is_some() as i32,
        ) != 0
    }
}

#[cfg(all(target_os = "macos", rusty_metal))]
#[allow(clippy::too_many_arguments)]
/// Calls the raw Metal fused two-projection shim or reports unsupported.
fn q4k_matvec2_raw(
    weights_a: &[u8],
    rows_a: usize,
    weights_b: &[u8],
    rows_b: usize,
    x: &[f32],
    cols: usize,
    out_a: &mut [f32],
    out_b: &mut [f32],
) -> bool {
    unsafe {
        ffi::rusty_metal_q4k_matvec2(
            weights_a.as_ptr(),
            weights_a.len(),
            rows_a,
            weights_b.as_ptr(),
            weights_b.len(),
            rows_b,
            x.as_ptr(),
            cols,
            out_a.as_mut_ptr(),
            out_b.as_mut_ptr(),
        ) != 0
    }
}

#[cfg(all(target_os = "macos", rusty_metal))]
/// Calls the raw Metal Q6_K projection shim or reports unsupported.
fn q6k_matvec_raw(weights: &[u8], x: &[f32], rows: usize, cols: usize, out: &mut [f32]) -> bool {
    unsafe {
        ffi::rusty_metal_q6k_matvec(
            weights.as_ptr(),
            weights.len(),
            x.as_ptr(),
            rows,
            cols,
            out.as_mut_ptr(),
        ) != 0
    }
}

#[cfg(all(target_os = "macos", rusty_metal))]
#[allow(clippy::too_many_arguments)]
/// Calls the raw Metal fused two-projection Q6_K shim or reports unsupported.
fn q6k_matvec2_raw(
    weights_a: &[u8],
    rows_a: usize,
    weights_b: &[u8],
    rows_b: usize,
    x: &[f32],
    cols: usize,
    out_a: &mut [f32],
    out_b: &mut [f32],
) -> bool {
    unsafe {
        ffi::rusty_metal_q6k_matvec2(
            weights_a.as_ptr(),
            weights_a.len(),
            rows_a,
            weights_b.as_ptr(),
            weights_b.len(),
            rows_b,
            x.as_ptr(),
            cols,
            out_a.as_mut_ptr(),
            out_b.as_mut_ptr(),
        ) != 0
    }
}

#[cfg(all(target_os = "macos", rusty_metal))]
#[allow(clippy::too_many_arguments)]
/// Calls the raw Metal fused three-projection Q6_K shim or reports unsupported.
fn q6k_matvec3_raw(
    weights_a: &[u8],
    rows_a: usize,
    weights_b: &[u8],
    rows_b: usize,
    weights_c: &[u8],
    rows_c: usize,
    x: &[f32],
    cols: usize,
    out_a: &mut [f32],
    out_b: &mut [f32],
    out_c: &mut [f32],
) -> bool {
    unsafe {
        ffi::rusty_metal_q6k_matvec3(
            weights_a.as_ptr(),
            weights_a.len(),
            rows_a,
            weights_b.as_ptr(),
            weights_b.len(),
            rows_b,
            weights_c.as_ptr(),
            weights_c.len(),
            rows_c,
            x.as_ptr(),
            cols,
            out_a.as_mut_ptr(),
            out_b.as_mut_ptr(),
            out_c.as_mut_ptr(),
        ) != 0
    }
}

#[cfg(all(target_os = "macos", rusty_metal))]
#[allow(clippy::too_many_arguments)]
/// Calls the raw Metal fused three-projection shim or reports unsupported.
fn q4k_matvec3_raw(
    weights_a: &[u8],
    rows_a: usize,
    weights_b: &[u8],
    rows_b: usize,
    weights_c: &[u8],
    rows_c: usize,
    x: &[f32],
    cols: usize,
    out_a: &mut [f32],
    out_b: &mut [f32],
    out_c: &mut [f32],
) -> bool {
    unsafe {
        ffi::rusty_metal_q4k_matvec3(
            weights_a.as_ptr(),
            weights_a.len(),
            rows_a,
            weights_b.as_ptr(),
            weights_b.len(),
            rows_b,
            weights_c.as_ptr(),
            weights_c.len(),
            rows_c,
            x.as_ptr(),
            cols,
            out_a.as_mut_ptr(),
            out_b.as_mut_ptr(),
            out_c.as_mut_ptr(),
        ) != 0
    }
}

#[cfg(all(target_os = "macos", rusty_metal))]
#[allow(clippy::too_many_arguments)]
/// Calls the raw mixed Q4_K/Q4_K/Q6_K Metal projection shim.
fn q4k_q4k_q6k_matvec3_raw(
    weights_a: &[u8],
    rows_a: usize,
    weights_b: &[u8],
    rows_b: usize,
    weights_c: &[u8],
    rows_c: usize,
    x: &[f32],
    cols: usize,
    out_a: &mut [f32],
    out_b: &mut [f32],
    out_c: &mut [f32],
) -> bool {
    unsafe {
        ffi::rusty_metal_q4k_q4k_q6k_matvec3(
            weights_a.as_ptr(),
            weights_a.len(),
            rows_a,
            weights_b.as_ptr(),
            weights_b.len(),
            rows_b,
            weights_c.as_ptr(),
            weights_c.len(),
            rows_c,
            x.as_ptr(),
            cols,
            out_a.as_mut_ptr(),
            out_b.as_mut_ptr(),
            out_c.as_mut_ptr(),
        ) != 0
    }
}

#[cfg(all(target_os = "macos", rusty_metal))]
#[allow(clippy::too_many_arguments)]
/// Calls the raw fused Mistral-style Q4_K/Q4_K/Q6_K FFN Metal shim.
fn q4k_q4k_q6k_ffn_raw(
    gate_weights: &[u8],
    up_weights: &[u8],
    down_weights: &[u8],
    x: &[f32],
    input_cols: usize,
    hidden_rows: usize,
    down_rows: usize,
    down_cols: usize,
    out: &mut [f32],
) -> bool {
    unsafe {
        ffi::rusty_metal_q4k_q4k_q6k_ffn(
            gate_weights.as_ptr(),
            gate_weights.len(),
            up_weights.as_ptr(),
            up_weights.len(),
            down_weights.as_ptr(),
            down_weights.len(),
            x.as_ptr(),
            input_cols,
            hidden_rows,
            down_rows,
            down_cols,
            out.as_mut_ptr(),
        ) != 0
    }
}

#[cfg(all(target_os = "macos", rusty_metal))]
#[allow(clippy::too_many_arguments)]
/// Calls the raw fused Mistral post-attention/FFN Metal shim.
fn mistral_post_attention_ffn_raw(
    wo_weights: &[u8],
    gate_weights: &[u8],
    up_weights: &[u8],
    down_weights: &[u8],
    x: &mut [f32],
    attn_out: &[f32],
    ffn_norm: &[f32],
    rms_eps: f32,
    dim: usize,
    attn_cols: usize,
    hidden_rows: usize,
    down_rows: usize,
    down_cols: usize,
) -> bool {
    unsafe {
        ffi::rusty_metal_mistral_post_attention_ffn(
            wo_weights.as_ptr(),
            wo_weights.len(),
            gate_weights.as_ptr(),
            gate_weights.len(),
            up_weights.as_ptr(),
            up_weights.len(),
            down_weights.as_ptr(),
            down_weights.len(),
            x.as_mut_ptr(),
            dim,
            attn_out.as_ptr(),
            attn_cols,
            ffn_norm.as_ptr(),
            rms_eps,
            hidden_rows,
            down_rows,
            down_cols,
        ) != 0
    }
}

#[cfg(not(all(target_os = "macos", rusty_metal)))]
/// Calls the raw Metal Q4_K projection shim or reports unsupported.
fn q4k_matvec_raw(
    _weights: &[u8],
    _x: &[f32],
    _rows: usize,
    _cols: usize,
    _out: &mut [f32],
) -> bool {
    false
}

#[cfg(not(all(target_os = "macos", rusty_metal)))]
#[allow(clippy::too_many_arguments)]
/// Calls the raw Metal attention shim or reports unsupported.
fn attention_raw(
    _query: &[f32],
    _keys: &[f32],
    _values: &[f32],
    _sinks: Option<&[f32]>,
    _out: &mut [f32],
    _heads: usize,
    _kv_mul: usize,
    _head_dim: usize,
    _value_dim: usize,
    _key_stride: usize,
    _value_stride: usize,
    _slot_count: usize,
    _start_t: usize,
    _end_t: usize,
    _scale: f32,
) -> bool {
    false
}

/// Reads an optional boolean-like environment variable.
fn env_flag(name: &str) -> Option<bool> {
    let value = std::env::var(name).ok()?;
    Some(parse_env_flag(&value))
}

/// Parses common truthy and falsey environment flag values.
fn parse_env_flag(value: &str) -> bool {
    match value.trim().to_ascii_lowercase().as_str() {
        "" | "1" | "true" | "yes" | "on" => true,
        "0" | "false" | "no" | "off" => false,
        _ => false,
    }
}

#[cfg(not(all(target_os = "macos", rusty_metal)))]
#[allow(clippy::too_many_arguments)]
/// Calls the raw Metal fused two-projection shim or reports unsupported.
fn q4k_matvec2_raw(
    _weights_a: &[u8],
    _rows_a: usize,
    _weights_b: &[u8],
    _rows_b: usize,
    _x: &[f32],
    _cols: usize,
    _out_a: &mut [f32],
    _out_b: &mut [f32],
) -> bool {
    false
}

#[cfg(not(all(target_os = "macos", rusty_metal)))]
/// Calls the raw Metal Q6_K projection shim or reports unsupported.
fn q6k_matvec_raw(
    _weights: &[u8],
    _x: &[f32],
    _rows: usize,
    _cols: usize,
    _out: &mut [f32],
) -> bool {
    false
}

#[cfg(not(all(target_os = "macos", rusty_metal)))]
#[allow(clippy::too_many_arguments)]
/// Calls the raw Metal fused two-projection Q6_K shim or reports unsupported.
fn q6k_matvec2_raw(
    _weights_a: &[u8],
    _rows_a: usize,
    _weights_b: &[u8],
    _rows_b: usize,
    _x: &[f32],
    _cols: usize,
    _out_a: &mut [f32],
    _out_b: &mut [f32],
) -> bool {
    false
}

#[cfg(not(all(target_os = "macos", rusty_metal)))]
#[allow(clippy::too_many_arguments)]
/// Calls the raw Metal fused three-projection Q6_K shim or reports unsupported.
fn q6k_matvec3_raw(
    _weights_a: &[u8],
    _rows_a: usize,
    _weights_b: &[u8],
    _rows_b: usize,
    _weights_c: &[u8],
    _rows_c: usize,
    _x: &[f32],
    _cols: usize,
    _out_a: &mut [f32],
    _out_b: &mut [f32],
    _out_c: &mut [f32],
) -> bool {
    false
}

#[cfg(not(all(target_os = "macos", rusty_metal)))]
#[allow(clippy::too_many_arguments)]
/// Calls the raw Metal fused three-projection shim or reports unsupported.
fn q4k_matvec3_raw(
    _weights_a: &[u8],
    _rows_a: usize,
    _weights_b: &[u8],
    _rows_b: usize,
    _weights_c: &[u8],
    _rows_c: usize,
    _x: &[f32],
    _cols: usize,
    _out_a: &mut [f32],
    _out_b: &mut [f32],
    _out_c: &mut [f32],
) -> bool {
    false
}

#[cfg(not(all(target_os = "macos", rusty_metal)))]
#[allow(clippy::too_many_arguments)]
/// Calls the raw mixed Q4_K/Q4_K/Q6_K Metal projection shim or reports unsupported.
fn q4k_q4k_q6k_matvec3_raw(
    _weights_a: &[u8],
    _rows_a: usize,
    _weights_b: &[u8],
    _rows_b: usize,
    _weights_c: &[u8],
    _rows_c: usize,
    _x: &[f32],
    _cols: usize,
    _out_a: &mut [f32],
    _out_b: &mut [f32],
    _out_c: &mut [f32],
) -> bool {
    false
}

#[cfg(not(all(target_os = "macos", rusty_metal)))]
#[allow(clippy::too_many_arguments)]
/// Calls the raw fused Mistral-style Q4_K/Q4_K/Q6_K FFN Metal shim.
fn q4k_q4k_q6k_ffn_raw(
    _gate_weights: &[u8],
    _up_weights: &[u8],
    _down_weights: &[u8],
    _x: &[f32],
    _input_cols: usize,
    _hidden_rows: usize,
    _down_rows: usize,
    _down_cols: usize,
    _out: &mut [f32],
) -> bool {
    false
}

#[cfg(not(all(target_os = "macos", rusty_metal)))]
#[allow(clippy::too_many_arguments)]
/// Calls the raw fused Mistral post-attention/FFN Metal shim.
fn mistral_post_attention_ffn_raw(
    _wo_weights: &[u8],
    _gate_weights: &[u8],
    _up_weights: &[u8],
    _down_weights: &[u8],
    _x: &mut [f32],
    _attn_out: &[f32],
    _ffn_norm: &[f32],
    _rms_eps: f32,
    _dim: usize,
    _attn_cols: usize,
    _hidden_rows: usize,
    _down_rows: usize,
    _down_cols: usize,
) -> bool {
    false
}

/// Attempts a fused Mistral post-attention + FFN Metal block.
#[allow(clippy::too_many_arguments)]
pub fn mistral_post_attention_ffn_into(
    wo: (&[u8], usize, usize),
    gate: (&[u8], usize, usize),
    up: (&[u8], usize, usize),
    down: (&[u8], usize, usize),
    x: &mut [f32],
    attn_out: &[f32],
    ffn_norm: &[f32],
    rms_eps: f32,
) -> bool {
    if !dispatch_enabled() || !post_attention_ffn_enabled() {
        return false;
    }
    let (wo_weights, wo_rows, wo_cols) = wo;
    let (gate_weights, gate_rows, gate_cols) = gate;
    let (up_weights, up_rows, up_cols) = up;
    let (down_weights, down_rows, down_cols) = down;
    if wo_rows == 0
        || wo_cols == 0
        || gate_rows == 0
        || down_rows == 0
        || wo_rows != x.len()
        || wo_cols != attn_out.len()
        || gate_cols != x.len()
        || up_cols != x.len()
        || gate_rows != up_rows
        || gate_rows != down_cols
        || down_rows != x.len()
        || ffn_norm.len() != x.len()
    {
        return false;
    }
    let dim = x.len();
    mistral_post_attention_ffn_raw(
        wo_weights,
        gate_weights,
        up_weights,
        down_weights,
        x,
        attn_out,
        ffn_norm,
        rms_eps,
        dim,
        wo_cols,
        gate_rows,
        down_rows,
        down_cols,
    )
}

pub const Q4_0_MIN_METAL_ROWS: usize = 512;
pub const Q8_0_MIN_METAL_ROWS: usize = 512;

/// Attempts a Metal Q4_0 matrix-vector multiply into the output buffer.
pub fn q4_0_matvec_into(
    weights: &[u8],
    x: &[f32],
    rows: usize,
    cols: usize,
    out: &mut Vec<f32>,
) -> bool {
    if !dispatch_enabled() || rows < Q4_0_MIN_METAL_ROWS || (cols % 32) != 0 {
        return false;
    }
    out.resize(rows, 0.0);
    q4_0_matvec_raw(weights, x, rows, cols, out)
}

#[inline]
fn q4_0_matrix_bytes(rows: usize, cols: usize) -> Option<usize> {
    if cols == 0 || (cols % 32) != 0 {
        return None;
    }
    rows.checked_mul(cols / 32)?.checked_mul(18)
}

/// Attempts two Q4_0 projections in one Metal command buffer.
pub fn q4_0_matvec2_into(
    a: (&[u8], usize, usize),
    b: (&[u8], usize, usize),
    x: &[f32],
    out_a: &mut Vec<f32>,
    out_b: &mut Vec<f32>,
) -> bool {
    let (a_weights, a_rows, a_cols) = a;
    let (b_weights, b_rows, b_cols) = b;
    if !dispatch_enabled()
        || a_rows < Q4_0_MIN_METAL_ROWS
        || b_rows < Q4_0_MIN_METAL_ROWS
        || a_cols != b_cols
        || a_cols != x.len()
        || q4_0_matrix_bytes(a_rows, a_cols).is_none_or(|len| a_weights.len() < len)
        || q4_0_matrix_bytes(b_rows, b_cols).is_none_or(|len| b_weights.len() < len)
    {
        return false;
    }
    out_a.resize(a_rows, 0.0);
    out_b.resize(b_rows, 0.0);
    q4_0_matvec2_raw(a_weights, a_rows, b_weights, b_rows, x, a_cols, out_a, out_b)
}

/// Attempts three Q4_0 projections in one Metal command buffer.
pub fn q4_0_matvec3_into(
    a: (&[u8], usize, usize),
    b: (&[u8], usize, usize),
    c: (&[u8], usize, usize),
    x: &[f32],
    out_a: &mut Vec<f32>,
    out_b: &mut Vec<f32>,
    out_c: &mut Vec<f32>,
) -> bool {
    let (a_weights, a_rows, a_cols) = a;
    let (b_weights, b_rows, b_cols) = b;
    let (c_weights, c_rows, c_cols) = c;
    if !dispatch_enabled()
        || a_rows < Q4_0_MIN_METAL_ROWS
        || b_rows < Q4_0_MIN_METAL_ROWS
        || c_rows < Q4_0_MIN_METAL_ROWS
        || a_cols != b_cols
        || a_cols != c_cols
        || a_cols != x.len()
        || q4_0_matrix_bytes(a_rows, a_cols).is_none_or(|len| a_weights.len() < len)
        || q4_0_matrix_bytes(b_rows, b_cols).is_none_or(|len| b_weights.len() < len)
        || q4_0_matrix_bytes(c_rows, c_cols).is_none_or(|len| c_weights.len() < len)
    {
        return false;
    }
    out_a.resize(a_rows, 0.0);
    out_b.resize(b_rows, 0.0);
    out_c.resize(c_rows, 0.0);
    q4_0_matvec3_raw(
        a_weights, a_rows, b_weights, b_rows, c_weights, c_rows, x, a_cols, out_a, out_b,
        out_c,
    )
}

/// Attempts a Q4_0 GELU feed-forward block without CPU-visible intermediates.
pub fn q4_0_gelu_ffn_into(
    gate: (&[u8], usize, usize),
    up: (&[u8], usize, usize),
    down: (&[u8], usize, usize),
    x: &[f32],
    out: &mut Vec<f32>,
) -> bool {
    let (gate_weights, gate_rows, gate_cols) = gate;
    let (up_weights, up_rows, up_cols) = up;
    let (down_weights, down_rows, down_cols) = down;
    if !dispatch_enabled()
        || gate_rows < Q4_0_MIN_METAL_ROWS
        || down_rows < Q4_0_MIN_METAL_ROWS
        || gate_cols != x.len()
        || up_cols != gate_cols
        || up_rows != gate_rows
        || down_cols != gate_rows
        || q4_0_matrix_bytes(gate_rows, gate_cols).is_none_or(|len| gate_weights.len() < len)
        || q4_0_matrix_bytes(up_rows, up_cols).is_none_or(|len| up_weights.len() < len)
        || q4_0_matrix_bytes(down_rows, down_cols).is_none_or(|len| down_weights.len() < len)
    {
        return false;
    }
    out.resize(down_rows, 0.0);
    q4_0_gelu_ffn_raw(
        gate_weights,
        up_weights,
        down_weights,
        x,
        gate_cols,
        gate_rows,
        down_rows,
        down_cols,
        out,
    )
}

/// Attempts a Metal Q8_0 matrix-vector multiply into the output buffer.
pub fn q8_0_matvec_into(
    weights: &[u8],
    x: &[f32],
    rows: usize,
    cols: usize,
    out: &mut Vec<f32>,
) -> bool {
    if !dispatch_enabled() || rows < Q8_0_MIN_METAL_ROWS || (cols % 32) != 0 {
        return false;
    }
    out.resize(rows, 0.0);
    q8_0_matvec_raw(weights, x, rows, cols, out)
}

#[cfg(all(target_os = "macos", rusty_metal))]
/// Calls the raw Metal Q4_0 projection shim.
fn q4_0_matvec_raw(weights: &[u8], x: &[f32], rows: usize, cols: usize, out: &mut [f32]) -> bool {
    unsafe {
        ffi::rusty_metal_q4_0_matvec(
            weights.as_ptr(),
            weights.len(),
            x.as_ptr(),
            rows,
            cols,
            out.as_mut_ptr(),
        ) != 0
    }
}

#[cfg(all(target_os = "macos", rusty_metal))]
#[allow(clippy::too_many_arguments)]
fn q4_0_matvec2_raw(
    a: &[u8],
    a_rows: usize,
    b: &[u8],
    b_rows: usize,
    x: &[f32],
    cols: usize,
    out_a: &mut [f32],
    out_b: &mut [f32],
) -> bool {
    unsafe {
        ffi::rusty_metal_q4_0_matvec2(
            a.as_ptr(), a.len(), a_rows, b.as_ptr(), b.len(), b_rows, x.as_ptr(), cols,
            out_a.as_mut_ptr(), out_b.as_mut_ptr(),
        ) != 0
    }
}

#[cfg(all(target_os = "macos", rusty_metal))]
#[allow(clippy::too_many_arguments)]
fn q4_0_matvec3_raw(
    a: &[u8],
    a_rows: usize,
    b: &[u8],
    b_rows: usize,
    c: &[u8],
    c_rows: usize,
    x: &[f32],
    cols: usize,
    out_a: &mut [f32],
    out_b: &mut [f32],
    out_c: &mut [f32],
) -> bool {
    unsafe {
        ffi::rusty_metal_q4_0_matvec3(
            a.as_ptr(), a.len(), a_rows, b.as_ptr(), b.len(), b_rows, c.as_ptr(), c.len(),
            c_rows, x.as_ptr(), cols, out_a.as_mut_ptr(), out_b.as_mut_ptr(), out_c.as_mut_ptr(),
        ) != 0
    }
}

#[cfg(all(target_os = "macos", rusty_metal))]
#[allow(clippy::too_many_arguments)]
fn q4_0_gelu_ffn_raw(
    gate: &[u8],
    up: &[u8],
    down: &[u8],
    x: &[f32],
    input_cols: usize,
    hidden_rows: usize,
    down_rows: usize,
    down_cols: usize,
    out: &mut [f32],
) -> bool {
    unsafe {
        ffi::rusty_metal_q4_0_gelu_ffn(
            gate.as_ptr(), gate.len(), up.as_ptr(), up.len(), down.as_ptr(), down.len(), x.as_ptr(),
            input_cols, hidden_rows, down_rows, down_cols, out.as_mut_ptr(),
        ) != 0
    }
}

#[cfg(all(target_os = "macos", rusty_metal))]
/// Calls the raw Metal Q8_0 projection shim.
fn q8_0_matvec_raw(weights: &[u8], x: &[f32], rows: usize, cols: usize, out: &mut [f32]) -> bool {
    unsafe {
        ffi::rusty_metal_q8_0_matvec(
            weights.as_ptr(),
            weights.len(),
            x.as_ptr(),
            rows,
            cols,
            out.as_mut_ptr(),
        ) != 0
    }
}

#[cfg(not(all(target_os = "macos", rusty_metal)))]
fn q4_0_matvec_raw(
    _weights: &[u8],
    _x: &[f32],
    _rows: usize,
    _cols: usize,
    _out: &mut [f32],
) -> bool {
    false
}

#[cfg(not(all(target_os = "macos", rusty_metal)))]
#[allow(clippy::too_many_arguments)]
fn q4_0_matvec2_raw(
    _a: &[u8],
    _a_rows: usize,
    _b: &[u8],
    _b_rows: usize,
    _x: &[f32],
    _cols: usize,
    _out_a: &mut [f32],
    _out_b: &mut [f32],
) -> bool {
    false
}

#[cfg(not(all(target_os = "macos", rusty_metal)))]
#[allow(clippy::too_many_arguments)]
fn q4_0_matvec3_raw(
    _a: &[u8],
    _a_rows: usize,
    _b: &[u8],
    _b_rows: usize,
    _c: &[u8],
    _c_rows: usize,
    _x: &[f32],
    _cols: usize,
    _out_a: &mut [f32],
    _out_b: &mut [f32],
    _out_c: &mut [f32],
) -> bool {
    false
}

#[cfg(not(all(target_os = "macos", rusty_metal)))]
#[allow(clippy::too_many_arguments)]
fn q4_0_gelu_ffn_raw(
    _gate: &[u8],
    _up: &[u8],
    _down: &[u8],
    _x: &[f32],
    _input_cols: usize,
    _hidden_rows: usize,
    _down_rows: usize,
    _down_cols: usize,
    _out: &mut [f32],
) -> bool {
    false
}

#[cfg(not(all(target_os = "macos", rusty_metal)))]
fn q8_0_matvec_raw(
    _weights: &[u8],
    _x: &[f32],
    _rows: usize,
    _cols: usize,
    _out: &mut [f32],
) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::parse_env_flag;

    #[cfg(all(target_os = "macos", rusty_metal))]
    #[test]
    /// Checks the lane-striped Metal Q4_K implementation against the native
    /// CPU implementation, including incomplete output and block-group tails.
    fn q4k_metal_matches_cpu_for_odd_shapes() {
        if !super::available() {
            return;
        }

        let rows = 5;
        let cols = 3 * 256;
        let blocks_per_row = cols / 256;
        let mut weights = vec![0u8; rows * blocks_per_row * 144];
        for block_index in 0..rows * blocks_per_row {
            let block = &mut weights[block_index * 144..(block_index + 1) * 144];
            // IEEE-754 binary16: d=0.5 and dmin=0.125.
            block[0..2].copy_from_slice(&0x3800u16.to_le_bytes());
            block[2..4].copy_from_slice(&0x3000u16.to_le_bytes());
            for (i, byte) in block[4..16].iter_mut().enumerate() {
                *byte = ((block_index * 29 + i * 17 + 11) & 0xff) as u8;
            }
            for (i, byte) in block[16..].iter_mut().enumerate() {
                *byte = ((block_index * 37 + i * 13 + 7) & 0xff) as u8;
            }
        }
        let x = (0..cols)
            .map(|i| ((i * 19 % 257) as f32 - 128.0) / 128.0)
            .collect::<Vec<_>>();
        let scale_min = |j: usize, scales: &[u8]| -> (u8, u8) {
            if j < 4 {
                (scales[j] & 63, scales[j + 4] & 63)
            } else {
                (
                    (scales[j + 4] & 15) | ((scales[j - 4] >> 6) << 4),
                    (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
                )
            }
        };
        let mut expected = vec![0.0f32; rows];
        for (row, expected_row) in expected.iter_mut().enumerate() {
            let row_base = row * blocks_per_row * 144;
            for block_index in 0..blocks_per_row {
                let block_start = row_base + block_index * 144;
                let block = &weights[block_start..block_start + 144];
                let d = crate::simd::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                let dmin = crate::simd::f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
                let scales = &block[4..16];
                let quants = &block[16..144];
                for chunk in 0..4 {
                    let (scale_lo, min_lo) = scale_min(chunk * 2, scales);
                    let (scale_hi, min_hi) = scale_min(chunk * 2 + 1, scales);
                    let x_lo = block_index * 256 + chunk * 64;
                    let q = &quants[chunk * 32..chunk * 32 + 32];
                    for i in 0..32 {
                        *expected_row += (d * f32::from(scale_lo) * f32::from(q[i] & 15)
                            - dmin * f32::from(min_lo))
                            * x[x_lo + i];
                        *expected_row += (d * f32::from(scale_hi) * f32::from(q[i] >> 4)
                            - dmin * f32::from(min_hi))
                            * x[x_lo + 32 + i];
                    }
                }
            }
        }
        let mut actual = vec![0.0f32; rows];
        assert!(super::q4k_matvec_raw(&weights, &x, rows, cols, &mut actual));

        for (row, (&cpu, &metal)) in expected.iter().zip(&actual).enumerate() {
            let tolerance = 5.0e-5 * cpu.abs().max(1.0);
            assert!(
                (cpu - metal).abs() <= tolerance,
                "row {row}: CPU={cpu}, Metal={metal}, tolerance={tolerance}"
            );
        }

        let rows_b = 3;
        let mut paired_a = vec![0.0f32; rows];
        let mut paired_b = vec![0.0f32; rows_b];
        assert!(super::q4k_matvec2_raw(
            &weights,
            rows,
            &weights[..rows_b * blocks_per_row * 144],
            rows_b,
            &x,
            cols,
            &mut paired_a,
            &mut paired_b,
        ));
        for (row, (&cpu, &metal)) in expected.iter().zip(&paired_a).enumerate() {
            let tolerance = 5.0e-5 * cpu.abs().max(1.0);
            assert!(
                (cpu - metal).abs() <= tolerance,
                "paired A row {row}: CPU={cpu}, Metal={metal}, tolerance={tolerance}"
            );
        }
        for (row, (&cpu, &metal)) in expected[..rows_b].iter().zip(&paired_b).enumerate() {
            let tolerance = 5.0e-5 * cpu.abs().max(1.0);
            assert!(
                (cpu - metal).abs() <= tolerance,
                "paired B row {row}: CPU={cpu}, Metal={metal}, tolerance={tolerance}"
            );
        }
    }

    #[cfg(all(target_os = "macos", rusty_metal))]
    #[test]
    fn q4_0_single_and_grouped_metal_paths_match_scalar() {
        if !super::available() {
            return;
        }
        let rows = 513;
        let cols = 3 * 32;
        let blocks_per_row = cols / 32;
        let mut weights = vec![0u8; rows * blocks_per_row * 18];
        for block_index in 0..rows * blocks_per_row {
            let block = &mut weights[block_index * 18..(block_index + 1) * 18];
            block[..2].copy_from_slice(&0x3400u16.to_le_bytes());
            for (i, value) in block[2..].iter_mut().enumerate() {
                *value = ((block_index * 29 + i * 17 + 3) & 0xff) as u8;
            }
        }
        let x = (0..cols)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 31.0)
            .collect::<Vec<_>>();
        let mut expected = vec![0.0f32; rows];
        for (row, value) in expected.iter_mut().enumerate() {
            for block_index in 0..blocks_per_row {
                let start = (row * blocks_per_row + block_index) * 18;
                let block = &weights[start..start + 18];
                let d = crate::simd::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                for i in 0..16 {
                    let packed = block[2 + i];
                    *value += d * f32::from((packed & 15) as i8 - 8) * x[block_index * 32 + i];
                    *value += d * f32::from((packed >> 4) as i8 - 8)
                        * x[block_index * 32 + 16 + i];
                }
            }
        }

        let mut single = vec![0.0; rows];
        assert!(super::q4_0_matvec_raw(
            &weights,
            &x,
            rows,
            cols,
            &mut single
        ));
        let mut pair_a = vec![0.0; rows];
        let mut pair_b = vec![0.0; rows];
        assert!(super::q4_0_matvec2_raw(
            &weights,
            rows,
            &weights,
            rows,
            &x,
            cols,
            &mut pair_a,
            &mut pair_b,
        ));
        let mut triple_a = vec![0.0; rows];
        let mut triple_b = vec![0.0; rows];
        let mut triple_c = vec![0.0; rows];
        assert!(super::q4_0_matvec3_raw(
            &weights,
            rows,
            &weights,
            rows,
            &weights,
            rows,
            &x,
            cols,
            &mut triple_a,
            &mut triple_b,
            &mut triple_c,
        ));

        for (row, expected) in expected.into_iter().enumerate() {
            let tolerance = 8.0e-5 * expected.abs().max(1.0);
            for (name, actual) in [
                ("single", single[row]),
                ("pair-a", pair_a[row]),
                ("pair-b", pair_b[row]),
                ("triple-a", triple_a[row]),
                ("triple-b", triple_b[row]),
                ("triple-c", triple_c[row]),
            ] {
                assert!(
                    (expected - actual).abs() <= tolerance,
                    "{name} row {row}: scalar={expected}, Metal={actual}, tolerance={tolerance}"
                );
            }
        }
    }

    #[cfg(all(target_os = "macos", rusty_metal))]
    #[test]
    fn q4_0_gelu_ffn_metal_path_matches_scalar() {
        if !super::available() {
            return;
        }
        fn make_weights(rows: usize, cols: usize, salt: usize) -> Vec<u8> {
            let blocks = cols / 32;
            let mut weights = vec![0u8; rows * blocks * 18];
            for block_index in 0..rows * blocks {
                let block = &mut weights[block_index * 18..(block_index + 1) * 18];
                block[..2].copy_from_slice(&0x2c00u16.to_le_bytes());
                for (i, value) in block[2..].iter_mut().enumerate() {
                    *value = ((block_index * 11 + i * 23 + salt) & 0xff) as u8;
                }
            }
            weights
        }
        fn scalar_matvec(weights: &[u8], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
            let blocks = cols / 32;
            let mut out = vec![0.0; rows];
            for (row, value) in out.iter_mut().enumerate() {
                for block_index in 0..blocks {
                    let start = (row * blocks + block_index) * 18;
                    let block = &weights[start..start + 18];
                    let d = crate::simd::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                    for i in 0..16 {
                        let packed = block[2 + i];
                        *value += d * f32::from((packed & 15) as i8 - 8)
                            * x[block_index * 32 + i];
                        *value += d * f32::from((packed >> 4) as i8 - 8)
                            * x[block_index * 32 + 16 + i];
                    }
                }
            }
            out
        }

        let input_cols = 32;
        let hidden_rows = 512;
        let down_rows = 512;
        let gate = make_weights(hidden_rows, input_cols, 3);
        let up = make_weights(hidden_rows, input_cols, 17);
        let down = make_weights(down_rows, hidden_rows, 41);
        let x = (0..input_cols)
            .map(|i| ((i * 19 % 43) as f32 - 21.0) / 17.0)
            .collect::<Vec<_>>();
        let gate_values = scalar_matvec(&gate, &x, hidden_rows, input_cols);
        let up_values = scalar_matvec(&up, &x, hidden_rows, input_cols);
        let hidden = gate_values
            .iter()
            .zip(&up_values)
            .map(|(&g, &u)| crate::model::gelu(g) * u)
            .collect::<Vec<_>>();
        let expected = scalar_matvec(&down, &hidden, down_rows, hidden_rows);
        let mut actual = vec![0.0; down_rows];
        assert!(super::q4_0_gelu_ffn_raw(
            &gate,
            &up,
            &down,
            &x,
            input_cols,
            hidden_rows,
            down_rows,
            hidden_rows,
            &mut actual,
        ));
        for (row, (&expected, &actual)) in expected.iter().zip(&actual).enumerate() {
            let tolerance = 2.0e-3 * expected.abs().max(1.0);
            assert!(
                (expected - actual).abs() <= tolerance,
                "row {row}: scalar={expected}, Metal={actual}, tolerance={tolerance}"
            );
        }
    }

    #[cfg(all(target_os = "macos", rusty_metal))]
    #[test]
    /// Checks the lane-striped Metal Q6_K implementation against direct scalar
    /// dequantization, including incomplete output and block-group tails.
    fn q6k_metal_matches_scalar_for_odd_shapes() {
        if !super::available() {
            return;
        }

        let rows = 5;
        let cols = 3 * 256;
        let blocks_per_row = cols / 256;
        let mut weights = vec![0u8; rows * blocks_per_row * 210];
        for block_index in 0..rows * blocks_per_row {
            let block = &mut weights[block_index * 210..(block_index + 1) * 210];
            for (i, byte) in block[..192].iter_mut().enumerate() {
                *byte = ((block_index * 41 + i * 23 + 5) & 0xff) as u8;
            }
            for (i, byte) in block[192..208].iter_mut().enumerate() {
                *byte = ((block_index * 11 + i * 7 + 109) & 0xff) as u8;
            }
            block[208..210].copy_from_slice(&0x3400u16.to_le_bytes()); // d=0.25
        }
        let x = (0..cols)
            .map(|i| ((i * 31 % 263) as f32 - 131.0) / 131.0)
            .collect::<Vec<_>>();
        let mut expected = vec![0.0f32; rows];
        for (row, expected_row) in expected.iter_mut().enumerate() {
            let row_base = row * blocks_per_row * 210;
            for block_index in 0..blocks_per_row {
                let block_start = row_base + block_index * 210;
                let block = &weights[block_start..block_start + 210];
                let ql = &block[..128];
                let qh = &block[128..192];
                let scales = &block[192..208];
                let d = crate::simd::f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
                for step in 0..2 {
                    for l in 0..32 {
                        let low = ql[step * 64 + l];
                        let high = ql[step * 64 + 32 + l];
                        let high_bits = qh[step * 32 + l];
                        let scale_group = step * 8 + usize::from(l >= 16);
                        let signed_scale = |offset: usize| -> f32 {
                            f32::from(scales[scale_group + offset] as i8)
                        };
                        let x_base = block_index * 256 + step * 128 + l;
                        *expected_row += d
                            * signed_scale(0)
                            * (((low & 15) | ((high_bits & 3) << 4)) as f32 - 32.0)
                            * x[x_base];
                        *expected_row += d
                            * signed_scale(2)
                            * (((high & 15) | (((high_bits >> 2) & 3) << 4)) as f32 - 32.0)
                            * x[x_base + 32];
                        *expected_row += d
                            * signed_scale(4)
                            * (((low >> 4) | (((high_bits >> 4) & 3) << 4)) as f32 - 32.0)
                            * x[x_base + 64];
                        *expected_row += d
                            * signed_scale(6)
                            * (((high >> 4) | (((high_bits >> 6) & 3) << 4)) as f32 - 32.0)
                            * x[x_base + 96];
                    }
                }
            }
        }
        let mut actual = vec![0.0f32; rows];
        assert!(super::q6k_matvec_raw(&weights, &x, rows, cols, &mut actual));

        for (row, (&scalar, &metal)) in expected.iter().zip(&actual).enumerate() {
            let tolerance = 5.0e-5 * scalar.abs().max(1.0);
            assert!(
                (scalar - metal).abs() <= tolerance,
                "row {row}: scalar={scalar}, Metal={metal}, tolerance={tolerance}"
            );
        }
    }

    #[cfg(all(target_os = "macos", rusty_metal))]
    #[test]
    fn greedy_argmax_matches_repeat_penalty_and_tie_breaking() {
        if !super::available() {
            return;
        }
        let mut logits = (0..777)
            .map(|index| -20.0 + (index % 17) as f32 * 0.01)
            .collect::<Vec<_>>();
        logits[42] = 10.0;
        logits[300] = 9.0;
        logits[301] = 9.0;
        logits[500] = f32::NAN;
        let recent = [42u32, 42];
        let mut selected = u32::MAX;
        let ok = unsafe {
            super::ffi::rusty_metal_test_greedy_argmax(
                logits.as_ptr(),
                logits.len() as u32,
                recent.as_ptr(),
                recent.len() as u32,
                2.0,
                &mut selected,
            ) != 0
        };
        assert!(ok);
        assert_eq!(selected, 300);
    }

    #[cfg(all(target_os = "macos", rusty_metal))]
    #[test]
    /// Checks the four-way temporal softmax merge against both the serial Metal
    /// kernel and a direct CPU softmax for GQA with non-trivial cache strides.
    fn parallel_resident_attention_matches_serial_and_cpu() {
        if !super::available() {
            return;
        }

        let heads = 4usize;
        let kv_mul = 2usize;
        let head_dim = 64usize;
        let value_dim = 48usize;
        let kv_heads = heads / kv_mul;
        let key_stride = kv_heads * head_dim;
        let value_stride = kv_heads * value_dim;
        let slot_count = 9usize;
        let start_t = 1usize;
        let end_t = 7usize;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let query = (0..heads * head_dim)
            .map(|i| ((i * 17 % 101) as f32 - 50.0) / 37.0)
            .collect::<Vec<_>>();
        let keys = (0..slot_count * key_stride)
            .map(|i| ((i * 29 % 127) as f32 - 63.0) / 43.0)
            .collect::<Vec<_>>();
        let values = (0..slot_count * value_stride)
            .map(|i| ((i * 31 % 137) as f32 - 68.0) / 47.0)
            .collect::<Vec<_>>();

        let run_metal = |parallel: bool| {
            let mut out = vec![0.0f32; heads * value_dim];
            let ok = unsafe {
                super::ffi::rusty_metal_test_resident_attention(
                    query.as_ptr(),
                    keys.as_ptr(),
                    values.as_ptr(),
                    out.as_mut_ptr(),
                    heads as u32,
                    kv_mul as u32,
                    head_dim as u32,
                    value_dim as u32,
                    key_stride as u32,
                    value_stride as u32,
                    slot_count as u32,
                    start_t as u32,
                    end_t as u32,
                    scale,
                    i32::from(parallel),
                ) != 0
            };
            assert!(ok);
            out
        };
        let serial = run_metal(false);
        let parallel = run_metal(true);

        let mut expected = vec![0.0f32; heads * value_dim];
        for head in 0..heads {
            let kv_head = head / kv_mul;
            let q = &query[head * head_dim..(head + 1) * head_dim];
            let mut scores = Vec::with_capacity(end_t - start_t + 1);
            for t in start_t..=end_t {
                let k = &keys[t * key_stride + kv_head * head_dim
                    ..t * key_stride + (kv_head + 1) * head_dim];
                scores.push(q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() * scale);
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights = scores
                .iter()
                .map(|score| (score - max).exp())
                .collect::<Vec<_>>();
            let denom = weights.iter().sum::<f32>();
            for (offset, &weight) in weights.iter().enumerate() {
                let t = start_t + offset;
                let v = &values[t * value_stride + kv_head * value_dim
                    ..t * value_stride + (kv_head + 1) * value_dim];
                for (dst, &value) in expected[head * value_dim..(head + 1) * value_dim]
                    .iter_mut()
                    .zip(v)
                {
                    *dst += weight * value / denom;
                }
            }
        }

        for (index, ((&cpu, &serial_value), &parallel_value)) in
            expected.iter().zip(&serial).zip(&parallel).enumerate()
        {
            let tolerance = 8.0e-5 * cpu.abs().max(1.0);
            assert!(
                (cpu - serial_value).abs() <= tolerance,
                "index {index}: CPU={cpu}, serial={serial_value}, tolerance={tolerance}"
            );
            assert!(
                (cpu - parallel_value).abs() <= tolerance,
                "index {index}: CPU={cpu}, parallel={parallel_value}, tolerance={tolerance}"
            );
        }
    }

    #[test]
    /// Verifies truthy environment values accepted by the Metal flag parser.
    fn metal_env_flag_accepts_explicit_truthy_values() {
        for value in ["", "1", "true", "TRUE", "yes", "on"] {
            assert!(parse_env_flag(value), "{value:?} should enable Metal");
        }
    }

    #[test]
    /// Verifies falsey environment values rejected by the Metal flag parser.
    fn metal_env_flag_rejects_explicit_false_values() {
        for value in ["0", "false", "FALSE", "no", "off", "maybe"] {
            assert!(!parse_env_flag(value), "{value:?} should disable Metal");
        }
    }

    #[test]
    /// Verifies that tiny Q4_K projections stay on the CPU path.
    fn q4k_single_metal_heuristic_skips_small_projections() {
        assert!(!super::q4k_single_should_use_metal(1024, 3072));
        assert!(super::q4k_single_should_use_metal(9216, 3072));
        assert!(super::q4k_single_should_use_metal(3072, 4096));
    }

    #[test]
    /// Verifies the fused K-quant heuristic gates on both a row floor and total
    /// work, keeping dispatch-bound small projections on CPU.
    fn fused_kquant_metal_heuristic_skips_tiny_projections() {
        // Below the row floor stays on CPU.
        assert!(!super::fused_kquant_should_use_metal(128, 3072));
        // Meets the row floor but not the ~12M work floor (gemma-4-E2B QKV
        // ~3072x2048 = 6.3M) — must stay on CPU (was the Metal-slower-than-CPU
        // regression).
        assert!(!super::fused_kquant_should_use_metal(3072, 2048));
        // Clears both floors (Ministral-3B QKV ~5120x3072 = 15.7M) — Metal.
        assert!(super::fused_kquant_should_use_metal(5120, 3072));
        // Wide-cols escape keeps large matvecs on the GPU regardless of work.
        assert!(super::fused_kquant_should_use_metal(128, 4096));
    }

    #[test]
    /// Verifies that ultra mode routes smaller Mistral projections to Metal.
    fn ultra_mode_lowers_metal_matvec_thresholds() {
        assert!(!super::ultra_mode_enabled());
        {
            let _guard = super::scoped_ultra_mode(true);
            assert!(super::ultra_mode_enabled());
            assert!(super::q4k_single_should_use_metal(1024, 3072));
            assert_eq!(super::q6k_min_metal_rows(), super::ULTRA_Q6K_MIN_METAL_ROWS);
        }
        assert!(!super::ultra_mode_enabled());
    }

    #[test]
    /// Verifies that the scoped dispatch policy restores the previous ultra state.
    fn dispatch_policy_restores_previous_ultra_state() {
        assert!(!super::ultra_mode_enabled());
        {
            let _guard = super::scoped_dispatch_policy(false, true);
            assert!(super::ultra_mode_enabled());
        }
        assert!(!super::ultra_mode_enabled());
    }

    #[test]
    /// Verifies the Metal attention threshold parser handles overrides and fallbacks.
    fn attention_min_tokens_parser_handles_overrides() {
        assert_eq!(
            super::parse_attention_min_metal_tokens(None),
            super::ATTENTION_MIN_METAL_TOKENS
        );
        assert_eq!(super::parse_attention_min_metal_tokens(Some("0")), 0);
        assert_eq!(super::parse_attention_min_metal_tokens(Some("512")), 512);
        assert_eq!(
            super::parse_attention_min_metal_tokens(Some("bogus")),
            super::ATTENTION_MIN_METAL_TOKENS
        );
        assert_eq!(
            super::parse_attention_min_metal_tokens(Some("  768  ")),
            768
        );
    }

    #[test]
    /// Verifies that short attention windows stay on the CPU path.
    fn attention_metal_heuristic_skips_short_windows() {
        assert!(!super::attention_scan_should_use_metal(0, 8_190));
        assert!(super::attention_scan_should_use_metal(0, 8_191));
    }

    #[test]
    /// Verifies that ultra mode lowers the attention Metal threshold.
    fn ultra_mode_lowers_attention_threshold() {
        assert!(!super::attention_scan_should_use_metal(
            0,
            super::ULTRA_ATTENTION_MIN_METAL_TOKENS - 1
        ));
        let _guard = super::scoped_ultra_mode(true);
        assert!(super::attention_scan_should_use_metal(
            0,
            super::ULTRA_ATTENTION_MIN_METAL_TOKENS - 1
        ));
    }
}
