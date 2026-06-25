use std::cell::Cell;
use std::sync::OnceLock;

pub const Q4K_MIN_METAL_ROWS: usize = 8_192;
pub const Q4K_MIN_METAL_COLS: usize = 4_096;
pub const Q6K_MIN_METAL_ROWS: usize = 2_048;
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
    }
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
    *NOCOPY_ENABLED.get_or_init(|| env_flag("RUSTY_LLM_METAL_NOCOPY") != Some(false))
}

/// Reports whether Mistral-style fused Metal FFN blocks are enabled.
pub fn fused_ffn_enabled() -> bool {
    static FUSED_FFN_ENABLED: OnceLock<bool> = OnceLock::new();
    *FUSED_FFN_ENABLED.get_or_init(|| env_flag("RUSTY_LLM_METAL_FUSED_FFN") != Some(false))
}

/// Reports whether the experimental fused Mistral post-attention/FFN block is enabled.
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
    if cols_a != cols_b || cols_a != x.len() {
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
    if cols_a != cols_b || cols_a != cols_c || cols_a != x.len() {
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
    if cols_a != cols_b || cols_a != cols_c || cols_a != x.len() {
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
