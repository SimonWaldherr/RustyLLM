use std::sync::OnceLock;

pub const Q6K_MIN_METAL_ROWS: usize = 2_048;
pub const ATTENTION_MIN_METAL_TOKENS: usize = 8_192;
static ATTENTION_MIN_METAL_TOKENS_RUNTIME: OnceLock<usize> = OnceLock::new();

fn parse_attention_min_metal_tokens(raw: Option<&str>) -> usize {
    raw.and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(ATTENTION_MIN_METAL_TOKENS)
}

/// Returns the attention window size threshold that enables Metal.
pub fn attention_min_metal_tokens() -> usize {
    *ATTENTION_MIN_METAL_TOKENS_RUNTIME.get_or_init(|| {
        let raw = std::env::var("RUSTY_LLM_METAL_ATTENTION_MIN_TOKENS").ok();
        parse_attention_min_metal_tokens(raw.as_deref())
    })
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
    *ENABLED.get_or_init(|| requested() != Some(false) && available())
}

/// Reads the environment flag that requests Metal acceleration.
pub fn requested() -> Option<bool> {
    env_flag("RUSTY_LLM_METAL")
}

/// Reads the environment flag for experimental Q6_K Metal acceleration.
pub fn q6k_enabled() -> bool {
    enabled()
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
    if !enabled() || !q4k_single_should_use_metal(rows, cols) {
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
    if !enabled() || rows < Q6K_MIN_METAL_ROWS {
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
    if !enabled() {
        return false;
    }
    let (weights_a, rows_a, cols_a) = a;
    let (weights_b, rows_b, cols_b) = b;
    if cols_a != cols_b || cols_a != x.len() || rows_a + rows_b < Q6K_MIN_METAL_ROWS {
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
    if !enabled() {
        return false;
    }
    let (weights_a, rows_a, cols_a) = a;
    let (weights_b, rows_b, cols_b) = b;
    let (weights_c, rows_c, cols_c) = c;
    if cols_a != cols_b
        || cols_a != cols_c
        || cols_a != x.len()
        || rows_a + rows_b + rows_c < Q6K_MIN_METAL_ROWS
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
    if !enabled() {
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
    if !enabled() {
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

/// Decides whether a single Q4_K projection is large enough for Metal dispatch.
fn q4k_single_should_use_metal(rows: usize, cols: usize) -> bool {
    rows >= 8_192 || cols >= 4_096
}

/// Decides whether a full attention scan is large enough for Metal dispatch.
#[cfg(any(all(target_os = "macos", rusty_metal), test))]
fn attention_scan_should_use_metal(start_t: usize, end_t: usize) -> bool {
    end_t
        .checked_sub(start_t)
        .map(|span| span + 1 >= attention_min_metal_tokens())
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
    if !enabled()
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
    if !enabled() || rows < Q4_0_MIN_METAL_ROWS || (cols % 32) != 0 {
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
    if !enabled() || rows < Q8_0_MIN_METAL_ROWS || (cols % 32) != 0 {
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
}
