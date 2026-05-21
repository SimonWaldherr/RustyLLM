use std::sync::OnceLock;

#[cfg(all(target_os = "macos", rusty_metal))]
mod ffi {
    unsafe extern "C" {
        pub fn rusty_metal_available() -> i32;
        pub fn rusty_metal_q4k_matvec(
            weights: *const u8,
            weights_len: usize,
            x: *const f32,
            rows: usize,
            cols: usize,
            out: *mut f32,
        ) -> i32;
        pub fn rusty_metal_q6k_matvec(
            weights: *const u8,
            weights_len: usize,
            x: *const f32,
            rows: usize,
            cols: usize,
            out: *mut f32,
        ) -> i32;
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
    }
}

#[cfg(all(target_os = "macos", rusty_metal))]
pub fn available() -> bool {
    unsafe { ffi::rusty_metal_available() != 0 }
}

#[cfg(not(all(target_os = "macos", rusty_metal)))]
pub fn available() -> bool {
    false
}

pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| requested() == Some(true) && available())
}

pub fn requested() -> Option<bool> {
    env_flag("RUSTY_LLM_METAL")
}

pub fn q6k_enabled() -> bool {
    enabled()
}

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

pub fn q6k_matvec_into(
    weights: &[u8],
    x: &[f32],
    rows: usize,
    cols: usize,
    out: &mut Vec<f32>,
) -> bool {
    if !enabled() || rows < 32_768 {
        return false;
    }
    out.resize(rows, 0.0);
    q6k_matvec_raw(weights, x, rows, cols, out)
}

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

fn q4k_single_should_use_metal(rows: usize, cols: usize) -> bool {
    rows >= 8_192 || cols >= 4_096
}

#[cfg(all(target_os = "macos", rusty_metal))]
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
fn q4k_matvec_raw(
    _weights: &[u8],
    _x: &[f32],
    _rows: usize,
    _cols: usize,
    _out: &mut [f32],
) -> bool {
    false
}

fn env_flag(name: &str) -> Option<bool> {
    let value = std::env::var(name).ok()?;
    Some(parse_env_flag(&value))
}

fn parse_env_flag(value: &str) -> bool {
    match value.trim().to_ascii_lowercase().as_str() {
        "" | "1" | "true" | "yes" | "on" => true,
        "0" | "false" | "no" | "off" => false,
        _ => false,
    }
}

#[cfg(not(all(target_os = "macos", rusty_metal)))]
#[allow(clippy::too_many_arguments)]
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

#[cfg(test)]
mod tests {
    use super::parse_env_flag;

    #[test]
    fn metal_env_flag_accepts_explicit_truthy_values() {
        for value in ["", "1", "true", "TRUE", "yes", "on"] {
            assert!(parse_env_flag(value), "{value:?} should enable Metal");
        }
    }

    #[test]
    fn metal_env_flag_rejects_explicit_false_values() {
        for value in ["0", "false", "FALSE", "no", "off", "maybe"] {
            assert!(!parse_env_flag(value), "{value:?} should disable Metal");
        }
    }

    #[test]
    fn q4k_single_metal_heuristic_skips_small_projections() {
        assert!(!super::q4k_single_should_use_metal(1024, 3072));
        assert!(super::q4k_single_should_use_metal(9216, 3072));
        assert!(super::q4k_single_should_use_metal(3072, 4096));
    }
}
