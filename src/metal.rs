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
    std::env::var_os("RUSTY_LLM_METAL").is_some() && available()
}

pub fn q4k_matvec_into(
    weights: &[u8],
    x: &[f32],
    rows: usize,
    cols: usize,
    out: &mut Vec<f32>,
) -> bool {
    if !enabled() {
        return false;
    }
    out.resize(rows, 0.0);
    q4k_matvec_raw(weights, x, rows, cols, out)
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
