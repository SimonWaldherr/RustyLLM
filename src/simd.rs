// simd.rs — Platform-specific SIMD kernels
//
// Provides optimized dot products for:
//   - f32 vectors (NEON / AVX2 / scalar)
//   - Q8_0 quantized vectors (fused dequant+dot)
//   - Q4_0 quantized vectors (fused dequant+dot)
//
// On Apple Silicon, NEON is always available — no feature detection needed.
// On x86_64, we runtime-detect AVX2 and fall back to scalar.

use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(not(target_family = "wasm"))]
use std::thread;

static NUM_THREADS: AtomicUsize = AtomicUsize::new(0);

// ─── f16 ↔ f32 conversion ────────────────────────────────────────────────────

#[inline(always)]
pub fn f16_to_f32(h: u16) -> f32 {
    // Use native hardware conversion where available
    #[cfg(target_arch = "aarch64")]
    {
        // ARM has native f16 support
        // But Rust doesn't expose vcvth_f32_f16 easily, so we do IEEE754 manually
        // Still very fast on Apple Silicon's pipeline
        f16_to_f32_soft(h)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        f16_to_f32_soft(h)
    }
}

pub fn set_num_threads(n: usize) {
    NUM_THREADS.store(n.max(1), Ordering::Relaxed);
}

#[inline]
fn num_threads() -> usize {
    let configured = NUM_THREADS.load(Ordering::Relaxed);
    if configured > 0 {
        configured
    } else if cfg!(target_family = "wasm") {
        1
    } else {
        {
            #[cfg(not(target_family = "wasm"))]
            {
                thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1)
            }
            #[cfg(target_family = "wasm")]
            {
                1
            }
        }
    }
}

fn parallel_matvec<T, F>(out: &mut [f32], rows: usize, row_span: usize, data: &[T], worker: F)
where
    T: Sync,
    F: Fn(&[T]) -> f32 + Sync,
{
    let threads = num_threads().min(rows);

    if threads <= 1 || rows < threads * 8 {
        for r in 0..rows {
            out[r] = worker(&data[r * row_span..(r + 1) * row_span]);
        }
        return;
    }

    #[cfg(target_family = "wasm")]
    {
        for r in 0..rows {
            out[r] = worker(&data[r * row_span..(r + 1) * row_span]);
        }
        return;
    }

    #[cfg(not(target_family = "wasm"))]
    {
        let chunk_rows = rows.div_ceil(threads);
        thread::scope(|scope| {
            for (chunk_idx, out_chunk) in out.chunks_mut(chunk_rows).enumerate() {
                let start_row = chunk_idx * chunk_rows;
                let start = start_row * row_span;
                let end = start + out_chunk.len() * row_span;
                let data_chunk = &data[start..end];
                let worker = &worker;
                scope.spawn(move || {
                    for (local_row, out_cell) in out_chunk.iter_mut().enumerate() {
                        let row = &data_chunk[local_row * row_span..(local_row + 1) * row_span];
                        *out_cell = worker(row);
                    }
                });
            }
        });
    }
}

#[inline(always)]
fn f16_to_f32_soft(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        // Subnormal
        let mut e = 0u32;
        let mut m = mant;
        while (m & 0x400) == 0 {
            m <<= 1;
            e += 1;
        }
        m &= 0x3ff;
        let e = 127 - 15 + 1 - e;
        return f32::from_bits((sign << 31) | (e << 23) | (m << 13));
    }
    if exp == 31 {
        return f32::from_bits((sign << 31) | (0xff << 23) | (mant << 13));
    }
    let e = exp + 127 - 15;
    f32::from_bits((sign << 31) | (e << 23) | (mant << 13))
}

// ─── Dispatch: pick best implementation at compile time ──────────────────────

/// f32 dot product of two slices (same length)
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { dot_f32_neon(a, b) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { dot_f32_avx2(a, b) }
        } else {
            dot_f32_scalar(a, b)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        dot_f32_scalar(a, b)
    }
}

/// Fused Q8_0 dot product: quantized_weight · f32_input
/// `qdata` is raw Q8_0 blocks, `x` is f32 input vector
/// `n` is the number of elements (must be multiple of 32)
#[inline]
pub fn dot_q8_0_f32(qdata: &[u8], x: &[f32], n: usize) -> f32 {
    debug_assert!(n % 32 == 0);
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { dot_q8_0_f32_neon(qdata, x, n) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { dot_q8_0_f32_avx2(qdata, x, n) }
        } else {
            dot_q8_0_f32_scalar(qdata, x, n)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        dot_q8_0_f32_scalar(qdata, x, n)
    }
}

/// Fused Q4_0 dot product
#[inline]
pub fn dot_q4_0_f32(qdata: &[u8], x: &[f32], n: usize) -> f32 {
    debug_assert!(n % 32 == 0);
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { dot_q4_0_f32_neon(qdata, x, n) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { dot_q4_0_f32_avx2(qdata, x, n) }
        } else {
            dot_q4_0_f32_scalar(qdata, x, n)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        dot_q4_0_f32_scalar(qdata, x, n)
    }
}

/// Fused Q4_K dot product
#[inline]
pub fn dot_q4_k_f32(qdata: &[u8], x: &[f32], n: usize) -> f32 {
    debug_assert!(n % 256 == 0);
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { dot_q4_k_f32_neon(qdata, x, n) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        dot_q4_k_f32_scalar(qdata, x, n)
    }
}

/// Fused Q6_K dot product
#[inline]
pub fn dot_q6_k_f32(qdata: &[u8], x: &[f32], n: usize) -> f32 {
    debug_assert!(n % 256 == 0);
    dot_q6_k_f32_scalar(qdata, x, n)
}

#[inline]
pub fn dot_mxfp4_f32(qdata: &[u8], x: &[f32], n: usize) -> f32 {
    debug_assert!(n % 32 == 0);
    dot_mxfp4_f32_scalar(qdata, x, n)
}

// ─── Simple matvec implementations ──────────────────────────────────────────

pub fn matvec_f32(weight: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows];
    parallel_matvec(&mut out, rows, cols, weight, |row| dot_f32(row, x));
    out
}

pub fn matvec_q8_0(qweight: &[u8], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let row_bytes = (cols / 32) * 34;
    let needed = row_bytes
        .checked_mul(rows)
        .expect("matvec_q8_0: rows*row_bytes overflow");
    assert!(
        qweight.len() >= needed,
        "matvec_q8_0: buffer too small (need {}, got {})",
        needed,
        qweight.len()
    );
    let mut out = vec![0.0f32; rows];
    parallel_matvec(&mut out, rows, row_bytes, qweight, |row| {
        dot_q8_0_f32(row, x, cols)
    });
    out
}

pub fn matvec_q4_0(qweight: &[u8], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let row_bytes = (cols / 32) * 18;
    let needed = row_bytes
        .checked_mul(rows)
        .expect("matvec_q4_0: rows*row_bytes overflow");
    assert!(
        qweight.len() >= needed,
        "matvec_q4_0: buffer too small (need {}, got {})",
        needed,
        qweight.len()
    );
    let mut out = vec![0.0f32; rows];
    parallel_matvec(&mut out, rows, row_bytes, qweight, |row| {
        dot_q4_0_f32(row, x, cols)
    });
    out
}

pub fn matvec_q4_k(qweight: &[u8], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let row_bytes = (cols / 256) * 144;
    let needed = row_bytes
        .checked_mul(rows)
        .expect("matvec_q4_k: rows*row_bytes overflow");
    assert!(
        qweight.len() >= needed,
        "matvec_q4_k: buffer too small (need {}, got {})",
        needed,
        qweight.len()
    );
    let mut out = vec![0.0f32; rows];
    parallel_matvec(&mut out, rows, row_bytes, qweight, |row| {
        dot_q4_k_f32(row, x, cols)
    });
    out
}

pub fn matvec_q6_k(qweight: &[u8], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let row_bytes = (cols / 256) * 210;
    let needed = row_bytes
        .checked_mul(rows)
        .expect("matvec_q6_k: rows*row_bytes overflow");
    assert!(
        qweight.len() >= needed,
        "matvec_q6_k: buffer too small (need {}, got {})",
        needed,
        qweight.len()
    );
    let mut out = vec![0.0f32; rows];
    parallel_matvec(&mut out, rows, row_bytes, qweight, |row| {
        dot_q6_k_f32(row, x, cols)
    });
    out
}

pub fn matvec_mxfp4(qweight: &[u8], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let row_bytes = (cols / 32) * 17;
    let needed = row_bytes
        .checked_mul(rows)
        .expect("matvec_mxfp4: rows*row_bytes overflow");
    assert!(
        qweight.len() >= needed,
        "matvec_mxfp4: buffer too small (need {}, got {})",
        needed,
        qweight.len()
    );
    let mut out = vec![0.0f32; rows];
    parallel_matvec(&mut out, rows, row_bytes, qweight, |row| {
        dot_mxfp4_f32(row, x, cols)
    });
    out
}

// ─── In-place matvec variants (write into caller buffer, no alloc) ───────────

pub fn matvec_f32_into(weight: &[f32], x: &[f32], rows: usize, cols: usize, out: &mut Vec<f32>) {
    out.resize(rows, 0.0);
    parallel_matvec(out, rows, cols, weight, |row| dot_f32(row, x));
}

pub fn matvec_q8_0_into(qweight: &[u8], x: &[f32], rows: usize, cols: usize, out: &mut Vec<f32>) {
    let row_bytes = (cols / 32) * 34;
    out.resize(rows, 0.0);
    parallel_matvec(out, rows, row_bytes, qweight, |row| {
        dot_q8_0_f32(row, x, cols)
    });
}

pub fn matvec_q4_0_into(qweight: &[u8], x: &[f32], rows: usize, cols: usize, out: &mut Vec<f32>) {
    let row_bytes = (cols / 32) * 18;
    out.resize(rows, 0.0);
    parallel_matvec(out, rows, row_bytes, qweight, |row| {
        dot_q4_0_f32(row, x, cols)
    });
}

pub fn matvec_q4_k_into(qweight: &[u8], x: &[f32], rows: usize, cols: usize, out: &mut Vec<f32>) {
    let row_bytes = (cols / 256) * 144;
    out.resize(rows, 0.0);
    parallel_matvec(out, rows, row_bytes, qweight, |row| {
        dot_q4_k_f32(row, x, cols)
    });
}

pub fn matvec_q6_k_into(qweight: &[u8], x: &[f32], rows: usize, cols: usize, out: &mut Vec<f32>) {
    let row_bytes = (cols / 256) * 210;
    out.resize(rows, 0.0);
    parallel_matvec(out, rows, row_bytes, qweight, |row| {
        dot_q6_k_f32(row, x, cols)
    });
}

pub fn matvec_mxfp4_into(qweight: &[u8], x: &[f32], rows: usize, cols: usize, out: &mut Vec<f32>) {
    let row_bytes = (cols / 32) * 17;
    out.resize(rows, 0.0);
    parallel_matvec(out, rows, row_bytes, qweight, |row| {
        dot_mxfp4_f32(row, x, cols)
    });
}

pub fn dequant_row_q8_0(qrow: &[u8], cols: usize) -> Vec<f32> {
    let n_blocks = cols / 32;
    let block_size = 34;
    let mut out = vec![0.0f32; cols];
    for b in 0..n_blocks {
        let block = &qrow[b * block_size..(b + 1) * block_size];
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for i in 0..32 {
            out[b * 32 + i] = scale * (block[2 + i] as i8) as f32;
        }
    }
    out
}

pub fn dequant_row_q4_0(qrow: &[u8], cols: usize) -> Vec<f32> {
    let n_blocks = cols / 32;
    let block_size = 18;
    let mut out = vec![0.0f32; cols];
    for b in 0..n_blocks {
        let block = &qrow[b * block_size..(b + 1) * block_size];
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for i in 0..16 {
            let byte = block[2 + i];
            let lo = ((byte & 0x0F) as i32 - 8) as f32;
            let hi = (((byte >> 4) & 0x0F) as i32 - 8) as f32;
            out[b * 32 + i * 2] = scale * lo;
            out[b * 32 + i * 2 + 1] = scale * hi;
        }
    }
    out
}

pub fn dequant_row_q4_k(qrow: &[u8], cols: usize) -> Vec<f32> {
    let n_blocks = cols / 256;
    let block_size = 144;
    let mut out = vec![0.0f32; cols];

    for b in 0..n_blocks {
        let block = &qrow[b * block_size..(b + 1) * block_size];
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales: &[u8; 12] = block[4..16].try_into().expect("q4_k scales size");
        let mut q = &block[16..];
        let yoff = b * 256;

        let mut is = 0usize;
        for j in (0..256).step_by(64) {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let d1 = d * sc1 as f32;
            let min1 = dmin * m1 as f32;

            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc2 as f32;
            let min2 = dmin * m2 as f32;

            for l in 0..32 {
                out[yoff + j + l] = d1 * (q[l] & 0x0F) as f32 - min1;
            }
            for l in 0..32 {
                out[yoff + j + 32 + l] = d2 * (q[l] >> 4) as f32 - min2;
            }

            q = &q[32..];
            is += 2;
        }
    }

    out
}

pub fn dequant_row_q6_k(qrow: &[u8], cols: usize) -> Vec<f32> {
    let n_blocks = cols / 256;
    let block_size = 210;
    let mut out = vec![0.0f32; cols];

    for b in 0..n_blocks {
        let block = &qrow[b * block_size..(b + 1) * block_size];
        let mut ql = &block[0..128];
        let mut qh = &block[128..192];
        let mut sc = &block[192..208];
        let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
        let yoff = b * 256;

        for n in (0..256).step_by(128) {
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((((ql[l] & 0x0F) | (((qh[l] >> 0) & 0x03) << 4)) as i32) - 32) as f32;
                let q2 =
                    ((((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 0x03) << 4)) as i32) - 32) as f32;
                let q3 = ((((ql[l] >> 4) | (((qh[l] >> 4) & 0x03) << 4)) as i32) - 32) as f32;
                let q4 = ((((ql[l + 32] >> 4) | (((qh[l] >> 6) & 0x03) << 4)) as i32) - 32) as f32;

                out[yoff + n + l] = d * (sc[is] as i8 as f32) * q1;
                out[yoff + n + 32 + l] = d * (sc[is + 2] as i8 as f32) * q2;
                out[yoff + n + 64 + l] = d * (sc[is + 4] as i8 as f32) * q3;
                out[yoff + n + 96 + l] = d * (sc[is + 6] as i8 as f32) * q4;
            }
            ql = &ql[64..];
            qh = &qh[32..];
            sc = &sc[8..];
        }
    }

    out
}

pub fn dequant_row_mxfp4(qrow: &[u8], cols: usize) -> Vec<f32> {
    let n_blocks = cols / 32;
    let block_size = 17;
    let mut out = vec![0.0f32; cols];

    for b in 0..n_blocks {
        let block = &qrow[b * block_size..(b + 1) * block_size];
        let scale = 2.0f32.powi(block[16] as i32 - 127);
        for i in 0..16 {
            let byte = block[i];
            out[b * 32 + i * 2] = mxfp4_nibble_to_f32(byte & 0x0F) * scale;
            out[b * 32 + i * 2 + 1] = mxfp4_nibble_to_f32(byte >> 4) * scale;
        }
    }

    out
}

// ─── Scalar fallbacks ────────────────────────────────────────────────────────

#[allow(dead_code)]
fn dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    // 4-way unrolled accumulator to exploit ILP
    let n = a.len();
    let chunks = n / 4;
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;

    for i in 0..chunks {
        let base = i * 4;
        s0 += a[base] * b[base];
        s1 += a[base + 1] * b[base + 1];
        s2 += a[base + 2] * b[base + 2];
        s3 += a[base + 3] * b[base + 3];
    }

    for i in (chunks * 4)..n {
        s0 += a[i] * b[i];
    }

    (s0 + s1) + (s2 + s3)
}

#[allow(dead_code)]
fn dot_q8_0_f32_scalar(qdata: &[u8], x: &[f32], n: usize) -> f32 {
    let n_blocks = n / 32;
    let block_size = 34; // 2 bytes scale (f16) + 32 bytes (i8)
    let mut sum = 0.0f32;

    for b in 0..n_blocks {
        let block = &qdata[b * block_size..];
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));

        let mut block_sum = 0.0f32;
        for i in 0..32 {
            block_sum += (block[2 + i] as i8) as f32 * x[b * 32 + i];
        }
        sum += scale * block_sum;
    }
    sum
}

#[allow(dead_code)]
fn dot_q4_0_f32_scalar(qdata: &[u8], x: &[f32], n: usize) -> f32 {
    let n_blocks = n / 32;
    let block_size = 18; // 2 bytes scale + 16 bytes (32 nibbles)
    let mut sum = 0.0f32;

    for b in 0..n_blocks {
        let block = &qdata[b * block_size..];
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));

        let mut block_sum = 0.0f32;
        for i in 0..16 {
            let byte = block[2 + i];
            let lo = ((byte & 0x0F) as i32 - 8) as f32;
            let hi = (((byte >> 4) & 0x0F) as i32 - 8) as f32;
            block_sum += lo * x[b * 32 + i * 2];
            block_sum += hi * x[b * 32 + i * 2 + 1];
        }
        sum += scale * block_sum;
    }
    sum
}

#[inline]
fn get_scale_min_k4(j: usize, q: &[u8; 12]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

#[allow(dead_code)]
fn dot_q4_k_f32_scalar(qdata: &[u8], x: &[f32], n: usize) -> f32 {
    let n_blocks = n / 256;
    let block_size = 144; // f16 d + f16 dmin + 12-byte scales + 128-byte nibbles
    let mut sum = 0.0f32;

    for b in 0..n_blocks {
        let block = &qdata[b * block_size..(b + 1) * block_size];
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales: &[u8; 12] = block[4..16].try_into().expect("q4_k scales size");
        let mut q = &block[16..];

        let xoff = b * 256;
        let mut is = 0usize;
        for j in (0..256).step_by(64) {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let d1 = d * sc1 as f32;
            let min1 = dmin * m1 as f32;

            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc2 as f32;
            let min2 = dmin * m2 as f32;

            let x1 = &x[xoff + j..xoff + j + 32];
            let x2 = &x[xoff + j + 32..xoff + j + 64];

            let mut qdot1_a = 0.0f32;
            let mut qdot1_b = 0.0f32;
            let mut qdot1_c = 0.0f32;
            let mut qdot1_d = 0.0f32;
            let mut qdot2_a = 0.0f32;
            let mut qdot2_b = 0.0f32;
            let mut qdot2_c = 0.0f32;
            let mut qdot2_d = 0.0f32;
            let mut xsum1_a = 0.0f32;
            let mut xsum1_b = 0.0f32;
            let mut xsum1_c = 0.0f32;
            let mut xsum1_d = 0.0f32;
            let mut xsum2_a = 0.0f32;
            let mut xsum2_b = 0.0f32;
            let mut xsum2_c = 0.0f32;
            let mut xsum2_d = 0.0f32;

            for l in (0..32).step_by(4) {
                let q0 = q[l];
                let q1 = q[l + 1];
                let q2 = q[l + 2];
                let q3 = q[l + 3];

                let x10 = x1[l];
                let x11 = x1[l + 1];
                let x12 = x1[l + 2];
                let x13 = x1[l + 3];
                let x20 = x2[l];
                let x21 = x2[l + 1];
                let x22 = x2[l + 2];
                let x23 = x2[l + 3];

                qdot1_a += (q0 & 0x0F) as f32 * x10;
                qdot1_b += (q1 & 0x0F) as f32 * x11;
                qdot1_c += (q2 & 0x0F) as f32 * x12;
                qdot1_d += (q3 & 0x0F) as f32 * x13;
                qdot2_a += (q0 >> 4) as f32 * x20;
                qdot2_b += (q1 >> 4) as f32 * x21;
                qdot2_c += (q2 >> 4) as f32 * x22;
                qdot2_d += (q3 >> 4) as f32 * x23;

                xsum1_a += x10;
                xsum1_b += x11;
                xsum1_c += x12;
                xsum1_d += x13;
                xsum2_a += x20;
                xsum2_b += x21;
                xsum2_c += x22;
                xsum2_d += x23;
            }

            let qdot1 = (qdot1_a + qdot1_b) + (qdot1_c + qdot1_d);
            let qdot2 = (qdot2_a + qdot2_b) + (qdot2_c + qdot2_d);
            let xsum1 = (xsum1_a + xsum1_b) + (xsum1_c + xsum1_d);
            let xsum2 = (xsum2_a + xsum2_b) + (xsum2_c + xsum2_d);

            sum += d1 * qdot1 - min1 * xsum1;
            sum += d2 * qdot2 - min2 * xsum2;

            q = &q[32..];
            is += 2;
        }
    }

    sum
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn dot_q4_k_f32_neon(qdata: &[u8], x: &[f32], n: usize) -> f32 {
    use std::arch::aarch64::*;

    let n_blocks = n / 256;
    let mask_low = vdup_n_u8(0x0F);
    let mut sum = 0.0f32;

    for b in 0..n_blocks {
        let block = qdata.as_ptr().add(b * 144);
        let d = f16_to_f32(u16::from_le_bytes([*block, *block.add(1)]));
        let dmin = f16_to_f32(u16::from_le_bytes([*block.add(2), *block.add(3)]));
        let scales: &[u8; 12] = std::slice::from_raw_parts(block.add(4), 12)
            .try_into()
            .expect("q4_k scales size");
        let mut q = block.add(16);
        let xbase = x.as_ptr().add(b * 256);

        let mut is = 0usize;
        for chunk in 0..4usize {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1 = d * sc1 as f32;
            let d2 = d * sc2 as f32;
            let min1 = dmin * m1 as f32;
            let min2 = dmin * m2 as f32;

            let x1 = xbase.add(chunk * 64);
            let x2 = x1.add(32);

            let mut qacc1a = vdupq_n_f32(0.0);
            let mut qacc1b = vdupq_n_f32(0.0);
            let mut qacc2a = vdupq_n_f32(0.0);
            let mut qacc2b = vdupq_n_f32(0.0);
            let mut xsum1a = vdupq_n_f32(0.0);
            let mut xsum1b = vdupq_n_f32(0.0);
            let mut xsum2a = vdupq_n_f32(0.0);
            let mut xsum2b = vdupq_n_f32(0.0);

            for i in 0..4usize {
                let nib = vld1_u8(q.add(i * 8));
                let lo = vand_u8(nib, mask_low);
                let hi = vshr_n_u8(nib, 4);

                let lo16 = vmovl_u8(lo);
                let hi16 = vmovl_u8(hi);

                let lo0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo16)));
                let lo1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo16)));
                let hi0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi16)));
                let hi1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi16)));

                let x1a = vld1q_f32(x1.add(i * 8));
                let x1b = vld1q_f32(x1.add(i * 8 + 4));
                let x2a = vld1q_f32(x2.add(i * 8));
                let x2b = vld1q_f32(x2.add(i * 8 + 4));

                qacc1a = vmlaq_f32(qacc1a, lo0, x1a);
                qacc1b = vmlaq_f32(qacc1b, lo1, x1b);
                qacc2a = vmlaq_f32(qacc2a, hi0, x2a);
                qacc2b = vmlaq_f32(qacc2b, hi1, x2b);

                xsum1a = vaddq_f32(xsum1a, x1a);
                xsum1b = vaddq_f32(xsum1b, x1b);
                xsum2a = vaddq_f32(xsum2a, x2a);
                xsum2b = vaddq_f32(xsum2b, x2b);
            }

            let qdot1 = vaddvq_f32(vaddq_f32(qacc1a, qacc1b));
            let qdot2 = vaddvq_f32(vaddq_f32(qacc2a, qacc2b));
            let xs1 = vaddvq_f32(vaddq_f32(xsum1a, xsum1b));
            let xs2 = vaddvq_f32(vaddq_f32(xsum2a, xsum2b));

            sum += d1 * qdot1 - min1 * xs1;
            sum += d2 * qdot2 - min2 * xs2;

            q = q.add(32);
            is += 2;
        }
    }

    sum
}

fn dot_q6_k_f32_scalar(qdata: &[u8], x: &[f32], n: usize) -> f32 {
    let n_blocks = n / 256;
    let block_size = 210;
    let mut sum = 0.0f32;

    for b in 0..n_blocks {
        let block = &qdata[b * block_size..(b + 1) * block_size];
        let mut ql = &block[0..128];
        let mut qh = &block[128..192];
        let mut sc = &block[192..208];
        let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
        let xoff = b * 256;

        for n in (0..256).step_by(128) {
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((((ql[l] & 0x0F) | (((qh[l] >> 0) & 0x03) << 4)) as i32) - 32) as f32;
                let q2 =
                    ((((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 0x03) << 4)) as i32) - 32) as f32;
                let q3 = ((((ql[l] >> 4) | (((qh[l] >> 4) & 0x03) << 4)) as i32) - 32) as f32;
                let q4 = ((((ql[l + 32] >> 4) | (((qh[l] >> 6) & 0x03) << 4)) as i32) - 32) as f32;

                sum += d * (sc[is] as i8 as f32) * q1 * x[xoff + n + l];
                sum += d * (sc[is + 2] as i8 as f32) * q2 * x[xoff + n + 32 + l];
                sum += d * (sc[is + 4] as i8 as f32) * q3 * x[xoff + n + 64 + l];
                sum += d * (sc[is + 6] as i8 as f32) * q4 * x[xoff + n + 96 + l];
            }
            ql = &ql[64..];
            qh = &qh[32..];
            sc = &sc[8..];
        }
    }

    sum
}

#[inline]
fn mxfp4_nibble_to_f32(v: u8) -> f32 {
    match v & 0x0F {
        0 => 0.0,
        1 => 0.5,
        2 => 1.0,
        3 => 1.5,
        4 => 2.0,
        5 => 3.0,
        6 => 4.0,
        7 => 6.0,
        8 => -0.0,
        9 => -0.5,
        10 => -1.0,
        11 => -1.5,
        12 => -2.0,
        13 => -3.0,
        14 => -4.0,
        15 => -6.0,
        _ => 0.0,
    }
}

fn dot_mxfp4_f32_scalar(qdata: &[u8], x: &[f32], n: usize) -> f32 {
    let n_blocks = n / 32;
    let block_size = 17;
    let mut sum = 0.0f32;

    for b in 0..n_blocks {
        let block = &qdata[b * block_size..(b + 1) * block_size];
        let scale = 2.0f32.powi(block[16] as i32 - 127);
        for i in 0..16 {
            let byte = block[i];
            sum += mxfp4_nibble_to_f32(byte & 0x0F) * scale * x[b * 32 + i * 2];
            sum += mxfp4_nibble_to_f32(byte >> 4) * scale * x[b * 32 + i * 2 + 1];
        }
    }

    sum
}

// ─── ARM NEON implementations (aarch64) ─────────────────────────────────────

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn dot_f32_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = a.len();
    let main = n / 16;
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);
    let mut ap = a.as_ptr();
    let mut bp = b.as_ptr();
    for _ in 0..main {
        acc0 = vmlaq_f32(acc0, vld1q_f32(ap), vld1q_f32(bp));
        acc1 = vmlaq_f32(acc1, vld1q_f32(ap.add(4)), vld1q_f32(bp.add(4)));
        acc2 = vmlaq_f32(acc2, vld1q_f32(ap.add(8)), vld1q_f32(bp.add(8)));
        acc3 = vmlaq_f32(acc3, vld1q_f32(ap.add(12)), vld1q_f32(bp.add(12)));
        ap = ap.add(16);
        bp = bp.add(16);
    }
    let acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    let mut sum = vaddvq_f32(acc);
    for i in (main * 16)..n {
        sum += a[i] * b[i];
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn dot_q8_0_f32_neon(qdata: &[u8], x: &[f32], n: usize) -> f32 {
    use std::arch::aarch64::*;
    let n_blocks = n / 32;
    let mut sum_acc = vdupq_n_f32(0.0);
    for b in 0..n_blocks {
        let block = qdata.as_ptr().add(b * 34);
        let scale = f16_to_f32(u16::from_le_bytes([*block, *block.add(1)]));
        let scale_v = vdupq_n_f32(scale);
        let q = block.add(2) as *const i8;
        let xp = x.as_ptr().add(b * 32);
        let mut bacc = vdupq_n_f32(0.0);
        // 32 i8 values in 4 groups of 8
        for i in 0..4_usize {
            let qi8 = vld1_s8(q.add(i * 8));
            let qi16 = vmovl_s8(qi8);
            let qlo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(qi16)));
            let qhi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(qi16)));
            bacc = vmlaq_f32(bacc, qlo, vld1q_f32(xp.add(i * 8)));
            bacc = vmlaq_f32(bacc, qhi, vld1q_f32(xp.add(i * 8 + 4)));
        }
        sum_acc = vmlaq_f32(sum_acc, bacc, scale_v);
    }
    vaddvq_f32(sum_acc)
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn dot_q4_0_f32_neon(qdata: &[u8], x: &[f32], n: usize) -> f32 {
    // Q4_0 layout per block (18 bytes): [f16 scale | 16 nibble bytes]
    // byte[i] lo nibble → weight for x[2i], hi nibble → weight for x[2i+1]
    use std::arch::aarch64::*;
    let n_blocks = n / 32;
    let mut sum_acc = vdupq_n_f32(0.0);
    let mask_low = vdupq_n_u8(0x0F);
    let eight = vdupq_n_u8(8);

    for b in 0..n_blocks {
        let block = qdata.as_ptr().add(b * 18);
        let scale = f16_to_f32(u16::from_le_bytes([*block, *block.add(1)]));
        let scale_v = vdupq_n_f32(scale);

        // Load 16 nibble bytes
        let nib = vld1q_u8(block.add(2));
        // lo nibbles (bits[3:0]): each nibble i → x[2i]
        let lo_i8 = vreinterpretq_s8_u8(vsubq_u8(vandq_u8(nib, mask_low), eight));
        // hi nibbles (bits[7:4]): each nibble i → x[2i+1]
        let hi_i8 = vreinterpretq_s8_u8(vsubq_u8(vshrq_n_u8(nib, 4), eight));

        // Widen to i16 (8 each per half)
        let lo_lo16 = vmovl_s8(vget_low_s8(lo_i8)); // lo nibbles 0..8  → i16x8
        let lo_hi16 = vmovl_s8(vget_high_s8(lo_i8)); // lo nibbles 8..16 → i16x8
        let hi_lo16 = vmovl_s8(vget_low_s8(hi_i8)); // hi nibbles 0..8  → i16x8
        let hi_hi16 = vmovl_s8(vget_high_s8(hi_i8)); // hi nibbles 8..16 → i16x8

        let xp = x.as_ptr().add(b * 32);
        let mut bacc = vdupq_n_f32(0.0);

        // Each "chunk" processes 4 lo+hi nibbles with 8 x values (even=lo, odd=hi)
        // using vuzp1q/vuzp2q to deinterleave x into even and odd lanes.
        macro_rules! chunk {
            ($lo4:expr, $hi4:expr, $xoff:expr) => {{
                let lof = vcvtq_f32_s32(vmovl_s16($lo4));
                let hif = vcvtq_f32_s32(vmovl_s16($hi4));
                let xa = vld1q_f32(xp.add($xoff));
                let xb = vld1q_f32(xp.add($xoff + 4));
                // deinterleave: even lanes = lo weight positions, odd = hi
                bacc = vmlaq_f32(bacc, lof, vuzp1q_f32(xa, xb));
                bacc = vmlaq_f32(bacc, hif, vuzp2q_f32(xa, xb));
            }};
        }
        chunk!(vget_low_s16(lo_lo16), vget_low_s16(hi_lo16), 0);
        chunk!(vget_high_s16(lo_lo16), vget_high_s16(hi_lo16), 8);
        chunk!(vget_low_s16(lo_hi16), vget_low_s16(hi_hi16), 16);
        chunk!(vget_high_s16(lo_hi16), vget_high_s16(hi_hi16), 24);

        sum_acc = vmlaq_f32(sum_acc, bacc, scale_v);
    }
    vaddvq_f32(sum_acc)
}

// ─── AVX2 + FMA implementations (x86_64) ────────────────────────────────────

/// Horizontal sum of 8 f32 in a __m256 register
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,sse3")]
#[inline]
unsafe fn hsum_avx(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let lo = _mm256_extractf128_ps(v, 0);
    let hi = _mm256_extractf128_ps(v, 1);
    let sum4 = _mm_add_ps(lo, hi);
    let sum2 = _mm_hadd_ps(sum4, sum4);
    let sum1 = _mm_hadd_ps(sum2, sum2);
    _mm_cvtss_f32(sum1)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let main = n / 8;
    let mut acc = _mm256_setzero_ps();
    let mut ap = a.as_ptr();
    let mut bp = b.as_ptr();
    for _ in 0..main {
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(ap), _mm256_loadu_ps(bp), acc);
        ap = ap.add(8);
        bp = bp.add(8);
    }
    let mut sum = hsum_avx(acc);
    for i in (main * 8)..n {
        sum += a[i] * b[i];
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q8_0_f32_avx2(qdata: &[u8], x: &[f32], n: usize) -> f32 {
    use std::arch::x86_64::*;
    let n_blocks = n / 32;
    let mut acc = _mm256_setzero_ps();
    for b in 0..n_blocks {
        let block = qdata.as_ptr().add(b * 34);
        let scale = f16_to_f32(u16::from_le_bytes([*block, *block.add(1)]));
        let sv = _mm256_set1_ps(scale);
        let qp = block.add(2) as *const i8;
        let xp = x.as_ptr().add(b * 32);
        // 32 i8 values in 4 groups of 8
        for i in 0..4_usize {
            // load 8 i8 into lower 64 bits of __m128i, sign-extend to i32x8
            let qi128 = _mm_loadl_epi64(qp.add(i * 8) as *const __m128i);
            let qi32 = _mm256_cvtepi8_epi32(qi128);
            let qf = _mm256_cvtepi32_ps(qi32);
            let xv = _mm256_loadu_ps(xp.add(i * 8));
            acc = _mm256_fmadd_ps(_mm256_mul_ps(sv, qf), xv, acc);
        }
    }
    hsum_avx(acc)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q4_0_f32_avx2(qdata: &[u8], x: &[f32], n: usize) -> f32 {
    // Q4_0 per block: [f16 scale | 16 nibble bytes]
    // byte[i]: lo=bits[3:0]→x[2i], hi=bits[7:4]→x[2i+1]
    // Strategy: interleave lo/hi nibbles so the reconstructed weights align with x[0..32].
    use std::arch::x86_64::*;
    let n_blocks = n / 32;
    let mut acc = _mm256_setzero_ps();
    let mask_0f = _mm_set1_epi8(0x0F_u8 as i8);
    let eight_ps = _mm256_set1_ps(8.0f32);

    for b in 0..n_blocks {
        let block = qdata.as_ptr().add(b * 18);
        let scale = f16_to_f32(u16::from_le_bytes([*block, *block.add(1)]));
        let sv = _mm256_set1_ps(scale);
        let xp = x.as_ptr().add(b * 32);

        // Load 16 nibble bytes
        let nib = _mm_loadu_si128(block.add(2) as *const __m128i);
        // lo nibbles: byte & 0x0F
        let lo = _mm_and_si128(nib, mask_0f);
        // hi nibbles: (byte >> 4) & 0x0F  — epi16 shift is safe here (see comments)
        let hi = _mm_and_si128(_mm_srli_epi16(nib, 4), mask_0f);

        // Interleave lo and hi into weight order matching x[0..32]:
        //   lo0,hi0,lo1,hi1,...,lo7,hi7  →  q[0..16]
        //   lo8,hi8,...,lo15,hi15         →  q[16..32]
        let q_lo16 = _mm_unpacklo_epi8(lo, hi); // q[0..16]
        let q_hi16 = _mm_unpackhi_epi8(lo, hi); // q[16..32]

        // cvtepu8_epi32 uses lower 8 bytes of __m128i → 8 × i32 (unsigned zero-extend)
        // Then subtract 8.0 in f32 to recover signed values in [-8, 7].
        macro_rules! process8 {
            ($qreg:expr, $xoff:expr) => {{
                let qf = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32($qreg)), eight_ps);
                let xv = _mm256_loadu_ps(xp.add($xoff));
                acc = _mm256_fmadd_ps(_mm256_mul_ps(sv, qf), xv, acc);
            }};
        }
        process8!(q_lo16, 0); // q[0..8]  · x[0..8]
        process8!(_mm_srli_si128(q_lo16, 8), 8); // q[8..16] · x[8..16]
        process8!(q_hi16, 16); // q[16..24]· x[16..24]
        process8!(_mm_srli_si128(q_hi16, 8), 24); // q[24..32]· x[24..32]
    }
    hsum_avx(acc)
}
