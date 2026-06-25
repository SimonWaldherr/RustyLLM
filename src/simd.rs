// simd.rs — Platform-specific SIMD kernels
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::needless_range_loop,
    clippy::needless_return
)]
//
// Provides optimized dot products for:
//   - f32 vectors (NEON / AVX2 / scalar)
//   - Q8_0 quantized vectors (fused dequant+dot)
//   - Q4_0 quantized vectors (fused dequant+dot)
//
// On Apple Silicon, NEON is always available — no feature detection needed.
// On x86_64, we runtime-detect AVX2 and fall back to scalar.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
#[cfg(not(target_family = "wasm"))]
use std::sync::{Arc, Condvar, Mutex, OnceLock};
#[cfg(not(target_family = "wasm"))]
use std::thread;

static NUM_THREADS: AtomicUsize = AtomicUsize::new(0);
static CPU_AFFINITY: AtomicBool = AtomicBool::new(false);
static WORKER_POLL_SPINS: AtomicUsize = AtomicUsize::new(2_000);
#[cfg(not(target_family = "wasm"))]
const MATVEC_CHUNK_ROWS: usize = 64;
#[cfg(not(target_family = "wasm"))]
const MIN_DYNAMIC_CHUNKS_PER_WORKER: usize = 4;

#[cfg(not(target_family = "wasm"))]
#[repr(align(64))]
struct CachePadded<T>(T);

#[cfg(target_arch = "x86_64")]
#[inline]
/// Detects whether AVX2/FMA kernels can be used on this CPU.
fn has_avx2_fma() -> bool {
    static HAS_AVX2_FMA: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *HAS_AVX2_FMA
        .get_or_init(|| is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"))
}

// ─── f16 ↔ f32 conversion ────────────────────────────────────────────────────

#[inline(always)]
/// Converts a half-precision bit pattern into `f32`.
pub fn f16_to_f32(h: u16) -> f32 {
    #[cfg(not(target_family = "wasm"))]
    {
        f16_lookup()[h as usize]
    }
    #[cfg(target_family = "wasm")]
    {
        f16_to_f32_soft(h)
    }
}

/// Sets the process-wide matrix-vector worker count.
pub fn set_num_threads(n: usize) {
    NUM_THREADS.store(n.max(1), Ordering::Relaxed);
}

#[cfg(not(target_family = "wasm"))]
/// Returns the operating-system reported parallelism.
pub fn available_threads() -> usize {
    thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .max(1)
}

#[cfg(target_family = "wasm")]
/// Returns the operating-system reported parallelism.
pub fn available_threads() -> usize {
    1
}

/// Enables best-effort worker CPU affinity for supported operating systems.
pub fn set_cpu_affinity_enabled(enabled: bool) {
    CPU_AFFINITY.store(enabled, Ordering::Relaxed);
}

/// Reports whether worker CPU affinity was requested.
pub fn cpu_affinity_enabled() -> bool {
    CPU_AFFINITY.load(Ordering::Relaxed)
}

/// Sets how long worker threads poll for new micro-jobs before sleeping.
pub fn set_worker_poll_spins(spins: usize) {
    WORKER_POLL_SPINS.store(spins, Ordering::Relaxed);
}

/// Returns the configured worker-poll spin count.
pub fn worker_poll_spins() -> usize {
    WORKER_POLL_SPINS.load(Ordering::Relaxed)
}

/// Pins the current compute thread when the target OS exposes a stable API.
pub fn pin_current_thread(worker_idx: usize) -> bool {
    if !cpu_affinity_enabled() {
        return false;
    }
    pin_current_thread_impl(worker_idx)
}

#[cfg(target_os = "macos")]
fn pin_current_thread_impl(worker_idx: usize) -> bool {
    const THREAD_AFFINITY_POLICY: i32 = 4;
    let tag = (worker_idx as i32).saturating_add(1);
    unsafe {
        let thread = mach_thread_self();
        thread_policy_set(thread, THREAD_AFFINITY_POLICY, &tag as *const i32, 1) == 0
    }
}

#[cfg(not(target_os = "macos"))]
fn pin_current_thread_impl(_worker_idx: usize) -> bool {
    false
}

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn mach_thread_self() -> u32;
    fn thread_policy_set(thread: u32, flavor: i32, policy_info: *const i32, count: u32) -> i32;
}

#[inline]
/// Returns the configured matrix-vector worker count.
pub fn num_threads() -> usize {
    let configured = NUM_THREADS.load(Ordering::Relaxed);
    if configured > 0 {
        configured
    } else if cfg!(target_family = "wasm") {
        1
    } else {
        #[cfg(not(target_family = "wasm"))]
        {
            static DEFAULT_THREADS: OnceLock<usize> = OnceLock::new();
            *DEFAULT_THREADS.get_or_init(available_threads)
        }
        #[cfg(target_family = "wasm")]
        {
            1
        }
    }
}

#[cfg(not(target_family = "wasm"))]
/// Returns the lazily initialized half-to-float lookup table.
fn f16_lookup() -> &'static [f32] {
    static F16_LOOKUP: OnceLock<Vec<f32>> = OnceLock::new();
    F16_LOOKUP.get_or_init(|| {
        let mut table = vec![0.0f32; 1 << 16];
        for (i, value) in table.iter_mut().enumerate() {
            *value = f16_to_f32_soft(i as u16);
        }
        table
    })
}

#[derive(Clone, Copy)]
enum MatvecKind {
    F32,
    Q8_0,
    Q8_1,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q4K,
    Q5K,
    Q6K,
    Mxfp4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KQuantMatvecKind {
    Q4K,
    Q5K,
    Q6K,
}

impl KQuantMatvecKind {
    #[inline]
    fn matvec_kind(self) -> MatvecKind {
        match self {
            Self::Q4K => MatvecKind::Q4K,
            Self::Q5K => MatvecKind::Q5K,
            Self::Q6K => MatvecKind::Q6K,
        }
    }

    #[inline]
    fn row_bytes(self, cols: usize) -> usize {
        match self {
            Self::Q4K => (cols / 256) * 144,
            Self::Q5K => (cols / 256) * 176,
            Self::Q6K => (cols / 256) * 210,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantMatvecKind {
    Q8_0,
    Q8_1,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q4K,
    Q5K,
    Q6K,
    Mxfp4,
}

impl QuantMatvecKind {
    #[inline]
    fn matvec_kind(self) -> MatvecKind {
        match self {
            Self::Q8_0 => MatvecKind::Q8_0,
            Self::Q8_1 => MatvecKind::Q8_1,
            Self::Q4_0 => MatvecKind::Q4_0,
            Self::Q4_1 => MatvecKind::Q4_1,
            Self::Q5_0 => MatvecKind::Q5_0,
            Self::Q5_1 => MatvecKind::Q5_1,
            Self::Q4K => MatvecKind::Q4K,
            Self::Q5K => MatvecKind::Q5K,
            Self::Q6K => MatvecKind::Q6K,
            Self::Mxfp4 => MatvecKind::Mxfp4,
        }
    }

    #[inline]
    fn block_size(self) -> usize {
        match self {
            Self::Q4K | Self::Q5K | Self::Q6K => 256,
            _ => 32,
        }
    }

    #[inline]
    fn row_bytes(self, cols: usize) -> Option<usize> {
        if cols == 0 || cols % self.block_size() != 0 {
            return None;
        }
        let blocks = cols / self.block_size();
        let bytes = match self {
            Self::Q8_0 => 34,
            Self::Q8_1 => 36,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Mxfp4 => 17,
        };
        blocks.checked_mul(bytes)
    }
}

#[inline]
fn is_kquant_kind(kind: QuantMatvecKind) -> bool {
    matches!(
        kind,
        QuantMatvecKind::Q4K | QuantMatvecKind::Q5K | QuantMatvecKind::Q6K
    )
}

#[inline]
fn is_mixed_kquant3(a: QuantMatvecKind, b: QuantMatvecKind, c: QuantMatvecKind) -> bool {
    is_kquant_kind(a) && is_kquant_kind(b) && is_kquant_kind(c) && !(a == b && b == c)
}

#[cfg(not(target_family = "wasm"))]
#[inline]
fn quant_kind_from_kquant(kind: KQuantMatvecKind) -> QuantMatvecKind {
    match kind {
        KQuantMatvecKind::Q4K => QuantMatvecKind::Q4K,
        KQuantMatvecKind::Q5K => QuantMatvecKind::Q5K,
        KQuantMatvecKind::Q6K => QuantMatvecKind::Q6K,
    }
}

#[cfg(not(target_family = "wasm"))]
#[inline]
fn quant_single_metal_into(
    kind: QuantMatvecKind,
    weights: &[u8],
    rows: usize,
    cols: usize,
    x: &[f32],
    out: &mut Vec<f32>,
) -> bool {
    match kind {
        QuantMatvecKind::Q4K => crate::metal::q4k_matvec_into(weights, x, rows, cols, out),
        QuantMatvecKind::Q6K => crate::metal::q6k_matvec_into(weights, x, rows, cols, out),
        _ => false,
    }
}

#[cfg(not(target_family = "wasm"))]
#[inline]
fn quant_cpu_into(
    kind: QuantMatvecKind,
    weights: &[u8],
    rows: usize,
    cols: usize,
    row_bytes: usize,
    x: &[f32],
    out: &mut Vec<f32>,
) {
    out.resize(rows, 0.0);
    parallel_matvec_u8(kind.matvec_kind(), out, rows, cols, row_bytes, weights, x);
}

#[cfg(not(target_family = "wasm"))]
#[allow(clippy::too_many_arguments)]
fn try_quant_metal3_into(
    a: (QuantMatvecKind, &[u8], usize, usize, usize),
    b: (QuantMatvecKind, &[u8], usize, usize, usize),
    c: (QuantMatvecKind, &[u8], usize, usize, usize),
    x: &[f32],
    out_a: &mut Vec<f32>,
    out_b: &mut Vec<f32>,
    out_c: &mut Vec<f32>,
) -> bool {
    let (kind_a, weights_a, rows_a, cols_a, row_bytes_a) = a;
    let (kind_b, weights_b, rows_b, cols_b, row_bytes_b) = b;
    let (kind_c, weights_c, rows_c, cols_c, row_bytes_c) = c;

    match (kind_a, kind_b, kind_c) {
        (QuantMatvecKind::Q4K, QuantMatvecKind::Q4K, QuantMatvecKind::Q4K) => {
            if crate::metal::q4k_matvec3_into(
                (weights_a, rows_a, cols_a),
                (weights_b, rows_b, cols_b),
                (weights_c, rows_c, cols_c),
                x,
                out_a,
                out_b,
                out_c,
            ) {
                return true;
            }
        }
        (QuantMatvecKind::Q6K, QuantMatvecKind::Q6K, QuantMatvecKind::Q6K) => {
            if crate::metal::q6k_matvec3_into(
                (weights_a, rows_a, cols_a),
                (weights_b, rows_b, cols_b),
                (weights_c, rows_c, cols_c),
                x,
                out_a,
                out_b,
                out_c,
            ) {
                return true;
            }
        }
        (QuantMatvecKind::Q4K, QuantMatvecKind::Q4K, QuantMatvecKind::Q6K) => {
            if crate::metal::q4k_q4k_q6k_matvec3_into(
                (weights_a, rows_a, cols_a),
                (weights_b, rows_b, cols_b),
                (weights_c, rows_c, cols_c),
                x,
                out_a,
                out_b,
                out_c,
            ) {
                return true;
            }
        }
        _ => {}
    }

    let used_a = quant_single_metal_into(kind_a, weights_a, rows_a, cols_a, x, out_a);
    let used_b = quant_single_metal_into(kind_b, weights_b, rows_b, cols_b, x, out_b);
    let used_c = quant_single_metal_into(kind_c, weights_c, rows_c, cols_c, x, out_c);
    if !(used_a || used_b || used_c) {
        return false;
    }

    if !used_a {
        quant_cpu_into(kind_a, weights_a, rows_a, cols_a, row_bytes_a, x, out_a);
    }
    if !used_b {
        quant_cpu_into(kind_b, weights_b, rows_b, cols_b, row_bytes_b, x, out_b);
    }
    if !used_c {
        quant_cpu_into(kind_c, weights_c, rows_c, cols_c, row_bytes_c, x, out_c);
    }
    true
}

#[cfg(not(target_family = "wasm"))]
#[allow(clippy::too_many_arguments)]
fn try_quant_metal2_into(
    a: (QuantMatvecKind, &[u8], usize, usize, usize),
    b: (QuantMatvecKind, &[u8], usize, usize, usize),
    x: &[f32],
    out_a: &mut Vec<f32>,
    out_b: &mut Vec<f32>,
) -> bool {
    let (kind_a, weights_a, rows_a, cols_a, row_bytes_a) = a;
    let (kind_b, weights_b, rows_b, cols_b, row_bytes_b) = b;

    match (kind_a, kind_b) {
        (QuantMatvecKind::Q4K, QuantMatvecKind::Q4K) => {
            if crate::metal::q4k_matvec2_into(
                (weights_a, rows_a, cols_a),
                (weights_b, rows_b, cols_b),
                x,
                out_a,
                out_b,
            ) {
                return true;
            }
        }
        (QuantMatvecKind::Q6K, QuantMatvecKind::Q6K) => {
            if crate::metal::q6k_matvec2_into(
                (weights_a, rows_a, cols_a),
                (weights_b, rows_b, cols_b),
                x,
                out_a,
                out_b,
            ) {
                return true;
            }
        }
        _ => {}
    }

    let used_a = quant_single_metal_into(kind_a, weights_a, rows_a, cols_a, x, out_a);
    let used_b = quant_single_metal_into(kind_b, weights_b, rows_b, cols_b, x, out_b);
    if !(used_a || used_b) {
        return false;
    }

    if !used_a {
        quant_cpu_into(kind_a, weights_a, rows_a, cols_a, row_bytes_a, x, out_a);
    }
    if !used_b {
        quant_cpu_into(kind_b, weights_b, rows_b, cols_b, row_bytes_b, x, out_b);
    }
    true
}

/// Runs an f32 matrix-vector job through the worker pool.
fn parallel_matvec_f32(out: &mut [f32], rows: usize, cols: usize, data: &[f32], x: &[f32]) {
    parallel_matvec(
        MatvecKind::F32,
        out,
        rows,
        cols,
        cols * std::mem::size_of::<f32>(),
        data.as_ptr() as *const u8,
        x.as_ptr(),
    );
}

/// Runs a quantized-byte matrix-vector job through the worker pool.
fn parallel_matvec_u8(
    kind: MatvecKind,
    out: &mut [f32],
    rows: usize,
    cols: usize,
    row_span: usize,
    data: &[u8],
    x: &[f32],
) {
    parallel_matvec(kind, out, rows, cols, row_span, data.as_ptr(), x.as_ptr());
}

/// Dispatches a matrix-vector job, using workers when the shape is large enough.
fn parallel_matvec(
    kind: MatvecKind,
    out: &mut [f32],
    rows: usize,
    cols: usize,
    row_span: usize,
    data: *const u8,
    x: *const f32,
) {
    let threads = num_threads().min(rows);

    if threads <= 1 || rows < threads * 8 {
        for r in 0..rows {
            out[r] = unsafe { dot_row(kind, data, x, r, cols, row_span) };
        }
        return;
    }

    #[cfg(target_family = "wasm")]
    {
        for r in 0..rows {
            out[r] = unsafe { dot_row(kind, data, x, r, cols, row_span) };
        }
        return;
    }

    #[cfg(not(target_family = "wasm"))]
    {
        worker_pool().run(MatvecJob {
            kind,
            data,
            x,
            out: out.as_mut_ptr(),
            rows,
            cols,
            row_span,
            workers: threads,
        });
    }
}

#[inline]
unsafe fn dot_row(
    kind: MatvecKind,
    data: *const u8,
    x: *const f32,
    row: usize,
    cols: usize,
    row_span: usize,
) -> f32 {
    let row_ptr = data.add(row * row_span);
    let x = std::slice::from_raw_parts(x, cols);
    match kind {
        MatvecKind::F32 => {
            let row = std::slice::from_raw_parts(row_ptr as *const f32, cols);
            dot_f32(row, x)
        }
        MatvecKind::Q8_0 => {
            let row = std::slice::from_raw_parts(row_ptr, row_span);
            dot_q8_0_f32(row, x, cols)
        }
        MatvecKind::Q8_1 => {
            let row = std::slice::from_raw_parts(row_ptr, row_span);
            dot_q8_1_f32(row, x, cols)
        }
        MatvecKind::Q4_0 => {
            let row = std::slice::from_raw_parts(row_ptr, row_span);
            dot_q4_0_f32(row, x, cols)
        }
        MatvecKind::Q4_1 => {
            let row = std::slice::from_raw_parts(row_ptr, row_span);
            dot_q4_1_f32(row, x, cols)
        }
        MatvecKind::Q5_0 => {
            let row = std::slice::from_raw_parts(row_ptr, row_span);
            dot_q5_0_f32(row, x, cols)
        }
        MatvecKind::Q5_1 => {
            let row = std::slice::from_raw_parts(row_ptr, row_span);
            dot_q5_1_f32(row, x, cols)
        }
        MatvecKind::Q4K => {
            let row = std::slice::from_raw_parts(row_ptr, row_span);
            dot_q4_k_f32(row, x, cols)
        }
        MatvecKind::Q5K => {
            let row = std::slice::from_raw_parts(row_ptr, row_span);
            dot_q5_k_f32(row, x, cols)
        }
        MatvecKind::Q6K => {
            let row = std::slice::from_raw_parts(row_ptr, row_span);
            dot_q6_k_f32(row, x, cols)
        }
        MatvecKind::Mxfp4 => {
            let row = std::slice::from_raw_parts(row_ptr, row_span);
            dot_mxfp4_f32(row, x, cols)
        }
    }
}

#[cfg(not(target_family = "wasm"))]
#[derive(Clone, Copy)]
struct MatvecJob {
    kind: MatvecKind,
    data: *const u8,
    x: *const f32,
    out: *mut f32,
    rows: usize,
    cols: usize,
    row_span: usize,
    workers: usize,
}

#[cfg(not(target_family = "wasm"))]
unsafe impl Send for MatvecJob {}
#[cfg(not(target_family = "wasm"))]
unsafe impl Sync for MatvecJob {}

#[cfg(not(target_family = "wasm"))]
#[inline]
fn matvec_chunk_count(work_items: usize) -> usize {
    work_items.div_ceil(MATVEC_CHUNK_ROWS)
}

#[cfg(not(target_family = "wasm"))]
#[inline]
fn use_dynamic_chunks(work_items: usize, workers: usize) -> bool {
    matvec_chunk_count(work_items) >= workers.saturating_mul(MIN_DYNAMIC_CHUNKS_PER_WORKER)
}

#[cfg(not(target_family = "wasm"))]
impl MatvecJob {
    unsafe fn run_static_worker(self, worker_idx: usize) {
        let start = self.rows * worker_idx / self.workers;
        let end = self.rows * (worker_idx + 1) / self.workers;
        self.run_rows(start, end);
    }

    unsafe fn run_dynamic_worker(self, worker_idx: usize, current_chunk: &AtomicUsize) {
        let chunks = matvec_chunk_count(self.rows);
        let mut chunk_idx = worker_idx;

        while chunk_idx < chunks {
            let start = chunk_idx * MATVEC_CHUNK_ROWS;
            let end = (start + MATVEC_CHUNK_ROWS).min(self.rows);
            self.run_rows(start, end);
            chunk_idx = current_chunk.fetch_add(1, Ordering::AcqRel);
        }
    }

    unsafe fn run_worker(self, worker_idx: usize, current_chunk: &AtomicUsize) {
        if use_dynamic_chunks(self.rows, self.workers) {
            self.run_dynamic_worker(worker_idx, current_chunk);
        } else {
            self.run_static_worker(worker_idx);
        }
    }

    unsafe fn run_rows(self, start: usize, end: usize) {
        for row in start..end {
            *self.out.add(row) =
                dot_row(self.kind, self.data, self.x, row, self.cols, self.row_span);
        }
    }
}

#[cfg(not(target_family = "wasm"))]
#[derive(Clone, Copy)]
struct Q4KMatvec3Job {
    kind_a: MatvecKind,
    kind_b: MatvecKind,
    kind_c: MatvecKind,
    a_data: *const u8,
    b_data: *const u8,
    c_data: *const u8,
    x: *const f32,
    out_a: *mut f32,
    out_b: *mut f32,
    out_c: *mut f32,
    rows_a: usize,
    rows_b: usize,
    rows_c: usize,
    cols: usize,
    row_span_a: usize,
    row_span_b: usize,
    row_span_c: usize,
    workers: usize,
}

#[cfg(not(target_family = "wasm"))]
unsafe impl Send for Q4KMatvec3Job {}
#[cfg(not(target_family = "wasm"))]
unsafe impl Sync for Q4KMatvec3Job {}

#[cfg(not(target_family = "wasm"))]
impl Q4KMatvec3Job {
    #[inline]
    /// Returns the amount of row work represented by this job.
    fn work_items(self) -> usize {
        self.rows_a + self.rows_b + self.rows_c
    }

    unsafe fn run_static_worker(self, worker_idx: usize) {
        let total = self.work_items();
        let start = total * worker_idx / self.workers;
        let end = total * (worker_idx + 1) / self.workers;
        self.run_items(start, end);
    }

    unsafe fn run_dynamic_worker(self, worker_idx: usize, current_chunk: &AtomicUsize) {
        let total = self.work_items();
        let chunks = matvec_chunk_count(total);
        let mut chunk_idx = worker_idx;

        while chunk_idx < chunks {
            let start = chunk_idx * MATVEC_CHUNK_ROWS;
            let end = (start + MATVEC_CHUNK_ROWS).min(total);
            self.run_items(start, end);
            chunk_idx = current_chunk.fetch_add(1, Ordering::AcqRel);
        }
    }

    unsafe fn run_worker(self, worker_idx: usize, current_chunk: &AtomicUsize) {
        if use_dynamic_chunks(self.work_items(), self.workers) {
            self.run_dynamic_worker(worker_idx, current_chunk);
        } else {
            self.run_static_worker(worker_idx);
        }
    }

    unsafe fn run_items(self, start: usize, end: usize) {
        let (a_start, a_end) = clipped_range(start, end, 0, self.rows_a);
        q4k_matvec3_rows(
            self.kind_a,
            self.a_data,
            self.x,
            self.out_a,
            self.cols,
            self.row_span_a,
            a_start,
            a_end,
        );

        let b_offset = self.rows_a;
        let (b_start, b_end) = clipped_range(start, end, b_offset, self.rows_b);
        q4k_matvec3_rows(
            self.kind_b,
            self.b_data,
            self.x,
            self.out_b,
            self.cols,
            self.row_span_b,
            b_start,
            b_end,
        );

        let c_offset = self.rows_a + self.rows_b;
        let (c_start, c_end) = clipped_range(start, end, c_offset, self.rows_c);
        q4k_matvec3_rows(
            self.kind_c,
            self.c_data,
            self.x,
            self.out_c,
            self.cols,
            self.row_span_c,
            c_start,
            c_end,
        );
    }
}

#[cfg(not(target_family = "wasm"))]
#[inline]
#[allow(clippy::too_many_arguments)]
/// Clips a worker row range to an output slice range.
fn clipped_range(start: usize, end: usize, offset: usize, len: usize) -> (usize, usize) {
    let local_start = start.saturating_sub(offset).min(len);
    let local_end = end.saturating_sub(offset).min(len);
    if local_end > local_start {
        (local_start, local_end)
    } else {
        (0, 0)
    }
}

#[cfg(not(target_family = "wasm"))]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn q4k_matvec3_rows(
    kind: MatvecKind,
    data: *const u8,
    x: *const f32,
    out: *mut f32,
    cols: usize,
    row_span: usize,
    start: usize,
    end: usize,
) {
    for row in start..end {
        *out.add(row) = dot_row(kind, data, x, row, cols, row_span);
    }
}

#[cfg(not(target_family = "wasm"))]
#[derive(Clone, Copy)]
enum WorkerJob {
    Matvec(MatvecJob),
    Q4KMatvec3(Q4KMatvec3Job),
}

#[cfg(not(target_family = "wasm"))]
impl WorkerJob {
    #[inline]
    /// Returns how many workers should execute this job.
    fn workers(self) -> usize {
        match self {
            WorkerJob::Matvec(job) => job.workers,
            WorkerJob::Q4KMatvec3(job) => job.workers,
        }
    }

    #[inline]
    unsafe fn run_worker(self, worker_idx: usize, current_chunk: &AtomicUsize) {
        match self {
            WorkerJob::Matvec(job) => job.run_worker(worker_idx, current_chunk),
            WorkerJob::Q4KMatvec3(job) => job.run_worker(worker_idx, current_chunk),
        }
    }
}

#[cfg(not(target_family = "wasm"))]
struct WorkerState {
    job: Option<WorkerJob>,
    job_id: u64,
}

#[cfg(not(target_family = "wasm"))]
struct WorkerPool {
    state: Mutex<WorkerState>,
    work_available: Condvar,
    completed: CachePadded<AtomicUsize>,
    current_chunk: CachePadded<AtomicUsize>,
    published_job_id: CachePadded<AtomicU64>,
    max_workers: usize,
}

#[cfg(not(target_family = "wasm"))]
impl WorkerPool {
    /// Spawns a fixed set of worker threads for shared matrix-vector jobs.
    fn new(max_workers: usize) -> Arc<Self> {
        let pool = Arc::new(Self {
            state: Mutex::new(WorkerState {
                job: None,
                job_id: 0,
            }),
            work_available: Condvar::new(),
            completed: CachePadded(AtomicUsize::new(0)),
            current_chunk: CachePadded(AtomicUsize::new(0)),
            published_job_id: CachePadded(AtomicU64::new(0)),
            max_workers,
        });

        for worker_idx in 0..max_workers {
            let pool = Arc::clone(&pool);
            thread::Builder::new()
                .name(format!("rusty-llm-matvec-{}", worker_idx))
                .spawn(move || worker_loop(pool, worker_idx))
                .expect("failed to start matvec worker");
        }

        pool
    }

    /// Executes a queued worker-pool job and waits for completion.
    fn run(&self, mut job: MatvecJob) {
        job.workers = job.workers.min(self.max_workers).min(job.rows).max(1);
        if job.workers <= 1 || job.rows < job.workers * 8 {
            for row in 0..job.rows {
                unsafe {
                    *job.out.add(row) =
                        dot_row(job.kind, job.data, job.x, row, job.cols, job.row_span);
                }
            }
            return;
        }

        self.run_job(WorkerJob::Matvec(job), job.workers);
    }

    /// Executes a fused Q4_K triple-matvec job and waits for completion.
    fn run_q4k_matvec3(&self, mut job: Q4KMatvec3Job) {
        let rows = job.work_items();
        job.workers = job.workers.min(self.max_workers).min(rows).max(1);
        if job.workers <= 1 || rows < job.workers * 8 {
            job.workers = 1;
            unsafe {
                job.run_static_worker(0);
            }
            return;
        }

        self.run_job(WorkerJob::Q4KMatvec3(job), job.workers);
    }

    /// Publishes a worker job and waits for all selected workers to finish it.
    fn run_job(&self, job: WorkerJob, workers: usize) {
        // Fast spin-wait if previous job is finishing (usually very quick)
        loop {
            let mut state = self.state.lock().expect("worker pool mutex poisoned");
            if state.job.is_none() {
                state.job_id = state.job_id.wrapping_add(1);
                self.completed.0.store(0, Ordering::Release);
                self.current_chunk.0.store(workers, Ordering::Release);
                state.job = Some(job);
                self.published_job_id
                    .0
                    .store(state.job_id, Ordering::Release);
                self.work_available.notify_all();
                break;
            }
            drop(state);
            std::hint::spin_loop();
        }

        // Wait for workers to finish current job using spin-loop
        // This is a micro-job, so sleeping via Condvar is too slow for LLM latencies.
        let mut spins = 0;
        loop {
            if self.completed.0.load(Ordering::Acquire) == workers {
                break;
            }

            if spins < 10000 {
                std::hint::spin_loop();
                spins += 1;
            } else {
                thread::yield_now();
            }
        }

        // Final cleanup
        let mut state = self.state.lock().expect("worker pool mutex poisoned");
        state.job = None;
    }
}

#[cfg(not(target_family = "wasm"))]
/// Processes worker-pool jobs on one background thread.
fn worker_loop(pool: Arc<WorkerPool>, worker_idx: usize) {
    let _ = pin_current_thread(worker_idx);
    let mut last_job_id = 0u64;
    loop {
        let poll_spins = worker_poll_spins();
        for _ in 0..poll_spins {
            if pool.published_job_id.0.load(Ordering::Acquire) != last_job_id {
                break;
            }
            std::hint::spin_loop();
        }

        let job = {
            let mut state = pool.state.lock().expect("worker pool mutex poisoned");
            while state.job_id == last_job_id || state.job.is_none() {
                state = pool
                    .work_available
                    .wait(state)
                    .expect("worker pool mutex poisoned");
            }
            last_job_id = state.job_id;
            state.job.expect("job should be available")
        };

        if worker_idx < job.workers() {
            unsafe {
                job.run_worker(worker_idx, &pool.current_chunk.0);
            }
            pool.completed.0.fetch_add(1, Ordering::Release);
        }
    }
}

#[cfg(not(target_family = "wasm"))]
/// Returns the lazily initialized global worker pool.
fn worker_pool() -> &'static WorkerPool {
    static POOL: OnceLock<Arc<WorkerPool>> = OnceLock::new();
    POOL.get_or_init(|| {
        let workers = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .max(1);
        WorkerPool::new(workers)
    })
}

#[inline(always)]
/// Converts half precision to f32 using portable scalar code.
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
/// Computes the dot product of two float vectors.
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { dot_f32_neon(a, b) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
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

/// Computes `out[i] += alpha * x[i]`.
#[inline]
/// Adds a scaled vector into another vector in place.
pub fn axpy_f32(out: &mut [f32], alpha: f32, x: &[f32]) {
    debug_assert_eq!(out.len(), x.len());
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { axpy_f32_neon(out, alpha, x) }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            unsafe { axpy_f32_avx2(out, alpha, x) }
        } else {
            axpy_f32_scalar(out, alpha, x)
        }
        return;
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        axpy_f32_scalar(out, alpha, x)
    }
}

/// Computes `out[i] *= scale`.
#[inline]
/// Scales a float vector in place.
pub fn scale_f32(out: &mut [f32], scale: f32) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { scale_f32_neon(out, scale) }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            unsafe { scale_f32_avx2(out, scale) }
        } else {
            scale_f32_scalar(out, scale)
        }
        return;
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scale_f32_scalar(out, scale)
    }
}

/// Computes `out[i] = out[i] * scale + add[i]`.
#[inline]
/// Scales a vector and adds another vector in place.
pub fn scale_add_f32(out: &mut [f32], scale: f32, add: &[f32]) {
    debug_assert_eq!(out.len(), add.len());
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { scale_add_f32_neon(out, scale, add) }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            unsafe { scale_add_f32_avx2(out, scale, add) }
        } else {
            scale_add_f32_scalar(out, scale, add)
        }
        return;
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scale_add_f32_scalar(out, scale, add)
    }
}

#[inline]
#[allow(dead_code)]
/// Scalar fallback for AXPY.
fn axpy_f32_scalar(out: &mut [f32], alpha: f32, x: &[f32]) {
    for (o, xi) in out.iter_mut().zip(x.iter()) {
        *o += alpha * *xi;
    }
}

#[inline]
#[allow(dead_code)]
/// Scalar fallback for vector scaling.
fn scale_f32_scalar(out: &mut [f32], scale: f32) {
    for o in out.iter_mut() {
        *o *= scale;
    }
}

#[inline]
#[allow(dead_code)]
/// Scalar fallback for fused scale-and-add.
fn scale_add_f32_scalar(out: &mut [f32], scale: f32, add: &[f32]) {
    for (o, a) in out.iter_mut().zip(add.iter()) {
        *o = *o * scale + *a;
    }
}

/// Fused Q8_0 dot product: quantized_weight · f32_input
/// `qdata` is raw Q8_0 blocks, `x` is f32 input vector
/// `n` is the number of elements (must be multiple of 32)
#[inline]
/// Computes a Q8_0 row dot product against an f32 vector.
pub fn dot_q8_0_f32(qdata: &[u8], x: &[f32], n: usize) -> f32 {
    debug_assert!(n % 32 == 0);
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { dot_q8_0_f32_neon(qdata, x, n) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
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

/// Computes a Q8_1 row dot product against an f32 vector.
#[inline]
pub fn dot_q8_1_f32(qdata: &[u8], x: &[f32], n: usize) -> f32 {
    debug_assert!(n % 32 == 0);
    dot_q8_1_f32_scalar(qdata, x, n)
}

/// Fused Q4_0 dot product
#[inline]
/// Computes a Q4_0 row dot product against an f32 vector.
pub fn dot_q4_0_f32(qdata: &[u8], x: &[f32], n: usize) -> f32 {
    debug_assert!(n % 32 == 0);
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { dot_q4_0_f32_neon(qdata, x, n) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
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

/// Computes a Q4_1 row dot product against an f32 vector.
#[inline]
pub fn dot_q4_1_f32(qdata: &[u8], x: &[f32], n: usize) -> f32 {
    debug_assert!(n % 32 == 0);
    dot_q4_1_f32_scalar(qdata, x, n)
}

/// Computes a Q5_0 row dot product against an f32 vector.
#[inline]
pub fn dot_q5_0_f32(qdata: &[u8], x: &[f32], n: usize) -> f32 {
    debug_assert!(n % 32 == 0);
    dot_q5_0_f32_scalar(qdata, x, n)
}

/// Computes a Q5_1 row dot product against an f32 vector.
#[inline]
pub fn dot_q5_1_f32(qdata: &[u8], x: &[f32], n: usize) -> f32 {
    debug_assert!(n % 32 == 0);
    dot_q5_1_f32_scalar(qdata, x, n)
}

/// Fused Q4_K dot product
#[inline]
/// Computes a Q4_K row dot product against an f32 vector.
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

/// Computes a Q5_K row dot product against an f32 vector.
#[inline]
pub fn dot_q5_k_f32(qdata: &[u8], x: &[f32], n: usize) -> f32 {
    debug_assert!(n % 256 == 0);
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { dot_q5_k_f32_neon(qdata, x, n) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        dot_q5_k_f32_scalar(qdata, x, n)
    }
}

/// Fused Q6_K dot product
#[inline]
/// Computes a Q6_K row dot product against an f32 vector.
pub fn dot_q6_k_f32(qdata: &[u8], x: &[f32], n: usize) -> f32 {
    debug_assert!(n % 256 == 0);
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            return dot_q6_k_f32_neon(qdata, x, n);
        }
    }
    #[allow(unreachable_code)]
    dot_q6_k_f32_scalar(qdata, x, n)
}

#[inline]
/// Computes an MXFP4 row dot product against an f32 vector.
pub fn dot_mxfp4_f32(qdata: &[u8], x: &[f32], n: usize) -> f32 {
    debug_assert!(n % 32 == 0);
    dot_mxfp4_f32_scalar(qdata, x, n)
}

// ─── Simple matvec implementations ──────────────────────────────────────────

pub fn matvec_f32(weight: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows];
    parallel_matvec_f32(&mut out, rows, cols, weight, x);
    out
}

/// Runs a Q8_0 matrix-vector multiply and returns a new vector.
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
    parallel_matvec_u8(
        MatvecKind::Q8_0,
        &mut out,
        rows,
        cols,
        row_bytes,
        qweight,
        x,
    );
    out
}

/// Runs a Q8_1 matrix-vector multiply and returns a new vector.
pub fn matvec_q8_1(qweight: &[u8], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let row_bytes = (cols / 32) * 36;
    let needed = row_bytes
        .checked_mul(rows)
        .expect("matvec_q8_1: rows*row_bytes overflow");
    assert!(
        qweight.len() >= needed,
        "matvec_q8_1: buffer too small (need {}, got {})",
        needed,
        qweight.len()
    );
    let mut out = vec![0.0f32; rows];
    parallel_matvec_u8(
        MatvecKind::Q8_1,
        &mut out,
        rows,
        cols,
        row_bytes,
        qweight,
        x,
    );
    out
}

/// Runs a Q4_0 matrix-vector multiply and returns a new vector.
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
    parallel_matvec_u8(
        MatvecKind::Q4_0,
        &mut out,
        rows,
        cols,
        row_bytes,
        qweight,
        x,
    );
    out
}

/// Runs a Q4_1 matrix-vector multiply and returns a new vector.
pub fn matvec_q4_1(qweight: &[u8], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let row_bytes = (cols / 32) * 20;
    let needed = row_bytes
        .checked_mul(rows)
        .expect("matvec_q4_1: rows*row_bytes overflow");
    assert!(
        qweight.len() >= needed,
        "matvec_q4_1: buffer too small (need {}, got {})",
        needed,
        qweight.len()
    );
    let mut out = vec![0.0f32; rows];
    parallel_matvec_u8(
        MatvecKind::Q4_1,
        &mut out,
        rows,
        cols,
        row_bytes,
        qweight,
        x,
    );
    out
}

/// Runs a Q5_0 matrix-vector multiply and returns a new vector.
pub fn matvec_q5_0(qweight: &[u8], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let row_bytes = (cols / 32) * 22;
    let needed = row_bytes
        .checked_mul(rows)
        .expect("matvec_q5_0: rows*row_bytes overflow");
    assert!(
        qweight.len() >= needed,
        "matvec_q5_0: buffer too small (need {}, got {})",
        needed,
        qweight.len()
    );
    let mut out = vec![0.0f32; rows];
    parallel_matvec_u8(
        MatvecKind::Q5_0,
        &mut out,
        rows,
        cols,
        row_bytes,
        qweight,
        x,
    );
    out
}

/// Runs a Q5_1 matrix-vector multiply and returns a new vector.
pub fn matvec_q5_1(qweight: &[u8], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let row_bytes = (cols / 32) * 24;
    let needed = row_bytes
        .checked_mul(rows)
        .expect("matvec_q5_1: rows*row_bytes overflow");
    assert!(
        qweight.len() >= needed,
        "matvec_q5_1: buffer too small (need {}, got {})",
        needed,
        qweight.len()
    );
    let mut out = vec![0.0f32; rows];
    parallel_matvec_u8(
        MatvecKind::Q5_1,
        &mut out,
        rows,
        cols,
        row_bytes,
        qweight,
        x,
    );
    out
}

/// Runs a Q4_K matrix-vector multiply and returns a new vector.
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
    #[cfg(not(target_family = "wasm"))]
    if crate::metal::q4k_matvec_into(qweight, x, rows, cols, &mut out) {
        return out;
    }
    parallel_matvec_u8(MatvecKind::Q4K, &mut out, rows, cols, row_bytes, qweight, x);
    out
}

/// Runs a Q5_K matrix-vector multiply and returns a new vector.
pub fn matvec_q5_k(qweight: &[u8], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let row_bytes = (cols / 256) * 176;
    let needed = row_bytes
        .checked_mul(rows)
        .expect("matvec_q5_k: rows*row_bytes overflow");
    assert!(
        qweight.len() >= needed,
        "matvec_q5_k: buffer too small (need {}, got {})",
        needed,
        qweight.len()
    );
    let mut out = vec![0.0f32; rows];
    parallel_matvec_u8(MatvecKind::Q5K, &mut out, rows, cols, row_bytes, qweight, x);
    out
}

/// Runs a Q6_K matrix-vector multiply and returns a new vector.
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
    #[cfg(not(target_family = "wasm"))]
    if crate::metal::q6k_matvec_into(qweight, x, rows, cols, &mut out) {
        return out;
    }
    parallel_matvec_u8(MatvecKind::Q6K, &mut out, rows, cols, row_bytes, qweight, x);
    out
}

/// Runs an MXFP4 matrix-vector multiply and returns a new vector.
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
    parallel_matvec_u8(
        MatvecKind::Mxfp4,
        &mut out,
        rows,
        cols,
        row_bytes,
        qweight,
        x,
    );
    out
}

// ─── In-place matvec variants (write into caller buffer, no alloc) ───────────

pub fn matvec_f32_into(weight: &[f32], x: &[f32], rows: usize, cols: usize, out: &mut Vec<f32>) {
    out.resize(rows, 0.0);
    parallel_matvec_f32(out, rows, cols, weight, x);
}

/// Runs a Q8_0 matrix-vector multiply into a reusable output buffer.
pub fn matvec_q8_0_into(qweight: &[u8], x: &[f32], rows: usize, cols: usize, out: &mut Vec<f32>) {
    let row_bytes = (cols / 32) * 34;
    out.resize(rows, 0.0);
    parallel_matvec_u8(MatvecKind::Q8_0, out, rows, cols, row_bytes, qweight, x);
}

/// Runs a Q8_1 matrix-vector multiply into a reusable output buffer.
pub fn matvec_q8_1_into(qweight: &[u8], x: &[f32], rows: usize, cols: usize, out: &mut Vec<f32>) {
    let row_bytes = (cols / 32) * 36;
    out.resize(rows, 0.0);
    parallel_matvec_u8(MatvecKind::Q8_1, out, rows, cols, row_bytes, qweight, x);
}

/// Runs a Q4_0 matrix-vector multiply into a reusable output buffer.
pub fn matvec_q4_0_into(qweight: &[u8], x: &[f32], rows: usize, cols: usize, out: &mut Vec<f32>) {
    let row_bytes = (cols / 32) * 18;
    out.resize(rows, 0.0);
    parallel_matvec_u8(MatvecKind::Q4_0, out, rows, cols, row_bytes, qweight, x);
}

/// Runs a Q4_1 matrix-vector multiply into a reusable output buffer.
pub fn matvec_q4_1_into(qweight: &[u8], x: &[f32], rows: usize, cols: usize, out: &mut Vec<f32>) {
    let row_bytes = (cols / 32) * 20;
    out.resize(rows, 0.0);
    parallel_matvec_u8(MatvecKind::Q4_1, out, rows, cols, row_bytes, qweight, x);
}

/// Runs a Q5_0 matrix-vector multiply into a reusable output buffer.
pub fn matvec_q5_0_into(qweight: &[u8], x: &[f32], rows: usize, cols: usize, out: &mut Vec<f32>) {
    let row_bytes = (cols / 32) * 22;
    out.resize(rows, 0.0);
    parallel_matvec_u8(MatvecKind::Q5_0, out, rows, cols, row_bytes, qweight, x);
}

/// Runs a Q5_1 matrix-vector multiply into a reusable output buffer.
pub fn matvec_q5_1_into(qweight: &[u8], x: &[f32], rows: usize, cols: usize, out: &mut Vec<f32>) {
    let row_bytes = (cols / 32) * 24;
    out.resize(rows, 0.0);
    parallel_matvec_u8(MatvecKind::Q5_1, out, rows, cols, row_bytes, qweight, x);
}

/// Runs a Q4_K matrix-vector multiply into a reusable output buffer.
pub fn matvec_q4_k_into(qweight: &[u8], x: &[f32], rows: usize, cols: usize, out: &mut Vec<f32>) {
    let row_bytes = (cols / 256) * 144;
    out.resize(rows, 0.0);
    #[cfg(not(target_family = "wasm"))]
    if crate::metal::q4k_matvec_into(qweight, x, rows, cols, out) {
        return;
    }
    parallel_matvec_u8(MatvecKind::Q4K, out, rows, cols, row_bytes, qweight, x);
}

/// Runs a Q5_K matrix-vector multiply into a reusable output buffer.
pub fn matvec_q5_k_into(qweight: &[u8], x: &[f32], rows: usize, cols: usize, out: &mut Vec<f32>) {
    let row_bytes = (cols / 256) * 176;
    out.resize(rows, 0.0);
    parallel_matvec_u8(MatvecKind::Q5K, out, rows, cols, row_bytes, qweight, x);
}

/// Runs three Q4_K projections against the same input vector.
pub fn matvec_q4_k3_into(
    a: (&[u8], usize, usize),
    b: (&[u8], usize, usize),
    c: (&[u8], usize, usize),
    x: &[f32],
    out_a: &mut Vec<f32>,
    out_b: &mut Vec<f32>,
    out_c: &mut Vec<f32>,
) -> bool {
    let (weights_a, rows_a, cols_a) = a;
    let (weights_b, rows_b, cols_b) = b;
    let (weights_c, rows_c, cols_c) = c;
    if cols_a == 0 || cols_a % 256 != 0 || cols_a != cols_b || cols_a != cols_c || cols_a != x.len()
    {
        return false;
    }

    let row_bytes = (cols_a / 256) * 144;
    let needed_a = match row_bytes.checked_mul(rows_a) {
        Some(v) => v,
        None => return false,
    };
    let needed_b = match row_bytes.checked_mul(rows_b) {
        Some(v) => v,
        None => return false,
    };
    let needed_c = match row_bytes.checked_mul(rows_c) {
        Some(v) => v,
        None => return false,
    };
    if weights_a.len() < needed_a || weights_b.len() < needed_b || weights_c.len() < needed_c {
        return false;
    }

    out_a.resize(rows_a, 0.0);
    out_b.resize(rows_b, 0.0);
    out_c.resize(rows_c, 0.0);

    #[cfg(not(target_family = "wasm"))]
    if crate::metal::q4k_matvec3_into(
        (weights_a, rows_a, cols_a),
        (weights_b, rows_b, cols_b),
        (weights_c, rows_c, cols_c),
        x,
        out_a,
        out_b,
        out_c,
    ) {
        return true;
    }

    #[cfg(target_family = "wasm")]
    {
        for row in 0..rows_a {
            out_a[row] = dot_q4_k_f32(
                &weights_a[row * row_bytes..(row + 1) * row_bytes],
                x,
                cols_a,
            );
        }
        for row in 0..rows_b {
            out_b[row] = dot_q4_k_f32(
                &weights_b[row * row_bytes..(row + 1) * row_bytes],
                x,
                cols_a,
            );
        }
        for row in 0..rows_c {
            out_c[row] = dot_q4_k_f32(
                &weights_c[row * row_bytes..(row + 1) * row_bytes],
                x,
                cols_a,
            );
        }
        return true;
    }

    #[cfg(not(target_family = "wasm"))]
    {
        let total_rows = rows_a + rows_b + rows_c;
        if total_rows == 0 {
            return true;
        }
        let workers = num_threads().min(total_rows);
        worker_pool().run_q4k_matvec3(Q4KMatvec3Job {
            kind_a: MatvecKind::Q4K,
            kind_b: MatvecKind::Q4K,
            kind_c: MatvecKind::Q4K,
            a_data: weights_a.as_ptr(),
            b_data: weights_b.as_ptr(),
            c_data: weights_c.as_ptr(),
            x: x.as_ptr(),
            out_a: out_a.as_mut_ptr(),
            out_b: out_b.as_mut_ptr(),
            out_c: out_c.as_mut_ptr(),
            rows_a,
            rows_b,
            rows_c,
            cols: cols_a,
            row_span_a: row_bytes,
            row_span_b: row_bytes,
            row_span_c: row_bytes,
            workers,
        });
        true
    }
}

/// Runs two Q4_K projections against the same input vector.
pub fn matvec_q4_k2_into(
    a: (&[u8], usize, usize),
    b: (&[u8], usize, usize),
    x: &[f32],
    out_a: &mut Vec<f32>,
    out_b: &mut Vec<f32>,
) -> bool {
    let (weights_a, rows_a, cols_a) = a;
    let (weights_b, rows_b, cols_b) = b;
    if cols_a == 0 || cols_a % 256 != 0 || cols_a != cols_b || cols_a != x.len() {
        return false;
    }

    let row_bytes = (cols_a / 256) * 144;
    let needed_a = match row_bytes.checked_mul(rows_a) {
        Some(v) => v,
        None => return false,
    };
    let needed_b = match row_bytes.checked_mul(rows_b) {
        Some(v) => v,
        None => return false,
    };
    if weights_a.len() < needed_a || weights_b.len() < needed_b {
        return false;
    }

    out_a.resize(rows_a, 0.0);
    out_b.resize(rows_b, 0.0);

    #[cfg(not(target_family = "wasm"))]
    if crate::metal::q4k_matvec2_into(
        (weights_a, rows_a, cols_a),
        (weights_b, rows_b, cols_b),
        x,
        out_a,
        out_b,
    ) {
        return true;
    }

    #[cfg(target_family = "wasm")]
    {
        for row in 0..rows_a {
            out_a[row] = dot_q4_k_f32(
                &weights_a[row * row_bytes..(row + 1) * row_bytes],
                x,
                cols_a,
            );
        }
        for row in 0..rows_b {
            out_b[row] = dot_q4_k_f32(
                &weights_b[row * row_bytes..(row + 1) * row_bytes],
                x,
                cols_a,
            );
        }
        return true;
    }

    #[cfg(not(target_family = "wasm"))]
    {
        let total_rows = rows_a + rows_b;
        if total_rows == 0 {
            return true;
        }
        let workers = num_threads().min(total_rows);
        worker_pool().run_q4k_matvec3(Q4KMatvec3Job {
            kind_a: MatvecKind::Q4K,
            kind_b: MatvecKind::Q4K,
            kind_c: MatvecKind::Q4K,
            a_data: weights_a.as_ptr(),
            b_data: weights_b.as_ptr(),
            c_data: weights_b.as_ptr(),
            x: x.as_ptr(),
            out_a: out_a.as_mut_ptr(),
            out_b: out_b.as_mut_ptr(),
            out_c: out_b.as_mut_ptr(),
            rows_a,
            rows_b,
            rows_c: 0,
            cols: cols_a,
            row_span_a: row_bytes,
            row_span_b: row_bytes,
            row_span_c: row_bytes,
            workers,
        });
        true
    }
}

#[allow(clippy::too_many_arguments)]
pub fn matvec_kquant3_into(
    a: (KQuantMatvecKind, &[u8], usize, usize),
    b: (KQuantMatvecKind, &[u8], usize, usize),
    c: (KQuantMatvecKind, &[u8], usize, usize),
    x: &[f32],
    out_a: &mut Vec<f32>,
    out_b: &mut Vec<f32>,
    out_c: &mut Vec<f32>,
) -> bool {
    let (kind_a, weights_a, rows_a, cols_a) = a;
    let (kind_b, weights_b, rows_b, cols_b) = b;
    let (kind_c, weights_c, rows_c, cols_c) = c;
    if cols_a == 0 || cols_a % 256 != 0 || cols_a != cols_b || cols_a != cols_c || cols_a != x.len()
    {
        return false;
    }
    let row_bytes_a = kind_a.row_bytes(cols_a);
    let row_bytes_b = kind_b.row_bytes(cols_b);
    let row_bytes_c = kind_c.row_bytes(cols_c);
    let needed_a = match row_bytes_a.checked_mul(rows_a) {
        Some(v) => v,
        None => return false,
    };
    let needed_b = match row_bytes_b.checked_mul(rows_b) {
        Some(v) => v,
        None => return false,
    };
    let needed_c = match row_bytes_c.checked_mul(rows_c) {
        Some(v) => v,
        None => return false,
    };
    if weights_a.len() < needed_a || weights_b.len() < needed_b || weights_c.len() < needed_c {
        return false;
    }

    out_a.resize(rows_a, 0.0);
    out_b.resize(rows_b, 0.0);
    out_c.resize(rows_c, 0.0);

    #[cfg(not(target_family = "wasm"))]
    if try_quant_metal3_into(
        (
            quant_kind_from_kquant(kind_a),
            weights_a,
            rows_a,
            cols_a,
            row_bytes_a,
        ),
        (
            quant_kind_from_kquant(kind_b),
            weights_b,
            rows_b,
            cols_b,
            row_bytes_b,
        ),
        (
            quant_kind_from_kquant(kind_c),
            weights_c,
            rows_c,
            cols_c,
            row_bytes_c,
        ),
        x,
        out_a,
        out_b,
        out_c,
    ) {
        return true;
    }

    #[cfg(target_family = "wasm")]
    {
        let kind_a = kind_a.matvec_kind();
        let kind_b = kind_b.matvec_kind();
        let kind_c = kind_c.matvec_kind();
        for row in 0..rows_a {
            out_a[row] = dot_row_from_kind(
                kind_a,
                &weights_a[row * row_bytes_a..(row + 1) * row_bytes_a],
                x,
                cols_a,
            );
        }
        for row in 0..rows_b {
            out_b[row] = dot_row_from_kind(
                kind_b,
                &weights_b[row * row_bytes_b..(row + 1) * row_bytes_b],
                x,
                cols_a,
            );
        }
        for row in 0..rows_c {
            out_c[row] = dot_row_from_kind(
                kind_c,
                &weights_c[row * row_bytes_c..(row + 1) * row_bytes_c],
                x,
                cols_a,
            );
        }
        return true;
    }

    #[cfg(not(target_family = "wasm"))]
    {
        let total_rows = rows_a + rows_b + rows_c;
        if total_rows == 0 {
            return true;
        }
        let workers = num_threads().min(total_rows);
        worker_pool().run_q4k_matvec3(Q4KMatvec3Job {
            kind_a: kind_a.matvec_kind(),
            kind_b: kind_b.matvec_kind(),
            kind_c: kind_c.matvec_kind(),
            a_data: weights_a.as_ptr(),
            b_data: weights_b.as_ptr(),
            c_data: weights_c.as_ptr(),
            x: x.as_ptr(),
            out_a: out_a.as_mut_ptr(),
            out_b: out_b.as_mut_ptr(),
            out_c: out_c.as_mut_ptr(),
            rows_a,
            rows_b,
            rows_c,
            cols: cols_a,
            row_span_a: row_bytes_a,
            row_span_b: row_bytes_b,
            row_span_c: row_bytes_c,
            workers,
        });
        true
    }
}

#[allow(clippy::too_many_arguments)]
pub fn matvec_quant3_into(
    a: (QuantMatvecKind, &[u8], usize, usize),
    b: (QuantMatvecKind, &[u8], usize, usize),
    c: (QuantMatvecKind, &[u8], usize, usize),
    x: &[f32],
    out_a: &mut Vec<f32>,
    out_b: &mut Vec<f32>,
    out_c: &mut Vec<f32>,
) -> bool {
    let (kind_a, weights_a, rows_a, cols_a) = a;
    let (kind_b, weights_b, rows_b, cols_b) = b;
    let (kind_c, weights_c, rows_c, cols_c) = c;
    if cols_a != cols_b || cols_a != cols_c || cols_a != x.len() {
        return false;
    }
    let Some(row_bytes_a) = kind_a.row_bytes(cols_a) else {
        return false;
    };
    let Some(row_bytes_b) = kind_b.row_bytes(cols_b) else {
        return false;
    };
    let Some(row_bytes_c) = kind_c.row_bytes(cols_c) else {
        return false;
    };
    let needed_a = match row_bytes_a.checked_mul(rows_a) {
        Some(v) => v,
        None => return false,
    };
    let needed_b = match row_bytes_b.checked_mul(rows_b) {
        Some(v) => v,
        None => return false,
    };
    let needed_c = match row_bytes_c.checked_mul(rows_c) {
        Some(v) => v,
        None => return false,
    };
    if weights_a.len() < needed_a || weights_b.len() < needed_b || weights_c.len() < needed_c {
        return false;
    }

    out_a.resize(rows_a, 0.0);
    out_b.resize(rows_b, 0.0);
    out_c.resize(rows_c, 0.0);

    #[cfg(not(target_family = "wasm"))]
    if try_quant_metal3_into(
        (kind_a, weights_a, rows_a, cols_a, row_bytes_a),
        (kind_b, weights_b, rows_b, cols_b, row_bytes_b),
        (kind_c, weights_c, rows_c, cols_c, row_bytes_c),
        x,
        out_a,
        out_b,
        out_c,
    ) {
        return true;
    }
    if is_mixed_kquant3(kind_a, kind_b, kind_c) {
        return false;
    }

    #[cfg(target_family = "wasm")]
    {
        let kind_a = kind_a.matvec_kind();
        let kind_b = kind_b.matvec_kind();
        let kind_c = kind_c.matvec_kind();
        for row in 0..rows_a {
            out_a[row] = dot_row_from_kind(
                kind_a,
                &weights_a[row * row_bytes_a..(row + 1) * row_bytes_a],
                x,
                cols_a,
            );
        }
        for row in 0..rows_b {
            out_b[row] = dot_row_from_kind(
                kind_b,
                &weights_b[row * row_bytes_b..(row + 1) * row_bytes_b],
                x,
                cols_a,
            );
        }
        for row in 0..rows_c {
            out_c[row] = dot_row_from_kind(
                kind_c,
                &weights_c[row * row_bytes_c..(row + 1) * row_bytes_c],
                x,
                cols_a,
            );
        }
        return true;
    }

    #[cfg(not(target_family = "wasm"))]
    {
        let total_rows = rows_a + rows_b + rows_c;
        if total_rows == 0 {
            return true;
        }
        let workers = num_threads().min(total_rows);
        worker_pool().run_q4k_matvec3(Q4KMatvec3Job {
            kind_a: kind_a.matvec_kind(),
            kind_b: kind_b.matvec_kind(),
            kind_c: kind_c.matvec_kind(),
            a_data: weights_a.as_ptr(),
            b_data: weights_b.as_ptr(),
            c_data: weights_c.as_ptr(),
            x: x.as_ptr(),
            out_a: out_a.as_mut_ptr(),
            out_b: out_b.as_mut_ptr(),
            out_c: out_c.as_mut_ptr(),
            rows_a,
            rows_b,
            rows_c,
            cols: cols_a,
            row_span_a: row_bytes_a,
            row_span_b: row_bytes_b,
            row_span_c: row_bytes_c,
            workers,
        });
        true
    }
}

pub fn matvec_quant2_into(
    a: (QuantMatvecKind, &[u8], usize, usize),
    b: (QuantMatvecKind, &[u8], usize, usize),
    x: &[f32],
    out_a: &mut Vec<f32>,
    out_b: &mut Vec<f32>,
) -> bool {
    let (kind_a, weights_a, rows_a, cols_a) = a;
    let (kind_b, weights_b, rows_b, cols_b) = b;
    if cols_a != cols_b || cols_a != x.len() {
        return false;
    }
    let Some(row_bytes_a) = kind_a.row_bytes(cols_a) else {
        return false;
    };
    let Some(row_bytes_b) = kind_b.row_bytes(cols_b) else {
        return false;
    };
    let needed_a = match row_bytes_a.checked_mul(rows_a) {
        Some(v) => v,
        None => return false,
    };
    let needed_b = match row_bytes_b.checked_mul(rows_b) {
        Some(v) => v,
        None => return false,
    };
    if weights_a.len() < needed_a || weights_b.len() < needed_b {
        return false;
    }

    out_a.resize(rows_a, 0.0);
    out_b.resize(rows_b, 0.0);

    #[cfg(not(target_family = "wasm"))]
    if try_quant_metal2_into(
        (kind_a, weights_a, rows_a, cols_a, row_bytes_a),
        (kind_b, weights_b, rows_b, cols_b, row_bytes_b),
        x,
        out_a,
        out_b,
    ) {
        return true;
    }

    #[cfg(target_family = "wasm")]
    {
        let kind_a = kind_a.matvec_kind();
        let kind_b = kind_b.matvec_kind();
        for row in 0..rows_a {
            out_a[row] = dot_row_from_kind(
                kind_a,
                &weights_a[row * row_bytes_a..(row + 1) * row_bytes_a],
                x,
                cols_a,
            );
        }
        for row in 0..rows_b {
            out_b[row] = dot_row_from_kind(
                kind_b,
                &weights_b[row * row_bytes_b..(row + 1) * row_bytes_b],
                x,
                cols_a,
            );
        }
        return true;
    }

    #[cfg(not(target_family = "wasm"))]
    {
        let total_rows = rows_a + rows_b;
        if total_rows == 0 {
            return true;
        }
        let workers = num_threads().min(total_rows);
        worker_pool().run_q4k_matvec3(Q4KMatvec3Job {
            kind_a: kind_a.matvec_kind(),
            kind_b: kind_b.matvec_kind(),
            kind_c: kind_b.matvec_kind(),
            a_data: weights_a.as_ptr(),
            b_data: weights_b.as_ptr(),
            c_data: weights_b.as_ptr(),
            x: x.as_ptr(),
            out_a: out_a.as_mut_ptr(),
            out_b: out_b.as_mut_ptr(),
            out_c: out_b.as_mut_ptr(),
            rows_a,
            rows_b,
            rows_c: 0,
            cols: cols_a,
            row_span_a: row_bytes_a,
            row_span_b: row_bytes_b,
            row_span_c: row_bytes_b,
            workers,
        });
        true
    }
}

#[allow(clippy::too_many_arguments)]
fn matvec_k3_into(
    kind: MatvecKind,
    _row_bytes: usize,
    a: (&[u8], usize, usize),
    b: (&[u8], usize, usize),
    c: (&[u8], usize, usize),
    x: &[f32],
    out_a: &mut Vec<f32>,
    out_b: &mut Vec<f32>,
    out_c: &mut Vec<f32>,
) -> bool {
    let kkind = match kind {
        MatvecKind::Q4K => KQuantMatvecKind::Q4K,
        MatvecKind::Q5K => KQuantMatvecKind::Q5K,
        MatvecKind::Q6K => KQuantMatvecKind::Q6K,
        _ => return false,
    };
    matvec_kquant3_into(
        (kkind, a.0, a.1, a.2),
        (kkind, b.0, b.1, b.2),
        (kkind, c.0, c.1, c.2),
        x,
        out_a,
        out_b,
        out_c,
    )
}

fn matvec_k2_into(
    kind: MatvecKind,
    row_bytes: usize,
    a: (&[u8], usize, usize),
    b: (&[u8], usize, usize),
    x: &[f32],
    out_a: &mut Vec<f32>,
    out_b: &mut Vec<f32>,
) -> bool {
    let (weights_a, rows_a, cols_a) = a;
    let (weights_b, rows_b, cols_b) = b;
    if cols_a == 0 || cols_a % 256 != 0 || cols_a != cols_b || cols_a != x.len() {
        return false;
    }
    let needed_a = match row_bytes.checked_mul(rows_a) {
        Some(v) => v,
        None => return false,
    };
    let needed_b = match row_bytes.checked_mul(rows_b) {
        Some(v) => v,
        None => return false,
    };
    if weights_a.len() < needed_a || weights_b.len() < needed_b {
        return false;
    }

    out_a.resize(rows_a, 0.0);
    out_b.resize(rows_b, 0.0);

    #[cfg(target_family = "wasm")]
    {
        for row in 0..rows_a {
            out_a[row] = dot_row_from_kind(
                kind,
                &weights_a[row * row_bytes..(row + 1) * row_bytes],
                x,
                cols_a,
            );
        }
        for row in 0..rows_b {
            out_b[row] = dot_row_from_kind(
                kind,
                &weights_b[row * row_bytes..(row + 1) * row_bytes],
                x,
                cols_a,
            );
        }
        return true;
    }

    #[cfg(not(target_family = "wasm"))]
    {
        let total_rows = rows_a + rows_b;
        if total_rows == 0 {
            return true;
        }
        let workers = num_threads().min(total_rows);
        worker_pool().run_q4k_matvec3(Q4KMatvec3Job {
            kind_a: kind,
            kind_b: kind,
            kind_c: kind,
            a_data: weights_a.as_ptr(),
            b_data: weights_b.as_ptr(),
            c_data: weights_b.as_ptr(),
            x: x.as_ptr(),
            out_a: out_a.as_mut_ptr(),
            out_b: out_b.as_mut_ptr(),
            out_c: out_b.as_mut_ptr(),
            rows_a,
            rows_b,
            rows_c: 0,
            cols: cols_a,
            row_span_a: row_bytes,
            row_span_b: row_bytes,
            row_span_c: row_bytes,
            workers,
        });
        true
    }
}

#[cfg(target_family = "wasm")]
fn dot_row_from_kind(kind: MatvecKind, row: &[u8], x: &[f32], cols: usize) -> f32 {
    match kind {
        MatvecKind::Q8_0 => dot_q8_0_f32(row, x, cols),
        MatvecKind::Q8_1 => dot_q8_1_f32(row, x, cols),
        MatvecKind::Q4_0 => dot_q4_0_f32(row, x, cols),
        MatvecKind::Q4_1 => dot_q4_1_f32(row, x, cols),
        MatvecKind::Q5_0 => dot_q5_0_f32(row, x, cols),
        MatvecKind::Q5_1 => dot_q5_1_f32(row, x, cols),
        MatvecKind::Q4K => dot_q4_k_f32(row, x, cols),
        MatvecKind::Q5K => dot_q5_k_f32(row, x, cols),
        MatvecKind::Q6K => dot_q6_k_f32(row, x, cols),
        MatvecKind::Mxfp4 => dot_mxfp4_f32(row, x, cols),
        _ => 0.0,
    }
}

/// Runs three Q5_K projections against the same input vector.
pub fn matvec_q5_k3_into(
    a: (&[u8], usize, usize),
    b: (&[u8], usize, usize),
    c: (&[u8], usize, usize),
    x: &[f32],
    out_a: &mut Vec<f32>,
    out_b: &mut Vec<f32>,
    out_c: &mut Vec<f32>,
) -> bool {
    matvec_k3_into(
        MatvecKind::Q5K,
        (a.2 / 256) * 176,
        a,
        b,
        c,
        x,
        out_a,
        out_b,
        out_c,
    )
}

/// Runs two Q5_K projections against the same input vector.
pub fn matvec_q5_k2_into(
    a: (&[u8], usize, usize),
    b: (&[u8], usize, usize),
    x: &[f32],
    out_a: &mut Vec<f32>,
    out_b: &mut Vec<f32>,
) -> bool {
    matvec_k2_into(MatvecKind::Q5K, (a.2 / 256) * 176, a, b, x, out_a, out_b)
}

/// Runs three Q6_K projections against the same input vector.
pub fn matvec_q6_k3_into(
    a: (&[u8], usize, usize),
    b: (&[u8], usize, usize),
    c: (&[u8], usize, usize),
    x: &[f32],
    out_a: &mut Vec<f32>,
    out_b: &mut Vec<f32>,
    out_c: &mut Vec<f32>,
) -> bool {
    #[cfg(not(target_family = "wasm"))]
    if crate::metal::q6k_matvec3_into(a, b, c, x, out_a, out_b, out_c) {
        return true;
    }
    matvec_k3_into(
        MatvecKind::Q6K,
        (a.2 / 256) * 210,
        a,
        b,
        c,
        x,
        out_a,
        out_b,
        out_c,
    )
}

/// Runs two Q6_K projections against the same input vector.
pub fn matvec_q6_k2_into(
    a: (&[u8], usize, usize),
    b: (&[u8], usize, usize),
    x: &[f32],
    out_a: &mut Vec<f32>,
    out_b: &mut Vec<f32>,
) -> bool {
    #[cfg(not(target_family = "wasm"))]
    if crate::metal::q6k_matvec2_into(a, b, x, out_a, out_b) {
        return true;
    }
    matvec_k2_into(MatvecKind::Q6K, (a.2 / 256) * 210, a, b, x, out_a, out_b)
}

/// Runs a Q6_K matrix-vector multiply into a reusable output buffer.
pub fn matvec_q6_k_into(qweight: &[u8], x: &[f32], rows: usize, cols: usize, out: &mut Vec<f32>) {
    let row_bytes = (cols / 256) * 210;
    out.resize(rows, 0.0);
    #[cfg(not(target_family = "wasm"))]
    if crate::metal::q6k_matvec_into(qweight, x, rows, cols, out) {
        return;
    }
    parallel_matvec_u8(MatvecKind::Q6K, out, rows, cols, row_bytes, qweight, x);
}

/// Runs an MXFP4 matrix-vector multiply into a reusable output buffer.
pub fn matvec_mxfp4_into(qweight: &[u8], x: &[f32], rows: usize, cols: usize, out: &mut Vec<f32>) {
    let row_bytes = (cols / 32) * 17;
    out.resize(rows, 0.0);
    parallel_matvec_u8(MatvecKind::Mxfp4, out, rows, cols, row_bytes, qweight, x);
}

macro_rules! dequant_row_vec {
    ($name:ident, $into:ident) => {
        pub fn $name(qrow: &[u8], cols: usize) -> Vec<f32> {
            let mut out = vec![0.0f32; cols];
            $into(qrow, &mut out);
            out
        }
    };
}

#[inline]
fn clear_dequant_tail(out: &mut [f32], written: usize) {
    if written < out.len() {
        out[written..].fill(0.0);
    }
}

/// Dequantizes one Q8_0 row into caller-owned storage.
pub fn dequant_row_q8_0_into(qrow: &[u8], out: &mut [f32]) {
    let n_blocks = out.len() / 32;
    clear_dequant_tail(out, n_blocks * 32);
    let block_size = 34;
    for b in 0..n_blocks {
        let block = &qrow[b * block_size..(b + 1) * block_size];
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for i in 0..32 {
            out[b * 32 + i] = scale * (block[2 + i] as i8) as f32;
        }
    }
}
dequant_row_vec!(dequant_row_q8_0, dequant_row_q8_0_into);

/// Dequantizes one Q8_1 row into caller-owned storage.
pub fn dequant_row_q8_1_into(qrow: &[u8], out: &mut [f32]) {
    let n_blocks = out.len() / 32;
    clear_dequant_tail(out, n_blocks * 32);
    let block_size = 36;
    for b in 0..n_blocks {
        let block = &qrow[b * block_size..(b + 1) * block_size];
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for i in 0..32 {
            out[b * 32 + i] = scale * (block[4 + i] as i8) as f32;
        }
    }
}
dequant_row_vec!(dequant_row_q8_1, dequant_row_q8_1_into);

/// Dequantizes one Q4_0 row into caller-owned storage.
pub fn dequant_row_q4_0_into(qrow: &[u8], out: &mut [f32]) {
    let n_blocks = out.len() / 32;
    clear_dequant_tail(out, n_blocks * 32);
    let block_size = 18;
    for b in 0..n_blocks {
        let block = &qrow[b * block_size..(b + 1) * block_size];
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for i in 0..16 {
            let byte = block[2 + i];
            let lo = ((byte & 0x0F) as i32 - 8) as f32;
            let hi = (((byte >> 4) & 0x0F) as i32 - 8) as f32;
            // ggml Q4_0 split layout: lo nibble of byte i → weight[i], hi → weight[i+16].
            out[b * 32 + i] = scale * lo;
            out[b * 32 + i + 16] = scale * hi;
        }
    }
}
dequant_row_vec!(dequant_row_q4_0, dequant_row_q4_0_into);

/// Dequantizes one Q4_1 row into caller-owned storage.
pub fn dequant_row_q4_1_into(qrow: &[u8], out: &mut [f32]) {
    let n_blocks = out.len() / 32;
    clear_dequant_tail(out, n_blocks * 32);
    let block_size = 20;
    for b in 0..n_blocks {
        let block = &qrow[b * block_size..(b + 1) * block_size];
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let min = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        for i in 0..16 {
            let byte = block[4 + i];
            // ggml split layout: lo nibble of byte i -> weight[i], hi -> weight[i+16].
            out[b * 32 + i] = scale * (byte & 0x0F) as f32 + min;
            out[b * 32 + i + 16] = scale * (byte >> 4) as f32 + min;
        }
    }
}
dequant_row_vec!(dequant_row_q4_1, dequant_row_q4_1_into);

/// Dequantizes one Q5_0 row into caller-owned storage.
pub fn dequant_row_q5_0_into(qrow: &[u8], out: &mut [f32]) {
    let n_blocks = out.len() / 32;
    clear_dequant_tail(out, n_blocks * 32);
    let block_size = 22;
    for b in 0..n_blocks {
        let block = &qrow[b * block_size..(b + 1) * block_size];
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        let qs = &block[6..22];
        for i in 0..16 {
            let byte = qs[i];
            let lo_hi = if ((qh >> i) & 1) != 0 { 16 } else { 0 };
            let hi_hi = if ((qh >> (i + 16)) & 1) != 0 { 16 } else { 0 };
            let lo = (((byte & 0x0F) | lo_hi) as i32 - 16) as f32;
            let hi = (((byte >> 4) | hi_hi) as i32 - 16) as f32;
            // ggml split layout: lo nibble of byte i -> weight[i], hi -> weight[i+16].
            out[b * 32 + i] = scale * lo;
            out[b * 32 + i + 16] = scale * hi;
        }
    }
}
dequant_row_vec!(dequant_row_q5_0, dequant_row_q5_0_into);

/// Dequantizes one Q5_1 row into caller-owned storage.
pub fn dequant_row_q5_1_into(qrow: &[u8], out: &mut [f32]) {
    let n_blocks = out.len() / 32;
    clear_dequant_tail(out, n_blocks * 32);
    let block_size = 24;
    for b in 0..n_blocks {
        let block = &qrow[b * block_size..(b + 1) * block_size];
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let min = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
        let qs = &block[8..24];
        for i in 0..16 {
            let byte = qs[i];
            let lo_hi = if ((qh >> i) & 1) != 0 { 16 } else { 0 };
            let hi_hi = if ((qh >> (i + 16)) & 1) != 0 { 16 } else { 0 };
            let lo = ((byte & 0x0F) | lo_hi) as f32;
            let hi = ((byte >> 4) | hi_hi) as f32;
            // ggml split layout: lo nibble of byte i -> weight[i], hi -> weight[i+16].
            out[b * 32 + i] = scale * lo + min;
            out[b * 32 + i + 16] = scale * hi + min;
        }
    }
}
dequant_row_vec!(dequant_row_q5_1, dequant_row_q5_1_into);

/// Dequantizes one Q4_K row into caller-owned storage.
pub fn dequant_row_q4_k_into(qrow: &[u8], out: &mut [f32]) {
    let n_blocks = out.len() / 256;
    clear_dequant_tail(out, n_blocks * 256);
    let block_size = 144;

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
}
dequant_row_vec!(dequant_row_q4_k, dequant_row_q4_k_into);

/// Dequantizes one Q5_K row into caller-owned storage.
pub fn dequant_row_q5_k_into(qrow: &[u8], out: &mut [f32]) {
    let n_blocks = out.len() / 256;
    clear_dequant_tail(out, n_blocks * 256);
    let block_size = 176;

    for b in 0..n_blocks {
        let block = &qrow[b * block_size..(b + 1) * block_size];
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales: &[u8; 12] = block[4..16].try_into().expect("q5_k scales size");
        let qh = &block[16..48];
        let mut qs = &block[48..176];
        let yoff = b * 256;

        let mut is = 0usize;
        let mut u1 = 1u8;
        let mut u2 = 2u8;
        for j in (0..256).step_by(64) {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let d1 = d * sc1 as f32;
            let min1 = dmin * m1 as f32;

            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc2 as f32;
            let min2 = dmin * m2 as f32;

            for l in 0..32 {
                let byte = qs[l];
                let hi0 = if (qh[l] & u1) != 0 { 16 } else { 0 };
                let hi1 = if (qh[l] & u2) != 0 { 16 } else { 0 };
                out[yoff + j + l] = d1 * ((byte & 0x0F) | hi0) as f32 - min1;
                out[yoff + j + 32 + l] = d2 * ((byte >> 4) | hi1) as f32 - min2;
            }

            qs = &qs[32..];
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}
dequant_row_vec!(dequant_row_q5_k, dequant_row_q5_k_into);

/// Dequantizes one Q6_K row into caller-owned storage.
pub fn dequant_row_q6_k_into(qrow: &[u8], out: &mut [f32]) {
    let n_blocks = out.len() / 256;
    clear_dequant_tail(out, n_blocks * 256);
    let block_size = 210;

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
                let q1 = ((((ql[l] & 0x0F) | ((qh[l] & 0x03) << 4)) as i32) - 32) as f32;
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
}
dequant_row_vec!(dequant_row_q6_k, dequant_row_q6_k_into);

/// Dequantizes one MXFP4 row into caller-owned storage.
pub fn dequant_row_mxfp4_into(qrow: &[u8], out: &mut [f32]) {
    let n_blocks = out.len() / 32;
    clear_dequant_tail(out, n_blocks * 32);
    let block_size = 17;

    for b in 0..n_blocks {
        let block = &qrow[b * block_size..(b + 1) * block_size];
        let scale = 2.0f32.powi(block[16] as i32 - 127);
        for i in 0..16 {
            let byte = block[i];
            out[b * 32 + i * 2] = mxfp4_nibble_to_f32(byte & 0x0F) * scale;
            out[b * 32 + i * 2 + 1] = mxfp4_nibble_to_f32(byte >> 4) * scale;
        }
    }
}
dequant_row_vec!(dequant_row_mxfp4, dequant_row_mxfp4_into);

// ─── Scalar fallbacks ────────────────────────────────────────────────────────

#[allow(dead_code)]
/// Portable scalar implementation of f32 dot product.
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
/// Portable scalar implementation of Q8_0 dot product.
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
/// Portable scalar implementation of Q4_0 dot product.
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
            // ggml Q4_0 split layout: lo nibble of byte i → weight[i], hi → weight[i+16].
            block_sum += lo * x[b * 32 + i];
            block_sum += hi * x[b * 32 + i + 16];
        }
        sum += scale * block_sum;
    }
    sum
}

/// Portable scalar implementation of Q8_1 dot product.
fn dot_q8_1_f32_scalar(qdata: &[u8], x: &[f32], n: usize) -> f32 {
    let n_blocks = n / 32;
    let block_size = 36;
    let mut sum = 0.0f32;

    for b in 0..n_blocks {
        let block = &qdata[b * block_size..];
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let mut block_sum = 0.0f32;
        for i in 0..32 {
            block_sum += (block[4 + i] as i8) as f32 * x[b * 32 + i];
        }
        sum += scale * block_sum;
    }

    sum
}

/// Portable scalar implementation of Q4_1 dot product.
fn dot_q4_1_f32_scalar(qdata: &[u8], x: &[f32], n: usize) -> f32 {
    let n_blocks = n / 32;
    let block_size = 20;
    let mut sum = 0.0f32;

    for b in 0..n_blocks {
        let block = &qdata[b * block_size..];
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let min = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let mut qsum = 0.0f32;
        let mut xsum = 0.0f32;
        for i in 0..16 {
            let byte = block[4 + i];
            let x0 = x[b * 32 + i];
            let x1 = x[b * 32 + i + 16];
            qsum += (byte & 0x0F) as f32 * x0;
            qsum += (byte >> 4) as f32 * x1;
            xsum += x0 + x1;
        }
        sum += scale * qsum + min * xsum;
    }

    sum
}

/// Portable scalar implementation of Q5_0 dot product.
fn dot_q5_0_f32_scalar(qdata: &[u8], x: &[f32], n: usize) -> f32 {
    let n_blocks = n / 32;
    let block_size = 22;
    let mut sum = 0.0f32;

    for b in 0..n_blocks {
        let block = &qdata[b * block_size..];
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        let qs = &block[6..22];
        let mut block_sum = 0.0f32;
        for i in 0..16 {
            let byte = qs[i];
            let lo_hi = if ((qh >> i) & 1) != 0 { 16 } else { 0 };
            let hi_hi = if ((qh >> (i + 16)) & 1) != 0 { 16 } else { 0 };
            let lo = (((byte & 0x0F) | lo_hi) as i32 - 16) as f32;
            let hi = (((byte >> 4) | hi_hi) as i32 - 16) as f32;
            block_sum += lo * x[b * 32 + i];
            block_sum += hi * x[b * 32 + i + 16];
        }
        sum += scale * block_sum;
    }

    sum
}

/// Portable scalar implementation of Q5_1 dot product.
fn dot_q5_1_f32_scalar(qdata: &[u8], x: &[f32], n: usize) -> f32 {
    let n_blocks = n / 32;
    let block_size = 24;
    let mut sum = 0.0f32;

    for b in 0..n_blocks {
        let block = &qdata[b * block_size..];
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let min = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
        let qs = &block[8..24];
        let mut qsum = 0.0f32;
        let mut xsum = 0.0f32;
        for i in 0..16 {
            let byte = qs[i];
            let lo_hi = if ((qh >> i) & 1) != 0 { 16 } else { 0 };
            let hi_hi = if ((qh >> (i + 16)) & 1) != 0 { 16 } else { 0 };
            let x0 = x[b * 32 + i];
            let x1 = x[b * 32 + i + 16];
            qsum += ((byte & 0x0F) | lo_hi) as f32 * x0;
            qsum += ((byte >> 4) | hi_hi) as f32 * x1;
            xsum += x0 + x1;
        }
        sum += scale * qsum + min * xsum;
    }

    sum
}

#[inline]
/// Extracts Q4_K scale and minimum values for one sub-block.
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

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn dot_q6_k_f32_neon(qdata: &[u8], x: &[f32], n: usize) -> f32 {
    use std::arch::aarch64::*;
    let n_blocks = n / 256;
    let block_size = 210;
    let mut total = vdupq_n_f32(0.0);
    let mask_lo4 = vdupq_n_u8(0x0F);
    let mask_03 = vdupq_n_u8(0x03);
    let sub32 = vdupq_n_u8(32);

    for b in 0..n_blocks {
        let block = qdata.as_ptr().add(b * block_size);
        let d = f16_to_f32(u16::from_le_bytes([*block.add(208), *block.add(209)]));
        let xbase = x.as_ptr().add(b * 256);
        let mut ql_ptr = block;
        let mut qh_ptr = block.add(128);
        let mut sc_ptr = block.add(192);
        let mut grp_x_base = 0usize;

        for _grp in 0..2 {
            for half in 0..2usize {
                let l = half * 16;
                let is = half;

                let sv1 = vdupq_n_f32(d * (*sc_ptr.add(is) as i8) as f32);
                let sv2 = vdupq_n_f32(d * (*sc_ptr.add(is + 2) as i8) as f32);
                let sv3 = vdupq_n_f32(d * (*sc_ptr.add(is + 4) as i8) as f32);
                let sv4 = vdupq_n_f32(d * (*sc_ptr.add(is + 6) as i8) as f32);

                let ql1 = vld1q_u8(ql_ptr.add(l));
                let ql2 = vld1q_u8(ql_ptr.add(l + 32));
                let qhv = vld1q_u8(qh_ptr.add(l));

                let lo1 = vandq_u8(ql1, mask_lo4);
                let hi1 = vshrq_n_u8(ql1, 4);
                let lo2 = vandq_u8(ql2, mask_lo4);
                let hi2 = vshrq_n_u8(ql2, 4);

                let h1 = vandq_u8(qhv, mask_03);
                let h2 = vandq_u8(vshrq_n_u8(qhv, 2), mask_03);
                let h3 = vandq_u8(vshrq_n_u8(qhv, 4), mask_03);
                let h4 = vshrq_n_u8(qhv, 6);

                let q1 = vreinterpretq_s8_u8(vsubq_u8(vorrq_u8(lo1, vshlq_n_u8(h1, 4)), sub32));
                let q2 = vreinterpretq_s8_u8(vsubq_u8(vorrq_u8(lo2, vshlq_n_u8(h2, 4)), sub32));
                let q3 = vreinterpretq_s8_u8(vsubq_u8(vorrq_u8(hi1, vshlq_n_u8(h3, 4)), sub32));
                let q4 = vreinterpretq_s8_u8(vsubq_u8(vorrq_u8(hi2, vshlq_n_u8(h4, 4)), sub32));

                macro_rules! dot16 {
                    ($qi8:expr, $xptr:expr, $sv:expr) => {{
                        let qlo = vmovl_s8(vget_low_s8($qi8));
                        let qhi = vmovl_s8(vget_high_s8($qi8));
                        let f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(qlo)));
                        let f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(qlo)));
                        let f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(qhi)));
                        let f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(qhi)));
                        total = vmlaq_f32(total, vmulq_f32(f0, $sv), vld1q_f32($xptr));
                        total = vmlaq_f32(total, vmulq_f32(f1, $sv), vld1q_f32($xptr.add(4)));
                        total = vmlaq_f32(total, vmulq_f32(f2, $sv), vld1q_f32($xptr.add(8)));
                        total = vmlaq_f32(total, vmulq_f32(f3, $sv), vld1q_f32($xptr.add(12)));
                    }};
                }

                let x_off = grp_x_base + l;
                dot16!(q1, xbase.add(x_off), sv1);
                dot16!(q2, xbase.add(x_off + 32), sv2);
                dot16!(q3, xbase.add(x_off + 64), sv3);
                dot16!(q4, xbase.add(x_off + 96), sv4);
            }
            ql_ptr = ql_ptr.add(64);
            qh_ptr = qh_ptr.add(32);
            sc_ptr = sc_ptr.add(8);
            grp_x_base += 128;
        }
    }
    vaddvq_f32(total)
}

#[allow(dead_code)]
/// Portable scalar implementation of Q4_K dot product.
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

#[allow(dead_code)]
/// Portable scalar implementation of Q5_K dot product.
fn dot_q5_k_f32_scalar(qdata: &[u8], x: &[f32], n: usize) -> f32 {
    let n_blocks = n / 256;
    let block_size = 176;
    let mut sum = 0.0f32;

    for b in 0..n_blocks {
        let block = &qdata[b * block_size..(b + 1) * block_size];
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales: &[u8; 12] = block[4..16].try_into().expect("q5_k scales size");
        let qh = &block[16..48];
        let mut qs = &block[48..176];
        let xoff = b * 256;

        let mut is = 0usize;
        let mut u1 = 1u8;
        let mut u2 = 2u8;
        for j in (0..256).step_by(64) {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let d1 = d * sc1 as f32;
            let min1 = dmin * m1 as f32;

            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc2 as f32;
            let min2 = dmin * m2 as f32;

            let mut qdot1 = 0.0f32;
            let mut qdot2 = 0.0f32;
            let mut xsum1 = 0.0f32;
            let mut xsum2 = 0.0f32;

            for l in 0..32 {
                let byte = qs[l];
                let idx1 = j + l;
                let idx2 = j + 32 + l;
                let hi1 = if (qh[l] & u1) != 0 { 16 } else { 0 };
                let hi2 = if (qh[l] & u2) != 0 { 16 } else { 0 };
                let x1 = x[xoff + idx1];
                let x2 = x[xoff + idx2];
                qdot1 += ((byte & 0x0F) | hi1) as f32 * x1;
                qdot2 += ((byte >> 4) | hi2) as f32 * x2;
                xsum1 += x1;
                xsum2 += x2;
            }

            sum += d1 * qdot1 - min1 * xsum1;
            sum += d2 * qdot2 - min2 * xsum2;

            qs = &qs[32..];
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }

    sum
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn dot_q5_k_f32_neon(qdata: &[u8], x: &[f32], n: usize) -> f32 {
    use std::arch::aarch64::*;

    let n_blocks = n / 256;
    let mask_low = vdupq_n_u8(0x0F);
    let high_bit = vdupq_n_u8(16);
    let zero = vdupq_n_u8(0);
    let mut total = vdupq_n_f32(0.0);

    for b in 0..n_blocks {
        let block = qdata.as_ptr().add(b * 176);
        let d = f16_to_f32(u16::from_le_bytes([*block, *block.add(1)]));
        let dmin = f16_to_f32(u16::from_le_bytes([*block.add(2), *block.add(3)]));
        let scales: &[u8; 12] = std::slice::from_raw_parts(block.add(4), 12)
            .try_into()
            .expect("q5_k scales size");
        let qh = block.add(16);
        let qs = block.add(48);
        let xbase = x.as_ptr().add(b * 256);

        let mut is = 0usize;
        let mut u1 = 1u8;
        let mut u2 = 2u8;
        for chunk in 0..4usize {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1 = vdupq_n_f32(d * sc1 as f32);
            let min1 = vdupq_n_f32(dmin * m1 as f32);
            let d2 = vdupq_n_f32(d * sc2 as f32);
            let min2 = vdupq_n_f32(dmin * m2 as f32);
            let u1v = vdupq_n_u8(u1);
            let u2v = vdupq_n_u8(u2);

            let qchunk = qs.add(chunk * 32);
            let x1 = xbase.add(chunk * 64);
            let x2 = x1.add(32);

            let mut qacc1 = vdupq_n_f32(0.0);
            let mut qacc2 = vdupq_n_f32(0.0);
            let mut xsum1 = vdupq_n_f32(0.0);
            let mut xsum2 = vdupq_n_f32(0.0);

            for lane in (0..32usize).step_by(16) {
                let packed = vld1q_u8(qchunk.add(lane));
                let high = vld1q_u8(qh.add(lane));
                let lo_high = vandq_u8(vcgtq_u8(vandq_u8(high, u1v), zero), high_bit);
                let hi_high = vandq_u8(vcgtq_u8(vandq_u8(high, u2v), zero), high_bit);
                let lo = vorrq_u8(vandq_u8(packed, mask_low), lo_high);
                let hi = vorrq_u8(vshrq_n_u8(packed, 4), hi_high);

                macro_rules! accumulate_u8x16 {
                    ($q:expr, $xptr:expr, $qacc:ident, $xsum:ident) => {{
                        let q16 = vmovl_u8(vget_low_u8($q));
                        let q32a = vcvtq_f32_u32(vmovl_u16(vget_low_u16(q16)));
                        let q32b = vcvtq_f32_u32(vmovl_u16(vget_high_u16(q16)));
                        let xa = vld1q_f32($xptr);
                        let xb = vld1q_f32($xptr.add(4));
                        $qacc = vmlaq_f32($qacc, q32a, xa);
                        $qacc = vmlaq_f32($qacc, q32b, xb);
                        $xsum = vaddq_f32($xsum, xa);
                        $xsum = vaddq_f32($xsum, xb);

                        let q16 = vmovl_u8(vget_high_u8($q));
                        let q32a = vcvtq_f32_u32(vmovl_u16(vget_low_u16(q16)));
                        let q32b = vcvtq_f32_u32(vmovl_u16(vget_high_u16(q16)));
                        let xa = vld1q_f32($xptr.add(8));
                        let xb = vld1q_f32($xptr.add(12));
                        $qacc = vmlaq_f32($qacc, q32a, xa);
                        $qacc = vmlaq_f32($qacc, q32b, xb);
                        $xsum = vaddq_f32($xsum, xa);
                        $xsum = vaddq_f32($xsum, xb);
                    }};
                }

                accumulate_u8x16!(lo, x1.add(lane), qacc1, xsum1);
                accumulate_u8x16!(hi, x2.add(lane), qacc2, xsum2);
            }

            let part1 = vsubq_f32(vmulq_f32(qacc1, d1), vmulq_f32(xsum1, min1));
            let part2 = vsubq_f32(vmulq_f32(qacc2, d2), vmulq_f32(xsum2, min2));
            total = vaddq_f32(total, vaddq_f32(part1, part2));

            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }

    vaddvq_f32(total)
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn dot_q4_k_f32_neon(qdata: &[u8], x: &[f32], n: usize) -> f32 {
    use std::arch::aarch64::*;

    let n_blocks = n / 256;
    let mask_low = vdup_n_u8(0x0F);
    let mut sum_acc = vdupq_n_f32(0.0);

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
            let d1 = vdupq_n_f32(d * sc1 as f32);
            let d2 = vdupq_n_f32(d * sc2 as f32);
            let min1 = vdupq_n_f32(dmin * m1 as f32);
            let min2 = vdupq_n_f32(dmin * m2 as f32);

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

            let part1 = vsubq_f32(
                vmulq_f32(vaddq_f32(qacc1a, qacc1b), d1),
                vmulq_f32(vaddq_f32(xsum1a, xsum1b), min1),
            );
            let part2 = vsubq_f32(
                vmulq_f32(vaddq_f32(qacc2a, qacc2b), d2),
                vmulq_f32(vaddq_f32(xsum2a, xsum2b), min2),
            );
            sum_acc = vaddq_f32(sum_acc, vaddq_f32(part1, part2));

            q = q.add(32);
            is += 2;
        }
    }

    vaddvq_f32(sum_acc)
}

/// Portable scalar implementation of Q6_K dot product.
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
                let q1 = ((((ql[l] & 0x0F) | ((qh[l] & 0x03) << 4)) as i32) - 32) as f32;
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

#[inline(always)]
/// Converts one MXFP4 nibble to its float value.
fn mxfp4_nibble_to_f32(v: u8) -> f32 {
    const LUT: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    LUT[(v & 0x0F) as usize]
}

#[inline(always)]
/// Converts one MXFP4 block scale byte to `f32`.
fn mxfp4_scale_to_f32(v: u8) -> f32 {
    #[cfg(not(target_family = "wasm"))]
    {
        mxfp4_scale_lookup()[v as usize]
    }
    #[cfg(target_family = "wasm")]
    {
        2.0f32.powi(v as i32 - 127)
    }
}

#[cfg(not(target_family = "wasm"))]
/// Returns the lazily initialized MXFP4 scale lookup table.
fn mxfp4_scale_lookup() -> &'static [f32; 256] {
    static MXFP4_SCALE_LOOKUP: OnceLock<[f32; 256]> = OnceLock::new();
    MXFP4_SCALE_LOOKUP.get_or_init(|| {
        let mut table = [0.0f32; 256];
        for (i, value) in table.iter_mut().enumerate() {
            *value = 2.0f32.powi(i as i32 - 127);
        }
        table
    })
}

/// Portable scalar implementation of MXFP4 dot product.
fn dot_mxfp4_f32_scalar(qdata: &[u8], x: &[f32], n: usize) -> f32 {
    let n_blocks = n / 32;
    let block_size = 17;
    let mut sum = 0.0f32;

    for b in 0..n_blocks {
        let block = &qdata[b * block_size..(b + 1) * block_size];
        let scale = mxfp4_scale_to_f32(block[16]);
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
    // ggml split layout: lo nibble of byte i → weight[i], hi nibble → weight[i+16].
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
        // lo nibbles (bits[3:0]): byte i → weight[i]      (weights 0..16)
        let lo_i8 = vreinterpretq_s8_u8(vsubq_u8(vandq_u8(nib, mask_low), eight));
        // hi nibbles (bits[7:4]): byte i → weight[i+16]   (weights 16..32)
        let hi_i8 = vreinterpretq_s8_u8(vsubq_u8(vshrq_n_u8(nib, 4), eight));

        // Widen to i16 (8 each per half), preserving weight order.
        let w0_8 = vmovl_s8(vget_low_s8(lo_i8)); // weights 0..8
        let w8_16 = vmovl_s8(vget_high_s8(lo_i8)); // weights 8..16
        let w16_24 = vmovl_s8(vget_low_s8(hi_i8)); // weights 16..24
        let w24_32 = vmovl_s8(vget_high_s8(hi_i8)); // weights 24..32

        let xp = x.as_ptr().add(b * 32);
        let mut bacc = vdupq_n_f32(0.0);

        // Each chunk multiplies 4 contiguous weights by 4 contiguous x values.
        macro_rules! chunk {
            ($w4:expr, $xoff:expr) => {{
                let wf = vcvtq_f32_s32(vmovl_s16($w4));
                let xv = vld1q_f32(xp.add($xoff));
                bacc = vmlaq_f32(bacc, wf, xv);
            }};
        }
        chunk!(vget_low_s16(w0_8), 0);
        chunk!(vget_high_s16(w0_8), 4);
        chunk!(vget_low_s16(w8_16), 8);
        chunk!(vget_high_s16(w8_16), 12);
        chunk!(vget_low_s16(w16_24), 16);
        chunk!(vget_high_s16(w16_24), 20);
        chunk!(vget_low_s16(w24_32), 24);
        chunk!(vget_high_s16(w24_32), 28);

        sum_acc = vmlaq_f32(sum_acc, bacc, scale_v);
    }
    vaddvq_f32(sum_acc)
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn axpy_f32_neon(out: &mut [f32], alpha: f32, x: &[f32]) {
    use std::arch::aarch64::*;
    let n = out.len();
    let main = n / 8;
    let av = vdupq_n_f32(alpha);
    let mut op = out.as_mut_ptr();
    let mut xp = x.as_ptr();
    for _ in 0..main {
        let o0 = vld1q_f32(op);
        let o1 = vld1q_f32(op.add(4));
        let x0 = vld1q_f32(xp);
        let x1 = vld1q_f32(xp.add(4));
        vst1q_f32(op, vmlaq_f32(o0, x0, av));
        vst1q_f32(op.add(4), vmlaq_f32(o1, x1, av));
        op = op.add(8);
        xp = xp.add(8);
    }
    for i in (main * 8)..n {
        out[i] += alpha * x[i];
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn scale_f32_neon(out: &mut [f32], scale: f32) {
    use std::arch::aarch64::*;
    let n = out.len();
    let main = n / 8;
    let sv = vdupq_n_f32(scale);
    let mut op = out.as_mut_ptr();
    for _ in 0..main {
        let o0 = vld1q_f32(op);
        let o1 = vld1q_f32(op.add(4));
        vst1q_f32(op, vmulq_f32(o0, sv));
        vst1q_f32(op.add(4), vmulq_f32(o1, sv));
        op = op.add(8);
    }
    for i in (main * 8)..n {
        out[i] *= scale;
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn scale_add_f32_neon(out: &mut [f32], scale: f32, add: &[f32]) {
    use std::arch::aarch64::*;
    let n = out.len();
    let main = n / 8;
    let sv = vdupq_n_f32(scale);
    let mut op = out.as_mut_ptr();
    let mut ap = add.as_ptr();
    for _ in 0..main {
        let o0 = vld1q_f32(op);
        let o1 = vld1q_f32(op.add(4));
        let a0 = vld1q_f32(ap);
        let a1 = vld1q_f32(ap.add(4));
        vst1q_f32(op, vmlaq_f32(a0, o0, sv));
        vst1q_f32(op.add(4), vmlaq_f32(a1, o1, sv));
        op = op.add(8);
        ap = ap.add(8);
    }
    for i in (main * 8)..n {
        out[i] = out[i] * scale + add[i];
    }
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
    let main = n / 32;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut ap = a.as_ptr();
    let mut bp = b.as_ptr();
    for _ in 0..main {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(ap), _mm256_loadu_ps(bp), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(8)), _mm256_loadu_ps(bp.add(8)), acc1);
        acc2 = _mm256_fmadd_ps(
            _mm256_loadu_ps(ap.add(16)),
            _mm256_loadu_ps(bp.add(16)),
            acc2,
        );
        acc3 = _mm256_fmadd_ps(
            _mm256_loadu_ps(ap.add(24)),
            _mm256_loadu_ps(bp.add(24)),
            acc3,
        );
        ap = ap.add(32);
        bp = bp.add(32);
    }
    let mut acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    let tail_start = main * 32;
    let tail_8 = (n - tail_start) / 8;
    for _ in 0..tail_8 {
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(ap), _mm256_loadu_ps(bp), acc);
        ap = ap.add(8);
        bp = bp.add(8);
    }
    let mut sum = hsum_avx(acc);
    for i in (tail_start + tail_8 * 8)..n {
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
    // ggml split layout: lo nibble of byte i → weight[i], hi nibble → weight[i+16].
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
        // lo nibbles (byte & 0x0F): bytes 0..16 → weights 0..16
        let lo = _mm_and_si128(nib, mask_0f);
        // hi nibbles ((byte >> 4) & 0x0F): bytes 0..16 → weights 16..32
        let hi = _mm_and_si128(_mm_srli_epi16(nib, 4), mask_0f);

        // cvtepu8_epi32 uses lower 8 bytes of __m128i → 8 × i32 (unsigned zero-extend)
        // Then subtract 8.0 in f32 to recover signed values in [-8, 7].
        macro_rules! process8 {
            ($qreg:expr, $xoff:expr) => {{
                let qf = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32($qreg)), eight_ps);
                let xv = _mm256_loadu_ps(xp.add($xoff));
                acc = _mm256_fmadd_ps(_mm256_mul_ps(sv, qf), xv, acc);
            }};
        }
        process8!(lo, 0); // weights 0..8   · x[0..8]
        process8!(_mm_srli_si128(lo, 8), 8); // weights 8..16  · x[8..16]
        process8!(hi, 16); // weights 16..24 · x[16..24]
        process8!(_mm_srli_si128(hi, 8), 24); // weights 24..32 · x[24..32]
    }
    hsum_avx(acc)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn axpy_f32_avx2(out: &mut [f32], alpha: f32, x: &[f32]) {
    use std::arch::x86_64::*;
    let n = out.len();
    let main = n / 8;
    let av = _mm256_set1_ps(alpha);
    let mut op = out.as_mut_ptr();
    let mut xp = x.as_ptr();
    for _ in 0..main {
        let o = _mm256_loadu_ps(op);
        let xv = _mm256_loadu_ps(xp);
        let y = _mm256_fmadd_ps(av, xv, o);
        _mm256_storeu_ps(op, y);
        op = op.add(8);
        xp = xp.add(8);
    }
    for i in (main * 8)..n {
        out[i] += alpha * x[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn scale_f32_avx2(out: &mut [f32], scale: f32) {
    use std::arch::x86_64::*;
    let n = out.len();
    let main = n / 8;
    let sv = _mm256_set1_ps(scale);
    let mut op = out.as_mut_ptr();
    for _ in 0..main {
        let o = _mm256_loadu_ps(op);
        _mm256_storeu_ps(op, _mm256_mul_ps(o, sv));
        op = op.add(8);
    }
    for i in (main * 8)..n {
        out[i] *= scale;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn scale_add_f32_avx2(out: &mut [f32], scale: f32, add: &[f32]) {
    use std::arch::x86_64::*;
    let n = out.len();
    let main = n / 8;
    let sv = _mm256_set1_ps(scale);
    let mut op = out.as_mut_ptr();
    let mut ap = add.as_ptr();
    for _ in 0..main {
        let o = _mm256_loadu_ps(op);
        let a = _mm256_loadu_ps(ap);
        let y = _mm256_fmadd_ps(o, sv, a);
        _mm256_storeu_ps(op, y);
        op = op.add(8);
        ap = ap.add(8);
    }
    for i in (main * 8)..n {
        out[i] = out[i] * scale + add[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds one Q4_0 block (18 bytes) with scale 1.0 and byte[i] = lo|(hi<<4),
    /// where lo = i and hi = 15 - i for i in 0..16.
    fn make_q4_0_block_known() -> Vec<u8> {
        let mut block = vec![0u8; 18];
        // f16 1.0 = 0x3c00, little-endian
        block[0] = 0x00;
        block[1] = 0x3c;
        for i in 0..16usize {
            let lo = i as u8; // 0..=15
            let hi = (15 - i) as u8; // 15..=0
            block[2 + i] = (hi << 4) | lo;
        }
        block
    }

    /// Builds deterministic synthetic Q4_0 matrix weights.
    fn make_q4_0_weights(rows: usize, cols: usize, seed: u8) -> Vec<u8> {
        let row_bytes = (cols / 32) * 18;
        let mut data = vec![0u8; rows * row_bytes];
        for row in 0..rows {
            for block in 0..(cols / 32) {
                let base = row * row_bytes + block * 18;
                data[base] = 0x00;
                data[base + 1] = 0x3c; // f16 1.0
                for i in 0..16 {
                    data[base + 2 + i] = seed.wrapping_add((row * 17 + block * 11 + i * 5) as u8);
                }
            }
        }
        data
    }

    #[test]
    fn q4_0_dequant_uses_split_layout() {
        // ggml Q4_0 split layout: lo nibble of byte i → weight[i],
        // hi nibble of byte i → weight[i+16]. Scale = 1.0.
        let block = make_q4_0_block_known();
        let got = dequant_row_q4_0(&block, 32);

        let mut expected = [0.0f32; 32];
        for i in 0..16usize {
            expected[i] = i as f32 - 8.0; // lo nibble: -8, -7, ..., 7
            expected[i + 16] = (15 - i) as f32 - 8.0; // hi nibble: 7, 6, ..., -8
        }
        for k in 0..32 {
            assert_eq!(got[k], expected[k], "weight[{k}] mismatch");
        }
    }

    #[test]
    fn q4_0_dot_matches_reference_dequant() {
        // The fused dot kernel must equal dequant-then-dot for an arbitrary x.
        let block = make_q4_0_block_known();
        let weights = dequant_row_q4_0(&block, 32);

        let x: Vec<f32> = (0..32).map(|k| (k as f32) * 0.5 - 7.0).collect();
        let reference: f32 = weights.iter().zip(&x).map(|(w, xv)| w * xv).sum();

        let fused = dot_q4_0_f32(&block, &x, 32);
        assert!(
            (fused - reference).abs() < 1e-3,
            "fused {fused} vs reference {reference}"
        );
    }

    #[test]
    fn q4_1_dequant_uses_split_layout() {
        let mut block = vec![0u8; 20];
        block[0] = 0x00;
        block[1] = 0x3c; // f16 scale = 1.0
        block[2] = 0x00;
        block[3] = 0x00; // f16 min = 0.0
        for i in 0..16usize {
            let lo = i as u8;
            let hi = (15 - i) as u8;
            block[4 + i] = (hi << 4) | lo;
        }
        let got = dequant_row_q4_1(&block, 32);
        let mut expected = [0.0f32; 32];
        for i in 0..16usize {
            expected[i] = i as f32;
            expected[i + 16] = (15 - i) as f32;
        }
        for k in 0..32 {
            assert_eq!(got[k], expected[k], "weight[{k}] mismatch");
        }
    }

    #[test]
    fn q4_1_dot_matches_reference_dequant() {
        let mut block = vec![0u8; 20];
        block[0] = 0x00;
        block[1] = 0x3c;
        block[2] = 0x00;
        block[3] = 0x00;
        for i in 0..16usize {
            block[4 + i] = (((15 - i) as u8) << 4) | (i as u8);
        }
        let weights = dequant_row_q4_1(&block, 32);
        let x: Vec<f32> = (0..32).map(|k| (k as f32) * 0.5 - 7.0).collect();
        let reference: f32 = weights.iter().zip(&x).map(|(w, xv)| w * xv).sum();
        let fused = dot_q4_1_f32(&block, &x, 32);
        assert!(
            (fused - reference).abs() < 1e-3,
            "fused {fused} vs reference {reference}"
        );
    }

    #[test]
    fn q5_0_dequant_uses_split_layout() {
        let mut block = vec![0u8; 22];
        block[0] = 0x00;
        block[1] = 0x3c; // f16 scale = 1.0
        // qh = 0 -> no high bits set
        for i in 0..16usize {
            let lo = i as u8;
            let hi = (15 - i) as u8;
            block[6 + i] = (hi << 4) | lo;
        }
        let got = dequant_row_q5_0(&block, 32);
        let mut expected = [0.0f32; 32];
        for i in 0..16usize {
            expected[i] = i as f32 - 16.0;
            expected[i + 16] = (15 - i) as f32 - 16.0;
        }
        for k in 0..32 {
            assert_eq!(got[k], expected[k], "weight[{k}] mismatch");
        }
    }

    #[test]
    fn q5_0_dot_matches_reference_dequant() {
        let mut block = vec![0u8; 22];
        block[0] = 0x00;
        block[1] = 0x3c;
        for i in 0..16usize {
            block[6 + i] = (((15 - i) as u8) << 4) | (i as u8);
        }
        let weights = dequant_row_q5_0(&block, 32);
        let x: Vec<f32> = (0..32).map(|k| (k as f32) * 0.5 - 7.0).collect();
        let reference: f32 = weights.iter().zip(&x).map(|(w, xv)| w * xv).sum();
        let fused = dot_q5_0_f32(&block, &x, 32);
        assert!(
            (fused - reference).abs() < 1e-3,
            "fused {fused} vs reference {reference}"
        );
    }

    #[test]
    fn q5_1_dequant_uses_split_layout() {
        let mut block = vec![0u8; 24];
        block[0] = 0x00;
        block[1] = 0x3c; // f16 scale = 1.0
        block[2] = 0x00;
        block[3] = 0x00; // f16 min = 0.0
        // qh = 0 -> no high bits set
        for i in 0..16usize {
            let lo = i as u8;
            let hi = (15 - i) as u8;
            block[8 + i] = (hi << 4) | lo;
        }
        let got = dequant_row_q5_1(&block, 32);
        let mut expected = [0.0f32; 32];
        for i in 0..16usize {
            expected[i] = i as f32;
            expected[i + 16] = (15 - i) as f32;
        }
        for k in 0..32 {
            assert_eq!(got[k], expected[k], "weight[{k}] mismatch");
        }
    }

    #[test]
    fn q5_1_dot_matches_reference_dequant() {
        let mut block = vec![0u8; 24];
        block[0] = 0x00;
        block[1] = 0x3c;
        block[2] = 0x00;
        block[3] = 0x00;
        for i in 0..16usize {
            block[8 + i] = (((15 - i) as u8) << 4) | (i as u8);
        }
        let weights = dequant_row_q5_1(&block, 32);
        let x: Vec<f32> = (0..32).map(|k| (k as f32) * 0.5 - 7.0).collect();
        let reference: f32 = weights.iter().zip(&x).map(|(w, xv)| w * xv).sum();
        let fused = dot_q5_1_f32(&block, &x, 32);
        assert!(
            (fused - reference).abs() < 1e-3,
            "fused {fused} vs reference {reference}"
        );
    }

    /// Builds deterministic synthetic Q4_K weights for fused-kernel tests.
    fn make_q4k_weights(rows: usize, cols: usize, seed: u8) -> Vec<u8> {
        let row_bytes = (cols / 256) * 144;
        let mut data = vec![0u8; rows * row_bytes];
        for row in 0..rows {
            for block in 0..(cols / 256) {
                let base = row * row_bytes + block * 144;
                data[base] = 0x00;
                data[base + 1] = 0x3c; // f16 1.0
                data[base + 2] = 0x00;
                data[base + 3] = 0x00; // f16 0.0 for dmin
                for i in 0..12 {
                    data[base + 4 + i] =
                        1 + ((seed.wrapping_add(row as u8).wrapping_add(i as u8)) & 0x07);
                }
                for i in 0..128 {
                    data[base + 16 + i] =
                        seed.wrapping_add((row * 17 + block * 29 + i * 3) as u8) & 0x7f;
                }
            }
        }
        data
    }

    /// Builds deterministic synthetic Q5_K weights for scalar/SIMD parity tests.
    fn make_q5k_blocks(blocks: usize, seed: u8) -> Vec<u8> {
        let mut data = vec![0u8; blocks * 176];
        for block in 0..blocks {
            let base = block * 176;
            data[base] = 0x00;
            data[base + 1] = 0x3c; // f16 1.0
            data[base + 2] = 0x00;
            data[base + 3] = 0x38; // f16 0.5 dmin
            for i in 0..12 {
                data[base + 4 + i] = seed.wrapping_add((block * 11 + i * 7) as u8) & 0x3f;
            }
            for i in 0..32 {
                data[base + 16 + i] = seed.wrapping_add((block * 13 + i * 5) as u8);
            }
            for i in 0..128 {
                data[base + 48 + i] = seed.wrapping_add((block * 17 + i * 3) as u8);
            }
        }
        data
    }

    /// Builds deterministic synthetic Q5_K matrix weights.
    fn make_q5k_weights(rows: usize, cols: usize, seed: u8) -> Vec<u8> {
        let row_bytes = (cols / 256) * 176;
        let mut data = vec![0u8; rows * row_bytes];
        for row in 0..rows {
            let row_data = make_q5k_blocks(cols / 256, seed.wrapping_add(row as u8));
            data[row * row_bytes..(row + 1) * row_bytes].copy_from_slice(&row_data);
        }
        data
    }

    /// Builds deterministic synthetic Q6_K matrix weights.
    fn make_q6k_weights(rows: usize, cols: usize, seed: u8) -> Vec<u8> {
        let row_bytes = (cols / 256) * 210;
        let mut data = vec![0u8; rows * row_bytes];
        for row in 0..rows {
            for block in 0..(cols / 256) {
                let base = row * row_bytes + block * 210;
                for i in 0..128 {
                    data[base + i] = seed.wrapping_add((row * 7 + block * 13 + i * 3) as u8);
                }
                for i in 0..64 {
                    data[base + 128 + i] = seed.wrapping_add((row * 11 + block * 17 + i * 5) as u8);
                }
                for i in 0..16 {
                    data[base + 192 + i] = 1 + ((seed.wrapping_add(i as u8)) & 0x07);
                }
                data[base + 208] = 0x00;
                data[base + 209] = 0x3c; // f16 1.0
            }
        }
        data
    }

    #[test]
    /// Verifies Q5_0 high-bit unpacking against explicit expected values.
    fn q5_0_dequant_and_dot_unpack_high_bits() {
        let mut row = vec![0u8; 22];
        row[0] = 0x00;
        row[1] = 0x3c; // f16 1.0
        let qh = (1u32 << 0) | (1u32 << 15) | (1u32 << 16) | (1u32 << 31);
        row[2..6].copy_from_slice(&qh.to_le_bytes());
        for i in 0..16 {
            row[6 + i] = (i as u8 & 0x0f) | ((15 - i as u8) << 4);
        }

        let deq = dequant_row_q5_0(&row, 32);
        assert_eq!(deq[0], 0.0);
        assert_eq!(deq[16], 15.0);
        assert_eq!(deq[15], 15.0);
        assert_eq!(deq[31], 0.0);

        let x = vec![1.0f32; 32];
        let expected: f32 = deq.iter().sum();
        assert_eq!(dot_q5_0_f32(&row, &x, 32), expected);
    }

    #[test]
    /// Verifies Q5_1 high-bit unpacking and min handling.
    fn q5_1_dequant_and_dot_unpack_high_bits() {
        let mut row = vec![0u8; 24];
        row[0] = 0x00;
        row[1] = 0x3c; // f16 1.0
        row[2] = 0x00;
        row[3] = 0x38; // f16 0.5
        let qh = (1u32 << 0) | (1u32 << 16);
        row[4..8].copy_from_slice(&qh.to_le_bytes());
        for i in 0..16 {
            row[8 + i] = (i as u8 & 0x0f) | ((i as u8 & 0x0f) << 4);
        }

        let deq = dequant_row_q5_1(&row, 32);
        assert_eq!(deq[0], 16.5);
        assert_eq!(deq[16], 16.5);
        assert_eq!(deq[1], 1.5);
        assert_eq!(deq[17], 1.5);

        let x = vec![0.25f32; 32];
        let expected: f32 = deq.iter().map(|v| v * 0.25).sum();
        assert_eq!(dot_q5_1_f32(&row, &x, 32), expected);
    }

    #[test]
    /// Verifies Q5_K reads one high-bit slot per quantized value.
    fn q5k_dot_uses_high_bit_plane() {
        let mut row = vec![0u8; 176];
        row[0] = 0x00;
        row[1] = 0x3c; // f16 1.0
        row[2] = 0x00;
        row[3] = 0x00; // f16 0.0 dmin
        for scale in &mut row[4..16] {
            *scale = 1;
        }
        for high in &mut row[16..48] {
            *high = 0x03;
        }

        let x = vec![1.0f32; 256];
        assert_eq!(dot_q5_k_f32(&row, &x, 256), 16.0 * 64.0);
        let deq = dequant_row_q5_k(&row, 256);
        assert_eq!(deq.iter().sum::<f32>(), 16.0 * 64.0);
    }

    #[test]
    /// Verifies optimized Q5_K dot products preserve scalar output.
    fn q5k_dot_matches_scalar_reference() {
        let n = 512;
        let row = make_q5k_blocks(n / 256, 23);
        let x: Vec<f32> = (0..n)
            .map(|i| ((i as f32 * 0.013).sin() * 0.75) + ((i % 17) as f32 * 0.01))
            .collect();

        let expected = dot_q5_k_f32_scalar(&row, &x, n);
        let actual = dot_q5_k_f32(&row, &x, n);
        let tolerance = expected.abs().max(1.0) * 1e-5;
        assert!((actual - expected).abs() <= tolerance);
    }

    #[test]
    /// Verifies MXFP4 scale decoding uses the same exponent mapping as the scalar formula.
    fn mxfp4_scale_lookup_matches_power_formula() {
        for scale in [1u8, 64, 127, 128, 191, 254] {
            assert_eq!(mxfp4_scale_to_f32(scale), 2.0f32.powi(scale as i32 - 127));
        }
    }

    #[test]
    /// Verifies that fused Q4_K two-projection output matches separate projections.
    fn q4k_matvec2_matches_separate_matvecs() {
        set_num_threads(3);
        let cols = 512;
        let x: Vec<f32> = (0..cols)
            .map(|i| ((i as f32 * 0.017).cos() * 0.5) + ((i % 11) as f32 * 0.01))
            .collect();
        let a = make_q4k_weights(5, cols, 3);
        let b = make_q4k_weights(7, cols, 19);

        let mut exp_a = Vec::new();
        let mut exp_b = Vec::new();
        matvec_q4_k_into(&a, &x, 5, cols, &mut exp_a);
        matvec_q4_k_into(&b, &x, 7, cols, &mut exp_b);

        let mut got_a = Vec::new();
        let mut got_b = Vec::new();
        assert!(matvec_q4_k2_into(
            (&a, 5, cols),
            (&b, 7, cols),
            &x,
            &mut got_a,
            &mut got_b,
        ));

        assert_close_slice(&got_a, &exp_a, 1e-5);
        assert_close_slice(&got_b, &exp_b, 1e-5);
    }

    #[test]
    /// Verifies that fused Q4_K three-projection output matches separate projections.
    fn q4k_matvec3_matches_separate_matvecs() {
        set_num_threads(3);
        let cols = 512;
        let x: Vec<f32> = (0..cols)
            .map(|i| ((i as f32 * 0.013).sin() * 0.5) + ((i % 7) as f32 * 0.01))
            .collect();
        let a = make_q4k_weights(5, cols, 3);
        let b = make_q4k_weights(7, cols, 19);
        let c = make_q4k_weights(4, cols, 41);

        let mut exp_a = Vec::new();
        let mut exp_b = Vec::new();
        let mut exp_c = Vec::new();
        matvec_q4_k_into(&a, &x, 5, cols, &mut exp_a);
        matvec_q4_k_into(&b, &x, 7, cols, &mut exp_b);
        matvec_q4_k_into(&c, &x, 4, cols, &mut exp_c);

        let mut got_a = Vec::new();
        let mut got_b = Vec::new();
        let mut got_c = Vec::new();
        assert!(matvec_q4_k3_into(
            (&a, 5, cols),
            (&b, 7, cols),
            (&c, 4, cols),
            &x,
            &mut got_a,
            &mut got_b,
            &mut got_c
        ));

        assert_close_slice(&got_a, &exp_a, 1e-5);
        assert_close_slice(&got_b, &exp_b, 1e-5);
        assert_close_slice(&got_c, &exp_c, 1e-5);
    }

    #[test]
    /// Verifies that fused Q4_K triple projection rejects incompatible shapes.
    fn q4k_matvec3_rejects_incompatible_shapes() {
        let cols = 512;
        let x = vec![0.0f32; cols];
        let a = make_q4k_weights(1, cols, 1);
        let b = make_q4k_weights(1, cols, 2);
        let c = make_q4k_weights(1, cols, 3);
        let mut out_a = Vec::new();
        let mut out_b = Vec::new();
        let mut out_c = Vec::new();

        assert!(!matvec_q4_k3_into(
            (&a, 1, cols),
            (&b, 1, cols),
            (&c, 1, cols - 256),
            &x,
            &mut out_a,
            &mut out_b,
            &mut out_c
        ));
    }

    #[test]
    /// Verifies generic Q4_0 two-projection fusion against separate matvecs.
    fn q4_0_quant_matvec2_matches_separate_matvecs() {
        set_num_threads(3);
        let cols = 64;
        let x: Vec<f32> = (0..cols)
            .map(|i| ((i as f32 * 0.037).sin() * 0.3) + ((i % 9) as f32 * 0.02))
            .collect();
        let a = make_q4_0_weights(5, cols, 11);
        let b = make_q4_0_weights(7, cols, 29);

        let mut exp_a = Vec::new();
        let mut exp_b = Vec::new();
        matvec_q4_0_into(&a, &x, 5, cols, &mut exp_a);
        matvec_q4_0_into(&b, &x, 7, cols, &mut exp_b);

        let mut got_a = Vec::new();
        let mut got_b = Vec::new();
        assert!(matvec_quant2_into(
            (QuantMatvecKind::Q4_0, &a, 5, cols),
            (QuantMatvecKind::Q4_0, &b, 7, cols),
            &x,
            &mut got_a,
            &mut got_b
        ));

        assert_close_slice(&got_a, &exp_a, 1e-5);
        assert_close_slice(&got_b, &exp_b, 1e-5);
    }

    #[test]
    /// Verifies generic Q4_0 three-projection fusion against separate matvecs.
    fn q4_0_quant_matvec3_matches_separate_matvecs() {
        set_num_threads(3);
        let cols = 64;
        let x: Vec<f32> = (0..cols)
            .map(|i| ((i as f32 * 0.041).cos() * 0.25) + ((i % 13) as f32 * 0.015))
            .collect();
        let a = make_q4_0_weights(5, cols, 3);
        let b = make_q4_0_weights(7, cols, 19);
        let c = make_q4_0_weights(4, cols, 41);

        let mut exp_a = Vec::new();
        let mut exp_b = Vec::new();
        let mut exp_c = Vec::new();
        matvec_q4_0_into(&a, &x, 5, cols, &mut exp_a);
        matvec_q4_0_into(&b, &x, 7, cols, &mut exp_b);
        matvec_q4_0_into(&c, &x, 4, cols, &mut exp_c);

        let mut got_a = Vec::new();
        let mut got_b = Vec::new();
        let mut got_c = Vec::new();
        assert!(matvec_quant3_into(
            (QuantMatvecKind::Q4_0, &a, 5, cols),
            (QuantMatvecKind::Q4_0, &b, 7, cols),
            (QuantMatvecKind::Q4_0, &c, 4, cols),
            &x,
            &mut got_a,
            &mut got_b,
            &mut got_c
        ));

        assert_close_slice(&got_a, &exp_a, 1e-5);
        assert_close_slice(&got_b, &exp_b, 1e-5);
        assert_close_slice(&got_c, &exp_c, 1e-5);
    }

    #[test]
    /// Verifies fused Q5_K projections match separate projections.
    fn q5k_matvec_fusion_matches_separate_matvecs() {
        let cols = 512;
        let x: Vec<f32> = (0..cols)
            .map(|i| ((i as f32 * 0.019).sin() * 0.25) + ((i % 13) as f32 * 0.02))
            .collect();
        let a = make_q5k_weights(4, cols, 5);
        let b = make_q5k_weights(6, cols, 17);
        let c = make_q5k_weights(3, cols, 29);

        let mut exp_a = Vec::new();
        let mut exp_b = Vec::new();
        let mut exp_c = Vec::new();
        matvec_q5_k_into(&a, &x, 4, cols, &mut exp_a);
        matvec_q5_k_into(&b, &x, 6, cols, &mut exp_b);
        matvec_q5_k_into(&c, &x, 3, cols, &mut exp_c);

        let mut out_a = Vec::new();
        let mut out_b = Vec::new();
        let mut out_c = Vec::new();
        assert!(matvec_q5_k3_into(
            (&a, 4, cols),
            (&b, 6, cols),
            (&c, 3, cols),
            &x,
            &mut out_a,
            &mut out_b,
            &mut out_c
        ));
        assert_eq!(out_a, exp_a);
        assert_eq!(out_b, exp_b);
        assert_eq!(out_c, exp_c);
    }

    #[test]
    /// Verifies Q6_K fused dot matches explicit dequantization.
    fn q6k_dot_matches_reference_dequant() {
        let cols = 512;
        let weights = make_q6k_weights(1, cols, 17);
        let x: Vec<f32> = (0..cols)
            .map(|i| ((i as f32 * 0.019).sin() * 0.35) - ((i % 11) as f32 * 0.017))
            .collect();
        let deq = dequant_row_q6_k(&weights, cols);
        let reference: f32 = deq.iter().zip(&x).map(|(w, xv)| w * xv).sum();
        let fused = dot_q6_k_f32(&weights, &x, cols);
        assert!(
            (fused - reference).abs() < 1e-2,
            "fused {fused} vs reference {reference}"
        );
    }

    #[test]
    /// Verifies fused Q6_K projections match separate projections.
    fn q6k_matvec_fusion_matches_separate_matvecs() {
        let cols = 512;
        let x: Vec<f32> = (0..cols)
            .map(|i| ((i as f32 * 0.023).cos() * 0.2) + ((i % 7) as f32 * 0.03))
            .collect();
        let a = make_q6k_weights(4, cols, 7);
        let b = make_q6k_weights(5, cols, 19);

        let mut exp_a = Vec::new();
        let mut exp_b = Vec::new();
        matvec_q6_k_into(&a, &x, 4, cols, &mut exp_a);
        matvec_q6_k_into(&b, &x, 5, cols, &mut exp_b);

        let mut out_a = Vec::new();
        let mut out_b = Vec::new();
        assert!(matvec_q6_k2_into(
            (&a, 4, cols),
            (&b, 5, cols),
            &x,
            &mut out_a,
            &mut out_b
        ));
        assert_eq!(out_a, exp_a);
        assert_eq!(out_b, exp_b);
    }

    #[test]
    /// Verifies mixed K-quant fusion covers Ministral-style Q4/Q4/Q6 attention.
    fn mixed_kquant_matvec3_matches_separate_matvecs() {
        let cols = 512;
        let x: Vec<f32> = (0..cols)
            .map(|i| ((i as f32 * 0.031).sin() * 0.2) + ((i % 5) as f32 * 0.04))
            .collect();
        let q = make_q4k_weights(4, cols, 3);
        let k = make_q4k_weights(2, cols, 13);
        let v = make_q6k_weights(2, cols, 23);

        let mut exp_q = Vec::new();
        let mut exp_k = Vec::new();
        let mut exp_v = Vec::new();
        matvec_q4_k_into(&q, &x, 4, cols, &mut exp_q);
        matvec_q4_k_into(&k, &x, 2, cols, &mut exp_k);
        matvec_q6_k_into(&v, &x, 2, cols, &mut exp_v);

        let mut out_q = Vec::new();
        let mut out_k = Vec::new();
        let mut out_v = Vec::new();
        assert!(matvec_kquant3_into(
            (KQuantMatvecKind::Q4K, &q, 4, cols),
            (KQuantMatvecKind::Q4K, &k, 2, cols),
            (KQuantMatvecKind::Q6K, &v, 2, cols),
            &x,
            &mut out_q,
            &mut out_k,
            &mut out_v
        ));
        assert_close_slice(&out_q, &exp_q, 1e-5);
        assert_close_slice(&out_k, &exp_k, 1e-5);
        assert_close_slice(&out_v, &exp_v, 1e-5);
    }

    fn assert_close_slice(actual: &[f32], expected: &[f32], relative: f32) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let tolerance = expected.abs().max(1.0) * relative;
            assert!(
                (*actual - *expected).abs() <= tolerance,
                "value[{idx}] actual {actual} expected {expected} tolerance {tolerance}"
            );
        }
    }
}
