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

use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(not(target_family = "wasm"))]
use std::sync::{Arc, Condvar, Mutex, OnceLock};
#[cfg(not(target_family = "wasm"))]
use std::thread;

static NUM_THREADS: AtomicUsize = AtomicUsize::new(0);

#[cfg(target_arch = "x86_64")]
#[inline]
fn has_avx2_fma() -> bool {
    static HAS_AVX2_FMA: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *HAS_AVX2_FMA
        .get_or_init(|| is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"))
}

// ─── f16 ↔ f32 conversion ────────────────────────────────────────────────────

#[inline(always)]
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
        #[cfg(not(target_family = "wasm"))]
        {
            static DEFAULT_THREADS: OnceLock<usize> = OnceLock::new();
            *DEFAULT_THREADS.get_or_init(|| {
                thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1)
            })
        }
        #[cfg(target_family = "wasm")]
        {
            1
        }
    }
}

#[cfg(not(target_family = "wasm"))]
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
    Q4_0,
    Q4K,
    Q6K,
    Mxfp4,
}

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
        MatvecKind::Q4_0 => {
            let row = std::slice::from_raw_parts(row_ptr, row_span);
            dot_q4_0_f32(row, x, cols)
        }
        MatvecKind::Q4K => {
            let row = std::slice::from_raw_parts(row_ptr, row_span);
            dot_q4_k_f32(row, x, cols)
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
impl MatvecJob {
    unsafe fn run_worker(self, worker_idx: usize) {
        let start = self.rows * worker_idx / self.workers;
        let end = self.rows * (worker_idx + 1) / self.workers;
        for row in start..end {
            *self.out.add(row) =
                dot_row(self.kind, self.data, self.x, row, self.cols, self.row_span);
        }
    }
}

#[cfg(not(target_family = "wasm"))]
#[derive(Clone, Copy)]
struct Q4KMatvec3Job {
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
    row_span: usize,
    workers: usize,
}

#[cfg(not(target_family = "wasm"))]
unsafe impl Send for Q4KMatvec3Job {}
#[cfg(not(target_family = "wasm"))]
unsafe impl Sync for Q4KMatvec3Job {}

#[cfg(not(target_family = "wasm"))]
impl Q4KMatvec3Job {
    #[inline]
    fn work_items(self) -> usize {
        self.rows_a + self.rows_b + self.rows_c
    }

    unsafe fn run_worker(self, worker_idx: usize) {
        let total = self.work_items();
        let start = total * worker_idx / self.workers;
        let end = total * (worker_idx + 1) / self.workers;

        let (a_start, a_end) = clipped_range(start, end, 0, self.rows_a);
        q4k_matvec3_rows(
            self.a_data,
            self.x,
            self.out_a,
            self.cols,
            self.row_span,
            a_start,
            a_end,
        );

        let b_offset = self.rows_a;
        let (b_start, b_end) = clipped_range(start, end, b_offset, self.rows_b);
        q4k_matvec3_rows(
            self.b_data,
            self.x,
            self.out_b,
            self.cols,
            self.row_span,
            b_start,
            b_end,
        );

        let c_offset = self.rows_a + self.rows_b;
        let (c_start, c_end) = clipped_range(start, end, c_offset, self.rows_c);
        q4k_matvec3_rows(
            self.c_data,
            self.x,
            self.out_c,
            self.cols,
            self.row_span,
            c_start,
            c_end,
        );
    }
}

#[cfg(not(target_family = "wasm"))]
#[inline]
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
unsafe fn q4k_matvec3_rows(
    data: *const u8,
    x: *const f32,
    out: *mut f32,
    cols: usize,
    row_span: usize,
    start: usize,
    end: usize,
) {
    for row in start..end {
        *out.add(row) = dot_row(MatvecKind::Q4K, data, x, row, cols, row_span);
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
    fn workers(self) -> usize {
        match self {
            WorkerJob::Matvec(job) => job.workers,
            WorkerJob::Q4KMatvec3(job) => job.workers,
        }
    }

    #[inline]
    unsafe fn run_worker(self, worker_idx: usize) {
        match self {
            WorkerJob::Matvec(job) => job.run_worker(worker_idx),
            WorkerJob::Q4KMatvec3(job) => job.run_worker(worker_idx),
        }
    }
}

#[cfg(not(target_family = "wasm"))]
struct WorkerState {
    job: Option<WorkerJob>,
    job_id: u64,
    completed: AtomicUsize,
}

#[cfg(not(target_family = "wasm"))]
struct WorkerPool {
    state: Mutex<WorkerState>,
    work_available: Condvar,
    max_workers: usize,
}

#[cfg(not(target_family = "wasm"))]
impl WorkerPool {
    fn new(max_workers: usize) -> Arc<Self> {
        let pool = Arc::new(Self {
            state: Mutex::new(WorkerState {
                job: None,
                job_id: 0,
                completed: AtomicUsize::new(0),
            }),
            work_available: Condvar::new(),
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

    fn run_q4k_matvec3(&self, mut job: Q4KMatvec3Job) {
        let rows = job.work_items();
        job.workers = job.workers.min(self.max_workers).min(rows).max(1);
        if job.workers <= 1 || rows < job.workers * 8 {
            job.workers = 1;
            unsafe {
                job.run_worker(0);
            }
            return;
        }

        self.run_job(WorkerJob::Q4KMatvec3(job), job.workers);
    }

    fn run_job(&self, job: WorkerJob, workers: usize) {
        // Fast spin-wait if previous job is finishing (usually very quick)
        loop {
            let mut state = self.state.lock().expect("worker pool mutex poisoned");
            if state.job.is_none() {
                state.job_id = state.job_id.wrapping_add(1);
                state.completed.store(0, Ordering::Release);
                state.job = Some(job);
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
            let state = self.state.lock().expect("worker pool mutex poisoned");
            if state.completed.load(Ordering::Acquire) == workers {
                break;
            }
            drop(state);

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
fn worker_loop(pool: Arc<WorkerPool>, worker_idx: usize) {
    let mut last_job_id = 0u64;
    loop {
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
                job.run_worker(worker_idx);
            }
            let state = pool.state.lock().expect("worker pool mutex poisoned");
            state.completed.fetch_add(1, Ordering::Release);
        }
    }
}

#[cfg(not(target_family = "wasm"))]
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

/// out[i] += alpha * x[i]
#[inline]
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

/// out[i] *= scale
#[inline]
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

/// out[i] = out[i] * scale + add[i]
#[inline]
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
fn axpy_f32_scalar(out: &mut [f32], alpha: f32, x: &[f32]) {
    for (o, xi) in out.iter_mut().zip(x.iter()) {
        *o += alpha * *xi;
    }
}

#[inline]
#[allow(dead_code)]
fn scale_f32_scalar(out: &mut [f32], scale: f32) {
    for o in out.iter_mut() {
        *o *= scale;
    }
}

#[inline]
#[allow(dead_code)]
fn scale_add_f32_scalar(out: &mut [f32], scale: f32, add: &[f32]) {
    for (o, a) in out.iter_mut().zip(add.iter()) {
        *o = *o * scale + *a;
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
    parallel_matvec_u8(MatvecKind::Q6K, &mut out, rows, cols, row_bytes, qweight, x);
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

pub fn matvec_q8_0_into(qweight: &[u8], x: &[f32], rows: usize, cols: usize, out: &mut Vec<f32>) {
    let row_bytes = (cols / 32) * 34;
    out.resize(rows, 0.0);
    parallel_matvec_u8(MatvecKind::Q8_0, out, rows, cols, row_bytes, qweight, x);
}

pub fn matvec_q4_0_into(qweight: &[u8], x: &[f32], rows: usize, cols: usize, out: &mut Vec<f32>) {
    let row_bytes = (cols / 32) * 18;
    out.resize(rows, 0.0);
    parallel_matvec_u8(MatvecKind::Q4_0, out, rows, cols, row_bytes, qweight, x);
}

pub fn matvec_q4_k_into(qweight: &[u8], x: &[f32], rows: usize, cols: usize, out: &mut Vec<f32>) {
    let row_bytes = (cols / 256) * 144;
    out.resize(rows, 0.0);
    #[cfg(not(target_family = "wasm"))]
    if crate::metal::q4k_matvec_into(qweight, x, rows, cols, out) {
        return;
    }
    parallel_matvec_u8(MatvecKind::Q4K, out, rows, cols, row_bytes, qweight, x);
}

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
            row_span: row_bytes,
            workers,
        });
        true
    }
}

pub fn matvec_q6_k_into(qweight: &[u8], x: &[f32], rows: usize, cols: usize, out: &mut Vec<f32>) {
    let row_bytes = (cols / 256) * 210;
    out.resize(rows, 0.0);
    parallel_matvec_u8(MatvecKind::Q6K, out, rows, cols, row_bytes, qweight, x);
}

pub fn matvec_mxfp4_into(qweight: &[u8], x: &[f32], rows: usize, cols: usize, out: &mut Vec<f32>) {
    let row_bytes = (cols / 32) * 17;
    out.resize(rows, 0.0);
    parallel_matvec_u8(MatvecKind::Mxfp4, out, rows, cols, row_bytes, qweight, x);
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
fn mxfp4_nibble_to_f32(v: u8) -> f32 {
    const LUT: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    LUT[(v & 0x0F) as usize]
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

    #[test]
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

        assert_eq!(got_a, exp_a);
        assert_eq!(got_b, exp_b);
        assert_eq!(got_c, exp_c);
    }

    #[test]
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
}
