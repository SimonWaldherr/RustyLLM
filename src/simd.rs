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

#[cfg(not(target_family = "wasm"))]
use std::cell::UnsafeCell;
#[cfg(not(target_family = "wasm"))]
use std::sync::atomic::AtomicU64;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
#[cfg(not(target_family = "wasm"))]
use std::sync::{Arc, Mutex, OnceLock};
#[cfg(not(target_family = "wasm"))]
use std::thread;

static NUM_THREADS: AtomicUsize = AtomicUsize::new(0);
static CPU_AFFINITY: AtomicBool = AtomicBool::new(false);
static WORKER_POLL_SPINS: AtomicUsize = AtomicUsize::new(2_000);
#[cfg(not(target_family = "wasm"))]
const MATVEC_CHUNK_ROWS: usize = 64;
#[cfg(not(target_family = "wasm"))]
const KQUANT_BATCH_CHUNK_ROWS: usize = 32;
/// How many tokens the batched K-quant kernels keep in flight against one
/// weight row. Unpacking a K-quant block — f16 scale conversion, 6-bit
/// scale/min extraction, nibble or 2-bit-plane reassembly — costs more vector
/// work than the `maddubs`/`sdot` products it feeds, and none of it depends on
/// the activation, so paying it once per four tokens instead of once per token
/// is what makes a prefill chunk cheaper than the same tokens run singly.
/// Four lanes keep the per-lane accumulators plus the unpacked weights inside
/// the 16 AVX2 registers. Eight was measured and rejected: the eight
/// `__m256i` accumulators spill, and end-to-end prefill dropped from 26.9 to
/// 19.2 tok/s on the i7-10850H (203-token prompt, paired round). Raising this
/// only makes sense together with a register-blocked kernel that reduces its
/// accumulator count, e.g. by narrowing to i16 before the widening add.
#[cfg(not(target_family = "wasm"))]
const KQUANT_BATCH_TOKENS: usize = 4;
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

// ─── bf16 ↔ f32 conversion ────────────────────────────────────────────────────

#[inline(always)]
/// Widens a bf16 bit pattern to `f32` (bf16 is just the top 16 bits of f32).
pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

#[inline(always)]
/// Narrows `f32` to bf16 with round-to-nearest-even, preserving NaN payloads.
pub fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        // Force the quiet bit so a narrowed NaN never turns into +/-Inf.
        return ((bits >> 16) as u16) | 0x0040;
    }
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    (rounded >> 16) as u16
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

/// Counts physical processor cores on Windows via
/// `GetLogicalProcessorInformationEx(RelationProcessorCore)`; each returned
/// record is one core.
#[cfg(all(windows, not(target_family = "wasm")))]
fn physical_cores() -> Option<usize> {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetLogicalProcessorInformationEx(
            relationship: u32,
            buffer: *mut u8,
            returned_length: *mut u32,
        ) -> i32;
    }
    const RELATION_PROCESSOR_CORE: u32 = 0;
    unsafe {
        let mut len: u32 = 0;
        GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, std::ptr::null_mut(), &mut len);
        if len == 0 {
            return None;
        }
        let mut buf = vec![0u8; len as usize];
        if GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, buf.as_mut_ptr(), &mut len)
            == 0
        {
            return None;
        }
        // Walk the variable-size SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX
        // records: u32 Relationship, u32 Size, then a relationship union.
        let mut count = 0usize;
        let mut off = 0usize;
        while off + 8 <= len as usize {
            let size = u32::from_le_bytes(buf[off + 4..off + 8].try_into().ok()?) as usize;
            if size == 0 {
                return None;
            }
            count += 1;
            off += size;
        }
        if count > 0 { Some(count) } else { None }
    }
}

/// Counts physical cores on Linux from unique `(physical id, core id)` pairs
/// in `/proc/cpuinfo`; returns `None` when the fields are absent (e.g. ARM).
#[cfg(all(target_os = "linux", not(target_family = "wasm")))]
fn physical_cores() -> Option<usize> {
    let info = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    let mut pairs = std::collections::HashSet::new();
    let mut phys: Option<u32> = None;
    let mut core: Option<u32> = None;
    let flush = |phys: &mut Option<u32>,
                 core: &mut Option<u32>,
                 pairs: &mut std::collections::HashSet<(u32, u32)>| {
        if let (Some(p), Some(c)) = (phys.take(), core.take()) {
            pairs.insert((p, c));
        }
    };
    for line in info.lines() {
        if line.trim().is_empty() {
            flush(&mut phys, &mut core, &mut pairs);
            continue;
        }
        let mut split = line.splitn(2, ':');
        let key = split.next().unwrap_or("").trim();
        let value = split.next().unwrap_or("").trim();
        match key {
            "physical id" => phys = value.parse().ok(),
            "core id" => core = value.parse().ok(),
            _ => {}
        }
    }
    flush(&mut phys, &mut core, &mut pairs);
    if pairs.is_empty() {
        None
    } else {
        Some(pairs.len())
    }
}

#[cfg(all(target_os = "macos", not(target_family = "wasm")))]
unsafe extern "C" {
    fn sysctlbyname(
        name: *const std::ffi::c_char,
        oldp: *mut std::ffi::c_void,
        oldlenp: *mut usize,
        newp: *const std::ffi::c_void,
        newlen: usize,
    ) -> i32;
}

/// Reads a single `i32` sysctl by name, returning `None` when the key is absent
/// (older macOS) or non-positive.
#[cfg(all(target_os = "macos", not(target_family = "wasm")))]
fn sysctl_i32(name: &std::ffi::CStr) -> Option<usize> {
    let mut value: i32 = 0;
    let mut len = std::mem::size_of::<i32>();
    let rc = unsafe {
        sysctlbyname(
            name.as_ptr(),
            &mut value as *mut i32 as *mut std::ffi::c_void,
            &mut len,
            std::ptr::null(),
            0,
        )
    };
    if rc == 0 && value > 0 {
        Some(value as usize)
    } else {
        None
    }
}

/// Counts every physical core, including Apple Silicon efficiency cores. This
/// is the ceiling for prefill thread widening: batched prefill is compute-bound
/// and dynamically chunked (see [`KQuantBatchSchedule::Rows`]), so a slower
/// E-core steals fewer chunks rather than stalling a barrier, and its
/// throughput still adds.
#[cfg(all(target_os = "macos", not(target_family = "wasm")))]
fn physical_cores() -> Option<usize> {
    sysctl_i32(c"hw.physicalcpu")
}

/// Counts the cores worth scheduling *decode* matvec shards on.
///
/// On Apple Silicon `hw.physicalcpu` counts performance **and** efficiency
/// cores (12 on an M2 Max: 8 Avalanche + 4 Blizzard), but an E-core has half
/// the NEON issue width at a lower clock, so a shard placed on one takes
/// roughly 3x as long. Decode dispatches are small and every one waits for its
/// slowest shard, so including E-cores sets the barrier to E-core speed and
/// makes the projection slower than using the P-cores alone. macOS 12+ exposes
/// the split through `hw.perflevel0.physicalcpu` (performance level 0 is the
/// fastest tier), which is the same key llama.cpp selects its default thread
/// count from. On a homogeneous CPU there is a single performance level, so this
/// key reports the full physical count and the result is unchanged from
/// `hw.physicalcpu`; the fallback covers macOS releases predating the key.
#[cfg(all(target_os = "macos", not(target_family = "wasm")))]
fn performance_cores() -> Option<usize> {
    sysctl_i32(c"hw.perflevel0.physicalcpu").or_else(physical_cores)
}

/// Outside Apple Silicon every core is equivalent for these kernels, so the
/// decode default and the prefill ceiling share one physical-core count.
#[cfg(all(not(target_os = "macos"), not(target_family = "wasm")))]
fn performance_cores() -> Option<usize> {
    physical_cores()
}

#[cfg(all(
    not(windows),
    not(target_os = "linux"),
    not(target_os = "macos"),
    not(target_family = "wasm")
))]
fn physical_cores() -> Option<usize> {
    None
}

/// Default matvec worker count: one per performance core. The quantized dot
/// kernels saturate the vector units, so SMT siblings contend rather than
/// help (measured: Ministral-3B decode 3.64 t/s at 6 threads vs 3.18 at 12 on
/// a 6C/12T i7-10850H), and on Apple Silicon an efficiency-core shard would set
/// the barrier for every dispatch. `--threads` still overrides.
#[cfg(not(target_family = "wasm"))]
fn default_worker_threads() -> usize {
    // Cached: `performance_cores()` is a `sysctl` syscall on macOS.
    static DEFAULT: OnceLock<usize> = OnceLock::new();
    *DEFAULT.get_or_init(|| {
        // Clamped to the parallelism the OS will actually give us. The core
        // counts come from hardware topology (sysctl, /proc/cpuinfo), which
        // ignores cpuset/affinity restrictions, so inside a two-CPU container on
        // a many-core host the raw count would oversubscribe the pool badly.
        performance_cores()
            .unwrap_or_else(available_threads)
            .min(available_threads())
            .max(1)
    })
}

/// Ceiling for prefill batch-thread widening: one thread per physical core.
///
/// This counts *all* physical cores, unlike the decode default, which on Apple
/// Silicon uses performance cores only. Prefill runs far more work per dispatch
/// and steals chunks dynamically, so an efficiency core contributes throughput
/// there instead of setting a barrier. On an M2 Max that means decode defaults
/// to 8 threads while a long prompt widens to 12 — the first time the widening
/// path does anything on Apple Silicon, since it previously compared 12 to 12.
pub fn physical_threads() -> usize {
    #[cfg(not(target_family = "wasm"))]
    {
        static PHYSICAL: OnceLock<usize> = OnceLock::new();
        *PHYSICAL.get_or_init(|| {
            physical_cores()
                .unwrap_or_else(available_threads)
                .min(available_threads())
                .max(1)
        })
    }
    #[cfg(target_family = "wasm")]
    {
        1
    }
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

/// Apple Silicon has no thread-pinning API: `THREAD_AFFINITY_POLICY` is gated
/// on `ml_get_max_affinity_sets() != 0` in XNU, which is 0 on arm64, so
/// `thread_policy_set` returns `KERN_NOT_SUPPORTED` there. Quality of Service is
/// the supported way to bias placement on that hardware, so `--cpu-affinity`
/// maps onto QoS on arm64 Macs.
///
/// It stays behind the flag rather than becoming the default because llama.cpp
/// gets its Apple Silicon placement purely from sizing the pool to the
/// performance cores and sets no QoS class at all, and this build has no Apple
/// hardware to measure the difference on.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn pin_current_thread_impl(_worker_idx: usize) -> bool {
    apply_compute_thread_qos()
}

/// Intel Macs do support affinity sets, so keep the original behaviour there.
/// The tag is an L2-sharing hint rather than a pin: distinct tags ask the
/// scheduler to spread these threads across separate cache groups, which suits
/// workers that stream disjoint weight ranges.
#[cfg(all(target_os = "macos", not(target_arch = "aarch64")))]
fn pin_current_thread_impl(worker_idx: usize) -> bool {
    const THREAD_AFFINITY_POLICY: i32 = 4;
    let tag = (worker_idx as i32).saturating_add(1);
    unsafe {
        let thread = mach_thread_self();
        thread_policy_set(thread, THREAD_AFFINITY_POLICY, &tag as *const i32, 1) == 0
    }
}

#[cfg(all(target_os = "macos", not(target_arch = "aarch64")))]
unsafe extern "C" {
    fn mach_thread_self() -> u32;
    fn thread_policy_set(thread: u32, flavor: i32, policy_info: *const i32, count: u32) -> i32;
}

#[cfg(not(target_os = "macos"))]
fn pin_current_thread_impl(_worker_idx: usize) -> bool {
    false
}

/// Requests performance-core placement for the calling compute thread.
///
/// On Apple Silicon the scheduler picks between performance and efficiency
/// cores partly from a thread's QoS class: `QOS_CLASS_BACKGROUND` is confined to
/// the efficiency cores, while the higher classes are eligible for the
/// performance cores. Because every pool dispatch waits for its slowest shard,
/// one shard on an E-core slows the entire projection, so a pool thread asks for
/// `QOS_CLASS_USER_INITIATED` — high enough to be preferred on the performance
/// cores, without claiming the UI-critical `QOS_CLASS_USER_INTERACTIVE` band.
///
/// This is a scheduler hint, not a pin: macOS may still migrate the thread. It
/// biases placement, which is the most Apple offers here. Returns whether the
/// request itself was accepted.
#[cfg(all(
    target_os = "macos",
    target_arch = "aarch64",
    not(target_family = "wasm")
))]
fn apply_compute_thread_qos() -> bool {
    /// `QOS_CLASS_USER_INITIATED` from `<sys/qos.h>`'s `qos_class_t`
    /// (`USER_INTERACTIVE` is `0x21`, `DEFAULT` is `0x15`).
    const QOS_CLASS_USER_INITIATED: u32 = 0x19;
    unsafe extern "C" {
        fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
    }
    unsafe { pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0) == 0 }
}

#[cfg(all(
    target_os = "macos",
    target_arch = "aarch64",
    not(target_family = "wasm")
))]
thread_local! {
    static COMPUTE_QOS_APPLIED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Applies the compute QoS request once per dispatching thread.
///
/// The pool's shard 0 runs on whichever thread dispatched the job, so under
/// `--cpu-affinity` that thread needs performance-core placement just as much as
/// the helpers do. Doing it lazily here covers the CLI's decode thread and each
/// of the server's per-connection threads without new public API.
///
/// Note that this raises the QoS of a thread the pool does not own, for the rest
/// of that thread's life. `--cpu-affinity` is an explicit request to place this
/// engine's compute threads, so that is within what the caller asked for, but it
/// is why the behaviour is not on by default.
#[cfg(all(
    target_os = "macos",
    target_arch = "aarch64",
    not(target_family = "wasm")
))]
#[inline]
fn ensure_compute_thread_qos() {
    if !cpu_affinity_enabled() {
        return;
    }
    COMPUTE_QOS_APPLIED.with(|applied| {
        if !applied.get() {
            applied.set(true);
            let _ = apply_compute_thread_qos();
        }
    });
}

#[cfg(all(
    not(all(target_os = "macos", target_arch = "aarch64")),
    not(target_family = "wasm")
))]
#[inline(always)]
fn ensure_compute_thread_qos() {}

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
            *DEFAULT_THREADS.get_or_init(default_worker_threads)
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

/// Borrowed view of an activation vector pre-quantized to int8 (Q8_K layout:
/// one f32 scale per 256-element super-block plus per-32 group sums). The
/// pointers reference a per-thread scratch buffer that the worker fills once
/// per matrix-vector job, so the K-quant dot kernels can use integer `sdot`
/// products instead of widening every weight to f32. A null `qs` means the
/// fast path is unavailable (no CPU dotprod support, or a non-K matvec) and
/// callers fall back to the f32 kernels.
// The fields/`present` are only read by the aarch64 sdot kernels; on other
// targets `XQuant` is always `NONE` and threaded through as a placeholder.
#[derive(Clone, Copy)]
#[allow(dead_code)]
struct XQuant {
    qs: *const i8,
    d: *const f32,
    bsums: *const i16,
}

#[allow(dead_code)]
impl XQuant {
    const NONE: XQuant = XQuant {
        qs: std::ptr::null(),
        d: std::ptr::null(),
        bsums: std::ptr::null(),
    };

    #[inline(always)]
    fn present(&self) -> bool {
        !self.qs.is_null()
    }
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

    #[inline]
    fn from_kquant(kind: KQuantMatvecKind) -> Self {
        match kind {
            KQuantMatvecKind::Q4K => Self::Q4K,
            KQuantMatvecKind::Q5K => Self::Q5K,
            KQuantMatvecKind::Q6K => Self::Q6K,
        }
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
        let xq = unsafe { prepare_xq(kind, x, cols) };
        for r in 0..rows {
            out[r] = unsafe { dot_row(kind, data, x, xq, r, cols, row_span) };
        }
        return;
    }

    #[cfg(target_family = "wasm")]
    {
        for r in 0..rows {
            out[r] = unsafe { dot_row(kind, data, x, XQuant::NONE, r, cols, row_span) };
        }
        return;
    }

    #[cfg(not(target_family = "wasm"))]
    {
        worker_pool().run(MatvecJob {
            kind,
            data,
            x,
            xq: XQuant::NONE,
            out: out.as_mut_ptr(),
            rows,
            cols,
            row_span,
            workers: threads,
        });
    }
}

/// Reports whether a matvec kind has an integer-dot kernel that consumes the
/// Q8_K-quantized activation scratch.
#[inline]
fn matvec_kind_uses_xq(kind: MatvecKind) -> bool {
    matches!(
        kind,
        MatvecKind::Q4K | MatvecKind::Q5K | MatvecKind::Q6K | MatvecKind::Q4_0 | MatvecKind::Q8_0
    )
}

/// Runs `func(ctx, start, end)` over disjoint sub-ranges of `0..len` on the
/// shared worker pool, blocking until all complete. Lets higher layers
/// parallelize embarrassingly-parallel loops (e.g. attention over KV heads)
/// with the same persistent threads used for matvecs.
///
/// # Safety
/// `func` runs on worker threads with the raw `ctx` pointer; the caller must
/// keep whatever `ctx` points at valid for the duration (this call blocks, so
/// a stack local is fine) and ensure each `[start, end)` range only touches
/// disjoint state (no data races across ranges).
pub unsafe fn parallel_range(len: usize, func: unsafe fn(*const (), usize, usize), ctx: *const ()) {
    if len == 0 {
        return;
    }
    #[cfg(target_family = "wasm")]
    {
        func(ctx, 0, len);
    }
    #[cfg(not(target_family = "wasm"))]
    {
        // A static even split finishes when its largest shard does, so workers
        // past the point where `ceil(len / workers)` stops shrinking add no
        // speed and only cost power and thermal headroom. Attention over 8 KV
        // heads on 6 threads runs two rounds either way, so it takes 4 workers
        // and leaves the other cores their clock budget.
        let threads = num_threads().min(len);
        let per_worker = len.div_ceil(threads.max(1));
        let workers = len.div_ceil(per_worker.max(1)).max(1);
        worker_pool().run_range(ctx, func, len, workers);
    }
}

/// Quantizes the activation vector to int8 for the integer fast path, but only
/// for kinds that have an integer-dot kernel (K-quants plus Q4_0/Q8_0). Returns
/// [`XQuant::NONE`] on platforms or CPUs without the fast path so callers
/// transparently fall back to the f32 kernels.
#[inline]
unsafe fn prepare_xq(kind: MatvecKind, x: *const f32, cols: usize) -> XQuant {
    if matvec_kind_uses_xq(kind) {
        prepare_xq_kquant(x, cols)
    } else {
        XQuant::NONE
    }
}

/// Quantizes the activation vector to the Q8_K layout for any K-quant matvec.
/// Used by the fused triple/double projections, whose sub-matrices share one
/// activation vector but may mix Q4_K/Q5_K/Q6_K (or Q4_0/Q8_0).
#[inline]
unsafe fn prepare_xq_kquant(x: *const f32, cols: usize) -> XQuant {
    #[cfg(all(target_arch = "aarch64", not(target_family = "wasm")))]
    {
        if cols >= 256 && cols % 256 == 0 && has_dotprod() {
            return fill_thread_xq_q8k(std::slice::from_raw_parts(x, cols));
        }
    }
    #[cfg(all(target_arch = "x86_64", not(target_family = "wasm")))]
    {
        if cols >= 256 && cols % 256 == 0 && has_avx2_fma() {
            return fill_thread_xq_q8k(std::slice::from_raw_parts(x, cols));
        }
    }
    let _ = (x, cols);
    XQuant::NONE
}

#[inline]
unsafe fn dot_row(
    kind: MatvecKind,
    data: *const u8,
    x: *const f32,
    xq: XQuant,
    row: usize,
    cols: usize,
    row_span: usize,
) -> f32 {
    let row_ptr = data.add(row * row_span);
    let x = std::slice::from_raw_parts(x, cols);
    // `xq` is only consumed by the aarch64 K-quant fast paths.
    let _ = xq;
    match kind {
        MatvecKind::F32 => {
            let row = std::slice::from_raw_parts(row_ptr as *const f32, cols);
            dot_f32(row, x)
        }
        MatvecKind::Q8_0 => {
            let row = std::slice::from_raw_parts(row_ptr, row_span);
            #[cfg(all(target_arch = "aarch64", not(target_family = "wasm")))]
            if xq.present() {
                return dot_q8_0_q8k_neon(row, xq, cols);
            }
            #[cfg(all(target_arch = "x86_64", not(target_family = "wasm")))]
            if xq.present() {
                return dot_q8_0_q8k_avx2(row, xq, cols);
            }
            dot_q8_0_f32(row, x, cols)
        }
        MatvecKind::Q8_1 => {
            let row = std::slice::from_raw_parts(row_ptr, row_span);
            dot_q8_1_f32(row, x, cols)
        }
        MatvecKind::Q4_0 => {
            let row = std::slice::from_raw_parts(row_ptr, row_span);
            #[cfg(all(target_arch = "aarch64", not(target_family = "wasm")))]
            if xq.present() {
                return dot_q4_0_q8k_neon(row, xq, cols);
            }
            #[cfg(all(target_arch = "x86_64", not(target_family = "wasm")))]
            if xq.present() {
                return dot_q4_0_q8k_avx2(row, xq, cols);
            }
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
            #[cfg(all(target_arch = "aarch64", not(target_family = "wasm")))]
            if xq.present() {
                return dot_q4_k_q8k_neon(row, xq, cols);
            }
            #[cfg(all(target_arch = "x86_64", not(target_family = "wasm")))]
            if xq.present() {
                return dot_q4_k_q8k_avx2(row, xq, cols);
            }
            dot_q4_k_f32(row, x, cols)
        }
        MatvecKind::Q5K => {
            let row = std::slice::from_raw_parts(row_ptr, row_span);
            #[cfg(all(target_arch = "aarch64", not(target_family = "wasm")))]
            if xq.present() {
                return dot_q5_k_q8k_neon(row, xq, cols);
            }
            #[cfg(all(target_arch = "x86_64", not(target_family = "wasm")))]
            if xq.present() {
                return dot_q5_k_q8k_avx2(row, xq, cols);
            }
            dot_q5_k_f32(row, x, cols)
        }
        MatvecKind::Q6K => {
            let row = std::slice::from_raw_parts(row_ptr, row_span);
            #[cfg(all(target_arch = "aarch64", not(target_family = "wasm")))]
            if xq.present() {
                return dot_q6_k_q8k_neon(row, xq, cols);
            }
            #[cfg(all(target_arch = "x86_64", not(target_family = "wasm")))]
            if xq.present() {
                return dot_q6_k_q8k_avx2(row, xq, cols);
            }
            dot_q6_k_f32(row, x, cols)
        }
        MatvecKind::Mxfp4 => {
            let row = std::slice::from_raw_parts(row_ptr, row_span);
            dot_mxfp4_f32(row, x, cols)
        }
    }
}

/// Computes one weight row against `lanes` (<= `KQUANT_BATCH_TOKENS`)
/// activations at once, writing lane `t` to `out[t]`.
///
/// Only the K-quant integer paths have a blocked kernel; every other dtype,
/// and any call with fewer than a full set of lanes, falls through to
/// `dot_row` per lane. The blocked kernels keep each lane's accumulation order
/// identical to the single-token version, so this returns bit-identical
/// results either way — the batch path is a pure scheduling change, not a
/// numerics change.
#[cfg(not(target_family = "wasm"))]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn dot_row_x4(
    kind: MatvecKind,
    data: *const u8,
    xs: &[*const f32; KQUANT_BATCH_TOKENS],
    xqs: &[XQuant; KQUANT_BATCH_TOKENS],
    lanes: usize,
    row: usize,
    cols: usize,
    row_span: usize,
    out: &mut [f32; KQUANT_BATCH_TOKENS],
) {
    if lanes == KQUANT_BATCH_TOKENS && xqs.iter().all(XQuant::present) {
        let row_ptr = data.add(row * row_span);
        let row_bytes = std::slice::from_raw_parts(row_ptr, row_span);
        #[cfg(all(target_arch = "x86_64", not(target_family = "wasm")))]
        match kind {
            MatvecKind::Q4K => {
                *out = dot_q4_k_q8k_avx2_x4(row_bytes, xqs, cols);
                return;
            }
            MatvecKind::Q5K => {
                *out = dot_q5_k_q8k_avx2_x4(row_bytes, xqs, cols);
                return;
            }
            MatvecKind::Q6K => {
                *out = dot_q6_k_q8k_avx2_x4(row_bytes, xqs, cols);
                return;
            }
            _ => {}
        }
        #[cfg(all(target_arch = "aarch64", not(target_family = "wasm")))]
        match kind {
            MatvecKind::Q4K => {
                *out = dot_q4_k_q8k_neon_x4(row_bytes, xqs, cols);
                return;
            }
            MatvecKind::Q5K => {
                *out = dot_q5_k_q8k_neon_x4(row_bytes, xqs, cols);
                return;
            }
            MatvecKind::Q6K => {
                *out = dot_q6_k_q8k_neon_x4(row_bytes, xqs, cols);
                return;
            }
            _ => {}
        }
        let _ = row_bytes;
    }
    for t in 0..lanes {
        out[t] = dot_row(kind, data, xs[t], xqs[t], row, cols, row_span);
    }
}

#[cfg(not(target_family = "wasm"))]
#[derive(Clone, Copy)]
struct MatvecJob {
    kind: MatvecKind,
    data: *const u8,
    x: *const f32,
    /// Activation quantized once by the publishing thread; workers only read
    /// it. The pointers stay valid for the job's lifetime because the caller
    /// blocks in `run_job` until every worker has finished.
    xq: XQuant,
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
        // The publishing thread quantized the shared activation once into its
        // own scratch; workers reuse those read-only pointers across all rows.
        let xq = self.xq;
        let start = self.rows * worker_idx / self.workers;
        let end = self.rows * (worker_idx + 1) / self.workers;
        self.run_rows(start, end, xq);
    }

    unsafe fn run_dynamic_worker(self, worker_idx: usize, current_chunk: &AtomicUsize) {
        let xq = self.xq;
        let chunks = matvec_chunk_count(self.rows);
        let mut chunk_idx = worker_idx;

        while chunk_idx < chunks {
            let start = chunk_idx * MATVEC_CHUNK_ROWS;
            let end = (start + MATVEC_CHUNK_ROWS).min(self.rows);
            self.run_rows(start, end, xq);
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

    unsafe fn run_rows(self, start: usize, end: usize, xq: XQuant) {
        // Deliberately no software prefetch here: the rows are fully sequential,
        // so the hardware stream prefetcher already covers them, and an explicit
        // `_mm_prefetch` ahead of the row measured 8-10% *slower* on the large
        // DRAM-bound projections (the LM head and ffn_down) because the extra
        // requests compete with demand loads for fill buffers.
        for row in start..end {
            *self.out.add(row) = dot_row(
                self.kind,
                self.data,
                self.x,
                xq,
                row,
                self.cols,
                self.row_span,
            );
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
    /// Shared activation quantized once by the publishing thread (see
    /// [`MatvecJob::xq`]).
    xq: XQuant,
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
        // All three projections share one activation vector that the
        // publishing thread quantized once; workers reuse it across a/b/c rows.
        let xq = self.xq;
        let total = self.work_items();
        let start = total * worker_idx / self.workers;
        let end = total * (worker_idx + 1) / self.workers;
        self.run_items(start, end, xq);
    }

    unsafe fn run_dynamic_worker(self, worker_idx: usize, current_chunk: &AtomicUsize) {
        let xq = self.xq;
        let total = self.work_items();
        let chunks = matvec_chunk_count(total);
        let mut chunk_idx = worker_idx;

        while chunk_idx < chunks {
            let start = chunk_idx * MATVEC_CHUNK_ROWS;
            let end = (start + MATVEC_CHUNK_ROWS).min(total);
            self.run_items(start, end, xq);
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

    unsafe fn run_items(self, start: usize, end: usize, xq: XQuant) {
        let (a_start, a_end) = clipped_range(start, end, 0, self.rows_a);
        q4k_matvec3_rows(
            self.kind_a,
            self.a_data,
            self.x,
            xq,
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
            xq,
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
            xq,
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
    xq: XQuant,
    out: *mut f32,
    cols: usize,
    row_span: usize,
    start: usize,
    end: usize,
) {
    for row in start..end {
        *out.add(row) = dot_row(kind, data, x, xq, row, cols, row_span);
    }
}

/// Scheduling policy for a batched K-quant projection.
#[cfg(not(target_family = "wasm"))]
#[derive(Clone, Copy)]
enum KQuantBatchSchedule {
    /// One worker owns a complete token and consumes its pre-quantized
    /// activation across every projection row.
    Tokens,
    /// One worker owns output rows across every token. This keeps the owned
    /// weight rows hot while the batch is consumed.
    Rows,
}

/// A K-quant matrix projection over many activation rows. It lets an encoder
/// process a sequence with one pool rendezvous per projection phase instead of
/// one rendezvous per token.
#[cfg(not(target_family = "wasm"))]
#[derive(Clone, Copy)]
struct KQuantBatch3Job {
    kind_a: MatvecKind,
    kind_b: MatvecKind,
    kind_c: MatvecKind,
    a_data: *const u8,
    b_data: *const u8,
    c_data: *const u8,
    inputs: *const f32,
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
    tokens: usize,
    workers: usize,
    schedule: KQuantBatchSchedule,
    batch_qs: *const i8,
    batch_d: *const f32,
    batch_bsums: *const i16,
    batch_blocks: usize,
}

#[cfg(not(target_family = "wasm"))]
unsafe impl Send for KQuantBatch3Job {}
#[cfg(not(target_family = "wasm"))]
unsafe impl Sync for KQuantBatch3Job {}

#[cfg(not(target_family = "wasm"))]
impl KQuantBatch3Job {
    unsafe fn run_worker(self, worker_idx: usize, current_chunk: &AtomicUsize) {
        match self.schedule {
            KQuantBatchSchedule::Tokens => self.run_token_worker(worker_idx),
            KQuantBatchSchedule::Rows if self.workers > 1 => {
                self.run_dynamic_row_worker(worker_idx, current_chunk)
            }
            KQuantBatchSchedule::Rows => self.run_row_worker(worker_idx),
        }
    }

    unsafe fn run_token_worker(self, worker_idx: usize) {
        let start = self.tokens * worker_idx / self.workers;
        let end = self.tokens * (worker_idx + 1) / self.workers;
        for token in start..end {
            let x = self.inputs.add(token * self.cols);
            // This worker owns the token and therefore its thread-local Q8_K
            // scratch. The scratch stays unchanged until all three matrices
            // have consumed the activation.
            let xq = self.xq_for_token(token, x);
            if self.rows_a > 0 {
                self.run_rows(
                    self.kind_a,
                    self.a_data,
                    self.out_a.add(token * self.rows_a),
                    self.rows_a,
                    self.row_span_a,
                    x,
                    xq,
                );
            }
            if self.rows_b > 0 {
                self.run_rows(
                    self.kind_b,
                    self.b_data,
                    self.out_b.add(token * self.rows_b),
                    self.rows_b,
                    self.row_span_b,
                    x,
                    xq,
                );
            }
            if self.rows_c > 0 {
                self.run_rows(
                    self.kind_c,
                    self.c_data,
                    self.out_c.add(token * self.rows_c),
                    self.rows_c,
                    self.row_span_c,
                    x,
                    xq,
                );
            }
        }
    }

    unsafe fn run_row_worker(self, worker_idx: usize) {
        let total_rows = self.rows_a + self.rows_b + self.rows_c;
        let start = total_rows * worker_idx / self.workers;
        let end = total_rows * (worker_idx + 1) / self.workers;
        self.run_row_range(start, end);
    }

    unsafe fn run_dynamic_row_worker(self, worker_idx: usize, current_chunk: &AtomicUsize) {
        let total_rows = self.rows_a + self.rows_b + self.rows_c;
        let chunks = total_rows.div_ceil(KQUANT_BATCH_CHUNK_ROWS);
        let mut chunk_idx = worker_idx;
        while chunk_idx < chunks {
            let start = chunk_idx * KQUANT_BATCH_CHUNK_ROWS;
            let end = (start + KQUANT_BATCH_CHUNK_ROWS).min(total_rows);
            self.run_row_range(start, end);
            chunk_idx = current_chunk.fetch_add(1, Ordering::AcqRel);
        }
    }

    unsafe fn run_row_range(self, start: usize, end: usize) {
        let (a_start, a_end) = clipped_range(start, end, 0, self.rows_a);
        let (b_start, b_end) = clipped_range(start, end, self.rows_a, self.rows_b);
        let (c_start, c_end) = clipped_range(start, end, self.rows_a + self.rows_b, self.rows_c);

        // Token-blocked: `KQUANT_BATCH_TOKENS` tokens are carried through the
        // row range together so each weight block is unpacked once for the
        // whole group (see `dot_row_x4`). A plain loop interchange to
        // row-outer/token-inner was measured SLOWER — a Q4_K row at cols=3072
        // is 1728 B but one token's quantized activation is 3072 B, so making
        // the activations the streamed operand moves the thrashing to the worse
        // side. Blocking keeps both resident: four activations (12 KiB) plus
        // one row stay in L1 while the row chunk (55 KiB) streams from L2.
        //
        // Lane grouping requires the batch-wide activation quantization: with
        // `batch_qs` null, `xq_for_token` hands out the *same* thread-local
        // scratch for every token, so only one lane can be live at a time.
        let lane_stride = if self.batch_qs.is_null() {
            1
        } else {
            KQUANT_BATCH_TOKENS
        };
        let mut token = 0usize;
        while token < self.tokens {
            let lanes = (self.tokens - token).min(lane_stride);
            let mut xs = [std::ptr::null::<f32>(); KQUANT_BATCH_TOKENS];
            let mut xqs = [XQuant::NONE; KQUANT_BATCH_TOKENS];
            for t in 0..lanes {
                xs[t] = self.inputs.add((token + t) * self.cols);
                // Row ownership reuses weight rows across tokens. Activations
                // are pre-quantized once for the whole batch when the
                // architecture has an integer-dot kernel, avoiding one
                // quantization per worker.
                xqs[t] = self.xq_for_token(token + t, xs[t]);
            }
            if a_end > a_start {
                self.run_rows_range_x4(
                    self.kind_a,
                    self.a_data,
                    self.out_a,
                    self.rows_a,
                    self.row_span_a,
                    a_start,
                    a_end,
                    token,
                    lanes,
                    &xs,
                    &xqs,
                );
            }
            if b_end > b_start {
                self.run_rows_range_x4(
                    self.kind_b,
                    self.b_data,
                    self.out_b,
                    self.rows_b,
                    self.row_span_b,
                    b_start,
                    b_end,
                    token,
                    lanes,
                    &xs,
                    &xqs,
                );
            }
            if c_end > c_start {
                self.run_rows_range_x4(
                    self.kind_c,
                    self.c_data,
                    self.out_c,
                    self.rows_c,
                    self.row_span_c,
                    c_start,
                    c_end,
                    token,
                    lanes,
                    &xs,
                    &xqs,
                );
            }
            token += lanes;
        }
    }

    /// Runs rows `start..end` of one matrix against `lanes` activations,
    /// scattering lane `t` to its own token row of `out`.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    unsafe fn run_rows_range_x4(
        self,
        kind: MatvecKind,
        data: *const u8,
        out: *mut f32,
        rows: usize,
        row_span: usize,
        start: usize,
        end: usize,
        token_base: usize,
        lanes: usize,
        xs: &[*const f32; KQUANT_BATCH_TOKENS],
        xqs: &[XQuant; KQUANT_BATCH_TOKENS],
    ) {
        let mut vals = [0.0f32; KQUANT_BATCH_TOKENS];
        for row in start..end {
            dot_row_x4(
                kind, data, xs, xqs, lanes, row, self.cols, row_span, &mut vals,
            );
            for t in 0..lanes {
                *out.add((token_base + t) * rows + row) = vals[t];
            }
        }
    }

    #[inline(always)]
    unsafe fn xq_for_token(self, token: usize, x: *const f32) -> XQuant {
        if self.batch_qs.is_null() {
            prepare_xq_kquant(x, self.cols)
        } else {
            XQuant {
                qs: self.batch_qs.add(token * self.cols),
                d: self.batch_d.add(token * self.batch_blocks),
                bsums: self.batch_bsums.add(token * self.batch_blocks * 8),
            }
        }
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    unsafe fn run_rows(
        self,
        kind: MatvecKind,
        data: *const u8,
        out: *mut f32,
        rows: usize,
        row_span: usize,
        x: *const f32,
        xq: XQuant,
    ) {
        self.run_rows_range(kind, data, out, row_span, 0, rows, x, xq);
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    unsafe fn run_rows_range(
        self,
        kind: MatvecKind,
        data: *const u8,
        out: *mut f32,
        row_span: usize,
        start: usize,
        end: usize,
        x: *const f32,
        xq: XQuant,
    ) {
        for row in start..end {
            *out.add(row) = dot_row(kind, data, x, xq, row, self.cols, row_span);
        }
    }
}

/// Generic "run a closure over disjoint sub-ranges of `0..len`" job. The
/// callback is a C-style trampoline (`ctx`, `start`, `end`) so higher layers
/// (e.g. attention in `model.rs`) can drive the worker pool without the pool
/// depending on them. Each worker gets a contiguous, non-overlapping range, so
/// the callback must only touch state disjoint per range.
#[cfg(not(target_family = "wasm"))]
#[derive(Clone, Copy)]
struct RangeJob {
    ctx: *const (),
    func: unsafe fn(*const (), usize, usize),
    len: usize,
    workers: usize,
}

#[cfg(not(target_family = "wasm"))]
unsafe impl Send for RangeJob {}
#[cfg(not(target_family = "wasm"))]
unsafe impl Sync for RangeJob {}

#[cfg(not(target_family = "wasm"))]
impl RangeJob {
    unsafe fn run_worker(self, worker_idx: usize, _current_chunk: &AtomicUsize) {
        // Static even split of 0..len; workers that map to an empty range skip.
        let start = self.len * worker_idx / self.workers;
        let end = self.len * (worker_idx + 1) / self.workers;
        if end > start {
            (self.func)(self.ctx, start, end);
        }
    }
}

#[cfg(not(target_family = "wasm"))]
#[derive(Clone, Copy)]
enum WorkerJob {
    Matvec(MatvecJob),
    Q4KMatvec3(Q4KMatvec3Job),
    KQuantBatch3(KQuantBatch3Job),
    Range(RangeJob),
}

#[cfg(not(target_family = "wasm"))]
impl WorkerJob {
    #[inline]
    /// Returns how many workers should execute this job.
    fn workers(self) -> usize {
        match self {
            WorkerJob::Matvec(job) => job.workers,
            WorkerJob::Q4KMatvec3(job) => job.workers,
            WorkerJob::KQuantBatch3(job) => job.workers,
            WorkerJob::Range(job) => job.workers,
        }
    }

    #[inline]
    unsafe fn run_worker(self, worker_idx: usize, current_chunk: &AtomicUsize) {
        match self {
            WorkerJob::Matvec(job) => job.run_worker(worker_idx, current_chunk),
            WorkerJob::Q4KMatvec3(job) => job.run_worker(worker_idx, current_chunk),
            WorkerJob::KQuantBatch3(job) => job.run_worker(worker_idx, current_chunk),
            WorkerJob::Range(job) => job.run_worker(worker_idx, current_chunk),
        }
    }
}

/// Per-helper synchronization slot. `wake` is written only by the dispatching
/// thread and `done` only by the helper itself, each on its own cache line, so
/// publishing a job never bounces one line between several helpers and the
/// dispatcher's completion polling cannot slow the helpers down.
#[cfg(not(target_family = "wasm"))]
struct WorkerSlot {
    wake: CachePadded<AtomicU64>,
    done: CachePadded<AtomicU64>,
    thread: OnceLock<thread::Thread>,
}

#[cfg(not(target_family = "wasm"))]
struct WorkerPool {
    /// Serializes dispatchers, since the HTTP server can decode on several
    /// threads. Uncontended in the common single-stream case, where it costs
    /// one atomic swap rather than the per-dispatch handshake it replaces.
    dispatch: Mutex<()>,
    /// The published job. Written by the dispatcher while it holds `dispatch`
    /// and only after every helper of the previous generation reported `done`,
    /// so no helper can still be reading it.
    job: UnsafeCell<Option<WorkerJob>>,
    /// Helper slots; `slots[i]` drives shard `i + 1`. Shard 0 is the caller.
    slots: Vec<WorkerSlot>,
    generation: AtomicU64,
    current_chunk: CachePadded<AtomicUsize>,
    max_workers: usize,
}

// Safety: `job` is only ever written by a dispatcher holding `dispatch`, and
// only while no helper is running, which `run_job` enforces by waiting for
// every helper's `done` before returning.
#[cfg(not(target_family = "wasm"))]
unsafe impl Sync for WorkerPool {}

#[cfg(not(target_family = "wasm"))]
impl WorkerPool {
    /// Spawns `max_workers - 1` helper threads. The dispatching thread runs the
    /// remaining shard itself, so an `N`-thread job occupies exactly `N` cores
    /// instead of `N` helpers plus a spinning dispatcher.
    fn new(max_workers: usize) -> Arc<Self> {
        let helpers = max_workers.saturating_sub(1);
        let pool = Arc::new(Self {
            dispatch: Mutex::new(()),
            job: UnsafeCell::new(None),
            slots: (0..helpers)
                .map(|_| WorkerSlot {
                    wake: CachePadded(AtomicU64::new(0)),
                    done: CachePadded(AtomicU64::new(0)),
                    thread: OnceLock::new(),
                })
                .collect(),
            generation: AtomicU64::new(0),
            current_chunk: CachePadded(AtomicUsize::new(0)),
            max_workers,
        });

        for helper_idx in 0..helpers {
            let pool = Arc::clone(&pool);
            thread::Builder::new()
                .name(format!("rusty-llm-matvec-{}", helper_idx + 1))
                .spawn(move || worker_loop(pool, helper_idx))
                .expect("failed to start matvec worker");
        }

        // `run_job` wakes helpers through their thread handle, so every slot has
        // to carry one before the first dispatch or a wake-up could be lost.
        for slot in &pool.slots {
            while slot.thread.get().is_none() {
                std::hint::spin_loop();
            }
        }

        pool
    }

    /// Executes a queued worker-pool job and waits for completion.
    fn run(&self, mut job: MatvecJob) {
        // Quantize the activation once on this thread; the scratch stays valid
        // until this thread's next job, which starts only after this one ends.
        job.xq = unsafe { prepare_xq(job.kind, job.x, job.cols) };
        job.workers = job.workers.min(self.max_workers).min(job.rows).max(1);
        if job.workers <= 1 || job.rows < job.workers * 8 {
            for row in 0..job.rows {
                unsafe {
                    *job.out.add(row) = dot_row(
                        job.kind,
                        job.data,
                        job.x,
                        job.xq,
                        row,
                        job.cols,
                        job.row_span,
                    );
                }
            }
            return;
        }

        self.run_job(WorkerJob::Matvec(job), job.workers);
    }

    /// Executes a fused Q4_K triple-matvec job and waits for completion.
    fn run_q4k_matvec3(&self, mut job: Q4KMatvec3Job) {
        // One shared activation for all three projections: quantize it once on
        // this thread, and only when a sub-matrix kind actually consumes it.
        let wants_xq = matvec_kind_uses_xq(job.kind_a)
            || matvec_kind_uses_xq(job.kind_b)
            || matvec_kind_uses_xq(job.kind_c);
        job.xq = if wants_xq {
            unsafe { prepare_xq_kquant(job.x, job.cols) }
        } else {
            XQuant::NONE
        };
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

    /// Executes a K-quant projection over multiple activation rows. Prefer row
    /// ownership whenever every worker receives projection rows: reusing those
    /// weights and the batch-wide quantized activations saves substantially
    /// more memory traffic than token ownership.
    fn run_kquant_batch3(&self, mut job: KQuantBatch3Job) {
        // Pick the schedule first, then bound the worker count by whatever that
        // schedule actually shards. Clamping to `tokens` up front (as this used
        // to) also throttled the row schedule, so a short microbatch — the tail
        // chunk of a prompt, or a small `--ubatch` — ran a projection over
        // thousands of rows on `tokens` workers and left the rest of the cores
        // idle.
        let available = job.workers.min(self.max_workers).max(1);
        let projection_rows = job.rows_a + job.rows_b + job.rows_c;
        job.schedule = if projection_rows >= available {
            KQuantBatchSchedule::Rows
        } else {
            KQuantBatchSchedule::Tokens
        };
        job.workers = match job.schedule {
            // Reached only when projection_rows >= available, so every worker
            // already receives rows.
            KQuantBatchSchedule::Rows => available,
            KQuantBatchSchedule::Tokens => available.min(job.tokens).max(1),
        };
        if job.workers <= 1 {
            unsafe {
                job.run_worker(0, &self.current_chunk.0);
            }
            return;
        }
        self.run_job(WorkerJob::KQuantBatch3(job), job.workers);
    }

    /// Runs `func` over disjoint sub-ranges of `0..len` across the pool and
    /// blocks until every range completes.
    fn run_range(
        &self,
        ctx: *const (),
        func: unsafe fn(*const (), usize, usize),
        len: usize,
        workers: usize,
    ) {
        let workers = workers.min(self.max_workers).min(len).max(1);
        if workers <= 1 {
            unsafe {
                func(ctx, 0, len);
            }
            return;
        }
        self.run_job(
            WorkerJob::Range(RangeJob {
                ctx,
                func,
                len,
                workers,
            }),
            workers,
        );
    }

    /// Publishes a worker job, runs shard 0 on the calling thread, and blocks
    /// until every helper shard reports completion.
    ///
    /// `workers` must equal the job's own worker count, since each shard derives
    /// its slice from that field. The calling thread executes kernel code while
    /// holding `dispatch`, so a job function must never dispatch again: nesting
    /// would deadlock on the non-reentrant mutex.
    fn run_job(&self, job: WorkerJob, workers: usize) {
        ensure_compute_thread_qos();
        let _dispatch = self.dispatch.lock().expect("worker pool mutex poisoned");
        let generation = self.generation.load(Ordering::Relaxed).wrapping_add(1);
        let helpers = workers.saturating_sub(1).min(self.slots.len());

        // Safety: every helper of the previous generation reported `done` before
        // that dispatch returned, and `dispatch` keeps other dispatchers out, so
        // nothing else can be reading or writing the job slot right now.
        unsafe {
            *self.job.get() = Some(job);
        }
        self.current_chunk.0.store(workers, Ordering::Release);

        // Release the job to exactly the helpers this job needs; the rest stay
        // parked instead of waking up only to discover they have no shard.
        for slot in &self.slots[..helpers] {
            slot.wake.0.store(generation, Ordering::Release);
        }
        for slot in &self.slots[..helpers] {
            if let Some(handle) = slot.thread.get() {
                handle.unpark();
            }
        }

        // The dispatcher is a full participant rather than a spinning observer.
        // On the dynamic-chunk path it also joins the work stealing, so a slow
        // helper no longer leaves this core idle.
        unsafe {
            job.run_worker(0, &self.current_chunk.0);
        }

        for slot in &self.slots[..helpers] {
            let mut spins = 0usize;
            while slot.done.0.load(Ordering::Acquire) != generation {
                if spins < 20_000 {
                    std::hint::spin_loop();
                    spins += 1;
                } else {
                    thread::yield_now();
                }
            }
        }

        self.generation.store(generation, Ordering::Relaxed);
    }
}

#[cfg(not(target_family = "wasm"))]
/// Runs helper shard `helper_idx + 1` for every job published to this pool.
fn worker_loop(pool: Arc<WorkerPool>, helper_idx: usize) {
    let shard = helper_idx + 1;
    let _ = pin_current_thread(shard);
    let slot = &pool.slots[helper_idx];
    let _ = slot.thread.set(thread::current());
    let mut last_seen = slot.wake.0.load(Ordering::Acquire);

    loop {
        // Spin briefly on this helper's own cache line so back-to-back layers
        // hand off without a syscall, then park so that an idle helper costs
        // nothing at all instead of burning a core polling a shared flag.
        let mut generation = slot.wake.0.load(Ordering::Acquire);
        if generation == last_seen {
            for _ in 0..worker_poll_spins() {
                std::hint::spin_loop();
                generation = slot.wake.0.load(Ordering::Acquire);
                if generation != last_seen {
                    break;
                }
            }
        }
        // `park` may return spuriously, and an `unpark` that lands before the
        // park makes it return immediately; re-checking `wake` covers both.
        while generation == last_seen {
            thread::park();
            generation = slot.wake.0.load(Ordering::Acquire);
        }

        // Safety: the dispatcher wrote the job before the release store to
        // `wake` that we just acquired, and it will not touch the slot again
        // until this helper publishes `done` below.
        let job = unsafe { (*pool.job.get()).expect("job should be available") };
        if shard < job.workers() {
            unsafe {
                job.run_worker(shard, &pool.current_chunk.0);
            }
        }
        last_seen = generation;
        slot.done.0.store(generation, Ordering::Release);
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

/// Computes four dot products against the same right-hand vector. GQA
/// attention uses this for its common four-query-head groups, letting the SIMD
/// loop load each cached key vector once rather than once per query head.
#[inline]
pub fn dot_f32x4(a0: &[f32], a1: &[f32], a2: &[f32], a3: &[f32], b: &[f32]) -> [f32; 4] {
    debug_assert_eq!(a0.len(), b.len());
    debug_assert_eq!(a1.len(), b.len());
    debug_assert_eq!(a2.len(), b.len());
    debug_assert_eq!(a3.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { dot_f32x4_neon(a0, a1, a2, a3, b) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            unsafe { dot_f32x4_avx2(a0, a1, a2, a3, b) }
        } else {
            dot_f32x4_scalar(a0, a1, a2, a3, b)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        dot_f32x4_scalar(a0, a1, a2, a3, b)
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

// ─── bf16 KV-cache kernels: f32 query/accumulator, bf16-stored K/V ───────────
// Used only by the bf16 KV-cache path (`KVCache::enable_bf16`): the query and
// the softmax accumulator stay f32 (unchanged numerics), only the cached
// key/value row read from RAM is bf16 and gets widened inline per element —
// no full-row dequant scratch buffer, so the DRAM traffic saved by storing
// half-width KV is not immediately spent again on an equally large fp32 copy.

/// Dot product of an f32 vector against a bf16-stored vector (same length).
#[inline]
pub fn dot_bf16_f32(a: &[f32], b: &[u16]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { dot_bf16_f32_neon(a, b) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            unsafe { dot_bf16_f32_avx2(a, b) }
        } else {
            dot_bf16_f32_scalar(a, b)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        dot_bf16_f32_scalar(a, b)
    }
}

/// Computes `out[i] += alpha * bf16_to_f32(x[i])`.
#[inline]
pub fn axpy_bf16_f32(out: &mut [f32], alpha: f32, x: &[u16]) {
    debug_assert_eq!(out.len(), x.len());
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { axpy_bf16_f32_neon(out, alpha, x) }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            unsafe { axpy_bf16_f32_avx2(out, alpha, x) }
        } else {
            axpy_bf16_f32_scalar(out, alpha, x)
        }
        return;
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        axpy_bf16_f32_scalar(out, alpha, x)
    }
}

/// Computes `out[i] = out[i] * scale + bf16_to_f32(add[i])`.
#[inline]
pub fn scale_add_bf16_f32(out: &mut [f32], scale: f32, add: &[u16]) {
    debug_assert_eq!(out.len(), add.len());
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { scale_add_bf16_f32_neon(out, scale, add) }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            unsafe { scale_add_bf16_f32_avx2(out, scale, add) }
        } else {
            scale_add_bf16_f32_scalar(out, scale, add)
        }
        return;
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scale_add_bf16_f32_scalar(out, scale, add)
    }
}

/// bf16-KV counterpart of `dot_f32x4`: widens each bf16 key element once and
/// reuses it across all four query-head dot products. Without this, GQA's
/// generic per-head loop would re-widen the same key row `kv_mul` times.
#[inline]
pub fn dot_bf16x4_f32(a0: &[f32], a1: &[f32], a2: &[f32], a3: &[f32], b: &[u16]) -> [f32; 4] {
    debug_assert_eq!(a0.len(), b.len());
    debug_assert_eq!(a1.len(), b.len());
    debug_assert_eq!(a2.len(), b.len());
    debug_assert_eq!(a3.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { dot_bf16x4_f32_neon(a0, a1, a2, a3, b) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            unsafe { dot_bf16x4_f32_avx2(a0, a1, a2, a3, b) }
        } else {
            dot_bf16x4_f32_scalar(a0, a1, a2, a3, b)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        dot_bf16x4_f32_scalar(a0, a1, a2, a3, b)
    }
}

/// bf16-KV counterpart of `affine_add_f32x4`: widens each bf16 value element
/// once and reuses it across all four heads' accumulator updates.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn affine_add_bf16x4_f32(
    out0: &mut [f32],
    out1: &mut [f32],
    out2: &mut [f32],
    out3: &mut [f32],
    multiplier: [f32; 4],
    alpha: [f32; 4],
    x: &[u16],
) {
    debug_assert_eq!(out0.len(), x.len());
    debug_assert_eq!(out1.len(), x.len());
    debug_assert_eq!(out2.len(), x.len());
    debug_assert_eq!(out3.len(), x.len());
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { affine_add_bf16x4_f32_neon(out0, out1, out2, out3, multiplier, alpha, x) }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            unsafe { affine_add_bf16x4_f32_avx2(out0, out1, out2, out3, multiplier, alpha, x) }
        } else {
            affine_add_bf16x4_f32_scalar(out0, out1, out2, out3, multiplier, alpha, x)
        }
        return;
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        affine_add_bf16x4_f32_scalar(out0, out1, out2, out3, multiplier, alpha, x)
    }
}

/// Applies four independent `out = out * multiplier + alpha * x` updates
/// while streaming the shared source vector only once. This is the value-cache
/// half of four-head grouped-query attention.
#[inline]
pub fn affine_add_f32x4(
    out0: &mut [f32],
    out1: &mut [f32],
    out2: &mut [f32],
    out3: &mut [f32],
    multiplier: [f32; 4],
    alpha: [f32; 4],
    x: &[f32],
) {
    debug_assert_eq!(out0.len(), x.len());
    debug_assert_eq!(out1.len(), x.len());
    debug_assert_eq!(out2.len(), x.len());
    debug_assert_eq!(out3.len(), x.len());
    #[cfg(target_arch = "aarch64")]
    unsafe {
        affine_add_f32x4_neon(out0, out1, out2, out3, multiplier, alpha, x)
    }
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        unsafe { affine_add_f32x4_avx2(out0, out1, out2, out3, multiplier, alpha, x) }
    } else {
        affine_add_f32x4_scalar(out0, out1, out2, out3, multiplier, alpha, x)
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        affine_add_f32x4_scalar(out0, out1, out2, out3, multiplier, alpha, x)
    }
}

/// Applies four independent affine updates and returns their dot products
/// against one shared vector. Combining both operations keeps every updated
/// output lane in a register for the dot product instead of reading the four
/// vectors from memory again.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn affine_add_dot_f32x4(
    out0: &mut [f32],
    out1: &mut [f32],
    out2: &mut [f32],
    out3: &mut [f32],
    multiplier: [f32; 4],
    alpha: [f32; 4],
    x: &[f32],
    dot: &[f32],
) -> [f32; 4] {
    debug_assert_eq!(out0.len(), x.len());
    debug_assert_eq!(out1.len(), x.len());
    debug_assert_eq!(out2.len(), x.len());
    debug_assert_eq!(out3.len(), x.len());
    debug_assert_eq!(dot.len(), x.len());
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { affine_add_dot_f32x4_neon(out0, out1, out2, out3, multiplier, alpha, x, dot) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            unsafe { affine_add_dot_f32x4_avx2(out0, out1, out2, out3, multiplier, alpha, x, dot) }
        } else {
            affine_add_dot_f32x4_scalar(out0, out1, out2, out3, multiplier, alpha, x, dot)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        affine_add_dot_f32x4_scalar(out0, out1, out2, out3, multiplier, alpha, x, dot)
    }
}

/// Computes `out[i] = silu(gate[i]) * up[i]` — the SwiGLU elementwise combine
/// between the fused gate/up projections and the down projection. The scalar
/// `expf` loop this replaces costs ~0.8 ms per layer at Ministral's
/// hidden_dim 9216 (≈7% of a decode token), so the SIMD paths use a
/// Cephes-style polynomial `expf` (max relative error ~2e-7, well below
/// quantization noise).
pub fn silu_mul_into(gate: &[f32], up: &[f32], out: &mut Vec<f32>) {
    debug_assert_eq!(gate.len(), up.len());
    let n = gate.len().min(up.len());
    out.resize(n, 0.0);
    silu_mul_slice_into(&gate[..n], &up[..n], &mut out[..n]);
}

/// Slice-output variant of [`silu_mul_into`] for row-major batch scratch.
/// Keeping the destination size fixed lets callers write directly into one
/// token row without allocating or copying a temporary `Vec`.
pub fn silu_mul_slice_into(gate: &[f32], up: &[f32], out: &mut [f32]) {
    debug_assert_eq!(gate.len(), up.len());
    debug_assert_eq!(gate.len(), out.len());
    let n = gate.len().min(up.len()).min(out.len());
    #[cfg(all(target_arch = "aarch64", not(target_family = "wasm")))]
    {
        unsafe { silu_mul_neon(&gate[..n], &up[..n], &mut out[..n]) }
        return;
    }
    #[cfg(all(target_arch = "x86_64", not(target_family = "wasm")))]
    {
        if has_avx2_fma() {
            unsafe { silu_mul_avx2(&gate[..n], &up[..n], &mut out[..n]) }
        } else {
            silu_mul_scalar(&gate[..n], &up[..n], &mut out[..n]);
        }
        return;
    }
    #[cfg(not(all(
        any(target_arch = "aarch64", target_arch = "x86_64"),
        not(target_family = "wasm")
    )))]
    {
        silu_mul_scalar(&gate[..n], &up[..n], &mut out[..n]);
    }
}

/// Multiplies `values[i]` by `sigmoid(gate[i])` in place.
///
/// Qwen3.5's full-attention blocks apply this gate after attention and before
/// their output projection. Keeping it in place avoids a scratch allocation
/// while sharing the same SIMD exp approximation as [`silu_mul_into`].
pub fn sigmoid_mul_in_place(values: &mut [f32], gate: &[f32]) {
    debug_assert_eq!(values.len(), gate.len());
    let n = values.len().min(gate.len());
    #[cfg(all(target_arch = "aarch64", not(target_family = "wasm")))]
    {
        unsafe { sigmoid_mul_neon(&mut values[..n], &gate[..n]) }
        return;
    }
    #[cfg(all(target_arch = "x86_64", not(target_family = "wasm")))]
    {
        if has_avx2_fma() {
            unsafe { sigmoid_mul_avx2(&mut values[..n], &gate[..n]) }
        } else {
            sigmoid_mul_scalar(&mut values[..n], &gate[..n]);
        }
        return;
    }
    #[cfg(not(all(
        any(target_arch = "aarch64", target_arch = "x86_64"),
        not(target_family = "wasm")
    )))]
    {
        sigmoid_mul_scalar(&mut values[..n], &gate[..n]);
    }
}

/// Scalar reference for `silu_mul_into` (also the wasm/non-AVX2 path).
fn silu_mul_scalar(gate: &[f32], up: &[f32], out: &mut [f32]) {
    for i in 0..out.len() {
        let g = gate[i];
        out[i] = g / (1.0 + (-g).exp()) * up[i];
    }
}

/// Scalar reference for [`sigmoid_mul_in_place`].
fn sigmoid_mul_scalar(values: &mut [f32], gate: &[f32]) {
    for (value, gate) in values.iter_mut().zip(gate) {
        *value *= 1.0 / (1.0 + (-*gate).exp());
    }
}

// Cephes-style expf constants, shared by the AVX2 and NEON kernels. The
// input is clamped to ±88.376 (beyond which f32 exp over/underflows), split
// as x = n·ln2 + r with a two-constant Cody–Waite reduction, exp(r) is a
// degree-5 polynomial, and 2^n is applied through the exponent bits.
const EXPF_HI: f32 = 88.376_26;
const EXPF_LO: f32 = -88.376_26;
const EXPF_LOG2E: f32 = std::f32::consts::LOG2_E;
const EXPF_C1: f32 = 0.693_359_4; // ln2 high part
const EXPF_C2: f32 = -2.121_944_4e-4; // ln2 low part
const EXPF_P0: f32 = 1.987_569_1e-4;
const EXPF_P1: f32 = 1.398_2e-3;
const EXPF_P2: f32 = 8.333_452e-3;
const EXPF_P3: f32 = 4.166_579_6e-2;
const EXPF_P4: f32 = 1.666_666_5e-1;
const EXPF_P5: f32 = 0.5;

#[cfg(all(target_arch = "x86_64", not(target_family = "wasm")))]
#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn exp_ps_avx2(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let x = _mm256_min_ps(
        _mm256_set1_ps(EXPF_HI),
        _mm256_max_ps(_mm256_set1_ps(EXPF_LO), x),
    );
    // n = round-toward-neg-inf(x * log2(e) + 0.5)
    let fx = _mm256_fmadd_ps(x, _mm256_set1_ps(EXPF_LOG2E), _mm256_set1_ps(0.5));
    let fx = _mm256_floor_ps(fx);
    // r = x - n*ln2 (two-step for accuracy)
    let x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(EXPF_C1), x);
    let x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(EXPF_C2), x);
    // Degree-5 polynomial for exp(r)
    let mut y = _mm256_set1_ps(EXPF_P0);
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(EXPF_P1));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(EXPF_P2));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(EXPF_P3));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(EXPF_P4));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(EXPF_P5));
    let x2 = _mm256_mul_ps(x, x);
    let y = _mm256_add_ps(_mm256_fmadd_ps(y, x2, x), _mm256_set1_ps(1.0));
    // * 2^n via exponent bits
    let n = _mm256_cvttps_epi32(fx);
    let pow2n = _mm256_castsi256_ps(_mm256_slli_epi32(
        _mm256_add_epi32(n, _mm256_set1_epi32(0x7f)),
        23,
    ));
    _mm256_mul_ps(y, pow2n)
}

#[cfg(all(target_arch = "x86_64", not(target_family = "wasm")))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn silu_mul_avx2(gate: &[f32], up: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = out.len();
    let main = n / 8 * 8;
    let one = _mm256_set1_ps(1.0);
    let mut i = 0usize;
    while i < main {
        let g = _mm256_loadu_ps(gate.as_ptr().add(i));
        let u = _mm256_loadu_ps(up.as_ptr().add(i));
        // silu(g) = g / (1 + exp(-g))
        let e = exp_ps_avx2(_mm256_sub_ps(_mm256_setzero_ps(), g));
        let s = _mm256_div_ps(g, _mm256_add_ps(one, e));
        _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_mul_ps(s, u));
        i += 8;
    }
    silu_mul_scalar(&gate[main..], &up[main..], &mut out[main..]);
}

#[cfg(all(target_arch = "x86_64", not(target_family = "wasm")))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn sigmoid_mul_avx2(values: &mut [f32], gate: &[f32]) {
    use std::arch::x86_64::*;
    let n = values.len();
    let main = n / 8 * 8;
    let one = _mm256_set1_ps(1.0);
    let mut i = 0usize;
    while i < main {
        let value = _mm256_loadu_ps(values.as_ptr().add(i));
        let g = _mm256_loadu_ps(gate.as_ptr().add(i));
        let e = exp_ps_avx2(_mm256_sub_ps(_mm256_setzero_ps(), g));
        let sigmoid = _mm256_div_ps(one, _mm256_add_ps(one, e));
        _mm256_storeu_ps(values.as_mut_ptr().add(i), _mm256_mul_ps(value, sigmoid));
        i += 8;
    }
    sigmoid_mul_scalar(&mut values[main..], &gate[main..]);
}

#[cfg(all(target_arch = "aarch64", not(target_family = "wasm")))]
#[inline]
unsafe fn exp_ps_neon(x: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    let x = vminq_f32(vdupq_n_f32(EXPF_HI), vmaxq_f32(vdupq_n_f32(EXPF_LO), x));
    let fx = vrndmq_f32(vfmaq_f32(vdupq_n_f32(0.5), x, vdupq_n_f32(EXPF_LOG2E)));
    let x = vfmsq_f32(x, fx, vdupq_n_f32(EXPF_C1));
    let x = vfmsq_f32(x, fx, vdupq_n_f32(EXPF_C2));
    let mut y = vdupq_n_f32(EXPF_P0);
    y = vfmaq_f32(vdupq_n_f32(EXPF_P1), y, x);
    y = vfmaq_f32(vdupq_n_f32(EXPF_P2), y, x);
    y = vfmaq_f32(vdupq_n_f32(EXPF_P3), y, x);
    y = vfmaq_f32(vdupq_n_f32(EXPF_P4), y, x);
    y = vfmaq_f32(vdupq_n_f32(EXPF_P5), y, x);
    let x2 = vmulq_f32(x, x);
    let y = vaddq_f32(vfmaq_f32(x, y, x2), vdupq_n_f32(1.0));
    let n = vcvtq_s32_f32(fx);
    let pow2n = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(n, vdupq_n_s32(0x7f)), 23));
    vmulq_f32(y, pow2n)
}

#[cfg(all(target_arch = "aarch64", not(target_family = "wasm")))]
unsafe fn silu_mul_neon(gate: &[f32], up: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::*;
    let n = out.len();
    let main = n / 4 * 4;
    let one = vdupq_n_f32(1.0);
    let mut i = 0usize;
    while i < main {
        let g = vld1q_f32(gate.as_ptr().add(i));
        let u = vld1q_f32(up.as_ptr().add(i));
        let e = exp_ps_neon(vnegq_f32(g));
        let s = vdivq_f32(g, vaddq_f32(one, e));
        vst1q_f32(out.as_mut_ptr().add(i), vmulq_f32(s, u));
        i += 4;
    }
    silu_mul_scalar(&gate[main..], &up[main..], &mut out[main..]);
}

#[cfg(all(target_arch = "aarch64", not(target_family = "wasm")))]
unsafe fn sigmoid_mul_neon(values: &mut [f32], gate: &[f32]) {
    use std::arch::aarch64::*;
    let n = values.len();
    let main = n / 4 * 4;
    let one = vdupq_n_f32(1.0);
    let mut i = 0usize;
    while i < main {
        let value = vld1q_f32(values.as_ptr().add(i));
        let g = vld1q_f32(gate.as_ptr().add(i));
        let e = exp_ps_neon(vnegq_f32(g));
        let sigmoid = vdivq_f32(one, vaddq_f32(one, e));
        vst1q_f32(values.as_mut_ptr().add(i), vmulq_f32(value, sigmoid));
        i += 4;
    }
    sigmoid_mul_scalar(&mut values[main..], &gate[main..]);
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

#[inline]
#[allow(dead_code)]
/// Scalar fallback for the bf16-KV dot product.
fn dot_bf16_f32_scalar(a: &[f32], b: &[u16]) -> f32 {
    let n = a.len();
    let chunks = n / 4;
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    for i in 0..chunks {
        let base = i * 4;
        s0 += a[base] * bf16_to_f32(b[base]);
        s1 += a[base + 1] * bf16_to_f32(b[base + 1]);
        s2 += a[base + 2] * bf16_to_f32(b[base + 2]);
        s3 += a[base + 3] * bf16_to_f32(b[base + 3]);
    }
    for i in (chunks * 4)..n {
        s0 += a[i] * bf16_to_f32(b[i]);
    }
    (s0 + s1) + (s2 + s3)
}

#[inline]
#[allow(dead_code)]
/// Scalar fallback for bf16-KV AXPY.
fn axpy_bf16_f32_scalar(out: &mut [f32], alpha: f32, x: &[u16]) {
    for (o, &xi) in out.iter_mut().zip(x.iter()) {
        *o += alpha * bf16_to_f32(xi);
    }
}

#[inline]
#[allow(dead_code)]
/// Scalar fallback for bf16-KV fused scale-and-add.
fn scale_add_bf16_f32_scalar(out: &mut [f32], scale: f32, add: &[u16]) {
    for (o, &a) in out.iter_mut().zip(add.iter()) {
        *o = *o * scale + bf16_to_f32(a);
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
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { dot_mxfp4_f32_neon(qdata, x, n) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        dot_mxfp4_f32_scalar(qdata, x, n)
    }
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
            xq: XQuant::NONE,
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
            xq: XQuant::NONE,
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
            xq: XQuant::NONE,
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

/// Runs one K-quant matrix against a row-major batch of activations.
///
/// The batch is CPU-oriented: workers own either disjoint input rows or
/// disjoint output rows, chosen by batch length, so an encoder avoids
/// publishing a separate pool job for every token. `inputs` must contain a
/// whole number of `cols`-wide rows and `out` is returned as the matching
/// row-major output matrix.
pub fn matvec_kquant_batch_into(
    a: (KQuantMatvecKind, &[u8], usize, usize),
    inputs: &[f32],
    out: &mut Vec<f32>,
) -> bool {
    let (_, _, _, cols) = a;
    let mut unused_b = Vec::new();
    let mut unused_c = Vec::new();
    matvec_kquant3_batch_into(
        a,
        (a.0, &[], 0, cols),
        (a.0, &[], 0, cols),
        inputs,
        out,
        &mut unused_b,
        &mut unused_c,
    )
}

/// Runs two K-quant matrices against a row-major batch of shared activations.
pub fn matvec_kquant2_batch_into(
    a: (KQuantMatvecKind, &[u8], usize, usize),
    b: (KQuantMatvecKind, &[u8], usize, usize),
    inputs: &[f32],
    out_a: &mut Vec<f32>,
    out_b: &mut Vec<f32>,
) -> bool {
    let mut unused_c = Vec::new();
    matvec_kquant3_batch_into(
        a,
        b,
        (a.0, &[], 0, a.3),
        inputs,
        out_a,
        out_b,
        &mut unused_c,
    )
}

/// Runs three K-quant matrices against a row-major batch of shared
/// activations. This deliberately stays on the CPU: dispatching one Metal
/// command buffer per token costs more than the saved arithmetic for the
/// encoder's narrow 768-wide projections.
#[allow(clippy::too_many_arguments)]
pub fn matvec_kquant3_batch_into(
    a: (KQuantMatvecKind, &[u8], usize, usize),
    b: (KQuantMatvecKind, &[u8], usize, usize),
    c: (KQuantMatvecKind, &[u8], usize, usize),
    inputs: &[f32],
    out_a: &mut Vec<f32>,
    out_b: &mut Vec<f32>,
    out_c: &mut Vec<f32>,
) -> bool {
    let (kind_a, weights_a, rows_a, cols_a) = a;
    let (kind_b, weights_b, rows_b, cols_b) = b;
    let (kind_c, weights_c, rows_c, cols_c) = c;
    matvec_quant3_batch_into(
        (
            QuantMatvecKind::from_kquant(kind_a),
            weights_a,
            rows_a,
            cols_a,
        ),
        (
            QuantMatvecKind::from_kquant(kind_b),
            weights_b,
            rows_b,
            cols_b,
        ),
        (
            QuantMatvecKind::from_kquant(kind_c),
            weights_c,
            rows_c,
            cols_c,
        ),
        inputs,
        out_a,
        out_b,
        out_c,
    )
}

/// Runs one quantized matrix against a row-major activation batch. Unlike the
/// narrower K-quant entry point, this also accepts the 32-value block formats
/// used by hybrid and fallback quantizations.
pub fn matvec_quant_batch_into(
    a: (QuantMatvecKind, &[u8], usize, usize),
    inputs: &[f32],
    out: &mut Vec<f32>,
) -> bool {
    let (_, _, _, cols) = a;
    let mut unused_b = Vec::new();
    let mut unused_c = Vec::new();
    matvec_quant3_batch_into(
        a,
        (a.0, &[], 0, cols),
        (a.0, &[], 0, cols),
        inputs,
        out,
        &mut unused_b,
        &mut unused_c,
    )
}

/// Runs two quantized matrices against the same row-major activation batch.
pub fn matvec_quant2_batch_into(
    a: (QuantMatvecKind, &[u8], usize, usize),
    b: (QuantMatvecKind, &[u8], usize, usize),
    inputs: &[f32],
    out_a: &mut Vec<f32>,
    out_b: &mut Vec<f32>,
) -> bool {
    let mut unused_c = Vec::new();
    matvec_quant3_batch_into(
        a,
        b,
        (a.0, &[], 0, a.3),
        inputs,
        out_a,
        out_b,
        &mut unused_c,
    )
}

/// Runs three quantized matrices against a row-major activation batch while
/// keeping each owned weight row hot across up to four activation rows.
#[allow(clippy::too_many_arguments)]
pub fn matvec_quant3_batch_into(
    a: (QuantMatvecKind, &[u8], usize, usize),
    b: (QuantMatvecKind, &[u8], usize, usize),
    c: (QuantMatvecKind, &[u8], usize, usize),
    inputs: &[f32],
    out_a: &mut Vec<f32>,
    out_b: &mut Vec<f32>,
    out_c: &mut Vec<f32>,
) -> bool {
    let (kind_a, weights_a, rows_a, cols_a) = a;
    let (kind_b, weights_b, rows_b, cols_b) = b;
    let (kind_c, weights_c, rows_c, cols_c) = c;
    if cols_a == 0 || cols_a != cols_b || cols_a != cols_c || inputs.len() % cols_a != 0 {
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

    let tokens = inputs.len() / cols_a;
    let out_len_a = match tokens.checked_mul(rows_a) {
        Some(v) => v,
        None => return false,
    };
    let out_len_b = match tokens.checked_mul(rows_b) {
        Some(v) => v,
        None => return false,
    };
    let out_len_c = match tokens.checked_mul(rows_c) {
        Some(v) => v,
        None => return false,
    };
    out_a.resize(out_len_a, 0.0);
    out_b.resize(out_len_b, 0.0);
    out_c.resize(out_len_c, 0.0);
    if tokens == 0 {
        return true;
    }

    #[cfg(target_family = "wasm")]
    {
        let mut row_a = Vec::new();
        let mut row_b = Vec::new();
        let mut row_c = Vec::new();
        for token in 0..tokens {
            let input = &inputs[token * cols_a..(token + 1) * cols_a];
            if !matvec_quant3_into(a, b, c, input, &mut row_a, &mut row_b, &mut row_c) {
                return false;
            }
            out_a[token * rows_a..(token + 1) * rows_a].copy_from_slice(&row_a);
            out_b[token * rows_b..(token + 1) * rows_b].copy_from_slice(&row_b);
            out_c[token * rows_c..(token + 1) * rows_c].copy_from_slice(&row_c);
        }
        return true;
    }

    #[cfg(not(target_family = "wasm"))]
    {
        // Pass the full thread count: `run_kquant_batch3` picks the schedule and
        // then bounds the workers by whatever that schedule shards. Clamping to
        // `tokens` here would throttle the row schedule, which shards thousands
        // of projection rows and does not care how many tokens the batch holds.
        let workers = num_threads();
        let batch_xq = prepare_batch_xq(inputs, cols_a);
        let (batch_qs, batch_d, batch_bsums) = batch_xq.as_ref().map_or(
            (std::ptr::null(), std::ptr::null(), std::ptr::null()),
            |batch| (batch.qs.as_ptr(), batch.d.as_ptr(), batch.bsums.as_ptr()),
        );
        worker_pool().run_kquant_batch3(KQuantBatch3Job {
            kind_a: kind_a.matvec_kind(),
            kind_b: kind_b.matvec_kind(),
            kind_c: kind_c.matvec_kind(),
            a_data: weights_a.as_ptr(),
            b_data: weights_b.as_ptr(),
            c_data: weights_c.as_ptr(),
            inputs: inputs.as_ptr(),
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
            tokens,
            workers,
            schedule: KQuantBatchSchedule::Tokens,
            batch_qs,
            batch_d,
            batch_bsums,
            batch_blocks: cols_a / 256,
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
            xq: XQuant::NONE,
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
            xq: XQuant::NONE,
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
            xq: XQuant::NONE,
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
fn dot_f32x4_scalar(a0: &[f32], a1: &[f32], a2: &[f32], a3: &[f32], b: &[f32]) -> [f32; 4] {
    let mut out = [0.0; 4];
    for i in 0..b.len() {
        let key = b[i];
        out[0] += a0[i] * key;
        out[1] += a1[i] * key;
        out[2] += a2[i] * key;
        out[3] += a3[i] * key;
    }
    out
}

#[allow(dead_code)]
fn affine_add_f32x4_scalar(
    out0: &mut [f32],
    out1: &mut [f32],
    out2: &mut [f32],
    out3: &mut [f32],
    multiplier: [f32; 4],
    alpha: [f32; 4],
    x: &[f32],
) {
    for i in 0..x.len() {
        out0[i] = out0[i] * multiplier[0] + alpha[0] * x[i];
        out1[i] = out1[i] * multiplier[1] + alpha[1] * x[i];
        out2[i] = out2[i] * multiplier[2] + alpha[2] * x[i];
        out3[i] = out3[i] * multiplier[3] + alpha[3] * x[i];
    }
}

#[allow(dead_code, clippy::too_many_arguments)]
fn affine_add_dot_f32x4_scalar(
    out0: &mut [f32],
    out1: &mut [f32],
    out2: &mut [f32],
    out3: &mut [f32],
    multiplier: [f32; 4],
    alpha: [f32; 4],
    x: &[f32],
    dot: &[f32],
) -> [f32; 4] {
    let mut sums = [0.0f32; 4];
    for i in 0..x.len() {
        out0[i] = out0[i] * multiplier[0] + alpha[0] * x[i];
        out1[i] = out1[i] * multiplier[1] + alpha[1] * x[i];
        out2[i] = out2[i] * multiplier[2] + alpha[2] * x[i];
        out3[i] = out3[i] * multiplier[3] + alpha[3] * x[i];
        sums[0] += out0[i] * dot[i];
        sums[1] += out1[i] * dot[i];
        sums[2] += out2[i] * dot[i];
        sums[3] += out3[i] * dot[i];
    }
    sums
}

#[allow(dead_code)]
fn dot_bf16x4_f32_scalar(a0: &[f32], a1: &[f32], a2: &[f32], a3: &[f32], b: &[u16]) -> [f32; 4] {
    let mut out = [0.0; 4];
    for i in 0..b.len() {
        let key = bf16_to_f32(b[i]);
        out[0] += a0[i] * key;
        out[1] += a1[i] * key;
        out[2] += a2[i] * key;
        out[3] += a3[i] * key;
    }
    out
}

#[allow(dead_code)]
fn affine_add_bf16x4_f32_scalar(
    out0: &mut [f32],
    out1: &mut [f32],
    out2: &mut [f32],
    out3: &mut [f32],
    multiplier: [f32; 4],
    alpha: [f32; 4],
    x: &[u16],
) {
    for i in 0..x.len() {
        let value = bf16_to_f32(x[i]);
        out0[i] = out0[i] * multiplier[0] + alpha[0] * value;
        out1[i] = out1[i] * multiplier[1] + alpha[1] * value;
        out2[i] = out2[i] * multiplier[2] + alpha[2] * value;
        out3[i] = out3[i] * multiplier[3] + alpha[3] * value;
    }
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

// ─── Integer (sdot) K-quant fast path (aarch64 + FEAT_DotProd) ───────────────
//
// llama.cpp's decode-critical path quantizes the activation vector to int8 once
// per matrix-vector product (Q8_K layout) and then evaluates each weight row
// with `sdot` integer dot products, applying the product of the weight and
// activation block scales only at the very end. This avoids widening every
// 4/6-bit weight to f32 in the inner loop — the dominant cost of the portable
// kernels. `sdot` (FEAT_DotProd) is present on Apple M-series and most ARMv8.2+
// cores; we detect it at runtime and fall back to the f32 kernels otherwise.

#[cfg(all(target_arch = "aarch64", not(target_family = "wasm")))]
#[inline]
/// Reports whether the CPU implements the `sdot`/`udot` dot-product extension.
fn has_dotprod() -> bool {
    static HAS: OnceLock<bool> = OnceLock::new();
    *HAS.get_or_init(|| std::arch::is_aarch64_feature_detected!("dotprod"))
}

/// Per-thread scratch holding one activation vector quantized to Q8_K: `qs`
/// int8 quants, one `d` scale per 256-element super-block, and one signed group
/// sum per 32 elements (used for the Q4_K/Q5_K `dmin` term).
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    not(target_family = "wasm")
))]
#[derive(Default)]
struct XqBuf {
    qs: Vec<i8>,
    d: Vec<f32>,
    bsums: Vec<i16>,
}

/// Owned Q8_K activations for an entire row-major batch. Batched projections
/// share this storage across workers, so row scheduling can keep weights hot
/// without quantizing every activation again on every worker.
#[cfg(not(target_family = "wasm"))]
struct BatchXqBuf {
    qs: Vec<i8>,
    d: Vec<f32>,
    bsums: Vec<i16>,
}

#[cfg(not(target_family = "wasm"))]
fn prepare_batch_xq(inputs: &[f32], cols: usize) -> Option<BatchXqBuf> {
    #[cfg(target_arch = "aarch64")]
    let supported = has_dotprod();
    #[cfg(target_arch = "x86_64")]
    let supported = has_avx2_fma();
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let supported = false;

    if !supported || cols == 0 || cols % 256 != 0 {
        return None;
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    {
        let tokens = inputs.len() / cols;
        let blocks = cols / 256;
        let mut batch = BatchXqBuf {
            qs: vec![0; inputs.len()],
            d: vec![0.0; tokens * blocks],
            bsums: vec![0; tokens * blocks * 8],
        };
        let mut row = XqBuf::default();
        for token in 0..tokens {
            let input = &inputs[token * cols..(token + 1) * cols];
            unsafe { quantize_row_q8k(input, &mut row) };
            batch.qs[token * cols..(token + 1) * cols].copy_from_slice(&row.qs);
            batch.d[token * blocks..(token + 1) * blocks].copy_from_slice(&row.d);
            batch.bsums[token * blocks * 8..(token + 1) * blocks * 8].copy_from_slice(&row.bsums);
        }
        Some(batch)
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        let _ = inputs;
        None
    }
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    not(target_family = "wasm")
))]
thread_local! {
    static XQ_SCRATCH: std::cell::RefCell<XqBuf> = std::cell::RefCell::new(XqBuf::default());
}

/// Quantizes `x` (length a multiple of 256) into this thread's Q8_K scratch and
/// returns borrowing pointers. The buffer lives for the thread's lifetime, so
/// the pointers stay valid until this thread quantizes another vector — which
/// only happens on its next job, after the current one has fully finished.
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    not(target_family = "wasm")
))]
#[inline]
unsafe fn fill_thread_xq_q8k(x: &[f32]) -> XQuant {
    XQ_SCRATCH.with(|cell| {
        let mut buf = cell.borrow_mut();
        quantize_row_q8k(x, &mut buf);
        XQuant {
            qs: buf.qs.as_ptr(),
            d: buf.d.as_ptr(),
            bsums: buf.bsums.as_ptr(),
        }
    })
}

/// Resizes the Q8_K scratch for an `n`-element activation (`n % 256 == 0`).
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    not(target_family = "wasm")
))]
#[inline]
fn resize_xq_buf(buf: &mut XqBuf, n: usize) {
    let nb = n / 256;
    if buf.qs.len() != n {
        buf.qs.resize(n, 0);
    }
    if buf.d.len() != nb {
        buf.d.resize(nb, 0.0);
    }
    if buf.bsums.len() != nb * 8 {
        buf.bsums.resize(nb * 8, 0);
    }
}

/// Quantizes one activation row to Q8_K (symmetric, 127-level, per-256 scale).
/// Portable reference used by tests and as the non-AVX2 x86 fallback; rounding
/// (ties-to-even) matches the NEON `vcvtnq` and AVX2 `cvtps` paths exactly.
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    not(target_family = "wasm")
))]
#[allow(dead_code)]
fn quantize_row_q8k_scalar(x: &[f32], buf: &mut XqBuf) {
    let n = x.len();
    let nb = n / 256;
    resize_xq_buf(buf, n);
    for b in 0..nb {
        let xb = &x[b * 256..(b + 1) * 256];
        let amax = xb.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        buf.d[b] = amax * (1.0 / 127.0);
        let id = if amax > 0.0 { 127.0 / amax } else { 0.0 };
        for sub in 0..8 {
            let mut sum = 0i32;
            for l in 0..32 {
                let q = (xb[sub * 32 + l] * id).round_ties_even() as i32;
                let q = q.clamp(-128, 127) as i8;
                buf.qs[b * 256 + sub * 32 + l] = q;
                sum += q as i32;
            }
            buf.bsums[b * 8 + sub] = sum as i16;
        }
    }
}

/// Quantizes one activation row to Q8_K (symmetric, 127-level, per-256 scale).
#[cfg(all(target_arch = "x86_64", not(target_family = "wasm")))]
#[inline]
unsafe fn quantize_row_q8k(x: &[f32], buf: &mut XqBuf) {
    if has_avx2_fma() {
        quantize_row_q8k_avx2(x, buf);
    } else {
        quantize_row_q8k_scalar(x, buf);
    }
}

/// AVX2 Q8_K activation quantizer: per-256 abs-max scale, round-to-nearest-even
/// int8 quants, and one signed sum per 32-element group.
#[cfg(all(target_arch = "x86_64", not(target_family = "wasm")))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn quantize_row_q8k_avx2(x: &[f32], buf: &mut XqBuf) {
    use std::arch::x86_64::*;
    let n = x.len();
    let nb = n / 256;
    resize_xq_buf(buf, n);
    let xp = x.as_ptr();
    let qp = buf.qs.as_mut_ptr();
    let sign_mask = _mm256_set1_ps(-0.0);
    let perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

    for b in 0..nb {
        let xb = xp.add(b * 256);
        let mut amax_v = _mm256_setzero_ps();
        for i in 0..32 {
            let v = _mm256_loadu_ps(xb.add(i * 8));
            amax_v = _mm256_max_ps(amax_v, _mm256_andnot_ps(sign_mask, v));
        }
        let hi = _mm256_extractf128_ps(amax_v, 1);
        let m4 = _mm_max_ps(_mm256_castps256_ps128(amax_v), hi);
        let m2 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        let m1 = _mm_max_ss(m2, _mm_shuffle_ps(m2, m2, 0x55));
        let amax = _mm_cvtss_f32(m1);

        buf.d[b] = amax * (1.0 / 127.0);
        let id = if amax > 0.0 { 127.0 / amax } else { 0.0 };
        let idv = _mm256_set1_ps(id);

        let qb = qp.add(b * 256);
        let bbase = b * 8;
        for sub in 0..8 {
            let off = sub * 32;
            // Default MXCSR rounding is nearest-even, matching NEON vcvtnq.
            let i0 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xb.add(off)), idv));
            let i1 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xb.add(off + 8)), idv));
            let i2 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xb.add(off + 16)), idv));
            let i3 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xb.add(off + 24)), idv));

            let sum_v = _mm256_add_epi32(_mm256_add_epi32(i0, i1), _mm256_add_epi32(i2, i3));
            let sum_hi = _mm256_extracti128_si256(sum_v, 1);
            let s4 = _mm_add_epi32(_mm256_castsi256_si128(sum_v), sum_hi);
            let s2 = _mm_add_epi32(s4, _mm_unpackhi_epi64(s4, s4));
            let s1 = _mm_add_epi32(s2, _mm_shuffle_epi32(s2, 0x55));
            buf.bsums[bbase + sub] = _mm_cvtsi128_si32(s1) as i16;

            let p01 = _mm256_packs_epi32(i0, i1);
            let p23 = _mm256_packs_epi32(i2, i3);
            let packed = _mm256_packs_epi16(p01, p23);
            let ordered = _mm256_permutevar8x32_epi32(packed, perm);
            _mm256_storeu_si256(qb.add(off) as *mut __m256i, ordered);
        }
    }
}

/// Quantizes one activation row to Q8_K (symmetric, 127-level, per-256 scale).
#[cfg(all(target_arch = "aarch64", not(target_family = "wasm")))]
#[inline]
unsafe fn quantize_row_q8k(x: &[f32], buf: &mut XqBuf) {
    use std::arch::aarch64::*;
    let n = x.len();
    let nb = n / 256;
    resize_xq_buf(buf, n);
    let xp = x.as_ptr();
    let qp = buf.qs.as_mut_ptr();

    for b in 0..nb {
        let xb = xp.add(b * 256);
        let mut amax = vdupq_n_f32(0.0);
        for i in 0..64 {
            amax = vmaxq_f32(amax, vabsq_f32(vld1q_f32(xb.add(i * 4))));
        }
        let amax = vmaxvq_f32(amax);
        buf.d[b] = amax * (1.0 / 127.0);
        let id = if amax > 0.0 { 127.0 / amax } else { 0.0 };
        let idv = vdupq_n_f32(id);

        let qb = qp.add(b * 256);
        let bbase = b * 8;
        for sub in 0..8 {
            let off = sub * 32;
            let lo = quantize16_q8(xb.add(off), idv);
            let hi = quantize16_q8(xb.add(off + 16), idv);
            vst1q_s8(qb.add(off), lo);
            vst1q_s8(qb.add(off + 16), hi);
            let s = vaddlvq_s8(lo) as i32 + vaddlvq_s8(hi) as i32;
            buf.bsums[bbase + sub] = s as i16;
        }
    }
}

/// Quantizes 16 contiguous floats to int8 (`round(x * id)`, saturating).
#[cfg(all(target_arch = "aarch64", not(target_family = "wasm")))]
#[inline(always)]
unsafe fn quantize16_q8(
    xp: *const f32,
    idv: std::arch::aarch64::float32x4_t,
) -> std::arch::aarch64::int8x16_t {
    use std::arch::aarch64::*;
    let q0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xp), idv));
    let q1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xp.add(4)), idv));
    let q2 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xp.add(8)), idv));
    let q3 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xp.add(12)), idv));
    let lo = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
    let hi = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
    vcombine_s8(vqmovn_s16(lo), vqmovn_s16(hi))
}

/// Single `sdot` instruction: `acc[j] += sum_k a[4j+k] * b[4j+k]`.
/// `vdotq_s32` is still unstable on stable Rust, so we emit it via inline asm.
/// Only reached from `#[target_feature(enable = "dotprod")]` kernels after a
/// runtime `has_dotprod()` check, so the instruction is always legal here.
#[cfg(all(target_arch = "aarch64", not(target_family = "wasm")))]
#[inline]
#[target_feature(enable = "dotprod")]
unsafe fn sdot(
    acc: std::arch::aarch64::int32x4_t,
    a: std::arch::aarch64::int8x16_t,
    b: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int32x4_t {
    let mut out = acc;
    core::arch::asm!(
        "sdot {out:v}.4s, {a:v}.16b, {b:v}.16b",
        out = inout(vreg) out,
        a = in(vreg) a,
        b = in(vreg) b,
        options(pure, nomem, nostack),
    );
    out
}

/// Q4_K · Q8_K integer dot product. Mirrors `dot_q4_k_f32_neon` but keeps the
/// inner products in int32 via `sdot`, folding the `dmin` term through the
/// pre-computed per-32 activation group sums (`bsums`).
#[cfg(all(target_arch = "aarch64", not(target_family = "wasm")))]
#[target_feature(enable = "dotprod")]
unsafe fn dot_q4_k_q8k_neon(qdata: &[u8], xq: XQuant, n: usize) -> f32 {
    use std::arch::aarch64::*;
    let nb = n / 256;
    let mask = vdupq_n_u8(0x0F);
    // Keep the four partial dot lanes live within each K block. Reducing every
    // 32-value sub-block with `vaddvq_s32` serializes the dot-product pipe;
    // one reduction per 256-value block leaves the independent sub-blocks in
    // flight without changing the scalar block accumulation semantics.
    let mut acc = 0.0f32;

    for b in 0..nb {
        let block = qdata.as_ptr().add(b * 144);
        let d = f16_to_f32(u16::from_le_bytes([*block, *block.add(1)]));
        let dmin = f16_to_f32(u16::from_le_bytes([*block.add(2), *block.add(3)]));
        let scales: &[u8; 12] = std::slice::from_raw_parts(block.add(4), 12)
            .try_into()
            .expect("q4_k scales size");
        let q = block.add(16);
        let dx = *xq.d.add(b);
        let xqb = xq.qs.add(b * 256);
        let bsb = xq.bsums.add(b * 8);

        let mut isum = vdupq_n_s32(0);
        let mut imin = 0i32;
        for c in 0..4usize {
            let is = c * 2;
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);

            // 32 nibble-bytes: low nibbles → sub-block A, high nibbles → sub-block B.
            let n0 = vld1q_u8(q.add(c * 32));
            let n1 = vld1q_u8(q.add(c * 32 + 16));
            let a0 = vreinterpretq_s8_u8(vandq_u8(n0, mask));
            let a1 = vreinterpretq_s8_u8(vandq_u8(n1, mask));
            let b0 = vreinterpretq_s8_u8(vshrq_n_u8(n0, 4));
            let b1 = vreinterpretq_s8_u8(vshrq_n_u8(n1, 4));

            let xa0 = vld1q_s8(xqb.add(c * 64));
            let xa1 = vld1q_s8(xqb.add(c * 64 + 16));
            let xb0 = vld1q_s8(xqb.add(c * 64 + 32));
            let xb1 = vld1q_s8(xqb.add(c * 64 + 48));

            let qa = sdot(sdot(vdupq_n_s32(0), a0, xa0), a1, xa1);
            let qb = sdot(sdot(vdupq_n_s32(0), b0, xb0), b1, xb1);

            isum = vaddq_s32(
                isum,
                vaddq_s32(
                    vmulq_s32(qa, vdupq_n_s32(sc1 as i32)),
                    vmulq_s32(qb, vdupq_n_s32(sc2 as i32)),
                ),
            );
            imin += m1 as i32 * (*bsb.add(is) as i32) + m2 as i32 * (*bsb.add(is + 1) as i32);
        }
        let isum = vaddvq_s32(isum) as f32;
        acc += dx * (d * isum - dmin * imin as f32);
    }
    acc
}

/// Q6_K · Q8_K integer dot product. Mirrors `dot_q6_k_f32_neon`, replacing the
/// dequant-to-f32 + FMA inner loop with `sdot`; the `-32` weight bias is folded
/// into the signed weights so no separate correction term is needed.
#[cfg(all(target_arch = "aarch64", not(target_family = "wasm")))]
#[target_feature(enable = "dotprod")]
unsafe fn dot_q6_k_q8k_neon(qdata: &[u8], xq: XQuant, n: usize) -> f32 {
    use std::arch::aarch64::*;
    let nb = n / 256;
    let mask_lo4 = vdupq_n_u8(0x0F);
    let mask_03 = vdupq_n_u8(0x03);
    let sub32 = vdupq_n_u8(32);
    let mut acc = 0.0f32;

    for b in 0..nb {
        let block = qdata.as_ptr().add(b * 210);
        let d = f16_to_f32(u16::from_le_bytes([*block.add(208), *block.add(209)]));
        let dx = *xq.d.add(b);
        let xqb = xq.qs.add(b * 256);
        let mut ql_ptr = block;
        let mut qh_ptr = block.add(128);
        let mut sc_ptr = block.add(192);
        let mut grp_x_base = 0usize;
        let mut isum = vdupq_n_s32(0);

        for _grp in 0..2 {
            for half in 0..2usize {
                let l = half * 16;
                let is = half;

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

                let x_off = grp_x_base + l;
                let p1 = sdot(vdupq_n_s32(0), q1, vld1q_s8(xqb.add(x_off)));
                let p2 = sdot(vdupq_n_s32(0), q2, vld1q_s8(xqb.add(x_off + 32)));
                let p3 = sdot(vdupq_n_s32(0), q3, vld1q_s8(xqb.add(x_off + 64)));
                let p4 = sdot(vdupq_n_s32(0), q4, vld1q_s8(xqb.add(x_off + 96)));

                let s1 = vdupq_n_s32((*sc_ptr.add(is) as i8) as i32);
                let s2 = vdupq_n_s32((*sc_ptr.add(is + 2) as i8) as i32);
                let s3 = vdupq_n_s32((*sc_ptr.add(is + 4) as i8) as i32);
                let s4 = vdupq_n_s32((*sc_ptr.add(is + 6) as i8) as i32);
                isum = vaddq_s32(
                    isum,
                    vaddq_s32(
                        vaddq_s32(vmulq_s32(p1, s1), vmulq_s32(p2, s2)),
                        vaddq_s32(vmulq_s32(p3, s3), vmulq_s32(p4, s4)),
                    ),
                );
            }
            ql_ptr = ql_ptr.add(64);
            qh_ptr = qh_ptr.add(32);
            sc_ptr = sc_ptr.add(8);
            grp_x_base += 128;
        }
        acc += d * dx * vaddvq_s32(isum) as f32;
    }
    acc
}

/// Q5_K · Q8_K integer dot product. Mirrors `dot_q5_k_f32_scalar`, keeping the
/// inner products in int32 via `sdot`; the `dmin` term folds through the
/// pre-computed per-32 activation group sums like the Q4_K kernel.
#[cfg(all(target_arch = "aarch64", not(target_family = "wasm")))]
#[target_feature(enable = "dotprod")]
unsafe fn dot_q5_k_q8k_neon(qdata: &[u8], xq: XQuant, n: usize) -> f32 {
    use std::arch::aarch64::*;
    let nb = n / 256;
    let mask = vdupq_n_u8(0x0F);
    let hbit = vdupq_n_u8(0x10);
    let mut acc = 0.0f32;

    for b in 0..nb {
        let block = qdata.as_ptr().add(b * 176);
        let d = f16_to_f32(u16::from_le_bytes([*block, *block.add(1)]));
        let dmin = f16_to_f32(u16::from_le_bytes([*block.add(2), *block.add(3)]));
        let scales: &[u8; 12] = std::slice::from_raw_parts(block.add(4), 12)
            .try_into()
            .expect("q5_k scales size");
        let qh0 = vld1q_u8(block.add(16));
        let qh1 = vld1q_u8(block.add(32));
        let q = block.add(48);
        let dx = *xq.d.add(b);
        let xqb = xq.qs.add(b * 256);
        let bsb = xq.bsums.add(b * 8);

        let mut isum = vdupq_n_s32(0);
        let mut imin = 0i32;
        for c in 0..4usize {
            let is = c * 2;
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let u1 = vdupq_n_u8(1u8 << (2 * c));
            let u2 = vdupq_n_u8(2u8 << (2 * c));

            // 32 nibble-bytes: low nibbles → sub-block A, high nibbles → B;
            // qh byte l supplies bit 4 for elements A[l] (mask u1) and B[l]
            // (mask u2). Values stay in 0..=31, safely positive as i8.
            let n0 = vld1q_u8(q.add(c * 32));
            let n1 = vld1q_u8(q.add(c * 32 + 16));
            let a0 = vorrq_u8(vandq_u8(n0, mask), vandq_u8(vtstq_u8(qh0, u1), hbit));
            let a1 = vorrq_u8(vandq_u8(n1, mask), vandq_u8(vtstq_u8(qh1, u1), hbit));
            let b0 = vorrq_u8(vshrq_n_u8(n0, 4), vandq_u8(vtstq_u8(qh0, u2), hbit));
            let b1 = vorrq_u8(vshrq_n_u8(n1, 4), vandq_u8(vtstq_u8(qh1, u2), hbit));

            let xa0 = vld1q_s8(xqb.add(c * 64));
            let xa1 = vld1q_s8(xqb.add(c * 64 + 16));
            let xb0 = vld1q_s8(xqb.add(c * 64 + 32));
            let xb1 = vld1q_s8(xqb.add(c * 64 + 48));

            let qa = sdot(
                sdot(vdupq_n_s32(0), vreinterpretq_s8_u8(a0), xa0),
                vreinterpretq_s8_u8(a1),
                xa1,
            );
            let qb = sdot(
                sdot(vdupq_n_s32(0), vreinterpretq_s8_u8(b0), xb0),
                vreinterpretq_s8_u8(b1),
                xb1,
            );

            isum = vaddq_s32(
                isum,
                vaddq_s32(
                    vmulq_s32(qa, vdupq_n_s32(sc1 as i32)),
                    vmulq_s32(qb, vdupq_n_s32(sc2 as i32)),
                ),
            );
            imin += m1 as i32 * (*bsb.add(is) as i32) + m2 as i32 * (*bsb.add(is + 1) as i32);
        }
        acc += dx * (d * vaddvq_s32(isum) as f32 - dmin * imin as f32);
    }
    acc
}

/// Q4_K · Q8_K for `KQUANT_BATCH_TOKENS` activations against one weight row.
///
/// Arithmetic per lane is identical to `dot_q4_k_q8k_neon`, in the same order:
/// only the token-independent weight work — the two f16 scale conversions, the
/// eight `get_scale_min_k4` unpacks and the nibble split — moves out of the
/// token loop, so each lane stays bit-identical to the single-token kernel
/// while that setup is paid once instead of four times. Not yet run on Apple
/// hardware; the parity test covers it wherever it does run.
#[cfg(all(target_arch = "aarch64", not(target_family = "wasm")))]
#[target_feature(enable = "dotprod")]
unsafe fn dot_q4_k_q8k_neon_x4(
    qdata: &[u8],
    xqs: &[XQuant; KQUANT_BATCH_TOKENS],
    n: usize,
) -> [f32; KQUANT_BATCH_TOKENS] {
    use std::arch::aarch64::*;
    let nb = n / 256;
    let mask = vdupq_n_u8(0x0F);
    let mut acc = [0.0f32; KQUANT_BATCH_TOKENS];

    for b in 0..nb {
        let block = qdata.as_ptr().add(b * 144);
        let d = f16_to_f32(u16::from_le_bytes([*block, *block.add(1)]));
        let dmin = f16_to_f32(u16::from_le_bytes([*block.add(2), *block.add(3)]));
        let scales: &[u8; 12] = std::slice::from_raw_parts(block.add(4), 12)
            .try_into()
            .expect("q4_k scales size");
        let q = block.add(16);

        let mut isum = [vdupq_n_s32(0); KQUANT_BATCH_TOKENS];
        let mut imin = [0i32; KQUANT_BATCH_TOKENS];
        for c in 0..4usize {
            let is = c * 2;
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let sc1_v = vdupq_n_s32(sc1 as i32);
            let sc2_v = vdupq_n_s32(sc2 as i32);

            let n0 = vld1q_u8(q.add(c * 32));
            let n1 = vld1q_u8(q.add(c * 32 + 16));
            let a0 = vreinterpretq_s8_u8(vandq_u8(n0, mask));
            let a1 = vreinterpretq_s8_u8(vandq_u8(n1, mask));
            let b0 = vreinterpretq_s8_u8(vshrq_n_u8(n0, 4));
            let b1 = vreinterpretq_s8_u8(vshrq_n_u8(n1, 4));

            for t in 0..KQUANT_BATCH_TOKENS {
                let xqb = xqs[t].qs.add(b * 256);
                let bsb = xqs[t].bsums.add(b * 8);
                let xa0 = vld1q_s8(xqb.add(c * 64));
                let xa1 = vld1q_s8(xqb.add(c * 64 + 16));
                let xb0 = vld1q_s8(xqb.add(c * 64 + 32));
                let xb1 = vld1q_s8(xqb.add(c * 64 + 48));

                let qa = sdot(sdot(vdupq_n_s32(0), a0, xa0), a1, xa1);
                let qb = sdot(sdot(vdupq_n_s32(0), b0, xb0), b1, xb1);
                isum[t] = vaddq_s32(
                    isum[t],
                    vaddq_s32(vmulq_s32(qa, sc1_v), vmulq_s32(qb, sc2_v)),
                );
                imin[t] +=
                    m1 as i32 * (*bsb.add(is) as i32) + m2 as i32 * (*bsb.add(is + 1) as i32);
            }
        }
        for t in 0..KQUANT_BATCH_TOKENS {
            let dx = *xqs[t].d.add(b);
            let total = vaddvq_s32(isum[t]) as f32;
            acc[t] += dx * (d * total - dmin * imin[t] as f32);
        }
    }
    acc
}

/// Q5_K · Q8_K for `KQUANT_BATCH_TOKENS` activations against one weight row.
/// Token-blocked mirror of `dot_q5_k_q8k_neon`; expanding the fifth weight bit
/// from the `qh` planes is weight-side work and hoists out with the rest.
#[cfg(all(target_arch = "aarch64", not(target_family = "wasm")))]
#[target_feature(enable = "dotprod")]
unsafe fn dot_q5_k_q8k_neon_x4(
    qdata: &[u8],
    xqs: &[XQuant; KQUANT_BATCH_TOKENS],
    n: usize,
) -> [f32; KQUANT_BATCH_TOKENS] {
    use std::arch::aarch64::*;
    let nb = n / 256;
    let mask = vdupq_n_u8(0x0F);
    let hbit = vdupq_n_u8(0x10);
    let mut acc = [0.0f32; KQUANT_BATCH_TOKENS];

    for b in 0..nb {
        let block = qdata.as_ptr().add(b * 176);
        let d = f16_to_f32(u16::from_le_bytes([*block, *block.add(1)]));
        let dmin = f16_to_f32(u16::from_le_bytes([*block.add(2), *block.add(3)]));
        let scales: &[u8; 12] = std::slice::from_raw_parts(block.add(4), 12)
            .try_into()
            .expect("q5_k scales size");
        let qh0 = vld1q_u8(block.add(16));
        let qh1 = vld1q_u8(block.add(32));
        let q = block.add(48);

        let mut isum = [vdupq_n_s32(0); KQUANT_BATCH_TOKENS];
        let mut imin = [0i32; KQUANT_BATCH_TOKENS];
        for c in 0..4usize {
            let is = c * 2;
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let sc1_v = vdupq_n_s32(sc1 as i32);
            let sc2_v = vdupq_n_s32(sc2 as i32);
            let u1 = vdupq_n_u8(1u8 << (2 * c));
            let u2 = vdupq_n_u8(2u8 << (2 * c));

            let n0 = vld1q_u8(q.add(c * 32));
            let n1 = vld1q_u8(q.add(c * 32 + 16));
            let a0 = vreinterpretq_s8_u8(vorrq_u8(
                vandq_u8(n0, mask),
                vandq_u8(vtstq_u8(qh0, u1), hbit),
            ));
            let a1 = vreinterpretq_s8_u8(vorrq_u8(
                vandq_u8(n1, mask),
                vandq_u8(vtstq_u8(qh1, u1), hbit),
            ));
            let b0 = vreinterpretq_s8_u8(vorrq_u8(
                vshrq_n_u8(n0, 4),
                vandq_u8(vtstq_u8(qh0, u2), hbit),
            ));
            let b1 = vreinterpretq_s8_u8(vorrq_u8(
                vshrq_n_u8(n1, 4),
                vandq_u8(vtstq_u8(qh1, u2), hbit),
            ));

            for t in 0..KQUANT_BATCH_TOKENS {
                let xqb = xqs[t].qs.add(b * 256);
                let bsb = xqs[t].bsums.add(b * 8);
                let xa0 = vld1q_s8(xqb.add(c * 64));
                let xa1 = vld1q_s8(xqb.add(c * 64 + 16));
                let xb0 = vld1q_s8(xqb.add(c * 64 + 32));
                let xb1 = vld1q_s8(xqb.add(c * 64 + 48));

                let qa = sdot(sdot(vdupq_n_s32(0), a0, xa0), a1, xa1);
                let qb = sdot(sdot(vdupq_n_s32(0), b0, xb0), b1, xb1);
                isum[t] = vaddq_s32(
                    isum[t],
                    vaddq_s32(vmulq_s32(qa, sc1_v), vmulq_s32(qb, sc2_v)),
                );
                imin[t] +=
                    m1 as i32 * (*bsb.add(is) as i32) + m2 as i32 * (*bsb.add(is + 1) as i32);
            }
        }
        for t in 0..KQUANT_BATCH_TOKENS {
            let dx = *xqs[t].d.add(b);
            let total = vaddvq_s32(isum[t]) as f32;
            acc[t] += dx * (d * total - dmin * imin[t] as f32);
        }
    }
    acc
}

/// Q6_K · Q8_K for `KQUANT_BATCH_TOKENS` activations against one weight row.
/// Token-blocked mirror of `dot_q6_k_q8k_neon`. Q6_K gains the most: each
/// 16-byte sub-vector costs a load plus five ops to reassemble the 6-bit
/// weights and fold the `-32` bias, and none of that depends on the token.
#[cfg(all(target_arch = "aarch64", not(target_family = "wasm")))]
#[target_feature(enable = "dotprod")]
unsafe fn dot_q6_k_q8k_neon_x4(
    qdata: &[u8],
    xqs: &[XQuant; KQUANT_BATCH_TOKENS],
    n: usize,
) -> [f32; KQUANT_BATCH_TOKENS] {
    use std::arch::aarch64::*;
    let nb = n / 256;
    let mask_lo4 = vdupq_n_u8(0x0F);
    let mask_03 = vdupq_n_u8(0x03);
    let sub32 = vdupq_n_u8(32);
    let mut acc = [0.0f32; KQUANT_BATCH_TOKENS];

    for b in 0..nb {
        let block = qdata.as_ptr().add(b * 210);
        let d = f16_to_f32(u16::from_le_bytes([*block.add(208), *block.add(209)]));
        let mut ql_ptr = block;
        let mut qh_ptr = block.add(128);
        let mut sc_ptr = block.add(192);
        let mut grp_x_base = 0usize;
        let mut isum = [vdupq_n_s32(0); KQUANT_BATCH_TOKENS];

        for _grp in 0..2 {
            for half in 0..2usize {
                let l = half * 16;
                let is = half;

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

                let s1 = vdupq_n_s32((*sc_ptr.add(is) as i8) as i32);
                let s2 = vdupq_n_s32((*sc_ptr.add(is + 2) as i8) as i32);
                let s3 = vdupq_n_s32((*sc_ptr.add(is + 4) as i8) as i32);
                let s4 = vdupq_n_s32((*sc_ptr.add(is + 6) as i8) as i32);

                let x_off = grp_x_base + l;
                for t in 0..KQUANT_BATCH_TOKENS {
                    let xqb = xqs[t].qs.add(b * 256);
                    let p1 = sdot(vdupq_n_s32(0), q1, vld1q_s8(xqb.add(x_off)));
                    let p2 = sdot(vdupq_n_s32(0), q2, vld1q_s8(xqb.add(x_off + 32)));
                    let p3 = sdot(vdupq_n_s32(0), q3, vld1q_s8(xqb.add(x_off + 64)));
                    let p4 = sdot(vdupq_n_s32(0), q4, vld1q_s8(xqb.add(x_off + 96)));
                    isum[t] = vaddq_s32(
                        isum[t],
                        vaddq_s32(
                            vaddq_s32(vmulq_s32(p1, s1), vmulq_s32(p2, s2)),
                            vaddq_s32(vmulq_s32(p3, s3), vmulq_s32(p4, s4)),
                        ),
                    );
                }
            }
            ql_ptr = ql_ptr.add(64);
            qh_ptr = qh_ptr.add(32);
            sc_ptr = sc_ptr.add(8);
            grp_x_base += 128;
        }
        for t in 0..KQUANT_BATCH_TOKENS {
            let dx = *xqs[t].d.add(b);
            acc[t] += d * dx * vaddvq_s32(isum[t]) as f32;
        }
    }
    acc
}

/// Q4_0 · Q8_K integer dot product: the `-8` nibble bias is folded into the
/// signed weights, so each 32-weight block is exactly two `sdot` instructions.
/// Eight Q4_0 blocks share one Q8_K activation super-block scale.
#[cfg(all(target_arch = "aarch64", not(target_family = "wasm")))]
#[target_feature(enable = "dotprod")]
unsafe fn dot_q4_0_q8k_neon(qdata: &[u8], xq: XQuant, n: usize) -> f32 {
    use std::arch::aarch64::*;
    let n_super = n / 256;
    let mask = vdupq_n_u8(0x0F);
    let eight = vdupq_n_u8(8);
    let mut acc = vdupq_n_f32(0.0);

    for sb in 0..n_super {
        let dx = *xq.d.add(sb);
        let xqb = xq.qs.add(sb * 256);
        let base = qdata.as_ptr().add(sb * 8 * 18);
        let mut inner = vdupq_n_f32(0.0);
        for pair in 0..4usize {
            let blk = pair * 2;
            let block0 = base.add(blk * 18);
            let block1 = base.add((blk + 1) * 18);
            let d0 = f16_to_f32(u16::from_le_bytes([*block0, *block0.add(1)]));
            let d1 = f16_to_f32(u16::from_le_bytes([*block1, *block1.add(1)]));

            // lo nibble of byte i → weight[i], hi nibble → weight[i+16]; the
            // u8 subtract wraps to the same bit pattern as a signed subtract.
            let nib0 = vld1q_u8(block0.add(2));
            let a0 = vreinterpretq_s8_u8(vsubq_u8(vandq_u8(nib0, mask), eight));
            let b0 = vreinterpretq_s8_u8(vsubq_u8(vshrq_n_u8(nib0, 4), eight));
            let nib1 = vld1q_u8(block1.add(2));
            let a1 = vreinterpretq_s8_u8(vsubq_u8(vandq_u8(nib1, mask), eight));
            let b1 = vreinterpretq_s8_u8(vsubq_u8(vshrq_n_u8(nib1, 4), eight));

            let x0 = vld1q_s8(xqb.add(blk * 32));
            let x1 = vld1q_s8(xqb.add(blk * 32 + 16));
            let x2 = vld1q_s8(xqb.add(blk * 32 + 32));
            let x3 = vld1q_s8(xqb.add(blk * 32 + 48));

            let s0 = sdot(sdot(vdupq_n_s32(0), a0, x0), b0, x1);
            let s1 = sdot(sdot(vdupq_n_s32(0), a1, x2), b1, x3);
            inner = vmlaq_f32(inner, vcvtq_f32_s32(s0), vdupq_n_f32(d0));
            inner = vmlaq_f32(inner, vcvtq_f32_s32(s1), vdupq_n_f32(d1));
        }
        acc = vmlaq_f32(acc, inner, vdupq_n_f32(dx));
    }
    vaddvq_f32(acc)
}

/// Q8_0 · Q8_K integer dot product: two `sdot` instructions per 32-weight
/// block; eight Q8_0 blocks share one Q8_K activation super-block scale.
#[cfg(all(target_arch = "aarch64", not(target_family = "wasm")))]
#[target_feature(enable = "dotprod")]
unsafe fn dot_q8_0_q8k_neon(qdata: &[u8], xq: XQuant, n: usize) -> f32 {
    use std::arch::aarch64::*;
    let n_super = n / 256;
    let mut acc = vdupq_n_f32(0.0);

    for sb in 0..n_super {
        let dx = *xq.d.add(sb);
        let xqb = xq.qs.add(sb * 256);
        let base = qdata.as_ptr().add(sb * 8 * 34);
        let mut inner = vdupq_n_f32(0.0);
        for pair in 0..4usize {
            let blk = pair * 2;
            let block0 = base.add(blk * 34);
            let block1 = base.add((blk + 1) * 34);
            let d0 = f16_to_f32(u16::from_le_bytes([*block0, *block0.add(1)]));
            let d1 = f16_to_f32(u16::from_le_bytes([*block1, *block1.add(1)]));

            let w0 = vld1q_s8(block0.add(2) as *const i8);
            let w1 = vld1q_s8(block0.add(18) as *const i8);
            let w2 = vld1q_s8(block1.add(2) as *const i8);
            let w3 = vld1q_s8(block1.add(18) as *const i8);

            let x0 = vld1q_s8(xqb.add(blk * 32));
            let x1 = vld1q_s8(xqb.add(blk * 32 + 16));
            let x2 = vld1q_s8(xqb.add(blk * 32 + 32));
            let x3 = vld1q_s8(xqb.add(blk * 32 + 48));

            let s0 = sdot(sdot(vdupq_n_s32(0), w0, x0), w1, x1);
            let s1 = sdot(sdot(vdupq_n_s32(0), w2, x2), w3, x3);
            inner = vmlaq_f32(inner, vcvtq_f32_s32(s0), vdupq_n_f32(d0));
            inner = vmlaq_f32(inner, vcvtq_f32_s32(s1), vdupq_n_f32(d1));
        }
        acc = vmlaq_f32(acc, inner, vdupq_n_f32(dx));
    }
    vaddvq_f32(acc)
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
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
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

/// NEON MXFP4 dot product. Each 17-byte block stores 16 packed nibbles plus a
/// shared exponent, with consecutive low/high nibbles mapping to consecutive
/// activation values. A table lookup turns the nonlinear E2M1 nibbles into
/// signed integer mantissas before the vector FMA path consumes them.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn dot_mxfp4_f32_neon(qdata: &[u8], x: &[f32], n: usize) -> f32 {
    use std::arch::aarch64::*;

    const MANTISSAS: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

    let n_blocks = n / 32;
    let mantissas = vld1q_s8(MANTISSAS.as_ptr());
    let nibble_mask = vdupq_n_u8(0x0f);
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut block_idx = 0usize;

    while block_idx + 1 < n_blocks {
        let block = qdata.as_ptr().add(block_idx * 17);
        let next_block = block.add(17);
        let x_ptr = x.as_ptr().add(block_idx * 32);
        // The lookup table stores E2M1 mantissas in half-units so they fit in
        // signed bytes; fold the 0.5 conversion into the shared exponent.
        let scale = mxfp4_scale_to_f32(*block.add(16)) * 0.5;
        let next_scale = mxfp4_scale_to_f32(*next_block.add(16)) * 0.5;
        acc0 = vmlaq_f32(
            acc0,
            dot_mxfp4_block_f32_neon(block, x_ptr, mantissas, nibble_mask),
            vdupq_n_f32(scale),
        );
        acc1 = vmlaq_f32(
            acc1,
            dot_mxfp4_block_f32_neon(next_block, x_ptr.add(32), mantissas, nibble_mask),
            vdupq_n_f32(next_scale),
        );
        block_idx += 2;
    }

    if block_idx < n_blocks {
        let block = qdata.as_ptr().add(block_idx * 17);
        let scale = mxfp4_scale_to_f32(*block.add(16)) * 0.5;
        acc0 = vmlaq_f32(
            acc0,
            dot_mxfp4_block_f32_neon(
                block,
                x.as_ptr().add(block_idx * 32),
                mantissas,
                nibble_mask,
            ),
            vdupq_n_f32(scale),
        );
    }

    vaddvq_f32(vaddq_f32(acc0, acc1))
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn dot_mxfp4_block_f32_neon(
    block: *const u8,
    x: *const f32,
    mantissas: std::arch::aarch64::int8x16_t,
    nibble_mask: std::arch::aarch64::uint8x16_t,
) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;

    let packed = vld1q_u8(block);
    let lo = vqtbl1q_s8(mantissas, vandq_u8(packed, nibble_mask));
    let hi = vqtbl1q_s8(mantissas, vshrq_n_u8(packed, 4));
    let x0 = vld2q_f32(x);
    let x1 = vld2q_f32(x.add(8));
    let x2 = vld2q_f32(x.add(16));
    let x3 = vld2q_f32(x.add(24));
    let mut acc = vdupq_n_f32(0.0);

    macro_rules! accumulate_nibbles {
        ($values:expr, $x0:expr, $x1:expr, $x2:expr, $x3:expr) => {{
            let first = vmovl_s8(vget_low_s8($values));
            let second = vmovl_s8(vget_high_s8($values));
            acc = vmlaq_f32(acc, vcvtq_f32_s32(vmovl_s16(vget_low_s16(first))), $x0);
            acc = vmlaq_f32(acc, vcvtq_f32_s32(vmovl_s16(vget_high_s16(first))), $x1);
            acc = vmlaq_f32(acc, vcvtq_f32_s32(vmovl_s16(vget_low_s16(second))), $x2);
            acc = vmlaq_f32(acc, vcvtq_f32_s32(vmovl_s16(vget_high_s16(second))), $x3);
        }};
    }

    accumulate_nibbles!(lo, x0.0, x1.0, x2.0, x3.0);
    accumulate_nibbles!(hi, x0.1, x1.1, x2.1, x3.1);
    acc
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
unsafe fn dot_f32x4_neon(a0: &[f32], a1: &[f32], a2: &[f32], a3: &[f32], b: &[f32]) -> [f32; 4] {
    use std::arch::aarch64::*;
    let n = b.len();
    let mut sums = [vdupq_n_f32(0.0); 4];
    let mut i = 0;
    while i + 4 <= n {
        let key = vld1q_f32(b.as_ptr().add(i));
        sums[0] = vmlaq_f32(sums[0], vld1q_f32(a0.as_ptr().add(i)), key);
        sums[1] = vmlaq_f32(sums[1], vld1q_f32(a1.as_ptr().add(i)), key);
        sums[2] = vmlaq_f32(sums[2], vld1q_f32(a2.as_ptr().add(i)), key);
        sums[3] = vmlaq_f32(sums[3], vld1q_f32(a3.as_ptr().add(i)), key);
        i += 4;
    }
    let mut out = [
        vaddvq_f32(sums[0]),
        vaddvq_f32(sums[1]),
        vaddvq_f32(sums[2]),
        vaddvq_f32(sums[3]),
    ];
    while i < n {
        let key = b[i];
        out[0] += a0[i] * key;
        out[1] += a1[i] * key;
        out[2] += a2[i] * key;
        out[3] += a3[i] * key;
        i += 1;
    }
    out
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn affine_add_f32x4_neon(
    out0: &mut [f32],
    out1: &mut [f32],
    out2: &mut [f32],
    out3: &mut [f32],
    multiplier: [f32; 4],
    alpha: [f32; 4],
    x: &[f32],
) {
    use std::arch::aarch64::*;
    let mul = [
        vdupq_n_f32(multiplier[0]),
        vdupq_n_f32(multiplier[1]),
        vdupq_n_f32(multiplier[2]),
        vdupq_n_f32(multiplier[3]),
    ];
    let add = [
        vdupq_n_f32(alpha[0]),
        vdupq_n_f32(alpha[1]),
        vdupq_n_f32(alpha[2]),
        vdupq_n_f32(alpha[3]),
    ];
    let mut i = 0;
    while i + 4 <= x.len() {
        let value = vld1q_f32(x.as_ptr().add(i));
        vst1q_f32(
            out0.as_mut_ptr().add(i),
            vmlaq_f32(
                vmulq_f32(vld1q_f32(out0.as_ptr().add(i)), mul[0]),
                value,
                add[0],
            ),
        );
        vst1q_f32(
            out1.as_mut_ptr().add(i),
            vmlaq_f32(
                vmulq_f32(vld1q_f32(out1.as_ptr().add(i)), mul[1]),
                value,
                add[1],
            ),
        );
        vst1q_f32(
            out2.as_mut_ptr().add(i),
            vmlaq_f32(
                vmulq_f32(vld1q_f32(out2.as_ptr().add(i)), mul[2]),
                value,
                add[2],
            ),
        );
        vst1q_f32(
            out3.as_mut_ptr().add(i),
            vmlaq_f32(
                vmulq_f32(vld1q_f32(out3.as_ptr().add(i)), mul[3]),
                value,
                add[3],
            ),
        );
        i += 4;
    }
    while i < x.len() {
        out0[i] = out0[i] * multiplier[0] + alpha[0] * x[i];
        out1[i] = out1[i] * multiplier[1] + alpha[1] * x[i];
        out2[i] = out2[i] * multiplier[2] + alpha[2] * x[i];
        out3[i] = out3[i] * multiplier[3] + alpha[3] * x[i];
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn affine_add_dot_f32x4_neon(
    out0: &mut [f32],
    out1: &mut [f32],
    out2: &mut [f32],
    out3: &mut [f32],
    multiplier: [f32; 4],
    alpha: [f32; 4],
    x: &[f32],
    dot: &[f32],
) -> [f32; 4] {
    use std::arch::aarch64::*;
    let mul = [
        vdupq_n_f32(multiplier[0]),
        vdupq_n_f32(multiplier[1]),
        vdupq_n_f32(multiplier[2]),
        vdupq_n_f32(multiplier[3]),
    ];
    let add = [
        vdupq_n_f32(alpha[0]),
        vdupq_n_f32(alpha[1]),
        vdupq_n_f32(alpha[2]),
        vdupq_n_f32(alpha[3]),
    ];
    let mut sums = [vdupq_n_f32(0.0); 4];
    let mut i = 0usize;
    while i + 4 <= x.len() {
        let value = vld1q_f32(x.as_ptr().add(i));
        let projection = vld1q_f32(dot.as_ptr().add(i));
        let updated0 = vmlaq_f32(
            vmulq_f32(vld1q_f32(out0.as_ptr().add(i)), mul[0]),
            value,
            add[0],
        );
        let updated1 = vmlaq_f32(
            vmulq_f32(vld1q_f32(out1.as_ptr().add(i)), mul[1]),
            value,
            add[1],
        );
        let updated2 = vmlaq_f32(
            vmulq_f32(vld1q_f32(out2.as_ptr().add(i)), mul[2]),
            value,
            add[2],
        );
        let updated3 = vmlaq_f32(
            vmulq_f32(vld1q_f32(out3.as_ptr().add(i)), mul[3]),
            value,
            add[3],
        );
        vst1q_f32(out0.as_mut_ptr().add(i), updated0);
        vst1q_f32(out1.as_mut_ptr().add(i), updated1);
        vst1q_f32(out2.as_mut_ptr().add(i), updated2);
        vst1q_f32(out3.as_mut_ptr().add(i), updated3);
        sums[0] = vmlaq_f32(sums[0], updated0, projection);
        sums[1] = vmlaq_f32(sums[1], updated1, projection);
        sums[2] = vmlaq_f32(sums[2], updated2, projection);
        sums[3] = vmlaq_f32(sums[3], updated3, projection);
        i += 4;
    }
    let mut result = [
        vaddvq_f32(sums[0]),
        vaddvq_f32(sums[1]),
        vaddvq_f32(sums[2]),
        vaddvq_f32(sums[3]),
    ];
    while i < x.len() {
        out0[i] = out0[i] * multiplier[0] + alpha[0] * x[i];
        out1[i] = out1[i] * multiplier[1] + alpha[1] * x[i];
        out2[i] = out2[i] * multiplier[2] + alpha[2] * x[i];
        out3[i] = out3[i] * multiplier[3] + alpha[3] * x[i];
        result[0] += out0[i] * dot[i];
        result[1] += out1[i] * dot[i];
        result[2] += out2[i] * dot[i];
        result[3] += out3[i] * dot[i];
        i += 1;
    }
    result
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn dot_q8_0_f32_neon(qdata: &[u8], x: &[f32], n: usize) -> f32 {
    use std::arch::aarch64::*;
    let n_blocks = n / 32;
    let mut sum_acc0 = vdupq_n_f32(0.0);
    let mut sum_acc1 = vdupq_n_f32(0.0);
    let mut b = 0usize;

    while b + 1 < n_blocks {
        let block = qdata.as_ptr().add(b * 34);
        let scale = f16_to_f32(u16::from_le_bytes([*block, *block.add(1)]));
        let block_next = qdata.as_ptr().add((b + 1) * 34);
        let scale_next = f16_to_f32(u16::from_le_bytes([*block_next, *block_next.add(1)]));
        let xptr = x.as_ptr().add(b * 32);

        sum_acc0 = vmlaq_f32(
            sum_acc0,
            dot_q8_0_block_f32_neon(block, xptr),
            vdupq_n_f32(scale),
        );
        sum_acc1 = vmlaq_f32(
            sum_acc1,
            dot_q8_0_block_f32_neon(block_next, xptr.add(32)),
            vdupq_n_f32(scale_next),
        );
        b += 2;
    }

    if b < n_blocks {
        let block = qdata.as_ptr().add(b * 34);
        let scale = f16_to_f32(u16::from_le_bytes([*block, *block.add(1)]));
        sum_acc0 = vmlaq_f32(
            sum_acc0,
            dot_q8_0_block_f32_neon(block, x.as_ptr().add(b * 32)),
            vdupq_n_f32(scale),
        );
    }

    vaddvq_f32(vaddq_f32(sum_acc0, sum_acc1))
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn dot_q8_0_block_f32_neon(
    block: *const u8,
    xp: *const f32,
) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    let q = block.add(2) as *const i8;
    let mut bacc = vdupq_n_f32(0.0);

    for i in 0..4_usize {
        let qi8 = vld1_s8(q.add(i * 8));
        let qi16 = vmovl_s8(qi8);
        let qlo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(qi16)));
        let qhi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(qi16)));
        bacc = vmlaq_f32(bacc, qlo, vld1q_f32(xp.add(i * 8)));
        bacc = vmlaq_f32(bacc, qhi, vld1q_f32(xp.add(i * 8 + 4)));
    }

    bacc
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn dot_q4_0_f32_neon(qdata: &[u8], x: &[f32], n: usize) -> f32 {
    // Q4_0 layout per block (18 bytes): [f16 scale | 16 nibble bytes]
    // ggml split layout: lo nibble of byte i → weight[i], hi nibble → weight[i+16].
    use std::arch::aarch64::*;
    let n_blocks = n / 32;
    let mut sum_acc0 = vdupq_n_f32(0.0);
    let mut sum_acc1 = vdupq_n_f32(0.0);
    let mask_low = vdupq_n_u8(0x0F);
    let eight = vdupq_n_u8(8);
    let mut b = 0usize;

    while b + 1 < n_blocks {
        let block = qdata.as_ptr().add(b * 18);
        let scale = f16_to_f32(u16::from_le_bytes([*block, *block.add(1)]));
        let block_next = qdata.as_ptr().add((b + 1) * 18);
        let scale_next = f16_to_f32(u16::from_le_bytes([*block_next, *block_next.add(1)]));
        let xptr = x.as_ptr().add(b * 32);

        sum_acc0 = vmlaq_f32(
            sum_acc0,
            dot_q4_0_block_f32_neon(block, xptr, mask_low, eight),
            vdupq_n_f32(scale),
        );
        sum_acc1 = vmlaq_f32(
            sum_acc1,
            dot_q4_0_block_f32_neon(block_next, xptr.add(32), mask_low, eight),
            vdupq_n_f32(scale_next),
        );
        b += 2;
    }

    if b < n_blocks {
        let block = qdata.as_ptr().add(b * 18);
        let scale = f16_to_f32(u16::from_le_bytes([*block, *block.add(1)]));
        sum_acc0 = vmlaq_f32(
            sum_acc0,
            dot_q4_0_block_f32_neon(block, x.as_ptr().add(b * 32), mask_low, eight),
            vdupq_n_f32(scale),
        );
    }

    vaddvq_f32(vaddq_f32(sum_acc0, sum_acc1))
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn dot_q4_0_block_f32_neon(
    block: *const u8,
    xp: *const f32,
    mask_low: std::arch::aarch64::uint8x16_t,
    eight: std::arch::aarch64::uint8x16_t,
) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;

    let nib = vld1q_u8(block.add(2));
    let lo_i8 = vreinterpretq_s8_u8(vsubq_u8(vandq_u8(nib, mask_low), eight));
    let hi_i8 = vreinterpretq_s8_u8(vsubq_u8(vshrq_n_u8(nib, 4), eight));

    let w0_8 = vmovl_s8(vget_low_s8(lo_i8));
    let w8_16 = vmovl_s8(vget_high_s8(lo_i8));
    let w16_24 = vmovl_s8(vget_low_s8(hi_i8));
    let w24_32 = vmovl_s8(vget_high_s8(hi_i8));

    let mut bacc = vdupq_n_f32(0.0);

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

    bacc
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

// ─── bf16 KV-cache kernels: NEON ──────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn load_bf16_widen_neon(ptr: *const u16) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    let half = vld1_u16(ptr);
    let widened = vmovl_u16(half);
    vreinterpretq_f32_u32(vshlq_n_u32(widened, 16))
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn dot_bf16_f32_neon(a: &[f32], b: &[u16]) -> f32 {
    use std::arch::aarch64::*;
    let n = a.len();
    let main = n / 4;
    let mut acc = vdupq_n_f32(0.0);
    let mut ap = a.as_ptr();
    let mut bp = b.as_ptr();
    for _ in 0..main {
        let bv = load_bf16_widen_neon(bp);
        acc = vmlaq_f32(acc, vld1q_f32(ap), bv);
        ap = ap.add(4);
        bp = bp.add(4);
    }
    let mut sum = vaddvq_f32(acc);
    for i in (main * 4)..n {
        sum += a[i] * bf16_to_f32(b[i]);
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn axpy_bf16_f32_neon(out: &mut [f32], alpha: f32, x: &[u16]) {
    use std::arch::aarch64::*;
    let n = out.len();
    let main = n / 4;
    let av = vdupq_n_f32(alpha);
    let mut op = out.as_mut_ptr();
    let mut xp = x.as_ptr();
    for _ in 0..main {
        let o = vld1q_f32(op);
        let xv = load_bf16_widen_neon(xp);
        vst1q_f32(op, vmlaq_f32(o, xv, av));
        op = op.add(4);
        xp = xp.add(4);
    }
    for i in (main * 4)..n {
        out[i] += alpha * bf16_to_f32(x[i]);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn scale_add_bf16_f32_neon(out: &mut [f32], scale: f32, add: &[u16]) {
    use std::arch::aarch64::*;
    let n = out.len();
    let main = n / 4;
    let sv = vdupq_n_f32(scale);
    let mut op = out.as_mut_ptr();
    let mut ap = add.as_ptr();
    for _ in 0..main {
        let o = vld1q_f32(op);
        let av = load_bf16_widen_neon(ap);
        vst1q_f32(op, vmlaq_f32(av, o, sv));
        op = op.add(4);
        ap = ap.add(4);
    }
    for i in (main * 4)..n {
        out[i] = out[i] * scale + bf16_to_f32(add[i]);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn dot_bf16x4_f32_neon(
    a0: &[f32],
    a1: &[f32],
    a2: &[f32],
    a3: &[f32],
    b: &[u16],
) -> [f32; 4] {
    use std::arch::aarch64::*;
    let n = b.len();
    let mut sums = [vdupq_n_f32(0.0); 4];
    let mut i = 0;
    while i + 4 <= n {
        let key = load_bf16_widen_neon(b.as_ptr().add(i));
        sums[0] = vmlaq_f32(sums[0], vld1q_f32(a0.as_ptr().add(i)), key);
        sums[1] = vmlaq_f32(sums[1], vld1q_f32(a1.as_ptr().add(i)), key);
        sums[2] = vmlaq_f32(sums[2], vld1q_f32(a2.as_ptr().add(i)), key);
        sums[3] = vmlaq_f32(sums[3], vld1q_f32(a3.as_ptr().add(i)), key);
        i += 4;
    }
    let mut out = [
        vaddvq_f32(sums[0]),
        vaddvq_f32(sums[1]),
        vaddvq_f32(sums[2]),
        vaddvq_f32(sums[3]),
    ];
    while i < n {
        let key = bf16_to_f32(b[i]);
        out[0] += a0[i] * key;
        out[1] += a1[i] * key;
        out[2] += a2[i] * key;
        out[3] += a3[i] * key;
        i += 1;
    }
    out
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn affine_add_bf16x4_f32_neon(
    out0: &mut [f32],
    out1: &mut [f32],
    out2: &mut [f32],
    out3: &mut [f32],
    multiplier: [f32; 4],
    alpha: [f32; 4],
    x: &[u16],
) {
    use std::arch::aarch64::*;
    let mul = [
        vdupq_n_f32(multiplier[0]),
        vdupq_n_f32(multiplier[1]),
        vdupq_n_f32(multiplier[2]),
        vdupq_n_f32(multiplier[3]),
    ];
    let add = [
        vdupq_n_f32(alpha[0]),
        vdupq_n_f32(alpha[1]),
        vdupq_n_f32(alpha[2]),
        vdupq_n_f32(alpha[3]),
    ];
    let mut i = 0;
    while i + 4 <= x.len() {
        let value = load_bf16_widen_neon(x.as_ptr().add(i));
        vst1q_f32(
            out0.as_mut_ptr().add(i),
            vmlaq_f32(
                vmulq_f32(vld1q_f32(out0.as_ptr().add(i)), mul[0]),
                value,
                add[0],
            ),
        );
        vst1q_f32(
            out1.as_mut_ptr().add(i),
            vmlaq_f32(
                vmulq_f32(vld1q_f32(out1.as_ptr().add(i)), mul[1]),
                value,
                add[1],
            ),
        );
        vst1q_f32(
            out2.as_mut_ptr().add(i),
            vmlaq_f32(
                vmulq_f32(vld1q_f32(out2.as_ptr().add(i)), mul[2]),
                value,
                add[2],
            ),
        );
        vst1q_f32(
            out3.as_mut_ptr().add(i),
            vmlaq_f32(
                vmulq_f32(vld1q_f32(out3.as_ptr().add(i)), mul[3]),
                value,
                add[3],
            ),
        );
        i += 4;
    }
    while i < x.len() {
        let value = bf16_to_f32(x[i]);
        out0[i] = out0[i] * multiplier[0] + alpha[0] * value;
        out1[i] = out1[i] * multiplier[1] + alpha[1] * value;
        out2[i] = out2[i] * multiplier[2] + alpha[2] * value;
        out3[i] = out3[i] * multiplier[3] + alpha[3] * value;
        i += 1;
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
unsafe fn dot_f32x4_avx2(a0: &[f32], a1: &[f32], a2: &[f32], a3: &[f32], b: &[f32]) -> [f32; 4] {
    use std::arch::x86_64::*;
    let mut sums = [_mm256_setzero_ps(); 4];
    let mut i = 0;
    while i + 8 <= b.len() {
        let key = _mm256_loadu_ps(b.as_ptr().add(i));
        sums[0] = _mm256_fmadd_ps(_mm256_loadu_ps(a0.as_ptr().add(i)), key, sums[0]);
        sums[1] = _mm256_fmadd_ps(_mm256_loadu_ps(a1.as_ptr().add(i)), key, sums[1]);
        sums[2] = _mm256_fmadd_ps(_mm256_loadu_ps(a2.as_ptr().add(i)), key, sums[2]);
        sums[3] = _mm256_fmadd_ps(_mm256_loadu_ps(a3.as_ptr().add(i)), key, sums[3]);
        i += 8;
    }
    let mut out = [
        hsum_avx(sums[0]),
        hsum_avx(sums[1]),
        hsum_avx(sums[2]),
        hsum_avx(sums[3]),
    ];
    while i < b.len() {
        let key = b[i];
        out[0] += a0[i] * key;
        out[1] += a1[i] * key;
        out[2] += a2[i] * key;
        out[3] += a3[i] * key;
        i += 1;
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn affine_add_f32x4_avx2(
    out0: &mut [f32],
    out1: &mut [f32],
    out2: &mut [f32],
    out3: &mut [f32],
    multiplier: [f32; 4],
    alpha: [f32; 4],
    x: &[f32],
) {
    use std::arch::x86_64::*;
    let mul = [
        _mm256_set1_ps(multiplier[0]),
        _mm256_set1_ps(multiplier[1]),
        _mm256_set1_ps(multiplier[2]),
        _mm256_set1_ps(multiplier[3]),
    ];
    let add = [
        _mm256_set1_ps(alpha[0]),
        _mm256_set1_ps(alpha[1]),
        _mm256_set1_ps(alpha[2]),
        _mm256_set1_ps(alpha[3]),
    ];
    let mut i = 0;
    while i + 8 <= x.len() {
        let value = _mm256_loadu_ps(x.as_ptr().add(i));
        let update = |out: &mut [f32], m, a| {
            let old = _mm256_loadu_ps(out.as_ptr().add(i));
            _mm256_storeu_ps(
                out.as_mut_ptr().add(i),
                _mm256_fmadd_ps(value, a, _mm256_mul_ps(old, m)),
            );
        };
        update(out0, mul[0], add[0]);
        update(out1, mul[1], add[1]);
        update(out2, mul[2], add[2]);
        update(out3, mul[3], add[3]);
        i += 8;
    }
    while i < x.len() {
        out0[i] = out0[i] * multiplier[0] + alpha[0] * x[i];
        out1[i] = out1[i] * multiplier[1] + alpha[1] * x[i];
        out2[i] = out2[i] * multiplier[2] + alpha[2] * x[i];
        out3[i] = out3[i] * multiplier[3] + alpha[3] * x[i];
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn affine_add_dot_f32x4_avx2(
    out0: &mut [f32],
    out1: &mut [f32],
    out2: &mut [f32],
    out3: &mut [f32],
    multiplier: [f32; 4],
    alpha: [f32; 4],
    x: &[f32],
    dot: &[f32],
) -> [f32; 4] {
    use std::arch::x86_64::*;
    let mul = [
        _mm256_set1_ps(multiplier[0]),
        _mm256_set1_ps(multiplier[1]),
        _mm256_set1_ps(multiplier[2]),
        _mm256_set1_ps(multiplier[3]),
    ];
    let add = [
        _mm256_set1_ps(alpha[0]),
        _mm256_set1_ps(alpha[1]),
        _mm256_set1_ps(alpha[2]),
        _mm256_set1_ps(alpha[3]),
    ];
    let mut sums = [_mm256_setzero_ps(); 4];
    let mut i = 0usize;
    while i + 8 <= x.len() {
        let value = _mm256_loadu_ps(x.as_ptr().add(i));
        let projection = _mm256_loadu_ps(dot.as_ptr().add(i));
        let updated0 = _mm256_fmadd_ps(
            value,
            add[0],
            _mm256_mul_ps(_mm256_loadu_ps(out0.as_ptr().add(i)), mul[0]),
        );
        let updated1 = _mm256_fmadd_ps(
            value,
            add[1],
            _mm256_mul_ps(_mm256_loadu_ps(out1.as_ptr().add(i)), mul[1]),
        );
        let updated2 = _mm256_fmadd_ps(
            value,
            add[2],
            _mm256_mul_ps(_mm256_loadu_ps(out2.as_ptr().add(i)), mul[2]),
        );
        let updated3 = _mm256_fmadd_ps(
            value,
            add[3],
            _mm256_mul_ps(_mm256_loadu_ps(out3.as_ptr().add(i)), mul[3]),
        );
        _mm256_storeu_ps(out0.as_mut_ptr().add(i), updated0);
        _mm256_storeu_ps(out1.as_mut_ptr().add(i), updated1);
        _mm256_storeu_ps(out2.as_mut_ptr().add(i), updated2);
        _mm256_storeu_ps(out3.as_mut_ptr().add(i), updated3);
        sums[0] = _mm256_fmadd_ps(updated0, projection, sums[0]);
        sums[1] = _mm256_fmadd_ps(updated1, projection, sums[1]);
        sums[2] = _mm256_fmadd_ps(updated2, projection, sums[2]);
        sums[3] = _mm256_fmadd_ps(updated3, projection, sums[3]);
        i += 8;
    }
    let mut result = [
        hsum_avx(sums[0]),
        hsum_avx(sums[1]),
        hsum_avx(sums[2]),
        hsum_avx(sums[3]),
    ];
    while i < x.len() {
        out0[i] = out0[i] * multiplier[0] + alpha[0] * x[i];
        out1[i] = out1[i] * multiplier[1] + alpha[1] * x[i];
        out2[i] = out2[i] * multiplier[2] + alpha[2] * x[i];
        out3[i] = out3[i] * multiplier[3] + alpha[3] * x[i];
        result[0] += out0[i] * dot[i];
        result[1] += out1[i] * dot[i];
        result[2] += out2[i] * dot[i];
        result[3] += out3[i] * dot[i];
        i += 1;
    }
    result
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

/// Horizontal sum of the eight int32 lanes of an AVX2 register.
#[cfg(all(target_arch = "x86_64", not(target_family = "wasm")))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hsum_i32_avx2(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::*;
    let hi = _mm256_extracti128_si256(v, 1);
    let s = _mm_add_epi32(_mm256_castsi256_si128(v), hi);
    let s = _mm_add_epi32(s, _mm_unpackhi_epi64(s, s));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x55));
    _mm_cvtsi128_si32(s)
}

/// Q4_K · Q8_K integer dot product on AVX2: unsigned 4-bit weights are
/// multiplied against the int8 activation via `maddubs`, per-32 scales fold in
/// through `madd`, and the `dmin` term uses the pre-computed group sums —
/// mirroring `dot_q4_k_q8k_neon`.
#[cfg(all(target_arch = "x86_64", not(target_family = "wasm")))]
#[target_feature(enable = "avx2")]
unsafe fn dot_q4_k_q8k_avx2(qdata: &[u8], xq: XQuant, n: usize) -> f32 {
    use std::arch::x86_64::*;
    let nb = n / 256;
    let lowmask = _mm256_set1_epi8(0x0F);
    let mut acc = 0.0f32;

    for b in 0..nb {
        let block = qdata.as_ptr().add(b * 144);
        let d = f16_to_f32(u16::from_le_bytes([*block, *block.add(1)]));
        let dmin = f16_to_f32(u16::from_le_bytes([*block.add(2), *block.add(3)]));
        let scales: &[u8; 12] = std::slice::from_raw_parts(block.add(4), 12)
            .try_into()
            .expect("q4_k scales size");
        let q = block.add(16);
        let dx = *xq.d.add(b);
        let xqb = xq.qs.add(b * 256);
        let bsb = xq.bsums.add(b * 8);

        let mut isum_v = _mm256_setzero_si256();
        let mut imin = 0i32;
        for c in 0..4usize {
            let is = c * 2;
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);

            let raw = _mm256_loadu_si256(q.add(c * 32) as *const __m256i);
            let lo = _mm256_and_si256(raw, lowmask);
            let hi = _mm256_and_si256(_mm256_srli_epi16(raw, 4), lowmask);
            let x_lo = _mm256_loadu_si256(xqb.add(c * 64) as *const __m256i);
            let x_hi = _mm256_loadu_si256(xqb.add(c * 64 + 32) as *const __m256i);

            let p_lo = _mm256_maddubs_epi16(lo, x_lo);
            let p_hi = _mm256_maddubs_epi16(hi, x_hi);
            isum_v = _mm256_add_epi32(
                isum_v,
                _mm256_madd_epi16(p_lo, _mm256_set1_epi16(sc1 as i16)),
            );
            isum_v = _mm256_add_epi32(
                isum_v,
                _mm256_madd_epi16(p_hi, _mm256_set1_epi16(sc2 as i16)),
            );
            imin += m1 as i32 * (*bsb.add(is) as i32) + m2 as i32 * (*bsb.add(is + 1) as i32);
        }
        let isum = hsum_i32_avx2(isum_v);
        acc += dx * (d * isum as f32 - dmin * imin as f32);
    }
    acc
}

/// Q5_K · Q8_K integer dot product on AVX2: like the Q4_K kernel with the
/// fifth weight bit OR'd in from the `qh` plane before `maddubs`.
#[cfg(all(target_arch = "x86_64", not(target_family = "wasm")))]
#[target_feature(enable = "avx2")]
unsafe fn dot_q5_k_q8k_avx2(qdata: &[u8], xq: XQuant, n: usize) -> f32 {
    use std::arch::x86_64::*;
    let nb = n / 256;
    let lowmask = _mm256_set1_epi8(0x0F);
    let hbit = _mm256_set1_epi8(0x10);
    let mut acc = 0.0f32;

    for b in 0..nb {
        let block = qdata.as_ptr().add(b * 176);
        let d = f16_to_f32(u16::from_le_bytes([*block, *block.add(1)]));
        let dmin = f16_to_f32(u16::from_le_bytes([*block.add(2), *block.add(3)]));
        let scales: &[u8; 12] = std::slice::from_raw_parts(block.add(4), 12)
            .try_into()
            .expect("q5_k scales size");
        let qh = _mm256_loadu_si256(block.add(16) as *const __m256i);
        let q = block.add(48);
        let dx = *xq.d.add(b);
        let xqb = xq.qs.add(b * 256);
        let bsb = xq.bsums.add(b * 8);

        let mut isum_v = _mm256_setzero_si256();
        let mut imin = 0i32;
        for c in 0..4usize {
            let is = c * 2;
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let u1 = _mm256_set1_epi8((1u8 << (2 * c)) as i8);
            let u2 = _mm256_set1_epi8((2u8 << (2 * c)) as i8);

            let raw = _mm256_loadu_si256(q.add(c * 32) as *const __m256i);
            let hi1 = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(qh, u1), u1), hbit);
            let hi2 = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(qh, u2), u2), hbit);
            let lo = _mm256_or_si256(_mm256_and_si256(raw, lowmask), hi1);
            let hi = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(raw, 4), lowmask), hi2);
            let x_lo = _mm256_loadu_si256(xqb.add(c * 64) as *const __m256i);
            let x_hi = _mm256_loadu_si256(xqb.add(c * 64 + 32) as *const __m256i);

            let p_lo = _mm256_maddubs_epi16(lo, x_lo);
            let p_hi = _mm256_maddubs_epi16(hi, x_hi);
            isum_v = _mm256_add_epi32(
                isum_v,
                _mm256_madd_epi16(p_lo, _mm256_set1_epi16(sc1 as i16)),
            );
            isum_v = _mm256_add_epi32(
                isum_v,
                _mm256_madd_epi16(p_hi, _mm256_set1_epi16(sc2 as i16)),
            );
            imin += m1 as i32 * (*bsb.add(is) as i32) + m2 as i32 * (*bsb.add(is + 1) as i32);
        }
        let isum = hsum_i32_avx2(isum_v);
        acc += dx * (d * isum as f32 - dmin * imin as f32);
    }
    acc
}

/// Q6_K · Q8_K integer dot product on AVX2. The `-32` bias is folded into the
/// signed weights, then `abs`/`sign` re-expresses the signed×signed product in
/// `maddubs` form; per-16 int8 scales fold in through `madd`.
#[cfg(all(target_arch = "x86_64", not(target_family = "wasm")))]
#[target_feature(enable = "avx2")]
unsafe fn dot_q6_k_q8k_avx2(qdata: &[u8], xq: XQuant, n: usize) -> f32 {
    use std::arch::x86_64::*;
    let nb = n / 256;
    let lowmask = _mm256_set1_epi8(0x0F);
    let m3 = _mm256_set1_epi8(0x03);
    let sub32 = _mm256_set1_epi8(32);
    let mut acc = 0.0f32;

    for b in 0..nb {
        let block = qdata.as_ptr().add(b * 210);
        let d = f16_to_f32(u16::from_le_bytes([*block.add(208), *block.add(209)]));
        let dx = *xq.d.add(b);
        let xqb = xq.qs.add(b * 256);
        let sc_base = block.add(192) as *const i8;

        let mut isum_v = _mm256_setzero_si256();
        for g in 0..2usize {
            let ql0 = _mm256_loadu_si256(block.add(g * 64) as *const __m256i);
            let ql1 = _mm256_loadu_si256(block.add(g * 64 + 32) as *const __m256i);
            let qhv = _mm256_loadu_si256(block.add(128 + g * 32) as *const __m256i);
            let scs = sc_base.add(g * 8);
            let x_base = g * 128;

            // Reassemble the four 32-element sub-vectors exactly like the NEON
            // kernel: x offsets base+0/32/64/96, high bits from 2-bit planes.
            let q0 = _mm256_or_si256(
                _mm256_and_si256(ql0, lowmask),
                _mm256_slli_epi16(_mm256_and_si256(qhv, m3), 4),
            );
            let q1 = _mm256_or_si256(
                _mm256_and_si256(ql1, lowmask),
                _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 2), m3), 4),
            );
            let q2 = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(ql0, 4), lowmask),
                _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 4), m3), 4),
            );
            let q3 = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(ql1, 4), lowmask),
                _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 6), m3), 4),
            );

            for (idx, qv) in [q0, q1, q2, q3].into_iter().enumerate() {
                let qs = _mm256_sub_epi8(qv, sub32);
                let xv = _mm256_loadu_si256(xqb.add(x_base + idx * 32) as *const __m256i);
                let p = _mm256_maddubs_epi16(_mm256_abs_epi8(qs), _mm256_sign_epi8(xv, qs));
                let sc_a = *scs.add(idx * 2) as i16;
                let sc_b = *scs.add(idx * 2 + 1) as i16;
                let sc_v = _mm256_set_m128i(_mm_set1_epi16(sc_b), _mm_set1_epi16(sc_a));
                isum_v = _mm256_add_epi32(isum_v, _mm256_madd_epi16(p, sc_v));
            }
        }
        acc += dx * d * hsum_i32_avx2(isum_v) as f32;
    }
    acc
}

/// Q4_K · Q8_K for `KQUANT_BATCH_TOKENS` activations against one weight row.
///
/// Arithmetic per lane is identical to `dot_q4_k_q8k_avx2`, in the same order:
/// only the token-independent weight work — the two f16 scale conversions, the
/// eight `get_scale_min_k4` unpacks, the scale broadcasts and the nibble split
/// — moves out of the token loop, so each lane stays bit-identical to the
/// single-token kernel while that setup is paid once instead of four times.
#[cfg(all(target_arch = "x86_64", not(target_family = "wasm")))]
#[target_feature(enable = "avx2")]
unsafe fn dot_q4_k_q8k_avx2_x4(
    qdata: &[u8],
    xqs: &[XQuant; KQUANT_BATCH_TOKENS],
    n: usize,
) -> [f32; KQUANT_BATCH_TOKENS] {
    use std::arch::x86_64::*;
    let nb = n / 256;
    let lowmask = _mm256_set1_epi8(0x0F);
    let mut acc = [0.0f32; KQUANT_BATCH_TOKENS];

    for b in 0..nb {
        let block = qdata.as_ptr().add(b * 144);
        let d = f16_to_f32(u16::from_le_bytes([*block, *block.add(1)]));
        let dmin = f16_to_f32(u16::from_le_bytes([*block.add(2), *block.add(3)]));
        let scales: &[u8; 12] = std::slice::from_raw_parts(block.add(4), 12)
            .try_into()
            .expect("q4_k scales size");
        let q = block.add(16);

        let mut isum_v = [_mm256_setzero_si256(); KQUANT_BATCH_TOKENS];
        let mut imin = [0i32; KQUANT_BATCH_TOKENS];
        for c in 0..4usize {
            let is = c * 2;
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let sc1_v = _mm256_set1_epi16(sc1 as i16);
            let sc2_v = _mm256_set1_epi16(sc2 as i16);

            let raw = _mm256_loadu_si256(q.add(c * 32) as *const __m256i);
            let lo = _mm256_and_si256(raw, lowmask);
            let hi = _mm256_and_si256(_mm256_srli_epi16(raw, 4), lowmask);

            for t in 0..KQUANT_BATCH_TOKENS {
                let xqb = xqs[t].qs.add(b * 256);
                let bsb = xqs[t].bsums.add(b * 8);
                let x_lo = _mm256_loadu_si256(xqb.add(c * 64) as *const __m256i);
                let x_hi = _mm256_loadu_si256(xqb.add(c * 64 + 32) as *const __m256i);

                let p_lo = _mm256_maddubs_epi16(lo, x_lo);
                let p_hi = _mm256_maddubs_epi16(hi, x_hi);
                isum_v[t] = _mm256_add_epi32(isum_v[t], _mm256_madd_epi16(p_lo, sc1_v));
                isum_v[t] = _mm256_add_epi32(isum_v[t], _mm256_madd_epi16(p_hi, sc2_v));
                imin[t] +=
                    m1 as i32 * (*bsb.add(is) as i32) + m2 as i32 * (*bsb.add(is + 1) as i32);
            }
        }
        for t in 0..KQUANT_BATCH_TOKENS {
            let dx = *xqs[t].d.add(b);
            let isum = hsum_i32_avx2(isum_v[t]);
            acc[t] += dx * (d * isum as f32 - dmin * imin[t] as f32);
        }
    }
    acc
}

/// Q5_K · Q8_K for `KQUANT_BATCH_TOKENS` activations against one weight row.
/// Token-blocked mirror of `dot_q5_k_q8k_avx2`; expanding the fifth weight bit
/// from the `qh` plane is weight-side work and hoists out with the rest.
#[cfg(all(target_arch = "x86_64", not(target_family = "wasm")))]
#[target_feature(enable = "avx2")]
unsafe fn dot_q5_k_q8k_avx2_x4(
    qdata: &[u8],
    xqs: &[XQuant; KQUANT_BATCH_TOKENS],
    n: usize,
) -> [f32; KQUANT_BATCH_TOKENS] {
    use std::arch::x86_64::*;
    let nb = n / 256;
    let lowmask = _mm256_set1_epi8(0x0F);
    let hbit = _mm256_set1_epi8(0x10);
    let mut acc = [0.0f32; KQUANT_BATCH_TOKENS];

    for b in 0..nb {
        let block = qdata.as_ptr().add(b * 176);
        let d = f16_to_f32(u16::from_le_bytes([*block, *block.add(1)]));
        let dmin = f16_to_f32(u16::from_le_bytes([*block.add(2), *block.add(3)]));
        let scales: &[u8; 12] = std::slice::from_raw_parts(block.add(4), 12)
            .try_into()
            .expect("q5_k scales size");
        let qh = _mm256_loadu_si256(block.add(16) as *const __m256i);
        let q = block.add(48);

        let mut isum_v = [_mm256_setzero_si256(); KQUANT_BATCH_TOKENS];
        let mut imin = [0i32; KQUANT_BATCH_TOKENS];
        for c in 0..4usize {
            let is = c * 2;
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let sc1_v = _mm256_set1_epi16(sc1 as i16);
            let sc2_v = _mm256_set1_epi16(sc2 as i16);
            let u1 = _mm256_set1_epi8((1u8 << (2 * c)) as i8);
            let u2 = _mm256_set1_epi8((2u8 << (2 * c)) as i8);

            let raw = _mm256_loadu_si256(q.add(c * 32) as *const __m256i);
            let hi1 = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(qh, u1), u1), hbit);
            let hi2 = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(qh, u2), u2), hbit);
            let lo = _mm256_or_si256(_mm256_and_si256(raw, lowmask), hi1);
            let hi = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(raw, 4), lowmask), hi2);

            for t in 0..KQUANT_BATCH_TOKENS {
                let xqb = xqs[t].qs.add(b * 256);
                let bsb = xqs[t].bsums.add(b * 8);
                let x_lo = _mm256_loadu_si256(xqb.add(c * 64) as *const __m256i);
                let x_hi = _mm256_loadu_si256(xqb.add(c * 64 + 32) as *const __m256i);

                let p_lo = _mm256_maddubs_epi16(lo, x_lo);
                let p_hi = _mm256_maddubs_epi16(hi, x_hi);
                isum_v[t] = _mm256_add_epi32(isum_v[t], _mm256_madd_epi16(p_lo, sc1_v));
                isum_v[t] = _mm256_add_epi32(isum_v[t], _mm256_madd_epi16(p_hi, sc2_v));
                imin[t] +=
                    m1 as i32 * (*bsb.add(is) as i32) + m2 as i32 * (*bsb.add(is + 1) as i32);
            }
        }
        for t in 0..KQUANT_BATCH_TOKENS {
            let dx = *xqs[t].d.add(b);
            let isum = hsum_i32_avx2(isum_v[t]);
            acc[t] += dx * (d * isum as f32 - dmin * imin[t] as f32);
        }
    }
    acc
}

/// Q6_K · Q8_K for `KQUANT_BATCH_TOKENS` activations against one weight row.
/// Token-blocked mirror of `dot_q6_k_q8k_avx2`. Q6_K gains the most from the
/// blocking: reassembling each 6-bit sub-vector costs four vector ops plus the
/// `-32` bias fold and the `abs` that puts the signed product into `maddubs`
/// form, and all of it is token independent. The four sub-vectors are rebuilt
/// per index instead of kept in an array so the four per-lane accumulators fit
/// alongside them without spilling.
#[cfg(all(target_arch = "x86_64", not(target_family = "wasm")))]
#[target_feature(enable = "avx2")]
unsafe fn dot_q6_k_q8k_avx2_x4(
    qdata: &[u8],
    xqs: &[XQuant; KQUANT_BATCH_TOKENS],
    n: usize,
) -> [f32; KQUANT_BATCH_TOKENS] {
    use std::arch::x86_64::*;
    let nb = n / 256;
    let lowmask = _mm256_set1_epi8(0x0F);
    let m3 = _mm256_set1_epi8(0x03);
    let sub32 = _mm256_set1_epi8(32);
    let mut acc = [0.0f32; KQUANT_BATCH_TOKENS];

    for b in 0..nb {
        let block = qdata.as_ptr().add(b * 210);
        let d = f16_to_f32(u16::from_le_bytes([*block.add(208), *block.add(209)]));
        let sc_base = block.add(192) as *const i8;

        let mut isum_v = [_mm256_setzero_si256(); KQUANT_BATCH_TOKENS];
        for g in 0..2usize {
            let ql0 = _mm256_loadu_si256(block.add(g * 64) as *const __m256i);
            let ql1 = _mm256_loadu_si256(block.add(g * 64 + 32) as *const __m256i);
            let qhv = _mm256_loadu_si256(block.add(128 + g * 32) as *const __m256i);
            let scs = sc_base.add(g * 8);
            let x_base = g * 128;

            for idx in 0..4usize {
                // Same reassembly as the single-token kernel: x offsets
                // base+0/32/64/96, high bits from the 2-bit planes.
                let qv = match idx {
                    0 => _mm256_or_si256(
                        _mm256_and_si256(ql0, lowmask),
                        _mm256_slli_epi16(_mm256_and_si256(qhv, m3), 4),
                    ),
                    1 => _mm256_or_si256(
                        _mm256_and_si256(ql1, lowmask),
                        _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 2), m3), 4),
                    ),
                    2 => _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16(ql0, 4), lowmask),
                        _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 4), m3), 4),
                    ),
                    _ => _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16(ql1, 4), lowmask),
                        _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 6), m3), 4),
                    ),
                };
                let qs = _mm256_sub_epi8(qv, sub32);
                let absq = _mm256_abs_epi8(qs);
                let sc_a = *scs.add(idx * 2) as i16;
                let sc_b = *scs.add(idx * 2 + 1) as i16;
                let sc_v = _mm256_set_m128i(_mm_set1_epi16(sc_b), _mm_set1_epi16(sc_a));

                for t in 0..KQUANT_BATCH_TOKENS {
                    let xqb = xqs[t].qs.add(b * 256);
                    let xv = _mm256_loadu_si256(xqb.add(x_base + idx * 32) as *const __m256i);
                    let p = _mm256_maddubs_epi16(absq, _mm256_sign_epi8(xv, qs));
                    isum_v[t] = _mm256_add_epi32(isum_v[t], _mm256_madd_epi16(p, sc_v));
                }
            }
        }
        for t in 0..KQUANT_BATCH_TOKENS {
            let dx = *xqs[t].d.add(b);
            acc[t] += dx * d * hsum_i32_avx2(isum_v[t]) as f32;
        }
    }
    acc
}

/// Q4_0 · Q8_K integer dot product on AVX2. Weights stay unsigned so `maddubs`
/// takes them directly, the `-8` bias folds into one term built from the
/// per-32 activation sums, and the per-block scale accumulates into an f32 lane
/// vector that is reduced once per row. Reached only behind `has_avx2_fma()`.
#[cfg(all(target_arch = "x86_64", not(target_family = "wasm")))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q4_0_q8k_avx2(qdata: &[u8], xq: XQuant, n: usize) -> f32 {
    use std::arch::x86_64::*;
    let n_super = n / 256;
    let lowmask = _mm_set1_epi8(0x0F);
    let ones = _mm256_set1_epi16(1);
    // One f32 lane accumulator for the whole row, reduced once at the end. The
    // previous version ran a full `hsum_i32_avx2` plus a scalar f32 multiply-add
    // for every 32-weight block, i.e. eight vector-to-scalar round trips per
    // super-block, and those latencies dominated a kernel that only reads 18
    // bytes per block.
    let mut acc = _mm256_setzero_ps();
    // The `-8` weight bias folds out of the inner loop: with `q` left unsigned,
    // sum((q-8)*x) == sum(q*x) - 8*sum(x), and sum(x) per 32-element group is
    // already tabulated in `xq.bsums` (Q4_0's block size is exactly that group
    // size). That removes the `abs`/`sign` pair the `maddubs` form needed.
    let mut bias = 0.0f32;

    for sb in 0..n_super {
        let dx = *xq.d.add(sb);
        let xqb = xq.qs.add(sb * 256);
        let bsums = xq.bsums.add(sb * 8);
        let base = qdata.as_ptr().add(sb * 8 * 18);
        for blk in 0..8usize {
            let block = base.add(blk * 18);
            let scale = dx * f16_to_f32(u16::from_le_bytes([*block, *block.add(1)]));

            // lo nibble of byte i → weight[i], hi nibble → weight[i+16].
            let nib = _mm_loadu_si128(block.add(2) as *const __m128i);
            let lo = _mm_and_si128(nib, lowmask);
            let hi = _mm_and_si128(_mm_srli_epi16(nib, 4), lowmask);
            // Unsigned 0..15 feeds `maddubs` directly; 15*128*2 stays inside i16.
            let q = _mm256_set_m128i(hi, lo);

            let xv = _mm256_loadu_si256(xqb.add(blk * 32) as *const __m256i);
            let p = _mm256_madd_epi16(_mm256_maddubs_epi16(q, xv), ones);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(scale), _mm256_cvtepi32_ps(p), acc);
            bias += scale * *bsums.add(blk) as f32;
        }
    }
    hsum_avx(acc) - 8.0 * bias
}

/// Q8_0 · Q8_K integer dot product on AVX2 via the `abs`/`sign` `maddubs` form,
/// accumulating scaled block sums in an f32 lane vector. Reached only behind
/// `has_avx2_fma()`.
#[cfg(all(target_arch = "x86_64", not(target_family = "wasm")))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q8_0_q8k_avx2(qdata: &[u8], xq: XQuant, n: usize) -> f32 {
    use std::arch::x86_64::*;
    let n_super = n / 256;
    let ones = _mm256_set1_epi16(1);
    // As in the Q4_0 kernel: keep the per-block scale in an f32 lane
    // accumulator instead of horizontally reducing to a scalar eight times per
    // super-block. Q8_0 weights are already signed with no offset, so the
    // `abs`/`sign` pair stays.
    let mut acc = _mm256_setzero_ps();

    for sb in 0..n_super {
        let dx = *xq.d.add(sb);
        let xqb = xq.qs.add(sb * 256);
        let base = qdata.as_ptr().add(sb * 8 * 34);
        for blk in 0..8usize {
            let block = base.add(blk * 34);
            let scale = dx * f16_to_f32(u16::from_le_bytes([*block, *block.add(1)]));

            let w = _mm256_loadu_si256(block.add(2) as *const __m256i);
            let xv = _mm256_loadu_si256(xqb.add(blk * 32) as *const __m256i);
            let p = _mm256_maddubs_epi16(_mm256_abs_epi8(w), _mm256_sign_epi8(xv, w));
            let s = _mm256_madd_epi16(p, ones);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(scale), _mm256_cvtepi32_ps(s), acc);
        }
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

// ─── bf16 KV-cache kernels: AVX2 ──────────────────────────────────────────────

/// Loads 8 packed bf16 values and widens them to an `__m256` of f32 by
/// zero-extending to u32 and shifting into the high 16 bits — bf16's only
/// difference from f32 is a truncated mantissa, so this is exact.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn load_bf16_widen_avx2(ptr: *const u16) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let half = _mm_loadu_si128(ptr as *const __m128i);
    let widened = _mm256_cvtepu16_epi32(half);
    let shifted = _mm256_slli_epi32(widened, 16);
    _mm256_castsi256_ps(shifted)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_bf16_f32_avx2(a: &[f32], b: &[u16]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let main = n / 8;
    let mut acc = _mm256_setzero_ps();
    let mut ap = a.as_ptr();
    let mut bp = b.as_ptr();
    for _ in 0..main {
        let bv = load_bf16_widen_avx2(bp);
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(ap), bv, acc);
        ap = ap.add(8);
        bp = bp.add(8);
    }
    let mut sum = hsum_avx(acc);
    for i in (main * 8)..n {
        sum += a[i] * bf16_to_f32(b[i]);
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn axpy_bf16_f32_avx2(out: &mut [f32], alpha: f32, x: &[u16]) {
    use std::arch::x86_64::*;
    let n = out.len();
    let main = n / 8;
    let av = _mm256_set1_ps(alpha);
    let mut op = out.as_mut_ptr();
    let mut xp = x.as_ptr();
    for _ in 0..main {
        let o = _mm256_loadu_ps(op);
        let xv = load_bf16_widen_avx2(xp);
        let y = _mm256_fmadd_ps(av, xv, o);
        _mm256_storeu_ps(op, y);
        op = op.add(8);
        xp = xp.add(8);
    }
    for i in (main * 8)..n {
        out[i] += alpha * bf16_to_f32(x[i]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn scale_add_bf16_f32_avx2(out: &mut [f32], scale: f32, add: &[u16]) {
    use std::arch::x86_64::*;
    let n = out.len();
    let main = n / 8;
    let sv = _mm256_set1_ps(scale);
    let mut op = out.as_mut_ptr();
    let mut ap = add.as_ptr();
    for _ in 0..main {
        let o = _mm256_loadu_ps(op);
        let a = load_bf16_widen_avx2(ap);
        let y = _mm256_fmadd_ps(o, sv, a);
        _mm256_storeu_ps(op, y);
        op = op.add(8);
        ap = ap.add(8);
    }
    for i in (main * 8)..n {
        out[i] = out[i] * scale + bf16_to_f32(add[i]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_bf16x4_f32_avx2(
    a0: &[f32],
    a1: &[f32],
    a2: &[f32],
    a3: &[f32],
    b: &[u16],
) -> [f32; 4] {
    use std::arch::x86_64::*;
    let mut sums = [_mm256_setzero_ps(); 4];
    let mut i = 0;
    while i + 8 <= b.len() {
        let key = load_bf16_widen_avx2(b.as_ptr().add(i));
        sums[0] = _mm256_fmadd_ps(_mm256_loadu_ps(a0.as_ptr().add(i)), key, sums[0]);
        sums[1] = _mm256_fmadd_ps(_mm256_loadu_ps(a1.as_ptr().add(i)), key, sums[1]);
        sums[2] = _mm256_fmadd_ps(_mm256_loadu_ps(a2.as_ptr().add(i)), key, sums[2]);
        sums[3] = _mm256_fmadd_ps(_mm256_loadu_ps(a3.as_ptr().add(i)), key, sums[3]);
        i += 8;
    }
    let mut out = [
        hsum_avx(sums[0]),
        hsum_avx(sums[1]),
        hsum_avx(sums[2]),
        hsum_avx(sums[3]),
    ];
    while i < b.len() {
        let key = bf16_to_f32(b[i]);
        out[0] += a0[i] * key;
        out[1] += a1[i] * key;
        out[2] += a2[i] * key;
        out[3] += a3[i] * key;
        i += 1;
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn affine_add_bf16x4_f32_avx2(
    out0: &mut [f32],
    out1: &mut [f32],
    out2: &mut [f32],
    out3: &mut [f32],
    multiplier: [f32; 4],
    alpha: [f32; 4],
    x: &[u16],
) {
    use std::arch::x86_64::*;
    let mul = [
        _mm256_set1_ps(multiplier[0]),
        _mm256_set1_ps(multiplier[1]),
        _mm256_set1_ps(multiplier[2]),
        _mm256_set1_ps(multiplier[3]),
    ];
    let add = [
        _mm256_set1_ps(alpha[0]),
        _mm256_set1_ps(alpha[1]),
        _mm256_set1_ps(alpha[2]),
        _mm256_set1_ps(alpha[3]),
    ];
    let mut i = 0;
    while i + 8 <= x.len() {
        let value = load_bf16_widen_avx2(x.as_ptr().add(i));
        let update = |out: &mut [f32], m, a| {
            let old = _mm256_loadu_ps(out.as_ptr().add(i));
            _mm256_storeu_ps(
                out.as_mut_ptr().add(i),
                _mm256_fmadd_ps(value, a, _mm256_mul_ps(old, m)),
            );
        };
        update(out0, mul[0], add[0]);
        update(out1, mul[1], add[1]);
        update(out2, mul[2], add[2]);
        update(out3, mul[3], add[3]);
        i += 8;
    }
    while i < x.len() {
        let value = bf16_to_f32(x[i]);
        out0[i] = out0[i] * multiplier[0] + alpha[0] * value;
        out1[i] = out1[i] * multiplier[1] + alpha[1] * value;
        out2[i] = out2[i] * multiplier[2] + alpha[2] * value;
        out3[i] = out3[i] * multiplier[3] + alpha[3] * value;
        i += 1;
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
    fn four_way_f32_dot_matches_individual_dots() {
        let key: Vec<f32> = (0..67).map(|i| i as f32 * 0.03125 - 1.0).collect();
        let queries: Vec<Vec<f32>> = (0..4)
            .map(|head| {
                (0..67)
                    .map(|i| (i as f32 * (head + 1) as f32 * 0.017) - 0.5)
                    .collect()
            })
            .collect();
        let got = dot_f32x4(&queries[0], &queries[1], &queries[2], &queries[3], &key);
        for head in 0..4 {
            let expected = dot_f32(&queries[head], &key);
            assert!(
                (got[head] - expected).abs() < 1e-4,
                "head {head}: {} vs {expected}",
                got[head]
            );
        }
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
    /// Verifies the MXFP4 NEON path against the scalar reference across
    /// several packed blocks and distinct shared exponents.
    fn mxfp4_dot_matches_scalar_reference() {
        let blocks = 10usize;
        let n = blocks * 32;
        let mut row = vec![0u8; blocks * 17];
        for block in 0..blocks {
            let base = block * 17;
            for i in 0..16 {
                let lo = ((block * 7 + i * 3 + 1) & 0x0f) as u8;
                let hi = ((block * 11 + i * 5 + 9) & 0x0f) as u8;
                row[base + i] = lo | (hi << 4);
            }
            row[base + 16] = 123 + (block as u8 % 9);
        }
        let x: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 0.071).sin() * 0.9 + (i % 13) as f32 * 0.02)
            .collect();

        let expected = dot_mxfp4_f32_scalar(&row, &x, n);
        let actual = dot_mxfp4_f32(&row, &x, n);
        let tolerance = expected.abs().max(1.0) * 2e-5;
        assert!(
            (actual - expected).abs() <= tolerance,
            "MXFP4 SIMD {actual} vs scalar {expected}, tolerance {tolerance}"
        );
    }

    #[test]
    #[ignore = "manual release benchmark; run cargo test --release --lib mxfp4_neon_speedup -- --ignored --nocapture"]
    /// Compares the dispatched MXFP4 kernel against the scalar reference on a
    /// representative GPT-OSS expert-row width. This stays ignored because
    /// wall-clock timing is intentionally not part of the regular test suite.
    fn mxfp4_neon_speedup() {
        const COLS: usize = 2880;
        const ROWS: usize = 192;
        const RUNS: usize = 100;
        const ROW_BYTES: usize = (COLS / 32) * 17;

        let mut weights = vec![0u8; ROWS * ROW_BYTES];
        for row in 0..ROWS {
            for block in 0..(COLS / 32) {
                let base = row * ROW_BYTES + block * 17;
                for i in 0..16 {
                    let lo = ((row * 3 + block * 5 + i * 7 + 1) & 0x0f) as u8;
                    let hi = ((row * 11 + block * 13 + i * 2 + 9) & 0x0f) as u8;
                    weights[base + i] = lo | (hi << 4);
                }
                weights[base + 16] = 123 + ((row + block) as u8 % 9);
            }
        }
        let x: Vec<f32> = (0..COLS)
            .map(|i| (i as f32 * 0.013).sin() * 0.8 + (i % 17) as f32 * 0.01)
            .collect();

        let measure = |kernel: fn(&[u8], &[f32], usize) -> f32| {
            let mut checksum = 0.0f32;
            let start = std::time::Instant::now();
            for _ in 0..RUNS {
                for row in 0..ROWS {
                    let offset = row * ROW_BYTES;
                    checksum += kernel(&weights[offset..offset + ROW_BYTES], &x, COLS);
                }
            }
            (start.elapsed(), std::hint::black_box(checksum))
        };

        let (scalar_time, scalar_sum) = measure(dot_mxfp4_f32_scalar);
        let (simd_time, simd_sum) = measure(dot_mxfp4_f32);
        let tolerance = scalar_sum.abs().max(1.0) * 2e-5;
        assert!(
            (simd_sum - scalar_sum).abs() <= tolerance,
            "MXFP4 checksum {simd_sum} vs scalar {scalar_sum}"
        );
        let speedup = scalar_time.as_secs_f64() / simd_time.as_secs_f64();
        eprintln!(
            "MXFP4 dot: scalar={:.3} ms, SIMD={:.3} ms, speedup={:.2}x",
            scalar_time.as_secs_f64() * 1000.0,
            simd_time.as_secs_f64() * 1000.0,
            speedup,
        );
    }

    #[test]
    #[ignore = "manual release benchmark; run cargo test --release --lib bf16_kv_neon_speedup -- --ignored --nocapture"]
    /// Compares the bf16-stored KV dot/axpy kernels against the equivalent
    /// f32-stored path at a representative head_dim, to confirm the bf16 KV
    /// cache (`KVCache::enable_bf16`) actually saves time on this machine
    /// rather than just DRAM traffic, given the per-element widen cost.
    fn bf16_kv_neon_speedup() {
        const HEAD_DIM: usize = 128;
        const POSITIONS: usize = 65536; // exceeds L2, matches long-context decode
        const RUNS: usize = 4;

        let query: Vec<f32> = (0..HEAD_DIM).map(|i| (i as f32 * 0.017).sin()).collect();
        let keys_f32: Vec<f32> = (0..POSITIONS * HEAD_DIM)
            .map(|i| (i as f32 * 0.0031).cos() * 0.5)
            .collect();
        let keys_bf16: Vec<u16> = keys_f32.iter().map(|&v| f32_to_bf16(v)).collect();

        let measure_f32 = || {
            let mut checksum = 0.0f32;
            let start = std::time::Instant::now();
            for _ in 0..RUNS {
                for pos in 0..POSITIONS {
                    let row = &keys_f32[pos * HEAD_DIM..(pos + 1) * HEAD_DIM];
                    checksum += dot_f32(&query, row);
                }
            }
            (start.elapsed(), std::hint::black_box(checksum))
        };
        let measure_bf16 = || {
            let mut checksum = 0.0f32;
            let start = std::time::Instant::now();
            for _ in 0..RUNS {
                for pos in 0..POSITIONS {
                    let row = &keys_bf16[pos * HEAD_DIM..(pos + 1) * HEAD_DIM];
                    checksum += dot_bf16_f32(&query, row);
                }
            }
            (start.elapsed(), std::hint::black_box(checksum))
        };

        // Correctness is checked per-position, not on the timing loops' summed
        // checksums: with RUNS*POSITIONS additions of signed near-cancelling
        // terms, an aggregate sum comparison is dominated by cancellation
        // noise rather than actual per-dot bf16 rounding error.
        let max_abs_err = (0..POSITIONS)
            .map(|pos| {
                let row_f32 = &keys_f32[pos * HEAD_DIM..(pos + 1) * HEAD_DIM];
                let row_bf16 = &keys_bf16[pos * HEAD_DIM..(pos + 1) * HEAD_DIM];
                (dot_bf16_f32(&query, row_bf16) - dot_f32(&query, row_f32)).abs()
            })
            .fold(0.0f32, f32::max);
        assert!(
            max_abs_err <= 0.1,
            "bf16 KV dot: max per-position error {max_abs_err} exceeds bf16 rounding budget"
        );

        let (f32_time, f32_sum) = measure_f32();
        let (bf16_time, bf16_sum) = measure_bf16();
        let speedup = f32_time.as_secs_f64() / bf16_time.as_secs_f64();
        eprintln!(
            "bf16 KV dot: f32={:.3} ms, bf16={:.3} ms, speedup={:.2}x (checksums f32={f32_sum:.3} bf16={bf16_sum:.3}, max per-position err={max_abs_err:.4})",
            f32_time.as_secs_f64() * 1000.0,
            bf16_time.as_secs_f64() * 1000.0,
            speedup,
        );
    }

    #[test]
    #[ignore = "manual release benchmark; run cargo test --release --lib bf16x4_kv_neon_speedup -- --ignored --nocapture"]
    /// Same comparison as `bf16_kv_neon_speedup` but for the fused 4-head GQA
    /// kernels (`dot_bf16x4_f32` vs `dot_f32x4`), which is the kernel pair
    /// actually used on the model's kv_mul == 4 decode path (model.rs:5662).
    fn bf16x4_kv_neon_speedup() {
        const HEAD_DIM: usize = 128;
        const POSITIONS: usize = 65536; // exceeds L2, matches long-context decode
        const RUNS: usize = 4;

        let q: [Vec<f32>; 4] = std::array::from_fn(|h| {
            (0..HEAD_DIM)
                .map(|i| (i as f32 * 0.017 + h as f32).sin())
                .collect()
        });
        let keys_f32: Vec<f32> = (0..POSITIONS * HEAD_DIM)
            .map(|i| (i as f32 * 0.0031).cos() * 0.5)
            .collect();
        let keys_bf16: Vec<u16> = keys_f32.iter().map(|&v| f32_to_bf16(v)).collect();

        let measure_f32 = || {
            let mut checksum = [0.0f32; 4];
            let start = std::time::Instant::now();
            for _ in 0..RUNS {
                for pos in 0..POSITIONS {
                    let row = &keys_f32[pos * HEAD_DIM..(pos + 1) * HEAD_DIM];
                    let s = dot_f32x4(&q[0], &q[1], &q[2], &q[3], row);
                    for i in 0..4 {
                        checksum[i] += s[i];
                    }
                }
            }
            (start.elapsed(), std::hint::black_box(checksum))
        };
        let measure_bf16 = || {
            let mut checksum = [0.0f32; 4];
            let start = std::time::Instant::now();
            for _ in 0..RUNS {
                for pos in 0..POSITIONS {
                    let row = &keys_bf16[pos * HEAD_DIM..(pos + 1) * HEAD_DIM];
                    let s = dot_bf16x4_f32(&q[0], &q[1], &q[2], &q[3], row);
                    for i in 0..4 {
                        checksum[i] += s[i];
                    }
                }
            }
            (start.elapsed(), std::hint::black_box(checksum))
        };

        // See bf16_kv_neon_speedup: compare per-position, not the timing
        // loops' summed checksums, which cancel down to noise-dominated values.
        let max_abs_err = (0..POSITIONS)
            .map(|pos| {
                let row_f32 = &keys_f32[pos * HEAD_DIM..(pos + 1) * HEAD_DIM];
                let row_bf16 = &keys_bf16[pos * HEAD_DIM..(pos + 1) * HEAD_DIM];
                let f32_scores = dot_f32x4(&q[0], &q[1], &q[2], &q[3], row_f32);
                let bf16_scores = dot_bf16x4_f32(&q[0], &q[1], &q[2], &q[3], row_bf16);
                (0..4)
                    .map(|h| (bf16_scores[h] - f32_scores[h]).abs())
                    .fold(0.0f32, f32::max)
            })
            .fold(0.0f32, f32::max);
        assert!(
            max_abs_err <= 0.1,
            "bf16x4 KV dot: max per-position error {max_abs_err} exceeds bf16 rounding budget"
        );

        let (f32_time, f32_sum) = measure_f32();
        let (bf16_time, bf16_sum) = measure_bf16();
        let speedup = f32_time.as_secs_f64() / bf16_time.as_secs_f64();
        eprintln!(
            "bf16x4 KV dot: f32={:.3} ms, bf16={:.3} ms, speedup={:.2}x (checksums f32={f32_sum:?} bf16={bf16_sum:?}, max per-position err={max_abs_err:.4})",
            f32_time.as_secs_f64() * 1000.0,
            bf16_time.as_secs_f64() * 1000.0,
            speedup,
        );
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

    /// Like `make_q4k_weights` but with a non-zero `dmin` so the `bsums`-based
    /// min term of the integer Q4_K kernel is actually exercised.
    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "x86_64"),
        not(target_family = "wasm")
    ))]
    fn make_q4k_weights_dmin(rows: usize, cols: usize, seed: u8) -> Vec<u8> {
        let row_bytes = (cols / 256) * 144;
        let mut data = vec![0u8; rows * row_bytes];
        for row in 0..rows {
            for block in 0..(cols / 256) {
                let base = row * row_bytes + block * 144;
                data[base] = 0x00;
                data[base + 1] = 0x3c; // f16 1.0 (d)
                data[base + 2] = 0x00;
                data[base + 3] = 0x38; // f16 0.5 (dmin)
                for i in 0..12 {
                    data[base + 4 + i] = seed.wrapping_add((row * 5 + block * 7 + i * 11) as u8);
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
    #[cfg(all(target_arch = "aarch64", not(target_family = "wasm")))]
    /// Verifies the integer (sdot) Q4_K kernel matches explicit dequantization,
    /// including the non-zero `dmin` min term, within int8-activation tolerance.
    fn q4k_sdot_matches_reference_dequant() {
        if !has_dotprod() {
            return;
        }
        let cols = 512;
        for seed in [3u8, 19, 101, 200] {
            let w = make_q4k_weights_dmin(1, cols, seed);
            let x: Vec<f32> = (0..cols)
                .map(|i| ((i as f32 * 0.013).sin() * 0.4) - ((i % 9) as f32 * 0.02))
                .collect();
            let deq = dequant_row_q4_k(&w, cols);
            let reference: f32 = deq.iter().zip(&x).map(|(a, b)| a * b).sum();

            let mut buf = XqBuf::default();
            unsafe { quantize_row_q8k(&x, &mut buf) };
            let xq = XQuant {
                qs: buf.qs.as_ptr(),
                d: buf.d.as_ptr(),
                bsums: buf.bsums.as_ptr(),
            };
            let got = unsafe { dot_q4_k_q8k_neon(&w, xq, cols) };
            let tol = 0.02 * reference.abs() + 1.0;
            assert!(
                (got - reference).abs() < tol,
                "seed {seed}: sdot {got} vs reference {reference}"
            );
        }
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", not(target_family = "wasm")))]
    /// Verifies the integer (sdot) Q6_K kernel matches explicit dequantization.
    fn q6k_sdot_matches_reference_dequant() {
        if !has_dotprod() {
            return;
        }
        let cols = 512;
        for seed in [7u8, 23, 99, 211] {
            let w = make_q6k_weights(1, cols, seed);
            let x: Vec<f32> = (0..cols)
                .map(|i| ((i as f32 * 0.019).sin() * 0.35) - ((i % 11) as f32 * 0.017))
                .collect();
            let deq = dequant_row_q6_k(&w, cols);
            let reference: f32 = deq.iter().zip(&x).map(|(a, b)| a * b).sum();

            let mut buf = XqBuf::default();
            unsafe { quantize_row_q8k(&x, &mut buf) };
            let xq = XQuant {
                qs: buf.qs.as_ptr(),
                d: buf.d.as_ptr(),
                bsums: buf.bsums.as_ptr(),
            };
            let got = unsafe { dot_q6_k_q8k_neon(&w, xq, cols) };
            let tol = 0.02 * reference.abs() + 1.0;
            assert!(
                (got - reference).abs() < tol,
                "seed {seed}: sdot {got} vs reference {reference}"
            );
        }
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", not(target_family = "wasm")))]
    /// Verifies the integer fast path stays consistent across the multi-row
    /// matvec entry point (which routes workers through `prepare_xq`).
    fn q4k_sdot_matvec_matches_reference() {
        if !has_dotprod() {
            return;
        }
        // SAFETY: single-threaded test; forces the CPU path by disabling Metal.
        unsafe { std::env::set_var("RUSTY_LLM_METAL", "0") };
        let cols = 512;
        let rows = 10;
        let w = make_q4k_weights_dmin(rows, cols, 37);
        let x: Vec<f32> = (0..cols)
            .map(|i| ((i as f32 * 0.021).cos() * 0.3) + ((i % 6) as f32 * 0.05))
            .collect();
        let mut out = Vec::new();
        matvec_q4_k_into(&w, &x, rows, cols, &mut out);
        let rb = (cols / 256) * 144;
        for r in 0..rows {
            let deq = dequant_row_q4_k(&w[r * rb..(r + 1) * rb], cols);
            let reference: f32 = deq.iter().zip(&x).map(|(a, b)| a * b).sum();
            let tol = 0.02 * reference.abs() + 1.0;
            assert!(
                (out[r] - reference).abs() < tol,
                "row {r}: matvec {} vs reference {reference}",
                out[r]
            );
        }
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

    #[test]
    /// The long-context batch API must preserve the per-token K-quant result
    /// while returning one contiguous row-major output per projection.
    fn kquant_batched_projections_match_token_matvecs() {
        let cols = 512;
        let tokens = 5;
        let q_rows = 40;
        let k_rows = 35;
        let v_rows = 33;
        let q = make_q4k_weights(q_rows, cols, 3);
        let k = make_q5k_weights(k_rows, cols, 13);
        let v = make_q6k_weights(v_rows, cols, 23);
        let inputs: Vec<f32> = (0..tokens * cols)
            .map(|i| {
                let token = i / cols;
                let col = i % cols;
                ((col as f32 * 0.017 + token as f32 * 0.13).sin() * 0.3)
                    + ((col % 11) as f32 * 0.015)
            })
            .collect();

        let mut expected_q = Vec::new();
        let mut expected_k = Vec::new();
        let mut expected_v = Vec::new();
        for input in inputs.chunks_exact(cols) {
            let mut q_row = Vec::new();
            let mut k_row = Vec::new();
            let mut v_row = Vec::new();
            assert!(matvec_kquant3_into(
                (KQuantMatvecKind::Q4K, &q, q_rows, cols),
                (KQuantMatvecKind::Q5K, &k, k_rows, cols),
                (KQuantMatvecKind::Q6K, &v, v_rows, cols),
                input,
                &mut q_row,
                &mut k_row,
                &mut v_row,
            ));
            expected_q.extend(q_row);
            expected_k.extend(k_row);
            expected_v.extend(v_row);
        }

        let mut out_q = Vec::new();
        let mut out_k = Vec::new();
        let mut out_v = Vec::new();
        assert!(matvec_kquant3_batch_into(
            (KQuantMatvecKind::Q4K, &q, q_rows, cols),
            (KQuantMatvecKind::Q5K, &k, k_rows, cols),
            (KQuantMatvecKind::Q6K, &v, v_rows, cols),
            &inputs,
            &mut out_q,
            &mut out_k,
            &mut out_v,
        ));
        assert_close_slice(&out_q, &expected_q, 1e-5);
        assert_close_slice(&out_k, &expected_k, 1e-5);
        assert_close_slice(&out_v, &expected_v, 1e-5);

        let mut out_q_single = Vec::new();
        let mut out_k_pair = Vec::new();
        assert!(matvec_kquant_batch_into(
            (KQuantMatvecKind::Q4K, &q, q_rows, cols),
            &inputs,
            &mut out_q_single,
        ));
        assert!(matvec_kquant2_batch_into(
            (KQuantMatvecKind::Q4K, &q, q_rows, cols),
            (KQuantMatvecKind::Q5K, &k, k_rows, cols),
            &inputs,
            &mut out_q,
            &mut out_k_pair,
        ));
        assert_close_slice(&out_q_single, &expected_q, 1e-5);
        assert_close_slice(&out_q, &expected_q, 1e-5);
        assert_close_slice(&out_k_pair, &expected_k, 1e-5);
    }

    #[test]
    #[cfg(not(target_family = "wasm"))]
    /// The token-blocked kernels must be a pure scheduling change: every lane
    /// keeps the single-token accumulation order, so results are bit-identical,
    /// not merely close. Token counts sweep 1..=9 so the partial tail group
    /// (1, 2 and 3 live lanes) is exercised alongside full groups of four.
    fn kquant_token_blocked_batch_is_bit_identical() {
        let cols = 512;
        let q_rows = 37;
        let k_rows = 9;
        let v_rows = 41;
        let q = make_q4k_weights(q_rows, cols, 5);
        let k = make_q5k_weights(k_rows, cols, 17);
        let v = make_q6k_weights(v_rows, cols, 29);

        for tokens in 1..=9usize {
            let inputs: Vec<f32> = (0..tokens * cols)
                .map(|i| {
                    let token = i / cols;
                    let col = i % cols;
                    ((col as f32 * 0.011 + token as f32 * 0.37).cos() * 0.4)
                        - ((col % 7) as f32 * 0.019)
                })
                .collect();

            let mut expected_q = Vec::new();
            let mut expected_k = Vec::new();
            let mut expected_v = Vec::new();
            for input in inputs.chunks_exact(cols) {
                let mut q_row = Vec::new();
                let mut k_row = Vec::new();
                let mut v_row = Vec::new();
                assert!(matvec_kquant3_into(
                    (KQuantMatvecKind::Q4K, &q, q_rows, cols),
                    (KQuantMatvecKind::Q5K, &k, k_rows, cols),
                    (KQuantMatvecKind::Q6K, &v, v_rows, cols),
                    input,
                    &mut q_row,
                    &mut k_row,
                    &mut v_row,
                ));
                expected_q.extend(q_row);
                expected_k.extend(k_row);
                expected_v.extend(v_row);
            }

            let mut out_q = Vec::new();
            let mut out_k = Vec::new();
            let mut out_v = Vec::new();
            assert!(matvec_kquant3_batch_into(
                (KQuantMatvecKind::Q4K, &q, q_rows, cols),
                (KQuantMatvecKind::Q5K, &k, k_rows, cols),
                (KQuantMatvecKind::Q6K, &v, v_rows, cols),
                &inputs,
                &mut out_q,
                &mut out_k,
                &mut out_v,
            ));

            for (label, actual, expected) in [
                ("q4_k", &out_q, &expected_q),
                ("q5_k", &out_k, &expected_k),
                ("q6_k", &out_v, &expected_v),
            ] {
                assert_eq!(actual.len(), expected.len(), "{label} length at {tokens}");
                for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        e.to_bits(),
                        "{label} tokens={tokens} value[{idx}] blocked {a} vs single {e}"
                    );
                }
            }
        }
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

    // ─── Int8 (Q8_K activation) fast-path parity tests ──────────────────────

    /// Exact f16 bit patterns so synthetic scales survive the round trip.
    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "x86_64"),
        not(target_family = "wasm")
    ))]
    const F16_SCALES: [u16; 6] = [0x3C00, 0x3800, 0x3400, 0x3E00, 0x4000, 0xB800];

    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "x86_64"),
        not(target_family = "wasm")
    ))]
    fn build_xq(x: &[f32], buf: &mut XqBuf) -> XQuant {
        unsafe { quantize_row_q8k(x, buf) };
        XQuant {
            qs: buf.qs.as_ptr(),
            d: buf.d.as_ptr(),
            bsums: buf.bsums.as_ptr(),
        }
    }

    /// Builds one Q4_0 row with per-block varying scales and nibble data.
    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "x86_64"),
        not(target_family = "wasm")
    ))]
    fn make_q4_0_row_scaled(cols: usize, seed: u8) -> Vec<u8> {
        let blocks = cols / 32;
        let mut data = vec![0u8; blocks * 18];
        for b in 0..blocks {
            let base = b * 18;
            let scale = F16_SCALES[(b + seed as usize) % F16_SCALES.len()];
            data[base..base + 2].copy_from_slice(&scale.to_le_bytes());
            for i in 0..16 {
                data[base + 2 + i] = seed.wrapping_add((b * 11 + i * 5) as u8);
            }
        }
        data
    }

    /// Builds one Q8_0 row with per-block varying scales and int8 data.
    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "x86_64"),
        not(target_family = "wasm")
    ))]
    fn make_q8_0_row_scaled(cols: usize, seed: u8) -> Vec<u8> {
        let blocks = cols / 32;
        let mut data = vec![0u8; blocks * 34];
        for b in 0..blocks {
            let base = b * 34;
            let scale = F16_SCALES[(b + seed as usize) % F16_SCALES.len()];
            data[base..base + 2].copy_from_slice(&scale.to_le_bytes());
            for i in 0..32 {
                data[base + 2 + i] = seed.wrapping_add((b * 13 + i * 7) as u8);
            }
        }
        data
    }

    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "x86_64"),
        not(target_family = "wasm")
    ))]
    fn test_activation(cols: usize) -> Vec<f32> {
        (0..cols)
            .map(|i| ((i as f32 * 0.017).sin() * 0.45) - ((i % 7) as f32 * 0.021))
            .collect()
    }

    /// The SIMD Q8_K activation quantizer must match the scalar reference
    /// exactly (same rounding mode, same group sums).
    #[test]
    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "x86_64"),
        not(target_family = "wasm")
    ))]
    fn q8k_quantizer_simd_matches_scalar() {
        for cols in [256usize, 512, 1024] {
            let x = test_activation(cols);
            let mut simd_buf = XqBuf::default();
            unsafe { quantize_row_q8k(&x, &mut simd_buf) };
            let mut scalar_buf = XqBuf::default();
            quantize_row_q8k_scalar(&x, &mut scalar_buf);
            assert_eq!(simd_buf.qs, scalar_buf.qs, "qs mismatch at cols={cols}");
            assert_eq!(simd_buf.d, scalar_buf.d, "d mismatch at cols={cols}");
            assert_eq!(
                simd_buf.bsums, scalar_buf.bsums,
                "bsums mismatch at cols={cols}"
            );
        }
    }

    // Scalar emulations of the int8-path math. Each mirrors the SIMD kernel's
    // integer accumulation exactly (same isum/imin formula, same per-block f32
    // combining), so SIMD kernels must agree to within accumulation-order
    // noise — a much tighter check than the f32 dequant reference, whose gap
    // to the int8 path is legitimate activation-quantization error.

    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "x86_64"),
        not(target_family = "wasm")
    ))]
    #[allow(dead_code)]
    fn emulate_q4_k_q8k(w: &[u8], buf: &XqBuf, cols: usize) -> f32 {
        let mut acc = 0.0f32;
        for b in 0..cols / 256 {
            let block = &w[b * 144..(b + 1) * 144];
            let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
            let scales: &[u8; 12] = block[4..16].try_into().unwrap();
            let q = &block[16..144];
            let dx = buf.d[b];
            let xq = &buf.qs[b * 256..(b + 1) * 256];
            let bs = &buf.bsums[b * 8..(b + 1) * 8];
            let mut isum = 0i32;
            let mut imin = 0i32;
            for c in 0..4usize {
                let is = c * 2;
                let (sc1, m1) = get_scale_min_k4(is, scales);
                let (sc2, m2) = get_scale_min_k4(is + 1, scales);
                let mut qa = 0i32;
                let mut qb = 0i32;
                for l in 0..32 {
                    let byte = q[c * 32 + l];
                    qa += (byte & 0xF) as i32 * xq[c * 64 + l] as i32;
                    qb += (byte >> 4) as i32 * xq[c * 64 + 32 + l] as i32;
                }
                isum += sc1 as i32 * qa + sc2 as i32 * qb;
                imin += m1 as i32 * bs[is] as i32 + m2 as i32 * bs[is + 1] as i32;
            }
            acc += dx * (d * isum as f32 - dmin * imin as f32);
        }
        acc
    }

    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "x86_64"),
        not(target_family = "wasm")
    ))]
    fn emulate_q5_k_q8k(w: &[u8], buf: &XqBuf, cols: usize) -> f32 {
        let mut acc = 0.0f32;
        for b in 0..cols / 256 {
            let block = &w[b * 176..(b + 1) * 176];
            let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
            let scales: &[u8; 12] = block[4..16].try_into().unwrap();
            let qh = &block[16..48];
            let q = &block[48..176];
            let dx = buf.d[b];
            let xq = &buf.qs[b * 256..(b + 1) * 256];
            let bs = &buf.bsums[b * 8..(b + 1) * 8];
            let mut isum = 0i32;
            let mut imin = 0i32;
            for c in 0..4usize {
                let is = c * 2;
                let (sc1, m1) = get_scale_min_k4(is, scales);
                let (sc2, m2) = get_scale_min_k4(is + 1, scales);
                let u1 = 1u8 << (2 * c);
                let u2 = 2u8 << (2 * c);
                let mut qa = 0i32;
                let mut qb = 0i32;
                for l in 0..32 {
                    let byte = q[c * 32 + l];
                    let a = (byte & 0xF) | if qh[l] & u1 != 0 { 0x10 } else { 0 };
                    let bv = (byte >> 4) | if qh[l] & u2 != 0 { 0x10 } else { 0 };
                    qa += a as i32 * xq[c * 64 + l] as i32;
                    qb += bv as i32 * xq[c * 64 + 32 + l] as i32;
                }
                isum += sc1 as i32 * qa + sc2 as i32 * qb;
                imin += m1 as i32 * bs[is] as i32 + m2 as i32 * bs[is + 1] as i32;
            }
            acc += dx * (d * isum as f32 - dmin * imin as f32);
        }
        acc
    }

    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "x86_64"),
        not(target_family = "wasm")
    ))]
    #[allow(dead_code)]
    fn emulate_q6_k_q8k(w: &[u8], buf: &XqBuf, cols: usize) -> f32 {
        let mut acc = 0.0f32;
        for b in 0..cols / 256 {
            let block = &w[b * 210..(b + 1) * 210];
            let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
            let dx = buf.d[b];
            let xq = &buf.qs[b * 256..(b + 1) * 256];
            let mut isum = 0i32;
            for g in 0..2usize {
                let ql = &block[g * 64..g * 64 + 64];
                let qh = &block[128 + g * 32..128 + g * 32 + 32];
                let scs = &block[192 + g * 8..192 + g * 8 + 8];
                let base = g * 128;
                for l in 0..32 {
                    let q0 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32;
                    let q1 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                    let q2 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                    let q3 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                    let sc_idx = l / 16;
                    isum += scs[sc_idx] as i8 as i32 * q0 * xq[base + l] as i32
                        + scs[sc_idx + 2] as i8 as i32 * q1 * xq[base + 32 + l] as i32
                        + scs[sc_idx + 4] as i8 as i32 * q2 * xq[base + 64 + l] as i32
                        + scs[sc_idx + 6] as i8 as i32 * q3 * xq[base + 96 + l] as i32;
                }
            }
            acc += dx * d * isum as f32;
        }
        acc
    }

    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "x86_64"),
        not(target_family = "wasm")
    ))]
    fn emulate_q4_0_q8k(w: &[u8], buf: &XqBuf, cols: usize) -> f32 {
        let mut acc = 0.0f32;
        for sb in 0..cols / 256 {
            let dx = buf.d[sb];
            let xq = &buf.qs[sb * 256..(sb + 1) * 256];
            let mut inner = 0.0f32;
            for blk in 0..8usize {
                let block = &w[(sb * 8 + blk) * 18..(sb * 8 + blk + 1) * 18];
                let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                let mut s = 0i32;
                for i in 0..16 {
                    let lo = (block[2 + i] & 0xF) as i32 - 8;
                    let hi = (block[2 + i] >> 4) as i32 - 8;
                    s += lo * xq[blk * 32 + i] as i32;
                    s += hi * xq[blk * 32 + 16 + i] as i32;
                }
                inner += d * s as f32;
            }
            acc += dx * inner;
        }
        acc
    }

    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "x86_64"),
        not(target_family = "wasm")
    ))]
    fn emulate_q8_0_q8k(w: &[u8], buf: &XqBuf, cols: usize) -> f32 {
        let mut acc = 0.0f32;
        for sb in 0..cols / 256 {
            let dx = buf.d[sb];
            let xq = &buf.qs[sb * 256..(sb + 1) * 256];
            let mut inner = 0.0f32;
            for blk in 0..8usize {
                let block = &w[(sb * 8 + blk) * 34..(sb * 8 + blk + 1) * 34];
                let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                let mut s = 0i32;
                for i in 0..32 {
                    s += block[2 + i] as i8 as i32 * xq[blk * 32 + i] as i32;
                }
                inner += d * s as f32;
            }
            acc += dx * inner;
        }
        acc
    }

    /// Shared macro: SIMD int8 kernel vs its scalar emulation (near-exact),
    /// plus a coarse sanity check against the f32 dequantized reference.
    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "x86_64"),
        not(target_family = "wasm")
    ))]
    macro_rules! int8_parity_test {
        ($name:ident, $gate:expr, $make:expr, $dequant:ident, $emulate:ident, $kernel:ident, $seeds:expr) => {
            #[test]
            fn $name() {
                if !$gate {
                    return;
                }
                let cols = 512usize;
                for seed in $seeds {
                    let w: Vec<u8> = $make(cols, seed);
                    let x = test_activation(cols);

                    let mut buf = XqBuf::default();
                    let xq = build_xq(&x, &mut buf);
                    let got = unsafe { $kernel(&w, xq, cols) };

                    let emu = $emulate(&w, &buf, cols);
                    let tol = 1e-4 * emu.abs().max(1.0);
                    assert!(
                        (got - emu).abs() <= tol,
                        "seed {seed}: SIMD {got} vs scalar-int8 emulation {emu}"
                    );

                    // Coarse sanity: the int8 scheme itself should stay within
                    // ~10% of full precision even on adversarial synthetic data.
                    let deq = $dequant(&w, cols);
                    let reference: f32 = deq.iter().zip(&x).map(|(a, b)| a * b).sum();
                    let sanity = 0.10 * reference.abs() + 1.0;
                    assert!(
                        (got - reference).abs() < sanity,
                        "seed {seed}: int8 {got} vs dequant reference {reference}"
                    );
                }
            }
        };
    }

    #[cfg(all(target_arch = "aarch64", not(target_family = "wasm")))]
    int8_parity_test!(
        q5k_sdot_matches_reference_dequant,
        has_dotprod(),
        |cols, seed| make_q5k_weights(1, cols, seed),
        dequant_row_q5_k,
        emulate_q5_k_q8k,
        dot_q5_k_q8k_neon,
        [5u8, 21, 103, 202]
    );

    #[cfg(all(target_arch = "aarch64", not(target_family = "wasm")))]
    int8_parity_test!(
        q4_0_sdot_matches_reference_dequant,
        has_dotprod(),
        make_q4_0_row_scaled,
        dequant_row_q4_0,
        emulate_q4_0_q8k,
        dot_q4_0_q8k_neon,
        [3u8, 19, 101, 200]
    );

    #[cfg(all(target_arch = "aarch64", not(target_family = "wasm")))]
    int8_parity_test!(
        q8_0_sdot_matches_reference_dequant,
        has_dotprod(),
        make_q8_0_row_scaled,
        dequant_row_q8_0,
        emulate_q8_0_q8k,
        dot_q8_0_q8k_neon,
        [7u8, 42, 133, 240]
    );

    #[cfg(all(target_arch = "x86_64", not(target_family = "wasm")))]
    int8_parity_test!(
        q4k_avx2_int8_matches_reference_dequant,
        has_avx2_fma(),
        |cols, seed| make_q4k_weights_dmin(1, cols, seed),
        dequant_row_q4_k,
        emulate_q4_k_q8k,
        dot_q4_k_q8k_avx2,
        [3u8, 19, 101, 200]
    );

    #[cfg(all(target_arch = "x86_64", not(target_family = "wasm")))]
    int8_parity_test!(
        q5k_avx2_int8_matches_reference_dequant,
        has_avx2_fma(),
        |cols, seed| make_q5k_weights(1, cols, seed),
        dequant_row_q5_k,
        emulate_q5_k_q8k,
        dot_q5_k_q8k_avx2,
        [5u8, 21, 103, 202]
    );

    #[cfg(all(target_arch = "x86_64", not(target_family = "wasm")))]
    int8_parity_test!(
        q6k_avx2_int8_matches_reference_dequant,
        has_avx2_fma(),
        |cols, seed| make_q6k_weights(1, cols, seed),
        dequant_row_q6_k,
        emulate_q6_k_q8k,
        dot_q6_k_q8k_avx2,
        [7u8, 23, 99, 211]
    );

    #[cfg(all(target_arch = "x86_64", not(target_family = "wasm")))]
    int8_parity_test!(
        q4_0_avx2_int8_matches_reference_dequant,
        has_avx2_fma(),
        make_q4_0_row_scaled,
        dequant_row_q4_0,
        emulate_q4_0_q8k,
        dot_q4_0_q8k_avx2,
        [3u8, 19, 101, 200]
    );

    #[cfg(all(target_arch = "x86_64", not(target_family = "wasm")))]
    int8_parity_test!(
        q8_0_avx2_int8_matches_reference_dequant,
        has_avx2_fma(),
        make_q8_0_row_scaled,
        dequant_row_q8_0,
        emulate_q8_0_q8k,
        dot_q8_0_q8k_avx2,
        [7u8, 42, 133, 240]
    );

    /// The SIMD SiLU-multiply (Cephes expf) must match the scalar libm
    /// reference within far-below-quantization-noise tolerance, including the
    /// clamp region (|x| near 90) and a non-multiple-of-lane-width tail.
    #[test]
    fn silu_mul_simd_matches_scalar() {
        let n = 1003usize;
        let gate: Vec<f32> = (0..n)
            .map(|i| (i as f32 / (n as f32 - 1.0)) * 180.0 - 90.0)
            .collect();
        let up: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.37).sin()) * 2.0).collect();
        let mut expect = vec![0.0f32; n];
        silu_mul_scalar(&gate, &up, &mut expect);
        let mut got = Vec::new();
        silu_mul_into(&gate, &up, &mut got);
        assert_eq!(got.len(), n);
        for i in 0..n {
            let tol = 1e-5f32.max(expect[i].abs() * 1e-5);
            assert!(
                (got[i] - expect[i]).abs() <= tol,
                "i={i} gate={} got {} expect {}",
                gate[i],
                got[i],
                expect[i]
            );
        }
    }

    #[test]
    fn sigmoid_mul_simd_matches_scalar() {
        let n = 1003usize;
        let gate: Vec<f32> = (0..n)
            .map(|i| (i as f32 / (n as f32 - 1.0)) * 180.0 - 90.0)
            .collect();
        let seed: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.37).sin()) * 2.0).collect();
        let mut expect = seed.clone();
        sigmoid_mul_scalar(&mut expect, &gate);
        let mut got = seed;
        sigmoid_mul_in_place(&mut got, &gate);
        for i in 0..n {
            let tol = 1e-5f32.max(expect[i].abs() * 1e-5);
            assert!(
                (got[i] - expect[i]).abs() <= tol,
                "i={i} gate={} got {} expect {}",
                gate[i],
                got[i],
                expect[i]
            );
        }
    }

    #[test]
    #[ignore = "manual release benchmark; run cargo test --release --lib qwen35_silu_neon_speedup -- --ignored --nocapture"]
    /// Measures the two SiLU shapes used by every Qwen3.5 Gated DeltaNet
    /// block: its 10,240-wide causal convolution and 6,144-wide output gate.
    /// Both runs cover all 48 recurrent blocks in one Qwen3.8 decode token.
    fn qwen35_silu_neon_speedup() {
        const CONV_DIM: usize = 10_240;
        const VALUE_DIM: usize = 6_144;
        const RECURRENT_LAYERS: usize = 48;
        const RUNS: usize = 64;

        let conv: Vec<f32> = (0..CONV_DIM)
            .map(|i| (i as f32 * 0.017).sin() * 5.0)
            .collect();
        let gate: Vec<f32> = (0..VALUE_DIM)
            .map(|i| (i as f32 * 0.023).cos() * 5.0)
            .collect();
        let value: Vec<f32> = (0..VALUE_DIM)
            .map(|i| (i as f32 * 0.031).sin() * 2.0)
            .collect();

        let mut scalar_conv = vec![0.0; CONV_DIM];
        let mut scalar_gate = vec![0.0; VALUE_DIM];
        silu_mul_scalar(&conv, &conv, &mut scalar_conv);
        silu_mul_scalar(&gate, &value, &mut scalar_gate);
        let mut simd_conv = Vec::new();
        let mut simd_gate = Vec::new();
        silu_mul_into(&conv, &conv, &mut simd_conv);
        silu_mul_into(&gate, &value, &mut simd_gate);
        for (actual, expected) in simd_conv.iter().zip(&scalar_conv) {
            assert!((actual - expected).abs() <= 1e-5f32.max(expected.abs() * 1e-5));
        }
        for (actual, expected) in simd_gate.iter().zip(&scalar_gate) {
            assert!((actual - expected).abs() <= 1e-5f32.max(expected.abs() * 1e-5));
        }

        let scalar_start = std::time::Instant::now();
        let mut scalar_sum = 0.0f32;
        for _ in 0..RUNS {
            for _ in 0..RECURRENT_LAYERS {
                silu_mul_scalar(&conv, &conv, &mut scalar_conv);
                silu_mul_scalar(&gate, &value, &mut scalar_gate);
                scalar_sum += scalar_conv[0] + scalar_gate[0];
            }
        }
        let scalar_time = scalar_start.elapsed();

        let simd_start = std::time::Instant::now();
        let mut simd_sum = 0.0f32;
        for _ in 0..RUNS {
            for _ in 0..RECURRENT_LAYERS {
                silu_mul_into(&conv, &conv, &mut simd_conv);
                silu_mul_into(&gate, &value, &mut simd_gate);
                simd_sum += simd_conv[0] + simd_gate[0];
            }
        }
        let simd_time = simd_start.elapsed();

        std::hint::black_box((scalar_sum, simd_sum));
        let speedup = scalar_time.as_secs_f64() / simd_time.as_secs_f64();
        eprintln!(
            "Qwen35 GDN SiLU: scalar={:.3} ms, SIMD={:.3} ms, speedup={speedup:.2}x",
            scalar_time.as_secs_f64() * 1000.0,
            simd_time.as_secs_f64() * 1000.0,
        );
    }

    /// End-to-end: the public Q4_0 matvec (worker pool + caller-side shared
    /// activation quantization) must match per-row dequantized references.
    #[test]
    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "x86_64"),
        not(target_family = "wasm")
    ))]
    fn q4_0_matvec_pooled_matches_reference() {
        let cols = 512usize;
        let rows = 256usize;
        let row_bytes = (cols / 32) * 18;
        let mut w = vec![0u8; rows * row_bytes];
        for r in 0..rows {
            let row = make_q4_0_row_scaled(cols, (r % 251) as u8);
            w[r * row_bytes..(r + 1) * row_bytes].copy_from_slice(&row);
        }
        let x = test_activation(cols);
        let mut out = Vec::new();
        matvec_q4_0_into(&w, &x, rows, cols, &mut out);
        for r in 0..rows {
            let deq = dequant_row_q4_0(&w[r * row_bytes..(r + 1) * row_bytes], cols);
            let reference: f32 = deq.iter().zip(&x).map(|(a, b)| a * b).sum();
            let tol = 0.02 * reference.abs() + 1.0;
            assert!(
                (out[r] - reference).abs() < tol,
                "row {r}: matvec {} vs reference {reference}",
                out[r]
            );
        }
    }

    /// The prefill widening ceiling must never sit below the decode default,
    /// otherwise `effective_prefill_threads` would ask to *narrow* the pool for
    /// long prompts. On Apple Silicon the two intentionally differ (performance
    /// cores for decode, every physical core for prefill); everywhere else they
    /// resolve to the same physical-core count.
    #[test]
    #[cfg(not(target_family = "wasm"))]
    fn prefill_thread_ceiling_never_below_decode_default() {
        let decode = super::default_worker_threads();
        let prefill = super::physical_threads();
        assert!(decode >= 1, "decode default must be positive, got {decode}");
        assert!(
            prefill >= 1,
            "prefill ceiling must be positive, got {prefill}"
        );
        assert!(
            prefill >= decode,
            "prefill ceiling {prefill} is below the decode default {decode}"
        );
    }

    /// Both core counts must stay within the machine's logical parallelism, so a
    /// bogus `sysctl` reading can never oversubscribe the pool.
    #[test]
    #[cfg(not(target_family = "wasm"))]
    fn core_counts_do_not_exceed_logical_parallelism() {
        let logical = super::available_threads();
        assert!(
            super::default_worker_threads() <= logical,
            "decode default {} exceeds {logical} logical threads",
            super::default_worker_threads()
        );
        assert!(
            super::physical_threads() <= logical,
            "prefill ceiling {} exceeds {logical} logical threads",
            super::physical_threads()
        );
    }

    #[test]
    fn bf16_roundtrip_is_exact_for_bf16_representable_values() {
        // Values whose low 16 mantissa bits are already zero survive
        // f32 -> bf16 -> f32 exactly.
        for bits in [0x3f80_0000u32, 0xbf00_0000, 0x4120_0000, 0x0000_0000] {
            let x = f32::from_bits(bits);
            assert_eq!(bf16_to_f32(f32_to_bf16(x)), x);
        }
    }

    #[test]
    fn bf16_roundtrip_rounds_to_nearest_even() {
        // 1.0 + 2^-8 rounds up (tie broken toward even mantissa); relative
        // error must stay within bf16's ~2^-8 precision.
        let x = 1.0f32 + 1.0 / 256.0;
        let y = bf16_to_f32(f32_to_bf16(x));
        assert!((y - x).abs() <= x * (1.0 / 128.0));
    }

    #[test]
    fn bf16_conversion_preserves_nan() {
        let n = f32::NAN;
        assert!(bf16_to_f32(f32_to_bf16(n)).is_nan());
    }

    fn bf16_row(values: &[f32]) -> Vec<u16> {
        values.iter().map(|&v| f32_to_bf16(v)).collect()
    }

    #[test]
    fn dot_bf16_matches_f32_reference_within_bf16_precision() {
        let n = 37; // deliberately not a multiple of 8/4 to exercise tails
        let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.037 - 0.6).collect();
        let b: Vec<f32> = (0..n).map(|i| ((i * 7 % 13) as f32) * 0.11 - 0.4).collect();
        let b_bf16 = bf16_row(&b);
        let b_widened: Vec<f32> = b_bf16.iter().map(|&h| bf16_to_f32(h)).collect();

        let got = dot_bf16_f32(&a, &b_bf16);
        let want = dot_f32(&a, &b_widened);
        assert!(
            (got - want).abs() <= 1e-3 * want.abs().max(1.0),
            "got {got}, want {want}"
        );
    }

    #[test]
    fn axpy_bf16_matches_f32_reference() {
        let n = 29;
        let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.21 - 1.3).collect();
        let x_bf16 = bf16_row(&x);
        let x_widened: Vec<f32> = x_bf16.iter().map(|&h| bf16_to_f32(h)).collect();

        let mut out_bf16 = vec![0.5f32; n];
        let mut out_f32 = vec![0.5f32; n];
        axpy_bf16_f32(&mut out_bf16, 1.7, &x_bf16);
        axpy_f32(&mut out_f32, 1.7, &x_widened);
        for (got, want) in out_bf16.iter().zip(out_f32.iter()) {
            assert!((got - want).abs() <= 1e-5, "got {got}, want {want}");
        }
    }

    #[test]
    fn scale_add_bf16_matches_f32_reference() {
        let n = 41;
        let add: Vec<f32> = (0..n).map(|i| (i as f32) * -0.09 + 2.0).collect();
        let add_bf16 = bf16_row(&add);
        let add_widened: Vec<f32> = add_bf16.iter().map(|&h| bf16_to_f32(h)).collect();

        let mut out_bf16 = vec![0.3f32; n];
        let mut out_f32 = vec![0.3f32; n];
        scale_add_bf16_f32(&mut out_bf16, 0.42, &add_bf16);
        scale_add_f32(&mut out_f32, 0.42, &add_widened);
        for (got, want) in out_bf16.iter().zip(out_f32.iter()) {
            assert!((got - want).abs() <= 1e-5, "got {got}, want {want}");
        }
    }

    #[test]
    fn dot_bf16x4_matches_four_individual_bf16_dots() {
        let n = 23;
        let a0: Vec<f32> = (0..n).map(|i| (i as f32) * 0.05 - 0.4).collect();
        let a1: Vec<f32> = (0..n).map(|i| (i as f32) * -0.03 + 0.2).collect();
        let a2: Vec<f32> = (0..n).map(|i| (i as f32) * 0.09).collect();
        let a3: Vec<f32> = (0..n).map(|i| 1.0 - (i as f32) * 0.02).collect();
        let b: Vec<f32> = (0..n).map(|i| ((i * 5 % 11) as f32) * 0.13 - 0.5).collect();
        let b_bf16 = bf16_row(&b);

        let got = dot_bf16x4_f32(&a0, &a1, &a2, &a3, &b_bf16);
        let want = [
            dot_bf16_f32(&a0, &b_bf16),
            dot_bf16_f32(&a1, &b_bf16),
            dot_bf16_f32(&a2, &b_bf16),
            dot_bf16_f32(&a3, &b_bf16),
        ];
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() <= 1e-4, "got {g}, want {w}");
        }
    }

    #[test]
    fn affine_add_bf16x4_matches_scalar_reference_formula() {
        let n = 19;
        let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.07 - 0.6).collect();
        let x_bf16 = bf16_row(&x);
        let multiplier = [0.0f32, 1.0, 0.5, 2.0];
        let alpha = [1.0f32, 0.3, 0.7, 1.5];
        let bases = [
            vec![0.2f32; n],
            vec![0.4f32; n],
            vec![-0.1f32; n],
            vec![0.9f32; n],
        ];

        let mut out0 = bases[0].clone();
        let mut out1 = bases[1].clone();
        let mut out2 = bases[2].clone();
        let mut out3 = bases[3].clone();
        affine_add_bf16x4_f32(
            &mut out0, &mut out1, &mut out2, &mut out3, multiplier, alpha, &x_bf16,
        );
        let got = [out0, out1, out2, out3];

        for g in 0..4 {
            for i in 0..n {
                let want = bases[g][i] * multiplier[g] + alpha[g] * bf16_to_f32(x_bf16[i]);
                assert!(
                    (got[g][i] - want).abs() <= 1e-5,
                    "head {g}, i {i}: {} vs {want}",
                    got[g][i]
                );
            }
        }
    }
}
