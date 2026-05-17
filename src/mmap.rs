// mmap.rs — Memory-mapped file support (works on macOS + Linux, no dependencies)
//
// Uses raw syscalls, no libc/CGo needed.
// On Apple Silicon, mmap is the fastest way to load weights — the unified memory
// architecture means mmap'd pages are directly accessible by CPU without copying.

use std::fs::File;
use std::io;
use std::os::unix::io::AsRawFd;

pub struct MmapFile {
    ptr: *mut u8,
    len: usize,
}

// SAFETY: The mmap'd region is read-only and lives for the lifetime of MmapFile.
// Multiple threads can safely read from it concurrently.
unsafe impl Send for MmapFile {}
unsafe impl Sync for MmapFile {}

impl MmapFile {
    /// Memory-map an entire file as read-only
    pub fn open(path: &str) -> io::Result<Self> {
        let file = File::open(path)?;
        let metadata = file.metadata()?;
        let len = metadata.len() as usize;

        if len == 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Empty file"));
        }

        let fd = file.as_raw_fd();

        // mmap(NULL, len, PROT_READ, MAP_PRIVATE, fd, 0)
        let ptr = unsafe { libc_mmap(std::ptr::null_mut(), len, PROT_READ, MAP_PRIVATE, fd, 0) };

        if ptr == MAP_FAILED {
            return Err(io::Error::last_os_error());
        }

        // Hint sequential access, then request eager page-fault to reduce
        // per-weight load latency during model loading.
        unsafe {
            libc_madvise(ptr, len, MADV_SEQUENTIAL);
            libc_madvise(ptr, len, MADV_WILLNEED);
        }

        Ok(Self {
            ptr: ptr as *mut u8,
            len,
        })
    }

    /// Get the full memory-mapped region as a byte slice
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Length in bytes
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for MmapFile {
    fn drop(&mut self) {
        unsafe {
            libc_munmap(self.ptr as *mut std::ffi::c_void, self.len);
        }
    }
}

// ─── Raw syscall wrappers (no libc crate needed) ────────────────────────────

const PROT_READ: i32 = 1;
const MAP_PRIVATE: i32 = 2;
const MAP_FAILED: *mut std::ffi::c_void = !0usize as *mut std::ffi::c_void;

// MADV_SEQUENTIAL = 2, MADV_WILLNEED = 3 on both Linux and macOS
const MADV_SEQUENTIAL: i32 = 2;
const MADV_WILLNEED: i32 = 3;

// These use the libc ABI directly — same on macOS and Linux
unsafe extern "C" {
    fn mmap(
        addr: *mut std::ffi::c_void,
        len: usize,
        prot: i32,
        flags: i32,
        fd: i32,
        offset: i64,
    ) -> *mut std::ffi::c_void;

    fn munmap(addr: *mut std::ffi::c_void, len: usize) -> i32;

    fn madvise(addr: *mut std::ffi::c_void, len: usize, advice: i32) -> i32;
}

unsafe fn libc_mmap(
    addr: *mut std::ffi::c_void,
    len: usize,
    prot: i32,
    flags: i32,
    fd: i32,
    offset: i64,
) -> *mut std::ffi::c_void {
    unsafe { mmap(addr, len, prot, flags, fd, offset) }
}

unsafe fn libc_munmap(addr: *mut std::ffi::c_void, len: usize) -> i32 {
    unsafe { munmap(addr, len) }
}

unsafe fn libc_madvise(addr: *mut std::ffi::c_void, len: usize, advice: i32) -> i32 {
    unsafe { madvise(addr, len, advice) }
}
