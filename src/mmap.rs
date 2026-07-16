// mmap.rs — Memory-mapped file support (macOS, Linux, and Windows — no dependencies)
//
// Uses raw syscalls on Unix and raw kernel32 bindings on Windows, no libc/CGo needed.
// On Apple Silicon, mmap is the fastest way to load weights — the unified memory
// architecture means mmap'd pages are directly accessible by CPU without copying.

use std::fs::File;
use std::io;

pub struct MmapFile {
    ptr: *mut u8,
    len: usize,
    locked: bool,
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

        let ptr = platform_map(&file, len)?;

        // Hint sequential access, then request eager page-fault to reduce
        // per-weight load latency during model loading.
        platform_prefetch(ptr, len);

        Ok(Self {
            ptr,
            len,
            locked: false,
        })
    }

    /// Best-effort lock for mapped model pages, matching llama.cpp's optional mlock path.
    pub fn lock_in_memory(&mut self) -> io::Result<()> {
        if self.locked {
            return Ok(());
        }
        platform_lock(self.ptr, self.len)?;
        self.locked = true;
        Ok(())
    }

    /// Get the full memory-mapped region as a byte slice
    #[inline]
    /// Returns the tensor bytes regardless of whether they are owned or borrowed.
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Length in bytes
    #[inline]
    /// Returns the number of stored items or bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    /// Reports whether the collection or mapping is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for MmapFile {
    /// Releases the resource represented by this guard or mapping.
    fn drop(&mut self) {
        if self.locked {
            platform_unlock(self.ptr, self.len);
        }
        platform_unmap(self.ptr, self.len);
    }
}

// ─── Unix: raw syscall wrappers (no libc crate needed) ──────────────────────

#[cfg(unix)]
mod unix_sys {
    use std::fs::File;
    use std::io;
    use std::os::unix::io::AsRawFd;

    const PROT_READ: i32 = 1;
    const MAP_PRIVATE: i32 = 2;
    const MAP_FAILED: *mut std::ffi::c_void = !0usize as *mut std::ffi::c_void;

    // MADV_SEQUENTIAL = 2, MADV_WILLNEED = 3 on both Linux and macOS
    const MADV_SEQUENTIAL: i32 = 2;
    const MADV_WILLNEED: i32 = 3;

    // These use the libc ABI directly — same on macOS and Linux
    unsafe extern "C" {
        /// Maps a file descriptor into this process address space through the libc ABI.
        fn mmap(
            addr: *mut std::ffi::c_void,
            len: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            offset: i64,
        ) -> *mut std::ffi::c_void;

        /// Releases a memory range previously returned by `mmap`.
        fn munmap(addr: *mut std::ffi::c_void, len: usize) -> i32;

        /// Provides sequential-read and prefetch advice for the mapped model bytes.
        fn madvise(addr: *mut std::ffi::c_void, len: usize, advice: i32) -> i32;

        /// Pins mapped model pages in physical memory when the OS allows it.
        fn mlock(addr: *const std::ffi::c_void, len: usize) -> i32;

        /// Releases a previous memory lock.
        fn munlock(addr: *const std::ffi::c_void, len: usize) -> i32;
    }

    /// Maps `len` bytes of `file` read-only and returns the base pointer.
    pub fn map(file: &File, len: usize) -> io::Result<*mut u8> {
        let fd = file.as_raw_fd();
        // mmap(NULL, len, PROT_READ, MAP_PRIVATE, fd, 0)
        let ptr = unsafe { mmap(std::ptr::null_mut(), len, PROT_READ, MAP_PRIVATE, fd, 0) };
        if ptr == MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        Ok(ptr as *mut u8)
    }

    /// Advises the kernel that the mapping will be read soon and sequentially.
    pub fn prefetch(ptr: *mut u8, len: usize) {
        unsafe {
            madvise(ptr as *mut std::ffi::c_void, len, MADV_SEQUENTIAL);
            madvise(ptr as *mut std::ffi::c_void, len, MADV_WILLNEED);
        }
    }

    /// Pins the mapping in physical memory.
    pub fn lock(ptr: *mut u8, len: usize) -> io::Result<()> {
        let rc = unsafe { mlock(ptr as *const std::ffi::c_void, len) };
        if rc != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    /// Releases a previous memory lock.
    pub fn unlock(ptr: *mut u8, len: usize) {
        unsafe {
            munlock(ptr as *const std::ffi::c_void, len);
        }
    }

    /// Unmaps the mapping.
    pub fn unmap(ptr: *mut u8, len: usize) {
        unsafe {
            munmap(ptr as *mut std::ffi::c_void, len);
        }
    }
}

#[cfg(unix)]
use unix_sys::{
    lock as platform_lock, map as platform_map, prefetch as platform_prefetch,
    unlock as platform_unlock, unmap as platform_unmap,
};

// ─── Windows: raw kernel32 bindings (no winapi/windows crates needed) ───────

#[cfg(windows)]
mod windows_sys {
    use std::fs::File;
    use std::io;
    use std::os::windows::io::AsRawHandle;

    type Handle = *mut std::ffi::c_void;

    const PAGE_READONLY: u32 = 0x02;
    const FILE_MAP_READ: u32 = 0x0004;

    #[repr(C)]
    struct MemoryRangeEntry {
        virtual_address: *mut std::ffi::c_void,
        number_of_bytes: usize,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        /// Creates a read-only file-mapping object over an open file handle.
        fn CreateFileMappingW(
            file: Handle,
            attributes: *mut std::ffi::c_void,
            protect: u32,
            maximum_size_high: u32,
            maximum_size_low: u32,
            name: *const u16,
        ) -> Handle;

        /// Maps a view of the file-mapping object into this process address space.
        fn MapViewOfFile(
            mapping: Handle,
            desired_access: u32,
            offset_high: u32,
            offset_low: u32,
            number_of_bytes: usize,
        ) -> *mut std::ffi::c_void;

        /// Releases a view previously returned by `MapViewOfFile`.
        fn UnmapViewOfFile(base_address: *const std::ffi::c_void) -> i32;

        /// Closes a kernel handle (the mapping handle; the view keeps the data alive).
        fn CloseHandle(handle: Handle) -> i32;

        /// Requests eager page-in of the mapped model bytes (Windows 8+).
        fn PrefetchVirtualMemory(
            process: Handle,
            number_of_entries: usize,
            virtual_addresses: *mut MemoryRangeEntry,
            flags: u32,
        ) -> i32;

        /// Returns the pseudo-handle for the current process.
        fn GetCurrentProcess() -> Handle;

        /// Pins mapped model pages in physical memory when the OS allows it.
        fn VirtualLock(address: *mut std::ffi::c_void, size: usize) -> i32;

        /// Releases a previous memory lock.
        fn VirtualUnlock(address: *mut std::ffi::c_void, size: usize) -> i32;
    }

    /// Maps `len` bytes of `file` read-only and returns the base pointer.
    pub fn map(file: &File, len: usize) -> io::Result<*mut u8> {
        let file_handle = file.as_raw_handle() as Handle;
        let mapping = unsafe {
            CreateFileMappingW(
                file_handle,
                std::ptr::null_mut(),
                PAGE_READONLY,
                0,
                0,
                std::ptr::null(),
            )
        };
        if mapping.is_null() {
            return Err(io::Error::last_os_error());
        }

        let view = unsafe { MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, len) };
        // The view (once created) keeps the underlying section alive; the
        // mapping handle itself is no longer needed either way.
        unsafe {
            CloseHandle(mapping);
        }
        if view.is_null() {
            return Err(io::Error::last_os_error());
        }
        Ok(view as *mut u8)
    }

    /// Requests eager page-in of the mapping; best-effort.
    pub fn prefetch(ptr: *mut u8, len: usize) {
        let mut range = MemoryRangeEntry {
            virtual_address: ptr as *mut std::ffi::c_void,
            number_of_bytes: len,
        };
        unsafe {
            PrefetchVirtualMemory(GetCurrentProcess(), 1, &mut range, 0);
        }
    }

    /// Pins the mapping in physical memory.
    pub fn lock(ptr: *mut u8, len: usize) -> io::Result<()> {
        let rc = unsafe { VirtualLock(ptr as *mut std::ffi::c_void, len) };
        if rc == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    /// Releases a previous memory lock.
    pub fn unlock(ptr: *mut u8, len: usize) {
        unsafe {
            VirtualUnlock(ptr as *mut std::ffi::c_void, len);
        }
    }

    /// Unmaps the view.
    pub fn unmap(ptr: *mut u8, _len: usize) {
        unsafe {
            UnmapViewOfFile(ptr as *const std::ffi::c_void);
        }
    }
}

#[cfg(windows)]
use windows_sys::{
    lock as platform_lock, map as platform_map, prefetch as platform_prefetch,
    unlock as platform_unlock, unmap as platform_unmap,
};
