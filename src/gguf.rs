// gguf.rs — Zero-copy GGUF parser operating on mmap'd bytes
//
// Reads the GGUF header, metadata, and tensor info directly from a byte slice.
// Tensor data stays in the mmap — we only dequantize on-demand or keep raw
// quantized pointers for fused SIMD kernels.

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(non_camel_case_types)]
#[repr(u32)]
pub enum GGMLType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    BF16 = 30,
    MXFP4 = 39,
    Unknown = 255,
}

impl From<u32> for GGMLType {
    /// converts raw GGUF tensor type IDs into `GGMLType` variants.
    fn from(v: u32) -> Self {
        match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            15 => Self::Q8_K,
            30 => Self::BF16,
            39 => Self::MXFP4,
            _ => Self::Unknown,
        }
    }
}

impl GGMLType {
    /// Bytes per block of quantized data
    pub fn block_bytes(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 => 18, // f16 scale + 16 packed nibbles
            Self::Q4_1 => 20, // f16 scale + f16 min + 16 packed nibbles
            Self::Q5_0 => 22, // f16 scale + 32 high bits + 16 packed nibbles
            Self::Q5_1 => 24, // f16 scale + f16 min + 32 high bits + 16 nibbles
            Self::Q8_0 => 34, // f16 scale + 32 i8 quants
            Self::Q8_1 => 36, // f16 scale + f16 sum + 32 i8 quants
            Self::Q4_K => 144,
            Self::Q5_K => 176,
            Self::Q6_K => 210,
            Self::MXFP4 => 17,
            _ => panic!("Unsupported: {:?}", self),
        }
    }

    /// Elements per block
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => 1,
            Self::Q4_K | Self::Q5_K | Self::Q6_K => 256,
            _ => 32,
        }
    }

    /// Total bytes needed for `n` elements
    pub fn data_size(&self, n: usize) -> Option<usize> {
        match self {
            Self::F32 => Some(n * 4),
            Self::F16 | Self::BF16 => Some(n * 2),
            Self::Q4_0
            | Self::Q4_1
            | Self::Q5_0
            | Self::Q5_1
            | Self::Q8_0
            | Self::Q8_1
            | Self::MXFP4 => {
                let blocks = n.div_ceil(self.block_size());
                Some(blocks * self.block_bytes())
            }
            // K-quant variants use a different super-block size.
            Self::Q4_K | Self::Q5_K | Self::Q6_K => {
                let blocks = n.div_ceil(self.block_size());
                Some(blocks * self.block_bytes())
            }
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    Str(String),
    Array(Vec<MetaValue>),
}

impl MetaValue {
    /// reads integer-like metadata as `u32`.
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::U32(v) => Some(*v),
            Self::I32(v) => Some(*v as u32),
            Self::U64(v) => Some(*v as u32),
            Self::I64(v) => Some(*v as u32),
            Self::U8(v) => Some(*v as u32),
            Self::U16(v) => Some(*v as u32),
            _ => None,
        }
    }

    /// reads numeric metadata as `f32`.
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::F32(v) => Some(*v),
            Self::F64(v) => Some(*v as f32),
            _ => None,
        }
    }

    /// reads string metadata.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(s) => Some(s),
            _ => None,
        }
    }

    /// reads string-array metadata.
    pub fn as_string_array(&self) -> Option<Vec<String>> {
        match self {
            Self::Array(arr) => Some(
                arr.iter()
                    .filter_map(|v| {
                        if let Self::Str(s) = v {
                            Some(s.clone())
                        } else {
                            None
                        }
                    })
                    .collect(),
            ),
            _ => None,
        }
    }

    /// reads float-array metadata.
    pub fn as_f32_array(&self) -> Option<Vec<f32>> {
        match self {
            Self::Array(arr) => Some(arr.iter().filter_map(|v| v.as_f32()).collect()),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub dtype: GGMLType,
    pub offset: u64, // offset from tensor data start
}

impl TensorInfo {
    /// returns the tensor element count from its dimensions.
    pub fn numel(&self) -> usize {
        self.dims.iter().map(|d| *d as usize).product()
    }
}

/// Cursor for reading through a byte slice
struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    /// Starts a little-endian reader at the beginning of a GGUF byte slice.
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// Reads one unsigned byte and advances the cursor.
    fn read_u8(&mut self) -> u8 {
        let v = self.data[self.pos];
        self.pos += 1;
        v
    }
    /// Reads one little-endian `u16` and advances the cursor.
    fn read_u16(&mut self) -> u16 {
        let v = u16::from_le_bytes([self.data[self.pos], self.data[self.pos + 1]]);
        self.pos += 2;
        v
    }
    /// Reads one little-endian `u32` and advances the cursor.
    fn read_u32(&mut self) -> u32 {
        let b = &self.data[self.pos..self.pos + 4];
        self.pos += 4;
        u32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }
    /// Reads one little-endian `i32` and advances the cursor.
    fn read_i32(&mut self) -> i32 {
        let b = &self.data[self.pos..self.pos + 4];
        self.pos += 4;
        i32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }
    /// Reads one little-endian `u64` and advances the cursor.
    fn read_u64(&mut self) -> u64 {
        let b = &self.data[self.pos..self.pos + 8];
        self.pos += 8;
        u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
    }
    /// Reads one little-endian `i64` and advances the cursor.
    fn read_i64(&mut self) -> i64 {
        let b = &self.data[self.pos..self.pos + 8];
        self.pos += 8;
        i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
    }
    /// Reads one little-endian `f32` and advances the cursor.
    fn read_f32(&mut self) -> f32 {
        let b = &self.data[self.pos..self.pos + 4];
        self.pos += 4;
        f32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }
    /// Reads one little-endian `f64` and advances the cursor.
    fn read_f64(&mut self) -> f64 {
        let b = &self.data[self.pos..self.pos + 8];
        self.pos += 8;
        f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
    }
    /// reads a GGUF length-prefixed UTF-8 string.
    fn read_string(&mut self) -> String {
        let len = self.read_u64() as usize;
        let s = String::from_utf8_lossy(&self.data[self.pos..self.pos + len]).to_string();
        self.pos += len;
        s
    }
    /// reads a GGUF boolean.
    fn read_bool(&mut self) -> bool {
        self.read_u8() != 0
    }

    /// reads one typed GGUF metadata value.
    fn read_value(&mut self, vtype: u32) -> Result<MetaValue, String> {
        match vtype {
            0 => Ok(MetaValue::U8(self.read_u8())),
            1 => Ok(MetaValue::I8(self.read_u8() as i8)),
            2 => Ok(MetaValue::U16(self.read_u16())),
            3 => Ok(MetaValue::I16(self.read_u16() as i16)),
            4 => Ok(MetaValue::U32(self.read_u32())),
            5 => Ok(MetaValue::I32(self.read_i32())),
            6 => Ok(MetaValue::F32(self.read_f32())),
            7 => Ok(MetaValue::Bool(self.read_bool())),
            8 => Ok(MetaValue::Str(self.read_string())),
            9 => {
                let elem_type = self.read_u32();
                let count = self.read_u64() as usize;
                let mut arr = Vec::with_capacity(count);
                for _ in 0..count {
                    arr.push(self.read_value(elem_type)?);
                }
                Ok(MetaValue::Array(arr))
            }
            10 => Ok(MetaValue::U64(self.read_u64())),
            11 => Ok(MetaValue::I64(self.read_i64())),
            12 => Ok(MetaValue::F64(self.read_f64())),
            _ => Err(format!(
                "Unknown GGUF metadata value type {}. The file may use a newer GGUF format version.",
                vtype
            )),
        }
    }
}

/// Parsed GGUF file — metadata + tensor layout info
pub struct GGUFFile {
    pub metadata: HashMap<String, MetaValue>,
    pub tensors: Vec<TensorInfo>,
    pub data_offset: usize, // byte offset where tensor data begins
}

impl GGUFFile {
    /// Parse GGUF header + metadata + tensor info from a byte slice (typically mmap'd)
    pub fn parse(data: &[u8]) -> Result<Self, String> {
        Self::parse_inner(data, true)
    }

    /// Parse GGUF metadata without printing header diagnostics.
    pub fn parse_quiet(data: &[u8]) -> Result<Self, String> {
        Self::parse_inner(data, false)
    }

    /// implements shared GGUF header, metadata, tensor, and.
    fn parse_inner(data: &[u8], verbose: bool) -> Result<Self, String> {
        let mut c = Cursor::new(data);

        if data.len() < 4 {
            return Err("File too small for GGUF header".to_string());
        }
        let magic_bytes = &data[0..4];
        if magic_bytes != b"GGUF" {
            let v = u32::from_le_bytes([
                magic_bytes[0],
                magic_bytes[1],
                magic_bytes[2],
                magic_bytes[3],
            ]);
            return Err(format!("Invalid GGUF magic: 0x{:08X}", v));
        }
        // advance cursor past magic
        c.pos = 4;

        let version = c.read_u32();
        let n_tensors = c.read_u64() as usize;
        let n_kv = c.read_u64() as usize;

        if verbose {
            eprintln!(
                "GGUF v{} — {} tensors, {} metadata entries",
                version, n_tensors, n_kv
            );
        }

        // Parse metadata
        let mut metadata = HashMap::with_capacity(n_kv);
        for _ in 0..n_kv {
            let key = c.read_string();
            let vtype = c.read_u32();
            let value = c.read_value(vtype)?;
            metadata.insert(key, value);
        }

        // Parse tensor infos
        let mut tensors = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = c.read_string();
            let n_dims = c.read_u32();
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(c.read_u64());
            }
            let dtype = GGMLType::from(c.read_u32());
            let offset = c.read_u64();
            tensors.push(TensorInfo {
                name,
                dims,
                dtype,
                offset,
            });
        }

        // Tensor data is aligned to 32 bytes after the header
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u32())
            .unwrap_or(32) as usize;
        let data_offset = c.pos.div_ceil(alignment) * alignment;

        Ok(Self {
            metadata,
            tensors,
            data_offset,
        })
    }

    /// returns a `u32` metadata value or a default.
    pub fn get_u32(&self, key: &str, default: u32) -> u32 {
        self.metadata
            .get(key)
            .and_then(|v| v.as_u32())
            .unwrap_or(default)
    }

    /// returns an `f32` metadata value or a default.
    pub fn get_f32(&self, key: &str, default: f32) -> f32 {
        self.metadata
            .get(key)
            .and_then(|v| v.as_f32())
            .unwrap_or(default)
    }

    /// returns a string metadata value when present.
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).and_then(|v| v.as_str())
    }
}
