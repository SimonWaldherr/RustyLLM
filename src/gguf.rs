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
    MXFP4 = 39,
    Unknown = 255,
}

impl From<u32> for GGMLType {
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
            Self::F16 => 2,
            Self::Q4_0 | Self::Q4_1 => 18, // 2 (f16 scale) + 16 (32 nibbles)
            Self::Q8_0 | Self::Q8_1 => 34, // 2 (f16 scale) + 32 (i8 quants)
            Self::Q5_0 | Self::Q5_1 => 34, // treat like Q8 layout for now
            _ => panic!("Unsupported: {:?}", self),
        }
    }

    /// Elements per block
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 | Self::F16 => 1,
            _ => 32,
        }
    }

    /// Total bytes needed for `n` elements
    pub fn data_size(&self, n: usize) -> Option<usize> {
        match self {
            Self::F32 => Some(n * 4),
            Self::F16 => Some(n * 2),
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => {
                Some((n / self.block_size()) * self.block_bytes())
            }
            // K-quant variants use a different super-block size.
            Self::Q4_K => Some((n / 256) * 144),
            Self::Q6_K => Some((n / 256) * 210),
            Self::MXFP4 => Some((n / 32) * 17),
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

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::F32(v) => Some(*v),
            Self::F64(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(s) => Some(s),
            _ => None,
        }
    }

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
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn read_u8(&mut self) -> u8 {
        let v = self.data[self.pos];
        self.pos += 1;
        v
    }
    fn read_u16(&mut self) -> u16 {
        let v = u16::from_le_bytes([self.data[self.pos], self.data[self.pos + 1]]);
        self.pos += 2;
        v
    }
    fn read_u32(&mut self) -> u32 {
        let b = &self.data[self.pos..self.pos + 4];
        self.pos += 4;
        u32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }
    fn read_i32(&mut self) -> i32 {
        let b = &self.data[self.pos..self.pos + 4];
        self.pos += 4;
        i32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }
    fn read_u64(&mut self) -> u64 {
        let b = &self.data[self.pos..self.pos + 8];
        self.pos += 8;
        u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
    }
    fn read_i64(&mut self) -> i64 {
        let b = &self.data[self.pos..self.pos + 8];
        self.pos += 8;
        i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
    }
    fn read_f32(&mut self) -> f32 {
        let b = &self.data[self.pos..self.pos + 4];
        self.pos += 4;
        f32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }
    fn read_f64(&mut self) -> f64 {
        let b = &self.data[self.pos..self.pos + 8];
        self.pos += 8;
        f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
    }
    fn read_string(&mut self) -> String {
        let len = self.read_u64() as usize;
        let s = String::from_utf8_lossy(&self.data[self.pos..self.pos + len]).to_string();
        self.pos += len;
        s
    }
    fn read_bool(&mut self) -> bool {
        self.read_u8() != 0
    }

    fn read_value(&mut self, vtype: u32) -> MetaValue {
        match vtype {
            0 => MetaValue::U8(self.read_u8()),
            1 => MetaValue::I8(self.read_u8() as i8),
            2 => MetaValue::U16(self.read_u16()),
            3 => MetaValue::I16(self.read_u16() as i16),
            4 => MetaValue::U32(self.read_u32()),
            5 => MetaValue::I32(self.read_i32()),
            6 => MetaValue::F32(self.read_f32()),
            7 => MetaValue::Bool(self.read_bool()),
            8 => MetaValue::Str(self.read_string()),
            9 => {
                let elem_type = self.read_u32();
                let count = self.read_u64() as usize;
                let mut arr = Vec::with_capacity(count);
                for _ in 0..count {
                    arr.push(self.read_value(elem_type));
                }
                MetaValue::Array(arr)
            }
            10 => MetaValue::U64(self.read_u64()),
            11 => MetaValue::I64(self.read_i64()),
            12 => MetaValue::F64(self.read_f64()),
            _ => panic!("Unknown GGUF value type: {}", vtype),
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

        eprintln!(
            "GGUF v{} — {} tensors, {} metadata entries",
            version, n_tensors, n_kv
        );

        // Parse metadata
        let mut metadata = HashMap::with_capacity(n_kv);
        for _ in 0..n_kv {
            let key = c.read_string();
            let vtype = c.read_u32();
            let value = c.read_value(vtype);
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
        let data_offset = (c.pos + alignment - 1) / alignment * alignment;

        Ok(Self {
            metadata,
            tensors,
            data_offset,
        })
    }

    pub fn get_u32(&self, key: &str, default: u32) -> u32 {
        self.metadata
            .get(key)
            .and_then(|v| v.as_u32())
            .unwrap_or(default)
    }

    pub fn get_f32(&self, key: &str, default: f32) -> f32 {
        self.metadata
            .get(key)
            .and_then(|v| v.as_f32())
            .unwrap_or(default)
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).and_then(|v| v.as_str())
    }
}
