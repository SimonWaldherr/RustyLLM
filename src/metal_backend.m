#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t rows;
    uint32_t cols;
    uint32_t row_bytes;
    uint32_t n_blocks;
    uint32_t rows_per_group;
} RustyQ4KParams;

typedef struct {
    uint32_t rows_a;
    uint32_t rows_b;
    uint32_t cols;
    uint32_t row_bytes;
    uint32_t n_blocks;
    uint32_t rows_per_group;
} RustyQ4KPairParams;

enum {
    RUSTY_MATVEC_ROWS_PER_GROUP = 4,
    RUSTY_MATVEC_THREADS_PER_GROUP = 32 * RUSTY_MATVEC_ROWS_PER_GROUP,
};

typedef struct {
    uint32_t heads;
    uint32_t kv_mul;
    uint32_t head_dim;
    uint32_t value_dim;
    uint32_t key_stride;
    uint32_t value_stride;
    uint32_t slot_count;
    uint32_t start_t;
    uint32_t end_t;
    uint32_t use_sink;
    float scale;
} RustyAttentionParams;

typedef struct {
    uint32_t heads;
    uint32_t kv_mul;
    uint32_t head_dim;
    uint32_t value_dim;
    uint32_t key_stride;
    uint32_t value_stride;
    uint32_t start_t;
    uint32_t end_t;
    float scale;
} RustyResidentAttentionParams;

typedef struct {
    uint32_t pos;
    uint32_t head_dim;
    uint32_t half_dim;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t value_dim;
    uint32_t kv_k_dim;
    uint32_t kv_v_dim;
    uint32_t slot;
    uint32_t neox;
} RustyRopeParams;

typedef struct {
    uint32_t len;
} RustyUnaryParams;

typedef struct {
    uint32_t len;
    float eps;
} RustyResidualNormParams;

typedef struct {
    const void *key;
    uintptr_t len;
    __strong id<MTLBuffer> buffer;
} RustyWeightCacheEntry;

enum {
    RUSTY_WEIGHT_CACHE_SIZE = 8192,
};

static id<MTLDevice> gDevice;
static id<MTLCommandQueue> gQueue;
static id<MTLComputePipelineState> gQ4KPipeline;
static id<MTLComputePipelineState> gQ4KPairPipeline;
static id<MTLComputePipelineState> gQ6KPipeline;
static id<MTLComputePipelineState> gQ4_0Pipeline;
static id<MTLComputePipelineState> gQ8_0Pipeline;
static id<MTLComputePipelineState> gAttentionPipeline;
static id<MTLComputePipelineState> gResidentAttentionPipeline;
static id<MTLComputePipelineState> gResidentParallelAttentionPipeline;
static id<MTLComputePipelineState> gResidentGroupedAttentionPipeline;
static id<MTLComputePipelineState> gSiluMulPipeline;
static id<MTLComputePipelineState> gGeluMulPipeline;
static id<MTLComputePipelineState> gResidualRmsPipeline;
static id<MTLComputePipelineState> gResidualAddPipeline;
static id<MTLComputePipelineState> gRopeStorePipeline;
static NSMutableDictionary<NSNumber *, id<MTLBuffer>> *gWeightBuffers;
static NSMutableDictionary<NSNumber *, id<MTLBuffer>> *gSharedBuffers;
static RustyWeightCacheEntry gWeightCache[RUSTY_WEIGHT_CACHE_SIZE];
static id<MTLBuffer> gAttentionZeroBuffer;
static const float gAttentionZero = 0.0f;
static uint64_t gMetalCommandBuffers;
static uint64_t gMetalDispatches;
static uint64_t gMetalCpuToGpuBytes;
static uint64_t gMetalGpuToCpuBytes;
static uint64_t gMetalBufferAllocations;
static uint64_t gMetalTemporaryAllocations;
static double gMetalCpuEncodeSeconds;
static double gMetalGpuSeconds;

static BOOL rusty_metal_private_weights_enabled(void);
static BOOL rusty_metal_profile_enabled(void);
static double rusty_metal_now_seconds(void);
static void rusty_metal_profile_command_buffer(id<MTLCommandBuffer> command_buffer,
                                               double encode_start,
                                               double encode_end);
static void rusty_metal_profile_dump(void);

static void rusty_metal_log_error(const char *step, NSError *error) {
    if (!getenv("RUSTY_LLM_METAL_DEBUG")) return;
    if (error) {
        fprintf(stderr, "RustyLLM Metal init failed at %s: %s\n", step, [[error localizedDescription] UTF8String]);
    } else {
        fprintf(stderr, "RustyLLM Metal init failed at %s\n", step);
    }
}

static NSString *const kQ4KSource =
@"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct Params { uint rows; uint cols; uint row_bytes; uint n_blocks; uint rows_per_group; };\n"
"inline uchar2 scale_min_k4(uint j, const device uchar* q) {\n"
"    if (j < 4) return uchar2(q[j] & 63, q[j + 4] & 63);\n"
"    return uchar2((q[j + 4] & 15) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4));\n"
"}\n"
"// Four independent eight-lane teams process four quant blocks at once.\n"
"// A lane loads four adjacent activations and reuses them for two rows.\n"
"kernel void q4k_matvec(device const uchar* weights [[buffer(0)]],\n"
"                       device const float* x [[buffer(1)]],\n"
"                       device float* out [[buffer(2)]],\n"
"                       constant Params& p [[buffer(3)]],\n"
"                       uint group [[threadgroup_position_in_grid]],\n"
"                       uint sg [[simdgroup_index_in_threadgroup]],\n"
"                       uint lane [[thread_index_in_simdgroup]]) {\n"
"    uint first_row = group * p.rows_per_group + sg * 2;\n"
"    uint block_team = lane >> 3;\n"
"    uint value_base = (lane & 7) * 4;\n"
"    float2 lane_sum = float2(0.0f);\n"
"    for (uint b = block_team; b < p.n_blocks; b += 4) {\n"
"        float4 inputs_lo[4];\n"
"        float4 inputs_hi[4];\n"
"        #pragma unroll\n"
"        for (uint segment = 0; segment < 4; ++segment) {\n"
"            uint input_base = b * 256 + segment * 64 + value_base;\n"
"            inputs_lo[segment] = *reinterpret_cast<const device float4*>(x + input_base);\n"
"            inputs_hi[segment] = *reinterpret_cast<const device float4*>(x + input_base + 32);\n"
"        }\n"
"        #pragma unroll\n"
"        for (uint row_slot = 0; row_slot < 2; ++row_slot) {\n"
"            uint row = first_row + row_slot;\n"
"            if (row >= p.rows) continue;\n"
"            const device uchar* block = weights + row * p.row_bytes + b * 144;\n"
"            const device half* multipliers = reinterpret_cast<const device half*>(block);\n"
"            const device uchar* packed_scales = block + 4;\n"
"            const device uchar* quants = block + 16;\n"
"            float weighted = 0.0f;\n"
"            float offsets = 0.0f;\n"
"            #pragma unroll\n"
"            for (uint segment = 0; segment < 4; ++segment) {\n"
"                uint low_group = segment * 2;\n"
"                uint quant_base = segment * 32 + value_base;\n"
"                ushort2 packed = *reinterpret_cast<const device ushort2*>(quants + quant_base);\n"
"                uchar2 sm_lo = scale_min_k4(low_group, packed_scales);\n"
"                uchar2 sm_hi = scale_min_k4(low_group + 1, packed_scales);\n"
"                float4 lo = inputs_lo[segment];\n"
"                float4 hi = inputs_hi[segment];\n"
"                float dot_lo = lo.x * float(packed.x & 0x000f) +\n"
"                               lo.y * float(packed.x & 0x0f00) * (1.0f / 256.0f) +\n"
"                               lo.z * float(packed.y & 0x000f) +\n"
"                               lo.w * float(packed.y & 0x0f00) * (1.0f / 256.0f);\n"
"                float dot_hi = hi.x * float(packed.x & 0x00f0) * (1.0f / 16.0f) +\n"
"                               hi.y * float(packed.x & 0xf000) * (1.0f / 4096.0f) +\n"
"                               hi.z * float(packed.y & 0x00f0) * (1.0f / 16.0f) +\n"
"                               hi.w * float(packed.y & 0xf000) * (1.0f / 4096.0f);\n"
"                weighted += float(sm_lo.x) * dot_lo + float(sm_hi.x) * dot_hi;\n"
"                offsets += float(sm_lo.y) * (lo.x + lo.y + lo.z + lo.w) +\n"
"                           float(sm_hi.y) * (hi.x + hi.y + hi.z + hi.w);\n"
"            }\n"
"            lane_sum[row_slot] += float(multipliers[0]) * weighted -\n"
"                                  float(multipliers[1]) * offsets;\n"
"        }\n"
"    }\n"
"    #pragma unroll\n"
"    for (uint row_slot = 0; row_slot < 2; ++row_slot) {\n"
"        float total = simd_sum(lane_sum[row_slot]);\n"
"        uint row = first_row + row_slot;\n"
"        if (lane == 0 && row < p.rows) out[row] = total;\n"
"    }\n"
"}\n"
"struct PairParams { uint rows_a; uint rows_b; uint cols; uint row_bytes; uint n_blocks; uint rows_per_group; };\n"
"// A single grid covers two matrices that share one activation vector.\n"
"kernel void q4k_matvec_pair(device const uchar* weights_a [[buffer(0)]],\n"
"                            device const uchar* weights_b [[buffer(1)]],\n"
"                            device const float* x [[buffer(2)]],\n"
"                            device float* out_a [[buffer(3)]],\n"
"                            device float* out_b [[buffer(4)]],\n"
"                            constant PairParams& pp [[buffer(5)]],\n"
"                            uint group [[threadgroup_position_in_grid]],\n"
"                            uint sg [[simdgroup_index_in_threadgroup]],\n"
"                            uint lane [[thread_index_in_simdgroup]]) {\n"
"    uint groups_a = (pp.rows_a + pp.rows_per_group - 1) / pp.rows_per_group;\n"
"    bool second = group >= groups_a;\n"
"    uint local_group = second ? group - groups_a : group;\n"
"    uint rows = second ? pp.rows_b : pp.rows_a;\n"
"    const device uchar* weights = second ? weights_b : weights_a;\n"
"    device float* out = second ? out_b : out_a;\n"
"    uint first_row = local_group * pp.rows_per_group + sg * 2;\n"
"    uint block_team = lane >> 3;\n"
"    uint value_base = (lane & 7) * 4;\n"
"    float2 lane_sum = float2(0.0f);\n"
"    for (uint b = block_team; b < pp.n_blocks; b += 4) {\n"
"        float4 inputs_lo[4];\n"
"        float4 inputs_hi[4];\n"
"        #pragma unroll\n"
"        for (uint segment = 0; segment < 4; ++segment) {\n"
"            uint input_base = b * 256 + segment * 64 + value_base;\n"
"            inputs_lo[segment] = *reinterpret_cast<const device float4*>(x + input_base);\n"
"            inputs_hi[segment] = *reinterpret_cast<const device float4*>(x + input_base + 32);\n"
"        }\n"
"        #pragma unroll\n"
"        for (uint row_slot = 0; row_slot < 2; ++row_slot) {\n"
"            uint row = first_row + row_slot;\n"
"            if (row >= rows) continue;\n"
"            const device uchar* block = weights + row * pp.row_bytes + b * 144;\n"
"            const device half* multipliers = reinterpret_cast<const device half*>(block);\n"
"            const device uchar* packed_scales = block + 4;\n"
"            const device uchar* quants = block + 16;\n"
"            float weighted = 0.0f;\n"
"            float offsets = 0.0f;\n"
"            #pragma unroll\n"
"            for (uint segment = 0; segment < 4; ++segment) {\n"
"                uint low_group = segment * 2;\n"
"                uint quant_base = segment * 32 + value_base;\n"
"                ushort2 packed = *reinterpret_cast<const device ushort2*>(quants + quant_base);\n"
"                uchar2 sm_lo = scale_min_k4(low_group, packed_scales);\n"
"                uchar2 sm_hi = scale_min_k4(low_group + 1, packed_scales);\n"
"                float4 lo = inputs_lo[segment];\n"
"                float4 hi = inputs_hi[segment];\n"
"                float dot_lo = lo.x * float(packed.x & 0x000f) +\n"
"                               lo.y * float(packed.x & 0x0f00) * (1.0f / 256.0f) +\n"
"                               lo.z * float(packed.y & 0x000f) +\n"
"                               lo.w * float(packed.y & 0x0f00) * (1.0f / 256.0f);\n"
"                float dot_hi = hi.x * float(packed.x & 0x00f0) * (1.0f / 16.0f) +\n"
"                               hi.y * float(packed.x & 0xf000) * (1.0f / 4096.0f) +\n"
"                               hi.z * float(packed.y & 0x00f0) * (1.0f / 16.0f) +\n"
"                               hi.w * float(packed.y & 0xf000) * (1.0f / 4096.0f);\n"
"                weighted += float(sm_lo.x) * dot_lo + float(sm_hi.x) * dot_hi;\n"
"                offsets += float(sm_lo.y) * (lo.x + lo.y + lo.z + lo.w) +\n"
"                           float(sm_hi.y) * (hi.x + hi.y + hi.z + hi.w);\n"
"            }\n"
"            lane_sum[row_slot] += float(multipliers[0]) * weighted -\n"
"                                  float(multipliers[1]) * offsets;\n"
"        }\n"
"    }\n"
"    #pragma unroll\n"
"    for (uint row_slot = 0; row_slot < 2; ++row_slot) {\n"
"        float total = simd_sum(lane_sum[row_slot]);\n"
"        uint row = first_row + row_slot;\n"
"        if (lane == 0 && row < rows) out[row] = total;\n"
"    }\n"
"}\n";

static NSString *const kQ6KSource =
@"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct Params { uint rows; uint cols; uint row_bytes; uint n_blocks; uint rows_per_group; };\n"
"// Four independent eight-lane teams process four quant blocks at once.\n"
"// Each lane rebuilds four adjacent values and applies them to two rows.\n"
"kernel void q6k_matvec(device const uchar* weights [[buffer(0)]],\n"
"                       device const float* x [[buffer(1)]],\n"
"                       device float* out [[buffer(2)]],\n"
"                       constant Params& p [[buffer(3)]],\n"
"                       uint group [[threadgroup_position_in_grid]],\n"
"                       uint sg [[simdgroup_index_in_threadgroup]],\n"
"                       uint lane [[thread_index_in_simdgroup]]) {\n"
"    uint first_row = group * p.rows_per_group + sg * 2;\n"
"    uint block_team = lane >> 3;\n"
"    uint value_base = (lane & 7) * 4;\n"
"    float2 lane_sum = float2(0.0f);\n"
"    for (uint b = block_team; b < p.n_blocks; b += 4) {\n"
"        float4 inputs[8];\n"
"        #pragma unroll\n"
"        for (uint quant_group = 0; quant_group < 8; ++quant_group) {\n"
"            uint input_base = b * 256 + quant_group * 32 + value_base;\n"
"            inputs[quant_group] = *reinterpret_cast<const device float4*>(x + input_base);\n"
"        }\n"
"        #pragma unroll\n"
"        for (uint row_slot = 0; row_slot < 2; ++row_slot) {\n"
"            uint row = first_row + row_slot;\n"
"            if (row >= p.rows) continue;\n"
"            const device uchar* block = weights + row * p.row_bytes + b * 210;\n"
"            const device uchar* low_bits = block;\n"
"            const device uchar* high_bits = block + 128;\n"
"            const device char* scales = reinterpret_cast<const device char*>(block + 192);\n"
"            float multiplier = float(*reinterpret_cast<const device half*>(block + 208));\n"
"            float weighted = 0.0f;\n"
"            #pragma unroll\n"
"            for (uint half_block = 0; half_block < 2; ++half_block) {\n"
"                uint low_base = half_block * 64 + value_base;\n"
"                uint high_base = half_block * 32 + value_base;\n"
"                uint scale_base = half_block * 8 + uint(value_base >= 16);\n"
"                uint input_group = half_block * 4;\n"
"                float4 x0 = inputs[input_group];\n"
"                float4 x1 = inputs[input_group + 1];\n"
"                float4 x2 = inputs[input_group + 2];\n"
"                float4 x3 = inputs[input_group + 3];\n"
"                uchar4 low_a = *reinterpret_cast<const device uchar4*>(low_bits + low_base);\n"
"                uchar4 low_b = *reinterpret_cast<const device uchar4*>(low_bits + low_base + 32);\n"
"                uchar4 high = *reinterpret_cast<const device uchar4*>(high_bits + high_base);\n"
"                int4 q0 = int4((low_a & 15) | ((high & 3) << 4)) - 32;\n"
"                int4 q1 = int4((low_b & 15) | (((high >> 2) & 3) << 4)) - 32;\n"
"                int4 q2 = int4((low_a >> 4) | (((high >> 4) & 3) << 4)) - 32;\n"
"                int4 q3 = int4((low_b >> 4) | ((high >> 6) << 4)) - 32;\n"
"                float dot0 = dot(float4(q0), x0);\n"
"                float dot1 = dot(float4(q1), x1);\n"
"                float dot2 = dot(float4(q2), x2);\n"
"                float dot3 = dot(float4(q3), x3);\n"
"                weighted += float(scales[scale_base]) * dot0 +\n"
"                            float(scales[scale_base + 2]) * dot1 +\n"
"                            float(scales[scale_base + 4]) * dot2 +\n"
"                            float(scales[scale_base + 6]) * dot3;\n"
"            }\n"
"            lane_sum[row_slot] += multiplier * weighted;\n"
"        }\n"
"    }\n"
"    #pragma unroll\n"
"    for (uint row_slot = 0; row_slot < 2; ++row_slot) {\n"
"        float total = simd_sum(lane_sum[row_slot]);\n"
"        uint row = first_row + row_slot;\n"
"        if (lane == 0 && row < p.rows) out[row] = total;\n"
"    }\n"
"}\n";

static NSString *const kQ4_0Source =
@"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct Params { uint rows; uint cols; uint row_bytes; uint n_blocks; uint rows_per_group; };\n"
"kernel void q4_0_matvec(device const uchar* weights [[buffer(0)]],\n"
"                        device const float* x [[buffer(1)]],\n"
"                        device float* out [[buffer(2)]],\n"
"                        constant Params& p [[buffer(3)]],\n"
"                        uint group [[threadgroup_position_in_grid]],\n"
"                        uint sg [[simdgroup_index_in_threadgroup]],\n"
"                        uint lane [[thread_index_in_simdgroup]]) {\n"
"    uint row = group * p.rows_per_group + sg;\n"
"    if (row >= p.rows) return;\n"
"    const device uchar* row_base = weights + row * p.row_bytes;\n"
"    float sum = 0.0f;\n"
"    for (uint b = lane; b < p.n_blocks; b += 32) {\n"
"        const device uchar* block = row_base + b * 18;\n"
"        ushort db = ushort(block[0]) | (ushort(block[1]) << 8);\n"
"        float d = float(as_type<half>(db));\n"
"        const device uchar* q = block + 2;\n"
"        uint xoff = b * 32;\n"
"        #pragma unroll(16)\n"
"        for (uint i = 0; i < 16; ++i) {\n"
"            uchar byte = q[i];\n"
"            float q0 = float(int(byte & 15) - 8);\n"
"            float q1 = float(int(byte >> 4) - 8);\n"
"            sum += d * q0 * x[xoff + i];\n"
"            sum += d * q1 * x[xoff + 16 + i];\n"
"        }\n"
"    }\n"
"    for (ushort offset = 16; offset > 0; offset >>= 1)\n"
"        sum += simd_shuffle_xor(sum, offset);\n"
"    if (lane == 0) out[row] = sum;\n"
"}\n";

static NSString *const kQ8_0Source =
@"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct Params { uint rows; uint cols; uint row_bytes; uint n_blocks; uint rows_per_group; };\n"
"kernel void q8_0_matvec(device const uchar* weights [[buffer(0)]],\n"
"                        device const float* x [[buffer(1)]],\n"
"                        device float* out [[buffer(2)]],\n"
"                        constant Params& p [[buffer(3)]],\n"
"                        uint group [[threadgroup_position_in_grid]],\n"
"                        uint sg [[simdgroup_index_in_threadgroup]],\n"
"                        uint lane [[thread_index_in_simdgroup]]) {\n"
"    uint row = group * p.rows_per_group + sg;\n"
"    if (row >= p.rows) return;\n"
"    const device uchar* row_base = weights + row * p.row_bytes;\n"
"    float sum = 0.0f;\n"
"    for (uint b = lane; b < p.n_blocks; b += 32) {\n"
"        const device uchar* block = row_base + b * 34;\n"
"        ushort db = ushort(block[0]) | (ushort(block[1]) << 8);\n"
"        float d = float(as_type<half>(db));\n"
"        const device char* q = (const device char*)(block + 2);\n"
"        uint xoff = b * 32;\n"
"        #pragma unroll(32)\n"
"        for (uint i = 0; i < 32; ++i) {\n"
"            sum += d * float(q[i]) * x[xoff + i];\n"
"        }\n"
"    }\n"
"    for (ushort offset = 16; offset > 0; offset >>= 1)\n"
"        sum += simd_shuffle_xor(sum, offset);\n"
"    if (lane == 0) out[row] = sum;\n"
"}\n";

static NSString *const kAttentionSource =
@"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct Params { uint heads; uint kv_mul; uint head_dim; uint value_dim; uint key_stride; uint value_stride; uint slot_count; uint start_t; uint end_t; uint use_sink; float scale; };\n"
"kernel void attention_scan(device const float* query [[buffer(0)]],\n"
"                           device const float* keys [[buffer(1)]],\n"
"                           device const float* values [[buffer(2)]],\n"
"                           device float* out [[buffer(3)]],\n"
"                           device const float* sinks [[buffer(4)]],\n"
"                           constant Params& p [[buffer(5)]],\n"
"                           uint head [[threadgroup_position_in_grid]],\n"
"                           uint lane [[thread_index_in_simdgroup]]) {\n"
"    constexpr uint MAX_LANE_VALUES = 8;\n"
"    if (head >= p.heads || p.head_dim > 256 || p.value_dim > 256 || p.slot_count == 0) return;\n"
"    const device float* q_row = query + head * p.head_dim;\n"
"    device float* out_row = out + head * p.value_dim;\n"
"    uint kv_head = head / p.kv_mul;\n"
"    float qreg[MAX_LANE_VALUES];\n"
"    float oreg[MAX_LANE_VALUES];\n"
"    #pragma unroll\n"
"    for (uint j = 0; j < MAX_LANE_VALUES; ++j) {\n"
"        uint i = lane + 32 * j;\n"
"        qreg[j] = i < p.head_dim ? q_row[i] : 0.0f;\n"
"        oreg[j] = 0.0f;\n"
"    }\n"
"    float max_score = -INFINITY;\n"
"    float denom = 0.0f;\n"
"    if (lane == 0 && p.use_sink != 0) {\n"
"        max_score = sinks[head];\n"
"        denom = 1.0f;\n"
"    }\n"
"    uint count = p.end_t >= p.start_t ? p.end_t - p.start_t + 1 : 0;\n"
"    uint slot = p.start_t % p.slot_count;\n"
"    for (uint n = 0; n < count; ++n) {\n"
"        const device float* k_row = keys + slot * p.key_stride + kv_head * p.head_dim;\n"
"        const device float* v_row = values + slot * p.value_stride + kv_head * p.value_dim;\n"
"        float partial = 0.0f;\n"
"        #pragma unroll\n"
"        for (uint j = 0; j < MAX_LANE_VALUES; ++j) {\n"
"            uint i = lane + 32 * j;\n"
"            if (i < p.head_dim) partial = fma(qreg[j], k_row[i], partial);\n"
"        }\n"
"        for (ushort offset = 16; offset > 0; offset >>= 1) partial += simd_shuffle_xor(partial, offset);\n"
"        float acc_scale = 1.0f;\n"
"        float value_scale = 0.0f;\n"
"        if (lane == 0) {\n"
"            float score = partial * p.scale;\n"
"            if (score > max_score) {\n"
"                acc_scale = isfinite(max_score) ? exp(max_score - score) : 0.0f;\n"
"                value_scale = 1.0f;\n"
"                denom = denom * acc_scale + 1.0f;\n"
"                max_score = score;\n"
"            } else {\n"
"                value_scale = exp(score - max_score);\n"
"                denom += value_scale;\n"
"            }\n"
"        }\n"
"        acc_scale = simd_broadcast_first(acc_scale);\n"
"        value_scale = simd_broadcast_first(value_scale);\n"
"        #pragma unroll\n"
"        for (uint j = 0; j < MAX_LANE_VALUES; ++j) {\n"
"            uint i = lane + 32 * j;\n"
"            if (i < p.value_dim) oreg[j] = fma(value_scale, v_row[i], oreg[j] * acc_scale);\n"
"        }\n"
"        ++slot;\n"
"        if (slot == p.slot_count) slot = 0;\n"
"    }\n"
"    float inv_denom = lane == 0 && denom > 0.0f ? 1.0f / denom : 0.0f;\n"
"    inv_denom = simd_broadcast_first(inv_denom);\n"
"    #pragma unroll\n"
"    for (uint j = 0; j < MAX_LANE_VALUES; ++j) {\n"
"        uint i = lane + 32 * j;\n"
"        if (i < p.value_dim) out_row[i] = oreg[j] * inv_denom;\n"
"    }\n"
"}\n";

static NSString *const kSiluMulSource =
@"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct Params { uint len; };\n"
"kernel void silu_mul(device const float* gate [[buffer(0)]],\n"
"                     device const float* up [[buffer(1)]],\n"
"                     device float* out [[buffer(2)]],\n"
"                     constant Params& p [[buffer(3)]],\n"
"                     uint gid [[thread_position_in_grid]]) {\n"
"    if (gid >= p.len) return;\n"
"    float g = gate[gid];\n"
"    out[gid] = (g / (1.0f + exp(-g))) * up[gid];\n"
"}\n";

static NSString *const kResidualSource =
@"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct NormParams { uint len; float eps; };\n"
"struct AddParams { uint len; };\n"
"kernel void residual_rms(device float* x [[buffer(0)]],\n"
"                         device const float* residual [[buffer(1)]],\n"
"                         device const float* weight [[buffer(2)]],\n"
"                         device float* out [[buffer(3)]],\n"
"                         constant NormParams& p [[buffer(4)]],\n"
"                         uint tid [[thread_index_in_threadgroup]]) {\n"
"    threadgroup float partial[256];\n"
"    float sum = 0.0f;\n"
"    for (uint i = tid; i < p.len; i += 256) {\n"
"        float v = x[i] + residual[i];\n"
"        x[i] = v;\n"
"        sum += v * v;\n"
"    }\n"
"    partial[tid] = sum;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = 128; stride > 0; stride >>= 1) {\n"
"        if (tid < stride) partial[tid] += partial[tid + stride];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    float scale = rsqrt(partial[0] / float(p.len) + p.eps);\n"
"    for (uint i = tid; i < p.len; i += 256) out[i] = x[i] * weight[i] * scale;\n"
"}\n"
"kernel void residual_add(device float* x [[buffer(0)]],\n"
"                         device const float* residual [[buffer(1)]],\n"
"                         constant AddParams& p [[buffer(2)]],\n"
"                         uint gid [[thread_position_in_grid]]) {\n"
"    if (gid >= p.len) return;\n"
"    x[gid] += residual[gid];\n"
"}\n";

// RoPE applied in-place to q and k, with rotated k and copied v written straight
// into the GPU-resident KV cache slot for this position. Handles both the
// interleaved (pairs i,i+1) and NeoX (pairs i,i+half) rotation conventions and
// grouped-query attention (n_kv_heads <= n_heads). Used by the resident decoder.
static NSString *const kRopeStoreSource =
@"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct RopeParams { uint pos; uint head_dim; uint half_dim; uint n_heads; uint n_kv_heads;\n"
"                    uint value_dim; uint kv_k_dim; uint kv_v_dim; uint slot; uint neox; };\n"
"kernel void rope_store(device float* q [[buffer(0)]],\n"
"                       device float* k [[buffer(1)]],\n"
"                       device const float* v [[buffer(2)]],\n"
"                       device const float* inv_freq [[buffer(3)]],\n"
"                       device float* k_cache [[buffer(4)]],\n"
"                       device float* v_cache [[buffer(5)]],\n"
"                       constant RopeParams& p [[buffer(6)]],\n"
"                       uint gid [[thread_position_in_grid]]) {\n"
"    uint pairs_per_head = p.half_dim;\n"
"    uint total = p.n_heads * pairs_per_head;\n"
"    if (gid < total) {\n"
"        uint h = gid / pairs_per_head;\n"
"        uint i = gid % pairs_per_head;\n"
"        float angle = float(p.pos) * inv_freq[i];\n"
"        float ca = cos(angle);\n"
"        float sa = sin(angle);\n"
"        uint off = h * p.head_dim;\n"
"        uint i0 = p.neox != 0 ? off + i : off + 2 * i;\n"
"        uint i1 = p.neox != 0 ? off + i + p.half_dim : off + 2 * i + 1;\n"
"        float v0 = q[i0];\n"
"        float v1 = q[i1];\n"
"        q[i0] = v0 * ca - v1 * sa;\n"
"        q[i1] = v0 * sa + v1 * ca;\n"
"        if (h < p.n_kv_heads) {\n"
"            float w0 = k[i0];\n"
"            float w1 = k[i1];\n"
"            float r0 = w0 * ca - w1 * sa;\n"
"            float r1 = w0 * sa + w1 * ca;\n"
"            k[i0] = r0;\n"
"            k[i1] = r1;\n"
"            k_cache[p.slot * p.kv_k_dim + i0] = r0;\n"
"            k_cache[p.slot * p.kv_k_dim + i1] = r1;\n"
"        }\n"
"    }\n"
"    // Copy V (unrotated) into the cache slot; one thread per element.\n"
"    if (gid < p.kv_v_dim) {\n"
"        v_cache[p.slot * p.kv_v_dim + gid] = v[gid];\n"
"    }\n"
"}\n";

// Correct GQA attention over a slot-major KV cache
// (k_cache[t*kv_k_dim + kv_head*head_dim + d], v_cache[t*kv_v_dim + kv_head*value_dim + d]),
// one query head per threadgroup, 32-lane online softmax. Used by the resident decoder.
static NSString *const kResidentAttnSource =
@"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct AttnParams { uint n_heads; uint kv_mul; uint head_dim; uint value_dim;\n"
"                    uint kv_k_dim; uint kv_v_dim; uint start_t; uint end_t; float scale; };\n"
"kernel void resident_attention(device const float* q [[buffer(0)]],\n"
"                               device const float* k_cache [[buffer(1)]],\n"
"                               device const float* v_cache [[buffer(2)]],\n"
"                               device float* out [[buffer(3)]],\n"
"                               constant AttnParams& p [[buffer(4)]],\n"
"                               uint head [[threadgroup_position_in_grid]],\n"
"                               uint lane [[thread_index_in_simdgroup]]) {\n"
"    if (head >= p.n_heads) return;\n"
"    threadgroup float qsh[256];\n"
"    threadgroup float osh[256];\n"
"    const device float* qrow = q + head * p.head_dim;\n"
"    uint kv_head = head / p.kv_mul;\n"
"    for (uint i = lane; i < p.head_dim; i += 32) qsh[i] = qrow[i];\n"
"    for (uint i = lane; i < p.value_dim; i += 32) osh[i] = 0.0f;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float maxs = -INFINITY;\n"
"    float denom = 0.0f;\n"
"    for (uint t = p.start_t; t <= p.end_t; ++t) {\n"
"        const device float* krow = k_cache + t * p.kv_k_dim + kv_head * p.head_dim;\n"
"        float partial = 0.0f;\n"
"        for (uint i = lane; i < p.head_dim; i += 32) partial += qsh[i] * krow[i];\n"
"        for (ushort o = 16; o > 0; o >>= 1) partial += simd_shuffle_xor(partial, o);\n"
"        float score = simd_broadcast_first(partial) * p.scale;\n"
"        const device float* vrow = v_cache + t * p.kv_v_dim + kv_head * p.value_dim;\n"
"        if (score > maxs) {\n"
"            float r = isfinite(maxs) ? exp(maxs - score) : 0.0f;\n"
"            denom = denom * r + 1.0f;\n"
"            for (uint i = lane; i < p.value_dim; i += 32) osh[i] = osh[i] * r + vrow[i];\n"
"            maxs = score;\n"
"        } else {\n"
"            float w = exp(score - maxs);\n"
"            denom += w;\n"
"            for (uint i = lane; i < p.value_dim; i += 32) osh[i] += w * vrow[i];\n"
"        }\n"
"    }\n"
"    float inv = denom > 0.0f ? 1.0f / denom : 0.0f;\n"
"    device float* orow = out + head * p.value_dim;\n"
"    for (uint i = lane; i < p.value_dim; i += 32) orow[i] = osh[i] * inv;\n"
"}\n";

// Short-context resident attention parallelized across four SIMD groups per
// query head. Each SIMD group scans a disjoint subset of KV positions with an
// online softmax; the four partial softmax states are merged exactly at the end.
// This trades additional query-head KV reads for 4x more temporal parallelism,
// which is profitable before the cache scan becomes memory-bandwidth bound.
static NSString *const kResidentParallelAttnSource =
@"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct AttnParams { uint n_heads; uint kv_mul; uint head_dim; uint value_dim;\n"
"                    uint kv_k_dim; uint kv_v_dim; uint start_t; uint end_t; float scale; };\n"
"kernel void resident_parallel_attention(device const float* q [[buffer(0)]],\n"
"                                        device const float* k_cache [[buffer(1)]],\n"
"                                        device const float* v_cache [[buffer(2)]],\n"
"                                        device float* out [[buffer(3)]],\n"
"                                        constant AttnParams& p [[buffer(4)]],\n"
"                                        uint head [[threadgroup_position_in_grid]],\n"
"                                        uint sg [[simdgroup_index_in_threadgroup]],\n"
"                                        uint lane [[thread_index_in_simdgroup]]) {\n"
"    constexpr uint nsg = 4;\n"
"    if (head >= p.n_heads) return;\n"
"    threadgroup float qsh[256];\n"
"    threadgroup float osh[nsg][256];\n"
"    threadgroup float maxsh[nsg];\n"
"    threadgroup float densh[nsg];\n"
"    threadgroup float merged_max;\n"
"    threadgroup float merged_den;\n"
"    uint local = sg * 32 + lane;\n"
"    const device float* qrow = q + head * p.head_dim;\n"
"    uint kv_head = head / p.kv_mul;\n"
"    for (uint i = local; i < p.head_dim; i += 128) qsh[i] = qrow[i];\n"
"    for (uint i = lane; i < p.value_dim; i += 32) osh[sg][i] = 0.0f;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float maxs = -INFINITY;\n"
"    float denom = 0.0f;\n"
"    for (uint t = p.start_t + sg; t <= p.end_t; t += nsg) {\n"
"        const device float* krow = k_cache + t * p.kv_k_dim + kv_head * p.head_dim;\n"
"        float partial = 0.0f;\n"
"        for (uint i = lane; i < p.head_dim; i += 32) partial += qsh[i] * krow[i];\n"
"        float score = simd_sum(partial) * p.scale;\n"
"        const device float* vrow = v_cache + t * p.kv_v_dim + kv_head * p.value_dim;\n"
"        if (score > maxs) {\n"
"            float r = isfinite(maxs) ? exp(maxs - score) : 0.0f;\n"
"            denom = denom * r + 1.0f;\n"
"            for (uint i = lane; i < p.value_dim; i += 32) osh[sg][i] = osh[sg][i] * r + vrow[i];\n"
"            maxs = score;\n"
"        } else {\n"
"            float w = exp(score - maxs);\n"
"            denom += w;\n"
"            for (uint i = lane; i < p.value_dim; i += 32) osh[sg][i] += w * vrow[i];\n"
"        }\n"
"    }\n"
"    if (lane == 0) { maxsh[sg] = maxs; densh[sg] = denom; }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (local == 0) {\n"
"        float m = max(max(maxsh[0], maxsh[1]), max(maxsh[2], maxsh[3]));\n"
"        float d = 0.0f;\n"
"        for (uint j = 0; j < nsg; ++j) d += densh[j] * exp(maxsh[j] - m);\n"
"        merged_max = m;\n"
"        merged_den = d;\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (sg == 0) {\n"
"        device float* orow = out + head * p.value_dim;\n"
"        float inv = merged_den > 0.0f ? 1.0f / merged_den : 0.0f;\n"
"        for (uint i = lane; i < p.value_dim; i += 32) {\n"
"            float total = 0.0f;\n"
"            for (uint j = 0; j < nsg; ++j) total += osh[j][i] * exp(maxsh[j] - merged_max);\n"
"            orow[i] = total * inv;\n"
"        }\n"
"    }\n"
"}\n";

// Four-head grouped-query attention for the resident decoder. Ministral 3
// maps four query heads to each KV head, so loading the KV row once into
// threadgroup memory avoids streaming the same keys and values four times.
static NSString *const kResidentGroupedAttnSource =
@"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct AttnParams { uint n_heads; uint kv_mul; uint head_dim; uint value_dim;\n"
"                    uint kv_k_dim; uint kv_v_dim; uint start_t; uint end_t; float scale; };\n"
"kernel void resident_gqa4_attention(device const float* q [[buffer(0)]],\n"
"                                    device const float* k_cache [[buffer(1)]],\n"
"                                    device const float* v_cache [[buffer(2)]],\n"
"                                    device float* out [[buffer(3)]],\n"
"                                    constant AttnParams& p [[buffer(4)]],\n"
"                                    uint kv_head [[threadgroup_position_in_grid]],\n"
"                                    uint sg [[simdgroup_index_in_threadgroup]],\n"
"                                    uint lane [[thread_index_in_simdgroup]]) {\n"
"    threadgroup float qsh[4][256];\n"
"    threadgroup float osh[4][256];\n"
"    threadgroup float ksh[256];\n"
"    threadgroup float vsh[256];\n"
"    uint head = kv_head * 4 + sg;\n"
"    const device float* qrow = q + head * p.head_dim;\n"
"    for (uint i = lane; i < p.head_dim; i += 32) qsh[sg][i] = qrow[i];\n"
"    for (uint i = lane; i < p.value_dim; i += 32) osh[sg][i] = 0.0f;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float maxs = -INFINITY;\n"
"    float denom = 0.0f;\n"
"    uint local = sg * 32 + lane;\n"
"    for (uint t = p.start_t; t <= p.end_t; ++t) {\n"
"        const device float* krow = k_cache + t * p.kv_k_dim + kv_head * p.head_dim;\n"
"        const device float* vrow = v_cache + t * p.kv_v_dim + kv_head * p.value_dim;\n"
"        for (uint i = local; i < p.head_dim; i += 128) ksh[i] = krow[i];\n"
"        for (uint i = local; i < p.value_dim; i += 128) vsh[i] = vrow[i];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        float partial = 0.0f;\n"
"        for (uint i = lane; i < p.head_dim; i += 32) partial += qsh[sg][i] * ksh[i];\n"
"        for (ushort o = 16; o > 0; o >>= 1) partial += simd_shuffle_xor(partial, o);\n"
"        float score = simd_broadcast_first(partial) * p.scale;\n"
"        if (score > maxs) {\n"
"            float r = isfinite(maxs) ? exp(maxs - score) : 0.0f;\n"
"            denom = denom * r + 1.0f;\n"
"            for (uint i = lane; i < p.value_dim; i += 32) osh[sg][i] = osh[sg][i] * r + vsh[i];\n"
"            maxs = score;\n"
"        } else {\n"
"            float w = exp(score - maxs);\n"
"            denom += w;\n"
"            for (uint i = lane; i < p.value_dim; i += 32) osh[sg][i] += w * vsh[i];\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    float inv = denom > 0.0f ? 1.0f / denom : 0.0f;\n"
"    device float* orow = out + head * p.value_dim;\n"
"    for (uint i = lane; i < p.value_dim; i += 32) orow[i] = osh[sg][i] * inv;\n"
"}\n";

static BOOL rusty_metal_init(void) {
    static dispatch_once_t once;
    static BOOL ok = NO;
    dispatch_once(&once, ^{
        gDevice = MTLCreateSystemDefaultDevice();
        if (!gDevice) {
            rusty_metal_log_error("MTLCreateSystemDefaultDevice", nil);
            return;
        }
        NSError *error = nil;
        MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
        options.fastMathEnabled = YES;
        id<MTLLibrary> library = [gDevice newLibraryWithSource:kQ4KSource options:options error:&error];
        if (!library) {
            rusty_metal_log_error("compile q4k library", error);
            return;
        }
        id<MTLFunction> function = [library newFunctionWithName:@"q4k_matvec"];
        if (!function) {
            rusty_metal_log_error("load q4k function", nil);
            return;
        }
        gQ4KPipeline = [gDevice newComputePipelineStateWithFunction:function error:&error];
        if (!gQ4KPipeline) {
            rusty_metal_log_error("create q4k pipeline", error);
            return;
        }
        id<MTLFunction> pair_function = [library newFunctionWithName:@"q4k_matvec_pair"];
        if (!pair_function) {
            rusty_metal_log_error("load q4k pair function", nil);
            return;
        }
        gQ4KPairPipeline = [gDevice newComputePipelineStateWithFunction:pair_function error:&error];
        if (!gQ4KPairPipeline) {
            rusty_metal_log_error("create q4k pair pipeline", error);
            return;
        }
        id<MTLLibrary> q6_library = [gDevice newLibraryWithSource:kQ6KSource options:options error:&error];
        if (!q6_library) {
            rusty_metal_log_error("compile q6k library", error);
            return;
        }
        id<MTLFunction> q6_function = [q6_library newFunctionWithName:@"q6k_matvec"];
        if (!q6_function) {
            rusty_metal_log_error("load q6k function", nil);
            return;
        }
        gQ6KPipeline = [gDevice newComputePipelineStateWithFunction:q6_function error:&error];
        if (!gQ6KPipeline) {
            rusty_metal_log_error("create q6k pipeline", error);
            return;
        }
        id<MTLLibrary> q4_0_library = [gDevice newLibraryWithSource:kQ4_0Source options:options error:&error];
        if (!q4_0_library) {
            rusty_metal_log_error("compile q4_0 library", error);
            return;
        }
        id<MTLFunction> q4_0_function = [q4_0_library newFunctionWithName:@"q4_0_matvec"];
        if (!q4_0_function) {
            rusty_metal_log_error("load q4_0 function", nil);
            return;
        }
        gQ4_0Pipeline = [gDevice newComputePipelineStateWithFunction:q4_0_function error:&error];
        if (!gQ4_0Pipeline) {
            rusty_metal_log_error("create q4_0 pipeline", error);
            return;
        }
        id<MTLLibrary> q8_0_library = [gDevice newLibraryWithSource:kQ8_0Source options:options error:&error];
        if (!q8_0_library) {
            rusty_metal_log_error("compile q8_0 library", error);
            return;
        }
        id<MTLFunction> q8_0_function = [q8_0_library newFunctionWithName:@"q8_0_matvec"];
        if (!q8_0_function) {
            rusty_metal_log_error("load q8_0 function", nil);
            return;
        }
        gQ8_0Pipeline = [gDevice newComputePipelineStateWithFunction:q8_0_function error:&error];
        if (!gQ8_0Pipeline) {
            rusty_metal_log_error("create q8_0 pipeline", error);
            return;
        }
        id<MTLLibrary> attention_library = [gDevice newLibraryWithSource:kAttentionSource options:options error:&error];
        if (!attention_library) {
            rusty_metal_log_error("compile attention library", error);
            return;
        }
        id<MTLFunction> attention_function = [attention_library newFunctionWithName:@"attention_scan"];
        if (!attention_function) {
            rusty_metal_log_error("load attention function", nil);
            return;
        }
        gAttentionPipeline = [gDevice newComputePipelineStateWithFunction:attention_function error:&error];
        if (!gAttentionPipeline) {
            rusty_metal_log_error("create attention pipeline", error);
            return;
        }
        id<MTLLibrary> resident_attention_library = [gDevice newLibraryWithSource:kResidentAttnSource options:options error:&error];
        if (!resident_attention_library) {
            rusty_metal_log_error("compile resident attention library", error);
            return;
        }
        id<MTLFunction> resident_attention_function = [resident_attention_library newFunctionWithName:@"resident_attention"];
        if (!resident_attention_function) {
            rusty_metal_log_error("load resident attention function", nil);
            return;
        }
        gResidentAttentionPipeline = [gDevice newComputePipelineStateWithFunction:resident_attention_function error:&error];
        if (!gResidentAttentionPipeline) {
            rusty_metal_log_error("create resident attention pipeline", error);
            return;
        }
        id<MTLLibrary> parallel_attention_library = [gDevice newLibraryWithSource:kResidentParallelAttnSource options:options error:&error];
        if (!parallel_attention_library) {
            rusty_metal_log_error("compile resident parallel attention library", error);
            return;
        }
        id<MTLFunction> parallel_attention_function = [parallel_attention_library newFunctionWithName:@"resident_parallel_attention"];
        if (!parallel_attention_function) {
            rusty_metal_log_error("load resident parallel attention function", nil);
            return;
        }
        gResidentParallelAttentionPipeline = [gDevice newComputePipelineStateWithFunction:parallel_attention_function error:&error];
        if (!gResidentParallelAttentionPipeline) {
            rusty_metal_log_error("create resident parallel attention pipeline", error);
            return;
        }
        id<MTLLibrary> grouped_attention_library = [gDevice newLibraryWithSource:kResidentGroupedAttnSource options:options error:&error];
        if (!grouped_attention_library) {
            rusty_metal_log_error("compile resident grouped attention library", error);
            return;
        }
        id<MTLFunction> grouped_attention_function = [grouped_attention_library newFunctionWithName:@"resident_gqa4_attention"];
        if (!grouped_attention_function) {
            rusty_metal_log_error("load resident grouped attention function", nil);
            return;
        }
        gResidentGroupedAttentionPipeline = [gDevice newComputePipelineStateWithFunction:grouped_attention_function error:&error];
        if (!gResidentGroupedAttentionPipeline) {
            rusty_metal_log_error("create resident grouped attention pipeline", error);
            return;
        }
        id<MTLLibrary> silu_library = [gDevice newLibraryWithSource:kSiluMulSource options:options error:&error];
        if (!silu_library) {
            rusty_metal_log_error("compile silu_mul library", error);
            return;
        }
        id<MTLFunction> silu_function = [silu_library newFunctionWithName:@"silu_mul"];
        if (!silu_function) {
            rusty_metal_log_error("load silu_mul function", nil);
            return;
        }
        gSiluMulPipeline = [gDevice newComputePipelineStateWithFunction:silu_function error:&error];
        if (!gSiluMulPipeline) {
            rusty_metal_log_error("create silu_mul pipeline", error);
            return;
        }
        id<MTLLibrary> residual_library = [gDevice newLibraryWithSource:kResidualSource options:options error:&error];
        if (!residual_library) {
            rusty_metal_log_error("compile residual kernels", error);
            return;
        }
        id<MTLFunction> residual_rms_function = [residual_library newFunctionWithName:@"residual_rms"];
        if (!residual_rms_function) {
            rusty_metal_log_error("load residual_rms function", nil);
            return;
        }
        gResidualRmsPipeline = [gDevice newComputePipelineStateWithFunction:residual_rms_function error:&error];
        if (!gResidualRmsPipeline) {
            rusty_metal_log_error("create residual_rms pipeline", error);
            return;
        }
        id<MTLFunction> residual_add_function = [residual_library newFunctionWithName:@"residual_add"];
        if (!residual_add_function) {
            rusty_metal_log_error("load residual_add function", nil);
            return;
        }
        gResidualAddPipeline = [gDevice newComputePipelineStateWithFunction:residual_add_function error:&error];
        if (!gResidualAddPipeline) {
            rusty_metal_log_error("create residual_add pipeline", error);
            return;
        }
        id<MTLLibrary> rope_library = [gDevice newLibraryWithSource:kRopeStoreSource options:nil error:&error];
        if (!rope_library) {
            rusty_metal_log_error("compile rope_store library", error);
            return;
        }
        id<MTLFunction> rope_function = [rope_library newFunctionWithName:@"rope_store"];
        if (!rope_function) {
            rusty_metal_log_error("load rope_store function", nil);
            return;
        }
        gRopeStorePipeline = [gDevice newComputePipelineStateWithFunction:rope_function error:&error];
        if (!gRopeStorePipeline) {
            rusty_metal_log_error("create rope_store pipeline", error);
            return;
        }
        gQueue = [gDevice newCommandQueue];
        if (!gQueue) {
            rusty_metal_log_error("create command queue", nil);
            return;
        }
        gWeightBuffers = [[NSMutableDictionary alloc] init];
        gSharedBuffers = [[NSMutableDictionary alloc] init];
        gAttentionZeroBuffer = [gDevice newBufferWithBytes:&gAttentionZero
                                                    length:sizeof(gAttentionZero)
                                                   options:MTLResourceStorageModeShared];
        if (!gAttentionZeroBuffer) {
            rusty_metal_log_error("create attention zero buffer", nil);
            return;
        }
        if (rusty_metal_profile_enabled()) {
            atexit(rusty_metal_profile_dump);
        }
        ok = YES;
    });
    return ok;
}

static id<MTLBuffer> rusty_metal_weight_buffer(const uint8_t *weights, uintptr_t weights_len) {
    NSUInteger cache_index = (((uintptr_t)weights) >> 6) & (RUSTY_WEIGHT_CACHE_SIZE - 1);
    RustyWeightCacheEntry *entry = &gWeightCache[cache_index];
    if (entry->key == weights && entry->len >= weights_len && entry->buffer) {
        return entry->buffer;
    }

    NSNumber *key = @((uintptr_t)weights);
    id<MTLBuffer> weight_buffer = [gWeightBuffers objectForKey:key];
    if (!weight_buffer || [weight_buffer length] < weights_len) {
        if (rusty_metal_private_weights_enabled()) {
            id<MTLBuffer> staging = [gDevice newBufferWithBytes:weights
                                                         length:(NSUInteger)weights_len
                                                        options:MTLResourceStorageModeShared];
            weight_buffer = [gDevice newBufferWithLength:(NSUInteger)weights_len
                                                 options:MTLResourceStorageModePrivate];
            gMetalBufferAllocations += staging ? 1 : 0;
            gMetalBufferAllocations += weight_buffer ? 1 : 0;
            gMetalCpuToGpuBytes += staging ? weights_len : 0;
            if (staging && weight_buffer) {
                double encode_start = rusty_metal_now_seconds();
        id<MTLCommandBuffer> command_buffer = [gQueue commandBuffer];
                id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
                [blit copyFromBuffer:staging
                         sourceOffset:0
                             toBuffer:weight_buffer
                    destinationOffset:0
                                 size:(NSUInteger)weights_len];
                [blit endEncoding];
                double encode_end = rusty_metal_now_seconds();
                [command_buffer commit];
                [command_buffer waitUntilCompleted];
                rusty_metal_profile_command_buffer(command_buffer, encode_start, encode_end);
                if ([command_buffer status] != MTLCommandBufferStatusCompleted) {
                    weight_buffer = nil;
                }
            } else {
                weight_buffer = nil;
            }
        }
        if (!weight_buffer) {
            weight_buffer = [gDevice newBufferWithBytes:weights
                                                 length:(NSUInteger)weights_len
                                                options:MTLResourceStorageModeShared];
            gMetalBufferAllocations += weight_buffer ? 1 : 0;
            gMetalCpuToGpuBytes += weight_buffer ? weights_len : 0;
        }
        if (!weight_buffer) return nil;
        [gWeightBuffers setObject:weight_buffer forKey:key];
    }
    entry->key = weights;
    entry->len = weights_len;
    entry->buffer = weight_buffer;
    return weight_buffer;
}

static id<MTLBuffer> rusty_metal_shared_buffer(const void *bytes, uintptr_t bytes_len) {
    NSNumber *key = @((uintptr_t)bytes);
    id<MTLBuffer> buffer = [gSharedBuffers objectForKey:key];
    if (!buffer || [buffer length] < bytes_len) {
        buffer = [gDevice newBufferWithBytesNoCopy:(void *)bytes
                                            length:(NSUInteger)bytes_len
                                           options:MTLResourceStorageModeShared
                                       deallocator:nil];
        if (!buffer) return nil;
        gMetalBufferAllocations += 1;
        [gSharedBuffers setObject:buffer forKey:key];
    }
    return buffer;
}

static id<MTLBuffer> rusty_metal_copy_buffer(const void *bytes, uintptr_t bytes_len) {
    id<MTLBuffer> buffer = [gDevice newBufferWithBytes:bytes
                                                length:(NSUInteger)bytes_len
                                               options:MTLResourceStorageModeShared];
    gMetalTemporaryAllocations += buffer ? 1 : 0;
    gMetalCpuToGpuBytes += buffer ? bytes_len : 0;
    return buffer;
}

static BOOL rusty_metal_env_disabled(const char *name) {
    const char *value = getenv(name);
    if (!value) return NO;
    return strcmp(value, "0") == 0 ||
           strcasecmp(value, "false") == 0 ||
           strcasecmp(value, "no") == 0 ||
           strcasecmp(value, "off") == 0;
}

static BOOL rusty_metal_env_enabled(const char *name) {
    const char *value = getenv(name);
    if (!value) return NO;
    return strcmp(value, "") == 0 ||
           strcmp(value, "1") == 0 ||
           strcasecmp(value, "true") == 0 ||
           strcasecmp(value, "yes") == 0 ||
           strcasecmp(value, "on") == 0;
}

static BOOL rusty_metal_nocopy_enabled(void) {
    return rusty_metal_env_enabled("RUSTY_LLM_METAL_NOCOPY");
}

static BOOL rusty_metal_private_weights_enabled(void) {
    return !rusty_metal_env_disabled("RUSTY_LLM_METAL_PRIVATE_WEIGHTS");
}

static BOOL rusty_metal_profile_enabled(void) {
    return rusty_metal_env_enabled("RUSTY_LLM_METAL_PROFILE");
}

static double rusty_metal_now_seconds(void) {
    return [[NSDate date] timeIntervalSinceReferenceDate];
}

static void rusty_metal_profile_command_buffer(id<MTLCommandBuffer> command_buffer,
                                               double encode_start,
                                               double encode_end) {
    if (!rusty_metal_profile_enabled()) return;
    gMetalCommandBuffers += 1;
    gMetalCpuEncodeSeconds += encode_end - encode_start;
    if ([command_buffer GPUStartTime] > 0.0 && [command_buffer GPUEndTime] > [command_buffer GPUStartTime]) {
        gMetalGpuSeconds += [command_buffer GPUEndTime] - [command_buffer GPUStartTime];
    }
}

static void rusty_metal_profile_dump(void) {
    if (!rusty_metal_profile_enabled()) return;
    fprintf(stderr,
            "Metal profile: command_buffers=%llu dispatches=%llu cpu_encode_ms=%.3f gpu_ms=%.3f cpu_to_gpu_bytes=%llu gpu_to_cpu_bytes=%llu buffer_allocations=%llu temporary_allocations=%llu\n",
            (unsigned long long)gMetalCommandBuffers,
            (unsigned long long)gMetalDispatches,
            gMetalCpuEncodeSeconds * 1000.0,
            gMetalGpuSeconds * 1000.0,
            (unsigned long long)gMetalCpuToGpuBytes,
            (unsigned long long)gMetalGpuToCpuBytes,
            (unsigned long long)gMetalBufferAllocations,
            (unsigned long long)gMetalTemporaryAllocations);
}

static NSUInteger rusty_metal_q4k_rows_per_group(NSUInteger rows) {
    const char *value = getenv("RUSTY_LLM_METAL_Q4K_ROWS_PER_GROUP");
    if (value && *value) {
        char *end = NULL;
        unsigned long parsed = strtoul(value, &end, 10);
        if (end != value && parsed >= 2 && parsed <= 16 && (parsed % 2) == 0) {
            return (NSUInteger)parsed;
        }
    }
    (void)rows;
    return 4;
}

static NSUInteger rusty_metal_q6k_rows_per_group(NSUInteger rows) {
    const char *value = getenv("RUSTY_LLM_METAL_Q6K_ROWS_PER_GROUP");
    if (value && *value) {
        char *end = NULL;
        unsigned long parsed = strtoul(value, &end, 10);
        if (end != value && parsed >= 2 && parsed <= 8 && (parsed % 2) == 0) {
            return (NSUInteger)parsed;
        }
    }
    // One simdgroup produces two rows; the smaller group wins for the wide
    // value and down projections used during token-serial decode.
    (void)rows;
    return 2;
}

static BOOL rusty_metal_ensure_buffer(id<MTLBuffer> __strong *buffer, NSUInteger size) {
    if (!*buffer || [*buffer length] < size) {
        *buffer = [gDevice newBufferWithLength:size options:MTLResourceStorageModeShared];
        gMetalBufferAllocations += *buffer ? 1 : 0;
    }
    return *buffer != nil;
}

static id<MTLBuffer> rusty_metal_input_buffer(const void *bytes,
                                              NSUInteger bytes_len,
                                              id<MTLBuffer> __strong *copy_buffer) {
    if (rusty_metal_nocopy_enabled()) {
        id<MTLBuffer> shared = rusty_metal_shared_buffer(bytes, (uintptr_t)bytes_len);
        if (shared) return shared;
    }
    if (!rusty_metal_ensure_buffer(copy_buffer, bytes_len)) return nil;
    memcpy([*copy_buffer contents], bytes, bytes_len);
    gMetalCpuToGpuBytes += bytes_len;
    return *copy_buffer;
}

static id<MTLBuffer> rusty_metal_output_buffer(void *bytes,
                                               NSUInteger bytes_len,
                                               id<MTLBuffer> __strong *copy_buffer,
                                               BOOL *needs_copy) {
    *needs_copy = YES;
    if (rusty_metal_nocopy_enabled()) {
        id<MTLBuffer> shared = rusty_metal_shared_buffer(bytes, (uintptr_t)bytes_len);
        if (shared) {
            *needs_copy = NO;
            return shared;
        }
    }
    if (!rusty_metal_ensure_buffer(copy_buffer, bytes_len)) return nil;
    return *copy_buffer;
}

static id<MTLBuffer> rusty_metal_inout_buffer(void *bytes,
                                              NSUInteger bytes_len,
                                              id<MTLBuffer> __strong *copy_buffer,
                                              BOOL *needs_copy) {
    *needs_copy = YES;
    if (rusty_metal_nocopy_enabled()) {
        id<MTLBuffer> shared = rusty_metal_shared_buffer(bytes, (uintptr_t)bytes_len);
        if (shared) {
            *needs_copy = NO;
            return shared;
        }
    }
    if (!rusty_metal_ensure_buffer(copy_buffer, bytes_len)) return nil;
    memcpy([*copy_buffer contents], bytes, bytes_len);
    gMetalCpuToGpuBytes += bytes_len;
    return *copy_buffer;
}

static void rusty_metal_encode_q4k(id<MTLComputeCommandEncoder> encoder,
                                   id<MTLBuffer> weight_buffer,
                                   id<MTLBuffer> x_buffer,
                                   id<MTLBuffer> out_buffer,
                                   uintptr_t rows,
                                   uintptr_t cols) {
    // Two SIMD groups, each producing two rows.
    const NSUInteger rows_per_group = rusty_metal_q4k_rows_per_group((NSUInteger)rows);
    RustyQ4KParams params = {
        .rows = (uint32_t)rows,
        .cols = (uint32_t)cols,
        .row_bytes = (uint32_t)((cols / 256) * 144),
        .n_blocks = (uint32_t)(cols / 256),
        .rows_per_group = (uint32_t)rows_per_group,
    };

    [encoder setComputePipelineState:gQ4KPipeline];
    [encoder setBuffer:weight_buffer offset:0 atIndex:0];
    [encoder setBuffer:x_buffer offset:0 atIndex:1];
    [encoder setBuffer:out_buffer offset:0 atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];

    MTLSize threads_per_group = MTLSizeMake(32 * (rows_per_group / 2), 1, 1);
    MTLSize threadgroups = MTLSizeMake(
        ((NSUInteger)rows + rows_per_group - 1) / rows_per_group,
        1,
        1
    );
    gMetalDispatches += 1;
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_group];
}

static void rusty_metal_encode_q4k_pair(id<MTLComputeCommandEncoder> encoder,
                                        id<MTLBuffer> weight_a,
                                        id<MTLBuffer> weight_b,
                                        id<MTLBuffer> x_buffer,
                                        id<MTLBuffer> out_a,
                                        id<MTLBuffer> out_b,
                                        uintptr_t rows_a,
                                        uintptr_t rows_b,
                                        uintptr_t cols) {
    const NSUInteger rows_per_group = rusty_metal_q4k_rows_per_group((NSUInteger)(rows_a + rows_b));
    RustyQ4KPairParams params = {
        .rows_a = (uint32_t)rows_a,
        .rows_b = (uint32_t)rows_b,
        .cols = (uint32_t)cols,
        .row_bytes = (uint32_t)((cols / 256) * 144),
        .n_blocks = (uint32_t)(cols / 256),
        .rows_per_group = (uint32_t)rows_per_group,
    };
    [encoder setComputePipelineState:gQ4KPairPipeline];
    [encoder setBuffer:weight_a offset:0 atIndex:0];
    [encoder setBuffer:weight_b offset:0 atIndex:1];
    [encoder setBuffer:x_buffer offset:0 atIndex:2];
    [encoder setBuffer:out_a offset:0 atIndex:3];
    [encoder setBuffer:out_b offset:0 atIndex:4];
    [encoder setBytes:&params length:sizeof(params) atIndex:5];
    NSUInteger groups_a = ((NSUInteger)rows_a + rows_per_group - 1) / rows_per_group;
    NSUInteger groups_b = ((NSUInteger)rows_b + rows_per_group - 1) / rows_per_group;
    MTLSize threads = MTLSizeMake(32 * (rows_per_group / 2), 1, 1);
    gMetalDispatches += 1;
    [encoder dispatchThreadgroups:MTLSizeMake(groups_a + groups_b, 1, 1)
            threadsPerThreadgroup:threads];
}

static void rusty_metal_encode_q6k(id<MTLComputeCommandEncoder> encoder,
                                   id<MTLBuffer> weight_buffer,
                                   id<MTLBuffer> x_buffer,
                                   id<MTLBuffer> out_buffer,
                                   uintptr_t rows,
                                   uintptr_t cols) {
    NSUInteger rows_per_group = rusty_metal_q6k_rows_per_group((NSUInteger)rows);
    RustyQ4KParams params = {
        .rows = (uint32_t)rows,
        .cols = (uint32_t)cols,
        .row_bytes = (uint32_t)((cols / 256) * 210),
        .n_blocks = (uint32_t)(cols / 256),
        .rows_per_group = (uint32_t)rows_per_group,
    };

    [encoder setComputePipelineState:gQ6KPipeline];
    [encoder setBuffer:weight_buffer offset:0 atIndex:0];
    [encoder setBuffer:x_buffer offset:0 atIndex:1];
    [encoder setBuffer:out_buffer offset:0 atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];

    MTLSize threads_per_group = MTLSizeMake(32 * (rows_per_group / 2), 1, 1);
    MTLSize threadgroups = MTLSizeMake(
        ((NSUInteger)rows + rows_per_group - 1) / rows_per_group,
        1,
        1
    );
    gMetalDispatches += 1;
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_group];
}

static void rusty_metal_encode_q4_0(id<MTLComputeCommandEncoder> encoder,
                                    id<MTLBuffer> weight_buffer,
                                    id<MTLBuffer> x_buffer,
                                    id<MTLBuffer> out_buffer,
                                    uintptr_t rows,
                                    uintptr_t cols) {
    RustyQ4KParams params = {
        .rows = (uint32_t)rows,
        .cols = (uint32_t)cols,
        .row_bytes = (uint32_t)((cols / 32) * 18),
        .n_blocks = (uint32_t)(cols / 32),
        .rows_per_group = RUSTY_MATVEC_ROWS_PER_GROUP,
    };

    [encoder setComputePipelineState:gQ4_0Pipeline];
    [encoder setBuffer:weight_buffer offset:0 atIndex:0];
    [encoder setBuffer:x_buffer offset:0 atIndex:1];
    [encoder setBuffer:out_buffer offset:0 atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];

    MTLSize threads_per_group = MTLSizeMake(RUSTY_MATVEC_THREADS_PER_GROUP, 1, 1);
    MTLSize threadgroups = MTLSizeMake(
        ((NSUInteger)rows + RUSTY_MATVEC_ROWS_PER_GROUP - 1) / RUSTY_MATVEC_ROWS_PER_GROUP,
        1,
        1
    );
    gMetalDispatches += 1;
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_group];
}

static void rusty_metal_encode_q8_0(id<MTLComputeCommandEncoder> encoder,
                                    id<MTLBuffer> weight_buffer,
                                    id<MTLBuffer> x_buffer,
                                    id<MTLBuffer> out_buffer,
                                    uintptr_t rows,
                                    uintptr_t cols) {
    RustyQ4KParams params = {
        .rows = (uint32_t)rows,
        .cols = (uint32_t)cols,
        .row_bytes = (uint32_t)((cols / 32) * 34),
        .n_blocks = (uint32_t)(cols / 32),
        .rows_per_group = RUSTY_MATVEC_ROWS_PER_GROUP,
    };

    [encoder setComputePipelineState:gQ8_0Pipeline];
    [encoder setBuffer:weight_buffer offset:0 atIndex:0];
    [encoder setBuffer:x_buffer offset:0 atIndex:1];
    [encoder setBuffer:out_buffer offset:0 atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];

    MTLSize threads_per_group = MTLSizeMake(RUSTY_MATVEC_THREADS_PER_GROUP, 1, 1);
    MTLSize threadgroups = MTLSizeMake(
        ((NSUInteger)rows + RUSTY_MATVEC_ROWS_PER_GROUP - 1) / RUSTY_MATVEC_ROWS_PER_GROUP,
        1,
        1
    );
    gMetalDispatches += 1;
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_group];
}

static void rusty_metal_encode_silu_mul(id<MTLComputeCommandEncoder> encoder,
                                        id<MTLBuffer> gate_buffer,
                                        id<MTLBuffer> up_buffer,
                                        id<MTLBuffer> out_buffer,
                                        uintptr_t len) {
    RustyUnaryParams params = {
        .len = (uint32_t)len,
    };
    [encoder setComputePipelineState:gSiluMulPipeline];
    [encoder setBuffer:gate_buffer offset:0 atIndex:0];
    [encoder setBuffer:up_buffer offset:0 atIndex:1];
    [encoder setBuffer:out_buffer offset:0 atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];
    MTLSize threads = MTLSizeMake(256, 1, 1);
    MTLSize groups = MTLSizeMake(((NSUInteger)len + 255) / 256, 1, 1);
    gMetalDispatches += 1;
    [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
}

static void rusty_metal_encode_residual_rms(id<MTLComputeCommandEncoder> encoder,
                                            id<MTLBuffer> x_buffer,
                                            id<MTLBuffer> residual_buffer,
                                            id<MTLBuffer> weight_buffer,
                                            id<MTLBuffer> out_buffer,
                                            uintptr_t len,
                                            float eps) {
    RustyResidualNormParams params = {
        .len = (uint32_t)len,
        .eps = eps,
    };
    [encoder setComputePipelineState:gResidualRmsPipeline];
    [encoder setBuffer:x_buffer offset:0 atIndex:0];
    [encoder setBuffer:residual_buffer offset:0 atIndex:1];
    [encoder setBuffer:weight_buffer offset:0 atIndex:2];
    [encoder setBuffer:out_buffer offset:0 atIndex:3];
    [encoder setBytes:&params length:sizeof(params) atIndex:4];
    gMetalDispatches += 1;
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void rusty_metal_encode_residual_add(id<MTLComputeCommandEncoder> encoder,
                                            id<MTLBuffer> x_buffer,
                                            id<MTLBuffer> residual_buffer,
                                            uintptr_t len) {
    RustyUnaryParams params = {
        .len = (uint32_t)len,
    };
    [encoder setComputePipelineState:gResidualAddPipeline];
    [encoder setBuffer:x_buffer offset:0 atIndex:0];
    [encoder setBuffer:residual_buffer offset:0 atIndex:1];
    [encoder setBytes:&params length:sizeof(params) atIndex:2];
    MTLSize threads = MTLSizeMake(256, 1, 1);
    MTLSize groups = MTLSizeMake(((NSUInteger)len + 255) / 256, 1, 1);
    gMetalDispatches += 1;
    [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
}

static void rusty_metal_encode_attention(id<MTLComputeCommandEncoder> encoder,
                                         id<MTLBuffer> query_buffer,
                                         id<MTLBuffer> keys_buffer,
                                         id<MTLBuffer> values_buffer,
                                         id<MTLBuffer> sinks_buffer,
                                         id<MTLBuffer> out_buffer,
                                         uintptr_t heads,
                                         uintptr_t kv_mul,
                                         uintptr_t head_dim,
                                         uintptr_t value_dim,
                                         uintptr_t key_stride,
                                         uintptr_t value_stride,
                                         uintptr_t slot_count,
                                         uintptr_t start_t,
                                         uintptr_t end_t,
                                         float scale,
                                         int use_sink) {
    RustyAttentionParams params = {
        .heads = (uint32_t)heads,
        .kv_mul = (uint32_t)kv_mul,
        .head_dim = (uint32_t)head_dim,
        .value_dim = (uint32_t)value_dim,
        .key_stride = (uint32_t)key_stride,
        .value_stride = (uint32_t)value_stride,
        .slot_count = (uint32_t)slot_count,
        .start_t = (uint32_t)start_t,
        .end_t = (uint32_t)end_t,
        .use_sink = (uint32_t)use_sink,
        .scale = scale,
    };
    [encoder setComputePipelineState:gAttentionPipeline];
    [encoder setBuffer:query_buffer offset:0 atIndex:0];
    [encoder setBuffer:keys_buffer offset:0 atIndex:1];
    [encoder setBuffer:values_buffer offset:0 atIndex:2];
    [encoder setBuffer:out_buffer offset:0 atIndex:3];
    [encoder setBuffer:sinks_buffer offset:0 atIndex:4];
    [encoder setBytes:&params length:sizeof(params) atIndex:5];
    gMetalDispatches += 1;
    [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)heads, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

int rusty_metal_available(void) {
    return rusty_metal_init() ? 1 : 0;
}

int rusty_metal_q4k_matvec(const uint8_t *weights,
                           uintptr_t weights_len,
                           const float *x,
                           uintptr_t rows,
                           uintptr_t cols,
                           float *out) {
    if (!rusty_metal_init() || !weights || !x || !out || rows == 0 || cols == 0 || (cols % 256) != 0) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> weight_buffer = rusty_metal_weight_buffer(weights, weights_len);
        if (!weight_buffer) return 0;

        static id<MTLBuffer> x_buffer = nil;
        static id<MTLBuffer> out_buffer = nil;
        NSUInteger x_size = (NSUInteger)(cols * sizeof(float));
        NSUInteger out_size = (NSUInteger)(rows * sizeof(float));
        BOOL out_needs_copy = YES;

        id<MTLBuffer> x_metal = rusty_metal_input_buffer(x, x_size, &x_buffer);
        id<MTLBuffer> out_metal = rusty_metal_output_buffer(out, out_size, &out_buffer, &out_needs_copy);
        if (!x_metal || !out_metal) return 0;

        double encode_start = rusty_metal_now_seconds();
        id<MTLCommandBuffer> command_buffer = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        rusty_metal_encode_q4k(encoder, weight_buffer, x_metal, out_metal, rows, cols);
        [encoder endEncoding];
        double encode_end = rusty_metal_now_seconds();
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        rusty_metal_profile_command_buffer(command_buffer, encode_start, encode_end);
        if ([command_buffer status] != MTLCommandBufferStatusCompleted) return 0;

        if (out_needs_copy) { gMetalGpuToCpuBytes += out_size; memcpy(out, [out_metal contents], out_size); }
        return 1;
    }
}

int rusty_metal_q4k_matvec2(const uint8_t *weights_a,
                            uintptr_t weights_a_len,
                            uintptr_t rows_a,
                            const uint8_t *weights_b,
                            uintptr_t weights_b_len,
                            uintptr_t rows_b,
                            const float *x,
                            uintptr_t cols,
                            float *out_a,
                            float *out_b) {
    if (!rusty_metal_init() || !weights_a || !weights_b || !x || !out_a || !out_b ||
        rows_a == 0 || rows_b == 0 || cols == 0 || (cols % 256) != 0) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> weight_a = rusty_metal_weight_buffer(weights_a, weights_a_len);
        id<MTLBuffer> weight_b = rusty_metal_weight_buffer(weights_b, weights_b_len);
        if (!weight_a || !weight_b) return 0;

        static id<MTLBuffer> x_buffer = nil;
        static id<MTLBuffer> out_a_buffer = nil;
        static id<MTLBuffer> out_b_buffer = nil;
        NSUInteger x_size = (NSUInteger)(cols * sizeof(float));
        NSUInteger out_a_size = (NSUInteger)(rows_a * sizeof(float));
        NSUInteger out_b_size = (NSUInteger)(rows_b * sizeof(float));
        BOOL out_a_needs_copy = YES;
        BOOL out_b_needs_copy = YES;

        id<MTLBuffer> x_metal = rusty_metal_input_buffer(x, x_size, &x_buffer);
        id<MTLBuffer> out_a_metal = rusty_metal_output_buffer(out_a, out_a_size, &out_a_buffer, &out_a_needs_copy);
        id<MTLBuffer> out_b_metal = rusty_metal_output_buffer(out_b, out_b_size, &out_b_buffer, &out_b_needs_copy);
        if (!x_metal || !out_a_metal || !out_b_metal) return 0;

        double encode_start = rusty_metal_now_seconds();
        id<MTLCommandBuffer> command_buffer = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        rusty_metal_encode_q4k_pair(
            encoder, weight_a, weight_b, x_metal, out_a_metal, out_b_metal,
            rows_a, rows_b, cols
        );
        [encoder endEncoding];
        double encode_end = rusty_metal_now_seconds();
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        rusty_metal_profile_command_buffer(command_buffer, encode_start, encode_end);
        if ([command_buffer status] != MTLCommandBufferStatusCompleted) return 0;

        if (out_a_needs_copy) { gMetalGpuToCpuBytes += out_a_size; memcpy(out_a, [out_a_metal contents], out_a_size); }
        if (out_b_needs_copy) { gMetalGpuToCpuBytes += out_b_size; memcpy(out_b, [out_b_metal contents], out_b_size); }
        return 1;
    }
}

int rusty_metal_q6k_matvec(const uint8_t *weights,
                           uintptr_t weights_len,
                           const float *x,
                           uintptr_t rows,
                           uintptr_t cols,
                           float *out) {
    if (!rusty_metal_init() || !weights || !x || !out || rows == 0 || cols == 0 || (cols % 256) != 0) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> weight_buffer = rusty_metal_weight_buffer(weights, weights_len);
        if (!weight_buffer) return 0;

        static id<MTLBuffer> x_buffer = nil;
        static id<MTLBuffer> out_buffer = nil;
        NSUInteger x_size = (NSUInteger)(cols * sizeof(float));
        NSUInteger out_size = (NSUInteger)(rows * sizeof(float));
        BOOL out_needs_copy = YES;

        id<MTLBuffer> x_metal = rusty_metal_input_buffer(x, x_size, &x_buffer);
        id<MTLBuffer> out_metal = rusty_metal_output_buffer(out, out_size, &out_buffer, &out_needs_copy);
        if (!x_metal || !out_metal) return 0;

        double encode_start = rusty_metal_now_seconds();
        id<MTLCommandBuffer> command_buffer = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        rusty_metal_encode_q6k(encoder, weight_buffer, x_metal, out_metal, rows, cols);
        [encoder endEncoding];
        double encode_end = rusty_metal_now_seconds();
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        rusty_metal_profile_command_buffer(command_buffer, encode_start, encode_end);
        if ([command_buffer status] != MTLCommandBufferStatusCompleted) return 0;

        if (out_needs_copy) { gMetalGpuToCpuBytes += out_size; memcpy(out, [out_metal contents], out_size); }
        return 1;
    }
}

int rusty_metal_q6k_matvec2(const uint8_t *weights_a,
                            uintptr_t weights_a_len,
                            uintptr_t rows_a,
                            const uint8_t *weights_b,
                            uintptr_t weights_b_len,
                            uintptr_t rows_b,
                            const float *x,
                            uintptr_t cols,
                            float *out_a,
                            float *out_b) {
    if (!rusty_metal_init() || !weights_a || !weights_b || !x || !out_a || !out_b ||
        rows_a == 0 || rows_b == 0 || cols == 0 || (cols % 256) != 0) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> weight_a = rusty_metal_weight_buffer(weights_a, weights_a_len);
        id<MTLBuffer> weight_b = rusty_metal_weight_buffer(weights_b, weights_b_len);
        if (!weight_a || !weight_b) return 0;

        static id<MTLBuffer> x_buffer = nil;
        static id<MTLBuffer> out_a_buffer = nil;
        static id<MTLBuffer> out_b_buffer = nil;
        NSUInteger x_size = (NSUInteger)(cols * sizeof(float));
        NSUInteger out_a_size = (NSUInteger)(rows_a * sizeof(float));
        NSUInteger out_b_size = (NSUInteger)(rows_b * sizeof(float));
        BOOL out_a_needs_copy = YES;
        BOOL out_b_needs_copy = YES;

        id<MTLBuffer> x_metal = rusty_metal_input_buffer(x, x_size, &x_buffer);
        id<MTLBuffer> out_a_metal = rusty_metal_output_buffer(out_a, out_a_size, &out_a_buffer, &out_a_needs_copy);
        id<MTLBuffer> out_b_metal = rusty_metal_output_buffer(out_b, out_b_size, &out_b_buffer, &out_b_needs_copy);
        if (!x_metal || !out_a_metal || !out_b_metal) return 0;

        double encode_start = rusty_metal_now_seconds();
        id<MTLCommandBuffer> command_buffer = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        rusty_metal_encode_q6k(encoder, weight_a, x_metal, out_a_metal, rows_a, cols);
        rusty_metal_encode_q6k(encoder, weight_b, x_metal, out_b_metal, rows_b, cols);
        [encoder endEncoding];
        double encode_end = rusty_metal_now_seconds();
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        rusty_metal_profile_command_buffer(command_buffer, encode_start, encode_end);
        if ([command_buffer status] != MTLCommandBufferStatusCompleted) return 0;

        if (out_a_needs_copy) { gMetalGpuToCpuBytes += out_a_size; memcpy(out_a, [out_a_metal contents], out_a_size); }
        if (out_b_needs_copy) { gMetalGpuToCpuBytes += out_b_size; memcpy(out_b, [out_b_metal contents], out_b_size); }
        return 1;
    }
}

int rusty_metal_q6k_matvec3(const uint8_t *weights_a,
                            uintptr_t weights_a_len,
                            uintptr_t rows_a,
                            const uint8_t *weights_b,
                            uintptr_t weights_b_len,
                            uintptr_t rows_b,
                            const uint8_t *weights_c,
                            uintptr_t weights_c_len,
                            uintptr_t rows_c,
                            const float *x,
                            uintptr_t cols,
                            float *out_a,
                            float *out_b,
                            float *out_c) {
    if (!rusty_metal_init() || !weights_a || !weights_b || !weights_c || !x || !out_a || !out_b || !out_c ||
        rows_a == 0 || rows_b == 0 || rows_c == 0 || cols == 0 || (cols % 256) != 0) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> weight_a = rusty_metal_weight_buffer(weights_a, weights_a_len);
        id<MTLBuffer> weight_b = rusty_metal_weight_buffer(weights_b, weights_b_len);
        id<MTLBuffer> weight_c = rusty_metal_weight_buffer(weights_c, weights_c_len);
        if (!weight_a || !weight_b || !weight_c) return 0;

        static id<MTLBuffer> x_buffer = nil;
        static id<MTLBuffer> out_a_buffer = nil;
        static id<MTLBuffer> out_b_buffer = nil;
        static id<MTLBuffer> out_c_buffer = nil;
        NSUInteger x_size = (NSUInteger)(cols * sizeof(float));
        NSUInteger out_a_size = (NSUInteger)(rows_a * sizeof(float));
        NSUInteger out_b_size = (NSUInteger)(rows_b * sizeof(float));
        NSUInteger out_c_size = (NSUInteger)(rows_c * sizeof(float));
        BOOL out_a_needs_copy = YES;
        BOOL out_b_needs_copy = YES;
        BOOL out_c_needs_copy = YES;

        id<MTLBuffer> x_metal = rusty_metal_input_buffer(x, x_size, &x_buffer);
        id<MTLBuffer> out_a_metal = rusty_metal_output_buffer(out_a, out_a_size, &out_a_buffer, &out_a_needs_copy);
        id<MTLBuffer> out_b_metal = rusty_metal_output_buffer(out_b, out_b_size, &out_b_buffer, &out_b_needs_copy);
        id<MTLBuffer> out_c_metal = rusty_metal_output_buffer(out_c, out_c_size, &out_c_buffer, &out_c_needs_copy);
        if (!x_metal || !out_a_metal || !out_b_metal || !out_c_metal) return 0;

        double encode_start = rusty_metal_now_seconds();
        id<MTLCommandBuffer> command_buffer = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        rusty_metal_encode_q6k(encoder, weight_a, x_metal, out_a_metal, rows_a, cols);
        rusty_metal_encode_q6k(encoder, weight_b, x_metal, out_b_metal, rows_b, cols);
        rusty_metal_encode_q6k(encoder, weight_c, x_metal, out_c_metal, rows_c, cols);
        [encoder endEncoding];
        double encode_end = rusty_metal_now_seconds();
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        rusty_metal_profile_command_buffer(command_buffer, encode_start, encode_end);
        if ([command_buffer status] != MTLCommandBufferStatusCompleted) return 0;

        if (out_a_needs_copy) { gMetalGpuToCpuBytes += out_a_size; memcpy(out_a, [out_a_metal contents], out_a_size); }
        if (out_b_needs_copy) { gMetalGpuToCpuBytes += out_b_size; memcpy(out_b, [out_b_metal contents], out_b_size); }
        if (out_c_needs_copy) { gMetalGpuToCpuBytes += out_c_size; memcpy(out_c, [out_c_metal contents], out_c_size); }
        return 1;
    }
}

int rusty_metal_q4k_matvec3(const uint8_t *weights_a,
                            uintptr_t weights_a_len,
                            uintptr_t rows_a,
                            const uint8_t *weights_b,
                            uintptr_t weights_b_len,
                            uintptr_t rows_b,
                            const uint8_t *weights_c,
                            uintptr_t weights_c_len,
                            uintptr_t rows_c,
                            const float *x,
                            uintptr_t cols,
                            float *out_a,
                            float *out_b,
                            float *out_c) {
    if (!rusty_metal_init() || !weights_a || !weights_b || !weights_c || !x || !out_a || !out_b || !out_c ||
        rows_a == 0 || rows_b == 0 || rows_c == 0 || cols == 0 || (cols % 256) != 0) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> weight_a = rusty_metal_weight_buffer(weights_a, weights_a_len);
        id<MTLBuffer> weight_b = rusty_metal_weight_buffer(weights_b, weights_b_len);
        id<MTLBuffer> weight_c = rusty_metal_weight_buffer(weights_c, weights_c_len);
        if (!weight_a || !weight_b || !weight_c) return 0;

        static id<MTLBuffer> x_buffer = nil;
        static id<MTLBuffer> out_a_buffer = nil;
        static id<MTLBuffer> out_b_buffer = nil;
        static id<MTLBuffer> out_c_buffer = nil;
        NSUInteger x_size = (NSUInteger)(cols * sizeof(float));
        NSUInteger out_a_size = (NSUInteger)(rows_a * sizeof(float));
        NSUInteger out_b_size = (NSUInteger)(rows_b * sizeof(float));
        NSUInteger out_c_size = (NSUInteger)(rows_c * sizeof(float));
        BOOL out_a_needs_copy = YES;
        BOOL out_b_needs_copy = YES;
        BOOL out_c_needs_copy = YES;

        id<MTLBuffer> x_metal = rusty_metal_input_buffer(x, x_size, &x_buffer);
        id<MTLBuffer> out_a_metal = rusty_metal_output_buffer(out_a, out_a_size, &out_a_buffer, &out_a_needs_copy);
        id<MTLBuffer> out_b_metal = rusty_metal_output_buffer(out_b, out_b_size, &out_b_buffer, &out_b_needs_copy);
        id<MTLBuffer> out_c_metal = rusty_metal_output_buffer(out_c, out_c_size, &out_c_buffer, &out_c_needs_copy);
        if (!x_metal || !out_a_metal || !out_b_metal || !out_c_metal) return 0;

        double encode_start = rusty_metal_now_seconds();
        id<MTLCommandBuffer> command_buffer = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        rusty_metal_encode_q4k(encoder, weight_a, x_metal, out_a_metal, rows_a, cols);
        rusty_metal_encode_q4k(encoder, weight_b, x_metal, out_b_metal, rows_b, cols);
        rusty_metal_encode_q4k(encoder, weight_c, x_metal, out_c_metal, rows_c, cols);
        [encoder endEncoding];
        double encode_end = rusty_metal_now_seconds();
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        rusty_metal_profile_command_buffer(command_buffer, encode_start, encode_end);
        if ([command_buffer status] != MTLCommandBufferStatusCompleted) return 0;

        if (out_a_needs_copy) { gMetalGpuToCpuBytes += out_a_size; memcpy(out_a, [out_a_metal contents], out_a_size); }
        if (out_b_needs_copy) { gMetalGpuToCpuBytes += out_b_size; memcpy(out_b, [out_b_metal contents], out_b_size); }
        if (out_c_needs_copy) { gMetalGpuToCpuBytes += out_c_size; memcpy(out_c, [out_c_metal contents], out_c_size); }
        return 1;
    }
}

int rusty_metal_q4k_q4k_q6k_matvec3(const uint8_t *weights_a,
                                    uintptr_t weights_a_len,
                                    uintptr_t rows_a,
                                    const uint8_t *weights_b,
                                    uintptr_t weights_b_len,
                                    uintptr_t rows_b,
                                    const uint8_t *weights_c,
                                    uintptr_t weights_c_len,
                                    uintptr_t rows_c,
                                    const float *x,
                                    uintptr_t cols,
                                    float *out_a,
                                    float *out_b,
                                    float *out_c) {
    if (!rusty_metal_init() || !weights_a || !weights_b || !weights_c || !x || !out_a || !out_b || !out_c ||
        rows_a == 0 || rows_b == 0 || rows_c == 0 || cols == 0 || (cols % 256) != 0) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> weight_a = rusty_metal_weight_buffer(weights_a, weights_a_len);
        id<MTLBuffer> weight_b = rusty_metal_weight_buffer(weights_b, weights_b_len);
        id<MTLBuffer> weight_c = rusty_metal_weight_buffer(weights_c, weights_c_len);
        if (!weight_a || !weight_b || !weight_c) return 0;

        static id<MTLBuffer> x_buffer = nil;
        static id<MTLBuffer> out_a_buffer = nil;
        static id<MTLBuffer> out_b_buffer = nil;
        static id<MTLBuffer> out_c_buffer = nil;
        NSUInteger x_size = (NSUInteger)(cols * sizeof(float));
        NSUInteger out_a_size = (NSUInteger)(rows_a * sizeof(float));
        NSUInteger out_b_size = (NSUInteger)(rows_b * sizeof(float));
        NSUInteger out_c_size = (NSUInteger)(rows_c * sizeof(float));
        BOOL out_a_needs_copy = YES;
        BOOL out_b_needs_copy = YES;
        BOOL out_c_needs_copy = YES;

        id<MTLBuffer> x_metal = rusty_metal_input_buffer(x, x_size, &x_buffer);
        id<MTLBuffer> out_a_metal = rusty_metal_output_buffer(out_a, out_a_size, &out_a_buffer, &out_a_needs_copy);
        id<MTLBuffer> out_b_metal = rusty_metal_output_buffer(out_b, out_b_size, &out_b_buffer, &out_b_needs_copy);
        id<MTLBuffer> out_c_metal = rusty_metal_output_buffer(out_c, out_c_size, &out_c_buffer, &out_c_needs_copy);
        if (!x_metal || !out_a_metal || !out_b_metal || !out_c_metal) return 0;

        double encode_start = rusty_metal_now_seconds();
        id<MTLCommandBuffer> command_buffer = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        rusty_metal_encode_q4k(encoder, weight_a, x_metal, out_a_metal, rows_a, cols);
        rusty_metal_encode_q4k(encoder, weight_b, x_metal, out_b_metal, rows_b, cols);
        rusty_metal_encode_q6k(encoder, weight_c, x_metal, out_c_metal, rows_c, cols);
        [encoder endEncoding];
        double encode_end = rusty_metal_now_seconds();
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        rusty_metal_profile_command_buffer(command_buffer, encode_start, encode_end);
        if ([command_buffer status] != MTLCommandBufferStatusCompleted) return 0;

        if (out_a_needs_copy) { gMetalGpuToCpuBytes += out_a_size; memcpy(out_a, [out_a_metal contents], out_a_size); }
        if (out_b_needs_copy) { gMetalGpuToCpuBytes += out_b_size; memcpy(out_b, [out_b_metal contents], out_b_size); }
        if (out_c_needs_copy) { gMetalGpuToCpuBytes += out_c_size; memcpy(out_c, [out_c_metal contents], out_c_size); }
        return 1;
    }
}

int rusty_metal_q4k_q4k_q6k_ffn(const uint8_t *gate_weights,
                                uintptr_t gate_weights_len,
                                const uint8_t *up_weights,
                                uintptr_t up_weights_len,
                                const uint8_t *down_weights,
                                uintptr_t down_weights_len,
                                const float *x,
                                uintptr_t input_cols,
                                uintptr_t hidden_rows,
                                uintptr_t down_rows,
                                uintptr_t down_cols,
                                float *out) {
    if (!rusty_metal_init() || !gate_weights || !up_weights || !down_weights || !x || !out ||
        input_cols == 0 || hidden_rows == 0 || down_rows == 0 || down_cols == 0 ||
        hidden_rows != down_cols || (input_cols % 256) != 0 || (down_cols % 256) != 0) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> gate_weight = rusty_metal_weight_buffer(gate_weights, gate_weights_len);
        id<MTLBuffer> up_weight = rusty_metal_weight_buffer(up_weights, up_weights_len);
        id<MTLBuffer> down_weight = rusty_metal_weight_buffer(down_weights, down_weights_len);
        if (!gate_weight || !up_weight || !down_weight) return 0;

        static id<MTLBuffer> x_buffer = nil;
        static id<MTLBuffer> gate_buffer = nil;
        static id<MTLBuffer> up_buffer = nil;
        static id<MTLBuffer> hidden_buffer = nil;
        static id<MTLBuffer> out_buffer = nil;
        NSUInteger x_size = (NSUInteger)(input_cols * sizeof(float));
        NSUInteger hidden_size = (NSUInteger)(hidden_rows * sizeof(float));
        NSUInteger out_size = (NSUInteger)(down_rows * sizeof(float));
        BOOL out_needs_copy = YES;

        id<MTLBuffer> x_metal = rusty_metal_input_buffer(x, x_size, &x_buffer);
        if (!x_metal ||
            !rusty_metal_ensure_buffer(&gate_buffer, hidden_size) ||
            !rusty_metal_ensure_buffer(&up_buffer, hidden_size) ||
            !rusty_metal_ensure_buffer(&hidden_buffer, hidden_size)) {
            return 0;
        }
        id<MTLBuffer> out_metal = rusty_metal_output_buffer(out, out_size, &out_buffer, &out_needs_copy);
        if (!out_metal) return 0;

        double encode_start = rusty_metal_now_seconds();
        id<MTLCommandBuffer> command_buffer = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        rusty_metal_encode_q4k(encoder, gate_weight, x_metal, gate_buffer, hidden_rows, input_cols);
        rusty_metal_encode_q4k(encoder, up_weight, x_metal, up_buffer, hidden_rows, input_cols);
        rusty_metal_encode_silu_mul(encoder, gate_buffer, up_buffer, hidden_buffer, hidden_rows);
        rusty_metal_encode_q6k(encoder, down_weight, hidden_buffer, out_metal, down_rows, down_cols);
        [encoder endEncoding];
        double encode_end = rusty_metal_now_seconds();
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        rusty_metal_profile_command_buffer(command_buffer, encode_start, encode_end);
        if ([command_buffer status] != MTLCommandBufferStatusCompleted) return 0;

        if (out_needs_copy) { gMetalGpuToCpuBytes += out_size; memcpy(out, [out_metal contents], out_size); }
        return 1;
    }
}

int rusty_metal_mistral_post_attention_ffn(const uint8_t *wo_weights,
                                           uintptr_t wo_weights_len,
                                           const uint8_t *gate_weights,
                                           uintptr_t gate_weights_len,
                                           const uint8_t *up_weights,
                                           uintptr_t up_weights_len,
                                           const uint8_t *down_weights,
                                           uintptr_t down_weights_len,
                                           float *x,
                                           uintptr_t dim,
                                           const float *attn_out,
                                           uintptr_t attn_cols,
                                           const float *ffn_norm,
                                           float rms_eps,
                                           uintptr_t hidden_rows,
                                           uintptr_t down_rows,
                                           uintptr_t down_cols) {
    if (!rusty_metal_init() || !wo_weights || !gate_weights || !up_weights || !down_weights ||
        !x || !attn_out || !ffn_norm || dim == 0 || attn_cols == 0 || hidden_rows == 0 ||
        down_rows == 0 || down_cols == 0 || down_rows != dim || hidden_rows != down_cols ||
        (dim % 256) != 0 || (attn_cols % 256) != 0 || (down_cols % 256) != 0) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> wo_weight = rusty_metal_weight_buffer(wo_weights, wo_weights_len);
        id<MTLBuffer> gate_weight = rusty_metal_weight_buffer(gate_weights, gate_weights_len);
        id<MTLBuffer> up_weight = rusty_metal_weight_buffer(up_weights, up_weights_len);
        id<MTLBuffer> down_weight = rusty_metal_weight_buffer(down_weights, down_weights_len);
        if (!wo_weight || !gate_weight || !up_weight || !down_weight) return 0;

        static id<MTLBuffer> x_buffer = nil;
        static id<MTLBuffer> attn_buffer = nil;
        static id<MTLBuffer> norm_weight_buffer = nil;
        static id<MTLBuffer> proj_buffer = nil;
        static id<MTLBuffer> norm_buffer = nil;
        static id<MTLBuffer> gate_buffer = nil;
        static id<MTLBuffer> up_buffer = nil;
        static id<MTLBuffer> hidden_buffer = nil;
        NSUInteger x_size = (NSUInteger)(dim * sizeof(float));
        NSUInteger attn_size = (NSUInteger)(attn_cols * sizeof(float));
        NSUInteger hidden_size = (NSUInteger)(hidden_rows * sizeof(float));
        BOOL x_needs_copy = YES;

        id<MTLBuffer> x_metal = rusty_metal_inout_buffer(x, x_size, &x_buffer, &x_needs_copy);
        id<MTLBuffer> attn_metal = rusty_metal_input_buffer(attn_out, attn_size, &attn_buffer);
        id<MTLBuffer> norm_weight_metal = rusty_metal_input_buffer(ffn_norm, x_size, &norm_weight_buffer);
        if (!x_metal || !attn_metal || !norm_weight_metal ||
            !rusty_metal_ensure_buffer(&proj_buffer, x_size) ||
            !rusty_metal_ensure_buffer(&norm_buffer, x_size) ||
            !rusty_metal_ensure_buffer(&gate_buffer, hidden_size) ||
            !rusty_metal_ensure_buffer(&up_buffer, hidden_size) ||
            !rusty_metal_ensure_buffer(&hidden_buffer, hidden_size)) {
            return 0;
        }

        double encode_start = rusty_metal_now_seconds();
        id<MTLCommandBuffer> command_buffer = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        rusty_metal_encode_q4k(encoder, wo_weight, attn_metal, proj_buffer, dim, attn_cols);
        rusty_metal_encode_residual_rms(encoder, x_metal, proj_buffer, norm_weight_metal, norm_buffer, dim, rms_eps);
        rusty_metal_encode_q4k(encoder, gate_weight, norm_buffer, gate_buffer, hidden_rows, dim);
        rusty_metal_encode_q4k(encoder, up_weight, norm_buffer, up_buffer, hidden_rows, dim);
        rusty_metal_encode_silu_mul(encoder, gate_buffer, up_buffer, hidden_buffer, hidden_rows);
        rusty_metal_encode_q6k(encoder, down_weight, hidden_buffer, proj_buffer, down_rows, down_cols);
        rusty_metal_encode_residual_add(encoder, x_metal, proj_buffer, dim);
        [encoder endEncoding];
        double encode_end = rusty_metal_now_seconds();
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        rusty_metal_profile_command_buffer(command_buffer, encode_start, encode_end);
        if ([command_buffer status] != MTLCommandBufferStatusCompleted) return 0;

        if (x_needs_copy) { gMetalGpuToCpuBytes += x_size; memcpy(x, [x_metal contents], x_size); }
        return 1;
    }
}

// ============================================================================
// Q4_0 matvec using simdgroup reduction.
// cols must be a multiple of 32.
// ============================================================================

int rusty_metal_q4_0_matvec(const uint8_t *weights,
                            uintptr_t weights_len,
                            const float *x,
                            uintptr_t rows,
                            uintptr_t cols,
                            float *out) {
    if (!rusty_metal_init() || !weights || !x || !out || rows == 0 || cols == 0 || (cols % 32) != 0) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> weight_buffer = rusty_metal_weight_buffer(weights, weights_len);
        if (!weight_buffer) return 0;

        static id<MTLBuffer> x_buffer = nil;
        static id<MTLBuffer> out_buffer = nil;
        NSUInteger x_size = (NSUInteger)(cols * sizeof(float));
        NSUInteger out_size = (NSUInteger)(rows * sizeof(float));
        BOOL out_needs_copy = YES;

        id<MTLBuffer> x_metal = rusty_metal_input_buffer(x, x_size, &x_buffer);
        id<MTLBuffer> out_metal = rusty_metal_output_buffer(out, out_size, &out_buffer, &out_needs_copy);
        if (!x_metal || !out_metal) return 0;

        double encode_start = rusty_metal_now_seconds();
        id<MTLCommandBuffer> command_buffer = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        rusty_metal_encode_q4_0(encoder, weight_buffer, x_metal, out_metal, rows, cols);
        [encoder endEncoding];
        double encode_end = rusty_metal_now_seconds();
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        rusty_metal_profile_command_buffer(command_buffer, encode_start, encode_end);
        if ([command_buffer status] != MTLCommandBufferStatusCompleted) return 0;

        if (out_needs_copy) { gMetalGpuToCpuBytes += out_size; memcpy(out, [out_metal contents], out_size); }
        return 1;
    }
}

// ============================================================================
// Q8_0 matvec using simdgroup reduction.
// cols must be a multiple of 32.
// ============================================================================

int rusty_metal_q8_0_matvec(const uint8_t *weights,
                            uintptr_t weights_len,
                            const float *x,
                            uintptr_t rows,
                            uintptr_t cols,
                            float *out) {
    if (!rusty_metal_init() || !weights || !x || !out || rows == 0 || cols == 0 || (cols % 32) != 0) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> weight_buffer = rusty_metal_weight_buffer(weights, weights_len);
        if (!weight_buffer) return 0;

        static id<MTLBuffer> x_buffer = nil;
        static id<MTLBuffer> out_buffer = nil;
        NSUInteger x_size = (NSUInteger)(cols * sizeof(float));
        NSUInteger out_size = (NSUInteger)(rows * sizeof(float));
        BOOL out_needs_copy = YES;

        id<MTLBuffer> x_metal = rusty_metal_input_buffer(x, x_size, &x_buffer);
        id<MTLBuffer> out_metal = rusty_metal_output_buffer(out, out_size, &out_buffer, &out_needs_copy);
        if (!x_metal || !out_metal) return 0;

        double encode_start = rusty_metal_now_seconds();
        id<MTLCommandBuffer> command_buffer = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        rusty_metal_encode_q8_0(encoder, weight_buffer, x_metal, out_metal, rows, cols);
        [encoder endEncoding];
        double encode_end = rusty_metal_now_seconds();
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        rusty_metal_profile_command_buffer(command_buffer, encode_start, encode_end);
        if ([command_buffer status] != MTLCommandBufferStatusCompleted) return 0;

        if (out_needs_copy) { gMetalGpuToCpuBytes += out_size; memcpy(out, [out_metal contents], out_size); }
        return 1;
    }
}

int rusty_metal_attention(const float *query,
                          uintptr_t query_len,
                          const float *keys,
                          uintptr_t keys_len,
                          const float *values,
                          uintptr_t values_len,
                          const float *sinks,
                          uintptr_t sinks_len,
                          float *out,
                          uintptr_t out_len,
                          uintptr_t heads,
                          uintptr_t kv_mul,
                          uintptr_t head_dim,
                          uintptr_t value_dim,
                          uintptr_t key_stride,
                          uintptr_t value_stride,
                          uintptr_t slot_count,
                          uintptr_t start_t,
                          uintptr_t end_t,
                          float scale,
                          int use_sink) {
    if (!rusty_metal_init() || !query || !keys || !values || !out || heads == 0 || kv_mul == 0 ||
        head_dim == 0 || value_dim == 0 || slot_count == 0 || head_dim > 256 || value_dim > 256) {
        return 0;
    }

    uintptr_t query_bytes = heads * head_dim * sizeof(float);
    uintptr_t keys_bytes = slot_count * key_stride * sizeof(float);
    uintptr_t values_bytes = slot_count * value_stride * sizeof(float);
    uintptr_t out_bytes = heads * value_dim * sizeof(float);
    uintptr_t sinks_bytes = heads * sizeof(float);

    if (query_len < query_bytes || keys_len < keys_bytes || values_len < values_bytes ||
        out_len < out_bytes || (use_sink && (!sinks || sinks_len < sinks_bytes))) {
        return 0;
    }

    @autoreleasepool {
        static id<MTLBuffer> query_copy_buffer = nil;
        static id<MTLBuffer> out_copy_buffer = nil;
        BOOL out_needs_copy = YES;
        id<MTLBuffer> query_buffer = rusty_metal_input_buffer(query, query_bytes, &query_copy_buffer);
        id<MTLBuffer> keys_buffer = rusty_metal_shared_buffer(keys, keys_len);
        id<MTLBuffer> values_buffer = rusty_metal_shared_buffer(values, values_len);
        id<MTLBuffer> out_buffer = rusty_metal_output_buffer(out, out_bytes, &out_copy_buffer, &out_needs_copy);
        if (!query_buffer || !keys_buffer || !values_buffer || !out_buffer) {
            return 0;
        }

        id<MTLBuffer> sinks_buffer = nil;
        if (use_sink) {
            static id<MTLBuffer> sinks_copy_buffer = nil;
            sinks_buffer = rusty_metal_input_buffer(sinks, sinks_bytes, &sinks_copy_buffer);
        } else {
            sinks_buffer = gAttentionZeroBuffer;
        }
        if (!sinks_buffer) return 0;

        double encode_start = rusty_metal_now_seconds();
        id<MTLCommandBuffer> command_buffer = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        rusty_metal_encode_attention(encoder,
                                     query_buffer,
                                     keys_buffer,
                                     values_buffer,
                                     sinks_buffer,
                                     out_buffer,
                                     heads,
                                     kv_mul,
                                     head_dim,
                                     value_dim,
                                     key_stride,
                                     value_stride,
                                     slot_count,
                                     start_t,
                                     end_t,
                                     scale,
                                     use_sink);
        [encoder endEncoding];
        double encode_end = rusty_metal_now_seconds();
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        rusty_metal_profile_command_buffer(command_buffer, encode_start, encode_end);
        if ([command_buffer status] != MTLCommandBufferStatusCompleted) return 0;

        if (out_needs_copy) memcpy(out, [out_buffer contents], out_bytes);
        return 1;
    }
}

// Test shim for comparing the resident serial and temporally parallel attention
// pipelines without configuring a complete transformer. Kept out of the public
// Rust API; the macOS parity test below is its only caller.
int rusty_metal_test_resident_attention(const float *query,
                                        const float *keys,
                                        const float *values,
                                        float *out,
                                        uint32_t heads,
                                        uint32_t kv_mul,
                                        uint32_t head_dim,
                                        uint32_t value_dim,
                                        uint32_t key_stride,
                                        uint32_t value_stride,
                                        uint32_t slot_count,
                                        uint32_t start_t,
                                        uint32_t end_t,
                                        float scale,
                                        int parallel) {
    if (!rusty_metal_init() || !query || !keys || !values || !out ||
        heads == 0 || kv_mul == 0 || (heads % kv_mul) != 0 ||
        head_dim == 0 || head_dim > 256 || value_dim == 0 || value_dim > 256 ||
        slot_count == 0 || start_t > end_t || end_t >= slot_count) {
        return 0;
    }

    @autoreleasepool {
        NSUInteger query_bytes = (NSUInteger)heads * head_dim * sizeof(float);
        NSUInteger key_bytes = (NSUInteger)slot_count * key_stride * sizeof(float);
        NSUInteger value_bytes = (NSUInteger)slot_count * value_stride * sizeof(float);
        NSUInteger out_bytes = (NSUInteger)heads * value_dim * sizeof(float);
        id<MTLBuffer> query_buffer = [gDevice newBufferWithBytes:query length:query_bytes
                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> key_buffer = [gDevice newBufferWithBytes:keys length:key_bytes
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> value_buffer = [gDevice newBufferWithBytes:values length:value_bytes
                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_buffer = [gDevice newBufferWithLength:out_bytes
                                                       options:MTLResourceStorageModeShared];
        if (!query_buffer || !key_buffer || !value_buffer || !out_buffer) return 0;

        RustyResidentAttentionParams p = {
            .heads = heads, .kv_mul = kv_mul, .head_dim = head_dim, .value_dim = value_dim,
            .key_stride = key_stride, .value_stride = value_stride,
            .start_t = start_t, .end_t = end_t, .scale = scale,
        };
        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:parallel ? gResidentParallelAttentionPipeline : gResidentAttentionPipeline];
        [enc setBuffer:query_buffer offset:0 atIndex:0];
        [enc setBuffer:key_buffer offset:0 atIndex:1];
        [enc setBuffer:value_buffer offset:0 atIndex:2];
        [enc setBuffer:out_buffer offset:0 atIndex:3];
        [enc setBytes:&p length:sizeof(p) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(heads, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(parallel ? 128 : 32, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if ([cb status] != MTLCommandBufferStatusCompleted) return 0;
        memcpy(out, [out_buffer contents], out_bytes);
        return 1;
    }
}

// ─── GPU-resident single-command-buffer decoder ─────────────────────────────
//
// Runs an entire token's forward pass (embedding → N layers → final norm →
// logits) as ONE command buffer with ONE waitUntilCompleted, keeping all
// intermediates and the KV cache resident on the GPU. This removes the per-op
// CPU↔GPU serialization that otherwise makes hybrid decode slower than the CPU.
// Supports the standard LLaMA-style transformer with Q4_K/Q6_K projections.

#define RUSTY_MAX_RESIDENT_LAYERS 200

// Layout mirrors the Rust `ResidentLayerDesc` (see metal.rs).
typedef struct {
    const uint8_t *w[7]; // wq,wk,wv,wo,gate,up,down
    uintptr_t w_len[7];
    uint32_t w_rows[7];
    uint32_t w_dt[7]; // 0 = Q4_K, 1 = Q6_K
    const float *attn_norm;
    const float *ffn_norm;
    const float *bq; uint32_t bq_len;
    const float *bk; uint32_t bk_len;
    const float *bv; uint32_t bv_len;
} RustyResidentLayerDesc;

typedef struct {
    __strong id<MTLBuffer> w[7];
    uint32_t rows[7];
    uint32_t cols[7];
    uint32_t dt[7];
    __strong id<MTLBuffer> attn_norm;
    __strong id<MTLBuffer> ffn_norm;
    __strong id<MTLBuffer> bias[3];
    uint32_t bias_len[3];
    __strong id<MTLBuffer> k_cache;
    __strong id<MTLBuffer> v_cache;
} ResidentLayer;

static BOOL gResidentReady;
static uint32_t gR_nlayers, gR_dim, gR_nheads, gR_nkv, gR_headdim, gR_valuedim;
static uint32_t gR_hidden, gR_vocab, gR_storage;
static uint32_t gR_qdim, gR_kdim, gR_vdim, gR_attndim, gR_half;
static float gR_eps;
static uint32_t gR_neox, gR_outrows, gR_outdt;
static ResidentLayer gRLayers[RUSTY_MAX_RESIDENT_LAYERS];
static __strong id<MTLBuffer> gR_x, gR_xn, gR_q, gR_k, gR_v, gR_attn;
static __strong id<MTLBuffer> gR_gate, gR_up, gR_hiddenbuf, gR_proj, gR_logits;
static __strong id<MTLBuffer> gR_zero, gR_invfreq, gR_outnorm, gR_outw;

static id<MTLBuffer> resident_alloc_shared(NSUInteger bytes) {
    return [gDevice newBufferWithLength:bytes options:MTLResourceStorageModeShared];
}

static id<MTLBuffer> resident_alloc_private(NSUInteger bytes) {
    return [gDevice newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
}

static id<MTLBuffer> resident_floats(const float *data, uint32_t len) {
    return [gDevice newBufferWithBytes:data
                               length:(NSUInteger)len * sizeof(float)
                              options:MTLResourceStorageModeShared];
}

static void resident_matvec(id<MTLComputeCommandEncoder> enc, uint32_t dt,
                            id<MTLBuffer> w, id<MTLBuffer> x, id<MTLBuffer> out,
                            uint32_t rows, uint32_t cols) {
    if (dt == 1) {
        rusty_metal_encode_q6k(enc, w, x, out, rows, cols);
    } else {
        rusty_metal_encode_q4k(enc, w, x, out, rows, cols);
    }
}

static void resident_q4_pair(id<MTLComputeCommandEncoder> enc,
                             id<MTLBuffer> weight_a, id<MTLBuffer> weight_b,
                             id<MTLBuffer> x, id<MTLBuffer> out_a, id<MTLBuffer> out_b,
                             uint32_t rows_a, uint32_t rows_b, uint32_t cols) {
    rusty_metal_encode_q4k_pair(
        enc, weight_a, weight_b, x, out_a, out_b, rows_a, rows_b, cols
    );
}

static void resident_rms(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> x,
                         id<MTLBuffer> residual, id<MTLBuffer> weight,
                         id<MTLBuffer> out, uint32_t len, float eps) {
    RustyResidualNormParams p = { .len = len, .eps = eps };
    [enc setComputePipelineState:gResidualRmsPipeline];
    [enc setBuffer:x offset:0 atIndex:0];
    [enc setBuffer:residual offset:0 atIndex:1];
    [enc setBuffer:weight offset:0 atIndex:2];
    [enc setBuffer:out offset:0 atIndex:3];
    [enc setBytes:&p length:sizeof(p) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void resident_add(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> x,
                         id<MTLBuffer> residual, uint32_t len) {
    RustyUnaryParams p = { .len = len };
    [enc setComputePipelineState:gResidualAddPipeline];
    [enc setBuffer:x offset:0 atIndex:0];
    [enc setBuffer:residual offset:0 atIndex:1];
    [enc setBytes:&p length:sizeof(p) atIndex:2];
    NSUInteger groups = ((NSUInteger)len + 255) / 256;
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void resident_silu(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> gate,
                          id<MTLBuffer> up, id<MTLBuffer> out, uint32_t len) {
    RustyUnaryParams p = { .len = len };
    [enc setComputePipelineState:gSiluMulPipeline];
    [enc setBuffer:gate offset:0 atIndex:0];
    [enc setBuffer:up offset:0 atIndex:1];
    [enc setBuffer:out offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    NSUInteger groups = ((NSUInteger)len + 255) / 256;
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void resident_rope(id<MTLComputeCommandEncoder> enc, ResidentLayer *L,
                          uint32_t pos, uint32_t slot) {
    RustyRopeParams p = {
        .pos = pos, .head_dim = gR_headdim, .half_dim = gR_half,
        .n_heads = gR_nheads, .n_kv_heads = gR_nkv, .value_dim = gR_valuedim,
        .kv_k_dim = gR_kdim, .kv_v_dim = gR_vdim, .slot = slot, .neox = gR_neox,
    };
    [enc setComputePipelineState:gRopeStorePipeline];
    [enc setBuffer:gR_q offset:0 atIndex:0];
    [enc setBuffer:gR_k offset:0 atIndex:1];
    [enc setBuffer:gR_v offset:0 atIndex:2];
    [enc setBuffer:gR_invfreq offset:0 atIndex:3];
    [enc setBuffer:L->k_cache offset:0 atIndex:4];
    [enc setBuffer:L->v_cache offset:0 atIndex:5];
    [enc setBytes:&p length:sizeof(p) atIndex:6];
    uint32_t work = gR_nheads * gR_half;
    if (gR_vdim > work) work = gR_vdim;
    NSUInteger groups = ((NSUInteger)work + 255) / 256;
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void resident_attn(id<MTLComputeCommandEncoder> enc, ResidentLayer *L,
                          uint32_t start_t, uint32_t end_t, float scale) {
    uint32_t kv_mul = gR_nheads / gR_nkv;
    RustyResidentAttentionParams p = {
        .heads = gR_nheads, .kv_mul = kv_mul, .head_dim = gR_headdim,
        .value_dim = gR_valuedim, .key_stride = gR_kdim, .value_stride = gR_vdim,
        .start_t = start_t, .end_t = end_t, .scale = scale,
    };
    // At short contexts the extra threadgroup barriers cost more than the KV
    // reuse saves. Once the attention scan reaches 128 tokens, grouping four
    // query heads around one KV head avoids enough repeated cache traffic to
    // win decisively on Ministral 3. The environment flag remains useful for
    // forcing an A/B comparison on other Apple GPUs.
    BOOL grouped_override_on = rusty_metal_env_enabled("RUSTY_LLM_METAL_GROUPED_GQA");
    BOOL grouped_override_off = rusty_metal_env_disabled("RUSTY_LLM_METAL_GROUPED_GQA");
    BOOL parallel_override_on = rusty_metal_env_enabled("RUSTY_LLM_METAL_PARALLEL_ATTN");
    BOOL parallel_override_off = rusty_metal_env_disabled("RUSTY_LLM_METAL_PARALLEL_ATTN");
    uint32_t context_tokens = end_t - start_t + 1;
    BOOL use_parallel = gResidentParallelAttentionPipeline != nil &&
                        (parallel_override_on ||
                         (!parallel_override_off && context_tokens >= 16));
    BOOL use_grouped_gqa = !use_parallel && kv_mul == 4 && gResidentGroupedAttentionPipeline != nil &&
                           (grouped_override_on || (!grouped_override_off && context_tokens >= 128));
    [enc setComputePipelineState:use_parallel ? gResidentParallelAttentionPipeline :
                                 (use_grouped_gqa ? gResidentGroupedAttentionPipeline : gResidentAttentionPipeline)];
    [enc setBuffer:gR_q offset:0 atIndex:0];
    [enc setBuffer:L->k_cache offset:0 atIndex:1];
    [enc setBuffer:L->v_cache offset:0 atIndex:2];
    [enc setBuffer:gR_attn offset:0 atIndex:3];
    [enc setBytes:&p length:sizeof(p) atIndex:4];
    if (use_parallel) {
        [enc dispatchThreadgroups:MTLSizeMake(gR_nheads, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    } else if (use_grouped_gqa) {
        [enc dispatchThreadgroups:MTLSizeMake(gR_nkv, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    } else {
        [enc dispatchThreadgroups:MTLSizeMake(gR_nheads, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }
}

int rusty_metal_resident_configure(uint32_t n_layers, uint32_t dim, uint32_t n_heads,
                                   uint32_t n_kv_heads, uint32_t head_dim, uint32_t value_dim,
                                   uint32_t hidden_dim, uint32_t vocab, uint32_t storage_len,
                                   float eps, uint32_t neox) {
    if (!rusty_metal_init()) return 0;
    if (n_layers == 0 || n_layers > RUSTY_MAX_RESIDENT_LAYERS) return 0;
    if (n_kv_heads == 0 || (n_heads % n_kv_heads) != 0) return 0;
    if (head_dim == 0 || head_dim > 256 || value_dim == 0 || value_dim > 256) return 0;
    if ((dim % 256) != 0 || (hidden_dim % 256) != 0 || storage_len == 0) return 0;
    gResidentReady = NO;
    gR_nlayers = n_layers; gR_dim = dim; gR_nheads = n_heads; gR_nkv = n_kv_heads;
    gR_headdim = head_dim; gR_valuedim = value_dim; gR_hidden = hidden_dim;
    gR_vocab = vocab; gR_storage = storage_len; gR_eps = eps; gR_neox = neox;
    gR_qdim = n_heads * head_dim; gR_kdim = n_kv_heads * head_dim;
    gR_vdim = n_kv_heads * value_dim; gR_attndim = n_heads * value_dim;
    gR_half = head_dim / 2;
    gR_x = resident_alloc_shared((NSUInteger)dim * sizeof(float));
    gR_xn = resident_alloc_private((NSUInteger)dim * sizeof(float));
    gR_q = resident_alloc_private((NSUInteger)gR_qdim * sizeof(float));
    gR_k = resident_alloc_private((NSUInteger)gR_kdim * sizeof(float));
    gR_v = resident_alloc_private((NSUInteger)gR_vdim * sizeof(float));
    gR_attn = resident_alloc_private((NSUInteger)gR_attndim * sizeof(float));
    gR_gate = resident_alloc_private((NSUInteger)hidden_dim * sizeof(float));
    gR_up = resident_alloc_private((NSUInteger)hidden_dim * sizeof(float));
    gR_hiddenbuf = resident_alloc_private((NSUInteger)hidden_dim * sizeof(float));
    gR_proj = resident_alloc_private((NSUInteger)dim * sizeof(float));
    gR_logits = resident_alloc_shared((NSUInteger)vocab * sizeof(float));
    gR_zero = resident_alloc_shared((NSUInteger)dim * sizeof(float));
    if (!gR_x || !gR_xn || !gR_q || !gR_k || !gR_v || !gR_attn || !gR_gate || !gR_up ||
        !gR_hiddenbuf || !gR_proj || !gR_logits || !gR_zero) {
        return 0;
    }
    memset([gR_zero contents], 0, (NSUInteger)dim * sizeof(float));
    return 1;
}

int rusty_metal_resident_set_layer(uint32_t l, const RustyResidentLayerDesc *d) {
    if (!gDevice || !d || l >= gR_nlayers) return 0;
    ResidentLayer *L = &gRLayers[l];
    uint32_t cols[7] = { gR_dim, gR_dim, gR_dim, gR_attndim, gR_dim, gR_dim, gR_hidden };
    for (int i = 0; i < 7; ++i) {
        if (!d->w[i] || (cols[i] % 256) != 0) return 0;
        id<MTLBuffer> wb = rusty_metal_weight_buffer(d->w[i], d->w_len[i]);
        if (!wb) return 0;
        L->w[i] = wb;
        L->rows[i] = d->w_rows[i];
        L->cols[i] = cols[i];
        L->dt[i] = d->w_dt[i];
    }
    if (!d->attn_norm || !d->ffn_norm) return 0;
    L->attn_norm = resident_floats(d->attn_norm, gR_dim);
    L->ffn_norm = resident_floats(d->ffn_norm, gR_dim);
    L->bias[0] = d->bq ? resident_floats(d->bq, d->bq_len) : nil;
    L->bias[1] = d->bk ? resident_floats(d->bk, d->bk_len) : nil;
    L->bias[2] = d->bv ? resident_floats(d->bv, d->bv_len) : nil;
    L->bias_len[0] = d->bq ? d->bq_len : 0;
    L->bias_len[1] = d->bk ? d->bk_len : 0;
    L->bias_len[2] = d->bv ? d->bv_len : 0;
    L->k_cache = resident_alloc_private((NSUInteger)gR_storage * gR_kdim * sizeof(float));
    L->v_cache = resident_alloc_private((NSUInteger)gR_storage * gR_vdim * sizeof(float));
    if (!L->attn_norm || !L->ffn_norm || !L->k_cache || !L->v_cache) return 0;
    return 1;
}

int rusty_metal_resident_set_output(const float *output_norm, const uint8_t *output_w,
                                    uintptr_t output_w_len, uint32_t output_rows,
                                    uint32_t output_dt, const float *inv_freq,
                                    uint32_t inv_freq_len) {
    if (!gDevice || !output_norm || !output_w || !inv_freq) return 0;
    gR_outnorm = resident_floats(output_norm, gR_dim);
    id<MTLBuffer> ow = rusty_metal_weight_buffer(output_w, output_w_len);
    if (!ow) return 0;
    gR_outw = ow;
    gR_outrows = output_rows;
    gR_outdt = output_dt;
    gR_invfreq = resident_floats(inv_freq, inv_freq_len);
    if (!gR_outnorm || !gR_invfreq) return 0;
    gResidentReady = YES;
    return 1;
}

// `want_logits` == 0 runs the layer stack for its KV-cache side effects only and
// skips the final norm plus the vocabulary projection. Prompt tokens need the
// cache filled but throw their logits away, and that projection is the single
// largest weight read in the model (330 MiB for a 131072-entry Q6_K vocabulary),
// so skipping it removes both the read and the device->host copy from every
// prefill token. It is safe to skip because each call re-seeds gR_x from the
// caller's embedding, so nothing carries over from the tail of the graph; only
// the in-layer RoPE/attention KV writes have to happen.
int rusty_metal_resident_decode(const float *x_embed, uint32_t pos, uint32_t start_t,
                                int want_logits, float *logits_out) {
    if (!gResidentReady || !x_embed || pos >= gR_storage) return 0;
    if (want_logits && !logits_out) return 0;
    @autoreleasepool {
        memcpy([gR_x contents], x_embed, (NSUInteger)gR_dim * sizeof(float));
        uint32_t slot = pos;
        float scale = 1.0f / sqrt((float)gR_headdim);
        double encode_start = rusty_metal_now_seconds();
        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        for (uint32_t l = 0; l < gR_nlayers; ++l) {
            ResidentLayer *L = &gRLayers[l];
            id<MTLBuffer> resid = (l == 0) ? gR_zero : gR_proj;
            resident_rms(enc, gR_x, resid, L->attn_norm, gR_xn, gR_dim, gR_eps);
            if (L->dt[0] == 0 && L->dt[1] == 0) {
                resident_q4_pair(
                    enc, L->w[0], L->w[1], gR_xn, gR_q, gR_k,
                    L->rows[0], L->rows[1], L->cols[0]
                );
            } else {
                resident_matvec(enc, L->dt[0], L->w[0], gR_xn, gR_q, L->rows[0], L->cols[0]);
                resident_matvec(enc, L->dt[1], L->w[1], gR_xn, gR_k, L->rows[1], L->cols[1]);
            }
            resident_matvec(enc, L->dt[2], L->w[2], gR_xn, gR_v, L->rows[2], L->cols[2]);
            if (L->bias_len[0]) resident_add(enc, gR_q, L->bias[0], L->bias_len[0]);
            if (L->bias_len[1]) resident_add(enc, gR_k, L->bias[1], L->bias_len[1]);
            if (L->bias_len[2]) resident_add(enc, gR_v, L->bias[2], L->bias_len[2]);
            resident_rope(enc, L, pos, slot);
            resident_attn(enc, L, start_t, pos, scale);
            resident_matvec(enc, L->dt[3], L->w[3], gR_attn, gR_proj, L->rows[3], L->cols[3]);
            resident_rms(enc, gR_x, gR_proj, L->ffn_norm, gR_xn, gR_dim, gR_eps);
            if (L->dt[4] == 0 && L->dt[5] == 0) {
                resident_q4_pair(
                    enc, L->w[4], L->w[5], gR_xn, gR_gate, gR_up,
                    L->rows[4], L->rows[5], L->cols[4]
                );
            } else {
                resident_matvec(enc, L->dt[4], L->w[4], gR_xn, gR_gate, L->rows[4], L->cols[4]);
                resident_matvec(enc, L->dt[5], L->w[5], gR_xn, gR_up, L->rows[5], L->cols[5]);
            }
            resident_silu(enc, gR_gate, gR_up, gR_hiddenbuf, gR_hidden);
            resident_matvec(enc, L->dt[6], L->w[6], gR_hiddenbuf, gR_proj, L->rows[6], L->cols[6]);
        }
        if (want_logits) {
            resident_rms(enc, gR_x, gR_proj, gR_outnorm, gR_xn, gR_dim, gR_eps);
            resident_matvec(enc, gR_outdt, gR_outw, gR_xn, gR_logits, gR_outrows, gR_dim);
        }
        [enc endEncoding];
        double encode_end = rusty_metal_now_seconds();
        [cb commit];
        [cb waitUntilCompleted];
        rusty_metal_profile_command_buffer(cb, encode_start, encode_end);
        if ([cb status] != MTLCommandBufferStatusCompleted) return 0;
        if (want_logits) {
            memcpy(logits_out, [gR_logits contents], (NSUInteger)gR_vocab * sizeof(float));
        }
        return 1;
    }
}
