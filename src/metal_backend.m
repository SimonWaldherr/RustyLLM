#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "rusty_metallib.h"

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
    uint32_t apply_gate;
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
    uint32_t vocab;
    uint32_t recent_len;
    uint32_t groups;
    float repeat_penalty;
} RustyArgmaxParams;

typedef struct {
    const void *key;
    uintptr_t len;
    __strong id<MTLBuffer> buffer;
} RustyWeightCacheEntry;

enum {
    RUSTY_WEIGHT_CACHE_SIZE = 8192,
    RUSTY_ARGMAX_GROUPS = 128,
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
static id<MTLComputePipelineState> gGreedyArgmaxPipeline;
static id<MTLComputePipelineState> gGreedyArgmaxStage1Pipeline;
static id<MTLComputePipelineState> gGreedyArgmaxStage2Pipeline;
static id<MTLComputePipelineState> gQwenConvSiluPipeline;
static id<MTLComputePipelineState> gQwenL2NormPipeline;
static id<MTLComputePipelineState> gQwenDeltaPipeline;
static id<MTLComputePipelineState> gQwenDeltaNormGatePipeline;
static id<MTLComputePipelineState> gQwenAttentionNormSplitPipeline;
static id<MTLComputePipelineState> gQwenSigmoidGatePipeline;
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
"struct Q4KParams { uint rows; uint cols; uint row_bytes; uint n_blocks; uint rows_per_group; };\n"
"inline uchar2 scale_min_k4(uint j, const device uchar* q) {\n"
"    if (j < 4) return uchar2(q[j] & 63, q[j + 4] & 63);\n"
"    return uchar2((q[j + 4] & 15) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4));\n"
"}\n"
"// Four independent eight-lane teams process four quant blocks at once.\n"
"// Match ggml's decode geometry: every SIMD group reuses its activation\n"
"// registers across eight rows, and two SIMD groups cover 16 rows.\n"
"kernel void q4k_matvec(device const uchar* weights [[buffer(0)]],\n"
"                       device const float* x [[buffer(1)]],\n"
"                       device float* out [[buffer(2)]],\n"
"                       constant Q4KParams& p [[buffer(3)]],\n"
"                       uint group [[threadgroup_position_in_grid]],\n"
"                       uint sg [[simdgroup_index_in_threadgroup]],\n"
"                       uint lane [[thread_index_in_simdgroup]]) {\n"
"    constexpr uint rows_per_simdgroup = 8;\n"
"    uint first_row = group * p.rows_per_group + sg * rows_per_simdgroup;\n"
"    uint block_team = lane >> 3;\n"
"    uint value_base = (lane & 7) * 4;\n"
"    float lane_sum[rows_per_simdgroup] = { 0.0f, 0.0f, 0.0f, 0.0f,\n"
"                                                0.0f, 0.0f, 0.0f, 0.0f };\n"
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
"        for (uint row_slot = 0; row_slot < rows_per_simdgroup; ++row_slot) {\n"
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
"    for (uint row_slot = 0; row_slot < rows_per_simdgroup; ++row_slot) {\n"
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
"    constexpr uint rows_per_simdgroup = 8;\n"
"    uint first_row = local_group * pp.rows_per_group + sg * rows_per_simdgroup;\n"
"    uint block_team = lane >> 3;\n"
"    uint value_base = (lane & 7) * 4;\n"
"    float lane_sum[rows_per_simdgroup] = { 0.0f, 0.0f, 0.0f, 0.0f,\n"
"                                                0.0f, 0.0f, 0.0f, 0.0f };\n"
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
"        for (uint row_slot = 0; row_slot < rows_per_simdgroup; ++row_slot) {\n"
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
"    for (uint row_slot = 0; row_slot < rows_per_simdgroup; ++row_slot) {\n"
"        float total = simd_sum(lane_sum[row_slot]);\n"
"        uint row = first_row + row_slot;\n"
"        if (lane == 0 && row < rows) out[row] = total;\n"
"    }\n"
"}\n";

static NSString *const kQ6KSource =
@"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct Q6KParams { uint rows; uint cols; uint row_bytes; uint n_blocks; uint rows_per_group; };\n"
"// Four independent eight-lane teams process four quant blocks at once.\n"
"// Each lane rebuilds four adjacent values and applies them to two rows.\n"
"kernel void q6k_matvec(device const uchar* weights [[buffer(0)]],\n"
"                       device const float* x [[buffer(1)]],\n"
"                       device float* out [[buffer(2)]],\n"
"                       constant Q6KParams& p [[buffer(3)]],\n"
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
"struct Q40Params { uint rows; uint cols; uint row_bytes; uint n_blocks; uint rows_per_group; };\n"
"kernel void q4_0_matvec(device const uchar* weights [[buffer(0)]],\n"
"                        device const float* x [[buffer(1)]],\n"
"                        device float* out [[buffer(2)]],\n"
"                        constant Q40Params& p [[buffer(3)]],\n"
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
"struct Q80Params { uint rows; uint cols; uint row_bytes; uint n_blocks; uint rows_per_group; };\n"
"kernel void q8_0_matvec(device const uchar* weights [[buffer(0)]],\n"
"                        device const float* x [[buffer(1)]],\n"
"                        device float* out [[buffer(2)]],\n"
"                        constant Q80Params& p [[buffer(3)]],\n"
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
"struct ScanParams { uint heads; uint kv_mul; uint head_dim; uint value_dim; uint key_stride; uint value_stride; uint slot_count; uint start_t; uint end_t; uint use_sink; float scale; };\n"
"kernel void attention_scan(device const float* query [[buffer(0)]],\n"
"                           device const float* keys [[buffer(1)]],\n"
"                           device const float* values [[buffer(2)]],\n"
"                           device float* out [[buffer(3)]],\n"
"                           device const float* sinks [[buffer(4)]],\n"
"                           constant ScanParams& p [[buffer(5)]],\n"
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
"struct ActivationParams { uint len; };\n"
"kernel void silu_mul(device const float* gate [[buffer(0)]],\n"
"                     device const float* up [[buffer(1)]],\n"
"                     device float* out [[buffer(2)]],\n"
"                     constant ActivationParams& p [[buffer(3)]],\n"
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
"    threadgroup float partial[8];\n"
"    uint lane = tid & 31;\n"
"    uint sg = tid >> 5;\n"
"    float sum = 0.0f;\n"
"    for (uint i = tid; i < p.len; i += 256) {\n"
"        float v = x[i] + residual[i];\n"
"        x[i] = v;\n"
"        sum += v * v;\n"
"    }\n"
"    sum = simd_sum(sum);\n"
"    if (lane == 0) partial[sg] = sum;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (sg == 0) {\n"
"        float total = lane < 8 ? partial[lane] : 0.0f;\n"
"        total = simd_sum(total);\n"
"        if (lane == 0) partial[0] = total;\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
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
"    // Partial RoPE leaves the remainder of every K head unchanged.  Those\n"
"    // values still belong in the cache; copying only the rotated prefix\n"
"    // makes attention incorrect from the second token onward.\n"
"    uint rotated_dim = 2 * p.half_dim;\n"
"    uint tail_dim = p.head_dim - rotated_dim;\n"
"    uint tail_total = p.n_kv_heads * tail_dim;\n"
"    if (gid < tail_total) {\n"
"        uint h = gid / tail_dim;\n"
"        uint d = rotated_dim + gid % tail_dim;\n"
"        uint off = h * p.head_dim + d;\n"
"        k_cache[p.slot * p.kv_k_dim + off] = k[off];\n"
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
"struct ResidentAttnParams { uint n_heads; uint kv_mul; uint head_dim; uint value_dim; uint apply_gate;\n"
"                    uint kv_k_dim; uint kv_v_dim; uint start_t; uint end_t; float scale; };\n"
"kernel void resident_attention(device const float* q [[buffer(0)]],\n"
"                               device const float* k_cache [[buffer(1)]],\n"
"                               device const float* v_cache [[buffer(2)]],\n"
"                               device float* out [[buffer(3)]],\n"
"                               constant ResidentAttnParams& p [[buffer(4)]],\n"
"                               device const float* gate [[buffer(5)]],\n"
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
"    for (uint i = lane; i < p.value_dim; i += 32) {\n"
"        float value = osh[i] * inv;\n"
"        if (p.apply_gate != 0) { float z = gate[head * p.value_dim + i]; value *= 1.0f / (1.0f + exp(-z)); }\n"
"        orow[i] = value;\n"
"    }\n"
"}\n";

// Short-context resident attention parallelized across four SIMD groups per
// query head. Each SIMD group scans a disjoint subset of KV positions with an
// online softmax; the four partial softmax states are merged exactly at the end.
// This trades additional query-head KV reads for 4x more temporal parallelism,
// which is profitable before the cache scan becomes memory-bandwidth bound.
static NSString *const kResidentParallelAttnSource =
@"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct ParallelAttnParams { uint n_heads; uint kv_mul; uint head_dim; uint value_dim; uint apply_gate;\n"
"                    uint kv_k_dim; uint kv_v_dim; uint start_t; uint end_t; float scale; };\n"
"kernel void resident_parallel_attention(device const float* q [[buffer(0)]],\n"
"                                        device const float* k_cache [[buffer(1)]],\n"
"                                        device const float* v_cache [[buffer(2)]],\n"
"                                        device float* out [[buffer(3)]],\n"
"                                        constant ParallelAttnParams& p [[buffer(4)]],\n"
"                                        device const float* gate [[buffer(5)]],\n"
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
"            float value = total * inv;\n"
"            if (p.apply_gate != 0) { float z = gate[head * p.value_dim + i]; value *= 1.0f / (1.0f + exp(-z)); }\n"
"            orow[i] = value;\n"
"        }\n"
"    }\n"
"}\n";

// Four-head grouped-query attention for the resident decoder. Ministral 3
// maps four query heads to each KV head, so loading the KV row once into
// threadgroup memory avoids streaming the same keys and values four times.
static NSString *const kResidentGroupedAttnSource =
@"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct GroupedAttnParams { uint n_heads; uint kv_mul; uint head_dim; uint value_dim; uint apply_gate;\n"
"                    uint kv_k_dim; uint kv_v_dim; uint start_t; uint end_t; float scale; };\n"
"kernel void resident_gqa4_attention(device const float* q [[buffer(0)]],\n"
"                                    device const float* k_cache [[buffer(1)]],\n"
"                                    device const float* v_cache [[buffer(2)]],\n"
"                                    device float* out [[buffer(3)]],\n"
"                                    constant GroupedAttnParams& p [[buffer(4)]],\n"
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

// Applies the exact greedy repetition penalty and reduces the vocabulary to a
// single token on-device. Equal logits select the lower token id.
static NSString *const kGreedyArgmaxSource =
@"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct GreedyParams { uint vocab; uint recent_len; uint groups; float repeat_penalty; };\n"
"kernel void greedy_argmax(device const float* logits [[buffer(0)]],\n"
"                          device const uint* recent [[buffer(1)]],\n"
"                          device uint* selected [[buffer(2)]],\n"
"                          constant GreedyParams& p [[buffer(3)]],\n"
"                          uint tid [[thread_index_in_threadgroup]],\n"
"                          uint lane [[thread_index_in_simdgroup]],\n"
"                          uint sg [[simdgroup_index_in_threadgroup]]) {\n"
"    float best = -INFINITY;\n"
"    uint best_id = 0;\n"
"    for (uint token = tid; token < p.vocab; token += 256) {\n"
"        float value = logits[token];\n"
"        if (!isfinite(value)) value = -INFINITY;\n"
"        if (p.repeat_penalty != 1.0f) {\n"
"            for (uint r = 0; r < p.recent_len; ++r) {\n"
"                if (recent[r] == token) {\n"
"                    value = value > 0.0f ? value / p.repeat_penalty : value * p.repeat_penalty;\n"
"                }\n"
"            }\n"
"        }\n"
"        if (value > best || (value == best && token < best_id)) { best = value; best_id = token; }\n"
"    }\n"
"    for (ushort delta = 16; delta > 0; delta >>= 1) {\n"
"        float other = simd_shuffle_down(best, delta);\n"
"        uint other_id = simd_shuffle_down(best_id, delta);\n"
"        if (other > best || (other == best && other_id < best_id)) { best = other; best_id = other_id; }\n"
"    }\n"
"    threadgroup float group_value[8];\n"
"    threadgroup uint group_id[8];\n"
"    if (lane == 0) { group_value[sg] = best; group_id[sg] = best_id; }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (sg == 0) {\n"
"        best = lane < 8 ? group_value[lane] : -INFINITY;\n"
"        best_id = lane < 8 ? group_id[lane] : 0xffffffffu;\n"
"        for (ushort delta = 16; delta > 0; delta >>= 1) {\n"
"            float other = simd_shuffle_down(best, delta);\n"
"            uint other_id = simd_shuffle_down(best_id, delta);\n"
"            if (other > best || (other == best && other_id < best_id)) { best = other; best_id = other_id; }\n"
"        }\n"
"        if (lane == 0) selected[0] = best_id;\n"
"    }\n"
"}\n"
"kernel void greedy_argmax_stage1(device const float* logits [[buffer(0)]],\n"
"                                 device const uint* recent [[buffer(1)]],\n"
"                                 device float* partial_value [[buffer(2)]],\n"
"                                 device uint* partial_id [[buffer(3)]],\n"
"                                 constant GreedyParams& p [[buffer(4)]],\n"
"                                 uint group [[threadgroup_position_in_grid]],\n"
"                                 uint tid [[thread_index_in_threadgroup]],\n"
"                                 uint lane [[thread_index_in_simdgroup]],\n"
"                                 uint sg [[simdgroup_index_in_threadgroup]]) {\n"
"    float best = -INFINITY;\n"
"    uint best_id = 0xffffffffu;\n"
"    uint stride = p.groups * 256;\n"
"    for (uint token = group * 256 + tid; token < p.vocab; token += stride) {\n"
"        float value = logits[token];\n"
"        if (!isfinite(value)) value = -INFINITY;\n"
"        if (p.repeat_penalty != 1.0f) {\n"
"            for (uint r = 0; r < p.recent_len; ++r) {\n"
"                if (recent[r] == token) value = value > 0.0f ? value / p.repeat_penalty : value * p.repeat_penalty;\n"
"            }\n"
"        }\n"
"        if (value > best || (value == best && token < best_id)) { best = value; best_id = token; }\n"
"    }\n"
"    for (ushort delta = 16; delta > 0; delta >>= 1) {\n"
"        float other = simd_shuffle_down(best, delta);\n"
"        uint other_id = simd_shuffle_down(best_id, delta);\n"
"        if (other > best || (other == best && other_id < best_id)) { best = other; best_id = other_id; }\n"
"    }\n"
"    threadgroup float group_value[8];\n"
"    threadgroup uint group_id[8];\n"
"    if (lane == 0) { group_value[sg] = best; group_id[sg] = best_id; }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (sg == 0) {\n"
"        best = lane < 8 ? group_value[lane] : -INFINITY;\n"
"        best_id = lane < 8 ? group_id[lane] : 0xffffffffu;\n"
"        for (ushort delta = 16; delta > 0; delta >>= 1) {\n"
"            float other = simd_shuffle_down(best, delta);\n"
"            uint other_id = simd_shuffle_down(best_id, delta);\n"
"            if (other > best || (other == best && other_id < best_id)) { best = other; best_id = other_id; }\n"
"        }\n"
"        if (lane == 0) { partial_value[group] = best; partial_id[group] = best_id; }\n"
"    }\n"
"}\n"
"kernel void greedy_argmax_stage2(device const float* partial_value [[buffer(0)]],\n"
"                                 device const uint* partial_id [[buffer(1)]],\n"
"                                 device uint* selected [[buffer(2)]],\n"
"                                 constant GreedyParams& p [[buffer(3)]],\n"
"                                 uint tid [[thread_index_in_threadgroup]],\n"
"                                 uint lane [[thread_index_in_simdgroup]],\n"
"                                 uint sg [[simdgroup_index_in_threadgroup]]) {\n"
"    float best = -INFINITY;\n"
"    uint best_id = 0xffffffffu;\n"
"    for (uint i = tid; i < p.groups; i += 256) {\n"
"        float value = partial_value[i];\n"
"        uint token = partial_id[i];\n"
"        if (value > best || (value == best && token < best_id)) { best = value; best_id = token; }\n"
"    }\n"
"    for (ushort delta = 16; delta > 0; delta >>= 1) {\n"
"        float other = simd_shuffle_down(best, delta);\n"
"        uint other_id = simd_shuffle_down(best_id, delta);\n"
"        if (other > best || (other == best && other_id < best_id)) { best = other; best_id = other_id; }\n"
"    }\n"
"    threadgroup float group_value[8];\n"
"    threadgroup uint group_id[8];\n"
"    if (lane == 0) { group_value[sg] = best; group_id[sg] = best_id; }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (sg == 0) {\n"
"        best = lane < 8 ? group_value[lane] : -INFINITY;\n"
"        best_id = lane < 8 ? group_id[lane] : 0xffffffffu;\n"
"        for (ushort delta = 16; delta > 0; delta >>= 1) {\n"
"            float other = simd_shuffle_down(best, delta);\n"
"            uint other_id = simd_shuffle_down(best_id, delta);\n"
"            if (other > best || (other == best && other_id < best_id)) { best = other; best_id = other_id; }\n"
"        }\n"
"        if (lane == 0) selected[0] = best_id;\n"
"    }\n"
"}\n";

// Qwen3.5/Qwen3.8 resident-only elementwise and recurrent kernels.  All large
// projections continue to use the shared Q4_K/Q6_K kernels above; these small
// kernels keep the hybrid block's activations and recurrent state on-device so
// the complete 64-layer graph can be encoded in one command buffer.
static NSString *const kQwenResidentSource =
@"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct QwenConvParams { uint conv_dim; uint d_conv; uint value_heads; uint key_heads; uint head_dim; float eps; };\n"
"struct QwenNormParams { uint heads; uint head_dim; float eps; };\n"
"struct QwenDeltaParams { uint value_heads; uint key_heads; uint head_dim; };\n"
"struct QwenAttentionNormParams { uint query_heads; uint key_heads; uint head_dim; float eps; };\n"
"struct QwenUnaryParams { uint len; };\n"
"kernel void qwen_conv_silu(device float* qkv [[buffer(0)]],\n"
"                            device const float* conv_w [[buffer(1)]],\n"
"                            device float* history [[buffer(2)]],\n"
"                            constant QwenConvParams& p [[buffer(3)]],\n"
"                            device float* alpha [[buffer(4)]],\n"
"                            device float* beta [[buffer(5)]],\n"
"                            device const float* a [[buffer(6)]],\n"
"                            device const float* dt_bias [[buffer(7)]],\n"
"                            uint group [[threadgroup_position_in_grid]],\n"
"                            uint lane [[thread_index_in_simdgroup]]) {\n"
"    uint head_count = 2 * p.key_heads + p.value_heads;\n"
"    if (group >= head_count || p.head_dim != 128 || p.d_conv < 2) return;\n"
"    uint history_len = p.d_conv - 1;\n"
"    float values[4];\n"
"    float sum = 0.0f;\n"
"    #pragma unroll\n"
"    for (uint j = 0; j < 4; ++j) {\n"
"        uint gid = group * p.head_dim + lane + j * 32;\n"
"        uint hbase = gid * history_len;\n"
"        uint wbase = gid * p.d_conv;\n"
"        float current = qkv[gid];\n"
"        float value = current * conv_w[wbase + history_len];\n"
"        for (uint i = 0; i < history_len; ++i) value += history[hbase + i] * conv_w[wbase + i];\n"
"        for (uint i = 1; i < history_len; ++i) history[hbase + i - 1] = history[hbase + i];\n"
"        history[hbase + history_len - 1] = current;\n"
"        value = value / (1.0f + exp(-value));\n"
"        values[j] = value;\n"
"        sum += value * value;\n"
"    }\n"
"    float denom = group < 2 * p.key_heads ? max(sqrt(simd_sum(sum)), p.eps) : 1.0f;\n"
"    #pragma unroll\n"
"    for (uint j = 0; j < 4; ++j) qkv[group * p.head_dim + lane + j * 32] = values[j] / denom;\n"
"    if (lane == 0 && group < p.value_heads) {\n"
"        float raw_alpha = alpha[group] + dt_bias[group];\n"
"        float step = raw_alpha > 20.0f ? raw_alpha : log(1.0f + exp(raw_alpha));\n"
"        alpha[group] = exp(a[group] * step);\n"
"        beta[group] = 1.0f / (1.0f + exp(-beta[group]));\n"
"    }\n"
"}\n"
"kernel void qwen_l2_norm_qk(device float* qkv [[buffer(0)]],\n"
"                            constant QwenNormParams& p [[buffer(1)]],\n"
"                            uint group [[threadgroup_position_in_grid]],\n"
"                            uint tid [[thread_index_in_threadgroup]]) {\n"
"    if (group >= 2 * p.heads) return;\n"
"    uint head = group % p.heads;\n"
"    uint base = head * p.head_dim + (group >= p.heads ? p.heads * p.head_dim : 0);\n"
"    float sum = 0.0f;\n"
"    for (uint i = tid; i < p.head_dim; i += 32) { float v = qkv[base + i]; sum += v * v; }\n"
"    float denom = max(sqrt(simd_sum(sum)), p.eps);\n"
"    for (uint i = tid; i < p.head_dim; i += 32) qkv[base + i] /= denom;\n"
"}\n"
"kernel void qwen_delta_step(device const float* qkv [[buffer(0)]],\n"
"                            device const float* alpha_in [[buffer(1)]],\n"
"                            device const float* beta_in [[buffer(2)]],\n"
"                            device float* state [[buffer(3)]],\n"
"                            device float* out [[buffer(4)]],\n"
"                            constant QwenDeltaParams& p [[buffer(5)]],\n"
"                            uint2 group [[threadgroup_position_in_grid]],\n"
"                            uint2 tid [[thread_position_in_threadgroup]]) {\n"
"    uint value_head = group.y;\n"
"    if (value_head >= p.value_heads) return;\n"
"    constexpr uint values_per_lane = 4;\n"
"    uint lane = tid.x;\n"
"    uint row = group.x * 4 + tid.y;\n"
"    if (row >= p.head_dim) return;\n"
"    uint key_head = value_head % p.key_heads;\n"
"    uint key_dim = p.key_heads * p.head_dim;\n"
"    const device float* q = qkv + key_head * p.head_dim;\n"
"    const device float* k = qkv + key_dim + key_head * p.head_dim;\n"
"    const device float* v = qkv + 2 * key_dim + value_head * p.head_dim;\n"
"    float decay = alpha_in[value_head];\n"
"    float beta = beta_in[value_head];\n"
"    float qscale = rsqrt(float(p.head_dim));\n"
"    uint state_base = (value_head * p.head_dim + row) * p.head_dim;\n"
"    uint col_base = lane * values_per_lane;\n"
"    float values[values_per_lane];\n"
"    float predicted_part = 0.0f;\n"
"    #pragma unroll\n"
"    for (uint j = 0; j < values_per_lane; ++j) {\n"
"        uint col = col_base + j;\n"
"        values[j] = state[state_base + col] * decay;\n"
"        predicted_part += values[j] * k[col];\n"
"    }\n"
"    float predicted = simd_sum(predicted_part);\n"
"    float delta = (v[row] - predicted) * beta;\n"
"    float projected_part = 0.0f;\n"
"    #pragma unroll\n"
"    for (uint j = 0; j < values_per_lane; ++j) {\n"
"        uint col = col_base + j;\n"
"        float updated = values[j] + delta * k[col];\n"
"        state[state_base + col] = updated;\n"
"        projected_part += updated * q[col];\n"
"    }\n"
"    float projected = simd_sum(projected_part) * qscale;\n"
"    if (lane == 0) out[value_head * p.head_dim + row] = projected;\n"
"}\n"
"kernel void qwen_delta_norm_gate(device const float* values [[buffer(0)]],\n"
"                                 device const float* gate [[buffer(1)]],\n"
"                                 device const float* norm [[buffer(2)]],\n"
"                                 device float* out [[buffer(3)]],\n"
"                                 constant QwenNormParams& p [[buffer(4)]],\n"
"                                 uint head [[threadgroup_position_in_grid]],\n"
"                                 uint tid [[thread_index_in_threadgroup]]) {\n"
"    if (head >= p.heads) return;\n"
"    uint base = head * p.head_dim;\n"
"    float sum = 0.0f;\n"
"    for (uint i = tid; i < p.head_dim; i += 32) { float v = values[base + i]; sum += v * v; }\n"
"    float scale = rsqrt(simd_sum(sum) / float(p.head_dim) + p.eps);\n"
"    for (uint i = tid; i < p.head_dim; i += 32) {\n"
"        float z = gate[base + i];\n"
"        out[base + i] = values[base + i] * scale * norm[i] * (z / (1.0f + exp(-z)));\n"
"    }\n"
"}\n"
"kernel void qwen_attention_norm_split(device const float* joint [[buffer(0)]],\n"
"                                      device float* keys [[buffer(1)]],\n"
"                                      device const float* q_norm [[buffer(2)]],\n"
"                                      device const float* k_norm [[buffer(3)]],\n"
"                                      device float* queries [[buffer(4)]],\n"
"                                      device float* gates [[buffer(5)]],\n"
"                                      constant QwenAttentionNormParams& p [[buffer(6)]],\n"
"                                      uint group [[threadgroup_position_in_grid]],\n"
"                                      uint tid [[thread_index_in_threadgroup]]) {\n"
"    if (group >= p.query_heads + p.key_heads) return;\n"
"    bool is_query = group < p.query_heads;\n"
"    uint head = is_query ? group : group - p.query_heads;\n"
"    uint base = head * p.head_dim;\n"
"    const device float* src = is_query ? joint + head * 2 * p.head_dim : keys + base;\n"
"    float sum = 0.0f;\n"
"    for (uint i = tid; i < p.head_dim; i += 32) { float v = src[i]; sum += v * v; }\n"
"    float scale = rsqrt(simd_sum(sum) / float(p.head_dim) + p.eps);\n"
"    for (uint i = tid; i < p.head_dim; i += 32) {\n"
"        if (is_query) {\n"
"            queries[base + i] = src[i] * scale * q_norm[i];\n"
"            gates[base + i] = src[p.head_dim + i];\n"
"        } else {\n"
"            keys[base + i] = src[i] * scale * k_norm[i];\n"
"        }\n"
"    }\n"
"}\n"
"kernel void qwen_sigmoid_gate(device float* values [[buffer(0)]],\n"
"                              device const float* gate [[buffer(1)]],\n"
"                              constant QwenUnaryParams& p [[buffer(2)]],\n"
"                              uint gid [[thread_position_in_grid]]) {\n"
"    if (gid >= p.len) return;\n"
"    values[gid] *= 1.0f / (1.0f + exp(-gate[gid]));\n"
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
        NSArray<NSString *> *sources = @[
            kQ4KSource,
            kQ6KSource,
            kQ4_0Source,
            kQ8_0Source,
            kAttentionSource,
            kSiluMulSource,
            kResidualSource,
            kRopeStoreSource,
            kResidentAttnSource,
            kResidentParallelAttnSource,
            kResidentGroupedAttnSource,
            kGreedyArgmaxSource,
            kQwenResidentSource,
        ];
        NSMutableString *combined_source = [[NSMutableString alloc] init];
        for (NSString *source in sources) {
            [combined_source appendString:source];
            [combined_source appendString:@"\n"];
        }
        id<MTLLibrary> library = nil;
        if (rusty_precompiled_metallib_len > 0) {
            dispatch_data_t library_data = dispatch_data_create(
                rusty_precompiled_metallib,
                rusty_precompiled_metallib_len,
                dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                ^{}
            );
            library = [gDevice newLibraryWithData:library_data error:&error];
            if (!library) {
                rusty_metal_log_error("load precompiled kernel library", error);
                error = nil;
            }
        }
        if (!library) {
            library = [gDevice newLibraryWithSource:combined_source options:options error:&error];
        }
        if (!library) {
            rusty_metal_log_error("compile combined kernel library", error);
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
        id<MTLFunction> q6_function = [library newFunctionWithName:@"q6k_matvec"];
        if (!q6_function) {
            rusty_metal_log_error("load q6k function", nil);
            return;
        }
        gQ6KPipeline = [gDevice newComputePipelineStateWithFunction:q6_function error:&error];
        if (!gQ6KPipeline) {
            rusty_metal_log_error("create q6k pipeline", error);
            return;
        }
        id<MTLFunction> q4_0_function = [library newFunctionWithName:@"q4_0_matvec"];
        if (!q4_0_function) {
            rusty_metal_log_error("load q4_0 function", nil);
            return;
        }
        gQ4_0Pipeline = [gDevice newComputePipelineStateWithFunction:q4_0_function error:&error];
        if (!gQ4_0Pipeline) {
            rusty_metal_log_error("create q4_0 pipeline", error);
            return;
        }
        id<MTLFunction> q8_0_function = [library newFunctionWithName:@"q8_0_matvec"];
        if (!q8_0_function) {
            rusty_metal_log_error("load q8_0 function", nil);
            return;
        }
        gQ8_0Pipeline = [gDevice newComputePipelineStateWithFunction:q8_0_function error:&error];
        if (!gQ8_0Pipeline) {
            rusty_metal_log_error("create q8_0 pipeline", error);
            return;
        }
        id<MTLFunction> attention_function = [library newFunctionWithName:@"attention_scan"];
        if (!attention_function) {
            rusty_metal_log_error("load attention function", nil);
            return;
        }
        gAttentionPipeline = [gDevice newComputePipelineStateWithFunction:attention_function error:&error];
        if (!gAttentionPipeline) {
            rusty_metal_log_error("create attention pipeline", error);
            return;
        }
        id<MTLFunction> resident_attention_function = [library newFunctionWithName:@"resident_attention"];
        if (!resident_attention_function) {
            rusty_metal_log_error("load resident attention function", nil);
            return;
        }
        gResidentAttentionPipeline = [gDevice newComputePipelineStateWithFunction:resident_attention_function error:&error];
        if (!gResidentAttentionPipeline) {
            rusty_metal_log_error("create resident attention pipeline", error);
            return;
        }
        id<MTLFunction> parallel_attention_function = [library newFunctionWithName:@"resident_parallel_attention"];
        if (!parallel_attention_function) {
            rusty_metal_log_error("load resident parallel attention function", nil);
            return;
        }
        gResidentParallelAttentionPipeline = [gDevice newComputePipelineStateWithFunction:parallel_attention_function error:&error];
        if (!gResidentParallelAttentionPipeline) {
            rusty_metal_log_error("create resident parallel attention pipeline", error);
            return;
        }
        id<MTLFunction> grouped_attention_function = [library newFunctionWithName:@"resident_gqa4_attention"];
        if (!grouped_attention_function) {
            rusty_metal_log_error("load resident grouped attention function", nil);
            return;
        }
        gResidentGroupedAttentionPipeline = [gDevice newComputePipelineStateWithFunction:grouped_attention_function error:&error];
        if (!gResidentGroupedAttentionPipeline) {
            rusty_metal_log_error("create resident grouped attention pipeline", error);
            return;
        }
        id<MTLFunction> silu_function = [library newFunctionWithName:@"silu_mul"];
        if (!silu_function) {
            rusty_metal_log_error("load silu_mul function", nil);
            return;
        }
        gSiluMulPipeline = [gDevice newComputePipelineStateWithFunction:silu_function error:&error];
        if (!gSiluMulPipeline) {
            rusty_metal_log_error("create silu_mul pipeline", error);
            return;
        }
        id<MTLFunction> residual_rms_function = [library newFunctionWithName:@"residual_rms"];
        if (!residual_rms_function) {
            rusty_metal_log_error("load residual_rms function", nil);
            return;
        }
        gResidualRmsPipeline = [gDevice newComputePipelineStateWithFunction:residual_rms_function error:&error];
        if (!gResidualRmsPipeline) {
            rusty_metal_log_error("create residual_rms pipeline", error);
            return;
        }
        id<MTLFunction> residual_add_function = [library newFunctionWithName:@"residual_add"];
        if (!residual_add_function) {
            rusty_metal_log_error("load residual_add function", nil);
            return;
        }
        gResidualAddPipeline = [gDevice newComputePipelineStateWithFunction:residual_add_function error:&error];
        if (!gResidualAddPipeline) {
            rusty_metal_log_error("create residual_add pipeline", error);
            return;
        }
        id<MTLFunction> rope_function = [library newFunctionWithName:@"rope_store"];
        if (!rope_function) {
            rusty_metal_log_error("load rope_store function", nil);
            return;
        }
        gRopeStorePipeline = [gDevice newComputePipelineStateWithFunction:rope_function error:&error];
        if (!gRopeStorePipeline) {
            rusty_metal_log_error("create rope_store pipeline", error);
            return;
        }
        id<MTLFunction> greedy_function = [library newFunctionWithName:@"greedy_argmax"];
        if (!greedy_function) {
            rusty_metal_log_error("load greedy_argmax function", nil);
            return;
        }
        gGreedyArgmaxPipeline = [gDevice newComputePipelineStateWithFunction:greedy_function error:&error];
        if (!gGreedyArgmaxPipeline) {
            rusty_metal_log_error("create greedy_argmax pipeline", error);
            return;
        }
        id<MTLFunction> greedy_stage1_function = [library newFunctionWithName:@"greedy_argmax_stage1"];
        id<MTLFunction> greedy_stage2_function = [library newFunctionWithName:@"greedy_argmax_stage2"];
        if (!greedy_stage1_function || !greedy_stage2_function) {
            rusty_metal_log_error("load parallel greedy argmax functions", nil);
            return;
        }
        gGreedyArgmaxStage1Pipeline = [gDevice newComputePipelineStateWithFunction:greedy_stage1_function error:&error];
        gGreedyArgmaxStage2Pipeline = [gDevice newComputePipelineStateWithFunction:greedy_stage2_function error:&error];
        if (!gGreedyArgmaxStage1Pipeline || !gGreedyArgmaxStage2Pipeline) {
            rusty_metal_log_error("create parallel greedy argmax pipelines", error);
            return;
        }
        id<MTLFunction> qwen_conv_function = [library newFunctionWithName:@"qwen_conv_silu"];
        id<MTLFunction> qwen_l2_function = [library newFunctionWithName:@"qwen_l2_norm_qk"];
        id<MTLFunction> qwen_delta_function = [library newFunctionWithName:@"qwen_delta_step"];
        id<MTLFunction> qwen_delta_norm_function = [library newFunctionWithName:@"qwen_delta_norm_gate"];
        id<MTLFunction> qwen_attention_norm_function = [library newFunctionWithName:@"qwen_attention_norm_split"];
        id<MTLFunction> qwen_sigmoid_function = [library newFunctionWithName:@"qwen_sigmoid_gate"];
        if (!qwen_conv_function || !qwen_l2_function || !qwen_delta_function ||
            !qwen_delta_norm_function || !qwen_attention_norm_function || !qwen_sigmoid_function) {
            // An installed precompiled metallib may predate these optional
            // Qwen kernels. Compile just this small source at runtime while
            // retaining the precompiled pipelines for the common operators.
            NSError *qwen_error = nil;
            id<MTLLibrary> qwen_library = [gDevice newLibraryWithSource:kQwenResidentSource
                                                               options:options
                                                                 error:&qwen_error];
            if (!qwen_library) {
                rusty_metal_log_error("compile qwen resident library", qwen_error);
                return;
            }
            qwen_conv_function = [qwen_library newFunctionWithName:@"qwen_conv_silu"];
            qwen_l2_function = [qwen_library newFunctionWithName:@"qwen_l2_norm_qk"];
            qwen_delta_function = [qwen_library newFunctionWithName:@"qwen_delta_step"];
            qwen_delta_norm_function = [qwen_library newFunctionWithName:@"qwen_delta_norm_gate"];
            qwen_attention_norm_function = [qwen_library newFunctionWithName:@"qwen_attention_norm_split"];
            qwen_sigmoid_function = [qwen_library newFunctionWithName:@"qwen_sigmoid_gate"];
            if (!qwen_conv_function || !qwen_l2_function || !qwen_delta_function ||
                !qwen_delta_norm_function || !qwen_attention_norm_function || !qwen_sigmoid_function) {
                rusty_metal_log_error("load qwen resident functions", nil);
                return;
            }
        }
        gQwenConvSiluPipeline = [gDevice newComputePipelineStateWithFunction:qwen_conv_function error:&error];
        gQwenL2NormPipeline = [gDevice newComputePipelineStateWithFunction:qwen_l2_function error:&error];
        gQwenDeltaPipeline = [gDevice newComputePipelineStateWithFunction:qwen_delta_function error:&error];
        gQwenDeltaNormGatePipeline = [gDevice newComputePipelineStateWithFunction:qwen_delta_norm_function error:&error];
        gQwenAttentionNormSplitPipeline = [gDevice newComputePipelineStateWithFunction:qwen_attention_norm_function error:&error];
        gQwenSigmoidGatePipeline = [gDevice newComputePipelineStateWithFunction:qwen_sigmoid_function error:&error];
        if (!gQwenConvSiluPipeline || !gQwenL2NormPipeline || !gQwenDeltaPipeline ||
            !gQwenDeltaNormGatePipeline || !gQwenAttentionNormSplitPipeline || !gQwenSigmoidGatePipeline) {
            rusty_metal_log_error("create qwen resident pipelines", error);
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
        if (end != value && parsed >= 8 && parsed <= 16 && (parsed % 8) == 0) {
            return (NSUInteger)parsed;
        }
    }
    (void)rows;
    // ggml's current Apple decode tuning processes eight rows in each of two
    // SIMD groups. This both cuts the number of groups and amortizes every
    // activation load across four times as many weight rows.
    return 16;
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
    // Two SIMD groups, each producing eight rows.
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

    MTLSize threads_per_group = MTLSizeMake(32 * (rows_per_group / 8), 1, 1);
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
    MTLSize threads = MTLSizeMake(32 * (rows_per_group / 8), 1, 1);
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
            .apply_gate = 0,
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
        [enc setBuffer:gAttentionZeroBuffer offset:0 atIndex:5];
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

int rusty_metal_test_greedy_argmax(const float *logits, uint32_t vocab,
                                   const uint32_t *recent, uint32_t recent_len,
                                   float repeat_penalty, uint32_t *token_out) {
    if (!rusty_metal_init() || !logits || vocab == 0 || recent_len > 64 ||
        (recent_len > 0 && !recent) || !token_out) return 0;
    @autoreleasepool {
        id<MTLBuffer> logits_buffer = [gDevice newBufferWithBytes:logits
            length:(NSUInteger)vocab * sizeof(float)
            options:MTLResourceStorageModeShared];
        uint32_t zero = 0;
        id<MTLBuffer> recent_buffer = [gDevice newBufferWithBytes:(recent_len ? recent : &zero)
            length:(NSUInteger)(recent_len ? recent_len : 1) * sizeof(uint32_t)
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> token_buffer = [gDevice newBufferWithLength:sizeof(uint32_t)
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> partial_value = [gDevice newBufferWithLength:RUSTY_ARGMAX_GROUPS * sizeof(float)
            options:MTLResourceStorageModePrivate];
        id<MTLBuffer> partial_id = [gDevice newBufferWithLength:RUSTY_ARGMAX_GROUPS * sizeof(uint32_t)
            options:MTLResourceStorageModePrivate];
        if (!logits_buffer || !recent_buffer || !token_buffer || !partial_value || !partial_id) return 0;
        uint32_t groups = MIN((vocab + 255) / 256, RUSTY_ARGMAX_GROUPS);
        RustyArgmaxParams p = {
            .vocab = vocab,
            .recent_len = recent_len,
            .groups = groups,
            .repeat_penalty = repeat_penalty,
        };
        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:gGreedyArgmaxStage1Pipeline];
        [enc setBuffer:logits_buffer offset:0 atIndex:0];
        [enc setBuffer:recent_buffer offset:0 atIndex:1];
        [enc setBuffer:partial_value offset:0 atIndex:2];
        [enc setBuffer:partial_id offset:0 atIndex:3];
        [enc setBytes:&p length:sizeof(p) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc setComputePipelineState:gGreedyArgmaxStage2Pipeline];
        [enc setBuffer:partial_value offset:0 atIndex:0];
        [enc setBuffer:partial_id offset:0 atIndex:1];
        [enc setBuffer:token_buffer offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if ([cb status] != MTLCommandBufferStatusCompleted) return 0;
        *token_out = *(const uint32_t *)[token_buffer contents];
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
static __strong id<MTLBuffer> gR_recent, gR_selected;

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

static void resident_greedy_argmax(id<MTLComputeCommandEncoder> enc,
                                   uint32_t recent_len, float repeat_penalty) {
    RustyArgmaxParams p = {
        .vocab = gR_vocab,
        .recent_len = recent_len,
        .groups = 1,
        .repeat_penalty = repeat_penalty,
    };
    [enc setComputePipelineState:gGreedyArgmaxPipeline];
    [enc setBuffer:gR_logits offset:0 atIndex:0];
    [enc setBuffer:gR_recent offset:0 atIndex:1];
    [enc setBuffer:gR_selected offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
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
        .value_dim = gR_valuedim, .apply_gate = 0,
        .key_stride = gR_kdim, .value_stride = gR_vdim,
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
    [enc setBuffer:gAttentionZeroBuffer offset:0 atIndex:5];
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
    gR_recent = resident_alloc_shared(64 * sizeof(uint32_t));
    gR_selected = resident_alloc_shared(sizeof(uint32_t));
    if (!gR_x || !gR_xn || !gR_q || !gR_k || !gR_v || !gR_attn || !gR_gate || !gR_up ||
        !gR_hiddenbuf || !gR_proj || !gR_logits || !gR_zero || !gR_recent || !gR_selected) {
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

// `output_mode` == 0 runs the layer stack for its KV-cache side effects only and
// skips the final norm plus the vocabulary projection. Prompt tokens need the
// cache filled but throw their logits away, and that projection is the single
// largest weight read in the model (330 MiB for a 131072-entry Q6_K vocabulary),
// so skipping it removes both the read and the device->host copy from every
// prefill token. It is safe to skip because each call re-seeds gR_x from the
// caller's embedding, so nothing carries over from the tail of the graph; only
// the in-layer RoPE/attention KV writes have to happen.
int rusty_metal_resident_decode(const float *x_embed, uint32_t pos, uint32_t start_t,
                                int output_mode, float *logits_out,
                                const uint32_t *recent, uint32_t recent_len,
                                float repeat_penalty, uint32_t *token_out) {
    if (!gResidentReady || !x_embed || pos >= gR_storage) return 0;
    if (output_mode == 1 && !logits_out) return 0;
    if (output_mode == 2 && (!token_out || recent_len > 64 || (recent_len > 0 && !recent))) return 0;
    if (output_mode < 0 || output_mode > 2) return 0;
    @autoreleasepool {
        memcpy([gR_x contents], x_embed, (NSUInteger)gR_dim * sizeof(float));
        if (output_mode == 2 && recent_len > 0) {
            memcpy([gR_recent contents], recent, (NSUInteger)recent_len * sizeof(uint32_t));
        }
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
        if (output_mode != 0) {
            resident_rms(enc, gR_x, gR_proj, gR_outnorm, gR_xn, gR_dim, gR_eps);
            resident_matvec(enc, gR_outdt, gR_outw, gR_xn, gR_logits, gR_outrows, gR_dim);
            if (output_mode == 2) {
                resident_greedy_argmax(enc, recent_len, repeat_penalty);
            }
        }
        [enc endEncoding];
        double encode_end = rusty_metal_now_seconds();
        [cb commit];
        [cb waitUntilCompleted];
        rusty_metal_profile_command_buffer(cb, encode_start, encode_end);
        if ([cb status] != MTLCommandBufferStatusCompleted) return 0;
        if (output_mode == 1) {
            memcpy(logits_out, [gR_logits contents], (NSUInteger)gR_vocab * sizeof(float));
        } else if (output_mode == 2) {
            *token_out = *(const uint32_t *)[gR_selected contents];
        }
        return 1;
    }
}

// ─── Qwen3.5/Qwen3.8 GPU-resident hybrid decoder ───────────────────────────

typedef struct {
    uint32_t layer_type;
    const uint8_t *w[8];
    uintptr_t w_len[8];
    uint32_t w_rows[8];
    uint32_t w_cols[8];
    uint32_t w_dt[8];
    const float *attn_norm; uint32_t attn_norm_len;
    const float *post_norm; uint32_t post_norm_len;
    const float *conv_w; uint32_t conv_w_len;
    const float *a; uint32_t a_len;
    const float *dt_bias; uint32_t dt_bias_len;
    const float *norm; uint32_t norm_len;
    const float *q_norm; uint32_t q_norm_len;
    const float *k_norm; uint32_t k_norm_len;
} RustyQwenResidentLayerDesc;

typedef struct {
    uint32_t conv_dim, d_conv, value_heads, key_heads, head_dim;
    float eps;
} RustyQwenConvParams;
typedef struct { uint32_t heads, head_dim; float eps; } RustyQwenNormParams;
typedef struct { uint32_t value_heads, key_heads, head_dim; } RustyQwenDeltaParams;
typedef struct { uint32_t query_heads, key_heads, head_dim; float eps; } RustyQwenAttentionNormParams;

typedef struct {
    uint32_t type;
    __strong id<MTLBuffer> w[8];
    NSUInteger w_offset[8];
    uint32_t rows[8], cols[8], dt[8];
    __strong id<MTLBuffer> attn_norm;
    __strong id<MTLBuffer> post_norm;
    __strong id<MTLBuffer> conv_w;
    __strong id<MTLBuffer> a;
    __strong id<MTLBuffer> dt_bias;
    __strong id<MTLBuffer> norm;
    __strong id<MTLBuffer> q_norm;
    __strong id<MTLBuffer> k_norm;
    __strong id<MTLBuffer> conv_state;
    __strong id<MTLBuffer> delta_state;
    __strong id<MTLBuffer> k_cache;
    __strong id<MTLBuffer> v_cache;
} QwenResidentLayer;

static BOOL gQwenResidentReady;
static uint32_t gQ_nlayers, gQ_dim, gQ_hidden, gQ_vocab, gQ_storage;
static uint32_t gQ_nheads, gQ_nkv, gQ_headdim, gQ_rotarydim;
static uint32_t gQ_valueheads, gQ_keyheads, gQ_statedim, gQ_dconv;
static uint32_t gQ_qdim, gQ_kdim, gQ_vdim, gQ_attndim, gQ_conv_dim;
static uint32_t gQ_outrows, gQ_outdt, gQ_registered_layers;
static float gQ_eps;
static QwenResidentLayer gQLayers[RUSTY_MAX_RESIDENT_LAYERS];
static __strong id<MTLBuffer> gQ_x, gQ_xn, gQ_joint, gQ_q, gQ_k, gQ_v;
static __strong id<MTLBuffer> gQ_attn, gQ_gate, gQ_up, gQ_hiddenbuf, gQ_proj;
static __strong id<MTLBuffer> gQ_alpha, gQ_beta, gQ_logits, gQ_zero;
static __strong id<MTLBuffer> gQ_recent, gQ_selected, gQ_argmax_value, gQ_argmax_id;
static __strong id<MTLBuffer> gQ_invfreq, gQ_outnorm, gQ_outw;
static NSUInteger gQ_outw_offset;

// GGUF tensors are slices of a process-lifetime read-only mmap.  Wrapping the
// encompassing VM pages avoids duplicating roughly 16 GiB of Qwen weights on
// a 32-GiB Mac.  The returned offset points from the page-aligned Metal buffer
// to the actual tensor start.  Fall back to the ordinary copied cache on a
// platform that rejects a read-only no-copy mapping.
static id<MTLBuffer> qwen_resident_weight_buffer(const uint8_t *bytes,
                                                 uintptr_t bytes_len,
                                                 NSUInteger *offset_out) {
    if (!bytes || bytes_len == 0 || !offset_out) return nil;
    if (rusty_metal_env_enabled("RUSTY_LLM_METAL_QWEN_COPY_WEIGHTS")) {
        *offset_out = 0;
        return rusty_metal_weight_buffer(bytes, bytes_len);
    }
    uintptr_t page_size = (uintptr_t)getpagesize();
    uintptr_t address = (uintptr_t)bytes;
    uintptr_t base = address - (address % page_size);
    uintptr_t offset = address - base;
    if (bytes_len > UINTPTR_MAX - offset - (page_size - 1)) return nil;
    uintptr_t mapped_len = (offset + bytes_len + page_size - 1) / page_size * page_size;
    id<MTLBuffer> buffer = [gDevice newBufferWithBytesNoCopy:(void *)base
                                                     length:(NSUInteger)mapped_len
                                                    options:MTLResourceStorageModeShared
                                                deallocator:nil];
    if (buffer) {
        *offset_out = (NSUInteger)offset;
        gMetalBufferAllocations += 1;
        return buffer;
    }
    buffer = rusty_metal_weight_buffer(bytes, bytes_len);
    *offset_out = 0;
    return buffer;
}

static void qwen_encode_matvec(id<MTLComputeCommandEncoder> enc, uint32_t dt,
                               id<MTLBuffer> weight, NSUInteger weight_offset,
                               id<MTLBuffer> x, id<MTLBuffer> out,
                               uint32_t rows, uint32_t cols) {
    NSUInteger rows_per_group = dt == 1
        ? rusty_metal_q6k_rows_per_group(rows)
        : rusty_metal_q4k_rows_per_group(rows);
    RustyQ4KParams params = {
        .rows = rows,
        .cols = cols,
        .row_bytes = (uint32_t)((cols / 256) * (dt == 1 ? 210 : 144)),
        .n_blocks = cols / 256,
        .rows_per_group = (uint32_t)rows_per_group,
    };
    [enc setComputePipelineState:dt == 1 ? gQ6KPipeline : gQ4KPipeline];
    [enc setBuffer:weight offset:weight_offset atIndex:0];
    [enc setBuffer:x offset:0 atIndex:1];
    [enc setBuffer:out offset:0 atIndex:2];
    [enc setBytes:&params length:sizeof(params) atIndex:3];
    NSUInteger rows_per_simd = dt == 1 ? 2 : 8;
    gMetalDispatches += 1;
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)rows + rows_per_group - 1) / rows_per_group, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32 * (rows_per_group / rows_per_simd), 1, 1)];
}

static void qwen_encode_q4_pair(id<MTLComputeCommandEncoder> enc,
                                id<MTLBuffer> weight_a, NSUInteger offset_a,
                                id<MTLBuffer> weight_b, NSUInteger offset_b,
                                id<MTLBuffer> x, id<MTLBuffer> out_a, id<MTLBuffer> out_b,
                                uint32_t rows_a, uint32_t rows_b, uint32_t cols) {
    NSUInteger rows_per_group = rusty_metal_q4k_rows_per_group(rows_a + rows_b);
    RustyQ4KPairParams params = {
        .rows_a = rows_a, .rows_b = rows_b, .cols = cols,
        .row_bytes = (uint32_t)((cols / 256) * 144), .n_blocks = cols / 256,
        .rows_per_group = (uint32_t)rows_per_group,
    };
    [enc setComputePipelineState:gQ4KPairPipeline];
    [enc setBuffer:weight_a offset:offset_a atIndex:0];
    [enc setBuffer:weight_b offset:offset_b atIndex:1];
    [enc setBuffer:x offset:0 atIndex:2];
    [enc setBuffer:out_a offset:0 atIndex:3];
    [enc setBuffer:out_b offset:0 atIndex:4];
    [enc setBytes:&params length:sizeof(params) atIndex:5];
    NSUInteger groups_a = ((NSUInteger)rows_a + rows_per_group - 1) / rows_per_group;
    NSUInteger groups_b = ((NSUInteger)rows_b + rows_per_group - 1) / rows_per_group;
    gMetalDispatches += 1;
    [enc dispatchThreadgroups:MTLSizeMake(groups_a + groups_b, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32 * (rows_per_group / 8), 1, 1)];
}

static void qwen_encode_conv(id<MTLComputeCommandEncoder> enc, QwenResidentLayer *L) {
    RustyQwenConvParams p = {
        .conv_dim = gQ_conv_dim, .d_conv = gQ_dconv, .value_heads = gQ_valueheads,
        .key_heads = gQ_keyheads, .head_dim = gQ_statedim, .eps = gQ_eps,
    };
    [enc setComputePipelineState:gQwenConvSiluPipeline];
    [enc setBuffer:gQ_joint offset:0 atIndex:0];
    [enc setBuffer:L->conv_w offset:0 atIndex:1];
    [enc setBuffer:L->conv_state offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    [enc setBuffer:gQ_alpha offset:0 atIndex:4];
    [enc setBuffer:gQ_beta offset:0 atIndex:5];
    [enc setBuffer:L->a offset:0 atIndex:6];
    [enc setBuffer:L->dt_bias offset:0 atIndex:7];
    gMetalDispatches += 1;
    [enc dispatchThreadgroups:MTLSizeMake(2 * gQ_keyheads + gQ_valueheads, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

static void qwen_encode_l2(id<MTLComputeCommandEncoder> enc) {
    RustyQwenNormParams p = { .heads = gQ_keyheads, .head_dim = gQ_statedim, .eps = gQ_eps };
    [enc setComputePipelineState:gQwenL2NormPipeline];
    [enc setBuffer:gQ_joint offset:0 atIndex:0];
    [enc setBytes:&p length:sizeof(p) atIndex:1];
    gMetalDispatches += 1;
    [enc dispatchThreadgroups:MTLSizeMake(2 * gQ_keyheads, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

static void qwen_encode_delta(id<MTLComputeCommandEncoder> enc, QwenResidentLayer *L) {
    RustyQwenDeltaParams p = {
        .value_heads = gQ_valueheads, .key_heads = gQ_keyheads, .head_dim = gQ_statedim,
    };
    [enc setComputePipelineState:gQwenDeltaPipeline];
    [enc setBuffer:gQ_joint offset:0 atIndex:0];
    [enc setBuffer:gQ_alpha offset:0 atIndex:1];
    [enc setBuffer:gQ_beta offset:0 atIndex:2];
    [enc setBuffer:L->delta_state offset:0 atIndex:3];
    [enc setBuffer:gQ_attn offset:0 atIndex:4];
    [enc setBytes:&p length:sizeof(p) atIndex:5];
    gMetalDispatches += 1;
    [enc dispatchThreadgroups:MTLSizeMake((gQ_statedim + 3) / 4, gQ_valueheads, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
}

static void qwen_encode_delta_norm_gate(id<MTLComputeCommandEncoder> enc,
                                         QwenResidentLayer *L) {
    RustyQwenNormParams p = { .heads = gQ_valueheads, .head_dim = gQ_statedim, .eps = gQ_eps };
    [enc setComputePipelineState:gQwenDeltaNormGatePipeline];
    [enc setBuffer:gQ_attn offset:0 atIndex:0];
    [enc setBuffer:gQ_gate offset:0 atIndex:1];
    [enc setBuffer:L->norm offset:0 atIndex:2];
    [enc setBuffer:gQ_hiddenbuf offset:0 atIndex:3];
    [enc setBytes:&p length:sizeof(p) atIndex:4];
    gMetalDispatches += 1;
    [enc dispatchThreadgroups:MTLSizeMake(gQ_valueheads, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

static void qwen_encode_attention_norm(id<MTLComputeCommandEncoder> enc,
                                        QwenResidentLayer *L) {
    RustyQwenAttentionNormParams p = {
        .query_heads = gQ_nheads, .key_heads = gQ_nkv, .head_dim = gQ_headdim, .eps = gQ_eps,
    };
    [enc setComputePipelineState:gQwenAttentionNormSplitPipeline];
    [enc setBuffer:gQ_joint offset:0 atIndex:0];
    [enc setBuffer:gQ_k offset:0 atIndex:1];
    [enc setBuffer:L->q_norm offset:0 atIndex:2];
    [enc setBuffer:L->k_norm offset:0 atIndex:3];
    [enc setBuffer:gQ_q offset:0 atIndex:4];
    [enc setBuffer:gQ_gate offset:0 atIndex:5];
    [enc setBytes:&p length:sizeof(p) atIndex:6];
    gMetalDispatches += 1;
    [enc dispatchThreadgroups:MTLSizeMake(gQ_nheads + gQ_nkv, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

static void qwen_encode_rope_store(id<MTLComputeCommandEncoder> enc,
                                    QwenResidentLayer *L, uint32_t pos) {
    RustyRopeParams p = {
        .pos = pos, .head_dim = gQ_headdim, .half_dim = gQ_rotarydim / 2,
        .n_heads = gQ_nheads, .n_kv_heads = gQ_nkv, .value_dim = gQ_headdim,
        .kv_k_dim = gQ_kdim, .kv_v_dim = gQ_vdim, .slot = pos, .neox = 1,
    };
    [enc setComputePipelineState:gRopeStorePipeline];
    [enc setBuffer:gQ_q offset:0 atIndex:0];
    [enc setBuffer:gQ_k offset:0 atIndex:1];
    [enc setBuffer:gQ_v offset:0 atIndex:2];
    [enc setBuffer:gQ_invfreq offset:0 atIndex:3];
    [enc setBuffer:L->k_cache offset:0 atIndex:4];
    [enc setBuffer:L->v_cache offset:0 atIndex:5];
    [enc setBytes:&p length:sizeof(p) atIndex:6];
    uint32_t work = gQ_nheads * (gQ_rotarydim / 2);
    if (gQ_vdim > work) work = gQ_vdim;
    gMetalDispatches += 1;
    [enc dispatchThreads:MTLSizeMake(work, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void qwen_encode_attention(id<MTLComputeCommandEncoder> enc,
                                   QwenResidentLayer *L, uint32_t start_t, uint32_t end_t) {
    RustyResidentAttentionParams p = {
        .heads = gQ_nheads, .kv_mul = gQ_nheads / gQ_nkv,
        .head_dim = gQ_headdim, .value_dim = gQ_headdim, .apply_gate = 1,
        .key_stride = gQ_kdim, .value_stride = gQ_vdim,
        .start_t = start_t, .end_t = end_t,
        .scale = 1.0f / sqrt((float)gQ_headdim),
    };
    uint32_t context_tokens = end_t - start_t + 1;
    BOOL use_parallel = gResidentParallelAttentionPipeline != nil && context_tokens >= 16;
    [enc setComputePipelineState:use_parallel ? gResidentParallelAttentionPipeline : gResidentAttentionPipeline];
    [enc setBuffer:gQ_q offset:0 atIndex:0];
    [enc setBuffer:L->k_cache offset:0 atIndex:1];
    [enc setBuffer:L->v_cache offset:0 atIndex:2];
    [enc setBuffer:gQ_attn offset:0 atIndex:3];
    [enc setBytes:&p length:sizeof(p) atIndex:4];
    [enc setBuffer:gQ_gate offset:0 atIndex:5];
    gMetalDispatches += 1;
    [enc dispatchThreadgroups:MTLSizeMake(gQ_nheads, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(use_parallel ? 128 : 32, 1, 1)];
}

static void qwen_encode_sigmoid_gate(id<MTLComputeCommandEncoder> enc) {
    RustyUnaryParams p = { .len = gQ_attndim };
    [enc setComputePipelineState:gQwenSigmoidGatePipeline];
    [enc setBuffer:gQ_attn offset:0 atIndex:0];
    [enc setBuffer:gQ_gate offset:0 atIndex:1];
    [enc setBytes:&p length:sizeof(p) atIndex:2];
    gMetalDispatches += 1;
    [enc dispatchThreads:MTLSizeMake(gQ_attndim, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void qwen_encode_greedy(id<MTLComputeCommandEncoder> enc,
                                uint32_t recent_len, float repeat_penalty) {
    if (rusty_metal_env_enabled("RUSTY_LLM_METAL_QWEN_LEGACY_ARGMAX")) {
        RustyArgmaxParams legacy = {
            .vocab = gQ_vocab, .recent_len = recent_len, .groups = 1,
            .repeat_penalty = repeat_penalty,
        };
        [enc setComputePipelineState:gGreedyArgmaxPipeline];
        [enc setBuffer:gQ_logits offset:0 atIndex:0];
        [enc setBuffer:gQ_recent offset:0 atIndex:1];
        [enc setBuffer:gQ_selected offset:0 atIndex:2];
        [enc setBytes:&legacy length:sizeof(legacy) atIndex:3];
        gMetalDispatches += 1;
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        return;
    }
    uint32_t groups = MIN((gQ_vocab + 255) / 256, RUSTY_ARGMAX_GROUPS);
    RustyArgmaxParams p = {
        .vocab = gQ_vocab, .recent_len = recent_len, .groups = groups,
        .repeat_penalty = repeat_penalty,
    };
    [enc setComputePipelineState:gGreedyArgmaxStage1Pipeline];
    [enc setBuffer:gQ_logits offset:0 atIndex:0];
    [enc setBuffer:gQ_recent offset:0 atIndex:1];
    [enc setBuffer:gQ_argmax_value offset:0 atIndex:2];
    [enc setBuffer:gQ_argmax_id offset:0 atIndex:3];
    [enc setBytes:&p length:sizeof(p) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc setComputePipelineState:gGreedyArgmaxStage2Pipeline];
    [enc setBuffer:gQ_argmax_value offset:0 atIndex:0];
    [enc setBuffer:gQ_argmax_id offset:0 atIndex:1];
    [enc setBuffer:gQ_selected offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    gMetalDispatches += 2;
}

int rusty_metal_qwen_resident_configure(uint32_t n_layers, uint32_t dim,
                                        uint32_t hidden_dim, uint32_t vocab,
                                        uint32_t storage_len, float eps,
                                        uint32_t n_heads, uint32_t n_kv_heads,
                                        uint32_t head_dim, uint32_t rotary_dim,
                                        uint32_t value_heads, uint32_t key_heads,
                                        uint32_t state_dim, uint32_t d_conv) {
    if (!rusty_metal_init()) return 0;
    if (n_layers == 0 || n_layers > RUSTY_MAX_RESIDENT_LAYERS || dim == 0 || hidden_dim == 0 ||
        vocab == 0 || storage_len == 0 || n_heads == 0 || n_kv_heads == 0 ||
        n_heads % n_kv_heads != 0 || head_dim == 0 || head_dim > 256 ||
        rotary_dim == 0 || rotary_dim > head_dim || (rotary_dim % 2) != 0 ||
        value_heads == 0 || key_heads == 0 || value_heads % key_heads != 0 ||
        state_dim != 128 || d_conv < 2 ||
        (dim % 256) != 0 || (hidden_dim % 256) != 0) return 0;
    uint64_t qdim64 = (uint64_t)n_heads * head_dim;
    uint64_t kdim64 = (uint64_t)n_kv_heads * head_dim;
    uint64_t value_dim64 = (uint64_t)value_heads * state_dim;
    uint64_t conv_dim64 = ((uint64_t)2 * key_heads + value_heads) * state_dim;
    if (qdim64 > UINT32_MAX / 2 || kdim64 > UINT32_MAX ||
        value_dim64 > UINT32_MAX || conv_dim64 > UINT32_MAX) return 0;

    gQwenResidentReady = NO;
    gQ_registered_layers = 0;
    gQ_nlayers = n_layers; gQ_dim = dim; gQ_hidden = hidden_dim; gQ_vocab = vocab;
    gQ_storage = storage_len; gQ_eps = eps; gQ_nheads = n_heads; gQ_nkv = n_kv_heads;
    gQ_headdim = head_dim; gQ_rotarydim = rotary_dim; gQ_valueheads = value_heads;
    gQ_keyheads = key_heads; gQ_statedim = state_dim; gQ_dconv = d_conv;
    gQ_qdim = (uint32_t)qdim64; gQ_kdim = (uint32_t)kdim64;
    gQ_vdim = (uint32_t)kdim64; gQ_attndim = (uint32_t)qdim64;
    gQ_conv_dim = (uint32_t)conv_dim64;
    uint32_t joint_len = MAX(gQ_conv_dim, 2 * gQ_qdim);
    uint32_t value_dim = (uint32_t)value_dim64;

    gQ_x = resident_alloc_shared((NSUInteger)dim * sizeof(float));
    gQ_xn = resident_alloc_private((NSUInteger)dim * sizeof(float));
    gQ_joint = resident_alloc_private((NSUInteger)joint_len * sizeof(float));
    gQ_q = resident_alloc_private((NSUInteger)gQ_qdim * sizeof(float));
    gQ_k = resident_alloc_private((NSUInteger)gQ_kdim * sizeof(float));
    gQ_v = resident_alloc_private((NSUInteger)gQ_vdim * sizeof(float));
    gQ_attn = resident_alloc_private((NSUInteger)MAX(gQ_attndim, value_dim) * sizeof(float));
    gQ_gate = resident_alloc_private((NSUInteger)MAX(MAX(hidden_dim, gQ_attndim), value_dim) * sizeof(float));
    gQ_up = resident_alloc_private((NSUInteger)hidden_dim * sizeof(float));
    gQ_hiddenbuf = resident_alloc_private((NSUInteger)MAX(hidden_dim, value_dim) * sizeof(float));
    gQ_proj = resident_alloc_private((NSUInteger)dim * sizeof(float));
    gQ_alpha = resident_alloc_private((NSUInteger)value_heads * sizeof(float));
    gQ_beta = resident_alloc_private((NSUInteger)value_heads * sizeof(float));
    gQ_logits = resident_alloc_shared((NSUInteger)vocab * sizeof(float));
    gQ_zero = resident_alloc_shared((NSUInteger)dim * sizeof(float));
    gQ_recent = resident_alloc_shared(64 * sizeof(uint32_t));
    gQ_selected = resident_alloc_shared(sizeof(uint32_t));
    gQ_argmax_value = resident_alloc_private(RUSTY_ARGMAX_GROUPS * sizeof(float));
    gQ_argmax_id = resident_alloc_private(RUSTY_ARGMAX_GROUPS * sizeof(uint32_t));
    if (!gQ_x || !gQ_xn || !gQ_joint || !gQ_q || !gQ_k || !gQ_v || !gQ_attn ||
        !gQ_gate || !gQ_up || !gQ_hiddenbuf || !gQ_proj || !gQ_alpha || !gQ_beta ||
        !gQ_logits || !gQ_zero || !gQ_recent || !gQ_selected ||
        !gQ_argmax_value || !gQ_argmax_id) return 0;
    memset([gQ_zero contents], 0, (NSUInteger)dim * sizeof(float));
    return 1;
}

int rusty_metal_qwen_resident_set_layer(uint32_t l,
                                        const RustyQwenResidentLayerDesc *d) {
    if (!gDevice || !d || l >= gQ_nlayers || d->layer_type > 1 ||
        !d->attn_norm || d->attn_norm_len != gQ_dim ||
        !d->post_norm || d->post_norm_len != gQ_dim) return 0;
    QwenResidentLayer *L = &gQLayers[l];
    L->type = d->layer_type;
    uint32_t required = d->layer_type == 0 ? 8 : 7;
    uint32_t value_dim = gQ_valueheads * gQ_statedim;
    uint32_t expected_rows[8] = {0};
    uint32_t expected_cols[8] = {0};
    if (d->layer_type == 0) {
        uint32_t rows[8] = {
            gQ_conv_dim, value_dim, gQ_valueheads, gQ_valueheads,
            gQ_dim, gQ_hidden, gQ_hidden, gQ_dim,
        };
        uint32_t cols[8] = {
            gQ_dim, gQ_dim, gQ_dim, gQ_dim,
            value_dim, gQ_dim, gQ_dim, gQ_hidden,
        };
        memcpy(expected_rows, rows, sizeof(rows));
        memcpy(expected_cols, cols, sizeof(cols));
    } else {
        uint32_t rows[8] = {
            2 * gQ_qdim, gQ_kdim, gQ_vdim, gQ_dim,
            gQ_hidden, gQ_hidden, gQ_dim, 0,
        };
        uint32_t cols[8] = {
            gQ_dim, gQ_dim, gQ_dim, gQ_attndim,
            gQ_dim, gQ_dim, gQ_hidden, 0,
        };
        memcpy(expected_rows, rows, sizeof(rows));
        memcpy(expected_cols, cols, sizeof(cols));
    }
    for (uint32_t i = 0; i < required; ++i) {
        if (!d->w[i] || d->w_len[i] == 0 || d->w_rows[i] == 0 || d->w_cols[i] == 0 ||
            d->w_dt[i] > 1 || (d->w_cols[i] % 256) != 0 ||
            d->w_rows[i] != expected_rows[i] || d->w_cols[i] != expected_cols[i]) return 0;
        uint64_t expected_len = (uint64_t)d->w_rows[i] * (d->w_cols[i] / 256) *
                                (d->w_dt[i] == 1 ? 210u : 144u);
        if (expected_len != d->w_len[i]) return 0;
        // These slots share the Q4_K pair kernel rather than selecting a
        // pipeline from their dtype at dispatch time.
        if ((d->layer_type == 0 && (i == 2 || i == 3 || i == 5 || i == 6) && d->w_dt[i] != 0) ||
            (d->layer_type == 1 && (i == 4 || i == 5) && d->w_dt[i] != 0)) return 0;
        L->w[i] = qwen_resident_weight_buffer(d->w[i], d->w_len[i], &L->w_offset[i]);
        if (!L->w[i]) return 0;
        L->rows[i] = d->w_rows[i]; L->cols[i] = d->w_cols[i]; L->dt[i] = d->w_dt[i];
    }
    L->attn_norm = resident_floats(d->attn_norm, d->attn_norm_len);
    L->post_norm = resident_floats(d->post_norm, d->post_norm_len);
    if (!L->attn_norm || !L->post_norm) return 0;

    if (d->layer_type == 0) {
        uint32_t conv_w_len = gQ_conv_dim * gQ_dconv;
        if (!d->conv_w || d->conv_w_len != conv_w_len || !d->a || d->a_len != gQ_valueheads ||
            !d->dt_bias || d->dt_bias_len != gQ_valueheads ||
            !d->norm || d->norm_len != gQ_statedim) return 0;
        L->conv_w = resident_floats(d->conv_w, d->conv_w_len);
        L->a = resident_floats(d->a, d->a_len);
        L->dt_bias = resident_floats(d->dt_bias, d->dt_bias_len);
        L->norm = resident_floats(d->norm, d->norm_len);
        NSUInteger conv_state_len = (NSUInteger)gQ_conv_dim * (gQ_dconv - 1);
        NSUInteger delta_state_len = (NSUInteger)gQ_valueheads * gQ_statedim * gQ_statedim;
        L->conv_state = resident_alloc_shared(conv_state_len * sizeof(float));
        L->delta_state = resident_alloc_shared(delta_state_len * sizeof(float));
        if (!L->conv_w || !L->a || !L->dt_bias || !L->norm ||
            !L->conv_state || !L->delta_state) return 0;
        memset([L->conv_state contents], 0, conv_state_len * sizeof(float));
        memset([L->delta_state contents], 0, delta_state_len * sizeof(float));
    } else {
        if (!d->q_norm || d->q_norm_len != gQ_headdim ||
            !d->k_norm || d->k_norm_len != gQ_headdim) return 0;
        L->q_norm = resident_floats(d->q_norm, d->q_norm_len);
        L->k_norm = resident_floats(d->k_norm, d->k_norm_len);
        L->k_cache = resident_alloc_private((NSUInteger)gQ_storage * gQ_kdim * sizeof(float));
        L->v_cache = resident_alloc_private((NSUInteger)gQ_storage * gQ_vdim * sizeof(float));
        if (!L->q_norm || !L->k_norm || !L->k_cache || !L->v_cache) return 0;
    }
    gQ_registered_layers += 1;
    return 1;
}

int rusty_metal_qwen_resident_set_output(const float *output_norm,
                                         const uint8_t *output_w,
                                         uintptr_t output_w_len,
                                         uint32_t output_rows,
                                         uint32_t output_dt,
                                         const float *inv_freq,
                                         uint32_t inv_freq_len) {
    if (!gDevice || !output_norm || !output_w || !inv_freq || output_dt > 1 ||
        output_rows != gQ_vocab || inv_freq_len < gQ_rotarydim / 2 ||
        gQ_registered_layers != gQ_nlayers) return 0;
    uint64_t expected_output_len = (uint64_t)output_rows * (gQ_dim / 256) *
                                   (output_dt == 1 ? 210u : 144u);
    if (expected_output_len != output_w_len) return 0;
    gQ_outnorm = resident_floats(output_norm, gQ_dim);
    gQ_invfreq = resident_floats(inv_freq, inv_freq_len);
    gQ_outw = qwen_resident_weight_buffer(output_w, output_w_len, &gQ_outw_offset);
    gQ_outrows = output_rows; gQ_outdt = output_dt;
    if (!gQ_outnorm || !gQ_invfreq || !gQ_outw) return 0;
    gQwenResidentReady = YES;
    return 1;
}

int rusty_metal_qwen_resident_decode(const float *x_embed, uint32_t pos,
                                     uint32_t start_t, int output_mode,
                                     float *logits_out, const uint32_t *recent,
                                     uint32_t recent_len, float repeat_penalty,
                                     uint32_t *token_out) {
    if (!gQwenResidentReady || !x_embed || pos >= gQ_storage || start_t > pos) return 0;
    if (output_mode == 1 && !logits_out) return 0;
    if (output_mode == 2 && (!token_out || recent_len > 64 || (recent_len && !recent))) return 0;
    if (output_mode < 0 || output_mode > 2) return 0;
    @autoreleasepool {
        if (pos == 0) {
            for (uint32_t l = 0; l < gQ_nlayers; ++l) {
                QwenResidentLayer *L = &gQLayers[l];
                if (L->type == 0) {
                    memset([L->conv_state contents], 0,
                           (NSUInteger)gQ_conv_dim * (gQ_dconv - 1) * sizeof(float));
                    memset([L->delta_state contents], 0,
                           (NSUInteger)gQ_valueheads * gQ_statedim * gQ_statedim * sizeof(float));
                }
            }
        }
        memcpy([gQ_x contents], x_embed, (NSUInteger)gQ_dim * sizeof(float));
        if (output_mode == 2 && recent_len) {
            memcpy([gQ_recent contents], recent, (NSUInteger)recent_len * sizeof(uint32_t));
        }
        double encode_start = rusty_metal_now_seconds();
        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        for (uint32_t l = 0; l < gQ_nlayers; ++l) {
            QwenResidentLayer *L = &gQLayers[l];
            id<MTLBuffer> residual = l == 0 ? gQ_zero : gQ_proj;
            resident_rms(enc, gQ_x, residual, L->attn_norm, gQ_xn, gQ_dim, gQ_eps);
            if (L->type == 0) {
                qwen_encode_matvec(enc, L->dt[0], L->w[0], L->w_offset[0],
                                   gQ_xn, gQ_joint, L->rows[0], L->cols[0]);
                qwen_encode_matvec(enc, L->dt[1], L->w[1], L->w_offset[1],
                                   gQ_xn, gQ_gate, L->rows[1], L->cols[1]);
                qwen_encode_q4_pair(enc, L->w[2], L->w_offset[2], L->w[3], L->w_offset[3],
                                    gQ_xn, gQ_alpha, gQ_beta,
                                    L->rows[2], L->rows[3], L->cols[2]);
                qwen_encode_conv(enc, L);
                qwen_encode_delta(enc, L);
                qwen_encode_delta_norm_gate(enc, L);
                qwen_encode_matvec(enc, L->dt[4], L->w[4], L->w_offset[4],
                                   gQ_hiddenbuf, gQ_proj, L->rows[4], L->cols[4]);
                resident_rms(enc, gQ_x, gQ_proj, L->post_norm, gQ_xn, gQ_dim, gQ_eps);
                qwen_encode_q4_pair(enc, L->w[5], L->w_offset[5], L->w[6], L->w_offset[6],
                                    gQ_xn, gQ_gate, gQ_up,
                                    L->rows[5], L->rows[6], L->cols[5]);
                resident_silu(enc, gQ_gate, gQ_up, gQ_hiddenbuf, gQ_hidden);
                qwen_encode_matvec(enc, L->dt[7], L->w[7], L->w_offset[7],
                                   gQ_hiddenbuf, gQ_proj, L->rows[7], L->cols[7]);
            } else {
                qwen_encode_matvec(enc, L->dt[0], L->w[0], L->w_offset[0],
                                   gQ_xn, gQ_joint, L->rows[0], L->cols[0]);
                qwen_encode_matvec(enc, L->dt[1], L->w[1], L->w_offset[1],
                                   gQ_xn, gQ_k, L->rows[1], L->cols[1]);
                qwen_encode_matvec(enc, L->dt[2], L->w[2], L->w_offset[2],
                                   gQ_xn, gQ_v, L->rows[2], L->cols[2]);
                qwen_encode_attention_norm(enc, L);
                qwen_encode_rope_store(enc, L, pos);
                qwen_encode_attention(enc, L, start_t, pos);
                qwen_encode_matvec(enc, L->dt[3], L->w[3], L->w_offset[3],
                                   gQ_attn, gQ_proj, L->rows[3], L->cols[3]);
                resident_rms(enc, gQ_x, gQ_proj, L->post_norm, gQ_xn, gQ_dim, gQ_eps);
                qwen_encode_q4_pair(enc, L->w[4], L->w_offset[4], L->w[5], L->w_offset[5],
                                    gQ_xn, gQ_gate, gQ_up,
                                    L->rows[4], L->rows[5], L->cols[4]);
                resident_silu(enc, gQ_gate, gQ_up, gQ_hiddenbuf, gQ_hidden);
                qwen_encode_matvec(enc, L->dt[6], L->w[6], L->w_offset[6],
                                   gQ_hiddenbuf, gQ_proj, L->rows[6], L->cols[6]);
            }
        }
        if (output_mode != 0) {
            resident_rms(enc, gQ_x, gQ_proj, gQ_outnorm, gQ_xn, gQ_dim, gQ_eps);
            qwen_encode_matvec(enc, gQ_outdt, gQ_outw, gQ_outw_offset,
                               gQ_xn, gQ_logits, gQ_outrows, gQ_dim);
            if (output_mode == 2) qwen_encode_greedy(enc, recent_len, repeat_penalty);
        }
        [enc endEncoding];
        double encode_end = rusty_metal_now_seconds();
        [cb commit];
        [cb waitUntilCompleted];
        rusty_metal_profile_command_buffer(cb, encode_start, encode_end);
        if ([cb status] != MTLCommandBufferStatusCompleted) return 0;
        if (output_mode == 1) {
            memcpy(logits_out, [gQ_logits contents], (NSUInteger)gQ_vocab * sizeof(float));
        } else if (output_mode == 2) {
            *token_out = *(const uint32_t *)[gQ_selected contents];
        }
        return 1;
    }
}
