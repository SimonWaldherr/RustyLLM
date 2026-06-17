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
    uint32_t len;
} RustyUnaryParams;

typedef struct {
    uint32_t len;
    float eps;
} RustyResidualNormParams;

static id<MTLDevice> gDevice;
static id<MTLCommandQueue> gQueue;
static id<MTLComputePipelineState> gQ4KPipeline;
static id<MTLComputePipelineState> gQ6KPipeline;
static id<MTLComputePipelineState> gQ4_0Pipeline;
static id<MTLComputePipelineState> gQ8_0Pipeline;
static id<MTLComputePipelineState> gAttentionPipeline;
static id<MTLComputePipelineState> gSiluMulPipeline;
static id<MTLComputePipelineState> gResidualRmsPipeline;
static id<MTLComputePipelineState> gResidualAddPipeline;
static NSMutableDictionary<NSNumber *, id<MTLBuffer>> *gWeightBuffers;
static NSMutableDictionary<NSNumber *, id<MTLBuffer>> *gSharedBuffers;
static const float gAttentionZero = 0.0f;

static BOOL rusty_metal_private_weights_enabled(void);

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
"// Simdgroup-reduced Q4_K matvec using simdgroup reduction.\n"
"// 32 threads collaborate on each output row; each lane owns ceil(n_blocks/32) blocks.\n"
"kernel void q4k_matvec(device const uchar* weights [[buffer(0)]],\n"
"                       device const float* x [[buffer(1)]],\n"
"                       device float* out [[buffer(2)]],\n"
"                       constant Params& p [[buffer(3)]],\n"
"                       uint group [[threadgroup_position_in_grid]],\n"
"                       uint sg [[simdgroup_index_in_threadgroup]],\n"
"                       uint lane [[thread_index_in_simdgroup]]) {\n"
"    uint row_half = lane >> 4;\n"
"    uint sublane = lane & 15;\n"
"    uint row = group * p.rows_per_group + sg * 2 + row_half;\n"
"    if (row >= p.rows) return;\n"
"    const device uchar* row_base = weights + row * p.row_bytes;\n"
"    float sum = 0.0f;\n"
"    for (uint b = sublane; b < p.n_blocks; b += 16) {\n"
"        const device uchar* block = row_base + b * 144;\n"
"        ushort db = ushort(block[0]) | (ushort(block[1]) << 8);\n"
"        ushort dmb = ushort(block[2]) | (ushort(block[3]) << 8);\n"
"        float d = float(as_type<half>(db));\n"
"        float dmin = float(as_type<half>(dmb));\n"
"        const device uchar* scales = block + 4;\n"
"        const device uchar* q = block + 16;\n"
"        uint xoff = b * 256;\n"
"        uint is = 0;\n"
"        #pragma unroll\n"
"        for (uint chunk = 0; chunk < 4; ++chunk) {\n"
"            uchar2 sm1 = scale_min_k4(is, scales);\n"
"            uchar2 sm2 = scale_min_k4(is + 1, scales);\n"
"            float d1 = d * float(sm1.x);\n"
"            float min1 = dmin * float(sm1.y);\n"
"            float d2 = d * float(sm2.x);\n"
"            float min2 = dmin * float(sm2.y);\n"
"            const device uchar* qchunk = q + chunk * 32;\n"
"            uint x1 = xoff + chunk * 64;\n"
"            uint x2 = x1 + 32;\n"
"            #pragma unroll(32)\n"
"            for (uint i = 0; i < 32; ++i) {\n"
"                uchar byte = qchunk[i];\n"
"                sum += (d1 * float(byte & 15) - min1) * x[x1 + i];\n"
"                sum += (d2 * float(byte >> 4) - min2) * x[x2 + i];\n"
"            }\n"
"            is += 2;\n"
"        }\n"
"    }\n"
"    for (ushort offset = 8; offset > 0; offset >>= 1)\n"
"        sum += simd_shuffle_xor(sum, offset);\n"
"    if (sublane == 0) out[row] = sum;\n"
"}\n";

static NSString *const kQ6KSource =
@"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct Params { uint rows; uint cols; uint row_bytes; uint n_blocks; uint rows_per_group; };\n"
"inline int i8(uchar v) { return v < 128 ? int(v) : int(v) - 256; }\n"
"// Simdgroup-reduced Q6_K matvec using simdgroup reduction.\n"
"kernel void q6k_matvec(device const uchar* weights [[buffer(0)]],\n"
"                       device const float* x [[buffer(1)]],\n"
"                       device float* out [[buffer(2)]],\n"
"                       constant Params& p [[buffer(3)]],\n"
"                       uint group [[threadgroup_position_in_grid]],\n"
"                       uint sg [[simdgroup_index_in_threadgroup]],\n"
"                       uint lane [[thread_index_in_simdgroup]]) {\n"
"    uint row_half = lane >> 4;\n"
"    uint sublane = lane & 15;\n"
"    uint row = group * p.rows_per_group + sg * 2 + row_half;\n"
"    if (row >= p.rows) return;\n"
"    const device uchar* row_base = weights + row * p.row_bytes;\n"
"    float sum = 0.0f;\n"
"    for (uint b = sublane; b < p.n_blocks; b += 16) {\n"
"        const device uchar* block = row_base + b * 210;\n"
"        const device uchar* ql = block;\n"
"        const device uchar* qh = block + 128;\n"
"        const device uchar* sc = block + 192;\n"
"        ushort db = ushort(block[208]) | (ushort(block[209]) << 8);\n"
"        float d = float(as_type<half>(db));\n"
"        uint xoff = b * 256;\n"
"        #pragma unroll\n"
"        for (uint step = 0; step < 2; ++step) {\n"
"            const device uchar* ql_sub = ql + step * 64;\n"
"            const device uchar* qh_sub = qh + step * 32;\n"
"            const device uchar* sc_sub = sc + step * 8;\n"
"            uint y = xoff + step * 128;\n"
"            float dsc0 = d * float(i8(sc_sub[0]));\n"
"            float dsc2 = d * float(i8(sc_sub[2]));\n"
"            float dsc4 = d * float(i8(sc_sub[4]));\n"
"            float dsc6 = d * float(i8(sc_sub[6]));\n"
"            #pragma unroll(16)\n"
"            for (uint l = 0; l < 16; ++l) {\n"
"                uchar ql0 = ql_sub[l];\n"
"                uchar ql32 = ql_sub[l + 32];\n"
"                uchar qh0 = qh_sub[l];\n"
"                sum += dsc0 * float(int((ql0 & 15) | ((qh0 & 3) << 4)) - 32) * x[y + l];\n"
"                sum += dsc2 * float(int((ql32 & 15) | (((qh0 >> 2) & 3) << 4)) - 32) * x[y + 32 + l];\n"
"                sum += dsc4 * float(int((ql0 >> 4) | (((qh0 >> 4) & 3) << 4)) - 32) * x[y + 64 + l];\n"
"                sum += dsc6 * float(int((ql32 >> 4) | (((qh0 >> 6) & 3) << 4)) - 32) * x[y + 96 + l];\n"
"            }\n"
"            float dsc1 = d * float(i8(sc_sub[1]));\n"
"            float dsc3 = d * float(i8(sc_sub[3]));\n"
"            float dsc5 = d * float(i8(sc_sub[5]));\n"
"            float dsc7 = d * float(i8(sc_sub[7]));\n"
"            #pragma unroll(16)\n"
"            for (uint l = 16; l < 32; ++l) {\n"
"                uchar ql0 = ql_sub[l];\n"
"                uchar ql32 = ql_sub[l + 32];\n"
"                uchar qh0 = qh_sub[l];\n"
"                sum += dsc1 * float(int((ql0 & 15) | ((qh0 & 3) << 4)) - 32) * x[y + l];\n"
"                sum += dsc3 * float(int((ql32 & 15) | (((qh0 >> 2) & 3) << 4)) - 32) * x[y + 32 + l];\n"
"                sum += dsc5 * float(int((ql0 >> 4) | (((qh0 >> 4) & 3) << 4)) - 32) * x[y + 64 + l];\n"
"                sum += dsc7 * float(int((ql32 >> 4) | (((qh0 >> 6) & 3) << 4)) - 32) * x[y + 96 + l];\n"
"            }\n"
"        }\n"
"    }\n"
"    for (ushort offset = 8; offset > 0; offset >>= 1)\n"
"        sum += simd_shuffle_xor(sum, offset);\n"
"    if (sublane == 0) out[row] = sum;\n"
"}\n";

// Q4_0: 18 bytes/block (2-byte f16 scale + 16 bytes of 32 nibbles), 32 elements/block.
// Simdgroup-reduced, same reduction pattern as Q4_K kernel above.
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

// Q8_0: 34 bytes/block (2-byte f16 scale + 32 bytes of int8), 32 elements/block.
// Simdgroup-reduced, same reduction pattern as Q4_K kernel above.
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
"    if (head >= p.heads) return;\n"
"    threadgroup float q_shared[2048];\n"
"    threadgroup float out_shared[2048];\n"
"    const device float* q_row = query + head * p.head_dim;\n"
"    device float* out_row = out + head * p.value_dim;\n"
"    uint kv_head = head / p.kv_mul;\n"
"    const device float* k_base = keys + kv_head * p.key_stride;\n"
"    const device float* v_base = values + kv_head * p.value_stride;\n"
"    for (uint i = lane; i < p.head_dim; i += 32) q_shared[i] = q_row[i];\n"
"    for (uint i = lane; i < p.value_dim; i += 32) out_shared[i] = 0.0f;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float max_score = p.use_sink != 0 ? sinks[head] : -INFINITY;\n"
"    float denom = p.use_sink != 0 ? 1.0f : 0.0f;\n"
"    for (uint t = p.start_t; t <= p.end_t; ++t) {\n"
"        uint slot = t % p.slot_count;\n"
"        const device float* k_row = k_base + slot * p.key_stride;\n"
"        const device float* v_row = v_base + slot * p.value_stride;\n"
"        float partial = 0.0f;\n"
"        for (uint i = lane; i < p.head_dim; i += 32) {\n"
"            partial += q_shared[i] * k_row[i];\n"
"        }\n"
"        for (ushort offset = 16; offset > 0; offset >>= 1) {\n"
"            partial += simd_shuffle_xor(partial, offset);\n"
"        }\n"
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
"                acc_scale = 1.0f;\n"
"                value_scale = exp(score - max_score);\n"
"                denom += value_scale;\n"
"            }\n"
"        }\n"
"        acc_scale = simd_broadcast_first(acc_scale);\n"
"        value_scale = simd_broadcast_first(value_scale);\n"
"        for (uint i = lane; i < p.value_dim; i += 32) {\n"
"            out_shared[i] = out_shared[i] * acc_scale + value_scale * v_row[i];\n"
"        }\n"
"    }\n"
"    float inv_denom = lane == 0 ? (denom > 0.0f ? 1.0f / denom : 0.0f) : 0.0f;\n"
"    inv_denom = simd_broadcast_first(inv_denom);\n"
"    for (uint i = lane; i < p.value_dim; i += 32) {\n"
"        out_row[i] = out_shared[i] * inv_denom;\n"
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
        gQueue = [gDevice newCommandQueue];
        if (!gQueue) {
            rusty_metal_log_error("create command queue", nil);
            return;
        }
        gWeightBuffers = [[NSMutableDictionary alloc] init];
        gSharedBuffers = [[NSMutableDictionary alloc] init];
        ok = YES;
    });
    return ok;
}

static id<MTLBuffer> rusty_metal_weight_buffer(const uint8_t *weights, uintptr_t weights_len) {
    NSNumber *key = @((uintptr_t)weights);
    id<MTLBuffer> weight_buffer = [gWeightBuffers objectForKey:key];
    if (!weight_buffer || [weight_buffer length] < weights_len) {
        if (rusty_metal_private_weights_enabled()) {
            id<MTLBuffer> staging = [gDevice newBufferWithBytes:weights
                                                         length:(NSUInteger)weights_len
                                                        options:MTLResourceStorageModeShared];
            weight_buffer = [gDevice newBufferWithLength:(NSUInteger)weights_len
                                                 options:MTLResourceStorageModePrivate];
            if (staging && weight_buffer) {
                id<MTLCommandBuffer> command_buffer = [gQueue commandBuffer];
                id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
                [blit copyFromBuffer:staging
                         sourceOffset:0
                             toBuffer:weight_buffer
                    destinationOffset:0
                                 size:(NSUInteger)weights_len];
                [blit endEncoding];
                [command_buffer commit];
                [command_buffer waitUntilCompleted];
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
        }
        if (!weight_buffer) return nil;
        [gWeightBuffers setObject:weight_buffer forKey:key];
    }
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
        [gSharedBuffers setObject:buffer forKey:key];
    }
    return buffer;
}

static id<MTLBuffer> rusty_metal_copy_buffer(const void *bytes, uintptr_t bytes_len) {
    return [gDevice newBufferWithBytes:bytes
                                length:(NSUInteger)bytes_len
                               options:MTLResourceStorageModeShared];
}

static BOOL rusty_metal_env_disabled(const char *name) {
    const char *value = getenv(name);
    if (!value) return NO;
    return strcmp(value, "0") == 0 ||
           strcasecmp(value, "false") == 0 ||
           strcasecmp(value, "no") == 0 ||
           strcasecmp(value, "off") == 0;
}

static BOOL rusty_metal_nocopy_enabled(void) {
    return !rusty_metal_env_disabled("RUSTY_LLM_METAL_NOCOPY");
}

static BOOL rusty_metal_private_weights_enabled(void) {
    const char *value = getenv("RUSTY_LLM_METAL_PRIVATE_WEIGHTS");
    return value && !rusty_metal_env_disabled("RUSTY_LLM_METAL_PRIVATE_WEIGHTS");
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
    (void)rows;
    return 8;
}

static BOOL rusty_metal_ensure_buffer(id<MTLBuffer> __strong *buffer, NSUInteger size) {
    if (!*buffer || [*buffer length] < size) {
        *buffer = [gDevice newBufferWithLength:size options:MTLResourceStorageModeShared];
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
    return *copy_buffer;
}

static void rusty_metal_encode_q4k(id<MTLComputeCommandEncoder> encoder,
                                   id<MTLBuffer> weight_buffer,
                                   id<MTLBuffer> x_buffer,
                                   id<MTLBuffer> out_buffer,
                                   uintptr_t rows,
                                   uintptr_t cols) {
    const NSUInteger rows_per_group = 8;
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

    MTLSize threads_per_group = MTLSizeMake(RUSTY_MATVEC_THREADS_PER_GROUP, 1, 1);
    MTLSize threadgroups = MTLSizeMake(
        ((NSUInteger)rows + rows_per_group - 1) / rows_per_group,
        1,
        1
    );
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_group];
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
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_group];
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

        id<MTLCommandBuffer> command_buffer = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        rusty_metal_encode_q4k(encoder, weight_buffer, x_metal, out_metal, rows, cols);
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if ([command_buffer status] != MTLCommandBufferStatusCompleted) return 0;

        if (out_needs_copy) memcpy(out, [out_metal contents], out_size);
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

        id<MTLCommandBuffer> command_buffer = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        rusty_metal_encode_q4k(encoder, weight_a, x_metal, out_a_metal, rows_a, cols);
        rusty_metal_encode_q4k(encoder, weight_b, x_metal, out_b_metal, rows_b, cols);
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if ([command_buffer status] != MTLCommandBufferStatusCompleted) return 0;

        if (out_a_needs_copy) memcpy(out_a, [out_a_metal contents], out_a_size);
        if (out_b_needs_copy) memcpy(out_b, [out_b_metal contents], out_b_size);
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

        id<MTLCommandBuffer> command_buffer = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        rusty_metal_encode_q6k(encoder, weight_buffer, x_metal, out_metal, rows, cols);
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if ([command_buffer status] != MTLCommandBufferStatusCompleted) return 0;

        if (out_needs_copy) memcpy(out, [out_metal contents], out_size);
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

        id<MTLCommandBuffer> command_buffer = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        rusty_metal_encode_q6k(encoder, weight_a, x_metal, out_a_metal, rows_a, cols);
        rusty_metal_encode_q6k(encoder, weight_b, x_metal, out_b_metal, rows_b, cols);
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if ([command_buffer status] != MTLCommandBufferStatusCompleted) return 0;

        if (out_a_needs_copy) memcpy(out_a, [out_a_metal contents], out_a_size);
        if (out_b_needs_copy) memcpy(out_b, [out_b_metal contents], out_b_size);
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

        id<MTLCommandBuffer> command_buffer = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        rusty_metal_encode_q6k(encoder, weight_a, x_metal, out_a_metal, rows_a, cols);
        rusty_metal_encode_q6k(encoder, weight_b, x_metal, out_b_metal, rows_b, cols);
        rusty_metal_encode_q6k(encoder, weight_c, x_metal, out_c_metal, rows_c, cols);
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if ([command_buffer status] != MTLCommandBufferStatusCompleted) return 0;

        if (out_a_needs_copy) memcpy(out_a, [out_a_metal contents], out_a_size);
        if (out_b_needs_copy) memcpy(out_b, [out_b_metal contents], out_b_size);
        if (out_c_needs_copy) memcpy(out_c, [out_c_metal contents], out_c_size);
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

        id<MTLCommandBuffer> command_buffer = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        rusty_metal_encode_q4k(encoder, weight_a, x_metal, out_a_metal, rows_a, cols);
        rusty_metal_encode_q4k(encoder, weight_b, x_metal, out_b_metal, rows_b, cols);
        rusty_metal_encode_q4k(encoder, weight_c, x_metal, out_c_metal, rows_c, cols);
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if ([command_buffer status] != MTLCommandBufferStatusCompleted) return 0;

        if (out_a_needs_copy) memcpy(out_a, [out_a_metal contents], out_a_size);
        if (out_b_needs_copy) memcpy(out_b, [out_b_metal contents], out_b_size);
        if (out_c_needs_copy) memcpy(out_c, [out_c_metal contents], out_c_size);
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

        id<MTLCommandBuffer> command_buffer = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        rusty_metal_encode_q4k(encoder, weight_a, x_metal, out_a_metal, rows_a, cols);
        rusty_metal_encode_q4k(encoder, weight_b, x_metal, out_b_metal, rows_b, cols);
        rusty_metal_encode_q6k(encoder, weight_c, x_metal, out_c_metal, rows_c, cols);
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if ([command_buffer status] != MTLCommandBufferStatusCompleted) return 0;

        if (out_a_needs_copy) memcpy(out_a, [out_a_metal contents], out_a_size);
        if (out_b_needs_copy) memcpy(out_b, [out_b_metal contents], out_b_size);
        if (out_c_needs_copy) memcpy(out_c, [out_c_metal contents], out_c_size);
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

        RustyUnaryParams silu_params = {
            .len = (uint32_t)hidden_rows,
        };

        id<MTLCommandBuffer> command_buffer = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        rusty_metal_encode_q4k(encoder, gate_weight, x_metal, gate_buffer, hidden_rows, input_cols);
        rusty_metal_encode_q4k(encoder, up_weight, x_metal, up_buffer, hidden_rows, input_cols);
        [encoder setComputePipelineState:gSiluMulPipeline];
        [encoder setBuffer:gate_buffer offset:0 atIndex:0];
        [encoder setBuffer:up_buffer offset:0 atIndex:1];
        [encoder setBuffer:hidden_buffer offset:0 atIndex:2];
        [encoder setBytes:&silu_params length:sizeof(silu_params) atIndex:3];
        MTLSize silu_threads = MTLSizeMake(256, 1, 1);
        MTLSize silu_groups = MTLSizeMake(((NSUInteger)hidden_rows + 255) / 256, 1, 1);
        [encoder dispatchThreadgroups:silu_groups threadsPerThreadgroup:silu_threads];
        rusty_metal_encode_q6k(encoder, down_weight, hidden_buffer, out_metal, down_rows, down_cols);
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if ([command_buffer status] != MTLCommandBufferStatusCompleted) return 0;

        if (out_needs_copy) memcpy(out, [out_metal contents], out_size);
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

        RustyResidualNormParams norm_params = {
            .len = (uint32_t)dim,
            .eps = rms_eps,
        };
        RustyUnaryParams hidden_params = {
            .len = (uint32_t)hidden_rows,
        };
        RustyUnaryParams add_params = {
            .len = (uint32_t)dim,
        };

        id<MTLCommandBuffer> command_buffer = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        rusty_metal_encode_q4k(encoder, wo_weight, attn_metal, proj_buffer, dim, attn_cols);
        [encoder setComputePipelineState:gResidualRmsPipeline];
        [encoder setBuffer:x_metal offset:0 atIndex:0];
        [encoder setBuffer:proj_buffer offset:0 atIndex:1];
        [encoder setBuffer:norm_weight_metal offset:0 atIndex:2];
        [encoder setBuffer:norm_buffer offset:0 atIndex:3];
        [encoder setBytes:&norm_params length:sizeof(norm_params) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        rusty_metal_encode_q4k(encoder, gate_weight, norm_buffer, gate_buffer, hidden_rows, dim);
        rusty_metal_encode_q4k(encoder, up_weight, norm_buffer, up_buffer, hidden_rows, dim);
        [encoder setComputePipelineState:gSiluMulPipeline];
        [encoder setBuffer:gate_buffer offset:0 atIndex:0];
        [encoder setBuffer:up_buffer offset:0 atIndex:1];
        [encoder setBuffer:hidden_buffer offset:0 atIndex:2];
        [encoder setBytes:&hidden_params length:sizeof(hidden_params) atIndex:3];
        MTLSize silu_threads = MTLSizeMake(256, 1, 1);
        MTLSize silu_groups = MTLSizeMake(((NSUInteger)hidden_rows + 255) / 256, 1, 1);
        [encoder dispatchThreadgroups:silu_groups threadsPerThreadgroup:silu_threads];
        rusty_metal_encode_q6k(encoder, down_weight, hidden_buffer, proj_buffer, down_rows, down_cols);
        [encoder setComputePipelineState:gResidualAddPipeline];
        [encoder setBuffer:x_metal offset:0 atIndex:0];
        [encoder setBuffer:proj_buffer offset:0 atIndex:1];
        [encoder setBytes:&add_params length:sizeof(add_params) atIndex:2];
        MTLSize add_threads = MTLSizeMake(256, 1, 1);
        MTLSize add_groups = MTLSizeMake(((NSUInteger)dim + 255) / 256, 1, 1);
        [encoder dispatchThreadgroups:add_groups threadsPerThreadgroup:add_threads];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if ([command_buffer status] != MTLCommandBufferStatusCompleted) return 0;

        if (x_needs_copy) memcpy(x, [x_metal contents], x_size);
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

        if (!rusty_metal_ensure_buffer(&x_buffer, x_size)) return 0;
        if (!rusty_metal_ensure_buffer(&out_buffer, out_size)) return 0;

        memcpy([x_buffer contents], x, x_size);

        id<MTLCommandBuffer> command_buffer = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        rusty_metal_encode_q4_0(encoder, weight_buffer, x_buffer, out_buffer, rows, cols);
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if ([command_buffer status] != MTLCommandBufferStatusCompleted) return 0;

        memcpy(out, [out_buffer contents], out_size);
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

        if (!rusty_metal_ensure_buffer(&x_buffer, x_size)) return 0;
        if (!rusty_metal_ensure_buffer(&out_buffer, out_size)) return 0;

        memcpy([x_buffer contents], x, x_size);

        id<MTLCommandBuffer> command_buffer = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        rusty_metal_encode_q8_0(encoder, weight_buffer, x_buffer, out_buffer, rows, cols);
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if ([command_buffer status] != MTLCommandBufferStatusCompleted) return 0;

        memcpy(out, [out_buffer contents], out_size);
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
        head_dim == 0 || value_dim == 0 || slot_count == 0 || head_dim > 2048 || value_dim > 2048) {
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
            sinks_buffer = rusty_metal_copy_buffer(sinks, sinks_len);
        } else {
            sinks_buffer = rusty_metal_copy_buffer(&gAttentionZero, sizeof(gAttentionZero));
        }
        if (!sinks_buffer) return 0;

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

        id<MTLCommandBuffer> command_buffer = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:gAttentionPipeline];
        [encoder setBuffer:query_buffer offset:0 atIndex:0];
        [encoder setBuffer:keys_buffer offset:0 atIndex:1];
        [encoder setBuffer:values_buffer offset:0 atIndex:2];
        [encoder setBuffer:out_buffer offset:0 atIndex:3];
        [encoder setBuffer:sinks_buffer offset:0 atIndex:4];
        [encoder setBytes:&params length:sizeof(params) atIndex:5];

        MTLSize threads_per_group = MTLSizeMake(32, 1, 1);
        MTLSize threadgroups = MTLSizeMake((NSUInteger)heads, 1, 1);
        [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_group];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if ([command_buffer status] != MTLCommandBufferStatusCompleted) return 0;

        if (out_needs_copy) memcpy(out, [out_buffer contents], out_bytes);
        return 1;
    }
}
