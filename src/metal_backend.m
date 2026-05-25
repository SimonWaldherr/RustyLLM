#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    uint32_t rows;
    uint32_t cols;
    uint32_t row_bytes;
    uint32_t n_blocks;
} RustyQ4KParams;

static id<MTLDevice> gDevice;
static id<MTLCommandQueue> gQueue;
static id<MTLComputePipelineState> gQ4KPipeline;
static id<MTLComputePipelineState> gQ6KPipeline;
static id<MTLComputePipelineState> gQ4_0Pipeline;
static id<MTLComputePipelineState> gQ8_0Pipeline;
static NSMutableDictionary<NSNumber *, id<MTLBuffer>> *gWeightBuffers;

static NSString *const kQ4KSource =
@"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct Params { uint rows; uint cols; uint row_bytes; uint n_blocks; };\n"
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
"                       uint row  [[threadgroup_position_in_grid]],\n"
"                       uint lane [[thread_index_in_simdgroup]]) {\n"
"    if (row >= p.rows) return;\n"
"    const device uchar* row_base = weights + row * p.row_bytes;\n"
"    float sum = 0.0f;\n"
"    for (uint b = lane; b < p.n_blocks; b += 32) {\n"
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
"    for (ushort offset = 16; offset > 0; offset >>= 1)\n"
"        sum += simd_shuffle_xor(sum, offset);\n"
"    if (lane == 0) out[row] = sum;\n"
"}\n";

static NSString *const kQ6KSource =
@"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct Params { uint rows; uint cols; uint row_bytes; uint n_blocks; };\n"
"inline int i8(uchar v) { return v < 128 ? int(v) : int(v) - 256; }\n"
"// Simdgroup-reduced Q6_K matvec using simdgroup reduction.\n"
"kernel void q6k_matvec(device const uchar* weights [[buffer(0)]],\n"
"                       device const float* x [[buffer(1)]],\n"
"                       device float* out [[buffer(2)]],\n"
"                       constant Params& p [[buffer(3)]],\n"
"                       uint row  [[threadgroup_position_in_grid]],\n"
"                       uint lane [[thread_index_in_simdgroup]]) {\n"
"    if (row >= p.rows) return;\n"
"    const device uchar* row_base = weights + row * p.row_bytes;\n"
"    float sum = 0.0f;\n"
"    for (uint b = lane; b < p.n_blocks; b += 32) {\n"
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
"    for (ushort offset = 16; offset > 0; offset >>= 1)\n"
"        sum += simd_shuffle_xor(sum, offset);\n"
"    if (lane == 0) out[row] = sum;\n"
"}\n";

// Q4_0: 18 bytes/block (2-byte f16 scale + 16 bytes of 32 nibbles), 32 elements/block.
// Simdgroup-reduced, same reduction pattern as Q4_K kernel above.
static NSString *const kQ4_0Source =
@"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct Params { uint rows; uint cols; uint row_bytes; uint n_blocks; };\n"
"kernel void q4_0_matvec(device const uchar* weights [[buffer(0)]],\n"
"                        device const float* x [[buffer(1)]],\n"
"                        device float* out [[buffer(2)]],\n"
"                        constant Params& p [[buffer(3)]],\n"
"                        uint row  [[threadgroup_position_in_grid]],\n"
"                        uint lane [[thread_index_in_simdgroup]]) {\n"
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
"struct Params { uint rows; uint cols; uint row_bytes; uint n_blocks; };\n"
"kernel void q8_0_matvec(device const uchar* weights [[buffer(0)]],\n"
"                        device const float* x [[buffer(1)]],\n"
"                        device float* out [[buffer(2)]],\n"
"                        constant Params& p [[buffer(3)]],\n"
"                        uint row  [[threadgroup_position_in_grid]],\n"
"                        uint lane [[thread_index_in_simdgroup]]) {\n"
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
static BOOL rusty_metal_init(void) {
    static dispatch_once_t once;
    static BOOL ok = NO;
    dispatch_once(&once, ^{
        gDevice = MTLCreateSystemDefaultDevice();
        if (!gDevice) return;
        NSError *error = nil;
        MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
        options.fastMathEnabled = YES;
        id<MTLLibrary> library = [gDevice newLibraryWithSource:kQ4KSource options:options error:&error];
        if (!library) return;
        id<MTLFunction> function = [library newFunctionWithName:@"q4k_matvec"];
        if (!function) return;
        gQ4KPipeline = [gDevice newComputePipelineStateWithFunction:function error:&error];
        if (!gQ4KPipeline) return;
        id<MTLLibrary> q6_library = [gDevice newLibraryWithSource:kQ6KSource options:options error:&error];
        if (!q6_library) return;
        id<MTLFunction> q6_function = [q6_library newFunctionWithName:@"q6k_matvec"];
        if (!q6_function) return;
        gQ6KPipeline = [gDevice newComputePipelineStateWithFunction:q6_function error:&error];
        if (!gQ6KPipeline) return;
        id<MTLLibrary> q4_0_library = [gDevice newLibraryWithSource:kQ4_0Source options:options error:&error];
        if (!q4_0_library) return;
        id<MTLFunction> q4_0_function = [q4_0_library newFunctionWithName:@"q4_0_matvec"];
        if (!q4_0_function) return;
        gQ4_0Pipeline = [gDevice newComputePipelineStateWithFunction:q4_0_function error:&error];
        if (!gQ4_0Pipeline) return;
        id<MTLLibrary> q8_0_library = [gDevice newLibraryWithSource:kQ8_0Source options:options error:&error];
        if (!q8_0_library) return;
        id<MTLFunction> q8_0_function = [q8_0_library newFunctionWithName:@"q8_0_matvec"];
        if (!q8_0_function) return;
        gQ8_0Pipeline = [gDevice newComputePipelineStateWithFunction:q8_0_function error:&error];
        if (!gQ8_0Pipeline) return;
        gQueue = [gDevice newCommandQueue];
        if (!gQueue) return;
        gWeightBuffers = [[NSMutableDictionary alloc] init];
        ok = YES;
    });
    return ok;
}

static id<MTLBuffer> rusty_metal_weight_buffer(const uint8_t *weights, uintptr_t weights_len) {
    NSNumber *key = @((uintptr_t)weights);
    id<MTLBuffer> weight_buffer = [gWeightBuffers objectForKey:key];
    if (!weight_buffer || [weight_buffer length] < weights_len) {
        weight_buffer = [gDevice newBufferWithBytes:weights
                                             length:(NSUInteger)weights_len
                                            options:MTLResourceStorageModeShared];
        if (!weight_buffer) return nil;
        [gWeightBuffers setObject:weight_buffer forKey:key];
    }
    return weight_buffer;
}

static BOOL rusty_metal_ensure_buffer(id<MTLBuffer> __strong *buffer, NSUInteger size) {
    if (!*buffer || [*buffer length] < size) {
        *buffer = [gDevice newBufferWithLength:size options:MTLResourceStorageModeShared];
    }
    return *buffer != nil;
}

static void rusty_metal_encode_q4k(id<MTLComputeCommandEncoder> encoder,
                                   id<MTLBuffer> weight_buffer,
                                   id<MTLBuffer> x_buffer,
                                   id<MTLBuffer> out_buffer,
                                   uintptr_t rows,
                                   uintptr_t cols) {
    RustyQ4KParams params = {
        .rows = (uint32_t)rows,
        .cols = (uint32_t)cols,
        .row_bytes = (uint32_t)((cols / 256) * 144),
        .n_blocks = (uint32_t)(cols / 256),
    };

    [encoder setComputePipelineState:gQ4KPipeline];
    [encoder setBuffer:weight_buffer offset:0 atIndex:0];
    [encoder setBuffer:x_buffer offset:0 atIndex:1];
    [encoder setBuffer:out_buffer offset:0 atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];

    // One simdgroup (32 threads) per row — simd_shuffle_xor reduction across lanes.
    MTLSize threads_per_group = MTLSizeMake(32, 1, 1);
    MTLSize threadgroups = MTLSizeMake((NSUInteger)rows, 1, 1);
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_group];
}

static void rusty_metal_encode_q6k(id<MTLComputeCommandEncoder> encoder,
                                   id<MTLBuffer> weight_buffer,
                                   id<MTLBuffer> x_buffer,
                                   id<MTLBuffer> out_buffer,
                                   uintptr_t rows,
                                   uintptr_t cols) {
    RustyQ4KParams params = {
        .rows = (uint32_t)rows,
        .cols = (uint32_t)cols,
        .row_bytes = (uint32_t)((cols / 256) * 210),
        .n_blocks = (uint32_t)(cols / 256),
    };

    [encoder setComputePipelineState:gQ6KPipeline];
    [encoder setBuffer:weight_buffer offset:0 atIndex:0];
    [encoder setBuffer:x_buffer offset:0 atIndex:1];
    [encoder setBuffer:out_buffer offset:0 atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];

    // One simdgroup (32 threads) per row — simd_shuffle_xor reduction across lanes.
    MTLSize threads_per_group = MTLSizeMake(32, 1, 1);
    MTLSize threadgroups = MTLSizeMake((NSUInteger)rows, 1, 1);
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
    };

    [encoder setComputePipelineState:gQ4_0Pipeline];
    [encoder setBuffer:weight_buffer offset:0 atIndex:0];
    [encoder setBuffer:x_buffer offset:0 atIndex:1];
    [encoder setBuffer:out_buffer offset:0 atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];

    MTLSize threads_per_group = MTLSizeMake(32, 1, 1);
    MTLSize threadgroups = MTLSizeMake((NSUInteger)rows, 1, 1);
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
    };

    [encoder setComputePipelineState:gQ8_0Pipeline];
    [encoder setBuffer:weight_buffer offset:0 atIndex:0];
    [encoder setBuffer:x_buffer offset:0 atIndex:1];
    [encoder setBuffer:out_buffer offset:0 atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];

    MTLSize threads_per_group = MTLSizeMake(32, 1, 1);
    MTLSize threadgroups = MTLSizeMake((NSUInteger)rows, 1, 1);
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

        if (!rusty_metal_ensure_buffer(&x_buffer, x_size)) return 0;
        if (!rusty_metal_ensure_buffer(&out_buffer, out_size)) return 0;

        memcpy([x_buffer contents], x, x_size);

        id<MTLCommandBuffer> command_buffer = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        rusty_metal_encode_q4k(encoder, weight_buffer, x_buffer, out_buffer, rows, cols);
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if ([command_buffer status] != MTLCommandBufferStatusCompleted) return 0;

        memcpy(out, [out_buffer contents], out_size);
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

        if (!rusty_metal_ensure_buffer(&x_buffer, x_size)) return 0;
        if (!rusty_metal_ensure_buffer(&out_a_buffer, out_a_size)) return 0;
        if (!rusty_metal_ensure_buffer(&out_b_buffer, out_b_size)) return 0;

        memcpy([x_buffer contents], x, x_size);

        id<MTLCommandBuffer> command_buffer = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        rusty_metal_encode_q4k(encoder, weight_a, x_buffer, out_a_buffer, rows_a, cols);
        rusty_metal_encode_q4k(encoder, weight_b, x_buffer, out_b_buffer, rows_b, cols);
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if ([command_buffer status] != MTLCommandBufferStatusCompleted) return 0;

        memcpy(out_a, [out_a_buffer contents], out_a_size);
        memcpy(out_b, [out_b_buffer contents], out_b_size);
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

        if (!rusty_metal_ensure_buffer(&x_buffer, x_size)) return 0;
        if (!rusty_metal_ensure_buffer(&out_buffer, out_size)) return 0;

        memcpy([x_buffer contents], x, x_size);

        id<MTLCommandBuffer> command_buffer = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        rusty_metal_encode_q6k(encoder, weight_buffer, x_buffer, out_buffer, rows, cols);
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if ([command_buffer status] != MTLCommandBufferStatusCompleted) return 0;

        memcpy(out, [out_buffer contents], out_size);
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

        if (!rusty_metal_ensure_buffer(&x_buffer, x_size)) return 0;
        if (!rusty_metal_ensure_buffer(&out_a_buffer, out_a_size)) return 0;
        if (!rusty_metal_ensure_buffer(&out_b_buffer, out_b_size)) return 0;

        memcpy([x_buffer contents], x, x_size);

        id<MTLCommandBuffer> command_buffer = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        rusty_metal_encode_q6k(encoder, weight_a, x_buffer, out_a_buffer, rows_a, cols);
        rusty_metal_encode_q6k(encoder, weight_b, x_buffer, out_b_buffer, rows_b, cols);
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if ([command_buffer status] != MTLCommandBufferStatusCompleted) return 0;

        memcpy(out_a, [out_a_buffer contents], out_a_size);
        memcpy(out_b, [out_b_buffer contents], out_b_size);
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

        if (!rusty_metal_ensure_buffer(&x_buffer, x_size)) return 0;
        if (!rusty_metal_ensure_buffer(&out_a_buffer, out_a_size)) return 0;
        if (!rusty_metal_ensure_buffer(&out_b_buffer, out_b_size)) return 0;
        if (!rusty_metal_ensure_buffer(&out_c_buffer, out_c_size)) return 0;

        memcpy([x_buffer contents], x, x_size);

        id<MTLCommandBuffer> command_buffer = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        rusty_metal_encode_q6k(encoder, weight_a, x_buffer, out_a_buffer, rows_a, cols);
        rusty_metal_encode_q6k(encoder, weight_b, x_buffer, out_b_buffer, rows_b, cols);
        rusty_metal_encode_q6k(encoder, weight_c, x_buffer, out_c_buffer, rows_c, cols);
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if ([command_buffer status] != MTLCommandBufferStatusCompleted) return 0;

        memcpy(out_a, [out_a_buffer contents], out_a_size);
        memcpy(out_b, [out_b_buffer contents], out_b_size);
        memcpy(out_c, [out_c_buffer contents], out_c_size);
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

        if (!rusty_metal_ensure_buffer(&x_buffer, x_size)) return 0;
        if (!rusty_metal_ensure_buffer(&out_a_buffer, out_a_size)) return 0;
        if (!rusty_metal_ensure_buffer(&out_b_buffer, out_b_size)) return 0;
        if (!rusty_metal_ensure_buffer(&out_c_buffer, out_c_size)) return 0;

        memcpy([x_buffer contents], x, x_size);

        id<MTLCommandBuffer> command_buffer = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        rusty_metal_encode_q4k(encoder, weight_a, x_buffer, out_a_buffer, rows_a, cols);
        rusty_metal_encode_q4k(encoder, weight_b, x_buffer, out_b_buffer, rows_b, cols);
        rusty_metal_encode_q4k(encoder, weight_c, x_buffer, out_c_buffer, rows_c, cols);
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if ([command_buffer status] != MTLCommandBufferStatusCompleted) return 0;

        memcpy(out_a, [out_a_buffer contents], out_a_size);
        memcpy(out_b, [out_b_buffer contents], out_b_size);
        memcpy(out_c, [out_c_buffer contents], out_c_size);
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
