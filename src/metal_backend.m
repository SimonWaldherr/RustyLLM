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
static NSMutableDictionary<NSNumber *, id<MTLBuffer>> *gWeightBuffers;

static NSString *const kQ4KSource =
@"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct Params { uint rows; uint cols; uint row_bytes; uint n_blocks; };\n"
"inline uchar2 scale_min_k4(uint j, const device uchar* q) {\n"
"    if (j < 4) return uchar2(q[j] & 63, q[j + 4] & 63);\n"
"    return uchar2((q[j + 4] & 15) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4));\n"
"}\n"
"kernel void q4k_matvec(device const uchar* weights [[buffer(0)]],\n"
"                       device const float* x [[buffer(1)]],\n"
"                       device float* out [[buffer(2)]],\n"
"                       constant Params& p [[buffer(3)]],\n"
"                       uint row [[thread_position_in_grid]]) {\n"
"    if (row >= p.rows) return;\n"
"    const device uchar* row_base = weights + row * p.row_bytes;\n"
"    float sum = 0.0f;\n"
"    for (uint b = 0; b < p.n_blocks; ++b) {\n"
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
"    out[row] = sum;\n"
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

    NSUInteger width = MIN((NSUInteger)gQ4KPipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)256);
    MTLSize threads_per_group = MTLSizeMake(width, 1, 1);
    MTLSize threads = MTLSizeMake((NSUInteger)rows, 1, 1);
    [encoder dispatchThreads:threads threadsPerThreadgroup:threads_per_group];
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
