// GPU Custom Call Prototype: Vector Add
// This validates that GPU custom calls work in EXLA before implementing FlashAttention.
//
// When CUDA_ENABLED is defined (nvcc detected), this registers a GPU handler.
// Otherwise, it's a no-op.

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

#ifdef CUDA_ENABLED
#include <cuda_runtime.h>

// Simple CUDA kernel for element-wise add
__global__ void vector_add_kernel(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

// GPU handler implementation - runs on CPU but launches GPU kernel
static ffi::Error gpu_add_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> a,
    ffi::Buffer<ffi::F32> b,
    ffi::Result<ffi::Buffer<ffi::F32>> out
) {
    // Get buffer dimensions
    auto dims = a.dimensions();
    int n = 1;
    for (auto d : dims) {
        n *= d;
    }

    // Get device pointers (already on GPU!)
    const float* a_ptr = a.typed_data();
    const float* b_ptr = b.typed_data();
    float* out_ptr = out->typed_data();

    // Launch kernel
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    vector_add_kernel<<<grid_size, block_size, 0, stream>>>(
        a_ptr, b_ptr, out_ptr, n
    );

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error::InvalidArgument(cudaGetErrorString(err));
    }

    return ffi::Error::Success();
}

// Define and register the GPU handler
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    gpu_add,
    gpu_add_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()  // a
        .Arg<ffi::Buffer<ffi::F32>>()  // b
        .Ret<ffi::Buffer<ffi::F32>>()  // out
);

XLA_FFI_REGISTER_HANDLER(
    ffi::GetXlaFfiApi(),
    "exla_gpu_add_f32",
    "CUDA",
    gpu_add
);

#endif  // CUDA_ENABLED
