// Fused DeltaProduct Scan Kernel
//
// Extends DeltaNet with multiple Householder transformation steps per token.
// At each timestep t, applies n_h sequential rank-1 updates to the state matrix:
//
//   For j in 0..n_h-1:
//     k_norm = k_{t,j} / ||k_{t,j}||_2        (L2 normalize key)
//     S = S - beta_{t,j} * (k_norm * k_norm^T @ S) + beta_{t,j} * (k_norm * v_{t,j}^T)
//
//   o_t = RMS_norm(S_t @ q_t)
//
// This is equivalent to: S = (I - beta * k*k^T) * S + beta * k*v^T
// which is a generalized Householder reflection + rank-1 update.
//
// Thread layout: one block per (batch, head), head_dim threads per block.
// Each thread owns one row of S[head_dim][head_dim] in shared memory.
//
// Inputs:
//   q:    [B, T, H, d]         — query vectors (shared across Householder steps)
//   k:    [B, T, n_h, H, d]    — key vectors per Householder step (pre-normalization)
//   v:    [B, T, n_h, H, d]    — value vectors per Householder step
//   beta: [B, T, n_h, H]       — scalar gate per head per step (post-sigmoid)
//
// Output:
//   out:  [B, T, H, d]         — RMS-normalized output
//
// Shared memory budget (head_dim=64):
//   S matrix:  64*64*4 = 16KB
//   k_shared:  64*4    = 256B
//   q_shared:  64*4    = 256B
//   rms_buf:   1*4     = 4B   (for warp reduction)
//   Total: ~17KB — well within 48KB limit

#include <cuda_runtime.h>
#include "precision.cuh"

constexpr float NORM_EPS = 1.0e-6f;

// ============================================================================
// Kernel
// ============================================================================

__global__ void fused_delta_product_scan_kernel(
    const io_type* __restrict__ q,       // [B, T, H, d]
    const io_type* __restrict__ k,       // [B, T, n_h, H, d]
    const io_type* __restrict__ v,       // [B, T, n_h, H, d]
    const io_type* __restrict__ beta,    // [B, T, n_h, H]
    io_type* __restrict__ output,        // [B, T, H, d]
    int seq_len,
    int num_householder,
    int num_heads,
    int head_dim
) {
    int b = blockIdx.x;   // batch index
    int h = blockIdx.y;   // head index
    int i = threadIdx.x;  // row index in S (0..head_dim-1)

    if (i >= head_dim) return;

    // Shared memory layout
    extern __shared__ float smem[];
    float* S = smem;                                    // [d][d]
    float* k_shared = smem + head_dim * head_dim;       // [d]
    float* q_shared = k_shared + head_dim;              // [d]
    float* rms_shared = q_shared + head_dim;            // [1] for RMS reduction

    // Initialize S to zero
    for (int j = 0; j < head_dim; j++) {
        S[i * head_dim + j] = 0.0f;
    }
    __syncthreads();

    // Strides for q: [B, T, H, d]
    int q_stride_B = seq_len * num_heads * head_dim;
    int q_stride_T = num_heads * head_dim;
    int q_stride_H = head_dim;

    // Strides for k/v: [B, T, n_h, H, d]
    int kv_stride_B = seq_len * num_householder * num_heads * head_dim;
    int kv_stride_T = num_householder * num_heads * head_dim;
    int kv_stride_J = num_heads * head_dim;
    int kv_stride_H = head_dim;

    // Strides for beta: [B, T, n_h, H]
    int beta_stride_B = seq_len * num_householder * num_heads;
    int beta_stride_T = num_householder * num_heads;
    int beta_stride_J = num_heads;

    int q_base = b * q_stride_B + h * q_stride_H;
    int kv_base = b * kv_stride_B + h * kv_stride_H;
    int beta_base = b * beta_stride_B + h;

    for (int t = 0; t < seq_len; t++) {
        // Apply n_h Householder updates
        for (int j = 0; j < num_householder; j++) {
            int kv_offset = kv_base + t * kv_stride_T + j * kv_stride_J;
            int beta_offset = beta_base + t * beta_stride_T + j * beta_stride_J;

            // Load k_{t,j}[i] into shared + register
            float k_i = IO_LOAD(k, kv_offset + i);
            k_shared[i] = k_i;
            __syncthreads();

            // L2 normalize k: compute ||k||^2 via shared memory reduction
            float k_sq = k_i * k_i;

            if (i == 0) rms_shared[0] = 0.0f;
            __syncthreads();
            atomicAdd(rms_shared, k_sq);
            __syncthreads();

            float k_norm_inv = rsqrtf(rms_shared[0] + NORM_EPS);
            float k_normed_i = k_i * k_norm_inv;
            k_shared[i] = k_normed_i;  // Store normalized k
            __syncthreads();

            float beta_val = IO_LOAD(beta, beta_offset);

            // Phase 1: Compute S^T @ k via shared memory accumulation
            if (i == 0) {
                for (int jj = 0; jj < head_dim; jj++) {
                    q_shared[jj] = 0.0f;
                }
            }
            __syncthreads();

            // Each thread i atomically adds k_normed[i] * S[i,j] for all j
            for (int jj = 0; jj < head_dim; jj++) {
                atomicAdd(&q_shared[jj], k_normed_i * S[i * head_dim + jj]);
            }
            __syncthreads();

            // Now q_shared[j] = (S^T @ k)[j]
            // S_new[i][j] = S[i][j] + beta * k[i] * (v[j] - (S^T @ k)[j])
            float beta_k_i = beta_val * k_normed_i;
            for (int jj = 0; jj < head_dim; jj++) {
                float v_j = IO_LOAD(v, kv_offset + jj);
                S[i * head_dim + jj] += beta_k_i * (v_j - q_shared[jj]);
            }
            __syncthreads();
        }

        // Output: o_t = S @ q_t with RMS normalization
        int q_offset = q_base + t * q_stride_T;

        // Load q_t into shared
        q_shared[i] = IO_LOAD(q, q_offset + i);
        __syncthreads();

        // Compute o_i = sum_j(S[i][j] * q[j])
        float o_i = 0.0f;
        for (int j = 0; j < head_dim; j++) {
            o_i += S[i * head_dim + j] * q_shared[j];
        }

        // RMS normalization: rms = sqrt(mean(o^2) + eps)
        if (i == 0) rms_shared[0] = 0.0f;
        __syncthreads();
        atomicAdd(rms_shared, o_i * o_i);
        __syncthreads();

        float rms_inv = rsqrtf(rms_shared[0] / (float)head_dim + NORM_EPS);
        float o_normed = o_i * rms_inv;

        // Write output
        IO_STORE(output, q_offset + i, o_normed);
        __syncthreads();
    }
}

// ============================================================================
// Standalone launch wrapper
// ============================================================================

#ifndef EXLA_FFI

extern "C" {

int fused_delta_product_scan_launch(
    cudaStream_t stream,
    const io_type* q, const io_type* k, const io_type* v, const io_type* beta,
    io_type* output,
    int batch, int seq_len, int num_householder, int num_heads, int head_dim
) {
    dim3 grid(batch, num_heads);
    dim3 block(head_dim);

    // S[d][d] + k_shared[d] + q_shared[d] + rms_shared[1]
    size_t smem_bytes = (size_t)head_dim * head_dim * sizeof(float)
                      + 2 * head_dim * sizeof(float)
                      + sizeof(float);

    fused_delta_product_scan_kernel<<<grid, block, smem_bytes, stream>>>(
        q, k, v, beta, output,
        seq_len, num_householder, num_heads, head_dim
    );

    return (int)cudaGetLastError();
}

}  // extern "C"

#endif  // !EXLA_FFI

// ============================================================================
// XLA FFI integration
// ============================================================================

#ifdef EXLA_FFI

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

ffi::Error fused_delta_product_scan_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<FFI_IO_TYPE> q,       // [B, T, H, d]
    ffi::Buffer<FFI_IO_TYPE> k,       // [B, T, n_h, H, d]
    ffi::Buffer<FFI_IO_TYPE> v,       // [B, T, n_h, H, d]
    ffi::Buffer<FFI_IO_TYPE> beta,    // [B, T, n_h, H]
    ffi::ResultBuffer<FFI_IO_TYPE> output  // [B, T, H, d]
) {
    // Extract dims from q: [B, T, H, d]
    auto q_dims = q.dimensions();
    int batch     = static_cast<int>(q_dims[0]);
    int seq_len   = static_cast<int>(q_dims[1]);
    int num_heads = static_cast<int>(q_dims[2]);
    int head_dim  = static_cast<int>(q_dims[3]);

    // Extract n_h from k: [B, T, n_h, H, d]
    auto k_dims = k.dimensions();
    int num_householder = static_cast<int>(k_dims[2]);

    dim3 grid(batch, num_heads);
    dim3 block(head_dim);
    size_t smem_bytes = (size_t)head_dim * head_dim * sizeof(float)
                      + 2 * head_dim * sizeof(float)
                      + sizeof(float);

    fused_delta_product_scan_kernel<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const io_type*>(q.untyped_data()),
        reinterpret_cast<const io_type*>(k.untyped_data()),
        reinterpret_cast<const io_type*>(v.untyped_data()),
        reinterpret_cast<const io_type*>(beta.untyped_data()),
        reinterpret_cast<io_type*>(output->untyped_data()),
        seq_len, num_householder, num_heads, head_dim
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    fused_delta_product_scan, fused_delta_product_scan_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // q
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // k
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // v
        .Arg<ffi::Buffer<FFI_IO_TYPE>>()   // beta
        .Ret<ffi::Buffer<FFI_IO_TYPE>>()   // output
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_delta_product_scan_" PRECISION_SUFFIX, "CUDA", fused_delta_product_scan);

#endif  // EXLA_FFI
