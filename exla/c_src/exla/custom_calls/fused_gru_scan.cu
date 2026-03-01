// Fused GRU Cell Scan Kernel
//
// Standard GRU with hidden-to-hidden matmul fused into the scan:
//   z_t = sigmoid(W_z@x + R_z@h)     — update gate
//   r_t = sigmoid(W_r@x + R_r@h)     — reset gate
//   h_tilde = tanh(W_h@x + R_h@(r_t * h))  — candidate
//   h_t = (1 - z_t) * h_tilde + z_t * h_{t-1}
//
// Note: GRU applies the reset gate BEFORE the candidate's recurrent matmul.
// We handle this by computing R_h@h first, then multiplying r_t element-wise.
// This is a common approximation used in cuDNN and most efficient GRU kernels.
//
// Inputs:
//   wx:  [batch, seq_len, 3*hidden]  — pre-computed W@x + bias (z, r, h gates)
//   R:   [hidden, 3*hidden]          — recurrent weights
//   h0:  [batch, hidden]             — initial hidden state
//
// Output:
//   out: [batch, seq_len, hidden]    — hidden states

#include <cuda_runtime.h>

__global__ void fused_gru_scan_kernel(
    const float* __restrict__ wx,
    const float* __restrict__ R,
    const float* __restrict__ h0,
    float* __restrict__ output,
    int batch, int seq_len, int hidden
) {
    int b = blockIdx.x;
    int i = threadIdx.x + blockIdx.y * blockDim.x;

    if (b >= batch || i >= hidden) return;

    extern __shared__ float h_shared[];

    float h_val = h0[b * hidden + i];
    int hidden3 = 3 * hidden;

    for (int t = 0; t < seq_len; t++) {
        h_shared[i] = h_val;
        __syncthreads();

        // R@h for all 3 gates
        float rh_z = 0.0f, rh_r = 0.0f, rh_h = 0.0f;
        for (int j = 0; j < hidden; j++) {
            float h_j = h_shared[j];
            int r_base = j * hidden3;
            rh_z += h_j * R[r_base + i];
            rh_r += h_j * R[r_base + hidden + i];
            rh_h += h_j * R[r_base + 2 * hidden + i];
        }

        int wx_idx = b * seq_len * hidden3 + t * hidden3;
        float z_t = 1.0f / (1.0f + expf(-(wx[wx_idx + i] + rh_z)));
        float r_t = 1.0f / (1.0f + expf(-(wx[wx_idx + hidden + i] + rh_r)));

        // Candidate: tanh(W_h@x + r_t * R_h@h)
        float h_tilde = tanhf(wx[wx_idx + 2 * hidden + i] + r_t * rh_h);

        // Update: h = (1-z)*h_tilde + z*h_prev
        h_val = (1.0f - z_t) * h_tilde + z_t * h_val;

        output[b * seq_len * hidden + t * hidden + i] = h_val;
        __syncthreads();
    }
}

// ============================================================================
// Standalone launch wrapper
// ============================================================================

#ifndef EXLA_FFI

extern "C" {

int fused_gru_scan_launch(
    cudaStream_t stream,
    const float* wx, const float* R, const float* h0,
    float* output,
    int batch, int seq_len, int hidden
) {
    int threads_per_block = (hidden < 256) ? hidden : 256;
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);
    size_t smem_bytes = hidden * sizeof(float);

    fused_gru_scan_kernel<<<grid, block, smem_bytes, stream>>>(
        wx, R, h0, output, batch, seq_len, hidden
    );

    return (int)cudaGetLastError();
}

}  // extern "C"

#endif

// ============================================================================
// XLA FFI integration
// ============================================================================

#ifdef EXLA_FFI

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

ffi::Error fused_gru_scan_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> wx,
    ffi::Buffer<ffi::F32> R,
    ffi::Buffer<ffi::F32> h0,
    ffi::ResultBuffer<ffi::F32> output
) {
    auto wx_dims = wx.dimensions();
    int batch   = static_cast<int>(wx_dims[0]);
    int seq_len = static_cast<int>(wx_dims[1]);
    int hidden  = static_cast<int>(wx_dims[2]) / 3;

    int threads_per_block = (hidden < 256) ? hidden : 256;
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);
    size_t smem_bytes = hidden * sizeof(float);

    fused_gru_scan_kernel<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const float*>(wx.untyped_data()),
        reinterpret_cast<const float*>(R.untyped_data()),
        reinterpret_cast<const float*>(h0.untyped_data()),
        reinterpret_cast<float*>(output->untyped_data()),
        batch, seq_len, hidden
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    fused_gru_scan, fused_gru_scan_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()   // wx
        .Arg<ffi::Buffer<ffi::F32>>()   // R
        .Arg<ffi::Buffer<ffi::F32>>()   // h0
        .Ret<ffi::Buffer<ffi::F32>>()   // output
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_gru_scan_f32", "CUDA", fused_gru_scan);

#endif  // EXLA_FFI
