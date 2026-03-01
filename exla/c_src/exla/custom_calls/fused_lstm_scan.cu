// Fused LSTM Cell Scan Kernel
//
// Standard LSTM with hidden-to-hidden matmul fused into the scan:
//   gates = W@x + R@h + bias  (W@x pre-computed, R@h in-kernel)
//   i_t = sigmoid(gates_i)
//   f_t = sigmoid(gates_f)
//   g_t = tanh(gates_g)
//   o_t = sigmoid(gates_o)
//   c_t = f_t * c_{t-1} + i_t * g_t
//   h_t = o_t * tanh(c_t)
//
// Thread layout: one thread per (batch, hidden) element.
// Shared memory holds h_prev for the R@h matmul reduction.
//
// Inputs:
//   wx:  [batch, seq_len, 4*hidden]  — pre-computed W@x + bias
//   R:   [hidden, 4*hidden]          — recurrent weight matrix
//   h0:  [batch, hidden]             — initial hidden state
//   c0:  [batch, hidden]             — initial cell state
//
// Output:
//   out: [batch, seq_len, hidden]    — hidden states

#include <cuda_runtime.h>

__global__ void fused_lstm_scan_kernel(
    const float* __restrict__ wx,
    const float* __restrict__ R,
    const float* __restrict__ h0,
    const float* __restrict__ c0,
    float* __restrict__ output,
    int batch, int seq_len, int hidden
) {
    int b = blockIdx.x;
    int i = threadIdx.x + blockIdx.y * blockDim.x;

    if (b >= batch || i >= hidden) return;

    extern __shared__ float h_shared[];

    float h_val = h0[b * hidden + i];
    float c_val = c0[b * hidden + i];
    int hidden4 = 4 * hidden;

    for (int t = 0; t < seq_len; t++) {
        h_shared[i] = h_val;
        __syncthreads();

        // R@h for all 4 gates
        float rh_i = 0.0f, rh_f = 0.0f, rh_g = 0.0f, rh_o = 0.0f;
        for (int j = 0; j < hidden; j++) {
            float h_j = h_shared[j];
            int r_base = j * hidden4;
            rh_i += h_j * R[r_base + i];
            rh_f += h_j * R[r_base + hidden + i];
            rh_g += h_j * R[r_base + 2 * hidden + i];
            rh_o += h_j * R[r_base + 3 * hidden + i];
        }

        int wx_idx = b * seq_len * hidden4 + t * hidden4;
        float i_t = 1.0f / (1.0f + expf(-(wx[wx_idx + i] + rh_i)));
        float f_t = 1.0f / (1.0f + expf(-(wx[wx_idx + hidden + i] + rh_f)));
        float g_t = tanhf(wx[wx_idx + 2 * hidden + i] + rh_g);
        float o_t = 1.0f / (1.0f + expf(-(wx[wx_idx + 3 * hidden + i] + rh_o)));

        c_val = f_t * c_val + i_t * g_t;
        h_val = o_t * tanhf(c_val);

        output[b * seq_len * hidden + t * hidden + i] = h_val;
        __syncthreads();
    }
}

// ============================================================================
// Standalone launch wrapper
// ============================================================================

#ifndef EXLA_FFI

extern "C" {

int fused_lstm_scan_launch(
    cudaStream_t stream,
    const float* wx, const float* R,
    const float* h0, const float* c0,
    float* output,
    int batch, int seq_len, int hidden
) {
    int threads_per_block = (hidden < 256) ? hidden : 256;
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);
    size_t smem_bytes = hidden * sizeof(float);

    fused_lstm_scan_kernel<<<grid, block, smem_bytes, stream>>>(
        wx, R, h0, c0, output, batch, seq_len, hidden
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

ffi::Error fused_lstm_scan_ffi_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> wx,
    ffi::Buffer<ffi::F32> R,
    ffi::Buffer<ffi::F32> h0,
    ffi::Buffer<ffi::F32> c0,
    ffi::ResultBuffer<ffi::F32> output
) {
    auto wx_dims = wx.dimensions();
    int batch   = static_cast<int>(wx_dims[0]);
    int seq_len = static_cast<int>(wx_dims[1]);
    int hidden  = static_cast<int>(wx_dims[2]) / 4;

    int threads_per_block = (hidden < 256) ? hidden : 256;
    int blocks_y = (hidden + threads_per_block - 1) / threads_per_block;
    dim3 grid(batch, blocks_y);
    dim3 block(threads_per_block);
    size_t smem_bytes = hidden * sizeof(float);

    fused_lstm_scan_kernel<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const float*>(wx.untyped_data()),
        reinterpret_cast<const float*>(R.untyped_data()),
        reinterpret_cast<const float*>(h0.untyped_data()),
        reinterpret_cast<const float*>(c0.untyped_data()),
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
    fused_lstm_scan, fused_lstm_scan_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()   // wx
        .Arg<ffi::Buffer<ffi::F32>>()   // R
        .Arg<ffi::Buffer<ffi::F32>>()   // h0
        .Arg<ffi::Buffer<ffi::F32>>()   // c0
        .Ret<ffi::Buffer<ffi::F32>>()   // output
);

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(),
    "exla_fused_lstm_scan_f32", "CUDA", fused_lstm_scan);

#endif  // EXLA_FFI
