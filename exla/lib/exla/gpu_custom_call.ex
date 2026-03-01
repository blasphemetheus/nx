defmodule EXLA.GPUCustomCall do
  @moduledoc """
  GPU custom call infrastructure for CUDA kernels in EXLA.

  This module documents how GPU custom calls work and serves as a reference
  for adding new CUDA kernels (e.g., fused scans, FlashAttention).

  ## Architecture

  GPU custom calls in EXLA involve two layers:

  1. **CUDA Kernel** (`c_src/exla/custom_calls/*.cu`):
     - Implements GPU computation, launched on XLA's CUDA stream
     - Registers via `XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), name, "CUDA", handler)`
     - Receives stream + device buffers from XLA (zero-copy)
     - Compiled by nvcc when detected (guarded by `#ifdef CUDA_ENABLED`)

  2. **MLIR Value Binding** (`lib/exla/mlir/value.ex`):
     - Builds `stablehlo.custom_call` ops with `api_version: 4` (typed FFI)
     - Routes to CUDA handler via `call_target_name`

  ## Adding a New CUDA Kernel

  1. Create `c_src/exla/custom_calls/my_kernel.cu` (see `gpu_add.cu` as template)
  2. Add `Value.my_kernel/N` in `lib/exla/mlir/value.ex`
  3. Rebuild: `EXLA_TARGET=cuda EXLA_FORCE_REBUILD=true mix compile`
  4. Add test in `test/exla/gpu_custom_call_test.exs`

  ## Important: XLA FFI API Name

  Use `XLA_FFI_GetApi()` (C function from `c_api.h`), NOT `ffi::GetXlaFfiApi()`
  which doesn't exist in XLA 0.10+ prebuilt headers.

  ## Status

  - [x] Makefile: `.cu` compilation via nvcc in `custom_calls/`
  - [x] Prototype kernel: `gpu_add.cu` (element-wise add)
  - [x] Value binding: `Value.gpu_add/3`
  - [x] Tested on CUDA hardware (NVIDIA T400, Compute 7.5)
  """

  @doc """
  Returns information about the GPU custom call infrastructure.
  """
  def status do
    %{
      cuda_kernels: ["gpu_add.cu"],
      value_bindings: ["gpu_add/3"],
      makefile_support: :enabled,
      tested: true,
      api_function: "XLA_FFI_GetApi()"
    }
  end
end
