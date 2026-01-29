defmodule EXLA.GPUCustomCall do
  @moduledoc """
  Prototype GPU custom call for testing XLA FFI GPU integration.

  This module documents the GPU custom call infrastructure we've added to EXLA.
  It provides a simple vector add via CUDA kernel to validate that GPU custom
  calls work before implementing FlashAttention.

  ## Architecture

  GPU custom calls in EXLA involve three layers:

  1. **CUDA Kernel** (`c_src/exla/custom_calls/gpu_add.cu`):
     - Implements the actual GPU computation
     - Registers handler with `XLA_FFI_REGISTER_HANDLER(..., "CUDA", handler)`
     - Receives CUDA stream from XLA for kernel execution

  2. **MLIR Value Binding** (`lib/exla/mlir/value.ex`):
     - `Value.gpu_add/3` builds the `stablehlo.custom_call` operation
     - Uses `call_target_name: "exla_gpu_add_f32"` to route to CUDA handler
     - Uses `api_version: 4` for typed FFI

  3. **Defn Integration** (`lib/exla/defn.ex`):
     - Pattern matches on Nx operations in `cached_recur_operator`
     - Routes to `Value.gpu_add` when on CUDA platform

  ## Testing

  To test on a machine with CUDA:

  ```bash
  cd exla
  # Ensure CUDA is available
  which nvcc

  # Compile with CUDA support
  mix deps.get
  EXLA_FORCE_REBUILD=true mix compile

  # Run tests
  XLA_TARGET=cuda mix test test/exla/gpu_custom_call_test.exs
  ```

  ## Next Steps for FlashAttention

  1. Create `flash_attention_fwd.cu` with forward kernel
  2. Create `flash_attention_bwd.cu` with backward kernel
  3. Add `Value.flash_attention/5` with forward + backward custom calls
  4. Add pattern match in `cached_recur_operator` for attention operation
  5. Use `Nx.Defn.Kernel.custom_grad` for gradient support

  ## Current Status

  - [x] CUDA kernel prototype (`gpu_add.cu`)
  - [x] Makefile support for `.cu` files in custom_calls
  - [x] Value binding (`Value.gpu_add/3`)
  - [ ] Defn integration (pattern matching in `cached_recur_operator`)
  - [ ] Test on CUDA hardware
  """

  @doc """
  Returns information about the GPU custom call prototype status.
  """
  def status do
    %{
      cuda_kernel: "c_src/exla/custom_calls/gpu_add.cu",
      value_binding: "lib/exla/mlir/value.ex (gpu_add/3)",
      makefile_support: "Makefile modified to compile .cu files",
      defn_integration: :pending,
      testing: :requires_cuda_hardware
    }
  end
end
