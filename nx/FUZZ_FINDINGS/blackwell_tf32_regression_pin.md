---
name: EXLA-on-Blackwell TF32 matmul divergence — regression pin
description: Blackwell RTX 5090 (sm_120) f32 matmul diverges from BinaryBackend at rtol≤1e-5 because XLA engages TF32 tensor cores by default. Confirmed by precision::highest control (re-passes). Already filed upstream (Nx #1702, XLA #39250). This finding is a local regression pin.
type: project
---

# Finding: Blackwell TF32 matmul divergence (regression pin)

## One-sentence summary

On Blackwell RTX 5090 (sm_120), EXLA's default `Nx.dot` for f32 diverges
from `Nx.BinaryBackend` at `rtol ≤ 1e-5` because XLA engages TF32 tensor
cores, truncating the mantissa from 23 to 10 bits. Setting
`precision: :highest` on the jit restores agreement.

This is a **regression pin**, not a new finding — the issue is already
filed upstream. It exists here so any future cross-backend differential
run on Blackwell lands in a known-explained state.

## Minimal repro

```elixir
# Run under devenv (so XLA_TARGET=cuda12, CUDA driver visible)
defmodule Probe do
  def run do
    :rand.seed(:exsss, {1, 2, 3})
    va = for _ <- 1..(32*32), do: (:rand.uniform() - 0.5) * 2.0
    vb = for _ <- 1..(32*32), do: (:rand.uniform() - 0.5) * 2.0

    Nx.default_backend(Nx.BinaryBackend)
    a_bin = Nx.tensor(va, type: :f32) |> Nx.reshape({32, 32})
    b_bin = Nx.tensor(vb, type: :f32) |> Nx.reshape({32, 32})
    r_bin = Nx.dot(a_bin, b_bin)

    Nx.default_backend({EXLA.Backend, client: :cuda})
    a_exla = Nx.tensor(va, type: :f32) |> Nx.reshape({32, 32})
    b_exla = Nx.tensor(vb, type: :f32) |> Nx.reshape({32, 32})
    r_exla = Nx.dot(a_exla, b_exla)

    diff = Nx.subtract(r_bin, Nx.backend_copy(r_exla)) |> Nx.abs() |> Nx.reduce_max()
    IO.inspect(diff, label: "max abs diff")
    # Typical: ~1e-3 absolute, ~1e-4 relative — classic TF32 signature.
  end
end
```

## Characterization

### Pure matmul diverges

| Shape | rtol threshold for agreement |
|---|---|
| 32×32 | fails at rtol=1e-5, passes at rtol=1e-3 |
| 128×128 | fails at rtol=1e-5 |

With `jit(fun, precision: :highest)` applied to the dot, both cases pass
at rtol=1e-5 — this is the clean experiment confirming TF32 as the cause.

### Nx.LinAlg ops DON'T diverge at rtol=1e-5 (surprising negative)

Tested at 32×32 and 128×128 — all agreed at tight tolerance:

| Op | 32×32 | 128×128 |
|---|---|---|
| `Nx.LinAlg.determinant` | agreed | agreed |
| `Nx.LinAlg.invert` | agreed | agreed |
| `Nx.LinAlg.qr` (Q matrix) | agreed | agreed |
| `Nx.LinAlg.svd` (s vector) | agreed | agreed |
| `Nx.LinAlg.cholesky` | agreed | (not tested; see `exla/test/differential_fuzz_test.exs`) |

**Interpretation:** EXLA's lowering of `Nx.LinAlg` routines evidently
does not route through TF32-engaged tensor-core GEMMs on Blackwell —
either because the lowering uses block/householder algorithms that
XLA doesn't autotune the same way, or because the LinAlg code paths
hit a different precision default. This is a **strong positive** for
the LinAlg module: common LinAlg usage on Blackwell is already
TF32-safe at these sizes, even without setting `precision: :highest`.

Open question: does this hold at larger sizes (≥256×256)? Not tested
because BinaryBackend's iterative QR/SVD is prohibitively slow at
that scale. Worth retrying when a faster reference becomes available
(Torchx CPU would be much faster than BinaryBackend).

## Related upstream

- Nx #1702 — adds `precision: :highest` to LinAlg ops to avoid TF32.
- XLA #39250 — GEMM / autotuner issue (TF32 default on Blackwell).
- Nx #1703, Triton #39253 — related Blackwell tracks.

## Classification

**Not a new bug.** This is a regression pin against a known upstream
issue. The value here is:

- `exla/test/differential_fuzz_test.exs` now exercises the default
  `Nx.dot` path on Blackwell every run — if future XLA versions fix
  the autotuner or Nx flips LinAlg defaults to `:highest`, the pin
  tests will break loudly and we flip them to positive assertions.
- The `precision: :highest` control test confirms the classification
  every run — if both default AND :highest start diverging, the cause
  is something other than TF32.

## Test pinning the behavior

`exla/test/differential_fuzz_test.exs`:

- `"32×32 f32 matmul diverges from BinaryBackend (pins TF32 bug)"`
- `"128×128 f32 matmul diverges (pins TF32 bug)"`
- `"matmul jit'd with precision: :highest agrees tightly (control)"`

Both "diverges" tests use `assert_raise ExUnit.AssertionError` on a
tight-tolerance diff — green today, breaks when the default agrees.
