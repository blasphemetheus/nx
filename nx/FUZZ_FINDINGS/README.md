# Fuzz findings — local backlog

Index of bugs surfaced by the fuzz probes in `nx/test/nx/fuzz_*.exs`. Each
entry below is either cross-referenced to an upstream issue (if filed) or
has a standalone md file in this directory (if not yet filed).

New findings go here first; they get their own `<topic>.md` with a minimal
repro, classification, and status. Once a cluster is coherent enough to be
worth the maintainer's attention, we file selectively.

## Filed upstream

| Class | Upstream | Summary |
|---|---|---|
| batched-input grad in `Nx.LinAlg` (multi-op) | [meta #1748](https://github.com/elixir-nx/nx/issues/1748), per-op #1741–#1746 | `custom_grad` formulas assume 2D input; break silently for batched |
| `Nx.LinAlg.eigh` grad | [#1740](https://github.com/elixir-nx/nx/issues/1740) | 2D and f64 inputs crash; only 3D batch-1 f32 works |
| `Nx.Defn.while` reverse-mode AD | [#1747](https://github.com/elixir-nx/nx/issues/1747) | Wrong grad when loop body's Jacobian wrt accumulator depends on differentiated variable |
| `cholesky_grad` gap in PR #1731 | comment on [PR #1731](https://github.com/elixir-nx/nx/pull/1731) | PR's fix is partial — still fails for batched input |
| Blackwell TF32 matmul divergence | Nx #1702, XLA #39250 | f32 GEMMs on sm_120 engage TF32 by default, diverging from BinaryBackend. Regression pin in `exla/test/differential_fuzz_test.exs` — see [blackwell_tf32_regression_pin.md](blackwell_tf32_regression_pin.md) |

## Local-only (not yet filed) — filing queue

Priority key:
- **HIGH** — real correctness bug, broad impact, small fix. File as PR with failing test + minimal fix.
- **MED** — real bug but narrow impact, or fix is nontrivial. File as issue, consider PR after discussion.
- **LOW** — UX / docs / polish. File when convenient; not blocking anyone.

| Priority | Class | Doc | Summary |
|---|---|---|---|
| **HIGH** | unary non-finite handling | [unary_nonfinite_crashes_and_wrong_values.md](unary_nonfinite_crashes_and_wrong_values.md) | `floor`/`ceil`/`round`/`atanh` crash on NaN/±Inf; `tanh(±Inf)` returns NaN (should be ±1), `sign(NaN)` returns 1.0 (should be NaN), `sign(-Inf)` returns **+1.0** (sign error). Found by T1.1 probe (2026-08-14). Pinned in `fuzz_float_edge_test.exs`. |
| **HIGH** | f64 binary-op overflow | [f64_binary_op_overflow_arithmetic_error.md](f64_binary_op_overflow_arithmetic_error.md) | `Nx.add(max, max)`, `Nx.multiply(max, 2.0)`, tensor-divisor `divide` raise `ArithmeticError` when the f64 result overflows — IEEE requires ±Inf. `pow` silently returns NaN. f32/unary/scalar paths handled, tensor f64 paths not. Found by T1.1 bit-pattern generator (2026-08-14). Pinned in `fuzz_float_edge_test.exs`. |
| **HIGH** | multi-tensor op dispatch uses `impl!/1` | [put_slice_grad_mixed_backend_dispatch.md](put_slice_grad_mixed_backend_dispatch.md) | `put_slice` / `gather` / `clip` / `reduce` / `window_reduce` dispatch on the first tensor's backend only; mixed concrete+Expr args crash in `BinaryBackend.to_binary/1`. **Not grad-specific** — also fires in plain `Nx.Defn.jit` closures and defn with `@module_attribute` tensors. One-line fix per op: use `impl!(a, b)`. Pinned in `fuzz_indexed_ops_test.exs`. |
| **HIGH** | `Expr.expr_block` doesn't normalize args | [take_grad_with_captured_indices.md](take_grad_with_captured_indices.md) | `Nx.take` / `Nx.take_along_axis` / `Nx.all_close` route correctly to `Expr.block/4`, but `expr_block` calls `parameter/2` on args without `to_expr/1` first — crashes in `parameter/2` on any concrete tensor. Fix: `Enum.map(args, &to_expr/1)`. Pinned in `fuzz_indexed_ops_test.exs`. |
| **MED** | `Nx.from_binary` rejects bitstrings from `Nx.to_binary` | [from_binary_rejects_sub_byte_bitstrings.md](from_binary_rejects_sub_byte_bitstrings.md) | `to_binary` returns bitstrings for sub-byte types with non-aligned bit counts (docs acknowledge this); `from_binary`'s `is_binary` guard rejects bitstrings. Documented inverse round-trip crashes. One-line fix: guard becomes `is_bitstring`. Pinned in `fuzz_serialization_test.exs`. |
| **MED** | `Nx.Backend.inspect` crashes on sub-byte int tensors | [inspect_crashes_on_sub_byte_int_tensors.md](inspect_crashes_on_sub_byte_int_tensors.md) | `IO.inspect` on any `u2/u4/s2/s4` tensor raises `MatchError` in `chunk/5` because `:s` and `:u` branches use `tail::binary` instead of `tail::bitstring`. Float branch already does it right. Affects BinaryBackend AND Torchx (the inspect code is shared). Pinned in `fuzz_serialization_test.exs`. |
| **LOW** | `Nx.reshape` with multiple `:auto` | [reshape_multiple_auto_error_message.md](reshape_multiple_auto_error_message.md) | Bare `ArithmeticError` instead of a helpful `ArgumentError`. UX-only, not correctness. Not pinned as test. |

### Suggested filing order

Two HIGH-priority findings are natural PR bundles — both "concrete + Expr tensor in the same call" class, one fix pattern per class:

1. **PR A: multi-tensor dispatch fix.** Edit `Nx.put_slice`, `Nx.clip`, `Nx.gather`, `Nx.reduce`, `Nx.window_reduce` to use `impl!(a, b)` or equivalent. Attach `fuzz_indexed_ops_test.exs` dispatch pins (flipped from `assert_raise` to positive). Single file diff, ~5 line changes, ~5 test flips.

2. **PR B: `Expr.expr_block` normalization.** One-line fix in `nx/defn/expr.ex` (`Enum.map(args, &to_expr/1)`). Attach `fuzz_indexed_ops_test.exs` expr_block pins (flipped).

Both could also be a single combined PR titled "Fix mixed-backend dispatch for multi-tensor ops" if a maintainer prefers.

The two MED findings (from_binary / inspect) are both one-liners in `Nx.Backend` and share the `tail::binary` → `tail::bitstring` pattern. They could be bundled into **PR C: sub-byte bitstring handling** with both fixes + their pins.

LOW (reshape) can wait or be bundled with anything.

## Adding a new finding

1. Write a failing test in the appropriate `fuzz_*.exs` file (or a new one).
2. Run it and capture the minimal reproducer.
3. Create `FUZZ_FINDINGS/<topic>.md` with:
   - One-sentence summary
   - Minimal repro (≤15 lines)
   - Expected vs observed
   - Root cause hypothesis (if known)
   - Classification: same as existing class? Or new?
   - Which test file pins the failure
4. Add the entry to the "Local-only" table above.

Batch-file only when a set of findings is coherent enough that a maintainer
can act on them as one unit.
