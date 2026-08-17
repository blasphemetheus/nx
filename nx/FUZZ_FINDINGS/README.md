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
| **FIXED 2026-08-17** — multi-tensor `impl!` dispatch | PR 1815 | put_slice/clip/gather/reduce/window_reduce now dispatch via impl!/2,3; pins flipped |
| **FIXED 2026-08-17** — batched pinv (n>=2) | PR 1816 | batched dot via batch_axes; n=1 still blocked by svd size-1 bug | 
| **FIXED 2026-08-17** — from_binary sub-byte bitstrings | PR 1817 | guard is is_bitstring; round trip works; pins flipped |
| **FIXED 2026-08-17** — inspect sub-byte crash | PR 1818 | tail::bitstring in :s/:u branches; pins flipped |
| **FIXED 2026-08-17** — reshape multiple :auto | PR 1819 | descriptive ArgumentError; error-contract table updated |
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
| **HIGH** | f64 binary-op overflow | [f64_binary_op_overflow_arithmetic_error.md](f64_binary_op_overflow_arithmetic_error.md) | `Nx.add(max, max)`, `Nx.multiply(max, 2.0)`, tensor-divisor `divide` raise `ArithmeticError` when the f64 result overflows — IEEE requires ±Inf. `pow` silently returns NaN. f32/unary/scalar paths handled, tensor f64 paths not. **Reduction flavor affects ALL float dtypes**: `Nx.product` of ~600 × 1e3 f16 values crashes (accumulator is a BEAM double; found by overnight FUZZ_SCALE=25, 2026-08-16). Pinned in `fuzz_float_edge_test.exs`. |
| **HIGH** | `clip` non-finite inconsistency (cross-backend divergence)  | [clip_nonfinite_inconsistent.md](clip_nonfinite_inconsistent.md) | `clip(NaN, 0, 2)` returns `0.0` — NaN input silently becomes an in-range value; each argument position handles NaN differently, contradicting the min/max composition. Found 2026-08-17, convention-frontier probes. Pinned in `fuzz_nonfinite_convention_test.exs`. |
| **MED** | `count_leading_zeros` sub-byte crash | [clz_sub_byte_crash.md](clz_sub_byte_crash.md) | element_clz/2 has no width-4/2 dispatcher clauses; crashes on any nonzero sub-byte element. The width-2 helper exists but is unreachable (found via coverage-guided targeting of that dead clause). |
| **MED** | argmax/argmin NaN tie divergence | [argmax_nan_tie_divergence.md](argmax_nan_tie_divergence.md) | With multiple NaNs, BinaryBackend argmax returns the LAST NaN index, EXLA the FIRST (documented tie_break: :low); silent cross-backend index divergence. Found 2026-08-17 backend-frontier differential. Pinned in exla differential. |
| **LOW-MED** | `svd` batched size-1 crash | [svd_batched_size1_crash.md](svd_batched_size1_crash.md) | `Nx.LinAlg.svd` crashes on any batched matrix with a size-1 dimension ({2,1,1}/{2,1,2}/{2,2,1}); root cause behind pinv n=1 mode. Found 2026-08-17 building PR B. |

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
