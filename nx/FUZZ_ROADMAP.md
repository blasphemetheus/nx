# Fuzz Roadmap — state of the campaign and what's next

Companion to `FUZZ_FINDINGS/` (bug backlog). This file tracks *coverage*: what the
fuzz suites test today, where they're blind, and the planned test types for the
Nx v1.0 push. Updated 2026-08-14 after a full audit of the suites vs the v0.13.1
API surface.

## Where we are

Campaign to date: ~28 issues filed (27 productively resolved), 14+ PRs merged.
Suites: 13 `fuzz_*.exs` files in `nx/test/nx/` (~5,300 lines, ~300 properties/tests)
plus cross-backend differentials in `torchx/test/` and `exla/test/`.

### What works (keep doing)

| Strategy | Where | Yield |
|---|---|---|
| Finite-difference grad differential | `fuzz_grad_test.exs` | Entire batched-LinAlg-grad cluster (#1740–#1748) |
| Closed-form grad oracles | `fuzz_second_order_grad_test.exs`, `fuzz_while_grad_test.exs` | BUG-1747 (while grad) |
| Cross-backend differential | `torchx/`, `exla/` differential suites | TF32/Blackwell divergence (#1702, XLA #39250) |
| Metamorphic identities | `fuzz_invariants_test.exs` | Numerically strongest oracle in tree |
| Round-trip properties | `fuzz_serialization_test.exs` | Sub-byte bitstring bugs |
| Bug-pin discipline | `assert_raise` pins + `FUZZ_FINDINGS/*.md` | Every finding stays reproducible |

### Known blind spots (audit 2026-08-14)

1. **Value blindness**: `Nx.iota` dominates as the value source (~160 sites) —
   monotone, non-negative, integral. The `edge_tensor` generator in
   `fuzz_random_values_test.exs` *sanitizes* NaN/Inf out before tensors are
   built, so property paths never feed special values into ops.
2. **Weak oracles where coverage is broadest**: `fuzz_test.exs` (77 properties)
   is crash+shape only; `fuzz_conv_test.exs` sweeps the full parameter space but
   never checks values; `fuzz_grad_test.exs` tolerance is disjunctive
   (atol OR rtol); some exla f16/bf16 differentials are informational-only.
3. **Surface gaps** (zero fuzz coverage): `Nx.Random` (all 14 fns), `Nx.block` +
   all 21 `Nx.Block` structs, `pad_outer`, `rfft`/`irfft`/`fft2`/`ifft2`,
   `shard_jit`/`Nx.Mesh`, `Nx.Serving`/`Nx.Batch`, `io_call` post-rename,
   f8/f8_e4m3fn (nowhere), s4 (nowhere), `median`/`mode`/`logsumexp`,
   `cond` under grad, multi-axis vectorization — 48 public functions total.
   Most shipped in v0.11–v0.13, i.e. after the original campaign.

## Planned test types

Status key: [ ] planned · [~] in progress · [x] landed

### Tier 1 — highest expected bug yield

- [x] **T1.1 Bit-pattern float generator.** Landed 2026-08-14: `FuzzGen`
  (`test/support/fuzz_gen.ex`) + `fuzz_float_edge_test.exs` (20 properties,
  NaN/Inf-aware oracles) + hostile-finite retrofit into `fuzz_test.exs` and
  `fuzz_sequence_test.exs`. **Found 2 HIGH bugs on first contact**:
  [f64 binary-op overflow](FUZZ_FINDINGS/f64_binary_op_overflow_arithmetic_error.md)
  and [unary non-finite crashes/wrong values](FUZZ_FINDINGS/unary_nonfinite_crashes_and_wrong_values.md).
- [x] **T1.2 `Nx.block` differential fuzz.** Landed 2026-08-14:
  `fuzz_block_test.exs` (15 properties) — jit-vs-eager route differential per
  block API + structural invariants (Q·R==A, L·Lᵀ==A, P·L·U==A, SVD/Eigh
  reconstruction, top_k traceability, cumulative vs Enum.scan, all_close vs
  Elixir reference, FFT2/RFFT round trips) with option sweeps (qr mode,
  full_matrices?, reverse, rtol/atol). No live bugs found — the historical
  expr_block dispatch bug was fixed upstream by the block rework. The
  *native-lowering* differential (EXLA custom calls vs default callback)
  needs a backend and belongs to T3.1's differential completion.
- [x] **T1.3 `Nx.Random` properties.** Landed 2026-08-14:
  `fuzz_random_props_test.exs` (12 properties) — bitwise determinism per
  seed, stream distinctness across seeds/new_key, split subkey stream
  independence, fold_in determinism+distinctness, uniform/randint
  range+shape+type contracts, moment checks at ~7 standard errors
  (deterministic per generated seed, so not flaky), shuffle permutation +
  determinism + non-identity, choice population membership + sample count,
  vectorized-key batch stream distinctness. No live bugs found.
- [x] **T1.4 Conv value oracle.** Landed 2026-08-14: `reference_conv` in
  `fuzz_conv_test.exs` — position-by-position reference built from trusted
  primitives (interior-padded `Nx.pad` for input dilation, strided `Nx.slice`
  receptive fields, `Nx.dot` contraction) — 7 value properties at f64/1e-9
  covering baseline, strides, explicit padding, input/kernel dilation,
  feature groups, and combined configs. No live bugs found. TODO: `:same`
  padding and `batch_group_size` in the reference; cross-backend conv values
  belong to T3.1.

### Tier 2 — v1.0-specific

- [x] **T2.1 Error-contract fuzz.** Landed 2026-08-14:
  `fuzz_error_contract_test.exs` — 24-case invalid-input table + randomized
  dimension-mismatch properties. Contract is nearly clean: only violation is
  the known reshape-two-`:auto` ArithmeticError leak (pinned). Documented
  non-raising semantics excluded: slice start-clamping, uneven split.
- [x] **T2.2 New-API metamorphic sweep.** Landed 2026-08-14:
  `fuzz_newapi_test.exs` — pad_outer 4 modes vs index-mapping reference (1-D
  and 2-D), rfft==fft-prefix, Parseval, fft2 vs composed 1-D, e4m3fn
  saturation to ±448 (never Inf) + NaN preservation, e5m2 overflow to ±Inf,
  f8 exact round trips, sub-byte modular arithmetic vs exact reference.
  No live bugs found.
- [x] **T2.3 `cond`-under-grad + multi-axis vectorization.** Landed 2026-08-14:
  `fuzz_cond_vectorized_grad_test.exs` — 7 properties with closed-form
  derivative oracles: grad through cond/nested-cond/data-dependent-pred,
  doubly-vectorized reductions and binary ops vs plain-axis reference,
  doubly-vectorized grad, and grad-through-cond-with-vectorized-input (the
  exact #1729/#1730 configuration — passes on current main; that bug class
  is confirmed dead).

### Tier 3 — infrastructure hardening

- [x] **T3.1 Differential completion.** Landed 2026-08-14:
  - EXLA differential: 14 informational (try/rescue + IO.puts) tests promoted
    to real assertions — all hold on Blackwell/cuda; new f64 describes
    (element-wise/reductions/linalg/grad at 1e-9..1e-13) and complex
    describes (c64 arithmetic/abs/phase/conjugate, fft/ifft, c128).
  - Torchx differential: mirrored f64 describes (complex already covered).
  - `fuzz_grad_test.exs`: conjunctive tolerance (|diff| <= atol + rtol*scale
    elementwise) replacing the disjunctive either/or check — passes, so the
    analytic grads were already accurate.
  - True three-way EXLA↔Torchx↔Binary in one test env is blocked by the
    umbrella layout (separate apps); both differential files share
    BinaryBackend as the hub, giving transitive coverage.
- [x] **T3.2 `Nx.Serving`/`Nx.Batch` metamorphic.** Landed 2026-08-15:
  `fuzz_serving_batch_test.exs` (6 properties + 3 tests) — Batch
  stack/split/pad structure, inline-serving topology invariance (run ==
  direct, batch_size splits, split-halves == whole, padded == plain),
  supervised-process serving with 16 concurrent batched_run requests
  (isolation across merge/split boundaries) and mixed request sizes,
  input-stream equality. No live bugs found. Partition concurrency and
  batch_keys left as follow-ups.
- [x] **T3.3 `shard_jit`/`Nx.Mesh` equivalence.** Landed 2026-08-16:
  `exla/test/exla/defn/sharding_fuzz_test.exs` (3 properties) — shard inputs
  by hand, run `EXLA.shard_jit` across the mesh, reassemble per-device
  outputs, demand exact equality with the unsharded computation. Covers a
  1-D mesh (axis-0 sharding, 4-function elementwise vocabulary), tuple
  outputs, and a 2×2 mesh with block sharding. No live bugs found.
  Premise correction vs the original plan: sharding is EXLA-only — the
  Evaluator's `__shard_jit__` raises by design — so the suite lives in
  exla/ and runs under `:multi_device`
  (`EXLA_TARGET=host XLA_FLAGS=--xla_force_host_platform_device_count=4`).

### Frontier suites (post-roadmap, 2026-08-17)

- [x] **Non-finite convention suite** (`fuzz_nonfinite_convention_test.exs`,
  14 properties): consistency oracles over the empirically-established NaN
  map (min/max family propagates; arg* points at reduce_*; sort orders
  -Inf < finite < +Inf < NaN; median follows sort). **Found
  [BUG-CLIP-NONFINITE]** — clip handles NaN differently per argument
  position; clip(NaN, 0, 2) returns 0.0.
- [x] **Integer semantics suite** (`fuzz_int_semantics_test.exs`, 40
  properties): exact unbounded-integer references wrapped to type for
  quotient/remainder/shifts/bitwise/popcount/clz; INT_MIN / -1 wrap and
  div-by-zero conventions pinned. Clean.
- [x] **Long-tail suite** (`fuzz_longtail_test.exs`, 16 properties): mode,
  weighted_mean, covariance, logsumexp (incl. overflow-regime stability),
  diagonal family, tri/tril/triu, to_batched/split, bitcast,
  cumulative_product, window_product, window_scatter_min metamorphic,
  logical ops. Clean.
- [x] **Backend/scale frontier** (2026-08-17): non-finite differential on
  cuda (conventions agree except two pinned divergences — clip NaN
  confirmed cross-backend with BinaryBackend the outlier, and NEW
  [argmax NaN-tie divergence](FUZZ_FINDINGS/argmax_nan_tie_divergence.md));
  large-shape GPU sweep (256x256 matmul @ precision :highest, softmax,
  1e6 f64 reductions, 100k transcendental chain); sharding collectives
  (sum over sharded axis all-reduces to replicated result, unsharded-axis
  sums reassemble, dot contracting the sharded axis). Donation semantics
  deferred.
- [x] **Coverage-guided gap finding** (2026-08-17): intersected
  never-executed lines across corpus + full-suite .coverdata; only ~130
  dark lines existed in the load-bearing modules. Targeted them with a
  comparison truth-table property, 12 dark-raise error-contract entries,
  complex as_boolean coverage, and sub-byte bit counting — the last found
  TWO bugs on first execution:
  [clz sub-byte crash + s2 count wrap](FUZZ_FINDINGS/clz_sub_byte_crash.md).
  Round 2 (same day): error-contract table to 46 entries,
  fuzz_darklines_test.exs for semantic arms (argmax tie_break witnesses,
  complex reduce acc, multi-axis aggregation, vectorized qr/lu, Expr/
  Evaluator/Grad trace contracts, Matrix complex conjugate). Dark lines
  ~130 -> 80; remainder is defensive/unreachable code and deep
  vectorized-grad reconciliation arms.

### Cheap wins

- [x] **Deep-run scaling.** Landed 2026-08-16: every property's budget is now
  `N * @fuzz_scale` with `@fuzz_scale` read from the `FUZZ_SCALE` env var
  (default 1 — CI unchanged; inner `max_runs: 1` value-generation clauses
  stay fixed to avoid quadratic blowup). Overnight harness: repeated
  corpus runs at `FUZZ_SCALE=25` with a fresh `--seed` per iteration
  (seed diversity finds more than single-seed depth), plus torchx/exla
  differential and sharding legs. Needs `--timeout 600000` — deep budgets
  exceed ExUnit's 60s default on the reduction properties.
- [x] **First overnight run** (2026-08-16, 9h, 150 iterations, fresh seed per
  iteration, 4 legs: nx corpus @ scale 25 / torchx diff / exla-cuda diff /
  sharding). 6 failing iterations, all in the nx leg, all triaged live:
  1 real bug (product-accumulator overflow — pinned, extends the
  [f64 overflow finding](FUZZ_FINDINGS/f64_binary_op_overflow_arithmetic_error.md)
  to every float dtype), 2 cost walls (shape generator now capped at 2048
  elements), 2 oracle-validity fixes (SVD reconstruction tolerance tail;
  FD-vs-tie separation for window min/max grads), 1 incomplete-fix catch.
  Final 53 iterations fully clean; torchx/exla/sharding legs never failed.
- [x] **Seed reproducibility** — already satisfied, no change needed:
  StreamData derives its value stream from ExUnit's seed, which every run
  prints (`Running ExUnit with seed: N`); rerun with `--seed N` to
  reproduce any property failure exactly.
