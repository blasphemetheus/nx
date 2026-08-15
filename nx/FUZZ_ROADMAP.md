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

- [ ] **T3.1 Differential completion.** f64 + complex on the EXLA differential,
  three-way EXLA↔Torchx↔Binary, grad differentials across backends; promote
  informational f16/bf16 tests to calibrated assertions; conjunctive grad
  tolerance.
- [ ] **T3.2 `Nx.Serving`/`Nx.Batch` metamorphic.** Results invariant to batch
  split/merge boundaries; streaming; partition concurrency.
- [ ] **T3.3 `shard_jit`/`Nx.Mesh` equivalence.** sharded == unsharded on the
  Evaluator; smoke coverage for the newest subsystem in the tree.

### Cheap wins (no new code)

- [ ] Bump `max_runs` on strong-oracle suites (invariants, grad, second-order)
  for overnight runs — current budgets are 8–40.
- [ ] Pin StreamData seeds in CI output for reproducibility of failures.
