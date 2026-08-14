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

- [~] **T1.1 Bit-pattern float generator.** Random bit patterns reinterpreted as
  floats (hits NaN payloads, ±Inf, denormals, −0.0, full exponent range) +
  cancellation-prone constructions. New shared generator in `test/support/`,
  new `fuzz_float_edge_test.exs` suite with NaN/Inf-aware oracles, then
  retrofit into existing smoke/invariant suites where domains allow.
- [ ] **T1.2 `Nx.block` differential fuzz.** Each of the 21 block structs
  carries a traced default implementation — compare backend lowering vs default
  callback, and sweep each struct's option matrix (`mode`, `full_matrices?`,
  `eps`, `k`, `axis`, …). Structural invariants where analytic (Q·R == A,
  top_k sorted, cumsum vs sum).
- [ ] **T1.3 `Nx.Random` properties.** split/fold_in determinism and
  independence, moment checks per distribution, choice/shuffle permutation
  invariants, vectorized keys.
- [ ] **T1.4 Conv value oracle.** Reference im2col+dot implementation (or
  cross-backend comparison) behind the existing parameter sweep — conv is
  currently shape-checked only.

### Tier 2 — v1.0-specific

- [ ] **T2.1 Error-contract fuzz.** Property: invalid inputs raise Nx-owned
  `ArgumentError` with a useful message — never bare
  `ArithmeticError`/`MatchError`/`FunctionClauseError` (the reshape-`:auto` /
  sub-byte-inspect class). API freeze makes error behavior contract.
- [ ] **T2.2 New-API metamorphic sweep.** `pad_outer` 4 modes (involution/
  periodicity identities), `rfft`/`irfft` round-trip + Parseval, `fft2` vs
  composed 1-D FFTs, f8/e4m3fn saturation (no-Inf type) + `Nx.Floating`
  round-trips, sub-byte arithmetic promotion.
- [ ] **T2.3 `cond`-under-grad + multi-axis vectorization.** Historical bug
  factory (#1729/#1730 were vectorized+cond); both configurations unfuzzed.

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
