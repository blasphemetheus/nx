# Nx fork — handoff (2026-10-06)

State of this fork's contribution work to elixir-nx/nx. Companion docs:
`WORKPLAN.md` (PR sequencing), `nx/FUZZ_ROADMAP.md` (test-coverage map),
`nx/FUZZ_FINDINGS/README.md` (bug queue).

## Where we are in one paragraph

Eight PRs merged upstream this cycle, all originating from a property-fuzz
campaign run on this fork. Nothing is currently in flight upstream. The
`integration` branch (synced with `origin/main` at v1.0, 2026-10-06) holds the fuzz
corpus — 36 files, ~16k lines, 698 properties + 668 tests, all green
against current main. Seven documented bugs remain unfiled, one of which
already has a tested fix branch waiting for a go-ahead.

## Merged upstream (this cycle)

| PR | What |
|---|---|
| 1813 | EXLA CUDA `OutputBuffer` default ctor — `XLA_TARGET=cuda12` builds compile again |
| 1815 | Multi-tensor `impl!` dispatch (put_slice/clip/gather/reduce/window_reduce) |
| 1816 | Batched `pinv` (n≥2) via `batch_axes` |
| 1817 | `from_binary` accepts sub-byte bitstrings |
| 1818 | `inspect` crash on sub-byte integer tensors |
| 1819 | `reshape` multiple `:auto` → descriptive ArgumentError |
| 1820 | Test coverage for `revectorize`'s multiple-`:auto` guard |
| 1821 | `eigh` dropping the batch for 1×1 matrices (unblocked svd/pinv n=1) |

Earlier cycles (already merged before this arc): 14 PRs including the
vectorized-grad boundary fix #1731, Blackwell tolerances, clip g², IPC shm
perms, CallbackServer.

## Open work, by readiness

### Fix built, tested, pushed — needs only a PR go-ahead

**`fix/pinv-zero-shape-batch`** (on fork, off current main). `pinv_zero_shape`
reverses the dim list to transpose the trailing two dims and never
re-reverses the batch prefix — unequal double-batch dims crash (equal ones
are a no-op and come out right). Two-line `put_elem` fix; tests cover `{3,2,2,2}`,
non-square `{5,4,2,3}` with Moore-Penrose identity, and the all-zeros
branch. Full nx suite green. Opened as draft PR
https://github.com/elixir-nx/nx/pull/1854 on 2026-10-07 for review.

**`fix/qr-f16-eps`** (on fork, off `origin/main` 9333caed, 1 commit). Found
2026-10-06 by the EXLA CUDA suite after the v1.0 merge: `Nx.LinAlg.qr` on
f16 returns all-NaN for wide matrices and for any rank-deficient column.
Root cause is upstream's "read float literals at the precision of the
surrounding expression" change (merged 2026-09-10): the default
`eps: 1.0e-10` is zero in f16, so the zero-norm guard in
`householder_reflector` never fires and it divides by zero. Reproduces on
the Evaluator too; EXLA host hides it via the CPU QR custom call. Only QR
is affected (cholesky/determinant/eigh/svd/pinv/lu probed clean in f16).
Fix clamps eps to `Nx.Constants.smallest_positive_normal(type)`, the idiom
`invert` already uses. Regression test in the existing `qr` describe,
verified failing-before/passing-after; full nx suite and the CUDA
`EXLA.MLIR.CustomCallTest` f16 case green. Opened as draft PR
https://github.com/elixir-nx/nx/pull/1853 on 2026-10-07 for the user to review
and edit before marking ready.

### Small focused PRs — fix direction is unambiguous

1. **clip non-finite** — **draft PR open**: https://github.com/elixir-nx/nx/pull/1855
   (2026-10-07, fork `fix/clip-nan`). BinaryBackend clip now composes
   `element_min(element_max(x, lo), hi)`; verified identical to EXLA/cuda on
   all three NaN positions plus float/int controls. Once merged, flip the
   `[BUG-CLIP-NONFINITE]` pins in `fuzz_nonfinite_convention_test.exs` and
   the EXLA `[DIVERGENCE-CLIP-NONFINITE]` pin to exact agreement.
2. **argmax/argmin NaN tie-break** (MED). Multiple NaNs: BinaryBackend
   returns the LAST NaN index, EXLA the FIRST, and the documented contract
   (`tie_break: :low`) says first. Fix the fold direction; flip the
   divergence pin.
3. **clz sub-byte dispatcher** (MED). `element_clz/2` has no width-4/2
   clauses so `count_leading_zeros` crashes on any nonzero sub-byte element
   — while the width-2 and width-4 helpers already exist in the file,
   unreachable. Scope the PR to the crash (u2/u4/s4) and surface the s2
   representability question separately (see below).
4. **custom_grad count validation** (MED). Extra returned gradients are
   **silently dropped** (plausible wrong gradient, no diagnostic); missing
   ones leak the internal `ERROR! grad for metadata returned N entries`.
   Validate length symmetrically and raise the existing friendly message.
5. **unary non-finite** (HIGH) — **Nx half open as draft PR**
   https://github.com/elixir-nx/nx/pull/1856 (2026-10-07, fork
   `fix/unary-nonfinite`): floor/ceil/round pass atoms through, sign has
   atom clauses; doctests run on all three backends (sign doctest uses only
   ±Inf since libtorch defines sign(NaN)=0). **Complex half NOT started**:
   tanh(±Inf)→NaN (want ±1), atanh(any non-finite) crashes, log/log1p/acosh
   on -Inf crash (Complex returns a complex value / raises for a real -Inf).
   Complex half built on fork `fix/nonfinite-tanh-atanh-log-acosh` (commit
   f0975fc, off complex main v1.0): tanh(±Inf)→±1.0, atanh(non-finite)→:nan,
   acosh(-Inf)→:nan (replaces a tested raise). Suite green; verified via dep
   swap that Nx tanh/atanh/acosh now match EXLA. **No PR yet — user said
   wait.** log/log1p(-Inf) deliberately left out: Complex has a test pinning
   log(:neg_infinity) == Inf+πi (complex-domain choice), so the real-tensor
   CaseClauseError is an Nx-layer question (BinaryBackend receiving a
   %Complex{} for a real dtype). Original note: `floor`/`ceil`/`round`/`atanh` crash on
   NaN/±Inf; `tanh(±Inf)` returns NaN (want ±1); `sign(NaN)` returns 1.0;
   **`sign(-Inf)` returns +1.0**. ~10 missing clauses, patterned on the
   `ieee754_fallback` table in `fork/fix/binary-backend-ieee754`. Minor
   digging first: confirm which ops route through the `complex` package vs
   BinaryBackend, since that decides where the diff lives.
6. **EXLA memory-tracking CI flake** (courtesy). The test captures a global
   memory baseline that async neighbours can free mid-test; it flaked on
   #1815's CI and will flake on others. Two-line fix (GC before baseline, or
   delta-based assertion).

### Discussion first — fix layer is a maintainer call

**f64 overflow + product-accumulator flavor** (HIGH) — **fix is in Complex,
draft PR open**: https://github.com/elixir-nx/complex/pull/31 (2026-10-07,
fork branch `fix/real-binary-op-overflow`). `Nx.add(max_f64, max_f64)`
raised `ArithmeticError` and `Nx.pow` returned NaN because
`Complex.add/subtract/multiply/divide` on reals re-raised BEAM float
overflow and `Complex.pow` mapped it to `:nan`. BinaryBackend routes ALL
of its element-wise arithmetic and the `Nx.product` accumulator through
these, so one Complex change fixes both the element-wise and the
reduction flavor (verified: with the patched dep swapped into the nx
build, all five repro cases return ±Inf, including 600×1e3 f16 product).
Nx follow-up once a Complex release ships and nx bumps the dep: flip the
`[BUG-F64-OVERFLOW]` pins in `fuzz_float_edge_test.exs` to value
assertions and drop the [1e-3, 1e3] magnitude cap in the NaN-propagation
property there. Earlier context: PR #1707 was closed with "defer to the
Complex library" — this is that deferral, done.

### Needs a real-world repro before it's worth chasing

**Grad's remaining ~24 dark lines** — deep vectorized-grad reconciliation
guards from the #1533-era rework plus defensive raises. Synthetic
construction attempts kept landing in the healthy arms. One concrete
by-product already recorded: `grad.ex:320-322` is **dead code** (the `true ->`
fallback re-tests a predicate the cond arm above already consumed) — an
upstream refactor note, not a bug.

**s2 bit-count representability**: bit-count ops return the input type and
s2 cannot represent the count 2, so `popcount(s2 -1)` returns `-2`. Fixing
means widening the output type, saturating, or documenting — a design
decision, bundled in `clz_sub_byte_crash.md`.

## Checkpoint (#765) — discussion prep done 2026-10-07, no code posted

Branch state: local `feat/gradient-checkpointing` == fork
`feat/gradient-checkpointing-v2` (d635c222), based on da1f4fb8, 30 commits
behind `origin/main`. The older fork branch `feat/gradient-checkpointing`
predates `Nx.block` and is obsolete. The branch adds a `:checkpoint` Expr op,
`Nx.Defn.checkpoint/2` (explicit input + fun/1), grad clauses that mirror
`:block`, an Evaluator `{:recompute, count, thunk}` cache entry so the output
is never cached, and a 1244-line `checkpoint_test.exs` (57 tests). No EXLA
support. polvalente asked to hold the PR until José reviews the spec, and
said it is a missing feature for Axon 1.0, to land after Nx 1.0 (now out).

### What was verified on 2026-10-07 (scripts in `scratch/checkpoint_*.exs`)

1. **`Nx.block` already has checkpoint semantics at the expression level.**
   `Nx.Defn.Grad.parents_args(:block)` re-invokes the callback on the
   original inputs and differentiates that fresh tree. The backward never
   references the forward body's intermediates. `debug_expr` of
   `grad(sum(block(f)(x)^2))` shows the forward as one `block` node and the
   recompute as separate inline `exp`/`cos` nodes. The checkpoint branch's
   grad clauses are a copy of block's.
2. **XLA undoes the recompute by CSE unless a barrier is present.** The
   unoptimised HLO has the forward body inside a `call` plus the inline
   recompute (two `exponential`). After optimisation on either backend the
   call is inlined and CSE merges them to one `exponential`.
3. **A `stablehlo.optimization_barrier` on the block's inputs prevents the
   CSE on CUDA but not on CPU.** Throwaway patch: 5-line
   `Value.optimization_barrier/1` using the generic `op/5` emitter, applied
   to `call_args` at the top of `EXLA.Defn.default_block_implementation/5`.
   Per-pass dump (`--xla_dump_hlo_pass_re=.*`):
   - CUDA, no barrier: CSE at pass 7 → 1 `exponential` in the final module.
   - CUDA, barrier: barrier expanded at pass 49 (`remat-pipeline`, the end
     of the pipeline, after every CSE) → 2 `exponential` survive.
   - CPU, barrier: `cse_barrier_expander` runs at pass 6, CSE at pass 15 →
     1 `exponential`. The CPU pipeline removes barriers before CSE, so the
     memory saving cannot exist on the host client and a regression test
     for it must run on a GPU or inspect pre-CSE HLO.
   Gradients were numerically identical in every configuration.
4. **Measurement tool**: the XLA dump also writes
   `module_*.{cpu,gpu}_after_optimizations-memory-usage-report.txt` with
   peak buffer-assignment bytes. That is how to show a peak-memory drop on a
   layered model without timing anything.

Corrections to earlier notes: the fence does *not* have to go in at re-trace
time. Putting the barrier on the *forward* block's inputs inside EXLA's
lowering is enough to defeat CSE, because forward becomes `f(barrier(x))`
and the recompute stays `f(x)`. What EXLA-only placement cannot do is tie
the recompute to the incoming cotangent so the scheduler is forced to run it
late; that needs the barrier in the gradient expression, built where `g` is
known (`update_grads`), which means a backend-neutral Nx-level node. The
issue's "Torchx: delegate to torch.utils.checkpoint" item is moot: Torchx is
a backend under the Evaluator and has no autograd of its own.

### Decision points to settle with polvalente/José before coding

1. Representation: new `:checkpoint` Expr op (branch) vs `Nx.block` with a
   `%Nx.Block.Checkpoint{}` struct (thin layer, grad machinery free,
   Evaluator body already uncached because block callbacks run eagerly).
2. Evaluator output semantics: never cache the output (polvalente's stated
   preference, branch does this with `{:recompute}`) vs refcounted like any
   node (JAX semantics; intermediates are already freed by refcounting).
3. Public API: `checkpoint(fun)` closure (issue text) vs `checkpoint(input,
   fun)` (branch) vs list of inputs; module (`Nx.Defn` vs `Nx.Defn.Kernel`
   next to `custom_grad`/`stop_grad`); name (checkpoint vs remat).
4. EXLA barrier placement: forward-input barrier in lowering only (verified,
   no Nx-core change, CSE only) vs cotangent-tied barrier in the grad
   expression (JAX-style scheduling guarantee, needs update_grads-time
   re-trace like `:while` and a backend-neutral barrier node).
5. Policy hook: opts/struct field reserved now (`policy:`), implemented later.
6. Tests: new file vs `describe "checkpoint"` in grad_test.exs; must include
   vectorized inputs (block's grad devectorizes the re-trace; the branch's
   clause does not) and a GPU-only HLO/memory assertion for the barrier.

### Found 2026-10-06, needs an EXLA fix (token chaining) — not yet filed

**io_call program order on CUDA**: independent io_calls are lowered as
unconnected side-effecting custom calls, so the GPU scheduler reorders them
(deterministically b, c, a for a three-call chain). Upstream's own
`EXLA.Defn.APITest` asserts program order and fails on CUDA. Root cause,
HLO evidence and the fix shape are in
`nx/FUZZ_FINDINGS/exla_io_call_order_not_preserved_on_gpu.md`.

## Infrastructure on `integration`

- **Fuzz corpus**: 36 `nx/test/nx/fuzz_*.exs` files. Highlights beyond the
  original campaign: bit-pattern float generator (`test/support/fuzz_gen.ex`)
  that feeds real NaN/Inf/denormals instead of iota; block jit-vs-eager
  differential; `Nx.Random` properties; conv value oracle; error-contract
  table (46 entries); non-finite convention consistency; integer semantics
  against exact unbounded-integer references; coverage-guided dark-line
  suite.
- **Cross-backend differentials**: `exla/test/differential_fuzz_test.exs`
  (f64 + complex + large-shape GPU sweeps, non-finite agreement, two pinned
  divergences) and `torchx/test/differential_fuzz_test.exs`. The EXLA file
  is CUDA-only in practice: with `EXLA_CLIENT=host` the same 5 tests fail
  before and after the v1.0 merge (3 QR Q-matrix sign flips, the
  clip-NaN divergence pin, one NaN reduction) — host artifacts, not bugs.
  Everything else in `exla/` and the 4-device sharding suite is green on host.
  On the CUDA client the full `exla/` suite has 24 long-standing failures
  (same set before and after the merge): the donation and memory-tracking
  tests build buffers on `:host` and jit on the default client, the
  `CustomCallAliasTest` asserts CPU custom-call names in MLIR, the
  `qr_cpu_custom_call` f32 case is too tight for GPU, the io_call
  program-order test, and nine last-ULP doctests. None are fork bugs.
- **Sharding**: `exla/test/exla/defn/sharding_fuzz_test.exs` — elementwise
  equivalence plus collectives (all-reduce over the sharded axis). Run with
  `EXLA_TARGET=host XLA_FLAGS=--xla_force_host_platform_device_count=4`.
- **Deep runs**: every property budget is `N * @fuzz_scale`, read from
  `FUZZ_SCALE` (default 1 = CI behavior). The overnight harness pattern is
  in the roadmap; needs `--timeout 600000` and a devenv shell. First 9-hour
  run did 150 iterations and found the product-accumulator bug.
- **Coverage tooling**: scratch scripts intersect `.coverdata` from the
  corpus and the full suite to list never-executed lines. Dark lines in the
  seven load-bearing modules went ~130 → 80; the residue is defensive or
  unreachable code, individually understood.

## Environment gotchas (cost real time before)

- `mix` must run from `nx/`, `exla/`, or `torchx/` — from the repo root it
  picks up the wrong project. Compound `cd nx && ... && git ...` breaks
  because git paths are repo-root-relative; keep them in separate calls.
- devenv 2.3 reads `devenv.yaml` keys in snake_case, and once a `nixpkgs:`
  block exists it takes the nixpkgs config from that block only — a top-level
  `allow_unfree` is silently ignored (error: "package 'cuda_nvcc' has an
  unfree license"). `devenv.yaml` (untracked) now sets `allow_unfree` and
  `cuda_capabilities: ["12.0"]` under `nixpkgs:`; the latter cuts NCCL from
  nine GPU archs to one (an hour+ → minutes) at the cost of no binary-cache
  hits for CUDA packages. Widen the list if another GPU ever joins.
- CUDA/EXLA work needs `devenv shell` (a bare shell lacks `make`, CUDA libs,
  and `python3`). `XLA_TARGET=cuda12`, not `cuda` — the bare name is no
  longer a valid target in xla 0.10.
- EXLA caches `libexla.so` under `~/.cache/xla/exla/<version-key>/` with a
  version-based key. Delete the cached `.so` when testing C++ changes or the
  "rebuild" silently reuses the old binary.
- `>>>>` in a binary comprehension parses as the shift operator; break the
  binary construction onto its own line.
- GitHub's API threw 503s repeatedly; PR-creation loops should check for an
  existing PR before retrying so a flake can't double-post.

## Standing conventions

- **Never open an upstream PR, issue, or comment without explicit
  permission** — including tests-only follow-ups prompted by a review note.
- One PR in flight at a time; prepare the next locally while waiting.
- Preview on the fork first (draft PR), then submit upstream on the word.
- One topic per PR — no bundling, even for fixes that share a discovery
  story.
- Every fix PR carries its flipped pins as regression tests.
- Additive commits for review feedback; no force-push once public.
- No historical comments in code ("this used to...") — that story belongs in
  the commit message and PR body.
- Issues: prefer small and specific over one bundled report.

## Immediate next actions

1. Review/edit draft PR 1853 (`fix/qr-f16-eps`) and mark it ready.
2. All suites verified against the v1.0 merge on 2026-10-06: nx, torchx,
   exla-host, 4-device sharding, and the CUDA differential suite (33
   properties, 31 tests, both GPU divergence pins intact).
3. Review/edit draft PR 1854 (`fix/pinv-zero-shape-batch`) and mark it ready.
4. Review/edit draft PR 1855 (clip NaN) and mark it ready; flip the clip pins after merge.
5. Review/edit draft Complex PR 31 (f64 overflow) and mark it ready; after a
   Complex release, bump nx's dep and flip the `[BUG-F64-OVERFLOW]` pins.
