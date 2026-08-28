# Nx fork — handoff (2026-08-28)

State of this fork's contribution work to elixir-nx/nx. Companion docs:
`WORKPLAN.md` (PR sequencing), `nx/FUZZ_ROADMAP.md` (test-coverage map),
`nx/FUZZ_FINDINGS/README.md` (bug queue).

## Where we are in one paragraph

Eight PRs merged upstream this cycle, all originating from a property-fuzz
campaign run on this fork. Nothing is currently in flight upstream. The
`integration` branch (74 commits ahead of `origin/main`) holds the fuzz
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
re-reverses the batch prefix — unequal double-batch dims crash, equal ones
silently broadcast. Two-line `put_elem` fix; tests cover `{3,2,2,2}`,
non-square `{5,4,2,3}` with Moore-Penrose identity, and the all-zeros
branch. Full nx suite green. Commit message is written for upstream.

### Small focused PRs — fix direction is unambiguous

1. **clip non-finite** (HIGH, do first). `clip(NaN, 0, 2)` returns `0.0` on
   BinaryBackend — launders a NaN into a legitimate-looking in-range value.
   **Confirmed cross-backend divergence**: EXLA/cuda propagates NaN in all
   three argument positions. EXLA, IEEE, and Nx's own min/max composition
   all agree on the correct answer, so there is nothing to debate. Fix is in
   BinaryBackend's clip comparison chain; tests come free by flipping
   `[BUG-CLIP-NONFINITE]` pins plus the EXLA `[DIVERGENCE-CLIP-NONFINITE]`
   pin into exact agreement.
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
5. **unary non-finite** (HIGH). `floor`/`ceil`/`round`/`atanh` crash on
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

**f64 overflow + product-accumulator flavor** (HIGH). `Nx.add(max_f64,
max_f64)` raises `ArithmeticError` where IEEE wants `Inf`; `pow` returns NaN.
The reduction flavor is worse: `Nx.product`'s accumulator is a BEAM double
regardless of dtype, so ~600 f16 elements of 1e3 crash — **this flavor
affects every float type**. Do NOT open a PR blind: PR #1707 was previously
closed with "IEEE 754 deferred to Complex library PR", so the rescues may
belong in the `complex` package rather than BinaryBackend. File one small
issue with the matrix and ask. The finding doc is effectively the issue text.

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

## Checkpoint (#765) — parked, deliberately

Gradient checkpointing has a working Evaluator implementation ported to
current main on `feat/gradient-checkpointing` (local) and
`feat/gradient-checkpointing-v2` (fork), with 57 tests including
rematerialization execution-count assertions. polvalente asked to hold for
after 1.0 as "purely additive", and said it's a missing feature for Axon 1.0.

Before posting anything to the issue, two things are worth doing locally:

1. **Prototype the block-route question.** The drafted reply claims
   checkpoint might become an `Nx.Block` struct since the grad machinery is
   now identical to `:block`'s. Unverified: block's evaluator path *caches*
   its output while checkpoint must never cache. Settle it in code so the
   comment carries evidence.
2. **EXLA barrier design.** The subtlety recorded in the test file: the
   barrier cannot live only in EXLA's lowering, because by then grad has
   already re-traced the body into ordinary ops indistinguishable from the
   forward pass — CSE would merge them and silently restore O(n) memory
   while every gradient test stays green. The fence must go in at re-trace
   time, mirroring JAX's `_remat_lowering` with `prevent_cse=True`. EXLA has
   no `:checkpoint` lowering and no `stablehlo.optimization_barrier` emitter
   today.

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
  divergences) and `torchx/test/differential_fuzz_test.exs`.
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

1. Push `integration` (2 local commits: main merge + pin flip).
2. Decide on `fix/pinv-zero-shape-batch` → upstream PR.
3. Build the clip fix, preview on fork, then submit.
4. Draft the f64-overflow issue on the fork for review before filing.
