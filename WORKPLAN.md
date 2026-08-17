# Work plan — fuzz findings + checkpoint (drafted 2026-08-16)

Nothing here is posted upstream yet. Ordering rationale: one PR in flight at
a time (polvalente's limit-WIP guidance); bug fixes are pre-1.0 material,
checkpoint is post-1.0 additive; discussion-first where the fix layer is a
maintainer call; silent-wrong-output severity beats crashes.

## Phase 1 — self-contained fix PRs (one at a time, each off clean main)

1. **PR A: multi-tensor `impl!` dispatch.** -> IN FLIGHT: draft
   https://github.com/elixir-nx/nx/pull/1815 (branch
   `fix/multi-tensor-dispatch`; preview was fork PR 12 - same head
   branch, pushes update both). put_slice/clip/gather/reduce/
   window_reduce dispatch on the first tensor only; mixed concrete+Expr args
   crash. ~5 one-line fixes + flip the [BUG-DISPATCH-*] pins in
   `fuzz_indexed_ops_test.exs`. Easiest review → goes first.
2. **PR B: batched pinv.** -> SUBMITTED: https://github.com/elixir-nx/nx/pull/1816 (open, not draft; preview was fork PR 13). n=1 excluded (svd bug, own finding doc).
   silently returns rank-4 (worst severity in the queue). Fix mirrors the
   #1748-era batched handling of its siblings. Flip the three
   [BUG-PINV-BATCHED] pins in `fuzz_linalg_test.exs`.
3. **PR C: sub-byte bitstring pair.** -> BUILT: fork draft PR 14 (branch `fix/sub-byte-bitstrings`, incl. reshape-:auto rider). HOLD upstream submission until PR A or B resolves (one-in-flight). `from_binary` is_binary→is_bitstring
   guard + `Nx.Backend` inspect `tail::binary`→`tail::bitstring` for :s/:u.
   Pins in `fuzz_serialization_test.exs`. The reshape-two-`:auto` LOW
   (ArgumentError instead of ArithmeticError) rides along.

## Phase 2 — non-finite/overflow cluster (discussion first)

4. **File ONE consolidated issue** with the full matrix from
   `nx/FUZZ_FINDINGS/{unary_nonfinite_crashes_and_wrong_values,
   f64_binary_op_overflow_arithmetic_error}.md`: floor/ceil/round/atanh
   crashes; tanh(±Inf)=NaN, sign(NaN)=1.0, sign(-Inf)=+1.0 wrong values;
   f64 binary-op overflow raises; product-accumulator overflow crashes for
   EVERY float dtype. Ask the layering question explicitly (fix in the
   `complex` package vs BinaryBackend) — precedent: PR #1707 was closed
   with "IEEE 754 deferred to Complex library PR".
5. **Fix PR(s) per the answer.** Likely: revive
   `fork/fix/binary-backend-ieee754`'s `ieee754_fallback` pattern for
   unary + add binary-op/reduction overflow rescue; possible companion PR
   to the complex package. Flip [BUG-UNARY-NONFINITE]/[BUG-F64-OVERFLOW]
   pins and remove the smoke-suite product clamp + NaN-prop caps in
   `fuzz_float_edge_test.exs`/`fuzz_test.exs`.

## Phase 3 — checkpoint #765 (prototype first; lands after 1.0)

6. **Local prototype: checkpoint as an Nx.Block struct.** The one
   unverified claim in the drafted reply to polvalente: block's evaluator
   path caches its output, checkpoint must never cache. Determine whether
   block can carry no-cache semantics or whether a dedicated node stays
   necessary. Evidence beats opinion in the eventual comment.
7. **Post the revised #765 comment** (with prototype findings + branch
   permalinks, no PR references) when given the go-ahead.
8. **EXLA barrier spike** on `feat/gradient-checkpointing-v2`:
   `stablehlo.optimization_barrier` emitter in `EXLA.MLIR.Value` + the
   grad-level barrier-node design (barrier must wrap the re-trace inputs at
   grad time — mirrors JAX `_remat_lowering` `prevent_cse=True`). Ready to
   move when the spec settles.

## Continuous (no posting)

- Occasional overnight runs (`FUZZ_SCALE=25` loop, devenv shell,
  `--timeout 600000`) — suite is deep-run-stable; FAILs are likely real.
- Flip pins as upstream fixes land (they self-announce by failing).

## Standing rules

- Confirm before any upstream PR/issue/comment.
- Additive commits for review feedback; no force-push.
- Every fix PR carries its flipped pins as regression tests.
- No side-findings bundled into focused PRs.
