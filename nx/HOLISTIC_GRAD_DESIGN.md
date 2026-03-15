# Holistic Vectorized Gradient Approach

Alternative to the per-op `adjust_vectorized_args` approach in PR #1697.

## What this branch does

Removes ~150 lines of per-op vectorized args adjustment. Instead:

1. **Devectorize the gradient seed** in `transform/3` — gradients flow through
   the expression tree in devectorized space
2. **Don't re-vectorize args** in `recur_to_grad` — args stay as the expression
   tree recorded them (devectorized)
3. **Don't adjust opts** — no `adjust_vectorized_args`, no per-op handlers
4. **Re-vectorize only in `to_grad`** — the final gradient output matches
   the original target's vectorization

## Why this works for most cases

The expression tree already stores devectorized operations (because
`apply_vectorized` devectorizes before calling the backend). The grad
clauses were originally written for devectorized tensors. So:

- Gradient seed: devectorized `{batch, ...}` shape
- Args: devectorized `{batch, ...}` shapes
- Opts (axes, padding, etc.): reference devectorized positions
- Everything is consistent — grad clauses work naturally

## What fails: non-vectorized target in vectorized context

9 tests fail. All share one pattern:

```elixir
# Non-vectorized y, vectorized x
grad(y_scalar, fn y -> f(x_vectorized, y) end)
```

**Root cause**: `unbroadcast` sums the gradient across ALL broadcast
dimensions, including the batch dimension. In devectorized space, the
batch dim looks like a regular broadcast dim.

Example: `grad(y, fn y -> add(x_vec, y) end)` where x is `[foo: 3]`:
- Expression tree: `add(x_devec{3}, y_devec{})` → result `{3}`
- Gradient g = `{3}` (values `[1.0, 1.0, 1.0]`)
- `unbroadcast(y{}, g{3}, ans{3})` → sums to scalar `3.0`
- Expected: `[1.0, 1.0, 1.0]` vectorized as `[foo: 3]`

The per-batch gradient info `[1.0, 1.0, 1.0]` is destroyed by
`unbroadcast` before `to_grad` can re-vectorize it.

## Approaches that DON'T work

### Modifying `unbroadcast` to skip batch dims

Tried: thread `vec_offset` through process dictionary, have `unbroadcast`
skip the first N dims when summing.

Result: 5 previously-passing tests broke. `unbroadcast` can't distinguish:
- **Grad targets**: should keep batch dims (per-batch gradient)
- **Constants**: should sum batch dims (correct mathematical unbroadcast)

Both appear identical to `unbroadcast` — a scalar broadcast to a larger shape.

### Hybrid: re-vectorize ans/g but not args

Tried: keep args devectorized, vectorize g and ans so `unbroadcast`
compares inner shapes.

Result: doesn't work. Nx operations (`broadcast`, `multiply`, etc.) call
`broadcast_vectors` which requires consistent vectorization between
operands. Mixing vectorized g with devectorized x corrupts shapes.

### Post-hoc recovery in `to_grad`

Analysis: by the time `to_grad` runs, the per-batch gradient has already
been summed to a scalar by `unbroadcast`. The information is irreversibly
lost. Can't recover `[1.0, 1.0, 1.0]` from `3.0`.

## The fundamental constraint

**Opts and tensor shapes must be consistent.** This creates two poles:

| Approach | Tensors | Opts | Consistency | Failures |
|----------|---------|------|-------------|----------|
| Per-op (PR #1697) | re-vectorized (inner shapes) | adjusted per-op | ✓ | 0 |
| Holistic (this branch) | devectorized (full shapes) | unadjusted (already correct) | ✓ | 9 (mixed-vec) |

There is no clean middle ground because:
- If tensors are re-vectorized → opts must reference inner positions → per-op adjustment needed
- If tensors stay devectorized → opts already correct → but `unbroadcast` can't distinguish batch from broadcast dims

## Score

- 256/265 tests pass (96.6%)
- 9 failures: all non-vectorized target + vectorized context
- 0 regressions on non-vectorization tests (225+ tests)
- ~150 lines of per-op code removed

## Open question for maintainers

1. Is mixed-vectorization (non-vec target + vec context) a supported use case?
   If not, the holistic approach is complete.
2. If yes, should we keep the per-op approach and address polvalente's review
   comments (missing handlers for window_product, fft, conv)?
3. Or is there a third approach we haven't considered?
