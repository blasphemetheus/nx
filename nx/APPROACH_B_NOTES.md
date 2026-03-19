# Approach B: Devectorized-space gradient computation

## Summary

Instead of the current strip-then-re-add-per-node pattern in `recur_to_grad`,
compute the entire gradient in devectorized space and only revectorize the
final result in `to_grad`.

## Current approach (A) — strip then re-add per node

```
recur_parents_tree: devectorize(node), record vectorized_names per node
recur_to_grad:
  for each node:
    revectorize_node(arg, vectorized_names)  ← re-add vec axes to args
    Nx.vectorize(ans, vectorized_names)       ← re-add vec axes to ans
    adjust_vectorized_args(op, args, offset)  ← fix axis indices
    grad(op, args, ans, g)                    ← compute gradient
```

Problems: double-vectorization collision when ops internally use
apply_vectorized (while, cond, cumsum chain, conv collapse/uncollapse).

## Proposed approach (B) — stay devectorized

```
recur_parents_tree: devectorize(node), record vectorized_names per node
recur_to_grad:
  for each node:
    DO NOT revectorize args or ans
    DO NOT adjust axes (they already include batch prefix)
    grad(op, args, ans, g)  ← compute gradient on devectorized shapes
to_grad:
  revectorize final gradient to match input's vectorized_axes
```

## What changes

1. **recur_to_grad (lines 243-258):** Remove the entire `if vectorized_names != []`
   block. Don't revectorize args, don't revectorize ans, don't adjust axes.

2. **to_grad (line 223):** After computing `res = sum_grad(...)`, revectorize
   `res` to match `arg.vectorized_axes` before broadcasting.

3. **All grad rules:** They now receive devectorized shapes (with batch dims
   baked in). Most rules "just work" because they're element-wise. Rules that
   use axes need to work correctly with the extra batch prefix.

4. **reduce_g:** `Nx.broadcast(g, x, axes: axes)` — the axes already include
   the batch dims since nothing was re-vectorized. Should work as-is.

5. **grad_broadcast:** Same — axes already reference devectorized shapes.

6. **unbroadcast:** Same.

7. **grad(:concatenate):** `elem(t.shape, axis)` — axis already includes batch
   prefix since shapes are devectorized. Should work.

8. **adjust_vectorized_args:** Not needed at all — delete the entire function
   and all its clauses.

9. **revectorize_node, compute_arg_vectorized_names:** Not needed — delete.

## What might break

- Grad rules that hard-code axis indices (e.g., `dot([0], ...)`) — these need
  to work with the actual devectorized shape, not inner-only indices.
- `Nx.broadcast(g, ans)` — if g and ans have different shapes due to
  vectorization, this could fail. But since everything is consistently
  devectorized, they should match.
- Custom grad callbacks (custom_grad) — these receive devectorized tensors
  and should produce devectorized results.

## Expected benefits

- No per-node revectorization → no collision
- No adjust_vectorized_args → simpler code
- while, cond, chained cumsum, conv all work because there's no
  vectorization to collide with
- Linalg grads (QR, Cholesky) may also just work since all tensors
  are consistently 3D (batch + matrix)

## Risk

- Many grad rules implicitly assume shapes match certain patterns.
  Devectorized shapes have extra batch dims that could cause issues.
- The `to_grad` revectorization needs to handle the batch→vectorized
  conversion correctly.

## Initial attempt findings

Tried removing the revectorization block. First test (`Nx.sum` on vectorized
input) failed with "expected length of axes (4) to match rank of shape (3)".

Root cause: the expr tree encodes axis indices from the FORWARD pass, where
ops independently devectorize their inputs (via `apply_vectorized` etc.) and
record devectorized axis indices. During backward pass, approach B's
"everything devectorized" scheme produces shapes with a DIFFERENT batch
prefix than what the forward pass recorded.

For example, `Nx.sum(x_vec)` internally devectorizes `x` to `{2, 3}`, sums
axes `[0, 1]` (all), and vectorizes the scalar result back to `[batch: 2] {}`.
In the expr tree, `opts[:axes]` might be `[0, 1]` or `nil`. But the backward
pass sees `x` with shape `{2, 3}` and the sum node also with different axes.

The issue is that each op's forward pass does its OWN devectorize/revectorize
cycle, so the axis indices in opts are relative to THAT op's devectorized
shape, not to a global devectorized shape.

## Verdict

Approach B needs the axis indices in the expr tree to be consistent with
the devectorized shapes. This requires either:
(a) Normalizing all axis indices during `recur_parents_tree` to be relative
    to the globally devectorized shape, OR
(b) Having the backward pass re-do the per-op devectorization (which is
    what approach A does).

Approach B is not a simple drop-in change — it requires rethinking how
axis indices are stored in the expression tree. It's an architectural
change to the whole defn expression system, not just to grad.ex.
