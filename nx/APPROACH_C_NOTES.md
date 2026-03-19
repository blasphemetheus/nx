# Approach C: Make apply_vectorized transparent to grad

## Concept

Instead of the grad system fighting with vectorize/devectorize nodes created
by apply_vectorized, recognize the devectorize→op→vectorize pattern and
handle vectorization at the boundary:

1. In `parents_args(:optional)`, when inner call has no vectorized axes
   (apply_vectorized pattern), pass `vectorized_names = []` to inner traversal
2. In `update_grads(:optional)`, devectorize incoming gradient `g` before
   passing to inner nodes
3. Inner nodes compute in devectorized space without collision
4. Result flows back out and gets revectorized by the outer optional node

## Status

Not tested yet — needs to be applied ON TOP of PR #1697 fixes (clean main
doesn't support vectorized grads at all). The approach is promising but needs:

1. Cherry-pick all PR #1697 infrastructure fixes first
2. Then apply the approach C changes to parents_args(:optional) and
   update_grads(:optional)
3. Test with cumsum, chained cumsum, while, cond

## Key code locations

- `parents_args(:optional)` at grad.ex:129 — where to stop propagating
  vectorized_names into inner trees
- `update_grads(:optional)` at grad.ex:307 — where to devectorize g at
  the boundary
- `recur_to_grad` at grad.ex:237 — where revectorization happens per-node
  (approach C would still use this for outer nodes, just not inner optional nodes)

## Relationship to other approaches

- Approach A (current): patches per collision — works for 15/17 cases
- Approach B: global devectorized space — fails because axis indices in
  expr tree are per-op, not global
- Approach C: per-optional devectorize boundary — conceptually cleanest,
  should fix while/cond/chained cumsum without needing global axis normalization
