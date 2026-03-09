# Nx Bugs & Gaps Tracker

Findings from auditing `nx/lib/nx/defn/grad.ex` and related code.

## Bugs

### 1. `select_and_scatter` crash with `:same` padding (#1675)
**Status:** Merged (#1676)

### 2. `window_max`/`window_min` gradients crash with `window_dilations` (#1679)
**Status:** Merged (closed #1679)

### 3. `triangular_solve` gradient missing `:conjugate` case
**Status:** PR open (#1681), under review
**Root cause:** Two `case opts[:transform_a]` blocks only handle `:none` and `:transpose`, but the forward pass also supports `:conjugate`.
**Fix:** Add `:conjugate -> Nx.conjugate(...)` to both case blocks.

### 4. `window_scatter_max`/`window_scatter_min` gradient drops `window_dilations`
**Status:** Merged (stacked on bug #2, closed with #1679)

### 5. Vectorized tensors crash in gradient computation (#1533)
**Status:** Partial fix on fork branch `fix/vectorized-grad-1533`
**Location:** Core gradient infrastructure — `grad.ex` (`to_grad`, `recur_to_grad`, `reduce_g`)
**Root cause:** The backward pass re-vectorizes args/ans but not `g`, and opts (axes, padding) reference devectorized axis numbers while tensors are re-vectorized.
**Fix so far:** Three changes in `grad.ex`:
  - `to_grad/4`: devectorize arg before broadcast, then re-vectorize result
  - `maybe_vectorize_grad/3`: broadcast `g` to devectorized ans shape, then vectorize
  - `reduce_g/3`: adjust axes by vectorization offset before broadcast
**Now works:** `grad(vectorized_x, &Nx.sum/1)`, `&Nx.mean/1`, elementwise ops, composed reductions
**Still broken:** Ops with shape-dependent grad clauses (squeeze, slice, pad) when vectorized input passes through them. Each needs per-clause axis adjustment.
**Complexity:** Medium for remaining fixes — systematic but repetitive

### 6. CallbackServer process leak on repeated JIT compilation (#1682)
**Status:** PR open (#1683), under review — José flagged for new release
**Root cause:** Every `__compile__` call unconditionally starts a `CallbackServer` process, even without `:runtime_call` nodes. Exhausts BEAM process limit after ~25K+ JIT calls.
**Fix:** Lazy `CallbackServer` startup, only when first `:runtime_call` is encountered.

## Gaps (not bugs, but missing features)

### 7. `window_product` has no gradient
**Status:** Documented as unsupported (`@error` list at grad.ex line 1194)
**Impact:** `Nx.Defn.grad` through `window_product` raises "cannot compute gradient" error.
**Notes:** Could be implemented as `window_product / element` but needs care around zeros.

### 8. `window_mean` has no explicit gradient
**Status:** Works via composition (defined as `window_sum / size`), so autodiff handles it.
**Impact:** None functionally, but no specialized optimization.
