# Vectorized Gradient Fix — Remaining Work

Partial fix on branch `fix/vectorized-grad-1533`. Tracks what's done and what's left for #1533.

## What Works Now

- `grad(vectorized_x, &Nx.sum/1)`
- `grad(vectorized_x, &Nx.mean/1)` (via composition)
- `grad(vectorized_x, fn x -> Nx.sum(Nx.multiply(x, x)) end)`
- Elementwise ops: sin, cos, exp, add, multiply, etc.
- All existing non-vectorized-target cases (unchanged)

## The Core Problem

`recur_to_grad/4` (line ~252) re-vectorizes tensor args but leaves `opts` unchanged.
The `opts` (axes, padding, strides) reference axis indices from the **devectorized** shape,
but the tensors now have vectorized axes prepended to their shape. This causes axis index
mismatches in any grad clause that uses `x.shape`, `opts[:axes]`, or `Nx.rank(x)`.

### Example

```
devectorized x: shape {2, 3}, sum axes: [1]
re-vectorized x: vectorized[x: 2] shape {3}, but opts still says axes: [1]
inner axis 0 in vectorized = axis 1 in devectorized
```

## Fix Strategy

Each affected grad clause needs its axis/shape references adjusted by `vec_offset = length(x.vectorized_axes)`. The pattern is:

```elixir
# Before (broken with vectorization):
axes = opts[:axes]
shape = x.shape

# After (vectorization-aware):
vec_offset = length(x.vectorized_axes)
axes = Enum.map(opts[:axes], &(&1 - vec_offset)) |> Enum.filter(&(&1 >= 0))
shape = x.shape  # .shape is already the inner shape when vectorized
```

Note: `x.shape` on a vectorized tensor IS the inner shape (vectorized dims are in `x.vectorized_axes`, not in `x.shape`). So `.shape` references are actually fine — the issue is only with **axis indices from opts** that were computed on the devectorized (flattened) shape.

## Affected Grad Clauses

### Priority 1 — Common in ML workloads

| Op | Line | Issue | Fix |
|----|------|-------|-----|
| `reduce_g` (helper) | 1461 | `opts[:axes]` from devec space | **DONE** — adjusted axes by vec_offset |
| `product` | 591 | `opts[:axes]`, `Nx.shape(x)` for unsqueeze | Adjust axes by vec_offset |
| `reduce_max/min` | 626 | `opts[:axes]`, shape iteration with `i in axes` | Adjust axes by vec_offset |
| `dot` | 641 | `Nx.rank(x.shape)`, `Nx.axes(x.shape)`, batch/contract axes | Adjust all axis lists by vec_offset |
| `squeeze` | 522 | `axes` from devec, `Nx.broadcast(g, x.shape, ...)` | Adjust axes by vec_offset |
| `reshape` | 526 | Uses `x.shape` in reshape — should be fine since `.shape` is inner |  Likely OK |
| `pad` | 534 | `padding_config` has wrong rank, `Nx.rank(unpadded)` | Slice padding_config to skip vec dims |
| `stack` | 724 | `Nx.rank(ans)`, axis index into shape list | Adjust axis by vec_offset |
| `concatenate` | 741 | Same as stack | Adjust axis by vec_offset |

### Priority 2 — Less common ops

| Op | Line | Issue | Fix |
|----|------|-------|-----|
| `put_slice` | 557 | `Nx.shape(update)` for slice lengths | Likely OK (`.shape` is inner) |
| `indexed_put` | 566 | `updates.shape` in reshape | Likely OK |
| `indexed_add` | 575 | `updates.shape` in reshape | Likely OK |
| `gather` | 765 | `opts[:axes]`, `tuple_size(i_shape)`, shape iteration | Adjust axes, use inner shape |
| `window_min/max` | 675 | `opts[:padding]`, `opts[:strides]` | Slice to skip vec dims |
| `window_sum` | 688 | `.shape`, `Nx.rank(x)`, padding/strides | Adjust rank, slice configs |
| `window_scatter` | 1146 | `Nx.axes()`, `.shape`, padding/strides | Filter vec axes from iteration |
| `fft/ifft` | 1495 | `Nx.rank(t) - 1`, last-axis assumption | Use inner rank |

### Priority 3 — Specialized ops

| Op | Line | Issue | Fix |
|----|------|-------|-----|
| `triangular_solve` | 1065 | `Nx.shape()` pattern match `{n}` | Match on inner shape |
| `grad_broadcast` helper | 1440 | `axes` on shape, `Nx.axes(shape)` | Adjust axes |
| `reshape_axis_into` helper | 1414 | `Nx.rank(x.shape)`, axis indices | Adjust by vec_offset |
| `reshape_axis_out_of` helper | 1423 | Axis indices on shape | Adjust by vec_offset |
| `conv_lhs_padding` helper | 1366 | Shape dims include vec axes | Ensure caller strips vec dims |
| `grad_scatter_window__gather_windows` | 1518 | `.shape` includes vec axes | Strip vec dims from shape |

## Key Insight

After re-reading the Nx.Tensor struct: `.shape` on a vectorized tensor is the **inner** shape (excludes vectorized dims). So most `.shape` references are actually correct. The real problem is only:

1. **`opts` axis indices** — computed on devectorized shape `{vec_sizes..., inner_sizes...}`, so they include vec dim offsets
2. **`Nx.rank(x)`** — returns inner rank, but `Nx.rank(x.shape)` also returns inner rank... need to verify which is used
3. **Configs** (padding, strides) — computed for all dims of devectorized shape, need to be sliced to inner dims only

## Testing Strategy

For each fixed op, add a test like:
```elixir
test "grad of vectorized [op]" do
  x = Nx.tensor(...) |> Nx.vectorize(:batch)

  # Verify mental model: vectorized grad == stack of individual grads
  individual_grads = for row <- ... do
    Nx.Defn.grad(row, &fun/1)
  end
  expected = Nx.stack(individual_grads) |> Nx.vectorize(:batch)

  actual = Nx.Defn.grad(x, &fun/1)
  assert_all_close(actual, expected)
end
```
