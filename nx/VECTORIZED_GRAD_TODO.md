# Vectorized Gradient Fix — Remaining Work

Partial fix on branch `fix/vectorized-grad-1533`. Tracks what's done and what's left for #1533.

## Background & Glossary

This section explains the key concepts needed to understand this document.

### Tensors, Shape, Rank, and Axes

A **tensor** is an n-dimensional array of numbers — the fundamental data structure in numerical computing and ML. Tensors generalize scalars (0D), vectors (1D), matrices (2D), and so on to arbitrary dimensions.

- **Shape**: A tuple describing the size of each dimension. A tensor with shape `{2, 3}` has 2 rows and 3 columns (6 elements total). A scalar has shape `{}`.
- **Rank**: The number of dimensions. Shape `{2, 3}` has rank 2. Shape `{5}` has rank 1. Shape `{}` has rank 0.
- **Axis**: A specific dimension, referenced by its integer index (0-based). In shape `{2, 3, 4}`, axis 0 has size 2, axis 1 has size 3, axis 2 has size 4.

### Vectorization (Batching)

**Vectorization** in Nx is a mechanism for applying the same operation across a "batch" of inputs simultaneously, without the operation needing to know about the batch dimension. Think of it like a `for` loop over independent computations, but expressed as a single batched operation for efficiency.

When you **vectorize** a tensor, you move one or more leading dimensions into a special `vectorized_axes` field. This makes them invisible to most operations:

```elixir
x = Nx.tensor([[1, 2, 3], [4, 5, 6]])  # shape {2, 3}
v = Nx.vectorize(x, :batch)             # vectorized[batch: 2] shape {3}
```

After vectorization, `v.shape` returns `{3}` (the **inner shape** — just the non-batch dimensions), and `v.vectorized_axes` returns `[batch: 2]`. Operations on `v` automatically apply independently to each of the 2 batch elements.

**Devectorization** (`Nx.devectorize/2`) is the reverse: it merges the vectorized axes back into the shape as leading dimensions, producing a normal tensor with shape `{2, 3}` again.

### Gradients and Automatic Differentiation (Autodiff)

A **gradient** measures how much a function's output changes when you nudge each input slightly. If `f(x) = x²`, the gradient is `f'(x) = 2x` — at `x = 3`, the gradient is 6, meaning a tiny increase in `x` causes the output to increase ~6x as much.

For multi-dimensional inputs, the gradient is a tensor of the same shape as the input, where each element says "how much does the output change if I nudge this specific element?"

**Automatic differentiation (autodiff)** computes gradients mechanically by applying the chain rule through a computation graph. Nx uses **reverse-mode autodiff** (also called **backpropagation** in ML), which works by:

1. **Forward pass**: Run the computation normally, recording each operation in a graph.
2. **Backward pass**: Walk the graph in reverse, propagating gradients from output back to inputs using the chain rule.

In `grad.ex`, each operation (sum, multiply, dot, etc.) has a **grad clause** — a rule for how gradients flow backward through it. For example, the gradient of `sum(x)` is broadcasting the output gradient back to the shape of `x`.

### Operations Referenced in This Document

- **Elementwise ops** (`sin`, `cos`, `exp`, `add`, `multiply`): Apply independently to each element. Gradient is straightforward (e.g., grad of `sin(x)` is `cos(x)`).
- **Reduction ops** (`sum`, `mean`, `product`, `reduce_max`): Collapse one or more axes into a single value (e.g., summing all elements along axis 1). The `axes` option specifies which dimensions to reduce.
- **Broadcasting**: Automatically expanding a smaller tensor to match a larger one's shape for element-wise operations. E.g., adding a shape-`{3}` tensor to a shape-`{2, 3}` tensor broadcasts the smaller one across the first dimension.
- **Padding**: Adding values (usually zeros) around the edges of a tensor. `padding_config` specifies how much to add on each side of each axis.
- **Strides**: Step size when sliding a window or sampling elements. A stride of 2 means "skip every other element."
- **Window operations** (`window_sum`, `window_max`): Slide a fixed-size window across the tensor, computing a result for each window position. Like a convolution but with simpler aggregation.
- **Dot product** (`dot`): Generalized matrix multiplication with specified batch and contraction axes.
- **Squeeze**: Remove axes of size 1 from a tensor's shape.
- **Stack/Concatenate**: Combine multiple tensors along a new or existing axis.
- **Gather**: Index into a tensor to extract elements at specified positions.
- **FFT/IFFT**: Fast Fourier Transform — converts between time and frequency domains.

### The `vec_offset` Pattern

This is the core fix pattern used throughout this document:

- **`vec_offset`**: `length(x.vectorized_axes)` — the number of vectorized (batch) dimensions. This is 0 for normal tensors.
- **Why it matters**: During the backward pass, `grad.ex` devectorizes all tensors (merging batch dims into the shape), computes the forward expression, then re-vectorizes the tensors for the grad clause. But the **opts** (axis indices, padding configs, etc.) were computed on the devectorized shape and are NOT adjusted. So axis index `1` in opts might actually refer to axis `0` in the re-vectorized tensor's inner shape.
- **The fix**: Subtract `vec_offset` from axis indices in opts, and/or slice configs to skip the first `vec_offset` entries.

### Code Locations

- **`grad.ex`**: `nx/lib/nx/defn/grad.ex` — the entire gradient computation engine
- **`recur_to_grad/4`**: The main backward-pass function that processes each operation node. Line ~252 is where it re-vectorizes tensors but not opts.
- **`to_grad/4`**: Handles leaf variable nodes — where the gradient is finally assigned to input variables.
- **`reduce_g/3`**: Helper that broadcasts a gradient back to a tensor's shape after a reduction operation.

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
