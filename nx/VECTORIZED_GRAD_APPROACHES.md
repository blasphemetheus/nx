# Vectorized Gradient Fix — Approach Exploration

Notes on different strategies attempted for fixing #1533.

## Approach B: Fully Devectorize the Backward Pass

**Idea:** Don't re-vectorize tensors in `recur_to_grad` at all. Keep everything in
devectorized space during the backward pass, then re-vectorize only at the very end in `to_grad`.

### Changes Made (3 edits to `grad.ex`)

#### 1. Gradient seed — use devectorized shape

```elixir
# Before:
grads = %{transformed_expr.data.id => [constant(1.0, transformed_expr)]}

# After:
devec_expr = Nx.devectorize(transformed_expr, keep_names: true)
grads = %{devec_expr.data.id => [constant(1.0, devec_expr)]}
```

`constant/2` uses `t.shape`, which for a vectorized tensor is the inner shape. By
devectorizing first, the gradient seed gets the full devectorized shape, matching the
devectorized `ans` stored by `parents_tree`.

#### 2. `recur_to_grad` — devectorize args instead of re-vectorizing

```elixir
# Before: re-vectorize tensor args
args = Enum.map(args, fn
  %T{} = arg -> revectorize_node(arg, vectorized_names)
  opt -> opt
end)
ans = Nx.vectorize(ans, vectorized_names)

# After: devectorize tensor args (they may still be vectorized from forward pass)
args =
  if vectorized_names != [] do
    Enum.map(args, fn
      %T{} = arg -> Nx.devectorize(arg, keep_names: true)
      opt -> opt
    end)
  else
    args
  end
# ans stays devectorized (already devectorized from parents_tree)
```

#### 3. `to_grad` — re-vectorize at the end

```elixir
# Before:
{Nx.broadcast(res, arg), {nodes, grads}}

# After:
res =
  case arg.vectorized_axes do
    [] -> Nx.broadcast(res, arg)
    vectorized_axes ->
      devec_arg = Nx.devectorize(arg, keep_names: true)
      res = Nx.broadcast(res, devec_arg)
      Nx.vectorize(res, vectorized_axes)
  end
{res, {nodes, grads}}
```

### Results: 225 pass, 3 fail

All non-vectorization tests pass. The opts-vs-tensor mismatch is fixed because both
opts and tensors are in devectorized space — axes match correctly. `reduce_g` needs
no changes. The `vec_offset` pattern is unnecessary.

### Why It Fails: Non-Vectorized Targets in Vectorized Computations

The 3 failing tests all involve a **non-vectorized grad target** participating in a
computation with vectorized tensors:

```elixir
x = Nx.tensor([[1, 2, 3], [4, 5, 6]]) |> Nx.vectorize(:x)
y = 1  # <-- non-vectorized target
grad = Nx.Defn.grad(y, fn y -> Nx.add(x, y) end)
# Expected: vectorized[x: 2] [3.0, 3.0]
# Got:      scalar 6.0
```

**Trace through the failure:**

1. In devectorized space: `x` is `{2, 3}`, `y` is `{}`, output is `{2, 3}`
2. Gradient seed: `ones({2, 3})`
3. `grad(:add, ...)` calls `unbroadcast(y, g, ans)` to reduce `g` from `{2, 3}` to `{}`
4. `unbroadcast` calls `grad_broadcast` which sums over ALL mismatched dims: `[0, 1]`
5. Result: `sum(ones({2, 3})) = 6.0` — a scalar

But the correct answer is `vectorized[x: 2] [3.0, 3.0]`:
- Batch 0: sum over inner dim `[1, 1, 1]` → 3.0
- Batch 1: sum over inner dim `[1, 1, 1]` → 3.0

**Root cause:** In devectorized space, `unbroadcast` can't distinguish batch dimensions
(axis 0, size 2) from data dimensions (axis 1, size 3). It sums over both, collapsing
the batch structure. With re-vectorization, axis 0 is hidden in `vectorized_axes`, so
`unbroadcast` only sees the inner shape `{3}` and sums over `[0]` → result per batch.

### Fundamental Lesson

**Re-vectorization is structurally necessary.** It's not just bookkeeping — it tells
gradient operations (unbroadcast, grad_broadcast, reduce_g) which dimensions are batch
dimensions vs. data dimensions. Without this distinction, reductions collapse batch
structure.

This rules out any approach that fully devectorizes during the backward pass. The fix
must preserve re-vectorization AND also adjust opts to match the re-vectorized shapes.

### What Approach B Got Right

The gradient computation itself (reduce_g, etc.) works correctly when opts and tensors
are in the same space. The consistency is the key — the bug is that re-vectorization
makes tensors and opts inconsistent.

## Approach C: Centralized Opts Adjustment (Current)

Keep re-vectorization (preserving batch structure), but adjust opts centrally in
`recur_to_grad` via `adjust_vectorized_args(op, args, vec_offset)` before dispatching
to individual grad clauses. See implementation on `fix/vectorized-grad-approach-c`.
