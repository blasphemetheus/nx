# Nx.LinAlg.pinv crashes on batched (3-D+) input in the forward pass

Found 2026-08-14 while repairing stale pins after the upstream #1740/#1748
fixes: the `[meta #1748]` pinv grad property turned out to crash in the
*forward* pass, not the grad.

## Minimal repro

```elixir
a = Nx.iota({2, 3, 3}, type: :f32) |> Nx.add(Nx.eye(3))
Nx.LinAlg.pinv(a)
# ** (ArgumentError) cannot broadcast tensor of dimensions {2, 3, 2, 3} to {2, 3, 3}
```

The failure mode depends on the inner matrix size — three different bugs:

| Input | Result |
|---|---|
| `{2, 1, 1}` | crash: `cannot reshape, current shape {1} is not compatible` |
| `{2, 2, 2}` | **silent wrong output**: returns shape `{2, 2, 2, 2}` (rank 4!) |
| `{2, n, n}`, n ≥ 3 | crash: `cannot broadcast {2, n, 2, n} to {2, n, n}` |

Non-square batched fails like n≥3. 2-D input works fine. The n=2 silent
rank-4 output is the worst variant — no error, wrong shape.

**Double-batch corner RESOLVED (2026-08-17)**: the "transposed batch
axes" failure for double-batch inputs ({3,2,n,n} -> "cannot broadcast
{3,2,..} to {2,3,..}") is NOT in svd — it is `pinv_zero_shape`
reversing the batch prefix (it transposes the last two dims via a
reversed dim list and never re-reverses the rest). Equal batch dims
mask it via silent broadcast. Trivial put_elem fix verified on
scratch/svd-batch-transpose. PR-ready, independent of the eigh fix
(test via {3,2,2,2}/{5,4,2,3} + zeros for the zero branch).

## Expected vs observed

Expected: batched pinv, like the rest of `Nx.LinAlg` (svd, qr, lu, solve,
determinant all support leading batch dimensions post-#1748).
Observed: `ArgumentError` from `Nx.Shape.binary_broadcast` inside a `cond`
broadcast clause — an intermediate acquires a doubled batch structure
(`{2, 3, 2, 3}`), suggesting an outer-product or dot inside `pinv` contracts
the wrong axes for rank-3 input.

## Classification

Same family as the #1741–#1746 batched-LinAlg class, but in the forward
pass. pinv was noted in the fuzz backlog as "same class, not separately
filed" — the upstream fixes covered the per-op grads but not pinv itself.
Not yet filed upstream.

## Pinned

`fuzz_linalg_test.exs` — `[BUG-PINV-BATCHED]` property asserts the current
crash; flip to a shape/value assertion when fixed.

## Priority

**HIGH** (upgraded from MED once the n=2 silent wrong output was found) —
batched pinv either crashes or silently returns a wrong-shape tensor; fix
likely mirrors the batched handling added to its siblings.
