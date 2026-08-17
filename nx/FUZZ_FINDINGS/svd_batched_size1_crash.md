# Nx.LinAlg.svd crashes on batched matrices with a size-1 dimension

Found 2026-08-17 while building the batched-pinv fix (PR B): the pinv n=1
failure mode turned out to be svd's, not pinv's.

## Minimal repro

```elixir
Nx.LinAlg.svd(Nx.iota({2, 1, 1}, type: :f64))
# ** (ArgumentError) cannot reshape, current shape {1} is not compatible ...
```

Crashes for any batched input where a matrix dimension is 1:
`{2, 1, 1}`, `{2, 1, 2}`, `{2, 2, 1}`. Unbatched `{1, 1}` and `{1, n}`
work fine.

## Root cause (diagnosed 2026-08-17)

Not svd itself: svd delegates to `Nx.LinAlg.eigh`, and the bug is the
degenerate branch of `Nx.LinAlg.BlockEigh.decompose` for n == 1:

    {Nx.take_diagonal(Nx.real(matrix)), Nx.tensor([1], type: matrix.type)}

The eigenvectors are a fresh CONSTANT with no vectorized axes, so the
batch (carried via collapsed vectorized axes) is silently dropped;
`revectorize_result` then cannot reshape 1 element into the batched
shape. Unbatched 1x1 survives only because 1 element happens to fit.

Candidate one-line fix (validated on scratch/eigh-batched-size1):
derive the eigenvector matrix from the input so it inherits the
vectorized axes, e.g. `Nx.multiply(matrix, 0) |> Nx.add(1)`. This
repairs eigh/svd/pinv for all single-batch size-1 shapes and
double-batch eigh. LEFTOVER CORNER: pinv on a double-batch of 1x1
({3,2,1,1}) still fails with transposed batch axes ("cannot broadcast
{3,2,1,1} to {2,3,1,1}") somewhere in the svd composition — separate,
deeper issue.

Not yet filed upstream; draft issue on the fork for review.

## Pinned

Covered indirectly by the `[BUG-PINV-BATCHED]` n=1 pin in
`fuzz_linalg_test.exs` (which should be repointed at svd once the pinv fix
lands upstream).

## Priority

**LOW-MED** — degenerate shapes (1×1/1×n matrices in a batch), clean error,
narrow blast radius; but it blocks completing batched pinv coverage.
