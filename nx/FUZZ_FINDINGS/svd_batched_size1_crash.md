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

## Classification

Separate from [pinv_batched_forward_crash.md](pinv_batched_forward_crash.md)
(whose n>=2 modes are fixed by the pinv batched-dot PR); this svd bug is
what remains behind pinv's n=1 mode. Likely in `Nx.LinAlg.SVD`'s batched
reshape plumbing when min(m, n) == 1. Not yet filed upstream.

## Pinned

Covered indirectly by the `[BUG-PINV-BATCHED]` n=1 pin in
`fuzz_linalg_test.exs` (which should be repointed at svd once the pinv fix
lands upstream).

## Priority

**LOW-MED** — degenerate shapes (1×1/1×n matrices in a batch), clean error,
narrow blast radius; but it blocks completing batched pinv coverage.
