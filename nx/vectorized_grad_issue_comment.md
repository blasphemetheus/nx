I've been digging into this and have three approaches on my fork — one works, two don't. Want to check which direction y'all prefer before I open a PR.

## What's Actually Broken

When the backward pass hits a vectorized node, [`recur_to_grad`](https://github.com/elixir-nx/nx/blob/c0cd8ea9/nx/lib/nx/defn/grad.ex#L244-L270) re-vectorizes the tensor args ([line 253](https://github.com/elixir-nx/nx/blob/c0cd8ea9/nx/lib/nx/defn/grad.ex#L253-L262)) but passes opts through unchanged. The opts (axes, padding, strides) still have axis indices from the devectorized shape, but the tensors now have their vectorized axes back. Mismatch.

```
Forward pass devectorizes x for the expression graph:
  vectorized[batch: 2] shape {3}  →  plain shape {2, 3}
                                       axis:  0  1
  sum is recorded with axes: [1] (pointing at the 3-wide dim)

Backward pass re-vectorizes x for the grad clause:
  plain shape {2, 3}  →  vectorized[batch: 2] shape {3}
                                                axis:  0
  but opts still says axes: [1] — that axis doesn't exist anymore!
  The 3-wide dim is now axis 0, not axis 1.
```

So anything in [`reduce_g`](https://github.com/elixir-nx/nx/blob/c0cd8ea9/nx/lib/nx/defn/grad.ex#L1585), [`grad_broadcast`](https://github.com/elixir-nx/nx/blob/c0cd8ea9/nx/lib/nx/defn/grad.ex#L1564), or any grad clause that reads `opts[:axes]` gets the wrong axis.

## What I Tried

### Approach A: Fix each grad clause individually

Go through each affected grad clause and subtract `vec_offset = length(x.vectorized_axes)` from the axis indices in opts. This works for the clauses I fixed (reduce_g covers sum/mean), but it's ~20 clauses total and every future grad clause would need to be vectorization-aware too.

Branch: [`fix/1533-approach-a`](https://github.com/blasphemetheus/nx/tree/fix/1533-approach-a) (partial — only reduce_g + to_grad + maybe_vectorize_grad)

### Approach B: Just don't re-vectorize

Idea: keep everything devectorized during the backward pass, re-vectorize only at the end in [`to_grad`](https://github.com/elixir-nx/nx/blob/c0cd8ea9/nx/lib/nx/defn/grad.ex#L210). Opts and tensors would be in the same space, no mismatch.

**Doesn't work.** The re-vectorization is load-bearing. [`unbroadcast`](https://github.com/elixir-nx/nx/blob/c0cd8ea9/nx/lib/nx/defn/grad.ex#L1557-L1562) and `grad_broadcast` need vectorized axes hidden so they don't sum over batch dims. Without that:

```elixir
grad(scalar_y, fn y -> Nx.add(vectorized_x, y) end)
# returns 6.0 instead of vectorized[batch: 2] [3.0, 3.0]
# because unbroadcast sums over ALL dims including batch
```

Writeup with the exact code changes and why they fail: [`VECTORIZED_GRAD_APPROACHES.md`](https://github.com/blasphemetheus/nx/blob/docs/1533-notes/nx/VECTORIZED_GRAD_APPROACHES.md)

### Approach C: Centralized opts adjustment (this one works)

Keep re-vectorization, but add one function — `adjust_vectorized_args(op, args, vec_offset)` — right after re-vectorization in [`recur_to_grad`](https://github.com/elixir-nx/nx/blob/c0cd8ea9/nx/lib/nx/defn/grad.ex#L244). It pattern-matches on the op name and adjusts the right args:

- **Keyword `opts[:axes]`**: sum, product, reduce_max/min, gather, sort — subtract `vec_offset` from each axis index
- **Plain axis lists**: squeeze, transpose, broadcast — same subtraction
- **Single integer axis**: stack, concatenate — subtract `vec_offset`
- **dot**: all 6 axis lists (contract + batch for both sides)
- **pad**: drop first `vec_offset` entries from `padding_config`
- **window ops**: drop leading entries from window_dimensions, strides, padding, dilations

Also adds:
- `maybe_vectorize_grad/3` — broadcasts `g` to match the re-vectorized `ans` when `g` isn't vectorized yet
- Updated `to_grad/4` — re-vectorizes final gradient for vectorized targets

**All 2563 existing tests pass + 5 new vectorized gradient tests** (sum, mean, product, reduce_max, composed `sum(x*x)`).

Branch: [`fix/1533-approach-c`](https://github.com/blasphemetheus/nx/tree/fix/1533-approach-c)

## Questions

1. **Is centralizing in `adjust_vectorized_args` the right call?** New grad clauses just need a line added there instead of each clause being vectorization-aware. Tradeoff is one function that knows every op's arg structure.

2. **Would it be better to fix this earlier** — like adjusting opts when nodes are stored in [`parents_tree`](https://github.com/elixir-nx/nx/blob/c0cd8ea9/nx/lib/nx/defn/grad.ex#L90), or in the expression-building code?

3. **Am I missing ops?** `conv` is stubbed as a no-op right now. Happy to add more tests for specific ops.

Background notes and a glossary of the jargon in this space: [`VECTORIZED_GRAD_TODO.md`](https://github.com/blasphemetheus/nx/blob/docs/1533-notes/nx/VECTORIZED_GRAD_TODO.md)
