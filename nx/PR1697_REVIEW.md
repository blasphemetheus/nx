# PR #1697 Review Response

Tracking Jose's and polvalente's feedback, applying the design recipe.

## The Big Question (Jose)

> I wonder if there is a way to do holistically? Like we remove vectorization
> before grading and then reapply it afterwards, or similar.

### Step 1: Data Definition — What IS the problem?

The expression tree already stores devectorized operations (because `apply_vectorized`
devectorizes before calling the backend). So the tree is naturally devectorized.

The current approach (PR as-is) FIGHTS this by re-vectorizing in `recur_to_grad`:
1. `revectorize_node` adds vectorized axes back to tensor args
2. `adjust_vectorized_args` undoes the re-vectorization for opts (axes, padding, etc.)
3. `maybe_vectorize_grad` vectorizes the gradient seed to match

This is per-op whack-a-mole. Every op with axis-related opts needs a handler.

### The Holistic Alternative: Don't re-vectorize during grad at all

Since the expression tree is already devectorized:
- Skip re-vectorization of args in `recur_to_grad` (remove lines 272-283)
- Skip `adjust_vectorized_args` entirely (remove ~100 lines)
- Skip `maybe_vectorize_grad` (remove ~10 lines)
- Grad clauses receive devectorized tensors — they were designed for this
- In `to_grad`, re-vectorize the final gradient to match the original target

What this removes:
- `adjust_vectorized_args` and ALL per-op clauses (~100 lines)
- `maybe_vectorize_grad` (~10 lines)
- Re-vectorization block in `recur_to_grad` (~15 lines)
- Gather-specific devectorize hack
- All per-op fixes (sort axis, squeeze, broadcast shape, window_sum padding)

What this adds:
- A few lines in `to_grad` to devectorize the target for broadcasting,
  then re-vectorize the result

### Why this should work

The grad clauses were written for devectorized tensors. The expression tree
stores devectorized ops. The `vectorized_names` tracking exists solely to
support re-vectorization — if we skip that, the tracking becomes just a
flag for `to_grad` to know the final output needs re-vectorization.

### Open questions for holistic approach

1. Does `parents_tree` vectorized_names tracking still need to propagate?
   Or can we just check `arg.vectorized_axes` in `to_grad`?

2. Do any grad clauses depend on having vectorized args?
   (They shouldn't — they predate vectorization support)

3. Edge cases: what about ops where the expression tree DOES store
   vectorized info? (gather's iota-prepended indices, etc.)

## polvalente's Code-Level Comments

### 1. `to_grad` vectorization handling (line 230)
> Is this really needed?

WITH holistic approach: Replaced by simpler devectorize-broadcast-revectorize.
WITHOUT holistic: Yes, needed for when res is scalar but arg is vectorized.

### 2. `maybe_vectorize_grad` → `broadcast_vectors` (line 323)
> take a look at Nx.broadcast_vectors, as it might solve the problem more cleanly

WITH holistic approach: Eliminated entirely.
WITHOUT holistic: Could simplify `maybe_vectorize_grad`.

### 3. `window_product` missing (line 393)
> Should window_product be here?

WITH holistic approach: Not needed — no per-op handlers.
WITHOUT holistic: Yes, needs to be added.

### 4. `fft`/`ifft` have axis options (line 415)
> There is an :axis option that should be adjusted. Likewise, fft2 and ifft2 has :axes

WITH holistic approach: Not needed.
WITHOUT holistic: Needs handlers for fft :axis and fft2 :axes.

### 5. `:conv` should raise (line 418)
> This should probably raise with unsupported

WITH holistic approach: Not needed (no passthrough handlers).
WITHOUT holistic: Good idea — raise instead of silently passing through.

### 6. Axes normalization (line 429)
> I'm not entirely sure that axes are normalized by this point.

WITH holistic approach: Not relevant.
WITHOUT holistic: Need to test negative/named axes.

### 7. Gather grad simplification (line 927)
> `g = Nx.devectorize(g, keep_names: false)` (remove if guard)

WITH holistic approach: Gather grad doesn't need special handling at all.
WITHOUT holistic: Good simplification — devectorize is a no-op when already devec.

### 8. Gather grad vectorize simplification (line 950)
> `result = Nx.vectorize(result, vec_axes)` (remove if guard)

Same as above — simplify by removing the guard.

### 9. Restore tests (line 4624)
> Let's restore these tests

Need to check which tests were removed.

## Decision

The holistic approach eliminates almost every per-op comment.
Should we prototype it before addressing individual comments?

## Next Steps

- [ ] Prototype the holistic approach (remove re-vectorization, adjust to_grad)
- [ ] Run existing 28 vectorized grad tests
- [ ] Run full grad test suite
- [ ] If holistic works: rewrite PR, address remaining comments
- [ ] If not: address comments individually on current approach
