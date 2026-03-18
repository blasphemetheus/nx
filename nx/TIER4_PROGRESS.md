# Tier 4: LLM-Guided Edge Case Generation — Progress

## Status: COMPLETE

All planned Tier 4 items have been implemented, including NablaFuzz/FreeFuzz pattern mining.

## Function Groups Analyzed

### Priority 1 — Complex validation logic
- [x] Slice/gather/indexed ops
- [x] Window ops
- [x] Reduction ops with axes
- [x] Custom reduce (Nx.reduce/4, Nx.window_reduce/5)

### Priority 2 — Shape manipulation boundaries
- [x] Reshape/transpose/squeeze/flatten
- [x] Pad/concatenate/stack
- [x] Broadcast/tile
- [x] Diagonal ops (take_diagonal, make_diagonal, put_diagonal)

### Priority 3 — Type and numerical boundaries
- [x] Sort/argsort, Top_k, Linspace, Reverse
- [x] Bitcast, Dot/outer
- [x] Type conversions (as_type, merge, normalize!, from_binary)
- [x] Vectorize/devectorize
- [x] Diff, Eye, Clip, Select, Covariance, FFT

### Priority 4 — Equivalence / differential tests
- [x] 31 algebraic identity properties across all files

### Priority 5 — Defn constructs
- [x] While loops (0/1/many iterations, element iteration)
- [x] Cond (each branch, boundary values)
- [x] Grad (identity, square, abs, relu, chain rule)
- [x] Hooks (named, with callback, override, containers, in chains)
- [x] Tokens (ordering, side-effect hooks, attach_token)
- [x] Nested JIT (on_conflict: :reuse)
- [x] Compile (template matching, incompatible rejection)
- [x] Hooks in while loops (multiple calls, 0-iteration case)
- [x] Hooks in cond (true/false branch selection)

### Priority 6 — Complex types
- [x] Supported operations (construction, real/imag, conjugate, abs, phase, arithmetic, exp, sum, reshape)
- [x] Rejected operations (sort, argsort, reduce_max/min, argmax/min, greater/less, clip, window_max/min, bitcast, erf)
- [x] Algebraic properties (conjugate involution, z*conj(z)=|z|², decomposition roundtrip)
- [x] Type promotion (f32+c64→c64, f64+c64→c128, s32+c64→c64)

### Priority 7 — Vectorized + edge case combinations
- [x] Vectorized slice, take, concatenate, pad, reshape, sum, sort, reverse
- [x] Vectorized as_type, abs, add broadcast, multiply same-axes
- [x] Vectorized dot, transpose
- [x] Vectorized argmax, reduce_max, all, any

### Priority 8 — Cross-backend edge cases (EXLA)
- [x] 48 tests comparing BinaryBackend vs EXLA for boundary operations
- [x] Covers: slice, put_slice, gather, take, take_along_axis, indexed_add/put
- [x] Covers: pad, window_sum/max/min/mean, sort, argsort, reverse, diff
- [x] Covers: clip, select, reshape, squeeze, flatten, tile, concatenate, stack
- [x] Covers: diagonal, norm, determinant, invert, solve
- [x] Covers: as_type, bitcast, FFT, covariance

---

## Bugs Found (Tier 4): 3 new bugs (total project: 7)

### Bug 5: BinaryBackend crashes on Nx.slice of scalar tensor
- `Nx.slice(Nx.tensor(42), [], [])` -> `:erlang.hd([])`
- File: `nx/lib/nx/binary_backend.ex:1852`

### Bug 6: Nx.linspace crashes with n=1
- `Nx.linspace(0, 10, n: 1)` -> ArithmeticError (divide by zero)
- File: `nx/lib/nx.ex:16841`

### Bug 7: Nx.gather gives wrong error on scalar indices
- `Nx.gather(Nx.iota({3}), Nx.tensor(0))` -> Erlang error instead of Nx message
- File: `nx/lib/nx.ex:14605`

---

## Test Files

| File | Properties | Tests | Skipped | Coverage |
|------|-----------|-------|---------|----------|
| fuzz_edge_cases_test.exs | 15 | 182 | 3 | slice/gather/indexed, window, pad, reshape, squeeze, sort, top_k, linspace, reverse, bitcast, dot, broadcast, tile, stack, concatenate, new_axis, f64 patterns, 15 equivalences |
| fuzz_edge_cases2_test.exs | 9 | 79 | 0 | diagonal, reduce, window_reduce, vectorize, to_batched, type conversion, defn while/cond/grad, 9 equivalences |
| fuzz_edge_cases3_test.exs | 7 | 58 | 0 | diff, eye, clip, select, covariance, FFT, LinAlg, 7 equivalences |
| fuzz_edge_cases4_test.exs | 0 | 72 | 0 | complex types (supported/rejected/properties), type promotion, vectorized+edge combos |
| fuzz_edge_cases5_test.exs | 0 | 72 | 0 | defn hooks, tokens, nested JIT, compile, hooks in while/cond |
| fuzz_edge_cases6_test.exs | 5 | 57 | 0 | NaN/Inf propagation through chains, broadcast at boundary-op intersections, multi-type special values |
| exla/fuzz_edge_cases_test.exs | 0 | 48 | 0 | cross-backend EXLA vs BinaryBackend for all edge cases |
| **Total** | **36** | **568** | **3** | |

## Grand Total (All Tiers)

| Metric | Count |
|--------|-------|
| Test files | 15 (10 Nx + 2 EXLA + 3 Tier 4 continued) |
| Properties | 258 |
| Tests | 445 (Nx) + 48 (EXLA edge) = 493+ |
| Total assertions | ~700+ |
| Bugs found | 7 |
| Skipped (documenting bugs) | 5 |
