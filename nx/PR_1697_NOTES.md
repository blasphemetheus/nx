# PR #1697 Notes — Support vectorize/devectorize inside gradients

## Summary of changes (fork/fix/1533-vectorized-grad-v2 branch)

Starting from PR tip (f5d4b9f7), added 11 commits:

### Code fixes
1. **Reverted broadcast_vectors in to_grad** — Nx.broadcast handles vectorized axes natively
2. **Added adjust_vectorized_args for window_scatter** — passthrough (raises not yet supported)
3. **Fixed reduce_g** — use Nx.axes(x) consistently for inner-shape coordinate space
4. **Fixed assert_all_close in test helpers** — devectorize result before comparing, supports vectorized tensors
5. **Added adjust_vectorized_args for slice/put_slice** — drop leading vectorized entries from start_indices/lengths/strides
6. **Added adjust_vectorized_args for indexed_add/put** — passthrough to let grad rule handle
7. **Fixed indexed_add/indexed_put grad rules** — devectorize g, compute, revectorize (same pattern as gather)

### Test changes
- All 17 review comments addressed
- 332 total vectorization tests (was ~45 originally)
- 17 skipped documenting known limitations
- Simplified test names per polvalente's request
- Restored composed grad tests to original file position
- Added comprehensive edge case coverage across all op categories

## Test results

332 tests, 0 failures, 17 skipped
Full nx test suite: 1350 doctests + 1313 tests, 0 failures

## Remaining 17 skipped tests

### Backend bug (2)
- partial axis reduction axis 1 on 2D inner — BinaryBackend.unary_broadcast crash
- partial axis reduction axis 1 on 3D inner — same

### Design/holistic approach needed (5)
- reshape+vectorize inside grad — grad devectorizes inputs, reshape sees wrong shape
- non-vectorized input, vectorized output — axes leak into gradient
- rename vectorized axes inside grad — original names lost
- devectorize then compute then return scalar — broadcast shape mismatch
- chained devectorize/vectorize with computation — lists.duplicate crash

### Need deep grad rule rewrite (5)
- window_scatter_max — source shape mismatch (needs full devec/revec in grad rule)
- window_scatter_min — same
- QR grad — cannot vectorize tensor of rank 0
- cholesky grad — lists.duplicate crash
- cumulative_sum — axis name collision (forward-pass issue, no grad rule)

### Mixed vectorized axes (2)
- dot with mixed vectorized axes — unbroadcast shape mismatch
- three different vectorized axes — broadcast failure

### Other (3)
- take_along_axis — parameter expression error
- conv — explicitly raises (by design)
- edge case where same name changes meaning — pre-existing skip

## Bugs found and fixed
1. assert_all_close in test helpers — didn't support vectorized tensors
2. reduce_g coordinate space mismatch — used devectorized axes with inner shapes
3. slice/put_slice missing vectorized args adjustment
4. indexed_add/put needed devectorize/revectorize in grad rule

## Holistic approach (tried, not adopted)
Commit 4e2980f4 prototyped removing all per-op handlers. Results: 96.6% tests pass
but 9 mixed-vectorization tests fail (non-vectorized target in vectorized context).
Since mixed-vec is needed, per-op approach was kept.
