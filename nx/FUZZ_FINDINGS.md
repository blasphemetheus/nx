# Nx Fuzz Testing Findings

## Summary

442 property tests + 203 explicit tests across 11 test files.
7 bugs found (5 BinaryBackend, 1 Nx.linspace, 1 validation ordering). 10 skipped tests documenting bugs.

---

## Bug 1: BinaryBackend crashes on float overflow

**Affected ops:** `exp`, `expm1`, `sinh`, `cosh`, `sigmoid`

**Reproduce:**
```elixir
Nx.exp(Nx.tensor(1000.0))
# ** (ArithmeticError) bad argument in arithmetic expression

Nx.sinh(Nx.tensor(1000.0))
# ** (ArithmeticError) bad argument in arithmetic expression

Nx.sigmoid(Nx.tensor(1.0e6))
# ** (ArithmeticError) bad argument in arithmetic expression
```

**Expected:** Should return `Inf` (or `1.0` for sigmoid) instead of crashing.

**Root cause:** BinaryBackend delegates to Erlang's `:math` module which raises
`ArithmeticError` on overflow. The backend doesn't rescue and convert to Inf.

**Note:** `Nx.exp(Nx.Constants.infinity())` correctly returns `Inf` — the special
constant path works, but the overflow-from-finite-value path doesn't.

**EXLA behavior:** EXLA correctly returns Inf for these inputs.

**Severity:** Medium. Any user whose data contains large values will hit this.
Common in ML when logits or activations explode during training.

**Fix approach:** Wrap `:math` calls in `try/rescue` in `binary_to_binary/4`
and return the appropriate Inf/NaN. Or use a pre-check: if the input exceeds
a threshold, return Inf directly without calling `:math`.

**Files to change:** `nx/lib/nx/binary_backend.ex` around line 2461
(`binary_to_binary/4` and the lambda that calls `:math` functions).

---

## Bug 2: BinaryBackend crashes on domain errors

**Affected ops:** `asin`, `acos` (inputs outside [-1, 1]),
`acosh` (inputs < 1), `atanh` (inputs outside (-1, 1))

**Reproduce:**
```elixir
Nx.asin(Nx.tensor(2.0))
# ** (ArithmeticError) bad argument in arithmetic expression

Nx.acos(Nx.tensor(2.0))
# ** (ArithmeticError) bad argument in arithmetic expression

Nx.acosh(Nx.tensor(0.5))
# ** (ArithmeticError) bad argument in arithmetic expression

Nx.atanh(Nx.tensor(2.0))
# ** (ArithmeticError) bad argument in arithmetic expression
```

**Expected:** Should return `NaN` instead of crashing. This is the IEEE 754
standard behavior for domain errors.

**Root cause:** Same as Bug 1 — `:math` module raises instead of returning NaN.

**EXLA behavior:** EXLA correctly returns NaN for these inputs.

**Severity:** Medium. Users doing inverse trig on unconstrained data will crash.

**Fix approach:** Same as Bug 1 — rescue ArithmeticError and return NaN.

**Files to change:** Same as Bug 1 (`nx/lib/nx/binary_backend.ex`).

---

## Bug 3: Nx.divide by zero crashes instead of returning Inf

**Reproduce:**
```elixir
Nx.divide(Nx.tensor(1.0), Nx.tensor(0.0))
# ** (ArithmeticError) bad argument in arithmetic expression

Nx.divide(Nx.tensor(1.0), Nx.tensor(-0.0))
# ** (ArithmeticError) bad argument in arithmetic expression
```

**Expected:** `Inf` for `1.0/0.0`, `-Inf` for `1.0/-0.0`, `NaN` for `0.0/0.0`
per IEEE 754.

**Root cause:** Same `:math` delegation issue.

**EXLA behavior:** EXLA correctly returns Inf/-Inf/NaN.

**Severity:** High. Division by zero is common in ML (e.g., normalizing by
variance which can be zero, reciprocal of small values).

**Fix approach:** Same as Bugs 1-2.

---

## Bug 4: window_scatter_max/min crashes on f64 tensors

*Renumbered from Bug 3 in previous version.*

**Affected ops:** `window_scatter_max`, `window_scatter_min`

**Reproduce:**
```elixir
t = Nx.iota({6}, type: :f64)
s = Nx.iota({3}, type: :f64)
init = Nx.tensor(0.0, type: :f64)
Nx.window_scatter_max(t, s, init, {2}, strides: [2], padding: :valid)
# ** (ArgumentError) unexpected size for tensor data, expected 384 bits got: 288 bits
```

**Expected:** Should work the same as f32 (which works correctly).

**Root cause:** Binary size calculation in `window_scatter` doesn't account
for f64's 8-byte elements correctly. The scatter result binary is constructed
with the wrong size.

**EXLA behavior:** Not tested for this specific case.

**Severity:** Medium. Any user using f64 with window_scatter hits this.

**Fix approach:** Find the binary size calculation in the window_scatter
implementation in `binary_backend.ex` and fix the byte-width handling.
Likely a hardcoded `4` (f32 bytes) instead of using `Nx.Type.size(type)`.

**Files to change:** `nx/lib/nx/binary_backend.ex`, search for
`window_scatter` implementation.

---

## Bug 4: Nx.iota rejects zero dimensions

**Reproduce:**
```elixir
Nx.iota({0})
# ** (ArgumentError) invalid dimension in axis 0 found in shape.
#    Each dimension must be a positive integer, got 0 in shape {0}

Nx.iota({3, 0, 4})
# ** (ArgumentError) invalid dimension in axis 1 found in shape.
```

**Expected behavior:** Unclear — this may be intentional. NumPy and JAX allow
zero-dimension tensors (`np.arange(0)` returns an empty array). XLA also
supports empty tensors.

**Assessment:** Design decision, not necessarily a bug. But worth discussing
whether Nx should support empty tensors for consistency with other frameworks.

**Severity:** Low. Empty tensors are rare in practice.

---

## Bug 5: BinaryBackend crashes on Nx.slice of scalar tensor

**Reproduce:**
```elixir
Nx.slice(Nx.tensor(42), [], [])
# ** (ArgumentError) errors were found at the given arguments:
#   * 1st argument: not a nonempty list
```

**Expected:** Should return the scalar tensor unchanged (no-op).

**Root cause:** `BinaryBackend.bin_slice/7` calls `hd(strides)` and `hd(start_indices)`
on line 1855, which crashes on empty lists when the tensor is scalar (rank 0).

**Severity:** Low. Slicing a scalar is unusual, but the validation in
`Nx.Shape.slice` passes, so the backend should handle it.

**Fix approach:** Add a scalar guard in `bin_slice/7`: if `start_indices == []`,
return the data unchanged.

**Files to change:** `nx/lib/nx/binary_backend.ex` around line 1852.

---

## Bug 6: Nx.linspace crashes with n=1

**Reproduce:**
```elixir
Nx.linspace(0, 10, n: 1)
# ** (ArithmeticError) bad argument in arithmetic expression
```

**Expected:** Should return a single-element tensor containing the start value,
like NumPy's `np.linspace(0, 10, 1)` which returns `[0.0]`.

**Root cause:** When `endpoint: true` (default), `divisor = n - 1 = 0`.
Then `Nx.divide(stop - start, 0)` triggers a divide-by-zero in BinaryBackend.
Line 16841 in `nx.ex`: `divisor = n - 1`.

**Severity:** Medium. `n=1` is a reasonable input (e.g., when generating a
single interpolation point).

**Fix approach:** Special-case `n == 1` before the divisor calculation:
return `Nx.broadcast(start, {1})` directly.

**Files to change:** `nx/lib/nx.ex` around line 16839.

---

## Bug 7: Nx.gather gives wrong error on scalar indices

**Reproduce:**
```elixir
Nx.gather(Nx.iota({3}), Nx.tensor(0))
# ** (ArgumentError) errors were found at the given arguments:
#   * 1st argument: out of range
```

**Expected:** Should raise `"expected indices rank to be at least 1, got: 0"`.

**Root cause:** `indexed_axes` (line 8027 in nx.ex) calls
`elem(indices.shape, tuple_size(indices.shape) - 1)` which is
`elem({}, -1)` — Erlang raises before the Nx validation fires.
The scalar check in `Nx.Shape.gather` (line 1641) never runs because
`indexed_axes` is called first.

**Severity:** Low. The function still rejects invalid input, just with
an unhelpful error message.

**Fix approach:** Move the `indices_shape == {}` check to before the
`indexed_axes` call in `Nx.gather/3`, or guard `indexed_axes` against
scalar indices.

**Files to change:** `nx/lib/nx.ex` around line 14605.

---

## Not a bug: Nx.select broadcasting

`Nx.select` with non-scalar predicate uses the predicate's shape as the output
shape, not a three-way broadcast. `on_true` and `on_false` must be broadcastable
TO the pred shape. This differs from NumPy's `np.where` which broadcasts all three.

This is intentional per the source code (line 7298-7303 of `nx.ex`).

---

## Not a bug: Nx.slice clamps out-of-bounds

`Nx.slice(tensor, [out_of_bounds], [length])` clamps the start index to the last
valid position instead of raising. This matches XLA behavior.

---

## Test coverage summary

| File | Properties | Tests | Skipped | Coverage |
|---|---|---|---|---|
| fuzz_test.exs | 140 | 2 | 6 | Core ops, all categories |
| fuzz_linalg_test.exs | 25 | 0 | 0 | QR, Cholesky, LU, SVD, eigh, solve, invert |
| fuzz_random_values_test.exs | 27 | 1 | 1 | Random floats, mathematical invariants |
| fuzz_broadcast_test.exs | 23 | 0 | 0 | Mismatched broadcastable shapes |
| fuzz_complex_test.exs | 29 | 0 | 0 | c64/c128 arithmetic, properties |
| fuzz_errors_test.exs | 15 | 7 | 0 | Invalid inputs, error messages |
| fuzz_vectorized_test.exs | 29 | 0 | 0 | Vectorized axes preservation |
| fuzz_types_test.exs | 18 | 9 | 0 | f16/bf16, type promotion, high-rank |
| fuzz_grad_test.exs | 61 | 2 | 0 | Finite difference gradient verification |
| fuzz_differential_test.exs | 60 | 0 | 0 | BinaryBackend vs EXLA cross-backend |
| fuzz_edge_cases_test.exs | 15 | 182 | 3 | Tier 4: source-derived boundary tests |
| fuzz_edge_cases2_test.exs | 9 | 79 | 0 | Tier 4: defn, diagonal, reduce, vectorize, type |
| fuzz_edge_cases3_test.exs | 7 | 58 | 0 | Tier 4: diff, eye, clip, FFT, LinAlg, equivalences |
| fuzz_edge_cases4_test.exs | 0 | 72 | 0 | Tier 4: complex types, vectorized+edge combos |
| fuzz_edge_cases5_test.exs | 0 | 72 | 0 | Tier 4: defn hooks, tokens, nested JIT, compile |
| fuzz_edge_cases6_test.exs | 5 | 57 | 0 | Tier 4: NaN/Inf propagation, broadcast+boundary combos |
| fuzz_sequence_test.exs | 32 | 3 | 0 | Tier 5: random/binary/JIT sequences, vectorized chains, backend/resource monitoring |
| fuzz_sequence2_test.exs | 1 | 43 | 0 | Tier 5: defn control flow, grad chains, concurrency, exotic types, high-rank, vectorized binary |
| exla/fuzz_edge_cases_test.exs | 0 | 48 | 0 | Tier 4: cross-backend edge cases (EXLA vs Binary) |
| exla/fuzz_sequence_test.exs | 12 | 0 | 0 | Tier 5: cross-backend op chain comparison (EXLA vs Binary) |
| **Total** | **508** | **635** | **10** | |
