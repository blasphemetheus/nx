# Nx Fuzz Testing Findings

## Summary

288 property tests + 10 explicit tests across 7 test files.
4 bugs found, all in BinaryBackend. 7 skipped tests documenting bugs.

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

## Bug 3: window_scatter_max/min crashes on f64 tensors

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
| **Total** | **288** | **10** | **7** | |
