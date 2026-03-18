# Nx Fuzz Testing Methodology

## Approach

Property-based fuzz testing using StreamData to generate random valid
tensor inputs and verify Nx operations don't crash, return correct shapes,
and return correct types.

## Generators

### Core generators (in `test/nx/fuzz_test.exs`)

- `tensor_shape()` — random shapes rank 0-3 with edge cases (scalar, singleton, empty)
- `tensor_dim()` — dimension sizes weighted toward edge cases: 0, 1, small, medium
- `tensor_type()` / `float_type()` / `numeric_type()` — type selectors
- `non_empty_shape()` — shapes where all dims > 0
- `tensor(shape, type)` — generates a tensor via `Nx.iota`
- `float_tensor()` — float tensor with random shape
- `valid_axes(shape)` — random valid axis subsets

### Domain-restricted generators

Float ops are split by domain requirements:
- **Safe**: sin, cos, tan, sigmoid, cbrt, erf, etc. — any float input
- **Positive**: log, sqrt, rsqrt — needs > 0
- **Unit domain**: asin, acos — needs [-1, 1]
- **Open unit**: atanh, erf_inv — needs (-1, 1)
- **GE one**: acosh — needs >= 1
- **Moderate**: asinh — needs bounded range to avoid overflow
- **Overflow-prone**: exp, sinh, cosh, expm1 — large inputs overflow

## Properties tested

### Per operation
1. **Crash oracle**: operation doesn't raise on valid inputs
2. **Shape oracle**: output shape matches expected shape
3. **Type oracle**: output type matches expected type

### Categories covered
- Unary element-wise (20+ ops across all domain categories)
- Binary element-wise (add, subtract, multiply, min, max, divide)
- Reductions (sum, product, reduce_max, reduce_min) — full and per-axis
- Shape ops (reshape, transpose, squeeze, new_axis)
- Type coercion (as_type across all numeric types)
- Broadcasting (same shape, scalar to shape)
- Creation (iota, eye, broadcast)
- Concatenation and stacking
- Comparison ops (equal, not_equal, greater, etc.)
- Empty tensor handling

## Findings

### Finding 1: BinaryBackend crashes on float overflow instead of returning Inf

**Affected ops**: `exp`, `expm1`, `sinh`, `cosh`
**Behavior**: `Nx.exp(Nx.tensor(710.0))` raises `ArithmeticError` instead of
returning `Nx.tensor(:infinity)`.
**Root cause**: BinaryBackend delegates to Erlang's `:math` module which raises
on overflow. The backend should catch this and return Inf/NaN.
**Severity**: Medium — affects any user whose data contains large values.
**Workaround**: Clip inputs before applying these ops.
**EXLA behavior**: EXLA correctly returns Inf for these inputs.

### Finding 2: BinaryBackend crashes on domain errors instead of returning NaN

**Affected ops**: `asin`, `acos` (with inputs outside [-1, 1]),
`acosh` (with inputs < 1), `atanh` (with inputs outside (-1, 1))
**Behavior**: Raises `ArithmeticError` instead of returning NaN.
**Root cause**: Same as Finding 1 — `:math` module raises.
**Severity**: Medium — silent crash instead of propagating NaN.

### Finding 3: Nx.iota rejects zero dimensions

**Behavior**: `Nx.iota({0})` raises `ArgumentError: invalid dimension in axis 0`.
**Assessment**: This may be intentional — Nx docs say dimensions must be positive.
But other tensor libraries (NumPy, JAX) allow zero-dimension tensors. Worth
discussing whether Nx should support them.

## Running the tests

```bash
cd nx
mix test test/nx/fuzz_test.exs
```

## Extending

To add fuzz tests for a new op:
1. Determine the op's domain requirements (what inputs are valid?)
2. Add it to the appropriate `@unary_float_*` or other category list
3. If it has unique domain requirements, add a new category
4. The property test is auto-generated from the category lists

For new categories (window ops, linalg, etc.):
1. Add a new `describe` block
2. Use the generators to produce valid inputs
3. Write properties: doesn't crash, correct shape, correct type
