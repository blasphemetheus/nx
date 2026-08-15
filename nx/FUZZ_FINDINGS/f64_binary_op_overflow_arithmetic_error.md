# f64 binary-op overflow raises ArithmeticError (or returns NaN) in BinaryBackend

Found 2026-08-14 by `fuzz_float_edge_test.exs` (FUZZ_ROADMAP T1.1, first run of
the bit-pattern generator).

## Summary

Binary arithmetic on f64 tensors crashes with `ArithmeticError` when the exact
result overflows f64 — IEEE 754 requires returning ±Infinity. `Nx.pow` returns
NaN instead of Inf for the same class. f32 is unaffected.

## Minimal repro

```elixir
max = Nx.from_binary(<<0x7FEFFFFFFFFFFFFF::64-native>>, {:f, 64})  # 1.7976931348623157e308

Nx.add(max, max)       # ** (ArithmeticError) bad argument in arithmetic expression
Nx.multiply(max, 2.0)  # ** (ArithmeticError)
Nx.multiply(max, max)  # ** (ArithmeticError)
Nx.divide(max, 0.5)    # ** (ArithmeticError)
Nx.pow(max, 2)         # #Nx.Tensor<f64 NaN>   (wrong value, no crash)
```

## Expected vs observed

Expected: `:infinity` (f64 +Inf) in all five cases.
Observed: `ArithmeticError` from `Complex` arithmetic / BEAM float overflow for
add/subtract/multiply/divide; silent `NaN` for `pow`.

## Inconsistency map (why this went unnoticed)

| Case | Result |
|---|---|
| `Nx.multiply(f32_max, f32_max)` | ✅ `:infinity` — BEAM float (f64) holds the intermediate; encode clamps |
| `Nx.exp(Nx.tensor(1000.0, type: {:f,64}))` | ✅ `:infinity` — unary path has overflow handling |
| `Nx.divide(max, 1.66e-113)` (scalar divisor) | ✅ `:infinity` — scalar path handled |
| `Nx.divide(max_t, tiny_t)` (tensor divisor) | ❌ `ArithmeticError` |
| `Nx.add(max, max)` | ❌ `ArithmeticError` |

## Root cause hypothesis

BEAM floats are f64 with no Inf/NaN representation — overflow in native
arithmetic raises. `Nx.BinaryBackend.element_wise_bin_op/4` routes through
`Complex` arithmetic (`Complex.divide/2` etc.) with no rescue, unlike the unary
path which has overflow fallbacks. f32 escapes because the f64 intermediate
can represent f32-overflowing values and `scalar_to_binary` clamps on encode.

## Classification

Same family as the fork/fix/binary-backend-ieee754 branch (commit 3ade4c44,
March 2026), which fixed the *unary* overflow class (`exp(1000)` etc.) and
division-by-zero signs — but binary-op overflow was not covered there. If that
branch gets revived for upstreaming, this belongs in it; the fix shape is the
same `rescue ArithmeticError -> ±infinity by sign` fallback, applied in
`element_wise_bin_op`, plus a value fix for `pow`.

## Pinned

`fuzz_float_edge_test.exs` — "known bugs: f64 overflow" describe block,
`[BUG-F64-OVERFLOW]` pins (flip `assert_raise` to value assertions when fixed).
The NaN-propagation property in the same file caps finite magnitudes into
[1e-3, 1e3] specifically to avoid tripping this bug — remove the cap when fixed.

## Priority

**HIGH** — real correctness bug in elementary ops (`add`, `multiply`) on the
reference backend, crash not wrong-value, and v1.0-relevant: BinaryBackend is
the semantic reference every other backend is judged against.
