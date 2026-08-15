# Unary op non-finite handling: crashes (floor/ceil/round/atanh) and wrong values (tanh, sign)

Found 2026-08-14 during FUZZ_ROADMAP T1.1 (probe while retrofitting the
bit-pattern generator). Same family as [f64_binary_op_overflow_arithmetic_error.md](f64_binary_op_overflow_arithmetic_error.md):
BinaryBackend's non-finite dispatch is incomplete, per-op.

## Crash class

```elixir
Nx.floor(Nx.tensor(:nan))          # ** crash — should be NaN
Nx.ceil(Nx.tensor(:infinity))      # ** crash — should be Inf
Nx.round(Nx.tensor(:neg_infinity)) # ** crash — should be -Inf
Nx.atanh(Nx.tensor(:nan))          # ** crash — should be NaN
```

floor/ceil/round must be the identity on NaN and ±Inf. atanh(NaN) is NaN;
atanh(±Inf) is a domain error → NaN.

## Wrong-value class

| Expression | Observed | IEEE-correct |
|---|---|---|
| `Nx.tanh(:infinity)` | `:nan` | `1.0` |
| `Nx.tanh(:neg_infinity)` | `:nan` | `-1.0` |
| `Nx.sign(:nan)` | `1.0` | `:nan` |
| `Nx.sign(:neg_infinity)` | `1.0` | `-1.0` |

`sign(-Inf) == 1.0` is the most alarming row — a sign error on a signed input.
Contrast with the ops that handle limits correctly on current main:
`sinh(±Inf) = ±Inf`, `cosh(±Inf) = +Inf`, `atan(±Inf) = ±π/2`,
`exp(-Inf) = 0.0`, `expm1(-Inf) = -1.0`, `abs`/`negate` correct.

## Classification

Root cause family: BinaryBackend decodes non-finites to atoms
(`:nan`/`:infinity`/`:neg_infinity`) and each unary op needs an explicit
non-finite clause; floor/ceil/round/atanh lack one entirely (atom hits
arithmetic → crash), tanh and sign have one that returns the wrong value.
The fork/fix/binary-backend-ieee754 branch (3ade4c44) built an
`ieee754_fallback` table for exactly this pattern — these ops belong in it.

## Pinned

`fuzz_float_edge_test.exs` — "unary limits" property asserts the
known-correct rows; `[BUG-UNARY-NONFINITE]` pins assert the current broken
behavior with flip-when-fixed comments.

## Priority

**HIGH** — `sign(-Inf) = 1.0` and `tanh(Inf) = NaN` are silent wrong values in
common activation-function territory on the reference backend.
