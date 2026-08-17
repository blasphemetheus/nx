# Nx.clip handles NaN differently per argument position, contradicting min/max

Found 2026-08-17 by the non-finite convention probes (first pass of the
convention-frontier fuzzing).

## Summary

`clip(x, lo, hi)` is definable as `min(max(x, lo), hi)`, and Nx's own
min/max propagate NaN. Under that composition, a NaN in ANY position must
yield NaN. Instead, each argument position currently does something
different:

| Expression | Observed | min/max composition says |
|---|---|---|
| `clip(NaN, 0, 2)` | `0.0` (the LOWER bound!) | NaN |
| `clip(1, NaN, 2)` | `2.0` (the upper bound) | NaN |
| `clip(1, 0, NaN)` | `NaN` | NaN |

The first row is the worst: a NaN *input value* comes out as a legitimate-
looking in-range number, silently laundering NaN out of a pipeline —
exactly where you'd want NaN to surface (clip is ubiquitous in training
loops as a guard).

## Classification

BinaryBackend comparison chain in clip presumably branches on `<`/`>`
comparisons that are all false for NaN, falling through to an arbitrary
arm per position. Convention-consistency class (the rest of the min/max
family — reduce/cumulative/arg — is coherent; see
`fuzz_nonfinite_convention_test.exs`'s moduledoc for the established map).

Cross-backend note: EXLA/XLA clamps via select chains that typically
propagate NaN — so this likely also diverges across backends (untested;
worth a differential probe before filing).

## Pinned

`fuzz_nonfinite_convention_test.exs` — `[BUG-CLIP-NONFINITE]` pins all
three positions plus a finite-bounds control asserting the min/max
composition. Flip to NaN assertions when fixed.

## Priority

**MED-HIGH** — silent wrong value in a guard-style op; narrow trigger
(NaN present) but that is precisely when clip's output matters. Not yet
filed upstream.
