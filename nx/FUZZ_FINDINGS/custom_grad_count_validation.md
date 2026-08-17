# custom_grad count mismatches: extras silently dropped, shortfalls leak an internal error

Found 2026-08-17 while targeting Grad's dark reconciliation arms.

## Summary

`custom_grad(expr, inputs, fun)` documents that `fun` must return "a list
of tensors that map directly to the inputs". The count is not validated
symmetrically:

| fun returns | inputs | behavior |
|---|---|---|
| non-list | 1 | proper raise: "custom_grad/3 must return a list of tensors..." |
| 2 entries | 1 | **extra silently dropped** — no error, first entry used |
| 0 entries | 1 | internal invariant leak: `ERROR! grad for metadata returned 0 entries but traversed 1` |
| 1 entry | 2 | same internal leak (1 vs 2) |

## Why it matters

The silent-drop case is the dangerous one: a user who miscounts inputs
(easy with containers) gets a *plausible* gradient with no diagnostic —
the extra entries they carefully computed are discarded. The shortfall
case fails loudly but with an internal "ERROR!" message that doesn't
mention custom_grad at all.

## Fix direction

Validate `length(returned) == length(inputs)` at the custom_grad
metadata grad rule and raise the existing user-facing message for both
directions.

## Pinned

`fuzz_darklines_test.exs` — `[BUG-CUSTOM-GRAD-COUNT]` pins both
behaviors; flip to the friendly ArgumentError/RuntimeError when fixed.

## Priority

**MED** — silent wrong-gradient hazard on a documented API contract,
trivial validation fix.
