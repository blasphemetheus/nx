# argmax/argmin NaN tie-breaking diverges between BinaryBackend and EXLA

Found 2026-08-17 by the backend-frontier non-finite differential (first
property run).

## Summary

With multiple NaNs present, `Nx.argmax` returns the index of the LAST NaN
on BinaryBackend and the FIRST NaN on EXLA:

```elixir
values = Nx.tensor([1.0, :nan, 2.0, :nan], type: {:f, 64})
Nx.argmax(values)  # BinaryBackend: 3    EXLA/cuda: 1
```

The default tie-break convention is documented as `tie_break: :low`
(first occurrence wins). EXLA honors it under NaN; BinaryBackend does
not. With at most one NaN the backends agree (both point at the NaN).

## Classification

Same convention-consistency family as
[clip_nonfinite_inconsistent.md](clip_nonfinite_inconsistent.md):
BinaryBackend's NaN comparison chain (`x > acc` false for NaN in both
directions) ends up keeping the latest NaN rather than the first.
Cross-backend divergence: code validated on one backend silently picks
different elements on the other (matters whenever argmax indexes into a
parallel structure).

## Pinned

`exla/test/differential_fuzz_test.exs` —
`[DIVERGENCE-ARGMAX-NAN-TIE]` asserts the current divergence; the
agreement property caps inputs at one NaN. Flip to exact agreement when
BinaryBackend is fixed.

## Priority

**MED** — needs multiple NaNs to trigger, but the failure is a silent
cross-backend index difference under the documented tie-break contract.
Not yet filed upstream.
