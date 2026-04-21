Thanks for the review — the 2.22 ULP figure changes how I'm thinking about this PR.

**Reconciling the numbers.** OCML.md lists `log` f32 at 3 ULP and XLA's ROCm accuracy budget (`kLogF32Budget.rocm_gpu.regular`) is 3, integer-typed — both documented bounds are fully consistent with a measured fractional max of 2.22 (integer ceiling). I initially read those as independent evidence for a ~3 ULP measurement, but on closer look they're just the integer rounding of whatever the true fractional bound is. So your 2.22 figure doesn't contradict either.

The one remaining question on the docs side: OCML.md compares `log` (3) to `log1p` (2). If `log` is actually 2.22 fractional and `log1p` is ~2.0, the gap is real but smaller than the integer rounding suggests, and OCML.md could reasonably be updated to reflect that — but that's a doc cleanup, not a reason to change `__ocml_log_f32`.

**Reframing on my side.** My "3 ULP → 2 ULP" pitch was based on the documented integer bound. If the current fractional max is ~2.22 and log1p is ~2.0, the actual improvement this PR would offer is ~0.22 ULP (roughly 10% relative), not a full ULP. That's a real but modest accuracy win, and it changes the perf-vs-accuracy calculus — especially against your point about satisfied users.

**What would unblock me.** Two measurements I'd like to see before pushing further:

1. **Actual fractional ULP max for the current `__ocml_log_f32`**, to confirm the 2.22 figure end-to-end.
2. **Actual fractional ULP max for the proposed `lnep`-routed path**, to confirm it really hits ≤ 2.0.

For (1), I put together a small self-contained sweep program that tests every normal positive f32 against an f64 reference and reports max fractional ULP: **[GIST_URL_HERE]**.

```
hipcc -O2 -o test_ocml_log_ulp test_ocml_log_ulp.hip.cpp
./test_ocml_log_ulp
```

Methodology is validated: the CPU fallback (`g++` against glibc `logf`) reports max fractional ULP = 0.818, matching glibc's known near-correctly-rounded behavior. Whatever the HIP run reports is a trustworthy exhaustive-sweep maximum, not a methodology artifact.

For (2), I can re-run the same sweep on a build of this PR once (1) is resolved.

**On perf.** If the lnep path shows a real regression on the workloads that matter, one option is to gate it behind an opt-in macro (e.g. `OCML_LOG_ACCURATE`) so the current `v_log_f32`-based fast path stays default. That keeps your users on today's behavior and lets applications that need tighter accuracy opt in. Happy to restructure the PR that way.

**Possible outcomes:**

- (1) confirms ≤ 2.22 **and** (2) shows no improvement or a perf regression → PR not worth it, I'll close.
- (1) is higher than 2.22 or (2) shows a real improvement → PR stands, possibly behind the opt-in flag.
- (1) confirms ~2.22 and (2) shows 2.0 → a 0.22 ULP win at some perf cost — worth a conversation about whether it's worth shipping.

Either way, measuring is cheap; let's do that before deciding.
