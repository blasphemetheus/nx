Thanks for both numbers — 2.22 ULP for `log` and 0.58 ULP for `log1p`. Taken together they reframe how I'm thinking about this PR.

**Reconciling the docs.** OCML.md lists `log` f32 at 3 ULP and `log1p` f32 at 2 ULP; XLA's `kLogF32Budget.rocm_gpu.regular` is 3 — all consistent with a measured 2.22 (documented bounds are integer-valued contracts).

**What log1p's 0.58 says about the PR's real target.** `log1p` beats its documented 2-ULP bound by ~3.4× in practice. The absolute measured gap between `log` (2.22) and `log1p` (0.58) is ~1.6 ULP. My PR's stated "3 ULP → 2 ULP" target was based on the documented integer bounds, not measured behavior — so if `lnep` routing for `log` lands near log1p's 0.58 the real improvement is much larger than the PR title advertises; if it lands near 2.0, it's much smaller. Only measuring will tell us which.

(Caveat: `log` and `log1p` are evaluated over different input ranges, so the 1.6 ULP "gap" is an analogy, not a guarantee the gap closes on the full `log` domain.)

**On "satisfied users" — a concrete unhappy user.** XLA's accuracy budgets have a ROCm-specific carve-out in [`accuracy_budget.h`](https://github.com/openxla/xla/blob/main/xla/codegen/intrinsic/accuracy/accuracy_budget.h):

```c
constexpr AccuracyBudget kLogF32Budget = {
    /*cpu=*/     {/*regular=*/1, /*subnormal=*/0},
    /*gpu=*/     {/*regular=*/1, /*subnormal=*/0},   // NVIDIA / generic GPU path
    /*rocm_gpu=*/UlpBudget{/*regular=*/3, /*subnormal=*/0},  // AMD carve-out
};
```

The `gpu = 1` line is the contract the NVIDIA path satisfies; the `rocm_gpu = 3` line exists specifically because ocml `log` is looser. That's a downstream user who is demonstrably *not* satisfied with the current behavior — they've had to carry an AMD-specific exception to keep their tests green. Tightening ocml `log` toward the 1-ULP contract NVIDIA meets would let this carve-out go away.

**What I'd like to measure.** Three fractional-ULP sweeps on current ocml:

1. `__ocml_log_f32` — confirms the 2.22 figure end-to-end.
2. `__ocml_log1p_f32` — confirms the 0.58 figure with the same methodology.
3. The PR's `lnep`-routed `__ocml_log_f32` — tells us where the new path actually lands.

Self-contained sweep program at **[GIST_URL_HERE]** — tests every normal positive f32 against an f64 reference and reports max fractional ULP. CPU fallback (`g++` against glibc `logf`) reports 0.818 max fractional ULP, which matches glibc's known near-correctly-rounded behavior — enough to trust the harness.

**Two asks on the perf side.**

1. If you can share approximate cycle counts or relative perf on the current `v_log_f32`-fastpath vs `lnep` path (or let me know which benchmark to run), I can size the accuracy-vs-perf tradeoff concretely.
2. If `lnep` routing is measurably slower on workloads that matter, I can gate it behind an opt-in macro (e.g. `OCML_LOG_ACCURATE`, paralleling the existing `ocml-accuracy` knobs) so today's fast path stays default and applications that need tighter accuracy opt in. Happy to restructure the PR that way if it's acceptable.

**Where I see this landing, by measurement (3):**

- `~0.58` (log1p-level accuracy) with small perf cost → merge as default. OCML.md's log row can be separately updated to reflect measured values.
- `~0.58` with real perf cost → merge behind an opt-in flag.
- `~2.0` (just clears the 2-ULP doc target) → the ~0.2 ULP real improvement isn't worth a perf cost; close.
- Doesn't beat 2.22 meaningfully → close.

The measurement side is inexpensive; I'd rather have numbers in hand than argue framings.
