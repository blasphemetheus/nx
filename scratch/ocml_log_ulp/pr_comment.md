Thanks for both numbers — 2.22 ULP for `log` and 0.58 ULP for `log1p`. Taken together they sharpen the PR body's own hedge ("*possibly closer to 1 ULP if lnep's actual bound is tighter than documented*"): the 0.58 figure confirms lnep is much tighter than its documented 2-ULP bound in practice.

**Reconciling the docs.** OCML.md listing `log` f32 at 3 ULP, `log1p` at 2 ULP, and XLA's `kLogF32Budget.rocm_gpu.regular = 3` — all consistent with measured 2.22 (documented bounds are integer-valued contracts).

**What the 0.58 figure changes for this PR.** `log1p` beats its documented bound by ~3.4× in practice. The absolute measured gap between `log` (2.22) and `log1p` (0.58) is ~1.6 ULP. My PR's stated "3 → 2 ULP" target used the documented integer bounds; if `lnep` routing for `log` lands near 0.58 (matching log1p's real behavior), the actual improvement is ~1.6 ULP — much more than the PR title advertises. If it lands near 2.0, it's ~0.2 ULP. Only measuring will tell.

(Caveat: `log` and `log1p` are evaluated over different input ranges — near 1 is where log1p's polynomial is strongest — so the 1.6 ULP gap is an intuition pump, not a guarantee the gap closes on the full `log` domain.)

**On the XLA carve-out I mentioned in the PR body.** I'm not trying to relitigate that — you've seen the PR body. I'll just note that the asymmetry it creates (ROCm `log` is the only backend where XLA has to carry a 2-ULP-above-parity budget exception) is exactly the thing this PR aims to close, and is what I mean by "worth doing" even against some perf cost.

**What I'd like to measure.** The PR already includes a FileCheck ISA test that asserts the new code path takes `lnep` (`v_frexp_mant_f32` + `v_fma_f32`, no `v_log_f32`) — so the code-path side is covered. What's missing is the numerical-ULP side. Three fractional-ULP sweeps on current ocml would settle it:

1. `__ocml_log_f32` — confirms the 2.22 figure end-to-end.
2. `__ocml_log1p_f32` — confirms the 0.58 figure with the same methodology.
3. The PR's `lnep`-routed `__ocml_log_f32` — the crux.

Self-contained sweep program at **https://gist.github.com/blasphemetheus/77197d676530280398afd943be618b5a** — tests every normal positive f32 against an f64 reference and reports max fractional ULP. CPU fallback (`g++` against glibc `logf`) reports 0.818 max fractional ULP, which matches glibc's known near-correctly-rounded behavior — enough to trust the harness.

**Two asks on the perf side.**

1. If you can share approximate cycle counts or relative perf for `v_log_f32`-fastpath vs `lnep` path (or point me at a benchmark to run), I can size the accuracy-vs-perf tradeoff concretely.
2. If `lnep` routing is measurably slower on workloads that matter, I can gate it behind an opt-in macro (e.g. `OCML_LOG_ACCURATE`, paralleling the existing `ocml-accuracy` knobs) so today's fast path stays default and applications that need tighter accuracy opt in. Happy to restructure the PR that way if it's acceptable.

**Tightening the merge threshold from the PR body.** My PR body's verification plan was "≤ 2 ULP merge / = 3 ULP adjust / > 3 ULP revert." Given your perf signal, I'd narrow that:

- Measurement (3) ≈ 0.58 (log1p-level) with small perf cost → merge as default. OCML.md's log row could be updated in a follow-up.
- ≈ 0.58 with real perf cost → merge behind an opt-in flag.
- ≈ 2.0 (clears the 2-ULP doc target but no real gain over current 2.22) → the ~0.2 ULP real improvement isn't worth a perf cost; close.
- Doesn't beat 2.22 meaningfully → close.

The measurement side is inexpensive; I'd rather have numbers in hand than argue framings.
