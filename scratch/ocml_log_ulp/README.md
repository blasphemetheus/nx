# Empirical ULP sweep for `__ocml_log_f32`

Self-contained fractional-ULP measurement over every normal positive
f32 input (~2.14B values). Compiles two ways from a single source:

| Build | Command | What it measures |
|---|---|---|
| **ROCm (primary)** | `hipcc -O2 -o test_ocml_log_ulp test_ocml_log_ulp.hip.cpp` | `__ocml_log_f32` via device `logf` |
| **CPU fallback** | `g++ -O2 -o test_ocml_log_ulp test_ocml_log_ulp.hip.cpp -lm` | host libm `logf` (methodology validation) |

CMake also supports the CPU build: `cmake -B build && cmake --build build`.

## Methodology

For each input `x` in `[0x00800000, 0x7f800000)` (every positive
normal f32):

```
expected_d     = log((double)x)          // IEEE 754 f64 log, ~0.5 ULP
actual         = device logf(x)           // __ocml_log_f32 under hipcc
fractional_ulp = |actual - expected_d| / ulp(float(expected_d))
```

Reports:
- max fractional ULP (comparable to `b-sumner`'s "2.22 ULP")
- max integer ULP (comparable to XLA's `kLogF32Budget.rocm_gpu.regular=3`)
- mean fractional ULP
- top-10 worst-case inputs with `%a` hex-float formatting

The f64 reference is not correctly-rounded but its error (~0.5 ULP
f64) is negligible relative to a 2-3 f32-ULP bound. For a crlibm-
level reference, link against MPFR or CORE-MATH.

## Why this PR needs it

[ROCm/llvm-project#2188](https://github.com/ROCm/llvm-project/pull/2188)
routes `__ocml_log_f32` through the `lnep` polynomial path. The PR's
premise (3 ULP → 2 ULP) is backed by:

- [AMD's OCML.md](https://github.com/ROCm/llvm-project/blob/amd-staging/amd/device-libs/doc/OCML.md)
  listing `log` f32 at **3 ULP**, `log1p` f32 at **2 ULP**.
- [openxla/xla#39048](https://github.com/openxla/xla/pull/39048) setting
  `kLogF32Budget.rocm_gpu.regular = 3` — by an AMD engineer, based on
  "OCML.md as well as empirical measurements."

A reviewer stated current impl hits 2.22 ULP maximum. This program
is the minimum-controversy reproducer.

## Validation of methodology

On a CPU build against glibc `logf` this reports:
```
tested             : 2130706432 values
max fractional ULP : 0.817664
max integer ULP    : 1
mean fractional ULP: 0.250009
```

That matches glibc's known ~correctly-rounded f32 log behavior, which
validates the framework.

## Expected ROCm output format

```
=== __ocml_log_f32 (via device logf) ULP sweep (all normal positive f32) ===
tested             : 2130706432 values
max fractional ULP : X.XX
max integer ULP    : N
mean fractional ULP: Y.YY

Top 10 worst cases:
  input=0x1.234567p+0 (1.137...)  expected=0x1...p+0  actual=0x1...p+0  frac_ulp=X.XX
  ...
```

- If `max fractional ULP ≤ 2.22`: reviewer's claim confirmed, PR's
  premise may need revising.
- If `max fractional ULP > 2.22`: PR's premise stands; reviewer's
  measurement was on a subset or used a different oracle.
- If `max fractional ULP ≈ 3`: aligns with OCML.md's documented bound.

## Files

- `test_ocml_log_ulp.hip.cpp` — the dual-mode test program.
- `CMakeLists.txt` — CPU build configuration.
- `.clangd` — editor-only config for Nix local dev (ignore when
  submitting).
