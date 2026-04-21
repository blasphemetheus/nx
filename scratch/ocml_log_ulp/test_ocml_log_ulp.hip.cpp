// test_ocml_log_ulp.hip.cpp
//
// Empirical fractional-ULP sweep for f32 log.
//
// Build modes:
//   hipcc -O2 -o test_ocml_log_ulp test_ocml_log_ulp.hip.cpp   # GPU
//   g++   -O2 -o test_ocml_log_ulp test_ocml_log_ulp.hip.cpp   # CPU fallback
//
// Under hipcc the device `logf()` dispatches to `__ocml_log_f32` —
// this is what PR #2188 modifies.
//
// Sweep: every normal positive f32 (2^31 - 2^23 ≈ 2.14B values).
// For each `x`:
//   expected_d    = log((double)x)        IEEE 754 f64 log, ~0.5 ULP
//   actual        = device logf(x)
//   fractional_ulp = |actual - expected_d| / ulp(float(expected_d))
//
// Output: max fractional ULP, max integer ULP, mean, top-10 worst.
// Compare max fractional ULP to b-sumner's claimed 2.22 figure.
//
// Runtime on a consumer ROCm GPU: ~30-120s dominated by ULP scoring.
// Runtime on CPU: ~60-180s (single-threaded).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#define HAS_HIP 1
#else
#define HAS_HIP 0
#endif

// ── Portable helpers ──────────────────────────────────────────────

static inline uint32_t to_bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  return u;
}

static inline float from_bits(uint32_t u) {
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

static inline double ulp_f32(float x) {
  if (!std::isfinite(x) || x == 0.0f) return 0.0;
  float n = std::nextafterf(x, std::copysignf(INFINITY, x));
  return std::fabs(static_cast<double>(n) - static_cast<double>(x));
}

static inline double fractional_ulp(float actual, double expected_d) {
  const double diff = std::fabs(static_cast<double>(actual) - expected_d);
  const float expected_f32 = static_cast<float>(expected_d);
  const double u = ulp_f32(expected_f32);
  return u == 0.0 ? 0.0 : diff / u;
}

struct Case {
  float input;
  double expected;
  float actual;
  double frac_ulp;
};

// ── Backend: GPU path under hipcc, CPU fallback otherwise ─────────

#if HAS_HIP
__global__ void log_kernel(const float* in, float* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = logf(in[i]);
}

static void compute_logf(const std::vector<float>& in,
                         std::vector<float>& out) {
  const size_t n = in.size();
  float* dev_in = nullptr;
  float* dev_out = nullptr;
  hipMalloc(&dev_in, n * sizeof(float));
  hipMalloc(&dev_out, n * sizeof(float));
  hipMemcpy(dev_in, in.data(), n * sizeof(float), hipMemcpyHostToDevice);

  const int block = 256;
  const int grid = static_cast<int>((n + block - 1) / block);
  hipLaunchKernelGGL(log_kernel, dim3(grid), dim3(block), 0, 0, dev_in, dev_out,
                     static_cast<int>(n));

  hipMemcpy(out.data(), dev_out, n * sizeof(float), hipMemcpyDeviceToHost);
  hipFree(dev_in);
  hipFree(dev_out);
}

constexpr const char* kImpl = "__ocml_log_f32 (via device logf)";
#else
static void compute_logf(const std::vector<float>& in,
                         std::vector<float>& out) {
  for (size_t i = 0; i < in.size(); ++i) out[i] = std::logf(in[i]);
}

constexpr const char* kImpl = "host libm logf (CPU fallback)";
#endif

// ── Sweep ─────────────────────────────────────────────────────────

int main() {
  constexpr uint32_t kStart = 0x00800000;  // smallest positive normal
  constexpr uint32_t kEnd = 0x7f800000;    // +inf (exclusive)
  constexpr size_t kBatch = 1u << 24;

  std::vector<float> host_in(kBatch);
  std::vector<float> host_out(kBatch);

  Case worst{0, 0, 0, 0};
  std::vector<Case> top;
  top.reserve(64);

  double mean_ulp = 0.0;
  uint64_t total = 0;
  int int_ulp_max = 0;

  for (uint32_t base = kStart; base < kEnd; base += kBatch) {
    const size_t n = std::min<size_t>(kBatch, kEnd - base);
    host_in.resize(n);
    host_out.resize(n);
    for (size_t i = 0; i < n; ++i) {
      host_in[i] = from_bits(base + static_cast<uint32_t>(i));
    }
    compute_logf(host_in, host_out);

    for (size_t i = 0; i < n; ++i) {
      const float x = host_in[i];
      const double expected = std::log(static_cast<double>(x));
      const float actual = host_out[i];

      if (!std::isfinite(expected) || !std::isfinite(actual)) continue;

      const double fu = fractional_ulp(actual, expected);
      const float expected_f32 = static_cast<float>(expected);
      const int iu = static_cast<int>(std::abs(
          static_cast<int64_t>(to_bits(actual)) -
          static_cast<int64_t>(to_bits(expected_f32))));

      mean_ulp += fu;
      ++total;
      if (iu > int_ulp_max) int_ulp_max = iu;
      if (fu > worst.frac_ulp) worst = Case{x, expected, actual, fu};
      if (top.size() < 64 || fu > top.back().frac_ulp) {
        top.push_back(Case{x, expected, actual, fu});
        std::sort(top.begin(), top.end(),
                  [](const Case& a, const Case& b) {
                    return a.frac_ulp > b.frac_ulp;
                  });
        if (top.size() > 64) top.pop_back();
      }
    }

    std::fprintf(stderr, "  swept 0x%08x / 0x%08x, max frac ULP = %.4f\n",
                 base + static_cast<uint32_t>(n), kEnd, worst.frac_ulp);
  }

  mean_ulp /= static_cast<double>(total);

  std::printf("\n=== %s ULP sweep (all normal positive f32) ===\n", kImpl);
  std::printf("tested             : %llu values\n",
              static_cast<unsigned long long>(total));
  std::printf("max fractional ULP : %.6f\n", worst.frac_ulp);
  std::printf("max integer ULP    : %d\n", int_ulp_max);
  std::printf("mean fractional ULP: %.6f\n", mean_ulp);

  std::printf("\nTop 10 worst cases:\n");
  for (size_t i = 0; i < std::min<size_t>(10, top.size()); ++i) {
    const Case& c = top[i];
    std::printf("  input=%a (%g)  expected=%a  actual=%a  frac_ulp=%.4f\n",
                c.input, c.input, static_cast<float>(c.expected), c.actual,
                c.frac_ulp);
  }

  return 0;
}
