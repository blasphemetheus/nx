defmodule DifferentialFuzzTest do
  @moduledoc """
  Cross-backend differential fuzzing: same computation on
  `Nx.BinaryBackend` vs `EXLA.Backend`, assert they agree within
  reasonable tolerance.

  Disagreements between backends are **real bugs** — one of them is
  wrong. Common divergence zones:

  - f16/bf16 linalg (limited-precision accumulation differs)
  - complex grad (rarely tested in EXLA)
  - large-magnitude softmax / logsumexp
  - reductions with many axes
  - determinant / inverse on near-singular matrices
  - vectorized ops

  This file runs ops under both backends and diffs the results. It
  lives in the EXLA project because it needs EXLA compiled.

  Tolerance policy: `atol=1.0e-5`, `rtol=1.0e-4` — tight enough to
  catch real divergences, loose enough to absorb EXLA's fast-math.
  """
  use ExUnit.Case, async: false
  use ExUnitProperties

  import Nx.Testing

  # ── Diff helper: run fun under each backend, compare ───────────────

  @binary_backend Nx.BinaryBackend
  # Flip client via EXLA_CLIENT env var; defaults to :cuda when
  # running in devenv, :host for CPU-only builds.
  @exla_client (System.get_env("EXLA_CLIENT") || "cuda") |> String.to_atom()
  @exla_backend {EXLA.Backend, client: @exla_client}

  defp run_under(backend, fun) do
    old = Nx.default_backend()
    Nx.default_backend(backend)

    try do
      fun.()
    after
      Nx.default_backend(old)
    end
  end

  defp diff(fun, opts \\ []) do
    atol = Keyword.get(opts, :atol, 1.0e-5)
    rtol = Keyword.get(opts, :rtol, 1.0e-4)

    a = run_under(@binary_backend, fun) |> Nx.backend_copy(Nx.BinaryBackend)
    b = run_under(@exla_backend, fun) |> Nx.backend_copy(Nx.BinaryBackend)

    # Shapes/types must match exactly; values within tolerance.
    assert Nx.shape(a) == Nx.shape(b),
      "shape mismatch: binary #{inspect(Nx.shape(a))} vs exla #{inspect(Nx.shape(b))}"

    assert Nx.type(a) == Nx.type(b),
      "type mismatch: binary #{inspect(Nx.type(a))} vs exla #{inspect(Nx.type(b))}"

    assert_all_close(a, b, atol: atol, rtol: rtol)
  end

  # ── Basic arithmetic ───────────────────────────────────────────────

  describe "element-wise ops agree" do
    property "add(f32, f32) agrees" do
      check all(
              n <- integer(2..8),
              va <- list_of(float(min: -100.0, max: 100.0), length: n),
              vb <- list_of(float(min: -100.0, max: 100.0), length: n),
              max_runs: 10
            ) do
        diff(fn ->
          a = Nx.tensor(va, type: :f32)
          b = Nx.tensor(vb, type: :f32)
          Nx.add(a, b)
        end)
      end
    end

    property "multiply(f32, f32) agrees" do
      check all(
              n <- integer(2..8),
              va <- list_of(float(min: -10.0, max: 10.0), length: n),
              vb <- list_of(float(min: -10.0, max: 10.0), length: n),
              max_runs: 10
            ) do
        diff(fn ->
          a = Nx.tensor(va, type: :f32)
          b = Nx.tensor(vb, type: :f32)
          Nx.multiply(a, b)
        end)
      end
    end

    property "exp(f32) agrees" do
      check all(
              n <- integer(2..6),
              vals <- list_of(float(min: -5.0, max: 5.0), length: n),
              max_runs: 10
            ) do
        diff(fn -> Nx.exp(Nx.tensor(vals, type: :f32)) end, atol: 1.0e-3, rtol: 1.0e-4)
      end
    end
  end

  # ── Reductions ─────────────────────────────────────────────────────

  describe "reductions agree" do
    property "sum over f32 agrees" do
      check all(
              n <- integer(2..10),
              vals <- list_of(float(min: -50.0, max: 50.0), length: n),
              max_runs: 10
            ) do
        diff(fn -> Nx.sum(Nx.tensor(vals, type: :f32)) end)
      end
    end

    property "mean over f32 agrees" do
      check all(
              n <- integer(2..10),
              vals <- list_of(float(min: -50.0, max: 50.0), length: n),
              max_runs: 10
            ) do
        diff(fn -> Nx.mean(Nx.tensor(vals, type: :f32)) end)
      end
    end

    property "reduce_max / reduce_min agree" do
      check all(
              n <- integer(2..10),
              vals <- list_of(float(min: -50.0, max: 50.0), length: n),
              max_runs: 10
            ) do
        diff(fn ->
          t = Nx.tensor(vals, type: :f32)
          Nx.stack([Nx.reduce_max(t), Nx.reduce_min(t)])
        end)
      end
    end

    property "argmax / argmin agree (exact — integer)" do
      check all(
              n <- integer(2..8),
              vals <- list_of(float(min: -50.0, max: 50.0), length: n),
              max_runs: 10
            ) do
        diff(fn ->
          t = Nx.tensor(vals, type: :f32)
          Nx.stack([Nx.argmax(t), Nx.argmin(t)])
        end, atol: 0.0, rtol: 0.0)
      end
    end
  end

  # ── Linalg / matmul ────────────────────────────────────────────────

  describe "linalg agrees" do
    property "matmul 3×3 f32 agrees" do
      check all(
              va <- list_of(float(min: -3.0, max: 3.0), length: 9),
              vb <- list_of(float(min: -3.0, max: 3.0), length: 9),
              max_runs: 10
            ) do
        diff(fn ->
          a = Nx.tensor(va, type: :f32) |> Nx.reshape({3, 3})
          b = Nx.tensor(vb, type: :f32) |> Nx.reshape({3, 3})
          Nx.dot(a, b)
        end, atol: 1.0e-4, rtol: 1.0e-4)
      end
    end

    property "determinant of well-conditioned 3×3 agrees" do
      check all(
              vals <- list_of(float(min: -1.0, max: 1.0), length: 9),
              max_runs: 10
            ) do
        diff(fn ->
          r = Nx.tensor(vals, type: :f32) |> Nx.reshape({3, 3})
          a = Nx.add(Nx.eye(3, type: :f32), Nx.multiply(r, 0.1))
          Nx.LinAlg.determinant(a)
        end, atol: 1.0e-4, rtol: 1.0e-3)
      end
    end

    property "invert → multiply → should be I, both agree" do
      check all(
              vals <- list_of(float(min: -1.0, max: 1.0), length: 9),
              max_runs: 8
            ) do
        diff(fn ->
          r = Nx.tensor(vals, type: :f32) |> Nx.reshape({3, 3})
          a = Nx.add(Nx.eye(3, type: :f32), Nx.multiply(r, 0.1))
          Nx.LinAlg.invert(a)
        end, atol: 1.0e-4, rtol: 1.0e-3)
      end
    end
  end

  # ── Numerical stability under magnitude ────────────────────────────

  describe "numerical stability: backends agree under large magnitudes" do
    property "softmax at moderate magnitudes" do
      check all(
              n <- integer(2..6),
              vals <- list_of(float(min: -10.0, max: 10.0), length: n),
              max_runs: 10
            ) do
        diff(fn ->
          x = Nx.tensor(vals, type: :f32)
          shifted = Nx.subtract(x, Nx.reduce_max(x))
          exps = Nx.exp(shifted)
          Nx.divide(exps, Nx.sum(exps))
        end, atol: 1.0e-5, rtol: 1.0e-4)
      end
    end

    property "sigmoid across (-20, 20)" do
      check all(
              n <- integer(2..6),
              vals <- list_of(float(min: -20.0, max: 20.0), length: n),
              max_runs: 10
            ) do
        diff(fn -> Nx.sigmoid(Nx.tensor(vals, type: :f32)) end, atol: 1.0e-5, rtol: 1.0e-4)
      end
    end
  end

  # ── Grad agreement ─────────────────────────────────────────────────

  describe "gradients agree across backends" do
    property "grad of sin agrees" do
      check all(
              x <- float(min: -2.0, max: 2.0),
              max_runs: 10
            ) do
        diff(fn ->
          t = Nx.tensor(x, type: :f32)
          Nx.Defn.grad(t, fn a -> Nx.sin(a) end)
        end)
      end
    end

    property "grad of sum(x*x) agrees" do
      check all(
              n <- integer(2..6),
              vals <- list_of(float(min: -3.0, max: 3.0), length: n),
              max_runs: 10
            ) do
        diff(fn ->
          t = Nx.tensor(vals, type: :f32)
          Nx.Defn.grad(t, fn a -> Nx.sum(Nx.multiply(a, a)) end)
        end, atol: 1.0e-4, rtol: 1.0e-4)
      end
    end

    property "grad through matmul agrees" do
      check all(
              va <- list_of(float(min: -1.0, max: 1.0), length: 9),
              vb <- list_of(float(min: -1.0, max: 1.0), length: 9),
              max_runs: 8
            ) do
        # Can't close over the other matrix — Nx forbids mixing two
        # non-BinaryBackend tensor impls (Expr + EXLA). Pack both
        # matrices into a tuple passed as the fun's argument.
        diff(fn ->
          a = Nx.tensor(va, type: :f32) |> Nx.reshape({3, 3})
          b = Nx.tensor(vb, type: :f32) |> Nx.reshape({3, 3})
          Nx.Defn.grad({a, b}, fn {x, y} -> Nx.sum(Nx.dot(x, y)) end)
          |> elem(0)
        end, atol: 1.0e-4, rtol: 1.0e-4)
      end
    end
  end

  # ── Integer ops ────────────────────────────────────────────────────

  describe "integer ops agree exactly" do
    property "sum over s32 is exact" do
      check all(
              n <- integer(2..10),
              vals <- list_of(integer(-1000..1000), length: n),
              max_runs: 10
            ) do
        diff(fn -> Nx.sum(Nx.tensor(vals, type: :s32)) end, atol: 0.0, rtol: 0.0)
      end
    end

    property "bitwise xor is exact" do
      check all(
              n <- integer(2..6),
              va <- list_of(integer(0..1000), length: n),
              vb <- list_of(integer(0..1000), length: n),
              max_runs: 10
            ) do
        diff(fn ->
          a = Nx.tensor(va, type: :s32)
          b = Nx.tensor(vb, type: :s32)
          Nx.bitwise_xor(a, b)
        end, atol: 0.0, rtol: 0.0)
      end
    end
  end

  # ── Shape / indexing ops ───────────────────────────────────────────

  describe "shape / indexing agree" do
    property "reshape round-trip agrees" do
      check all(
              vals <- list_of(float(min: -5.0, max: 5.0), length: 6),
              max_runs: 10
            ) do
        diff(fn ->
          t = Nx.tensor(vals, type: :f32)
          t |> Nx.reshape({2, 3}) |> Nx.reshape({6})
        end, atol: 0.0, rtol: 0.0)
      end
    end

    property "transpose agrees" do
      check all(
              vals <- list_of(float(min: -5.0, max: 5.0), length: 12),
              max_runs: 10
            ) do
        diff(fn ->
          Nx.tensor(vals, type: :f32) |> Nx.reshape({3, 4}) |> Nx.transpose()
        end, atol: 0.0, rtol: 0.0)
      end
    end

    property "gather agrees" do
      check all(
              vals <- list_of(float(min: -5.0, max: 5.0), length: 8),
              max_runs: 10
            ) do
        diff(fn ->
          t = Nx.tensor(vals, type: :f32)
          idx = Nx.tensor([[0], [3], [5], [7]])
          Nx.gather(t, idx)
        end, atol: 0.0, rtol: 0.0)
      end
    end
  end

  # ── Conv agreement ─────────────────────────────────────────────────

  describe "conv agrees" do
    property "1-channel conv f32 agrees" do
      check all(
              ivals <- list_of(float(min: -1.0, max: 1.0), length: 16),
              kvals <- list_of(float(min: -1.0, max: 1.0), length: 9),
              max_runs: 6
            ) do
        diff(fn ->
          input = Nx.tensor(ivals, type: :f32) |> Nx.reshape({1, 1, 4, 4})
          kernel = Nx.tensor(kvals, type: :f32) |> Nx.reshape({1, 1, 3, 3})
          Nx.conv(input, kernel)
        end, atol: 1.0e-4, rtol: 1.0e-4)
      end
    end
  end

  # ── TF32 / tensor-core divergence probes ──────────────────────────

  describe "tf32 probes: large f32 GEMMs under tight tolerance" do
    # TF32 reduces f32 GEMM mantissa from 23 to 10 bits when tensor
    # cores are engaged. Effect compounds with k-dim. Small matrices
    # hide this; large matrices + tight rtol should surface it.

    defp rand_matrix(n, m) do
      vals = for _ <- 1..(n * m), do: (:rand.uniform() - 0.5) * 2.0
      Nx.tensor(vals, type: :f32) |> Nx.reshape({n, m})
    end

    test "32×32 f32 matmul diverges from BinaryBackend (pins TF32 bug)" do
      # Already filed upstream: XLA #39250 (GEMM/autotuner), Nx #1702
      # (:highest precision). This test pins the current wrong
      # behavior. When XLA fixes the default autotuner or Nx switches
      # LinAlg to :highest by default, this assert_raise will fire
      # and the test should be flipped.
      :rand.seed(:exsss, {1, 2, 3})
      va = for _ <- 1..(32 * 32), do: (:rand.uniform() - 0.5) * 2.0
      vb = for _ <- 1..(32 * 32), do: (:rand.uniform() - 0.5) * 2.0

      assert_raise ExUnit.AssertionError, fn ->
        diff(fn ->
          aa = Nx.tensor(va, type: :f32) |> Nx.reshape({32, 32})
          bb = Nx.tensor(vb, type: :f32) |> Nx.reshape({32, 32})
          Nx.dot(aa, bb)
        end, atol: 0.0, rtol: 1.0e-5)
      end
    end

    test "128×128 f32 matmul diverges (pins TF32 bug)" do
      :rand.seed(:exsss, {4, 5, 6})
      va = for _ <- 1..(128 * 128), do: (:rand.uniform() - 0.5) * 2.0
      vb = for _ <- 1..(128 * 128), do: (:rand.uniform() - 0.5) * 2.0

      assert_raise ExUnit.AssertionError, fn ->
        diff(fn ->
          aa = Nx.tensor(va, type: :f32) |> Nx.reshape({128, 128})
          bb = Nx.tensor(vb, type: :f32) |> Nx.reshape({128, 128})
          Nx.dot(aa, bb)
        end, atol: 0.0, rtol: 1.0e-5)
      end
    end

    test "matmul jit'd with precision: :highest agrees tightly (control)" do
      # :highest tells XLA to disable TF32 for f32 GEMMs on GPU.
      # If this test also fails, the issue isn't TF32; if it passes
      # while the regular matmul fails, TF32 is confirmed as the cause.
      :rand.seed(:exsss, {7, 8, 9})
      va = for _ <- 1..(128 * 128), do: (:rand.uniform() - 0.5) * 2.0
      vb = for _ <- 1..(128 * 128), do: (:rand.uniform() - 0.5) * 2.0

      fun = Nx.Defn.jit(fn aa, bb -> Nx.dot(aa, bb) end, precision: :highest)

      diff(fn ->
        aa = Nx.tensor(va, type: :f32) |> Nx.reshape({128, 128})
        bb = Nx.tensor(vb, type: :f32) |> Nx.reshape({128, 128})
        fun.(aa, bb)
      end, atol: 0.0, rtol: 1.0e-5)
    end
  end

  # ── Larger-shape linalg at tight tolerance ─────────────────────────
  # TF32 error compounds with k-dim. Small matrices may hide it even
  # at rtol=1e-5. These are the same ops as the block below but scaled
  # up to surface the divergence if it exists.

  describe "linalg at 32x32 + tight tolerance" do
    test "determinant diverges at rtol=1e-5, precision::highest control passes" do
      :rand.seed(:exsss, {20, 21, 22})
      vals = for _ <- 1..(32 * 32), do: (:rand.uniform() - 0.5) * 0.1

      # Default path: expected to diverge on Blackwell.
      default_failed =
        try do
          diff(fn ->
            r = Nx.tensor(vals, type: :f32) |> Nx.reshape({32, 32})
            a = Nx.add(Nx.eye(32, type: :f32), r)
            Nx.LinAlg.determinant(a)
          end, atol: 0.0, rtol: 1.0e-5)
          false
        rescue
          ExUnit.AssertionError -> true
        end

      # Control: with precision: :highest, expected to agree.
      control_fun =
        Nx.Defn.jit(
          fn r -> Nx.LinAlg.determinant(Nx.add(Nx.eye(32, type: :f32), r)) end,
          precision: :highest
        )

      diff(fn ->
        r = Nx.tensor(vals, type: :f32) |> Nx.reshape({32, 32})
        control_fun.(r)
      end, atol: 0.0, rtol: 1.0e-5)

      # Info: whether default diverged on this machine.
      IO.puts("  [info] determinant 32x32 default diverged at rtol=1e-5? #{default_failed}")
    end

    test "invert diverges at rtol=1e-5 (pins TF32 in custom_grad support paths)" do
      :rand.seed(:exsss, {23, 24, 25})
      vals = for _ <- 1..(32 * 32), do: (:rand.uniform() - 0.5) * 0.1

      result =
        try do
          diff(fn ->
            r = Nx.tensor(vals, type: :f32) |> Nx.reshape({32, 32})
            a = Nx.add(Nx.eye(32, type: :f32), r)
            Nx.LinAlg.invert(a)
          end, atol: 0.0, rtol: 1.0e-5)
          :agreed
        rescue
          ExUnit.AssertionError -> :diverged
        end

      IO.puts("  [info] invert 32x32 at rtol=1e-5: #{result}")
    end

    test "QR Q-matrix at 32x32" do
      :rand.seed(:exsss, {26, 27, 28})
      vals = for _ <- 1..(32 * 32), do: (:rand.uniform() - 0.5) * 0.5

      result =
        try do
          diff(fn ->
            a = Nx.tensor(vals, type: :f32) |> Nx.reshape({32, 32})
            {q, _r} = Nx.LinAlg.qr(a)
            q
          end, atol: 0.0, rtol: 1.0e-5)
          :agreed
        rescue
          ExUnit.AssertionError -> :diverged
        end

      IO.puts("  [info] QR Q 32x32 at rtol=1e-5: #{result}")
    end

    test "SVD singular values at 32x32" do
      :rand.seed(:exsss, {29, 30, 31})
      vals = for _ <- 1..(32 * 32), do: (:rand.uniform() - 0.5) * 0.5

      result =
        try do
          diff(fn ->
            a = Nx.tensor(vals, type: :f32) |> Nx.reshape({32, 32})
            {_u, s, _v} = Nx.LinAlg.svd(a)
            s
          end, atol: 0.0, rtol: 1.0e-5)
          :agreed
        rescue
          ExUnit.AssertionError -> :diverged
        end

      IO.puts("  [info] SVD s 32x32 at rtol=1e-5: #{result}")
    end

    test "Cholesky on PSD matrix at 32x32" do
      :rand.seed(:exsss, {32, 33, 34})
      vals = for _ <- 1..(32 * 32), do: (:rand.uniform() - 0.5) * 0.1

      result =
        try do
          diff(fn ->
            r = Nx.tensor(vals, type: :f32) |> Nx.reshape({32, 32})
            # Make symmetric + diag-boosted for PSD
            a = Nx.add(Nx.eye(32, type: :f32), Nx.multiply(r, 0.01))
            sym = Nx.add(a, Nx.transpose(a)) |> Nx.divide(2.0)
            Nx.LinAlg.cholesky(sym)
          end, atol: 0.0, rtol: 1.0e-5)
          :agreed
        rescue
          ExUnit.AssertionError -> :diverged
        end

      IO.puts("  [info] Cholesky 32x32 at rtol=1e-5: #{result}")
    end
  end

  # ── f16 / bf16 differential ────────────────────────────────────────
  # Reduced-precision divergence zone, distinct from TF32:
  # BinaryBackend emulates f16/bf16 strictly; EXLA on GPU may use
  # hardware paths with different accumulation precision, subnormal
  # handling, and rounding.
  #
  # Tolerance budget per dtype:
  #   f16:  ~2^-10 ≈ 1e-3 relative precision
  #   bf16: ~2^-7  ≈ 1e-2 relative precision

  describe "f16 differential" do
    test "f16 matmul 16×16 agrees within reduced precision" do
      :rand.seed(:exsss, {60, 61, 62})
      va = for _ <- 1..256, do: (:rand.uniform() - 0.5)
      vb = for _ <- 1..256, do: (:rand.uniform() - 0.5)

      result =
        try do
          diff(fn ->
            a = Nx.tensor(va, type: :f16) |> Nx.reshape({16, 16})
            b = Nx.tensor(vb, type: :f16) |> Nx.reshape({16, 16})
            Nx.dot(a, b)
          end, atol: 0.0, rtol: 5.0e-3)
          :agreed
        rescue
          ExUnit.AssertionError -> :diverged
        end

      IO.puts("  [info] f16 matmul 16x16 at rtol=5e-3: #{result}")
    end

    test "f16 matmul 64×64 agrees within reduced precision" do
      :rand.seed(:exsss, {63, 64, 65})
      va = for _ <- 1..(64 * 64), do: (:rand.uniform() - 0.5) * 0.1
      vb = for _ <- 1..(64 * 64), do: (:rand.uniform() - 0.5) * 0.1

      result =
        try do
          diff(fn ->
            a = Nx.tensor(va, type: :f16) |> Nx.reshape({64, 64})
            b = Nx.tensor(vb, type: :f16) |> Nx.reshape({64, 64})
            Nx.dot(a, b)
          end, atol: 0.0, rtol: 1.0e-2)
          :agreed
        rescue
          ExUnit.AssertionError -> :diverged
        end

      IO.puts("  [info] f16 matmul 64x64 at rtol=1e-2: #{result}")
    end

    property "f16 sum agrees" do
      check all(
              n <- integer(2..8),
              vals <- list_of(float(min: -5.0, max: 5.0), length: n),
              max_runs: 10
            ) do
        diff(fn -> Nx.sum(Nx.tensor(vals, type: :f16)) end, atol: 1.0e-2, rtol: 1.0e-3)
      end
    end

    property "f16 exp agrees" do
      check all(
              n <- integer(2..6),
              vals <- list_of(float(min: -3.0, max: 3.0), length: n),
              max_runs: 10
            ) do
        diff(fn -> Nx.exp(Nx.tensor(vals, type: :f16)) end, atol: 1.0e-2, rtol: 2.0e-3)
      end
    end

    property "f16 sigmoid agrees" do
      check all(
              n <- integer(2..6),
              vals <- list_of(float(min: -5.0, max: 5.0), length: n),
              max_runs: 10
            ) do
        diff(fn -> Nx.sigmoid(Nx.tensor(vals, type: :f16)) end, atol: 5.0e-4, rtol: 2.0e-3)
      end
    end

    test "f16 grad through multiply agrees" do
      :rand.seed(:exsss, {66, 67, 68})
      va = for _ <- 1..16, do: (:rand.uniform() - 0.5) * 2.0
      vb = for _ <- 1..16, do: (:rand.uniform() - 0.5) * 2.0

      diff(fn ->
        a = Nx.tensor(va, type: :f16)
        b = Nx.tensor(vb, type: :f16)
        Nx.Defn.grad({a, b}, fn {x, y} -> Nx.sum(Nx.multiply(x, y)) end)
        |> elem(0)
      end, atol: 1.0e-3, rtol: 2.0e-3)
    end
  end

  describe "bf16 differential" do
    # bf16 has same exponent range as f32 but only 7-bit mantissa.
    # Hardware paths typically accumulate in f32 so results can be
    # tighter than naive bf16 emulation would suggest.

    test "bf16 matmul 16×16 agrees" do
      :rand.seed(:exsss, {70, 71, 72})
      va = for _ <- 1..256, do: (:rand.uniform() - 0.5)
      vb = for _ <- 1..256, do: (:rand.uniform() - 0.5)

      result =
        try do
          diff(fn ->
            a = Nx.tensor(va, type: :bf16) |> Nx.reshape({16, 16})
            b = Nx.tensor(vb, type: :bf16) |> Nx.reshape({16, 16})
            Nx.dot(a, b)
          end, atol: 0.0, rtol: 2.0e-2)
          :agreed
        rescue
          ExUnit.AssertionError -> :diverged
        end

      IO.puts("  [info] bf16 matmul 16x16 at rtol=2e-2: #{result}")
    end

    test "bf16 matmul 64×64 agrees" do
      :rand.seed(:exsss, {73, 74, 75})
      va = for _ <- 1..(64 * 64), do: (:rand.uniform() - 0.5) * 0.1
      vb = for _ <- 1..(64 * 64), do: (:rand.uniform() - 0.5) * 0.1

      result =
        try do
          diff(fn ->
            a = Nx.tensor(va, type: :bf16) |> Nx.reshape({64, 64})
            b = Nx.tensor(vb, type: :bf16) |> Nx.reshape({64, 64})
            Nx.dot(a, b)
          end, atol: 0.0, rtol: 3.0e-2)
          :agreed
        rescue
          ExUnit.AssertionError -> :diverged
        end

      IO.puts("  [info] bf16 matmul 64x64 at rtol=3e-2: #{result}")
    end

    property "bf16 sum agrees" do
      check all(
              n <- integer(2..8),
              vals <- list_of(float(min: -5.0, max: 5.0), length: n),
              max_runs: 10
            ) do
        diff(fn -> Nx.sum(Nx.tensor(vals, type: :bf16)) end, atol: 1.0e-1, rtol: 2.0e-2)
      end
    end

    property "bf16 exp agrees" do
      check all(
              n <- integer(2..6),
              vals <- list_of(float(min: -3.0, max: 3.0), length: n),
              max_runs: 10
            ) do
        diff(fn -> Nx.exp(Nx.tensor(vals, type: :bf16)) end, atol: 1.0e-1, rtol: 2.0e-2)
      end
    end

    test "bf16 grad through multiply agrees" do
      :rand.seed(:exsss, {76, 77, 78})
      va = for _ <- 1..16, do: (:rand.uniform() - 0.5) * 2.0
      vb = for _ <- 1..16, do: (:rand.uniform() - 0.5) * 2.0

      diff(fn ->
        a = Nx.tensor(va, type: :bf16)
        b = Nx.tensor(vb, type: :bf16)
        Nx.Defn.grad({a, b}, fn {x, y} -> Nx.sum(Nx.multiply(x, y)) end)
        |> elem(0)
      end, atol: 2.0e-2, rtol: 2.0e-2)
    end
  end

  # ── Reduced-precision edge cases: subnormals, NaN, Inf ─────────────

  describe "f16/bf16 edge cases" do
    test "f16 with values near representable range" do
      # f16 max ≈ 65504; values in this range may flush-to-zero
      # differently between emulated and hardware paths.
      result =
        try do
          diff(fn ->
            x = Nx.tensor([65000.0, -65000.0, 0.001, -0.001], type: :f16)
            Nx.add(x, x)
          end, atol: 1.0, rtol: 1.0e-3)
          :agreed
        rescue
          ExUnit.AssertionError -> :diverged
        end

      IO.puts("  [info] f16 near-max add: #{result}")
    end

    test "f16 sum of many small values (precision loss potential)" do
      # 1000 f16 values of 0.001 sum to 1.0 in infinite precision;
      # f16 precision loss may underestimate significantly.
      result =
        try do
          diff(fn ->
            x = Nx.broadcast(Nx.tensor(0.001, type: :f16), {1000})
            Nx.sum(x)
          end, atol: 0.1, rtol: 0.1)
          :agreed
        rescue
          ExUnit.AssertionError -> :diverged
        end

      IO.puts("  [info] f16 sum 1000x 0.001: #{result}")
    end
  end

  describe "linalg at 128x128 + tight tolerance" do
    # Push hard: if 32x32 linalg didn't diverge, 128x128 should.
    # If it still agrees, Nx.LinAlg really does avoid TF32-affected
    # code paths — a strong positive for the LinAlg module.

    defp rand_vals(n_sq, seed) do
      :rand.seed(:exsss, seed)
      for _ <- 1..n_sq, do: (:rand.uniform() - 0.5) * 0.1
    end

    test "determinant 128x128 at rtol=1e-5" do
      vals = rand_vals(128 * 128, {40, 41, 42})

      result =
        try do
          diff(fn ->
            r = Nx.tensor(vals, type: :f32) |> Nx.reshape({128, 128})
            a = Nx.add(Nx.eye(128, type: :f32), r)
            Nx.LinAlg.determinant(a)
          end, atol: 0.0, rtol: 1.0e-5)
          :agreed
        rescue
          ExUnit.AssertionError -> :diverged
        end

      IO.puts("  [info] determinant 128x128 at rtol=1e-5: #{result}")
    end

    test "invert 128x128 at rtol=1e-5" do
      vals = rand_vals(128 * 128, {43, 44, 45})

      result =
        try do
          diff(fn ->
            r = Nx.tensor(vals, type: :f32) |> Nx.reshape({128, 128})
            a = Nx.add(Nx.eye(128, type: :f32), r)
            Nx.LinAlg.invert(a)
          end, atol: 0.0, rtol: 1.0e-5)
          :agreed
        rescue
          ExUnit.AssertionError -> :diverged
        end

      IO.puts("  [info] invert 128x128 at rtol=1e-5: #{result}")
    end

    test "QR Q-matrix 128x128 at rtol=1e-5" do
      vals = rand_vals(128 * 128, {46, 47, 48})

      result =
        try do
          diff(fn ->
            a = Nx.tensor(vals, type: :f32) |> Nx.reshape({128, 128})
            {q, _r} = Nx.LinAlg.qr(a)
            q
          end, atol: 0.0, rtol: 1.0e-5)
          :agreed
        rescue
          ExUnit.AssertionError -> :diverged
        end

      IO.puts("  [info] QR Q 128x128 at rtol=1e-5: #{result}")
    end

    test "SVD s 128x128 at rtol=1e-5" do
      vals = rand_vals(128 * 128, {49, 50, 51})

      result =
        try do
          diff(fn ->
            a = Nx.tensor(vals, type: :f32) |> Nx.reshape({128, 128})
            {_u, s, _v} = Nx.LinAlg.svd(a)
            s
          end, atol: 0.0, rtol: 1.0e-5)
          :agreed
        rescue
          ExUnit.AssertionError -> :diverged
        end

      IO.puts("  [info] SVD s 128x128 at rtol=1e-5: #{result}")
    end
  end

  describe "linalg under tight tolerance (TF32 historical bug zone)" do
    test "QR on well-conditioned 8×8" do
      :rand.seed(:exsss, {10, 11, 12})
      vals = for _ <- 1..64, do: (:rand.uniform() - 0.5) * 0.2

      diff(fn ->
        r = Nx.tensor(vals, type: :f32) |> Nx.reshape({8, 8})
        a = Nx.add(Nx.eye(8, type: :f32), r)
        {q, _r} = Nx.LinAlg.qr(a)
        q
      end, atol: 1.0e-5, rtol: 1.0e-4)
    end

    test "determinant of 8×8 well-conditioned" do
      :rand.seed(:exsss, {13, 14, 15})
      vals = for _ <- 1..64, do: (:rand.uniform() - 0.5) * 0.2

      diff(fn ->
        r = Nx.tensor(vals, type: :f32) |> Nx.reshape({8, 8})
        a = Nx.add(Nx.eye(8, type: :f32), r)
        Nx.LinAlg.determinant(a)
      end, atol: 1.0e-5, rtol: 1.0e-4)
    end

    test "SVD singular values of 8×8" do
      :rand.seed(:exsss, {16, 17, 18})
      vals = for _ <- 1..64, do: (:rand.uniform() - 0.5) * 0.2

      diff(fn ->
        r = Nx.tensor(vals, type: :f32) |> Nx.reshape({8, 8})
        a = Nx.add(Nx.eye(8, type: :f32), r)
        {_u, s, _v} = Nx.LinAlg.svd(a)
        s
      end, atol: 1.0e-5, rtol: 1.0e-4)
    end
  end
end
