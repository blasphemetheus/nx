defmodule TorchxDifferentialFuzzTest do
  @moduledoc """
  Cross-backend differential fuzzing: `Nx.BinaryBackend` vs
  `Torchx.Backend`. Both CPU — disagreements are real bugs in one of
  them.

  Torchx wraps libtorch. Expected divergence zones (historical):

  - Complex ops (libtorch complex paths have edge cases)
  - Reductions with axes (libtorch's ordering may differ from
    BinaryBackend's iteration)
  - Specific linalg where Torchx delegates directly to libtorch
    (QR, SVD, eigh) — different algorithm than BinaryBackend's
    hand-rolled Elixir
  - Type promotion rules for edge dtype pairs
  - Grad through ops where BinaryBackend and Torchx disagree on
    derivative formulas

  Complement to `exla/test/differential_fuzz_test.exs`: EXLA catches
  XLA/TF32/Blackwell issues, Torchx catches libtorch-specific issues.
  """
  use ExUnit.Case, async: false
  use ExUnitProperties

  import Nx.Testing

  @binary_backend Nx.BinaryBackend
  @torchx_backend Torchx.Backend

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
    b = run_under(@torchx_backend, fun) |> Nx.backend_copy(Nx.BinaryBackend)

    assert Nx.shape(a) == Nx.shape(b),
      "shape mismatch: binary #{inspect(Nx.shape(a))} vs torchx #{inspect(Nx.shape(b))}"

    assert Nx.type(a) == Nx.type(b),
      "type mismatch: binary #{inspect(Nx.type(a))} vs torchx #{inspect(Nx.type(b))}"

    assert_all_close(a, b, atol: atol, rtol: rtol)
  end

  # ── Element-wise basics ────────────────────────────────────────────

  describe "element-wise agreement" do
    property "add f32" do
      check all(
              n <- integer(2..8),
              va <- list_of(float(min: -100.0, max: 100.0), length: n),
              vb <- list_of(float(min: -100.0, max: 100.0), length: n),
              max_runs: 10
            ) do
        diff(fn ->
          Nx.add(Nx.tensor(va, type: :f32), Nx.tensor(vb, type: :f32))
        end)
      end
    end

    property "multiply f32" do
      check all(
              n <- integer(2..8),
              va <- list_of(float(min: -10.0, max: 10.0), length: n),
              vb <- list_of(float(min: -10.0, max: 10.0), length: n),
              max_runs: 10
            ) do
        diff(fn ->
          Nx.multiply(Nx.tensor(va, type: :f32), Nx.tensor(vb, type: :f32))
        end)
      end
    end

    property "exp / log agree" do
      check all(
              n <- integer(2..6),
              vals <- list_of(float(min: 0.1, max: 3.0), length: n),
              max_runs: 10
            ) do
        diff(fn ->
          x = Nx.tensor(vals, type: :f32)
          Nx.log(Nx.exp(x))
        end, atol: 1.0e-5, rtol: 1.0e-5)
      end
    end
  end

  # ── Reductions (Torchx reduction ordering may differ) ──────────────

  describe "reductions agree" do
    property "sum" do
      check all(
              n <- integer(2..10),
              vals <- list_of(float(min: -50.0, max: 50.0), length: n),
              max_runs: 10
            ) do
        diff(fn -> Nx.sum(Nx.tensor(vals, type: :f32)) end)
      end
    end

    property "mean" do
      check all(
              n <- integer(2..10),
              vals <- list_of(float(min: -50.0, max: 50.0), length: n),
              max_runs: 10
            ) do
        diff(fn -> Nx.mean(Nx.tensor(vals, type: :f32)) end)
      end
    end

    property "argmax / argmin (exact — integer)" do
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

    property "cumulative_sum" do
      check all(
              n <- integer(2..8),
              vals <- list_of(float(min: -10.0, max: 10.0), length: n),
              max_runs: 10
            ) do
        diff(fn -> Nx.cumulative_sum(Nx.tensor(vals, type: :f32)) end)
      end
    end
  end

  # ── Linalg via libtorch (different algorithm than BinaryBackend) ──

  describe "linalg agreement" do
    property "matmul 3×3 agrees" do
      check all(
              va <- list_of(float(min: -3.0, max: 3.0), length: 9),
              vb <- list_of(float(min: -3.0, max: 3.0), length: 9),
              max_runs: 10
            ) do
        diff(fn ->
          a = Nx.tensor(va, type: :f32) |> Nx.reshape({3, 3})
          b = Nx.tensor(vb, type: :f32) |> Nx.reshape({3, 3})
          Nx.dot(a, b)
        end, atol: 1.0e-5, rtol: 1.0e-5)
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
        end, atol: 1.0e-5, rtol: 1.0e-4)
      end
    end

    property "invert of well-conditioned 3×3 agrees" do
      check all(
              vals <- list_of(float(min: -1.0, max: 1.0), length: 9),
              max_runs: 10
            ) do
        diff(fn ->
          r = Nx.tensor(vals, type: :f32) |> Nx.reshape({3, 3})
          a = Nx.add(Nx.eye(3, type: :f32), Nx.multiply(r, 0.1))
          Nx.LinAlg.invert(a)
        end, atol: 1.0e-4, rtol: 1.0e-4)
      end
    end

    property "cholesky on SPD matrix" do
      check all(
              vals <- list_of(float(min: -1.0, max: 1.0), length: 9),
              max_runs: 8
            ) do
        diff(fn ->
          r = Nx.tensor(vals, type: :f32) |> Nx.reshape({3, 3})
          sym = Nx.add(r, Nx.transpose(r)) |> Nx.divide(2.0)
          spd = Nx.add(Nx.eye(3, type: :f32), Nx.multiply(sym, 0.1))
          Nx.LinAlg.cholesky(spd)
        end, atol: 1.0e-5, rtol: 1.0e-4)
      end
    end
  end

  # ── Complex (historical flaky zone in Torchx) ─────────────────────

  describe "complex agreement" do
    test "c64 add" do
      diff(fn ->
        a = Nx.tensor([Complex.new(1.0, 2.0), Complex.new(3.0, -1.0)], type: :c64)
        b = Nx.tensor([Complex.new(0.5, 1.0), Complex.new(-1.0, 2.0)], type: :c64)
        Nx.add(a, b)
      end)
    end

    test "c64 multiply" do
      diff(fn ->
        a = Nx.tensor([Complex.new(1.0, 2.0), Complex.new(3.0, -1.0)], type: :c64)
        b = Nx.tensor([Complex.new(0.5, 1.0), Complex.new(-1.0, 2.0)], type: :c64)
        Nx.multiply(a, b)
      end)
    end

    test "c64 conjugate" do
      diff(fn ->
        a = Nx.tensor([Complex.new(1.0, 2.0), Complex.new(-3.0, 0.5)], type: :c64)
        Nx.conjugate(a)
      end)
    end

    test "c64 exp" do
      diff(fn ->
        a = Nx.tensor([Complex.new(0.5, 1.0), Complex.new(-1.0, 2.0)], type: :c64)
        Nx.exp(a)
      end, atol: 1.0e-5, rtol: 1.0e-4)
    end

    test "c64 abs" do
      diff(fn ->
        a = Nx.tensor([Complex.new(3.0, 4.0), Complex.new(-5.0, 12.0)], type: :c64)
        Nx.abs(a)
      end)
    end
  end

  # ── Grad agreement ─────────────────────────────────────────────────

  describe "grad agreement" do
    property "grad of sin" do
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

    property "grad of sum(x*x)" do
      check all(
              n <- integer(2..6),
              vals <- list_of(float(min: -3.0, max: 3.0), length: n),
              max_runs: 10
            ) do
        diff(fn ->
          t = Nx.tensor(vals, type: :f32)
          Nx.Defn.grad(t, fn a -> Nx.sum(Nx.multiply(a, a)) end)
        end)
      end
    end

    property "grad through matmul (tuple arg)" do
      check all(
              va <- list_of(float(min: -1.0, max: 1.0), length: 9),
              vb <- list_of(float(min: -1.0, max: 1.0), length: 9),
              max_runs: 8
            ) do
        diff(fn ->
          a = Nx.tensor(va, type: :f32) |> Nx.reshape({3, 3})
          b = Nx.tensor(vb, type: :f32) |> Nx.reshape({3, 3})
          Nx.Defn.grad({a, b}, fn {x, y} -> Nx.sum(Nx.dot(x, y)) end)
          |> elem(0)
        end, atol: 1.0e-5, rtol: 1.0e-4)
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

    property "bitwise xor exact" do
      check all(
              n <- integer(2..6),
              va <- list_of(integer(0..1000), length: n),
              vb <- list_of(integer(0..1000), length: n),
              max_runs: 10
            ) do
        diff(fn ->
          Nx.bitwise_xor(Nx.tensor(va, type: :s32), Nx.tensor(vb, type: :s32))
        end, atol: 0.0, rtol: 0.0)
      end
    end
  end

  # ── Harder differential zones ──────────────────────────────────────

  describe "linalg with algorithm-divergent paths" do
    # Torchx delegates QR/SVD/eigh to libtorch LAPACK routines, while
    # BinaryBackend uses its own Elixir/Householder implementation.
    # These are different algorithms — tight tolerance should catch
    # deviations beyond floating point accumulation.

    property "QR Q-matrix orthonormality agrees" do
      check all(
              vals <- list_of(float(min: -2.0, max: 2.0), length: 16),
              max_runs: 8
            ) do
        # Q^T Q should be identity; both backends should agree.
        diff(fn ->
          a = Nx.tensor(vals, type: :f32) |> Nx.reshape({4, 4})
          {q, _r} = Nx.LinAlg.qr(a)
          Nx.dot(Nx.transpose(q), q)
        end, atol: 1.0e-5, rtol: 1.0e-4)
      end
    end

    property "SVD singular values (sorted) agree" do
      check all(
              vals <- list_of(float(min: -1.0, max: 1.0), length: 16),
              max_runs: 8
            ) do
        diff(fn ->
          a = Nx.tensor(vals, type: :f32) |> Nx.reshape({4, 4})
          {_u, s, _v} = Nx.LinAlg.svd(a)
          s
        end, atol: 1.0e-4, rtol: 1.0e-4)
      end
    end

    test "eigh symmetric matrix eigenvalues" do
      :rand.seed(:exsss, {80, 81, 82})
      vals = for _ <- 1..16, do: (:rand.uniform() - 0.5) * 2.0

      diff(fn ->
        a = Nx.tensor(vals, type: :f32) |> Nx.reshape({4, 4})
        sym = Nx.divide(Nx.add(a, Nx.transpose(a)), 2.0)
        {eigvals, _eigvecs} = Nx.LinAlg.eigh(sym)
        # Sort to avoid ordering differences.
        Nx.sort(eigvals)
      end, atol: 1.0e-4, rtol: 1.0e-4)
    end
  end

  describe "sort tie-breaking" do
    # Different backends may break ties (equal values) differently.
    # argsort on a tensor with ties is a likely divergence.

    test "argsort of unique values agrees" do
      diff(fn ->
        t = Nx.tensor([3.0, 1.0, 4.0, 1.5, 9.0, 2.0, 6.0], type: :f32)
        Nx.argsort(t)
      end, atol: 0.0, rtol: 0.0)
    end

    test "argsort with ties: BinaryBackend and Torchx may differ" do
      # [1, 1, 2, 2, 3] has ties. Different impls may give [0,1,2,3,4]
      # or [1,0,3,2,4] — both are valid answers.
      result =
        try do
          diff(fn ->
            t = Nx.tensor([1, 1, 2, 2, 3], type: :s32)
            Nx.argsort(t)
          end, atol: 0.0, rtol: 0.0)
          :agreed
        rescue
          ExUnit.AssertionError -> :diverged
        end

      IO.puts("  [info] argsort with ties: #{result}")
    end

    test "sort (not argsort) with ties agrees" do
      # Sort itself produces the same sorted values regardless of
      # tie-breaking.
      diff(fn ->
        t = Nx.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 1.5, 1.5], type: :f32)
        Nx.sort(t)
      end, atol: 0.0, rtol: 0.0)
    end
  end

  describe "sub-byte dtypes" do
    # Torchx may not support u2/u4/s2/s4 at all — probe.

    test "u4 add (does Torchx support it?)" do
      result =
        try do
          diff(fn ->
            a = Nx.tensor([0, 1, 2, 3], type: :u4)
            b = Nx.tensor([1, 1, 1, 1], type: :u4)
            Nx.add(a, b)
          end, atol: 0.0, rtol: 0.0)
          :agreed
        rescue
          e -> {:raised, Exception.message(e) |> String.slice(0, 100)}
        end

      IO.puts("  [info] u4 add: #{inspect(result)}")
    end

    test "s2 addition probe" do
      result =
        try do
          diff(fn ->
            a = Nx.tensor([-2, -1, 0, 1], type: :s2)
            b = Nx.tensor([1, 1, 0, -1], type: :s2)
            Nx.add(a, b)
          end, atol: 0.0, rtol: 0.0)
          :agreed
        rescue
          e -> {:raised, Exception.message(e) |> String.slice(0, 100)}
        end

      IO.puts("  [info] s2 add: #{inspect(result)}")
    end
  end

  describe "scatter / indexed ops" do
    test "indexed_add with overlapping indices" do
      diff(fn ->
        t = Nx.broadcast(Nx.tensor(0.0, type: :f32), {5})
        idx = Nx.tensor([[0], [0], [1], [2], [2]])
        updates = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0], type: :f32)
        Nx.indexed_add(t, idx, updates)
      end)
    end

    test "put_slice agreement" do
      diff(fn ->
        t = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0], type: :f32)
        patch = Nx.tensor([100.0, 200.0], type: :f32)
        Nx.put_slice(t, [1], patch)
      end)
    end
  end

  # ── Shape / indexing ───────────────────────────────────────────────

  describe "shape / indexing agree" do
    property "reshape round-trip" do
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
end
