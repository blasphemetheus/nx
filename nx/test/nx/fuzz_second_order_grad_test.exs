defmodule Nx.FuzzSecondOrderGradTest do
  @moduledoc """
  Property-based fuzz tests for second-order gradients (grad of grad).

  For each input f : ℝⁿ → ℝ with a known analytical second derivative,
  assert that nested `Nx.Defn.grad` produces the expected result.

  These tests target a part of the autograd machinery that is rarely
  exercised by existing suites: nothing in the fuzz branch had
  second-order coverage at the time of writing.
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  @fuzz_scale String.to_integer(System.get_env("FUZZ_SCALE", "1"))

  import Nx.Testing

  # ── Generators ─────────────────────────────────────────────────────

  defp float_in(a, b) do
    map(integer(0..1000), fn k -> a + (b - a) * k / 1000 end)
  end

  defp nonzero_float(a, b) do
    bind(float_in(a, b), fn v ->
      if abs(v) < 1.0e-3, do: constant(1.0), else: constant(v)
    end)
  end

  # ── Scalar second derivatives (closed-form check) ─────────────────

  describe "scalar second derivatives match closed form" do
    property "d²/dx² (x²) == 2" do
      check all(x <- float_in(-5.0, 5.0), max_runs: 20 * @fuzz_scale) do
        t = Nx.tensor(x, type: :f32)

        d2 =
          Nx.Defn.grad(t, fn a ->
            Nx.Defn.grad(a, fn b -> Nx.multiply(b, b) end)
          end)

        assert_all_close(d2, Nx.tensor(2.0, type: :f32), atol: 1.0e-3)
      end
    end

    property "d²/dx² (x³) == 6x" do
      check all(x <- float_in(-5.0, 5.0), max_runs: 20 * @fuzz_scale) do
        t = Nx.tensor(x, type: :f32)

        d2 =
          Nx.Defn.grad(t, fn a ->
            Nx.Defn.grad(a, fn b -> Nx.multiply(Nx.multiply(b, b), b) end)
          end)

        expected = Nx.tensor(6 * x, type: :f32)
        assert_all_close(d2, expected, atol: 1.0e-3)
      end
    end

    property "d²/dx² sin(x) == -sin(x)" do
      check all(x <- float_in(-3.0, 3.0), max_runs: 20 * @fuzz_scale) do
        t = Nx.tensor(x, type: :f32)

        d2 =
          Nx.Defn.grad(t, fn a ->
            Nx.Defn.grad(a, fn b -> Nx.sin(b) end)
          end)

        assert_all_close(d2, Nx.tensor(-:math.sin(x), type: :f32), atol: 1.0e-3)
      end
    end

    property "d²/dx² exp(x) == exp(x)" do
      check all(x <- float_in(-3.0, 3.0), max_runs: 20 * @fuzz_scale) do
        t = Nx.tensor(x, type: :f32)

        d2 =
          Nx.Defn.grad(t, fn a ->
            Nx.Defn.grad(a, fn b -> Nx.exp(b) end)
          end)

        assert_all_close(d2, Nx.tensor(:math.exp(x), type: :f32), atol: 1.0e-2)
      end
    end

    property "d²/dx² log(x) == -1/x² (for x > 0)" do
      check all(x <- nonzero_float(0.1, 5.0), max_runs: 20 * @fuzz_scale) do
        t = Nx.tensor(x, type: :f32)

        d2 =
          Nx.Defn.grad(t, fn a ->
            Nx.Defn.grad(a, fn b -> Nx.log(b) end)
          end)

        expected = Nx.tensor(-1.0 / (x * x), type: :f32)
        assert_all_close(d2, expected, atol: 1.0e-3)
      end
    end

    property "d²/dx² tanh(x) == -2 tanh(x) sech²(x)" do
      check all(x <- float_in(-2.0, 2.0), max_runs: 20 * @fuzz_scale) do
        t = Nx.tensor(x, type: :f32)

        d2 =
          Nx.Defn.grad(t, fn a ->
            Nx.Defn.grad(a, fn b -> Nx.tanh(b) end)
          end)

        tanh_x = :math.tanh(x)
        sech2_x = 1.0 - tanh_x * tanh_x
        expected = Nx.tensor(-2.0 * tanh_x * sech2_x, type: :f32)
        assert_all_close(d2, expected, atol: 1.0e-3)
      end
    end
  end

  # ── Vector second-order: Hessian diagonal ─────────────────────────

  describe "vector second-order (Hessian diagonal)" do
    property "∂²/∂x² sum(x²) vector is [2, 2, ...]" do
      check all(n <- integer(2..5), max_runs: 10 * @fuzz_scale) do
        t = Nx.tensor(for(i <- 1..n, do: i / 1.0), type: :f32)

        # grad of sum(x²) is 2x. Its grad wrt x is Jacobian [2, 0; 0, 2; ...].
        # Taking the diagonal via elementwise: grad(2x wrt x) = 2 (vector of 2's).
        d2 =
          Nx.Defn.grad(t, fn a ->
            Nx.sum(Nx.Defn.grad(a, fn b -> Nx.sum(Nx.multiply(b, b)) end))
          end)

        assert_all_close(d2, Nx.broadcast(Nx.tensor(2.0, type: :f32), {n}), atol: 1.0e-3)
      end
    end

    property "∂²/∂x² sum(sin(x)) == -sin(x) (elementwise diagonal)" do
      check all(n <- integer(2..5), max_runs: 10 * @fuzz_scale) do
        xs = for(i <- 1..n, do: i * 0.5)
        t = Nx.tensor(xs, type: :f32)

        d2 =
          Nx.Defn.grad(t, fn a ->
            Nx.sum(Nx.Defn.grad(a, fn b -> Nx.sum(Nx.sin(b)) end))
          end)

        expected = Nx.tensor(Enum.map(xs, &(-:math.sin(&1))), type: :f32)
        assert_all_close(d2, expected, atol: 1.0e-3)
      end
    end
  end

  # ── Grad-of-grad through composed functions ───────────────────────

  describe "composed second derivatives" do
    property "d²/dx² (sin(x) * x) == 2cos(x) - x sin(x)" do
      check all(x <- float_in(-3.0, 3.0), max_runs: 20 * @fuzz_scale) do
        t = Nx.tensor(x, type: :f32)

        d2 =
          Nx.Defn.grad(t, fn a ->
            Nx.Defn.grad(a, fn b -> Nx.multiply(Nx.sin(b), b) end)
          end)

        expected = Nx.tensor(2 * :math.cos(x) - x * :math.sin(x), type: :f32)
        assert_all_close(d2, expected, atol: 1.0e-3)
      end
    end
  end

  # ── Harder: second-order on batched / vectorized / linalg inputs ──
  #
  # These are the cases most likely to surface bugs. Second-order grads
  # exercise the autograd infrastructure twice, so any rank/batch/
  # vectorization bug that first-order tolerates gets amplified.

  describe "second-order on batched inputs" do
    property "d²/dx² sum(x²) over 2-D batched input is all 2s" do
      check all(b <- integer(2..3), n <- integer(2..4), max_runs: 6 * @fuzz_scale) do
        t = Nx.iota({b, n}, type: :f32)

        d2 =
          Nx.Defn.grad(t, fn a ->
            Nx.sum(Nx.Defn.grad(a, fn c -> Nx.sum(Nx.multiply(c, c)) end))
          end)

        assert_all_close(d2, Nx.broadcast(Nx.tensor(2.0, type: :f32), {b, n}), atol: 1.0e-3)
      end
    end

    property "d²/dx² sum(exp(x)) over batched input == exp(x) (elementwise)" do
      check all(b <- integer(2..3), n <- integer(2..4), max_runs: 6 * @fuzz_scale) do
        xs = Nx.divide(Nx.iota({b, n}, type: :f32), Nx.tensor(10.0, type: :f32))

        d2 =
          Nx.Defn.grad(xs, fn a ->
            Nx.sum(Nx.Defn.grad(a, fn c -> Nx.sum(Nx.exp(c)) end))
          end)

        expected = Nx.exp(xs)
        assert_all_close(d2, expected, atol: 1.0e-2)
      end
    end
  end

  describe "second-order at non-smooth / edge points" do
    # d²/dx² |x| is 0 almost everywhere (and undefined at 0). Autograd
    # typically returns 0 for this, but the behavior at x = 0 is an
    # implementation choice worth pinning.
    property "d²/dx² abs(x) == 0 for x != 0" do
      check all(x <- float_in(0.5, 5.0), max_runs: 10 * @fuzz_scale) do
        t = Nx.tensor(x, type: :f32)

        d2 =
          Nx.Defn.grad(t, fn a ->
            Nx.Defn.grad(a, fn b -> Nx.abs(b) end)
          end)

        assert_all_close(d2, Nx.tensor(0.0, type: :f32), atol: 1.0e-4)
      end
    end

    test "d²/dx² abs(x) at x = 0 should not crash" do
      t = Nx.tensor(0.0, type: :f32)

      d2 =
        Nx.Defn.grad(t, fn a ->
          Nx.Defn.grad(a, fn b -> Nx.abs(b) end)
        end)

      # Either 0.0 or NaN is defensible; the main assertion is "doesn't
      # crash the grad pipeline."
      _ = Nx.to_number(d2)
    end
  end

  describe "cross-partial derivatives (different variables)" do
    # d²/dxdy (x * y) = 1 — a basic sanity check for multi-variable
    # second derivatives.
    property "d/dx d/dy (x * y) == 1" do
      check all(x <- float_in(-3.0, 3.0), y <- float_in(-3.0, 3.0), max_runs: 10 * @fuzz_scale) do
        tx = Nx.tensor(x, type: :f32)
        ty = Nx.tensor(y, type: :f32)

        # The outer grad is wrt tx; the inner grad is wrt ty.
        cross =
          Nx.Defn.grad(tx, fn a ->
            Nx.Defn.grad(ty, fn b -> Nx.multiply(a, b) end)
          end)

        assert_all_close(cross, Nx.tensor(1.0, type: :f32), atol: 1.0e-4)
      end
    end
  end

  describe "second-order through vectorized inputs" do
    # First-order vec-grad already has known gaps (see #1706/#1729/#1730).
    # Second-order through vectorize is strictly harder and is a likely
    # bug zone — if this doesn't crash it means the boundary-wrapper
    # approach in #1731 actually composes correctly.
    test "d²/dx² x² works through Nx.vectorize on a batch" do
      v = Nx.tensor([1.0, 2.0, 3.0], type: :f32) |> Nx.vectorize(:batch)

      d2 =
        Nx.Defn.grad(v, fn x ->
          Nx.Defn.grad(x, fn y -> Nx.multiply(y, y) end)
        end)

      assert d2.vectorized_axes == [batch: 3]
      devec = Nx.devectorize(d2, keep_names: false)
      assert_all_close(devec, Nx.broadcast(Nx.tensor(2.0, type: :f32), {3}), atol: 1.0e-4)
    end

    test "d²/dx² sin(x) works through Nx.vectorize" do
      xs = [0.5, 1.0, 1.5]
      v = Nx.tensor(xs, type: :f32) |> Nx.vectorize(:batch)

      d2 =
        Nx.Defn.grad(v, fn x ->
          Nx.Defn.grad(x, fn y -> Nx.sin(y) end)
        end)

      expected_vals = Enum.map(xs, &(-:math.sin(&1)))
      devec = Nx.devectorize(d2, keep_names: false)
      assert_all_close(devec, Nx.tensor(expected_vals, type: :f32), atol: 1.0e-3)
    end
  end

  describe "Hessian-vector products" do
    # For quadratic form f(x) = 0.5 * xᵀ A x, Hessian is A, so H·v = A·v.
    # Implement via `grad(x, grad(x, f) · v)` and compare.
    test "HVP of quadratic form 0.5 xᵀAx matches A·v" do
      a = Nx.tensor([[2.0, 1.0], [1.0, 3.0]], type: :f32)
      x = Nx.tensor([1.0, 2.0], type: :f32)
      v = Nx.tensor([0.5, -0.3], type: :f32)

      hvp =
        Nx.Defn.grad(x, fn xi ->
          grad_f =
            Nx.Defn.grad(xi, fn y ->
              Nx.multiply(0.5, Nx.dot(y, Nx.dot(a, y)))
            end)

          Nx.dot(grad_f, v)
        end)

      expected = Nx.dot(a, v)
      assert_all_close(hvp, expected, atol: 1.0e-4)
    end
  end

  describe "second-order through linalg ops" do
    # First-order grad through Cholesky on batched input is already
    # broken post-#1731 (see PR thread). Check 2-D case: does a 2-D
    # second-order grad through Cholesky even work?
    test "d²/dx² sum(cholesky(A)) for 2-D input doesn't crash" do
      a = Nx.tensor([[4.0, 2.0], [2.0, 5.0]], type: :f32)

      # If this raises, we've found a second-order-through-linalg bug.
      # Don't assert the specific value; just that it completes.
      d2 =
        Nx.Defn.grad(a, fn x ->
          Nx.sum(Nx.Defn.grad(x, fn y -> Nx.sum(Nx.LinAlg.cholesky(y)) end))
        end)

      assert Nx.shape(d2) == Nx.shape(a)
    end

    # First-order sqrt grad is well-tested. Second-order might surface
    # any subtle issue with how grad composes through sqrt.
    property "d²/dx² sqrt(x) == -0.25 * x^(-3/2) for x > 0" do
      check all(x <- nonzero_float(0.5, 5.0), max_runs: 10 * @fuzz_scale) do
        t = Nx.tensor(x, type: :f32)

        d2 =
          Nx.Defn.grad(t, fn a ->
            Nx.Defn.grad(a, fn b -> Nx.sqrt(b) end)
          end)

        expected = Nx.tensor(-0.25 * :math.pow(x, -1.5), type: :f32)
        assert_all_close(d2, expected, atol: 1.0e-3)
      end
    end
  end
end
