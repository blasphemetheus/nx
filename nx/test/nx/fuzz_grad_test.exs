defmodule Nx.FuzzGradTest do
  @moduledoc """
  Tier 2: Gradient verification via finite differences.

  For each differentiable op, compares Nx.Defn.grad against
  numerical finite differences: (f(x+eps) - f(x-eps)) / (2*eps).

  Based on the NablaFuzz approach (ICSE 2023) which found 107 AD bugs.
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  import Nx.Defn

  @step 1.0e-4
  @atol 1.0e-2
  @rtol 1.0e-2

  # ── Helper: numerical gradient via finite differences ─────────────

  defp numerical_grad(fun, x) do
    # Centered finite differences: (f(x+h/2) - f(x-h/2)) / h
    # Works for scalar-output functions of vector inputs
    x_flat = Nx.to_flat_list(x)
    shape = Nx.shape(x)
    type = Nx.type(x)

    grads =
      x_flat
      |> Enum.with_index()
      |> Enum.map(fn {_val, i} ->
        perturbation = List.duplicate(0.0, length(x_flat))
        perturbation = List.replace_at(perturbation, i, @step / 2.0)
        h = Nx.tensor(perturbation, type: type) |> Nx.reshape(shape)

        f_plus = fun.(Nx.add(x, h)) |> Nx.to_number()
        f_minus = fun.(Nx.subtract(x, h)) |> Nx.to_number()
        (f_plus - f_minus) / @step
      end)

    Nx.tensor(grads, type: type) |> Nx.reshape(shape)
  end

  defp check_grad(fun, x, opts \\ []) do
    atol = opts[:atol] || @atol
    rtol = opts[:rtol] || @rtol

    analytical = Nx.Defn.grad(x, fun)
    numerical = numerical_grad(fun, x)

    diff = Nx.subtract(analytical, numerical) |> Nx.abs()
    scale = Nx.max(Nx.abs(numerical), 1.0)
    rel_diff = Nx.divide(diff, scale)

    max_abs = Nx.reduce_max(diff) |> Nx.to_number()
    max_rel = Nx.reduce_max(rel_diff) |> Nx.to_number()

    if max_abs > atol and max_rel > rtol do
      flunk("""
      Gradient mismatch!
      max_abs_diff: #{max_abs}
      max_rel_diff: #{max_rel}
      analytical: #{inspect(analytical)}
      numerical:  #{inspect(numerical)}
      input:      #{inspect(x)}
      """)
    end
  end

  defp random_input(n, opts \\ []) do
    min = opts[:min] || -2.0
    max = opts[:max] || 2.0
    range = max - min

    Nx.tensor(for(_ <- 1..n, do: :rand.uniform() * range + min), type: :f64)
  end

  # ── Unary element-wise gradients ──────────────────────────────────

  describe "unary element-wise gradients" do
    property "grad of sum(sin(x))" do
      check all(n <- integer(1..8), max_runs: 10) do
        x = random_input(n)
        check_grad(fn x -> Nx.sum(Nx.sin(x)) end, x)
      end
    end

    property "grad of sum(cos(x))" do
      check all(n <- integer(1..8), max_runs: 10) do
        x = random_input(n)
        check_grad(fn x -> Nx.sum(Nx.cos(x)) end, x)
      end
    end

    property "grad of sum(tanh(x))" do
      check all(n <- integer(1..8), max_runs: 10) do
        x = random_input(n)
        check_grad(fn x -> Nx.sum(Nx.tanh(x)) end, x)
      end
    end

    property "grad of sum(sigmoid(x))" do
      check all(n <- integer(1..8), max_runs: 10) do
        x = random_input(n, min: -3.0, max: 3.0)
        check_grad(fn x -> Nx.sum(Nx.sigmoid(x)) end, x)
      end
    end

    property "grad of sum(exp(x))" do
      check all(n <- integer(1..8), max_runs: 10) do
        x = random_input(n, min: -2.0, max: 2.0)
        check_grad(fn x -> Nx.sum(Nx.exp(x)) end, x)
      end
    end

    property "grad of sum(log(x))" do
      check all(n <- integer(1..8), max_runs: 10) do
        x = random_input(n, min: 0.5, max: 5.0)
        check_grad(fn x -> Nx.sum(Nx.log(x)) end, x)
      end
    end

    property "grad of sum(sqrt(x))" do
      check all(n <- integer(1..8), max_runs: 10) do
        x = random_input(n, min: 0.5, max: 5.0)
        check_grad(fn x -> Nx.sum(Nx.sqrt(x)) end, x)
      end
    end

    property "grad of sum(abs(x))" do
      check all(n <- integer(1..8), max_runs: 10) do
        # Avoid x=0 where abs is not differentiable
        x = random_input(n, min: 0.5, max: 5.0)
        check_grad(fn x -> Nx.sum(Nx.abs(x)) end, x)
      end
    end

    property "grad of sum(negate(x))" do
      check all(n <- integer(1..8), max_runs: 10) do
        x = random_input(n)
        check_grad(fn x -> Nx.sum(Nx.negate(x)) end, x)
      end
    end

    property "grad of sum(cbrt(x))" do
      check all(n <- integer(1..8), max_runs: 10) do
        x = random_input(n, min: 0.5, max: 5.0)
        check_grad(fn x -> Nx.sum(Nx.cbrt(x)) end, x)
      end
    end

    property "grad of sum(rsqrt(x))" do
      check all(n <- integer(1..8), max_runs: 10) do
        x = random_input(n, min: 0.5, max: 5.0)
        check_grad(fn x -> Nx.sum(Nx.rsqrt(x)) end, x)
      end
    end

    property "grad of sum(erf(x))" do
      check all(n <- integer(1..8), max_runs: 10) do
        x = random_input(n, min: -2.0, max: 2.0)
        check_grad(fn x -> Nx.sum(Nx.erf(x)) end, x)
      end
    end
  end

  # ── Binary element-wise gradients ─────────────────────────────────

  describe "binary element-wise gradients" do
    property "grad of sum(x + y) wrt x" do
      check all(n <- integer(1..8), max_runs: 10) do
        x = random_input(n)
        y = random_input(n)
        check_grad(fn x -> Nx.sum(Nx.add(x, y)) end, x)
      end
    end

    property "grad of sum(x * y) wrt x" do
      check all(n <- integer(1..8), max_runs: 10) do
        x = random_input(n)
        y = random_input(n)
        check_grad(fn x -> Nx.sum(Nx.multiply(x, y)) end, x)
      end
    end

    property "grad of sum(x / y) wrt x" do
      check all(n <- integer(1..8), max_runs: 10) do
        x = random_input(n)
        y = random_input(n, min: 0.5, max: 5.0)
        check_grad(fn x -> Nx.sum(Nx.divide(x, y)) end, x)
      end
    end

    property "grad of sum(x - y) wrt x" do
      check all(n <- integer(1..8), max_runs: 10) do
        x = random_input(n)
        y = random_input(n)
        check_grad(fn x -> Nx.sum(Nx.subtract(x, y)) end, x)
      end
    end

    property "grad of sum(x^2)" do
      check all(n <- integer(1..8), max_runs: 10) do
        x = random_input(n)
        check_grad(fn x -> Nx.sum(Nx.pow(x, 2)) end, x)
      end
    end

    property "grad of sum(x^3)" do
      check all(n <- integer(1..8), max_runs: 10) do
        x = random_input(n)
        check_grad(fn x -> Nx.sum(Nx.pow(x, 3)) end, x)
      end
    end
  end

  # ── Reduction gradients ───────────────────────────────────────────

  describe "reduction gradients" do
    property "grad of sum(x)" do
      check all(n <- integer(1..8), max_runs: 10) do
        x = random_input(n)
        check_grad(&Nx.sum/1, x)
      end
    end

    property "grad of mean(x)" do
      check all(n <- integer(1..8), max_runs: 10) do
        x = random_input(n)
        check_grad(&Nx.mean/1, x)
      end
    end

    property "grad of sum(x^2) (variance-like)" do
      check all(n <- integer(2..8), max_runs: 10) do
        x = random_input(n)
        check_grad(fn x -> Nx.sum(Nx.pow(x, 2)) end, x)
      end
    end
  end

  # ── Composition gradients ─────────────────────────────────────────

  describe "composition gradients" do
    property "grad of sum(sin(x) * cos(x))" do
      check all(n <- integer(1..8), max_runs: 10) do
        x = random_input(n)
        check_grad(fn x -> Nx.sum(Nx.multiply(Nx.sin(x), Nx.cos(x))) end, x)
      end
    end

    property "grad of sum(exp(sin(x)))" do
      check all(n <- integer(1..8), max_runs: 10) do
        x = random_input(n, min: -1.0, max: 1.0)
        check_grad(fn x -> Nx.sum(Nx.exp(Nx.sin(x))) end, x)
      end
    end

    property "grad of sum(log(1 + exp(x))) (softplus)" do
      check all(n <- integer(1..8), max_runs: 10) do
        x = random_input(n, min: -3.0, max: 3.0)
        check_grad(fn x -> Nx.sum(Nx.log(Nx.add(1, Nx.exp(x)))) end, x)
      end
    end

    property "grad of sum(tanh(x)^2)" do
      check all(n <- integer(1..8), max_runs: 10) do
        x = random_input(n)
        check_grad(fn x -> Nx.sum(Nx.pow(Nx.tanh(x), 2)) end, x)
      end
    end
  end

  # ── Dot product gradients ─────────────────────────────────────────

  describe "dot product gradients" do
    property "grad of dot(x, y) wrt x" do
      check all(n <- integer(1..6), max_runs: 10) do
        x = random_input(n)
        y = random_input(n)
        check_grad(fn x -> Nx.dot(x, y) end, x)
      end
    end

    property "grad of sum(matmul(x, w)) wrt x" do
      check all(
              m <- integer(1..4),
              k <- integer(1..4),
              max_runs: 8
            ) do
        x = random_input(m * k) |> Nx.reshape({m, k})
        w = random_input(k * 2) |> Nx.reshape({k, 2})

        check_grad(fn x -> Nx.sum(Nx.dot(x, w)) end, x)
      end
    end
  end

  # ── Shape op gradients ────────────────────────────────────────────

  describe "shape op gradients" do
    property "grad through reshape" do
      check all(n <- integer(1..8), max_runs: 10) do
        x = random_input(n)
        check_grad(fn x -> Nx.sum(Nx.reshape(x, {1, n})) end, x)
      end
    end

    property "grad through squeeze" do
      check all(n <- integer(1..8), max_runs: 10) do
        x = random_input(n) |> Nx.reshape({1, n})
        check_grad(fn x -> Nx.sum(Nx.squeeze(x, axes: [0])) end, x)
      end
    end

    property "grad through transpose" do
      check all(
              m <- integer(1..4),
              n <- integer(1..4),
              max_runs: 8
            ) do
        x = random_input(m * n) |> Nx.reshape({m, n})
        check_grad(fn x -> Nx.sum(Nx.transpose(x)) end, x)
      end
    end

    property "grad through concatenate" do
      check all(n <- integer(1..6), max_runs: 8) do
        x = random_input(n)
        check_grad(fn x -> Nx.sum(Nx.concatenate([x, x])) end, x)
      end
    end

    property "grad through slice" do
      check all(n <- integer(3..8), max_runs: 8) do
        x = random_input(n)
        check_grad(fn x -> Nx.sum(Nx.slice(x, [1], [n - 2])) end, x)
      end
    end

    property "grad through pad" do
      check all(n <- integer(1..6), max_runs: 8) do
        x = random_input(n)
        check_grad(fn x -> Nx.sum(Nx.pad(x, 0, [{1, 1, 0}])) end, x)
      end
    end

    property "grad through reverse" do
      check all(n <- integer(1..8), max_runs: 8) do
        x = random_input(n)
        check_grad(fn x -> Nx.sum(Nx.reverse(x)) end, x)
      end
    end
  end

  # ── Higher-order gradients ────────────────────────────────────────

  describe "higher-order gradients" do
    property "second derivative of sum(x^3)" do
      check all(n <- integer(1..6), max_runs: 8) do
        x = random_input(n)

        # d/dx sum(x^3) = 3x^2, d²/dx² sum(x^3) = 6x
        second_grad =
          Nx.Defn.grad(x, fn x ->
            Nx.sum(Nx.Defn.grad(x, fn x -> Nx.sum(Nx.pow(x, 3)) end))
          end)

        expected = Nx.multiply(6, x)

        diff =
          Nx.subtract(second_grad, expected)
          |> Nx.abs()
          |> Nx.reduce_max()
          |> Nx.to_number()

        assert diff < 0.1
      end
    end

    property "second derivative of sum(sin(x))" do
      check all(n <- integer(1..6), max_runs: 8) do
        x = random_input(n)

        # d/dx sum(sin(x)) = cos(x), d²/dx² sum(sin(x)) = -sin(x)
        second_grad =
          Nx.Defn.grad(x, fn x ->
            Nx.sum(Nx.Defn.grad(x, fn x -> Nx.sum(Nx.sin(x)) end))
          end)

        expected = Nx.negate(Nx.sin(x))

        diff =
          Nx.subtract(second_grad, expected)
          |> Nx.abs()
          |> Nx.reduce_max()
          |> Nx.to_number()

        assert diff < 0.1
      end
    end
  end
end
