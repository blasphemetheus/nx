defmodule Nx.FuzzCondVectorizedGradTest do
  @moduledoc """
  Fuzz for `cond`-under-grad and multi-axis vectorization (FUZZ_ROADMAP T2.3).

  Both configurations were historical bug factories (#1729/#1730 came from
  vectorized+cond interactions; #1533 was the vectorized-grad cluster) yet
  neither was property-fuzzed. Oracles are closed-form derivatives, so every
  assertion is exact up to float tolerance:

    d/dx sum(sin x) = cos x     d/dx sum(x²) = 2x
    d/dx sum(cos x) = -sin x    d/dx sum(exp x) = exp x
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  @fuzz_scale String.to_integer(System.get_env("FUZZ_SCALE", "1"))

  import Nx.Defn
  import Nx.Testing

  defn cond_fun(x, flag) do
    if flag > 0 do
      Nx.sum(Nx.sin(x))
    else
      Nx.sum(Nx.cos(x))
    end
  end

  defn grad_cond_fun(x, flag), do: grad(x, &cond_fun(&1, flag))

  defn nested_cond(x, f1, f2) do
    if f1 > 0 do
      if f2 > 0 do
        Nx.sum(Nx.multiply(x, x))
      else
        Nx.sum(Nx.exp(x))
      end
    else
      Nx.sum(Nx.sin(x))
    end
  end

  defn grad_nested_cond(x, f1, f2), do: grad(x, &nested_cond(&1, f1, f2))

  defn data_dependent_pred(x) do
    if Nx.sum(x) > 0 do
      Nx.sum(Nx.multiply(x, x))
    else
      Nx.sum(Nx.negate(x))
    end
  end

  defn grad_data_dependent_pred(x), do: grad(x, &data_dependent_pred/1)

  defn grad_sin_sum(x), do: grad(x, &Nx.sum(Nx.sin(&1)))

  defp float_tensor(shape) do
    count = Tuple.product(shape)

    bind(list_of(float(min: -3.0, max: 3.0), length: count), fn vals ->
      constant(vals |> Nx.tensor(type: {:f, 64}) |> Nx.reshape(shape))
    end)
  end

  describe "cond under grad" do
    property "grad through cond equals the taken branch's closed-form derivative" do
      check all(x <- float_tensor({4}), flag <- member_of([-1, 1]), max_runs: 20 * @fuzz_scale) do
        result = grad_cond_fun(x, Nx.tensor(flag))

        expected = if flag > 0, do: Nx.cos(x), else: Nx.negate(Nx.sin(x))
        assert_all_close(result, expected, atol: 1.0e-9)
      end
    end

    property "grad through nested cond selects the right closed form" do
      check all(
              x <- float_tensor({4}),
              f1 <- member_of([-1, 1]),
              f2 <- member_of([-1, 1]),
              max_runs: 20 * @fuzz_scale
            ) do
        result = grad_nested_cond(x, Nx.tensor(f1), Nx.tensor(f2))

        expected =
          cond do
            f1 > 0 and f2 > 0 -> Nx.multiply(x, 2.0)
            f1 > 0 -> Nx.exp(x)
            true -> Nx.cos(x)
          end

        assert_all_close(result, expected, atol: 1.0e-9)
      end
    end

    property "grad with a data-dependent predicate follows the taken branch" do
      check all(x <- float_tensor({4}), max_runs: 20 * @fuzz_scale) do
        result = grad_data_dependent_pred(x)

        expected =
          if Nx.to_number(Nx.sum(x)) > 0 do
            Nx.multiply(x, 2.0)
          else
            Nx.broadcast(Nx.tensor(-1.0, type: {:f, 64}), Nx.shape(x))
          end

        assert_all_close(result, expected, atol: 1.0e-9)
      end
    end
  end

  describe "multi-axis vectorization (forward)" do
    property "reduction over a doubly-vectorized tensor equals a plain axis reduction" do
      check all(
              a <- integer(2..3),
              b <- integer(2..3),
              n <- integer(2..4),
              max_runs: 15 * @fuzz_scale
            ) do
        check all(t <- float_tensor({a, b, n}), max_runs: 1) do
          vec = Nx.vectorize(t, [:a, :b])

          for {op, axes_fun} <- [
                {&Nx.sum/1, fn p -> Nx.sum(p, axes: [2]) end},
                {&Nx.mean/1, fn p -> Nx.mean(p, axes: [2]) end},
                {&Nx.reduce_max/1, fn p -> Nx.reduce_max(p, axes: [2]) end}
              ] do
            vectorized = op.(vec) |> Nx.devectorize(keep_names: false)
            assert_all_close(vectorized, axes_fun.(t), atol: 1.0e-9)
          end
        end
      end
    end

    property "binary op between two doubly-vectorized tensors matches the plain op" do
      check all(
              a <- integer(2..3),
              b <- integer(2..3),
              n <- integer(2..4),
              max_runs: 15 * @fuzz_scale
            ) do
        check all(t <- float_tensor({a, b, n}), u <- float_tensor({a, b, n}), max_runs: 1) do
          vt = Nx.vectorize(t, [:a, :b])
          vu = Nx.vectorize(u, [:a, :b])

          result = Nx.multiply(vt, vu) |> Nx.devectorize(keep_names: false)
          assert_all_close(result, Nx.multiply(t, u), atol: 1.0e-9)
        end
      end
    end
  end

  describe "multi-axis vectorized grad" do
    property "grad over a doubly-vectorized input matches the closed form per slice" do
      check all(
              a <- integer(2..3),
              b <- integer(2..3),
              n <- integer(2..4),
              max_runs: 15 * @fuzz_scale
            ) do
        check all(t <- float_tensor({a, b, n}), max_runs: 1) do
          vec = Nx.vectorize(t, [:a, :b])

          result = grad_sin_sum(vec) |> Nx.devectorize(keep_names: false)

          # closed form: d/dx sum(sin x) = cos x, per batch slice
          assert_all_close(result, Nx.cos(t), atol: 1.0e-9)
        end
      end
    end

    property "grad through cond with a doubly-vectorized input follows the taken branch" do
      check all(
              a <- integer(2..3),
              b <- integer(2..3),
              n <- integer(2..4),
              flag <- member_of([-1, 1]),
              max_runs: 15 * @fuzz_scale
            ) do
        check all(t <- float_tensor({a, b, n}), max_runs: 1) do
          vec = Nx.vectorize(t, [:a, :b])

          result = grad_cond_fun(vec, Nx.tensor(flag)) |> Nx.devectorize(keep_names: false)

          expected = if flag > 0, do: Nx.cos(t), else: Nx.negate(Nx.sin(t))
          assert_all_close(result, expected, atol: 1.0e-9)
        end
      end
    end
  end
end
