defmodule Nx.FuzzWhileGradTest do
  @moduledoc """
  Fuzz tests for gradients through `Nx.Defn.while` loops.

  `while` is heavily used internally (QR iterations, eigendecomposition,
  Cholesky) but rarely appears in user code today. The grad path through
  a while loop has to reverse-accumulate across dynamic iteration counts
  which is algorithmically tricky and often under-tested.

  Note on `defn` syntax: while loops cannot access outer-scope variables
  directly; all needed values must be threaded through the state tuple.
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  import Nx.Defn
  import Nx.Testing

  # ── defn functions under test ──────────────────────────────────────

  # Returns x + x + ... + x (n_steps times) = n_steps * x.
  # Gradient wrt x is n_steps.
  defn sum_n_copies(x, n_steps) do
    {acc, _, _} =
      while {acc = Nx.multiply(x, 0.0), i = 0, x = x}, Nx.less(i, n_steps) do
        {Nx.add(acc, x), i + 1, x}
      end

    acc
  end

  # Returns x * x * ... * x (n_steps times) = x^n_steps.
  # Gradient wrt x is n_steps * x^(n_steps - 1).
  defn power_via_loop(x, n_steps) do
    {acc, _, _} =
      while {acc = Nx.divide(x, x), i = 0, x = x}, Nx.less(i, n_steps) do
        {Nx.multiply(acc, x), i + 1, x}
      end

    acc
  end

  # sum of i * x for i in 0..n-1 = n*(n-1)/2 * x.
  # Gradient wrt x is n*(n-1)/2.
  defn weighted_sum(x, n_steps) do
    {acc, _, _} =
      while {acc = Nx.multiply(x, 0.0), i = 0, x = x}, Nx.less(i, n_steps) do
        {Nx.add(acc, Nx.multiply(x, i)), i + 1, x}
      end

    acc
  end

  # Loop-driven array accumulation: sum up elements of t.
  defn sum_of_elements(t) do
    n = Nx.axis_size(t, 0)

    {acc, _, _} =
      while {acc = Nx.tensor(0.0), i = 0, t = t}, Nx.less(i, n) do
        {Nx.add(acc, t[i]), i + 1, t}
      end

    acc
  end

  # ── Tests ──────────────────────────────────────────────────────────

  describe "grad through simple while loops (scalar state)" do
    property "d/dx sum_n_copies(x, n) == n" do
      check all(
              x <- float(min: -5.0, max: 5.0),
              n <- integer(1..5),
              max_runs: 8
            ) do
        t = Nx.tensor(x, type: :f32)
        grad = Nx.Defn.grad(t, fn a -> sum_n_copies(a, n) end)
        assert_all_close(grad, Nx.tensor(n * 1.0, type: :f32), atol: 1.0e-3)
      end
    end

    property "d/dx power_via_loop(x, n) == n * x^(n-1)" do
      check all(
              x <- float(min: 0.5, max: 2.0),
              n <- integer(1..4),
              max_runs: 8
            ) do
        t = Nx.tensor(x, type: :f32)
        grad = Nx.Defn.grad(t, fn a -> power_via_loop(a, n) end)
        expected = Nx.tensor(n * :math.pow(x, n - 1), type: :f32)
        assert_all_close(grad, expected, atol: 1.0e-2)
      end
    end

    property "d/dx weighted_sum(x, n) == n*(n-1)/2" do
      check all(
              x <- float(min: -3.0, max: 3.0),
              n <- integer(1..6),
              max_runs: 8
            ) do
        t = Nx.tensor(x, type: :f32)
        grad = Nx.Defn.grad(t, fn a -> weighted_sum(a, n) end)
        expected = Nx.tensor(n * (n - 1) / 2.0, type: :f32)
        assert_all_close(grad, expected, atol: 1.0e-3)
      end
    end
  end

  describe "grad through while with tensor state" do
    property "d/dt sum_of_elements(t) is all ones" do
      check all(n <- integer(2..5), max_runs: 6) do
        t = Nx.iota({n}, type: :f32)
        grad = Nx.Defn.grad(t, fn a -> sum_of_elements(a) end)
        expected = Nx.broadcast(Nx.tensor(1.0, type: :f32), {n})
        assert_all_close(grad, expected, atol: 1.0e-4)
      end
    end
  end

  describe "grad through zero-iteration while" do
    test "d/dx sum_n_copies(x, 0) == 0" do
      t = Nx.tensor(2.0, type: :f32)
      grad = Nx.Defn.grad(t, fn a -> sum_n_copies(a, 0) end)
      assert_all_close(grad, Nx.tensor(0.0, type: :f32), atol: 1.0e-4)
    end
  end

  describe "grad through nested while" do
    defn nested_loops(x, outer, inner) do
      {acc, _, _, _} =
        while {acc = Nx.multiply(x, 0.0), o = 0, x = x, inner = inner},
              Nx.less(o, outer) do
          {inner_acc, _, _} =
            while {ia = Nx.multiply(x, 0.0), i = 0, x = x}, Nx.less(i, inner) do
              {Nx.add(ia, x), i + 1, x}
            end

          {Nx.add(acc, inner_acc), o + 1, x, inner}
        end

      acc
    end

    test "d/dx nested_loops(x, 3, 4) == 12" do
      t = Nx.tensor(1.5, type: :f32)
      grad = Nx.Defn.grad(t, fn a -> nested_loops(a, 3, 4) end)
      assert_all_close(grad, Nx.tensor(12.0, type: :f32), atol: 1.0e-3)
    end
  end

  describe "grad composition with while" do
    defn sin_of_loop_sum(x, n_steps) do
      looped = sum_n_copies(x, n_steps)
      Nx.sin(looped)
    end

    property "d/dx sin(sum_n_copies(x, n)) == n * cos(n*x)" do
      check all(
              x <- float(min: -1.0, max: 1.0),
              n <- integer(1..4),
              max_runs: 8
            ) do
        t = Nx.tensor(x, type: :f32)
        grad = Nx.Defn.grad(t, fn a -> sin_of_loop_sum(a, n) end)
        expected = Nx.tensor(n * :math.cos(n * x), type: :f32)
        assert_all_close(grad, expected, atol: 1.0e-3)
      end
    end
  end

  describe "grad through while with batched input" do
    defn batched_loop(x, n_steps) do
      {acc, _, _} =
        while {acc = Nx.multiply(x, 0.0), i = 0, x = x}, Nx.less(i, n_steps) do
          {Nx.add(acc, x), i + 1, x}
        end

      Nx.sum(acc)
    end

    property "d/dt batched_loop(t, n) is all n's" do
      check all(
              rows <- integer(2..4),
              cols <- integer(2..4),
              n <- integer(1..3),
              max_runs: 6
            ) do
        t = Nx.iota({rows, cols}, type: :f32)
        grad = Nx.Defn.grad(t, fn a -> batched_loop(a, n) end)
        expected = Nx.broadcast(Nx.tensor(n * 1.0, type: :f32), {rows, cols})
        assert_all_close(grad, expected, atol: 1.0e-3)
      end
    end
  end
end
