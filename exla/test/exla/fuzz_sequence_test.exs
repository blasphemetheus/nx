defmodule EXLA.FuzzSequenceTest do
  @moduledoc """
  Tier 5: Cross-backend sequence testing.
  Runs the same op chain on BinaryBackend and EXLA, compares results.
  """
  use ExUnit.Case, async: false
  use ExUnitProperties

  import Nx.Testing
  import Nx.Defn

  # ── Fixed op chains as defn functions ──────────────────────────────
  # We define fixed chains as defn functions so they can be JIT'd by EXLA

  defn chain_math(t) do
    t
    |> Nx.multiply(2)
    |> Nx.add(1)
    |> Nx.tanh()
    |> Nx.abs()
  end

  defn chain_trig(t) do
    t
    |> Nx.sin()
    |> Nx.cos()
    |> Nx.multiply(3)
    |> Nx.subtract(1)
  end

  defn chain_reduce(t) do
    t
    |> Nx.multiply(2)
    |> Nx.add(1)
    |> Nx.sum()
  end

  defn chain_reshape_reduce(t) do
    t
    |> Nx.flatten()
    |> Nx.multiply(2)
    |> Nx.sum()
  end

  defn chain_exp_log(t) do
    t
    |> Nx.abs()
    |> Nx.add(1)
    |> Nx.log()
    |> Nx.exp()
    |> Nx.subtract(1)
  end

  defn chain_sigmoid(t) do
    t
    |> Nx.subtract(Nx.mean(t))
    |> Nx.sigmoid()
  end

  defn chain_sort_reverse(t) do
    t
    |> Nx.sort()
    |> Nx.reverse()
    |> Nx.multiply(2)
  end

  defn chain_window(t) do
    t
    |> Nx.multiply(1)
    |> Nx.window_sum({2})
    |> Nx.add(1)
  end

  defn chain_cumulative(t) do
    t
    |> Nx.cumulative_sum()
    |> Nx.subtract(Nx.mean(t))
  end

  defn chain_multi_step(t) do
    a = Nx.multiply(t, 2)
    b = Nx.add(t, 1)
    Nx.add(a, b)
  end

  # ── Compare helper ─────────────────────────────────────────────────

  defp compare_chain(defn_fun, inputs, opts \\ []) do
    atol = opts[:atol] || 1.0e-4
    rtol = opts[:rtol] || 1.0e-4

    binary_inputs = Enum.map(inputs, &Nx.backend_transfer(&1, Nx.BinaryBackend))

    binary_result =
      Nx.with_default_backend(Nx.BinaryBackend, fn ->
        apply(defn_fun, binary_inputs)
      end)

    exla_fun = Nx.Defn.jit(defn_fun, compiler: EXLA, client: :host)
    exla_result = apply(exla_fun, binary_inputs) |> Nx.backend_transfer(Nx.BinaryBackend)

    assert_all_close(binary_result, exla_result, atol: atol, rtol: rtol)
  end

  # ── Tests ──────────────────────────────────────────────────────────

  describe "cross-backend op chains" do
    property "math chain agrees" do
      check all(n <- integer(1..8), max_runs: 15) do
        t = Nx.iota({n}, type: :f32)
        compare_chain(&chain_math/1, [t])
      end
    end

    property "trig chain agrees" do
      check all(n <- integer(1..8), max_runs: 15) do
        t = Nx.iota({n}, type: :f32)
        compare_chain(&chain_trig/1, [t])
      end
    end

    property "reduction chain agrees" do
      check all(
              m <- integer(1..5),
              n <- integer(1..5),
              max_runs: 15
            ) do
        t = Nx.iota({m, n}, type: :f32)
        compare_chain(&chain_reduce/1, [t])
      end
    end

    property "reshape + reduce chain agrees" do
      check all(
              m <- integer(1..5),
              n <- integer(1..5),
              max_runs: 15
            ) do
        t = Nx.iota({m, n}, type: :f32)
        compare_chain(&chain_reshape_reduce/1, [t])
      end
    end

    property "exp/log chain agrees" do
      check all(n <- integer(1..8), max_runs: 15) do
        t = Nx.iota({n}, type: :f32)
        compare_chain(&chain_exp_log/1, [t], atol: 1.0e-3, rtol: 1.0e-3)
      end
    end

    property "sigmoid chain agrees" do
      check all(n <- integer(2..8), max_runs: 15) do
        t = Nx.iota({n}, type: :f32)
        compare_chain(&chain_sigmoid/1, [t])
      end
    end

    property "sort + reverse chain agrees" do
      check all(n <- integer(2..8), max_runs: 15) do
        t = Nx.subtract(Nx.tensor(n, type: :f32), Nx.iota({n}, type: :f32))
        compare_chain(&chain_sort_reverse/1, [t])
      end
    end

    property "window chain agrees" do
      check all(n <- integer(3..10), max_runs: 15) do
        t = Nx.iota({n}, type: :f32)
        compare_chain(&chain_window/1, [t])
      end
    end

    property "cumulative chain agrees" do
      check all(n <- integer(2..8), max_runs: 15) do
        t = Nx.iota({n}, type: :f32)
        compare_chain(&chain_cumulative/1, [t], atol: 1.0e-3, rtol: 1.0e-3)
      end
    end

    property "multi-step (two paths merged) chain agrees" do
      check all(n <- integer(1..8), max_runs: 15) do
        t = Nx.iota({n}, type: :f32)
        compare_chain(&chain_multi_step/1, [t])
      end
    end
  end

  describe "cross-backend 2D chains" do
    property "math chain on 2D agrees" do
      check all(
              m <- integer(2..5),
              n <- integer(2..5),
              max_runs: 10
            ) do
        t = Nx.iota({m, n}, type: :f32)
        compare_chain(&chain_math/1, [t])
      end
    end

    property "sigmoid on 2D agrees" do
      check all(
              m <- integer(2..5),
              n <- integer(2..5),
              max_runs: 10
            ) do
        t = Nx.iota({m, n}, type: :f32)
        compare_chain(&chain_sigmoid/1, [t])
      end
    end
  end
end
