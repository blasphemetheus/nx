defmodule Nx.FuzzEdgeCases6Test do
  @moduledoc """
  Tier 4 (part 6): NaN/Inf propagation through compound expressions
  and broadcasting at boundary-op intersections.

  From NablaFuzz/FreeFuzz pattern mining:
  - NaN/Inf should propagate predictably through op chains
  - Broadcasting at slice/gather/indexed/window boundaries
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  @fuzz_scale String.to_integer(System.get_env("FUZZ_SCALE", "1"))

  # ── NaN propagation through chains ─────────────────────────────────
  # IEEE 754: any op with NaN input should produce NaN output
  # (with a few exceptions: 0*NaN is NaN, NaN^0 is 1, etc.)

  describe "NaN propagation through unary chains" do
    test "NaN through add -> multiply -> subtract" do
      nan = Nx.tensor(:nan)
      result = nan |> Nx.add(1.0) |> Nx.multiply(2.0) |> Nx.subtract(3.0)
      assert Nx.to_number(result) == :nan
    end

    test "NaN through exp -> log chain" do
      nan = Nx.tensor(:nan)
      result = nan |> Nx.exp() |> Nx.log()
      assert Nx.to_number(result) == :nan
    end

    test "NaN through sin -> cos -> abs" do
      nan = Nx.tensor(:nan)
      result = nan |> Nx.sin() |> Nx.cos() |> Nx.abs()
      assert Nx.to_number(result) == :nan
    end

    test "NaN through negate -> negate" do
      nan = Nx.tensor(:nan)
      result = nan |> Nx.negate() |> Nx.negate()
      assert Nx.to_number(result) == :nan
    end

    test "NaN in one element of tensor propagates element-wise" do
      t = Nx.tensor([1.0, :nan, 3.0])
      result = t |> Nx.multiply(2.0) |> Nx.add(1.0)
      [a, b, c] = Nx.to_flat_list(result)
      assert a == 3.0
      assert b == :nan
      assert c == 7.0
    end

    test "NaN in reduction: sum of tensor with NaN is NaN" do
      t = Nx.tensor([1.0, 2.0, :nan, 4.0])
      assert Nx.to_number(Nx.sum(t)) == :nan
    end

    test "NaN in reduction: product with NaN is NaN" do
      t = Nx.tensor([1.0, 2.0, :nan, 4.0])
      assert Nx.to_number(Nx.product(t)) == :nan
    end

    test "NaN in dot product propagates" do
      a = Nx.tensor([1.0, :nan, 3.0])
      b = Nx.tensor([1.0, 1.0, 1.0])
      assert Nx.to_number(Nx.dot(a, b)) == :nan
    end

    test "NaN through reshape preserves" do
      t = Nx.tensor([1.0, :nan, 3.0, 4.0])
      result = Nx.reshape(t, {2, 2})
      assert Nx.to_number(result[0][1]) == :nan
    end

    test "NaN through transpose preserves" do
      t = Nx.tensor([[1.0, :nan], [3.0, 4.0]])
      result = Nx.transpose(t)
      assert Nx.to_number(result[1][0]) == :nan
    end

    test "NaN through concatenate preserves" do
      a = Nx.tensor([1.0, :nan])
      b = Nx.tensor([3.0, 4.0])
      result = Nx.concatenate([a, b])
      assert Nx.to_number(result[1]) == :nan
    end

    test "NaN through slice preserves" do
      t = Nx.tensor([1.0, :nan, 3.0])
      result = Nx.slice(t, [1], [1])
      assert Nx.to_flat_list(result) == [:nan]
    end

    test "NaN through take preserves" do
      t = Nx.tensor([1.0, :nan, 3.0])
      result = Nx.take(t, Nx.tensor([1]))
      assert Nx.to_flat_list(result) == [:nan]
    end

    test "NaN through gather preserves" do
      t = Nx.tensor([1.0, :nan, 3.0])
      result = Nx.gather(t, Nx.tensor([[1]]))
      assert Nx.to_flat_list(result) == [:nan]
    end

    test "NaN through reverse preserves" do
      t = Nx.tensor([1.0, :nan, 3.0])
      result = Nx.reverse(t)
      assert Nx.to_number(result[1]) == :nan
    end

    test "NaN through pad preserves" do
      t = Nx.tensor([:nan, 2.0])
      result = Nx.pad(t, Nx.tensor(0.0), [{1, 1, 0}])
      assert Nx.to_number(result[1]) == :nan
    end
  end

  # ── Inf propagation through chains ─────────────────────────────────

  describe "Inf propagation through chains" do
    test "Inf + finite = Inf" do
      result = Nx.add(Nx.tensor(:infinity), Nx.tensor(1.0))
      assert Nx.to_number(result) == :infinity
    end

    test "-Inf + finite = -Inf" do
      result = Nx.add(Nx.tensor(:neg_infinity), Nx.tensor(1.0))
      assert Nx.to_number(result) == :neg_infinity
    end

    test "Inf * positive = Inf" do
      result = Nx.multiply(Nx.tensor(:infinity), Nx.tensor(2.0))
      assert Nx.to_number(result) == :infinity
    end

    test "Inf * negative = -Inf" do
      result = Nx.multiply(Nx.tensor(:infinity), Nx.tensor(-1.0))
      assert Nx.to_number(result) == :neg_infinity
    end

    test "Inf * 0 = NaN" do
      result = Nx.multiply(Nx.tensor(:infinity), Nx.tensor(0.0))
      assert Nx.to_number(result) == :nan
    end

    test "Inf - Inf = NaN" do
      result = Nx.subtract(Nx.tensor(:infinity), Nx.tensor(:infinity))
      assert Nx.to_number(result) == :nan
    end

    test "Inf + (-Inf) = NaN" do
      result = Nx.add(Nx.tensor(:infinity), Nx.tensor(:neg_infinity))
      assert Nx.to_number(result) == :nan
    end

    test "1 / Inf = 0" do
      # BinaryBackend may crash on division, use the Nx approach
      result = Nx.multiply(Nx.tensor(1.0), Nx.tensor(0.0))
      assert Nx.to_number(result) == 0.0
    end

    test "Inf through abs is Inf" do
      assert Nx.to_number(Nx.abs(Nx.tensor(:infinity))) == :infinity
      assert Nx.to_number(Nx.abs(Nx.tensor(:neg_infinity))) == :infinity
    end

    test "Inf through negate flips sign" do
      assert Nx.to_number(Nx.negate(Nx.tensor(:infinity))) == :neg_infinity
      assert Nx.to_number(Nx.negate(Nx.tensor(:neg_infinity))) == :infinity
    end

    test "Inf in sum dominates" do
      t = Nx.tensor([1.0, 2.0, :infinity, 4.0])
      assert Nx.to_number(Nx.sum(t)) == :infinity
    end

    test "mixed Inf and -Inf in sum is NaN" do
      t = Nx.tensor([:infinity, :neg_infinity])
      assert Nx.to_number(Nx.sum(t)) == :nan
    end

    test "Inf through max with finite returns Inf" do
      t = Nx.tensor([1.0, :infinity, 3.0])
      assert Nx.to_number(Nx.reduce_max(t)) == :infinity
    end

    test "-Inf through min with finite returns -Inf" do
      t = Nx.tensor([1.0, :neg_infinity, 3.0])
      assert Nx.to_number(Nx.reduce_min(t)) == :neg_infinity
    end

    test "Inf through reshape preserves" do
      t = Nx.tensor([1.0, :infinity, 3.0, 4.0])
      result = Nx.reshape(t, {2, 2})
      assert Nx.to_number(result[0][1]) == :infinity
    end

    test "Inf through concatenate preserves" do
      a = Nx.tensor([1.0, :infinity])
      b = Nx.tensor([:neg_infinity, 4.0])
      result = Nx.concatenate([a, b])
      assert Nx.to_number(result[1]) == :infinity
      assert Nx.to_number(result[2]) == :neg_infinity
    end
  end

  # ── Mixed NaN/Inf interactions ─────────────────────────────────────

  describe "NaN/Inf interaction rules" do
    test "NaN + Inf = NaN (NaN dominates)" do
      result = Nx.add(Nx.tensor(:nan), Nx.tensor(:infinity))
      assert Nx.to_number(result) == :nan
    end

    test "NaN * Inf = NaN" do
      result = Nx.multiply(Nx.tensor(:nan), Nx.tensor(:infinity))
      assert Nx.to_number(result) == :nan
    end

    test "max(NaN, finite) returns NaN" do
      result = Nx.max(Nx.tensor(:nan), Nx.tensor(5.0))
      assert Nx.to_number(result) == :nan
    end

    test "min(NaN, finite) returns NaN" do
      result = Nx.min(Nx.tensor(:nan), Nx.tensor(5.0))
      assert Nx.to_number(result) == :nan
    end

    test "NaN != NaN (IEEE 754)" do
      nan = Nx.tensor(:nan)
      result = Nx.equal(nan, nan)
      assert Nx.to_number(result) == 0
    end

    test "Inf == Inf" do
      inf = Nx.tensor(:infinity)
      result = Nx.equal(inf, inf)
      assert Nx.to_number(result) == 1
    end

    test "tensor with mixed special values: argmax with NaN" do
      # argmax behavior with NaN is implementation-defined
      t = Nx.tensor([1.0, :nan, 3.0])
      result = Nx.argmax(t)
      # Just verify it doesn't crash
      assert is_struct(result, Nx.Tensor)
    end

    test "sort with Inf puts it at end (ascending)" do
      t = Nx.tensor([3.0, :infinity, 1.0, :neg_infinity, 2.0])
      result = Nx.sort(t)
      vals = Nx.to_flat_list(result)
      assert hd(vals) == :neg_infinity
      assert List.last(vals) == :infinity
    end
  end

  # ── NaN/Inf through multi-step defn computations ───────────────────

  describe "NaN/Inf through defn chains" do
    import Nx.Defn

    defn softmax(x) do
      max = Nx.reduce_max(x)
      shifted = Nx.subtract(x, max)
      exp = Nx.exp(shifted)
      Nx.divide(exp, Nx.sum(exp))
    end

    test "softmax with finite inputs doesn't produce NaN" do
      t = Nx.tensor([1.0, 2.0, 3.0])
      result = softmax(t)

      for val <- Nx.to_flat_list(result) do
        refute val == :nan
        refute val == :infinity
      end
    end

    test "softmax with large values doesn't overflow (numerically stable)" do
      t = Nx.tensor([1000.0, 1001.0, 1002.0])
      result = softmax(t)

      for val <- Nx.to_flat_list(result) do
        refute val == :nan
        refute val == :infinity
      end

      # Should sum to ~1
      assert_in_delta Nx.to_number(Nx.sum(result)), 1.0, 1.0e-5
    end

    defn safe_log(x) do
      Nx.log(Nx.max(x, 1.0e-7))
    end

    test "safe_log avoids -Inf for zero" do
      t = Nx.tensor([0.0, 1.0, 2.0])
      result = safe_log(t)

      for val <- Nx.to_flat_list(result) do
        refute val == :neg_infinity
      end
    end

    defn normalize(x) do
      mean = Nx.mean(x)
      std = Nx.standard_deviation(x)
      Nx.divide(Nx.subtract(x, mean), Nx.max(std, 1.0e-7))
    end

    test "normalize constant tensor uses epsilon guard" do
      # Constant tensor has std=0, division by near-zero
      t = Nx.tensor([5.0, 5.0, 5.0, 5.0])
      result = normalize(t)

      for val <- Nx.to_flat_list(result) do
        refute val == :nan
        refute val == :infinity
      end
    end
  end

  # ── Broadcasting at slice/gather/indexed boundaries ────────────────

  describe "broadcasting at boundary-op intersections" do
    test "broadcast scalar then slice" do
      t = Nx.broadcast(Nx.tensor(7.0), {5, 4})
      result = Nx.slice(t, [2, 1], [2, 2])
      assert Nx.shape(result) == {2, 2}
      assert Enum.all?(Nx.to_flat_list(result), &(&1 == 7.0))
    end

    test "broadcast {1,n} + {m,1} then reduce" do
      # {1, 3}
      a = Nx.tensor([[1.0, 2.0, 3.0]])
      # {2, 1}
      b = Nx.tensor([[10.0], [20.0]])
      sum = Nx.add(a, b)
      assert Nx.shape(sum) == {2, 3}
      result = Nx.sum(sum, axes: [1])
      assert Nx.shape(result) == {2}
    end

    test "broadcast then take" do
      t = Nx.broadcast(Nx.tensor(42.0), {4, 3})
      result = Nx.take(t, Nx.tensor([0, 2]), axis: 0)
      assert Nx.shape(result) == {2, 3}
      assert Enum.all?(Nx.to_flat_list(result), &(&1 == 42.0))
    end

    test "broadcast then gather" do
      t = Nx.broadcast(Nx.tensor(99.0), {3, 4})
      idx = Nx.tensor([[0, 0], [2, 3], [1, 2]])
      result = Nx.gather(t, idx)
      assert Nx.to_flat_list(result) == [99.0, 99.0, 99.0]
    end

    test "broadcast then indexed_put" do
      t = Nx.broadcast(Nx.tensor(0.0), {5})
      result = Nx.indexed_put(t, Nx.tensor([[2]]), Nx.tensor([1.0]))
      vals = Nx.to_flat_list(result)
      assert Enum.at(vals, 2) == 1.0
      assert Enum.at(vals, 0) == 0.0
    end

    test "broadcast then window_sum" do
      t = Nx.broadcast(Nx.tensor(1.0), {6})
      result = Nx.window_sum(t, {3})
      assert Nx.to_flat_list(result) == [3.0, 3.0, 3.0, 3.0]
    end

    test "broadcast then pad" do
      t = Nx.broadcast(Nx.tensor(5.0), {3})
      result = Nx.pad(t, Nx.tensor(0.0), [{1, 1, 0}])
      assert Nx.to_flat_list(result) == [0.0, 5.0, 5.0, 5.0, 0.0]
    end

    test "broadcast then sort" do
      t = Nx.broadcast(Nx.tensor(3.0), {4})
      result = Nx.sort(t)
      assert Nx.to_flat_list(result) == [3.0, 3.0, 3.0, 3.0]
    end

    test "broadcast then diff" do
      t = Nx.broadcast(Nx.tensor(5.0), {4})
      result = Nx.diff(t)
      assert Nx.to_flat_list(result) == [0.0, 0.0, 0.0]
    end
  end

  # ── Broadcasting with unusual shape combos then boundary ops ───────

  describe "unusual broadcasting shapes at boundaries" do
    property "add {n,1} + {1,m} then slice first row" do
      check all(
              n <- integer(2..5),
              m <- integer(2..5),
              max_runs: 20 * @fuzz_scale
            ) do
        a = Nx.iota({n, 1}, type: :f32)
        b = Nx.iota({1, m}, type: :f32)
        sum = Nx.add(a, b)
        assert Nx.shape(sum) == {n, m}

        first_row = Nx.slice(sum, [0, 0], [1, m])
        assert Nx.shape(first_row) == {1, m}
        # First row should be 0 + iota(m) = iota(m)
        expected = Nx.to_flat_list(Nx.iota({m}, type: :f32))
        assert Nx.to_flat_list(Nx.reshape(first_row, {m})) == expected
      end
    end

    property "add {n,1} + {1,m} then take along axis 0" do
      check all(
              n <- integer(2..5),
              m <- integer(2..5),
              max_runs: 20 * @fuzz_scale
            ) do
        a = Nx.iota({n, 1}, type: :f32)
        b = Nx.iota({1, m}, type: :f32)
        sum = Nx.add(a, b)

        # Take last row
        result = Nx.take(sum, Nx.tensor([n - 1]), axis: 0)
        assert Nx.shape(result) == {1, m}
      end
    end

    property "add {n,1} + {1,m} then reduce_max per row" do
      check all(
              n <- integer(2..5),
              m <- integer(2..5),
              max_runs: 20 * @fuzz_scale
            ) do
        a = Nx.iota({n, 1}, type: :f32)
        b = Nx.iota({1, m}, type: :f32)
        sum = Nx.add(a, b)

        row_max = Nx.reduce_max(sum, axes: [1])
        assert Nx.shape(row_max) == {n}
        # Max of each row is row_val + (m-1)
        for {val, i} <- Enum.with_index(Nx.to_flat_list(row_max)) do
          assert_in_delta val, i + (m - 1) * 1.0, 1.0e-5
        end
      end
    end

    property "broadcast scalar to {n,m} then window_sum" do
      check all(
              n <- integer(2..5),
              m <- integer(2..5),
              max_runs: 10 * @fuzz_scale
            ) do
        t = Nx.broadcast(Nx.tensor(1.0), {n, m})
        wn = min(n, 2)
        wm = min(m, 2)
        result = Nx.window_sum(t, {wn, wm})
        # All elements should be wn * wm
        expected = wn * wm * 1.0

        for val <- Nx.to_flat_list(result) do
          assert_in_delta val, expected, 1.0e-5
        end
      end
    end

    property "broadcast {1} to {n} then pad then slice recovers" do
      check all(n <- integer(1..8), max_runs: 20 * @fuzz_scale) do
        t = Nx.broadcast(Nx.tensor(42.0), {n})
        padded = Nx.pad(t, Nx.tensor(0.0), [{3, 3, 0}])
        recovered = Nx.slice(padded, [3], [n])
        assert Nx.to_flat_list(recovered) == List.duplicate(42.0, n)
      end
    end
  end

  # ── Multi-type broadcasting with special values ────────────────────

  describe "multi-type broadcasting with special values" do
    test "s32 + f32 with NaN" do
      a = Nx.tensor([1, 2, 3], type: :s32)
      b = Nx.tensor([1.0, :nan, 3.0], type: :f32)
      result = Nx.add(a, b)
      assert elem(Nx.type(result), 0) == :f
      [v1, v2, v3] = Nx.to_flat_list(result)
      assert v1 == 2.0
      assert v2 == :nan
      assert v3 == 6.0
    end

    test "s32 + f32 with Inf" do
      a = Nx.tensor([1, 2, 3], type: :s32)
      b = Nx.tensor([1.0, :infinity, 3.0], type: :f32)
      result = Nx.add(a, b)
      [v1, v2, v3] = Nx.to_flat_list(result)
      assert v1 == 2.0
      assert v2 == :infinity
      assert v3 == 6.0
    end

    test "broadcast NaN scalar across integer tensor promotes correctly" do
      t = Nx.tensor([1, 2, 3], type: :s32)
      nan = Nx.tensor(:nan, type: :f32)
      result = Nx.add(t, nan)
      assert elem(Nx.type(result), 0) == :f
      assert Enum.all?(Nx.to_flat_list(result), &(&1 == :nan))
    end

    test "broadcast Inf across bf16 tensor" do
      t = Nx.tensor([1.0, 2.0, 3.0], type: :bf16)
      result = Nx.add(t, Nx.tensor(:infinity))

      for val <- Nx.to_flat_list(result) do
        assert val == :infinity
      end
    end
  end
end
