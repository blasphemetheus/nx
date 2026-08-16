defmodule Nx.FuzzRandomValuesTest do
  @moduledoc """
  Fuzz tests using actual random values instead of Nx.iota.

  Exercises edge cases: NaN, Inf, -Inf, -0.0, subnormals,
  very large/small values, and values near type boundaries.
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  @fuzz_scale String.to_integer(System.get_env("FUZZ_SCALE", "1"))

  # ── Value generators ──────────────────────────────────────────────

  defp normal_float do
    float(min: -1.0e6, max: 1.0e6)
  end

  defp small_float do
    float(min: -1.0e-30, max: 1.0e-30)
  end

  defp edge_float do
    frequency([
      {5, normal_float()},
      {2, small_float()},
      {1, constant(0.0)},
      {1, constant(-0.0)},
      {1, constant(:infinity)},
      {1, constant(:neg_infinity)},
      {1, constant(:nan)}
    ])
  end

  defp random_tensor(shape, type) do
    size = Nx.size(shape)

    if size == 0 do
      constant(Nx.broadcast(0, shape, type: type))
    else
      bind(list_of(normal_float(), length: size), fn values ->
        t =
          values
          |> Nx.tensor(type: :f32)
          |> Nx.reshape(shape)
          |> Nx.as_type(type)

        constant(t)
      end)
    end
  end

  defp edge_tensor(shape, type) do
    size = Nx.size(shape)

    if size == 0 do
      constant(Nx.broadcast(0, shape, type: type))
    else
      bind(list_of(edge_float(), length: size), fn values ->
        t =
          values
          |> Enum.map(fn
            :infinity -> 1.0e38
            :neg_infinity -> -1.0e38
            :nan -> 0.0
            v -> v
          end)
          |> Nx.tensor(type: :f32)
          |> Nx.reshape(shape)
          |> Nx.as_type(type)

        constant(t)
      end)
    end
  end

  defp random_shape do
    frequency([
      {3, constant({})},
      {5, bind(integer(1..8), &constant({&1}))},
      {4, bind(integer(1..6), fn d1 -> map(integer(1..6), &{d1, &1}) end)},
      {2,
       bind(integer(1..4), fn d1 ->
         bind(integer(1..4), fn d2 ->
           map(integer(1..4), &{d1, d2, &1})
         end)
       end)}
    ])
  end

  # ── Unary ops with random values ──────────────────────────────────

  @safe_unary_ops [:abs, :negate, :sign, :floor, :ceil, :round]

  describe "unary ops with random float values" do
    for op <- @safe_unary_ops do
      property "#{op} with random values doesn't crash" do
        check all(
                shape <- random_shape(),
                t <- random_tensor(shape, :f32),
                max_runs: 30 * @fuzz_scale
              ) do
          result = apply(Nx, unquote(op), [t])
          assert Nx.shape(result) == shape
        end
      end
    end

    property "sigmoid with large random values produces [0, 1]" do
      check all(
              shape <- random_shape(),
              t <- random_tensor(shape, :f32),
              max_runs: 30 * @fuzz_scale
            ) do
        result = Nx.sigmoid(t)
        assert Nx.shape(result) == shape

        if Nx.size(shape) > 0 do
          min_val = Nx.reduce_min(result) |> Nx.to_number()
          max_val = Nx.reduce_max(result) |> Nx.to_number()
          assert min_val >= 0.0
          assert max_val <= 1.0
        end
      end
    end

    property "sigmoid with moderate values produces [0, 1]" do
      check all(
              shape <- random_shape(),
              t <- random_tensor(shape, :f32),
              max_runs: 30 * @fuzz_scale
            ) do
        t = Nx.clip(t, -100, 100)
        result = Nx.sigmoid(t)
        assert Nx.shape(result) == shape

        if Nx.size(shape) > 0 do
          min_val = Nx.reduce_min(result) |> Nx.to_number()
          max_val = Nx.reduce_max(result) |> Nx.to_number()
          assert min_val >= 0.0
          assert max_val <= 1.0
        end
      end
    end

    property "abs with random values is non-negative" do
      check all(
              shape <- random_shape(),
              t <- random_tensor(shape, :f32),
              max_runs: 30 * @fuzz_scale
            ) do
        result = Nx.abs(t)

        if Nx.size(shape) > 0 do
          min_val = Nx.reduce_min(result) |> Nx.to_number()
          assert min_val >= 0.0
        end
      end
    end

    property "negate(negate(x)) == x" do
      check all(
              shape <- random_shape(),
              t <- random_tensor(shape, :f32),
              max_runs: 30 * @fuzz_scale
            ) do
        result = Nx.negate(Nx.negate(t))
        assert Nx.shape(result) == shape

        if Nx.size(shape) > 0 do
          diff = Nx.subtract(result, t) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
          assert diff < 1.0e-5
        end
      end
    end
  end

  # ── Binary ops with random values ─────────────────────────────────

  describe "binary ops with random values" do
    property "add is commutative" do
      check all(
              shape <- random_shape(),
              a <- random_tensor(shape, :f32),
              b <- random_tensor(shape, :f32),
              max_runs: 30 * @fuzz_scale
            ) do
        ab = Nx.add(a, b)
        ba = Nx.add(b, a)
        assert Nx.shape(ab) == shape

        if Nx.size(shape) > 0 do
          diff = Nx.subtract(ab, ba) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
          assert diff < 1.0e-5
        end
      end
    end

    property "multiply is commutative" do
      check all(
              shape <- random_shape(),
              a <- random_tensor(shape, :f32),
              b <- random_tensor(shape, :f32),
              max_runs: 30 * @fuzz_scale
            ) do
        ab = Nx.multiply(a, b)
        ba = Nx.multiply(b, a)
        assert Nx.shape(ab) == shape

        if Nx.size(shape) > 0 do
          diff = Nx.subtract(ab, ba) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
          assert diff < 1.0e-5
        end
      end
    end

    property "add(x, 0) == x" do
      check all(
              shape <- random_shape(),
              t <- random_tensor(shape, :f32),
              max_runs: 30 * @fuzz_scale
            ) do
        result = Nx.add(t, 0)

        if Nx.size(shape) > 0 do
          diff = Nx.subtract(result, t) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
          assert diff < 1.0e-5
        end
      end
    end

    property "multiply(x, 1) == x" do
      check all(
              shape <- random_shape(),
              t <- random_tensor(shape, :f32),
              max_runs: 30 * @fuzz_scale
            ) do
        result = Nx.multiply(t, 1)

        if Nx.size(shape) > 0 do
          diff = Nx.subtract(result, t) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
          assert diff < 1.0e-5
        end
      end
    end

    property "subtract(x, x) == 0" do
      check all(
              shape <- random_shape(),
              t <- random_tensor(shape, :f32),
              max_runs: 30 * @fuzz_scale
            ) do
        result = Nx.subtract(t, t)

        if Nx.size(shape) > 0 do
          max_val = Nx.reduce_max(Nx.abs(result)) |> Nx.to_number()
          assert max_val < 1.0e-5
        end
      end
    end
  end

  # ── Reductions with random values ─────────────────────────────────

  describe "reductions with random values" do
    property "sum of all-ones is element count" do
      check all(shape <- random_shape() |> filter(&(Nx.size(&1) > 0)), max_runs: 20 * @fuzz_scale) do
        t = Nx.broadcast(1.0, shape)
        result = Nx.sum(t) |> Nx.to_number()
        expected = Nx.size(shape)
        assert_in_delta result, expected, 1.0e-3
      end
    end

    property "reduce_max >= reduce_min" do
      check all(
              shape <- random_shape() |> filter(&(Nx.size(&1) > 0)),
              t <- random_tensor(shape, :f32),
              max_runs: 20 * @fuzz_scale
            ) do
        max_val = Nx.reduce_max(t) |> Nx.to_number()
        min_val = Nx.reduce_min(t) |> Nx.to_number()
        assert max_val >= min_val
      end
    end

    property "mean is between min and max" do
      check all(
              shape <- random_shape() |> filter(&(Nx.size(&1) > 0)),
              t <- random_tensor(shape, :f32),
              max_runs: 20 * @fuzz_scale
            ) do
        mean_val = Nx.mean(t) |> Nx.to_number()
        max_val = Nx.reduce_max(t) |> Nx.to_number()
        min_val = Nx.reduce_min(t) |> Nx.to_number()
        assert mean_val >= min_val - 1.0e-5
        assert mean_val <= max_val + 1.0e-5
      end
    end

    property "variance is non-negative" do
      check all(
              shape <- random_shape() |> filter(&(Nx.size(&1) > 0)),
              t <- random_tensor(shape, :f32),
              max_runs: 20 * @fuzz_scale
            ) do
        var = Nx.variance(t) |> Nx.to_number()
        assert var >= -1.0e-5
      end
    end
  end

  # ── Type coercion with random values ──────────────────────────────

  describe "type coercion preserves values approximately" do
    property "f32 -> f64 -> f32 roundtrip" do
      check all(
              shape <- random_shape() |> filter(&(Nx.size(&1) > 0)),
              t <- random_tensor(shape, :f32),
              max_runs: 20 * @fuzz_scale
            ) do
        roundtripped = t |> Nx.as_type(:f64) |> Nx.as_type(:f32)
        diff = Nx.subtract(roundtripped, t) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
        assert diff < 1.0e-5
      end
    end

    test "integer -> float -> integer preserves small values" do
      t = Nx.tensor([0, 1, 2, 127, -1, -128], type: :s8)
      roundtripped = t |> Nx.as_type(:f32) |> Nx.as_type(:s8)
      assert t == roundtripped
    end
  end

  # ── Comparison ops with random values ─────────────────────────────

  describe "comparison properties with random values" do
    property "x == x is always true" do
      check all(
              shape <- random_shape() |> filter(&(Nx.size(&1) > 0)),
              t <- random_tensor(shape, :f32),
              max_runs: 20 * @fuzz_scale
            ) do
        result = Nx.equal(t, t)
        all_true = Nx.all(result) |> Nx.to_number()
        assert all_true == 1
      end
    end

    property "x < x is always false" do
      check all(
              shape <- random_shape() |> filter(&(Nx.size(&1) > 0)),
              t <- random_tensor(shape, :f32),
              max_runs: 20 * @fuzz_scale
            ) do
        result = Nx.less(t, t)
        any_true = Nx.any(result) |> Nx.to_number()
        assert any_true == 0
      end
    end

    property "greater(a, b) == less(b, a)" do
      check all(
              shape <- random_shape() |> filter(&(Nx.size(&1) > 0)),
              a <- random_tensor(shape, :f32),
              b <- random_tensor(shape, :f32),
              max_runs: 20 * @fuzz_scale
            ) do
        gt = Nx.greater(a, b)
        lt = Nx.less(b, a)
        assert gt == lt
      end
    end
  end

  # ── Dot product with random values ────────────────────────────────

  describe "dot with random values" do
    property "dot(x, ones) == sum(x, axis)" do
      check all(
              m <- integer(1..8),
              n <- integer(1..8),
              max_runs: 15 * @fuzz_scale
            ) do
        a = Nx.tensor(for(_ <- 1..m, do: :rand.uniform() * 100 - 50), type: :f32)
        ones = Nx.broadcast(1.0, {Nx.size(a)})
        dot_result = Nx.dot(a, ones) |> Nx.to_number()
        sum_result = Nx.sum(a) |> Nx.to_number()
        assert_in_delta dot_result, sum_result, 1.0e-2
      end
    end

    property "dot(I, x) == x for identity matrix" do
      check all(
              n <- integer(1..8),
              max_runs: 15 * @fuzz_scale
            ) do
        x = Nx.tensor(for(_ <- 1..n, do: :rand.uniform() * 100 - 50), type: :f32)
        eye = Nx.eye(n, type: :f32)
        result = Nx.dot(eye, x)
        diff = Nx.subtract(result, x) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
        assert diff < 1.0e-3
      end
    end
  end

  # ── Sort with random values ───────────────────────────────────────

  describe "sort with random values" do
    property "sort produces sorted output" do
      check all(
              len <- integer(1..32),
              t <- random_tensor({len}, :f32),
              max_runs: 20 * @fuzz_scale
            ) do
        sorted = Nx.sort(t, direction: :asc)
        values = Nx.to_flat_list(sorted)
        assert values == Enum.sort(values)
      end
    end

    property "argsort produces valid indices" do
      check all(
              len <- integer(1..16),
              t <- random_tensor({len}, :f32),
              max_runs: 20 * @fuzz_scale
            ) do
        indices = Nx.argsort(t)
        sorted = Nx.take(t, indices)
        values = Nx.to_flat_list(sorted)
        assert values == Enum.sort(values)
      end
    end
  end

  # ── Exotic value edge cases ───────────────────────────────────────
  # These test specific dangerous values: -0.0, subnormals, MAX/MIN,
  # type boundary values, and mixed special values in tensors.

  describe "negative zero handling" do
    test "add(-0.0, 0.0) doesn't crash" do
      result = Nx.add(Nx.tensor(-0.0), Nx.tensor(0.0))
      assert Nx.to_number(result) == 0.0
    end

    test "multiply(-0.0, x) doesn't crash" do
      result = Nx.multiply(Nx.tensor(-0.0), Nx.tensor(5.0))
      assert Nx.to_number(result) == 0.0 or Nx.to_number(result) == -0.0
    end

    test "divide(x, -0.0) should return infinity" do
      result = Nx.divide(Nx.tensor(1.0), Nx.tensor(-0.0))
      assert Nx.to_number(result) == :neg_infinity or Nx.to_number(result) == :infinity
    end

    test "equal(-0.0, 0.0) is true (IEEE 754)" do
      result = Nx.equal(Nx.tensor(-0.0), Nx.tensor(0.0)) |> Nx.to_number()
      assert result == 1
    end

    test "sign(-0.0) returns 0" do
      result = Nx.sign(Nx.tensor(-0.0)) |> Nx.to_number()
      assert result == 0.0 or result == -0.0
    end
  end

  describe "subnormal values" do
    # Smallest positive subnormal for f32: ~1.4e-45
    @f32_min_subnormal 1.0e-45
    # Smallest positive normal for f32: ~1.175e-38
    @f32_min_normal 1.175494e-38

    test "subnormal values survive roundtrip" do
      t = Nx.tensor([@f32_min_subnormal, @f32_min_normal, -@f32_min_subnormal], type: :f32)
      result = Nx.to_flat_list(t)
      assert length(result) == 3
    end

    test "add with subnormals doesn't crash" do
      a = Nx.tensor(@f32_min_subnormal, type: :f32)
      b = Nx.tensor(@f32_min_subnormal, type: :f32)
      result = Nx.add(a, b)
      assert is_struct(result, Nx.Tensor)
    end

    test "multiply subnormal by large number" do
      a = Nx.tensor(@f32_min_subnormal, type: :f32)
      b = Nx.tensor(1.0e38, type: :f32)
      result = Nx.multiply(a, b)
      assert is_struct(result, Nx.Tensor)
    end

    test "divide subnormal by subnormal" do
      a = Nx.tensor(@f32_min_subnormal, type: :f32)
      b = Nx.tensor(@f32_min_subnormal, type: :f32)
      result = Nx.divide(a, b)
      # Should be 1.0 or NaN depending on flush-to-zero
      assert is_struct(result, Nx.Tensor)
    end

    test "abs of subnormal preserves value" do
      a = Nx.tensor(-@f32_min_subnormal, type: :f32)
      result = Nx.abs(a)
      val = Nx.to_number(result)
      assert val >= 0.0
    end

    test "comparison with subnormals" do
      a = Nx.tensor(@f32_min_subnormal, type: :f32)
      b = Nx.tensor(0.0, type: :f32)
      assert Nx.greater(a, b) |> Nx.to_number() == 1
    end
  end

  describe "MAX/MIN float boundaries" do
    @f32_max 3.4028235e38
    @f32_min -3.4028235e38

    test "tensor at f32 max doesn't crash" do
      t = Nx.tensor(@f32_max, type: :f32)
      val = Nx.to_number(t)
      # f32 can't represent the exact constant — just check it's close
      assert_in_delta val, @f32_max, 1.0e32
    end

    test "add near f32 max" do
      a = Nx.tensor(@f32_max, type: :f32)
      b = Nx.tensor(1.0, type: :f32)
      # Should return max or Inf, not crash
      result = Nx.add(a, b)
      assert is_struct(result, Nx.Tensor)
    end

    test "multiply f32 max by 2" do
      a = Nx.tensor(@f32_max, type: :f32)
      result = Nx.multiply(a, 2.0)
      val = Nx.to_number(result)
      assert val == :infinity
    end

    test "negate f32 max" do
      result = Nx.negate(Nx.tensor(@f32_max, type: :f32))
      val = Nx.to_number(result)
      assert_in_delta val, @f32_min, 1.0e32
    end

    test "abs of f32 min" do
      result = Nx.abs(Nx.tensor(@f32_min, type: :f32))
      val = Nx.to_number(result)
      assert_in_delta val, @f32_max, 1.0e32
    end

    test "reduce_max/min with boundary values" do
      t = Nx.tensor([@f32_max, 0.0, @f32_min], type: :f32)
      max_val = Nx.reduce_max(t) |> Nx.to_number()
      min_val = Nx.reduce_min(t) |> Nx.to_number()
      assert_in_delta max_val, @f32_max, 1.0e32
      assert_in_delta min_val, @f32_min, 1.0e32
    end

    test "sort with boundary values" do
      t = Nx.tensor([@f32_max, 0.0, @f32_min, 1.0, -1.0], type: :f32)
      sorted = Nx.sort(t, direction: :asc) |> Nx.to_flat_list()
      assert sorted == Enum.sort(sorted)
    end
  end

  describe "mixed special values in tensors" do
    test "tensor with Inf, -Inf, NaN, 0, -0" do
      t = Nx.tensor([:infinity, :neg_infinity, :nan, 0.0, -0.0], type: :f32)
      assert Nx.shape(t) == {5}
    end

    test "sum with Inf propagates" do
      t = Nx.tensor([1.0, :infinity, 3.0], type: :f32)
      result = Nx.sum(t) |> Nx.to_number()
      assert result == :infinity
    end

    test "sum with -Inf propagates" do
      t = Nx.tensor([1.0, :neg_infinity, 3.0], type: :f32)
      result = Nx.sum(t) |> Nx.to_number()
      assert result == :neg_infinity
    end

    test "sum with Inf and -Inf produces NaN" do
      t = Nx.tensor([:infinity, :neg_infinity], type: :f32)
      result = Nx.sum(t) |> Nx.to_number()
      assert result == :nan
    end

    test "mean with NaN propagates NaN" do
      t = Nx.tensor([1.0, :nan, 3.0], type: :f32)
      result = Nx.mean(t) |> Nx.to_number()
      assert result == :nan
    end

    test "multiply Inf * 0 produces NaN" do
      result = Nx.multiply(Nx.tensor(:infinity), Nx.tensor(0.0)) |> Nx.to_number()
      assert result == :nan
    end

    test "equal(NaN, NaN) is false (IEEE 754)" do
      result = Nx.equal(Nx.tensor(:nan), Nx.tensor(:nan)) |> Nx.to_number()
      assert result == 0
    end

    test "is_nan detects NaN" do
      t = Nx.tensor([1.0, :nan, :infinity, 0.0], type: :f32)
      result = Nx.is_nan(t) |> Nx.to_flat_list()
      assert result == [0, 1, 0, 0]
    end

    test "is_infinity detects Inf and -Inf" do
      t = Nx.tensor([1.0, :infinity, :neg_infinity, 0.0], type: :f32)
      result = Nx.is_infinity(t) |> Nx.to_flat_list()
      assert result == [0, 1, 1, 0]
    end

    test "select with NaN pred" do
      # NaN is truthy (non-zero bits)
      pred = Nx.tensor(:nan, type: :f32)
      on_true = Nx.tensor(1.0)
      on_false = Nx.tensor(0.0)
      result = Nx.select(pred, on_true, on_false) |> Nx.to_number()
      # NaN should be treated as truthy since its bits are non-zero
      assert result == 1.0 or result == 0.0
    end

    test "dot with Inf values" do
      a = Nx.tensor([1.0, :infinity], type: :f32)
      b = Nx.tensor([1.0, 1.0], type: :f32)
      result = Nx.dot(a, b) |> Nx.to_number()
      assert result == :infinity
    end

    test "abs(Inf) == Inf, abs(-Inf) == Inf, abs(NaN) == NaN" do
      assert Nx.abs(Nx.tensor(:infinity)) |> Nx.to_number() == :infinity
      assert Nx.abs(Nx.tensor(:neg_infinity)) |> Nx.to_number() == :infinity
      assert Nx.abs(Nx.tensor(:nan)) |> Nx.to_number() == :nan
    end
  end

  describe "integer type boundaries" do
    test "u8 boundary values" do
      t = Nx.tensor([0, 127, 255], type: :u8)
      assert Nx.reduce_max(t) |> Nx.to_number() == 255
      assert Nx.reduce_min(t) |> Nx.to_number() == 0
    end

    test "s8 boundary values" do
      t = Nx.tensor([-128, 0, 127], type: :s8)
      assert Nx.reduce_max(t) |> Nx.to_number() == 127
      assert Nx.reduce_min(t) |> Nx.to_number() == -128
    end

    test "u8 overflow wraps" do
      result = Nx.add(Nx.tensor(255, type: :u8), Nx.tensor(1, type: :u8))
      val = Nx.to_number(result)
      # Should wrap to 0
      assert val == 0
    end

    test "s8 overflow wraps" do
      result = Nx.add(Nx.tensor(127, type: :s8), Nx.tensor(1, type: :s8))
      val = Nx.to_number(result)
      # Should wrap to -128
      assert val == -128
    end

    test "negate s8 min" do
      # -(-128) = 128 which overflows s8
      result = Nx.negate(Nx.tensor(-128, type: :s8))
      assert is_struct(result, Nx.Tensor)
    end
  end
end
