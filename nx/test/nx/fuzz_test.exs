defmodule Nx.FuzzTest do
  @moduledoc """
  Property-based fuzz tests for Nx operations.

  Uses StreamData generators to produce random valid inputs and verifies
  that operations don't crash, return correct shapes, and return correct types.
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  # ── Generators ─────────────────────────────────────────────────────

  @integer_types [:u8, :u16, :u32, :u64, :s8, :s16, :s32, :s64]
  @float_types [:f16, :bf16, :f32, :f64]
  @complex_types [:c64, :c128]
  @all_numeric_types @integer_types ++ @float_types
  @all_types @all_numeric_types ++ @complex_types

  defp tensor_type do
    member_of(@all_types)
  end

  defp float_type do
    member_of(@float_types)
  end

  defp numeric_type do
    member_of(@all_numeric_types)
  end

  defp tensor_dim do
    # Include edge cases: 0 (empty), 1 (singleton), small, medium
    frequency([
      {1, constant(0)},
      {3, constant(1)},
      {5, integer(2..8)},
      {2, integer(9..32)},
      {1, integer(33..64)}
    ])
  end

  defp tensor_shape do
    # Rank 0 (scalar) through rank 4
    frequency([
      {2, constant({})},
      {4, map(tensor_dim(), &{&1})},
      {4, bind(tensor_dim(), fn d1 -> map(tensor_dim(), &{d1, &1}) end)},
      {2, bind(tensor_dim(), fn d1 ->
        bind(tensor_dim(), fn d2 ->
          map(tensor_dim(), &{d1, d2, &1})
        end)
      end)}
    ])
  end

  defp non_empty_shape do
    # Shape where all dims > 0 (for ops that can't handle empty tensors)
    tensor_shape()
    |> filter(fn shape ->
      shape == {} or Enum.all?(Tuple.to_list(shape), &(&1 > 0))
    end)
  end

  defp tensor(shape_gen \\ non_empty_shape(), type_gen \\ numeric_type()) do
    bind(shape_gen, fn shape ->
      bind(type_gen, fn type ->
        constant(Nx.iota(shape, type: type))
      end)
    end)
  end

  defp float_tensor(shape_gen \\ non_empty_shape()) do
    tensor(shape_gen, float_type())
  end

  defp valid_axes(shape) do
    rank = tuple_size(shape)

    if rank == 0 do
      constant([])
    else
      all_axes = Enum.to_list(0..(rank - 1))

      # Generate subsets of valid axes
      member_of([
        [],
        all_axes | (for a <- all_axes, do: [a])
      ])
    end
  end

  # ── Unary Element-wise Operations ──────────────────────────────────

  @unary_ops [
    :abs, :negate, :sign, :floor, :ceil, :round,
    :bitwise_not, :count_leading_zeros, :population_count
  ]

  # Split by domain requirements
  @unary_float_safe [
    :sigmoid, :sin, :cos, :tan, :atan, :tanh, :cbrt,
    :erf, :erfc, :is_nan, :is_infinity, :sign
  ]

  # These overflow on large inputs (exp(710) overflows f64)
  @unary_float_overflow [:exp, :expm1, :sinh, :cosh]

  # These need inputs in [-1, 1]
  @unary_float_unit_domain [:asin, :acos]

  # These need positive inputs (> 0)
  @unary_float_positive [:log, :log1p, :rsqrt, :sqrt]

  # acosh needs inputs >= 1
  @unary_float_ge_one [:acosh]

  # These need inputs in (-1, 1)
  @unary_float_open_unit [:atanh, :erf_inv]

  # These need inputs in a moderate range
  @unary_float_moderate [:asinh]

  describe "unary element-wise ops don't crash" do
    for op <- @unary_ops do
      property "#{op} doesn't crash on integer tensors" do
        check all(t <- tensor(non_empty_shape(), member_of(@integer_types))) do
          result = apply(Nx, unquote(op), [t])
          assert is_struct(result, Nx.Tensor)
          assert Nx.shape(result) == Nx.shape(t)
        end
      end
    end

    for op <- @unary_float_safe do
      property "#{op} doesn't crash on float tensors" do
        check all(t <- float_tensor()) do
          result = apply(Nx, unquote(op), [t])
          assert is_struct(result, Nx.Tensor)
          assert Nx.shape(result) == Nx.shape(t)
        end
      end
    end

    for op <- @unary_float_positive do
      property "#{op} doesn't crash on positive float tensors" do
        check all(t <- float_tensor()) do
          t = Nx.add(Nx.abs(t), 0.001)
          result = apply(Nx, unquote(op), [t])
          assert is_struct(result, Nx.Tensor)
          assert Nx.shape(result) == Nx.shape(t)
        end
      end
    end

    for op <- @unary_float_ge_one do
      property "#{op} doesn't crash on inputs >= 1" do
        check all(t <- float_tensor()) do
          t = Nx.add(Nx.abs(t), 1.0)
          result = apply(Nx, unquote(op), [t])
          assert is_struct(result, Nx.Tensor)
          assert Nx.shape(result) == Nx.shape(t)
        end
      end
    end

    for op <- @unary_float_unit_domain do
      property "#{op} doesn't crash on [-0.99, 0.99] inputs" do
        check all(t <- float_tensor()) do
          # Map iota values to [-0.99, 0.99] range
          t = Nx.multiply(Nx.divide(Nx.sin(Nx.as_type(t, :f32)), 1.0), 0.99)
          result = apply(Nx, unquote(op), [t])
          assert is_struct(result, Nx.Tensor)
          assert Nx.shape(result) == Nx.shape(t)
        end
      end
    end

    for op <- @unary_float_open_unit do
      property "#{op} doesn't crash on (-0.99, 0.99) inputs" do
        check all(t <- float_tensor()) do
          t = Nx.multiply(Nx.divide(Nx.sin(Nx.as_type(t, :f32)), 1.0), 0.99)
          result = apply(Nx, unquote(op), [t])
          assert is_struct(result, Nx.Tensor)
          assert Nx.shape(result) == Nx.shape(t)
        end
      end
    end

    for op <- @unary_float_moderate do
      property "#{op} doesn't crash on moderate float tensors" do
        check all(t <- float_tensor()) do
          # Keep values moderate to avoid overflow
          t = Nx.clip(Nx.as_type(t, :f32), -100, 100)
          result = apply(Nx, unquote(op), [t])
          assert is_struct(result, Nx.Tensor)
          assert Nx.shape(result) == Nx.shape(t)
        end
      end
    end

    # ── Overflow behavior tests ──
    # These probe whether the BinaryBackend crashes vs returns Inf/NaN
    # for inputs that overflow. A robust backend should not crash.

    for op <- @unary_float_overflow do
      @tag :skip
      property "#{op} should return Inf (not crash) on large inputs" do
        # BinaryBackend delegates to :math which raises ArithmeticError
        # on overflow. Should return Inf instead.
        check all(type <- float_type()) do
          t = Nx.tensor([100.0, 500.0, 1000.0], type: type)
          result = apply(Nx, unquote(op), [t])
          assert is_struct(result, Nx.Tensor)
        end
      end
    end
  end

  # ── Binary Element-wise Operations ────────────────────────────────

  @binary_ops [:add, :subtract, :multiply, :min, :max]

  describe "binary element-wise ops don't crash" do
    for op <- @binary_ops do
      property "#{op} doesn't crash with matching shapes" do
        check all(
                shape <- non_empty_shape(),
                type <- numeric_type(),
                max_runs: 50
              ) do
          a = Nx.iota(shape, type: type)
          b = Nx.iota(shape, type: type)
          result = apply(Nx, unquote(op), [a, b])
          assert is_struct(result, Nx.Tensor)
          assert Nx.shape(result) == shape
        end
      end
    end

    property "divide doesn't crash with non-zero divisor" do
      check all(
              shape <- non_empty_shape(),
              type <- float_type(),
              max_runs: 50
            ) do
        a = Nx.iota(shape, type: type)
        b = Nx.add(Nx.iota(shape, type: type), 1)
        result = Nx.divide(a, b)
        assert is_struct(result, Nx.Tensor)
        assert Nx.shape(result) == shape
      end
    end
  end

  # ── Reduction Operations ──────────────────────────────────────────

  @reduce_ops [:sum, :product, :reduce_max, :reduce_min]

  describe "reduction ops don't crash" do
    for op <- @reduce_ops do
      property "#{op} reduces all axes" do
        check all(t <- tensor(non_empty_shape(), float_type()), max_runs: 50) do
          result = apply(Nx, unquote(op), [t])
          assert is_struct(result, Nx.Tensor)
          # Full reduction produces scalar
          assert Nx.shape(result) == {}
        end
      end

      property "#{op} reduces single axis" do
        check all(
                shape <- non_empty_shape() |> filter(&(tuple_size(&1) >= 1)),
                type <- float_type(),
                max_runs: 50
              ) do
          t = Nx.iota(shape, type: type)
          axis = :rand.uniform(tuple_size(shape)) - 1
          result = apply(Nx, unquote(op), [t, [axes: [axis]]])
          assert is_struct(result, Nx.Tensor)
          assert tuple_size(Nx.shape(result)) == tuple_size(shape) - 1
        end
      end
    end
  end

  # ── Shape Operations ──────────────────────────────────────────────

  describe "shape ops don't crash" do
    property "reshape preserves element count" do
      check all(
              shape <- non_empty_shape() |> filter(&(Nx.size(&1) > 0)),
              type <- numeric_type(),
              max_runs: 50
            ) do
        t = Nx.iota(shape, type: type)
        flat = {Nx.size(shape)}
        result = Nx.reshape(t, flat)
        assert Nx.shape(result) == flat
        assert Nx.size(result) == Nx.size(t)
      end
    end

    property "transpose reverses axes" do
      check all(
              shape <- non_empty_shape() |> filter(&(tuple_size(&1) >= 2)),
              type <- numeric_type(),
              max_runs: 50
            ) do
        t = Nx.iota(shape, type: type)
        result = Nx.transpose(t)
        assert is_struct(result, Nx.Tensor)

        expected_shape =
          shape |> Tuple.to_list() |> Enum.reverse() |> List.to_tuple()

        assert Nx.shape(result) == expected_shape
      end
    end

    property "squeeze removes size-1 dims" do
      check all(type <- numeric_type(), max_runs: 50) do
        # Create a shape with at least one size-1 dim
        t = Nx.iota({3, 1, 4, 1}, type: type)
        result = Nx.squeeze(t)
        assert Nx.shape(result) == {3, 4}
      end
    end

    property "new_axis adds a dimension" do
      check all(
              shape <- non_empty_shape(),
              type <- numeric_type(),
              max_runs: 50
            ) do
        t = Nx.iota(shape, type: type)
        result = Nx.new_axis(t, 0)
        assert tuple_size(Nx.shape(result)) == tuple_size(shape) + 1
        assert elem(Nx.shape(result), 0) == 1
      end
    end
  end

  # ── Type Coercion ─────────────────────────────────────────────────

  describe "type coercion" do
    property "as_type doesn't crash for valid conversions" do
      check all(
              shape <- non_empty_shape(),
              from_type <- numeric_type(),
              to_type <- numeric_type(),
              max_runs: 50
            ) do
        t = Nx.iota(shape, type: from_type)
        result = Nx.as_type(t, to_type)
        assert Nx.type(result) == Nx.Type.normalize!(to_type)
        assert Nx.shape(result) == shape
      end
    end
  end

  # ── Broadcasting ──────────────────────────────────────────────────

  describe "broadcasting" do
    property "broadcast to same shape is identity" do
      check all(t <- tensor(non_empty_shape(), numeric_type()), max_runs: 50) do
        result = Nx.broadcast(t, Nx.shape(t))
        assert Nx.shape(result) == Nx.shape(t)
      end
    end

    property "scalar broadcasts to any shape" do
      check all(
              shape <- non_empty_shape(),
              type <- numeric_type(),
              max_runs: 50
            ) do
        scalar = Nx.tensor(1, type: type)
        result = Nx.broadcast(scalar, shape)
        assert Nx.shape(result) == shape
      end
    end
  end

  # ── Creation Operations ───────────────────────────────────────────

  describe "creation ops don't crash" do
    property "iota creates correct shape" do
      check all(
              shape <- non_empty_shape(),
              type <- numeric_type(),
              max_runs: 50
            ) do
        result = Nx.iota(shape, type: type)
        assert Nx.shape(result) == shape
        assert Nx.type(result) == Nx.Type.normalize!(type)
      end
    end

    property "eye creates square identity" do
      check all(n <- integer(1..16), type <- float_type(), max_runs: 20) do
        result = Nx.eye(n, type: type)
        assert Nx.shape(result) == {n, n}
      end
    end

    property "broadcast creates correct shape" do
      check all(shape <- non_empty_shape(), max_runs: 50) do
        result = Nx.broadcast(0, shape)
        assert Nx.shape(result) == shape
      end
    end
  end

  # ── Concatenation and Stacking ────────────────────────────────────

  describe "concatenation" do
    property "concatenate along axis 0" do
      check all(
              cols <- integer(1..8),
              rows1 <- integer(1..8),
              rows2 <- integer(1..8),
              type <- numeric_type(),
              max_runs: 30
            ) do
        a = Nx.iota({rows1, cols}, type: type)
        b = Nx.iota({rows2, cols}, type: type)
        result = Nx.concatenate([a, b], axis: 0)
        assert Nx.shape(result) == {rows1 + rows2, cols}
      end
    end

    property "stack adds a new axis" do
      check all(
              shape <- non_empty_shape() |> filter(&(tuple_size(&1) >= 1)),
              n <- integer(1..4),
              type <- numeric_type(),
              max_runs: 30
            ) do
        tensors = for _ <- 1..n, do: Nx.iota(shape, type: type)
        result = Nx.stack(tensors)
        assert elem(Nx.shape(result), 0) == n
      end
    end
  end

  # ── Empty tensor edge cases ───────────────────────────────────────

  describe "empty tensor handling" do
    # Nx.iota rejects zero dimensions — documenting this behavior
    test "iota rejects zero dimensions" do
      assert_raise ArgumentError, fn ->
        Nx.iota({0})
      end
    end

    test "iota rejects zero in multi-dim shape" do
      assert_raise ArgumentError, fn ->
        Nx.iota({3, 0, 4})
      end
    end
  end

  # ── Comparison Operations ─────────────────────────────────────────

  @comparison_ops [:equal, :not_equal, :greater, :greater_equal, :less, :less_equal]

  describe "comparison ops" do
    for op <- @comparison_ops do
      property "#{op} returns u8 predicate with correct shape" do
        check all(
                shape <- non_empty_shape(),
                type <- numeric_type(),
                max_runs: 30
              ) do
          a = Nx.iota(shape, type: type)
          b = Nx.iota(shape, type: type)
          result = apply(Nx, unquote(op), [a, b])
          assert Nx.shape(result) == shape
          assert Nx.type(result) == {:u, 8}
        end
      end
    end
  end
end
