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
      {2,
       bind(tensor_dim(), fn d1 ->
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
        all_axes | for(a <- all_axes, do: [a])
      ])
    end
  end

  # ── Unary Element-wise Operations ──────────────────────────────────

  @unary_ops [
    :abs,
    :negate,
    :sign,
    :floor,
    :ceil,
    :round,
    :bitwise_not,
    :count_leading_zeros,
    :population_count
  ]

  # Split by domain requirements
  @unary_float_safe [
    :sigmoid,
    :sin,
    :cos,
    :tan,
    :atan,
    :tanh,
    :cbrt,
    :erf,
    :erfc,
    :is_nan,
    :is_infinity,
    :sign
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

  # ── Window Operations ─────────────────────────────────────────────

  describe "window ops don't crash" do
    for op <- [:window_sum, :window_max, :window_min, :window_product] do
      property "#{op} with 1D window" do
        check all(
                len <- integer(2..32),
                win <- integer(1..4),
                type <- float_type(),
                max_runs: 30
              ) do
          win = min(win, len)
          t = Nx.iota({len}, type: type)
          result = apply(Nx, unquote(op), [t, {win}])
          assert is_struct(result, Nx.Tensor)
          expected_len = len - win + 1
          assert Nx.shape(result) == {expected_len}
        end
      end

      property "#{op} with 2D window" do
        check all(
                rows <- integer(2..16),
                cols <- integer(2..16),
                win_r <- integer(1..3),
                win_c <- integer(1..3),
                type <- float_type(),
                max_runs: 20
              ) do
          win_r = min(win_r, rows)
          win_c = min(win_c, cols)
          t = Nx.iota({rows, cols}, type: type)
          result = apply(Nx, unquote(op), [t, {win_r, win_c}])
          assert is_struct(result, Nx.Tensor)
          assert Nx.shape(result) == {rows - win_r + 1, cols - win_c + 1}
        end
      end
    end

    for op <- [:window_sum, :window_max, :window_min] do
      property "#{op} with strides" do
        check all(
                len <- integer(4..32),
                type <- float_type(),
                max_runs: 20
              ) do
          t = Nx.iota({len}, type: type)
          result = apply(Nx, unquote(op), [t, {2}, [strides: [2]]])
          assert is_struct(result, Nx.Tensor)
        end
      end
    end
  end

  # ── Pad Operations ────────────────────────────────────────────────

  describe "pad ops don't crash" do
    property "pad 1D with positive padding" do
      check all(
              len <- integer(1..16),
              pad_lo <- integer(0..4),
              pad_hi <- integer(0..4),
              type <- float_type(),
              max_runs: 30
            ) do
        t = Nx.iota({len}, type: type)
        result = Nx.pad(t, 0, [{pad_lo, pad_hi, 0}])
        assert Nx.shape(result) == {len + pad_lo + pad_hi}
      end
    end

    property "pad 2D" do
      check all(
              rows <- integer(1..8),
              cols <- integer(1..8),
              type <- float_type(),
              max_runs: 20
            ) do
        t = Nx.iota({rows, cols}, type: type)
        result = Nx.pad(t, 0, [{1, 1, 0}, {0, 2, 0}])
        assert Nx.shape(result) == {rows + 2, cols + 2}
      end
    end
  end

  # ── Slice Operations ──────────────────────────────────────────────

  describe "slice ops don't crash" do
    property "slice 1D" do
      check all(
              len <- integer(2..32),
              type <- numeric_type(),
              max_runs: 30
            ) do
        t = Nx.iota({len}, type: type)
        start = :rand.uniform(len) - 1
        slice_len = :rand.uniform(len - start)
        result = Nx.slice(t, [start], [slice_len])
        assert Nx.shape(result) == {slice_len}
      end
    end

    property "slice 2D" do
      check all(
              rows <- integer(2..16),
              cols <- integer(2..16),
              type <- numeric_type(),
              max_runs: 20
            ) do
        t = Nx.iota({rows, cols}, type: type)
        sr = :rand.uniform(rows) - 1
        sc = :rand.uniform(cols) - 1
        lr = :rand.uniform(rows - sr)
        lc = :rand.uniform(cols - sc)
        result = Nx.slice(t, [sr, sc], [lr, lc])
        assert Nx.shape(result) == {lr, lc}
      end
    end

    property "put_slice 1D" do
      check all(
              len <- integer(2..16),
              type <- float_type(),
              max_runs: 20
            ) do
        t = Nx.iota({len}, type: type)
        start = :rand.uniform(len) - 1
        update_len = :rand.uniform(len - start)
        update = Nx.broadcast(99.0, {update_len})
        result = Nx.put_slice(t, [start], update)
        assert Nx.shape(result) == {len}
      end
    end
  end

  # ── Gather / Take Operations ──────────────────────────────────────

  describe "gather and take ops don't crash" do
    property "take from 1D" do
      check all(
              len <- integer(1..32),
              n_idx <- integer(1..8),
              type <- numeric_type(),
              max_runs: 30
            ) do
        t = Nx.iota({len}, type: type)
        indices = Nx.remainder(Nx.iota({n_idx}, type: :s64), len)
        result = Nx.take(t, indices)
        assert Nx.shape(result) == {n_idx}
      end
    end

    property "gather with indices" do
      check all(
              len <- integer(2..16),
              n_idx <- integer(1..8),
              type <- numeric_type(),
              max_runs: 20
            ) do
        t = Nx.iota({len}, type: type)
        indices = Nx.remainder(Nx.iota({n_idx, 1}, type: :s64), len)
        result = Nx.gather(t, indices)
        assert is_struct(result, Nx.Tensor)
      end
    end
  end

  # ── Reverse / Sort Operations ─────────────────────────────────────

  describe "reverse and sort ops don't crash" do
    property "reverse 1D" do
      check all(t <- tensor(non_empty_shape(), numeric_type()), max_runs: 30) do
        if tuple_size(Nx.shape(t)) >= 1 do
          result = Nx.reverse(t)
          assert Nx.shape(result) == Nx.shape(t)
        end
      end
    end

    property "sort 1D" do
      check all(
              len <- integer(1..32),
              type <- numeric_type(),
              max_runs: 20
            ) do
        t = Nx.iota({len}, type: type)
        result = Nx.sort(t)
        assert Nx.shape(result) == {len}
      end
    end

    property "argsort 1D" do
      check all(
              len <- integer(1..32),
              type <- numeric_type(),
              max_runs: 20
            ) do
        t = Nx.iota({len}, type: type)
        result = Nx.argsort(t)
        assert Nx.shape(result) == {len}
      end
    end
  end

  # ── Cumulative Operations ─────────────────────────────────────────

  describe "cumulative ops don't crash" do
    for op <- [:cumulative_sum, :cumulative_product, :cumulative_min, :cumulative_max] do
      property "#{op} preserves shape" do
        check all(
                len <- integer(1..32),
                type <- float_type(),
                max_runs: 20
              ) do
          t = Nx.iota({len}, type: type)
          result = apply(Nx, unquote(op), [t])
          assert Nx.shape(result) == {len}
        end
      end
    end
  end

  # ── Dot / Tensordot ───────────────────────────────────────────────

  describe "dot product ops don't crash" do
    property "dot 1D vectors" do
      check all(
              len <- integer(1..32),
              type <- float_type(),
              max_runs: 20
            ) do
        a = Nx.iota({len}, type: type)
        b = Nx.iota({len}, type: type)
        result = Nx.dot(a, b)
        assert Nx.shape(result) == {}
      end
    end

    property "dot 2D matrix multiply" do
      check all(
              m <- integer(1..16),
              n <- integer(1..16),
              k <- integer(1..16),
              type <- float_type(),
              max_runs: 20
            ) do
        a = Nx.iota({m, k}, type: type)
        b = Nx.iota({k, n}, type: type)
        result = Nx.dot(a, b)
        assert Nx.shape(result) == {m, n}
      end
    end

    property "dot with batched matmul" do
      check all(
              batch <- integer(1..4),
              m <- integer(1..8),
              n <- integer(1..8),
              k <- integer(1..8),
              type <- float_type(),
              max_runs: 15
            ) do
        a = Nx.iota({batch, m, k}, type: type)
        b = Nx.iota({batch, k, n}, type: type)
        result = Nx.dot(a, [2], [0], b, [1], [0])
        assert Nx.shape(result) == {batch, m, n}
      end
    end
  end

  # ── Select / Where ────────────────────────────────────────────────

  describe "select ops don't crash" do
    property "select with random predicate" do
      check all(
              shape <- non_empty_shape(),
              type <- float_type(),
              max_runs: 20
            ) do
        pred = Nx.greater(Nx.iota(shape, type: :f32), Nx.size(shape) / 2)
        on_true = Nx.iota(shape, type: type)
        on_false = Nx.broadcast(0, shape) |> Nx.as_type(type)
        result = Nx.select(pred, on_true, on_false)
        assert Nx.shape(result) == shape
      end
    end
  end

  # ── Indexed Operations ────────────────────────────────────────────

  describe "indexed ops don't crash" do
    property "indexed_add 1D" do
      check all(
              len <- integer(2..16),
              type <- float_type(),
              max_runs: 20
            ) do
        t = Nx.iota({len}, type: type)
        idx = Nx.remainder(Nx.iota({2, 1}, type: :s64), len)
        updates = Nx.broadcast(1.0, {2})
        result = Nx.indexed_add(t, idx, updates)
        assert Nx.shape(result) == {len}
      end
    end

    property "indexed_put 1D" do
      check all(
              len <- integer(2..16),
              type <- float_type(),
              max_runs: 20
            ) do
        t = Nx.iota({len}, type: type)
        idx = Nx.remainder(Nx.iota({2, 1}, type: :s64), len)
        updates = Nx.broadcast(99.0, {2})
        result = Nx.indexed_put(t, idx, updates)
        assert Nx.shape(result) == {len}
      end
    end
  end

  # ── Clip ──────────────────────────────────────────────────────────

  describe "clip doesn't crash" do
    property "clip with random bounds" do
      check all(
              shape <- non_empty_shape(),
              type <- float_type(),
              max_runs: 20
            ) do
        t = Nx.iota(shape, type: type)
        result = Nx.clip(t, 2, 5)
        assert Nx.shape(result) == shape
      end
    end
  end

  # ── Logical Operations ────────────────────────────────────────────

  @logical_ops [:logical_and, :logical_or, :logical_xor]

  describe "logical ops don't crash" do
    for op <- @logical_ops do
      property "#{op} returns correct shape" do
        check all(
                shape <- non_empty_shape(),
                max_runs: 20
              ) do
          a = Nx.greater(Nx.iota(shape, type: :f32), 2)
          b = Nx.less(Nx.iota(shape, type: :f32), 5)
          result = apply(Nx, unquote(op), [a, b])
          assert Nx.shape(result) == shape
        end
      end
    end

    property "logical_not returns correct shape" do
      check all(shape <- non_empty_shape(), max_runs: 20) do
        a = Nx.greater(Nx.iota(shape, type: :f32), 2)
        result = Nx.logical_not(a)
        assert Nx.shape(result) == shape
      end
    end
  end

  # ── Bitwise Operations ────────────────────────────────────────────

  describe "bitwise binary ops don't crash" do
    for op <- [:bitwise_and, :bitwise_or, :bitwise_xor] do
      property "#{op} with matching integer shapes" do
        check all(
                shape <- non_empty_shape(),
                type <- member_of(@integer_types),
                max_runs: 20
              ) do
          a = Nx.iota(shape, type: type)
          b = Nx.iota(shape, type: type)
          result = apply(Nx, unquote(op), [a, b])
          assert Nx.shape(result) == shape
        end
      end
    end

    for op <- [:left_shift, :right_shift] do
      property "#{op} with small shift amounts" do
        check all(
                shape <- non_empty_shape(),
                type <- member_of([:u8, :u16, :s8, :s16, :s32]),
                max_runs: 20
              ) do
          a = Nx.iota(shape, type: type)
          shift = Nx.broadcast(Nx.tensor(1, type: type), shape)
          result = apply(Nx, unquote(op), [a, shift])
          assert Nx.shape(result) == shape
        end
      end
    end
  end

  # ── all/any Aggregation ───────────────────────────────────────────

  describe "all/any aggregation" do
    for op <- [:all, :any] do
      property "#{op} reduces to scalar" do
        check all(t <- tensor(non_empty_shape(), numeric_type()), max_runs: 20) do
          result = apply(Nx, unquote(op), [t])
          assert Nx.shape(result) == {}
        end
      end
    end

    property "argmax returns scalar index" do
      check all(t <- tensor(non_empty_shape(), numeric_type()), max_runs: 20) do
        result = Nx.argmax(t)
        assert Nx.shape(result) == {}
      end
    end

    property "argmin returns scalar index" do
      check all(t <- tensor(non_empty_shape(), numeric_type()), max_runs: 20) do
        result = Nx.argmin(t)
        assert Nx.shape(result) == {}
      end
    end
  end

  # ── Statistical Operations ────────────────────────────────────────

  describe "statistical ops don't crash" do
    for op <- [:mean, :variance, :standard_deviation] do
      property "#{op} reduces to scalar" do
        check all(
                shape <- non_empty_shape() |> filter(&(Nx.size(&1) > 0)),
                type <- float_type(),
                max_runs: 20
              ) do
          t = Nx.iota(shape, type: type)
          result = apply(Nx, unquote(op), [t])
          assert Nx.shape(result) == {}
        end
      end
    end

    property "covariance doesn't crash" do
      check all(
              rows <- integer(2..16),
              cols <- integer(2..8),
              type <- float_type(),
              max_runs: 15
            ) do
        t = Nx.iota({rows, cols}, type: type)
        result = Nx.covariance(t)
        assert Nx.shape(result) == {cols, cols}
      end
    end

    property "weighted_mean doesn't crash" do
      check all(
              len <- integer(1..16),
              type <- float_type(),
              max_runs: 15
            ) do
        t = Nx.iota({len}, type: type)
        w = Nx.add(Nx.iota({len}, type: type), 1)
        result = Nx.weighted_mean(t, w)
        assert Nx.shape(result) == {}
      end
    end
  end

  # ── Shape Manipulation (more) ─────────────────────────────────────

  describe "more shape ops don't crash" do
    property "flatten" do
      check all(
              shape <- non_empty_shape() |> filter(&(Nx.size(&1) > 0)),
              type <- numeric_type(),
              max_runs: 20
            ) do
        t = Nx.iota(shape, type: type)
        result = Nx.flatten(t)
        assert Nx.shape(result) == {Nx.size(shape)}
      end
    end

    property "tile 1D" do
      check all(
              len <- integer(1..8),
              reps <- integer(1..4),
              type <- numeric_type(),
              max_runs: 20
            ) do
        t = Nx.iota({len}, type: type)
        result = Nx.tile(t, [reps])
        assert Nx.shape(result) == {len * reps}
      end
    end

    property "tile 2D" do
      check all(
              rows <- integer(1..6),
              cols <- integer(1..6),
              rep_r <- integer(1..3),
              rep_c <- integer(1..3),
              type <- numeric_type(),
              max_runs: 15
            ) do
        t = Nx.iota({rows, cols}, type: type)
        result = Nx.tile(t, [rep_r, rep_c])
        assert Nx.shape(result) == {rows * rep_r, cols * rep_c}
      end
    end

    property "reflect 1D" do
      check all(
              len <- integer(3..16),
              type <- float_type(),
              max_runs: 15
            ) do
        t = Nx.iota({len}, type: type)
        pad = min(:rand.uniform(len - 1), len - 1)
        result = Nx.reflect(t, padding_config: [{pad, pad}])
        assert Nx.shape(result) == {len + 2 * pad}
      end
    end

    property "diff 1D" do
      check all(
              len <- integer(2..16),
              type <- float_type(),
              max_runs: 15
            ) do
        t = Nx.iota({len}, type: type)
        result = Nx.diff(t)
        assert Nx.shape(result) == {len - 1}
      end
    end

    property "split" do
      check all(
              type <- numeric_type(),
              max_runs: 15
            ) do
        t = Nx.iota({6}, type: type)
        {left, right} = Nx.split(t, 3)
        assert Nx.shape(left) == {3}
        assert Nx.shape(right) == {3}
      end
    end

    property "slice_along_axis" do
      check all(
              len <- integer(4..16),
              type <- numeric_type(),
              max_runs: 15
            ) do
        t = Nx.iota({len}, type: type)
        start = :rand.uniform(div(len, 2)) - 1
        slice_len = :rand.uniform(len - start)
        result = Nx.slice_along_axis(t, start, slice_len)
        assert Nx.shape(result) == {slice_len}
      end
    end
  end

  # ── Matrix Diagonal Operations ────────────────────────────────────

  describe "diagonal ops don't crash" do
    property "take_diagonal" do
      check all(
              n <- integer(2..8),
              type <- numeric_type(),
              max_runs: 15
            ) do
        t = Nx.iota({n, n}, type: type)
        result = Nx.take_diagonal(t)
        assert Nx.shape(result) == {n}
      end
    end

    property "make_diagonal" do
      check all(
              n <- integer(1..8),
              type <- numeric_type(),
              max_runs: 15
            ) do
        t = Nx.iota({n}, type: type)
        result = Nx.make_diagonal(t)
        assert Nx.shape(result) == {n, n}
      end
    end

    property "triu" do
      check all(
              n <- integer(2..8),
              type <- numeric_type(),
              max_runs: 15
            ) do
        t = Nx.iota({n, n}, type: type)
        result = Nx.triu(t)
        assert Nx.shape(result) == {n, n}
      end
    end

    property "tril" do
      check all(
              n <- integer(2..8),
              type <- numeric_type(),
              max_runs: 15
            ) do
        t = Nx.iota({n, n}, type: type)
        result = Nx.tril(t)
        assert Nx.shape(result) == {n, n}
      end
    end

    property "tri" do
      check all(
              n <- integer(2..8),
              max_runs: 15
            ) do
        result = Nx.tri(n, n)
        assert Nx.shape(result) == {n, n}
      end
    end
  end

  # ── Complex Number Operations ─────────────────────────────────────

  describe "complex ops don't crash" do
    property "conjugate" do
      check all(
              shape <- non_empty_shape(),
              type <- member_of([:f32, :f64]),
              max_runs: 15
            ) do
        t = Nx.iota(shape, type: type)
        result = Nx.conjugate(t)
        assert Nx.shape(result) == shape
      end
    end

    property "real and imag on float tensors" do
      check all(
              shape <- non_empty_shape(),
              type <- member_of([:f32, :f64]),
              max_runs: 15
            ) do
        t = Nx.iota(shape, type: type)
        r = Nx.real(t)
        i = Nx.imag(t)
        assert Nx.shape(r) == shape
        assert Nx.shape(i) == shape
      end
    end
  end

  # ── FFT ───────────────────────────────────────────────────────────

  describe "FFT ops don't crash" do
    property "fft and ifft roundtrip" do
      check all(
              # FFT requires power-of-2 or the library handles padding
              exp <- integer(1..6),
              type <- member_of([:f32, :f64]),
              max_runs: 10
            ) do
        len = Integer.pow(2, exp)
        t = Nx.iota({len}, type: type)
        ft = Nx.fft(t)
        assert is_struct(ft, Nx.Tensor)
        ift = Nx.ifft(ft)
        assert is_struct(ift, Nx.Tensor)
      end
    end
  end

  # ── Top-K ─────────────────────────────────────────────────────────

  describe "top_k" do
    property "returns correct shapes" do
      check all(
              len <- integer(2..32),
              k <- integer(1..4),
              type <- numeric_type(),
              max_runs: 15
            ) do
        k = min(k, len)
        t = Nx.iota({len}, type: type)
        {values, indices} = Nx.top_k(t, k: k)
        assert Nx.shape(values) == {k}
        assert Nx.shape(indices) == {k}
      end
    end
  end

  # ── Window Scatter ────────────────────────────────────────────────

  describe "window scatter ops don't crash" do
    for op <- [:window_scatter_max, :window_scatter_min] do
      property "#{op} 1D f32" do
        check all(_ <- constant(:ok), max_runs: 10) do
          t = Nx.iota({6}, type: :f32)
          source = Nx.iota({3}, type: :f32)
          init = Nx.tensor(0.0, type: :f32)

          result =
            apply(Nx, unquote(op), [t, source, init, {2}, [strides: [2], padding: :valid]])

          assert Nx.shape(result) == {6}
        end
      end

      # BUG: window_scatter on f64 crashes with binary size mismatch
      @tag :skip
      property "#{op} 1D f64 (crashes — binary size bug)" do
        check all(_ <- constant(:ok), max_runs: 5) do
          t = Nx.iota({6}, type: :f64)
          source = Nx.iota({3}, type: :f64)
          init = Nx.tensor(0.0, type: :f64)

          result =
            apply(Nx, unquote(op), [t, source, init, {2}, [strides: [2], padding: :valid]])

          assert Nx.shape(result) == {6}
        end
      end
    end

    property "window_mean 1D" do
      check all(
              len <- integer(2..16),
              type <- float_type(),
              max_runs: 15
            ) do
        t = Nx.iota({len}, type: type)
        win = min(2, len)
        result = Nx.window_mean(t, {win})
        assert is_struct(result, Nx.Tensor)
      end
    end
  end

  # ── Take Along Axis ───────────────────────────────────────────────

  describe "take_along_axis" do
    property "1D take_along_axis" do
      check all(
              len <- integer(2..16),
              type <- numeric_type(),
              max_runs: 15
            ) do
        t = Nx.iota({len}, type: type)
        # Sort indices as a valid use case
        indices = Nx.argsort(t, direction: :desc)
        result = Nx.take_along_axis(t, indices, axis: 0)
        assert Nx.shape(result) == {len}
      end
    end
  end

  # ── Bitcast ───────────────────────────────────────────────────────

  describe "bitcast" do
    property "bitcast preserves byte size" do
      check all(
              shape <- non_empty_shape() |> filter(&(Nx.size(&1) > 0)),
              max_runs: 15
            ) do
        t = Nx.iota(shape, type: :f32)
        result = Nx.bitcast(t, :s32)
        assert Nx.shape(result) == shape
        assert Nx.type(result) == {:s, 32}
      end
    end
  end
end
