defmodule Nx.FuzzEdgeCases2Test do
  @moduledoc """
  Tier 4 (continued): Edge case tests for defn constructs, diagonal ops,
  reduce, vectorize/devectorize, type conversions, and to_batched.
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  import Nx.Defn

  # ── Diagonal boundary conditions ───────────────────────────────────
  # Source: shape.ex:1382 (take_diagonal), shape.ex:1397 (make_diagonal),
  #         shape.ex:1416 (put_diagonal)
  # Boundaries:
  #   - take_diagonal: rank >= 2
  #   - make_diagonal: rank == 1
  #   - put_diagonal: tensor rank 2, diagonal rank 1, length match, offset bounds

  describe "take_diagonal boundary conditions" do
    test "take_diagonal on square matrix" do
      t = Nx.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
      result = Nx.take_diagonal(t)
      assert Nx.to_flat_list(result) == [1, 5, 9]
    end

    test "take_diagonal on non-square matrix (wide)" do
      t = Nx.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
      result = Nx.take_diagonal(t)
      assert Nx.to_flat_list(result) == [1, 6]
    end

    test "take_diagonal on non-square matrix (tall)" do
      t = Nx.tensor([[1, 2], [3, 4], [5, 6]])
      result = Nx.take_diagonal(t)
      assert Nx.to_flat_list(result) == [1, 4]
    end

    test "take_diagonal with positive offset" do
      t = Nx.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
      result = Nx.take_diagonal(t, offset: 1)
      assert Nx.to_flat_list(result) == [2, 6]
    end

    test "take_diagonal with negative offset" do
      t = Nx.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
      result = Nx.take_diagonal(t, offset: -1)
      assert Nx.to_flat_list(result) == [4, 8]
    end

    test "take_diagonal with max positive offset" do
      t = Nx.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
      result = Nx.take_diagonal(t, offset: 2)
      assert Nx.to_flat_list(result) == [3]
    end

    test "take_diagonal with max negative offset" do
      t = Nx.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
      result = Nx.take_diagonal(t, offset: -2)
      assert Nx.to_flat_list(result) == [7]
    end

    test "take_diagonal with 3D batched tensor" do
      t = Nx.iota({2, 3, 3})
      result = Nx.take_diagonal(t)
      assert Nx.shape(result) == {2, 3}
    end

    test "take_diagonal raises on rank 1" do
      t = Nx.tensor([1, 2, 3])
      assert_raise ArgumentError, ~r/rank 2 or higher/, fn ->
        Nx.take_diagonal(t)
      end
    end

    test "take_diagonal raises on scalar" do
      t = Nx.tensor(42)
      assert_raise ArgumentError, ~r/rank 2 or higher/, fn ->
        Nx.take_diagonal(t)
      end
    end
  end

  describe "make_diagonal boundary conditions" do
    test "make_diagonal from 1D" do
      t = Nx.tensor([1, 2, 3])
      result = Nx.make_diagonal(t)
      expected = Nx.tensor([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
      assert Nx.to_flat_list(result) == Nx.to_flat_list(expected)
    end

    test "make_diagonal single element" do
      t = Nx.tensor([42])
      result = Nx.make_diagonal(t)
      assert Nx.shape(result) == {1, 1}
      assert Nx.to_flat_list(result) == [42]
    end

    test "make_diagonal with positive offset" do
      t = Nx.tensor([1, 2])
      result = Nx.make_diagonal(t, offset: 1)
      assert Nx.shape(result) == {3, 3}
    end

    test "make_diagonal with negative offset" do
      t = Nx.tensor([1, 2])
      result = Nx.make_diagonal(t, offset: -1)
      assert Nx.shape(result) == {3, 3}
    end

    test "make_diagonal raises on rank 2" do
      t = Nx.tensor([[1, 2], [3, 4]])
      assert_raise ArgumentError, ~r/rank 1/, fn ->
        Nx.make_diagonal(t)
      end
    end
  end

  describe "put_diagonal boundary conditions" do
    test "put_diagonal replaces main diagonal" do
      t = Nx.broadcast(Nx.tensor(0), {3, 3})
      d = Nx.tensor([1, 2, 3])
      result = Nx.put_diagonal(t, d)
      expected = Nx.tensor([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
      assert Nx.to_flat_list(result) == Nx.to_flat_list(expected)
    end

    test "put_diagonal on non-square matrix" do
      t = Nx.broadcast(Nx.tensor(0), {3, 4})
      d = Nx.tensor([1, 2, 3])
      result = Nx.put_diagonal(t, d)
      assert Nx.shape(result) == {3, 4}
    end

    test "put_diagonal with positive offset" do
      t = Nx.broadcast(Nx.tensor(0), {3, 3})
      d = Nx.tensor([10, 20])
      result = Nx.put_diagonal(t, d, offset: 1)
      assert Nx.to_number(result[0][1]) == 10
      assert Nx.to_number(result[1][2]) == 20
    end

    test "put_diagonal raises on wrong diagonal length" do
      t = Nx.broadcast(Nx.tensor(0), {3, 3})
      d = Nx.tensor([1, 2])

      assert_raise ArgumentError, ~r/expected diagonal tensor of length/, fn ->
        Nx.put_diagonal(t, d)
      end
    end

    test "put_diagonal raises on rank 3 tensor" do
      t = Nx.iota({2, 3, 3})
      d = Nx.tensor([1, 2, 3])

      assert_raise ArgumentError, ~r/rank 2/, fn ->
        Nx.put_diagonal(t, d)
      end
    end

    test "put_diagonal raises on offset too large" do
      t = Nx.broadcast(Nx.tensor(0), {3, 3})
      d = Nx.tensor([1, 2, 3])

      assert_raise ArgumentError, ~r/offset must be less than/, fn ->
        Nx.put_diagonal(t, d, offset: 4)
      end
    end

    test "take_diagonal(make_diagonal(v)) == v" do
      v = Nx.tensor([1, 2, 3, 4, 5])
      result = v |> Nx.make_diagonal() |> Nx.take_diagonal()
      assert Nx.to_flat_list(result) == Nx.to_flat_list(v)
    end
  end

  # ── Reduce boundary conditions ─────────────────────────────────────
  # Source: nx.ex:11901
  # Boundaries:
  #   - accumulator must be non-vectorized scalar (line 11909)
  #   - reducer function must return scalar (expr.ex:945)

  describe "reduce boundary conditions" do
    test "reduce with sum as custom reducer" do
      t = Nx.tensor([1, 2, 3, 4, 5])
      result = Nx.reduce(t, Nx.tensor(0), fn x, acc -> Nx.add(x, acc) end)
      assert Nx.to_number(result) == 15
    end

    test "reduce with product as custom reducer" do
      t = Nx.tensor([1.0, 2.0, 3.0, 4.0])
      result = Nx.reduce(t, Nx.tensor(1.0), fn x, acc -> Nx.multiply(x, acc) end)
      assert_in_delta Nx.to_number(result), 24.0, 1.0e-5
    end

    test "reduce on specific axes" do
      t = Nx.iota({2, 3}, type: :f32)
      result = Nx.reduce(t, Nx.tensor(0.0), [axes: [1]], fn x, acc -> Nx.add(x, acc) end)
      assert Nx.shape(result) == {2}
    end

    test "reduce with keep_axes" do
      t = Nx.iota({3, 4}, type: :f32)
      result = Nx.reduce(t, Nx.tensor(0.0), [axes: [1], keep_axes: true], fn x, acc ->
        Nx.add(x, acc)
      end)
      assert Nx.shape(result) == {3, 1}
    end

    test "reduce raises on non-scalar accumulator" do
      t = Nx.tensor([1, 2, 3])
      assert_raise ArgumentError, ~r/accumulator must be a non-vectorized scalar/, fn ->
        Nx.reduce(t, Nx.tensor([0, 0]), fn x, acc -> Nx.add(x, acc) end)
      end
    end

    test "reduce single element returns that element (via acc)" do
      t = Nx.tensor([42.0])
      result = Nx.reduce(t, Nx.tensor(0.0), fn x, acc -> Nx.add(x, acc) end)
      assert Nx.to_number(result) == 42.0
    end
  end

  # ── Window_reduce boundary conditions ──────────────────────────────
  # Source: nx.ex:12049
  # Boundaries:
  #   - accumulator cannot be vectorized (line 12055)
  #   - reducer must return scalar

  describe "window_reduce boundary conditions" do
    test "window_reduce custom max" do
      t = Nx.tensor([3.0, 1.0, 4.0, 1.0, 5.0, 9.0])
      result = Nx.window_reduce(t, Nx.Constants.neg_infinity(), {2}, fn x, acc ->
        Nx.max(x, acc)
      end)
      assert Nx.shape(result) == {5}
    end

    test "window_reduce with strides" do
      t = Nx.iota({6}, type: :f32)
      result = Nx.window_reduce(t, Nx.tensor(0.0), {3}, [strides: [2]], fn x, acc ->
        Nx.add(x, acc)
      end)
      assert Nx.shape(result) == {2}
    end

    test "window_reduce 2D" do
      t = Nx.iota({4, 4}, type: :f32)
      result = Nx.window_reduce(t, Nx.tensor(0.0), {2, 2}, fn x, acc ->
        Nx.add(x, acc)
      end)
      assert Nx.shape(result) == {3, 3}
    end

    test "window_reduce raises on vectorized accumulator" do
      t = Nx.iota({6}, type: :f32)
      vec_acc = Nx.tensor([0.0, 0.0]) |> Nx.vectorize(:batch)

      assert_raise ArgumentError, ~r/accumulator .* cannot be vectorized/, fn ->
        Nx.window_reduce(t, vec_acc, {2}, fn x, acc -> Nx.add(x, acc) end)
      end
    end
  end

  # ── Vectorize / devectorize boundary conditions ────────────────────
  # Source: nx.ex:4869 (vectorize), nx.ex:5009 (devectorize)
  # Boundaries:
  #   - cannot vectorize rank-0 tensor (line 4878)
  #   - n vectorized axes must not exceed shape size (line 4889)
  #   - name conflicts rejected (lines 4916, 4921)

  describe "vectorize boundary conditions" do
    test "vectorize rank-1 tensor" do
      t = Nx.tensor([1, 2, 3])
      result = Nx.vectorize(t, :batch)
      assert result.vectorized_axes == [batch: 3]
      assert result.shape == {}
    end

    test "vectorize rank-2 tensor (1 axis)" do
      t = Nx.iota({2, 3})
      result = Nx.vectorize(t, :batch)
      assert result.vectorized_axes == [batch: 2]
      assert result.shape == {3}
    end

    test "vectorize then devectorize roundtrip" do
      t = Nx.iota({3, 4})
      result = t |> Nx.vectorize(:batch) |> Nx.devectorize()
      assert Nx.shape(result) == {3, 4}
      assert Nx.to_flat_list(result) == Nx.to_flat_list(t)
    end

    test "vectorize raises on scalar" do
      t = Nx.tensor(42)
      assert_raise ArgumentError, ~r/cannot vectorize tensor of rank 0/, fn ->
        Nx.vectorize(t, :batch)
      end
    end

    test "vectorize raises on name conflict with existing vectorized axis" do
      t = Nx.iota({2, 3, 4})
      v1 = Nx.vectorize(t, :batch)  # vectorized_axes: [batch: 2], shape: {3, 4}

      assert_raise ArgumentError, ~r/already a vectorized axis/, fn ->
        Nx.vectorize(v1, :batch)  # tries to add :batch again
      end
    end

    test "devectorize with keep_names: false drops vectorized axis name" do
      t = Nx.iota({2, 3}, names: [:x, :y])
      v = Nx.vectorize(t, :batch)
      result = Nx.devectorize(v, keep_names: false)
      assert Nx.shape(result) == {2, 3}
      # keep_names: false makes the vectorized axis name nil, keeps inner names
      assert result.names == [nil, :y]
    end

    test "devectorize with keep_names: true preserves vectorized name" do
      t = Nx.iota({2, 3}, names: [:x, :y])
      v = Nx.vectorize(t, :batch)
      result = Nx.devectorize(v, keep_names: true)
      assert result.names == [:batch, :y]
    end
  end

  # ── To_batched boundary conditions ─────────────────────────────────
  # Source: nx.ex:2520
  # Boundaries:
  #   - batch_size >= 1
  #   - cannot batch scalar
  #   - cannot batch beyond tensor size

  describe "to_batched boundary conditions" do
    test "to_batched evenly divides" do
      t = Nx.iota({6, 3})
      batches = Nx.to_batched(t, 2) |> Enum.to_list()
      assert length(batches) == 3
      assert Nx.shape(hd(batches)) == {2, 3}
    end

    test "to_batched with remainder repeats" do
      t = Nx.iota({5, 3})
      batches = Nx.to_batched(t, 2) |> Enum.to_list()
      assert length(batches) == 3
      # Last batch should be padded to size 2
      assert Nx.shape(List.last(batches)) == {2, 3}
    end

    test "to_batched with leftover :discard" do
      t = Nx.iota({5, 3})
      batches = Nx.to_batched(t, 2, leftover: :discard) |> Enum.to_list()
      assert length(batches) == 2
    end

    test "to_batched batch_size == tensor_size" do
      t = Nx.iota({4, 3})
      batches = Nx.to_batched(t, 4) |> Enum.to_list()
      assert length(batches) == 1
      assert Nx.to_flat_list(hd(batches)) == Nx.to_flat_list(t)
    end

    test "to_batched batch_size == 1" do
      t = Nx.iota({3})
      batches = Nx.to_batched(t, 1) |> Enum.to_list()
      assert length(batches) == 3
    end

    test "to_batched raises on scalar" do
      t = Nx.tensor(42)
      assert_raise ArgumentError, ~r/cannot batch .* scalar/, fn ->
        Nx.to_batched(t, 1) |> Enum.to_list()
      end
    end
  end

  # ── Type conversion edge cases ─────────────────────────────────────
  # Source: nx.ex:2744 (as_type), type.ex:201 (normalize!),
  #         nx.ex:1998 (from_binary), nx.ex:2819 (bitcast)

  describe "type conversion edge cases" do
    test "as_type same type is identity" do
      t = Nx.tensor([1.0, 2.0, 3.0], type: :f32)
      result = Nx.as_type(t, :f32)
      assert Nx.to_flat_list(result) == [1.0, 2.0, 3.0]
    end

    test "as_type f64 to f16 loses precision" do
      t = Nx.tensor(3.141592653589793, type: :f64)
      result = Nx.as_type(t, :f16)
      val = Nx.to_number(Nx.as_type(result, :f64))
      # f16 has ~3 decimal digits of precision
      assert_in_delta val, 3.14, 0.01
    end

    test "as_type f32 to s32 truncates" do
      t = Nx.tensor(3.7, type: :f32)
      result = Nx.as_type(t, :s32)
      assert Nx.to_number(result) == 3
    end

    test "as_type negative float to unsigned wraps" do
      t = Nx.tensor(-1.0, type: :f32)
      result = Nx.as_type(t, :u8)
      val = Nx.to_number(result)
      # Implementation-defined: may wrap or clamp
      assert is_integer(val)
    end

    test "as_type integer widening preserves value" do
      t = Nx.tensor(42, type: :s8)
      result = Nx.as_type(t, :s64)
      assert Nx.to_number(result) == 42
    end

    test "as_type u8 max to s8" do
      t = Nx.tensor(255, type: :u8)
      result = Nx.as_type(t, :s8)
      val = Nx.to_number(result)
      # 255 doesn't fit in s8 (-128..127), wraps to -1
      assert val == -1
    end

    test "Type.normalize! accepts atoms" do
      assert Nx.Type.normalize!(:f32) == {:f, 32}
      assert Nx.Type.normalize!(:s64) == {:s, 64}
      assert Nx.Type.normalize!(:u8) == {:u, 8}
      assert Nx.Type.normalize!(:bf16) == {:bf, 16}
      assert Nx.Type.normalize!(:c64) == {:c, 64}
      assert Nx.Type.normalize!(:c128) == {:c, 128}
    end

    test "Type.normalize! rejects invalid sizes" do
      assert_raise ArgumentError, fn -> Nx.Type.normalize!({:s, 0}) end
      assert_raise ArgumentError, fn -> Nx.Type.normalize!({:f, 48}) end
      assert_raise ArgumentError, fn -> Nx.Type.normalize!({:u, 128}) end
    end

    test "Type.merge promotes correctly" do
      assert Nx.Type.merge({:s, 8}, {:u, 8}) == {:s, 16}
      assert Nx.Type.merge({:f, 32}, {:s, 32}) == {:f, 32}
      assert Nx.Type.merge({:f, 32}, {:f, 64}) == {:f, 64}
      assert Nx.Type.merge({:bf, 16}, {:f, 32}) == {:f, 32}
    end

    test "from_binary rejects empty binary" do
      assert_raise ArgumentError, ~r/cannot build an empty tensor/, fn ->
        Nx.from_binary(<<>>, :f32)
      end
    end

    test "from_binary rejects misaligned binary" do
      assert_raise ArgumentError, ~r/binary does not match/, fn ->
        # 3 bytes can't be divided into 4-byte f32 elements
        Nx.from_binary(<<1, 2, 3>>, :f32)
      end
    end

    test "bitcast rejects complex types" do
      t = Nx.tensor(Complex.new(1.0, 2.0), type: :c64)
      assert_raise ArgumentError, ~r/does not support complex/, fn ->
        Nx.bitcast(t, :s64)
      end
    end

    test "to_heatmap rejects scalar" do
      t = Nx.tensor(42)
      assert_raise ArgumentError, ~r/cannot show heatmap for scalar/, fn ->
        Nx.to_heatmap(t)
      end
    end
  end

  # ── Defn edge cases ────────────────────────────────────────────────
  # Source: defn/expr.ex (while, cond), defn.ex (jit, grad)

  describe "defn while edge cases" do
    defn while_countdown(n) do
      count = Nx.tensor(0)

      {count, _} = while {count, n}, Nx.greater(n, 0) do
        {count + 1, n - 1}
      end

      count
    end

    test "while with 0 iterations" do
      result = while_countdown(Nx.tensor(0))
      assert Nx.to_number(result) == 0
    end

    test "while with 1 iteration" do
      result = while_countdown(Nx.tensor(1))
      assert Nx.to_number(result) == 1
    end

    test "while with many iterations" do
      result = while_countdown(Nx.tensor(100))
      assert Nx.to_number(result) == 100
    end

    defn while_sum_elements(values) do
      n = Nx.axis_size(values, 0)
      sum = Nx.tensor(0.0)
      i = Nx.tensor(0)

      {sum, _i, _v} =
        while {sum, i, values}, Nx.less(i, n) do
          {sum + values[i], i + 1, values}
        end

      sum
    end

    test "while iterates over all elements" do
      result = while_sum_elements(Nx.tensor([1.0, 2.0, 3.0, 4.0]))
      assert Nx.to_number(result) == 10.0
    end

    test "while with single-element tensor" do
      result = while_sum_elements(Nx.tensor([42.0]))
      assert Nx.to_number(result) == 42.0
    end
  end

  describe "defn cond edge cases" do
    defn cond_sign(x) do
      cond do
        Nx.greater(x, 0) -> Nx.tensor(1)
        Nx.less(x, 0) -> Nx.tensor(-1)
        true -> Nx.tensor(0)
      end
    end

    test "cond positive" do
      assert Nx.to_number(cond_sign(Nx.tensor(5))) == 1
    end

    test "cond negative" do
      assert Nx.to_number(cond_sign(Nx.tensor(-3))) == -1
    end

    test "cond zero" do
      assert Nx.to_number(cond_sign(Nx.tensor(0))) == 0
    end

    defn cond_clamp(x, lo, hi) do
      cond do
        Nx.less(x, lo) -> lo
        Nx.greater(x, hi) -> hi
        true -> x
      end
    end

    test "cond clamp at lower bound" do
      assert Nx.to_number(cond_clamp(Nx.tensor(-5.0), Nx.tensor(0.0), Nx.tensor(10.0))) == 0.0
    end

    test "cond clamp at upper bound" do
      assert Nx.to_number(cond_clamp(Nx.tensor(15.0), Nx.tensor(0.0), Nx.tensor(10.0))) == 10.0
    end

    test "cond clamp in range" do
      assert Nx.to_number(cond_clamp(Nx.tensor(5.0), Nx.tensor(0.0), Nx.tensor(10.0))) == 5.0
    end
  end

  describe "defn grad edge cases" do
    defn grad_identity(x), do: Nx.Defn.grad(x, &Nx.add(&1, 0))
    defn grad_square(x), do: Nx.Defn.grad(x, &Nx.multiply(&1, &1))
    defn grad_abs(x), do: Nx.Defn.grad(x, &Nx.abs/1)
    defn grad_relu(x), do: Nx.Defn.grad(x, &Nx.max(&1, 0))
    defn grad_gaussian(x), do: Nx.Defn.grad(x, fn x -> Nx.exp(Nx.negate(Nx.multiply(x, x))) end)

    test "grad of identity is 1" do
      result = grad_identity(Nx.tensor(5.0))
      assert Nx.to_number(result) == 1.0
    end

    test "grad of square at x=3 is 2x=6" do
      result = grad_square(Nx.tensor(3.0))
      assert_in_delta Nx.to_number(result), 6.0, 1.0e-5
    end

    test "grad of abs at positive is 1" do
      result = grad_abs(Nx.tensor(3.0))
      assert Nx.to_number(result) == 1.0
    end

    test "grad of abs at negative is -1" do
      result = grad_abs(Nx.tensor(-3.0))
      assert Nx.to_number(result) == -1.0
    end

    test "grad of relu at positive is 1" do
      result = grad_relu(Nx.tensor(3.0))
      assert Nx.to_number(result) == 1.0
    end

    test "grad of relu at negative is 0" do
      result = grad_relu(Nx.tensor(-3.0))
      assert Nx.to_number(result) == 0.0
    end

    test "grad of exp(-x^2) at x=0 is 0" do
      result = grad_gaussian(Nx.tensor(0.0))
      assert_in_delta Nx.to_number(result), 0.0, 1.0e-6
    end

    test "grad of exp(-x^2) at x=1 is -2*exp(-1)" do
      result = grad_gaussian(Nx.tensor(1.0))
      expected = -2.0 * :math.exp(-1.0)
      assert_in_delta Nx.to_number(result), expected, 1.0e-5
    end
  end

  # ── More algebraic equivalence tests ───────────────────────────────

  describe "matrix algebraic equivalences" do
    property "make_diagonal then take_diagonal recovers original" do
      check all(n <- integer(1..8), max_runs: 20) do
        v = Nx.iota({n})
        result = v |> Nx.make_diagonal() |> Nx.take_diagonal()
        assert Nx.to_flat_list(result) == Nx.to_flat_list(v)
      end
    end

    property "sum of diagonal equals trace" do
      check all(n <- integer(1..6), max_runs: 20) do
        t = Nx.iota({n, n}, type: :f32)
        trace = Nx.take_diagonal(t) |> Nx.sum() |> Nx.to_number()
        # Manual trace: sum of t[i][i] = sum of i*(n+1) for i in 0..n-1
        expected = Enum.sum(for i <- 0..(n - 1), do: i * (n + 1))
        assert_in_delta trace, expected * 1.0, 1.0e-4
      end
    end

    property "reverse(reverse(t)) == t for any axis" do
      check all(
              m <- integer(2..6),
              n <- integer(2..6),
              axis <- member_of([0, 1]),
              max_runs: 20
            ) do
        t = Nx.iota({m, n})
        result = t |> Nx.reverse(axes: [axis]) |> Nx.reverse(axes: [axis])
        assert Nx.to_flat_list(result) == Nx.to_flat_list(t)
      end
    end

    property "sort(sort(t)) == sort(t) (idempotent)" do
      check all(n <- integer(1..10), max_runs: 20) do
        # Use iota in reverse for interesting values
        t = Nx.subtract(Nx.tensor(n), Nx.iota({n}))
        sorted_once = Nx.sort(t)
        sorted_twice = Nx.sort(sorted_once)
        assert Nx.to_flat_list(sorted_once) == Nx.to_flat_list(sorted_twice)
      end
    end

    property "argsort then take recovers sorted tensor" do
      check all(n <- integer(1..10), max_runs: 20) do
        t = Nx.subtract(Nx.tensor(n), Nx.iota({n}))
        indices = Nx.argsort(t)
        recovered = Nx.take(t, indices)
        expected = Nx.sort(t)
        assert Nx.to_flat_list(recovered) == Nx.to_flat_list(expected)
      end
    end

    property "top_k values are first k of sorted descending" do
      check all(
              n <- integer(2..10),
              max_runs: 20
            ) do
        k = div(n, 2) + 1
        t = Nx.iota({n}, type: :f32)
        {values, _indices} = Nx.top_k(t, k: k)
        sorted_desc = Nx.sort(t, direction: :desc)
        top_sorted = Nx.slice(sorted_desc, [0], [k])
        assert Nx.to_flat_list(values) == Nx.to_flat_list(top_sorted)
      end
    end

    property "tile([n]) then slice recovers original" do
      check all(
              size <- integer(1..6),
              reps <- integer(2..4),
              max_runs: 20
            ) do
        t = Nx.iota({size}, type: :f32)
        tiled = Nx.tile(t, [reps])
        recovered = Nx.slice(tiled, [0], [size])
        assert Nx.to_flat_list(recovered) == Nx.to_flat_list(t)
      end
    end

    property "flatten then reshape recovers original" do
      check all(
              m <- integer(1..5),
              n <- integer(1..5),
              max_runs: 20
            ) do
        t = Nx.iota({m, n})
        result = t |> Nx.flatten() |> Nx.reshape({m, n})
        assert Nx.to_flat_list(result) == Nx.to_flat_list(t)
      end
    end

    property "broadcast({1,n}, {m,n}) rows are identical" do
      check all(
              m <- integer(2..5),
              n <- integer(1..5),
              max_runs: 20
            ) do
        row = Nx.iota({1, n}, type: :f32)
        result = Nx.broadcast(row, {m, n})
        first_row = Nx.slice(result, [0, 0], [1, n]) |> Nx.to_flat_list()
        last_row = Nx.slice(result, [m - 1, 0], [1, n]) |> Nx.to_flat_list()
        assert first_row == last_row
      end
    end
  end

  # ── LazyContainer edge cases ───────────────────────────────────────

  describe "LazyContainer edge cases" do
    test "boolean raises with helpful message" do
      assert_raise Protocol.UndefinedError, ~r/booleans are not valid tensors/, fn ->
        Nx.LazyContainer.traverse(true, nil, fn _, _ -> nil end)
      end
    end

    test "list raises with helpful message" do
      assert_raise Protocol.UndefinedError, ~r/lists are not valid tensors/, fn ->
        Nx.LazyContainer.traverse([1, 2, 3], nil, fn _, _ -> nil end)
      end
    end
  end
end
