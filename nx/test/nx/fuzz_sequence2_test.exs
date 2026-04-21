defmodule Nx.FuzzSequence2Test do
  @moduledoc """
  Tier 5 (extended): Defn control flow sequences, grad through chains,
  concurrent ops, exotic types, high-rank tensors, vectorized binary ops.
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  import Nx.Defn

  # ── Task #25: Defn while/cond inside op sequences ──────────────────

  describe "defn control flow sequences" do
    defn while_accumulate(t) do
      n = Nx.axis_size(t, 0)
      acc = Nx.tensor(0.0)
      i = Nx.tensor(0)

      {acc, _i, _t} =
        while {acc, i, t}, Nx.less(i, n) do
          {acc + t[i], i + 1, t}
        end

      acc
    end

    test "while loop sum matches Nx.sum" do
      for n <- [1, 3, 5, 8] do
        t = Nx.iota({n}, type: :f32)
        while_result = Nx.to_number(while_accumulate(t))
        sum_result = Nx.to_number(Nx.sum(t))
        assert_in_delta while_result, sum_result, 1.0e-4
      end
    end

    defn cond_abs(x) do
      if Nx.greater_equal(x, 0) do
        x
      else
        Nx.negate(x)
      end
    end

    test "cond abs matches Nx.abs for various values" do
      for val <- [-5.0, -1.0, -0.0, 0.0, 1.0, 5.0] do
        t = Nx.tensor(val)
        cond_result = Nx.to_number(cond_abs(t))
        abs_result = Nx.to_number(Nx.abs(t))
        assert_in_delta cond_result, abs_result, 1.0e-6
      end
    end

    defn cond_relu_chain(x) do
      x = Nx.multiply(x, 2)

      x =
        if Nx.greater(x, 0) do
          x
        else
          Nx.tensor(0.0)
        end

      Nx.add(x, 1)
    end

    test "cond_relu_chain: positive path" do
      assert Nx.to_number(cond_relu_chain(Nx.tensor(3.0))) == 7.0
    end

    test "cond_relu_chain: negative path" do
      assert Nx.to_number(cond_relu_chain(Nx.tensor(-3.0))) == 1.0
    end

    test "cond_relu_chain: zero path" do
      assert Nx.to_number(cond_relu_chain(Nx.tensor(0.0))) == 1.0
    end

    defn while_then_reduce(t) do
      n = Nx.axis_size(t, 0)
      doubled = Nx.tensor(0.0)
      i = Nx.tensor(0)

      {doubled, _i, _t} =
        while {doubled, i, t}, Nx.less(i, n) do
          {doubled + t[i] * 2, i + 1, t}
        end

      doubled
    end

    test "while_then_reduce matches 2*sum" do
      for n <- [1, 4, 7] do
        t = Nx.iota({n}, type: :f32)
        result = Nx.to_number(while_then_reduce(t))
        expected = Nx.to_number(Nx.sum(Nx.multiply(t, 2)))
        assert_in_delta result, expected, 1.0e-3
      end
    end

    defn nested_cond(x) do
      cond do
        Nx.greater(x, 10) -> Nx.tensor(3)
        Nx.greater(x, 0) -> Nx.tensor(2)
        Nx.equal(x, 0) -> Nx.tensor(1)
        true -> Nx.tensor(0)
      end
    end

    test "nested cond hits each branch" do
      assert Nx.to_number(nested_cond(Nx.tensor(20.0))) == 3
      assert Nx.to_number(nested_cond(Nx.tensor(5.0))) == 2
      assert Nx.to_number(nested_cond(Nx.tensor(0.0))) == 1
      assert Nx.to_number(nested_cond(Nx.tensor(-5.0))) == 0
    end

    defn while_with_cond(x) do
      {x, count} =
        while {x, count = Nx.tensor(0)}, Nx.less(count, 5) do
          new_x =
            if Nx.greater(x, 0) do
              x - 1
            else
              x + 1
            end

          {new_x, count + 1}
        end

      x
    end

    test "while with cond inside: positive start" do
      result = Nx.to_number(while_with_cond(Nx.tensor(3.0)))
      # 3 -> 2 -> 1 -> 0 -> 1 -> 0 (5 iterations)
      assert_in_delta result, 0.0, 1.0e-5
    end

    test "while with cond inside: negative start" do
      result = Nx.to_number(while_with_cond(Nx.tensor(-3.0)))
      # -3 -> -2 -> -1 -> 0 -> 1 -> 0 (5 iterations)
      assert_in_delta result, 0.0, 1.0e-5
    end
  end

  # ── Task #26: Grad through op chains ───────────────────────────────

  describe "grad through op chains" do
    defn grad_chain_1(x) do
      Nx.Defn.grad(x, fn x ->
        x |> Nx.multiply(3) |> Nx.add(2) |> Nx.sum()
      end)
    end

    test "grad of linear chain is constant" do
      result = grad_chain_1(Nx.tensor([1.0, 2.0, 3.0]))
      # d/dx (3x + 2) = 3 for each element
      assert Nx.to_flat_list(result) == [3.0, 3.0, 3.0]
    end

    defn grad_chain_2(x) do
      Nx.Defn.grad(x, fn x ->
        x |> Nx.multiply(x) |> Nx.sum()
      end)
    end

    test "grad of x^2 chain is 2x" do
      result = grad_chain_2(Nx.tensor([1.0, 2.0, 3.0]))
      assert Nx.to_flat_list(result) == [2.0, 4.0, 6.0]
    end

    defn grad_exp_chain(x) do
      Nx.Defn.grad(x, fn x ->
        x |> Nx.exp() |> Nx.sum()
      end)
    end

    test "grad of exp chain is exp(x)" do
      t = Nx.tensor([0.0, 1.0, 2.0])
      result = grad_exp_chain(t)
      expected = Nx.exp(t)

      for {r, e} <- Enum.zip(Nx.to_flat_list(result), Nx.to_flat_list(expected)) do
        assert_in_delta r, e, 1.0e-5
      end
    end

    defn grad_tanh_chain(x) do
      Nx.Defn.grad(x, fn x ->
        x |> Nx.tanh() |> Nx.sum()
      end)
    end

    test "grad of tanh doesn't produce NaN for moderate values" do
      t = Nx.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
      result = grad_tanh_chain(t)

      for val <- Nx.to_flat_list(result) do
        refute val == :nan
        refute val == :infinity
        # tanh' = 1 - tanh^2, always in (0, 1]
        assert val >= 0.0 and val <= 1.0 + 1.0e-6
      end
    end

    defn grad_composition(x) do
      Nx.Defn.grad(x, fn x ->
        x
        |> Nx.multiply(2)
        |> Nx.sin()
        |> Nx.abs()
        |> Nx.sum()
      end)
    end

    test "grad of composed chain doesn't crash" do
      t = Nx.tensor([0.5, 1.0, 1.5])
      result = grad_composition(t)
      assert Nx.shape(result) == {3}

      for val <- Nx.to_flat_list(result) do
        refute val == :nan
      end
    end

    defn grad_through_cond(x) do
      Nx.Defn.grad(x, fn x ->
        if Nx.greater(Nx.sum(x), 0) do
          Nx.sum(Nx.multiply(x, x))
        else
          Nx.sum(Nx.negate(x))
        end
      end)
    end

    test "grad through cond: positive branch (quadratic)" do
      t = Nx.tensor([1.0, 2.0, 3.0])
      result = grad_through_cond(t)
      # d/dx(x^2) = 2x
      assert Nx.to_flat_list(result) == [2.0, 4.0, 6.0]
    end

    test "grad through cond: negative branch (linear)" do
      t = Nx.tensor([-1.0, -2.0, -3.0])
      result = grad_through_cond(t)
      # d/dx(-x) = -1
      assert Nx.to_flat_list(result) == [-1.0, -1.0, -1.0]
    end

    defn grad_through_while(x) do
      Nx.Defn.grad(x, fn x ->
        {result, _i} =
          while {x, i = Nx.tensor(0)}, Nx.less(i, 3) do
            {Nx.multiply(x, x), i + 1}
          end

        Nx.sum(result)
      end)
    end

    test "grad through while loop (x^(2^3) = x^8)" do
      t = Nx.tensor([2.0])
      result = grad_through_while(t)
      # x^8, grad = 8*x^7 = 8*128 = 1024
      [val] = Nx.to_flat_list(result)
      assert_in_delta val, 1024.0, 1.0
    end
  end

  # ── Task #27: Concurrent tensor operations ─────────────────────────

  describe "concurrent tensor operations" do
    test "concurrent independent computations don't crash" do
      tasks =
        for i <- 1..20 do
          Task.async(fn ->
            t = Nx.iota({10, 10}, type: :f32)

            t
            |> Nx.multiply(i)
            |> Nx.add(1)
            |> Nx.tanh()
            |> Nx.sum()
            |> Nx.to_number()
          end)
        end

      results = Task.await_many(tasks, 10_000)
      assert length(results) == 20
      assert Enum.all?(results, &is_float/1)
    end

    test "concurrent reads of same tensor" do
      shared = Nx.iota({100}, type: :f32)

      tasks =
        for _ <- 1..20 do
          Task.async(fn ->
            Nx.to_flat_list(shared)
          end)
        end

      results = Task.await_many(tasks, 10_000)
      expected = Nx.to_flat_list(shared)

      for result <- results do
        assert result == expected
      end
    end

    test "concurrent reductions on same tensor" do
      shared = Nx.iota({50, 50}, type: :f32)

      tasks =
        for op <- [:sum, :product, :reduce_max, :reduce_min, :mean] do
          Task.async(fn ->
            result = apply(Nx, op, [Nx.as_type(shared, :f32)])
            {op, Nx.to_number(result)}
          end)
        end

      results = Task.await_many(tasks, 10_000)
      assert length(results) == 5

      for {op, val} <- results do
        assert is_number(val) or val == :infinity or val == :neg_infinity,
          "#{op} returned non-number: #{inspect(val)}"
      end
    end

    test "concurrent op chains on independent tensors" do
      tasks =
        for i <- 1..10 do
          Task.async(fn ->
            t = Nx.iota({i + 2, 3}, type: :f32)

            result =
              t
              |> Nx.multiply(2)
              |> Nx.add(1)
              |> Nx.abs()
              |> Nx.sum(axes: [1])
              |> Nx.to_flat_list()

            {i, result}
          end)
        end

      results = Task.await_many(tasks, 10_000)
      assert length(results) == 10

      for {i, vals} <- results do
        assert length(vals) == i + 2
      end
    end
  end

  # ── Task #28: Exotic type conversion chains ────────────────────────

  describe "exotic type conversion chains" do
    test "f16 -> bf16 -> f32 -> f64 preserves approximate value" do
      t = Nx.tensor(3.14, type: :f16)

      result =
        t
        |> Nx.as_type(:bf16)
        |> Nx.as_type(:f32)
        |> Nx.as_type(:f64)

      assert Nx.type(result) == {:f, 64}
      assert_in_delta Nx.to_number(result), 3.14, 0.02
    end

    test "u8 -> s32 -> f16 -> bf16 -> f64 chain" do
      t = Nx.tensor(42, type: :u8)

      result =
        t
        |> Nx.as_type(:s32)
        |> Nx.as_type(:f16)
        |> Nx.as_type(:bf16)
        |> Nx.as_type(:f64)

      assert Nx.type(result) == {:f, 64}
      assert_in_delta Nx.to_number(result), 42.0, 0.5
    end

    test "s8 -> u16 -> f32 -> s64 chain" do
      t = Nx.tensor(100, type: :s8)

      result =
        t
        |> Nx.as_type(:u16)
        |> Nx.as_type(:f32)
        |> Nx.as_type(:s64)

      assert Nx.type(result) == {:s, 64}
      assert Nx.to_number(result) == 100
    end

    test "float -> complex -> real roundtrip" do
      t = Nx.tensor([1.0, 2.0, 3.0], type: :f32)
      complex = Nx.as_type(t, :c64)
      assert Nx.type(complex) == {:c, 64}

      back = Nx.real(complex)
      assert Nx.type(back) == {:f, 32}
      assert Nx.to_flat_list(back) == [1.0, 2.0, 3.0]
    end

    test "f64 -> c128 -> real -> f32 chain" do
      t = Nx.tensor(2.718, type: :f64)

      result =
        t
        |> Nx.as_type(:c128)
        |> Nx.real()
        |> Nx.as_type(:f32)

      assert Nx.type(result) == {:f, 32}
      assert_in_delta Nx.to_number(result), 2.718, 1.0e-3
    end

    test "type chain preserves tensor shape" do
      t = Nx.iota({3, 4}, type: :s32)

      result =
        t
        |> Nx.as_type(:f32)
        |> Nx.as_type(:f16)
        |> Nx.as_type(:bf16)
        |> Nx.as_type(:f64)
        |> Nx.as_type(:s64)
        |> Nx.as_type(:f32)

      assert Nx.shape(result) == {3, 4}
      assert Nx.type(result) == {:f, 32}
    end

    property "random type conversion chain preserves shape" do
      types = [:f32, :f64, :s32, :s64, :f16, :bf16]

      check all(
              n <- integer(2..6),
              chain <- list_of(member_of(types), length: 5),
              max_runs: 20
            ) do
        t = Nx.iota({n}, type: :f32)
        result = Enum.reduce(chain, t, &Nx.as_type(&2, &1))
        assert Nx.shape(result) == {n}
      end
    end
  end

  # ── Task #29: High-rank tensors through op sequences ───────────────

  describe "high-rank tensor sequences" do
    test "rank 5 through math chain" do
      t = Nx.iota({2, 2, 2, 2, 2}, type: :f32)

      result =
        t
        |> Nx.multiply(2)
        |> Nx.add(1)
        |> Nx.tanh()
        |> Nx.abs()

      assert Nx.shape(result) == {2, 2, 2, 2, 2}
      assert Nx.size(result) == 32
    end

    test "rank 6 through math chain" do
      t = Nx.iota({2, 2, 2, 2, 2, 2}, type: :f32)

      result =
        t
        |> Nx.add(1)
        |> Nx.negate()
        |> Nx.abs()
        |> Nx.sign()

      assert Nx.shape(result) == {2, 2, 2, 2, 2, 2}
      assert Nx.size(result) == 64
    end

    test "rank 5 flatten then reshape back" do
      t = Nx.iota({2, 2, 2, 2, 2}, type: :f32)
      flat = Nx.flatten(t)
      assert Nx.shape(flat) == {32}

      back = Nx.reshape(flat, {2, 2, 2, 2, 2})
      assert Nx.to_flat_list(back) == Nx.to_flat_list(t)
    end

    test "rank 5 reduction along various axes" do
      t = Nx.iota({2, 3, 2, 2, 2}, type: :f32)

      r0 = Nx.sum(t, axes: [0])
      assert Nx.shape(r0) == {3, 2, 2, 2}

      r_last = Nx.sum(t, axes: [4])
      assert Nx.shape(r_last) == {2, 3, 2, 2}

      r_mid = Nx.sum(t, axes: [2])
      assert Nx.shape(r_mid) == {2, 3, 2, 2}
    end

    test "rank 5 transpose" do
      t = Nx.iota({2, 3, 2, 2, 2}, type: :f32)
      result = Nx.transpose(t)
      assert Nx.shape(result) == {2, 2, 2, 3, 2}
    end

    test "rank 5 reverse along multiple axes" do
      t = Nx.iota({2, 2, 2, 2, 2}, type: :f32)
      result = Nx.reverse(t, axes: [0, 2, 4])
      assert Nx.shape(result) == {2, 2, 2, 2, 2}

      # Double reverse is identity
      back = Nx.reverse(result, axes: [0, 2, 4])
      assert Nx.to_flat_list(back) == Nx.to_flat_list(t)
    end

    test "rank 5 slice" do
      t = Nx.iota({3, 3, 3, 3, 3}, type: :f32)
      result = Nx.slice(t, [1, 1, 1, 1, 1], [2, 2, 2, 2, 2])
      assert Nx.shape(result) == {2, 2, 2, 2, 2}
    end

    test "rank 5 new_axis then squeeze" do
      t = Nx.iota({2, 2, 2, 2, 2}, type: :f32)
      expanded = Nx.new_axis(t, 0)
      assert Nx.shape(expanded) == {1, 2, 2, 2, 2, 2}
      assert tuple_size(Nx.shape(expanded)) == 6

      squeezed = Nx.squeeze(expanded, axes: [0])
      assert Nx.shape(squeezed) == {2, 2, 2, 2, 2}
      assert Nx.to_flat_list(squeezed) == Nx.to_flat_list(t)
    end

    test "rank 5 window_sum" do
      t = Nx.iota({3, 3, 3, 3, 3}, type: :f32)
      result = Nx.window_sum(t, {2, 2, 2, 2, 2})
      assert Nx.shape(result) == {2, 2, 2, 2, 2}
    end
  end

  # ── Task #30: Vectorized binary op sequences ───────────────────────

  describe "vectorized binary op sequences" do
    test "two vectorized tensors with same axis: element-wise ops" do
      a = Nx.iota({3, 4}, type: :f32) |> Nx.vectorize(:batch)
      b = Nx.broadcast(Nx.tensor(1.0), {3, 4}) |> Nx.vectorize(:batch)

      result = a |> Nx.add(b) |> Nx.multiply(b) |> Nx.subtract(b)
      assert result.vectorized_axes == [batch: 3]
      assert result.shape == {4}

      devec = Nx.devectorize(result)
      assert Nx.shape(devec) == {3, 4}
    end

    test "vectorized tensor + scalar broadcasting" do
      t = Nx.iota({2, 5}, type: :f32) |> Nx.vectorize(:batch)

      result =
        t
        |> Nx.add(10)
        |> Nx.multiply(2)
        |> Nx.subtract(5)

      assert result.vectorized_axes == [batch: 2]
      devec = Nx.devectorize(result)
      assert Nx.shape(devec) == {2, 5}
    end

    test "vectorized chain: add -> tanh -> multiply -> sum" do
      a = Nx.iota({3, 4}, type: :f32) |> Nx.vectorize(:batch)
      b = Nx.broadcast(Nx.tensor(0.5), {3, 4}) |> Nx.vectorize(:batch)

      result =
        a
        |> Nx.add(b)
        |> Nx.tanh()
        |> Nx.multiply(b)
        |> Nx.sum()

      assert result.vectorized_axes == [batch: 3]
      assert result.shape == {}
    end

    test "vectorized binary ops preserve vectorized axes through chain" do
      t = Nx.iota({2, 3}, type: :f32) |> Nx.vectorize(:batch)

      result =
        t
        |> Nx.multiply(2)
        |> Nx.add(Nx.broadcast(Nx.tensor(1.0), {2, 3}) |> Nx.vectorize(:batch))
        |> Nx.abs()
        |> Nx.negate()

      assert result.vectorized_axes == [batch: 2]
      assert result.shape == {3}
    end

    test "double vectorized binary ops" do
      a = Nx.iota({2, 3, 4}, type: :f32)
        |> Nx.vectorize(:outer)
        |> Nx.vectorize(:inner)

      b = Nx.broadcast(Nx.tensor(1.0), {2, 3, 4})
        |> Nx.vectorize(:outer)
        |> Nx.vectorize(:inner)

      result = Nx.add(a, b)
      assert result.vectorized_axes == [outer: 2, inner: 3]
      assert result.shape == {4}
    end

    test "vectorized dot product" do
      # Batch of vector dot products
      a = Nx.iota({3, 4}, type: :f32) |> Nx.vectorize(:batch)
      b = Nx.broadcast(Nx.tensor(1.0), {3, 4}) |> Nx.vectorize(:batch)

      result = Nx.dot(a, b)
      assert result.vectorized_axes == [batch: 3]
      assert result.shape == {}

      devec = Nx.devectorize(result)
      # batch 0: dot([0,1,2,3], [1,1,1,1]) = 6
      assert_in_delta Nx.to_number(devec[0]), 6.0, 1.0e-4
    end

    test "vectorized with mismatched axis sizes raises" do
      a = Nx.iota({2, 3}) |> Nx.vectorize(:batch)
      b = Nx.iota({4, 3}) |> Nx.vectorize(:batch)

      assert_raise ArgumentError, ~r/expected vectorized axis :batch/, fn ->
        Nx.add(a, b)
      end
    end
  end
end
