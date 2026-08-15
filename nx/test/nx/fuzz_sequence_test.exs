defmodule Nx.FuzzSequenceTest do
  @moduledoc """
  Tier 5: Stateful / sequence testing.

  Generates random sequences of valid tensor operations and checks
  that invariants hold after each step:
  - Shape consistency: output shape matches Nx.shape/1
  - Type consistency: output type matches Nx.type/1
  - No crashes: valid op sequences never crash
  - Data integrity: values accessible via to_flat_list without crash
  - Vectorized axes: preserved correctly through chains
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  # ── Op definitions ─────────────────────────────────────────────────
  # Each op is an atom name. apply_op/2 dispatches to the implementation.

  @op_names [
    :abs,
    :negate,
    :sign,
    :sin,
    :cos,
    :tanh,
    :sigmoid,
    :exp,
    :log,
    :sqrt,
    :rsqrt,
    :flatten,
    :squeeze_ones,
    :new_axis_0,
    :new_axis_last,
    :reverse,
    :sum,
    :mean,
    :product,
    :sum_axis0,
    :reduce_max_axis0,
    :add_1,
    :sub_1,
    :mul_2,
    :add_0,
    :mul_1,
    :to_f32,
    :to_s32,
    :to_f64,
    # Richer ops (task #20)
    :window_sum_2,
    :pad_1,
    :diff_1,
    :tile_2,
    :gather_first,
    :indexed_put_zero,
    :random_reshape,
    :sort_asc,
    :cumulative_sum
  ]

  defp apply_op(:abs, t), do: Nx.abs(t)
  defp apply_op(:negate, t), do: Nx.negate(t)
  defp apply_op(:sign, t), do: Nx.sign(t)
  defp apply_op(:sin, t), do: Nx.sin(Nx.as_type(t, :f32))
  defp apply_op(:cos, t), do: Nx.cos(Nx.as_type(t, :f32))
  defp apply_op(:tanh, t), do: Nx.tanh(Nx.as_type(t, :f32))
  defp apply_op(:sigmoid, t), do: Nx.sigmoid(Nx.as_type(t, :f32))
  defp apply_op(:exp, t), do: Nx.exp(Nx.clip(Nx.as_type(t, :f32), -10.0, 10.0))
  defp apply_op(:log, t), do: Nx.log(Nx.add(Nx.abs(Nx.as_type(t, :f32)), 1.0))
  defp apply_op(:sqrt, t), do: Nx.sqrt(Nx.abs(Nx.as_type(t, :f32)))
  defp apply_op(:rsqrt, t), do: Nx.rsqrt(Nx.add(Nx.abs(Nx.as_type(t, :f32)), 1.0))

  defp apply_op(:flatten, t), do: Nx.flatten(t)

  defp apply_op(:squeeze_ones, t) do
    ones = for {d, i} <- Enum.with_index(Tuple.to_list(Nx.shape(t))), d == 1, do: i
    if ones == [], do: t, else: Nx.squeeze(t, axes: ones)
  end

  defp apply_op(:new_axis_0, t), do: Nx.new_axis(t, 0)
  defp apply_op(:new_axis_last, t), do: Nx.new_axis(t, -1)

  defp apply_op(:reverse, t) do
    if tuple_size(Nx.shape(t)) == 0, do: t, else: Nx.reverse(t)
  end

  defp apply_op(:sum, t), do: Nx.sum(Nx.as_type(t, :f32))
  defp apply_op(:mean, t), do: Nx.mean(Nx.as_type(t, :f32))
  defp apply_op(:product, t), do: Nx.product(Nx.clip(Nx.as_type(t, :f32), -2.0, 2.0))

  defp apply_op(:sum_axis0, t) do
    if tuple_size(Nx.shape(t)) == 0, do: t, else: Nx.sum(Nx.as_type(t, :f32), axes: [0])
  end

  defp apply_op(:reduce_max_axis0, t) do
    if tuple_size(Nx.shape(t)) == 0, do: t, else: Nx.reduce_max(t, axes: [0])
  end

  defp apply_op(:add_1, t), do: Nx.add(t, 1)
  defp apply_op(:sub_1, t), do: Nx.subtract(t, 1)
  defp apply_op(:mul_2, t), do: Nx.multiply(t, 2)
  defp apply_op(:add_0, t), do: Nx.add(t, 0)
  defp apply_op(:mul_1, t), do: Nx.multiply(t, 1)

  defp apply_op(:to_f32, t), do: Nx.as_type(t, :f32)
  defp apply_op(:to_s32, t), do: Nx.as_type(t, :s32)
  defp apply_op(:to_f64, t), do: Nx.as_type(t, :f64)

  # Richer ops (task #20)
  defp apply_op(:window_sum_2, t) do
    rank = tuple_size(Nx.shape(t))

    if rank == 0 do
      t
    else
      # Window of size 2 (or 1 if dim is 1) along each axis
      window = List.to_tuple(for d <- Tuple.to_list(Nx.shape(t)), do: min(d, 2))
      Nx.window_sum(Nx.as_type(t, :f32), window)
    end
  end

  defp apply_op(:pad_1, t) do
    rank = tuple_size(Nx.shape(t))

    if rank == 0 do
      t
    else
      config = List.duplicate({1, 0, 0}, rank)
      Nx.pad(t, Nx.tensor(0, type: Nx.type(t)), config)
    end
  end

  defp apply_op(:diff_1, t) do
    rank = tuple_size(Nx.shape(t))

    if rank == 0 or elem(Nx.shape(t), rank - 1) < 2 do
      t
    else
      Nx.diff(t)
    end
  end

  defp apply_op(:tile_2, t) do
    rank = tuple_size(Nx.shape(t))

    if rank == 0 or Nx.size(t) > 500 do
      t
    else
      reps = List.duplicate(1, max(rank - 1, 0)) ++ [2]
      Nx.tile(t, reps)
    end
  end

  defp apply_op(:gather_first, t) do
    rank = tuple_size(Nx.shape(t))

    if rank == 0 do
      t
    else
      # Gather the first element along first axis
      idx = Nx.tensor([List.duplicate(0, rank)])
      Nx.gather(t, idx)
    end
  end

  defp apply_op(:indexed_put_zero, t) do
    rank = tuple_size(Nx.shape(t))

    if rank == 0 do
      t
    else
      idx = Nx.tensor([List.duplicate(0, rank)])
      val = Nx.tensor([0], type: Nx.type(t))
      Nx.indexed_put(t, idx, val)
    end
  end

  # Random reshape to compatible shape (task #21)
  defp apply_op(:random_reshape, t) do
    size = Nx.size(t)

    if size <= 1 do
      t
    else
      new_shape = random_compatible_shape(size)
      Nx.reshape(t, new_shape)
    end
  end

  defp apply_op(:sort_asc, t) do
    rank = tuple_size(Nx.shape(t))
    if rank == 0, do: t, else: Nx.sort(t)
  end

  defp apply_op(:cumulative_sum, t) do
    rank = tuple_size(Nx.shape(t))
    if rank == 0, do: t, else: Nx.cumulative_sum(Nx.as_type(t, :f32))
  end

  # Generate a random compatible shape for a given total size
  defp random_compatible_shape(size) do
    factors = factorize(size)
    # Pick a random grouping of factors
    case Enum.random(1..min(length(factors), 4)) do
      1 ->
        {size}

      n_dims ->
        dims = group_factors(factors, n_dims)
        List.to_tuple(dims)
    end
  end

  defp factorize(1), do: [1]

  defp factorize(n) when n > 1 do
    Enum.reduce(2..n, {n, []}, fn f, {remaining, factors} ->
      do_factor(remaining, f, factors)
    end)
    |> elem(1)
    |> Enum.reverse()
  end

  defp do_factor(remaining, f, factors) when rem(remaining, f) == 0 do
    do_factor(div(remaining, f), f, [f | factors])
  end

  defp do_factor(remaining, _f, factors), do: {remaining, factors}

  defp group_factors(factors, n_dims) when length(factors) <= n_dims do
    padding = List.duplicate(1, n_dims - length(factors))
    Enum.shuffle(factors ++ padding)
  end

  defp group_factors(factors, n_dims) do
    # Randomly merge adjacent factors until we have n_dims groups
    factors
    |> Enum.shuffle()
    |> Enum.chunk_every(ceil(length(factors) / n_dims))
    |> Enum.map(&Enum.product/1)
    |> Enum.take(n_dims)
  end

  # ── Generators ─────────────────────────────────────────────────────

  defp op_gen do
    member_of(@op_names)
  end

  defp op_sequence(min_len, max_len) do
    bind(integer(min_len..max_len), fn len ->
      list_of(op_gen(), length: len)
    end)
  end

  defp initial_shape do
    member_of([
      {3},
      {4},
      {2, 3},
      {3, 4},
      {2, 3, 2},
      {4, 2}
    ])
  end

  defp initial_type do
    member_of([:f32, :s32, :f64, :s64])
  end

  # ── Invariant checks ───────────────────────────────────────────────

  defp check_invariants(tensor, step_name) do
    # 1. Shape is a valid tuple
    shape = Nx.shape(tensor)
    assert is_tuple(shape), "#{step_name}: shape is not a tuple"

    # 2. All dimensions are non-negative integers
    for {d, i} <- Enum.with_index(Tuple.to_list(shape)) do
      assert is_integer(d) and d >= 0,
             "#{step_name}: dimension #{i} is #{inspect(d)}"
    end

    # 3. Type is valid
    {type_class, type_bits} = Nx.type(tensor)

    assert type_class in [:u, :s, :f, :bf, :c],
           "#{step_name}: invalid type class #{inspect(type_class)}"

    assert is_integer(type_bits) and type_bits > 0,
           "#{step_name}: invalid type bits #{inspect(type_bits)}"

    # 4. Size matches shape product
    expected_size = if shape == {}, do: 1, else: Tuple.product(shape)

    assert Nx.size(tensor) == expected_size,
           "#{step_name}: size #{Nx.size(tensor)} != expected #{expected_size}"

    # 5. Rank matches shape tuple_size
    assert Nx.rank(tensor) == tuple_size(shape),
           "#{step_name}: rank mismatch"

    # 6. Can extract data without crashing (if reasonably sized)
    if Nx.size(tensor) <= 1000 do
      flat = Nx.to_flat_list(tensor)
      assert is_list(flat), "#{step_name}: to_flat_list returned non-list"

      assert length(flat) == expected_size,
             "#{step_name}: flat list length #{length(flat)} != #{expected_size}"
    end

    tensor
  end

  # ── Sequence runner ────────────────────────────────────────────────

  defp run_sequence(tensor, ops) do
    Enum.reduce(ops, {tensor, 0}, fn op_name, {t, step} ->
      result = apply_op(op_name, t)
      check_invariants(result, "step #{step} (#{op_name})")
      {result, step + 1}
    end)
  end

  # ── Tests ──────────────────────────────────────────────────────────

  describe "random op sequences" do
    property "random sequence of 3-8 ops preserves invariants" do
      check all(
              shape <- initial_shape(),
              type <- initial_type(),
              ops <- op_sequence(3, 8),
              # T1.1 retrofit: hostile-but-finite values; max_mag keeps
              # exp() in the chain vocabulary finite (exp(20) ~ 4.8e8)
              tensor <-
                FuzzGen.value_mixed_tensor(shape, Nx.Type.normalize!(type), max_mag: 20.0),
              max_runs: 50
            ) do
        check_invariants(tensor, "initial")
        {final, _steps} = run_sequence(tensor, ops)
        assert is_struct(final, Nx.Tensor)
      end
    end

    property "long sequence of 8-15 ops preserves invariants" do
      check all(
              shape <- initial_shape(),
              type <- initial_type(),
              ops <- op_sequence(8, 15),
              tensor <-
                FuzzGen.value_mixed_tensor(shape, Nx.Type.normalize!(type), max_mag: 20.0),
              max_runs: 30
            ) do
        {final, _steps} = run_sequence(tensor, ops)
        assert is_struct(final, Nx.Tensor)
      end
    end
  end

  describe "targeted sequence patterns" do
    property "shape-changing ops chain: new_axis -> flatten -> new_axis -> squeeze" do
      check all(
              shape <- initial_shape(),
              type <- initial_type(),
              max_runs: 20
            ) do
        tensor = Nx.iota(shape, type: type)

        result =
          tensor
          |> Nx.new_axis(0)
          |> Nx.flatten()
          |> Nx.new_axis(-1)

        check_invariants(result, "shape chain")

        # Squeeze the added dim
        result2 = Nx.squeeze(result, axes: [1])
        check_invariants(result2, "after squeeze")

        # Should have same number of elements
        assert Nx.size(result2) == Nx.size(tensor)
      end
    end

    property "type conversion chain preserves element count" do
      check all(
              shape <- initial_shape(),
              max_runs: 20
            ) do
        tensor = Nx.iota(shape, type: :s32)

        result =
          tensor
          |> Nx.as_type(:f32)
          |> Nx.as_type(:f64)
          |> Nx.as_type(:s64)
          |> Nx.as_type(:f32)

        check_invariants(result, "type chain")
        assert Nx.shape(result) == shape
        assert Nx.type(result) == {:f, 32}
      end
    end

    property "reduction then broadcast recovers shape (different values)" do
      check all(
              m <- integer(2..5),
              n <- integer(2..5),
              max_runs: 20
            ) do
        tensor = Nx.iota({m, n}, type: :f32)

        # Reduce along axis 1, then broadcast back
        reduced = Nx.sum(tensor, axes: [1], keep_axes: true)
        check_invariants(reduced, "after reduce")
        assert Nx.shape(reduced) == {m, 1}

        broadcasted = Nx.broadcast(reduced, {m, n})
        check_invariants(broadcasted, "after broadcast")
        assert Nx.shape(broadcasted) == {m, n}
      end
    end

    property "math ops chain doesn't produce NaN from safe inputs" do
      check all(
              n <- integer(1..8),
              max_runs: 30
            ) do
        # Start with values in [1, 9] to avoid domain issues
        tensor = Nx.add(Nx.iota({n}, type: :f32), 1.0)

        result =
          tensor
          |> Nx.log()
          |> Nx.exp()
          |> Nx.tanh()
          |> Nx.abs()
          |> Nx.sqrt()

        check_invariants(result, "math chain")

        for val <- Nx.to_flat_list(result) do
          refute val == :nan, "NaN appeared in math chain"
          refute val == :infinity, "Inf appeared in math chain"
        end
      end
    end

    property "slice -> pad -> slice roundtrip preserves size" do
      check all(
              n <- integer(4..10),
              max_runs: 20
            ) do
        tensor = Nx.iota({n}, type: :f32)

        # Slice middle portion
        start = div(n, 4)
        len = div(n, 2)
        sliced = Nx.slice(tensor, [start], [len])
        check_invariants(sliced, "after slice")

        # Pad back to original size
        pad_before = start
        pad_after = n - start - len
        padded = Nx.pad(sliced, Nx.tensor(0.0), [{pad_before, pad_after, 0}])
        check_invariants(padded, "after pad")
        assert Nx.shape(padded) == {n}
      end
    end

    property "concatenate then slice recovers parts" do
      check all(
              a_len <- integer(1..5),
              b_len <- integer(1..5),
              max_runs: 20
            ) do
        a = Nx.iota({a_len}, type: :f32)
        b = Nx.add(Nx.iota({b_len}, type: :f32), 100.0)

        combined = Nx.concatenate([a, b])
        check_invariants(combined, "after concat")
        assert Nx.shape(combined) == {a_len + b_len}

        recovered_a = Nx.slice(combined, [0], [a_len])
        recovered_b = Nx.slice(combined, [a_len], [b_len])

        assert Nx.to_flat_list(recovered_a) == Nx.to_flat_list(a)
        assert Nx.to_flat_list(recovered_b) == Nx.to_flat_list(b)
      end
    end
  end

  # ── Vectorized sequence tests ──────────────────────────────────────

  describe "vectorized op sequences" do
    property "vectorize -> ops -> devectorize preserves element count" do
      check all(
              batch <- integer(2..4),
              n <- integer(2..6),
              max_runs: 20
            ) do
        tensor = Nx.iota({batch, n}, type: :f32)
        vec = Nx.vectorize(tensor, :batch)

        # Apply some ops
        result =
          vec
          |> Nx.multiply(2)
          |> Nx.add(1)
          |> Nx.abs()

        assert result.vectorized_axes == [batch: batch]
        assert result.shape == {n}

        devec = Nx.devectorize(result)
        check_invariants(devec, "after devectorize")
        assert Nx.shape(devec) == {batch, n}
      end
    end

    property "vectorize -> reshape inner -> devectorize" do
      check all(
              batch <- integer(2..3),
              m <- integer(2..4),
              n <- integer(2..4),
              max_runs: 20
            ) do
        tensor = Nx.iota({batch, m * n}, type: :f32)
        vec = Nx.vectorize(tensor, :batch)

        reshaped = Nx.reshape(vec, {m, n})
        assert reshaped.vectorized_axes == [batch: batch]
        assert reshaped.shape == {m, n}

        devec = Nx.devectorize(reshaped)
        assert Nx.shape(devec) == {batch, m, n}
        assert Nx.size(devec) == batch * m * n
      end
    end

    property "vectorize -> reduce inner -> devectorize" do
      check all(
              batch <- integer(2..4),
              n <- integer(2..6),
              max_runs: 20
            ) do
        tensor = Nx.iota({batch, n}, type: :f32)
        vec = Nx.vectorize(tensor, :batch)

        reduced = Nx.sum(vec)
        assert reduced.vectorized_axes == [batch: batch]
        assert reduced.shape == {}

        devec = Nx.devectorize(reduced)
        assert Nx.shape(devec) == {batch}

        # Each batch row i has values [i*n, i*n+1, ..., i*n+n-1]
        # sum = n*(i*n) + n*(n-1)/2 = i*n² + n*(n-1)/2
        for {val, i} <- Enum.with_index(Nx.to_flat_list(devec)) do
          expected = i * n * n + n * (n - 1) / 2
          assert_in_delta val, expected, 1.0e-3
        end
      end
    end

    property "double vectorize -> ops -> double devectorize" do
      check all(
              b1 <- integer(2..3),
              b2 <- integer(2..3),
              n <- integer(2..4),
              max_runs: 15
            ) do
        tensor = Nx.iota({b1, b2, n}, type: :f32)
        vec = tensor |> Nx.vectorize(:outer) |> Nx.vectorize(:inner)

        assert vec.vectorized_axes == [outer: b1, inner: b2]
        assert vec.shape == {n}

        result = Nx.add(vec, 10)
        assert result.vectorized_axes == [outer: b1, inner: b2]

        devec = result |> Nx.devectorize()
        assert Nx.shape(devec) == {b1, b2, n}
      end
    end
  end

  # ── Backend consistency sequences ──────────────────────────────────

  describe "backend consistency" do
    property "backend_transfer roundtrip preserves values" do
      check all(
              shape <- initial_shape(),
              type <- initial_type(),
              max_runs: 20
            ) do
        tensor = Nx.iota(shape, type: type)

        # Transfer to binary backend explicitly and back
        transferred = Nx.backend_transfer(tensor, Nx.BinaryBackend)
        assert Nx.shape(transferred) == shape
        assert Nx.type(transferred) == Nx.Type.normalize!(type)

        assert Nx.to_flat_list(tensor) == Nx.to_flat_list(transferred)
      end
    end

    property "backend_copy preserves values" do
      check all(
              shape <- initial_shape(),
              type <- initial_type(),
              max_runs: 20
            ) do
        tensor = Nx.iota(shape, type: type)

        copied = Nx.backend_copy(tensor, Nx.BinaryBackend)
        assert Nx.to_flat_list(tensor) == Nx.to_flat_list(copied)
      end
    end

    property "ops after backend_transfer produce same results" do
      check all(
              n <- integer(2..8),
              max_runs: 20
            ) do
        tensor = Nx.iota({n}, type: :f32)

        result_before = tensor |> Nx.multiply(2) |> Nx.add(1) |> Nx.to_flat_list()

        transferred = Nx.backend_transfer(tensor, Nx.BinaryBackend)
        result_after = transferred |> Nx.multiply(2) |> Nx.add(1) |> Nx.to_flat_list()

        assert result_before == result_after
      end
    end
  end

  # ── Idempotency and involution sequences ───────────────────────────

  describe "idempotency and involution sequences" do
    property "sort is idempotent: sort(sort(x)) == sort(x)" do
      check all(n <- integer(1..10), max_runs: 20) do
        t = Nx.subtract(Nx.tensor(n), Nx.iota({n}))
        once = Nx.sort(t)
        twice = Nx.sort(once)
        assert Nx.to_flat_list(once) == Nx.to_flat_list(twice)
      end
    end

    property "abs is idempotent: abs(abs(x)) == abs(x)" do
      check all(n <- integer(1..10), max_runs: 20) do
        t = Nx.subtract(Nx.iota({n}, type: :f32), Nx.tensor(n / 2.0))
        once = Nx.abs(t)
        twice = Nx.abs(once)
        assert Nx.to_flat_list(once) == Nx.to_flat_list(twice)
      end
    end

    property "negate is involution: negate(negate(x)) == x" do
      check all(n <- integer(1..10), max_runs: 20) do
        t = Nx.iota({n}, type: :f32)
        assert Nx.to_flat_list(t |> Nx.negate() |> Nx.negate()) == Nx.to_flat_list(t)
      end
    end

    property "transpose is involution on 2D: transpose(transpose(x)) == x" do
      check all(
              m <- integer(1..5),
              n <- integer(1..5),
              max_runs: 20
            ) do
        t = Nx.iota({m, n}, type: :f32)
        assert Nx.to_flat_list(t |> Nx.transpose() |> Nx.transpose()) == Nx.to_flat_list(t)
      end
    end

    property "reverse is involution: reverse(reverse(x)) == x" do
      check all(n <- integer(1..10), max_runs: 20) do
        t = Nx.iota({n}, type: :f32)
        assert Nx.to_flat_list(t |> Nx.reverse() |> Nx.reverse()) == Nx.to_flat_list(t)
      end
    end

    property "flatten then reshape recovers original" do
      check all(
              m <- integer(1..5),
              n <- integer(1..5),
              max_runs: 20
            ) do
        t = Nx.iota({m, n})
        assert Nx.to_flat_list(t |> Nx.flatten() |> Nx.reshape({m, n})) == Nx.to_flat_list(t)
      end
    end

    property "new_axis then squeeze is identity" do
      check all(n <- integer(1..8), max_runs: 20) do
        t = Nx.iota({n}, type: :f32)
        result = t |> Nx.new_axis(0) |> Nx.squeeze(axes: [0])
        assert Nx.to_flat_list(result) == Nx.to_flat_list(t)
        assert Nx.shape(result) == Nx.shape(t)
      end
    end
  end

  # ── Stress test: long random chains ────────────────────────────────

  describe "stress: long random chains" do
    property "20-op random chain doesn't crash" do
      check all(
              shape <- initial_shape(),
              type <- member_of([:f32, :f64]),
              ops <- op_sequence(15, 20),
              max_runs: 15
            ) do
        tensor = Nx.iota(shape, type: type)
        {final, steps} = run_sequence(tensor, ops)
        assert is_struct(final, Nx.Tensor)
        assert steps >= 15
      end
    end
  end

  # ── Binary op sequences (task #19) ─────────────────────────────────
  # Two tensors interacting through binary ops

  describe "binary op sequences" do
    property "add/multiply chain with two tensors" do
      check all(
              n <- integer(2..8),
              ops <- list_of(member_of([:add, :subtract, :multiply, :min, :max]), length: 5),
              max_runs: 30
            ) do
        a = Nx.iota({n}, type: :f32)
        b = Nx.add(Nx.iota({n}, type: :f32), 1.0)

        result =
          Enum.reduce(ops, a, fn op, acc ->
            apply(Nx, op, [acc, b])
          end)

        check_invariants(result, "binary chain")
        assert Nx.shape(result) == {n}
      end
    end

    property "binary ops with broadcasting: {n,m} and scalar" do
      check all(
              n <- integer(2..5),
              m <- integer(2..5),
              ops <- list_of(member_of([:add, :subtract, :multiply]), length: 4),
              max_runs: 20
            ) do
        a = Nx.iota({n, m}, type: :f32)

        result =
          Enum.reduce(ops, a, fn op, acc ->
            # Use row vector broadcast: {1,m} broadcasts with {n,m}
            b = Nx.iota({1, elem(Nx.shape(acc), tuple_size(Nx.shape(acc)) - 1)}, type: :f32)
            apply(Nx, op, [acc, b])
          end)

        check_invariants(result, "broadcast binary chain")
      end
    end

    property "binary ops with scalar" do
      check all(
              shape <- initial_shape(),
              scalars <- list_of(float(min: -10.0, max: 10.0), length: 4),
              ops <- list_of(member_of([:add, :subtract, :multiply]), length: 4),
              max_runs: 20
            ) do
        tensor = Nx.iota(shape, type: :f32)

        result =
          Enum.zip(ops, scalars)
          |> Enum.reduce(tensor, fn {op, scalar}, acc ->
            apply(Nx, op, [acc, scalar])
          end)

        check_invariants(result, "scalar binary chain")
        assert Nx.shape(result) == shape
      end
    end

    property "dot sequence: matmul chain" do
      check all(
              dims <- list_of(integer(2..5), length: 4),
              max_runs: 15
            ) do
        # Create a chain of matrices that can be multiplied
        [d0, d1, d2, d3] = dims
        a = Nx.iota({d0, d1}, type: :f32)
        b = Nx.iota({d1, d2}, type: :f32)
        c = Nx.iota({d2, d3}, type: :f32)

        ab = Nx.dot(a, b)
        check_invariants(ab, "dot step 1")
        assert Nx.shape(ab) == {d0, d2}

        abc = Nx.dot(ab, c)
        check_invariants(abc, "dot step 2")
        assert Nx.shape(abc) == {d0, d3}
      end
    end

    property "outer product then reshape" do
      check all(
              m <- integer(1..5),
              n <- integer(1..5),
              max_runs: 20
            ) do
        a = Nx.iota({m}, type: :f32)
        b = Nx.iota({n}, type: :f32)

        result = Nx.outer(a, b)
        check_invariants(result, "outer")
        assert Nx.shape(result) == {m, n}

        flat = Nx.flatten(result)
        check_invariants(flat, "outer then flatten")
        assert Nx.shape(flat) == {m * n}
      end
    end
  end

  # ── JIT vs eager comparison (task #22) ─────────────────────────────

  describe "JIT vs eager comparison" do
    import Nx.Defn

    defn jit_chain_1(t) do
      t
      |> Nx.multiply(2)
      |> Nx.add(1)
      |> Nx.abs()
      |> Nx.tanh()
    end

    defn jit_chain_2(t) do
      t
      |> Nx.sin()
      |> Nx.cos()
      |> Nx.negate()
      |> Nx.exp()
    end

    defn jit_chain_3(t) do
      t
      |> Nx.as_type(:f32)
      |> Nx.sqrt()
      |> Nx.multiply(3)
      |> Nx.subtract(1)
      |> Nx.sigmoid()
    end

    defn jit_reduce_chain(t) do
      t
      |> Nx.multiply(2)
      |> Nx.add(1)
      |> Nx.sum()
    end

    property "jit_chain_1 matches eager" do
      check all(n <- integer(1..8), max_runs: 20) do
        t = Nx.iota({n}, type: :f32)
        eager = jit_chain_1(t)
        jitted = Nx.Defn.jit(&jit_chain_1/1).(t)
        assert Nx.to_flat_list(eager) == Nx.to_flat_list(jitted)
      end
    end

    property "jit_chain_2 matches eager" do
      check all(n <- integer(1..8), max_runs: 20) do
        t = Nx.iota({n}, type: :f32)
        eager = jit_chain_2(t)
        jitted = Nx.Defn.jit(&jit_chain_2/1).(t)

        for {e, j} <- Enum.zip(Nx.to_flat_list(eager), Nx.to_flat_list(jitted)) do
          assert_in_delta e, j, 1.0e-6
        end
      end
    end

    property "jit_chain_3 matches eager for positive inputs" do
      check all(n <- integer(1..8), max_runs: 20) do
        t = Nx.add(Nx.iota({n}, type: :f32), 1.0)
        eager = jit_chain_3(t)
        jitted = Nx.Defn.jit(&jit_chain_3/1).(t)

        for {e, j} <- Enum.zip(Nx.to_flat_list(eager), Nx.to_flat_list(jitted)) do
          assert_in_delta e, j, 1.0e-6
        end
      end
    end

    property "jit reduction matches eager" do
      check all(
              m <- integer(1..5),
              n <- integer(1..5),
              max_runs: 20
            ) do
        t = Nx.iota({m, n}, type: :f32)
        eager = Nx.to_number(jit_reduce_chain(t))
        jitted = Nx.to_number(Nx.Defn.jit(&jit_reduce_chain/1).(t))
        assert_in_delta eager, jitted, 1.0e-4
      end
    end
  end

  # ── Resource monitoring (task #24) ──────────────────────────────────

  describe "resource monitoring" do
    test "50-op sequence doesn't leak processes" do
      initial_processes = length(Process.list())

      # Run several long sequences
      for _ <- 1..5 do
        tensor = Nx.iota({4, 3}, type: :f32)
        ops = for _ <- 1..50, do: Enum.random(@op_names)

        {_final, _steps} = run_sequence(tensor, ops)
      end

      :erlang.garbage_collect()
      Process.sleep(50)

      final_processes = length(Process.list())
      # Allow some variance (up to 10 processes) but no unbounded growth
      assert final_processes - initial_processes < 10,
             "Process count grew from #{initial_processes} to #{final_processes}"
    end

    test "100-op sequence doesn't crash" do
      tensor = Nx.iota({3, 4}, type: :f32)
      ops = for _ <- 1..100, do: Enum.random(@op_names)

      {final, steps} = run_sequence(tensor, ops)
      assert is_struct(final, Nx.Tensor)
      assert steps == 100
    end

    test "repeated backend_transfer doesn't leak" do
      initial_processes = length(Process.list())

      tensor = Nx.iota({10, 10}, type: :f32)

      for _ <- 1..100 do
        tensor
        |> Nx.backend_transfer(Nx.BinaryBackend)
        |> Nx.multiply(2)
        |> Nx.backend_transfer(Nx.BinaryBackend)
      end

      :erlang.garbage_collect()
      Process.sleep(50)

      final_processes = length(Process.list())

      assert final_processes - initial_processes < 10,
             "Process leak: #{initial_processes} -> #{final_processes}"
    end
  end
end
