defmodule Nx.FuzzBroadcastTest do
  @moduledoc """
  Fuzz tests for broadcasting with mismatched shapes.

  Tests binary ops with broadcastable but different shapes:
  {3,1}+{1,4}, scalar+matrix, etc.
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  @fuzz_scale String.to_integer(System.get_env("FUZZ_SCALE", "1"))

  # ── Generators ─────────────────────────────────────────────────────

  # Generate a pair of shapes that are broadcastable with each other
  defp broadcastable_pair do
    frequency([
      # scalar + anything
      {3,
       bind(
         member_of([{}, {3}, {2, 3}, {4, 1, 3}]),
         &constant({{}, &1})
       )},
      # 1D + 1D same
      {2,
       bind(integer(1..8), fn n ->
         constant({{n}, {n}})
       end)},
      # {n, 1} + {1, m}
      {3,
       bind(integer(1..6), fn n ->
         map(integer(1..6), fn m ->
           {{n, 1}, {1, m}}
         end)
       end)},
      # {n, m} + {m}
      {3,
       bind(integer(1..6), fn n ->
         map(integer(1..6), fn m ->
           {{n, m}, {m}}
         end)
       end)},
      # {n, m} + {1, m}
      {2,
       bind(integer(1..6), fn n ->
         map(integer(1..6), fn m ->
           {{n, m}, {1, m}}
         end)
       end)},
      # {a, b, c} + {c}
      {2,
       bind(integer(1..4), fn a ->
         bind(integer(1..4), fn b ->
           map(integer(1..4), fn c ->
             {{a, b, c}, {c}}
           end)
         end)
       end)},
      # {a, 1, c} + {1, b, 1}
      {2,
       bind(integer(1..4), fn a ->
         bind(integer(1..4), fn b ->
           map(integer(1..4), fn c ->
             {{a, 1, c}, {1, b, 1}}
           end)
         end)
       end)}
    ])
  end

  defp expected_broadcast_shape({}, shape), do: shape
  defp expected_broadcast_shape(shape, {}), do: shape

  defp expected_broadcast_shape(s1, s2) do
    l1 = Tuple.to_list(s1) |> Enum.reverse()
    l2 = Tuple.to_list(s2) |> Enum.reverse()
    max_len = max(length(l1), length(l2))
    l1 = List.duplicate(1, max_len - length(l1)) ++ Enum.reverse(l1)
    l2 = List.duplicate(1, max_len - length(l2)) ++ Enum.reverse(l2)

    Enum.zip(l1, l2)
    |> Enum.map(fn {a, b} -> max(a, b) end)
    |> List.to_tuple()
  end

  defp make_tensor(shape, type) do
    if shape == {} do
      Nx.tensor(1, type: type)
    else
      Nx.iota(shape, type: type)
    end
  end

  # ── Binary ops with broadcasting ──────────────────────────────────

  @broadcast_ops [:add, :subtract, :multiply, :min, :max]

  describe "binary ops with broadcast shapes" do
    for op <- @broadcast_ops do
      property "#{op} broadcasts correctly" do
        check all(
                {s1, s2} <- broadcastable_pair(),
                type <- member_of([:f32, :f64]),
                max_runs: 30 * @fuzz_scale
              ) do
          a = make_tensor(s1, type)
          b = make_tensor(s2, type)
          result = apply(Nx, unquote(op), [a, b])
          expected_shape = expected_broadcast_shape(s1, s2)
          assert Nx.shape(result) == expected_shape
        end
      end
    end

    property "divide broadcasts correctly" do
      check all(
              {s1, s2} <- broadcastable_pair(),
              type <- member_of([:f32, :f64]),
              max_runs: 30 * @fuzz_scale
            ) do
        a = make_tensor(s1, type)
        b = Nx.add(make_tensor(s2, type), 1)
        result = Nx.divide(a, b)
        expected_shape = expected_broadcast_shape(s1, s2)
        assert Nx.shape(result) == expected_shape
      end
    end

    property "pow broadcasts correctly" do
      check all(
              {s1, s2} <- broadcastable_pair(),
              max_runs: 20 * @fuzz_scale
            ) do
        a = Nx.add(make_tensor(s1, :f32), 1)
        b = make_tensor(s2, :f32)
        result = Nx.pow(a, b)
        expected_shape = expected_broadcast_shape(s1, s2)
        assert Nx.shape(result) == expected_shape
      end
    end

    property "atan2 broadcasts correctly" do
      check all(
              {s1, s2} <- broadcastable_pair(),
              max_runs: 20 * @fuzz_scale
            ) do
        a = make_tensor(s1, :f32)
        b = Nx.add(make_tensor(s2, :f32), 1)
        result = Nx.atan2(a, b)
        expected_shape = expected_broadcast_shape(s1, s2)
        assert Nx.shape(result) == expected_shape
      end
    end

    property "remainder broadcasts correctly" do
      check all(
              {s1, s2} <- broadcastable_pair(),
              max_runs: 20 * @fuzz_scale
            ) do
        a = make_tensor(s1, :f32)
        b = Nx.add(make_tensor(s2, :f32), 1)
        result = Nx.remainder(a, b)
        expected_shape = expected_broadcast_shape(s1, s2)
        assert Nx.shape(result) == expected_shape
      end
    end
  end

  # ── Comparison ops with broadcasting ──────────────────────────────

  @comparison_ops [:equal, :not_equal, :greater, :greater_equal, :less, :less_equal]

  describe "comparison ops with broadcast shapes" do
    for op <- @comparison_ops do
      property "#{op} broadcasts correctly" do
        check all(
                {s1, s2} <- broadcastable_pair(),
                type <- member_of([:f32, :s32]),
                max_runs: 20 * @fuzz_scale
              ) do
          a = make_tensor(s1, type)
          b = make_tensor(s2, type)
          result = apply(Nx, unquote(op), [a, b])
          expected_shape = expected_broadcast_shape(s1, s2)
          assert Nx.shape(result) == expected_shape
          assert Nx.type(result) == {:u, 8}
        end
      end
    end
  end

  # ── Broadcasting properties ───────────────────────────────────────

  describe "broadcasting mathematical properties" do
    property "add(x, broadcast(0, x.shape)) == x" do
      check all(
              shape <- member_of([{3}, {2, 3}, {4, 1, 3}]),
              type <- member_of([:f32, :f64]),
              max_runs: 20 * @fuzz_scale
            ) do
        x = Nx.iota(shape, type: type)
        zero = Nx.broadcast(0, shape) |> Nx.as_type(type)
        result = Nx.add(x, zero)

        diff =
          Nx.subtract(result, x) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

        assert diff < 1.0e-5
      end
    end

    property "scalar broadcast then op == op with scalar" do
      check all(
              shape <- member_of([{3}, {2, 3}, {2, 3, 4}]),
              max_runs: 15 * @fuzz_scale
            ) do
        x = Nx.iota(shape, type: :f32)
        # x + 5.0 via scalar
        via_scalar = Nx.add(x, 5.0)
        # x + broadcast(5.0, shape)
        via_broadcast = Nx.add(x, Nx.broadcast(5.0, shape))

        diff =
          Nx.subtract(via_scalar, via_broadcast)
          |> Nx.abs()
          |> Nx.reduce_max()
          |> Nx.to_number()

        assert diff < 1.0e-5
      end
    end

    property "broadcast + reduce_sum along broadcast axis recovers original" do
      check all(
              n <- integer(1..8),
              m <- integer(1..8),
              max_runs: 15 * @fuzz_scale
            ) do
        # Column vector {n, 1} broadcast to {n, m}
        col = Nx.iota({n, 1}, type: :f32)
        broadcasted = Nx.broadcast(col, {n, m})
        # Sum along axis 1 should give col * m
        summed = Nx.sum(broadcasted, axes: [1], keep_axes: true)
        expected = Nx.multiply(col, m)

        diff =
          Nx.subtract(summed, expected)
          |> Nx.abs()
          |> Nx.reduce_max()
          |> Nx.to_number()

        assert diff < 1.0e-3
      end
    end
  end

  # ── Select with broadcast ─────────────────────────────────────────

  describe "select with broadcast" do
    property "select with matching shapes" do
      check all(
              n <- integer(1..8),
              m <- integer(1..8),
              max_runs: 15 * @fuzz_scale
            ) do
        pred = Nx.greater(Nx.iota({n, m}, type: :f32), n * m / 2)
        on_true = Nx.iota({n, m}, type: :f32)
        on_false = Nx.broadcast(0.0, {n, m})

        result = Nx.select(pred, on_true, on_false)
        assert Nx.shape(result) == {n, m}
      end
    end

    property "select with scalar pred" do
      check all(
              n <- integer(1..8),
              max_runs: 15 * @fuzz_scale
            ) do
        pred = Nx.tensor(1, type: :u8)
        on_true = Nx.iota({n}, type: :f32)
        on_false = Nx.broadcast(0.0, {n})

        result = Nx.select(pred, on_true, on_false)
        assert Nx.shape(result) == {n}
      end
    end
  end

  # ── Logical ops with broadcast ────────────────────────────────────

  describe "logical ops with broadcast" do
    for op <- [:logical_and, :logical_or, :logical_xor] do
      property "#{op} broadcasts correctly" do
        check all(
                {s1, s2} <- broadcastable_pair(),
                max_runs: 15 * @fuzz_scale
              ) do
          a = Nx.greater(make_tensor(s1, :f32), 0)
          b = Nx.greater(make_tensor(s2, :f32), 0)
          result = apply(Nx, unquote(op), [a, b])
          expected_shape = expected_broadcast_shape(s1, s2)
          assert Nx.shape(result) == expected_shape
        end
      end
    end
  end
end
