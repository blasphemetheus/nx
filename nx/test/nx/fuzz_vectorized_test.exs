defmodule Nx.FuzzVectorizedTest do
  @moduledoc """
  Fuzz tests for operations on vectorized tensors.

  Verifies that vectorized results match per-element computation
  and that ops preserve vectorized axes correctly.
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  # ── Generators ─────────────────────────────────────────────────────

  defp vectorized_tensor do
    bind(integer(1..6), fn batch ->
      bind(member_of([{}, {3}, {2, 3}, {4}]), fn inner_shape ->
        bind(member_of([:f32, :f64]), fn type ->
          full_shape = Tuple.insert_at(inner_shape, 0, batch)
          constant(Nx.iota(full_shape, type: type) |> Nx.vectorize(:batch))
        end)
      end)
    end)
  end

  defp vectorized_tensor_with_shape(batch, inner_shape, type) do
    full_shape = Tuple.insert_at(inner_shape, 0, batch)
    Nx.iota(full_shape, type: type) |> Nx.vectorize(:batch)
  end

  # ── Unary ops preserve vectorization ──────────────────────────────

  @unary_ops [:abs, :negate, :sign, :floor, :ceil, :round, :sigmoid]

  describe "unary ops preserve vectorized axes" do
    for op <- @unary_ops do
      property "#{op} preserves vectorization" do
        check all(t <- vectorized_tensor(), max_runs: 20) do
          result = apply(Nx, unquote(op), [t])
          assert result.vectorized_axes == t.vectorized_axes
          assert Nx.shape(result) == Nx.shape(t)
        end
      end
    end
  end

  # ── Vectorized matches per-element ────────────────────────────────

  describe "vectorized results match per-element" do
    property "sum matches per-batch sum" do
      check all(
              batch <- integer(1..6),
              cols <- integer(1..8),
              type <- member_of([:f32, :f64]),
              max_runs: 15
            ) do
        t = vectorized_tensor_with_shape(batch, {cols}, type)

        vec_result = Nx.sum(t)
        assert vec_result.vectorized_axes == [batch: batch]

        # Compare against per-element computation
        devec = Nx.devectorize(t, keep_names: false)

        devec_result = Nx.devectorize(vec_result, keep_names: false)

        for i <- 0..(batch - 1) do
          row = Nx.slice_along_axis(devec, i, 1, axis: 0) |> Nx.squeeze(axes: [0])
          expected = Nx.sum(row) |> Nx.to_number()
          actual = devec_result[i] |> Nx.to_number()
          assert_in_delta actual, expected, 1.0e-4
        end
      end
    end

    property "multiply matches per-batch multiply" do
      check all(
              batch <- integer(1..4),
              cols <- integer(1..6),
              type <- member_of([:f32]),
              max_runs: 10
            ) do
        a = vectorized_tensor_with_shape(batch, {cols}, type)
        b = vectorized_tensor_with_shape(batch, {cols}, type)

        vec_result = Nx.multiply(a, b)
        assert vec_result.vectorized_axes == [batch: batch]

        devec_a = Nx.devectorize(a, keep_names: false)
        devec_b = Nx.devectorize(b, keep_names: false)
        devec_result = Nx.devectorize(vec_result, keep_names: false)

        for i <- 0..(batch - 1) do
          row_a = Nx.slice_along_axis(devec_a, i, 1, axis: 0) |> Nx.squeeze(axes: [0])
          row_b = Nx.slice_along_axis(devec_b, i, 1, axis: 0) |> Nx.squeeze(axes: [0])
          expected = Nx.multiply(row_a, row_b)
          actual = Nx.slice_along_axis(devec_result, i, 1, axis: 0) |> Nx.squeeze(axes: [0])

          diff =
            Nx.subtract(actual, expected) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

          assert diff < 1.0e-4
        end
      end
    end
  end

  # ── Binary ops with same vectorized axes ──────────────────────────

  describe "binary ops with same vectorized axes" do
    for op <- [:add, :subtract, :multiply, :min, :max] do
      property "#{op} with same axes preserves vectorization" do
        check all(
                batch <- integer(1..6),
                cols <- integer(1..6),
                type <- member_of([:f32, :f64]),
                max_runs: 15
              ) do
          a = vectorized_tensor_with_shape(batch, {cols}, type)
          b = vectorized_tensor_with_shape(batch, {cols}, type)
          result = apply(Nx, unquote(op), [a, b])
          assert result.vectorized_axes == [batch: batch]
          assert Nx.shape(result) == {cols}
        end
      end
    end
  end

  # ── Reductions preserve vectorization ─────────────────────────────

  describe "reductions preserve vectorization" do
    for op <- [:sum, :product, :reduce_max, :reduce_min, :mean] do
      property "#{op} reduces inner dims, keeps batch" do
        check all(
                batch <- integer(1..6),
                cols <- integer(1..8),
                type <- member_of([:f32]),
                max_runs: 15
              ) do
          t = vectorized_tensor_with_shape(batch, {cols}, type)
          result = apply(Nx, unquote(op), [t])
          assert result.vectorized_axes == [batch: batch]
          assert Nx.shape(result) == {}
        end
      end
    end

    property "sum with axis preserves batch and reduces correct axis" do
      check all(
              batch <- integer(1..4),
              rows <- integer(1..4),
              cols <- integer(1..4),
              type <- member_of([:f32]),
              max_runs: 10
            ) do
        t = vectorized_tensor_with_shape(batch, {rows, cols}, type)
        result = Nx.sum(t, axes: [0])
        assert result.vectorized_axes == [batch: batch]
        assert Nx.shape(result) == {cols}
      end
    end
  end

  # ── Shape ops preserve vectorization ──────────────────────────────

  describe "shape ops preserve vectorization" do
    property "reshape preserves batch" do
      check all(
              batch <- integer(1..4),
              type <- member_of([:f32]),
              max_runs: 15
            ) do
        t = vectorized_tensor_with_shape(batch, {2, 3}, type)
        result = Nx.reshape(t, {6})
        assert result.vectorized_axes == [batch: batch]
        assert Nx.shape(result) == {6}
      end
    end

    property "transpose preserves batch" do
      check all(
              batch <- integer(1..4),
              rows <- integer(1..4),
              cols <- integer(1..4),
              type <- member_of([:f32]),
              max_runs: 10
            ) do
        t = vectorized_tensor_with_shape(batch, {rows, cols}, type)
        result = Nx.transpose(t)
        assert result.vectorized_axes == [batch: batch]
        assert Nx.shape(result) == {cols, rows}
      end
    end

    property "squeeze preserves batch" do
      check all(
              batch <- integer(1..4),
              n <- integer(1..6),
              type <- member_of([:f32]),
              max_runs: 15
            ) do
        t = vectorized_tensor_with_shape(batch, {1, n}, type)
        result = Nx.squeeze(t, axes: [0])
        assert result.vectorized_axes == [batch: batch]
        assert Nx.shape(result) == {n}
      end
    end
  end

  # ── Vectorized + scalar broadcasting ──────────────────────────────

  describe "vectorized with scalar" do
    property "add scalar to vectorized" do
      check all(
              batch <- integer(1..6),
              cols <- integer(1..8),
              max_runs: 15
            ) do
        t = vectorized_tensor_with_shape(batch, {cols}, :f32)
        result = Nx.add(t, 1.0)
        assert result.vectorized_axes == [batch: batch]
        assert Nx.shape(result) == {cols}
      end
    end

    property "multiply vectorized by scalar" do
      check all(
              batch <- integer(1..6),
              cols <- integer(1..8),
              max_runs: 15
            ) do
        t = vectorized_tensor_with_shape(batch, {cols}, :f32)
        result = Nx.multiply(t, 2.0)
        assert result.vectorized_axes == [batch: batch]
      end
    end
  end

  # ── Comparison ops on vectorized ──────────────────────────────────

  describe "comparison ops on vectorized" do
    for op <- [:equal, :greater, :less] do
      property "#{op} preserves vectorization" do
        check all(
                batch <- integer(1..4),
                cols <- integer(1..6),
                max_runs: 10
              ) do
          a = vectorized_tensor_with_shape(batch, {cols}, :f32)
          b = vectorized_tensor_with_shape(batch, {cols}, :f32)
          result = apply(Nx, unquote(op), [a, b])
          assert result.vectorized_axes == [batch: batch]
          assert Nx.type(result) == {:u, 8}
        end
      end
    end
  end

  # ── Devectorize / revectorize ─────────────────────────────────────

  describe "devectorize/vectorize roundtrip" do
    property "devectorize then vectorize is identity" do
      check all(
              batch <- integer(1..6),
              cols <- integer(1..8),
              type <- member_of([:f32, :f64]),
              max_runs: 15
            ) do
        t = vectorized_tensor_with_shape(batch, {cols}, type)
        devec = Nx.devectorize(t, keep_names: false)
        revec = Nx.vectorize(devec, :batch)

        assert revec.vectorized_axes == [batch: batch]

        diff =
          Nx.subtract(Nx.devectorize(revec), Nx.devectorize(t))
          |> Nx.abs()
          |> Nx.reduce_max()
          |> Nx.to_number()

        assert diff < 1.0e-5
      end
    end
  end
end
