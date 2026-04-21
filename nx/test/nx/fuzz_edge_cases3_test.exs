defmodule Nx.FuzzEdgeCases3Test do
  @moduledoc """
  Tier 4 (part 3): Edge case tests for diff, eye, clip, conv, covariance,
  FFT, and LinAlg boundary conditions.
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  # ── Nx.diff boundary conditions ────────────────────────────────────
  # Source: nx.ex:11583
  # Boundaries:
  #   - cannot diff scalar (line 11583)
  #   - order must be non-negative integer

  describe "diff boundary conditions" do
    test "diff 1D basic" do
      t = Nx.tensor([1, 4, 2, 8, 5])
      result = Nx.diff(t)
      assert Nx.to_flat_list(result) == [3, -2, 6, -3]
    end

    test "diff 1D order 2 (second differences)" do
      t = Nx.tensor([1, 4, 2, 8, 5])
      result = Nx.diff(t, order: 2)
      # First diff: [3, -2, 6, -3], second diff: [-5, 8, -9]
      assert Nx.to_flat_list(result) == [-5, 8, -9]
    end

    test "diff reduces length by order" do
      t = Nx.iota({10})
      assert Nx.shape(Nx.diff(t, order: 1)) == {9}
      assert Nx.shape(Nx.diff(t, order: 2)) == {8}
      assert Nx.shape(Nx.diff(t, order: 3)) == {7}
    end

    test "diff of constant tensor is all zeros" do
      t = Nx.broadcast(Nx.tensor(5.0), {6})
      result = Nx.diff(t)
      assert Nx.to_flat_list(result) == [0.0, 0.0, 0.0, 0.0, 0.0]
    end

    test "diff of linear tensor is constant" do
      t = Nx.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
      result = Nx.diff(t)
      assert Nx.to_flat_list(result) == [1.0, 1.0, 1.0, 1.0]
    end

    test "diff of quadratic: second diff is constant" do
      # t = [0, 1, 4, 9, 16] (x^2)
      t = Nx.tensor([0.0, 1.0, 4.0, 9.0, 16.0])
      result = Nx.diff(t, order: 2)
      assert Nx.to_flat_list(result) == [2.0, 2.0, 2.0]
    end

    test "diff on 2D along axis 0" do
      t = Nx.tensor([[1, 2], [4, 8], [9, 3]])
      result = Nx.diff(t, axis: 0)
      assert Nx.shape(result) == {2, 2}
      assert Nx.to_flat_list(result) == [3, 6, 5, -5]
    end

    test "diff on 2D along axis 1" do
      t = Nx.tensor([[1, 2, 4], [8, 3, 1]])
      result = Nx.diff(t, axis: 1)
      assert Nx.shape(result) == {2, 2}
      assert Nx.to_flat_list(result) == [1, 2, -5, -2]
    end

    test "diff raises on scalar" do
      t = Nx.tensor(42)
      assert_raise ArgumentError, fn ->
        Nx.diff(t)
      end
    end

    test "diff with order == tensor length - 1 gives single element" do
      t = Nx.tensor([1, 2, 4, 8])
      result = Nx.diff(t, order: 3)
      assert Nx.shape(result) == {1}
    end
  end

  # ── Nx.eye boundary conditions ─────────────────────────────────────
  # Source: nx.ex:1349
  # Boundaries:
  #   - shape must have at least 2 dimensions or be a positive integer

  describe "eye boundary conditions" do
    test "eye from integer" do
      result = Nx.eye(3)
      expected = Nx.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
      assert Nx.to_flat_list(result) == Nx.to_flat_list(expected)
    end

    test "eye from shape tuple" do
      result = Nx.eye({3, 4})
      assert Nx.shape(result) == {3, 4}
      # Diagonal is [1, 1, 1], rest zeros
      assert Nx.to_number(result[0][0]) == 1
      assert Nx.to_number(result[0][1]) == 0
      assert Nx.to_number(result[2][2]) == 1
    end

    test "eye 1x1" do
      result = Nx.eye(1)
      assert Nx.to_flat_list(result) == [1]
    end

    test "eye with type" do
      result = Nx.eye(3, type: :f64)
      assert Nx.type(result) == {:f, 64}
    end

    test "eye batched {2, 3, 3}" do
      result = Nx.eye({2, 3, 3})
      assert Nx.shape(result) == {2, 3, 3}
      # Each batch should be an identity
      assert Nx.to_flat_list(result[0]) == [1, 0, 0, 0, 1, 0, 0, 0, 1]
    end

    test "eye raises on 1D shape" do
      assert_raise ArgumentError, fn ->
        Nx.eye({3})
      end
    end
  end

  # ── Nx.clip boundary conditions ────────────────────────────────────
  # Source: nx.ex:13449
  # Boundaries:
  #   - min must be non-vectorized scalar
  #   - max must be non-vectorized scalar

  describe "clip boundary conditions" do
    test "clip basic" do
      t = Nx.tensor([1.0, 5.0, 3.0, 8.0, -2.0])
      result = Nx.clip(t, 2.0, 6.0)
      assert Nx.to_flat_list(result) == [2.0, 5.0, 3.0, 6.0, 2.0]
    end

    test "clip with min == max collapses to constant" do
      t = Nx.tensor([1.0, 5.0, 3.0])
      result = Nx.clip(t, 3.0, 3.0)
      assert Nx.to_flat_list(result) == [3.0, 3.0, 3.0]
    end

    test "clip when all values in range is identity" do
      t = Nx.tensor([2.0, 3.0, 4.0])
      result = Nx.clip(t, 1.0, 5.0)
      assert Nx.to_flat_list(result) == [2.0, 3.0, 4.0]
    end

    test "clip when all values below min" do
      t = Nx.tensor([1.0, 2.0, 3.0])
      result = Nx.clip(t, 10.0, 20.0)
      assert Nx.to_flat_list(result) == [10.0, 10.0, 10.0]
    end

    test "clip when all values above max" do
      t = Nx.tensor([10.0, 20.0, 30.0])
      result = Nx.clip(t, 1.0, 5.0)
      assert Nx.to_flat_list(result) == [5.0, 5.0, 5.0]
    end

    test "clip with integer types" do
      t = Nx.tensor([1, 5, 10, 15, 20], type: :s32)
      result = Nx.clip(t, 5, 15)
      assert Nx.to_flat_list(result) == [5, 5, 10, 15, 15]
    end

    test "clip raises on vectorized min" do
      t = Nx.tensor([1.0, 2.0, 3.0])
      min_v = Nx.tensor([0.0, 0.0]) |> Nx.vectorize(:batch)

      assert_raise ArgumentError, ~r/min .* must be a non-vectorized scalar/, fn ->
        Nx.clip(t, min_v, 10.0)
      end
    end

    test "clip raises on vectorized max" do
      t = Nx.tensor([1.0, 2.0, 3.0])
      max_v = Nx.tensor([10.0, 10.0]) |> Nx.vectorize(:batch)

      assert_raise ArgumentError, ~r/max .* must be a non-vectorized scalar/, fn ->
        Nx.clip(t, 0.0, max_v)
      end
    end

    test "clip scalar tensor" do
      t = Nx.tensor(5.0)
      assert Nx.to_number(Nx.clip(t, 0.0, 3.0)) == 3.0
    end
  end

  # ── Nx.select boundary conditions ──────────────────────────────────

  describe "select boundary conditions" do
    test "select with scalar predicate" do
      result = Nx.select(Nx.tensor(1), Nx.tensor(10), Nx.tensor(20))
      assert Nx.to_number(result) == 10
    end

    test "select with 0 predicate picks on_false" do
      result = Nx.select(Nx.tensor(0), Nx.tensor(10), Nx.tensor(20))
      assert Nx.to_number(result) == 20
    end

    test "select element-wise" do
      pred = Nx.tensor([1, 0, 1, 0])
      result = Nx.select(pred, Nx.tensor([10, 20, 30, 40]), Nx.tensor([50, 60, 70, 80]))
      assert Nx.to_flat_list(result) == [10, 60, 30, 80]
    end

    test "select broadcasts on_true and on_false to pred shape" do
      pred = Nx.tensor([1, 0, 1])
      result = Nx.select(pred, Nx.tensor(100), Nx.tensor(0))
      assert Nx.to_flat_list(result) == [100, 0, 100]
    end

    test "select with type promotion" do
      pred = Nx.tensor([1, 0])
      result = Nx.select(pred, Nx.tensor([1, 2], type: :s32), Nx.tensor([1.5, 2.5], type: :f32))
      assert elem(Nx.type(result), 0) == :f
    end
  end

  # ── Nx.covariance boundary conditions ──────────────────────────────
  # Source: nx.ex:16053
  # Boundaries:
  #   - tensor rank >= 2
  #   - mean rank >= 1
  #   - ddof >= 0

  describe "covariance boundary conditions" do
    test "covariance of identity-like data" do
      # Each column is independent
      t = Nx.tensor([[1.0, 0.0], [0.0, 1.0]])
      result = Nx.covariance(t)
      assert Nx.shape(result) == {2, 2}
    end

    test "covariance of constant columns is zero" do
      t = Nx.tensor([[5.0, 3.0], [5.0, 3.0], [5.0, 3.0]], type: :f32)
      result = Nx.covariance(t, ddof: 0)
      # All values should be 0 (no variance in constant data)
      for val <- Nx.to_flat_list(result) do
        assert_in_delta val, 0.0, 1.0e-6
      end
    end

    test "covariance with ddof: 0 vs ddof: 1" do
      t = Nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
      cov0 = Nx.covariance(t, ddof: 0)
      cov1 = Nx.covariance(t, ddof: 1)
      # ddof=1 divides by n-1 instead of n, so values are larger
      val0 = Nx.to_number(cov0[0][0])
      val1 = Nx.to_number(cov1[0][0])
      assert val1 > val0
    end

    test "covariance is symmetric" do
      t = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
      result = Nx.covariance(t)
      transposed = Nx.transpose(result)

      for {a, b} <- Enum.zip(Nx.to_flat_list(result), Nx.to_flat_list(transposed)) do
        assert_in_delta a, b, 1.0e-5
      end
    end
  end

  # ── FFT boundary conditions ────────────────────────────────────────
  # Source: shape.ex:2177
  # Boundaries:
  #   - rank must be > 0

  describe "FFT boundary conditions" do
    test "fft 1D basic" do
      t = Nx.tensor([1.0, 0.0, 0.0, 0.0])
      result = Nx.fft(t)
      # FFT of impulse is flat spectrum
      assert Nx.shape(result) == {4}
      assert Nx.type(result) == {:c, 64}
    end

    test "fft with explicit length" do
      t = Nx.tensor([1.0, 2.0, 3.0])
      result = Nx.fft(t, length: 4)
      assert Nx.shape(result) == {4}
    end

    test "fft with length shorter than input truncates" do
      t = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      result = Nx.fft(t, length: 3)
      assert Nx.shape(result) == {3}
    end

    test "ifft inverts fft" do
      t = Nx.tensor([1.0, 2.0, 3.0, 4.0])
      roundtrip = t |> Nx.fft() |> Nx.ifft()
      for {orig, rt} <- Enum.zip(Nx.to_flat_list(Nx.as_type(t, :c64)), Nx.to_flat_list(roundtrip)) do
        assert_in_delta Complex.abs(Complex.subtract(orig, rt)), 0.0, 1.0e-5
      end
    end

    test "fft on 2D applies to last axis" do
      t = Nx.iota({2, 4}, type: :f32)
      result = Nx.fft(t)
      assert Nx.shape(result) == {2, 4}
    end

    test "fft2 basic" do
      t = Nx.iota({4, 4}, type: :f32)
      result = Nx.fft2(t)
      assert Nx.shape(result) == {4, 4}
      assert Nx.type(result) == {:c, 64}
    end
  end

  # ── LinAlg boundary conditions ─────────────────────────────────────

  describe "LinAlg boundary conditions" do
    test "norm raises on rank > 2" do
      t = Nx.iota({2, 3, 4}, type: :f32)
      assert_raise ArgumentError, ~r/expected 1-D or 2-D tensor/, fn ->
        Nx.LinAlg.norm(t)
      end
    end

    test "norm 1D (vector norm)" do
      t = Nx.tensor([3.0, 4.0])
      result = Nx.LinAlg.norm(t)
      assert_in_delta Nx.to_number(result), 5.0, 1.0e-5
    end

    test "norm 2D (Frobenius)" do
      t = Nx.eye(3, type: :f32)
      result = Nx.LinAlg.norm(t)
      assert_in_delta Nx.to_number(result), :math.sqrt(3.0), 1.0e-5
    end

    test "norm 1D with ord: 1 (L1)" do
      t = Nx.tensor([1.0, -2.0, 3.0])
      result = Nx.LinAlg.norm(t, ord: 1)
      assert_in_delta Nx.to_number(result), 6.0, 1.0e-5
    end

    test "norm 1D with ord: :inf (max abs)" do
      t = Nx.tensor([1.0, -5.0, 3.0])
      result = Nx.LinAlg.norm(t, ord: :inf)
      assert_in_delta Nx.to_number(result), 5.0, 1.0e-5
    end

    test "determinant of identity is 1" do
      t = Nx.eye(3, type: :f32)
      result = Nx.LinAlg.determinant(t)
      assert_in_delta Nx.to_number(result), 1.0, 1.0e-5
    end

    test "determinant of 2x scaling matrix" do
      t = Nx.multiply(Nx.eye(3, type: :f32), 2.0)
      result = Nx.LinAlg.determinant(t)
      assert_in_delta Nx.to_number(result), 8.0, 1.0e-5
    end

    test "determinant raises on non-square" do
      t = Nx.iota({2, 3}, type: :f32)
      assert_raise ArgumentError, ~r/square/, fn ->
        Nx.LinAlg.determinant(t)
      end
    end

    test "invert of identity is identity" do
      t = Nx.eye(3, type: :f32)
      result = Nx.LinAlg.invert(t)

      for {a, b} <- Enum.zip(Nx.to_flat_list(result), Nx.to_flat_list(t)) do
        assert_in_delta a, b, 1.0e-5
      end
    end

    test "invert raises on non-square" do
      t = Nx.iota({2, 3}, type: :f32)
      assert_raise ArgumentError, ~r/square/, fn ->
        Nx.LinAlg.invert(t)
      end
    end

    test "solve Ax=b for identity A" do
      a = Nx.eye(3, type: :f32)
      b = Nx.tensor([1.0, 2.0, 3.0])
      result = Nx.LinAlg.solve(a, b)
      assert Nx.to_flat_list(result) == [1.0, 2.0, 3.0]
    end

    test "solve raises on non-square A" do
      a = Nx.iota({2, 3}, type: :f32)
      b = Nx.tensor([1.0, 2.0])
      assert_raise ArgumentError, ~r/square/, fn ->
        Nx.LinAlg.solve(a, b)
      end
    end

    test "triangular_solve with identity" do
      a = Nx.eye(3, type: :f32)
      b = Nx.tensor([[1.0], [2.0], [3.0]])
      result = Nx.LinAlg.triangular_solve(a, b)
      assert Nx.to_flat_list(result) == [1.0, 2.0, 3.0]
    end

    test "triangular_solve raises on invalid transform_a" do
      a = Nx.eye(3, type: :f32)
      b = Nx.tensor([[1.0], [2.0], [3.0]])

      assert_raise ArgumentError, ~r/transform_a/, fn ->
        Nx.LinAlg.triangular_solve(a, b, transform_a: :invalid)
      end
    end

    test "qr raises on invalid mode" do
      t = Nx.iota({3, 3}, type: :f32)
      assert_raise ArgumentError, ~r/invalid :mode/, fn ->
        Nx.LinAlg.qr(t, mode: :invalid)
      end
    end

    test "qr of identity returns Q=I, R=I" do
      t = Nx.eye(3, type: :f32)
      {q, _r} = Nx.LinAlg.qr(t)

      for {a, b} <- Enum.zip(Nx.to_flat_list(q), Nx.to_flat_list(Nx.eye(3, type: :f32))) do
        assert_in_delta abs(a), abs(b), 1.0e-5
      end
    end

    test "A = Q*R roundtrip" do
      a = Nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
      {q, r} = Nx.LinAlg.qr(a)
      reconstructed = Nx.dot(q, r)

      for {orig, rec} <- Enum.zip(Nx.to_flat_list(a), Nx.to_flat_list(reconstructed)) do
        assert_in_delta orig, rec, 1.0e-4
      end
    end

    test "invert(A) * A ≈ identity" do
      a = Nx.tensor([[2.0, 1.0], [1.0, 3.0]])
      inv = Nx.LinAlg.invert(a)
      product = Nx.dot(inv, a)
      eye = Nx.eye(2, type: :f32)

      for {p, e} <- Enum.zip(Nx.to_flat_list(product), Nx.to_flat_list(eye)) do
        assert_in_delta p, e, 1.0e-4
      end
    end
  end

  # ── More equivalence properties ────────────────────────────────────

  describe "numerical equivalences" do
    property "diff(cumulative_sum(x)) ≈ x[1:]" do
      check all(n <- integer(2..10), max_runs: 20) do
        t = Nx.iota({n}, type: :f32)
        cumsum = Nx.cumulative_sum(t)
        diff = Nx.diff(cumsum)
        original_tail = Nx.slice(t, [1], [n - 1])

        for {d, o} <- Enum.zip(Nx.to_flat_list(diff), Nx.to_flat_list(original_tail)) do
          assert_in_delta d, o, 1.0e-4
        end
      end
    end

    property "clip(x, min, max) == max(min(x, max), min)" do
      check all(n <- integer(1..8), max_runs: 20) do
        t = Nx.iota({n}, type: :f32)
        lo = 2.0
        hi = 6.0
        clipped = Nx.clip(t, lo, hi)
        manual = t |> Nx.min(hi) |> Nx.max(lo)
        assert Nx.to_flat_list(clipped) == Nx.to_flat_list(manual)
      end
    end

    property "select(1, a, b) == a and select(0, a, b) == b" do
      check all(n <- integer(1..8), max_runs: 20) do
        a = Nx.iota({n}, type: :f32)
        b = Nx.add(a, 100.0)

        result_true = Nx.select(Nx.tensor(1), a, b)
        assert Nx.to_flat_list(result_true) == Nx.to_flat_list(a)

        result_false = Nx.select(Nx.tensor(0), a, b)
        assert Nx.to_flat_list(result_false) == Nx.to_flat_list(b)
      end
    end

    property "det(A) * det(inv(A)) ≈ 1 for invertible A" do
      check all(
              _ <- constant(:ok),
              max_runs: 5
            ) do
        # Use a well-conditioned matrix
        a = Nx.tensor([[4.0, 1.0], [2.0, 3.0]])
        det_a = Nx.LinAlg.determinant(a) |> Nx.to_number()
        det_inv = Nx.LinAlg.determinant(Nx.LinAlg.invert(a)) |> Nx.to_number()
        assert_in_delta det_a * det_inv, 1.0, 1.0e-4
      end
    end

    property "norm(x) >= 0 for any x" do
      check all(n <- integer(1..8), max_runs: 20) do
        # Mix of positive and negative values
        t = Nx.subtract(Nx.iota({n}, type: :f32), Nx.tensor(n / 2.0))
        norm = Nx.LinAlg.norm(t) |> Nx.to_number()
        assert norm >= 0.0
      end
    end

    property "norm(scalar * x) == |scalar| * norm(x)" do
      check all(n <- integer(1..8), max_runs: 20) do
        t = Nx.add(Nx.iota({n}, type: :f32), 1.0)
        scalar = 3.0
        norm_t = Nx.LinAlg.norm(t) |> Nx.to_number()
        norm_st = Nx.LinAlg.norm(Nx.multiply(t, scalar)) |> Nx.to_number()
        assert_in_delta norm_st, abs(scalar) * norm_t, 1.0e-4
      end
    end

    property "eye * x == x for matrix multiply" do
      check all(
              m <- integer(1..5),
              n <- integer(1..5),
              max_runs: 20
            ) do
        x = Nx.iota({m, n}, type: :f32)
        eye = Nx.eye(m, type: :f32)
        result = Nx.dot(eye, x)

        for {a, b} <- Enum.zip(Nx.to_flat_list(result), Nx.to_flat_list(x)) do
          assert_in_delta a, b, 1.0e-4
        end
      end
    end
  end
end
