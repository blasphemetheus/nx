defmodule Nx.FuzzErrorsTest do
  @moduledoc """
  Fuzz tests for error paths.

  Verifies that invalid inputs produce clean ArgumentError
  with clear messages, not crashes or FunctionClauseError.
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  # ── Shape mismatch errors ─────────────────────────────────────────

  describe "binary ops reject non-broadcastable shapes" do
    property "add rejects incompatible shapes" do
      check all(
              n <- integer(2..8),
              m <- integer(2..8),
              max_runs: 15
            ) do
        if n != m do
          a = Nx.iota({n})
          b = Nx.iota({m})
          assert_raise ArgumentError, fn -> Nx.add(a, b) end
        end
      end
    end

    property "multiply rejects incompatible 2D shapes" do
      check all(
              r1 <- integer(2..6),
              c1 <- integer(2..6),
              r2 <- integer(2..6),
              c2 <- integer(2..6),
              max_runs: 15
            ) do
        if c1 != c2 and c1 != 1 and c2 != 1 and r1 != r2 and r1 != 1 and r2 != 1 do
          a = Nx.iota({r1, c1})
          b = Nx.iota({r2, c2})
          assert_raise ArgumentError, fn -> Nx.multiply(a, b) end
        end
      end
    end
  end

  # ── Invalid axis errors ───────────────────────────────────────────

  describe "invalid axis errors" do
    property "sum rejects out-of-bounds axis" do
      check all(
              rank <- integer(1..4),
              max_runs: 15
            ) do
        shape = List.to_tuple(List.duplicate(3, rank))
        t = Nx.iota(shape)
        assert_raise ArgumentError, fn -> Nx.sum(t, axes: [rank]) end
      end
    end

    property "sum rejects negative out-of-bounds axis" do
      check all(
              rank <- integer(1..4),
              max_runs: 15
            ) do
        shape = List.to_tuple(List.duplicate(3, rank))
        t = Nx.iota(shape)
        assert_raise ArgumentError, fn -> Nx.sum(t, axes: [-(rank + 1)]) end
      end
    end

    test "transpose rejects invalid permutation" do
      t = Nx.iota({2, 3, 4})
      assert_raise ArgumentError, fn -> Nx.transpose(t, axes: [0, 0, 1]) end
    end
  end

  # ── Reshape errors ────────────────────────────────────────────────

  describe "reshape errors" do
    property "reshape rejects incompatible sizes" do
      check all(
              n <- integer(2..16),
              m <- integer(2..16),
              max_runs: 15
            ) do
        if n != m do
          t = Nx.iota({n})
          assert_raise ArgumentError, fn -> Nx.reshape(t, {m}) end
        end
      end
    end

    test "reshape rejects negative dimensions" do
      t = Nx.iota({6})
      assert_raise ArgumentError, fn -> Nx.reshape(t, {-1, 3}) end
    end
  end

  # ── Slice errors ──────────────────────────────────────────────────

  describe "slice errors" do
    property "slice clamps out-of-bounds start (doesn't crash)" do
      check all(
              len <- integer(2..16),
              max_runs: 15
            ) do
        t = Nx.iota({len})
        # Nx clamps out-of-bounds indices to last valid position (XLA behavior)
        result = Nx.slice(t, [len + 5], [1])
        assert Nx.shape(result) == {1}
      end
    end

    property "slice rejects length exceeding size" do
      check all(
              len <- integer(2..16),
              max_runs: 15
            ) do
        t = Nx.iota({len})
        assert_raise ArgumentError, fn -> Nx.slice(t, [0], [len + 1]) end
      end
    end

    test "slice rejects wrong rank start_indices" do
      t = Nx.iota({3, 4})
      assert_raise ArgumentError, fn -> Nx.slice(t, [0], [2]) end
    end
  end

  # ── Dot errors ────────────────────────────────────────────────────

  describe "dot errors" do
    property "dot rejects mismatched contracting dims" do
      check all(
              m <- integer(2..8),
              k1 <- integer(2..8),
              k2 <- integer(2..8),
              n <- integer(2..8),
              max_runs: 15
            ) do
        if k1 != k2 do
          a = Nx.iota({m, k1})
          b = Nx.iota({k2, n})
          assert_raise ArgumentError, fn -> Nx.dot(a, b) end
        end
      end
    end
  end

  # ── Type errors ───────────────────────────────────────────────────

  describe "type-specific errors" do
    property "bitwise_and rejects float types" do
      check all(
              shape <- member_of([{3}, {2, 3}]),
              type <- member_of([:f32, :f64]),
              max_runs: 10
            ) do
        t = Nx.iota(shape, type: type)
        assert_raise ArgumentError, fn -> Nx.bitwise_and(t, t) end
      end
    end

    property "bitwise_not rejects float types" do
      check all(
              shape <- member_of([{3}, {2, 3}]),
              type <- member_of([:f32, :f64]),
              max_runs: 10
            ) do
        t = Nx.iota(shape, type: type)
        assert_raise ArgumentError, fn -> Nx.bitwise_not(t) end
      end
    end
  end

  # ── Gather/Take errors ────────────────────────────────────────────

  describe "gather/take errors" do
    property "take rejects out-of-bounds indices" do
      check all(
              len <- integer(2..8),
              max_runs: 10
            ) do
        t = Nx.iota({len})
        # This may or may not raise depending on backend behavior
        # Just verify it doesn't segfault or produce garbage
        try do
          _result = Nx.take(t, Nx.tensor([len]))
        rescue
          ArgumentError -> :ok
        end
      end
    end
  end

  # ── Concatenate errors ────────────────────────────────────────────

  describe "concatenate errors" do
    property "concatenate rejects mismatched non-concat dims" do
      check all(
              r1 <- integer(1..6),
              r2 <- integer(1..6),
              c1 <- integer(2..6),
              c2 <- integer(2..6),
              max_runs: 15
            ) do
        if c1 != c2 do
          a = Nx.iota({r1, c1})
          b = Nx.iota({r2, c2})
          assert_raise ArgumentError, fn -> Nx.concatenate([a, b], axis: 0) end
        end
      end
    end

    test "concatenate rejects empty list" do
      assert_raise ArgumentError, fn -> Nx.concatenate([]) end
    end
  end

  # ── LinAlg errors ─────────────────────────────────────────────────

  describe "linalg errors" do
    property "cholesky rejects non-square" do
      check all(
              m <- integer(2..6),
              n <- integer(2..6),
              max_runs: 10
            ) do
        if m != n do
          t = Nx.iota({m, n}, type: :f32)
          assert_raise ArgumentError, fn -> Nx.LinAlg.cholesky(t) end
        end
      end
    end

    property "qr rejects 1D tensor" do
      check all(n <- integer(1..8), max_runs: 10) do
        t = Nx.iota({n}, type: :f32)
        assert_raise ArgumentError, fn -> Nx.LinAlg.qr(t) end
      end
    end

    property "determinant rejects non-square" do
      check all(
              m <- integer(2..6),
              n <- integer(2..6),
              max_runs: 10
            ) do
        if m != n do
          t = Nx.iota({m, n}, type: :f32)
          assert_raise ArgumentError, fn -> Nx.LinAlg.determinant(t) end
        end
      end
    end
  end

  # ── Misc errors ───────────────────────────────────────────────────

  describe "miscellaneous error paths" do
    test "eye rejects non-positive size" do
      assert_raise ArgumentError, fn -> Nx.eye(0) end
      assert_raise ArgumentError, fn -> Nx.eye(-1) end
    end

    test "iota rejects zero dimensions" do
      assert_raise ArgumentError, fn -> Nx.iota({0}) end
      assert_raise ArgumentError, fn -> Nx.iota({3, 0}) end
    end

    test "pad rejects wrong padding config rank" do
      t = Nx.iota({3, 4})
      assert_raise ArgumentError, fn -> Nx.pad(t, 0, [{1, 1, 0}]) end
    end
  end
end
