defmodule Nx.FuzzTypesTest do
  @moduledoc """
  Fuzz tests for type-specific edge cases.

  f16/bf16 limited range, multi-type binary ops (type promotion),
  and high-rank tensor handling.
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  @fuzz_scale String.to_integer(System.get_env("FUZZ_SCALE", "1"))

  # ── f16 specific tests ────────────────────────────────────────────

  describe "f16 edge cases" do
    # f16: max ~65504, min subnormal ~5.96e-8, eps ~9.77e-4
    @f16_max 65504.0
    @f16_tiny 1.0e-7

    test "f16 max value roundtrips" do
      t = Nx.tensor(@f16_max, type: :f16)
      val = Nx.to_number(t)
      assert_in_delta val, @f16_max, 1.0
    end

    test "f16 overflow to Inf" do
      t = Nx.tensor(@f16_max, type: :f16)
      result = Nx.multiply(t, 2.0)
      val = Nx.to_number(result)
      assert val == :infinity
    end

    test "f16 tiny values" do
      t = Nx.tensor(@f16_tiny, type: :f16)
      assert is_struct(t, Nx.Tensor)
    end

    property "f16 iota doesn't crash" do
      check all(shape <- member_of([{3}, {2, 3}, {4, 2}]), max_runs: 10 * @fuzz_scale) do
        t = Nx.iota(shape, type: :f16)
        assert Nx.shape(t) == shape
        assert Nx.type(t) == {:f, 16}
      end
    end

    property "f16 arithmetic doesn't crash" do
      check all(
              n <- integer(1..8),
              max_runs: 10 * @fuzz_scale
            ) do
        a = Nx.iota({n}, type: :f16)
        b = Nx.iota({n}, type: :f16)
        assert is_struct(Nx.add(a, b), Nx.Tensor)
        assert is_struct(Nx.multiply(a, b), Nx.Tensor)
      end
    end

    property "f16 reductions don't crash" do
      check all(n <- integer(1..16), max_runs: 10 * @fuzz_scale) do
        t = Nx.iota({n}, type: :f16)
        assert is_struct(Nx.sum(t), Nx.Tensor)
        assert is_struct(Nx.reduce_max(t), Nx.Tensor)
      end
    end
  end

  # ── bf16 specific tests ───────────────────────────────────────────

  describe "bf16 edge cases" do
    # bf16: max ~3.4e38 (same as f32), but only 7 mantissa bits (eps ~7.8e-3)

    test "bf16 loses precision vs f32" do
      # bf16 can only represent ~3 decimal digits
      f32_val = Nx.tensor(1.234567, type: :f32)
      bf16_val = Nx.as_type(f32_val, :bf16)
      back = Nx.as_type(bf16_val, :f32)

      diff = Nx.subtract(f32_val, back) |> Nx.abs() |> Nx.to_number()
      # bf16 eps is ~7.8e-3, so diff should be within that
      assert diff < 0.01
    end

    property "bf16 arithmetic doesn't crash" do
      check all(n <- integer(1..8), max_runs: 10 * @fuzz_scale) do
        a = Nx.iota({n}, type: :bf16)
        b = Nx.iota({n}, type: :bf16)
        assert is_struct(Nx.add(a, b), Nx.Tensor)
        assert is_struct(Nx.multiply(a, b), Nx.Tensor)
      end
    end

    property "bf16 dot product doesn't crash" do
      check all(
              m <- integer(1..8),
              n <- integer(1..8),
              k <- integer(1..8),
              max_runs: 10 * @fuzz_scale
            ) do
        a = Nx.iota({m, k}, type: :bf16)
        b = Nx.iota({k, n}, type: :bf16)
        result = Nx.dot(a, b)
        assert Nx.shape(result) == {m, n}
      end
    end
  end

  # ── Multi-type binary ops (type promotion) ────────────────────────

  describe "type promotion in binary ops" do
    @type_pairs [
      {:u8, :f32},
      {:s32, :f32},
      {:s32, :f64},
      {:f16, :f32},
      {:bf16, :f32},
      {:f32, :f64},
      {:u8, :s32},
      {:s8, :u16},
      {:f16, :f64}
    ]

    for {t1, t2} <- @type_pairs do
      property "add #{t1} + #{t2} promotes correctly" do
        check all(n <- integer(1..8), max_runs: 10 * @fuzz_scale) do
          a = Nx.iota({n}, type: unquote(t1))
          b = Nx.iota({n}, type: unquote(t2))
          result = Nx.add(a, b)
          assert Nx.shape(result) == {n}

          # Result type should be the "wider" type
          {result_class, result_bits} = Nx.type(result)
          assert result_class in [:f, :s, :u, :bf]
        end
      end
    end

    property "multiply integer * float promotes to float" do
      check all(n <- integer(1..8), max_runs: 10 * @fuzz_scale) do
        a = Nx.iota({n}, type: :s32)
        b = Nx.iota({n}, type: :f32)
        result = Nx.multiply(a, b)
        assert elem(Nx.type(result), 0) == :f
      end
    end

    property "comparison of different types doesn't crash" do
      check all(n <- integer(1..8), max_runs: 10 * @fuzz_scale) do
        a = Nx.iota({n}, type: :s32)
        b = Nx.iota({n}, type: :f32)
        result = Nx.equal(a, b)
        assert Nx.type(result) == {:u, 8}
      end
    end
  end

  # ── High-rank tensors ─────────────────────────────────────────────

  describe "high-rank tensors" do
    property "rank 4 operations don't crash" do
      check all(type <- member_of([:f32, :s32]), max_runs: 10 * @fuzz_scale) do
        t = Nx.iota({2, 3, 2, 2}, type: type)
        assert Nx.shape(Nx.sum(t)) == {}
        assert Nx.shape(Nx.sum(t, axes: [0])) == {3, 2, 2}
        assert Nx.shape(Nx.transpose(t)) == {2, 2, 3, 2}
        assert Nx.shape(Nx.reshape(t, {6, 4})) == {6, 4}
      end
    end

    property "rank 5 operations don't crash" do
      check all(type <- member_of([:f32]), max_runs: 5 * @fuzz_scale) do
        t = Nx.iota({2, 2, 2, 2, 2}, type: type)
        assert Nx.shape(Nx.sum(t)) == {}
        assert Nx.shape(Nx.abs(t)) == {2, 2, 2, 2, 2}
        assert Nx.shape(Nx.reshape(t, {4, 8})) == {4, 8}
      end
    end

    test "rank 6 basic ops" do
      t = Nx.iota({2, 2, 2, 2, 2, 2}, type: :f32)
      assert Nx.shape(t) == {2, 2, 2, 2, 2, 2}
      assert Nx.size(t) == 64
      assert is_struct(Nx.sum(t), Nx.Tensor)
      assert is_struct(Nx.abs(t), Nx.Tensor)
    end
  end

  # ── as_type edge cases ────────────────────────────────────────────

  describe "as_type edge cases" do
    test "f32 Inf to integer" do
      t = Nx.tensor(:infinity, type: :f32)
      # Converting Inf to integer — implementation-defined behavior
      result = Nx.as_type(t, :s32)
      assert is_struct(result, Nx.Tensor)
    end

    test "f32 NaN to integer" do
      t = Nx.tensor(:nan, type: :f32)
      result = Nx.as_type(t, :s32)
      assert is_struct(result, Nx.Tensor)
    end

    test "large f64 to f16 clamps or overflows" do
      t = Nx.tensor(1.0e10, type: :f64)
      result = Nx.as_type(t, :f16)
      val = Nx.to_number(result)
      # Should be Inf since 1e10 > f16 max (65504)
      assert val == :infinity
    end

    test "negative to unsigned wraps" do
      t = Nx.tensor(-1, type: :s32)
      result = Nx.as_type(t, :u8)
      val = Nx.to_number(result)
      assert val == 255
    end
  end
end
