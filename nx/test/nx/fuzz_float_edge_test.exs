defmodule Nx.FuzzFloatEdgeTest do
  @moduledoc """
  Value-aware float fuzzing (FUZZ_ROADMAP T1.1).

  Tensors are built from raw bit patterns via `FuzzGen`, so NaN payloads,
  ±Infinity, denormals, and −0.0 occur in every property. Oracles are chosen
  to be independent of implementation-defined NaN semantics:

    * bit-level classification computed in Elixir (no Nx involved)
    * bit-exactness of data-movement ops (they must not reinterpret values)
    * permutation invariance (sort must not lose or invent values)
    * one-directional IEEE guarantees (NaN in => NaN out for arithmetic)
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  import Nx.Testing

  @types [{:f, 32}, {:f, 64}]

  defp words(t) do
    {:f, size} = Nx.type(t)
    for <<w::size(size)-unsigned-native <- Nx.to_binary(t)>>, do: w
  end

  defp sign_bit_mask({:f, 32}), do: 0x8000_0000
  defp sign_bit_mask({:f, 64}), do: 0x8000_0000_0000_0000

  # Clamp finite magnitudes into [1e-3, 1e3]; NaN survives both selects
  # (comparisons against NaN are false, keeping the original element).
  defp cap_finite(t) do
    type = Nx.type(t)
    hi = Nx.tensor(1.0e3, type: type)
    lo = Nx.tensor(1.0, type: type)
    t = Nx.select(Nx.greater(Nx.abs(t), hi), Nx.multiply(Nx.sign(t), hi), t)
    Nx.select(Nx.less(Nx.abs(t), 1.0e-3), lo, t)
  end

  for type <- @types do
    describe "classification oracle #{inspect(type)}" do
      property "is_nan agrees with independent bit-level classification" do
        check all(
                t <- FuzzGen.bit_tensor(FuzzGen.shape(), unquote(Macro.escape(type))),
                max_runs: 50
              ) do
          expected = for c <- FuzzGen.classify(t), do: if(c == :nan, do: 1, else: 0)
          assert Nx.to_flat_list(Nx.is_nan(t)) == expected
        end
      end

      property "is_infinity agrees with independent bit-level classification" do
        check all(
                t <- FuzzGen.bit_tensor(FuzzGen.shape(), unquote(Macro.escape(type))),
                max_runs: 50
              ) do
          expected = for c <- FuzzGen.classify(t), do: if(c == :infinity, do: 1, else: 0)
          assert Nx.to_flat_list(Nx.is_infinity(t)) == expected
        end
      end
    end

    describe "data movement is bit-exact #{inspect(type)}" do
      property "reshape/transpose/reverse round-trips preserve every bit" do
        check all(
                t <- FuzzGen.bit_tensor(FuzzGen.shape(), unquote(Macro.escape(type))),
                max_runs: 50
              ) do
          flat = Nx.reshape(t, {Nx.size(t)})
          assert Nx.to_binary(flat) == Nx.to_binary(t)

          assert Nx.to_binary(t |> Nx.transpose() |> Nx.transpose()) == Nx.to_binary(t)
          assert Nx.to_binary(flat |> Nx.reverse() |> Nx.reverse()) == Nx.to_binary(flat)
        end
      end

      property "serialization round-trips preserve NaN payloads and -0.0" do
        check all(
                t <- FuzzGen.bit_tensor(FuzzGen.shape(), unquote(Macro.escape(type))),
                max_runs: 50
              ) do
          bin_rt = t |> Nx.to_binary() |> Nx.from_binary(Nx.type(t))
          assert Nx.to_binary(bin_rt) == Nx.to_binary(t)

          ser_rt = t |> Nx.serialize() |> Nx.deserialize()
          assert Nx.to_binary(ser_rt) == Nx.to_binary(t)
        end
      end
    end

    describe "sign-bit ops on finite values #{inspect(type)}" do
      property "negate flips exactly the sign bit (including -0.0 and denormals)" do
        check all(
                t <- FuzzGen.finite_bit_tensor(FuzzGen.shape(), unquote(Macro.escape(type))),
                max_runs: 50
              ) do
          mask = sign_bit_mask(unquote(Macro.escape(type)))
          expected = for w <- words(t), do: Bitwise.bxor(w, mask)
          assert words(Nx.negate(t)) == expected
        end
      end

      property "abs clears exactly the sign bit (including -0.0 and denormals)" do
        check all(
                t <- FuzzGen.finite_bit_tensor(FuzzGen.shape(), unquote(Macro.escape(type))),
                max_runs: 50
              ) do
          mask = sign_bit_mask(unquote(Macro.escape(type)))
          expected = for w <- words(t), do: Bitwise.band(w, Bitwise.bnot(mask))
          assert words(Nx.abs(t)) == expected
        end
      end
    end

    describe "IEEE one-directional guarantees #{inspect(type)}" do
      property "NaN in => NaN out for add/subtract/multiply/divide" do
        check all(
                t <- FuzzGen.bit_tensor(FuzzGen.shape(), unquote(Macro.escape(type))),
                u <- FuzzGen.bit_tensor(constant({}), unquote(Macro.escape(type))),
                max_runs: 50
              ) do
          # Cap finite magnitudes into [1e-3, 1e3] (NaN passes through both
          # selects untouched) so this property doesn't trip
          # [BUG-F64-OVERFLOW] — remove the caps when that bug is fixed.
          t = cap_finite(t)
          u = cap_finite(u)

          nan_in = Nx.is_nan(t)

          for op <- [:add, :subtract, :multiply, :divide] do
            nan_out = Nx.is_nan(apply(Nx, op, [t, u]))
            # wherever the input was NaN, the output must be NaN
            violation = Nx.logical_and(nan_in, Nx.logical_not(nan_out))
            assert Nx.to_number(Nx.any(violation)) == 0, "#{op} lost a NaN"
          end
        end
      end

      property "equal(t, t) is false exactly on NaN elements" do
        check all(
                t <- FuzzGen.bit_tensor(FuzzGen.shape(), unquote(Macro.escape(type))),
                max_runs: 50
              ) do
          assert Nx.to_flat_list(Nx.equal(t, t)) ==
                   Nx.to_flat_list(Nx.logical_not(Nx.is_nan(t)))
        end
      end
    end

    describe "permutation invariance #{inspect(type)}" do
      property "sort neither loses nor invents values, even with NaN/Inf present" do
        shapes = StreamData.filter(FuzzGen.shape(), &(tuple_size(&1) >= 1))

        check all(
                t <- FuzzGen.bit_tensor(shapes, unquote(Macro.escape(type))),
                max_runs: 50
              ) do
          sorted = Nx.sort(t, axis: 0)
          assert Enum.sort(words(sorted)) == Enum.sort(words(t))
        end
      end
    end
  end

  describe "infinity propagation through reductions" do
    property "sum of finite values plus a single +Inf is +Inf" do
      shapes = StreamData.filter(FuzzGen.shape(), &(tuple_size(&1) >= 1))

      check all(t <- FuzzGen.finite_bit_tensor(shapes, {:f, 64}), max_runs: 50) do
        flat = Nx.reshape(t, {Nx.size(t)})
        inf = Nx.tensor([:infinity], type: {:f, 64})
        with_inf = Nx.concatenate([inf, flat])

        total = Nx.sum(with_inf)
        assert Nx.to_number(Nx.is_infinity(total)) == 1
        assert Nx.to_number(Nx.greater(total, 0)) == 1
      end
    end
  end

  describe "known bugs: f64 overflow ([BUG-F64-OVERFLOW])" do
    # See FUZZ_FINDINGS/f64_binary_op_overflow_arithmetic_error.md.
    # IEEE 754 requires +Infinity in all four cases below. Flip these pins to
    # `assert Nx.to_flat_list(...) == [:infinity]` when the bug is fixed.

    @f64_max Nx.from_binary(<<0x7FEFFFFFFFFFFFFF::64-native>>, {:f, 64})

    test "add overflow raises instead of returning infinity" do
      assert_raise ArithmeticError, fn -> Nx.add(@f64_max, @f64_max) end
    end

    test "multiply overflow raises instead of returning infinity" do
      assert_raise ArithmeticError, fn -> Nx.multiply(@f64_max, 2.0) end
    end

    test "tensor-divisor divide overflow raises instead of returning infinity" do
      tiny = Nx.tensor(1.0e-113, type: {:f, 64})
      assert_raise ArithmeticError, fn -> Nx.divide(@f64_max, tiny) end
    end

    test "pow overflow returns NaN instead of infinity" do
      result = Nx.pow(@f64_max, 2)
      assert Nx.to_flat_list(result) == [:nan]
    end

    test "control: f32 overflow correctly produces infinity" do
      f32_max = Nx.from_binary(<<0x7F7FFFFF::32-native>>, {:f, 32})
      assert Nx.to_flat_list(Nx.multiply(f32_max, f32_max)) == [:infinity]
    end
  end

  describe "cancellation" do
    property "summing (x, -x(1+eps)) pairs matches the analytic residual" do
      check all(t <- FuzzGen.cancellation_tensor(), max_runs: 50) do
        # each pair contributes -x * 1.0e-14; f64 has plenty of headroom to
        # represent both the terms and the residual, so a correct summation
        # is close to the analytic value at loose relative tolerance
        xs = t |> Nx.to_flat_list() |> Enum.take_every(2)
        expected = -1.0e-14 * Enum.sum(xs)

        assert_all_close(Nx.sum(t), Nx.tensor(expected, type: {:f, 64}),
          rtol: 1.0e-2,
          atol: 1.0e-30
        )
      end
    end
  end
end
