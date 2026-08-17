defmodule Nx.FuzzIntSemanticsTest do
  @moduledoc """
  Exact-oracle fuzz for integer op semantics.

  Every property compares against an Elixir reference computed in
  unbounded integers and wrapped to the tensor type (two's complement).
  Conventions verified: quotient/remainder truncate toward zero (matching
  `Integer.div/rem`); INT_MIN / -1 wraps; shifts wrap on overflow with
  arithmetic right shift for signed types; division by zero raises
  ArithmeticError (matching Elixir's own `div/2`).
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  import Bitwise

  @fuzz_scale String.to_integer(System.get_env("FUZZ_SCALE", "1"))

  @int_types [{:u, 8}, {:u, 16}, {:u, 32}, {:u, 64}, {:s, 8}, {:s, 16}, {:s, 32}, {:s, 64}]

  defp range({:u, bits}), do: {0, (1 <<< bits) - 1}
  defp range({:s, bits}), do: {-(1 <<< (bits - 1)), (1 <<< (bits - 1)) - 1}

  defp wrap(x, {:u, bits}), do: Integer.mod(x, 1 <<< bits)

  defp wrap(x, {:s, bits}) do
    m = 1 <<< bits
    v = Integer.mod(x, m)
    if v >= m >>> 1, do: v - m, else: v
  end

  # unsigned bit pattern of a value in this type
  defp pattern(x, {_, bits}), do: Integer.mod(x, 1 <<< bits)

  defp int_value(type) do
    {lo, hi} = range(type)

    frequency([
      {4, integer(lo..hi)},
      {1, member_of(Enum.uniq([lo, hi, 0, 1, -1, div(hi, 2)]) |> Enum.filter(&(&1 in lo..hi)))}
    ])
  end

  for type <- @int_types do
    describe "quotient/remainder #{inspect(type)}" do
      property "match Integer.div/rem wrapped to the type" do
        check all(
                a <- int_value(unquote(Macro.escape(type))),
                b <- int_value(unquote(Macro.escape(type))),
                b != 0,
                max_runs: 40 * @fuzz_scale
              ) do
          type = unquote(Macro.escape(type))
          ta = Nx.tensor(a, type: type)
          tb = Nx.tensor(b, type: type)

          # quotient truncates toward zero, matching Elixir div/2
          assert Nx.to_number(Nx.quotient(ta, tb)) == wrap(div(a, b), type)
          assert Nx.to_number(Nx.remainder(ta, tb)) == wrap(rem(a, b), type)
        end
      end

      property "q * b + r reconstructs a (in wrapped arithmetic)" do
        check all(
                a <- int_value(unquote(Macro.escape(type))),
                b <- int_value(unquote(Macro.escape(type))),
                b != 0,
                max_runs: 40 * @fuzz_scale
              ) do
          type = unquote(Macro.escape(type))
          ta = Nx.tensor(a, type: type)
          tb = Nx.tensor(b, type: type)

          q = Nx.to_number(Nx.quotient(ta, tb))
          r = Nx.to_number(Nx.remainder(ta, tb))

          assert wrap(q * b + r, type) == a
        end
      end
    end

    describe "shifts #{inspect(type)}" do
      property "left/right shift match Bitwise references wrapped to the type" do
        check all(
                a <- int_value(unquote(Macro.escape(type))),
                s <- integer(0..(2 * elem(unquote(Macro.escape(type)), 1))),
                max_runs: 40 * @fuzz_scale
              ) do
          type = unquote(Macro.escape(type))
          ta = Nx.tensor(a, type: type)

          assert Nx.to_number(Nx.left_shift(ta, s)) == wrap(a <<< s, type),
                 "left_shift(#{a}, #{s})"

          # arithmetic right shift: Elixir's >>> on unbounded integers is
          # already arithmetic (sign-extending) for negatives
          assert Nx.to_number(Nx.right_shift(ta, s)) == wrap(a >>> s, type),
                 "right_shift(#{a}, #{s})"
        end
      end
    end

    describe "bitwise ops #{inspect(type)}" do
      property "and/or/xor/not match Bitwise references; De Morgan holds" do
        check all(
                a <- int_value(unquote(Macro.escape(type))),
                b <- int_value(unquote(Macro.escape(type))),
                max_runs: 40 * @fuzz_scale
              ) do
          type = unquote(Macro.escape(type))
          ta = Nx.tensor(a, type: type)
          tb = Nx.tensor(b, type: type)

          assert Nx.to_number(Nx.bitwise_and(ta, tb)) == wrap(a &&& b, type)
          assert Nx.to_number(Nx.bitwise_or(ta, tb)) == wrap(a ||| b, type)
          assert Nx.to_number(Nx.bitwise_xor(ta, tb)) == wrap(bxor(a, b), type)
          assert Nx.to_number(Nx.bitwise_not(ta)) == wrap(bnot(a), type)

          # De Morgan: not(a and b) == not(a) or not(b)
          assert Nx.bitwise_not(Nx.bitwise_and(ta, tb)) ==
                   Nx.bitwise_or(Nx.bitwise_not(ta), Nx.bitwise_not(tb))
        end
      end
    end

    describe "bit counting #{inspect(type)}" do
      property "population_count and count_leading_zeros match bit-pattern references" do
        check all(a <- int_value(unquote(Macro.escape(type))), max_runs: 40 * @fuzz_scale) do
          type = unquote(Macro.escape(type))
          {_, bits} = type
          ta = Nx.tensor(a, type: type)
          pat = pattern(a, type)

          pat_bin = <<pat::size(bits)>>
          expected_pop = for(<<bit::1 <- pat_bin>>, do: bit) |> Enum.sum()
          expected_clz = bits - if pat == 0, do: 0, else: Integer.digits(pat, 2) |> length()

          assert Nx.to_number(Nx.population_count(ta)) == expected_pop,
                 "population_count(#{a})"

          assert Nx.to_number(Nx.count_leading_zeros(ta)) == expected_clz,
                 "count_leading_zeros(#{a})"
        end
      end
    end
  end

  describe "documented edge conventions" do
    test "INT_MIN / -1 wraps instead of overflowing" do
      for {:s, bits} = type <- [s: 8, s: 16, s: 32, s: 64] do
        {min, _} = range(type)
        result = Nx.quotient(Nx.tensor(min, type: type), Nx.tensor(-1, type: type))
        assert Nx.to_number(result) == min, "s#{bits}"
      end
    end

    test "integer division by zero raises ArithmeticError, matching Elixir's div/2" do
      assert_raise ArithmeticError, fn -> Nx.quotient(Nx.tensor(1), Nx.tensor(0)) end
      assert_raise ArithmeticError, fn -> Nx.remainder(Nx.tensor(1), Nx.tensor(0)) end
    end

    test "negative shift amounts raise ArgumentError" do
      assert_raise ArgumentError, fn -> Nx.left_shift(Nx.tensor(1), -1) end
      assert_raise ArgumentError, fn -> Nx.right_shift(Nx.tensor(1), -1) end
    end
  end
end
