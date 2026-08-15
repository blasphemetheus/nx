defmodule Nx.FuzzNewApiTest do
  @moduledoc """
  Metamorphic sweep of API surface added in v0.11–v0.13 (FUZZ_ROADMAP T2.2) —
  the least-fuzzed code in the tree: `pad_outer` modes, `rfft`/`irfft`/`fft2`,
  f8 float types (E5M2 and E4M3FN), and sub-byte integer arithmetic.

  Oracles are index-mapping references computed in Elixir, exact modular
  arithmetic, and cross-API consistency (rfft vs fft prefix, fft2 vs composed
  1-D transforms, Parseval's identity).
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  import Nx.Testing

  describe "pad_outer modes vs index-mapping reference" do
    # Index mapping into the source for position i in -pl..(n+pr-1):
    defp source_index(:cyclic, i, n), do: Integer.mod(i, n)
    defp source_index(:replicate, i, n), do: i |> max(0) |> min(n - 1)
    defp source_index(:reflect, i, _n) when i < 0, do: -i
    defp source_index(:reflect, i, n) when i >= n, do: 2 * n - 2 - i
    defp source_index(:reflect, i, _n), do: i
    defp source_index(:mirror, i, _n) when i < 0, do: -i - 1
    defp source_index(:mirror, i, n) when i >= n, do: 2 * n - 1 - i
    defp source_index(:mirror, i, _n), do: i

    property "1-D pad_outer matches the reference for all four modes" do
      check all(
              n <- integer(2..8),
              pl <- integer(0..3),
              pr <- integer(0..3),
              mode <- member_of([:cyclic, :replicate, :reflect, :mirror]),
              max_runs: 40
            ) do
        # reflect/mirror can only reach back as far as the source allows
        pl = if mode == :reflect, do: min(pl, n - 1), else: min(pl, n)
        pr = if mode == :reflect, do: min(pr, n - 1), else: min(pr, n)

        t = Nx.iota({n}, type: {:s, 32})
        values = Nx.to_flat_list(t)

        result = Nx.pad_outer(t, mode, [{pl, pr}])

        expected =
          for i <- -pl..(n + pr - 1) do
            Enum.at(values, source_index(mode, i, n))
          end

        assert Nx.to_flat_list(result) == expected,
               "mode #{mode}, pads {#{pl}, #{pr}}"
      end
    end

    property "2-D pad_outer applies the mode independently per axis" do
      check all(
              rows <- integer(2..4),
              cols <- integer(2..4),
              p <- integer(1..2),
              mode <- member_of([:cyclic, :replicate, :mirror]),
              max_runs: 20
            ) do
        t = Nx.iota({rows, cols}, type: {:s, 32})
        nested = Nx.to_list(t)

        result = Nx.pad_outer(t, mode, [{p, p}, {p, p}])

        expected =
          for i <- -p..(rows + p - 1) do
            row = Enum.at(nested, source_index(mode, i, rows))

            for j <- -p..(cols + p - 1) do
              Enum.at(row, source_index(mode, j, cols))
            end
          end

        assert Nx.to_list(result) == expected
      end
    end
  end

  describe "FFT family consistency" do
    defp real_signal(n) do
      bind(list_of(float(min: -10.0, max: 10.0), length: n), fn vals ->
        constant(Nx.tensor(vals, type: {:f, 64}))
      end)
    end

    property "rfft is the first n/2+1 bins of fft for real input" do
      check all(n <- member_of([4, 8, 16]), x <- real_signal(n), max_runs: 15) do
        full = Nx.fft(x)
        half = Nx.rfft(x)

        assert_all_close(Nx.slice(full, [0], [div(n, 2) + 1]), half, atol: 1.0e-9)
      end
    end

    property "Parseval: sum(x^2) == mean(|fft(x)|^2)" do
      check all(n <- member_of([4, 8, 16]), x <- real_signal(n), max_runs: 15) do
        time_energy = Nx.sum(Nx.multiply(x, x))

        spectrum = Nx.fft(x)

        freq_energy =
          spectrum |> Nx.abs() |> then(&Nx.multiply(&1, &1)) |> Nx.sum() |> Nx.divide(n)

        assert_all_close(time_energy, Nx.real(freq_energy), rtol: 1.0e-9, atol: 1.0e-9)
      end
    end

    property "fft2 equals composed 1-D ffts along each axis" do
      check all(
              rows <- member_of([2, 4]),
              cols <- member_of([2, 4]),
              max_runs: 10
            ) do
        check all(
                vals <- list_of(float(min: -10.0, max: 10.0), length: rows * cols),
                max_runs: 1
              ) do
          t = vals |> Nx.tensor(type: {:f, 64}) |> Nx.reshape({rows, cols})

          # fft along last axis (cols), then along rows via double transpose
          composed =
            t
            |> Nx.fft()
            |> Nx.transpose()
            |> Nx.fft()
            |> Nx.transpose()

          assert_all_close(Nx.real(Nx.fft2(t)), Nx.real(composed), atol: 1.0e-8)
          assert_all_close(Nx.imag(Nx.fft2(t)), Nx.imag(composed), atol: 1.0e-8)
        end
      end
    end
  end

  describe "f8 float types" do
    property "e4m3fn saturates to +/-448 on overflow — never infinity" do
      check all(
              mag <- float(min: 449.0, max: 1.0e30),
              sign <- member_of([1.0, -1.0]),
              max_runs: 30
            ) do
        t = Nx.tensor(sign * mag, type: {:f, 32})
        converted = Nx.as_type(t, :f8_e4m3fn)

        assert Nx.to_number(Nx.is_infinity(converted)) == 0
        assert_all_close(converted, Nx.tensor(sign * 448.0), atol: 0.0)
      end
    end

    property "e5m2 overflows to +/-infinity like a standard float" do
      check all(
              mag <- float(min: 1.0e5, max: 1.0e30),
              sign <- member_of([1.0, -1.0]),
              max_runs: 30
            ) do
        t = Nx.tensor(sign * mag, type: {:f, 32})
        converted = Nx.as_type(t, {:f, 8})

        assert Nx.to_number(Nx.is_infinity(converted)) == 1
        assert Nx.to_number(Nx.greater(converted, 0)) == if(sign > 0, do: 1, else: 0)
      end
    end

    property "exactly-representable values survive the f32 -> f8 -> f32 round trip" do
      # small integers and powers of two within both f8 ranges
      exact = [0.0, 1.0, -1.0, 2.0, -2.0, 4.0, 8.0, 16.0, 0.5, -0.5, 0.25, 3.0, -3.0]

      check all(v <- member_of(exact), type <- member_of([:f8_e4m3fn, {:f, 8}]), max_runs: 30) do
        t = Nx.tensor(v, type: {:f, 32})
        round_tripped = t |> Nx.as_type(type) |> Nx.as_type({:f, 32})
        assert Nx.to_number(round_tripped) == v
      end
    end

    test "e4m3fn NaN is preserved through conversion" do
      t = Nx.as_type(Nx.tensor(:nan, type: {:f, 32}), :f8_e4m3fn)
      assert Nx.to_number(Nx.is_nan(t)) == 1
    end
  end

  describe "sub-byte integer arithmetic" do
    property "u2/u4 add/multiply/subtract wrap modulo 2^bits" do
      check all(
              type <- member_of([{:u, 2}, {:u, 4}]),
              op <- member_of([:add, :multiply, :subtract]),
              max_runs: 40
            ) do
        {:u, bits} = type
        modulus = Bitwise.bsl(1, bits)

        check all(
                a <- integer(0..(modulus - 1)),
                b <- integer(0..(modulus - 1)),
                max_runs: 4
              ) do
          ta = Nx.tensor(a, type: type)
          tb = Nx.tensor(b, type: type)

          result = apply(Nx, op, [ta, tb])
          assert Nx.type(result) == type

          reference =
            case op do
              :add -> Integer.mod(a + b, modulus)
              :multiply -> Integer.mod(a * b, modulus)
              :subtract -> Integer.mod(a - b, modulus)
            end

          assert Nx.to_number(result) == reference,
                 "#{inspect(type)} #{op}(#{a}, #{b})"
        end
      end
    end

    property "in-range sub-byte values survive the round trip through s32" do
      check all(type <- member_of([{:u, 2}, {:s, 2}, {:u, 4}, {:s, 4}]), max_runs: 20) do
        {sign, bits} = type

        {lo, hi} =
          case sign do
            :u -> {0, Bitwise.bsl(1, bits) - 1}
            :s -> {-Bitwise.bsl(1, bits - 1), Bitwise.bsl(1, bits - 1) - 1}
          end

        check all(v <- integer(lo..hi), max_runs: 4) do
          t = Nx.tensor(v, type: type)
          round_tripped = t |> Nx.as_type({:s, 32}) |> Nx.as_type(type)
          assert Nx.to_number(round_tripped) == v
        end
      end
    end
  end
end
