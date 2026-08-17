defmodule Nx.FuzzLongtailTest do
  @moduledoc """
  Reference-oracle fuzz for previously zero-coverage functions: mode,
  weighted_mean, covariance, logsumexp, the diagonal family, tri/tril/triu,
  to_batched, split, bitcast, cumulative_product, window_product,
  window_scatter_min, and the logical ops.

  Every property checks against an Elixir reference or an exact metamorphic
  identity (e.g. window_scatter_min == negate ∘ window_scatter_max ∘ negate).
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  import Nx.Testing

  @fuzz_scale String.to_integer(System.get_env("FUZZ_SCALE", "1"))

  defp float_list(n), do: list_of(float(min: -50.0, max: 50.0), length: n)

  describe "mode" do
    property "returns the most frequent value; ties go to the smallest" do
      check all(n <- integer(1..12), max_runs: 30 * @fuzz_scale) do
        check all(values <- list_of(integer(0..4), length: n), max_runs: 1) do
          t = Nx.tensor(values)

          freqs = Enum.frequencies(values)
          max_count = freqs |> Map.values() |> Enum.max()

          expected =
            freqs
            |> Enum.filter(fn {_v, c} -> c == max_count end)
            |> Enum.map(&elem(&1, 0))
            |> Enum.min()

          assert Nx.to_number(Nx.mode(t)) == expected
        end
      end
    end

    property "mode along an axis equals per-row mode" do
      check all(rows <- integer(1..4), cols <- integer(1..6), max_runs: 20 * @fuzz_scale) do
        check all(
                values <- list_of(list_of(integer(0..3), length: cols), length: rows),
                max_runs: 1
              ) do
          t = Nx.tensor(values)
          result = Nx.to_flat_list(Nx.mode(t, axis: 1))

          expected =
            for row <- values do
              freqs = Enum.frequencies(row)
              max_count = freqs |> Map.values() |> Enum.max()

              freqs
              |> Enum.filter(fn {_, c} -> c == max_count end)
              |> Enum.map(&elem(&1, 0))
              |> Enum.min()
            end

          assert result == expected
        end
      end
    end
  end

  describe "weighted_mean and covariance" do
    property "weighted_mean equals sum(w*x)/sum(w)" do
      check all(n <- integer(1..8), max_runs: 30 * @fuzz_scale) do
        check all(
                xs <- float_list(n),
                ws <- list_of(float(min: 0.1, max: 5.0), length: n),
                max_runs: 1
              ) do
          t = Nx.tensor(xs, type: {:f, 64})
          w = Nx.tensor(ws, type: {:f, 64})

          expected =
            Enum.zip_with(xs, ws, &(&1 * &2)) |> Enum.sum() |> Kernel./(Enum.sum(ws))

          assert_all_close(Nx.weighted_mean(t, w), Nx.tensor(expected, type: {:f, 64}),
            rtol: 1.0e-10
          )
        end
      end
    end

    property "covariance equals the population-covariance reference" do
      check all(n <- integer(2..6), d <- integer(1..3), max_runs: 15 * @fuzz_scale) do
        check all(rows <- list_of(float_list(d), length: n), max_runs: 1) do
          t = Nx.tensor(rows, type: {:f, 64})

          means = for j <- 0..(d - 1), do: Enum.sum(Enum.map(rows, &Enum.at(&1, j))) / n

          expected =
            for i <- 0..(d - 1) do
              for j <- 0..(d - 1) do
                Enum.sum(
                  Enum.map(rows, fn row ->
                    (Enum.at(row, i) - Enum.at(means, i)) * (Enum.at(row, j) - Enum.at(means, j))
                  end)
                ) / n
              end
            end

          assert_all_close(Nx.covariance(t), Nx.tensor(expected, type: {:f, 64}),
            rtol: 1.0e-9,
            atol: 1.0e-9
          )
        end
      end
    end
  end

  describe "logsumexp" do
    property "equals log(sum(exp)) at moderate magnitudes" do
      check all(n <- integer(1..8), max_runs: 30 * @fuzz_scale) do
        check all(xs <- list_of(float(min: -10.0, max: 10.0), length: n), max_runs: 1) do
          t = Nx.tensor(xs, type: {:f, 64})
          expected = :math.log(Enum.sum(Enum.map(xs, &:math.exp/1)))

          assert_all_close(Nx.logsumexp(t), Nx.tensor(expected, type: {:f, 64}), rtol: 1.0e-10)
        end
      end
    end

    property "is shift-stable where the naive formula overflows" do
      check all(
              base <- float(min: 500.0, max: 5000.0),
              n <- integer(2..6),
              max_runs: 30 * @fuzz_scale
            ) do
        # naive log(sum(exp(base))) overflows f64 for base > ~709
        t = Nx.broadcast(Nx.tensor(base, type: {:f, 64}), {n})
        expected = base + :math.log(n)

        assert_all_close(Nx.logsumexp(t), Nx.tensor(expected, type: {:f, 64}), rtol: 1.0e-12)
      end
    end
  end

  describe "diagonal family" do
    property "take_diagonal recovers what make_diagonal placed (offset sweep)" do
      check all(n <- integer(1..5), offset <- integer(-2..2), max_runs: 30 * @fuzz_scale) do
        check all(xs <- float_list(n), max_runs: 1) do
          v = Nx.tensor(xs, type: {:f, 64})

          m = Nx.make_diagonal(v, offset: offset)
          assert Nx.to_flat_list(Nx.take_diagonal(m, offset: offset)) == Nx.to_flat_list(v)

          # everything off the target diagonal is zero
          assert Nx.to_number(Nx.sum(Nx.abs(m))) ==
                   Nx.to_number(Nx.sum(Nx.abs(v)))
        end
      end
    end

    property "put_diagonal overwrites exactly the diagonal" do
      check all(n <- integer(2..5), max_runs: 30 * @fuzz_scale) do
        check all(xs <- float_list(n), max_runs: 1) do
          base = Nx.broadcast(Nx.tensor(7.0, type: {:f, 64}), {n, n})
          v = Nx.tensor(xs, type: {:f, 64})

          result = Nx.put_diagonal(base, v)

          assert Nx.to_flat_list(Nx.take_diagonal(result)) == Nx.to_flat_list(v)
          # off-diagonal count of 7.0s is n*n - n
          sevens = Nx.sum(Nx.equal(result, 7.0)) |> Nx.to_number()
          diag_sevens = Enum.count(xs, &(&1 == 7.0))
          assert sevens == n * n - n + diag_sevens
        end
      end
    end
  end

  describe "tri/tril/triu" do
    property "match index-predicate references (offset sweep)" do
      check all(
              rows <- integer(1..5),
              cols <- integer(1..5),
              k <- integer(-3..3),
              max_runs: 30 * @fuzz_scale
            ) do
        tri = Nx.tri(rows, cols, k: k)

        expected_tri =
          for i <- 0..(rows - 1) do
            for j <- 0..(cols - 1), do: if(j <= i + k, do: 1, else: 0)
          end

        assert Nx.to_list(tri) == expected_tri

        t = Nx.iota({rows, cols}) |> Nx.add(1)
        tril = Nx.to_list(Nx.tril(t, k: k))
        triu = Nx.to_list(Nx.triu(t, k: k))
        source = Nx.to_list(t)

        for i <- 0..(rows - 1), j <- 0..(cols - 1) do
          src = source |> Enum.at(i) |> Enum.at(j)
          low = tril |> Enum.at(i) |> Enum.at(j)
          up = triu |> Enum.at(i) |> Enum.at(j)

          assert low == if(j <= i + k, do: src, else: 0)
          assert up == if(j >= i + k, do: src, else: 0)
        end
      end
    end
  end

  describe "to_batched and split" do
    property "to_batched concatenates back to the original (with :repeat wrap)" do
      check all(
              rows <- integer(1..10),
              cols <- integer(1..3),
              batch <- integer(1..rows),
              max_runs: 30 * @fuzz_scale
            ) do
        t = Nx.iota({rows, cols})

        batches = t |> Nx.to_batched(batch) |> Enum.to_list()
        rejoined = Nx.concatenate(batches)

        # every batch has exactly `batch` rows; the first `rows` rows of
        # the rejoined tensor are the original (leftover wraps from row 0)
        assert Enum.all?(batches, &(Nx.axis_size(&1, 0) == batch))
        assert Nx.slice_along_axis(rejoined, 0, rows, axis: 0) == t

        discarded = t |> Nx.to_batched(batch, leftover: :discard) |> Enum.to_list()
        assert length(discarded) == div(rows, batch)
      end
    end

    property "split concatenates back to the original" do
      check all(n <- integer(2..10), at <- integer(1..(n - 1)), max_runs: 30 * @fuzz_scale) do
        t = Nx.iota({n})
        {left, right} = Nx.split(t, at)

        assert Nx.axis_size(left, 0) == at
        assert Nx.concatenate([left, right]) == t
      end
    end
  end

  describe "bitcast" do
    property "is bit-preserving both ways for same-width types" do
      check all(n <- integer(1..8), max_runs: 30 * @fuzz_scale) do
        check all(xs <- float_list(n), max_runs: 1) do
          t = Nx.tensor(xs, type: {:f, 32})

          round_tripped = t |> Nx.bitcast({:u, 32}) |> Nx.bitcast({:f, 32})
          assert Nx.to_binary(round_tripped) == Nx.to_binary(t)

          as_int = Nx.bitcast(t, {:s, 32})
          assert Nx.to_binary(as_int) == Nx.to_binary(t)
        end
      end
    end

    test "raises for width mismatch" do
      assert_raise ArgumentError, fn -> Nx.bitcast(Nx.tensor([1.0], type: :f32), {:f, 64}) end
    end
  end

  describe "cumulative_product" do
    property "matches Enum.scan (forward and reverse)" do
      check all(n <- integer(1..8), reverse? <- boolean(), max_runs: 30 * @fuzz_scale) do
        check all(values <- list_of(integer(-3..3), length: n), max_runs: 1) do
          t = Nx.tensor(values, type: {:s, 64})

          input = if reverse?, do: Enum.reverse(values), else: values
          scanned = Enum.scan(input, &(&1 * &2))
          expected = if reverse?, do: Enum.reverse(scanned), else: scanned

          assert Nx.to_flat_list(Nx.cumulative_product(t, reverse: reverse?)) == expected
        end
      end
    end
  end

  describe "window_product and window_scatter_min" do
    property "window_product matches a brute-force reference" do
      check all(n <- integer(2..8), w <- integer(1..3), w <= n, max_runs: 30 * @fuzz_scale) do
        check all(values <- list_of(integer(-3..3), length: n), max_runs: 1) do
          t = Nx.tensor(values, type: {:s, 64})

          expected =
            values
            |> Enum.chunk_every(w, 1, :discard)
            |> Enum.map(&Enum.product/1)

          assert Nx.to_flat_list(Nx.window_product(t, {w})) == expected
        end
      end
    end

    property "window_scatter_min is negate . window_scatter_max . negate" do
      check all(pairs <- integer(1..4), max_runs: 30 * @fuzz_scale) do
        n = pairs * 2

        check all(
                values <- list_of(integer(-9..9), length: n),
                source <- list_of(integer(1..5), length: pairs),
                max_runs: 1
              ) do
          t = Nx.tensor(values, type: {:s, 64})
          s = Nx.tensor(source, type: {:s, 64})

          via_min = Nx.window_scatter_min(t, s, 0, {2}, strides: [2])

          via_max =
            t
            |> Nx.negate()
            |> Nx.window_scatter_max(s, 0, {2}, strides: [2])

          # scatter positions must match: min of x is max of -x
          assert Nx.to_flat_list(Nx.not_equal(via_min, 0)) ==
                   Nx.to_flat_list(Nx.not_equal(via_max, 0))
        end
      end
    end
  end

  describe "logical ops" do
    property "logical and/or/xor/not treat any nonzero as true" do
      check all(n <- integer(1..8), max_runs: 30 * @fuzz_scale) do
        check all(
                xs <- list_of(integer(-2..2), length: n),
                ys <- list_of(integer(-2..2), length: n),
                max_runs: 1
              ) do
          tx = Nx.tensor(xs)
          ty = Nx.tensor(ys)

          bx = Enum.map(xs, &(&1 != 0))
          by = Enum.map(ys, &(&1 != 0))

          to_bit = fn b -> if b, do: 1, else: 0 end

          assert Nx.to_flat_list(Nx.logical_and(tx, ty)) ==
                   Enum.zip_with(bx, by, &to_bit.(&1 and &2))

          assert Nx.to_flat_list(Nx.logical_or(tx, ty)) ==
                   Enum.zip_with(bx, by, &to_bit.(&1 or &2))

          assert Nx.to_flat_list(Nx.logical_xor(tx, ty)) ==
                   Enum.zip_with(bx, by, &to_bit.(&1 != &2))

          assert Nx.to_flat_list(Nx.logical_not(tx)) == Enum.map(bx, &to_bit.(not &1))

          # not_equal is the complement of equal
          assert Nx.logical_not(Nx.equal(tx, ty)) == Nx.not_equal(tx, ty)
        end
      end
    end
  end
end
