defmodule Nx.FuzzRandomPropsTest do
  @moduledoc """
  Property fuzz for `Nx.Random` (FUZZ_ROADMAP T1.3). Zero coverage existed.

  Everything here is deterministic given the StreamData-generated seed, so
  there is no statistical flakiness beyond the (generous) moment tolerances:
  with n = 2048 samples the tolerances sit at ~7 standard errors.

  Failure modes these guard against: key reuse / stream correlation between
  split subkeys (silently corrupts training), non-determinism across calls
  (breaks reproducible experiments), out-of-range samples, and shuffle/choice
  losing or inventing elements.
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  @n 2048

  defp seed, do: integer(0..1_000_000)

  describe "determinism" do
    property "same seed produces bitwise-identical uniform/normal/randint streams" do
      check all(s <- seed(), max_runs: 25) do
        for fun <- [
              fn k -> Nx.Random.uniform(k, shape: {16}) |> elem(0) end,
              fn k -> Nx.Random.normal(k, shape: {16}) |> elem(0) end,
              fn k -> Nx.Random.randint(k, 0, 1000, shape: {16}) |> elem(0) end
            ] do
          a = fun.(Nx.Random.key(s))
          b = fun.(Nx.Random.key(s))
          assert Nx.to_binary(a) == Nx.to_binary(b)
        end
      end
    end

    property "different seeds produce different streams" do
      check all(s <- seed(), delta <- integer(1..1000), max_runs: 25) do
        {a, _} = Nx.Random.uniform(Nx.Random.key(s), shape: {64})
        {b, _} = Nx.Random.uniform(Nx.Random.key(s + delta), shape: {64})
        refute Nx.to_binary(a) == Nx.to_binary(b)
      end
    end

    property "the returned new_key differs from the input key and advances the stream" do
      check all(s <- seed(), max_runs: 25) do
        key = Nx.Random.key(s)
        {a, key2} = Nx.Random.uniform(key, shape: {64})
        refute Nx.to_binary(key2) == Nx.to_binary(key)

        {b, _key3} = Nx.Random.uniform(key2, shape: {64})
        refute Nx.to_binary(a) == Nx.to_binary(b)
      end
    end
  end

  describe "split and fold_in" do
    property "split produces pairwise-distinct subkeys with pairwise-distinct streams" do
      check all(s <- seed(), parts <- integer(2..5), max_runs: 20) do
        keys = Nx.Random.split(Nx.Random.key(s), parts: parts)

        streams =
          for i <- 0..(parts - 1) do
            Nx.Random.uniform_split(keys[i], 0.0, 1.0, shape: {32}) |> Nx.to_binary()
          end

        assert length(Enum.uniq(streams)) == parts, "split subkeys produced colliding streams"
      end
    end

    property "fold_in is deterministic and distinct per folded data" do
      check all(s <- seed(), a <- integer(0..1_000_000), b <- integer(0..1_000_000), max_runs: 20) do
        key = Nx.Random.key(s)

        folded_a1 = Nx.Random.fold_in(key, a)
        folded_a2 = Nx.Random.fold_in(key, a)
        assert Nx.to_binary(folded_a1) == Nx.to_binary(folded_a2)

        if a != b do
          folded_b = Nx.Random.fold_in(key, b)

          {sa, _} = Nx.Random.uniform(folded_a1, shape: {32})
          {sb, _} = Nx.Random.uniform(folded_b, shape: {32})
          refute Nx.to_binary(sa) == Nx.to_binary(sb)
        end
      end
    end
  end

  describe "range and shape contracts" do
    property "uniform(min, max) stays within bounds; shape and type honored" do
      check all(
              s <- seed(),
              min_v <- float(min: -100.0, max: 99.0),
              width <- float(min: 0.001, max: 100.0),
              type <- member_of([{:f, 32}, {:f, 64}]),
              max_runs: 25
            ) do
        max_v = min_v + width

        {t, _} =
          Nx.Random.uniform(Nx.Random.key(s), min_v, max_v, shape: {4, 8}, type: type)

        assert Nx.shape(t) == {4, 8}
        assert Nx.type(t) == type

        assert Nx.to_number(Nx.all(Nx.greater_equal(t, min_v))) == 1
        assert Nx.to_number(Nx.all(Nx.less_equal(t, max_v))) == 1
      end
    end

    property "randint(min, max) stays within bounds with integer type" do
      check all(
              s <- seed(),
              min_v <- integer(-1000..999),
              width <- integer(1..1000),
              type <- member_of([{:s, 32}, {:s, 64}, {:u, 32}]),
              max_runs: 25
            ) do
        {min_v, max_v} =
          case type do
            {:u, _} -> {abs(min_v), abs(min_v) + width}
            _ -> {min_v, min_v + width}
          end

        {t, _} =
          Nx.Random.randint(Nx.Random.key(s), min_v, max_v, shape: {32}, type: type)

        assert Nx.type(t) == type
        assert Nx.to_number(Nx.all(Nx.greater_equal(t, min_v))) == 1
        assert Nx.to_number(Nx.all(Nx.less_equal(t, max_v))) == 1
      end
    end
  end

  describe "moments (n = #{@n}, tolerances ~7 standard errors)" do
    property "uniform(0, 1) has mean ~0.5 and variance ~1/12" do
      check all(s <- seed(), max_runs: 15) do
        {t, _} = Nx.Random.uniform(Nx.Random.key(s), shape: {@n})

        mean = Nx.to_number(Nx.mean(t))
        # std error of mean = 1/sqrt(12n) ~ 0.0064; 7 SE ~ 0.045
        assert abs(mean - 0.5) < 0.045

        var = Nx.to_number(Nx.variance(t))
        assert abs(var - 1 / 12) < 0.02
      end
    end

    property "normal(mu, sigma) has mean ~mu and std ~sigma" do
      check all(
              s <- seed(),
              mu <- float(min: -10.0, max: 10.0),
              sigma <- float(min: 0.1, max: 5.0),
              max_runs: 15
            ) do
        {t, _} = Nx.Random.normal(Nx.Random.key(s), mu, sigma, shape: {@n})

        mean = Nx.to_number(Nx.mean(t))
        # std error of mean = sigma/sqrt(n) ~ sigma/45; allow ~7 SE
        assert abs(mean - mu) < 0.16 * sigma

        std = Nx.to_number(Nx.standard_deviation(t))
        assert abs(std - sigma) < 0.15 * sigma
      end
    end
  end

  describe "shuffle and choice" do
    property "shuffle is a permutation, deterministic per key, not the identity" do
      check all(s <- seed(), n <- integer(16..64), max_runs: 20) do
        t = Nx.iota({n}, type: {:f, 32})
        key = Nx.Random.key(s)

        {shuffled, _} = Nx.Random.shuffle(key, t)

        # permutation: same multiset
        assert Enum.sort(Nx.to_flat_list(shuffled)) == Nx.to_flat_list(t)

        # deterministic
        {shuffled2, _} = Nx.Random.shuffle(key, t)
        assert Nx.to_binary(shuffled) == Nx.to_binary(shuffled2)

        # identity permutation has probability 1/n! <= 1/16! ~ 5e-14
        refute Nx.to_binary(shuffled) == Nx.to_binary(t)
      end
    end

    property "choice samples only from the population, honors sample count" do
      check all(s <- seed(), n <- integer(4..32), k <- integer(1..8), max_runs: 20) do
        # distinct population values so membership is meaningful
        population = Nx.multiply(Nx.iota({n}, type: {:f, 32}), 3.0)

        {picked, _} = Nx.Random.choice(Nx.Random.key(s), population, samples: k)

        assert Nx.axis_size(picked, 0) == k

        pop = MapSet.new(Nx.to_flat_list(population))

        for v <- Nx.to_flat_list(picked) do
          assert MapSet.member?(pop, v), "choice invented value #{v}"
        end
      end
    end
  end

  describe "vectorized keys" do
    property "a vectorized batch of split keys yields pairwise-distinct batch streams" do
      check all(s <- seed(), parts <- integer(2..4), max_runs: 15) do
        keys = Nx.Random.split(Nx.Random.key(s), parts: parts)
        vkeys = Nx.vectorize(keys, :batch)

        sample = Nx.Random.uniform_split(vkeys, 0.0, 1.0, shape: {16})

        assert Keyword.keys(sample.vectorized_axes) == [:batch]

        devec = Nx.devectorize(sample, keep_names: false)

        streams =
          for i <- 0..(parts - 1) do
            devec |> Nx.slice_along_axis(i, 1, axis: 0) |> Nx.to_binary()
          end

        assert length(Enum.uniq(streams)) == parts
      end
    end
  end
end
