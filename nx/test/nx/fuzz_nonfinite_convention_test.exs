defmodule Nx.FuzzNonfiniteConventionTest do
  @moduledoc """
  Convention-consistency fuzz for NaN/Inf through order-statistic ops.

  Nx's NaN conventions (established empirically 2026-08-17):

    * min/max/reduce_min/reduce_max/cumulative_min/cumulative_max
      PROPAGATE NaN
    * argmin/argmax point at the NaN — i.e. at what reduce_min/reduce_max
      return; the pairs are mutually consistent
    * sort treats NaN as the LARGEST value (last ascending, first
      descending), with -Inf < finite < +Inf < NaN
    * median follows the sort-based definition under that order

  The oracle is consistency: related ops must agree with each other and
  with an Elixir reference that implements the same convention. The one
  known violation is clip — pinned as [BUG-CLIP-NONFINITE].
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  @fuzz_scale String.to_integer(System.get_env("FUZZ_SCALE", "1"))

  @types [{:f, 32}, {:f, 64}]

  # total order implementing Nx's sort convention
  defp conv_key(:neg_infinity), do: {0, 0}
  defp conv_key(:nan), do: {3, 0}
  defp conv_key(:infinity), do: {2, 0}
  defp conv_key(x) when is_number(x), do: {1, x}

  defp conv_sort(values), do: Enum.sort_by(values, &conv_key/1)

  defp nonfinite_vector(type) do
    bind(integer(2..10), fn n ->
      FuzzGen.bit_tensor(constant({n}), type)
    end)
  end

  for type <- @types do
    describe "NaN propagation and arg-consistency #{inspect(type)}" do
      property "min/max propagate NaN elementwise" do
        check all(
                t <- nonfinite_vector(unquote(Macro.escape(type))),
                u <- nonfinite_vector(unquote(Macro.escape(type))),
                max_runs: 40 * @fuzz_scale
              ) do
          u = Nx.broadcast(Nx.take(u, Nx.tensor(0)), Nx.shape(t))

          either_nan = Nx.logical_or(Nx.is_nan(t), Nx.is_nan(u))

          for op <- [:min, :max] do
            result_nan = Nx.is_nan(apply(Nx, op, [t, u]))
            assert Nx.to_flat_list(result_nan) == Nx.to_flat_list(either_nan), "#{op}"
          end
        end
      end

      property "reduce_min/reduce_max return NaN iff a NaN is present" do
        check all(t <- nonfinite_vector(unquote(Macro.escape(type))), max_runs: 40 * @fuzz_scale) do
          any_nan = Nx.to_number(Nx.any(Nx.is_nan(t)))

          for op <- [:reduce_min, :reduce_max] do
            assert Nx.to_number(Nx.is_nan(apply(Nx, op, [t]))) == any_nan, "#{op}"
          end
        end
      end

      property "argmin/argmax point at what reduce_min/reduce_max return" do
        check all(t <- nonfinite_vector(unquote(Macro.escape(type))), max_runs: 40 * @fuzz_scale) do
          for {arg_op, red_op} <- [{:argmin, :reduce_min}, {:argmax, :reduce_max}] do
            idx = Nx.to_number(apply(Nx, arg_op, [t]))
            picked = Nx.to_flat_list(t) |> Enum.at(idx)
            reduced = Nx.to_flat_list(apply(Nx, red_op, [t])) |> hd()

            # both NaN, or equal values
            assert (picked == :nan and reduced == :nan) or picked == reduced,
                   "#{arg_op}: picked #{inspect(picked)}, #{red_op} #{inspect(reduced)}"
          end
        end
      end

      property "cumulative_min/max poison everything after the first NaN" do
        check all(t <- nonfinite_vector(unquote(Macro.escape(type))), max_runs: 40 * @fuzz_scale) do
          values = Nx.to_flat_list(t)

          case Enum.find_index(values, &(&1 == :nan)) do
            nil ->
              :ok

            first_nan ->
              for op <- [:cumulative_min, :cumulative_max] do
                result = Nx.to_flat_list(apply(Nx, op, [t]))

                assert result |> Enum.drop(first_nan) |> Enum.all?(&(&1 == :nan)),
                       "#{op}: not poisoned after index #{first_nan}"
              end
          end
        end
      end
    end

    describe "sort order convention #{inspect(type)}" do
      property "sort ascending follows -Inf < finite < +Inf < NaN" do
        check all(t <- nonfinite_vector(unquote(Macro.escape(type))), max_runs: 40 * @fuzz_scale) do
          values = Nx.to_flat_list(t)
          sorted = Nx.to_flat_list(Nx.sort(t, axis: 0))

          assert sorted == conv_sort(values)
        end
      end

      property "sort descending is the exact reverse of ascending" do
        check all(t <- nonfinite_vector(unquote(Macro.escape(type))), max_runs: 40 * @fuzz_scale) do
          asc = Nx.to_flat_list(Nx.sort(t, axis: 0))
          desc = Nx.to_flat_list(Nx.sort(t, axis: 0, direction: :desc))

          assert desc == Enum.reverse(asc)
        end
      end
    end

    describe "median follows the sort-based definition #{inspect(type)}" do
      property "median equals the reference under the sort convention" do
        check all(t <- nonfinite_vector(unquote(Macro.escape(type))), max_runs: 40 * @fuzz_scale) do
          values = Nx.to_flat_list(t)
          sorted = conv_sort(values)
          n = length(sorted)

          expected =
            if rem(n, 2) == 1 do
              Enum.at(sorted, div(n, 2))
            else
              a = Enum.at(sorted, div(n, 2) - 1)
              b = Enum.at(sorted, div(n, 2))

              cond do
                a == :nan or b == :nan -> :nan
                a == :infinity and b == :infinity -> :infinity
                a == :neg_infinity and b == :neg_infinity -> :neg_infinity
                a == :neg_infinity and b == :infinity -> :nan
                a == :neg_infinity -> :neg_infinity
                b == :infinity -> :infinity
                true -> (a + b) / 2
              end
            end

          result = Nx.to_flat_list(Nx.median(t)) |> hd()

          # the even-count mean is computed at tensor precision; tolerance
          # must be relative to magnitude (half-ulp at f32 is ~6e-8 rel)
          rtol = if unquote(Macro.escape(type)) == {:f, 32}, do: 1.0e-6, else: 1.0e-12

          case {result, expected} do
            {r, e} when is_number(r) and is_number(e) ->
              assert_in_delta r, e, max(abs(e) * rtol, 1.0e-9)

            {r, e} ->
              assert r == e
          end
        end
      end
    end
  end

  describe "comparison truth table over non-finites" do
    # Coverage-guided: the BinaryBackend comparison clauses for
    # finite-vs-Inf and NaN arms were dark under the whole test suite.
    # IEEE reference: NaN compares false to everything (not_equal true);
    # -Inf < finite < +Inf.
    defp cmp_key(:neg_infinity), do: {0, 0}
    defp cmp_key(:infinity), do: {2, 0}
    defp cmp_key(x), do: {1, x}

    defp ref_cmp(op, a, b) do
      if a == :nan or b == :nan do
        if op == :not_equal, do: 1, else: 0
      else
        result =
          case op do
            :equal -> a == b
            :not_equal -> a != b
            :greater -> cmp_key(a) > cmp_key(b)
            :less -> cmp_key(a) < cmp_key(b)
            :greater_equal -> cmp_key(a) >= cmp_key(b)
            :less_equal -> cmp_key(a) <= cmp_key(b)
          end

        if result, do: 1, else: 0
      end
    end

    property "all six comparisons match the IEEE reference" do
      specials = [:nan, :infinity, :neg_infinity]

      check all(
              a <- one_of([float(min: -100.0, max: 100.0), member_of(specials)]),
              b <- one_of([float(min: -100.0, max: 100.0), member_of(specials)]),
              max_runs: 60 * @fuzz_scale
            ) do
        ta = Nx.tensor(a, type: {:f, 64})
        tb = Nx.tensor(b, type: {:f, 64})

        for op <- [:equal, :not_equal, :greater, :less, :greater_equal, :less_equal] do
          assert Nx.to_number(apply(Nx, op, [ta, tb])) == ref_cmp(op, a, b),
                 "#{op}(#{inspect(a)}, #{inspect(b)})"
        end
      end
    end

    test "exhaustive special-pair matrix" do
      values = [:nan, :infinity, :neg_infinity, -1.0, 0.0, 1.0]

      for a <- values,
          b <- values,
          op <- [:equal, :not_equal, :greater, :less, :greater_equal, :less_equal] do
        ta = Nx.tensor(a, type: {:f, 64})
        tb = Nx.tensor(b, type: {:f, 64})

        assert Nx.to_number(apply(Nx, op, [ta, tb])) == ref_cmp(op, a, b),
               "#{op}(#{inspect(a)}, #{inspect(b)})"
      end
    end
  end

  describe "known bugs: clip non-finite handling ([BUG-CLIP-NONFINITE])" do
    # See FUZZ_FINDINGS/clip_nonfinite_inconsistent.md. clip is definable
    # as min(max(x, lo), hi); under Nx's own NaN-propagating min/max that
    # composition returns NaN whenever any argument is NaN. Instead each
    # argument position currently does something different. Flip these to
    # NaN assertions when fixed.

    test "clip(NaN, lo, hi) returns the LOWER BOUND instead of NaN" do
      result = Nx.clip(Nx.tensor(:nan, type: {:f, 64}), 0.0, 2.0)
      assert Nx.to_flat_list(result) == [0.0]
    end

    test "clip(x, NaN, hi) returns the UPPER BOUND instead of NaN" do
      result = Nx.clip(Nx.tensor(1.0, type: {:f, 64}), Nx.tensor(:nan, type: {:f, 64}), 2.0)
      assert Nx.to_flat_list(result) == [2.0]
    end

    test "clip(x, lo, NaN) returns NaN (the only position that propagates)" do
      result = Nx.clip(Nx.tensor(1.0, type: {:f, 64}), 0.0, Nx.tensor(:nan, type: {:f, 64}))
      assert Nx.to_flat_list(result) == [:nan]
    end

    test "control: clip with finite bounds equals the min/max composition" do
      t = Nx.tensor([-5.0, 0.5, 5.0], type: {:f, 64})
      composed = Nx.min(Nx.max(t, 0.0), 2.0)
      assert Nx.clip(t, 0.0, 2.0) == composed
    end
  end
end
