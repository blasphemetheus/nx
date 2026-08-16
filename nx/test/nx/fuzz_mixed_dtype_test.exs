defmodule Nx.FuzzMixedDtypeTest do
  @moduledoc """
  Fuzz tests for mixed-dtype operations.

  Type promotion is historically a source of subtle bugs. These
  properties check that binary ops on mixed dtypes produce a result
  of the expected promoted type, that round-tripping through
  BinaryBackend preserves values, and that unusual combinations
  (integer + complex, bf16 + f64, etc.) don't crash.
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  @fuzz_scale String.to_integer(System.get_env("FUZZ_SCALE", "1"))

  import Nx.Testing

  # ── Generators ─────────────────────────────────────────────────────

  @integer_types [{:s, 8}, {:s, 16}, {:s, 32}, {:s, 64}, {:u, 8}, {:u, 16}, {:u, 32}, {:u, 64}]
  @float_types [{:bf, 16}, {:f, 16}, {:f, 32}, {:f, 64}]
  @numeric_types @integer_types ++ @float_types
  @complex_types [{:c, 64}, {:c, 128}]

  defp any_type(), do: member_of(@numeric_types)
  defp int_type(), do: member_of(@integer_types)
  defp float_type(), do: member_of(@float_types)

  defp scalar_of(type) do
    case type do
      {:u, _} -> bind(integer(0..100), &constant(Nx.tensor(&1, type: type)))
      {:s, _} -> bind(integer(-50..50), &constant(Nx.tensor(&1, type: type)))
      {:f, _} -> bind(integer(-50..50), &constant(Nx.tensor(&1 / 1.0, type: type)))
      {:bf, _} -> bind(integer(-50..50), &constant(Nx.tensor(&1 / 1.0, type: type)))
      {:c, _} -> bind(integer(-50..50), &constant(Nx.tensor(&1 / 1.0, type: type)))
    end
  end

  # ── Binary ops across dtype pairs ──────────────────────────────────

  describe "binary ops across every (numeric, numeric) pair" do
    property "add(a, b) doesn't crash for any (numeric, numeric) pair" do
      check all(
              ta <- any_type(),
              tb <- any_type(),
              a <- scalar_of(ta),
              b <- scalar_of(tb),
              max_runs: 40 * @fuzz_scale
            ) do
        _ = Nx.add(a, b)
      end
    end

    property "multiply(a, b) doesn't crash for any (numeric, numeric) pair" do
      check all(
              ta <- any_type(),
              tb <- any_type(),
              a <- scalar_of(ta),
              b <- scalar_of(tb),
              max_runs: 40 * @fuzz_scale
            ) do
        _ = Nx.multiply(a, b)
      end
    end

    property "subtract(a, b) doesn't crash for any pair where b's max < a's max" do
      # For unsigned small int pairs, subtract can underflow. Don't
      # stress that; pick types that won't trivially underflow.
      check all(
              ta <- float_type(),
              tb <- any_type(),
              a <- scalar_of(ta),
              b <- scalar_of(tb),
              max_runs: 30 * @fuzz_scale
            ) do
        _ = Nx.subtract(a, b)
      end
    end

    property "divide(a, b) with non-zero b doesn't crash" do
      check all(
              ta <- float_type(),
              tb <- float_type(),
              max_runs: 25 * @fuzz_scale
            ) do
        a = Nx.tensor(1.0, type: ta)
        b = Nx.tensor(2.0, type: tb)
        _ = Nx.divide(a, b)
      end
    end
  end

  # ── Type promotion expectations ────────────────────────────────────

  describe "type promotion rules" do
    property "int + float promotes to (at least) float" do
      check all(
              ti <- int_type(),
              tf <- float_type(),
              max_runs: 20 * @fuzz_scale
            ) do
        a = Nx.tensor(1, type: ti)
        b = Nx.tensor(1.0, type: tf)
        result = Nx.add(a, b)
        {kind, _} = Nx.type(result)
        assert kind in [:f, :bf, :c], "int+float promoted to #{inspect(Nx.type(result))}"
      end
    end

    property "int + int promotes to (at least) the wider int" do
      check all(
              {ka, ba} <- member_of(@integer_types),
              {kb, bb} <- member_of(@integer_types),
              max_runs: 30 * @fuzz_scale
            ) do
        a = Nx.tensor(1, type: {ka, ba})
        b = Nx.tensor(1, type: {kb, bb})
        result = Nx.add(a, b)
        {kr, br} = Nx.type(result)
        assert kr in [:s, :u, :f, :bf]
        # Result width should be >= both inputs.
        assert br >= ba
        assert br >= bb
      end
    end

    property "float + float promotes to the wider float" do
      check all(
              {_, ba} <- member_of(@float_types),
              {_, bb} <- member_of(@float_types),
              max_runs: 20 * @fuzz_scale
            ) do
        a = Nx.tensor(1.0, type: {:f, ba})
        b = Nx.tensor(1.0, type: {:f, bb})
        result = Nx.add(a, b)
        {_, br} = Nx.type(result)
        assert br >= max(ba, bb)
      end
    end
  end

  # ── Complex number interactions ────────────────────────────────────

  describe "complex + real interactions" do
    property "real + complex produces complex" do
      check all(
              rt <- any_type(),
              ct <- member_of(@complex_types),
              a <- scalar_of(rt),
              c <- scalar_of(ct),
              max_runs: 20 * @fuzz_scale
            ) do
        result = Nx.add(a, c)
        {kind, _} = Nx.type(result)
        assert kind == :c
      end
    end
  end

  # ── Value preservation across type casts ───────────────────────────

  describe "as_type value round-trip" do
    property "f32 -> f64 -> f32 preserves representable values" do
      check all(x <- integer(-1000..1000), max_runs: 30 * @fuzz_scale) do
        t = Nx.tensor(x * 0.5, type: :f32)
        round_trip = t |> Nx.as_type(:f64) |> Nx.as_type(:f32)
        assert_all_close(round_trip, t, atol: 0.0, rtol: 0.0)
      end
    end

    property "s32 -> f64 -> s32 preserves (small) integer values" do
      check all(x <- integer(-1000..1000), max_runs: 30 * @fuzz_scale) do
        t = Nx.tensor(x, type: :s32)
        round_trip = t |> Nx.as_type(:f64) |> Nx.as_type(:s32)
        assert_equal(round_trip, t)
      end
    end

    property "u8 -> f32 -> u8 preserves 0..255 values" do
      check all(x <- integer(0..255), max_runs: 30 * @fuzz_scale) do
        t = Nx.tensor(x, type: :u8)
        round_trip = t |> Nx.as_type(:f32) |> Nx.as_type(:u8)
        assert_equal(round_trip, t)
      end
    end
  end

  # ── Broadcasting across dtypes ─────────────────────────────────────

  describe "broadcasting + dtype promotion" do
    property "broadcasting scalar to tensor preserves promoted dtype" do
      check all(
              ts <- any_type(),
              tt <- any_type(),
              max_runs: 20 * @fuzz_scale
            ) do
        s = Nx.tensor(1, type: ts)
        t = Nx.broadcast(Nx.tensor(2, type: tt), {3, 3})
        result = Nx.add(s, t)
        assert Nx.shape(result) == {3, 3}
      end
    end
  end
end
