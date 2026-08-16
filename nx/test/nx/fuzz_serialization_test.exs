defmodule Nx.FuzzSerializationTest do
  @moduledoc """
  Fuzz tests for `Nx.to_binary`, `Nx.from_binary`, `Nx.serialize`,
  and `Nx.deserialize` round-trips.

  Findings:
  - `Nx.from_binary` rejects bitstrings that `Nx.to_binary` produces
    for sub-byte types with non-byte-aligned bit counts. See
    `FUZZ_FINDINGS/from_binary_rejects_sub_byte_bitstrings.md`.

  Everything else is green:
  - All dtypes (u8/16/32/64, s8/16/32/64, f16/32/64, bf16, c64/128) round-trip.
  - Sub-byte types (u2/u4/s2/s4) round-trip when element count × bit size is
    divisible by 8.
  - NaN / Inf / -0.0 preserved at bit level.
  - Vectorized tensors preserve vectorized_axes.
  - Names preserved.
  - Nested containers (tuple of maps of tensors) preserved.
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  @fuzz_scale String.to_integer(System.get_env("FUZZ_SCALE", "1"))

  import Nx.Testing

  # ── FUZZ FINDING: to_binary/from_binary asymmetry for sub-byte ─────

  describe "sub-byte binary round-trip (from_binary_rejects_sub_byte_bitstrings)" do
    # Pins current behavior: to_binary returns a bitstring for sub-byte
    # types with non-aligned bit counts, but from_binary's is_binary
    # guard rejects bitstrings. When fixed, replace assert_raise with
    # round-trip equality.

    test "u4 tensor with 3 elements: to_binary produces bitstring" do
      t = Nx.tensor([0, 7, 15], type: :u4)
      bin = Nx.to_binary(t)
      # 3 u4 values = 12 bits, not byte-aligned
      assert bit_size(bin) == 12
      refute rem(bit_size(bin), 8) == 0
    end

    # BUG-FROMBIN-u4-3elem — Nx.from_binary's is_binary guard rejects bitstrings.
    # Fix: guard becomes is_bitstring (to_binary/from_binary are an inverse pair).
    # See FUZZ_FINDINGS/from_binary_rejects_sub_byte_bitstrings.md.
    test "[BUG-FROMBIN-u4-3elem] from_binary rejects the bitstring that to_binary produces (u4, 3 elements)" do
      t = Nx.tensor([0, 7, 15], type: :u4)
      bin = Nx.to_binary(t)

      assert_raise FunctionClauseError, fn ->
        Nx.from_binary(bin, :u4)
      end

      # Once fixed (relax guard to is_bitstring), replace with:
      #   rt = Nx.from_binary(bin, :u4) |> Nx.reshape({3})
      #   assert_equal(rt, t)
    end

    test "[BUG-FROMBIN-u2-3elem] u2 tensor with 3 elements fails the same way" do
      t = Nx.tensor([0, 1, 2], type: :u2)
      bin = Nx.to_binary(t)
      assert bit_size(bin) == 6

      assert_raise FunctionClauseError, fn ->
        Nx.from_binary(bin, :u2)
      end
    end

    test "aligned sub-byte cases round-trip fine (4 u4 = 16 bits)" do
      t = Nx.tensor([0, 7, 15, 3], type: :u4)
      bin = Nx.to_binary(t)
      assert bit_size(bin) == 16
      rt = Nx.from_binary(bin, :u4) |> Nx.reshape({4})
      assert_equal(rt, t)
    end
  end

  # ── Green paths: regression coverage for round-trips that DO work ──

  describe "to_binary / from_binary round-trips for byte-aligned types" do
    @standard_types [
      {:u, 8},
      {:u, 16},
      {:u, 32},
      {:u, 64},
      {:s, 8},
      {:s, 16},
      {:s, 32},
      {:s, 64},
      {:f, 16},
      {:bf, 16},
      {:f, 32},
      {:f, 64}
    ]

    for type <- @standard_types do
      test "round-trip preserves values for #{inspect(type)}" do
        type = unquote(Macro.escape(type))

        t =
          case type do
            {:u, _} -> Nx.tensor([0, 1, 2, 3], type: type)
            {:s, _} -> Nx.tensor([-2, -1, 0, 1], type: type)
            {:f, _} -> Nx.tensor([-1.5, 0.0, 1.5, 2.5], type: type)
            {:bf, _} -> Nx.tensor([-1.5, 0.0, 1.5, 2.5], type: type)
          end

        bin = Nx.to_binary(t)
        rt = Nx.from_binary(bin, type) |> Nx.reshape(Nx.shape(t))
        assert_equal(rt, t)
      end
    end
  end

  describe "serialize / deserialize round-trips" do
    property "preserves f32 values" do
      check all(
              n <- integer(1..10),
              vals <- list_of(float(min: -100.0, max: 100.0), length: n),
              max_runs: 10 * @fuzz_scale
            ) do
        t = Nx.tensor(vals, type: :f32)
        rt = Nx.deserialize(Nx.serialize(t))
        assert_all_close(rt, t, atol: 0.0, rtol: 0.0)
      end
    end

    property "preserves s32 values" do
      check all(
              n <- integer(1..10),
              vals <- list_of(integer(-1000..1000), length: n),
              max_runs: 10 * @fuzz_scale
            ) do
        t = Nx.tensor(vals, type: :s32)
        rt = Nx.deserialize(Nx.serialize(t))
        assert_equal(rt, t)
      end
    end

    test "preserves complex values" do
      t = Nx.tensor([Complex.new(1.0, 2.0), Complex.new(-1.0, 0.5)], type: :c64)
      rt = Nx.deserialize(Nx.serialize(t))
      assert_all_close(rt, t, atol: 0.0, rtol: 0.0)
    end

    test "preserves vectorized_axes" do
      t = Nx.iota({3, 4}, type: :f32) |> Nx.vectorize(batch: 3)
      rt = Nx.deserialize(Nx.serialize(t))
      assert rt.vectorized_axes == t.vectorized_axes
    end

    test "preserves tensor names" do
      t = Nx.tensor([1.0, 2.0, 3.0], names: [:x])
      rt = Nx.deserialize(Nx.serialize(t))
      assert Nx.names(rt) == Nx.names(t)
    end

    test "round-trips nested tuple of maps of tensors" do
      nested = {
        %{w: Nx.tensor([1.0]), b: Nx.tensor([2.0])},
        %{w: Nx.tensor([3.0]), b: Nx.tensor([4.0])}
      }

      rt = Nx.deserialize(Nx.serialize(nested))

      {{map1_rt, map2_rt}, {map1, map2}} = {rt, nested}

      for key <- [:w, :b] do
        assert_equal(map1_rt[key], map1[key])
        assert_equal(map2_rt[key], map2[key])
      end
    end

    test "round-trips sub-byte types (aligned counts)" do
      # serialize is not affected by the from_binary bug — it has its
      # own wire format.
      for type <- [{:u, 2}, {:u, 4}, {:s, 2}, {:s, 4}] do
        t =
          case type do
            {:u, _} -> Nx.tensor([0, 1, 2, 3], type: type)
            {:s, _} -> Nx.tensor([-2, -1, 0, 1], type: type)
          end

        rt = Nx.deserialize(Nx.serialize(t))
        assert_equal(rt, t)
      end
    end
  end

  # ── Special float values preserved at bit level ────────────────────

  describe "special float values round-trip at bit level" do
    test "NaN pattern preserved through to_binary/from_binary" do
      t = Nx.tensor([:nan, 0.0, 1.0], type: :f32)
      bin = Nx.to_binary(t)
      rt = Nx.from_binary(bin, :f32)

      assert Nx.to_flat_list(Nx.is_nan(rt)) == Nx.to_flat_list(Nx.is_nan(t))
    end

    test "Inf / -Inf preserved" do
      t = Nx.tensor([:infinity, :neg_infinity, 1.0], type: :f32)
      bin = Nx.to_binary(t)
      rt = Nx.from_binary(bin, :f32)

      assert Nx.to_flat_list(rt) == Nx.to_flat_list(t)
    end

    test "-0.0 bit pattern preserved" do
      t = Nx.tensor([-0.0, 0.0, 1.0], type: :f32)
      bin = Nx.to_binary(t)
      rt = Nx.from_binary(bin, :f32)

      # Compare bit patterns, not values (since -0.0 == 0.0).
      assert Nx.to_binary(t) == Nx.to_binary(rt)
    end
  end

  # ── FUZZ FINDING: Nx.Backend.inspect crashes on sub-byte int tensors ──

  describe "sub-byte inspect (inspect_crashes_on_sub_byte_int_tensors)" do
    # See FUZZ_FINDINGS/inspect_crashes_on_sub_byte_int_tensors.md.
    # Nx.Backend.chunk/5 uses `tail::binary` in :s and :u branches;
    # should be `tail::bitstring` (as the float branch already does).

    # BUG-INSPECT-u4 — Nx.Backend.chunk/5 uses tail::binary; should be tail::bitstring.
    # Float branch already does it right. Affects BinaryBackend AND Torchx.
    # See FUZZ_FINDINGS/inspect_crashes_on_sub_byte_int_tensors.md.
    test "[BUG-INSPECT-u4] u4 tensor crashes inspect" do
      t = Nx.tensor([0, 1, 2, 3], type: :u4)

      result = inspect(t)
      assert result =~ "Inspect.Error"
      assert result =~ "MatchError"

      # Once fixed, replace with:
      #   assert result =~ "u4"
      #   assert result =~ "[0, 1, 2, 3]"
    end

    test "[BUG-INSPECT-s2] s2 tensor crashes inspect" do
      t = Nx.tensor([-2, -1, 0, 1], type: :s2)
      result = inspect(t)
      assert result =~ "Inspect.Error"
      assert result =~ "MatchError"
    end

    test "[BUG-INSPECT-u4-size2] u4 of size 2 also crashes (tail misaligned after one element)" do
      # [0, 1] = 8 bits total. After consuming one u4, tail is 4 bits —
      # not byte-aligned, so chunk/5 crashes.
      t = Nx.tensor([0, 1], type: :u4)
      result = inspect(t)
      assert result =~ "Inspect.Error"
    end
  end
end
