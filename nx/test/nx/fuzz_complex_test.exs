defmodule Nx.FuzzComplexTest do
  @moduledoc """
  Fuzz tests for complex number operations (c64, c128).

  Tests that ops handle complex inputs correctly — arithmetic,
  reductions, transcendentals, type coercion, and properties
  like conjugate(conjugate(z)) == z.
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  # ── Generators ─────────────────────────────────────────────────────

  defp complex_type do
    member_of([:c64, :c128])
  end

  defp complex_tensor(shape) do
    bind(complex_type(), fn type ->
      # Build complex from real iota — gives predictable non-zero values
      constant(
        Nx.complex(
          Nx.iota(shape, type: if(type == :c64, do: :f32, else: :f64)),
          Nx.add(Nx.iota(shape, type: if(type == :c64, do: :f32, else: :f64)), 1)
        )
      )
    end)
  end

  defp complex_shape do
    frequency([
      {3, constant({})},
      {5, bind(integer(1..8), &constant({&1}))},
      {4, bind(integer(1..6), fn d1 -> map(integer(1..6), &{d1, &1}) end)}
    ])
  end

  # ── Basic arithmetic ──────────────────────────────────────────────

  describe "complex arithmetic doesn't crash" do
    property "add complex tensors" do
      check all(
              shape <- complex_shape() |> filter(&(&1 != {})),
              a <- complex_tensor(shape),
              max_runs: 20
            ) do
        b = Nx.complex(Nx.broadcast(1.0, shape), Nx.broadcast(2.0, shape))
        result = Nx.add(a, b)
        assert Nx.shape(result) == shape
        assert elem(Nx.type(result), 0) == :c
      end
    end

    property "subtract complex tensors" do
      check all(
              shape <- complex_shape() |> filter(&(&1 != {})),
              a <- complex_tensor(shape),
              max_runs: 20
            ) do
        result = Nx.subtract(a, a)
        assert Nx.shape(result) == shape
      end
    end

    property "multiply complex tensors" do
      check all(
              shape <- complex_shape() |> filter(&(&1 != {})),
              a <- complex_tensor(shape),
              max_runs: 20
            ) do
        one = Nx.complex(Nx.broadcast(1.0, shape), Nx.broadcast(0.0, shape))
        result = Nx.multiply(a, one)
        assert Nx.shape(result) == shape
      end
    end

    property "divide complex tensors" do
      check all(
              shape <- complex_shape() |> filter(&(&1 != {})),
              a <- complex_tensor(shape),
              max_runs: 20
            ) do
        # Divide by non-zero complex
        b = Nx.complex(Nx.broadcast(2.0, shape), Nx.broadcast(1.0, shape))
        result = Nx.divide(a, b)
        assert Nx.shape(result) == shape
      end
    end
  end

  # ── Conjugate / real / imag ───────────────────────────────────────

  describe "conjugate, real, imag" do
    property "conjugate(conjugate(z)) == z" do
      check all(
              shape <- complex_shape() |> filter(&(&1 != {})),
              z <- complex_tensor(shape),
              max_runs: 20
            ) do
        result = Nx.conjugate(Nx.conjugate(z))

        diff =
          Nx.subtract(result, z)
          |> Nx.abs()
          |> Nx.reduce_max()
          |> Nx.to_number()

        assert diff < 1.0e-5
      end
    end

    property "real(z) + imag(z)*i == z" do
      check all(
              shape <- complex_shape() |> filter(&(&1 != {})),
              z <- complex_tensor(shape),
              max_runs: 20
            ) do
        r = Nx.real(z)
        i = Nx.imag(z)
        reconstructed = Nx.complex(r, i)

        diff =
          Nx.subtract(reconstructed, z)
          |> Nx.abs()
          |> Nx.reduce_max()
          |> Nx.to_number()

        assert diff < 1.0e-5
      end
    end

    property "real and imag have float type" do
      check all(
              shape <- complex_shape() |> filter(&(&1 != {})),
              z <- complex_tensor(shape),
              max_runs: 15
            ) do
        r = Nx.real(z)
        i = Nx.imag(z)
        assert elem(Nx.type(r), 0) == :f
        assert elem(Nx.type(i), 0) == :f
        assert Nx.shape(r) == shape
        assert Nx.shape(i) == shape
      end
    end

    property "abs of complex is non-negative" do
      check all(
              shape <- complex_shape() |> filter(&(&1 != {})),
              z <- complex_tensor(shape),
              max_runs: 15
            ) do
        result = Nx.abs(z)
        assert elem(Nx.type(result), 0) == :f
        min_val = Nx.reduce_min(result) |> Nx.to_number()
        assert min_val >= 0.0
      end
    end

    property "|z|^2 == real(z)^2 + imag(z)^2" do
      check all(
              shape <- complex_shape() |> filter(&(&1 != {})),
              z <- complex_tensor(shape),
              max_runs: 15
            ) do
        abs_sq = Nx.pow(Nx.abs(z), 2)
        re_sq = Nx.pow(Nx.real(z), 2)
        im_sq = Nx.pow(Nx.imag(z), 2)
        expected = Nx.add(re_sq, im_sq)

        diff =
          Nx.subtract(abs_sq, expected)
          |> Nx.abs()
          |> Nx.reduce_max()
          |> Nx.to_number()

        assert diff < 1.0e-2
      end
    end
  end

  # ── Reductions on complex ─────────────────────────────────────────

  describe "reductions on complex tensors" do
    property "sum doesn't crash" do
      check all(
              shape <- complex_shape() |> filter(&(&1 != {} and Nx.size(&1) > 0)),
              z <- complex_tensor(shape),
              max_runs: 15
            ) do
        result = Nx.sum(z)
        assert Nx.shape(result) == {}
        assert elem(Nx.type(result), 0) == :c
      end
    end

    property "mean doesn't crash" do
      check all(
              shape <- complex_shape() |> filter(&(&1 != {} and Nx.size(&1) > 0)),
              z <- complex_tensor(shape),
              max_runs: 15
            ) do
        result = Nx.mean(z)
        assert Nx.shape(result) == {}
      end
    end

    property "sum of ones equals element count" do
      check all(
              shape <- complex_shape() |> filter(&(&1 != {} and Nx.size(&1) > 0)),
              max_runs: 15
            ) do
        ones = Nx.complex(Nx.broadcast(1.0, shape), Nx.broadcast(0.0, shape))
        result = Nx.sum(ones)
        real_part = Nx.real(result) |> Nx.to_number()
        imag_part = Nx.imag(result) |> Nx.to_number()
        assert_in_delta real_part, Nx.size(shape), 1.0e-3
        assert_in_delta imag_part, 0.0, 1.0e-3
      end
    end
  end

  # ── Type coercion ─────────────────────────────────────────────────

  describe "complex type coercion" do
    property "float to complex preserves real part" do
      check all(
              shape <- complex_shape() |> filter(&(&1 != {})),
              max_runs: 15
            ) do
        f = Nx.iota(shape, type: :f32)
        c = Nx.as_type(f, :c64)
        assert elem(Nx.type(c), 0) == :c

        real_back = Nx.real(c)

        diff =
          Nx.subtract(real_back, f) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

        assert diff < 1.0e-5
      end
    end

    property "float to complex has zero imaginary" do
      check all(
              shape <- complex_shape() |> filter(&(&1 != {})),
              max_runs: 15
            ) do
        f = Nx.iota(shape, type: :f32)
        c = Nx.as_type(f, :c64)
        imag_part = Nx.imag(c)
        max_imag = Nx.reduce_max(Nx.abs(imag_part)) |> Nx.to_number()
        assert max_imag < 1.0e-5
      end
    end

    property "c64 to c128 roundtrip" do
      check all(
              shape <- complex_shape() |> filter(&(&1 != {})),
              z <- complex_tensor(shape),
              max_runs: 10
            ) do
        # Only test c64 inputs
        z64 = Nx.as_type(z, :c64)
        roundtripped = z64 |> Nx.as_type(:c128) |> Nx.as_type(:c64)

        diff =
          Nx.subtract(roundtripped, z64)
          |> Nx.abs()
          |> Nx.reduce_max()
          |> Nx.to_number()

        assert diff < 1.0e-5
      end
    end
  end

  # ── Complex transcendentals ───────────────────────────────────────

  describe "complex transcendentals don't crash" do
    for op <- [:exp, :log, :sqrt, :sin, :cos, :tan, :sinh, :cosh, :tanh] do
      property "#{op} on complex" do
        check all(
                shape <- complex_shape() |> filter(&(&1 != {})),
                max_runs: 10
              ) do
          # Small values to avoid overflow
          z =
            Nx.complex(
              Nx.divide(Nx.iota(shape, type: :f32), Nx.size(shape) + 1),
              Nx.divide(Nx.iota(shape, type: :f32), Nx.size(shape) + 1)
            )

          result = apply(Nx, unquote(op), [z])
          assert is_struct(result, Nx.Tensor)
          assert Nx.shape(result) == shape
        end
      end
    end
  end

  # ── Complex dot product ───────────────────────────────────────────

  describe "complex dot product" do
    property "dot of complex vectors" do
      check all(
              n <- integer(1..8),
              max_runs: 10
            ) do
        a = Nx.complex(Nx.iota({n}, type: :f32), Nx.broadcast(1.0, {n}))
        b = Nx.complex(Nx.broadcast(1.0, {n}), Nx.iota({n}, type: :f32))
        result = Nx.dot(a, b)
        assert Nx.shape(result) == {}
        assert elem(Nx.type(result), 0) == :c
      end
    end

    property "dot of complex matrices" do
      check all(
              m <- integer(1..6),
              n <- integer(1..6),
              k <- integer(1..6),
              max_runs: 10
            ) do
        a = Nx.complex(Nx.iota({m, k}, type: :f32), Nx.broadcast(1.0, {m, k}))
        b = Nx.complex(Nx.iota({k, n}, type: :f32), Nx.broadcast(1.0, {k, n}))
        result = Nx.dot(a, b)
        assert Nx.shape(result) == {m, n}
        assert elem(Nx.type(result), 0) == :c
      end
    end
  end

  # ── Shape ops on complex ──────────────────────────────────────────

  describe "shape ops on complex" do
    property "reshape preserves values" do
      check all(
              n <- integer(1..16),
              max_runs: 15
            ) do
        z = Nx.complex(Nx.iota({n}, type: :f32), Nx.broadcast(1.0, {n}))
        flat = Nx.reshape(z, {n})
        assert Nx.shape(flat) == {n}

        diff =
          Nx.subtract(flat, z) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

        assert diff < 1.0e-5
      end
    end

    property "transpose on complex matrix" do
      check all(
              m <- integer(1..6),
              n <- integer(1..6),
              max_runs: 10
            ) do
        z = Nx.complex(Nx.iota({m, n}, type: :f32), Nx.broadcast(1.0, {m, n}))
        result = Nx.transpose(z)
        assert Nx.shape(result) == {n, m}
      end
    end

    property "concatenate complex" do
      check all(
              n <- integer(1..8),
              max_runs: 10
            ) do
        a = Nx.complex(Nx.iota({n}, type: :f32), Nx.broadcast(1.0, {n}))
        b = Nx.complex(Nx.broadcast(0.0, {n}), Nx.iota({n}, type: :f32))
        result = Nx.concatenate([a, b])
        assert Nx.shape(result) == {2 * n}
      end
    end
  end
end
