defmodule FuzzGen do
  @moduledoc """
  Shared StreamData generators for value-aware fuzzing (FUZZ_ROADMAP T1.1).

  The existing fuzz suites overwhelmingly use `Nx.iota` values — monotone,
  non-negative, integral — which cannot expose cancellation, sign-handling,
  ordering, or NaN-propagation bugs. These generators build tensors from raw
  bit patterns instead, so NaN payloads, ±Infinity, denormals, negative zero,
  and the full exponent range all occur naturally.

  Elixir floats cannot represent NaN/Infinity, so tensors are constructed via
  `Nx.from_binary/2` and inspected via `Nx.to_binary/1` or classification
  helpers — never through `Nx.to_number/1`.
  """

  import StreamData

  @doc """
  A tensor of the given float type whose elements are random bit patterns,
  with special values (NaN variants, ±Inf, ±0.0, denormals, exact powers of
  two, max-finite) mixed in at elevated frequency.
  """
  def bit_tensor(shape_gen \\ shape(), type \\ {:f, 32}) do
    bind(shape_gen, fn shape ->
      count = Tuple.product(shape)

      bind(list_of(bits(type), length: count), fn words ->
        bin = for w <- words, into: <<>>, do: encode_word(w, type)
        constant(bin |> Nx.from_binary(type) |> Nx.reshape(shape))
      end)
    end)
  end

  @doc "Random shapes, rank 0-3, dims 1-6 (small on purpose: values carry the load)."
  def shape do
    frequency([
      {1, constant({})},
      {3, tuple({integer(1..6)})},
      {3, tuple({integer(1..4), integer(1..4)})},
      {2, tuple({integer(1..3), integer(1..3), integer(1..3)})}
    ])
  end

  @doc """
  A single element as an integer bit pattern for `type`. Mixes uniform random
  bits with forced special patterns.
  """
  def bits({:f, bits_size} = type) do
    frequency([
      {5, integer(0..(Bitwise.bsl(1, bits_size) - 1))},
      {3, member_of(special_bits(type))}
    ])
  end

  # Special bit patterns per float width: ±0, ±Inf, quiet/signaling NaN,
  # smallest/largest denormal, smallest/largest normal, 1.0, -1.0.
  defp special_bits({:f, 32}) do
    [
      0x0000_0000,
      0x8000_0000,
      0x7F80_0000,
      0xFF80_0000,
      0x7FC0_0001,
      0x7F80_0001,
      0xFFC0_1234,
      0x0000_0001,
      0x007F_FFFF,
      0x0080_0000,
      0x7F7F_FFFF,
      0x3F80_0000,
      0xBF80_0000
    ]
  end

  defp special_bits({:f, 64}) do
    [
      0x0000_0000_0000_0000,
      0x8000_0000_0000_0000,
      0x7FF0_0000_0000_0000,
      0xFFF0_0000_0000_0000,
      0x7FF8_0000_0000_0001,
      0x7FF0_0000_0000_0001,
      0xFFF8_0000_0000_BEEF,
      0x0000_0000_0000_0001,
      0x000F_FFFF_FFFF_FFFF,
      0x0010_0000_0000_0000,
      0x7FEF_FFFF_FFFF_FFFF,
      0x3FF0_0000_0000_0000,
      0xBFF0_0000_0000_0000
    ]
  end

  defp encode_word(w, {:f, 32}), do: <<w::32-native>>
  defp encode_word(w, {:f, 64}), do: <<w::64-native>>

  @doc """
  A finite-valued tensor built from bit patterns with NaN/Inf excluded but
  denormals, −0.0, and extreme exponents kept. For ops whose NaN semantics
  are implementation-defined but which should be exercised at range extremes.
  """
  def finite_bit_tensor(shape_gen \\ shape(), type \\ {:f, 32}) do
    map(bit_tensor(shape_gen, type), fn t ->
      finite = Nx.logical_and(Nx.logical_not(Nx.is_nan(t)), Nx.logical_not(Nx.is_infinity(t)))
      Nx.select(finite, t, Nx.tensor(1.0, type: Nx.type(t)))
    end)
  end

  @doc """
  A cancellation-prone 1-D f64 tensor: pairs (x, -x(1+ε)) whose naive sum is
  catastrophically smaller than the summands.
  """
  def cancellation_tensor do
    bind(integer(1..8), fn pairs ->
      bind(list_of(float(min: 1.0e-3, max: 1.0e12), length: pairs), fn xs ->
        values = Enum.flat_map(xs, fn x -> [x, -x * (1.0 + 1.0e-14)] end)
        constant(Nx.tensor(values, type: {:f, 64}))
      end)
    end)
  end

  ## Bit-level classification (independent oracle — no Nx involved)

  @doc "Classify every element of a float tensor from its raw bits."
  def classify(%Nx.Tensor{} = t) do
    {:f, size} = Nx.type(t)

    for <<word::size(size)-native <- Nx.to_binary(t)>> do
      classify_bits(word, size)
    end
  end

  defp classify_bits(word, size) do
    {exp_bits, frac_bits} = if size == 32, do: {8, 23}, else: {11, 52}
    <<_sign::1, exp::size(exp_bits), frac::size(frac_bits)>> = <<word::size(size)>>

    max_exp = Bitwise.bsl(1, exp_bits) - 1

    cond do
      exp == max_exp and frac != 0 -> :nan
      exp == max_exp -> :infinity
      exp == 0 and frac != 0 -> :denormal
      exp == 0 -> :zero
      true -> :normal
    end
  end
end
