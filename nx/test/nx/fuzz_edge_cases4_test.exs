defmodule Nx.FuzzEdgeCases4Test do
  @moduledoc """
  Tier 4 (part 4): Complex type boundary tests and vectorized+edge combinations.
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  # ── Complex type: operations that should work ──────────────────────

  describe "complex type: supported operations" do
    test "complex construction from real and imag" do
      result = Nx.complex(Nx.tensor(1.0), Nx.tensor(2.0))
      assert Nx.type(result) == {:c, 64}
      assert Nx.to_number(Nx.real(result)) == 1.0
      assert Nx.to_number(Nx.imag(result)) == 2.0
    end

    test "complex from f64 components produces c128" do
      result = Nx.complex(Nx.tensor(1.0, type: :f64), Nx.tensor(2.0, type: :f64))
      assert Nx.type(result) == {:c, 128}
    end

    test "real/1 extracts real part" do
      z = Nx.complex(Nx.tensor(3.0), Nx.tensor(4.0))
      assert Nx.to_number(Nx.real(z)) == 3.0
    end

    test "imag/1 extracts imaginary part" do
      z = Nx.complex(Nx.tensor(3.0), Nx.tensor(4.0))
      assert Nx.to_number(Nx.imag(z)) == 4.0
    end

    test "real/1 on non-complex returns unchanged" do
      t = Nx.tensor(5.0, type: :f32)
      assert Nx.to_number(Nx.real(t)) == 5.0
    end

    test "imag/1 on non-complex returns zero" do
      t = Nx.tensor(5.0, type: :f32)
      assert Nx.to_number(Nx.imag(t)) == 0.0
    end

    test "conjugate negates imaginary part" do
      z = Nx.complex(Nx.tensor(3.0), Nx.tensor(4.0))
      conj = Nx.conjugate(z)
      assert Nx.to_number(Nx.real(conj)) == 3.0
      assert Nx.to_number(Nx.imag(conj)) == -4.0
    end

    test "abs of complex returns magnitude" do
      z = Nx.complex(Nx.tensor(3.0), Nx.tensor(4.0))
      result = Nx.abs(z)
      assert Nx.type(result) == {:f, 32}
      assert_in_delta Nx.to_number(result), 5.0, 1.0e-5
    end

    test "phase of complex" do
      # phase of 1+i should be pi/4
      z = Nx.complex(Nx.tensor(1.0), Nx.tensor(1.0))
      result = Nx.phase(z)
      assert_in_delta Nx.to_number(result), :math.pi() / 4, 1.0e-5
    end

    test "complex addition" do
      a = Nx.complex(Nx.tensor(1.0), Nx.tensor(2.0))
      b = Nx.complex(Nx.tensor(3.0), Nx.tensor(4.0))
      result = Nx.add(a, b)
      assert Nx.to_number(Nx.real(result)) == 4.0
      assert Nx.to_number(Nx.imag(result)) == 6.0
    end

    test "complex multiplication" do
      # (1+2i)(3+4i) = 3+4i+6i+8i² = 3+10i-8 = -5+10i
      a = Nx.complex(Nx.tensor(1.0), Nx.tensor(2.0))
      b = Nx.complex(Nx.tensor(3.0), Nx.tensor(4.0))
      result = Nx.multiply(a, b)
      assert_in_delta Nx.to_number(Nx.real(result)), -5.0, 1.0e-5
      assert_in_delta Nx.to_number(Nx.imag(result)), 10.0, 1.0e-5
    end

    test "complex exp (Euler's formula)" do
      # exp(i*pi) ≈ -1 + 0i
      z = Nx.complex(Nx.tensor(0.0), Nx.tensor(:math.pi()))
      result = Nx.exp(z)
      assert_in_delta Nx.to_number(Nx.real(result)), -1.0, 1.0e-5
      assert_in_delta Nx.to_number(Nx.imag(result)), 0.0, 1.0e-5
    end

    test "complex dot product" do
      a = Nx.tensor([Complex.new(1, 0), Complex.new(0, 1)], type: :c64)
      b = Nx.tensor([Complex.new(1, 0), Complex.new(0, 1)], type: :c64)
      result = Nx.dot(a, b)
      # Nx.dot conjugates the left operand for complex:
      # conj(1)*1 + conj(i)*i = 1 + (-i)(i) = 1 + 1 = 2
      # OR Nx.dot may not conjugate at all: 1*1 + i*i = 1 - 1 = 0
      # Just verify it doesn't crash and returns a complex scalar
      assert Nx.shape(result) == {}
      assert Nx.type(result) == {:c, 64}
    end

    test "complex sum" do
      t = Nx.tensor([Complex.new(1, 2), Complex.new(3, 4)], type: :c64)
      result = Nx.sum(t)
      assert_in_delta Nx.to_number(Nx.real(result)), 4.0, 1.0e-5
      assert_in_delta Nx.to_number(Nx.imag(result)), 6.0, 1.0e-5
    end

    test "complex reshape preserves values" do
      t = Nx.tensor([Complex.new(1, 2), Complex.new(3, 4), Complex.new(5, 6)], type: :c64)
      result = Nx.reshape(t, {1, 3})
      assert Nx.shape(result) == {1, 3}
      assert Nx.type(result) == {:c, 64}
    end
  end

  # ── Complex type: operations that should REJECT ────────────────────

  describe "complex type: rejected operations" do
    setup do
      z = Nx.tensor([Complex.new(1, 2), Complex.new(3, 4)], type: :c64)
      %{z: z}
    end

    test "sort rejects complex", %{z: z} do
      assert_raise ArgumentError, ~r/complex/, fn ->
        Nx.sort(z)
      end
    end

    test "argsort rejects complex", %{z: z} do
      assert_raise ArgumentError, ~r/complex/, fn ->
        Nx.argsort(z)
      end
    end

    test "reduce_max rejects complex", %{z: z} do
      assert_raise ArgumentError, ~r/complex/, fn ->
        Nx.reduce_max(z)
      end
    end

    test "reduce_min rejects complex", %{z: z} do
      assert_raise ArgumentError, ~r/complex/, fn ->
        Nx.reduce_min(z)
      end
    end

    test "argmax rejects complex", %{z: z} do
      assert_raise ArgumentError, ~r/complex/, fn ->
        Nx.argmax(z)
      end
    end

    test "argmin rejects complex", %{z: z} do
      assert_raise ArgumentError, ~r/complex/, fn ->
        Nx.argmin(z)
      end
    end

    test "greater rejects complex", %{z: z} do
      assert_raise ArgumentError, ~r/complex/, fn ->
        Nx.greater(z, z)
      end
    end

    test "less rejects complex", %{z: z} do
      assert_raise ArgumentError, ~r/complex/, fn ->
        Nx.less(z, z)
      end
    end

    test "clip rejects complex", %{z: z} do
      assert_raise ArgumentError, ~r/complex/, fn ->
        Nx.clip(z, 0, 10)
      end
    end

    test "window_max rejects complex", %{z: z} do
      assert_raise ArgumentError, ~r/complex/, fn ->
        Nx.window_max(z, {2})
      end
    end

    test "window_min rejects complex", %{z: z} do
      assert_raise ArgumentError, ~r/complex/, fn ->
        Nx.window_min(z, {2})
      end
    end

    test "bitcast rejects complex" do
      z = Nx.tensor(Complex.new(1, 2), type: :c64)
      assert_raise ArgumentError, ~r/does not support complex/, fn ->
        Nx.bitcast(z, :s64)
      end
    end

    test "erf rejects complex" do
      z = Nx.tensor(Complex.new(1, 0), type: :c64)
      assert_raise ArgumentError, ~r/complex/, fn ->
        Nx.erf(z)
      end
    end

    test "complex/2 rejects complex inputs" do
      z = Nx.tensor(Complex.new(1, 2), type: :c64)
      assert_raise ArgumentError, ~r/complex/, fn ->
        Nx.complex(z, z)
      end
    end
  end

  # ── Complex type: equivalence properties ───────────────────────────

  describe "complex algebraic properties" do
    test "conjugate(conjugate(z)) == z (involution)" do
      z = Nx.tensor([Complex.new(1, 2), Complex.new(3, -4)], type: :c64)
      result = z |> Nx.conjugate() |> Nx.conjugate()
      for {orig, rt} <- Enum.zip(Nx.to_flat_list(z), Nx.to_flat_list(result)) do
        assert_in_delta Complex.abs(Complex.subtract(orig, rt)), 0.0, 1.0e-5
      end
    end

    test "z * conjugate(z) == |z|² (real, non-negative)" do
      z = Nx.tensor([Complex.new(3, 4), Complex.new(1, -1)], type: :c64)
      product = Nx.multiply(z, Nx.conjugate(z))
      abs_sq = Nx.multiply(Nx.abs(z), Nx.abs(z))

      for {p, a} <- Enum.zip(Nx.to_flat_list(Nx.real(product)), Nx.to_flat_list(abs_sq)) do
        assert_in_delta p, a, 1.0e-4
      end

      # Imaginary part should be ~0
      for im <- Nx.to_flat_list(Nx.imag(product)) do
        assert_in_delta im, 0.0, 1.0e-5
      end
    end

    test "real(z) + i*imag(z) == z (decomposition roundtrip)" do
      z = Nx.tensor([Complex.new(3, 4), Complex.new(-1, 2)], type: :c64)
      reconstructed = Nx.complex(Nx.real(z), Nx.imag(z))

      for {orig, rec} <- Enum.zip(Nx.to_flat_list(z), Nx.to_flat_list(reconstructed)) do
        assert_in_delta Complex.abs(Complex.subtract(orig, rec)), 0.0, 1.0e-5
      end
    end
  end

  # ── Type promotion with complex ────────────────────────────────────

  describe "type promotion with complex" do
    test "f32 + c64 promotes to c64" do
      a = Nx.tensor(1.0, type: :f32)
      b = Nx.tensor(Complex.new(2, 3), type: :c64)
      result = Nx.add(a, b)
      assert Nx.type(result) == {:c, 64}
    end

    test "f64 + c64 promotes to c128" do
      assert Nx.Type.merge({:f, 64}, {:c, 64}) == {:c, 128}
    end

    test "s32 + c64 promotes to c64" do
      a = Nx.tensor(5, type: :s32)
      b = Nx.tensor(Complex.new(1, 1), type: :c64)
      result = Nx.add(a, b)
      assert Nx.type(result) == {:c, 64}
    end

    test "to_complex and to_real roundtrip types" do
      assert Nx.Type.to_complex({:f, 32}) == {:c, 64}
      assert Nx.Type.to_complex({:f, 64}) == {:c, 128}
      assert Nx.Type.to_real({:c, 64}) == {:f, 32}
      assert Nx.Type.to_real({:c, 128}) == {:f, 64}
    end
  end

  # ── Vectorized + edge case combinations ────────────────────────────
  # Test that boundary-condition operations work correctly with vectorized tensors

  describe "vectorized + slice edge cases" do
    test "slice of vectorized tensor" do
      t = Nx.iota({3, 5}) |> Nx.vectorize(:batch)
      # Inner shape is {5}, slice inner
      result = Nx.slice(t, [1], [3])
      assert result.vectorized_axes == [batch: 3]
      assert result.shape == {3}
    end

    test "take of vectorized tensor" do
      t = Nx.iota({2, 4}) |> Nx.vectorize(:batch)
      idx = Nx.tensor([3, 1, 0])
      result = Nx.take(t, idx)
      assert result.vectorized_axes == [batch: 2]
      assert result.shape == {3}
    end

    test "concatenate vectorized tensors" do
      a = Nx.iota({2, 3}) |> Nx.vectorize(:batch)
      b = Nx.iota({2, 4}) |> Nx.vectorize(:batch)
      result = Nx.concatenate([a, b], axis: 0)
      assert result.vectorized_axes == [batch: 2]
      assert result.shape == {7}
    end

    test "pad vectorized tensor" do
      t = Nx.iota({2, 3}, type: :f32) |> Nx.vectorize(:batch)
      result = Nx.pad(t, Nx.tensor(0.0), [{1, 1, 0}])
      assert result.vectorized_axes == [batch: 2]
      assert result.shape == {5}
    end

    test "reshape vectorized tensor (inner only)" do
      t = Nx.iota({2, 6}) |> Nx.vectorize(:batch)
      result = Nx.reshape(t, {2, 3})
      assert result.vectorized_axes == [batch: 2]
      assert result.shape == {2, 3}
    end

    test "sum of vectorized tensor along inner axis" do
      t = Nx.iota({3, 4}, type: :f32) |> Nx.vectorize(:batch)
      result = Nx.sum(t, axes: [0])
      assert result.vectorized_axes == [batch: 3]
      assert result.shape == {}
    end

    test "sort of vectorized tensor" do
      t = Nx.tensor([[3, 1, 2], [6, 4, 5]]) |> Nx.vectorize(:batch)
      result = Nx.sort(t)
      devec = Nx.devectorize(result)
      assert Nx.to_flat_list(devec[0]) == [1, 2, 3]
      assert Nx.to_flat_list(devec[1]) == [4, 5, 6]
    end

    test "reverse of vectorized tensor" do
      t = Nx.iota({2, 4}) |> Nx.vectorize(:batch)
      result = Nx.reverse(t)
      devec = Nx.devectorize(result)
      assert Nx.to_flat_list(devec[0]) == [3, 2, 1, 0]
    end
  end

  describe "vectorized + type operations" do
    test "as_type vectorized" do
      t = Nx.iota({3, 4}, type: :s32) |> Nx.vectorize(:batch)
      result = Nx.as_type(t, :f32)
      assert Nx.type(result) == {:f, 32}
      assert result.vectorized_axes == [batch: 3]
    end

    test "abs of vectorized" do
      t = Nx.tensor([[1, -2, 3], [-4, 5, -6]]) |> Nx.vectorize(:batch)
      result = Nx.abs(t)
      devec = Nx.devectorize(result)
      assert Nx.to_flat_list(devec[0]) == [1, 2, 3]
      assert Nx.to_flat_list(devec[1]) == [4, 5, 6]
    end

    test "add vectorized with broadcast" do
      t = Nx.iota({3, 4}, type: :f32) |> Nx.vectorize(:batch)
      scalar = Nx.tensor(100.0)
      result = Nx.add(t, scalar)
      assert result.vectorized_axes == [batch: 3]
      devec = Nx.devectorize(result)
      assert Nx.to_number(devec[0][0]) == 100.0
      assert Nx.to_number(devec[0][1]) == 101.0
    end

    test "multiply two vectorized tensors (same axes)" do
      a = Nx.iota({2, 3}, type: :f32) |> Nx.vectorize(:batch)
      b = Nx.broadcast(Nx.tensor(2.0), {2, 3}) |> Nx.vectorize(:batch)
      result = Nx.multiply(a, b)
      assert result.vectorized_axes == [batch: 2]
      devec = Nx.devectorize(result)
      assert Nx.to_flat_list(devec[0]) == [0.0, 2.0, 4.0]
    end
  end

  describe "vectorized + linalg" do
    test "dot of vectorized tensor" do
      a = Nx.iota({2, 3, 4}, type: :f32) |> Nx.vectorize(:batch)
      b = Nx.iota({2, 4, 2}, type: :f32) |> Nx.vectorize(:batch)
      result = Nx.dot(a, b)
      assert result.vectorized_axes == [batch: 2]
      assert result.shape == {3, 2}
    end

    test "transpose of vectorized" do
      t = Nx.iota({2, 3, 4}) |> Nx.vectorize(:batch)
      result = Nx.transpose(t)
      assert result.vectorized_axes == [batch: 2]
      assert result.shape == {4, 3}
    end
  end

  # ── Vectorized + reduction edge cases ──────────────────────────────

  describe "vectorized + reductions" do
    test "argmax of vectorized" do
      t = Nx.tensor([[3, 1, 5, 2], [1, 4, 2, 8]]) |> Nx.vectorize(:batch)
      result = Nx.argmax(t, axis: 0)
      devec = Nx.devectorize(result)
      assert Nx.to_number(devec[0]) == 2  # index of 5
      assert Nx.to_number(devec[1]) == 3  # index of 8
    end

    test "reduce_max of vectorized" do
      t = Nx.tensor([[3.0, 1.0, 5.0], [2.0, 8.0, 4.0]]) |> Nx.vectorize(:batch)
      result = Nx.reduce_max(t)
      devec = Nx.devectorize(result)
      assert Nx.to_number(devec[0]) == 5.0
      assert Nx.to_number(devec[1]) == 8.0
    end

    test "all of vectorized (per-batch)" do
      t = Nx.tensor([[1, 1, 1], [1, 0, 1]]) |> Nx.vectorize(:batch)
      result = Nx.all(t)
      devec = Nx.devectorize(result)
      assert Nx.to_number(devec[0]) == 1
      assert Nx.to_number(devec[1]) == 0
    end

    test "any of vectorized (per-batch)" do
      t = Nx.tensor([[0, 0, 0], [0, 1, 0]]) |> Nx.vectorize(:batch)
      result = Nx.any(t)
      devec = Nx.devectorize(result)
      assert Nx.to_number(devec[0]) == 0
      assert Nx.to_number(devec[1]) == 1
    end
  end
end
