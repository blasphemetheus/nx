defmodule EXLA.Defn.FFTAutodiffTest do
  use ExUnit.Case, async: true

  import Nx.Defn

  @moduletag :fft_autodiff

  # Reproduce: EXLA autodiff through FFT -> LayerNorm backward pass
  # Edifice's FNet and FNO use real-valued DFT matrix multiply as workaround
  # because this pattern reportedly fails on EXLA with Nx.less/2 errors.

  defn fft_then_sum(x) do
    x |> Nx.fft() |> Nx.real() |> Nx.sum()
  end

  defn fft_then_layernorm(x, gamma, beta) do
    fourier = x |> Nx.fft() |> Nx.real()

    mean = Nx.mean(fourier, axes: [-1], keep_axes: true)
    var = Nx.variance(fourier, axes: [-1], keep_axes: true)
    normalized = Nx.divide(Nx.subtract(fourier, mean), Nx.sqrt(Nx.add(var, 1.0e-5)))
    Nx.sum(Nx.add(Nx.multiply(normalized, gamma), beta))
  end

  defn fft_then_softmax(x) do
    fourier = x |> Nx.fft() |> Nx.real()

    max = Nx.reduce_max(fourier, axes: [-1], keep_axes: true)
    exp = Nx.exp(Nx.subtract(fourier, max))
    probs = Nx.divide(exp, Nx.sum(exp, axes: [-1], keep_axes: true))
    Nx.sum(probs)
  end

  describe "fft forward pass" do
    test "fft forward works" do
      x = Nx.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
      result = fft_then_sum(x)
      assert result.shape == {}
    end
  end

  describe "fft gradient - simple" do
    test "grad through fft then real then sum" do
      x = Nx.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

      g = Nx.Defn.grad(x, &fft_then_sum/1)
      assert g.shape == {2, 4}
    end
  end

  describe "fft gradient - layernorm (the pattern that reportedly breaks EXLA)" do
    test "grad through fft then layernorm" do
      x = Nx.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
      gamma = Nx.tensor([1.0, 1.0, 1.0, 1.0])
      beta = Nx.tensor([0.0, 0.0, 0.0, 0.0])

      g = Nx.Defn.grad(x, fn x -> fft_then_layernorm(x, gamma, beta) end)
      assert g.shape == {2, 4}
    end
  end

  describe "fft gradient - softmax" do
    test "grad through fft then softmax" do
      x = Nx.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

      g = Nx.Defn.grad(x, &fft_then_softmax/1)
      assert g.shape == {2, 4}
    end
  end
end
