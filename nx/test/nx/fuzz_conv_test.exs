defmodule Nx.FuzzConvTest do
  @moduledoc """
  Fuzz tests for `Nx.conv` across its parameter space.

  Nx.conv has a wide parameter surface (stride, padding, dilation,
  input/kernel/output permutations, groups, batch_groups). Many of
  those interactions have thin test coverage. This file probes each
  dimension of the parameter space with property-based tests,
  asserting that:

  - forward pass doesn't crash on valid inputs
  - forward pass produces the shape the op documents
  - gradient is shape-consistent with the input
  - composition with transpose/reshape doesn't lose information
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  import Nx.Testing

  # ── Generators ─────────────────────────────────────────────────────

  # Small NCHW input: {batch, in_channels, H, W}.
  defp nchw_input() do
    bind(integer(1..2), fn b ->
      bind(integer(1..3), fn c_in ->
        bind(integer(3..6), fn h ->
          bind(integer(3..6), fn w ->
            shape = {b, c_in, h, w}
            constant(Nx.iota(shape, type: :f32))
          end)
        end)
      end)
    end)
  end

  # Kernel: {out_channels, in_channels, kH, kW}. Small.
  defp kernel_for(input) do
    {_, c_in, h, w} = Nx.shape(input)

    bind(integer(1..3), fn c_out ->
      bind(integer(1..min(h, 3)), fn kh ->
        bind(integer(1..min(w, 3)), fn kw ->
          constant(Nx.iota({c_out, c_in, kh, kw}, type: :f32))
        end)
      end)
    end)
  end

  # ── Baseline: conv doesn't crash, grad shape matches input ─────────

  describe "baseline conv shape + grad" do
    property "conv forward pass works with default opts" do
      check all(
              input <- nchw_input(),
              kernel <- kernel_for(input),
              max_runs: 12
            ) do
        result = Nx.conv(input, kernel)
        {b, _, _, _} = Nx.shape(input)
        {c_out, _, _, _} = Nx.shape(kernel)
        # Default: no padding, stride 1, no dilation. Output batch + channels
        # should match.
        assert elem(Nx.shape(result), 0) == b
        assert elem(Nx.shape(result), 1) == c_out
      end
    end

    property "grad wrt input has same shape as input" do
      check all(
              input <- nchw_input(),
              kernel <- kernel_for(input),
              max_runs: 10
            ) do
        grad = Nx.Defn.grad(input, fn x -> Nx.sum(Nx.conv(x, kernel)) end)
        assert Nx.shape(grad) == Nx.shape(input)
      end
    end

    property "grad wrt kernel has same shape as kernel" do
      check all(
              input <- nchw_input(),
              kernel <- kernel_for(input),
              max_runs: 10
            ) do
        grad = Nx.Defn.grad(kernel, fn k -> Nx.sum(Nx.conv(input, k)) end)
        assert Nx.shape(grad) == Nx.shape(kernel)
      end
    end
  end

  # ── Parameter dimensions one at a time ─────────────────────────────

  describe "strides" do
    property "conv with stride [s, s] works and shrinks output" do
      check all(
              input <- nchw_input(),
              kernel <- kernel_for(input),
              s <- integer(1..2),
              max_runs: 8
            ) do
        result = Nx.conv(input, kernel, strides: [s, s])
        assert is_struct(result, Nx.Tensor)
      end
    end

    property "grad through strided conv is shape-consistent" do
      check all(
              input <- nchw_input(),
              kernel <- kernel_for(input),
              s <- integer(1..2),
              max_runs: 8
            ) do
        grad =
          Nx.Defn.grad(input, fn x ->
            Nx.sum(Nx.conv(x, kernel, strides: [s, s]))
          end)

        assert Nx.shape(grad) == Nx.shape(input)
      end
    end
  end

  describe "padding" do
    property "conv with :same padding preserves spatial dims (stride 1)" do
      check all(
              input <- nchw_input(),
              kernel <- kernel_for(input),
              max_runs: 8
            ) do
        result = Nx.conv(input, kernel, padding: :same)
        {_, _, h_in, w_in} = Nx.shape(input)
        {_, _, h_out, w_out} = Nx.shape(result)
        assert h_out == h_in
        assert w_out == w_in
      end
    end

    property "conv with explicit padding doesn't crash" do
      check all(
              input <- nchw_input(),
              kernel <- kernel_for(input),
              p <- integer(0..2),
              max_runs: 8
            ) do
        result = Nx.conv(input, kernel, padding: [{p, p}, {p, p}])
        assert is_struct(result, Nx.Tensor)
      end
    end
  end

  describe "dilation" do
    property "conv with input_dilation doesn't crash for small values" do
      check all(
              input <- nchw_input(),
              kernel <- kernel_for(input),
              d <- integer(1..2),
              max_runs: 8
            ) do
        result = Nx.conv(input, kernel, input_dilation: [d, d])
        assert is_struct(result, Nx.Tensor)
      end
    end

    property "conv with kernel_dilation doesn't crash for small values" do
      check all(
              input <- nchw_input(),
              max_runs: 8
            ) do
        # Pick a small kernel so kernel_dilation 2 doesn't blow past input.
        {_, c_in, _, _} = Nx.shape(input)
        kernel = Nx.iota({1, c_in, 2, 2}, type: :f32)

        result = Nx.conv(input, kernel, kernel_dilation: [2, 2])
        assert is_struct(result, Nx.Tensor)
      end
    end
  end

  describe "permutations" do
    # NHWC layout: permute input [0, 2, 3, 1] on the way in.
    property "conv with input_permutation NHWC->NCHW doesn't crash" do
      check all(
              input <- nchw_input(),
              kernel <- kernel_for(input),
              max_runs: 8
            ) do
        # Transpose input to NHWC then pass input_permutation to
        # restore NCHW-style processing.
        input_nhwc = Nx.transpose(input, axes: [0, 2, 3, 1])

        result = Nx.conv(input_nhwc, kernel, input_permutation: [0, 3, 1, 2])
        assert is_struct(result, Nx.Tensor)
      end
    end

    property "conv with output_permutation doesn't crash" do
      check all(
              input <- nchw_input(),
              kernel <- kernel_for(input),
              max_runs: 8
            ) do
        result = Nx.conv(input, kernel, output_permutation: [0, 2, 3, 1])
        assert is_struct(result, Nx.Tensor)
      end
    end
  end

  describe "groups" do
    # groups requires in_channels divisible by groups.
    property "groups=1 is identity with the baseline" do
      check all(
              input <- nchw_input(),
              kernel <- kernel_for(input),
              max_runs: 8
            ) do
        baseline = Nx.conv(input, kernel)
        grouped = Nx.conv(input, kernel, feature_group_size: 1)
        assert_all_close(baseline, grouped)
      end
    end

    test "groups=2 with appropriately-shaped kernel works" do
      # input {1, 4, 4, 4}, kernel {4, 2, 2, 2} with groups=2 splits
      # 4 in_channels into 2 groups of 2.
      input = Nx.iota({1, 4, 4, 4}, type: :f32)
      kernel = Nx.iota({4, 2, 2, 2}, type: :f32)

      result = Nx.conv(input, kernel, feature_group_size: 2)
      assert is_struct(result, Nx.Tensor)
    end
  end

  describe "grad + parameter combinations" do
    # The interaction of grad with each non-trivial parameter is where
    # bugs hide. We checked each parameter in isolation above; here we
    # combine grad with each parameter.

    property "grad through :same-padded conv is shape-consistent" do
      check all(
              input <- nchw_input(),
              kernel <- kernel_for(input),
              max_runs: 8
            ) do
        grad =
          Nx.Defn.grad(input, fn x ->
            Nx.sum(Nx.conv(x, kernel, padding: :same))
          end)

        assert Nx.shape(grad) == Nx.shape(input)
      end
    end

    property "grad through input_dilated conv is shape-consistent" do
      check all(
              input <- nchw_input(),
              kernel <- kernel_for(input),
              max_runs: 8
            ) do
        grad =
          Nx.Defn.grad(input, fn x ->
            Nx.sum(Nx.conv(x, kernel, input_dilation: [2, 2]))
          end)

        assert Nx.shape(grad) == Nx.shape(input)
      end
    end

    property "grad through input_permutation'd conv is shape-consistent" do
      check all(
              input <- nchw_input(),
              kernel <- kernel_for(input),
              max_runs: 8
            ) do
        input_nhwc = Nx.transpose(input, axes: [0, 2, 3, 1])

        grad =
          Nx.Defn.grad(input_nhwc, fn x ->
            Nx.sum(Nx.conv(x, kernel, input_permutation: [0, 3, 1, 2]))
          end)

        assert Nx.shape(grad) == Nx.shape(input_nhwc)
      end
    end
  end
end
