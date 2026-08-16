defmodule EXLA.Defn.ShardingFuzzTest do
  @moduledoc """
  Sharding equivalence fuzz (FUZZ_ROADMAP T3.3).

  Oracle: shard the inputs by hand, run `EXLA.shard_jit` across the mesh,
  reassemble the per-device outputs, and demand exact equality with the
  same computation run unsharded. Any partitioning, collective, or
  reassembly defect breaks the equality.

  Runs only with a multi-device host client:

      XLA_FLAGS=--xla_force_host_platform_device_count=4 mix test \\
        test/exla/defn/sharding_fuzz_test.exs
  """
  use EXLA.Case, async: true
  use ExUnitProperties

  alias Nx.Mesh

  @moduletag :multi_device

  @devices 4

  defp shard_axis0(t, parts) do
    rows = div(Nx.axis_size(t, 0), parts)
    for i <- 0..(parts - 1), do: Nx.slice_along_axis(t, i * rows, rows, axis: 0)
  end

  defp random_tensor(rows, cols, gen_vals) do
    gen_vals |> Nx.tensor(type: {:f, 32}) |> Nx.reshape({rows, cols})
  end

  # Elementwise function vocabulary — output sharding matches input
  # sharding, so axis-0 reassembly is plain concatenation.
  defp fun_for(:add_mul), do: fn x, y -> Nx.multiply(Nx.add(x, y), y) end
  defp fun_for(:tanh_sub), do: fn x, y -> Nx.subtract(Nx.tanh(x), y) end
  defp fun_for(:sin_scale), do: fn x, y -> Nx.multiply(Nx.sin(x), Nx.add(y, 1.0)) end
  defp fun_for(:min_max), do: fn x, y -> Nx.min(Nx.max(x, y), Nx.multiply(x, y)) end

  describe "1-D mesh, axis-0 sharding" do
    property "reassembled sharded result equals the unsharded computation" do
      check all(
              rows_per_dev <- integer(1..3),
              cols <- integer(1..4),
              fun_key <- member_of([:add_mul, :tanh_sub, :sin_scale, :min_max]),
              max_runs: 10
            ) do
        rows = rows_per_dev * @devices

        check all(
                xs <- list_of(float(min: -10.0, max: 10.0), length: rows * cols),
                ys <- list_of(float(min: -10.0, max: 10.0), length: rows * cols),
                max_runs: 1
              ) do
          x = random_tensor(rows, cols, xs)
          y = random_tensor(rows, cols, ys)
          fun = fun_for(fun_key)

          mesh = %Mesh{name: "mesh", shape: {@devices}}
          shardings = [%{0 => [0]}, %{0 => [0]}]

          args =
            Enum.zip_with(shard_axis0(x, @devices), shard_axis0(y, @devices), fn xi, yi ->
              [xi, yi]
            end)

          results = EXLA.shard_jit(fun, mesh, input_shardings: shardings).(args)
          reassembled = Nx.concatenate(results)

          expected = Nx.Defn.jit_apply(fun, [x, y])
          assert_equal(to_binary_backend(reassembled), to_binary_backend(expected))
        end
      end
    end

    property "tuple outputs reassemble element-wise" do
      check all(rows_per_dev <- integer(1..3), cols <- integer(1..3), max_runs: 10) do
        rows = rows_per_dev * @devices

        check all(
                xs <- list_of(float(min: -10.0, max: 10.0), length: rows * cols),
                ys <- list_of(float(min: -10.0, max: 10.0), length: rows * cols),
                max_runs: 1
              ) do
          x = random_tensor(rows, cols, xs)
          y = random_tensor(rows, cols, ys)
          fun = fn a, b -> {Nx.add(a, b), Nx.multiply(a, b)} end

          mesh = %Mesh{name: "mesh", shape: {@devices}}
          shardings = [%{0 => [0]}, %{0 => [0]}]

          args =
            Enum.zip_with(shard_axis0(x, @devices), shard_axis0(y, @devices), fn xi, yi ->
              [xi, yi]
            end)

          results = EXLA.shard_jit(fun, mesh, input_shardings: shardings).(args)

          sum_reassembled = results |> Enum.map(&elem(&1, 0)) |> Nx.concatenate()
          prod_reassembled = results |> Enum.map(&elem(&1, 1)) |> Nx.concatenate()

          assert_equal(to_binary_backend(sum_reassembled), to_binary_backend(Nx.add(x, y)))

          assert_equal(
            to_binary_backend(prod_reassembled),
            to_binary_backend(Nx.multiply(x, y))
          )
        end
      end
    end
  end

  describe "2-D mesh, block sharding" do
    property "2x2 block-sharded elementwise op reassembles to the unsharded result" do
      check all(
              half_rows <- integer(1..3),
              half_cols <- integer(1..3),
              max_runs: 10
            ) do
        rows = half_rows * 2
        cols = half_cols * 2

        check all(
                xs <- list_of(float(min: -10.0, max: 10.0), length: rows * cols),
                max_runs: 1
              ) do
          x = random_tensor(rows, cols, xs)
          fun = fn t -> Nx.multiply(Nx.tanh(t), 3.0) end

          mesh = %Mesh{name: "mesh", shape: {2, 2}}
          shardings = [%{0 => [0], 1 => [1]}]

          # device order is row-major over the mesh: (r0,c0) (r0,c1) (r1,c0) (r1,c1)
          blocks =
            for r <- 0..1, c <- 0..1 do
              x
              |> Nx.slice_along_axis(r * half_rows, half_rows, axis: 0)
              |> Nx.slice_along_axis(c * half_cols, half_cols, axis: 1)
            end

          args = Enum.map(blocks, &[&1])

          results = EXLA.shard_jit(fun, mesh, input_shardings: shardings).(args)

          [d0, d1, d2, d3] = results
          top = Nx.concatenate([d0, d1], axis: 1)
          bottom = Nx.concatenate([d2, d3], axis: 1)
          reassembled = Nx.concatenate([top, bottom], axis: 0)

          expected = Nx.Defn.jit_apply(fun, [x])
          assert_equal(to_binary_backend(reassembled), to_binary_backend(expected))
        end
      end
    end
  end
end
