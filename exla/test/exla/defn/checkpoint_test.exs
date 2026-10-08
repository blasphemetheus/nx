defmodule EXLA.Defn.CheckpointTest do
  use EXLA.Case, async: true

  import Nx.Defn

  defn body(x), do: x |> Nx.exp() |> Nx.sin()

  defn loss_with_checkpoint(x) do
    y = checkpoint(x, &body/1)
    Nx.sum(y * y)
  end

  defn loss_without_checkpoint(x) do
    y = body(x)
    Nx.sum(y * y)
  end

  defn pair(x, w1, w2) do
    x |> Nx.dot(w1) |> Nx.max(0) |> Nx.dot(w2) |> Nx.max(0)
  end

  defn mlp_without_checkpoint(ws, x) do
    x = pair(x, ws[0], ws[1])
    x = pair(x, ws[2], ws[3])
    x = pair(x, ws[4], ws[5])
    x = pair(x, ws[6], ws[7])
    Nx.sum(x)
  end

  defn mlp_with_checkpoint(ws, x) do
    x = checkpoint([x, ws[0], ws[1]], &pair/3)
    x = checkpoint([x, ws[2], ws[3]], &pair/3)
    x = checkpoint([x, ws[4], ws[5]], &pair/3)
    x = checkpoint([x, ws[6], ws[7]], &pair/3)
    Nx.sum(x)
  end

  defn mlp_with_captured_weights(ws, x) do
    x = checkpoint(x, fn x -> pair(x, ws[0], ws[1]) end)
    x = checkpoint(x, fn x -> pair(x, ws[2], ws[3]) end)
    x = checkpoint(x, fn x -> pair(x, ws[4], ws[5]) end)
    x = checkpoint(x, fn x -> pair(x, ws[6], ws[7]) end)
    Nx.sum(x)
  end

  defp count(hlo, needle), do: hlo |> String.split(needle) |> length() |> Kernel.-(1)

  test "computes the same gradient as the plain function" do
    x = Nx.iota({16}, type: :f32) |> Nx.divide(16)

    assert_all_close(
      Nx.Defn.grad(x, &loss_with_checkpoint/1),
      Nx.Defn.grad(x, &loss_without_checkpoint/1)
    )
  end

  test "checkpoints with equal shapes and different bodies keep their own body" do
    {ws, _} = Nx.Random.normal(Nx.Random.key(0), shape: {8, 16, 16}, type: :f32)
    x = Nx.iota({8, 16}, type: :f32) |> Nx.divide(128)

    assert_equal(
      Nx.Defn.grad(ws, &mlp_with_captured_weights(&1, x)),
      Nx.Defn.grad(ws, &mlp_without_checkpoint(&1, x))
    )
  end

  @tag :rematerialization
  test "keeps the recomputed body in the compiled gradient" do
    x = Nx.iota({1024}, type: :f32) |> Nx.divide(1024)

    with_checkpoint =
      EXLA.to_executable(fn x -> Nx.Defn.grad(x, &loss_with_checkpoint/1) end, [x])

    without_checkpoint =
      EXLA.to_executable(fn x -> Nx.Defn.grad(x, &loss_without_checkpoint/1) end, [x])

    assert count(EXLA.Executable.optimized_hlo(without_checkpoint), "exponential(") == 1
    assert count(EXLA.Executable.optimized_hlo(with_checkpoint), "exponential(") == 2
  end

  @tag :rematerialization
  test "lowers the peak scratch memory of the gradient" do
    n = 512
    batch = 4096
    ws = Nx.broadcast(Nx.tensor(0.01, type: :f32), {8, n, n})
    x = Nx.broadcast(Nx.tensor(0.5, type: :f32), {batch, n})

    with_checkpoint =
      EXLA.to_executable(fn ws, x -> Nx.Defn.grad(ws, &mlp_with_checkpoint(&1, x)) end, [ws, x])

    without_checkpoint =
      EXLA.to_executable(fn ws, x -> Nx.Defn.grad(ws, &mlp_without_checkpoint(&1, x)) end, [
        ws,
        x
      ])

    %{temp_size_in_bytes: with_temp} = EXLA.Executable.memory_stats(with_checkpoint)
    %{temp_size_in_bytes: without_temp} = EXLA.Executable.memory_stats(without_checkpoint)

    assert with_temp < without_temp
  end
end
