defmodule EXLA.Defn.CheckpointTest do
  use EXLA.Case, async: true

  import Nx.Defn
  import Nx.Testing, only: [assert_equal: 2]

  defn pair(x, w1, w2), do: x |> Nx.dot(w1) |> Nx.max(0) |> Nx.dot(w2) |> Nx.max(0)

  defn mlp_without_checkpoint(ws, x) do
    x = pair(x, ws[0], ws[1])
    x = pair(x, ws[2], ws[3])
    x = pair(x, ws[4], ws[5])
    x = pair(x, ws[6], ws[7])
    Nx.sum(x)
  end

  defn mlp_with_checkpoint(ws, x) do
    x = checkpoint(x, fn x -> pair(x, ws[0], ws[1]) end)
    x = checkpoint(x, fn x -> pair(x, ws[2], ws[3]) end)
    x = checkpoint(x, fn x -> pair(x, ws[4], ws[5]) end)
    x = checkpoint(x, fn x -> pair(x, ws[6], ws[7]) end)
    Nx.sum(x)
  end

  test "checkpoints with equal shapes and different bodies keep their own body" do
    {ws, _} = Nx.Random.normal(Nx.Random.key(0), shape: {8, 16, 16}, type: :f32)
    x = Nx.iota({8, 16}, type: :f32) |> Nx.divide(128)

    with_checkpoint =
      EXLA.jit(fn ws, x -> Nx.Defn.grad(ws, &mlp_with_checkpoint(&1, x)) end).(ws, x)

    without_checkpoint =
      EXLA.jit(fn ws, x -> Nx.Defn.grad(ws, &mlp_without_checkpoint(&1, x)) end).(ws, x)

    assert_equal(with_checkpoint, without_checkpoint)
  end
end
