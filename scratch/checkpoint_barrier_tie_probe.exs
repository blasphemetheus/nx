# Hand-written forward and backward for four dense-relu pairs, no grad transform.
# Three ways to feed the recompute of each block's intermediates:
#   none  - recompute reads x directly (XLA merges it with the forward copy)
#   input - recompute reads barrier({x})          (what Variant A/B do)
#   tied  - recompute reads barrier({x, g})       (what JAX does)
# Compare XLA's peak buffer bytes and where the recompute lands in the schedule.
defmodule Manual do
  import Nx.Defn

  defn fwd(x, w1, w2), do: x |> Nx.dot(w1) |> Nx.max(0) |> Nx.dot(w2) |> Nx.max(0)

  defn bwd_body(x, w1, w2, g) do
    h1 = Nx.dot(x, w1)
    a1 = Nx.max(h1, 0)
    h2 = Nx.dot(a1, w2)
    gh2 = Nx.select(h2 > 0, g, 0)
    dw2 = Nx.dot(a1, [0], gh2, [0])
    ga1 = Nx.dot(gh2, [1], w2, [1])
    gh1 = Nx.select(h1 > 0, ga1, 0)
    dw1 = Nx.dot(x, [0], gh1, [0])
    gx = Nx.dot(gh1, [1], w1, [1])
    {gx, dw1, dw2}
  end

  defn bwd_none(x, w1, w2, g), do: bwd_body(x, w1, w2, g)

  defn bwd_input(x, w1, w2, g) do
    {x} = optimization_barrier({x})
    bwd_body(x, w1, w2, g)
  end

  defn bwd_tied(x, w1, w2, g) do
    {x, g} = optimization_barrier({x, g})
    bwd_body(x, w1, w2, g)
  end

  defn run_none(ws, x), do: run(ws, x, &bwd_none/4)
  defn run_input(ws, x), do: run(ws, x, &bwd_input/4)
  defn run_tied(ws, x), do: run(ws, x, &bwd_tied/4)

  deftransformp run(ws, x, bwd) do
    x1 = fwd(x, ws[0], ws[1])
    x2 = fwd(x1, ws[2], ws[3])
    x3 = fwd(x2, ws[4], ws[5])
    y = fwd(x3, ws[6], ws[7])
    g = Nx.broadcast(Nx.tensor(1.0, type: :f32), y)
    {g3, dw6, dw7} = bwd.(x3, ws[6], ws[7], g)
    {g2, dw4, dw5} = bwd.(x2, ws[4], ws[5], g3)
    {g1, dw2, dw3} = bwd.(x1, ws[2], ws[3], g2)
    {_g0, dw0, dw1} = bwd.(x, ws[0], ws[1], g1)
    Nx.stack([dw0, dw1, dw2, dw3, dw4, dw5, dw6, dw7])
  end
end

n = 2048
batch = String.to_integer(System.get_env("BATCH", "16384"))
key = Nx.Random.key(0)
{ws, _} = Nx.Random.normal(key, shape: {8, n, n}, type: :f32)
ws = Nx.multiply(ws, 0.02)
x = Nx.iota({batch, n}, type: :f32) |> Nx.divide(batch * n)

mode = System.get_env("PROBE_MODE", "none")
fun = Map.fetch!(%{"none" => &Manual.run_none/2, "input" => &Manual.run_input/2, "tied" => &Manual.run_tied/2}, mode)
g = Nx.Defn.jit(fun, compiler: EXLA, client: :cuda).(ws, x)
IO.puts("#{mode}: grad sum = #{Nx.to_number(Nx.sum(g))}")
