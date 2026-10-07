# Peak-memory comparison on XLA's buffer-assignment report.
# Each checkpoint wraps two dense+relu sub-layers, so the body has real residuals
# (relu's gradient needs its pre-activation input) that a checkpoint can drop.
defmodule MLP do
  import Nx.Defn

  defn pair(x, w1, w2) do
    x |> Nx.dot(w1) |> Nx.max(0) |> Nx.dot(w2) |> Nx.max(0)
  end

  defn plain(ws, x) do
    x = pair(x, ws[0], ws[1])
    x = pair(x, ws[2], ws[3])
    x = pair(x, ws[4], ws[5])
    x = pair(x, ws[6], ws[7])
    Nx.sum(x)
  end

  defn ckpt(ws, x) do
    x = Nx.Defn.checkpoint(x, &pair(&1, ws[0], ws[1]))
    x = Nx.Defn.checkpoint(x, &pair(&1, ws[2], ws[3]))
    x = Nx.Defn.checkpoint(x, &pair(&1, ws[4], ws[5]))
    x = Nx.Defn.checkpoint(x, &pair(&1, ws[6], ws[7]))
    Nx.sum(x)
  end
end

n = 2048
batch = String.to_integer(System.get_env("BATCH", "16384"))
key = Nx.Random.key(0)
{ws, _} = Nx.Random.normal(key, shape: {8, n, n}, type: :f32)
ws = Nx.multiply(ws, 0.02)
x = Nx.iota({batch, n}, type: :f32) |> Nx.divide(batch * n)

which = System.get_env("WHICH", "plain")
fun = if which == "plain", do: &MLP.plain/2, else: &MLP.ckpt/2
g = Nx.Defn.jit(fn ws, x -> Nx.Defn.grad(ws, &fun.(&1, x)) end, compiler: EXLA, client: :cuda).(ws, x)
IO.puts("#{which}: grad sum = #{Nx.to_number(Nx.sum(g))}")
