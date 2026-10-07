defmodule Probe do
  import Nx.Defn

  defn body(x), do: x |> Nx.exp() |> Nx.sin()

  defn with_ckpt(x) do
    y = Nx.Defn.checkpoint(x, &body/1)
    Nx.sum(y * y)
  end

  defn plain(x) do
    y = body(x)
    Nx.sum(y * y)
  end
end

x = Nx.iota({1024}, type: :f32) |> Nx.divide(1024)
client = String.to_atom(System.get_env("PROBE_CLIENT", "cuda"))
g1 = Nx.Defn.jit(fn x -> Nx.Defn.grad(x, &Probe.with_ckpt/1) end, compiler: EXLA, client: client).(x)
g2 = Nx.Defn.jit(fn x -> Nx.Defn.grad(x, &Probe.plain/1) end, compiler: EXLA, client: client).(x)
IO.puts("grads all_close: #{Nx.to_number(Nx.all_close(g1, g2))}")
IO.inspect(Nx.Defn.debug_expr(fn x -> Nx.Defn.grad(x, &Probe.with_ckpt/1) end).(x), label: "grad expr")
