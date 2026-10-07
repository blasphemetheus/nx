defmodule Probe.Blk do
  defstruct []
end

defmodule Probe do
  import Nx.Defn

  defn body(x) do
    x |> Nx.exp() |> Nx.sin() |> Nx.sum()
  end

  defn via_block(x) do
    Nx.block(%Probe.Blk{}, [x], Nx.template({}, :f32), fn %Probe.Blk{}, x -> body(x) end)
  end

  defn plain(x), do: body(x)
end

x = Nx.iota({4}, type: :f32)
expr_block = Nx.Defn.debug_expr(fn x -> Nx.Defn.grad(x, &Probe.via_block/1) end).(x)
expr_plain = Nx.Defn.debug_expr(fn x -> Nx.Defn.grad(x, &Probe.plain/1) end).(x)

count = fn expr, op ->
  {_, n} =
    Nx.Defn.Composite.reduce(expr, 0, fn t, acc ->
      {_, n} = Nx.Defn.Tree.apply_args(t, :all, acc, fn _a, acc -> {nil, acc} end)
      acc + n
    end)
  n
end

ops = fn expr ->
  {_, acc} =
    Nx.Defn.Tree.scope_ids(expr)
    |> then(fn ids -> {nil, map_size(ids)} end)
  acc
end

IO.puts("=== grad via block")
IO.inspect(expr_block, limit: :infinity)
IO.puts("=== grad plain")
IO.inspect(expr_plain, limit: :infinity)
