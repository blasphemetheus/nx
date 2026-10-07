# Usage (from a worktree with EXLA built): XLA_FLAGS="$XLA_FLAGS --xla_dump_to=DIR --xla_dump_hlo_as_text --xla_dump_hlo_pass_re=.*" MIX_ENV=test EXLA_TARGET=cuda mix run scratch/checkpoint_block_cse_probe.exs
# Then count exponential( in DIR/module_0001.main.*after_optimizations.txt: 2 = recompute kept, 1 = CSE folded it.
# Swap client: :host for :cuda to see the CPU pipeline. Pair with the throwaway optimization_barrier patch described in HANDOFF.md.
defmodule Probe.Blk do
  defstruct []
end

defmodule Probe do
  import Nx.Defn

  defn body(x), do: x |> Nx.exp() |> Nx.sin()

  defn via_block(x) do
    y = Nx.block(%Probe.Blk{}, [x], x, fn %Probe.Blk{}, x -> body(x) end)
    Nx.sum(y * y)
  end

  defn plain(x) do
    y = body(x)
    Nx.sum(y * y)
  end
end

x = Nx.iota({1024}, type: :f32) |> Nx.divide(1024)
g1 = Nx.Defn.jit(fn x -> Nx.Defn.grad(x, &Probe.via_block/1) end, compiler: EXLA, client: :host).(x)
g2 = Nx.Defn.jit(fn x -> Nx.Defn.grad(x, &Probe.plain/1) end, compiler: EXLA, client: :host).(x)
IO.puts("grads equal: #{inspect(Nx.all_close(g1, g2) |> Nx.to_number())}")
IO.puts("=== pre-opt expr (via_block)")
IO.inspect(Nx.Defn.debug_expr(fn x -> Nx.Defn.grad(x, &Probe.via_block/1) end).(x), limit: :infinity)
