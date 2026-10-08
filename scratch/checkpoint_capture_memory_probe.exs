# Same MLP as checkpoint_mlp_memory_probe.exs, but each checkpoint names only
# the activation and picks the weights up from the enclosing scope through the
# checkpoint macro. Peak memory and gemm count should match the explicit form.
defmodule MLPCapture do
  import Nx.Defn

  defn pair(x, w1, w2) do
    x |> Nx.dot(w1) |> Nx.max(0) |> Nx.dot(w2) |> Nx.max(0)
  end

  defn explicit(ws, x) do
    x = checkpoint([x, ws[0], ws[1]], &pair/3)
    x = checkpoint([x, ws[2], ws[3]], &pair/3)
    x = checkpoint([x, ws[4], ws[5]], &pair/3)
    x = checkpoint([x, ws[6], ws[7]], &pair/3)
    Nx.sum(x)
  end

  defn ckpt(ws, x) do
    x = checkpoint(x, fn x -> pair(x, ws[0], ws[1]) end)
    x = checkpoint(x, fn x -> pair(x, ws[2], ws[3]) end)
    x = checkpoint(x, fn x -> pair(x, ws[4], ws[5]) end)
    x = checkpoint(x, fn x -> pair(x, ws[6], ws[7]) end)
    Nx.sum(x)
  end
end

n = 2048
batch = String.to_integer(System.get_env("BATCH", "16384"))
key = Nx.Random.key(0)
{ws, _} = Nx.Random.normal(key, shape: {8, n, n}, type: :f32)
ws = Nx.multiply(ws, 0.02)
x = Nx.iota({batch, n}, type: :f32) |> Nx.divide(batch * n)

for {name, fun} <- [explicit: &MLPCapture.explicit/2, capture: &MLPCapture.ckpt/2] do
  exec = EXLA.to_executable(fn ws, x -> Nx.Defn.grad(ws, &fun.(&1, x)) end, [ws, x], client: :cuda)
  %{temp_size_in_bytes: temp} = EXLA.Executable.memory_stats(exec)
  hlo = EXLA.Executable.optimized_hlo(exec)
  gemms = hlo |> String.split("\n") |> Enum.count(&String.contains?(&1, "gemm"))
  IO.puts("#{name}: temp #{Float.round(temp / 1024 / 1024 / 1024, 3)} GiB, gemm lines #{gemms}")
end
