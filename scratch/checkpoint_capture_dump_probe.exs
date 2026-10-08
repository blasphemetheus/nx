# Peak-memory comparison for the recompute-at-every-use block, explicit input
# list against closure captures through the checkpoint macro. Run with
# XLA_FLAGS="$XLA_FLAGS --xla_dump_to=DIR --xla_dump_hlo_as_text" and read
# "Total bytes used" from the memory-usage-report in DIR. WHICH=explicit|capture.
defmodule MLPDump do
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

  defn explicit(ws, x) do
    x = checkpoint([x, ws[0], ws[1]], &pair/3)
    x = checkpoint([x, ws[2], ws[3]], &pair/3)
    x = checkpoint([x, ws[4], ws[5]], &pair/3)
    x = checkpoint([x, ws[6], ws[7]], &pair/3)
    Nx.sum(x)
  end

  defn capture_slices(ws, x) do
    {w1, w2, w3, w4, w5, w6, w7, w8} = {ws[0], ws[1], ws[2], ws[3], ws[4], ws[5], ws[6], ws[7]}
    x = checkpoint(x, fn x -> pair(x, w1, w2) end)
    x = checkpoint(x, fn x -> pair(x, w3, w4) end)
    x = checkpoint(x, fn x -> pair(x, w5, w6) end)
    x = checkpoint(x, fn x -> pair(x, w7, w8) end)
    Nx.sum(x)
  end

  defn capture(ws, x) do
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

which = System.get_env("WHICH", "explicit")
fun =
  case which do
    "plain" -> &MLPDump.plain/2
    "explicit" -> &MLPDump.explicit/2
    "capture" -> &MLPDump.capture/2
    "capture_slices" -> &MLPDump.capture_slices/2
  end
g = Nx.Defn.jit(fn ws, x -> Nx.Defn.grad(ws, &fun.(&1, x)) end, compiler: EXLA, client: :cuda).(ws, x)
IO.puts("#{which}: grad sum = #{Nx.to_number(Nx.sum(g))}, abs sum = #{Nx.to_number(Nx.sum(Nx.abs(g)))}")

if System.get_env("COMPARE") do
  ref = Nx.Defn.jit(fn ws, x -> Nx.Defn.grad(ws, &MLPDump.plain(&1, x)) end, compiler: EXLA, client: :cuda).(ws, x)
  diff = Nx.to_number(Nx.reduce_max(Nx.abs(Nx.subtract(g, ref))))
  scale = Nx.to_number(Nx.reduce_max(Nx.abs(ref)))
  IO.puts("#{which} vs plain: max abs diff #{diff}, max abs ref #{scale}, ratio #{diff / scale}")
end
