Mix.install([
  {:complex, github: "elixir-nx/complex", override: true},
  {:nx, path: "./nx", override: true},
  {:exla, path: "./exla"}
])

cases = [
  {"2D f32", fn -> Nx.tensor([[4.0, 2.0], [2.0, 5.0]], type: :f32) end},
  {"2D f64", fn -> Nx.tensor([[4.0, 2.0], [2.0, 5.0]], type: :f64) end},
  {"3D f32", fn -> Nx.tensor([[[4.0, 2.0], [2.0, 5.0]]], type: :f32) end},
  {"3D f64", fn -> Nx.tensor([[[4.0, 2.0], [2.0, 5.0]]], type: :f64) end}
]

grad_fn = fn a ->
  {s, _v} = Nx.LinAlg.eigh(a)
  Nx.sum(s)
end

run_case = fn {label, build_x} ->
  try do
    x = build_x.()
    g = Nx.Defn.grad(x, grad_fn)
    {label, :ok, inspect(Nx.shape(g))}
  rescue
    e ->
      msg =
        e
        |> Exception.message()
        |> String.split("\n")
        |> Enum.take(2)
        |> Enum.join(" / ")
        |> String.slice(0, 160)

      {label, :fail, msg}
  end
end

format_row = fn {label, status, detail} ->
  symbol = if status == :ok, do: "ok", else: "FAIL"
  "| #{label} | #{symbol} | `#{detail}` |"
end

section = fn name, results ->
  IO.puts("\n## #{name}")
  IO.puts("| Case | Status | Detail |")
  IO.puts("|------|--------|--------|")
  Enum.each(results, fn row -> IO.puts(format_row.(row)) end)
end

IO.puts("=== BinaryBackend ===")
Nx.global_default_backend(Nx.BinaryBackend)
Nx.Defn.global_default_options(compiler: Nx.Defn.Evaluator)
binary_results = Enum.map(cases, run_case)
section.("BinaryBackend", binary_results)

IO.puts("\n=== EXLA ===")
Nx.global_default_backend({EXLA.Backend, client: :host})
Nx.Defn.global_default_options(compiler: EXLA)
exla_results = Enum.map(cases, run_case)
section.("EXLA (client: :host, compiler: EXLA)", exla_results)

IO.puts("\n## Combined matrix")
IO.puts("| Case | BinaryBackend | EXLA |")
IO.puts("|------|---------------|------|")

Enum.zip(binary_results, exla_results)
|> Enum.each(fn {{label, b_status, b_detail}, {_, e_status, e_detail}} ->
  b = if b_status == :ok, do: "ok (#{b_detail})", else: "FAIL: #{b_detail}"
  e = if e_status == :ok, do: "ok (#{e_detail})", else: "FAIL: #{e_detail}"
  IO.puts("| #{label} | #{b} | #{e} |")
end)
