Mix.install([{:nx, path: "./nx", override: true}])

IO.puts("\n=== LU batched grad ===")

try do
  x =
    Nx.tensor([
      [[4.0, 3.0], [6.0, 3.0]],
      [[2.0, 1.0], [5.0, 7.0]]
    ])

  grad = Nx.Defn.grad(x, fn t ->
    {_p, l, u} = Nx.LinAlg.lu(t)
    Nx.add(Nx.sum(l), Nx.sum(u))
  end)

  IO.puts("OK shape: #{inspect(Nx.shape(grad))}")
  IO.inspect(grad, label: "grad")
rescue
  e -> IO.puts("FAILED: #{Exception.format(:error, e, __STACKTRACE__) |> String.split("\n") |> Enum.take(5) |> Enum.join("\n")}")
end

IO.puts("\n=== LU vectorized grad ===")

try do
  x =
    Nx.tensor([
      [[4.0, 3.0], [6.0, 3.0]],
      [[2.0, 1.0], [5.0, 7.0]]
    ])
    |> Nx.vectorize(:batch)

  grad = Nx.Defn.grad(x, fn t ->
    {_p, l, u} = Nx.LinAlg.lu(t)
    Nx.add(Nx.sum(l), Nx.sum(u))
  end)

  IO.puts("OK shape: #{inspect(Nx.shape(grad))}, vec axes: #{inspect(grad.vectorized_axes)}")
  IO.inspect(grad, label: "grad")
rescue
  e -> IO.puts("FAILED: #{Exception.format(:error, e, __STACKTRACE__) |> String.split("\n") |> Enum.take(5) |> Enum.join("\n")}")
end

IO.puts("\n=== SVD batched grad ===")

try do
  x =
    Nx.tensor([
      [[4.0, 3.0], [6.0, 3.0]],
      [[2.0, 1.0], [5.0, 7.0]]
    ])

  grad = Nx.Defn.grad(x, fn t ->
    {u, s, vt} = Nx.LinAlg.svd(t)
    Nx.add(Nx.add(Nx.sum(u), Nx.sum(s)), Nx.sum(vt))
  end)

  IO.puts("OK shape: #{inspect(Nx.shape(grad))}")
  IO.inspect(grad, label: "grad")
rescue
  e -> IO.puts("FAILED: #{Exception.format(:error, e, __STACKTRACE__) |> String.split("\n") |> Enum.take(5) |> Enum.join("\n")}")
end

IO.puts("\n=== SVD vectorized grad ===")

try do
  x =
    Nx.tensor([
      [[4.0, 3.0], [6.0, 3.0]],
      [[2.0, 1.0], [5.0, 7.0]]
    ])
    |> Nx.vectorize(:batch)

  grad = Nx.Defn.grad(x, fn t ->
    {u, s, vt} = Nx.LinAlg.svd(t)
    Nx.add(Nx.add(Nx.sum(u), Nx.sum(s)), Nx.sum(vt))
  end)

  IO.puts("OK shape: #{inspect(Nx.shape(grad))}, vec axes: #{inspect(grad.vectorized_axes)}")
  IO.inspect(grad, label: "grad")
rescue
  e -> IO.puts("FAILED: #{Exception.format(:error, e, __STACKTRACE__) |> String.split("\n") |> Enum.take(5) |> Enum.join("\n")}")
end
