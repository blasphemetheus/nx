# AI-generated. See CHECKPOINT_COMPILER_REVIEW.md for interpretation and authorship.
# Run the same file from each revision's nx/ directory:
#   MIX_ENV=test mix run /absolute/path/checkpoint_overhead.exs --label candidate --out /tmp/candidate.csv
# From exla/, add --exla to include uncached EXLA compilation and synchronized execution.
defmodule CheckpointOverhead do
  @compile {:no_warn_undefined, [EXLA]}
  alias Nx.Defn.{Composite, Tree}

  def run do
    {opts, [], []} =
      OptionParser.parse(System.argv(),
        strict: [label: :string, out: :string, exla: :boolean, samples: :integer, depth: :integer]
      )

    samples = opts[:samples] || 15
    depths = if opts[:depth], do: [opts[:depth]], else: [8, 32, 128]

    modes =
      if function_exported?(Nx.Defn.Kernel, :checkpoint_capture, 2),
        do: [:plain, :explicit, :capture],
        else: [:plain, :explicit]

    label = opts[:label] || "unlabelled"
    compiler = if opts[:exla], do: EXLA, else: Nx.Defn.Evaluator
    if opts[:exla], do: Application.put_env(:exla, :default_client, :host)

    compiler_opts =
      if opts[:exla],
        do: [compiler: EXLA, client: :host, cache: false],
        else: [compiler: Nx.Defn.Evaluator]

    {sha, 0} = System.cmd("git", ["rev-parse", "--short", "HEAD"])

    IO.puts(
      "revision=#{String.trim(sha)} elixir=#{System.version()} otp=#{System.otp_release()} schedulers=#{System.schedulers_online()} samples=#{samples}"
    )

    IO.puts("compiler=#{inspect(compiler)} sizes=64 depths=#{inspect(depths)}")

    IO.puts(
      "Times are microseconds; reductions are for this BEAM process, not native compiler work."
    )

    header =
      "label,revision,compiler,depth,mode,phase,median_us,p25_us,p75_us,median_reductions,nodes"

    rows =
      for depth <- depths, mode <- modes, reduce: [] do
        rows ->
          fun = workload(mode, depth)
          args = [Nx.broadcast(0.2, {64}), Nx.tensor(0.7)]

          grad_fun = fn x, weight ->
            Nx.Defn.grad({x, weight}, fn {x, weight} -> fun.(x, weight) end)
          end

          expected =
            apply(Nx.Defn.jit(workload(:plain, depth), compiler: Nx.Defn.Evaluator), args)

          actual = apply(Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator), args)
          Nx.Testing.assert_all_close(actual, expected)

          phases = [
            {"trace_forward", fn -> Nx.Defn.debug_expr_apply(fun, args) end, fun},
            {"trace_gradient", fn -> Nx.Defn.debug_expr_apply(grad_fun, args) end, grad_fun},
            {"compile_forward", fn -> Nx.Defn.compile(fun, args, compiler_opts) end, fun},
            {"compile_gradient", fn -> Nx.Defn.compile(grad_fun, args, compiler_opts) end,
             grad_fun}
          ]

          rows =
            Enum.reduce(phases, rows, fn {phase, action, traced_fun}, rows ->
              nodes = count_nodes(Nx.Defn.debug_expr_apply(traced_fun, args))
              result = measure(action, samples)
              row = format_row(label, sha, compiler, depth, mode, phase, result, nodes)
              IO.puts(row)
              rows ++ [row]
            end)

          for {phase, target} <- [{"execute_forward", fun}, {"execute_gradient", grad_fun}],
              reduce: rows do
            rows ->
              compiled = Nx.Defn.compile(target, args, compiler_opts)

              action = fn ->
                apply(compiled, args)
                |> Composite.reduce(:ok, fn tensor, :ok ->
                  Nx.to_binary(tensor)
                  :ok
                end)
              end

              result = measure(action, samples)
              row = format_row(label, sha, compiler, depth, mode, phase, result, 0)
              IO.puts(row)
              rows ++ [row]
          end
      end

    if opts[:out], do: File.write!(opts[:out], Enum.join([header | rows], "\n") <> "\n")
  end

  defp workload(mode, depth) do
    fn x, weight ->
      Enum.reduce(1..depth, x, fn _, value ->
        case mode do
          :plain ->
            stage(value, weight)

          :explicit ->
            Nx.Defn.Kernel.checkpoint([value, weight], &stage/2)

          :capture ->
            apply(Nx.Defn.Kernel, :checkpoint_capture, [value, fn v -> stage(v, weight) end])
        end
      end)
      |> Nx.sum()
    end
  end

  defp stage(value, weight), do: Nx.sin(Nx.add(Nx.multiply(value, weight), 0.01))

  defp measure(action, samples) do
    for _ <- 1..3, do: action.()

    results =
      for _ <- 1..samples do
        :erlang.garbage_collect()
        {:reductions, before} = Process.info(self(), :reductions)
        {elapsed, _result} = :timer.tc(action)
        {:reductions, after_count} = Process.info(self(), :reductions)
        {elapsed, after_count - before}
      end

    times = Enum.map(results, &elem(&1, 0)) |> Enum.sort()
    reductions = Enum.map(results, &elem(&1, 1)) |> Enum.sort()

    {quantile(times, 0.5), quantile(times, 0.25), quantile(times, 0.75),
     quantile(reductions, 0.5)}
  end

  defp quantile(sorted, fraction), do: Enum.at(sorted, floor((length(sorted) - 1) * fraction))

  defp format_row(
         label,
         sha,
         compiler,
         depth,
         mode,
         phase,
         {median, p25, p75, reductions},
         nodes
       ),
       do:
         Enum.join(
           [
             label,
             String.trim(sha),
             inspect(compiler),
             depth,
             mode,
             phase,
             median,
             p25,
             p75,
             reductions,
             nodes
           ],
           ","
         )

  defp count_nodes(expr) do
    Composite.reduce(expr, MapSet.new(), &visit/2) |> MapSet.size()
  end

  defp visit(%Nx.Tensor{data: %Nx.Defn.Expr{id: id}} = tensor, seen) do
    if MapSet.member?(seen, id),
      do: seen,
      else:
        Tree.apply_args(tensor, :all, MapSet.put(seen, id), fn child, seen ->
          {child, visit(child, seen)}
        end)
        |> elem(1)
  end
end

Code.ensure_loaded!(Nx.Defn.Kernel)
CheckpointOverhead.run()
