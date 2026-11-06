# AI-generated. Summarize the recorded runs without external dependencies.
# elixir nx/bench/checkpoint_compare.exs nx/bench/checkpoint_results
directory = List.first(System.argv()) || Path.join(__DIR__, "checkpoint_results")
median = fn values -> Enum.at(Enum.sort(values), div(length(values), 2)) end

rows =
  for file <- Path.wildcard(Path.join(directory, "*.csv")),
      line <- File.read!(file) |> String.split("\n", trim: true) |> tl() do
    [label, revision, compiler, depth, mode, phase, time, p25, p75, reductions, nodes] =
      String.split(line, ",")

    %{
      label: label,
      revision: revision,
      compiler: compiler,
      depth: String.to_integer(depth),
      mode: mode,
      phase: phase,
      time: String.to_integer(time),
      p25: String.to_integer(p25),
      p75: String.to_integer(p75),
      reductions: String.to_integer(reductions),
      nodes: String.to_integer(nodes)
    }
  end

groups =
  Enum.group_by(rows, fn row ->
    kind = if String.contains?(row.label, "baseline"), do: :baseline, else: :candidate
    {row.compiler, row.depth, row.mode, row.phase, kind}
  end)

IO.puts(
  "| Compiler | Depth | Phase | Baseline mode | Candidate mode | Baseline µs | Candidate µs | Time change | Reduction change |"
)

IO.puts("|---|---:|---|---|---|---:|---:|---:|---:|")

for compiler <- ["Nx.Defn.Evaluator", "EXLA"],
    depth <- [8, 32, 128],
    phase <- [
      "trace_forward",
      "trace_gradient",
      "compile_forward",
      "compile_gradient",
      "execute_forward",
      "execute_gradient"
    ],
    {base_mode, candidate_mode} <- [
      {"plain", "plain"},
      {"explicit", "explicit"},
      {"explicit", "capture"}
    ] do
  base = groups[{compiler, depth, base_mode, phase, :baseline}]
  candidate = groups[{compiler, depth, candidate_mode, phase, :candidate}]

  if base && candidate do
    base_time = median.(Enum.map(base, & &1.time))
    candidate_time = median.(Enum.map(candidate, & &1.time))
    base_reductions = median.(Enum.map(base, & &1.reductions))
    candidate_reductions = median.(Enum.map(candidate, & &1.reductions))
    time_delta = Float.round(100 * (candidate_time / base_time - 1), 1)
    reductions_delta = Float.round(100 * (candidate_reductions / base_reductions - 1), 1)

    IO.puts(
      "| #{compiler} | #{depth} | #{phase} | #{base_mode} | #{candidate_mode} | #{base_time} | #{candidate_time} | #{time_delta}% | #{reductions_delta}% |"
    )
  end
end
