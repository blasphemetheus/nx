# AI-generated. See ../RECOMPUTE_PER_USE.md for the experiment and authorship poem.
# Run from nx/: mix run ../examples/recompute_each_use.exs
# Run from exla/: mix run ../examples/recompute_each_use.exs --exla

defmodule RecomputeExample do
  import Nx.Defn
  @compile {:no_warn_undefined, [EXLA, EXLA.Executable]}

  defn expanded(row, column) do
    Nx.exp(Nx.new_axis(row, 1) + Nx.new_axis(column, 0))
  end

  defn plain(row, column) do
    weights = expanded(row, column)
    consume(weights)
  end

  defn rebuilt(row, column) do
    weights = recompute_each_use([row, column], &expanded/2)
    consume(weights)
  end

  defnp consume(weights) do
    first = Nx.mean(weights, axes: [1])
    Nx.sum(weights * Nx.new_axis(Nx.sin(first), 1))
  end

  def counts do
    IO.puts("Evaluator: a chain of recipes, then y + y")

    for depth <- [1, 2, 5, 10] do
      Process.put(:recipe_calls, 0)

      result =
        Nx.Defn.jit(fn input ->
          output =
            Enum.reduce(1..depth, input, fn _, value ->
              Nx.Defn.Kernel.recompute_each_use(value, fn tensor ->
                # Test instrumentation counts execution, not symbolic tracing.
                unless match?(%Nx.Tensor{data: %Nx.Defn.Expr{}}, tensor) do
                  Process.put(:recipe_calls, Process.get(:recipe_calls) + 1)
                end

                Nx.sin(tensor)
              end)
            end)

          Nx.add(output, output)
        end).(Nx.tensor(0.2))

      IO.puts(
        "depth=#{depth} body_calls=#{Process.get(:recipe_calls)} result=#{Nx.to_number(result)}"
      )
    end
  end

  def compare(use_exla) do
    size = 512
    row = Nx.iota({size}, type: :f32) |> Nx.divide(size * 4)
    column = Nx.negate(row)
    compiler = if use_exla, do: EXLA, else: Nx.Defn.Evaluator
    Nx.Defn.default_options(compiler: compiler)
    IO.puts("Two #{size}-element inputs: #{2 * size * 4} bytes")
    IO.puts("One #{size}x#{size} derived value: #{size * size * 4} bytes")
    expected = plain(row, column)
    actual = rebuilt(row, column)
    Nx.Testing.assert_all_close(actual, expected)

    for {name, fun} <- [{"plain", &plain/2}, {"per_use", &rebuilt/2}] do
      compiled = Nx.Defn.compile(fun, [row, column], compiler: compiler)
      compiled.(row, column) |> Nx.to_number()

      times =
        for _ <- 1..5 do
          {time, _} = :timer.tc(fn -> compiled.(row, column) |> Nx.to_number() end)
          time
        end

      IO.puts("#{name}: median warm execution #{Enum.at(Enum.sort(times), 2)} us")

      if use_exla do
        executable = EXLA.to_executable(fun, [row, column])
        IO.inspect(EXLA.Executable.memory_stats(executable), label: "#{name} compiled memory")
        text = EXLA.Executable.optimized_hlo(executable)
        count = length(String.split(text, "exponential(")) - 1
        IO.puts("#{name}: optimized exponential instructions=#{count}")
      end
    end
  end
end

RecomputeExample.counts()
use_exla = "--exla" in System.argv()
if use_exla, do: Application.put_env(:exla, :default_client, :host)
RecomputeExample.compare(use_exla)
