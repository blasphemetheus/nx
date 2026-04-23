Mix.install([
  {:nx, path: "./nx", override: true},
  {:exla, path: "./exla"}
])

Nx.global_default_backend({EXLA.Backend, client: :host})
Nx.Defn.global_default_options(compiler: EXLA, client: :host)

IO.puts("=== f64 `divide(1, sqrt(x))` CPU sweep ===")
IO.puts("Measuring fraction of inputs where the simplifier's rewrite to")
IO.puts("rsqrt changes the bit pattern from the correctly-rounded")
IO.puts("composition reference `1.0 / :math.sqrt(x)`.\n")

defmodule V do
  def u64(f) when is_float(f) do
    <<u::64>> = <<f::float-64>>
    u
  end

  def ulp_diff(a, b) do
    ua = u64(a)
    ub = u64(b)
    if ua >= ub, do: ua - ub, else: ub - ua
  end

  def log_inputs(n, lo, hi) do
    for i <- 1..n do
      e = lo + (hi - lo) * (i - 1) / (n - 1)
      :math.pow(10.0, e)
    end
  end
end

# 1000 log-uniform inputs over the regular range.
inputs = V.log_inputs(1000, -200, 200)

# Curated inputs that showed divergence in the bit-exact unit test.
curated = [2.0, 3.0, 0.5, 1.5, 7.0, :math.pi(), :math.exp(1)]
all_inputs = curated ++ inputs

batch = Nx.tensor(all_inputs, type: :f64)

# User-written form: `1.0 / sqrt(x)`. On XLA main (which EXLA bundles)
# the algebraic simplifier rewrites this to `multiply(1, rsqrt(x))`,
# so what runs is effectively the rsqrt emit path.
div_sqrt_results = batch |> Nx.sqrt() |> then(&Nx.divide(1.0, &1)) |> Nx.to_list()

# For reference, compute rsqrt directly — should land on the same path
# post-simplifier since they lower to the same HLO.
rsqrt_results = batch |> Nx.rsqrt() |> Nx.to_list()

# Host reference: correctly-rounded composition, computed outside XLA
# using Erlang's :math.sqrt (which delegates to libm's sqrt, also
# correctly rounded on glibc).
references = Enum.map(all_inputs, fn x -> 1.0 / :math.sqrt(x) end)

# Count divergences for each EXLA path against the reference.
stats =
  Enum.zip([all_inputs, div_sqrt_results, rsqrt_results, references])
  |> Enum.reduce(
    %{div_mismatch: 0, rsqrt_mismatch: 0, div_ulp_hist: %{}, rsqrt_ulp_hist: %{}},
    fn {x, dv, rv, ref}, acc ->
      d_ulp = V.ulp_diff(dv, ref)
      r_ulp = V.ulp_diff(rv, ref)

      acc
      |> Map.update!(:div_mismatch, fn c -> c + if(d_ulp > 0, do: 1, else: 0) end)
      |> Map.update!(:rsqrt_mismatch, fn c -> c + if(r_ulp > 0, do: 1, else: 0) end)
      |> Map.update!(:div_ulp_hist, fn h -> Map.update(h, d_ulp, 1, &(&1 + 1)) end)
      |> Map.update!(:rsqrt_ulp_hist, fn h -> Map.update(h, r_ulp, 1, &(&1 + 1)) end)
      |> tap(fn _ ->
        if x in curated and d_ulp > 0 do
          IO.puts(
            "  curated #{x}: Nx.divide(1, Nx.sqrt(x)) = #{dv}, reference = #{ref}, ULP diff = #{d_ulp}"
          )
        end
      end)
    end
  )

total = length(all_inputs)

IO.puts("\n--- Totals (sample size: #{total}) ---")

div_pct = :erlang.float_to_binary(100 * stats.div_mismatch / total, decimals: 2)
rsqrt_pct = :erlang.float_to_binary(100 * stats.rsqrt_mismatch / total, decimals: 2)

IO.puts("Nx.divide(1.0, Nx.sqrt(x)) vs reference:  #{stats.div_mismatch} / #{total}  (#{div_pct}%) mismatched")

IO.puts("Nx.rsqrt(x)                vs reference:  #{stats.rsqrt_mismatch} / #{total}  (#{rsqrt_pct}%) mismatched")

IO.puts("\nULP histogram for Nx.divide(1, Nx.sqrt(x)):")

for {ulp, count} <- Enum.sort(stats.div_ulp_hist) do
  pct = :erlang.float_to_binary(100 * count / total, decimals: 2)
  IO.puts("  #{ulp} ULP: #{count} (#{pct}%)")
end

IO.puts("\nULP histogram for Nx.rsqrt(x):")

for {ulp, count} <- Enum.sort(stats.rsqrt_ulp_hist) do
  pct = :erlang.float_to_binary(100 * count / total, decimals: 2)
  IO.puts("  #{ulp} ULP: #{count} (#{pct}%)")
end
