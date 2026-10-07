# EXLA: independent io_calls fire out of program order on the CUDA client

Found 2026-10-06 by the first run of upstream's own `exla/` suite on CUDA
after the v1.0 merge (`EXLA.Defn.APITest` "executes independent io_calls in
program order"). Deterministic across runs; passes on the host client.

## Repro

```elixir
defn chain(a, b, c) do
  a = io_call(a, :a)
  b = io_call(b, :b)
  c = io_call(c, :c)
  a + c + b
end
# host: callbacks fire a, b, c
# cuda: callbacks fire b, c, a  (every run)
```

Scratch script: `scratchpad/io_order.exs` pattern — jit with hooks that
`send` their name to the test process, then drain the mailbox.

## Mechanism

`EXLA.Defn` lowers each `io_call` to its own `stablehlo.custom_call`
(`exla_runtime_callback`, `has_side_effect = true`) with no data or token
dependency between successive calls. In the post-optimization HLO the three
custom-calls are mutually independent and the ROOT fusion consumes all
three. XLA only promises that side-effecting ops are not removed or
duplicated; relative order of independent side-effecting ops is up to the
backend scheduler. The CPU emitter keeps emission order; the GPU
latency-hiding scheduler does not (`module_0001.main.sm_12.0a_gpu_after_optimizations.txt`
lists them as `custom-call.5, .4, .3`).

The io_call docs show two io_calls printing in program order and the
upstream test asserts it, so program order is the intended contract.

## Fix shape (EXLA, not Nx)

Thread a token through the callbacks: `Value.host_callback` takes the
current token as an operand and returns a fresh one alongside the data
outputs; `defn.ex` keeps the token in the cache the way the infeed path
already does (`Outfeed.with_token`, `get_token`/`update_token`,
`Value.create_token`). The FFI handlers in
`c_src/exla/custom_calls/runtime_callback{,_cuda}.cc` use `RemainingArgs`
/ `RemainingRets`, so they need a token arg/ret added. Token-chained
custom calls form a dependency chain the GPU scheduler must respect.

Not yet reported upstream.
