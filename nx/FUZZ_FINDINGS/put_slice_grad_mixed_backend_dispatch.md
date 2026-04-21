---
name: Multi-tensor Nx op dispatch uses impl!(tensor) — crashes on mixed concrete+Expr args
description: Nx.put_slice, Nx.gather, and Nx.clip dispatch via impl!(first_tensor) only. When that one tensor is concrete BinaryBackend and a co-tensor argument is Expr (as under grad re-trace with a captured outer tensor), BinaryBackend is picked and crashes on to_binary(expr).
type: project
---

# Finding: multi-tensor op dispatch ignores co-tensor args (class)

## One-sentence summary

Five `Nx` ops — `put_slice`, `gather`, `clip`, `reduce`,
`window_reduce` — dispatch via `impl!(first_tensor)` only. When the
first tensor is concrete (BinaryBackend) and a co-tensor argument is
`Nx.Defn.Expr`, the dispatch picks `BinaryBackend`, which immediately
crashes in `to_binary/1` on the Expr.

## Scope is broader than grad

This is **not** a grad-specific bug. The trigger is "any defn
compilation where the body sees a closure-captured concrete tensor
alongside a defn parameter." That includes:

- `Nx.Defn.grad(p, fn p -> f(captured_t, p) end)` — grad re-trace
- `Nx.Defn.jit(fn p -> f(captured_t, p) end).(p)` — plain jit with a
  closure
- `defn foo(p), do: put_slice(@captured, [0], p)` — module attribute
  holding a concrete tensor (both compile-time literals and runtime
  assignments)

All three forms route a concrete tensor + Expr param into the same
`Nx.put_slice` / `Nx.clip` / etc. call and crash. Normal user
patterns like "load weights at module load, pass batches as defn
params" can hit this if `put_slice`/`clip`/`gather`/`reduce`/
`window_reduce` appears in the body.

## Minimal repro

```elixir
defmodule P do
  import Nx.Defn
  defn put_slice_sum(t, patch), do: Nx.sum(Nx.put_slice(t, [1], patch))
end

t = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
patch = Nx.tensor([10.0, 20.0])

# grad wrt target (t) — OK
Nx.Defn.grad(t, fn x -> P.put_slice_sum(x, patch) end)
#=> #Nx.Tensor< f32[5] [1.0, 0.0, 0.0, 1.0, 1.0] >

# grad wrt update (patch) — crashes
Nx.Defn.grad(patch, fn p -> P.put_slice_sum(t, p) end)
#=> ** (FunctionClauseError) no function clause matching in
#     Nx.BinaryBackend.to_binary/1
#    Nx.BinaryBackend.put_slice/5
#    Nx.put_slice/3
#    P."__defn:put_slice_sum__"/2
#    Nx.Defn.Grad.transform/3
```

The crash also reproduces with both `t` and `patch` as defn parameters
of the outer fun, and with a bare anonymous fn (no nested defn). It
does **not** reproduce when `t` is constructed *inside* the defn (e.g.
`t = Nx.tensor([...])` or `Nx.broadcast(0.0, {5})`), because the inner
literal is an Expr node rather than a concrete BinaryBackend tensor.

## Expected vs observed

Expected: `Nx.Defn.grad(patch, fn p -> P.put_slice_sum(t, p) end)`
returns `Nx.tensor([1.0, 1.0])` (same shape as patch; sum's derivative
is 1 at each inserted cell).

Observed: `FunctionClauseError` in `Nx.BinaryBackend.to_binary/1` with
the Expr parameter as the offending argument.

## Root cause hypothesis

`nx/lib/nx.ex:14046` dispatches `Nx.put_slice` via

```elixir
result =
  impl!(tensor).put_slice(
    %{tensor | shape: ..., names: ..., type: ...},
    tensor,
    start_indices,
    slice
  )
```

Compare to `Nx.pad` at `nx/lib/nx.ex:4117`:

```elixir
impl!(tensor, pad_value).pad(out, tensor, pad_value, padding_config)
```

`Nx.Shared.impl!/2` calls `pick_struct/2` which always prefers a
non-`Nx.BinaryBackend` struct (see `nx/lib/nx/shared.ex:467-468`). So
when put_slice looks only at `tensor`'s struct, a concrete target +
Expr update picks `BinaryBackend`, and
`Nx.BinaryBackend.put_slice/5` at `nx/lib/nx/binary_backend.ex:1922`
immediately calls `to_binary(slice)` on the Expr — crash.

Proposed fix (single-line):

```elixir
result = impl!(tensor, slice).put_slice(...)
```

This matches the `Nx.pad` pattern and lets `pick_struct` choose the
Expr backend whenever either argument is an Expr.

## Sibling ops with the same bug

Quick audit of `impl!/1` sites in `nx/lib/nx.ex` that take a co-tensor
argument — all reproduced with the same `to_binary/1` crash pattern:

| Op | Site | Co-tensor arg | Grad repro that crashes |
|---|---|---|---|
| `Nx.put_slice` | `nx.ex:14046` | `slice` | grad wrt update with captured target |
| `Nx.gather` | `nx.ex:14724` | `indices` | grad wrt indices with captured source (even if indices gradient is semantically meaningless, dispatch should still route to Expr, not crash) |
| `Nx.clip` | `nx.ex:13520` | `min`, `max` | grad wrt `min` or `max` with captured tensor |

Ops that already dispatch correctly:

| Op | Site | Pattern |
|---|---|---|
| `Nx.pad` | `nx.ex:4117` | `impl!(tensor, pad_value).pad(...)` |
| `Nx.conv` | `nx.ex:13375` | `impl!(tensor, kernel).conv(...)` |
| `Nx.dot` | `nx.ex:12679,12682` | `impl!(t1, t2).dot(...)` |
| `Nx.concatenate` | `nx.ex:14881` | `list_impl!(tensors).concatenate(...)` |

## Classification

**New class.** Not related to the batched-input `custom_grad` family
(meta #1748) or the `Nx.Defn.while` reverse-mode AD bug (#1747). This
is a dispatch/routing bug — the `impl!/2-3` and `list_impl!/1` helpers
in `Nx.Shared` exist specifically to handle this case (see
`pick_struct/2` preferring any non-BinaryBackend struct), but several
multi-tensor ops call the single-arg `impl!/1` variant instead.

Remaining sites to audit for the same pattern (not yet probed):

- `nx.ex:13814` `Nx.slice(tensor, start_indices, ...)` — `start_indices`
  are normalized via `to_indices/1` first, so it may be immune; needs
  an Expr-index probe to confirm.
- `nx.ex:12005` `reduce(out, tensor, acc, ..., fun)` — `acc` is a
  tensor.
- `nx.ex:12142` `window_reduce(out, tensor, acc, ...)` — `acc` is a
  tensor.
- `nx.ex:15237,15493` `sort/argsort` — single tensor, probably OK.

## Test pinning the failure

`nx/test/nx/fuzz_indexed_ops_test.exs` — see new describe block
`"Nx.put_slice grad wrt update (#put_slice_grad_mixed_backend)"`.
