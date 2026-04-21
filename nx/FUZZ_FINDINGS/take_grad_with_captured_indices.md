---
name: Nx.take grad crashes when indices are a defn param or captured concrete tensor
description: Expr.expr_block/3 assumes all args are already Expr and calls parameter/2 on them. When Nx.take's indices are a concrete tensor (closure capture) or a defn parameter re-traced under grad, the concrete indices skip the Expr conversion and parameter/2's pattern match fails with FunctionClauseError.
type: project
---

# Finding: `Nx.take` under grad crashes on concrete-indices inputs

## One-sentence summary

`Nx.Defn.grad` through `Nx.take(tensor, indices)` crashes with
`FunctionClauseError` in `Nx.Defn.Expr.parameter/2` whenever
`indices` reaches `take` as a concrete (BinaryBackend) tensor — which
happens in two common cases: (a) `indices` is closed over from
outer scope in the fun passed to `grad`, or (b) `indices` is a
parameter of the defn being differentiated.

## Minimal repro

```elixir
defmodule P do
  import Nx.Defn
  defn take_sum(t, idx), do: Nx.sum(Nx.take(t, idx))
end

t = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
idx = Nx.tensor([0, 2, 4], type: :s32)

# grad wrt t with indices as defn param — crashes
Nx.Defn.grad(t, fn x -> P.take_sum(x, idx) end)
#=> ** (FunctionClauseError) no function clause matching in
#     Nx.Defn.Expr.parameter/2
#    Nx.Defn.Expr.parameter(#Nx.Tensor< s32[3] [0, 2, 4] >, 1)
#    Enum.with_index_list/3
#    Nx.Defn.Expr.expr_block/3
#    P."__defn:take_sum__"/2
#    Nx.Defn.Grad.transform/3
```

Control that works: indices constructed inline inside the defn body.

```elixir
defn take_sum_inline(t), do: Nx.sum(Nx.take(t, Nx.tensor([0, 2, 4])))
Nx.Defn.grad(t, &take_sum_inline/1)
#=> #Nx.Tensor< f32[5] [1.0, 0.0, 1.0, 0.0, 1.0] >
```

## Expected vs observed

Expected: `Nx.tensor([1.0, 0.0, 1.0, 0.0, 1.0])` — one at each
gathered position.

Observed: `FunctionClauseError` in `Nx.Defn.Expr.parameter/2`.

## Root cause hypothesis

`Nx.take` (`nx/lib/nx.ex:14330`) calls `block(struct, [tensor, indices],
out, fn ... end)`. `Nx.block/4` routes via
`Nx.Shared.list_impl!(args)` (correct — picks `Nx.Defn.Expr` over
`BinaryBackend`). Control then reaches `Nx.Defn.Expr.block/4` →
`Nx.Defn.Expr.expr_block/3` at `nx/lib/nx/defn/expr.ex:423`:

```elixir
defp expr_block(struct, in_args, fun) do
  {args, opts} = Enum.split_while(in_args, &(not is_list(&1)))
  params = Enum.with_index(args, &parameter/2)  # ← crash
  ...
```

`parameter/2` at `expr.ex:72` pattern-matches
`%T{data: %Expr{context: context}}` — a concrete `BinaryBackend`
tensor has `data: %Nx.BinaryBackend.Buffer{}`, so the match fails.

Proposed fix: normalize args to Expr before building parameters:

```elixir
defp expr_block(struct, in_args, fun) do
  {args, opts} = Enum.split_while(in_args, &(not is_list(&1)))
  args = Enum.map(args, &to_expr/1)
  params = Enum.with_index(args, &parameter/2)
  ...
```

The other use of `block/4` in `Nx.ex` is `all_close` (`nx.ex:8938`)
which almost certainly has the same issue when one side is captured.

## Affected ops

Full audit of `Nx.block/4` call sites confirms the bug affects
exactly the three multi-tensor block users:

| Op | Site | Block struct | Notes |
|---|---|---|---|
| `Nx.take` | `nx.ex:14330` | `Nx.Block.Take` | `[tensor, indices]` |
| `Nx.take_along_axis` | `nx.ex:14510` | `Nx.Block.TakeAlongAxis` | `[tensor, indices]` |
| `Nx.all_close` | `nx.ex:8938` | `Nx.Block.AllClose` | `[a, b]` |

All other `block/4` call sites pass a single tensor, so they can't
hit this bug.

## Classification

**New class.** Distinct from the `impl!/1`-dispatch class
([put_slice_grad_mixed_backend_dispatch.md](put_slice_grad_mixed_backend_dispatch.md))
— that one routes to BinaryBackend and crashes in `to_binary/1`. This
one routes correctly to Expr but `expr_block` doesn't normalize its
input tensors, so the crash is in `parameter/2` instead.

Both classes share the same trigger shape (concrete + Expr tensor in
the same call under grad), but land in different crash paths.

## Test pinning the failure

`nx/test/nx/fuzz_indexed_ops_test.exs` — describe block
`"Nx.take grad with concrete indices"` (and mirror test for
`take_along_axis`).
