---
name: Nx.reshape with multiple :auto raises bare ArithmeticError
description: Nx.Shape.reshape/2 resolves the first :auto by dividing by Tuple.product of the rest, but doesn't guard against a second :auto in the shape — the arithmetic hits :auto * 1 and raises a bare ArithmeticError with no Nx context.
type: project
---

# Finding: `Nx.reshape({:auto, :auto})` raises bare `ArithmeticError`

## One-sentence summary

`Nx.reshape(t, {:auto, :auto})` (multiple `:auto` placeholders — a
user error) raises a bare `ArithmeticError: bad argument in
arithmetic expression` from `Tuple.product/2` in `Nx.Shape.reshape/2`,
rather than a helpful `ArgumentError` explaining only one `:auto` is
allowed.

## Minimal repro

```elixir
Nx.iota({12}) |> Nx.reshape({:auto, :auto})
#=> ** (ArithmeticError) bad argument in arithmetic expression
#    :erlang.*(:auto, 1)
#    Tuple.product/2
#    Nx.Shape.reshape/2  (nx/lib/nx/shape.ex:172)
#    Nx.reshape/3
```

## Root cause

`nx/lib/nx/shape.ex:165-181`:

```elixir
case Enum.find_index(Tuple.to_list(new_shape), &(&1 == :auto)) do
  nil -> new_shape
  idx ->
    shape_without_auto = Tuple.delete_at(new_shape, idx)
    inferred_dim = div(old_size, Tuple.product(shape_without_auto))
    ...
end
```

Only the first `:auto` is removed. `Tuple.product(shape_without_auto)`
then multiplies the remaining `:auto` with other dims, hitting the
Erlang `*` operator on an atom.

Proposed fix: count `:auto` occurrences and raise an `ArgumentError`
if `> 1`, e.g.

```elixir
if Enum.count(Tuple.to_list(new_shape), &(&1 == :auto)) > 1 do
  raise ArgumentError,
    "reshape accepts at most one :auto placeholder, got #{inspect(new_shape)}"
end
```

## Classification

**Minor UX.** Not a correctness bug — the user's input is indeed
invalid (only one `:auto` is resolvable). But the error message is
unhelpful and traces through `:erlang.*` rather than naming the
contract violation.

## Test pinning the failure

Not pinned as a test — low-priority UX issue. Fix can land alongside
other shape validation improvements.
