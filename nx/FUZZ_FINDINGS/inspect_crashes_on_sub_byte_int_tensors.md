---
name: Inspect crashes on sub-byte integer tensors (u2/u4/s2/s4)
description: Nx.Backend.inspect/3's chunk helper uses `tail::binary` in the :s and :u branches, which requires byte-aligned remainder. After consuming any sub-byte element, the remaining bits aren't byte-aligned, triggering MatchError. Affects any attempt to print or inspect a u2/u4/s2/s4 tensor on BinaryBackend AND Torchx.
type: project
---

# Finding: `Nx.Backend.inspect/3` crashes on sub-byte integer tensors

## One-sentence summary

`IO.inspect` (and anything that calls the `Inspect` protocol) on any
`u2`/`u4`/`s2`/`s4` tensor crashes with `MatchError` inside
`Nx.Backend.chunk/5` because the integer branches use `tail::binary`
instead of `tail::bitstring`, and sub-byte integers leave the
bitstring tail misaligned after consuming one element.

## Minimal repro

```elixir
iex> t = Nx.tensor([0, 1, 2, 3], type: :u4)
iex> IO.inspect(t)
#Inspect.Error<
  got MatchError with message:

      """
      no match of right hand side value: <<1, 35>>
      """

  while inspecting:

      %{
        data: %Nx.BinaryBackend{state: <<1, 35>>},
        type: {:u, 4},
        ...
      }

  Stacktrace:
    (nx 0.11.0) lib/nx/backend.ex:174: Nx.Backend.chunk/5
    (nx 0.11.0) lib/nx/backend.ex:221: Nx.Backend.chunk_each/5
    (nx 0.11.0) lib/nx/backend.ex:192: Nx.Backend.chunk/5
    (nx 0.11.0) lib/nx/backend.ex:161: Nx.Backend.inspect/3
>
```

Affects every backend — the inspect code is in the shared
`Nx.Backend` module. Verified on both `Nx.BinaryBackend` and
`Torchx.Backend` with types `:u4` and `:s2`.

## Root cause

`nx/lib/nx/backend.ex:166-188`:

```elixir
defp chunk([], data, type, limit, _docs) do
  {doc, tail} =
    case type do
      {:s, size} ->
        <<seg::size(^size)-signed-integer-native, tail::binary>> = data
        # ^ tail::binary requires byte alignment — fails on sub-byte
        {Integer.to_string(seg), tail}

      {:u, size} ->
        <<seg::size(^size)-unsigned-integer-native, tail::binary>> = data
        # ^ same bug
        {Integer.to_string(seg), tail}

      {:c, size} ->
        # complex halves are always byte-aligned (c64 = 32+32, c128 = 64+64)
        ...

      {type, size} ->
        <<float::size(^size)-bitstring, tail::bitstring>> = data
        # ^ correct — uses bitstring
        ...
    end
  ...
end
```

For a `u4` tensor with N elements: total bits = 4N. After consuming
one 4-bit element, the tail has `4(N-1)` bits. For `N = 2`: 4 bits.
For `N = 3`: 8 bits (byte-aligned, works by accident). For `N = 4`:
12 bits — not byte-aligned, `tail::binary` match fails.

So certain element counts happen to work; others crash. Users get
unexpected inspect failures depending on shape.

Proposed fix: use `tail::bitstring` in both integer branches,
matching the float branch pattern at line 183.

```elixir
{:s, size} ->
  <<seg::size(^size)-signed-integer-native, tail::bitstring>> = data
  {Integer.to_string(seg), tail}

{:u, size} ->
  <<seg::size(^size)-unsigned-integer-native, tail::bitstring>> = data
  {Integer.to_string(seg), tail}
```

## Related finding

Same theme as [from_binary_rejects_sub_byte_bitstrings.md](from_binary_rejects_sub_byte_bitstrings.md):
Nx's sub-byte support has several places where `is_binary`/`tail::binary`
is used when it should be `is_bitstring`/`tail::bitstring`. A full
audit would be worthwhile.

## Classification

**New bug.** Sub-byte-specific. Crashes a core user operation
(inspect) on any u2/u4/s2/s4 tensor with element count
`N` where `N * bit_size mod 8 != 0` in any intermediate step of
chunked inspection.

## Test pinning the failure

`nx/test/nx/fuzz_serialization_test.exs` — a new describe block
`"sub-byte inspect (inspect_crashes_on_sub_byte_int_tensors)"`.
