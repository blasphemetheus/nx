---
name: Nx.from_binary rejects bitstrings that Nx.to_binary produces for sub-byte types
description: Nx.to_binary's docs acknowledge it may return a bitstring (not byte-aligned) for u2/u4/s2/s4 tensors; Nx.from_binary's guard is `when is_binary(binary)` (byte-aligned only) so the documented inverse round-trip fails with FunctionClauseError for any sub-byte tensor whose total bit count isn't a multiple of 8.
type: project
---

# Finding: `to_binary` → `from_binary` asymmetry for sub-byte types

## One-sentence summary

`Nx.to_binary/1` returns a **bitstring** (not byte-aligned) for
`u2`/`u4`/`s2`/`s4` tensors whose total bit count isn't a multiple
of 8 — but `Nx.from_binary/3` has a `when is_binary(binary)` guard
that rejects bitstrings, so the documented round-trip crashes with
`FunctionClauseError`.

## Minimal repro

```elixir
# 3 u4 values = 12 bits (not byte-aligned)
t = Nx.tensor([0, 7, 15], type: :u4)
bin = Nx.to_binary(t)
#=> <<7, 15::size(4)>>   — a 12-bit bitstring

Nx.from_binary(bin, :u4)
#=> ** (FunctionClauseError) no function clause matching in Nx.from_binary/3
#     Nx.from_binary(<<7, 15::size(4)>>, :u4, [])
```

Affected combinations (any sub-byte type whose `n_elements * bit_size`
isn't divisible by 8):

| type | element counts that fail |
|---|---|
| `:u2` / `:s2` | 1, 2, 3, 5, 6, 7, 9, … (anything where `2n % 8 ≠ 0`) |
| `:u4` / `:s4` | 1, 3, 5, 7, … (odd counts) |

Aligned cases work — e.g. 4 `:u4` elements (16 bits) round-trip fine.

## Expected vs observed

Expected: `Nx.from_binary(Nx.to_binary(t), type) |> Nx.reshape(Nx.shape(t))`
returns a tensor equal to `t`. This is the standard round-trip and
is the only way to interop with raw byte-level storage.

Observed: `FunctionClauseError` on any sub-byte tensor with a
non-aligned bit count.

## Root cause

`nx/lib/nx.ex:1998`:

```elixir
def from_binary(binary, type, opts \\ []) when is_binary(binary) do
  ...
  if rem(Kernel.bit_size(binary), size) != 0 do
    raise ArgumentError, "binary does not match the given size"
  end
  ...
```

The `is_binary` guard rejects bitstrings outright — before the
internal `bit_size/size` check even runs. Meanwhile `to_binary`
(`nx/lib/nx.ex:2034`) explicitly documents that it may return a
bitstring for sub-byte types.

Proposed fix: relax the guard to `is_bitstring` — the existing
`rem(bit_size, size) != 0` check already handles malformed inputs.

```elixir
def from_binary(binary, type, opts \\ []) when is_bitstring(binary) do
  ...
```

## Classification

**New class.** Unrelated to the dispatch bugs — this is an API-surface
asymmetry. `to_binary` is generous (bitstring for sub-byte),
`from_binary` is strict (binary only). The documented idiom is
supposed to be symmetric.

`Nx.serialize`/`Nx.deserialize` are NOT affected — they have their
own wire format and handle sub-byte types correctly.

## Test pinning the failure

`nx/test/nx/fuzz_serialization_test.exs` — describe block
`"sub-byte binary round-trip (from_binary_rejects_sub_byte_bitstrings)"`.
