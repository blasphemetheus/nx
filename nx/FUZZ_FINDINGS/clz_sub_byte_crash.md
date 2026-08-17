# Sub-byte bit counting is broken twice: clz crashes, s2 counts wrap

Found 2026-08-17 by coverage-guided gap finding — the very first execution
of a dark line. `element_clz2/1` showed as never-executed under the whole
test suite; targeting it revealed it is UNREACHABLE: dead code behind a
missing dispatcher arm.

## Minimal repro

```elixir
Nx.count_leading_zeros(Nx.tensor(1, type: :u2))
# ** (FunctionClauseError) no function clause matching in Nx.BinaryBackend.element_clz/2
```

Crashes for any u2/u4/s2/s4 tensor containing a nonzero element. Zero
values survive via the `element_clz(0, size)` clause; `population_count`
is unaffected (width-agnostic implementation).

## Cause

`Nx.BinaryBackend.element_clz/2` dispatches on bit width with clauses for
64/32/16/8 only:

```elixir
defp element_clz(0, size), do: size
defp element_clz(n, 64), do: element_clz64(n)
defp element_clz(n, 32), do: element_clz32(n)
defp element_clz(n, 16), do: element_clz16(n)
defp element_clz(n, 8),  do: element_clz8(n)
# sizes 4 and 2 fall through -> FunctionClauseError
```

The width-2 helper (`element_clz2/1`) already exists further down the
module (as is `element_clz4/1` — both are used internally by the clz8
chain). Fix shape: two dispatcher clauses, `element_clz(n, 4)` and
`element_clz(n, 2)`, delegating to the existing helpers.

## Second defect: s2 results wrap (silent wrong value)

Bit-count ops return the INPUT type, and s2 cannot represent count 2:

```elixir
Nx.population_count(Nx.tensor(-1, type: :s2))
# bit pattern 11, true count 2 -> returns -2 (wrapped into s2)
```

Also affects clz(0) on s2 (true answer 2 -> -2), and will affect any
fixed clz on s2. u2/u4/s4 can represent their maximum counts; only s2
cannot. Fix direction is a design decision: widen the output type for
sub-byte inputs, or saturate, or document.

## Pinned

`fuzz_int_semantics_test.exs` — popcount property covers u2/u4/s4;
`[BUG-CLZ-SUBBYTE]` pins the nonzero crash and the s2 zero-wrap;
`[BUG-POPCOUNT-S2-WRAP]` pins the s2 popcount wrap. Flip when fixed.

## Priority

**MED** — clean crash (not silent), narrow surface (sub-byte clz), but a
complete feature hole with the fix half-written in the codebase already.
