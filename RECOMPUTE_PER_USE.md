# Per-use recomputation experiment

**AI-generated — implementation and explanation by Codex, 2026-11-06.**

This is a separate feature from gradient checkpointing. It asks the evaluator and EXLA to rebuild a marked tensor at each operand occurrence. Its explicit inputs remain shared. There is no new gradient rule.

## Start with the behavior

```elixir
defn example(x) do
  y = recompute_each_use(x, &Nx.exp/1)
  y * y
end
```

An ordinary expression evaluates the exponential once and shares its result. This prototype's evaluator runs the recipe twice, once for each multiplication operand. Returning a marked tensor is also a use. Assigning it to another variable is not a use: variable names do not create graph operations.

A consumer means an operation in the traced graph, not each textual mention in the source. If you compute an ordinary shared expression from the marked tensor, that ordinary expression still has normal caching. The marker does not automatically propagate through subsequent operations.

Without tracing, the call executes the callback normally and returns a tensor. Other compilers may treat the marker as an ordinary block. EXLA lowers distinct copies and uses dependencies to discourage their merging or early scheduling; arbitrary optimized programs do not have an exact instruction-count guarantee.

## A possible use: small inputs, a large derived result

The runnable example creates a matrix from two vectors:

```elixir
weights =
  recompute_each_use([row, column], fn r, c ->
    Nx.exp(Nx.new_axis(r, 1) + Nx.new_axis(c, 0))
  end)

first = Nx.mean(weights, axes: [1])
Nx.sum(weights * Nx.new_axis(Nx.sin(first), 1))
```

The second consumer depends on the first reduction. Keeping the matrix between those consumers costs more storage than keeping its two input vectors. Rebuilding the matrix near the later multiplication may let the compiler combine the reconstruction with that multiplication and reduction.

For 512-element f32 vectors, the inputs total 4,096 bytes and the matrix contains 1,048,576 bytes. On one local EXLA CPU run:

| Version | Compiled scratch bytes | Optimized exponential instructions | Median of five warm executions |
|---|---:|---:|---:|
| Plain | 1,050,624 | 1 | 362 µs |
| Per-use | 4,096 | 2 | 211 µs |

Values agree. These are measurements of this example on this machine, not a general performance claim. The timing sample is small. Scratch excludes arguments and outputs; it is not total process memory. Fusion and scheduling explain why doing more arithmetic can sometimes also run faster.

The evaluator run was slower: roughly 112 ms plain versus 179 ms per-use. It performs the extra work directly and does not gain EXLA fusion. That contrast is part of the tradeoff.

Potential applications include generated masks, coordinate-dependent fields, or other cheap expanded values. Those applications remain hypotheses until measured. Expensive matrix products or recipes with inputs as large as their outputs are less attractive.

## The deliberate chain limit

The recipe keeps evaluated inputs. For a chain of ten marked sine operations followed by `y + y`, the evaluator runs eleven bodies: nine upstream bodies once, then the last body twice. The tests and runner cover depths 1, 2, 5, and 10.

That bounded work has a cost: each downstream recipe holds its upstream result as an input. This is not recursively discarding every activation. Replacing those retained inputs with recipes would be another policy, and could repeat much more work. This is why this experiment should not replace A's backward checkpointing.

## How the implementation works

1. `recompute_each_use/2` creates a `Nx.Block.Recompute` with explicit inputs. The callback returns one tensor.
2. The evaluator's cache stores the callback and evaluated inputs, rather than the output. Each read executes the recipe. Inputs and results are kept non-donatable so a backend cannot overwrite buffers still needed by a later execution.
3. EXLA caches the recipe's input operands but lowers its body again at a use. For ordinary consumer operations, it lowers other operands first and places the recipe inputs and those operands through one optimization barrier.
4. The consumer uses the barrier's output operands too. Using its original operands would leave the dependency outputs unused, allowing optimization to remove the intended ordering.
5. For repeated operands such as `y * y`, the next reconstruction is tied to the earlier reconstruction. This preserves separate work in the tested compiled example, but can restrict parallel scheduling.

Start in `nx/lib/nx/defn/kernel.ex`, then the `:recompute` cache cases in `nx/lib/nx/defn/evaluator.ex`. The EXLA changes are `lower_recompute_block/4` and `lower_recompute_args/3`. Unlike the closure prototypes, this does not discover implicit dependencies or rewrite callback source.

The earlier local per-use block experiment informed this implementation. This version has its own name and marker, keeps A unchanged, retains recipe inputs in both implementations, and tests the chosen operand-occurrence contract.

## Limits and validation

The body must be pure, straight-line tensor computation. This prototype rejects nested blocks, conditionals, loops, runtime calls, IO calls, custom-gradient metadata, and non-tensor outputs inside the recipe. Static configuration can be captured, but runtime tensors must be explicit inputs. Validation cannot detect arbitrary Elixir side effects executed while tracing; callback purity remains the caller's responsibility.

The EXLA prototype also adds operand inspection to generic lowering; its compilation overhead has not been measured.

A recipe can be used by a surrounding conditional; the evaluator test covers a use in a branch followed by another outer use. Broader combinations of loops, specialized compiler operations, compiler integrations, and donation need further work. Random keys should be explicit; a body that constructs control flow or nested blocks remains unsupported.

Gradients use the existing ordinary block rule. Value and nonlinear second-derivative tests pass, but this does not promise backward residual checkpointing or backward memory savings.

Validation:
- Nx: 1,388 doctests and 1,604 tests, zero failures with seed 0.
- One preceding run hit an unrelated randomized least-squares tolerance failure. No tolerances were changed.
- EXLA CPU defn suite: 459 tests, zero failures, 43 excluded; after adding the expanded-matrix regression, the focused checkpoint suite passes all 17 tests.
- Tests distinguish evaluator execution counts, chain growth, branch reuse, vectorized/container inputs, boolean outputs, nonlinear gradients, unsupported effects/outputs, and retained compiled reconstruction.
- No CUDA run was made for this experiment.

## Run it

From `nx/`:

```sh
mix run ../examples/recompute_each_use.exs
```

From `exla/`, after setting up EXLA dependencies:

```sh
mix run ../examples/recompute_each_use.exs --exla
```

The script selects the host client for the EXLA comparison, prints recipe counts and input/output sizes, checks numerical agreement, and reports warm execution times and compiled memory. The evaluator callback counter is test instrumentation, not part of the public API.

**AI-generated poem**

> Keep two vectors, let fields fade;
> Rebuild the values where they're weighed.
> Count the work and count the space;
> Choose the trade for this one place.
