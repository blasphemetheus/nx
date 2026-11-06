# Checkpoint compiler integration and overhead

**AI-generated — review of the dedicated checkpoint-node prototype.**

## Integration review

The review follows the stored region through tracing, tree traversal, differentiation,
evaluator preparation and execution, and EXLA lowering. It examines the compiler
implementations present in this repository. It is not a compatibility audit of
external Nx compilers.

| Area | Finding and evidence |
|---|---|
| Graph traversal | The node exposes its inputs to scope-local walks and includes its body and parameters in full walks. This matters for loop/branch sharing and for locating aliases. |
| Differentiation | The direct rule reconstructs the body from tied inputs. Prior boundary tests cover nested scopes, repeated inputs and saved custom-gradient values. |
| Evaluator caches | A region gets its own prepared body cache and parameter binding. It does not read outer tensors through undeclared parameters. |
| Buffer donation | Checkpoint inputs are marked as potential aliases, results are not donated, and body parameters are bound without donation. New recording-backend tests protect explicit inputs and captured tuple aliases still used outside the region. |
| EXLA lowering | The node's stored body is substituted with its outer inputs and lowered into the current computation. Tests compare evaluator and EXLA values and gradients for checkpoints inside loops, loops and branches inside checkpoints, nested captures, and mixed vectorized captures. |
| Compile cache identity | New tests use equal-shaped callback bodies with different operations and repeated calls with changed captured parameter values. The compiled values remain correct. |
| Custom derivatives at boundaries | A saved custom-gradient value keeps its deliberately nonstandard derivative and second derivative under EXLA. |
| Effects and opaque callbacks | Internal IO and runtime calls are now rejected explicitly. Saved values produced by callbacks outside the region remain supported and are tested with the evaluator and EXLA. |
| Other in-repository backends | Torchx is a tensor backend rather than a separate defn compiler; it would use evaluator execution here. No native Torchx run was made. The debug compiler is exercised by graph inspection and tracing measurements. |

### Why internal callbacks are rejected

A probe created two independent IO calls in a checkpoint and consumed their
outputs in reverse order. EXLA executed them in reverse order, whereas the
same ordinary EXLA graph executed them in creation order. EXLA's ordered-IO
prepass follows scope-local edges; it does not enter the stored region.
Inlining the region later therefore bypassed that prepass for its internal
callbacks. The numeric result was correct, so numeric tests alone missed this.

Supporting effects requires more than making the prepass enter another graph.
Reconstruction can repeat effects, and runtime callbacks can hold tensors hidden
from graph substitution. The prototype now requires a pure tensor body and
reports an error for internal IO/runtime calls. Computing such a value before
the region preserves its boundary and does not require rewriting its callback.

This restriction does not detect arbitrary Elixir effects executed during
tracing. Callback purity remains part of the contract.

### Limits of this review

The new tests cover specific scope, cache, aliasing and differentiation paths.
They are not proof of compatibility with every operation or compiler. No
multi-device/sharding audit, native Torchx validation, or new CUDA run was made.
Custom gradients created inside the region remain unsupported.

## Overhead measurement

Measurements and reproduction commands will be added after the integration
changes have passed validation. The runner separately measures forward tracing,
gradient tracing, uncached compilation, and synchronized warm execution.
The baseline is A on upstream main, before the capture tracing hooks exist.
Plain workloads therefore reveal costs paid even when checkpointing is unused.

**AI-generated poem**

> Follow graphs through every gate;
> Measure costs before you state.
