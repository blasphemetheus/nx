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

The source revisions are A at `b63a692c` and the reviewed candidate at
`31a02ce8`. The benchmark is an elementwise chain of
`sin(value * weight + 0.01)`, ending in a sum, on an f32 vector of length 64
and one scalar weight. Every stage is either plain, an explicit-input
checkpoint, or a captured checkpoint region.

Environment: AMD Ryzen 9 9950X3D, Linux 6.18.56, Elixir 1.18.5, OTP 28,
32 online BEAM schedulers. EXLA used the host client with XLA 0.11.0.
No GPU timing was performed.

Each row below is the median of three run medians from fresh VMs, alternating
baseline and candidate sequentially. Each evaluator run has 15 measured samples
after three warmups per phase; each EXLA run has seven measured samples.
Garbage collection runs before each sample, outside its timer. This is a
small local experiment, without CPU pinning or a statistical significance
claim. The raw CSVs include each run's median and quartiles.

### Cost when checkpointing is unused

For the 128-stage plain workload:

| Phase | Baseline | Candidate | Time change | BEAM reduction change |
|---|---:|---:|---:|---:|
| Forward trace | 162 µs | 184 µs | +13.6% | +10.6% |
| Gradient trace | 945 µs | 1,047 µs | +10.8% | +10.2% |
| Evaluator forward compilation | 477 µs | 503 µs | +5.5% | +7.2% |
| Evaluator gradient compilation | 2,138 µs | 2,195 µs | +2.7% | +7.0% |
| Warm evaluator forward execution | 1,334 µs | 1,313 µs | −1.6% | 0.0% |

The tracing hooks have a measurable cost even for code without checkpoints.
Expression IDs check process-local recording state, expression construction
checks scope state, and context merging consults the checkpoint scope helper.
The plain graphs have the same 388 nodes in both revisions. Increased BEAM
reductions support the conclusion that extra tracing work exists, rather than
interpreting small timing differences alone.

These measurements do not isolate each hook's share of that cost. Reductions
are a BEAM scheduling-work counter, not elapsed CPU cycles, and do not measure
native XLA compilation. Warm execution does not repeatedly trace the graph.

The smaller plain forward traces were 10 versus 17 µs at depth 8 and 39 versus
45 µs at depth 32. Their time increases are 70% and 15.4%, respectively, while
reduction counts increased about 12% at both depths. Percentages from very
short timings are sensitive to fixed costs and noise; the results do not support
one universal percentage for tracing overhead.

### Captured regions versus A's explicit checkpoints

For the 128-stage evaluator workload:

| Phase | A: explicit inputs | Candidate: captures |
|---|---:|---:|
| Forward trace | 306 µs | 517 µs |
| Gradient trace | 1,461 µs | 1,679 µs |
| Forward compilation | 606 µs | 1,038 µs |
| Gradient compilation | 3,715 µs | 3,973 µs |
| Warm forward execution | 1,295 µs | 1,305 µs |
| Warm gradient execution | 5,404 µs | 5,341 µs |

Forward tracing is about 69% slower, an additional 211 µs in this small graph.
Gradient tracing is about 15% slower. Forward graphs contain 899 distinct nodes
for the capture form versus 772 for A; gradient graphs contain 2,557 versus
2,303. These counts include stored region bodies and parameters; they are not
optimized instruction counts or byte-allocation measurements.

For comparison, A's explicit API on the candidate revision traces forward in
338 µs. Thus the 517 µs capture time includes both the branch-wide hooks and
the additional capture/region work. Warm execution is close in this workload;
the small timing differences are not evidence of a runtime speedup.

### EXLA CPU

For the 32-stage workload:

| Phase | A: explicit inputs | Candidate: captures |
|---|---:|---:|
| Forward trace | 79 µs | 129 µs |
| Gradient trace | 384 µs | 438 µs |
| Uncached forward compilation | 33.147 ms | 30.877 ms |
| Uncached gradient compilation | 97.799 ms | 94.844 ms |
| Warm forward call, including result transfer | 32 µs | 30 µs |
| Warm gradient call, including result transfer | 112 µs | 113 µs |

The larger tracing cost does not translate into larger total native compilation
time in this example. The node and block forms have different lowering paths;
these results do not prove that the node generally compiles faster. Ordinary
plain EXLA forward compilation was 30.711 ms baseline versus 29.908 ms candidate.

Both compile phases set `cache: false`; they include tracing and preparation as
well as native compilation. Warm calls use an already compiled function and
materialize every result with `Nx.to_binary/1` to wait for completion. Their
times include dispatch and result transfer, not just tensor kernel execution.
This tiny workload is not a model-training throughput benchmark.

### Reproduce and inspect

The scripts need no new dependency:

- [checkpoint_overhead.exs](nx/bench/checkpoint_overhead.exs) runs one revision.
- [checkpoint_compare.exs](nx/bench/checkpoint_compare.exs) summarizes CSVs.
- [checkpoint_results/](nx/bench/checkpoint_results/) contains the recorded runs.

Use the same runner file from both worktrees. From each revision's `nx/`:

```sh
MIX_ENV=test mix run /absolute/path/checkpoint_overhead.exs --label baseline_1 --out /tmp/baseline_1.csv
```

Change the label/output for the candidate and repeat sequentially in fresh VMs.
The default evaluator depths are 8, 32, and 128. For EXLA, run from `exla/`
with the local EXLA build environment configured:

```sh
MIX_ENV=test mix run /absolute/path/checkpoint_overhead.exs --exla --depth 32 --samples 7 --label exla_candidate --out /tmp/exla_candidate.csv
```

Summarize the committed results from the repository root:

```sh
elixir nx/bench/checkpoint_compare.exs nx/bench/checkpoint_results
```

Labels containing `baseline` identify baseline runs in the comparison script.
Other labels identify candidate runs. The full summary includes unchanged
explicit-API comparisons as well as plain and captured forms.

## Assessment

The pure tensor paths examined here work through the evaluator and EXLA,
including the tested control-flow, cache, gradient and aliasing combinations.
Rejecting internal callbacks closes a known unsupported case; it does not add
general effect support.

The main performance concern is the cost paid by plain code during tracing.
Before proposing this implementation broadly, a useful next experiment would
reduce or avoid those inactive tracing/scope checks and rerun the same benchmark.
This report does not implement that optimization or claim its feasibility.

Large models, deeply nested regions, large captured containers, multi-device
execution, peak BEAM allocations, and GPU throughput remain unmeasured.
External compilers would need to understand or lower the new node; this review
does not provide an automatic fallback for them.

Validation after the integration changes: 1,388 doctests and 1,622 Nx tests pass;
the EXLA CPU defn suite passes 465 tests with 43 excluded by its configuration.
Warnings-as-errors compilation and changed-file formatting pass.

**AI-generated poem**

> Follow graphs through every gate;
> Measure costs before you state.
