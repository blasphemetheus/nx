# Checkpoint recomputation placed by XLA scheduling groups

An experiment, not a proposal. It answers one question about the gradient
checkpointing design: can the barrier in the checkpoint gradient rule take
the saved inputs alone, with the incoming gradient left out, if XLA is told
where to schedule the recomputation some other way?

## The problem

The checkpoint gradient rule recomputes the block body from its saved inputs
during the backward pass. Two things can defeat the memory saving:

1. XLA's common subexpression elimination sees that the recomputed body is a
   copy of the forward body and merges them. Any barrier on the inputs stops
   this.
2. The recomputation depends only on the saved inputs, which exist as soon as
   the forward pass finishes. The scheduler is free to run the recomputation
   right there, next to the original, so both sets of intermediates are alive
   at once and nothing is saved. A barrier on the inputs alone does not stop
   this. Putting the incoming gradient through the same barrier does, because
   the recomputation then cannot start before the backward pass arrives.

The checkpoint PR uses the second form, one barrier over inputs and incoming
gradient together. This branch tries the first form plus a scheduling hint.

## What this branch changes

* `nx/lib/nx/defn/grad.ex`: the checkpoint rule puts only the inputs through
  the barrier and wraps the whole backward body of the checkpoint, from the
  barrier outputs to the input gradients, in a metadata node with a
  `scheduling_group` key.
* `exla/lib/exla/defn.ex`, `exla/lib/exla/mlir/value.ex`: every op lowered
  inside such a node gets the `_scheduling_group_id` frontend attribute that
  XLA's latency hiding scheduler reads.
* `xla_scheduling_groups.patch`: three changes to XLA at the commit EXLA
  pins (bb760b04), needed for the attribute to have any effect on these
  programs. The GPU pass that legalizes scheduling annotations keeps them
  only on cuBLAS and Triton GEMM fusions; the latency hiding scheduler skips
  computations with no async collectives; a fusion that mixes two groups is
  a compile error. The patch keeps all annotations, runs the scheduler for
  annotated computations, and gives a mixed fusion the group of its root.
* `build_xla.sh`: the Bazel invocation used to build the patched
  `xla_extension` archive on the machine where this was measured. Paths are
  that machine's.
* `probe.exs`: the hand-written forward and backward used for the first
  measurements, before the gradient rule was changed. `PROBE_MODE` selects
  no barrier, inputs-only barrier, tied barrier, recompute-only group or
  whole-body group.

## Results

Two sets of numbers. The first comes from `probe.exs`, a hand-written
forward and backward for four dense-relu pairs, 2048 wide, batch 8192, where
only the activation goes through the barrier and the weights are read
directly. Peak scratch is XLA's `preallocated-temp` from the memory usage
report, latency hiding scheduler on, patched XLA where groups are used.

| probe variant | peak scratch |
|---|---|
| no barrier | 528 MiB |
| barrier on the activation only | 560 MiB |
| barrier on activation and incoming gradient (what the checkpoint PR does) | 336 MiB |
| activation-only barrier, recomputation grouped | 912 MiB |
| activation-only barrier, whole backward body grouped | 336 MiB |

The recompute-only group placed the recomputation correctly but moved every
weight-gradient GEMM to the end of the program, because running the latency
hiding scheduler replaces the default memory-aware ordering and nothing in a
recompute-only group constrains the rest. Grouping the whole body put the
weight gradients back next to their layer. That is the one configuration
that matched the tied barrier, and it never beat it.

The second set comes from this branch's gradient rule, through
`checkpoint/2` itself, with `EXLA.Executable.memory_stats/1` on the gradient
of the four-layer MLP from the EXLA checkpoint tests. Here every checkpoint
input goes through the barrier, the weights included, which is how the
checkpoint PR is written and differs from the probe. The small-width case is
the one the memory test uses: the activation is much larger than any GEMM
workspace, so the numbers isolate activations.

| gradient rule variant | n=256, batch 65536 | n=2048, batch 8192 |
|---|---|---|
| no checkpoint | 576 MiB | 528 to 560 MiB |
| tied barrier (checkpoint PR) | 449 MiB | 448 MiB |
| this branch, stock XLA | 704 MiB | 384 MiB |
| this branch, patched XLA, whole-body groups | 800 MiB | 384 MiB |

At the activation-dominated size this branch uses more memory than no
checkpoint at all, patched or not, while the tied barrier saves two
activations. The wide case is dominated by GEMM workspaces, which vary with
autotuning between runs, and is not a reliable signal at this scale. The
scheduled HLO for the patched wide case shows the groups in place and the
weight gradients interleaved per layer, so the placement worked and the
memory did not follow; the cause was not isolated. The difference from the
probe that stands out is the weights passing through the barrier.

The patched XLA also crashed the compiler on one test of the EXLA checkpoint
suite, a `CHECK` failure in `latency_hiding_scheduler.h` with "No state for
fused_computation". The patch's handling of a fusion that mixes two groups
leaves annotations inside the fused computation, and the patched scheduler
entry point then tries to schedule the fusion body. The stock XLA ran the
suite with one failure, the memory test, as expected for an inputs-only
barrier.

## Why the tied barrier stays

* The scheduling group is a preference for the scheduler, not a dependency
  in the graph. The tied barrier is a dependency: no output exists until
  every operand does, so no scheduler can move the recomputation early.
* Scheduling annotations are a GPU feature. The CPU client has nothing
  equivalent.
* It needs three changes inside XLA, an XLA release carrying them, and an
  EXLA pin bump, plus a new Nx to EXLA channel for the group ids.
* The best patched result matched the tied barrier and never beat it.
