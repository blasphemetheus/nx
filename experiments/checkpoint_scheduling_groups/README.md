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

RESULTS_PLACEHOLDER

## Why the tied barrier stays

* The scheduling group is a preference for the scheduler, not a dependency
  in the graph. The tied barrier is a dependency: no output exists until
  every operand does, so no scheduler can move the recomputation early.
* Scheduling annotations are a GPU feature. The CPU client has nothing
  equivalent.
* It needs three changes inside XLA, an XLA release carrying them, and an
  EXLA pin bump, plus a new Nx to EXLA channel for the group ids.
* The best patched result matched the tied barrier and never beat it.
