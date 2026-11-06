# Checkpoint capture boundaries

**AI-generated — design notes for the checkpoint-node prototype.**

The purpose of a checkpoint boundary is to identify what can be rebuilt
and which values must be available to rebuild it. A convenient callback
does not remove that distinction. This prototype discovers the callback's
tensor dependencies and stores them as explicit region inputs.

This document explains the current algorithm and the rules the tests should
establish. It does not claim that all combinations of scopes or custom
gradients have already been validated.

## Begin with two different computations

```elixir
prepared = Nx.sin(weight)
checkpoint_capture(x, fn value -> value * prepared end)
```

The region receives `x` and `prepared`. It rebuilds the multiplication.
It does not include the sine that produced `prepared`.

```elixir
checkpoint_capture(x, fn value -> value * Nx.sin(weight) end)
```

The region receives `x` and `weight`. It rebuilds the sine and multiplication.
The callback's execution during tracing determines this difference.

An assignment or a second variable name does not create a tensor operation.
Capturing the same tensor twice should produce one discovered input. Passing
the same tensor twice explicitly still produces two callback parameters.
If that tensor is also captured by its original outer name, it may occupy a
third input position. Gradients from all positions must accumulate under
the original tensor's identity.

## The data the prototype records

A region stores:

- **Inputs:** tensor expressions from the enclosing computation, flattened
  from explicit argument containers and followed by discovered captures.
- **Parameters:** fresh symbolic tensors representing those inputs inside
  the region.
- **Body:** one tensor or a flat tuple of tensors computed from parameters.
- **Scope:** an identity that distinguishes the region's parameters and
  computations from those in an enclosing or nested region.

Static configuration, such as an axis number or an Elixir function used while
tracing, is not a runtime tensor input. A function can itself invoke another
closure; the resulting traced tensor dependencies are what matter.
Context-free tensor constants remain embedded expressions.

## How the boundary is found

1. Convert explicit inputs to expressions and replace them with fresh
   parameters before calling the callback.
2. While that callback runs, record the IDs of newly constructed expressions
   in process-local state. This is the prototype's unusual mechanism:
   expression construction history determines which work belongs inside.
3. Walk backward from the outputs. Visit each tensor ID once.
4. For a context-free expression, leave it embedded. For an expression created
   during this trace, inspect its arguments. For a pre-existing expression,
   stop and record that tensor as a captured input.
5. Give each capture a fresh parameter and substitute parameters into the
   stored body. Repeated uses of one expression continue to share one rewritten
   expression within that body.

A pre-existing captured expression is a boundary even when it contains a large
subgraph. Capture discovery must not walk through it and silently move that
earlier computation into the checkpoint.

## Nested regions and control flow

A nested checkpoint performs its own discovery. Its node exposes explicit
inputs to its parent. The parent walks those inputs rather than treating the
nested body's private parameters as outer dependencies. Expressions created
by the nested trace are included in the parent's construction record when
that trace finishes.

During substitution, the walker can enter nested bodies to reconnect graph
references, but must preserve each body's parameter ownership and sharing.
Only expressions in the source scope change to the substitution's target
context. The target is selected from the replacement inputs. If all replacements
have no context tag, the rebuilt expressions must also have no context tag;
falling back to the old checkpoint scope can make nested gradient contributions
incompatible. The nested region keeps its own scope identity.

Conditional branches can refer to enclosing values, so capture discovery visits
both traced branches, including the branch not selected at runtime. A capture
needed only by one branch is still a dependency of the traced region.

A while body has its own scope. Values crossing that boundary must follow Nx's
existing loop-input rules. A checkpoint inside a loop may capture values from
that loop's current scope. A checkpoint does not grant permission to reach
through an ordinary block or while boundary that would otherwise reject a
tensor capture.

Tracing must restore both the current scope and the recorded-expression set
after an exception. A failed nested trace must not corrupt a subsequent
checkpoint in the enclosing trace.

## What differentiation does

The enclosing gradient traversal initially sees only the region inputs.
Once output gradients arrive, the direct checkpoint rule ties those gradients
and the saved inputs together, substitutes the tied inputs for the stored
parameters, and differentiates the reconstructed body.

That local derivative walk stops at the substituted inputs. Its results are
added to the enclosing gradients under the original input IDs. Inputs unused
by the body receive zero; shared inputs and uses outside the checkpoint still
receive all of their contributions.

## Custom gradients: the boundary matters

A custom gradient stores an Elixir derivative callback as well as its forward
value. That callback may hold tensors from the original trace. Rewriting
expression arguments does not rewrite tensors hidden in the callback.

There are three different cases to investigate:

| Placement | Required behavior |
|---|---|
| Custom-gradient value computed before the checkpoint | Treat the value as an ordinary saved input. Its custom derivative belongs to the enclosing gradient traversal. |
| Custom gradient created inside the checkpoint | Reconstruct its derivative dependencies correctly, or reject it explicitly. Merely copying its callback is insufficient. |
| Custom gradient around the checkpoint's output | The outer custom rule controls differentiation, as it does around an ordinary computation. |

The initial prototype rejects custom-gradient metadata during discovery and
substitution without distinguishing all these placements. The investigation
will test those cases before changing that restriction.

For an internal rule, tracing only the forward callback can miss dependencies
used solely by the derivative callback. Supporting that case may require a
separate trace of the derivative, with symbolic incoming gradients and an
explicit representation of any additional tensor dependencies. That is a
larger design than removing the rejection.

## Boundary checks now covered

The tests cover shared precomputed expressions crossing nested regions and
outer uses, sibling regions with vectorized captures, branch-only captures,
checkpoints inside loop bodies, mixed explicit/captured tensor identities,
ordinary block scope restrictions, and recovery after a nested tracing exception.
They compare forward values and nonlinear first/second derivatives where
supported, and inspect region inputs where numeric agreement alone cannot
show the chosen boundary.

The nested shared-capture test exposed a stale-scope bug when substituting
context-free inputs. It fails on the original implementation and passes after
using the replacement inputs' context, including its absence.

Compiler integration review and tracing-overhead measurement are separate,
deferred work. Passing these boundary tests will not establish either of them.

**AI-generated poem**

> Name what enters, trace what stays;
> Keep each boundary through the maze.
