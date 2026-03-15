# Gradient Checkpointing Design Document

Following the [HtDP Design Recipe](https://htdp.org/2026-2-25//Book/index.html) for implementing `Nx.Defn.checkpoint/2` ([Nx #765](https://github.com/elixir-nx/nx/issues/765), [Axon #372](https://github.com/elixir-nx/axon/issues/372)).

## Step 1: Data Definitions

### CheckpointExpr

A `CheckpointExpr` is an `Nx.Defn.Expr` where:

```
%Nx.Defn.Expr{
  op:      :checkpoint,
  args:    [input_exprs, body_expr, body_fun],
  context: <inherited from body's tracing context>,
  id:      <unique ref>
}
```

| Field | Type | Description |
|-------|------|-------------|
| `input_exprs` | `[%Nx.Tensor{data: %Expr{}}]` | Tensor expressions saved for recomputation. Passed as explicit arguments to `checkpoint/2`. These are the only values retained during the forward pass. |
| `body_expr` | `%Nx.Tensor{data: %Expr{}}` or `tuple()` | The traced result of `body_fun` applied to parameter nodes. Used for shape/type inference during tracing and as the forward computation path. |
| `body_fun` | `(Tensor, ... -> Tensor \| tuple())` | A function with arity = `length(input_exprs)`. Re-invoked with fresh parameter expressions during gradient backpropagation to produce a new expression tree for differentiation. Must be a pure function of its arguments. |

### Design decision: explicit inputs

The public API takes explicit inputs rather than detecting captured variables from an arity-0 closure:

```elixir
Nx.Defn.checkpoint(input_or_inputs, fun)
```

Rationale:
- Matches how `:optional` works internally (explicit `in_args` + `fun`)
- The user always knows what data they're saving
- An arity-0 convenience wrapper can be added later as sugar (the internal expression node is the same)
- Explicit over implicit

### Relationship to existing expression nodes

| Aspect | `:checkpoint` | `:optional` | `:while` |
|--------|--------------|-------------|----------|
| Stores body fun for re-tracing | Yes | Yes | No (uses stored expressions) |
| Stores traced body expression | Yes | Yes | Yes |
| Stores explicit inputs | Yes | Yes (as `call`) | Yes (as `initial`) |
| Scope semantics in tree traversal | Body is scoped | Body is scoped | Body is scoped |
| Gradient strategy | Re-trace fun, differentiate fresh tree | Passthrough to body children | Build companion while loop |

## Step 2: Signature, Purpose, Header

### `Nx.Defn.checkpoint/2`

```
checkpoint : (Tensor | [Tensor] | Container, (Tensor, ... -> Tensor | tuple())) -> Tensor | tuple()
```

**Purpose**: Mark a computation region for gradient checkpointing. During the forward pass, behaves as `apply(fun, inputs)`. During the backward pass, the function is re-executed with saved inputs to recompute intermediates, rather than storing them.

**Header**:
```elixir
def checkpoint(input_or_inputs, fun)
```

### `Nx.Defn.Expr.checkpoint/2` (expression node constructor)

```
checkpoint : ([%Tensor{}], (Tensor, ... -> Tensor | tuple())) -> %Tensor{} | tuple()
```

**Purpose**: Trace `fun` with parameter nodes derived from `input_exprs`, producing the body expression. Wrap the result in a `:checkpoint` expression node that stores input expressions, the traced body, and the original function for re-tracing during gradient computation.

**Header**:
```elixir
def checkpoint(in_args, fun)
```

### `Nx.Defn.Tree.apply_args/4` clause for `:checkpoint`

```
apply_args : (:checkpoint, [input_exprs, body_expr, body_fun], acc, fun) -> {args, acc}
```

**Purpose**: Traverse the expression tree. In `:scope` mode, traverse only `input_exprs` (the external-facing inputs). In `:all` mode, also traverse `body_expr` (the internal computation). Never traverse `body_fun`.

### `Nx.Defn.Grad` — `parents_args` clause for `:checkpoint`

```
parents_args : (:checkpoint, %Tensor{}, id, {parents, nodes}, vectorized_names) -> {parents, nodes}
```

**Purpose**: Re-trace `body_fun` with fresh parameters to produce a new expression tree. Build the parent-child relationships through this re-traced tree so gradients can flow through it during backpropagation. Store the re-traced body expression back into the node (like `:optional` does).

### `Nx.Defn.Grad` — `update_grads` clause for `:checkpoint`

```
update_grads : (:checkpoint, [input_exprs, body_expr, body_fun], ans, gs, to_grad_ids, grads) -> grads
```

**Purpose**: Propagate gradients through the checkpoint boundary. Seed the body expression outputs with incoming gradients `gs`, then call `to_grad` on each input expression to compute their gradients through the re-traced body. This is where recomputation conceptually happens — the fresh expression tree from `parents_args` is what gets differentiated.

### `Nx.Defn.Evaluator` clause for `:checkpoint`

```
eval_apply : (:checkpoint, [input_exprs, body_expr, body_fun]) -> result
```

**Purpose**: Pass-through evaluation. Evaluate `body_expr` using the evaluated `input_exprs` as parameters. No memory optimization in eager mode — that only matters in compiled backends (EXLA).

### `Nx.Defn.Grad` — `reduce_args` clause for `:checkpoint`

```
reduce_args : (:checkpoint, %Tensor{}, acc, fun) -> acc
```

**Purpose**: Identify which args participate in gradient computation. For `:checkpoint`, ALL input expressions participate (they all flow through the body function).

## Step 3: Examples / Tests

33 tests written in `test/nx/defn/checkpoint_test.exs` covering:

- Forward pass is a no-op (3 tests)
- Gradient correctness for elementwise, reductions, compositions (3 tests)
- Multi-layer dense chain - the primary use case (1 test)
- Nested checkpoints (1 test)
- Interaction with `cond` (2 tests), `while` (1 test), `custom_grad` (1 test), `stop_grad` (1 test)
- Container (tuple) outputs (1 test)
- `value_and_grad` (1 test)
- Gradient w.r.t. captured weights / param maps (2 tests)
- Diamond/shared input pattern (1 test)
- Partial checkpointing - ops before and after boundary (1 test)
- Higher-order gradients (1 test)
- Numerical precision (1 test)
- Multiple outputs consumed independently (1 test)
- Many sequential checkpoints (2 tests)
- Zero gradient / constant output (1 test)
- Broadcasting inside checkpoint (1 test)
- Dtype preservation - f64, bf16 (2 tests)
- Shape-changing ops - reshape, transpose (2 tests)

All tests pass with the pass-through stub since checkpoint is semantically transparent.

## Step 4: Template / Inventory

*TODO — derive function outlines from the data definitions*

## Step 5: Function Definition

*TODO — fill in the templates*

## Step 6: Testing

*TODO — verify all 33 tests pass with the real implementation*

---

## Implementation Phases

### Phase 1: Nx core (Evaluator backend)
- [ ] `:checkpoint` expression node in `Nx.Defn.Expr`
- [ ] `Nx.Defn.checkpoint/2` public API
- [ ] `Nx.Defn.Tree` traversal clause
- [ ] Gradient rule in `Nx.Defn.Grad`
- [ ] `Nx.Defn.Evaluator` support

### Phase 2: EXLA backend
- [ ] Re-add `optimization_barrier` NIF for CSE prevention
- [ ] EXLA compiler support for `:checkpoint` nodes

### Phase 3: Axon integration
- [ ] `Axon.checkpoint/1` layer

### Phase 4: Policies (future)
- [ ] Selective save/recompute decisions

## References

- [Nx #765 - Support checkpoints in gradients](https://github.com/elixir-nx/nx/issues/765)
- [Axon #372 - Support gradient checkpoints in models](https://github.com/elixir-nx/axon/issues/372)
- [Nx #946 - Replace optional callbacks with Nx.block](https://github.com/elixir-nx/nx/issues/946) (complementary)
- [JAX jax.checkpoint / jax.remat](https://docs.jax.dev/en/latest/gradient-checkpointing.html)
- [PyTorch torch.utils.checkpoint](https://docs.pytorch.org/docs/stable/checkpoint.html)
- Chen et al., "Training Deep Nets with Sublinear Memory Cost" (2016)
