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

### Design decisions

| Question | Decision | Rationale |
|----------|----------|-----------|
| Re-tracing location | `parents_args` (like `:optional`) | Checkpoint is a boundary, not a loop. Re-trace the function, let normal gradient flow through the fresh tree. `update_grads` just assigns gradients to body outputs. |
| Context scoping | Inherit body's context (no dedicated scope) | Checkpoint's body result IS returned to the outer scope (unlike while's looping body). No risk of scope escape. |
| Public API mechanism | `deftransform` calling `Expr.checkpoint/2` | Runs at trace time, creates expression nodes. Same pattern as `custom_grad`. Falls back to `fun.(input)` outside defn. |
| Container handling | No flattening — pass through as-is | Not iterative like while. Input goes in, output comes out. No shape matching across iterations. |
| `reduce_args` | All inputs participate in gradient | Every input potentially has a gradient path through the body. |

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

### `Nx.Defn.checkpoint/2` (public API — deftransform)

```elixir
# In Nx.Defn or Nx.Defn.Kernel
deftransform checkpoint(input, fun) do
  # If inside defn (input is an Expr tensor), create expression node
  # Otherwise, pass through: fun.(input)
  ...
end
```

### `Nx.Defn.Expr.checkpoint/2` (expression node constructor)

Template follows `:optional` pattern at line 374 of expr.ex:

```elixir
def checkpoint(input, fun) do
  # 1. Create parameter node from input
  param = parameter(input, 0)

  # 2. Trace fun with the parameter to get body expression
  body_expr = fun.(param)

  # 3. Get context from body expression
  context = body_expr.data.context  # (or handle tuple output)

  # 4. Build :checkpoint expression node
  #    args = [input, body_expr, fun]
  expr(body_expr, context, :checkpoint, [input, body_expr, fun])

  # 5. If body_expr is a tuple, wrap with :elem extraction (like :optional)
end
```

### `Nx.Defn.Tree.apply_args/4` clause

Template follows `:optional` pattern at line 182 of tree.ex:

```elixir
# In apply_args, :scope mode — only traverse input (external-facing)
defp apply_args(:checkpoint, [input, _body_expr, _fun], acc, fun, _mode = :scope) do
  {input, acc} = fun.(input, acc)
  {[input, _body_expr, _fun], acc}
end

# In apply_args, :all mode — also traverse body_expr
defp apply_args(:checkpoint, [input, body_expr, fun_arg], acc, fun, _mode = :all) do
  {input, acc} = fun.(input, acc)
  {body_expr, acc} = Composite.traverse(body_expr, acc, fun)
  {[input, body_expr, fun_arg], acc}
end
```

### `Nx.Defn.Grad.parents_args/5` clause

Template follows `:optional` at line 129 of grad.ex:

```elixir
defp parents_args(
       :checkpoint,
       %{data: %{args: [_input, _body_expr, body_fun]}} = t,
       id,
       acc,
       parent_vectorized_names
     ) do
  # 1. Re-trace body_fun with parameter created from input
  #    (like :optional does: apply(callback, call.data.args))
  expr = body_fun.(... fresh param from input ...)

  # 2. Traverse the re-traced expression, building parent-child edges
  #    (like :optional's Composite.reduce loop)
  {parents, nodes} = ... traverse expr, linking to id ...

  # 3. Store re-traced expr back into the node
  updated_node = {put_in(t.data.args, [input, expr, body_fun]), parent_vectorized_names}
  {parents, Map.put(nodes, id, updated_node)}
end
```

### `Nx.Defn.Grad.update_grads/6` clause

Template follows `:optional` at line 295 of grad.ex:

```elixir
defp update_grads(:checkpoint, [_input, expr, _fun], _ans, gs, _to_grad_ids, grads) do
  # Assign incoming gradients to body expression output nodes
  # (identical to :optional's update_grads)
  gs = List.wrap(gs)

  {grads, []} =
    Composite.reduce(expr, {grads, gs}, fn child, {grads, [g | gs]} ->
      {Map.update(grads, child.data.id, [g], &[g | &1]), gs}
    end)

  grads
end
```

### `Nx.Defn.Grad.reduce_args/4` clause

```elixir
defp reduce_args(:checkpoint, %{data: %{args: [input | _]}}, acc, fun) do
  # All inputs participate in gradient
  fun.(input, acc)
end
```

### `Nx.Defn.Evaluator` clause

Template follows `:optional` evaluator pattern:

```elixir
# In compute_cache: separate input (outer scope) from body (inner scope)
# In eval_apply: evaluate input, set as param, evaluate body_expr
```

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
