# Nx Fuzz Testing Plan

## Background

Nx has 1,261 hand-written tests across 30 files but no property-based or fuzz testing.
No StreamData dependency. Research shows crash oracles find the most bugs with least effort.

Key references:
- NablaFuzz (ICSE 2023): 173 bugs across PyTorch/TF/JAX via AD differential testing
- FreeFuzz (ICSE 2022): 49 bugs via mining + mutation of test inputs
- ConFL (2023): 84 CVEs in TensorFlow via constraint-guided fuzzing
- DocTer (ISSTA 2022): 94 bugs via documentation-extracted constraints

## Tier 1: "Does it crash?" — StreamData property tests (CURRENT)

### Goal
For each Nx op, verify: given valid random inputs, the op doesn't crash and
returns the expected shape/type.

### Steps
1. Add `stream_data` as test dependency
2. Build core generators:
   - `tensor_shape()` — random shapes including edge cases: `{}`, `{0}`, `{1}`, `{0, 3}`, large dims
   - `tensor_type()` — all Nx types: `:u8`, `:s32`, `:f16`, `:f32`, `:f64`, `:bf16`, `:c64`, `:c128`
   - `tensor_value(shape, type)` — random values including: 0, -0, NaN, Inf, -Inf, subnormals, MAX, MIN
   - `broadcastable_shapes(target_shape)` — shapes that broadcast with the target
   - `valid_axes(shape)` — random valid axis selections
3. Write properties per op category:
   - Unary element-wise: sin, cos, exp, abs, negate, etc.
   - Binary element-wise: add, multiply, divide, pow, etc.
   - Reductions: sum, product, reduce_max, reduce_min
   - Shape ops: reshape, transpose, squeeze, pad, slice, concatenate
   - Creation: iota, eye, broadcast
   - Comparison: equal, greater, less, etc.
4. Properties to check:
   - Does not crash (crash oracle)
   - Output shape matches expected shape (shape oracle)
   - Output type matches expected type (type oracle)
   - Output has no unexpected NaN (for well-defined inputs)

### Expected bug categories
- Zero-dimensional tensor edge cases
- Empty tensor (shape contains 0) handling
- Type coercion surprises
- Broadcasting edge cases
- Invalid axis handling

### Effort: 1-2 sessions

---

## Tier 2: "Is the gradient correct?" — Finite difference verification

### Goal
For each differentiable op, compare `Nx.Defn.grad` result against numerical
finite differences: `(f(x+eps) - f(x-eps)) / (2*eps)`.

### Steps
1. Extend the existing `check_grads!` helper in `test/support/helpers.ex`
2. Use Tier 1 generators to produce random inputs
3. For each differentiable op:
   - Generate random input in valid domain (e.g., positive for log, [-1,1] for asin)
   - Compute analytical gradient via `Nx.Defn.grad`
   - Compute numerical gradient via centered finite differences
   - Assert they match within tolerance
4. Test higher-order derivatives (grad of grad)
5. Test gradient through compositions (chains of ops)

### Expected bug categories
- Gradient returning NaN for valid inputs
- Wrong gradient magnitude
- Missing gradient rules
- Higher-order derivative failures
- Complex number gradient issues

### Effort: 1-2 sessions

---

## Tier 3: "Do backends agree?" — Differential testing

### Goal
Run the same operation on BinaryBackend vs EXLA and compare results.

### Steps
1. Use Tier 1 generators for inputs
2. For each op, run on both backends
3. Compare with appropriate tolerance (1-2 ULP for element-wise, larger for reductions)
4. Special handling for TF32 (use precision: :highest for comparison)
5. Log any divergences with full input/output details

### Expected bug categories
- Backend-specific numerical issues
- Missing backend implementations
- Type handling differences
- Shape handling differences
- Precision issues (TF32, etc.)

### Effort: 2-3 sessions (needs EXLA integration)

---

## Tier 4: LLM-guided edge case generation

### Goal
Use Claude Code to systematically read Nx source code, identify validation
checks, and generate targeted edge case tests.

### Approach
1. For each Nx function:
   - Read the source code
   - Identify all validation checks (guards, raise conditions)
   - Generate inputs that hit each boundary
   - Generate inputs that are "just barely valid" and "just barely invalid"
2. Mine patterns from NablaFuzz/FreeFuzz findings:
   - Zero-dim tensors passed to ops expecting >= 1 dim
   - NaN/Inf propagation through compound expressions
   - Type promotion chains
   - Broadcasting with unusual dimension combinations
3. Generate differential tests (compute same result two ways, compare)

### Expected bug categories
- Validation gaps (inputs that should be rejected but aren't)
- Edge case crashes (inputs near validation boundaries)
- Documentation-implementation mismatches

### Effort: 2-3 sessions

---

## Tier 5: Stateful / sequence testing

### Goal
Test sequences of operations (not just individual ops).

### Approach
1. Use PropCheck for stateful model-based testing
2. Model tensor state machine: create → transform → reduce → compare
3. Generate random sequences of valid operations
4. Check invariants hold throughout the sequence:
   - Memory not leaked
   - Backend state consistent
   - Vectorized axes preserved correctly

### Expected bug categories
- State corruption across operations
- Memory leaks in long chains
- Backend state inconsistencies
- Vectorization axis tracking bugs

### Effort: 3-4 sessions

---

## Bug tracking

All bugs found will be:
1. Reduced to minimal reproduction
2. Filed as GitHub issues on elixir-nx/nx
3. Fix PRs submitted where feasible

## Priority

Tier 1 > Tier 2 > Tier 4 > Tier 3 > Tier 5

Tier 1 has the highest bug-finding ROI per effort. Tier 2 is directly relevant
to our vectorized grad work. Tier 4 leverages our tooling. Tier 3 needs
backend setup. Tier 5 is most complex.
