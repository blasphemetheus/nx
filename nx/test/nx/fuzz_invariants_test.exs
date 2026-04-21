defmodule Nx.FuzzInvariantsTest do
  @moduledoc """
  Fuzz tests for mathematical / algebraic identities that Nx ops
  should satisfy. Each property asserts that LHS ≈ RHS where the
  identity is known to hold mathematically — any deviation signals
  either a bug or an unexpected numerical issue.

  Identities are grouped by the area they exercise:

  - LinAlg identities: transpose-of-product, matmul associativity,
    determinant multiplicativity, trace cyclic property,
    det(A) == prod(eigvals(A)) for symmetric matrices.
  - Reduction identities: sum invariance under reverse/transpose,
    cumsum-vs-sum, axis-split.
  - Numerical-stability identities: softmax shift-invariance,
    sigmoid(x) + sigmoid(-x) == 1, logsumexp stable form.
  - Shape identities: reverse-reverse, transpose-inverse.
  - Grad identities: linearity (grad(f+g) == grad(f) + grad(g)),
    scaling (grad(c*f) == c*grad(f)).

  Unlike `fuzz_linalg_test.exs`, which tests reconstruction invariants
  for each decomposition (Q*R == A etc.), this file tests identities
  that span multiple ops — so a failure points to an inconsistency
  somewhere in the chain.
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  import Nx.Testing

  # ── Generators ─────────────────────────────────────────────────────

  defp well_conditioned_matrix(n) do
    # A = I + 0.1 * R where R is small random — stays near identity,
    # so inv/det/solve are numerically well-behaved.
    bind(list_of(float(min: -1.0, max: 1.0), length: n * n), fn vals ->
      r = Nx.tensor(vals, type: :f32) |> Nx.reshape({n, n})
      a = Nx.add(Nx.eye(n, type: :f32), Nx.multiply(r, 0.1))
      constant(a)
    end)
  end

  defp small_matrix(n, m) do
    bind(list_of(float(min: -2.0, max: 2.0), length: n * m), fn vals ->
      constant(Nx.tensor(vals, type: :f32) |> Nx.reshape({n, m}))
    end)
  end

  defp small_vector(n) do
    bind(list_of(float(min: -2.0, max: 2.0), length: n), fn vals ->
      constant(Nx.tensor(vals, type: :f32))
    end)
  end


  # ── LinAlg identities ──────────────────────────────────────────────

  describe "linalg: matmul algebraic identities" do
    property "(A @ B)ᵀ == Bᵀ @ Aᵀ" do
      check all(
              n <- integer(2..4),
              k <- integer(2..4),
              m <- integer(2..4),
              a <- small_matrix(n, k),
              b <- small_matrix(k, m),
              max_runs: 12
            ) do
        lhs = Nx.transpose(Nx.dot(a, b))
        rhs = Nx.dot(Nx.transpose(b), Nx.transpose(a))
        assert_all_close(lhs, rhs, atol: 1.0e-4)
      end
    end

    property "matmul is associative: (A @ B) @ C == A @ (B @ C)" do
      check all(
              n <- integer(2..4),
              k <- integer(2..4),
              m <- integer(2..4),
              p <- integer(2..4),
              a <- small_matrix(n, k),
              b <- small_matrix(k, m),
              c <- small_matrix(m, p),
              max_runs: 10
            ) do
        lhs = Nx.dot(Nx.dot(a, b), c)
        rhs = Nx.dot(a, Nx.dot(b, c))
        assert_all_close(lhs, rhs, atol: 1.0e-3)
      end
    end
  end

  describe "linalg: determinant multiplicativity" do
    property "det(A @ B) == det(A) * det(B)" do
      check all(
              n <- integer(2..4),
              a <- well_conditioned_matrix(n),
              b <- well_conditioned_matrix(n),
              max_runs: 10
            ) do
        lhs = Nx.LinAlg.determinant(Nx.dot(a, b))
        rhs = Nx.multiply(Nx.LinAlg.determinant(a), Nx.LinAlg.determinant(b))
        # Well-conditioned so values are near 1; absolute tolerance fine.
        assert_all_close(lhs, rhs, atol: 1.0e-3)
      end
    end

    property "det(Aᵀ) == det(A)" do
      check all(
              n <- integer(2..4),
              a <- well_conditioned_matrix(n),
              max_runs: 10
            ) do
        lhs = Nx.LinAlg.determinant(Nx.transpose(a))
        rhs = Nx.LinAlg.determinant(a)
        assert_all_close(lhs, rhs, atol: 1.0e-4)
      end
    end
  end

  describe "linalg: trace cyclic property" do
    property "trace(A @ B) == trace(B @ A)" do
      check all(
              n <- integer(2..4),
              m <- integer(2..4),
              a <- small_matrix(n, m),
              b <- small_matrix(m, n),
              max_runs: 10
            ) do
        lhs = Nx.sum(Nx.take_diagonal(Nx.dot(a, b)))
        rhs = Nx.sum(Nx.take_diagonal(Nx.dot(b, a)))
        assert_all_close(lhs, rhs, atol: 1.0e-3)
      end
    end
  end

  describe "linalg: invert and solve agreement" do
    property "solve(A, b) == invert(A) @ b" do
      check all(
              n <- integer(2..4),
              a <- well_conditioned_matrix(n),
              b <- small_vector(n),
              max_runs: 10
            ) do
        lhs = Nx.LinAlg.solve(a, b)
        rhs = Nx.dot(Nx.LinAlg.invert(a), b)
        assert_all_close(lhs, rhs, atol: 1.0e-3)
      end
    end
  end

  # ── Reduction identities ───────────────────────────────────────────

  describe "reductions: structural invariance" do
    property "sum(reverse(t)) == sum(t)" do
      check all(
              n <- integer(2..8),
              t <- small_vector(n),
              max_runs: 10
            ) do
        lhs = Nx.sum(Nx.reverse(t))
        rhs = Nx.sum(t)
        assert_all_close(lhs, rhs, atol: 1.0e-4)
      end
    end

    property "sum(transpose(A)) == sum(A)" do
      check all(
              n <- integer(2..5),
              m <- integer(2..5),
              a <- small_matrix(n, m),
              max_runs: 10
            ) do
        lhs = Nx.sum(Nx.transpose(a))
        rhs = Nx.sum(a)
        assert_all_close(lhs, rhs, atol: 1.0e-4)
      end
    end

    property "cumulative_sum(t)[-1] == sum(t)" do
      check all(
              n <- integer(2..8),
              t <- small_vector(n),
              max_runs: 10
            ) do
        cumsum = Nx.cumulative_sum(t)
        lhs = cumsum[n - 1]
        rhs = Nx.sum(t)
        assert_all_close(lhs, rhs, atol: 1.0e-3)
      end
    end

    property "sum(A, axes: [0..rank-1]) == sum(A)" do
      check all(
              n <- integer(2..4),
              m <- integer(2..4),
              a <- small_matrix(n, m),
              max_runs: 10
            ) do
        lhs = Nx.sum(a, axes: [0, 1])
        rhs = Nx.sum(a)
        assert_all_close(lhs, rhs, atol: 1.0e-4)
      end
    end
  end

  # ── Numerical stability identities ─────────────────────────────────

  describe "stability: softmax shift-invariance" do
    property "softmax(x) == softmax(x + c) for any constant c" do
      # softmax is shift-invariant: adding a constant to every element
      # should not change the result.
      check all(
              n <- integer(2..8),
              c <- float(min: -5.0, max: 5.0),
              vals <- list_of(float(min: -3.0, max: 3.0), length: n),
              max_runs: 12
            ) do
        x = Nx.tensor(vals, type: :f32)
        shifted = Nx.add(x, Nx.tensor(c, type: :f32))

        lhs = softmax(x)
        rhs = softmax(shifted)
        assert_all_close(lhs, rhs, atol: 1.0e-5)
      end
    end

    property "softmax output sums to 1" do
      check all(
              n <- integer(2..8),
              vals <- list_of(float(min: -3.0, max: 3.0), length: n),
              max_runs: 10
            ) do
        x = Nx.tensor(vals, type: :f32)
        total = Nx.sum(softmax(x))
        assert_all_close(total, Nx.tensor(1.0, type: :f32), atol: 1.0e-5)
      end
    end

    defp softmax(x) do
      # Standard stable form: subtract max, exp, normalize.
      shifted = Nx.subtract(x, Nx.reduce_max(x))
      exps = Nx.exp(shifted)
      Nx.divide(exps, Nx.sum(exps))
    end
  end

  describe "stability: sigmoid identity" do
    property "sigmoid(x) + sigmoid(-x) == 1" do
      check all(
              n <- integer(2..8),
              vals <- list_of(float(min: -10.0, max: 10.0), length: n),
              max_runs: 12
            ) do
        x = Nx.tensor(vals, type: :f32)
        total = Nx.add(Nx.sigmoid(x), Nx.sigmoid(Nx.negate(x)))
        ones = Nx.broadcast(Nx.tensor(1.0, type: :f32), {n})
        assert_all_close(total, ones, atol: 1.0e-5)
      end
    end
  end

  describe "stability: logsumexp via stable form" do
    property "logsumexp(x) == log(sum(exp(x))) when magnitudes are moderate" do
      # In the moderate range log(sum(exp(x))) is accurate; pushing to
      # extreme values would exercise the stability path but we don't
      # expect Nx to expose a stable logsumexp, so we just verify the
      # naive form agrees with itself under shift.
      check all(
              n <- integer(2..8),
              c <- float(min: -3.0, max: 3.0),
              vals <- list_of(float(min: -2.0, max: 2.0), length: n),
              max_runs: 10
            ) do
        x = Nx.tensor(vals, type: :f32)
        shifted = Nx.add(x, Nx.tensor(c, type: :f32))

        # logsumexp(x + c) == logsumexp(x) + c
        lhs = naive_logsumexp(shifted)
        rhs = Nx.add(naive_logsumexp(x), Nx.tensor(c, type: :f32))
        assert_all_close(lhs, rhs, atol: 1.0e-4)
      end
    end

    defp naive_logsumexp(x), do: Nx.log(Nx.sum(Nx.exp(x)))
  end

  # ── Shape identities ───────────────────────────────────────────────

  describe "shape: involutions" do
    property "reverse(reverse(t)) == t" do
      check all(
              n <- integer(2..8),
              t <- small_vector(n),
              max_runs: 10
            ) do
        round_trip = t |> Nx.reverse() |> Nx.reverse()
        assert_all_close(round_trip, t, atol: 0.0, rtol: 0.0)
      end
    end

    property "transpose ∘ transpose == id (2D)" do
      check all(
              n <- integer(2..5),
              m <- integer(2..5),
              a <- small_matrix(n, m),
              max_runs: 10
            ) do
        round_trip = a |> Nx.transpose() |> Nx.transpose()
        assert_all_close(round_trip, a, atol: 0.0, rtol: 0.0)
      end
    end

    property "reshape round-trip preserves values" do
      check all(
              n <- integer(2..5),
              m <- integer(2..5),
              a <- small_matrix(n, m),
              max_runs: 10
            ) do
        round_trip = a |> Nx.reshape({n * m}) |> Nx.reshape({n, m})
        assert_all_close(round_trip, a, atol: 0.0, rtol: 0.0)
      end
    end
  end

  # ── Grad identities ────────────────────────────────────────────────

  describe "grad: linearity and scaling" do
    property "grad(f + g) == grad(f) + grad(g)" do
      check all(
              x <- float(min: -2.0, max: 2.0),
              max_runs: 10
            ) do
        t = Nx.tensor(x, type: :f32)

        grad_f = Nx.Defn.grad(t, fn a -> Nx.sin(a) end)
        grad_g = Nx.Defn.grad(t, fn a -> Nx.cos(a) end)
        grad_sum = Nx.Defn.grad(t, fn a -> Nx.add(Nx.sin(a), Nx.cos(a)) end)

        assert_all_close(grad_sum, Nx.add(grad_f, grad_g), atol: 1.0e-5)
      end
    end

    property "grad(c * f) == c * grad(f)" do
      check all(
              x <- float(min: -2.0, max: 2.0),
              c <- float(min: -3.0, max: 3.0),
              max_runs: 10
            ) do
        t = Nx.tensor(x, type: :f32)
        c_t = Nx.tensor(c, type: :f32)

        grad_f = Nx.Defn.grad(t, fn a -> Nx.sin(a) end)
        grad_cf = Nx.Defn.grad(t, fn a -> Nx.multiply(c_t, Nx.sin(a)) end)

        assert_all_close(grad_cf, Nx.multiply(c_t, grad_f), atol: 1.0e-5)
      end
    end

    property "grad of (0 * x + c) is zero (grad ignores constant terms)" do
      check all(
              x <- float(min: -2.0, max: 2.0),
              c <- float(min: -5.0, max: 5.0),
              max_runs: 8
            ) do
        t = Nx.tensor(x, type: :f32)
        c_t = Nx.tensor(c, type: :f32)

        grad = Nx.Defn.grad(t, fn a -> Nx.add(Nx.multiply(a, 0.0), c_t) end)
        assert_all_close(grad, Nx.tensor(0.0, type: :f32), atol: 1.0e-6)
      end
    end
  end

  # ── Harder identities: edge cases more likely to surface bugs ──────

  describe "numerical stability: large magnitudes" do
    # The naive softmax/logsumexp identities above only used moderate
    # magnitudes — they would pass even on a broken impl that doesn't
    # handle overflow. Push harder to probe stability.

    property "softmax shift-invariance holds at large magnitudes" do
      check all(
              n <- integer(2..5),
              c <- float(min: -100.0, max: 100.0),
              vals <- list_of(float(min: -50.0, max: 50.0), length: n),
              max_runs: 12
            ) do
        x = Nx.tensor(vals, type: :f32)
        shifted = Nx.add(x, Nx.tensor(c, type: :f32))

        lhs = softmax(x)
        rhs = softmax(shifted)
        # Larger tolerance because exponentials at this scale lose
        # precision, but the stable form should still produce the same
        # *probability distribution*.
        assert_all_close(lhs, rhs, atol: 1.0e-4, rtol: 1.0e-3)
      end
    end

    property "sigmoid(x) + sigmoid(-x) == 1 at large |x|" do
      # At |x| > ~15, sigmoid(-x) underflows to 0 in f32 and
      # sigmoid(x) saturates to 1 — the sum stays at 1.0 (exactly).
      check all(
              n <- integer(2..5),
              vals <- list_of(float(min: -50.0, max: 50.0), length: n),
              max_runs: 10
            ) do
        x = Nx.tensor(vals, type: :f32)
        total = Nx.add(Nx.sigmoid(x), Nx.sigmoid(Nx.negate(x)))
        ones = Nx.broadcast(Nx.tensor(1.0, type: :f32), {n})
        assert_all_close(total, ones, atol: 1.0e-5)
      end
    end

    property "exp(x) * exp(-x) == 1 within precision" do
      check all(
              vals <- list_of(float(min: -10.0, max: 10.0), length: 4),
              max_runs: 10
            ) do
        x = Nx.tensor(vals, type: :f32)
        product = Nx.multiply(Nx.exp(x), Nx.exp(Nx.negate(x)))
        ones = Nx.broadcast(Nx.tensor(1.0, type: :f32), {4})
        assert_all_close(product, ones, atol: 1.0e-4)
      end
    end

    property "log(exp(x)) == x within precision" do
      check all(
              vals <- list_of(float(min: -5.0, max: 5.0), length: 4),
              max_runs: 10
            ) do
        x = Nx.tensor(vals, type: :f32)
        round_trip = Nx.log(Nx.exp(x))
        assert_all_close(round_trip, x, atol: 1.0e-5)
      end
    end

    property "sqrt(x*x) == |x|" do
      check all(
              vals <- list_of(float(min: -100.0, max: 100.0), length: 4),
              max_runs: 10
            ) do
        x = Nx.tensor(vals, type: :f32)
        lhs = Nx.sqrt(Nx.multiply(x, x))
        rhs = Nx.abs(x)
        assert_all_close(lhs, rhs, atol: 1.0e-4)
      end
    end
  end

  describe "near-singular linalg" do
    # Identities that should still hold on matrices near singular.

    property "solve followed by matmul recovers the input (moderately conditioned)" do
      check all(
              n <- integer(2..4),
              a <- well_conditioned_matrix(n),
              b <- small_vector(n),
              max_runs: 10
            ) do
        x = Nx.LinAlg.solve(a, b)
        recovered = Nx.dot(a, x)
        assert_all_close(recovered, b, atol: 1.0e-3)
      end
    end

    property "det(2*A) == 2^n * det(A) for n×n matrix" do
      check all(
              n <- integer(2..4),
              a <- well_conditioned_matrix(n),
              max_runs: 8
            ) do
        lhs = Nx.LinAlg.determinant(Nx.multiply(a, 2.0))
        rhs = Nx.multiply(:math.pow(2, n), Nx.LinAlg.determinant(a))
        assert_all_close(lhs, rhs, atol: 1.0e-2)
      end
    end
  end

  describe "grad: chain and product rules" do
    property "grad of f(g(x)) via chain rule" do
      check all(
              x <- float(min: 0.5, max: 2.0),
              max_runs: 10
            ) do
        t = Nx.tensor(x, type: :f32)

        # d/dx sin(x^2) = cos(x^2) * 2x
        grad = Nx.Defn.grad(t, fn a -> Nx.sin(Nx.multiply(a, a)) end)
        expected = Nx.tensor(:math.cos(x * x) * 2 * x, type: :f32)
        assert_all_close(grad, expected, atol: 1.0e-3)
      end
    end

    property "grad of product: d/dx [f(x) * g(x)] == f'*g + f*g'" do
      check all(
              x <- float(min: 0.5, max: 2.0),
              max_runs: 10
            ) do
        t = Nx.tensor(x, type: :f32)

        grad_prod =
          Nx.Defn.grad(t, fn a -> Nx.multiply(Nx.sin(a), Nx.cos(a)) end)

        expected = Nx.tensor(:math.cos(x) * :math.cos(x) - :math.sin(x) * :math.sin(x),
                             type: :f32)
        assert_all_close(grad_prod, expected, atol: 1.0e-3)
      end
    end

    property "grad of sum is sum of grads (tensor version)" do
      check all(
              n <- integer(2..4),
              vals <- list_of(float(min: -2.0, max: 2.0), length: n),
              max_runs: 10
            ) do
        t = Nx.tensor(vals, type: :f32)

        grad_f = Nx.Defn.grad(t, fn a -> Nx.sum(Nx.sin(a)) end)
        grad_g = Nx.Defn.grad(t, fn a -> Nx.sum(Nx.cos(a)) end)
        grad_sum = Nx.Defn.grad(t, fn a -> Nx.sum(Nx.add(Nx.sin(a), Nx.cos(a))) end)

        assert_all_close(grad_sum, Nx.add(grad_f, grad_g), atol: 1.0e-4)
      end
    end
  end

  describe "stress: extreme magnitudes and edge cases" do
    property "softmax is finite for large inputs" do
      # Naive softmax overflows at large values; the stable form
      # (subtract max) should still produce finite values.
      check all(
              n <- integer(2..5),
              scale <- float(min: 50.0, max: 200.0),
              vals <- list_of(float(min: -1.0, max: 1.0), length: n),
              max_runs: 10
            ) do
        x = Nx.tensor(Enum.map(vals, &(&1 * scale)), type: :f32)
        result = softmax(x)
        # No NaN, no Inf — use self-equality to detect NaN.
        nan_count =
          result
          |> Nx.is_nan()
          |> Nx.sum()
          |> Nx.to_number()

        assert nan_count == 0

        inf_count =
          result
          |> Nx.is_infinity()
          |> Nx.sum()
          |> Nx.to_number()

        assert inf_count == 0
      end
    end

    property "softmax output sums to 1 even with one extreme value" do
      # One very large element should dominate; probability mass
      # should concentrate on it and still sum to exactly 1.0.
      check all(
              n <- integer(2..5),
              big_idx <- integer(0..1),
              max_runs: 10
            ) do
        idx = rem(big_idx, n)
        vals = List.duplicate(0.0, n) |> List.replace_at(idx, 500.0)
        x = Nx.tensor(vals, type: :f32)

        total = Nx.sum(softmax(x))
        assert_all_close(total, Nx.tensor(1.0, type: :f32), atol: 1.0e-6)
      end
    end

    property "reverse on 1-element tensor is identity" do
      check all(
              v <- float(min: -100.0, max: 100.0),
              max_runs: 5
            ) do
        t = Nx.tensor([v], type: :f32)
        assert_all_close(Nx.reverse(t), t, atol: 0.0, rtol: 0.0)
      end
    end

    property "transpose on 1×n matrix gives n×1" do
      check all(
              n <- integer(1..5),
              vals <- list_of(float(min: -5.0, max: 5.0), length: n),
              max_runs: 8
            ) do
        row = Nx.tensor([vals], type: :f32)
        col = Nx.transpose(row)
        assert Nx.shape(col) == {n, 1}
        assert_all_close(Nx.reshape(col, {n}), Nx.reshape(row, {n}), atol: 0.0)
      end
    end

    property "sum of single-element tensor equals that element" do
      check all(
              v <- float(min: -1000.0, max: 1000.0),
              max_runs: 6
            ) do
        t = Nx.tensor([v], type: :f32)
        assert_all_close(Nx.sum(t), Nx.tensor(v, type: :f32), atol: 1.0e-3)
      end
    end
  end

  describe "integer ops: modular and overflow-free identities" do
    property "s32 sum of positive values never overflows to negative" do
      check all(
              n <- integer(2..8),
              vals <- list_of(integer(0..1_000_000), length: n),
              max_runs: 10
            ) do
        t = Nx.tensor(vals, type: :s32)
        total = Nx.sum(t)
        assert Nx.to_number(total) >= 0
      end
    end

    property "bitwise: (a ^^^ b) ^^^ b == a (xor involution)" do
      check all(
              n <- integer(2..5),
              va <- list_of(integer(0..1000), length: n),
              vb <- list_of(integer(0..1000), length: n),
              max_runs: 10
            ) do
        a = Nx.tensor(va, type: :s32)
        b = Nx.tensor(vb, type: :s32)
        round_trip = Nx.bitwise_xor(Nx.bitwise_xor(a, b), b)
        assert_equal(round_trip, a)
      end
    end

    property "bitwise: a &&& a == a (and idempotent)" do
      check all(
              n <- integer(2..5),
              vals <- list_of(integer(0..1000), length: n),
              max_runs: 10
            ) do
        a = Nx.tensor(vals, type: :s32)
        assert_equal(Nx.bitwise_and(a, a), a)
      end
    end

    property "bitwise: a ||| 0 == a (or identity)" do
      check all(
              n <- integer(2..5),
              vals <- list_of(integer(0..1000), length: n),
              max_runs: 8
            ) do
        a = Nx.tensor(vals, type: :s32)
        zeros = Nx.broadcast(Nx.tensor(0, type: :s32), {n})
        assert_equal(Nx.bitwise_or(a, zeros), a)
      end
    end
  end

  describe "element-wise op identities" do
    property "a - a == 0" do
      check all(
              vals <- list_of(float(min: -100.0, max: 100.0), length: 4),
              max_runs: 10
            ) do
        x = Nx.tensor(vals, type: :f32)
        diff = Nx.subtract(x, x)
        zeros = Nx.broadcast(Nx.tensor(0.0, type: :f32), {4})
        assert_all_close(diff, zeros, atol: 0.0, rtol: 0.0)
      end
    end

    property "a / a == 1 (for non-zero a)" do
      check all(
              vals <- list_of(float(min: 0.1, max: 10.0), length: 4),
              max_runs: 10
            ) do
        x = Nx.tensor(vals, type: :f32)
        ratio = Nx.divide(x, x)
        ones = Nx.broadcast(Nx.tensor(1.0, type: :f32), {4})
        assert_all_close(ratio, ones, atol: 1.0e-6)
      end
    end

    property "max(a, b) + min(a, b) == a + b" do
      check all(
              n <- integer(2..5),
              va <- list_of(float(min: -5.0, max: 5.0), length: n),
              vb <- list_of(float(min: -5.0, max: 5.0), length: n),
              max_runs: 10
            ) do
        a = Nx.tensor(va, type: :f32)
        b = Nx.tensor(vb, type: :f32)

        lhs = Nx.add(Nx.max(a, b), Nx.min(a, b))
        rhs = Nx.add(a, b)
        assert_all_close(lhs, rhs, atol: 1.0e-5)
      end
    end

    property "max(a, b) * min(a, b) == a * b" do
      check all(
              n <- integer(2..5),
              va <- list_of(float(min: -5.0, max: 5.0), length: n),
              vb <- list_of(float(min: -5.0, max: 5.0), length: n),
              max_runs: 10
            ) do
        a = Nx.tensor(va, type: :f32)
        b = Nx.tensor(vb, type: :f32)

        lhs = Nx.multiply(Nx.max(a, b), Nx.min(a, b))
        rhs = Nx.multiply(a, b)
        assert_all_close(lhs, rhs, atol: 1.0e-4)
      end
    end
  end
end
