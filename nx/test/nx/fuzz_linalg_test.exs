defmodule Nx.FuzzLinAlgTest do
  @moduledoc """
  Property-based fuzz tests for Nx.LinAlg operations.

  Tests that decompositions don't crash on valid inputs and satisfy
  mathematical properties (e.g., A ≈ Q*R for QR decomposition).
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  import Nx.Testing

  # ── Generators ─────────────────────────────────────────────────────

  defp square_matrix(max_n \\ 8) do
    bind(integer(2..max_n), fn n ->
      bind(member_of([:f32, :f64]), fn type ->
        constant(Nx.add(Nx.iota({n, n}, type: type), Nx.eye(n, type: type)))
      end)
    end)
  end

  defp positive_definite(max_n \\ 8) do
    bind(integer(2..max_n), fn n ->
      bind(member_of([:f32, :f64]), fn type ->
        # A^T * A + n*I is always positive definite
        a = Nx.iota({n, n}, type: type)
        at = Nx.transpose(a)
        pd = Nx.add(Nx.dot(at, a), Nx.multiply(n, Nx.eye(n, type: type)))
        constant(pd)
      end)
    end)
  end

  defp upper_triangular(max_n \\ 8) do
    bind(integer(2..max_n), fn n ->
      bind(member_of([:f32, :f64]), fn type ->
        a = Nx.add(Nx.iota({n, n}, type: type), Nx.multiply(n, Nx.eye(n, type: type)))
        # Zero out below diagonal
        mask =
          Nx.less_equal(
            Nx.iota({n, n}, axis: 0, type: :s32),
            Nx.iota({n, n}, axis: 1, type: :s32)
          )

        constant(Nx.select(mask, a, 0))
      end)
    end)
  end

  defp lower_triangular(max_n \\ 8) do
    bind(integer(2..max_n), fn n ->
      bind(member_of([:f32, :f64]), fn type ->
        a = Nx.add(Nx.iota({n, n}, type: type), Nx.multiply(n, Nx.eye(n, type: type)))

        mask =
          Nx.greater_equal(
            Nx.iota({n, n}, axis: 0, type: :s32),
            Nx.iota({n, n}, axis: 1, type: :s32)
          )

        constant(Nx.select(mask, a, 0))
      end)
    end)
  end

  defp tall_matrix(max_m \\ 8, max_n \\ 6) do
    bind(integer(2..max_m), fn m ->
      bind(integer(2..min(m, max_n)), fn n ->
        bind(member_of([:f32, :f64]), fn type ->
          constant(Nx.iota({m, n}, type: type))
        end)
      end)
    end)
  end

  # ── QR Decomposition ──────────────────────────────────────────────

  describe "QR decomposition" do
    property "doesn't crash on square matrices" do
      check all(a <- square_matrix(), max_runs: 15) do
        {q, r} = Nx.LinAlg.qr(a)
        assert is_struct(q, Nx.Tensor)
        assert is_struct(r, Nx.Tensor)
      end
    end

    property "doesn't crash on tall matrices" do
      check all(a <- tall_matrix(), max_runs: 15) do
        {q, r} = Nx.LinAlg.qr(a, mode: :reduced)
        assert is_struct(q, Nx.Tensor)
        assert is_struct(r, Nx.Tensor)
      end
    end

    property "Q*R reconstructs A" do
      check all(a <- square_matrix(6), max_runs: 10) do
        {q, r} = Nx.LinAlg.qr(a)
        reconstructed = Nx.dot(q, r)
        assert_all_close(reconstructed, a, atol: 1.0e-3, rtol: 1.0e-3)
      end
    end

    property "Q is orthogonal (Q^T * Q ≈ I)" do
      check all(a <- square_matrix(6), max_runs: 10) do
        {q, _r} = Nx.LinAlg.qr(a)
        n = elem(Nx.shape(q), 0)
        qtq = Nx.dot(Nx.transpose(q), q)
        assert_all_close(qtq, Nx.eye(n), atol: 1.0e-3, rtol: 1.0e-3)
      end
    end
  end

  # ── Cholesky Decomposition ────────────────────────────────────────

  describe "Cholesky decomposition" do
    property "doesn't crash on positive definite matrices" do
      check all(a <- positive_definite(), max_runs: 15) do
        l = Nx.LinAlg.cholesky(a)
        assert is_struct(l, Nx.Tensor)
        assert Nx.shape(l) == Nx.shape(a)
      end
    end

    property "L * L^T reconstructs A" do
      check all(a <- positive_definite(6), max_runs: 10) do
        l = Nx.LinAlg.cholesky(a)
        reconstructed = Nx.dot(l, Nx.transpose(l))
        assert_all_close(reconstructed, a, atol: 1.0e-2, rtol: 1.0e-2)
      end
    end

    property "L is lower triangular" do
      check all(a <- positive_definite(), max_runs: 10) do
        l = Nx.LinAlg.cholesky(a)
        n = elem(Nx.shape(l), 0)

        # Check upper triangle is zero
        upper_mask =
          Nx.less(
            Nx.iota({n, n}, axis: 0, type: :s32),
            Nx.iota({n, n}, axis: 1, type: :s32)
          )

        upper_vals = Nx.select(upper_mask, Nx.abs(l), 0)
        max_upper = Nx.reduce_max(upper_vals) |> Nx.to_number()
        assert max_upper < 1.0e-6
      end
    end
  end

  # ── LU Decomposition ──────────────────────────────────────────────

  describe "LU decomposition" do
    property "doesn't crash on square matrices" do
      check all(a <- square_matrix(), max_runs: 15) do
        {p, l, u} = Nx.LinAlg.lu(a)
        assert is_struct(p, Nx.Tensor)
        assert is_struct(l, Nx.Tensor)
        assert is_struct(u, Nx.Tensor)
      end
    end

    property "P * L * U reconstructs A" do
      check all(a <- square_matrix(6), max_runs: 10) do
        {p, l, u} = Nx.LinAlg.lu(a)
        reconstructed = p |> Nx.dot(l) |> Nx.dot(u)
        assert_all_close(reconstructed, a, atol: 1.0e-2, rtol: 1.0e-2)
      end
    end
  end

  # ── SVD ───────────────────────────────────────────────────────────

  describe "SVD" do
    property "doesn't crash on square matrices" do
      check all(a <- square_matrix(6), max_runs: 10) do
        {u, s, v} = Nx.LinAlg.svd(a)
        assert is_struct(u, Nx.Tensor)
        assert is_struct(s, Nx.Tensor)
        assert is_struct(v, Nx.Tensor)
      end
    end

    property "singular values are non-negative" do
      check all(a <- square_matrix(6), max_runs: 10) do
        {_u, s, _v} = Nx.LinAlg.svd(a)
        min_s = Nx.reduce_min(s) |> Nx.to_number()
        assert min_s >= -1.0e-5
      end
    end

    property "U and V are orthogonal" do
      check all(a <- square_matrix(5), max_runs: 8) do
        {u, _s, v} = Nx.LinAlg.svd(a)
        n = elem(Nx.shape(u), 0)
        utu = Nx.dot(Nx.transpose(u), u)
        vtv = Nx.dot(Nx.transpose(v), v)
        assert_all_close(utu, Nx.eye(n), atol: 1.0e-2, rtol: 1.0e-2)
        assert_all_close(vtv, Nx.eye(elem(Nx.shape(v), 0)), atol: 1.0e-2, rtol: 1.0e-2)
      end
    end
  end

  # ── Eigenvalue Decomposition ──────────────────────────────────────

  describe "eigh (symmetric eigendecomposition)" do
    property "doesn't crash on symmetric matrices" do
      check all(n <- integer(2..6), type <- member_of([:f32, :f64]), max_runs: 10) do
        a = Nx.iota({n, n}, type: type)
        sym = Nx.add(a, Nx.transpose(a)) |> Nx.divide(2)
        {evals, evecs} = Nx.LinAlg.eigh(sym)
        assert is_struct(evals, Nx.Tensor)
        assert is_struct(evecs, Nx.Tensor)
        assert Nx.shape(evals) == {n}
        assert Nx.shape(evecs) == {n, n}
      end
    end

    property "eigenvalues are real for symmetric matrices" do
      check all(n <- integer(2..5), max_runs: 8) do
        a = Nx.iota({n, n}, type: :f32)
        sym = Nx.add(a, Nx.transpose(a)) |> Nx.divide(2)
        {evals, _evecs} = Nx.LinAlg.eigh(sym)
        # Eigenvalues of symmetric matrices are always real
        assert Nx.type(evals) == {:f, 32}
      end
    end
  end

  # ── Triangular Solve ──────────────────────────────────────────────

  describe "triangular_solve" do
    property "doesn't crash with upper triangular" do
      check all(a <- upper_triangular(), max_runs: 10) do
        n = elem(Nx.shape(a), 0)
        b = Nx.iota({n}, type: Nx.type(a))
        result = Nx.LinAlg.triangular_solve(a, b, lower: false)
        assert Nx.shape(result) == {n}
      end
    end

    property "doesn't crash with lower triangular" do
      check all(a <- lower_triangular(), max_runs: 10) do
        n = elem(Nx.shape(a), 0)
        b = Nx.iota({n}, type: Nx.type(a))
        result = Nx.LinAlg.triangular_solve(a, b, lower: true)
        assert Nx.shape(result) == {n}
      end
    end

    property "A * x ≈ b (solution verification)" do
      check all(a <- upper_triangular(5), max_runs: 8) do
        n = elem(Nx.shape(a), 0)
        b = Nx.add(Nx.iota({n}, type: Nx.type(a)), 1)
        x = Nx.LinAlg.triangular_solve(a, b, lower: false)
        reconstructed = Nx.dot(a, x)
        assert_all_close(reconstructed, b, atol: 1.0e-2, rtol: 1.0e-2)
      end
    end
  end

  # ── Determinant ───────────────────────────────────────────────────

  describe "determinant" do
    property "doesn't crash on square matrices" do
      check all(a <- square_matrix(), max_runs: 15) do
        result = Nx.LinAlg.determinant(a)
        assert Nx.shape(result) == {}
      end
    end

    property "det(I) = 1" do
      check all(n <- integer(2..8), type <- member_of([:f32, :f64]), max_runs: 10) do
        eye = Nx.eye(n, type: type)
        det = Nx.LinAlg.determinant(eye) |> Nx.to_number()
        assert_in_delta det, 1.0, 1.0e-4
      end
    end
  end

  # ── Norm ──────────────────────────────────────────────────────────

  describe "norm" do
    property "vector norm is non-negative" do
      check all(
              len <- integer(1..32),
              type <- member_of([:f32, :f64]),
              max_runs: 15
            ) do
        t = Nx.iota({len}, type: type)
        norm = Nx.LinAlg.norm(t) |> Nx.to_number()
        assert norm >= 0.0
      end
    end

    property "matrix frobenius norm is non-negative" do
      check all(a <- square_matrix(), max_runs: 10) do
        norm = Nx.LinAlg.norm(a) |> Nx.to_number()
        assert norm >= 0.0
      end
    end
  end

  # ── Solve ─────────────────────────────────────────────────────────

  describe "solve" do
    property "doesn't crash on invertible matrices" do
      check all(a <- positive_definite(6), max_runs: 10) do
        n = elem(Nx.shape(a), 0)
        b = Nx.add(Nx.iota({n}, type: Nx.type(a)), 1)
        result = Nx.LinAlg.solve(a, b)
        assert Nx.shape(result) == {n}
      end
    end

    property "A * solve(A, b) ≈ b" do
      check all(a <- positive_definite(5), max_runs: 8) do
        n = elem(Nx.shape(a), 0)
        b = Nx.add(Nx.iota({n}, type: Nx.type(a)), 1)
        x = Nx.LinAlg.solve(a, b)
        reconstructed = Nx.dot(a, x)

        assert_all_close(reconstructed, Nx.as_type(b, Nx.type(reconstructed)),
          atol: 1.0e-1,
          rtol: 1.0e-1
        )
      end
    end
  end

  # ── Invert ────────────────────────────────────────────────────────

  describe "invert" do
    property "doesn't crash on invertible matrices" do
      check all(a <- positive_definite(6), max_runs: 10) do
        inv = Nx.LinAlg.invert(a)
        assert Nx.shape(inv) == Nx.shape(a)
      end
    end

    property "A * A^-1 ≈ I" do
      check all(a <- positive_definite(5), max_runs: 8) do
        inv = Nx.LinAlg.invert(a)
        n = elem(Nx.shape(a), 0)
        product = Nx.dot(a, inv)

        assert_all_close(product, Nx.eye(n, type: Nx.type(product)),
          atol: 1.0e-1,
          rtol: 1.0e-1
        )
      end
    end
  end

  # ── Generators for batched inputs ──────────────────────────────────

  # {B, N, N} batched square matrix, safely invertible (diagonally dominant).
  defp batched_square_matrix(max_n \\ 6, max_b \\ 3) do
    bind(integer(2..max_n), fn n ->
      bind(integer(2..max_b), fn b ->
        bind(member_of([:f32, :f64]), fn type ->
          # B copies of (iota + n*I) → diagonally dominant, well-conditioned.
          single = Nx.add(Nx.iota({n, n}, type: type), Nx.multiply(n, Nx.eye(n, type: type)))

          batched =
            0..(b - 1)
            |> Enum.map(fn i -> Nx.add(single, Nx.multiply(i + 1, Nx.eye(n, type: type))) end)
            |> Nx.stack()

          constant(batched)
        end)
      end)
    end)
  end

  # {B, N, N} batched positive-definite matrix.
  defp batched_positive_definite(max_n \\ 5, max_b \\ 3) do
    bind(integer(2..max_n), fn n ->
      bind(integer(2..max_b), fn b ->
        bind(member_of([:f32, :f64]), fn type ->
          batched =
            0..(b - 1)
            |> Enum.map(fn i ->
              a = Nx.add(Nx.iota({n, n}, type: type), Nx.multiply(i + 1, Nx.eye(n, type: type)))
              # A^T * A + n*I is always positive definite.
              Nx.add(Nx.dot(Nx.transpose(a), a), Nx.multiply(n, Nx.eye(n, type: type)))
            end)
            |> Nx.stack()

          constant(batched)
        end)
      end)
    end)
  end

  # ── Gradients through linalg decompositions (batched inputs) ──────
  #
  # The invariant: `Nx.Defn.grad(x, fn ... decomp(x) ... end)` must work
  # and return a tensor of the same shape as `x`, for `x` with a leading
  # batch dimension. Decompositions with hand-written `custom_grad`
  # formulas have historically been coded for 2D and silently broken for
  # batched inputs; this block exercises every decomposition against that
  # expectation. Each failing property is an observed bug that wants its
  # own tracking issue / fix PR.

  describe "gradients through linalg (batched)" do
    # Tracks cholesky_grad gap in PR #1731 — the PR's fix is partial and still
    # fails for batched input. Class: same batched-grad class as #1741-#1746.
    property "[meta #1748] grad of sum(Cholesky(A)) on batched positive-definite input" do
      check all(a <- batched_positive_definite(4, 3), max_runs: 5) do
        grad = Nx.Defn.grad(a, fn x -> Nx.sum(Nx.LinAlg.cholesky(x)) end)
        assert Nx.shape(grad) == Nx.shape(a)
      end
    end

    # Class: same batched-grad class as #1741-#1746. No QR-specific issue filed.
    property "[meta #1748] grad of sum(Q + R) for QR on batched square input" do
      check all(a <- batched_square_matrix(4, 3), max_runs: 5) do
        grad =
          Nx.Defn.grad(a, fn x ->
            {q, r} = Nx.LinAlg.qr(x)
            Nx.add(Nx.sum(q), Nx.sum(r))
          end)

        assert Nx.shape(grad) == Nx.shape(a)
      end
    end

    # Upstream: https://github.com/elixir-nx/nx/issues/1742
    # Root cause: lu_grad uses unqualified Nx.dot; wrong axis contractions with leading batch dim.
    property "[#1742] grad of sum(L + U) for LU on batched square input" do
      check all(a <- batched_square_matrix(4, 3), max_runs: 5) do
        grad =
          Nx.Defn.grad(a, fn x ->
            {_p, l, u} = Nx.LinAlg.lu(x)
            Nx.add(Nx.sum(l), Nx.sum(u))
          end)

        assert Nx.shape(grad) == Nx.shape(a)
      end
    end

    # Upstream: https://github.com/elixir-nx/nx/issues/1743
    # Root cause: svd_grad pattern-matches {m, n} = Nx.shape(input); fails on 3D+.
    property "[#1743] grad of sum(U + s + V) for SVD on batched square input" do
      check all(a <- batched_square_matrix(4, 3), max_runs: 5) do
        grad =
          Nx.Defn.grad(a, fn x ->
            {u, s, vt} = Nx.LinAlg.svd(x)
            Nx.add(Nx.add(Nx.sum(u), Nx.sum(s)), Nx.sum(vt))
          end)

        assert Nx.shape(grad) == Nx.shape(a)
      end
    end

    # Upstream: https://github.com/elixir-nx/nx/issues/1740
    # See BUG-1740-A / BUG-1740-B pins below for specific reproducers.
    property "[#1740] grad of sum(eigvals) for eigh on batched symmetric input" do
      check all(a <- batched_positive_definite(4, 3), max_runs: 5) do
        # positive_definite matrices are symmetric; eigh is defined for symmetric inputs.
        grad =
          Nx.Defn.grad(a, fn x ->
            {eigvals, _eigvecs} = Nx.LinAlg.eigh(x)
            Nx.sum(eigvals)
          end)

        assert Nx.shape(grad) == Nx.shape(a)
      end
    end

    # Upstream: https://github.com/elixir-nx/nx/issues/1741
    # Root cause: Nx.dot inside triangular_solve grad has shape-mismatch on batched input.
    property "[#1741] grad of sum(triangular_solve(A, b)) on batched lower-triangular A" do
      check all(a <- batched_positive_definite(4, 3), max_runs: 5) do
        # Use cholesky(a) as a batched lower-triangular matrix, and an arbitrary b.
        batch = elem(Nx.shape(a), 0)
        n = elem(Nx.shape(a), 1)
        b = Nx.broadcast(Nx.tensor(1.0, type: Nx.type(a)), {batch, n})

        grad =
          Nx.Defn.grad(a, fn x ->
            l = Nx.LinAlg.cholesky(x)
            Nx.sum(Nx.LinAlg.triangular_solve(l, b))
          end)

        assert Nx.shape(grad) == Nx.shape(a)
      end
    end
  end

  # ── eigh grad: targeted coverage after the batched-grad probe ──────
  #
  # Running the batched-grad probe above surfaced that eigh grad has
  # three distinct behaviors depending on rank + dtype:
  #
  #   2D any dtype     → fails with "cannot reshape {} to {1, N, N}"
  #                      (eigh grad path assumes 3D input)
  #   3D batch=1 f32   → works
  #   3D any batch f64 → fails with "expected 32 bits got 64 bits"
  #                      (hardcoded f32 somewhere in eigh grad)
  #
  # The parameterized property below makes the full (rank, dtype) map
  # visible in a single test run. The two minimal-reproducer tests
  # below it are the concrete artifacts to point fix PRs at.

  describe "eigh grad: (rank, dtype) coverage" do
    defp symmetric_2d(n, type) do
      # Diagonally-dominant symmetric matrix.
      a = Nx.iota({n, n}, type: type)
      sym = Nx.add(Nx.transpose(a), a)
      Nx.add(sym, Nx.multiply(n * 4, Nx.eye(n, type: type)))
    end

    defp symmetric_batched(batch, n, type) do
      0..(batch - 1)
      |> Enum.map(fn i ->
        Nx.add(symmetric_2d(n, type), Nx.multiply(i + 1, Nx.eye(n, type: type)))
      end)
      |> Nx.stack()
    end

    property "eigh grad handles every (rank, dtype) combination" do
      check all(
              batch <- integer(1..3),
              dtype <- member_of([:f32, :f64]),
              use_batch <- boolean(),
              max_runs: 12
            ) do
        n = 3
        input = if use_batch, do: symmetric_batched(batch, n, dtype), else: symmetric_2d(n, dtype)

        grad =
          Nx.Defn.grad(input, fn x ->
            {eigvals, _eigvecs} = Nx.LinAlg.eigh(x)
            Nx.sum(eigvals)
          end)

        assert Nx.shape(grad) == Nx.shape(input),
               "eigh grad shape mismatch for input shape=#{inspect(Nx.shape(input))} dtype=#{inspect(dtype)}"
      end
    end

    # BUG-1740-A — 2D eigh grad raises reshape error.
    # Upstream: https://github.com/elixir-nx/nx/issues/1740
    # Class: defn/grad formula bug. Fires on BinaryBackend AND EXLA,
    # independent of the :verify_binary_size compile-time flag.
    # Root cause hypothesis: the grad path reshapes a scalar intermediate
    # into {1, N, N}, assuming 3D input. A 2D input produces a scalar
    # cotangent with shape {} that can't be reshaped into {2, 2}.
    test "[FIXED-1740-A] eigh grad: 2D input works, d(sum eigvals)/dA = I" do
      # #1740 fixed upstream. Sum of eigenvalues is the trace, so the
      # grad wrt a symmetric input is the identity matrix.
      x = Nx.tensor([[4.0, 2.0], [2.0, 5.0]], type: :f32)

      grad =
        Nx.Defn.grad(x, fn a ->
          {s, _} = Nx.LinAlg.eigh(a)
          Nx.sum(s)
        end)

      assert_all_close(grad, Nx.eye(2, type: :f32), atol: 1.0e-3)
    end

    # BUG-1740-B — f64 eigh grad has an internal type-tag mismatch in an
    # intermediate tensor: somewhere in the grad path, a scalar tensor is
    # declared `{:f, 32}` but receives 64-bit binary data.
    # Upstream: https://github.com/elixir-nx/nx/issues/1740
    # Class: BinaryBackend bug — surfaces as an ArgumentError only when
    # the :verify_binary_size compile-time flag is on (see
    # nx/config/config.exs and nx/lib/nx/binary_backend.ex:117).
    #
    # IMPORTANT NUANCE: probing with the flag OFF (the default for end-
    # user apps depending on :nx as a hex dep) shows the final grad VALUE
    # is numerically identical to the f32 grad for simple test cases
    # (d(trace)/dA = I, d(sum λ²)/dA). The malformed intermediate tensor
    # doesn't corrupt observable output in the cases we can easily
    # construct. The assert_raise below pins the "loud" behavior only —
    # a value-based f64-specific pin isn't easily writable because f32
    # and f64 produce the same numbers. The bug is real (type tag is
    # wrong) but the severity is "latent internal inconsistency," not
    # "silent wrong results for users."
    test "[FIXED-1740-B] eigh grad: f64 batched input works" do
      # #1740 fixed upstream (eps constant no longer f32-typed).
      x = Nx.tensor([[[4.0, 2.0], [2.0, 5.0]]], type: :f64)

      grad =
        Nx.Defn.grad(x, fn a ->
          {s, _} = Nx.LinAlg.eigh(a)
          Nx.sum(s)
        end)

      assert Nx.type(grad) == {:f, 64}
      assert_all_close(grad, Nx.broadcast(Nx.eye(2, type: :f64), {1, 2, 2}), atol: 1.0e-3)
    end

    # The one case that works — kept as a positive assertion to catch
    # regressions in the opposite direction.
    test "eigh grad: 3D batch=1 f32 works" do
      x = Nx.tensor([[[4.0, 2.0], [2.0, 5.0]]], type: :f32)

      grad =
        Nx.Defn.grad(x, fn a ->
          {s, _} = Nx.LinAlg.eigh(a)
          Nx.sum(s)
        end)

      assert Nx.shape(grad) == {1, 2, 2}
    end

    # BUG-1740-B-value — value-level exposure of the f64 grad bug at 4×4.
    # Upstream: https://github.com/elixir-nx/nx/issues/1740
    # The [BUG-1740-B] pin above captures the loud verifier error at 2×2
    # with :verify_binary_size on. This test captures the *observable*
    # failure at 4×4: the f64 grad of sum(log(eigvals)) returns entries
    # with magnitude > 1e15 where the f32 grad returns values in the
    # 1e-4 to 1e-5 range. Probed with the flag OFF: max |f64 - f32|
    # is ~5.5e25 on this input.
    #
    # Root cause (confirmed via Nx.Defn.debug_expr on the grad):
    # nx/lib/nx/lin_alg.ex:1388 — eigh's default `eps: 1.0e-4` is a raw
    # float literal that defn types as {:f, 32}. The constant flows into
    # block_eigh's while-body convergence checks and emerges in the
    # backward pass as f32 intermediates (e.g. `reshape 0.0001 f32[1][1]`
    # and multiple `elem fd, N f32[1][2][2]` nodes in the grad expr),
    # which then collide with the f64 data stride in a scalar tensor.
    # Fix direction: scale the eps default to the input tensor's float
    # type (f32 → 1.0e-4; f64 → ~1.0e-10 or similar) and/or cast the
    # literal via Nx.tensor(1.0e-4, type: output_type) before threading
    # it into the while body.
    #
    # Runs under mix test (flag ON): currently fails either via the
    # assert_raise path or via the assert_all_close below. When #1740
    # is fixed, the grad should return values comparable to f32 within
    # reasonable precision, and this test will pass.
    # Un-skipped 2026-08-14: #1740 fixed upstream; probe shows max diff
    # 4.5e-8 on this input (was ~5.5e25).
    test "[FIXED-1740-B-value] f64 eigh grad on 4×4 agrees with f32" do
      y64 =
        Nx.tensor(
          [
            [
              [5.0, 1.3, 0.7, 0.2],
              [1.3, 6.0, 0.5, 0.1],
              [0.7, 0.5, 7.0, 0.4],
              [0.2, 0.1, 0.4, 8.0]
            ]
          ],
          type: :f64
        )

      y32 = Nx.as_type(y64, :f32)

      obj = fn a ->
        {s, _v} = Nx.LinAlg.eigh(a)
        Nx.sum(Nx.log(s))
      end

      grad64 = Nx.Defn.grad(y64, obj) |> Nx.as_type(:f32)
      grad32 = Nx.Defn.grad(y32, obj)

      # Correct contract: f64 grad should agree with f32 grad to ~1e-3.
      # Currently: max diff observed ≈ 5.5e25 on this host.
      assert_all_close(grad64, grad32, atol: 1.0e-3, rtol: 1.0e-3)
    end
  end

  # ── Task #15: extend batched-grad probes to the remaining linalg ops
  #
  # Covers ops that weren't in the first batched-grad sweep: invert,
  # solve, determinant, norm, pinv, matrix_power, matrix_rank. Each
  # property exercises the op's grad on a batched input and asserts the
  # grad has the same shape as the input. Failures are follow-up issues
  # / fix-PR candidates of the same class as #1740/#1741/#1742/#1743.

  describe "gradients through linalg (batched) — extended" do
    # Upstream: https://github.com/elixir-nx/nx/issues/1744
    # Root cause: custom_grad at nx/lib/nx/lin_alg.ex:866 uses Nx.dot without batch axes.
    property "[#1744] grad of sum(invert(A)) on batched square input" do
      check all(a <- batched_square_matrix(4, 3), max_runs: 5) do
        grad = Nx.Defn.grad(a, fn x -> Nx.sum(Nx.LinAlg.invert(x)) end)
        assert Nx.shape(grad) == Nx.shape(a)
      end
    end

    # Upstream: https://github.com/elixir-nx/nx/issues/1745
    property "[#1745] grad of sum(solve(A, b)) on batched square input" do
      check all(a <- batched_square_matrix(4, 3), max_runs: 5) do
        batch = elem(Nx.shape(a), 0)
        n = elem(Nx.shape(a), 1)
        b = Nx.broadcast(Nx.tensor(1.0, type: Nx.type(a)), {batch, n})
        grad = Nx.Defn.grad(a, fn x -> Nx.sum(Nx.LinAlg.solve(x, b)) end)
        assert Nx.shape(grad) == Nx.shape(a)
      end
    end

    # Upstream: https://github.com/elixir-nx/nx/issues/1746
    # Note: 3×3 determinant grad happens to work; 4×4 and larger fail.
    property "[#1746] grad of determinant(A) on batched square input" do
      check all(a <- batched_square_matrix(4, 3), max_runs: 5) do
        grad = Nx.Defn.grad(a, fn x -> Nx.sum(Nx.LinAlg.determinant(x)) end)
        assert Nx.shape(grad) == Nx.shape(a)
      end
    end

    # Resolution of the #1748 class for norm: batched input is rejected
    # up front with a clean ArgumentError (norm is documented 1-D/2-D
    # only) rather than gaining batch support.
    property "[meta #1748] norm(A) on batched input raises cleanly" do
      check all(a <- batched_square_matrix(4, 3), max_runs: 5) do
        assert_raise ArgumentError, ~r/expected 1-D or 2-D tensor/, fn ->
          Nx.Defn.grad(a, fn x -> Nx.sum(Nx.LinAlg.norm(x)) end)
        end
      end
    end

    # LIVE BUG: pinv is broken on batched input in the FORWARD pass —
    # the upstream #1748 fixes covered its siblings but not pinv. Three
    # size-dependent failure modes; the n=2 silent rank-4 output is the
    # worst (wrong result, no error). See
    # FUZZ_FINDINGS/pinv_batched_forward_crash.md. Flip all three to
    # shape assertions ({batch, n, n}) when fixed.
    test "[BUG-PINV-BATCHED] batched pinv: n=1 crashes on reshape" do
      a = Nx.iota({2, 1, 1}, type: :f32) |> Nx.add(Nx.eye(1))
      assert_raise ArgumentError, ~r/cannot reshape/, fn -> Nx.LinAlg.pinv(a) end
    end

    test "[BUG-PINV-BATCHED] batched pinv: n=2 silently returns rank-4 output" do
      a = Nx.iota({2, 2, 2}, type: :f32) |> Nx.add(Nx.eye(2))
      # WRONG: should be {2, 2, 2}
      assert Nx.shape(Nx.LinAlg.pinv(a)) == {2, 2, 2, 2}
    end

    test "[BUG-PINV-BATCHED] batched pinv: n>=3 crashes on broadcast" do
      a = Nx.iota({2, 3, 3}, type: :f32) |> Nx.add(Nx.eye(3))
      assert_raise ArgumentError, ~r/cannot broadcast tensor/, fn -> Nx.LinAlg.pinv(a) end
    end

    # Class: same batched-grad class as #1741-#1746. Not separately filed.
    property "[meta #1748] grad of sum(matrix_power(A, 2)) on batched square input" do
      check all(a <- batched_square_matrix(4, 3), max_runs: 5) do
        grad = Nx.Defn.grad(a, fn x -> Nx.sum(Nx.LinAlg.matrix_power(x, 2)) end)
        assert Nx.shape(grad) == Nx.shape(a)
      end
    end

    # Class: same batched-grad class as #1741-#1746. Not separately filed.
    property "[meta #1748] grad of sum(adjoint(A)) on batched square input" do
      check all(a <- batched_square_matrix(4, 3), max_runs: 5) do
        grad = Nx.Defn.grad(a, fn x -> Nx.sum(Nx.LinAlg.adjoint(x)) end)
        assert Nx.shape(grad) == Nx.shape(a)
      end
    end
  end
end
