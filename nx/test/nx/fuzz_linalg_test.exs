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
end
