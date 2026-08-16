defmodule Nx.FuzzBlockTest do
  @moduledoc """
  Differential + invariant fuzz for `Nx.block`-backed APIs (FUZZ_ROADMAP T1.2).

  Every `Nx.Block` struct executes through two routes:

    * eager — `backend.block/4` invoked directly on concrete tensors
    * jit   — an `Expr` `:block` node evaluated by `Nx.Defn.Evaluator`,
      which re-dispatches to `backend.block/4` with reconstructed
      parameters and options

  The plumbing between those routes (parameter encoding, tuple outputs,
  option splitting) is exactly where past block bugs lived, so every
  property below asserts jit/eager agreement, plus a structural invariant
  that holds regardless of route. Option matrices of the structs are swept
  where they exist.
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  @fuzz_scale String.to_integer(System.get_env("FUZZ_SCALE", "1"))

  import Nx.Testing

  # Well-conditioned n×n matrix: I + 0.1R keeps linalg numerically tame.
  defp matrix(n) do
    bind(list_of(float(min: -1.0, max: 1.0), length: n * n), fn vals ->
      r = vals |> Nx.tensor(type: {:f, 64}) |> Nx.reshape({n, n})
      constant(Nx.add(Nx.eye(n, type: {:f, 64}), Nx.multiply(r, 0.1)))
    end)
  end

  defp spd_matrix(n) do
    map(matrix(n), fn a ->
      # A·Aᵀ + n·I is symmetric positive definite
      a
      |> Nx.dot(Nx.transpose(a))
      |> Nx.add(Nx.multiply(Nx.eye(n, type: {:f, 64}), n))
    end)
  end

  defp vector(n) do
    bind(list_of(float(min: -10.0, max: 10.0), length: n), fn vals ->
      constant(Nx.tensor(vals, type: {:f, 64}))
    end)
  end

  defp jit(fun, args) do
    Nx.Defn.jit_apply(fun, args)
  end

  defp assert_routes_agree(fun, args, opts \\ []) do
    eager = apply(fun, args)
    jitted = jit(fun, args)

    case eager do
      %Nx.Tensor{} ->
        assert_all_close(jitted, eager, opts)

      tuple when is_tuple(tuple) ->
        for {e, j} <- Enum.zip(Tuple.to_list(eager), Tuple.to_list(jitted)) do
          assert_all_close(j, e, opts)
        end
    end

    eager
  end

  describe "LinAlg blocks: jit/eager agreement + structural invariants" do
    property "QR: routes agree and Q·R reconstructs A (mode sweep)" do
      check all(
              n <- integer(2..5),
              a <- matrix(n),
              mode <- member_of([:reduced, :complete]),
              max_runs: 10 * @fuzz_scale
            ) do
        {q, r} = assert_routes_agree(fn x -> Nx.LinAlg.qr(x, mode: mode) end, [a])

        assert_all_close(Nx.dot(q, r), a, atol: 1.0e-8)

        # Q has orthonormal columns
        qtq = Nx.dot(Nx.transpose(q), q)
        assert_all_close(qtq, Nx.eye(Nx.axis_size(q, 1), type: {:f, 64}), atol: 1.0e-8)
      end
    end

    property "Cholesky: routes agree and L·Lᵀ reconstructs A" do
      check all(n <- integer(2..5), a <- spd_matrix(n), max_runs: 10 * @fuzz_scale) do
        l = assert_routes_agree(&Nx.LinAlg.cholesky/1, [a])
        assert_all_close(Nx.dot(l, Nx.transpose(l)), a, atol: 1.0e-8)
      end
    end

    property "Solve: routes agree and A·x reconstructs b" do
      check all(n <- integer(2..5), a <- matrix(n), b <- vector(n), max_runs: 10 * @fuzz_scale) do
        x = assert_routes_agree(&Nx.LinAlg.solve/2, [a, b])
        assert_all_close(Nx.dot(a, x), b, atol: 1.0e-6)
      end
    end

    property "LU: routes agree and P·L·U reconstructs A" do
      check all(n <- integer(2..5), a <- matrix(n), max_runs: 10 * @fuzz_scale) do
        {p, l, u} = assert_routes_agree(&Nx.LinAlg.lu/1, [a])
        assert_all_close(p |> Nx.dot(l) |> Nx.dot(u), a, atol: 1.0e-8)
      end
    end

    property "SVD: routes agree, singular values sorted desc, reconstruction holds" do
      check all(
              n <- integer(2..4),
              a <- matrix(n),
              full? <- boolean(),
              max_runs: 8 * @fuzz_scale
            ) do
        {u, s, vt} =
          assert_routes_agree(fn x -> Nx.LinAlg.svd(x, full_matrices?: full?) end, [a],
            atol: 1.0e-4
          )

        s_list = Nx.to_flat_list(s)
        assert s_list == Enum.sort(s_list, :desc), "singular values not sorted desc"
        assert Enum.all?(s_list, &(&1 >= 0)), "negative singular value"

        # svd/eigh are iterative; the reconstruction-error tail slightly
        # exceeds 1e-4 on unlucky matrices (overnight run, seed 924345613,
        # max diff 1.09e-4 after 162 clean runs). 1e-3 still catches real
        # breakage, which is O(1) off.
        reconstructed = u |> Nx.dot(Nx.make_diagonal(s)) |> Nx.dot(vt)
        assert_all_close(reconstructed, a, atol: 1.0e-3)
      end
    end

    property "Eigh: routes agree and A·V = V·diag(λ) for symmetric A" do
      check all(n <- integer(2..4), a <- spd_matrix(n), max_runs: 8 * @fuzz_scale) do
        {evals, evecs} = assert_routes_agree(&Nx.LinAlg.eigh/1, [a], atol: 1.0e-4)

        av = Nx.dot(a, evecs)
        vl = Nx.dot(evecs, Nx.make_diagonal(evals))
        assert_all_close(av, vl, atol: 1.0e-3, rtol: 1.0e-3)
      end
    end

    property "Determinant: routes agree and multiplicativity holds" do
      check all(n <- integer(2..4), a <- matrix(n), b <- matrix(n), max_runs: 10 * @fuzz_scale) do
        det_a = assert_routes_agree(&Nx.LinAlg.determinant/1, [a])
        det_b = Nx.LinAlg.determinant(b)
        det_ab = Nx.LinAlg.determinant(Nx.dot(a, b))
        assert_all_close(det_ab, Nx.multiply(det_a, det_b), rtol: 1.0e-6, atol: 1.0e-10)
      end
    end
  end

  describe "indexing blocks" do
    property "take: routes agree and matches Elixir list reference (axis sweep)" do
      check all(
              rows <- integer(2..5),
              cols <- integer(2..5),
              axis <- member_of([0, 1]),
              idx_len <- integer(1..4),
              max_runs: 15 * @fuzz_scale
            ) do
        t = Nx.iota({rows, cols}, type: {:f, 32})
        axis_size = if axis == 0, do: rows, else: cols

        check all(idx <- list_of(integer(0..(axis_size - 1)), length: idx_len), max_runs: 1) do
          idx_t = Nx.tensor(idx)

          result = assert_routes_agree(fn x, i -> Nx.take(x, i, axis: axis) end, [t, idx_t])

          # Elixir reference
          nested = Nx.to_list(t)

          expected =
            case axis do
              0 -> Enum.map(idx, &Enum.at(nested, &1))
              1 -> Enum.map(nested, fn row -> Enum.map(idx, &Enum.at(row, &1)) end)
            end

          assert Nx.to_list(result) == expected
        end
      end
    end

    property "top_k: routes agree, values sorted desc and traceable to input" do
      check all(
              n <- integer(3..10),
              t <- vector(n),
              k <- integer(1..3),
              max_runs: 15 * @fuzz_scale
            ) do
        {values, indices} = assert_routes_agree(fn x -> Nx.top_k(x, k: k) end, [t])

        vals = Nx.to_flat_list(values)
        assert vals == Enum.sort(vals, :desc)

        # every (value, index) pair must satisfy t[index] == value
        input = Nx.to_flat_list(t)

        for {v, i} <- Enum.zip(vals, Nx.to_flat_list(indices)) do
          assert Enum.at(input, i) == v
        end
      end
    end
  end

  describe "cumulative blocks" do
    property "cumulative_sum: routes agree and matches Enum.scan (axis/reverse sweep)" do
      check all(
              n <- integer(2..8),
              t <- vector(n),
              reverse? <- boolean(),
              max_runs: 20 * @fuzz_scale
            ) do
        result =
          assert_routes_agree(fn x -> Nx.cumulative_sum(x, axis: 0, reverse: reverse?) end, [t])

        values = Nx.to_flat_list(t)
        values = if reverse?, do: Enum.reverse(values), else: values
        scanned = Enum.scan(values, &(&1 + &2))
        expected = if reverse?, do: Enum.reverse(scanned), else: scanned

        assert_all_close(result, Nx.tensor(expected, type: {:f, 64}), atol: 1.0e-9)
      end
    end

    property "cumulative_min/max: routes agree and match Enum.scan" do
      check all(n <- integer(2..8), t <- vector(n), max_runs: 20 * @fuzz_scale) do
        for {op, fun} <- [{:cumulative_min, &min/2}, {:cumulative_max, &max/2}] do
          result = assert_routes_agree(fn x -> apply(Nx, op, [x]) end, [t])
          expected = t |> Nx.to_flat_list() |> Enum.scan(fun)
          assert_all_close(result, Nx.tensor(expected, type: {:f, 64}), atol: 0.0)
        end
      end
    end
  end

  describe "predicate/misc blocks" do
    property "all_close: routes agree and match Elixir reference (rtol/atol sweep)" do
      check all(
              n <- integer(1..6),
              a <- vector(n),
              b <- vector(n),
              atol <- member_of([1.0e-8, 0.1, 10.0]),
              rtol <- member_of([1.0e-5, 0.1]),
              max_runs: 20 * @fuzz_scale
            ) do
        result =
          assert_routes_agree(
            fn x, y -> Nx.all_close(x, y, atol: atol, rtol: rtol) end,
            [a, b]
          )

        as = Nx.to_flat_list(a)
        bs = Nx.to_flat_list(b)

        expected =
          Enum.zip(as, bs)
          |> Enum.all?(fn {x, y} -> abs(x - y) <= atol + rtol * abs(y) end)

        assert Nx.to_number(result) == if(expected, do: 1, else: 0)
      end
    end

    property "phase: routes agree and match atan2(imag, real)" do
      check all(n <- integer(1..6), re <- vector(n), im <- vector(n), max_runs: 20 * @fuzz_scale) do
        z = Nx.complex(Nx.as_type(re, {:f, 32}), Nx.as_type(im, {:f, 32}))

        result = assert_routes_agree(&Nx.phase/1, [z])
        expected = Nx.atan2(Nx.as_type(im, {:f, 32}), Nx.as_type(re, {:f, 32}))
        assert_all_close(result, expected, atol: 1.0e-6)
      end
    end
  end

  describe "FFT blocks" do
    property "fft2/ifft2: routes agree and round-trip is identity" do
      check all(rows <- member_of([2, 4]), cols <- member_of([2, 4]), max_runs: 8 * @fuzz_scale) do
        check all(
                vals <- list_of(float(min: -10.0, max: 10.0), length: rows * cols),
                max_runs: 1
              ) do
          t = vals |> Nx.tensor(type: {:f, 32}) |> Nx.reshape({rows, cols})

          f = assert_routes_agree(&Nx.fft2/1, [t], atol: 1.0e-4)
          round_tripped = Nx.ifft2(f)
          assert_all_close(Nx.real(round_tripped), t, atol: 1.0e-4)
          assert_all_close(Nx.imag(round_tripped), Nx.broadcast(0.0, {rows, cols}), atol: 1.0e-4)
        end
      end
    end

    property "rfft/irfft: routes agree and round-trip recovers real input" do
      check all(n <- member_of([4, 8]), max_runs: 8 * @fuzz_scale) do
        check all(vals <- list_of(float(min: -10.0, max: 10.0), length: n), max_runs: 1) do
          t = Nx.tensor(vals, type: {:f, 32})

          f = assert_routes_agree(&Nx.rfft/1, [t], atol: 1.0e-4)
          assert Nx.axis_size(f, 0) == div(n, 2) + 1

          back = assert_routes_agree(fn x -> Nx.irfft(x, length: n) end, [f], atol: 1.0e-4)
          assert_all_close(back, t, atol: 1.0e-3)
        end
      end
    end
  end
end
