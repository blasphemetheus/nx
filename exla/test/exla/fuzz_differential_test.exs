defmodule EXLA.FuzzDifferentialTest do
  @moduledoc """
  Tier 3: Differential testing — BinaryBackend vs EXLA.

  Runs the same operations on both backends and compares results.
  Uses precision: :highest to avoid TF32 divergence for matmul ops.
  """
  use ExUnit.Case, async: false
  use ExUnitProperties

  import Nx.Testing

  setup do
    :ok
  end

  # ── Helpers ───────────────────────────────────────────────────────

  defp on_binary_backend(fun) do
    Nx.with_default_backend(Nx.BinaryBackend, fun)
  end

  defp on_exla(fun) do
    Nx.Defn.jit(fun, compiler: EXLA, client: :host)
  end

  defp compare_backends(fun, inputs, opts \\ []) do
    atol = opts[:atol] || 1.0e-4
    rtol = opts[:rtol] || 1.0e-4

    binary_result = apply(fun, inputs)
    exla_result = on_exla(fn -> apply(fun, inputs) end).()

    # Transfer EXLA result to binary backend for comparison
    exla_on_binary = Nx.backend_transfer(exla_result, Nx.BinaryBackend)

    assert_all_close(binary_result, exla_on_binary, atol: atol, rtol: rtol)
  end

  defp random_f32(shape) do
    size = Nx.size(shape)

    for(_ <- 1..size, do: :rand.uniform() * 4 - 2)
    |> Nx.tensor(type: :f32)
    |> Nx.reshape(shape)
  end

  # ── Unary element-wise ────────────────────────────────────────────

  @unary_ops [:abs, :negate, :sign, :floor, :ceil, :round, :sigmoid]

  describe "unary ops: BinaryBackend vs EXLA" do
    for op <- @unary_ops do
      property "#{op} agrees across backends" do
        check all(n <- integer(1..8), max_runs: 10) do
          x = random_f32({n})

          binary_result = on_binary_backend(fn -> apply(Nx, unquote(op), [x]) end)

          exla_result =
            Nx.Defn.jit(fn x -> apply(Nx, unquote(op), [x]) end,
              compiler: EXLA,
              client: :host
            ).(x)

          assert_all_close(binary_result, exla_result, atol: 1.0e-5, rtol: 1.0e-5)
        end
      end
    end
  end

  # ── Transcendentals ──────────────────────────────────────────────

  @transcendental_ops [:sin, :cos, :tan, :tanh, :erf, :erfc]

  describe "transcendental ops: BinaryBackend vs EXLA" do
    for op <- @transcendental_ops do
      property "#{op} agrees across backends" do
        check all(n <- integer(1..8), max_runs: 10) do
          x = random_f32({n})

          binary_result = on_binary_backend(fn -> apply(Nx, unquote(op), [x]) end)

          exla_result =
            Nx.Defn.jit(fn x -> apply(Nx, unquote(op), [x]) end,
              compiler: EXLA,
              client: :host
            ).(x)

          assert_all_close(binary_result, exla_result, atol: 1.0e-5, rtol: 1.0e-5)
        end
      end
    end

    property "exp agrees (moderate values)" do
      check all(n <- integer(1..8), max_runs: 10) do
        x =
          for(_ <- 1..n, do: :rand.uniform() * 6 - 3)
          |> Nx.tensor(type: :f32)

        binary_result = on_binary_backend(fn -> Nx.exp(x) end)
        exla_result = Nx.Defn.jit(fn x -> Nx.exp(x) end, compiler: EXLA, client: :host).(x)
        assert_all_close(binary_result, exla_result, atol: 1.0e-4, rtol: 1.0e-4)
      end
    end

    property "log agrees (positive values)" do
      check all(n <- integer(1..8), max_runs: 10) do
        x =
          for(_ <- 1..n, do: :rand.uniform() * 5 + 0.1)
          |> Nx.tensor(type: :f32)

        binary_result = on_binary_backend(fn -> Nx.log(x) end)
        exla_result = Nx.Defn.jit(fn x -> Nx.log(x) end, compiler: EXLA, client: :host).(x)
        assert_all_close(binary_result, exla_result, atol: 1.0e-5, rtol: 1.0e-5)
      end
    end

    property "sqrt agrees (positive values)" do
      check all(n <- integer(1..8), max_runs: 10) do
        x =
          for(_ <- 1..n, do: :rand.uniform() * 10 + 0.1)
          |> Nx.tensor(type: :f32)

        binary_result = on_binary_backend(fn -> Nx.sqrt(x) end)
        exla_result = Nx.Defn.jit(fn x -> Nx.sqrt(x) end, compiler: EXLA, client: :host).(x)
        assert_all_close(binary_result, exla_result, atol: 1.0e-5, rtol: 1.0e-5)
      end
    end
  end

  # ── Binary ops ────────────────────────────────────────────────────

  @binary_ops [:add, :subtract, :multiply, :min, :max]

  describe "binary ops: BinaryBackend vs EXLA" do
    for op <- @binary_ops do
      property "#{op} agrees across backends" do
        check all(n <- integer(1..8), max_runs: 10) do
          a = random_f32({n})
          b = random_f32({n})

          binary_result = on_binary_backend(fn -> apply(Nx, unquote(op), [a, b]) end)

          exla_result =
            Nx.Defn.jit(fn {a, b} -> apply(Nx, unquote(op), [a, b]) end,
              compiler: EXLA,
              client: :host
            ).({a, b})

          assert_all_close(binary_result, exla_result, atol: 1.0e-5, rtol: 1.0e-5)
        end
      end
    end

    property "divide agrees (non-zero divisor)" do
      check all(n <- integer(1..8), max_runs: 10) do
        a = random_f32({n})

        b =
          for(_ <- 1..n, do: :rand.uniform() * 4 + 0.5)
          |> Nx.tensor(type: :f32)

        binary_result = on_binary_backend(fn -> Nx.divide(a, b) end)

        exla_result =
          Nx.Defn.jit(fn {a, b} -> Nx.divide(a, b) end, compiler: EXLA, client: :host).({a, b})

        assert_all_close(binary_result, exla_result, atol: 1.0e-5, rtol: 1.0e-5)
      end
    end
  end

  # ── Reductions ────────────────────────────────────────────────────

  describe "reductions: BinaryBackend vs EXLA" do
    for op <- [:sum, :product, :reduce_max, :reduce_min, :mean] do
      property "#{op} agrees across backends" do
        check all(n <- integer(1..16), max_runs: 10) do
          x = random_f32({n})
          binary_result = on_binary_backend(fn -> apply(Nx, unquote(op), [x]) end)

          exla_result =
            Nx.Defn.jit(fn x -> apply(Nx, unquote(op), [x]) end,
              compiler: EXLA,
              client: :host
            ).(x)

          assert_all_close(binary_result, exla_result, atol: 1.0e-4, rtol: 1.0e-4)
        end
      end
    end
  end

  # ── Dot product ───────────────────────────────────────────────────

  describe "dot: BinaryBackend vs EXLA" do
    property "vector dot agrees" do
      check all(n <- integer(1..16), max_runs: 10) do
        a = random_f32({n})
        b = random_f32({n})

        binary_result = on_binary_backend(fn -> Nx.dot(a, b) end)

        exla_result =
          Nx.Defn.jit(fn {a, b} -> Nx.dot(a, b) end, compiler: EXLA, client: :host).({a, b})

        # CPU EXLA should match BinaryBackend closely (no TF32)
        assert_all_close(binary_result, exla_result, atol: 1.0e-4, rtol: 1.0e-4)
      end
    end

    property "matmul agrees" do
      check all(
              m <- integer(1..8),
              n <- integer(1..8),
              k <- integer(1..8),
              max_runs: 8
            ) do
        a = random_f32({m, k})
        b = random_f32({k, n})

        binary_result = on_binary_backend(fn -> Nx.dot(a, b) end)

        exla_result =
          Nx.Defn.jit(fn {a, b} -> Nx.dot(a, b) end, compiler: EXLA, client: :host).({a, b})

        assert_all_close(binary_result, exla_result, atol: 1.0e-3, rtol: 1.0e-3)
      end
    end
  end

  # ── Shape ops ─────────────────────────────────────────────────────

  describe "shape ops: BinaryBackend vs EXLA" do
    property "sort agrees" do
      check all(n <- integer(1..16), max_runs: 10) do
        x = random_f32({n})
        binary_result = on_binary_backend(fn -> Nx.sort(x) end)
        exla_result = Nx.Defn.jit(fn x -> Nx.sort(x) end, compiler: EXLA, client: :host).(x)
        assert_all_close(binary_result, exla_result)
      end
    end

    property "transpose agrees" do
      check all(
              m <- integer(1..6),
              n <- integer(1..6),
              max_runs: 10
            ) do
        x = random_f32({m, n})
        binary_result = on_binary_backend(fn -> Nx.transpose(x) end)

        exla_result =
          Nx.Defn.jit(fn x -> Nx.transpose(x) end, compiler: EXLA, client: :host).(x)

        assert_all_close(binary_result, exla_result)
      end
    end

    property "concatenate agrees" do
      check all(
              n <- integer(1..8),
              m <- integer(1..8),
              max_runs: 10
            ) do
        a = random_f32({n})
        b = random_f32({m})
        binary_result = on_binary_backend(fn -> Nx.concatenate([a, b]) end)

        exla_result =
          Nx.Defn.jit(fn {a, b} -> Nx.concatenate([a, b]) end, compiler: EXLA, client: :host).(
            {a, b}
          )

        assert_all_close(binary_result, exla_result)
      end
    end
  end

  # ── LinAlg ────────────────────────────────────────────────────────

  describe "linalg: BinaryBackend vs EXLA" do
    property "QR reconstruction agrees" do
      check all(n <- integer(2..6), max_runs: 5) do
        a = random_f32({n, n})

        binary_qr =
          on_binary_backend(fn ->
            {q, r} = Nx.LinAlg.qr(a)
            Nx.dot(q, r)
          end)

        exla_qr =
          Nx.Defn.jit(
            fn a ->
              {q, r} = Nx.LinAlg.qr(a)
              Nx.dot(q, r)
            end,
            compiler: EXLA,
            client: :host
          ).(a)

        # Both should reconstruct A similarly
        assert_all_close(binary_qr, exla_qr, atol: 1.0e-3, rtol: 1.0e-3)
      end
    end

    property "determinant agrees" do
      check all(n <- integer(2..5), max_runs: 5) do
        a = Nx.add(random_f32({n, n}), Nx.multiply(n, Nx.eye(n, type: :f32)))

        binary_det = on_binary_backend(fn -> Nx.LinAlg.determinant(a) end)

        exla_det =
          Nx.Defn.jit(fn a -> Nx.LinAlg.determinant(a) end, compiler: EXLA, client: :host).(a)

        assert_all_close(binary_det, exla_det, atol: 1.0e-1, rtol: 1.0e-1)
      end
    end
  end

  # ── Gradient agreement ────────────────────────────────────────────

  describe "gradient: BinaryBackend vs EXLA" do
    property "grad of sum(sin(x)) agrees" do
      check all(n <- integer(1..8), max_runs: 8) do
        x = random_f32({n})

        binary_grad = Nx.Defn.grad(x, fn x -> Nx.sum(Nx.sin(x)) end)

        exla_grad =
          Nx.Defn.jit(fn x -> Nx.Defn.grad(x, fn x -> Nx.sum(Nx.sin(x)) end) end,
            compiler: EXLA,
            client: :host
          ).(x)

        assert_all_close(binary_grad, exla_grad, atol: 1.0e-5, rtol: 1.0e-5)
      end
    end

    property "grad of sum(x^2) agrees" do
      check all(n <- integer(1..8), max_runs: 8) do
        x = random_f32({n})
        binary_grad = Nx.Defn.grad(x, fn x -> Nx.sum(Nx.pow(x, 2)) end)

        exla_grad =
          Nx.Defn.jit(fn x -> Nx.Defn.grad(x, fn x -> Nx.sum(Nx.pow(x, 2)) end) end,
            compiler: EXLA,
            client: :host
          ).(x)

        assert_all_close(binary_grad, exla_grad, atol: 1.0e-5, rtol: 1.0e-5)
      end
    end

    property "grad of sum(matmul(x, w)) agrees" do
      check all(
              m <- integer(1..4),
              k <- integer(1..4),
              max_runs: 5
            ) do
        x = random_f32({m, k})
        w = random_f32({k, 2})

        binary_grad = Nx.Defn.grad(x, fn x -> Nx.sum(Nx.dot(x, w)) end)

        exla_grad =
          Nx.Defn.jit(fn x -> Nx.Defn.grad(x, fn x -> Nx.sum(Nx.dot(x, w)) end) end,
            compiler: EXLA,
            client: :host
          ).(x)

        assert_all_close(binary_grad, exla_grad, atol: 1.0e-3, rtol: 1.0e-3)
      end
    end
  end

  # ── GPU differential tests ────────────────────────────────────────
  # These test BinaryBackend vs EXLA GPU (CUDA). Using precision: :highest
  # for matmul ops to avoid TF32 divergence.

  @tag :cuda_required
  describe "GPU: unary ops agree" do
    for op <- [:abs, :negate, :sin, :cos, :tanh, :sigmoid, :exp] do
      @tag :cuda_required
      property "GPU #{op} agrees with BinaryBackend" do
        check all(n <- integer(1..8), max_runs: 8) do
          x =
            for(_ <- 1..n, do: :rand.uniform() * 4 - 2)
            |> Nx.tensor(type: :f32)

          binary_result = on_binary_backend(fn -> apply(Nx, unquote(op), [x]) end)

          exla_result =
            Nx.Defn.jit(fn x -> apply(Nx, unquote(op), [x]) end,
              compiler: EXLA
            ).(x)

          assert_all_close(binary_result, exla_result, atol: 1.0e-4, rtol: 1.0e-4)
        end
      end
    end
  end

  @tag :cuda_required
  describe "GPU: matmul agrees (precision: :highest)" do
    @tag :cuda_required
    property "GPU matmul with highest precision" do
      check all(
              m <- integer(1..8),
              n <- integer(1..8),
              k <- integer(1..8),
              max_runs: 5
            ) do
        a = random_f32({m, k})
        b = random_f32({k, n})

        binary_result = on_binary_backend(fn -> Nx.dot(a, b) end)

        exla_result =
          Nx.Defn.jit(fn {a, b} -> Nx.dot(a, b) end,
            compiler: EXLA,
            precision: :highest
          ).({a, b})

        assert_all_close(binary_result, exla_result, atol: 1.0e-3, rtol: 1.0e-3)
      end
    end
  end

  @tag :cuda_required
  describe "GPU: reductions agree" do
    for op <- [:sum, :reduce_max, :reduce_min, :mean] do
      @tag :cuda_required
      property "GPU #{op} agrees" do
        check all(n <- integer(1..16), max_runs: 8) do
          x = random_f32({n})
          binary_result = on_binary_backend(fn -> apply(Nx, unquote(op), [x]) end)

          exla_result =
            Nx.Defn.jit(fn x -> apply(Nx, unquote(op), [x]) end,
              compiler: EXLA
            ).(x)

          assert_all_close(binary_result, exla_result, atol: 1.0e-4, rtol: 1.0e-4)
        end
      end
    end
  end

  # ── More linalg cross-backend ─────────────────────────────────────

  describe "linalg cross-backend: more ops" do
    property "cholesky agrees" do
      check all(n <- integer(2..5), max_runs: 5) do
        a = random_f32({n, n})
        pd = Nx.add(Nx.dot(Nx.transpose(a), a), Nx.multiply(n, Nx.eye(n, type: :f32)))

        binary_l = on_binary_backend(fn -> Nx.LinAlg.cholesky(pd) end)

        exla_l =
          Nx.Defn.jit(fn a -> Nx.LinAlg.cholesky(a) end, compiler: EXLA, client: :host).(pd)

        assert_all_close(binary_l, exla_l, atol: 1.0e-3, rtol: 1.0e-3)
      end
    end

    property "SVD singular values agree" do
      check all(n <- integer(2..5), max_runs: 5) do
        a = random_f32({n, n})

        binary_s =
          on_binary_backend(fn ->
            {_u, s, _v} = Nx.LinAlg.svd(a)
            s
          end)

        exla_s =
          Nx.Defn.jit(
            fn a ->
              {_u, s, _v} = Nx.LinAlg.svd(a)
              s
            end,
            compiler: EXLA,
            client: :host
          ).(a)

        assert_all_close(binary_s, exla_s, atol: 1.0e-2, rtol: 1.0e-2)
      end
    end

    property "solve agrees" do
      check all(n <- integer(2..5), max_runs: 5) do
        a = random_f32({n, n})
        pd = Nx.add(Nx.dot(Nx.transpose(a), a), Nx.multiply(n, Nx.eye(n, type: :f32)))

        b =
          for(_ <- 1..n, do: :rand.uniform() * 4 - 2)
          |> Nx.tensor(type: :f32)

        binary_x = on_binary_backend(fn -> Nx.LinAlg.solve(pd, b) end)

        exla_x =
          Nx.Defn.jit(fn {a, b} -> Nx.LinAlg.solve(a, b) end, compiler: EXLA, client: :host).(
            {pd, b}
          )

        assert_all_close(binary_x, exla_x, atol: 1.0e-2, rtol: 1.0e-2)
      end
    end

    property "invert agrees" do
      check all(n <- integer(2..5), max_runs: 5) do
        a = random_f32({n, n})
        pd = Nx.add(Nx.dot(Nx.transpose(a), a), Nx.multiply(n, Nx.eye(n, type: :f32)))

        binary_inv = on_binary_backend(fn -> Nx.LinAlg.invert(pd) end)

        exla_inv =
          Nx.Defn.jit(fn a -> Nx.LinAlg.invert(a) end, compiler: EXLA, client: :host).(pd)

        assert_all_close(binary_inv, exla_inv, atol: 1.0e-2, rtol: 1.0e-2)
      end
    end

    property "eigh eigenvalues agree" do
      check all(n <- integer(2..5), max_runs: 5) do
        a = random_f32({n, n})
        sym = Nx.add(a, Nx.transpose(a)) |> Nx.divide(2)

        binary_evals =
          on_binary_backend(fn ->
            {evals, _} = Nx.LinAlg.eigh(sym)
            evals
          end)

        exla_evals =
          Nx.Defn.jit(
            fn a ->
              {evals, _} = Nx.LinAlg.eigh(a)
              evals
            end,
            compiler: EXLA,
            client: :host
          ).(sym)

        assert_all_close(binary_evals, exla_evals, atol: 1.0e-2, rtol: 1.0e-2)
      end
    end

    property "triangular_solve agrees" do
      check all(n <- integer(2..5), max_runs: 5) do
        a = random_f32({n, n})
        upper = Nx.add(Nx.triu(a), Nx.multiply(n, Nx.eye(n, type: :f32)))

        b =
          for(_ <- 1..n, do: :rand.uniform() * 4 - 2)
          |> Nx.tensor(type: :f32)

        binary_x = on_binary_backend(fn -> Nx.LinAlg.triangular_solve(upper, b, lower: false) end)

        exla_x =
          Nx.Defn.jit(
            fn {a, b} -> Nx.LinAlg.triangular_solve(a, b, lower: false) end,
            compiler: EXLA,
            client: :host
          ).({upper, b})

        assert_all_close(binary_x, exla_x, atol: 1.0e-3, rtol: 1.0e-3)
      end
    end
  end

  # ── More cross-backend gradient agreement ─────────────────────────

  describe "more gradient cross-backend agreement" do
    property "grad of tanh agrees" do
      check all(n <- integer(1..6), max_runs: 6) do
        x = random_f32({n})
        binary_grad = Nx.Defn.grad(x, fn x -> Nx.sum(Nx.tanh(x)) end)

        exla_grad =
          Nx.Defn.jit(fn x -> Nx.Defn.grad(x, fn x -> Nx.sum(Nx.tanh(x)) end) end,
            compiler: EXLA,
            client: :host
          ).(x)

        assert_all_close(binary_grad, exla_grad, atol: 1.0e-4, rtol: 1.0e-4)
      end
    end

    property "grad of exp agrees" do
      check all(n <- integer(1..6), max_runs: 6) do
        x =
          for(_ <- 1..n, do: :rand.uniform() * 4 - 2)
          |> Nx.tensor(type: :f32)

        binary_grad = Nx.Defn.grad(x, fn x -> Nx.sum(Nx.exp(x)) end)

        exla_grad =
          Nx.Defn.jit(fn x -> Nx.Defn.grad(x, fn x -> Nx.sum(Nx.exp(x)) end) end,
            compiler: EXLA,
            client: :host
          ).(x)

        assert_all_close(binary_grad, exla_grad, atol: 1.0e-4, rtol: 1.0e-4)
      end
    end

    property "grad of sigmoid agrees" do
      check all(n <- integer(1..6), max_runs: 6) do
        x = random_f32({n})
        binary_grad = Nx.Defn.grad(x, fn x -> Nx.sum(Nx.sigmoid(x)) end)

        exla_grad =
          Nx.Defn.jit(fn x -> Nx.Defn.grad(x, fn x -> Nx.sum(Nx.sigmoid(x)) end) end,
            compiler: EXLA,
            client: :host
          ).(x)

        assert_all_close(binary_grad, exla_grad, atol: 1.0e-4, rtol: 1.0e-4)
      end
    end

    property "grad of sin*cos agrees" do
      check all(n <- integer(1..6), max_runs: 6) do
        x = random_f32({n})
        fun = fn x -> Nx.sum(Nx.multiply(Nx.sin(x), Nx.cos(x))) end
        binary_grad = Nx.Defn.grad(x, fun)

        exla_grad =
          Nx.Defn.jit(
            fn x -> Nx.Defn.grad(x, fn x -> Nx.sum(Nx.multiply(Nx.sin(x), Nx.cos(x))) end) end,
            compiler: EXLA,
            client: :host
          ).(x)

        assert_all_close(binary_grad, exla_grad, atol: 1.0e-4, rtol: 1.0e-4)
      end
    end

    property "grad of multiply(x, y) wrt both agrees" do
      check all(n <- integer(1..6), max_runs: 6) do
        x = random_f32({n})
        y = random_f32({n})

        {binary_gx, binary_gy} =
          Nx.Defn.grad({x, y}, fn {x, y} -> Nx.sum(Nx.multiply(x, y)) end)

        {exla_gx, exla_gy} =
          Nx.Defn.jit(
            fn {x, y} ->
              Nx.Defn.grad({x, y}, fn {x, y} -> Nx.sum(Nx.multiply(x, y)) end)
            end,
            compiler: EXLA,
            client: :host
          ).({x, y})

        assert_all_close(binary_gx, exla_gx, atol: 1.0e-4, rtol: 1.0e-4)
        assert_all_close(binary_gy, exla_gy, atol: 1.0e-4, rtol: 1.0e-4)
      end
    end
  end
end
