defmodule Nx.FuzzDarklinesTest do
  @moduledoc """
  Coverage-guided round 2: targets for lines that no test in the tree had
  ever executed — semantic branches, non-ArgumentError raises, inspect
  paths, and defn trace-time validations. Each test names the module:line
  it lights.
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  import Nx.Testing

  @fuzz_scale String.to_integer(System.get_env("FUZZ_SCALE", "1"))

  # Note: BinaryBackend.sort has scalar-shape branches, but Nx validates
  # the axis against the rank first, so they are unreachable defensive
  # code from the public API.

  describe "argmax/argmin tie_break comparators (non_finite_lt/gt arms)" do
    property "tie_break :low/:high pick first/last among ties, incl. non-finites" do
      check all(
              tie_break <- member_of([:low, :high]),
              max_runs: 20 * @fuzz_scale
            ) do
        check all(
                values <-
                  list_of(
                    frequency([
                      {3, member_of([1.0, 2.0, 2.0, -1.0])},
                      {1, member_of([:nan, :infinity, :neg_infinity])}
                    ]),
                    length: 6
                  ),
                max_runs: 1
              ) do
          t = Nx.tensor(values, type: {:f, 64})

          for op <- [:argmax, :argmin] do
            idx = Nx.to_number(apply(Nx, op, [t, [tie_break: tie_break]]))
            picked = Enum.at(values, idx)

            # the picked element must be a witness of the extremum: for
            # :low the FIRST index with that value, for :high the LAST
            witness_indices =
              values
              |> Enum.with_index()
              |> Enum.filter(fn {v, _} -> v == picked or (v == :nan and picked == :nan) end)
              |> Enum.map(&elem(&1, 1))

            case tie_break do
              :low -> assert idx == Enum.min(witness_indices), "#{op} #{inspect(values)}"
              :high -> assert idx == Enum.max(witness_indices), "#{op} #{inspect(values)}"
            end
          end
        end
      end
    end
  end

  describe "complex reduce accumulator (scalar_to_number Complex arm)" do
    test "reduce over c64 with a complex init equals sum" do
      z = Nx.complex(Nx.tensor([1.0, 2.0, 3.0]), Nx.tensor([-1.0, 0.5, 2.0]))
      init = Nx.complex(Nx.tensor(0.0), Nx.tensor(0.0))

      reduced = Nx.reduce(z, init, fn a, b -> Nx.add(a, b) end)
      assert_all_close(Nx.real(reduced), Nx.real(Nx.sum(z)), atol: 1.0e-6)
      assert_all_close(Nx.imag(reduced), Nx.imag(Nx.sum(z)), atol: 1.0e-6)
    end
  end

  describe "multi-axis aggregation paths (aggregate_read recursion)" do
    property "multi-axis sum equals iterated single-axis sums" do
      check all(
              axes <- member_of([[0, 1], [1, 2], [0, 2], [0, 1, 2], [2, 3], [1, 3]]),
              max_runs: 20 * @fuzz_scale
            ) do
        t = Nx.iota({2, 3, 4, 5}, type: {:f, 64})

        via_multi = Nx.sum(t, axes: axes)

        via_iterated =
          axes
          |> Enum.sort(:desc)
          |> Enum.reduce(t, fn axis, acc -> Nx.sum(acc, axes: [axis]) end)

        assert via_multi == via_iterated

        assert Nx.reduce_max(t, axes: axes) ==
                 axes
                 |> Enum.sort(:desc)
                 |> Enum.reduce(t, fn axis, acc -> Nx.reduce_max(acc, axes: [axis]) end)
      end
    end
  end

  describe "vectorized linalg tuple re-vectorization (LinAlg apply_vectorized arms)" do
    test "vectorized qr (2-tuple) and lu (3-tuple) keep the vectorized axis" do
      t = Nx.iota({2, 3, 3}, type: {:f, 64}) |> Nx.add(Nx.eye(3)) |> Nx.vectorize(:b)

      {q, r} = Nx.LinAlg.qr(t)
      assert Keyword.keys(q.vectorized_axes) == [:b]
      assert Keyword.keys(r.vectorized_axes) == [:b]

      {p, l, u} = Nx.LinAlg.lu(t)

      for part <- [p, l, u] do
        assert Keyword.keys(part.vectorized_axes) == [:b]
      end
    end
  end

  describe "Expr trace-time contracts" do
    test "Nx.to_binary inside defn raises the cannot-invoke error" do
      assert_raise ArgumentError, ~r/cannot invoke to_binary\/2 on Nx.Defn.Expr/, fn ->
        Nx.Defn.jit(fn x -> Nx.to_binary(x) end).(Nx.tensor([1.0]))
      end
    end

    test "backend_transfer inside defn: to Expr is identity, to others raises" do
      result = Nx.Defn.jit(fn x -> Nx.backend_transfer(x, Nx.Defn.Expr) end).(Nx.tensor([1.0]))
      assert result == Nx.tensor([1.0])

      assert_raise ArgumentError, ~r/cannot invoke backend_transfer/, fn ->
        Nx.Defn.jit(fn x -> Nx.backend_transfer(x, Nx.BinaryBackend) end).(Nx.tensor([1.0]))
      end
    end

    test "window_reduce fun returning a non-scalar raises at trace time" do
      assert_raise RuntimeError, ~r/window_reduce function must return a scalar/, fn ->
        Nx.Defn.jit(fn x ->
          Nx.window_reduce(x, 0.0, {2}, fn a, b -> Nx.stack([a, b]) end)
        end).(Nx.tensor([1.0, 2.0, 3.0]))
      end
    end
  end

  describe "Expr inspect paths" do
    test "print_id renders the expression ref header" do
      expr_tensor = Nx.Defn.debug_expr(fn x -> Nx.add(x, 1) end).(Nx.tensor(1.0))
      assert inspect(expr_tensor, custom_options: [print_id: true]) =~ "Nx.Defn.Expr<"
    end

    test "expressions with more than 26 nodes use two-letter names" do
      # sin chain defeats constant folding, giving 30 distinct nodes
      fun = fn x ->
        Enum.reduce(1..30, x, fn _, acc -> Nx.sin(acc) end)
      end

      rendered = inspect(Nx.Defn.debug_expr(fun).(Nx.tensor(1.0)))
      # base-26 with a as the zero digit: z is followed by ba
      assert rendered =~ " ba = "
    end
  end

  describe "Evaluator contracts" do
    test "passing an Expr tensor as a jit argument raises" do
      expr_tensor = Nx.Defn.debug_expr(fn x -> Nx.add(x, 1) end).(Nx.tensor(1.0))

      assert_raise ArgumentError, ~r/cannot pass a tensor expression as argument/, fn ->
        Nx.Defn.jit_apply(fn y -> Nx.add(y, 1) end, [expr_tensor])
      end
    end

    test "shard_jit on the Evaluator raises its unsupported error" do
      mesh = %Nx.Mesh{name: "m", shape: {1}}

      assert_raise RuntimeError, ~r/sharding is not supported by Nx.Defn.Evaluator/, fn ->
        Nx.Defn.shard_jit_apply(fn x -> x end, mesh, [[Nx.tensor(1.0)]],
          compiler: Nx.Defn.Evaluator
        )
      end
    end
  end

  describe "Grad dark arms" do
    test "grad of a closure-returned concrete tensor raises" do
      constant = Nx.tensor(2.0)

      assert_raise ArgumentError, ~r/can only compute gradients of tensor expressions/, fn ->
        Nx.Defn.grad(Nx.tensor(1.0), fn _ -> constant end)
      end
    end

    test "closure environment containing a map of tensors stops grads through it" do
      captured = %{weight: Nx.tensor(3.0)}

      grad = Nx.Defn.grad(Nx.tensor(2.0), fn x -> Nx.multiply(x, captured.weight) end)
      assert_all_close(grad, Nx.tensor(3.0), atol: 1.0e-6)
    end

    test "grad treats runtime_call output as a constant leaf" do
      grad =
        Nx.Defn.grad(Nx.tensor([1.0, 2.0]), fn x ->
          observed = Nx.runtime_call(Nx.template({2}, :f32), x, [], fn t, _opts -> t end)
          Nx.sum(Nx.add(Nx.multiply(observed, 0.0), x))
        end)

      assert_all_close(grad, Nx.tensor([1.0, 1.0]), atol: 1.0e-6)
    end

    test "grad wrt a plain tensor against a vectorized closure tensor" do
      x_vec = Nx.tensor([[1.0, 2.0], [3.0, 4.0]]) |> Nx.vectorize(:b)

      grad = Nx.Defn.grad(Nx.tensor(2.0), fn w -> Nx.sum(Nx.multiply(x_vec, w)) end)

      # d/dw sum(x * w) = sum(x) per batch: [3.0, 7.0]
      assert Keyword.keys(grad.vectorized_axes) == [:b]

      assert_all_close(Nx.devectorize(grad, keep_names: false), Nx.tensor([3.0, 7.0]),
        atol: 1.0e-6
      )
    end
  end

  describe "Matrix complex-conjugate arms" do
    test "triangular_solve on complex tensors satisfies A x = b" do
      a =
        Nx.complex(
          Nx.tensor([[2.0, 0.0], [1.0, 3.0]]),
          Nx.tensor([[0.5, 0.0], [-0.5, 1.0]])
        )

      b = Nx.complex(Nx.tensor([1.0, 2.0]), Nx.tensor([0.0, -1.0]))

      x = Nx.LinAlg.triangular_solve(a, b)
      reconstructed = Nx.dot(a, x)

      assert_all_close(Nx.real(reconstructed), Nx.real(b), atol: 1.0e-5)
      assert_all_close(Nx.imag(reconstructed), Nx.imag(b), atol: 1.0e-5)
    end

    test "triangular_solve with transform_a: :conjugate on complex tensors" do
      a = Nx.complex(Nx.tensor([[2.0, 0.0], [1.0, 3.0]]), Nx.tensor([[0.5, 0.0], [-0.5, 1.0]]))
      b = Nx.complex(Nx.tensor([1.0, 2.0]), Nx.tensor([0.0, -1.0]))

      x = Nx.LinAlg.triangular_solve(a, b, transform_a: :conjugate)
      conj_a = Nx.conjugate(a)
      reconstructed = Nx.dot(conj_a, x)

      assert_all_close(Nx.real(reconstructed), Nx.real(b), atol: 1.0e-5)
      assert_all_close(Nx.imag(reconstructed), Nx.imag(b), atol: 1.0e-5)
    end
  end

  describe "defn while validations (trace-time CompileError arms)" do
    defmodule ScalarGenerator do
      import Nx.Defn

      defn loop_over(x, gen) do
        {acc, _} =
          while {acc = x, gen}, _i <- gen do
            {acc, gen}
          end

        acc
      end
    end

    test "scalar tensor as while generator raises at trace time" do
      assert_raise CompileError, ~r/cannot have a scalar tensor as generator/, fn ->
        ScalarGenerator.loop_over(Nx.tensor(1.0), Nx.tensor(5))
      end
    end
  end
end
