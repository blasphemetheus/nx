defmodule Nx.Defn.RecomputeTest do
  use ExUnit.Case, async: true

  import Nx.Defn
  import Nx.Testing

  # The evaluator calls these with tensors, tracing calls them with
  # expressions. Only the evaluator calls are counted.
  def counted_exp(x), do: counted(:exp, x, &Nx.exp/1)
  def counted_sin(y), do: counted(:sin, y, &Nx.sin/1)

  defp counted(name, %Nx.Tensor{data: %Nx.Defn.Expr{}} = x, fun) when is_atom(name), do: fun.(x)

  defp counted(name, x, fun) do
    send(self(), {:ran, name})
    fun.(x)
  end

  defp runs(name) do
    receive do
      {:ran, ^name} -> 1 + runs(name)
    after
      0 -> 0
    end
  end

  defn two_uses(x) do
    y = recompute(x, &counted_exp/1)
    Nx.sum(y) + Nx.reduce_max(y)
  end

  defn two_uses_plain(x) do
    y = Nx.exp(x)
    Nx.sum(y) + Nx.reduce_max(y)
  end

  defn chain(x) do
    y = recompute(x, &counted_exp/1)
    z = recompute(y, &counted_sin/1)
    Nx.sum(z) + Nx.reduce_max(z)
  end

  defn chain_plain(x) do
    z = x |> Nx.exp() |> Nx.sin()
    Nx.sum(z) + Nx.reduce_max(z)
  end

  defn dense(x, w1, w2), do: x |> Nx.dot(w1) |> Nx.max(0) |> Nx.dot(w2) |> Nx.tanh()

  defn mlp(weights, x) do
    y = recompute([x, weights], fn x, weights -> dense(x, weights.a, weights.b) end)
    z = recompute([y, weights], fn y, weights -> dense(y, weights.c, weights.d) end)
    Nx.sum(z * z)
  end

  defn mlp_plain(weights, x) do
    z = x |> dense(weights.a, weights.b) |> dense(weights.c, weights.d)
    Nx.sum(z * z)
  end

  defn tuple_output(x) do
    {a, b} = recompute(x, fn x -> {Nx.sin(x), Nx.cos(x)} end)
    Nx.sum(a * b)
  end

  defn tuple_output_plain(x), do: Nx.sum(Nx.sin(x) * Nx.cos(x))

  defn recompute_in_checkpoint(weights, x) do
    z =
      checkpoint([x, weights], fn x, weights ->
        y = recompute([x, weights], fn x, weights -> dense(x, weights.a, weights.b) end)
        dense(y, weights.c, weights.d) + Nx.sum(y)
      end)

    Nx.sum(z * z)
  end

  defn checkpoint_in_recompute(weights, x) do
    z =
      recompute([x, weights], fn x, weights ->
        y = checkpoint([x, weights], fn x, weights -> dense(x, weights.a, weights.b) end)
        dense(y, weights.c, weights.d) + Nx.sum(y)
      end)

    Nx.sum(z * z)
  end

  defn nested_plain(weights, x) do
    y = dense(x, weights.a, weights.b)
    z = dense(y, weights.c, weights.d) + Nx.sum(y)
    Nx.sum(z * z)
  end

  defn recompute_then_checkpoint(weights, x) do
    y = recompute([x, weights], fn x, weights -> dense(x, weights.a, weights.b) end)
    z = checkpoint([y, weights], fn y, weights -> dense(y, weights.c, weights.d) end)
    Nx.sum(z * z)
  end

  defn checkpoint_then_recompute(weights, x) do
    y = checkpoint([x, weights], fn x, weights -> dense(x, weights.a, weights.b) end)
    z = recompute([y, weights], fn y, weights -> dense(y, weights.c, weights.d) end)
    Nx.sum(z * z)
  end

  defn counted_recompute_then_checkpoint(x) do
    y = recompute(x, &counted_exp/1)
    z = checkpoint(y, &counted_sin/1)
    Nx.sum(z) + Nx.reduce_max(z)
  end

  defn counted_checkpoint_then_recompute(x) do
    y = checkpoint(x, &counted_exp/1)
    z = recompute(y, &counted_sin/1)
    Nx.sum(z) + Nx.reduce_max(z)
  end

  defn counted_plain(x) do
    z = x |> Nx.exp() |> Nx.sin()
    Nx.sum(z) + Nx.reduce_max(z)
  end

  defp assert_same_grads(fun, plain) do
    x = Nx.iota({2, 3}, type: :f32) |> Nx.divide(6)
    weights = weights()

    assert_all_close(fun.(weights, x), plain.(weights, x))

    grads = Nx.Defn.grad(weights, &fun.(&1, x))
    plain_grads = Nx.Defn.grad(weights, &plain.(&1, x))

    for name <- [:a, :b, :c, :d] do
      assert_all_close(grads[name], plain_grads[name])
    end

    assert_all_close(Nx.Defn.grad(x, &fun.(weights, &1)), Nx.Defn.grad(x, &plain.(weights, &1)))
  end

  defp weights do
    key = Nx.Random.key(7)
    {a, key} = Nx.Random.normal(key, shape: {3, 4})
    {b, key} = Nx.Random.normal(key, shape: {4, 3})
    {c, key} = Nx.Random.normal(key, shape: {3, 4})
    {d, _key} = Nx.Random.normal(key, shape: {4, 3})
    %{a: a, b: b, c: c, d: d}
  end

  describe "recompute/2" do
    test "returns the value of the function" do
      x = Nx.tensor([0.1, 0.2, 0.3])
      assert_all_close(two_uses(x), two_uses_plain(x))
      assert_all_close(chain(x), chain_plain(x))
    end

    test "runs the function once for every use of the result" do
      two_uses(Nx.tensor([0.1, 0.2, 0.3]))
      assert runs(:exp) == 2
    end

    # The evaluator keeps the evaluated inputs of a recompute block, so the
    # first block of a chain runs once. EXLA lowers it again for every use.
    test "the evaluator keeps the inputs of a recompute block" do
      chain(Nx.tensor([0.1, 0.2, 0.3]))
      assert runs(:sin) == 2
      assert runs(:exp) == 1
    end

    test "the gradient of a chain matches the plain function" do
      x = Nx.iota({2, 3}, type: :f32) |> Nx.divide(6)
      weights = weights()

      grads = Nx.Defn.grad(weights, &mlp(&1, x))
      plain_grads = Nx.Defn.grad(weights, &mlp_plain(&1, x))

      for name <- [:a, :b, :c, :d] do
        assert_all_close(grads[name], plain_grads[name])
      end

      assert_all_close(
        Nx.Defn.grad(x, &mlp(weights, &1)),
        Nx.Defn.grad(x, &mlp_plain(weights, &1))
      )
    end

    test "returns a tuple of tensors" do
      x = Nx.tensor([0.1, 0.2, 0.3])
      assert_all_close(tuple_output(x), tuple_output_plain(x))

      assert_all_close(
        Nx.Defn.grad(x, &tuple_output/1),
        Nx.Defn.grad(x, &tuple_output_plain/1)
      )
    end

    test "is the function itself outside defn" do
      x = Nx.tensor([0.1, 0.2, 0.3])
      assert_all_close(Nx.Defn.Kernel.recompute(x, &Nx.exp/1), Nx.exp(x))
    end

    test "raises when the function arity does not match the inputs" do
      assert_raise ArgumentError, ~r/expected a function of arity 2/, fn ->
        Nx.Defn.Kernel.recompute([Nx.tensor(1), Nx.tensor(2)], &Nx.exp/1)
      end
    end
  end

  describe "recompute/2 with checkpoint/2" do
    test "recompute inside a checkpoint matches the plain function" do
      assert_same_grads(&recompute_in_checkpoint/2, &nested_plain/2)
    end

    test "checkpoint inside a recompute matches the plain function" do
      assert_same_grads(&checkpoint_in_recompute/2, &nested_plain/2)
    end

    test "a recompute result as a checkpoint input matches the plain function" do
      assert_same_grads(&recompute_then_checkpoint/2, &mlp_plain/2)
    end

    test "a checkpoint result as a recompute input matches the plain function" do
      assert_same_grads(&checkpoint_then_recompute/2, &mlp_plain/2)
    end

    test "the second-order gradient of a mixed chain matches the plain function" do
      x = Nx.iota({2, 3}, type: :f32) |> Nx.divide(6)
      weights = weights()

      second = fn fun ->
        Nx.Defn.grad(x, fn x -> Nx.sum(Nx.Defn.grad(x, &fun.(weights, &1))) end)
      end

      assert_all_close(second.(&recompute_then_checkpoint/2), second.(&mlp_plain/2))
      assert_all_close(second.(&checkpoint_then_recompute/2), second.(&mlp_plain/2))
    end

    # Outside grad a checkpoint is the plain function, so its result is kept
    # and its recompute input has one use.
    test "a checkpoint keeps its result, so its recompute input runs once" do
      x = Nx.tensor([0.1, 0.2, 0.3])
      assert_all_close(counted_recompute_then_checkpoint(x), counted_plain(x))
      assert runs(:sin) == 1
      assert runs(:exp) == 1
    end

    test "a recompute runs at every use and reads the kept checkpoint result" do
      x = Nx.tensor([0.1, 0.2, 0.3])
      assert_all_close(counted_checkpoint_then_recompute(x), counted_plain(x))
      assert runs(:sin) == 2
      assert runs(:exp) == 1
    end
  end
end
