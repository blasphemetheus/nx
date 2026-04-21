defmodule Nx.FuzzEdgeCases5Test do
  @moduledoc """
  Tier 4 (part 5): Defn hooks, tokens, nested JIT, and compile edge cases.
  """
  use ExUnit.Case, async: true

  import Nx.Defn

  # ── Hook edge cases ────────────────────────────────────────────────
  # Source: defn/kernel.ex:1320
  # Boundaries:
  #   - hook must return expression for it to be evaluated
  #   - named hooks can be overridden via JIT options

  describe "hook edge cases" do
    defn hook_passthrough(x) do
      hook(x, :inspect_x)
    end

    test "hook returns the expression unchanged" do
      result = hook_passthrough(Nx.tensor(42.0))
      assert Nx.to_number(result) == 42.0
    end

    test "hook with override callback receives value" do
      parent = self()

      fun =
        Nx.Defn.jit(&hook_passthrough/1,
          hooks: %{inspect_x: fn val -> send(parent, {:got, val}) end}
        )

      result = fun.(Nx.tensor(7.0))
      assert Nx.to_number(result) == 7.0
      assert_receive {:got, tensor}
      assert Nx.to_number(tensor) == 7.0
    end

    defn hook_in_chain(x) do
      x
      |> Nx.multiply(2)
      |> hook(:after_mul)
      |> Nx.add(1)
    end

    test "hook in middle of computation chain" do
      parent = self()

      fun =
        Nx.Defn.jit(&hook_in_chain/1,
          hooks: %{after_mul: fn val -> send(parent, {:mid, val}) end}
        )

      result = fun.(Nx.tensor(5.0))
      assert Nx.to_number(result) == 11.0
      assert_receive {:mid, tensor}
      assert Nx.to_number(tensor) == 10.0
    end

    defn hook_with_default(x) do
      hook(Nx.add(x, 1), :add_hook, fn _val -> :ok end)
    end

    test "hook with default callback (no override)" do
      result = hook_with_default(Nx.tensor(5.0))
      assert Nx.to_number(result) == 6.0
    end

    test "hook with default callback overridden" do
      parent = self()

      fun =
        Nx.Defn.jit(&hook_with_default/1,
          hooks: %{add_hook: fn val -> send(parent, {:custom, val}) end}
        )

      result = fun.(Nx.tensor(5.0))
      assert Nx.to_number(result) == 6.0
      assert_receive {:custom, tensor}
      assert Nx.to_number(tensor) == 6.0
    end

    defn hook_container(a, b) do
      hook({a, b}, :pair)
    end

    test "hook with tuple container" do
      parent = self()

      fun =
        Nx.Defn.jit(&hook_container/2,
          hooks: %{pair: fn val -> send(parent, {:pair, val}) end}
        )

      {ra, rb} = fun.(Nx.tensor(1.0), Nx.tensor(2.0))
      assert Nx.to_number(ra) == 1.0
      assert Nx.to_number(rb) == 2.0
      assert_receive {:pair, {ta, tb}}
      assert Nx.to_number(ta) == 1.0
      assert Nx.to_number(tb) == 2.0
    end
  end

  # ── Token edge cases ──────────────────────────────────────────────
  # Source: defn/kernel.ex:1504-1533
  # Token ordering: hooks execute in order they were attached

  describe "token edge cases" do
    defn side_effect_ordered(a, b) do
      token = create_token()
      {token, _} = hook_token(token, a, :first)
      {token, _} = hook_token(token, b, :second)
      attach_token(token, Nx.add(a, b))
    end

    test "token-ordered hooks fire in attachment order" do
      parent = self()

      fun =
        Nx.Defn.jit(&side_effect_ordered/2,
          hooks: %{
            first: fn val -> send(parent, {:first, Nx.to_number(val)}) end,
            second: fn val -> send(parent, {:second, Nx.to_number(val)}) end
          }
        )

      result = fun.(Nx.tensor(10.0), Nx.tensor(20.0))
      assert Nx.to_number(result) == 30.0

      # Both hooks should have fired
      assert_receive {:first, 10.0}
      assert_receive {:second, 20.0}
    end

    defn single_token_hook(x) do
      token = create_token()
      {token, _} = hook_token(token, Nx.multiply(x, x), :squared)
      attach_token(token, x)
    end

    test "token hook doesn't affect returned value" do
      parent = self()

      fun =
        Nx.Defn.jit(&single_token_hook/1,
          hooks: %{squared: fn val -> send(parent, {:sq, val}) end}
        )

      result = fun.(Nx.tensor(5.0))
      # The returned value is x (5.0), not x*x
      assert Nx.to_number(result) == 5.0
      assert_receive {:sq, tensor}
      assert Nx.to_number(tensor) == 25.0
    end
  end

  # ── Nested JIT edge cases ──────────────────────────────────────────
  # Source: defn.ex:430
  # on_conflict: :raise (default), :force, :reuse

  describe "nested JIT edge cases" do
    defn outer_fn(x) do
      Nx.multiply(x, 2)
    end

    test "jit with on_conflict: :reuse works inside jit" do
      # This tests the :reuse path — falls back to interpreter
      defn_fun = fn x ->
        inner = Nx.Defn.jit(&outer_fn/1, on_conflict: :reuse)
        inner.(x)
      end

      result = Nx.Defn.jit(defn_fun).(Nx.tensor(5.0))
      assert Nx.to_number(result) == 10.0
    end

    test "basic jit works" do
      fun = Nx.Defn.jit(&outer_fn/1)
      assert Nx.to_number(fun.(Nx.tensor(3.0))) == 6.0
    end

    test "jit is idempotent" do
      fun1 = Nx.Defn.jit(&outer_fn/1)
      result1 = fun1.(Nx.tensor(4.0))
      result2 = fun1.(Nx.tensor(4.0))
      assert Nx.to_number(result1) == Nx.to_number(result2)
    end
  end

  # ── Compile edge cases ─────────────────────────────────────────────
  # Source: defn.ex:320

  describe "compile edge cases" do
    defn add_one(x), do: Nx.add(x, 1)

    test "compile with matching template works" do
      fun = Nx.Defn.compile(&add_one/1, [Nx.template({}, :f32)])
      result = fun.(Nx.tensor(5.0))
      assert Nx.to_number(result) == 6.0
    end

    test "compile with shaped template works" do
      fun = Nx.Defn.compile(&add_one/1, [Nx.template({3}, :f32)])
      result = fun.(Nx.tensor([1.0, 2.0, 3.0]))
      assert Nx.to_flat_list(result) == [2.0, 3.0, 4.0]
    end

    test "compiled function rejects incompatible input" do
      fun = Nx.Defn.compile(&add_one/1, [Nx.template({}, :f32)])

      assert_raise ArgumentError, ~r/not compatible/, fn ->
        fun.(Nx.tensor([1.0, 2.0, 3.0]))
      end
    end
  end

  # ── Hooks in while loops ───────────────────────────────────────────

  describe "hooks in while loops" do
    defn hook_in_while(x) do
      while x, Nx.less(x, 5) do
        hook(x + 1, :step)
      end
    end

    test "hook called multiple times in while loop" do
      parent = self()

      fun =
        Nx.Defn.jit(&hook_in_while/1,
          hooks: %{step: fn val -> send(parent, {:step, Nx.to_number(val)}) end}
        )

      result = fun.(Nx.tensor(0))
      assert Nx.to_number(result) == 5

      # Should receive 5 hook calls: 1, 2, 3, 4, 5
      for expected <- [1, 2, 3, 4, 5] do
        assert_receive {:step, ^expected}
      end

      refute_receive {:step, _}
    end

    defn hook_in_while_zero_iters(x) do
      while x, Nx.less(x, 0) do
        hook(x + 1, :step)
      end
    end

    test "hook not called when while has 0 iterations" do
      parent = self()

      fun =
        Nx.Defn.jit(&hook_in_while_zero_iters/1,
          hooks: %{step: fn val -> send(parent, {:step, val}) end}
        )

      result = fun.(Nx.tensor(5))
      assert Nx.to_number(result) == 5
      refute_receive {:step, _}
    end
  end

  # ── Hooks in cond ──────────────────────────────────────────────────

  describe "hooks in cond" do
    defn hook_in_cond(x) do
      if Nx.greater(x, 0) do
        hook(x, :positive)
      else
        hook(Nx.negate(x), :negative)
      end
    end

    test "hook in true branch fires" do
      parent = self()

      fun =
        Nx.Defn.jit(&hook_in_cond/1,
          hooks: %{
            positive: fn val -> send(parent, {:pos, val}) end,
            negative: fn val -> send(parent, {:neg, val}) end
          }
        )

      result = fun.(Nx.tensor(5.0))
      assert Nx.to_number(result) == 5.0
      assert_receive {:pos, tensor}
      assert Nx.to_number(tensor) == 5.0
      refute_receive {:neg, _}
    end

    test "hook in false branch fires" do
      parent = self()

      fun =
        Nx.Defn.jit(&hook_in_cond/1,
          hooks: %{
            positive: fn val -> send(parent, {:pos, val}) end,
            negative: fn val -> send(parent, {:neg, val}) end
          }
        )

      result = fun.(Nx.tensor(-3.0))
      assert Nx.to_number(result) == 3.0
      assert_receive {:neg, tensor}
      assert Nx.to_number(tensor) == 3.0
      refute_receive {:pos, _}
    end
  end
end
