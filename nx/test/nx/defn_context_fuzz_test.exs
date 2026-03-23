defmodule Nx.DefnContextFuzzTest do
  @moduledoc """
  Fuzz testing for defn context validation gaps.

  The while-tuple-capture bug showed that captured tensors in while
  return tuples silently become 0. This test suite probes for similar
  context validation gaps in other defn constructs.

  Tests assert CORRECT behavior. Failures indicate bugs.
  """
  use ExUnit.Case, async: true
  import Nx.Defn

  # ── Nested tuples in while return ──────────────────────────────────

  # Nested tuples change the while return structure — the compiler
  # catches this as a shape mismatch. Not the same bug.

  # ── Captured tensor used in computation inside while return ────────

  defn capture_computed_in_return(x, y) do
    {result, _} =
      while {x, count = Nx.tensor(0)}, Nx.less(count, 1) do
        # y is captured but used in Nx.multiply first
        # Does the Nx op catch it, or does the result still go in the tuple?
        {Nx.multiply(x, y), count + 1}
      end

    result
  end

  test "captured tensor used in computation in while return" do
    try do
      result = Nx.to_number(capture_computed_in_return(Nx.tensor(2.0), Nx.tensor(5.0)))
      # If it doesn't raise, should be 10.0
      assert result == 10.0
    rescue
      RuntimeError -> :ok
    end
  end

  # ── Multiple captured tensors in while return ──────────────────────

  defn multi_capture(x, y, z) do
    {_, leaked_y, leaked_z, _} =
      while {x, _p1 = Nx.tensor(0.0), _p2 = Nx.tensor(0.0), count = Nx.tensor(0)},
            Nx.less(count, 1) do
        {x, y, z, count + 1}
      end

    Nx.add(leaked_y, leaked_z)
  end

  test "multiple captured tensors in while return" do
    result = Nx.to_number(multi_capture(Nx.tensor(1.0), Nx.tensor(10.0), Nx.tensor(20.0)))
    assert result == 30.0
  end

  # ── Captured tensor survives multiple while iterations ─────────────

  defn capture_multi_iter(x, y) do
    {_, leaked_y, _} =
      while {x, _p = Nx.tensor(0.0), count = Nx.tensor(0)}, Nx.less(count, 3) do
        {Nx.add(x, 1), y, count + 1}
      end

    leaked_y
  end

  test "captured tensor across multiple while iterations" do
    result = Nx.to_number(capture_multi_iter(Nx.tensor(0.0), Nx.tensor(42.0)))
    assert result == 42.0
  end

  # ── Cond inside while with captured tensor ─────────────────────────

  defn cond_inside_while_capture(x, y) do
    {result, _} =
      while {x, count = Nx.tensor(0)}, Nx.less(count, 1) do
        val =
          if Nx.greater(x, 0) do
            y
          else
            Nx.tensor(0.0)
          end

        {val, count + 1}
      end

    result
  end

  test "captured tensor in cond branch inside while" do
    try do
      result = Nx.to_number(cond_inside_while_capture(Nx.tensor(5.0), Nx.tensor(42.0)))
      assert result == 42.0
    rescue
      RuntimeError -> :ok
    end
  end

  # ── While inside cond with captured tensor ─────────────────────────

  defn while_inside_cond_capture(x, y) do
    if Nx.greater(x, 0) do
      {result, _} =
        while {x, count = Nx.tensor(0)}, Nx.less(count, 1) do
          {Nx.add(x, 1), count + 1}
        end

      Nx.add(result, y)
    else
      x
    end
  end

  test "captured tensor used after while inside cond" do
    result = Nx.to_number(while_inside_cond_capture(Nx.tensor(1.0), Nx.tensor(10.0)))
    assert result == 12.0
  end

  # ── Two sequential whiles sharing same captured tensor ─────────────

  defn two_whiles_same_capture(x, _y) do
    {r1, _} =
      while {x, count = Nx.tensor(0)}, Nx.less(count, 1) do
        {Nx.add(x, 1), count + 1}
      end

    {r2, _} =
      while {r1, count = Nx.tensor(0)}, Nx.less(count, 1) do
        {Nx.add(r1, 1), count + 1}
      end

    r2
  end

  test "two sequential whiles" do
    result = Nx.to_number(two_whiles_same_capture(Nx.tensor(0.0), Nx.tensor(10.0)))
    assert result == 2.0
  end

  # ── Tensor created inside while used in return tuple ───────────────

  defn new_tensor_in_while_tuple(x) do
    {_x, created, _} =
      while {x, _p = Nx.tensor(0.0), count = Nx.tensor(0)}, Nx.less(count, 1) do
        # Tensor created inside while body — valid context
        {x, Nx.tensor(99.0), count + 1}
      end

    created
  end

  test "tensor created inside while body in return tuple" do
    # This should work — tensor is created in the while context
    result = Nx.to_number(new_tensor_in_while_tuple(Nx.tensor(1.0)))
    assert result == 99.0
  end

  # ── hook inside while with captured tensor ─────────────────────────

  defn hook_capture_in_while(x, _y) do
    {result, _} =
      while {x, count = Nx.tensor(0)}, Nx.less(count, 1) do
        hooked = hook(x, :my_hook)
        {Nx.add(hooked, 1), count + 1}
      end

    result
  end

  test "hook inside while works" do
    result = Nx.to_number(hook_capture_in_while(Nx.tensor(1.0), Nx.tensor(10.0)))
    assert result == 2.0
  end

  # ── Returned tensor from while used in further computation ─────────

  defn while_result_in_computation(x) do
    {result, _} =
      while {x, count = Nx.tensor(0)}, Nx.less(count, 3) do
        {Nx.add(x, 1), count + 1}
      end

    # Use while result in further computation
    Nx.multiply(result, 2)
  end

  test "while result used in further computation" do
    result = Nx.to_number(while_result_in_computation(Nx.tensor(0.0)))
    assert result == 6.0
  end
end
