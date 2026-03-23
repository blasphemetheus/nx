defmodule Nx.WhileTupleCaptureTest do
  @moduledoc """
  Root cause investigation: captured tensors in while return tuples
  silently become 0 instead of raising "different contexts" error.

  Investigation path:
  1. Found: runtime_call with tuple input {x, y} inside while returns
     wrong value (y becomes 0)
  2. Narrowed: runtime_call with SINGLE captured input correctly raises
     "different contexts" — only tuple wrapping bypasses the check
  3. Root cause: the defn compiler doesn't validate context for tensors
     placed directly in while return tuples. A captured y in {x, y, count+1}
     silently becomes 0 — no runtime_call needed to reproduce.
  """
  use ExUnit.Case, async: true
  import Nx.Defn

  # ── Root cause: captured tensor in while return tuple becomes 0 ────

  defn captured_in_while_tuple(x, y) do
    {_x, leaked_y, _count} =
      while {x, _placeholder = Nx.tensor(0.0), count = Nx.tensor(0)}, Nx.less(count, 1) do
        {x, y, count + 1}
      end

    leaked_y
  end

  test "captured tensor in while return tuple should be 42.0 or raise" do
    # Root cause: y is captured (not in while state), placed directly
    # in the while body return tuple. The defn compiler should either
    # raise "different contexts" (like it does for Nx.add(x, y)) or
    # correctly thread y through. Instead it silently becomes 0.
    result = Nx.to_number(captured_in_while_tuple(Nx.tensor(1.0), Nx.tensor(42.0)))
    assert result == 42.0
  end

  # ── Downstream symptom: runtime_call with tuple input ──────────────

  def sum_pair({a, b}, _opts), do: Nx.add(a, b)

  defn runtime_call_tuple_captured(x, y) do
    {result, _} =
      while {x, count = Nx.tensor(0)}, Nx.less(count, 1) do
        summed = Nx.runtime_call(x, {x, y}, &sum_pair/2)
        {summed, count + 1}
      end

    result
  end

  test "runtime_call with tuple input containing captured tensor" do
    # Downstream of root cause: {x, y} tuple input to runtime_call
    # passes y through the while body tuple, which zeroes it.
    # Callback receives {1.0, 0} instead of {1.0, 10.0}.
    result = Nx.to_number(runtime_call_tuple_captured(Nx.tensor(1.0), Nx.tensor(10.0)))
    assert result == 11.0
  end

  # ── Control: direct capture correctly raises ───────────────────────

  defn direct_capture_in_while(x, y) do
    {result, _} =
      while {x, count = Nx.tensor(0)}, Nx.less(count, 1) do
        {Nx.add(x, y), count + 1}
      end

    result
  end

  test "direct captured tensor in while correctly raises" do
    # This is the CORRECT behavior — using y directly in Nx.add
    # raises "different contexts". The bug is that placing y in
    # a tuple bypasses this check.
    assert_raise RuntimeError, ~r/different contexts/, fn ->
      direct_capture_in_while(Nx.tensor(1.0), Nx.tensor(10.0))
    end
  end

  # ── Control: both tensors in while state works ─────────────────────

  defn both_in_state(x, y) do
    {_x, result, _count} =
      while {x, y, count = Nx.tensor(0)}, Nx.less(count, 1) do
        {x, y, count + 1}
      end

    result
  end

  test "tensor in while state is preserved correctly" do
    result = Nx.to_number(both_in_state(Nx.tensor(1.0), Nx.tensor(42.0)))
    assert result == 42.0
  end

  # ── Scoping: where else does capture work correctly? ─────────────

  defn capture_in_condition(x, limit) do
    {result, _} =
      while {x, count = Nx.tensor(0)}, Nx.less(count, limit) do
        {Nx.add(x, 1), count + 1}
      end

    result
  end

  test "captured tensor in while condition correctly raises" do
    assert_raise RuntimeError, ~r/different contexts/, fn ->
      capture_in_condition(Nx.tensor(0.0), Nx.tensor(3))
    end
  end

  defn cond_capture(x, y) do
    if Nx.greater(x, 0) do
      {x, y}
    else
      {Nx.negate(x), y}
    end
  end

  test "captured tensor in cond branches works correctly" do
    # Cond does NOT have this bug — captured tensors are preserved
    {_a, b} = cond_capture(Nx.tensor(5.0), Nx.tensor(42.0))
    assert Nx.to_number(b) == 42.0
  end
end
