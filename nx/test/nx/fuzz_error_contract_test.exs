defmodule Nx.FuzzErrorContractTest do
  @moduledoc """
  Error-contract fuzz (FUZZ_ROADMAP T2.1).

  v1.0 freezes error behavior into API contract: invalid input must raise an
  Nx-owned `ArgumentError` with a real message — never a leaked internal
  error (`ArithmeticError`, `MatchError`, `FunctionClauseError`,
  `UndefinedFunctionError`, `CaseClauseError`). This suite enumerates one
  invalid construction per API area and asserts the contract, with
  randomized variants for the dimension-mismatch classes.

  Non-raising cases that look surprising but are documented semantics
  (excluded here on purpose): `Nx.slice/4` clamps out-of-range start
  indices (XLA dynamic-slice semantics); `Nx.split/3` allows uneven splits.
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  # {label, fun} — every entry must raise ArgumentError
  defp contract_cases do
    [
      {"reshape wrong size", fn -> Nx.reshape(Nx.iota({6}), {4}) end},
      {"broadcast incompatible", fn -> Nx.broadcast(Nx.iota({3}), {2, 4}) end},
      {"add incompatible shapes", fn -> Nx.add(Nx.iota({3}), Nx.iota({4})) end},
      {"dot incompatible", fn -> Nx.dot(Nx.iota({2, 3}), Nx.iota({4, 2})) end},
      {"concatenate rank mismatch", fn -> Nx.concatenate([Nx.iota({2}), Nx.iota({2, 2})]) end},
      {"slice negative length", fn -> Nx.slice(Nx.iota({3}), [0], [-1]) end},
      {"transpose bad axes", fn -> Nx.transpose(Nx.iota({2, 3}), axes: [0, 5]) end},
      {"squeeze non-1 axis", fn -> Nx.squeeze(Nx.iota({2, 3}), axes: [0]) end},
      {"sum bad axis", fn -> Nx.sum(Nx.iota({2, 3}), axes: [7]) end},
      {"take bad axis", fn -> Nx.take(Nx.iota({2, 3}), Nx.tensor([0]), axis: 9) end},
      {"take float indices", fn -> Nx.take(Nx.iota({3}), Nx.tensor([0.5])) end},
      {"as_type bad type", fn -> Nx.as_type(Nx.iota({2}), :f100) end},
      {"pad wrong config length", fn -> Nx.pad(Nx.iota({2, 2}), 0, [{0, 0, 0}]) end},
      {"from_binary wrong byte count", fn -> Nx.from_binary(<<1, 2, 3>>, {:f, 32}) end},
      {"cholesky non-square", fn -> Nx.LinAlg.cholesky(Nx.iota({2, 3})) end},
      {"solve dim mismatch",
       fn -> Nx.LinAlg.solve(Nx.iota({2, 2}, type: :f32), Nx.iota({3}, type: :f32)) end},
      {"window rank mismatch", fn -> Nx.window_sum(Nx.iota({2, 2}), {2}) end},
      {"iota bad axis", fn -> Nx.iota({2, 2}, axis: 5) end},
      {"tensor ragged list", fn -> Nx.tensor([[1, 2], [3]]) end},
      {"eye negative dim", fn -> Nx.eye({-1, 2}) end},
      {"tri negative", fn -> Nx.tri(-2, -2) end},
      {"stack empty list", fn -> Nx.stack([]) end},
      {"new_axis out of range", fn -> Nx.new_axis(Nx.iota({2}), 5) end},
      {"indexed_put shape mismatch",
       fn -> Nx.indexed_put(Nx.iota({3}), Nx.tensor([[0]]), Nx.tensor([1, 2])) end}
    ]
  end

  test "invalid inputs raise Nx-owned ArgumentError with a message" do
    for {label, fun} <- contract_cases() do
      error =
        try do
          fun.()
          flunk("#{label}: expected a raise, got a value")
        rescue
          e -> e
        end

      assert %ArgumentError{} = error,
             "#{label}: raised #{inspect(error.__struct__)} instead of ArgumentError"

      message = Exception.message(error)
      assert is_binary(message) and byte_size(message) > 10, "#{label}: unhelpful message"
    end
  end

  describe "randomized dimension mismatches" do
    property "reshape to any wrong total size raises ArgumentError" do
      check all(n <- integer(2..30), m <- integer(2..30), n != m, max_runs: 30) do
        assert_raise ArgumentError, fn -> Nx.reshape(Nx.iota({n}), {m}) end
      end
    end

    property "binary ops on any incompatible 1-D shapes raise ArgumentError" do
      check all(n <- integer(2..20), delta <- integer(1..10), max_runs: 30) do
        a = Nx.iota({n})
        b = Nx.iota({n + delta})

        for op <- [:add, :multiply, :max, :equal] do
          assert_raise ArgumentError, fn -> apply(Nx, op, [a, b]) end
        end
      end
    end

    property "reductions along any out-of-range axis raise ArgumentError" do
      check all(rank <- integer(1..3), extra <- integer(0..5), max_runs: 30) do
        shape = List.to_tuple(List.duplicate(2, rank))
        t = Nx.iota(shape)
        bad_axis = rank + extra

        for op <- [:sum, :mean, :product] do
          assert_raise ArgumentError, fn -> apply(Nx, op, [t, [axes: [bad_axis]]]) end
        end
      end
    end
  end

  describe "known violations ([BUG-ERROR-CONTRACT])" do
    # See FUZZ_FINDINGS/reshape_multiple_auto_error_message.md. Flip to
    # assert_raise ArgumentError when fixed.
    test "reshape with two :auto leaks a bare ArithmeticError" do
      assert_raise ArithmeticError, fn ->
        Nx.reshape(Nx.iota({12}), {:auto, :auto})
      end
    end
  end
end
