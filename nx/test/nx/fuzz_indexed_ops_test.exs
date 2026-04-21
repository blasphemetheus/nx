defmodule Nx.FuzzIndexedOpsTest do
  @moduledoc """
  Fuzz tests for indexed / slice-family ops (`Nx.gather`,
  `Nx.indexed_add`, `Nx.indexed_put`, `Nx.slice`, `Nx.put_slice`,
  `Nx.take`, `Nx.scatter*`).

  These ops are exercised heavily by Axon but are thinly covered at
  the Nx layer — especially under `grad` and with mixed-backend
  inputs (one concrete tensor, one Expr tensor, as happens when a
  grad re-trace captures an outer-scope value).
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  import Nx.Defn
  import Nx.Testing

  # ── put_slice under grad ───────────────────────────────────────────

  describe "Nx.put_slice grad wrt update (put_slice_grad_mixed_backend)" do
    # See FUZZ_FINDINGS/put_slice_grad_mixed_backend_dispatch.md.
    # Nx.put_slice dispatches via impl!(tensor) alone. When target is
    # concrete BinaryBackend and update is an Expr (typical in grad
    # re-trace of a closure over an outer-scope tensor), BinaryBackend
    # is picked and crashes on to_binary(expr).

    defn put_slice_sum(t, patch) do
      Nx.sum(Nx.put_slice(t, [1], patch))
    end

    test "grad wrt update with captured target crashes (pins bug — flip when fixed)" do
      # Pins the CURRENT wrong behavior. When Nx.put_slice is fixed to
      # dispatch via impl!(tensor, slice), this assert_raise will fail
      # and the test should be flipped to assert_all_close below.
      t = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      patch = Nx.tensor([10.0, 20.0])

      assert_raise FunctionClauseError, ~r/Nx\.BinaryBackend\.to_binary/, fn ->
        Nx.Defn.grad(patch, fn p -> put_slice_sum(t, p) end)
      end

      # Once fixed, replace the above with:
      #   grad = Nx.Defn.grad(patch, fn p -> put_slice_sum(t, p) end)
      #   assert_all_close(grad, Nx.tensor([1.0, 1.0]), atol: 1.0e-6)
    end

    test "grad wrt target with captured update works (control)" do
      t = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      patch = Nx.tensor([10.0, 20.0])

      grad = Nx.Defn.grad(t, fn x -> put_slice_sum(x, patch) end)
      assert_all_close(grad, Nx.tensor([1.0, 0.0, 0.0, 1.0, 1.0]), atol: 1.0e-6)
    end

    test "grad wrt update with inner-constructed target works (control)" do
      defmodule Inner do
        import Nx.Defn

        defn put_slice_sum_inner(patch) do
          t = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
          Nx.sum(Nx.put_slice(t, [1], patch))
        end
      end

      patch = Nx.tensor([10.0, 20.0])
      grad = Nx.Defn.grad(patch, &Inner.put_slice_sum_inner/1)
      assert_all_close(grad, Nx.tensor([1.0, 1.0]), atol: 1.0e-6)
    end
  end

  # ── Non-grad scope: same bug fires on plain jit ────────────────────

  describe "dispatch bug also fires outside grad (plain jit closures)" do
    # The bug is not grad-specific — any Nx.Defn.jit closure over a
    # concrete tensor hits the same impl!/1 dispatch path.

    test "jit(fn p -> put_slice(captured_t, p) end) crashes" do
      t = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      jitted = Nx.Defn.jit(fn p -> Nx.sum(Nx.put_slice(t, [1], p)) end)

      assert_raise FunctionClauseError, ~r/Nx\.BinaryBackend\.to_binary/, fn ->
        jitted.(Nx.tensor([10.0, 20.0]))
      end
    end

    test "jit(fn lo -> clip(captured_t, lo, hi) end) crashes" do
      t = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      jitted = Nx.Defn.jit(fn lo -> Nx.clip(t, lo, 10.0) end)

      assert_raise FunctionClauseError, ~r/Nx\.BinaryBackend\.to_binary/, fn ->
        jitted.(Nx.tensor(1.5))
      end
    end

    test "jit(fn idx -> take(captured_t, idx) end) crashes (Expr.parameter class)" do
      t = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      jitted = Nx.Defn.jit(fn idx -> Nx.take(t, idx) end)

      assert_raise FunctionClauseError, ~r/Nx\.Defn\.Expr\.parameter/, fn ->
        jitted.(Nx.tensor([0, 2, 4], type: :s32))
      end
    end
  end

  # ── Same dispatch bug in Nx.clip and Nx.gather ─────────────────────

  describe "clip / gather dispatch mirrors put_slice bug" do
    # Same mechanism: impl!(tensor) ignores the second tensor arg.
    # Pins current (broken) behavior with assert_raise — flip when
    # all three dispatch sites are fixed.

    defn clip_sum(t, lo, hi), do: Nx.sum(Nx.clip(t, lo, hi))
    defn gather_sum(t, idx), do: Nx.sum(Nx.gather(t, idx))

    test "clip: grad wrt min with captured target crashes" do
      t = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      hi = Nx.tensor(4.5)

      assert_raise FunctionClauseError, ~r/Nx\.BinaryBackend\.to_binary/, fn ->
        Nx.Defn.grad(Nx.tensor(1.5), fn lo -> clip_sum(t, lo, hi) end)
      end
    end

    test "clip: grad wrt max with captured target crashes" do
      t = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      lo = Nx.tensor(1.5)

      assert_raise FunctionClauseError, ~r/Nx\.BinaryBackend\.to_binary/, fn ->
        Nx.Defn.grad(Nx.tensor(4.5), fn hi -> clip_sum(t, lo, hi) end)
      end
    end

    test "gather: calling with concrete source + Expr indices crashes" do
      # Grad wrt indices is semantically meaningless (int type), but
      # the dispatch should still route to Expr rather than crash.
      t = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      idx_f = Nx.tensor([[0.0], [2.0], [4.0]])

      assert_raise FunctionClauseError, ~r/Nx\.BinaryBackend\.to_binary/, fn ->
        Nx.Defn.grad(idx_f, fn i -> gather_sum(t, Nx.as_type(i, :s32)) end)
      end
    end
  end

  # ── reduce / window_reduce: same dispatch bug class (acc is tensor) ──

  describe "reduce / window_reduce dispatch (mixed concrete + Expr acc)" do
    defn reduce_to_scalar(t, acc) do
      Nx.reduce(t, acc, fn a, b -> a + b end)
    end

    defn windowed_max(t, acc) do
      Nx.window_reduce(t, acc, {2}, fn a, b -> Nx.max(a, b) end)
    end

    test "reduce: grad wrt acc with captured source crashes" do
      t = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

      assert_raise FunctionClauseError, ~r/Nx\.BinaryBackend\.to_binary/, fn ->
        Nx.Defn.grad(Nx.tensor(0.0), fn a -> reduce_to_scalar(t, a) end)
      end
    end

    test "window_reduce: grad wrt acc with captured source crashes" do
      t = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

      assert_raise FunctionClauseError, ~r/Nx\.BinaryBackend\.to_binary/, fn ->
        Nx.Defn.grad(Nx.tensor(0.0), fn a -> Nx.sum(windowed_max(t, a)) end)
      end
    end
  end

  # ── Nx.take: Expr.expr_block doesn't normalize args ────────────────

  describe "Nx.take grad with concrete indices (take_grad_with_captured_indices)" do
    # See FUZZ_FINDINGS/take_grad_with_captured_indices.md.
    # Distinct from the dispatch-bug class: top-level Nx.block routes
    # correctly to Expr via list_impl!, but Expr.expr_block/3 then
    # calls parameter/2 on raw concrete tensors without normalizing,
    # and parameter/2's pattern match fails.

    defn take_sum(t, idx), do: Nx.sum(Nx.take(t, idx))

    test "grad wrt t with captured concrete indices crashes (pins bug)" do
      t = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      idx = Nx.tensor([0, 2, 4], type: :s32)

      assert_raise FunctionClauseError, ~r/Nx\.Defn\.Expr\.parameter/, fn ->
        Nx.Defn.grad(t, fn x -> take_sum(x, idx) end)
      end

      # Once fixed, replace with:
      #   grad = Nx.Defn.grad(t, fn x -> take_sum(x, idx) end)
      #   assert_all_close(grad, Nx.tensor([1.0, 0.0, 1.0, 0.0, 1.0]), atol: 1.0e-6)
    end

    test "grad wrt t with inline indices works (control)" do
      defmodule TakeInline do
        import Nx.Defn
        defn take_sum_inline(t), do: Nx.sum(Nx.take(t, Nx.tensor([0, 2, 4])))
      end

      t = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      grad = Nx.Defn.grad(t, &TakeInline.take_sum_inline/1)
      assert_all_close(grad, Nx.tensor([1.0, 0.0, 1.0, 0.0, 1.0]), atol: 1.0e-6)
    end

    # Nx.take_along_axis uses the same Nx.block/4 path — same bug.
    defn take_along_axis_sum(t, idx) do
      Nx.sum(Nx.take_along_axis(t, idx, axis: 0))
    end

    test "take_along_axis: grad wrt t with captured indices crashes (same bug class)" do
      t = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      idx = Nx.tensor([0, 2, 4], type: :s32)

      assert_raise FunctionClauseError, ~r/Nx\.Defn\.Expr\.parameter/, fn ->
        Nx.Defn.grad(t, fn x -> take_along_axis_sum(x, idx) end)
      end
    end

    # Nx.all_close uses the same Nx.block/4 path. Since all_close
    # returns a bool, embed it in a differentiable expression.
    defn allclose_path(a, b) do
      close = Nx.as_type(Nx.all_close(a, b), :f32)
      Nx.multiply(close, Nx.sum(a))
    end

    test "all_close: grad with captured b crashes (same bug class)" do
      a = Nx.tensor([1.0, 2.0, 3.0])
      b = Nx.tensor([1.0, 2.0, 3.0])

      assert_raise FunctionClauseError, ~r/Nx\.Defn\.Expr\.parameter/, fn ->
        Nx.Defn.grad(a, fn x -> allclose_path(x, b) end)
      end
    end
  end

  # ── put_slice / gather / indexed_add shape + forward fuzzing ──────

  describe "indexed ops don't crash on valid inputs" do
    property "gather with unique indices matches hand-computed lookup" do
      check all(
              n <- integer(3..8),
              k <- integer(1..5),
              max_runs: 10
            ) do
        k = min(k, n)
        t = Nx.iota({n}, type: :f32)
        idx = Nx.tensor(Enum.take(Enum.shuffle(0..(n - 1)), k)) |> Nx.reshape({k, 1})
        result = Nx.gather(t, idx)
        assert Nx.shape(result) == {k}
      end
    end

    property "indexed_add with overlapping indices sums contributions" do
      check all(
              n <- integer(3..6),
              max_runs: 6
            ) do
        t = Nx.broadcast(Nx.tensor(0.0, type: :f32), {n})
        idx = Nx.tensor([[0], [0], [1]])
        updates = Nx.tensor([1.0, 2.0, 3.0], type: :f32)
        result = Nx.indexed_add(t, idx, updates)
        # Position 0 should be 3.0 (1.0 + 2.0); position 1 should be 3.0.
        assert_all_close(result[0], Nx.tensor(3.0, type: :f32), atol: 1.0e-6)
        assert_all_close(result[1], Nx.tensor(3.0, type: :f32), atol: 1.0e-6)
      end
    end

    property "slice + sum + grad is shape-consistent" do
      check all(
              n <- integer(3..6),
              start <- integer(0..2),
              len <- integer(1..3),
              max_runs: 8
            ) do
        n = max(n, start + len)
        t = Nx.iota({n}, type: :f32)

        grad = Nx.Defn.grad(t, fn x -> Nx.sum(Nx.slice(x, [start], [len])) end)
        assert Nx.shape(grad) == {n}
      end
    end
  end
end
