defmodule EXLA.FuzzEdgeCasesTest do
  @moduledoc """
  Tier 4: Cross-backend edge case tests.
  Runs boundary-condition operations on EXLA and compares with BinaryBackend.
  """
  use ExUnit.Case, async: false

  import Nx.Testing

  defp compare(fun, inputs, opts \\ []) do
    atol = opts[:atol] || 1.0e-4
    rtol = opts[:rtol] || 1.0e-4

    # Ensure inputs are on BinaryBackend for reference computation
    binary_inputs = Enum.map(inputs, &Nx.backend_transfer(&1, Nx.BinaryBackend))

    binary_result =
      Nx.with_default_backend(Nx.BinaryBackend, fn ->
        apply(fun, binary_inputs)
      end)

    # Wrap in a defn-compatible way for EXLA JIT
    exla_fun =
      Nx.Defn.jit(
        fn inputs_tuple ->
          inputs_list = Tuple.to_list(inputs_tuple)
          apply(fun, inputs_list)
        end,
        compiler: EXLA,
        client: :host
      )

    exla_result =
      exla_fun.(List.to_tuple(binary_inputs))
      |> Nx.backend_transfer(Nx.BinaryBackend)

    assert_all_close(binary_result, exla_result, atol: atol, rtol: rtol)
  end

  # ── Slice boundary cases on EXLA ───────────────────────────────────

  describe "slice edge cases: EXLA vs BinaryBackend" do
    test "slice full tensor" do
      t = Nx.iota({4, 3}, type: :f32)
      compare(&Nx.slice(&1, [0, 0], [4, 3]), [t])
    end

    test "slice last element" do
      t = Nx.iota({5}, type: :f32)
      compare(&Nx.slice(&1, [4], [1]), [t])
    end

    test "slice with strides" do
      t = Nx.iota({12}, type: :f32)
      compare(&Nx.slice(&1, [0], [12], strides: [3]), [t])
    end

    test "slice clamped start" do
      t = Nx.iota({5}, type: :f32)
      compare(&Nx.slice(&1, [10], [2]), [t])
    end
  end

  # ── Put_slice on EXLA ──────────────────────────────────────────────

  describe "put_slice edge cases: EXLA vs BinaryBackend" do
    test "put_slice at beginning" do
      t = Nx.broadcast(Nx.tensor(0.0), {5})
      s = Nx.tensor([1.0, 2.0])
      compare(fn t, s -> Nx.put_slice(t, [0], s) end, [t, s])
    end

    test "put_slice at end" do
      t = Nx.broadcast(Nx.tensor(0.0), {5})
      s = Nx.tensor([8.0, 9.0])
      compare(fn t, s -> Nx.put_slice(t, [3], s) end, [t, s])
    end
  end

  # ── Gather/take on EXLA ────────────────────────────────────────────

  describe "gather/take edge cases: EXLA vs BinaryBackend" do
    test "gather single element" do
      t = Nx.iota({3, 4}, type: :f32)
      idx = Nx.tensor([[2, 3]])
      compare(&Nx.gather(&1, &2), [t, idx])
    end

    test "gather partial (rows)" do
      t = Nx.iota({4, 5}, type: :f32)
      idx = Nx.tensor([[0], [3]])
      compare(&Nx.gather(&1, &2), [t, idx])
    end

    test "take with negative axis" do
      t = Nx.iota({3, 4}, type: :f32)
      idx = Nx.tensor([1, 0])
      compare(&Nx.take(&1, &2, axis: -1), [t, idx])
    end

    test "take with duplicate indices" do
      t = Nx.tensor([10.0, 20.0, 30.0])
      idx = Nx.tensor([0, 0, 0, 0])
      compare(&Nx.take(&1, &2), [t, idx])
    end

    test "take_along_axis" do
      t = Nx.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
      idx = Nx.tensor([[2, 0], [1, 2]])
      compare(&Nx.take_along_axis(&1, &2, axis: 1), [t, idx])
    end
  end

  # ── Indexed ops on EXLA ────────────────────────────────────────────

  describe "indexed ops edge cases: EXLA vs BinaryBackend" do
    test "indexed_add overlapping" do
      t = Nx.tensor([0.0, 0.0, 0.0])
      idx = Nx.tensor([[1], [1], [1]])
      updates = Nx.tensor([1.0, 2.0, 3.0])
      compare(fn t, i, u -> Nx.indexed_add(t, i, u) end, [t, idx, updates])
    end

    test "indexed_put single" do
      t = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      idx = Nx.tensor([[2]])
      updates = Nx.tensor([99.0])
      compare(fn t, i, u -> Nx.indexed_put(t, i, u) end, [t, idx, updates])
    end
  end

  # ── Pad on EXLA ────────────────────────────────────────────────────

  describe "pad edge cases: EXLA vs BinaryBackend" do
    test "pad with negative edge" do
      t = Nx.iota({5}, type: :f32)
      compare(fn t -> Nx.pad(t, Nx.tensor(0.0), [{-1, -1, 0}]) end, [t])
    end

    test "pad with interior" do
      t = Nx.tensor([1.0, 2.0, 3.0])
      compare(fn t -> Nx.pad(t, Nx.tensor(0.0), [{0, 0, 1}]) end, [t])
    end

    test "pad with edge + interior" do
      t = Nx.tensor([1.0, 2.0])
      compare(fn t -> Nx.pad(t, Nx.tensor(0.0), [{1, 1, 2}]) end, [t])
    end
  end

  # ── Window ops on EXLA ─────────────────────────────────────────────

  describe "window ops edge cases: EXLA vs BinaryBackend" do
    test "window_sum full tensor window" do
      t = Nx.iota({4, 3}, type: :f32)
      compare(fn t -> Nx.window_sum(t, {4, 3}) end, [t])
    end

    test "window_max with strides" do
      t = Nx.tensor([1.0, 5.0, 2.0, 8.0, 3.0, 7.0])
      compare(fn t -> Nx.window_max(t, {2}, strides: [2]) end, [t])
    end

    test "window_min with same padding" do
      t = Nx.iota({4, 4}, type: :f32)
      compare(fn t -> Nx.window_min(t, {3, 3}, padding: :same) end, [t])
    end

    test "window_mean 2D" do
      t = Nx.iota({4, 4}, type: :f32)
      compare(fn t -> Nx.window_mean(t, {2, 2}) end, [t])
    end
  end

  # ── Sort/reverse/diff on EXLA ──────────────────────────────────────

  describe "sort/reverse/diff edge cases: EXLA vs BinaryBackend" do
    test "sort 1D" do
      t = Nx.tensor([3.0, 1.0, 4.0, 1.0, 5.0])
      compare(&Nx.sort/1, [t])
    end

    test "sort descending" do
      t = Nx.tensor([3.0, 1.0, 4.0, 1.0, 5.0])
      compare(&Nx.sort(&1, direction: :desc), [t])
    end

    test "argsort" do
      t = Nx.tensor([30.0, 10.0, 20.0])
      compare(&Nx.argsort/1, [t])
    end

    test "reverse 2D all axes" do
      t = Nx.iota({3, 4}, type: :f32)
      compare(&Nx.reverse(&1, axes: [0, 1]), [t])
    end

    test "diff 1D" do
      t = Nx.tensor([1.0, 4.0, 2.0, 8.0, 5.0])
      compare(&Nx.diff/1, [t])
    end

    test "diff order 2" do
      t = Nx.tensor([1.0, 4.0, 2.0, 8.0, 5.0])
      compare(&Nx.diff(&1, order: 2), [t])
    end
  end

  # ── Clip / select on EXLA ──────────────────────────────────────────

  describe "clip/select edge cases: EXLA vs BinaryBackend" do
    test "clip basic" do
      t = Nx.tensor([1.0, 5.0, 3.0, 8.0, -2.0])
      lo = Nx.tensor(2.0)
      hi = Nx.tensor(6.0)
      compare(fn t, lo, hi -> Nx.clip(t, lo, hi) end, [t, lo, hi])
    end

    test "clip min == max" do
      t = Nx.tensor([1.0, 5.0, 3.0])
      v = Nx.tensor(3.0)
      compare(fn t, v, v2 -> Nx.clip(t, v, v2) end, [t, v, v])
    end

    test "select element-wise" do
      pred = Nx.tensor([1, 0, 1, 0])
      a = Nx.tensor([10.0, 20.0, 30.0, 40.0])
      b = Nx.tensor([50.0, 60.0, 70.0, 80.0])
      compare(fn p, a, b -> Nx.select(p, a, b) end, [pred, a, b])
    end
  end

  # ── Reshape / squeeze / flatten on EXLA ────────────────────────────

  describe "shape ops edge cases: EXLA vs BinaryBackend" do
    test "reshape with :auto" do
      t = Nx.iota({12}, type: :f32)
      compare(fn t -> Nx.reshape(t, {3, :auto}) end, [t])
    end

    test "squeeze all 1-dims" do
      t = Nx.iota({1, 3, 1, 4, 1}, type: :f32)
      compare(fn t -> Nx.squeeze(t, axes: [0, 2, 4]) end, [t])
    end

    test "flatten" do
      t = Nx.iota({2, 3, 4}, type: :f32)
      compare(&Nx.flatten/1, [t])
    end

    test "tile" do
      t = Nx.tensor([1.0, 2.0, 3.0])
      compare(fn t -> Nx.tile(t, [3]) end, [t])
    end

    test "concatenate" do
      a = Nx.tensor([1.0, 2.0])
      b = Nx.tensor([3.0, 4.0, 5.0])
      compare(fn a, b -> Nx.concatenate([a, b]) end, [a, b])
    end

    test "stack" do
      a = Nx.tensor([1.0, 2.0])
      b = Nx.tensor([3.0, 4.0])
      compare(fn a, b -> Nx.stack([a, b]) end, [a, b])
    end
  end

  # ── Diagonal on EXLA ───────────────────────────────────────────────

  describe "diagonal ops: EXLA vs BinaryBackend" do
    test "take_diagonal" do
      t = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
      compare(&Nx.take_diagonal/1, [t])
    end

    test "take_diagonal with offset" do
      t = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
      compare(&Nx.take_diagonal(&1, offset: 1), [t])
    end

    test "make_diagonal" do
      v = Nx.tensor([1.0, 2.0, 3.0])
      compare(&Nx.make_diagonal/1, [v])
    end
  end

  # ── LinAlg on EXLA ─────────────────────────────────────────────────

  describe "linalg edge cases: EXLA vs BinaryBackend" do
    test "determinant of identity" do
      t = Nx.eye(3, type: :f32)
      compare(&Nx.LinAlg.determinant/1, [t])
    end

    test "norm 1D" do
      t = Nx.tensor([3.0, 4.0])
      compare(&Nx.LinAlg.norm/1, [t])
    end

    test "invert of simple matrix" do
      t = Nx.tensor([[2.0, 1.0], [1.0, 3.0]])
      compare(&Nx.LinAlg.invert/1, [t], atol: 1.0e-3, rtol: 1.0e-3)
    end

    test "solve identity" do
      a = Nx.eye(3, type: :f32)
      b = Nx.tensor([1.0, 2.0, 3.0])
      compare(&Nx.LinAlg.solve(&1, &2), [a, b])
    end
  end

  # ── Type conversion on EXLA ────────────────────────────────────────

  describe "type conversion edge cases: EXLA vs BinaryBackend" do
    test "as_type f32 to s32" do
      t = Nx.tensor([1.5, 2.7, 3.1], type: :f32)
      compare(fn t -> Nx.as_type(t, :s32) end, [t])
    end

    test "as_type s32 to f32" do
      t = Nx.tensor([1, 2, 3], type: :s32)
      compare(fn t -> Nx.as_type(t, :f32) end, [t])
    end

    test "bitcast f32 to u32 roundtrip" do
      t = Nx.tensor([1.0, 2.0, 3.0], type: :f32)
      compare(fn t -> t |> Nx.bitcast(:u32) |> Nx.bitcast(:f32) end, [t])
    end
  end

  # ── FFT on EXLA ────────────────────────────────────────────────────

  describe "FFT edge cases: EXLA vs BinaryBackend" do
    test "fft 1D" do
      t = Nx.tensor([1.0, 0.0, 0.0, 0.0])
      compare(&Nx.fft/1, [t], atol: 1.0e-4, rtol: 1.0e-4)
    end

    test "fft then ifft roundtrip" do
      t = Nx.tensor([1.0, 2.0, 3.0, 4.0])
      compare(fn t -> t |> Nx.fft() |> Nx.ifft() end, [t], atol: 1.0e-4, rtol: 1.0e-4)
    end
  end

  # ── Covariance on EXLA ─────────────────────────────────────────────

  describe "covariance: EXLA vs BinaryBackend" do
    test "covariance basic" do
      t = Nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
      compare(&Nx.covariance/1, [t], atol: 1.0e-3, rtol: 1.0e-3)
    end
  end
end
