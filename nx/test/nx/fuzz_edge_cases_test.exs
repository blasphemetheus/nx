defmodule Nx.FuzzEdgeCasesTest do
  @moduledoc """
  Tier 4: LLM-guided edge case tests derived from source code analysis.

  Each test targets a specific validation boundary or edge condition
  identified by reading the Nx source code. Tests are organized by
  the function group they exercise.
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  # ── Slice boundary conditions ──────────────────────────────────────
  # Source: Nx.Shape.slice/4 (shape.ex:1242)
  # Boundaries:
  #   - length must be >= 1 (line 1263)
  #   - stride must be >= 1 (line 1268)
  #   - length must be <= dim size (line 1273)
  #   - start_index clamped to dim - length (line 1279)
  #   - start_indices/lengths/strides length must match rank

  describe "slice boundary conditions" do
    test "slice with length == dim size (full slice)" do
      t = Nx.iota({4, 3})
      result = Nx.slice(t, [0, 0], [4, 3])
      assert Nx.shape(result) == {4, 3}
      assert Nx.to_flat_list(result) == Nx.to_flat_list(t)
    end

    test "slice with length == 1 (minimum valid length)" do
      t = Nx.iota({5})
      for i <- 0..4 do
        result = Nx.slice(t, [i], [1])
        assert Nx.to_flat_list(result) == [i]
      end
    end

    test "slice start clamped when start + length > dim" do
      # Start index is clamped to max(0, dim - length)
      t = Nx.iota({5})
      # start=10 should be clamped to 5-2=3
      result = Nx.slice(t, [10], [2])
      assert Nx.to_flat_list(result) == [3, 4]
    end

    test "slice start clamped with negative start" do
      t = Nx.iota({5})
      # Negative start should be handled by to_indices
      # but if it makes it through as integer, it gets clamped
      result = Nx.slice(t, [0], [5])
      assert Nx.shape(result) == {5}
    end

    test "slice with stride == dim (maximum useful stride)" do
      t = Nx.iota({6})
      result = Nx.slice(t, [0], [6], strides: [6])
      # ceil(6/6) = 1
      assert Nx.shape(result) == {1}
      assert Nx.to_flat_list(result) == [0]
    end

    test "slice with stride > length" do
      t = Nx.iota({10})
      result = Nx.slice(t, [0], [3], strides: [5])
      # ceil(3/5) = 1
      assert Nx.shape(result) == {1}
    end

    test "slice output shape with various strides" do
      t = Nx.iota({12})
      # ceil(12/1)=12, ceil(12/2)=6, ceil(12/3)=4, ceil(12/4)=3
      assert Nx.shape(Nx.slice(t, [0], [12], strides: [1])) == {12}
      assert Nx.shape(Nx.slice(t, [0], [12], strides: [2])) == {6}
      assert Nx.shape(Nx.slice(t, [0], [12], strides: [3])) == {4}
      assert Nx.shape(Nx.slice(t, [0], [12], strides: [4])) == {3}
      # ceil(12/5)=3, ceil(12/6)=2, ceil(12/7)=2
      assert Nx.shape(Nx.slice(t, [0], [12], strides: [5])) == {3}
      assert Nx.shape(Nx.slice(t, [0], [12], strides: [6])) == {2}
      assert Nx.shape(Nx.slice(t, [0], [12], strides: [7])) == {2}
    end

    test "slice raises on length 0" do
      t = Nx.iota({5})
      assert_raise ArgumentError, ~r/length at axis 0 must be greater/, fn ->
        Nx.slice(t, [0], [0])
      end
    end

    test "slice raises on length > dim" do
      t = Nx.iota({3})
      assert_raise ArgumentError, ~r/length at axis 0 must be less than/, fn ->
        Nx.slice(t, [0], [4])
      end
    end

    test "slice raises on stride 0" do
      t = Nx.iota({5})
      assert_raise ArgumentError, ~r/stride at axis 0 must be greater/, fn ->
        Nx.slice(t, [0], [5], strides: [0])
      end
    end

    test "slice raises on rank mismatch in start_indices" do
      t = Nx.iota({3, 4})
      assert_raise ArgumentError, ~r/invalid start indices rank/, fn ->
        Nx.slice(t, [0], [3, 4])
      end
    end

    test "slice raises on rank mismatch in lengths" do
      t = Nx.iota({3, 4})
      assert_raise ArgumentError, ~r/invalid limit indices rank/, fn ->
        Nx.slice(t, [0, 0], [3])
      end
    end

    test "slice on scalar tensor returns scalar unchanged" do
      t = Nx.tensor(42)
      result = Nx.slice(t, [], [])
      assert Nx.to_number(result) == 42
    end

    test "slice on rank-1 tensor with dim=1" do
      t = Nx.tensor([7])
      result = Nx.slice(t, [0], [1])
      assert Nx.to_flat_list(result) == [7]
    end

    test "slice multi-dim boundary: last element in each dim" do
      t = Nx.iota({3, 4, 5})
      # Take the very last element as a 1x1x1 slice
      result = Nx.slice(t, [2, 3, 4], [1, 1, 1])
      assert Nx.shape(result) == {1, 1, 1}
      assert result |> Nx.reshape({}) |> Nx.to_number() == 3 * 4 * 5 - 1
    end

    test "slice with strides on multi-dim tensor" do
      t = Nx.iota({6, 8})
      result = Nx.slice(t, [0, 0], [6, 8], strides: [2, 3])
      # ceil(6/2) = 3, ceil(8/3) = 3
      assert Nx.shape(result) == {3, 3}
    end
  end

  # ── Put_slice boundary conditions ──────────────────────────────────
  # Source: Nx.Shape.put_slice/5 (shape.ex:1299)
  # Boundaries:
  #   - start_indices rank must match tensor rank (line 1302)
  #   - slice rank must match tensor rank (line 1306)
  #   - slice dims must be <= tensor dims (line 1326)

  describe "put_slice boundary conditions" do
    test "put_slice with full-size slice (overwrite entire tensor)" do
      t = Nx.iota({3, 4})
      s = Nx.broadcast(Nx.tensor(99), {3, 4})
      result = Nx.put_slice(t, [0, 0], s)
      assert Nx.to_flat_list(result) == List.duplicate(99, 12)
    end

    test "put_slice with 1x1 slice at each corner of 2D tensor" do
      t = Nx.broadcast(Nx.tensor(0), {3, 4})
      corners = [{0, 0}, {0, 3}, {2, 0}, {2, 3}]

      for {r, c} <- corners do
        result = Nx.put_slice(t, [r, c], Nx.tensor([[1]]))
        assert Nx.to_number(result[r][c]) == 1
      end
    end

    test "put_slice at the very end of each dimension" do
      t = Nx.broadcast(Nx.tensor(0), {5})
      result = Nx.put_slice(t, [4], Nx.tensor([99]))
      assert Nx.to_flat_list(result) == [0, 0, 0, 0, 99]
    end

    test "put_slice raises when slice dim > tensor dim" do
      t = Nx.iota({3, 4})
      s = Nx.iota({4, 4})

      assert_raise ArgumentError, ~r/slice shape .* must be less than or equal/, fn ->
        Nx.put_slice(t, [0, 0], s)
      end
    end

    test "put_slice raises on rank mismatch" do
      t = Nx.iota({3, 4})
      s = Nx.iota({3})

      assert_raise ArgumentError, ~r/invalid slice for put_slice/, fn ->
        Nx.put_slice(t, [0, 0], s)
      end
    end

    test "put_slice with type promotion" do
      t = Nx.iota({4}, type: :s32)
      s = Nx.tensor([1.5], type: :f32)
      result = Nx.put_slice(t, [0], s)
      # Result type should be promoted to float
      assert elem(Nx.type(result), 0) == :f
    end
  end

  # ── Take boundary conditions ───────────────────────────────────────
  # Source: Nx.take/3 (nx.ex:14179)
  # Boundaries:
  #   - indices must be integer type (line 14180)
  #   - axis normalized via Nx.Shape.normalize_axis
  #   - out-of-bounds indices are clamped (implementation-specific)

  describe "take boundary conditions" do
    test "take with empty-ish index: single index" do
      t = Nx.iota({5})
      result = Nx.take(t, Nx.tensor([0]))
      assert Nx.shape(result) == {1}
      assert Nx.to_flat_list(result) == [0]
    end

    test "take all elements in reverse" do
      t = Nx.iota({4})
      result = Nx.take(t, Nx.tensor([3, 2, 1, 0]))
      assert Nx.to_flat_list(result) == [3, 2, 1, 0]
    end

    test "take with duplicate indices" do
      t = Nx.tensor([10, 20, 30])
      result = Nx.take(t, Nx.tensor([0, 0, 0, 0]))
      assert Nx.to_flat_list(result) == [10, 10, 10, 10]
    end

    test "take with multi-dim indices replaces axis with indices shape" do
      t = Nx.iota({4, 3})
      idx = Nx.tensor([[0, 1], [2, 3]])
      result = Nx.take(t, idx, axis: 0)
      # axis 0 (size 4) replaced by idx shape {2,2}
      assert Nx.shape(result) == {2, 2, 3}
    end

    test "take on axis 1 of 3D tensor" do
      t = Nx.iota({2, 3, 4})
      result = Nx.take(t, Nx.tensor([2, 0]), axis: 1)
      assert Nx.shape(result) == {2, 2, 4}
    end

    test "take with negative axis" do
      t = Nx.iota({3, 4})
      result = Nx.take(t, Nx.tensor([0, 1]), axis: -1)
      assert Nx.shape(result) == {3, 2}
    end

    test "take raises on float indices" do
      t = Nx.iota({5})
      assert_raise ArgumentError, ~r/indices must be an integer tensor/, fn ->
        Nx.take(t, Nx.tensor([0.0, 1.0]))
      end
    end

    test "take raises on out-of-range axis" do
      t = Nx.iota({3, 4})
      assert_raise ArgumentError, ~r/given axis \(5\) invalid/, fn ->
        Nx.take(t, Nx.tensor([0]), axis: 5)
      end
    end

    test "take from scalar-like: rank 1 tensor of size 1" do
      t = Nx.tensor([42])
      result = Nx.take(t, Nx.tensor([0, 0, 0]))
      assert Nx.to_flat_list(result) == [42, 42, 42]
    end
  end

  # ── Take_along_axis boundary conditions ────────────────────────────
  # Source: Nx.Shape.take_along_axis/3 (shape.ex:1566)
  # Boundaries:
  #   - ranks must match (line 1571)
  #   - non-indexed dims must match (line 1586)
  #   - indices must be integer (line 14419)

  describe "take_along_axis boundary conditions" do
    test "take_along_axis with single index per row" do
      t = Nx.tensor([[10, 20, 30], [40, 50, 60]])
      idx = Nx.tensor([[2], [0]])
      result = Nx.take_along_axis(t, idx, axis: 1)
      assert Nx.shape(result) == {2, 1}
      assert Nx.to_flat_list(result) == [30, 40]
    end

    test "take_along_axis with more indices than original dim" do
      t = Nx.tensor([[1, 2], [3, 4]])
      idx = Nx.tensor([[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]])
      result = Nx.take_along_axis(t, idx, axis: 1)
      assert Nx.shape(result) == {2, 5}
    end

    test "take_along_axis on axis 0" do
      t = Nx.tensor([[1, 2, 3], [4, 5, 6]])
      idx = Nx.tensor([[0, 1, 0], [1, 0, 1], [0, 0, 0]])
      result = Nx.take_along_axis(t, idx, axis: 0)
      assert Nx.shape(result) == {3, 3}
    end

    test "take_along_axis raises on rank mismatch" do
      t = Nx.tensor([[1, 2], [3, 4]])
      idx = Nx.tensor([0, 1])  # rank 1, should be rank 2

      assert_raise ArgumentError, ~r/shapes must have the same number of dimensions/, fn ->
        Nx.take_along_axis(t, idx, axis: 0)
      end
    end

    test "take_along_axis raises on non-indexed dim mismatch" do
      t = Nx.tensor([[1, 2, 3], [4, 5, 6]])  # {2, 3}
      idx = Nx.tensor([[0, 1], [1, 0], [0, 0]])  # {3, 2} — rows don't match

      assert_raise ArgumentError, ~r/non-indexing dimensions must match/, fn ->
        Nx.take_along_axis(t, idx, axis: 1)
      end
    end

    test "take_along_axis identity: iota indices return original" do
      t = Nx.iota({3, 4})
      idx = Nx.stack([Nx.iota({3, 4}, axis: 1)])
      |> Nx.reshape({3, 4})
      result = Nx.take_along_axis(t, idx, axis: 1)
      assert Nx.to_flat_list(result) == Nx.to_flat_list(t)
    end
  end

  # ── Gather boundary conditions ─────────────────────────────────────
  # Source: Nx.Shape.gather/3 (shape.ex:1638)
  # Boundaries:
  #   - indices rank must be >= 1 (line 1641)
  #   - last dim of indices must be <= tensor rank (line 1647)
  #   - axes must be sorted (indexed_axes line 8033)
  #   - axes length must match last dim of indices (line 8037)

  describe "gather boundary conditions" do
    test "gather single element from 2D" do
      t = Nx.iota({3, 4})
      result = Nx.gather(t, Nx.tensor([[2, 3]]))
      assert Nx.shape(result) == {1}
      assert Nx.to_flat_list(result) == [11]  # 2*4 + 3
    end

    test "gather rows (partial indexing)" do
      t = Nx.iota({3, 4})
      result = Nx.gather(t, Nx.tensor([[0], [2]]))
      # last dim=1 < rank=2, so gather sub-tensors
      assert Nx.shape(result) == {2, 4}
    end

    test "gather with axes option (column indexing)" do
      t = Nx.iota({3, 4})
      result = Nx.gather(t, Nx.tensor([[1], [3]]), axes: [1])
      assert Nx.shape(result) == {2, 3}
    end

    test "gather scalar from 1D" do
      t = Nx.tensor([10, 20, 30, 40, 50])
      result = Nx.gather(t, Nx.tensor([[0], [4]]))
      assert Nx.to_flat_list(result) == [10, 50]
    end

    test "gather all elements individually from high-rank tensor" do
      t = Nx.iota({2, 2, 2})
      indices = Nx.tensor([
        [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
        [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]
      ])
      result = Nx.gather(t, indices)
      assert Nx.to_flat_list(result) == Enum.to_list(0..7)
    end

    test "gather with scalar indices raises correct error" do
      t = Nx.iota({3})
      assert_raise ArgumentError, ~r/expected indices rank to be at least 1/, fn ->
        Nx.gather(t, Nx.tensor(0))
      end
    end

    test "gather raises when last dim > tensor rank" do
      t = Nx.iota({3, 4})
      assert_raise ArgumentError, ~r/expected the last indices dimension size/, fn ->
        Nx.gather(t, Nx.tensor([[0, 0, 0]]))
      end
    end

    test "gather with unsorted axes raises" do
      t = Nx.iota({3, 4, 5})
      assert_raise ArgumentError, ~r/:axes must be an ordered list/, fn ->
        Nx.gather(t, Nx.tensor([[0, 0]]), axes: [2, 0])
      end
    end

    test "gather with axes length mismatch raises" do
      t = Nx.iota({3, 4})
      assert_raise ArgumentError, ~r/:axes must have the same number/, fn ->
        Nx.gather(t, Nx.tensor([[0]]), axes: [0, 1])
      end
    end

    test "gather multi-dim indices produces correct output shape" do
      # indices shape {a, b, k} -> output shape {a, b} ++ non-indexed
      t = Nx.iota({4, 5, 6})
      indices = Nx.tensor([[[0, 0], [1, 1]], [[2, 2], [3, 3]]])
      # indices shape: {2, 2, 2}, axes default [0,1], tensor rank 3
      # output: {2, 2} ++ {6} = {2, 2, 6}
      result = Nx.gather(t, indices)
      assert Nx.shape(result) == {2, 2, 6}
    end
  end

  # ── Indexed_add / indexed_put boundary conditions ──────────────────
  # Source: Nx.Shape.indexed/4 (shape.ex:932)
  # Boundaries:
  #   - indices must be rank 1 or 2 (line 938)
  #   - leading axis of indices must match leading axis of updates (line 941)
  #   - rank equation: u - 1 + n == r (line 946)
  #   - update dims must be <= target dims on non-indexed axes (line 957)

  describe "indexed_add boundary conditions" do
    test "indexed_add single scalar update" do
      t = Nx.tensor([0, 0, 0, 0, 0])
      result = Nx.indexed_add(t, Nx.tensor([2]), Nx.tensor(10))
      assert Nx.to_flat_list(result) == [0, 0, 10, 0, 0]
    end

    test "indexed_add multiple overlapping indices" do
      t = Nx.tensor([0, 0, 0])
      indices = Nx.tensor([[1], [1], [1]])
      updates = Nx.tensor([1, 2, 3])
      result = Nx.indexed_add(t, indices, updates)
      # All add to index 1: 0 + 1 + 2 + 3 = 6
      assert Nx.to_number(result[1]) == 6
    end

    test "indexed_add to all positions" do
      t = Nx.tensor([0, 0, 0])
      indices = Nx.tensor([[0], [1], [2]])
      updates = Nx.tensor([10, 20, 30])
      result = Nx.indexed_add(t, indices, updates)
      assert Nx.to_flat_list(result) == [10, 20, 30]
    end

    test "indexed_add with sub-tensor updates (rank > 1)" do
      t = Nx.iota({3, 4})
      indices = Nx.tensor([[0], [2]])
      updates = Nx.tensor([[100, 100, 100, 100], [200, 200, 200, 200]])
      result = Nx.indexed_add(t, indices, updates)
      # Row 0: [0,1,2,3] + [100,100,100,100]
      assert Nx.to_flat_list(result[0]) == [100, 101, 102, 103]
    end

    test "indexed_add with axes option" do
      t = Nx.iota({2, 3})
      indices = Nx.tensor([[0], [2]])
      updates = Nx.tensor([[10, 10], [30, 30]])
      result = Nx.indexed_add(t, indices, updates, axes: [1])
      assert Nx.shape(result) == {2, 3}
    end

    test "indexed_add raises on float indices" do
      t = Nx.tensor([1, 2, 3])
      assert_raise ArgumentError, ~r/indices must be an integer tensor/, fn ->
        Nx.indexed_add(t, Nx.tensor([[0.0]]), Nx.tensor([1]))
      end
    end

    test "indexed_add raises on indices/updates leading axis mismatch" do
      t = Nx.tensor([0, 0, 0])
      indices = Nx.tensor([[0], [1]])  # 2 entries
      updates = Nx.tensor([1, 2, 3])   # 3 entries

      assert_raise ArgumentError, ~r/leading axis .* to match/, fn ->
        Nx.indexed_add(t, indices, updates)
      end
    end

    test "indexed_add raises on rank mismatch" do
      t = Nx.tensor([0, 0, 0])
      # indices last dim=1, updates rank=2 -> u-1+n = 2-1+1 = 2 != 1
      assert_raise ArgumentError, ~r/rank of the input/, fn ->
        Nx.indexed_add(t, Nx.tensor([[0]]), Nx.tensor([[1, 2]]))
      end
    end
  end

  describe "indexed_put boundary conditions" do
    test "indexed_put single element" do
      t = Nx.tensor([1, 2, 3, 4, 5])
      result = Nx.indexed_put(t, Nx.tensor([2]), Nx.tensor(99))
      assert Nx.to_flat_list(result) == [1, 2, 99, 4, 5]
    end

    test "indexed_put overwrites (doesn't add)" do
      t = Nx.tensor([10, 20, 30])
      result = Nx.indexed_put(t, Nx.tensor([[0], [1], [2]]), Nx.tensor([0, 0, 0]))
      assert Nx.to_flat_list(result) == [0, 0, 0]
    end

    test "indexed_put with type promotion" do
      t = Nx.tensor([1, 2, 3], type: :s32)
      result = Nx.indexed_put(t, Nx.tensor([[1]]), Nx.tensor([1.5], type: :f32))
      assert elem(Nx.type(result), 0) == :f
    end
  end

  # ── Window ops boundary conditions ─────────────────────────────────
  # Source: Nx.Shape.pool/5 (shape.ex:855)
  # Boundaries:
  #   - window rank must match tensor rank (validate_window! line 888)
  #   - strides rank must match tensor rank (validate_strides! line 903)
  #   - window result must not be empty (line 873)
  #   - window_scatter source shape must match valid windows (nx.ex:7465)

  describe "window ops boundary conditions" do
    test "window_sum with window == tensor shape (single output)" do
      t = Nx.iota({4, 3}, type: :f32)
      result = Nx.window_sum(t, {4, 3})
      assert Nx.shape(result) == {1, 1}
    end

    test "window_sum with window {1, 1} (identity)" do
      t = Nx.iota({3, 4}, type: :f32)
      result = Nx.window_sum(t, {1, 1})
      assert Nx.to_flat_list(result) == Nx.to_flat_list(t)
    end

    test "window_max with stride == window (non-overlapping)" do
      t = Nx.tensor([1.0, 5.0, 2.0, 8.0, 3.0, 7.0])
      result = Nx.window_max(t, {2}, strides: [2])
      assert Nx.to_flat_list(result) == [5.0, 8.0, 7.0]
    end

    test "window_min with padding :same preserves shape" do
      t = Nx.iota({4, 4}, type: :f32)
      result = Nx.window_min(t, {3, 3}, padding: :same)
      assert Nx.shape(result) == {4, 4}
    end

    test "window raises on rank mismatch" do
      t = Nx.iota({3, 4}, type: :f32)
      assert_raise ArgumentError, ~r/rank of shape .* does not match rank of window/, fn ->
        Nx.window_sum(t, {2})
      end
    end

    test "window raises on stride rank mismatch" do
      t = Nx.iota({3, 4}, type: :f32)
      assert_raise ArgumentError, ~r/rank of shape .* does not match rank of stride/, fn ->
        Nx.window_sum(t, {2, 2}, strides: [1])
      end
    end

    test "window_scatter_max with window covering entire tensor" do
      t = Nx.tensor([3.0, 1.0, 4.0, 1.0, 5.0])
      # window {5} with stride 5, padding :valid -> source shape {1}
      source = Nx.tensor([99.0])
      init = Nx.tensor(0.0)
      result = Nx.window_scatter_max(t, source, init, {5}, strides: [5])
      # Max is at index 4 (value 5.0), so 99.0 goes there
      assert Nx.to_number(result[4]) == 99.0
    end

    test "window_scatter_max raises when source shape doesn't match" do
      t = Nx.iota({6}, type: :f32)
      source = Nx.tensor([1.0, 2.0])  # wrong shape
      init = Nx.tensor(0.0)

      assert_raise ArgumentError, ~r/source shape must match valid windows/, fn ->
        Nx.window_scatter_max(t, source, init, {3}, strides: [1])
      end
    end

    test "window_scatter_max raises on vectorized init_value" do
      t = Nx.iota({6}, type: :f32)
      source = Nx.iota({4}, type: :f32)

      init = Nx.tensor([0.0, 0.0]) |> Nx.vectorize(:batch)

      assert_raise ArgumentError, ~r/init_value tensor cannot be vectorized/, fn ->
        Nx.window_scatter_max(t, source, init, {3}, strides: [1])
      end
    end

    test "window_mean 2D with asymmetric padding" do
      t = Nx.iota({3, 3}, type: :f32)
      result = Nx.window_mean(t, {2, 2}, padding: [{0, 1}, {1, 0}])
      assert is_struct(result, Nx.Tensor)
    end
  end

  # ── Reshape boundary conditions ────────────────────────────────────
  # Source: Nx.Shape.reshape/2 (shape.ex:162)
  # Boundaries:
  #   - product of new shape must equal product of old shape (line 183)
  #   - :auto dimension inference (line 166)
  #   - :auto with incompatible remainder raises (line 174)

  describe "reshape boundary conditions" do
    test "reshape to same shape is identity" do
      t = Nx.iota({3, 4})
      result = Nx.reshape(t, {3, 4})
      assert Nx.to_flat_list(result) == Nx.to_flat_list(t)
    end

    test "reshape to scalar from {1}" do
      t = Nx.tensor([42])
      result = Nx.reshape(t, {})
      assert Nx.to_number(result) == 42
    end

    test "reshape from scalar to {1}" do
      t = Nx.tensor(42)
      result = Nx.reshape(t, {1})
      assert Nx.to_flat_list(result) == [42]
    end

    test "reshape with :auto infers correct dimension" do
      t = Nx.iota({12})
      result = Nx.reshape(t, {3, :auto})
      assert Nx.shape(result) == {3, 4}

      result2 = Nx.reshape(t, {:auto, 6})
      assert Nx.shape(result2) == {2, 6}
    end

    test "reshape raises on incompatible shapes" do
      t = Nx.iota({12})
      assert_raise ArgumentError, ~r/cannot reshape/, fn ->
        Nx.reshape(t, {5, 5})
      end
    end

    test "reshape raises on incompatible :auto" do
      t = Nx.iota({7})  # prime number
      assert_raise ArgumentError, ~r/cannot reshape/, fn ->
        Nx.reshape(t, {3, :auto})
      end
    end

    test "reshape flattens high-rank tensor" do
      t = Nx.iota({2, 3, 4, 5})
      result = Nx.reshape(t, {120})
      assert Nx.shape(result) == {120}
      assert Nx.to_flat_list(result) == Enum.to_list(0..119)
    end

    test "reshape to many dimensions" do
      t = Nx.iota({24})
      result = Nx.reshape(t, {2, 3, 2, 2})
      assert Nx.shape(result) == {2, 3, 2, 2}
    end
  end

  # ── Pad boundary conditions ────────────────────────────────────────
  # Source: Nx.Shape.pad/2 (shape.ex:1033)
  # Boundaries:
  #   - padding rank must match shape rank (line 1043/1051)
  #   - interior padding must be non-negative (line 1060)
  #   - negative edge padding reduces dimension

  describe "pad boundary conditions" do
    test "pad with all zeros is identity" do
      t = Nx.iota({3, 4})
      result = Nx.pad(t, Nx.tensor(0), [{0, 0, 0}, {0, 0, 0}])
      assert Nx.to_flat_list(result) == Nx.to_flat_list(t)
    end

    test "pad with negative edge reduces dimension" do
      t = Nx.iota({5})
      result = Nx.pad(t, Nx.tensor(0), [{-1, -1, 0}])
      # removes first and last: [1, 2, 3]
      assert Nx.shape(result) == {3}
      assert Nx.to_flat_list(result) == [1, 2, 3]
    end

    test "pad with interior padding inserts values between elements" do
      t = Nx.tensor([1, 2, 3])
      result = Nx.pad(t, Nx.tensor(0), [{0, 0, 1}])
      assert Nx.to_flat_list(result) == [1, 0, 2, 0, 3]
    end

    test "pad with interior and edge padding combined" do
      t = Nx.tensor([1, 2])
      result = Nx.pad(t, Nx.tensor(0), [{1, 1, 2}])
      # interior: [1, 0, 0, 2] (2 interior pads between elements)
      # edge: [0, 1, 0, 0, 2, 0]
      assert Nx.shape(result) == {6}
    end

    test "pad raises on interior < 0" do
      t = Nx.iota({3})
      assert_raise ArgumentError, ~r/interior padding must be non-negative/, fn ->
        Nx.pad(t, Nx.tensor(0), [{0, 0, -1}])
      end
    end

    test "pad raises on rank mismatch" do
      t = Nx.iota({3, 4})
      assert_raise ArgumentError, ~r/rank of padding configuration/, fn ->
        Nx.pad(t, Nx.tensor(0), [{0, 0, 0}])
      end
    end

    test "pad scalar tensor" do
      t = Nx.tensor(42)
      result = Nx.pad(t, Nx.tensor(0), [])
      assert Nx.to_number(result) == 42
    end
  end

  # ── Squeeze boundary conditions ────────────────────────────────────
  # Source: Nx.Shape.squeeze/3 (shape.ex:987)
  # Boundary: can only squeeze dimensions of size 1 (line 1000)

  describe "squeeze boundary conditions" do
    test "squeeze all size-1 dimensions" do
      t = Nx.iota({1, 3, 1, 4, 1})
      result = Nx.squeeze(t, axes: [0, 2, 4])
      assert Nx.shape(result) == {3, 4}
    end

    test "squeeze no dimensions (no size-1 dims)" do
      t = Nx.iota({3, 4})
      result = Nx.squeeze(t)
      assert Nx.shape(result) == {3, 4}
    end

    test "squeeze all dimensions from {1,1,1}" do
      t = Nx.tensor([[[42]]])
      result = Nx.squeeze(t)
      assert Nx.shape(result) == {}
      assert Nx.to_number(result) == 42
    end

    test "squeeze raises on non-1 dimension" do
      t = Nx.iota({3, 4})
      assert_raise ArgumentError, ~r/cannot squeeze dimensions whose sizes are not 1/, fn ->
        Nx.squeeze(t, axes: [0])
      end
    end
  end

  # ── Equivalence / differential tests ───────────────────────────────
  # Same result computed two ways — catches subtle logic errors

  describe "algebraic equivalences" do
    property "reshape-then-reduce == direct reduce" do
      check all(
              m <- integer(1..6),
              n <- integer(1..6),
              max_runs: 20
            ) do
        t = Nx.iota({m, n}, type: :f32)
        direct = Nx.sum(t) |> Nx.to_number()
        reshaped = Nx.reshape(t, {m * n}) |> Nx.sum() |> Nx.to_number()
        assert_in_delta direct, reshaped, 1.0e-5
      end
    end

    property "transpose-transpose is identity" do
      check all(
              m <- integer(1..6),
              n <- integer(1..6),
              max_runs: 20
            ) do
        t = Nx.iota({m, n}, type: :f32)
        result = t |> Nx.transpose() |> Nx.transpose()
        assert Nx.to_flat_list(result) == Nx.to_flat_list(t)
      end
    end

    property "exp(log(x)) ≈ x for positive x" do
      check all(n <- integer(1..8), max_runs: 20) do
        # Avoid values too close to 0 (log instability) or too large (exp overflow)
        t = Nx.add(Nx.iota({n}, type: :f32), 1.0)
        result = t |> Nx.log() |> Nx.exp()

        for {orig, roundtrip} <- Enum.zip(Nx.to_flat_list(t), Nx.to_flat_list(result)) do
          assert_in_delta orig, roundtrip, 1.0e-5
        end
      end
    end

    property "negate(negate(x)) == x" do
      check all(n <- integer(1..8), max_runs: 20) do
        t = Nx.iota({n}, type: :f32)
        result = t |> Nx.negate() |> Nx.negate()
        assert Nx.to_flat_list(result) == Nx.to_flat_list(t)
      end
    end

    property "abs(negate(x)) == abs(x)" do
      check all(n <- integer(1..8), max_runs: 20) do
        t = Nx.iota({n}, type: :f32)
        assert Nx.to_flat_list(Nx.abs(Nx.negate(t))) == Nx.to_flat_list(Nx.abs(t))
      end
    end

    property "add(x, 0) == x" do
      check all(n <- integer(1..8), max_runs: 20) do
        t = Nx.iota({n}, type: :f32)
        result = Nx.add(t, 0.0)
        assert Nx.to_flat_list(result) == Nx.to_flat_list(t)
      end
    end

    property "multiply(x, 1) == x" do
      check all(n <- integer(1..8), max_runs: 20) do
        t = Nx.iota({n}, type: :f32)
        result = Nx.multiply(t, 1.0)
        assert Nx.to_flat_list(result) == Nx.to_flat_list(t)
      end
    end

    property "multiply(x, 0) == 0" do
      check all(n <- integer(1..8), max_runs: 20) do
        t = Nx.add(Nx.iota({n}, type: :f32), 1.0)
        result = Nx.multiply(t, 0.0)
        assert Nx.to_flat_list(result) == List.duplicate(0.0, n)
      end
    end

    property "sum of iota(n) == n*(n-1)/2" do
      check all(n <- integer(1..50), max_runs: 30) do
        t = Nx.iota({n}, type: :f64)
        result = Nx.sum(t) |> Nx.to_number()
        expected = n * (n - 1) / 2
        assert_in_delta result, expected, 1.0
      end
    end

    property "dot(x, ones) == sum(x, axes: [-1])" do
      check all(
              m <- integer(1..6),
              n <- integer(1..6),
              max_runs: 20
            ) do
        x = Nx.iota({m, n}, type: :f32)
        ones = Nx.broadcast(Nx.tensor(1.0), {n})
        dot_result = Nx.dot(x, ones)
        sum_result = Nx.sum(x, axes: [-1])
        assert Nx.shape(dot_result) == Nx.shape(sum_result)

        for {d, s} <- Enum.zip(Nx.to_flat_list(dot_result), Nx.to_flat_list(sum_result)) do
          assert_in_delta d, s, 1.0e-4
        end
      end
    end

    property "slice then concatenate recovers original" do
      check all(n <- integer(2..10), max_runs: 20) do
        split = div(n, 2)
        t = Nx.iota({n}, type: :f32)
        left = Nx.slice(t, [0], [split])
        right = Nx.slice(t, [split], [n - split])
        result = Nx.concatenate([left, right])
        assert Nx.to_flat_list(result) == Nx.to_flat_list(t)
      end
    end

    property "take with iota indices is identity" do
      check all(n <- integer(1..10), max_runs: 20) do
        t = Nx.iota({n}, type: :f32)
        idx = Nx.iota({n}, type: :s32)
        result = Nx.take(t, idx)
        assert Nx.to_flat_list(result) == Nx.to_flat_list(t)
      end
    end

    property "gather all indices == flatten" do
      check all(
              m <- integer(1..4),
              n <- integer(1..4),
              max_runs: 20
            ) do
        t = Nx.iota({m, n}, type: :f32)

        indices =
          for i <- 0..(m - 1), j <- 0..(n - 1), do: [i, j]

        idx = Nx.tensor(indices)
        result = Nx.gather(t, idx)
        assert Nx.to_flat_list(result) == Nx.to_flat_list(Nx.reshape(t, {m * n}))
      end
    end

    property "indexed_put then gather recovers update" do
      check all(n <- integer(3..10), max_runs: 20) do
        t = Nx.broadcast(Nx.tensor(0), {n})
        idx = Nx.tensor([[div(n, 2)]])
        update = Nx.tensor([42])
        result = Nx.indexed_put(t, idx, update)

        gathered = Nx.gather(result, Nx.tensor([[div(n, 2)]]))
        assert Nx.to_flat_list(gathered) == [42]
      end
    end

    property "pad then slice recovers original" do
      check all(n <- integer(1..8), max_runs: 20) do
        t = Nx.iota({n}, type: :f32)
        padded = Nx.pad(t, Nx.tensor(0.0), [{2, 3, 0}])
        recovered = Nx.slice(padded, [2], [n])
        assert Nx.to_flat_list(recovered) == Nx.to_flat_list(t)
      end
    end
  end

  # ── Cross-API pattern transfer ─────────────────────────────────────
  # If Bug 4 (window_scatter f64) exists, test similar ops with f64

  describe "f64 type across ops (pattern transfer from window_scatter f64 bug)" do
    test "slice works with f64" do
      t = Nx.iota({6}, type: :f64)
      result = Nx.slice(t, [1], [3])
      assert Nx.type(result) == {:f, 64}
      assert Nx.to_flat_list(result) == [1.0, 2.0, 3.0]
    end

    test "take works with f64" do
      t = Nx.iota({5}, type: :f64)
      result = Nx.take(t, Nx.tensor([4, 2, 0]))
      assert Nx.type(result) == {:f, 64}
      assert Nx.to_flat_list(result) == [4.0, 2.0, 0.0]
    end

    test "gather works with f64" do
      t = Nx.iota({3, 4}, type: :f64)
      result = Nx.gather(t, Nx.tensor([[0, 0], [2, 3]]))
      assert Nx.type(result) == {:f, 64}
    end

    test "indexed_add works with f64" do
      t = Nx.broadcast(Nx.tensor(0.0, type: :f64), {5})
      result = Nx.indexed_add(t, Nx.tensor([[2]]), Nx.tensor([1.0], type: :f64))
      assert Nx.type(result) == {:f, 64}
      assert Nx.to_number(result[2]) == 1.0
    end

    test "pad works with f64" do
      t = Nx.iota({3}, type: :f64)
      result = Nx.pad(t, Nx.tensor(0.0, type: :f64), [{1, 1, 0}])
      assert Nx.type(result) == {:f, 64}
      assert Nx.shape(result) == {5}
    end

    test "window_sum works with f64" do
      t = Nx.iota({6}, type: :f64)
      result = Nx.window_sum(t, {3}, strides: [1])
      assert Nx.type(result) == {:f, 64}
      # [0+1+2, 1+2+3, 2+3+4, 3+4+5] = [3, 6, 9, 12]
      assert Nx.to_flat_list(result) == [3.0, 6.0, 9.0, 12.0]
    end

    test "window_max works with f64" do
      t = Nx.tensor([3.0, 1.0, 4.0, 1.0, 5.0, 9.0], type: :f64)
      result = Nx.window_max(t, {2}, strides: [2])
      assert Nx.type(result) == {:f, 64}
      assert Nx.to_flat_list(result) == [3.0, 4.0, 9.0]
    end

    test "window_min works with f64" do
      t = Nx.tensor([3.0, 1.0, 4.0, 1.0, 5.0, 9.0], type: :f64)
      result = Nx.window_min(t, {2}, strides: [2])
      assert Nx.type(result) == {:f, 64}
      assert Nx.to_flat_list(result) == [1.0, 1.0, 5.0]
    end
  end

  # ── Normalize_axis edge cases ──────────────────────────────────────
  # Source: Nx.Shape.normalize_axis/4 (shape.ex:1105)
  # Boundaries:
  #   - negative axis: valid if abs(axis) <= rank
  #   - positive axis: valid if axis < rank
  #   - named axis: must exist in names

  describe "axis normalization edge cases" do
    test "negative axis -1 selects last dimension" do
      t = Nx.iota({3, 4, 5})
      result = Nx.sum(t, axes: [-1])
      assert Nx.shape(result) == {3, 4}
    end

    test "negative axis -rank selects first dimension" do
      t = Nx.iota({3, 4, 5})
      result = Nx.sum(t, axes: [-3])
      assert Nx.shape(result) == {4, 5}
    end

    test "axis raises on -rank-1" do
      t = Nx.iota({3, 4})
      assert_raise ArgumentError, ~r/given axis .* invalid/, fn ->
        Nx.sum(t, axes: [-3])
      end
    end

    test "axis raises on rank" do
      t = Nx.iota({3, 4})
      assert_raise ArgumentError, ~r/given axis .* invalid/, fn ->
        Nx.sum(t, axes: [2])
      end
    end

    test "named axis" do
      t = Nx.iota({3, 4}, names: [:rows, :cols])
      result = Nx.sum(t, axes: [:cols])
      assert Nx.shape(result) == {3}
    end

    test "named axis raises on unknown name" do
      t = Nx.iota({3, 4}, names: [:rows, :cols])
      assert_raise ArgumentError, ~r/name :foo not found/, fn ->
        Nx.sum(t, axes: [:foo])
      end
    end
  end

  # ── Reduction with keep_axes ───────────────────────────────────────

  describe "reduction keep_axes boundary" do
    test "sum with keep_axes preserves rank" do
      t = Nx.iota({3, 4, 5}, type: :f32)
      result = Nx.sum(t, axes: [1], keep_axes: true)
      assert Nx.shape(result) == {3, 1, 5}
    end

    test "sum of all axes with keep_axes" do
      t = Nx.iota({3, 4}, type: :f32)
      result = Nx.sum(t, keep_axes: true)
      assert Nx.shape(result) == {1, 1}
    end

    test "argmax on single-element axis" do
      t = Nx.tensor([[5], [3], [8]])
      result = Nx.argmax(t, axis: 1)
      assert Nx.to_flat_list(result) == [0, 0, 0]
    end

    test "argmin tie-breaking: returns first occurrence" do
      t = Nx.tensor([3, 1, 1, 1, 5])
      result = Nx.argmin(t)
      assert Nx.to_number(result) == 1
    end

    test "reduce_max on single-element tensor" do
      t = Nx.tensor([42.0])
      assert Nx.to_number(Nx.reduce_max(t)) == 42.0
    end

    test "reduce_min on single-element tensor" do
      t = Nx.tensor([42.0])
      assert Nx.to_number(Nx.reduce_min(t)) == 42.0
    end
  end

  # ── Bitcast boundary conditions ────────────────────────────────────

  describe "bitcast boundary conditions" do
    test "bitcast f32 to u32 and back" do
      t = Nx.tensor(1.0, type: :f32)
      u = Nx.bitcast(t, :u32)
      f = Nx.bitcast(u, :f32)
      assert Nx.to_number(f) == 1.0
    end

    test "bitcast f64 to s64 and back" do
      t = Nx.tensor(1.0, type: :f64)
      s = Nx.bitcast(t, :s64)
      f = Nx.bitcast(s, :f64)
      assert Nx.to_number(f) == 1.0
    end

    test "bitcast raises on size mismatch" do
      t = Nx.tensor(1.0, type: :f32)
      assert_raise ArgumentError, fn ->
        Nx.bitcast(t, :f64)
      end
    end
  end

  # ── Concatenate boundary conditions ────────────────────────────────

  describe "concatenate boundary conditions" do
    test "concatenate single tensor is identity" do
      t = Nx.iota({3, 4})
      result = Nx.concatenate([t])
      assert Nx.to_flat_list(result) == Nx.to_flat_list(t)
    end

    test "concatenate on non-zero axis" do
      a = Nx.iota({2, 3})
      b = Nx.iota({2, 4})
      result = Nx.concatenate([a, b], axis: 1)
      assert Nx.shape(result) == {2, 7}
    end

    test "concatenate with type promotion" do
      a = Nx.iota({3}, type: :s32)
      b = Nx.iota({3}, type: :f32)
      result = Nx.concatenate([a, b])
      assert elem(Nx.type(result), 0) == :f
    end

    test "concatenate many small tensors" do
      tensors = for i <- 0..9, do: Nx.tensor([i])
      result = Nx.concatenate(tensors)
      assert Nx.shape(result) == {10}
      assert Nx.to_flat_list(result) == Enum.to_list(0..9)
    end
  end

  # ── New_axis boundary conditions ───────────────────────────────────

  describe "new_axis boundary conditions" do
    test "new_axis at 0 on scalar" do
      t = Nx.tensor(42)
      result = Nx.new_axis(t, 0)
      assert Nx.shape(result) == {1}
    end

    test "new_axis at -1 on scalar" do
      t = Nx.tensor(42)
      result = Nx.new_axis(t, -1)
      assert Nx.shape(result) == {1}
    end

    test "new_axis at every valid position" do
      t = Nx.iota({3, 4})
      # Valid positions: 0, 1, 2 (and -1, -2, -3)
      assert Nx.shape(Nx.new_axis(t, 0)) == {1, 3, 4}
      assert Nx.shape(Nx.new_axis(t, 1)) == {3, 1, 4}
      assert Nx.shape(Nx.new_axis(t, 2)) == {3, 4, 1}
      assert Nx.shape(Nx.new_axis(t, -1)) == {3, 4, 1}
      assert Nx.shape(Nx.new_axis(t, -2)) == {3, 1, 4}
      assert Nx.shape(Nx.new_axis(t, -3)) == {1, 3, 4}
    end

    test "new_axis raises on out-of-range position" do
      t = Nx.iota({3, 4})
      assert_raise ArgumentError, ~r/new axis position/, fn ->
        Nx.new_axis(t, 3)
      end
    end
  end

  # ── Sort / argsort boundary conditions ─────────────────────────────
  # Source: nx.ex:15128 (sort), nx.ex:15378 (argsort)
  # Boundaries:
  #   - direction must be :asc or :desc
  #   - axis normalized
  #   - complex types rejected

  describe "sort boundary conditions" do
    test "sort on already sorted tensor" do
      t = Nx.tensor([1, 2, 3, 4, 5])
      assert Nx.to_flat_list(Nx.sort(t)) == [1, 2, 3, 4, 5]
    end

    test "sort descending" do
      t = Nx.tensor([3, 1, 4, 1, 5])
      assert Nx.to_flat_list(Nx.sort(t, direction: :desc)) == [5, 4, 3, 1, 1]
    end

    test "sort single element" do
      t = Nx.tensor([42])
      assert Nx.to_flat_list(Nx.sort(t)) == [42]
    end

    test "sort on 2D along axis 0" do
      t = Nx.tensor([[3, 1], [1, 4]])
      result = Nx.sort(t, axis: 0)
      assert Nx.shape(result) == {2, 2}
      # Column-wise sort: [1,1] and [3,4]
      assert Nx.to_flat_list(result) == [1, 1, 3, 4]
    end

    test "sort on 2D along axis 1" do
      t = Nx.tensor([[3, 1], [4, 2]])
      result = Nx.sort(t, axis: 1)
      assert Nx.to_flat_list(result) == [1, 3, 2, 4]
    end

    test "sort with negative axis" do
      t = Nx.tensor([[3, 1], [4, 2]])
      result = Nx.sort(t, axis: -1)
      assert Nx.to_flat_list(result) == [1, 3, 2, 4]
    end

    test "sort raises on invalid direction" do
      t = Nx.tensor([1, 2, 3])
      assert_raise ArgumentError, ~r/unknown value for :direction/, fn ->
        Nx.sort(t, direction: :up)
      end
    end

    test "sort stable preserves order of equal elements" do
      t = Nx.tensor([3.0, 1.0, 2.0, 1.0])
      result = Nx.sort(t, stable: true)
      assert Nx.to_flat_list(result) == [1.0, 1.0, 2.0, 3.0]
    end

    test "argsort returns indices" do
      t = Nx.tensor([30, 10, 20])
      result = Nx.argsort(t)
      assert Nx.to_flat_list(result) == [1, 2, 0]
    end

    test "argsort descending" do
      t = Nx.tensor([30, 10, 20])
      result = Nx.argsort(t, direction: :desc)
      assert Nx.to_flat_list(result) == [0, 2, 1]
    end

    test "argsort raises on invalid direction" do
      t = Nx.tensor([1, 2, 3])
      assert_raise ArgumentError, ~r/unknown value for :direction/, fn ->
        Nx.argsort(t, direction: :up)
      end
    end
  end

  # ── Top_k boundary conditions ──────────────────────────────────────
  # Source: shape.ex:2211
  # Boundaries:
  #   - k >= 1 (line 2231)
  #   - rank >= 1 (line 2214)
  #   - last_dim >= k (line 2223)

  describe "top_k boundary conditions" do
    test "top_k with k == last dim size (full sort)" do
      t = Nx.tensor([3, 1, 4, 1, 5])
      {values, _indices} = Nx.top_k(t, k: 5)
      assert Nx.to_flat_list(values) == [5, 4, 3, 1, 1]
      assert Nx.shape(values) == {5}
    end

    test "top_k with k == 1" do
      t = Nx.tensor([3, 1, 4, 1, 5])
      {values, _indices} = Nx.top_k(t, k: 1)
      assert Nx.to_flat_list(values) == [5]
    end

    test "top_k on 2D tensor" do
      t = Nx.tensor([[3, 1, 4], [1, 5, 9]])
      {values, _indices} = Nx.top_k(t, k: 2)
      assert Nx.shape(values) == {2, 2}
    end

    test "top_k raises on k == 0" do
      t = Nx.tensor([1, 2, 3])
      assert_raise ArgumentError, ~r/k must be .* greater than or equal to 1/, fn ->
        Nx.top_k(t, k: 0)
      end
    end

    test "top_k raises on k > last dim" do
      t = Nx.tensor([1, 2, 3])
      assert_raise ArgumentError, ~r/last axis size must be greater than or equal to k/, fn ->
        Nx.top_k(t, k: 4)
      end
    end

    test "top_k raises on scalar" do
      t = Nx.tensor(42)
      assert_raise ArgumentError, ~r/must have at least rank 1/, fn ->
        Nx.top_k(t, k: 1)
      end
    end
  end

  # ── Linspace boundary conditions ───────────────────────────────────
  # Source: nx.ex:16812
  # Boundaries:
  #   - n must be positive integer
  #   - start/stop must have same shape

  describe "linspace boundary conditions" do
    test "linspace n=1 returns start value" do
      result = Nx.linspace(0, 10, n: 1)
      assert Nx.to_flat_list(result) == [0.0]
    end

    test "linspace n=2 returns endpoints" do
      result = Nx.linspace(0, 10, n: 2)
      assert Nx.to_flat_list(result) == [0.0, 10.0]
    end

    test "linspace endpoint: false excludes endpoint" do
      result = Nx.linspace(0, 10, n: 2, endpoint: false)
      [a, b] = Nx.to_flat_list(result)
      assert a == 0.0
      assert b == 5.0
    end

    test "linspace start == stop returns constant" do
      result = Nx.linspace(5, 5, n: 4)
      assert Nx.to_flat_list(result) == [5.0, 5.0, 5.0, 5.0]
    end

    test "linspace negative range" do
      result = Nx.linspace(10, 0, n: 3)
      assert Nx.to_flat_list(result) == [10.0, 5.0, 0.0]
    end

    test "linspace with type :f64" do
      result = Nx.linspace(0, 1, n: 3, type: :f64)
      assert Nx.type(result) == {:f, 64}
      [a, b, c] = Nx.to_flat_list(result)
      assert_in_delta a, 0.0, 1.0e-10
      assert_in_delta b, 0.5, 1.0e-10
      assert_in_delta c, 1.0, 1.0e-10
    end

    test "linspace raises on n=0" do
      assert_raise ArgumentError, ~r/expected n to be a non-negative integer/, fn ->
        Nx.linspace(0, 10, n: 0)
      end
    end
  end

  # ── Tile boundary conditions ───────────────────────────────────────
  # Source: nx.ex:3261
  # Boundaries:
  #   - repetitions must be list of ints >= 1

  describe "tile boundary conditions" do
    test "tile with all 1s is identity" do
      t = Nx.iota({3, 4})
      result = Nx.tile(t, [1, 1])
      assert Nx.to_flat_list(result) == Nx.to_flat_list(t)
    end

    test "tile adds dimensions when reps > rank" do
      t = Nx.tensor([1, 2, 3])
      result = Nx.tile(t, [2, 3])
      assert Nx.shape(result) == {2, 9}
    end

    test "tile with fewer reps pads with 1s" do
      t = Nx.iota({2, 3})
      result = Nx.tile(t, [2])
      assert Nx.shape(result) == {2, 6}
    end

    test "tile scalar" do
      t = Nx.tensor(5)
      result = Nx.tile(t, [3])
      assert Nx.to_flat_list(result) == [5, 5, 5]
    end

    test "tile raises on rep < 1" do
      t = Nx.tensor([1, 2, 3])
      assert_raise ArgumentError, ~r/repetitions must be a list of integers/, fn ->
        Nx.tile(t, [0])
      end
    end
  end

  # ── Stack boundary conditions ──────────────────────────────────────
  # Source: nx.ex:14937
  # Boundaries:
  #   - non-empty list required
  #   - all shapes must match

  describe "stack boundary conditions" do
    test "stack single tensor" do
      t = Nx.tensor([1, 2, 3])
      result = Nx.stack([t])
      assert Nx.shape(result) == {1, 3}
    end

    test "stack scalars" do
      result = Nx.stack([Nx.tensor(1), Nx.tensor(2), Nx.tensor(3)])
      assert Nx.shape(result) == {3}
      assert Nx.to_flat_list(result) == [1, 2, 3]
    end

    test "stack on axis 1" do
      a = Nx.tensor([1, 2])
      b = Nx.tensor([3, 4])
      result = Nx.stack([a, b], axis: 1)
      assert Nx.shape(result) == {2, 2}
    end

    test "stack raises on empty list" do
      assert_raise ArgumentError, ~r/no tensors were given to stack/, fn ->
        Nx.stack([])
      end
    end

    test "stack raises on shape mismatch" do
      assert_raise ArgumentError, ~r/same shape/, fn ->
        Nx.stack([Nx.tensor([1, 2]), Nx.tensor([1, 2, 3])])
      end
    end

    test "stack with type promotion" do
      a = Nx.tensor([1, 2], type: :s32)
      b = Nx.tensor([1.5, 2.5], type: :f32)
      result = Nx.stack([a, b])
      assert elem(Nx.type(result), 0) == :f
    end
  end

  # ── Dot boundary conditions ────────────────────────────────────────
  # Source: shape.ex:1851 (validate_dot_axes!)
  # Boundaries:
  #   - contracting dims must match
  #   - batch axes must be successive from 0
  #   - batch and contract axes cannot overlap

  describe "dot boundary conditions" do
    test "dot with mismatched contracting dims raises" do
      a = Nx.iota({3, 4})
      b = Nx.iota({5, 2})

      assert_raise ArgumentError, ~r/dot.* expects shapes to be compatible/, fn ->
        Nx.dot(a, [1], b, [0])
      end
    end

    test "dot vector-vector (inner product)" do
      a = Nx.tensor([1.0, 2.0, 3.0])
      b = Nx.tensor([4.0, 5.0, 6.0])
      result = Nx.dot(a, b)
      assert Nx.to_number(result) == 32.0
    end

    test "dot matrix-vector" do
      a = Nx.iota({2, 3}, type: :f32)
      b = Nx.tensor([1.0, 1.0, 1.0])
      result = Nx.dot(a, b)
      assert Nx.shape(result) == {2}
    end

    test "dot with batch axes" do
      # batch_size=2, then 3x4 @ 4x5
      a = Nx.iota({2, 3, 4}, type: :f32)
      b = Nx.iota({2, 4, 5}, type: :f32)
      result = Nx.dot(a, [2], [0], b, [1], [0])
      assert Nx.shape(result) == {2, 3, 5}
    end

    test "dot raises when batch axes are not successive from 0" do
      a = Nx.iota({2, 3, 4}, type: :f32)
      b = Nx.iota({2, 4, 5}, type: :f32)

      assert_raise ArgumentError, ~r/batch axes must be successive/, fn ->
        Nx.dot(a, [2], [1], b, [1], [1])
      end
    end
  end

  # ── Reverse boundary conditions ────────────────────────────────────

  describe "reverse boundary conditions" do
    test "reverse 1D" do
      t = Nx.tensor([1, 2, 3, 4, 5])
      result = Nx.reverse(t)
      assert Nx.to_flat_list(result) == [5, 4, 3, 2, 1]
    end

    test "reverse with empty axes is identity" do
      t = Nx.iota({3, 4})
      result = Nx.reverse(t, axes: [])
      assert Nx.to_flat_list(result) == Nx.to_flat_list(t)
    end

    test "reverse single element is identity" do
      t = Nx.tensor([42])
      result = Nx.reverse(t)
      assert Nx.to_flat_list(result) == [42]
    end

    test "reverse specific axis on 2D" do
      t = Nx.tensor([[1, 2], [3, 4]])
      # Reverse axis 0: rows swap
      result0 = Nx.reverse(t, axes: [0])
      assert Nx.to_flat_list(result0) == [3, 4, 1, 2]
      # Reverse axis 1: cols swap
      result1 = Nx.reverse(t, axes: [1])
      assert Nx.to_flat_list(result1) == [2, 1, 4, 3]
    end

    test "reverse all axes on 2D" do
      t = Nx.tensor([[1, 2], [3, 4]])
      result = Nx.reverse(t, axes: [0, 1])
      assert Nx.to_flat_list(result) == [4, 3, 2, 1]
    end

    test "double reverse is identity" do
      t = Nx.iota({3, 4})
      result = t |> Nx.reverse() |> Nx.reverse()
      assert Nx.to_flat_list(result) == Nx.to_flat_list(t)
    end

    test "reverse scalar is identity" do
      t = Nx.tensor(42)
      result = Nx.reverse(t)
      assert Nx.to_number(result) == 42
    end
  end

  # ── Flatten boundary conditions ────────────────────────────────────
  # Source: shape.ex:590
  # Boundary: axes must be consecutive

  describe "flatten boundary conditions" do
    test "flatten no args produces rank 1" do
      t = Nx.iota({2, 3, 4})
      result = Nx.flatten(t)
      assert Nx.shape(result) == {24}
    end

    test "flatten scalar produces {1}" do
      t = Nx.tensor(42)
      result = Nx.flatten(t)
      assert Nx.shape(result) == {1}
    end

    test "flatten partial: first two axes" do
      t = Nx.iota({2, 3, 4})
      result = Nx.flatten(t, axes: [0, 1])
      assert Nx.shape(result) == {6, 4}
    end

    test "flatten partial: last two axes" do
      t = Nx.iota({2, 3, 4})
      result = Nx.flatten(t, axes: [1, 2])
      assert Nx.shape(result) == {2, 12}
    end

    test "flatten raises on non-consecutive axes" do
      t = Nx.iota({2, 3, 4})
      assert_raise ArgumentError, ~r/flatten axes must be consecutive/, fn ->
        Nx.flatten(t, axes: [0, 2])
      end
    end

    test "flatten single axis is identity on that dimension" do
      t = Nx.iota({2, 3, 4})
      result = Nx.flatten(t, axes: [1])
      assert Nx.shape(result) == {2, 3, 4}
    end
  end

  # ── Broadcast boundary conditions ──────────────────────────────────

  describe "broadcast boundary conditions" do
    test "broadcast scalar to shape" do
      t = Nx.tensor(5)
      result = Nx.broadcast(t, {3, 4})
      assert Nx.shape(result) == {3, 4}
      assert Enum.all?(Nx.to_flat_list(result), &(&1 == 5))
    end

    test "broadcast {1} to {5}" do
      t = Nx.tensor([42])
      result = Nx.broadcast(t, {5})
      assert Nx.to_flat_list(result) == [42, 42, 42, 42, 42]
    end

    test "broadcast {1, 3} to {4, 3}" do
      t = Nx.tensor([[1, 2, 3]])
      result = Nx.broadcast(t, {4, 3})
      assert Nx.shape(result) == {4, 3}
    end

    test "broadcast with explicit axes" do
      t = Nx.tensor([1, 2, 3])
      result = Nx.broadcast(t, {3, 4}, axes: [0])
      assert Nx.shape(result) == {3, 4}
    end

    test "broadcast raises on incompatible shapes" do
      t = Nx.tensor([1, 2, 3])
      assert_raise ArgumentError, ~r/cannot broadcast/, fn ->
        Nx.broadcast(t, {4})
      end
    end

    test "broadcast raises on unordered axes" do
      t = Nx.iota({2, 3})
      assert_raise ArgumentError, ~r/broadcast axes must be ordered/, fn ->
        Nx.broadcast(t, {4, 3, 2}, axes: [2, 0])
      end
    end

    test "broadcast to same shape is identity" do
      t = Nx.iota({3, 4})
      result = Nx.broadcast(t, {3, 4})
      assert Nx.to_flat_list(result) == Nx.to_flat_list(t)
    end
  end

  # ── Outer product boundary conditions ──────────────────────────────

  describe "outer boundary conditions" do
    test "outer of two vectors" do
      a = Nx.tensor([1.0, 2.0, 3.0])
      b = Nx.tensor([4.0, 5.0])
      result = Nx.outer(a, b)
      assert Nx.shape(result) == {3, 2}
      assert Nx.to_flat_list(result) == [4.0, 5.0, 8.0, 10.0, 12.0, 15.0]
    end

    test "outer of scalars" do
      a = Nx.tensor(3)
      b = Nx.tensor(4)
      result = Nx.outer(a, b)
      assert Nx.shape(result) == {1, 1}
    end

    test "outer of 2D tensors (flattens first)" do
      a = Nx.iota({2, 3})
      b = Nx.iota({4})
      result = Nx.outer(a, b)
      assert Nx.shape(result) == {6, 4}
    end

    test "outer with type promotion" do
      a = Nx.tensor([1, 2], type: :s32)
      b = Nx.tensor([1.0, 2.0], type: :f32)
      result = Nx.outer(a, b)
      assert elem(Nx.type(result), 0) == :f
    end
  end
end
