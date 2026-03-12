defmodule EXLA.Defn.VectorizeTest do
  use EXLA.Case, async: true

  import Nx.Defn
  import Nx, only: :sigils

  setup do
    Nx.default_backend(EXLA.Backend)

    base =
      Nx.tensor([
        [[0, 1, 2]],
        [[3, 4, 5]],
        [[6, 7, 8]]
      ])

    vectorized = Nx.vectorize(base, :rows)
    %{base: base, vectorized: vectorized}
  end

  defn add_n(x, y), do: Nx.add(x, y)

  def add(x, y) do
    EXLA.jit_apply(&add_n/2, [x, y])
  end

  describe "addition" do
    test "left addition by scalar", %{vectorized: vectorized} do
      result = add(2, vectorized)
      assert result.shape == {1, 3}

      assert_equal(
        result,
        Nx.vectorize(
          Nx.tensor([[[2, 3, 4]], [[5, 6, 7]], [[8, 9, 10]]]),
          :rows
        )
      )
    end

    test "right addition by scalar", %{vectorized: vectorized} do
      result = add(vectorized, 2)
      assert result.shape == {1, 3}

      assert_equal(
        result,
        Nx.vectorize(
          Nx.tensor([[[2, 3, 4]], [[5, 6, 7]], [[8, 9, 10]]]),
          :rows
        )
      )
    end

    test "left addition by rank-1", %{vectorized: vectorized} do
      result = add(Nx.tensor([2]), vectorized)
      assert result.shape == {1, 3}

      assert_equal(
        result,
        Nx.vectorize(
          Nx.tensor([[[2, 3, 4]], [[5, 6, 7]], [[8, 9, 10]]]),
          :rows
        )
      )
    end

    test "right addition by rank-1", %{vectorized: vectorized} do
      result = add(vectorized, Nx.tensor([2]))
      assert result.shape == {1, 3}

      assert_equal(
        result,
        Nx.vectorize(
          Nx.tensor([[[2, 3, 4]], [[5, 6, 7]], [[8, 9, 10]]]),
          :rows
        )
      )
    end

    test "left addition by rank-2", %{vectorized: vectorized} do
      result = add(Nx.tensor([[1], [2]]), vectorized)
      assert result.shape == {2, 3}

      assert_equal(
        result,
        Nx.vectorize(
          Nx.tensor([
            [[1, 2, 3], [2, 3, 4]],
            [[4, 5, 6], [5, 6, 7]],
            [[7, 8, 9], [8, 9, 10]]
          ]),
          :rows
        )
      )
    end

    test "right addition by rank-2", %{vectorized: vectorized} do
      result = add(vectorized, Nx.tensor([[1], [2]]))
      assert result.shape == {2, 3}

      assert_equal(
        result,
        Nx.vectorize(
          Nx.tensor([
            [[1, 2, 3], [2, 3, 4]],
            [[4, 5, 6], [5, 6, 7]],
            [[7, 8, 9], [8, 9, 10]]
          ]),
          :rows
        )
      )
    end

    test "addition by vectorized with same axes", %{base: base, vectorized: vectorized} do
      assert_equal(
        Nx.vectorize(Nx.add(base, base), :rows),
        add(vectorized, vectorized)
      )
    end

    test "addition by vectorized with different axes", %{vectorized: vectorized, base: base} do
      v2 = Nx.vectorize(base, :cols)

      result =
        Nx.stack([
          Nx.add(base, Nx.tensor([[0, 1, 2]])),
          Nx.add(base, Nx.tensor([[3, 4, 5]])),
          Nx.add(base, Nx.tensor([[6, 7, 8]]))
        ])
        |> Nx.vectorize(:rows)
        |> Nx.vectorize(:cols)

      assert_equal(result, add(vectorized, v2))
    end

    test "addition by vectorized with common axes", %{vectorized: vectorized, base: base} do
      base_2 = Nx.iota({1, 2, 3, 1, 3})
      v2 = base_2 |> Nx.vectorize(:x) |> Nx.vectorize(:y) |> Nx.vectorize(:rows)

      base = Nx.concatenate([base, base], axis: 1) |> Nx.reshape({1, 2, 3, 1, 3})

      result =
        Nx.add(base, base_2)
        |> Nx.reshape({3, 1, 2, 1, 3})
        |> Nx.vectorize(:rows)
        |> Nx.vectorize(:x)
        |> Nx.vectorize(:y)

      assert_equal(result, add(vectorized, v2))
    end
  end

  test "squeeze" do
    assert_equal(
      EXLA.jit(&Nx.squeeze/1).(Nx.iota({1, 1, 2}) |> Nx.vectorize(:x)),
      Nx.iota({1, 2}) |> Nx.vectorize(:x)
    )
  end

  describe "cond" do
    deftransformp send_value(val, opts \\ []) do
      Nx.Defn.Kernel.hook(
        val,
        &send(opts[:pid] || self(), {:vectorization_test, &1, clause: opts[:clause]})
      )
    end

    defn vectorized_if(pred, then, other, opts \\ []) do
      cond do
        pred -> send_value(then, pid: opts[:pid], clause: "if")
        true -> send_value(other, pid: opts[:pid], clause: "else")
      end
    end

    defn vectorized_cond(pred1, clause1, pred2, clause2, clause3, opts \\ []) do
      cond do
        pred1 -> send_value(clause1, pid: opts[:pid], clause: "clause_1")
        pred2 -> send_value(clause2, pid: opts[:pid], clause: "clause_2")
        true -> send_value(clause3, pid: opts[:pid], clause: "clause_3")
      end
    end

    test "simple if" do
      # this tests the case where we have a single vectorized predicate
      pred = Nx.vectorize(~VEC[0 1 0], :pred)

      assert_equal(vectorized_if(pred, 1, 2, pid: self()), Nx.vectorize(~VEC[2 1 2], :pred))

      assert_received {:vectorization_test, t, clause: "if"}
      assert_equal(t, Nx.tensor(1))
      assert_received {:vectorization_test, t, clause: "else"}
      assert_equal(t, Nx.tensor(2))
      refute_received {:vectorization_test, _, _}
    end

    test "simple cond" do
      # This tests the case where we have two vectorized predicates
      pred1 = Nx.vectorize(~VEC[1 0 0], :pred)
      pred2 = Nx.vectorize(~VEC[0 0 0], :pred)

      assert_equal(
        vectorized_cond(pred1, 1, pred2, 2, 3, pid: self()),
        Nx.vectorize(~VEC[1 3 3], :pred)
      )

      assert_received {:vectorization_test, t, clause: "clause_1"}
      assert_equal(t, Nx.tensor(1))
      assert_received {:vectorization_test, t, clause: "clause_3"}
      assert_equal(t, Nx.tensor(3))
      refute_received {:vectorization_test, _, _}
    end

    test "if with container result" do
      pred1 = Nx.vectorize(~VEC[2 0 0], :pred)

      result =
        vectorized_if(
          pred1,
          {1, 2, 3},
          {7, 8, Nx.vectorize(~VEC[9 10 11], :x)},
          pid: self()
        )

      assert_equal(result, {
        Nx.vectorize(~VEC[1 7 7], :pred),
        Nx.vectorize(~VEC[2 8 8], :pred),
        Nx.vectorize(~MAT[
                  3 3 3
                  9 10 11
                  9 10 11
                ], pred: 3, x: 3)
      })

      assert_received {:vectorization_test, t, clause: "if"}
      assert_equal(t, {Nx.tensor(1), Nx.tensor(2), Nx.tensor(3)})
      assert_received {:vectorization_test, t, clause: "else"}
      assert_equal(t, {Nx.tensor(7), Nx.tensor(8), Nx.vectorize(Nx.tensor([9, 10, 11]), :x)})
      refute_received {:vectorization_test, _, _}
    end

    defn cond4(p1, c1, p2, c2, p3, c3, c4, opts \\ []) do
      cond do
        p1 -> send_value(c1, pid: opts[:pid], clause: "c1")
        p2 -> send_value(c2, pid: opts[:pid], clause: "c2")
        p3 -> send_value(c3, pid: opts[:pid], clause: "c3")
        true -> send_value(c4, pid: opts[:pid], clause: "c4")
      end
    end

    test "only executes selected branches" do
      t = Nx.vectorize(~VEC[1], :pred)
      f = Nx.vectorize(~VEC[0], :pred)

      assert = fn res, val, clause ->
        t = Nx.tensor(val)
        assert_equal(Nx.vectorize(Nx.new_axis(t, 0), :pred), res)
        assert_received {:vectorization_test, rec_t, clause: ^clause}
        assert_equal(rec_t, t)
        refute_received {:vectorization_test, _, _}
      end

      assert.(cond4(t, 10, 0, 20, 0, 30, 40, pid: self()), 10, "c1")
      assert.(cond4(0, 10, t, 20, 0, 30, 40, pid: self()), 20, "c2")
      assert.(cond4(0, 10, 0, 20, t, 30, 40, pid: self()), 30, "c3")
      assert.(cond4(f, 10, 0, 20, 0, 30, 40, pid: self()), 40, "c4")
    end

    test "1 vectorized pred in the beginning" do
      assert_equal(
        cond4(Nx.vectorize(~VEC[0 1], :pred), 10, 0, 20, 0, 30, 40),
        Nx.vectorize(~VEC[40 10], :pred)
      )

      assert_equal(
        cond4(Nx.vectorize(~VEC[0 0], :pred), 10, 1, 20, 0, 30, 40),
        Nx.vectorize(~VEC[20 20], :pred)
      )

      assert_equal(
        cond4(Nx.vectorize(~VEC[0 0], :pred), 10, 0, 20, 1, 30, 40),
        Nx.vectorize(~VEC[30 30], :pred)
      )

      assert_equal(
        cond4(Nx.vectorize(~VEC[0 0], :pred), 10, 0, 20, 0, 30, 40),
        Nx.vectorize(~VEC[40 40], :pred)
      )
    end

    test "1 vectorized pred in the second but not last position" do
      assert_equal(
        cond4(0, 10, Nx.vectorize(~VEC[0 1], :pred), 20, 0, 30, 40),
        Nx.vectorize(~VEC[40 20], :pred)
      )

      assert_equal(
        cond4(1, 10, Nx.vectorize(~VEC[0 1], :pred), 20, 0, 30, 40),
        Nx.vectorize(~VEC[10 10], :pred)
      )

      assert_equal(
        cond4(0, 10, Nx.vectorize(~VEC[0 0], :pred), 20, 1, 30, 40),
        Nx.vectorize(~VEC[30 30], :pred)
      )

      assert_equal(
        cond4(0, 10, Nx.vectorize(~VEC[0 0], :pred), 20, 0, 30, 40),
        Nx.vectorize(~VEC[40 40], :pred)
      )
    end

    test "1 vectorized pred in the last position" do
      assert_equal(
        cond4(0, 10, 0, 20, Nx.vectorize(~VEC[0 1], :pred), 30, 40),
        Nx.vectorize(~VEC[40 30], :pred)
      )

      assert_equal(
        cond4(1, 10, 0, 20, Nx.vectorize(~VEC[0 1], :pred), 30, 40),
        Nx.vectorize(~VEC[10 10], :pred)
      )

      assert_equal(
        cond4(0, 10, 1, 20, Nx.vectorize(~VEC[0 1], :pred), 30, 40),
        Nx.vectorize(~VEC[20 20], :pred)
      )

      assert_equal(
        cond4(0, 10, 0, 20, Nx.vectorize(~VEC[0 0], :pred), 30, 40),
        Nx.vectorize(~VEC[40 40], :pred)
      )
    end

    # Without hooks, vectorized cond with different axes works correctly.
    # This confirms the bug is in the outfeed system, not the cond logic.
    defn unhook_cond_different_axes(p1, p2) do
      cond do
        p1 -> 1
        p2 -> 2
        true -> 0
      end
    end

    test "cond with different vectorization axes works without hooks" do
      result =
        unhook_cond_different_axes(
          Nx.vectorize(~VEC[1 0], :a),
          Nx.vectorize(~VEC[0 1 0], :b)
        )

      assert_equal(
        result,
        Nx.vectorize(~MAT[
              1 1 1
              0 2 0
            ], a: 2, b: 3)
      )
    end

    # Minimal reproduction for #1689: outfeed buffer size mismatch when
    # hooks are used inside cond with predicates on different vectorization
    # axes. The outfeed typespecs are computed from the devectorized shape
    # at compile time, but at runtime the vectorized axes cause different
    # buffer sizes in different branches, crashing XLA.
    defn hooked_cond_different_axes(p1, p2) do
      cond do
        p1 -> send_value(1, clause: "p1")
        p2 -> send_value(2, clause: "p2")
        true -> send_value(0, clause: "default")
      end
    end

    test "hook inside cond with different vectorization axes (issue #1689)" do
      result =
        hooked_cond_different_axes(
          Nx.vectorize(~VEC[1 0], :a),
          Nx.vectorize(~VEC[0 1 0], :b)
        )

      assert_equal(
        result,
        Nx.vectorize(~MAT[
              1 1 1
              0 2 0
            ], a: 2, b: 3)
      )
    end

    test "2 vectorized preds with different axes" do
      assert_equal(
        cond4(
          Nx.vectorize(~VEC[0 1 0], :pred1),
          10,
          Nx.vectorize(~VEC[1 0], :pred2),
          20,
          0,
          30,
          40
        ),
        Nx.vectorize(~MAT[
              20 40
              10 10
              20 40
            ], pred1: 3, pred2: 2)
      )
    end

    # Stress tests: more complex vectorization scenarios for outfeed robustness

    defn hooked_cond_3axes(p1, p2, p3) do
      cond do
        p1 -> send_value(1, clause: "p1")
        p2 -> send_value(2, clause: "p2")
        p3 -> send_value(3, clause: "p3")
        true -> send_value(0, clause: "default")
      end
    end

    test "hooked cond with 3 different vectorization axes" do
      # p1: a=0 -> 1, a=1 -> 0
      # p2: b=0 -> 0, b=1 -> 1
      # p3: c=0 -> 0, c=1 -> 0, c=2 -> 1
      #
      # Result is vectorized[a: 2][b: 2][c: 3]
      # a=0: p1=1 for all b,c → all 1s
      # a=1, b=0: p1=0, p2=0, check p3 → c=0->0, c=1->0, c=2->3
      # a=1, b=1: p1=0, p2=1 → all 2s
      result =
        hooked_cond_3axes(
          Nx.vectorize(~VEC[1 0], :a),
          Nx.vectorize(~VEC[0 1], :b),
          Nx.vectorize(~VEC[0 0 1], :c)
        )

      expected =
        Nx.tensor([
          [[1, 1, 1], [1, 1, 1]],
          [[0, 0, 3], [2, 2, 2]]
        ])
        |> Nx.vectorize(:a)
        |> Nx.vectorize(:b)
        |> Nx.vectorize(:c)

      assert_equal(result, expected)
    end

    defn hooked_cond_tensor_result(p1, p2) do
      cond do
        p1 -> send_value(Nx.tensor([[1, 2], [3, 4]]), clause: "p1")
        p2 -> send_value(Nx.tensor([[5, 6], [7, 8]]), clause: "p2")
        true -> send_value(Nx.tensor([[0, 0], [0, 0]]), clause: "default")
      end
    end

    test "hooked cond with higher-rank tensor results and different axes" do
      # p1: a=0 -> 1, a=1 -> 0
      # p2: b=0 -> 0, b=1 -> 1
      #
      # Result is vectorized[a: 2][b: 2] with inner shape {2, 2}
      # a=0: p1=1 → [[1,2],[3,4]] for all b
      # a=1, b=0: p1=0, p2=0 → [[0,0],[0,0]]
      # a=1, b=1: p1=0, p2=1 → [[5,6],[7,8]]
      result =
        hooked_cond_tensor_result(
          Nx.vectorize(~VEC[1 0], :a),
          Nx.vectorize(~VEC[0 1], :b)
        )

      expected =
        Nx.tensor([
          [[[1, 2], [3, 4]], [[1, 2], [3, 4]]],
          [[[0, 0], [0, 0]], [[5, 6], [7, 8]]]
        ])
        |> Nx.vectorize(:a)
        |> Nx.vectorize(:b)

      assert_equal(result, expected)
    end

    defn unhook_cond_3axes(p1, p2, p3) do
      cond do
        p1 -> 1
        p2 -> 2
        p3 -> 3
        true -> 0
      end
    end

    test "unhook cond with 3 different vectorization axes" do
      result =
        unhook_cond_3axes(
          Nx.vectorize(~VEC[1 0], :a),
          Nx.vectorize(~VEC[0 1], :b),
          Nx.vectorize(~VEC[0 0 1], :c)
        )

      expected =
        Nx.tensor([
          [[1, 1, 1], [1, 1, 1]],
          [[0, 0, 3], [2, 2, 2]]
        ])
        |> Nx.vectorize(:a)
        |> Nx.vectorize(:b)
        |> Nx.vectorize(:c)

      assert_equal(result, expected)
    end

    test "2 vectorized preds with different axes + clauses that match either" do
      assert_equal(
        cond4(
          Nx.vectorize(~VEC[0 1 0], :pred1),
          Nx.vectorize(~VEC[10 100], :pred2),
          Nx.vectorize(~VEC[1 0], :pred2),
          Nx.vectorize(~VEC[20 200 2000], :pred1),
          0,
          30,
          40
        ),
        Nx.vectorize(~MAT[
              20 40
              10 100
              2000 40
            ], pred1: 3, pred2: 2)
      )
    end

    # --- Additional stress tests for outfeed robustness ---

    # 1. Container (tuple) results with cross-axis predicates
    defn hooked_container_cross_axis(pred, then_a, then_b, else_a, else_b, opts \\ []) do
      cond do
        pred -> send_value({then_a, then_b}, pid: opts[:pid], clause: "if")
        true -> send_value({else_a, else_b}, pid: opts[:pid], clause: "else")
      end
    end

    test "container result with cross-axis predicates" do
      # pred on :a, results are scalars → output vectorized[a: 3] tuple
      # a=0: pred=1 → {10, 20}
      # a=1: pred=0 → {30, 40}
      # a=2: pred=1 → {10, 20}
      {ra, rb} =
        hooked_container_cross_axis(
          Nx.vectorize(~VEC[1 0 1], :a),
          10, 20, 30, 40,
          pid: self()
        )

      assert_equal(ra, Nx.vectorize(~VEC[10 30 10], :a))
      assert_equal(rb, Nx.vectorize(~VEC[20 40 20], :a))

      assert_received {:vectorization_test, t, clause: "if"}
      assert_equal(t, {Nx.tensor(10), Nx.tensor(20)})
      assert_received {:vectorization_test, t, clause: "else"}
      assert_equal(t, {Nx.tensor(30), Nx.tensor(40)})
      refute_received {:vectorization_test, _, _}
    end

    test "container result with two different vectorization axes" do
      # pred on :a (size 2), result values on :b (size 3) via broadcast
      # a=0: pred=1 → {10, 20}
      # a=1: pred=0 → {Nx.vectorize(~VEC[30 31 32], :b), 40}
      #
      # Output: tuple of vectorized[a: 2] tensors, second element of first
      # has vectorized[a: 2][b: 3]
      {ra, rb} =
        hooked_container_cross_axis(
          Nx.vectorize(~VEC[1 0], :a),
          10, 20,
          Nx.vectorize(~VEC[30 31 32], :b), 40,
          pid: self()
        )

      expected_a =
        Nx.tensor([[10, 10, 10], [30, 31, 32]])
        |> Nx.vectorize(:a)
        |> Nx.vectorize(:b)

      assert_equal(ra, expected_a)
      assert_equal(rb, Nx.vectorize(~VEC[20 40], :a))
    end

    # 2. Multiple hooks per branch
    defn multi_hook_branch(pred) do
      cond do
        pred ->
          a = send_value(1, clause: "first_if")
          send_value(Nx.add(a, 10), clause: "second_if")

        true ->
          a = send_value(2, clause: "first_else")
          send_value(Nx.add(a, 20), clause: "second_else")
      end
    end

    test "multiple hooks in same branch with different axes" do
      result = multi_hook_branch(Nx.vectorize(~VEC[1 0 1], :a))

      # a=0: pred=1 → 1+10=11
      # a=1: pred=0 → 2+20=22
      # a=2: pred=1 → 1+10=11
      assert_equal(result, Nx.vectorize(~VEC[11 22 11], :a))
    end

    # 3. Large vectorization sizes
    test "hooked cond with large vectorization sizes" do
      # 10 elements on :a, 7 on :b
      p1_data = List.duplicate(0, 10) |> List.replace_at(0, 1) |> List.replace_at(5, 1)
      p2_data = List.duplicate(0, 7) |> List.replace_at(2, 1) |> List.replace_at(6, 1)

      p1 = Nx.tensor(p1_data) |> Nx.vectorize(:a)
      p2 = Nx.tensor(p2_data) |> Nx.vectorize(:b)

      result = hooked_cond_different_axes(p1, p2)

      # Build expected 10x7 matrix
      # a=0,5: p1=1 → all 1s (row of 7 ones)
      # other a: p1=0, check p2 → b=2,6: p2=1 → 2, rest → 0
      expected =
        for a <- 0..9 do
          for b <- 0..6 do
            cond do
              Enum.at(p1_data, a) == 1 -> 1
              Enum.at(p2_data, b) == 1 -> 2
              true -> 0
            end
          end
        end
        |> Nx.tensor()
        |> Nx.vectorize(:a)
        |> Nx.vectorize(:b)

      assert_equal(result, expected)
    end

    # 4. Computed predicates (Nx.greater instead of literal vectors)
    defn hooked_cond_computed_preds(x, y) do
      cond do
        Nx.greater(x, 5) -> send_value(1, clause: "gt5")
        Nx.greater(y, 3) -> send_value(2, clause: "gt3")
        true -> send_value(0, clause: "default")
      end
    end

    test "hooked cond with computed predicates on different axes" do
      # x vectorized on :a: [2, 8, 3] → greater(x,5): [0, 1, 0]
      # y vectorized on :b: [1, 5, 4, 2] → greater(y,3): [0, 1, 1, 0]
      #
      # Result is vectorized[a: 3][b: 4]
      # a=0 (x=2, gt5=0): check y → b=0->0, b=1->2, b=2->2, b=3->0
      # a=1 (x=8, gt5=1): all → 1
      # a=2 (x=3, gt5=0): check y → b=0->0, b=1->2, b=2->2, b=3->0
      result =
        hooked_cond_computed_preds(
          Nx.vectorize(Nx.tensor([2, 8, 3]), :a),
          Nx.vectorize(Nx.tensor([1, 5, 4, 2]), :b)
        )

      expected =
        Nx.tensor([
          [0, 2, 2, 0],
          [1, 1, 1, 1],
          [0, 2, 2, 0]
        ])
        |> Nx.vectorize(:a)
        |> Nx.vectorize(:b)

      assert_equal(result, expected)
    end

    # 5. Vectorized values in branch results (result vectorized on different axis than pred)
    defn hooked_cond_vectorized_results(pred, vec_result) do
      cond do
        pred -> send_value(vec_result, clause: "if")
        true -> send_value(Nx.tensor(0), clause: "else")
      end
    end

    test "branch result vectorized on different axis than predicate" do
      # pred on :a (size 2): [1, 0]
      # vec_result on :b (size 3): [10, 20, 30]
      #
      # Result is vectorized[a: 2][b: 3]
      # a=0: pred=1 → [10, 20, 30]
      # a=1: pred=0 → [0, 0, 0]
      result =
        hooked_cond_vectorized_results(
          Nx.vectorize(~VEC[1 0], :a),
          Nx.vectorize(Nx.tensor([10, 20, 30]), :b)
        )

      expected =
        Nx.tensor([[10, 20, 30], [0, 0, 0]])
        |> Nx.vectorize(:a)
        |> Nx.vectorize(:b)

      assert_equal(result, expected)
    end

    # 6. All-true / all-false predicate vectors (every element hits same branch)
    test "hooked cond with all-true predicate vector" do
      result =
        hooked_cond_different_axes(
          Nx.vectorize(~VEC[1 1 1], :a),
          Nx.vectorize(~VEC[0 1], :b)
        )

      # All a values are true → always branch 1, regardless of b
      expected =
        Nx.tensor([[1, 1], [1, 1], [1, 1]])
        |> Nx.vectorize(:a)
        |> Nx.vectorize(:b)

      assert_equal(result, expected)
    end

    test "hooked cond with all-false predicates (always default)" do
      result =
        hooked_cond_different_axes(
          Nx.vectorize(~VEC[0 0], :a),
          Nx.vectorize(~VEC[0 0 0], :b)
        )

      # All predicates false → always 0
      expected =
        Nx.tensor([[0, 0, 0], [0, 0, 0]])
        |> Nx.vectorize(:a)
        |> Nx.vectorize(:b)

      assert_equal(result, expected)
    end

    # 7. While loop containing vectorized cond with hooks
    defn while_with_vectorized_cond(pred, n) do
      {_, result} =
        while {i = Nx.tensor(0), acc = Nx.tensor(0)}, Nx.less(i, n) do
          val =
            cond do
              pred -> send_value(1, clause: "loop_if")
              true -> send_value(2, clause: "loop_else")
            end

          {i + 1, acc + val}
        end

      result
    end

    test "while loop with vectorized cond and hooks" do
      # pred on :a: [1, 0], loop 3 times
      # a=0: pred=1, accumulates 1*3 = 3
      # a=1: pred=0, accumulates 2*3 = 6
      result =
        while_with_vectorized_cond(
          Nx.vectorize(~VEC[1 0], :a),
          3
        )

      assert_equal(result, Nx.vectorize(Nx.tensor([3, 6]), :a))
    end

    # 8. Nested cond with hooks on different axes
    defn nested_cond_hooked(p_outer, p_inner) do
      cond do
        p_outer ->
          cond do
            p_inner -> send_value(1, clause: "outer_true_inner_true")
            true -> send_value(2, clause: "outer_true_inner_false")
          end

        true ->
          send_value(3, clause: "outer_false")
      end
    end

    test "nested cond with hooks on different axes" do
      # p_outer on :a: [1, 0, 1]
      # p_inner on :b: [0, 1]
      #
      # Result is vectorized[a: 3][b: 2]
      # a=0 (outer=1): b=0 (inner=0) → 2, b=1 (inner=1) → 1
      # a=1 (outer=0): → 3 for all b
      # a=2 (outer=1): b=0 (inner=0) → 2, b=1 (inner=1) → 1
      result =
        nested_cond_hooked(
          Nx.vectorize(~VEC[1 0 1], :a),
          Nx.vectorize(~VEC[0 1], :b)
        )

      expected =
        Nx.tensor([[2, 1], [3, 3], [2, 1]])
        |> Nx.vectorize(:a)
        |> Nx.vectorize(:b)

      assert_equal(result, expected)
    end
  end
end
