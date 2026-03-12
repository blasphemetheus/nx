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
  end
end
