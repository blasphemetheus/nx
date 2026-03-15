defmodule Nx.Defn.CheckpointTest do
  use ExUnit.Case, async: true

  import Nx.Defn

  # --- Forward pass: checkpoint is a no-op ---

  describe "forward pass (outside grad)" do
    defn checkpoint_identity(x) do
      Nx.Defn.checkpoint(x, fn x -> x end)
    end

    test "returns same result as calling the function directly" do
      x = Nx.tensor([1.0, 2.0, 3.0])
      assert checkpoint_identity(x) == x
    end

    defn checkpoint_computation(x) do
      Nx.Defn.checkpoint(x, fn x -> Nx.sin(Nx.add(x, 1.0)) end)
    end

    defn no_checkpoint_computation(x) do
      Nx.sin(Nx.add(x, 1.0))
    end

    test "produces identical result to non-checkpointed computation" do
      x = Nx.tensor([0.5, 1.0, 1.5])
      assert checkpoint_computation(x) == no_checkpoint_computation(x)
    end

    defn checkpoint_chain(x) do
      x
      |> Nx.Defn.checkpoint(fn x -> Nx.multiply(x, 2.0) end)
      |> Nx.Defn.checkpoint(fn x -> Nx.add(x, 1.0) end)
      |> Nx.Defn.checkpoint(fn x -> Nx.pow(x, 2) end)
    end

    defn no_checkpoint_chain(x) do
      x |> Nx.multiply(2.0) |> Nx.add(1.0) |> Nx.pow(2)
    end

    test "chained checkpoints produce identical result" do
      x = Nx.tensor(3.0)
      assert checkpoint_chain(x) == no_checkpoint_chain(x)
    end
  end

  # --- Gradient correctness: checkpoint produces same gradients ---

  describe "gradient correctness" do
    defn grad_with_checkpoint(x) do
      grad(x, fn x ->
        Nx.Defn.checkpoint(x, fn x -> Nx.sin(x) end)
      end)
    end

    defn grad_without_checkpoint(x) do
      grad(x, fn x -> Nx.sin(x) end)
    end

    test "simple elementwise: sin" do
      x = Nx.tensor([0.5, 1.0, 1.5])
      assert grad_with_checkpoint(x) == grad_without_checkpoint(x)
    end

    defn grad_checkpoint_multiply(x) do
      grad(x, fn x ->
        Nx.Defn.checkpoint(x, fn x -> Nx.sum(Nx.multiply(x, x)) end)
      end)
    end

    defn grad_no_checkpoint_multiply(x) do
      grad(x, fn x -> Nx.sum(Nx.multiply(x, x)) end)
    end

    test "reduction: sum of squares" do
      x = Nx.tensor([1.0, 2.0, 3.0])
      assert grad_checkpoint_multiply(x) == grad_no_checkpoint_multiply(x)
    end

    defn grad_checkpoint_composed(x) do
      grad(x, fn x ->
        x
        |> Nx.Defn.checkpoint(fn x -> Nx.tanh(x) end)
        |> Nx.sum()
      end)
    end

    defn grad_no_checkpoint_composed(x) do
      grad(x, fn x -> x |> Nx.tanh() |> Nx.sum() end)
    end

    test "composed: tanh then sum" do
      x = Nx.tensor([0.5, 1.0, 2.0])
      assert grad_checkpoint_composed(x) == grad_no_checkpoint_composed(x)
    end
  end

  # --- Multi-layer chain (the primary use case) ---

  describe "multi-layer checkpoint" do
    defn dense_block(x, w) do
      Nx.dot(x, w) |> Nx.max(0)
    end

    defn grad_checkpointed_layers(w1, w2, x) do
      grad(x, fn x ->
        x
        |> Nx.Defn.checkpoint(fn x -> dense_block(x, w1) end)
        |> Nx.Defn.checkpoint(fn x -> dense_block(x, w2) end)
        |> Nx.sum()
      end)
    end

    defn grad_plain_layers(w1, w2, x) do
      grad(x, fn x ->
        x
        |> dense_block(w1)
        |> dense_block(w2)
        |> Nx.sum()
      end)
    end

    test "dense layers produce same gradient with and without checkpoint" do
      w1 = Nx.tensor([[0.5, -0.3], [0.2, 0.8]])
      w2 = Nx.tensor([[0.1, 0.4], [-0.2, 0.3]])
      x = Nx.tensor([1.0, 2.0])

      assert grad_checkpointed_layers(w1, w2, x) == grad_plain_layers(w1, w2, x)
    end
  end

  # --- Nested checkpoints ---

  describe "nested checkpoints" do
    defn grad_nested_checkpoint(x) do
      grad(x, fn x ->
        Nx.Defn.checkpoint(x, fn x ->
          y = Nx.multiply(x, x)

          Nx.Defn.checkpoint(y, fn y ->
            Nx.sum(Nx.sin(y))
          end)
        end)
      end)
    end

    defn grad_no_nested(x) do
      grad(x, fn x -> Nx.sum(Nx.sin(Nx.multiply(x, x))) end)
    end

    test "nested checkpoints produce correct gradient" do
      x = Nx.tensor([1.0, 2.0, 3.0])
      assert grad_nested_checkpoint(x) == grad_no_nested(x)
    end
  end

  # --- Interaction with control flow ---

  describe "interaction with cond" do
    defn grad_checkpoint_with_cond(x) do
      grad(x, fn x ->
        Nx.Defn.checkpoint(x, fn x ->
          if Nx.greater(Nx.sum(x), 0) do
            Nx.sum(Nx.sin(x))
          else
            Nx.sum(Nx.cos(x))
          end
        end)
      end)
    end

    defn grad_cond_no_checkpoint(x) do
      grad(x, fn x ->
        if Nx.greater(Nx.sum(x), 0) do
          Nx.sum(Nx.sin(x))
        else
          Nx.sum(Nx.cos(x))
        end
      end)
    end

    test "checkpoint with cond (true branch)" do
      x = Nx.tensor([1.0, 2.0, 3.0])
      assert grad_checkpoint_with_cond(x) == grad_cond_no_checkpoint(x)
    end

    test "checkpoint with cond (false branch)" do
      x = Nx.tensor([-1.0, -2.0, -3.0])
      assert grad_checkpoint_with_cond(x) == grad_cond_no_checkpoint(x)
    end
  end

  describe "interaction with while" do
    defn grad_checkpoint_with_while(x) do
      grad(x, fn x ->
        Nx.Defn.checkpoint(x, fn x ->
          {_i, acc} =
            while {i = 0, acc = x}, Nx.less(i, 3) do
              {i + 1, Nx.sin(acc)}
            end

          Nx.sum(acc)
        end)
      end)
    end

    defn grad_while_no_checkpoint(x) do
      grad(x, fn x ->
        {_i, acc} =
          while {i = 0, acc = x}, Nx.less(i, 3) do
            {i + 1, Nx.sin(acc)}
          end

        Nx.sum(acc)
      end)
    end

    test "checkpoint wrapping while produces correct gradient" do
      x = Nx.tensor([0.5, 1.0])
      assert grad_checkpoint_with_while(x) == grad_while_no_checkpoint(x)
    end
  end

  # --- Interaction with custom_grad ---

  describe "interaction with custom_grad" do
    defn my_relu(x) do
      custom_grad(
        Nx.max(x, 0),
        [x],
        fn g -> [Nx.select(Nx.greater(x, 0), g, 0.0)] end
      )
    end

    defn grad_checkpoint_custom_grad(x) do
      grad(x, fn x ->
        Nx.Defn.checkpoint(x, fn x ->
          Nx.sum(my_relu(x))
        end)
      end)
    end

    defn grad_custom_grad_no_checkpoint(x) do
      grad(x, fn x -> Nx.sum(my_relu(x)) end)
    end

    test "checkpoint with custom_grad produces correct gradient" do
      x = Nx.tensor([-1.0, 0.5, 2.0])
      assert grad_checkpoint_custom_grad(x) == grad_custom_grad_no_checkpoint(x)
    end
  end

  # --- Container inputs/outputs ---

  describe "container support" do
    defn grad_checkpoint_tuple_output(x) do
      grad(x, fn x ->
        {a, b} =
          Nx.Defn.checkpoint(x, fn x ->
            {Nx.sin(x), Nx.cos(x)}
          end)

        Nx.sum(Nx.add(a, b))
      end)
    end

    defn grad_tuple_no_checkpoint(x) do
      grad(x, fn x ->
        Nx.sum(Nx.add(Nx.sin(x), Nx.cos(x)))
      end)
    end

    test "checkpoint returning tuple produces correct gradient" do
      x = Nx.tensor([1.0, 2.0, 3.0])
      assert grad_checkpoint_tuple_output(x) == grad_tuple_no_checkpoint(x)
    end
  end

  # --- Value and grad ---

  describe "value_and_grad with checkpoint" do
    defn vag_with_checkpoint(x) do
      value_and_grad(x, fn x ->
        x
        |> Nx.Defn.checkpoint(fn x -> Nx.sin(x) end)
        |> Nx.sum()
      end)
    end

    defn vag_without_checkpoint(x) do
      value_and_grad(x, fn x -> x |> Nx.sin() |> Nx.sum() end)
    end

    test "value_and_grad produces same value and gradient" do
      x = Nx.tensor([1.0, 2.0, 3.0])
      {val_cp, grad_cp} = vag_with_checkpoint(x)
      {val_no, grad_no} = vag_without_checkpoint(x)
      assert val_cp == val_no
      assert grad_cp == grad_no
    end
  end

  # --- Edge cases ---

  describe "edge cases" do
    defn grad_checkpoint_scalar(x) do
      grad(x, fn x ->
        Nx.Defn.checkpoint(x, fn x -> Nx.multiply(x, x) end)
      end)
    end

    test "scalar input/output" do
      x = Nx.tensor(3.0)
      assert grad_checkpoint_scalar(x) == Nx.tensor(6.0)
    end

    defn grad_checkpoint_no_grad_path(x, y) do
      grad(x, fn x ->
        # y is captured via closure, not a checkpoint input
        Nx.Defn.checkpoint(x, fn x -> Nx.sum(Nx.multiply(x, y)) end)
      end)
    end

    test "captured non-grad variable" do
      x = Nx.tensor([1.0, 2.0, 3.0])
      y = Nx.tensor([4.0, 5.0, 6.0])
      assert grad_checkpoint_no_grad_path(x, y) == y
    end

    defn grad_checkpoint_high_rank(x) do
      grad(x, fn x ->
        Nx.Defn.checkpoint(x, fn x ->
          x |> Nx.sin() |> Nx.sum()
        end)
      end)
    end

    test "high-rank tensor" do
      x = Nx.iota({2, 3, 4}, type: :f32)
      expected = Nx.Defn.grad(x, &Nx.sum(Nx.sin(&1)))
      assert grad_checkpoint_high_rank(x) == expected
    end
  end

  # --- Gradient w.r.t. weights (training use case) ---

  describe "gradient w.r.t. captured parameters" do
    defn dense_layer(x, w) do
      Nx.dot(x, w) |> Nx.max(0)
    end

    defn grad_weights_with_checkpoint(w1, w2, x) do
      grad({w1, w2}, fn {w1, w2} ->
        x
        |> Nx.Defn.checkpoint(fn x -> dense_layer(x, w1) end)
        |> Nx.Defn.checkpoint(fn x -> dense_layer(x, w2) end)
        |> Nx.sum()
      end)
    end

    defn grad_weights_no_checkpoint(w1, w2, x) do
      grad({w1, w2}, fn {w1, w2} ->
        x
        |> dense_layer(w1)
        |> dense_layer(w2)
        |> Nx.sum()
      end)
    end

    test "gradient flows to weights captured in checkpoint closures" do
      w1 = Nx.tensor([[0.5, -0.3], [0.2, 0.8]])
      w2 = Nx.tensor([[0.1, 0.4], [-0.2, 0.3]])
      x = Nx.tensor([1.0, 2.0])

      assert grad_weights_with_checkpoint(w1, w2, x) ==
               grad_weights_no_checkpoint(w1, w2, x)
    end

    defn vag_params_with_checkpoint(params, x) do
      value_and_grad(params, fn params ->
        x
        |> Nx.Defn.checkpoint(fn x -> dense_layer(x, params.w1) end)
        |> Nx.Defn.checkpoint(fn x -> dense_layer(x, params.w2) end)
        |> Nx.sum()
      end)
    end

    defn vag_params_no_checkpoint(params, x) do
      value_and_grad(params, fn params ->
        x |> dense_layer(params.w1) |> dense_layer(params.w2) |> Nx.sum()
      end)
    end

    test "value_and_grad w.r.t. map of params (training pattern)" do
      params = %{
        w1: Nx.tensor([[0.5, -0.3], [0.2, 0.8]]),
        w2: Nx.tensor([[0.1, 0.4], [-0.2, 0.3]])
      }

      x = Nx.tensor([1.0, 2.0])

      {val_cp, grad_cp} = vag_params_with_checkpoint(params, x)
      {val_no, grad_no} = vag_params_no_checkpoint(params, x)
      assert val_cp == val_no
      assert grad_cp == grad_no
    end
  end

  # --- Diamond/shared input pattern ---

  describe "shared input (diamond pattern)" do
    defn grad_diamond_checkpoint(x) do
      grad(x, fn x ->
        a = Nx.Defn.checkpoint(x, fn x -> Nx.sin(x) end)
        b = Nx.Defn.checkpoint(x, fn x -> Nx.cos(x) end)
        Nx.sum(Nx.multiply(a, b))
      end)
    end

    defn grad_diamond_no_checkpoint(x) do
      grad(x, fn x ->
        Nx.sum(Nx.multiply(Nx.sin(x), Nx.cos(x)))
      end)
    end

    test "two checkpoints sharing the same input" do
      x = Nx.tensor([1.0, 2.0, 3.0])
      assert grad_diamond_checkpoint(x) == grad_diamond_no_checkpoint(x)
    end
  end

  # --- Checkpoint in the middle of a chain ---

  describe "partial checkpointing" do
    defn grad_middle_checkpoint(x) do
      grad(x, fn x ->
        x
        |> Nx.multiply(2.0)
        |> Nx.Defn.checkpoint(fn x -> Nx.sin(Nx.exp(x)) end)
        |> Nx.sum()
      end)
    end

    defn grad_middle_no_checkpoint(x) do
      grad(x, fn x ->
        x |> Nx.multiply(2.0) |> Nx.exp() |> Nx.sin() |> Nx.sum()
      end)
    end

    test "ops before and after checkpoint boundary" do
      x = Nx.tensor([0.1, 0.2, 0.3])
      assert grad_middle_checkpoint(x) == grad_middle_no_checkpoint(x)
    end
  end

  # --- stop_grad interaction ---

  describe "interaction with stop_grad" do
    defn grad_checkpoint_with_stop_grad(x) do
      grad(x, fn x ->
        Nx.Defn.checkpoint(x, fn x ->
          Nx.sum(Nx.multiply(x, stop_grad(Nx.sin(x))))
        end)
      end)
    end

    defn grad_stop_grad_no_checkpoint(x) do
      grad(x, fn x ->
        Nx.sum(Nx.multiply(x, stop_grad(Nx.sin(x))))
      end)
    end

    test "stop_grad inside checkpoint" do
      x = Nx.tensor([1.0, 2.0, 3.0])
      assert grad_checkpoint_with_stop_grad(x) == grad_stop_grad_no_checkpoint(x)
    end
  end

  # --- Higher-order gradients ---

  describe "higher-order gradients" do
    defn grad_of_grad_checkpoint(x) do
      grad(x, fn x ->
        grad(x, fn x ->
          Nx.Defn.checkpoint(x, fn x -> Nx.sum(Nx.pow(x, 3)) end)
        end)
        |> Nx.sum()
      end)
    end

    defn grad_of_grad_no_checkpoint(x) do
      grad(x, fn x ->
        grad(x, fn x ->
          Nx.sum(Nx.pow(x, 3))
        end)
        |> Nx.sum()
      end)
    end

    test "second-order gradient through checkpoint" do
      x = Nx.tensor([1.0, 2.0, 3.0])
      assert grad_of_grad_checkpoint(x) == grad_of_grad_no_checkpoint(x)
    end
  end

  # --- Numerical precision ---

  describe "numerical precision" do
    defn grad_checkpoint_exp_log(x) do
      grad(x, fn x ->
        Nx.Defn.checkpoint(x, fn x ->
          Nx.sum(Nx.log(Nx.exp(x)))
        end)
      end)
    end

    defn grad_exp_log_no_checkpoint(x) do
      grad(x, fn x -> Nx.sum(Nx.log(Nx.exp(x))) end)
    end

    test "exp/log chain produces bitwise identical gradient" do
      x = Nx.tensor([0.1, 1.0, 5.0])
      assert grad_checkpoint_exp_log(x) == grad_exp_log_no_checkpoint(x)
    end
  end

  # --- Multiple outputs consumed independently ---

  describe "multiple outputs consumed separately" do
    defn grad_multi_output_checkpoint(x) do
      grad(x, fn x ->
        {a, b} =
          Nx.Defn.checkpoint(x, fn x ->
            {Nx.sin(x), Nx.cos(x)}
          end)

        Nx.sum(Nx.pow(a, 2)) + Nx.sum(Nx.pow(b, 3))
      end)
    end

    defn grad_multi_output_no_checkpoint(x) do
      grad(x, fn x ->
        a = Nx.sin(x)
        b = Nx.cos(x)
        Nx.sum(Nx.pow(a, 2)) + Nx.sum(Nx.pow(b, 3))
      end)
    end

    test "tuple outputs used in independent expressions" do
      x = Nx.tensor([0.5, 1.0, 1.5])
      assert grad_multi_output_checkpoint(x) == grad_multi_output_no_checkpoint(x)
    end
  end

  # --- Many sequential checkpoints (stress test) ---

  describe "many sequential checkpoints" do
    defn apply_checkpointed_sins(x) do
      x
      |> Nx.Defn.checkpoint(&Nx.sin/1)
      |> Nx.Defn.checkpoint(&Nx.sin/1)
      |> Nx.Defn.checkpoint(&Nx.sin/1)
      |> Nx.Defn.checkpoint(&Nx.sin/1)
      |> Nx.Defn.checkpoint(&Nx.sin/1)
    end

    defn apply_plain_sins(x) do
      x |> Nx.sin() |> Nx.sin() |> Nx.sin() |> Nx.sin() |> Nx.sin()
    end

    test "many sequential checkpointed layers forward" do
      x = Nx.tensor([0.5, 1.0])
      assert apply_checkpointed_sins(x) == apply_plain_sins(x)
    end

    test "gradient through many sequential checkpointed layers" do
      x = Nx.tensor([0.5, 1.0])

      grad_cp = Nx.Defn.grad(x, fn x -> Nx.sum(apply_checkpointed_sins(x)) end)
      grad_plain = Nx.Defn.grad(x, fn x -> Nx.sum(apply_plain_sins(x)) end)
      assert grad_cp == grad_plain
    end
  end

  # --- Additional edge cases ---

  describe "zero gradient through checkpoint" do
    defn grad_checkpoint_constant(x) do
      grad(x, fn x ->
        Nx.Defn.checkpoint(x, fn _x -> Nx.tensor(42.0) end)
      end)
    end

    test "checkpoint returning constant gives zero gradient" do
      x = Nx.tensor([1.0, 2.0, 3.0])
      result = grad_checkpoint_constant(x)
      assert result == Nx.broadcast(0.0, {3})
    end
  end

  describe "broadcasting inside checkpoint" do
    defn grad_checkpoint_broadcast(x) do
      grad(x, fn x ->
        Nx.Defn.checkpoint(x, fn x ->
          w = Nx.tensor([[1.0, 2.0, 3.0]])
          Nx.sum(Nx.multiply(x, w))
        end)
      end)
    end

    defn grad_broadcast_no_checkpoint(x) do
      grad(x, fn x ->
        w = Nx.tensor([[1.0, 2.0, 3.0]])
        Nx.sum(Nx.multiply(x, w))
      end)
    end

    test "broadcasting inside checkpoint" do
      x = Nx.tensor([[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]])
      assert grad_checkpoint_broadcast(x) == grad_broadcast_no_checkpoint(x)
    end
  end

  describe "dtype preservation" do
    defn grad_checkpoint_f64(x) do
      grad(x, fn x ->
        Nx.Defn.checkpoint(x, fn x -> Nx.sum(Nx.sin(x)) end)
      end)
    end

    defn grad_f64_no_checkpoint(x) do
      grad(x, fn x -> Nx.sum(Nx.sin(x)) end)
    end

    test "f64 tensors" do
      x = Nx.tensor([1.0, 2.0, 3.0], type: :f64)
      assert grad_checkpoint_f64(x) == grad_f64_no_checkpoint(x)
    end

    test "bf16 tensors" do
      x = Nx.tensor([1.0, 2.0, 3.0], type: :bf16)
      result = grad_checkpoint_f64(x)
      expected = grad_f64_no_checkpoint(x)
      assert result == expected
    end
  end

  describe "shape-changing ops inside checkpoint" do
    defn grad_checkpoint_reshape(x) do
      grad(x, fn x ->
        Nx.Defn.checkpoint(x, fn x ->
          x |> Nx.reshape({6}) |> Nx.sin() |> Nx.sum()
        end)
      end)
    end

    defn grad_reshape_no_checkpoint(x) do
      grad(x, fn x ->
        x |> Nx.reshape({6}) |> Nx.sin() |> Nx.sum()
      end)
    end

    test "reshape inside checkpoint" do
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      assert grad_checkpoint_reshape(x) == grad_reshape_no_checkpoint(x)
    end

    defn grad_checkpoint_transpose(x) do
      grad(x, fn x ->
        Nx.Defn.checkpoint(x, fn x ->
          x |> Nx.transpose() |> Nx.sin() |> Nx.sum()
        end)
      end)
    end

    defn grad_transpose_no_checkpoint(x) do
      grad(x, fn x ->
        x |> Nx.transpose() |> Nx.sin() |> Nx.sum()
      end)
    end

    test "transpose inside checkpoint" do
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      assert grad_checkpoint_transpose(x) == grad_transpose_no_checkpoint(x)
    end
  end
end
