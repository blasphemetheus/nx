defmodule EXLA.GPUCustomCallTest do
  use ExUnit.Case, async: false

  alias EXLA.BinaryBuffer
  alias EXLA.DeviceBuffer
  alias EXLA.Typespec
  alias EXLA.MLIR.Value
  import EXLAHelpers

  @moduletag platform: :cuda

  defp f32_vec_typespec(n), do: Typespec.tensor({:f, 32}, {n})
  defp f32_3d_typespec(b, t, h), do: Typespec.tensor({:f, 32}, {b, t, h})
  defp f32_2d_typespec(b, h), do: Typespec.tensor({:f, 32}, {b, h})
  defp f32_4d_typespec(b, t, h, d), do: Typespec.tensor({:f, 32}, {b, t, h, d})

  describe "gpu_add CUDA custom call" do
    test "adds two f32 vectors via CUDA kernel" do
      ts = f32_vec_typespec(4)

      t1 = BinaryBuffer.from_binary(Nx.to_binary(Nx.tensor([1.0, 2.0, 3.0, 4.0], type: :f32)), ts)
      t2 = BinaryBuffer.from_binary(Nx.to_binary(Nx.tensor([10.0, 20.0, 30.0, 40.0], type: :f32)), ts)

      assert [result = %DeviceBuffer{}] =
               run_one([t1, t2], [], [ts], fn _builder, a, b ->
                 [Value.gpu_add(a, b, ts)]
               end)

      result_binary = DeviceBuffer.read(result)
      result_tensor = Nx.from_binary(result_binary, :f32)

      expected = Nx.tensor([11.0, 22.0, 33.0, 44.0])
      assert Nx.all_close(result_tensor, expected, atol: 1.0e-6) == Nx.tensor(1, type: :u8)
    end

    test "adds two f32 matrices via CUDA kernel" do
      ts = Typespec.tensor({:f, 32}, {2, 3})

      a_data = Nx.to_binary(Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], type: :f32))
      b_data = Nx.to_binary(Nx.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], type: :f32))

      t1 = BinaryBuffer.from_binary(a_data, ts)
      t2 = BinaryBuffer.from_binary(b_data, ts)

      assert [result = %DeviceBuffer{}] =
               run_one([t1, t2], [], [ts], fn _builder, a, b ->
                 [Value.gpu_add(a, b, ts)]
               end)

      result_tensor = Nx.from_binary(DeviceBuffer.read(result), :f32) |> Nx.reshape({2, 3})
      expected = Nx.tensor([[11.0, 22.0, 33.0], [44.0, 55.0, 66.0]])
      assert Nx.all_close(result_tensor, expected, atol: 1.0e-6) == Nx.tensor(1, type: :u8)
    end
  end

  describe "fused_mingru_scan CUDA custom call" do
    test "scans a single-step sequence (reduces to one MinGRU update)" do
      # batch=1, seq_len=1, hidden=4
      # h_new = (1 - z) * h0 + z * candidate
      # z=0.5: h_new = 0.5 * [1,1,1,1] + 0.5 * [2,4,6,8] = [1.5, 2.5, 3.5, 4.5]
      out_ts = f32_3d_typespec(1, 1, 4)
      gates_ts = f32_3d_typespec(1, 1, 4)
      cand_ts = f32_3d_typespec(1, 1, 4)
      h0_ts = f32_2d_typespec(1, 4)

      gates = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[[0.5, 0.5, 0.5, 0.5]]], type: :f32)), gates_ts)
      candidates = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[[2.0, 4.0, 6.0, 8.0]]], type: :f32)), cand_ts)
      h0 = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[1.0, 1.0, 1.0, 1.0]], type: :f32)), h0_ts)

      assert [result = %DeviceBuffer{}] =
               run_one([gates, candidates, h0], [], [out_ts], fn _builder, g, c, h ->
                 [Value.fused_mingru_scan(g, c, h, out_ts)]
               end)

      result_tensor = Nx.from_binary(DeviceBuffer.read(result), :f32)
      expected = Nx.tensor([1.5, 2.5, 3.5, 4.5])
      assert Nx.all_close(result_tensor, expected, atol: 1.0e-5) == Nx.tensor(1, type: :u8)
    end

    test "scans a multi-step sequence" do
      # batch=1, seq_len=3, hidden=2
      # h0 = [0, 0]
      # Step 0: z=[1,1], cand=[1,2] → h = (1-1)*0 + 1*[1,2] = [1, 2]
      # Step 1: z=[0,0], cand=[9,9] → h = (1-0)*[1,2] + 0*[9,9] = [1, 2]
      # Step 2: z=[0.5,0.5], cand=[3,4] → h = 0.5*[1,2] + 0.5*[3,4] = [2, 3]
      out_ts = f32_3d_typespec(1, 3, 2)
      gates_ts = f32_3d_typespec(1, 3, 2)
      cand_ts = f32_3d_typespec(1, 3, 2)
      h0_ts = f32_2d_typespec(1, 2)

      gates = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[[1.0, 1.0], [0.0, 0.0], [0.5, 0.5]]], type: :f32)), gates_ts)
      candidates = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[[1.0, 2.0], [9.0, 9.0], [3.0, 4.0]]], type: :f32)), cand_ts)
      h0 = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[0.0, 0.0]], type: :f32)), h0_ts)

      assert [result = %DeviceBuffer{}] =
               run_one([gates, candidates, h0], [], [out_ts], fn _builder, g, c, h ->
                 [Value.fused_mingru_scan(g, c, h, out_ts)]
               end)

      result_tensor = Nx.from_binary(DeviceBuffer.read(result), :f32) |> Nx.reshape({1, 3, 2})
      expected = Nx.tensor([[[1.0, 2.0], [1.0, 2.0], [2.0, 3.0]]])
      assert Nx.all_close(result_tensor, expected, atol: 1.0e-5) == Nx.tensor(1, type: :u8)
    end

    test "handles multiple batches" do
      # batch=2, seq_len=2, hidden=2
      out_ts = f32_3d_typespec(2, 2, 2)
      gates_ts = f32_3d_typespec(2, 2, 2)
      cand_ts = f32_3d_typespec(2, 2, 2)
      h0_ts = f32_2d_typespec(2, 2)

      # Batch 0: z=1 everywhere, so h = candidate at each step
      # Batch 1: z=0 everywhere, so h = h0 at each step
      gates = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([
          [[1.0, 1.0], [1.0, 1.0]],
          [[0.0, 0.0], [0.0, 0.0]]
        ], type: :f32)), gates_ts)
      candidates = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([
          [[5.0, 6.0], [7.0, 8.0]],
          [[99.0, 99.0], [99.0, 99.0]]
        ], type: :f32)), cand_ts)
      h0 = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[0.0, 0.0], [1.0, 2.0]], type: :f32)), h0_ts)

      assert [result = %DeviceBuffer{}] =
               run_one([gates, candidates, h0], [], [out_ts], fn _builder, g, c, h ->
                 [Value.fused_mingru_scan(g, c, h, out_ts)]
               end)

      result_tensor = Nx.from_binary(DeviceBuffer.read(result), :f32) |> Nx.reshape({2, 2, 2})
      expected = Nx.tensor([
        [[5.0, 6.0], [7.0, 8.0]],
        [[1.0, 2.0], [1.0, 2.0]]
      ])
      assert Nx.all_close(result_tensor, expected, atol: 1.0e-5) == Nx.tensor(1, type: :u8)
    end
  end

  describe "fused_minlstm_scan CUDA custom call" do
    test "scans a single-step sequence" do
      # batch=1, seq_len=1, hidden=4
      # f=0.6, i=0.4, eps=1e-6 → f'≈0.6, i'≈0.4
      # c_new = f' * h0 + i' * cand ≈ 0.6*[1,1,1,1] + 0.4*[10,20,30,40]
      #       ≈ [4.6, 8.6, 12.6, 16.6]
      out_ts = f32_3d_typespec(1, 1, 4)
      fg_ts = f32_3d_typespec(1, 1, 4)
      ig_ts = f32_3d_typespec(1, 1, 4)
      cand_ts = f32_3d_typespec(1, 1, 4)
      h0_ts = f32_2d_typespec(1, 4)

      forget_gates = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[[0.6, 0.6, 0.6, 0.6]]], type: :f32)), fg_ts)
      input_gates = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[[0.4, 0.4, 0.4, 0.4]]], type: :f32)), ig_ts)
      candidates = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[[10.0, 20.0, 30.0, 40.0]]], type: :f32)), cand_ts)
      h0 = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[1.0, 1.0, 1.0, 1.0]], type: :f32)), h0_ts)

      assert [result = %DeviceBuffer{}] =
               run_one([forget_gates, input_gates, candidates, h0], [], [out_ts],
                 fn _builder, fg, ig, c, h ->
                   [Value.fused_minlstm_scan(fg, ig, c, h, out_ts)]
                 end)

      result_tensor = Nx.from_binary(DeviceBuffer.read(result), :f32)
      # f'=0.6/1.0=0.6, i'=0.4/1.0=0.4
      expected = Nx.tensor([4.6, 8.6, 12.6, 16.6])
      assert Nx.all_close(result_tensor, expected, atol: 1.0e-4) == Nx.tensor(1, type: :u8)
    end

    test "scans a multi-step sequence with gate normalization" do
      # batch=1, seq_len=2, hidden=2
      # h0 = [0, 0]
      # Step 0: f=0.8, i=0.2 → f'=0.8, i'=0.2, c = 0.8*0 + 0.2*[10,20] = [2, 4]
      # Step 1: f=0.5, i=0.5 → f'=0.5, i'=0.5, c = 0.5*[2,4] + 0.5*[6,8] = [4, 6]
      out_ts = f32_3d_typespec(1, 2, 2)
      fg_ts = f32_3d_typespec(1, 2, 2)
      ig_ts = f32_3d_typespec(1, 2, 2)
      cand_ts = f32_3d_typespec(1, 2, 2)
      h0_ts = f32_2d_typespec(1, 2)

      forget_gates = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[[0.8, 0.8], [0.5, 0.5]]], type: :f32)), fg_ts)
      input_gates = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[[0.2, 0.2], [0.5, 0.5]]], type: :f32)), ig_ts)
      candidates = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[[10.0, 20.0], [6.0, 8.0]]], type: :f32)), cand_ts)
      h0 = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[0.0, 0.0]], type: :f32)), h0_ts)

      assert [result = %DeviceBuffer{}] =
               run_one([forget_gates, input_gates, candidates, h0], [], [out_ts],
                 fn _builder, fg, ig, c, h ->
                   [Value.fused_minlstm_scan(fg, ig, c, h, out_ts)]
                 end)

      result_tensor = Nx.from_binary(DeviceBuffer.read(result), :f32) |> Nx.reshape({1, 2, 2})
      expected = Nx.tensor([[[2.0, 4.0], [4.0, 6.0]]])
      assert Nx.all_close(result_tensor, expected, atol: 1.0e-4) == Nx.tensor(1, type: :u8)
    end

    test "handles multiple batches" do
      # batch=2, seq_len=1, hidden=2
      # Both batches: f=0.5, i=0.5 → f'=0.5, i'=0.5
      # Batch 0: h0=[2,4], cand=[6,8] → c = 0.5*[2,4] + 0.5*[6,8] = [4, 6]
      # Batch 1: h0=[10,10], cand=[0,0] → c = 0.5*[10,10] + 0.5*[0,0] = [5, 5]
      out_ts = f32_3d_typespec(2, 1, 2)
      fg_ts = f32_3d_typespec(2, 1, 2)
      ig_ts = f32_3d_typespec(2, 1, 2)
      cand_ts = f32_3d_typespec(2, 1, 2)
      h0_ts = f32_2d_typespec(2, 2)

      forget_gates = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[[0.5, 0.5]], [[0.5, 0.5]]], type: :f32)), fg_ts)
      input_gates = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[[0.5, 0.5]], [[0.5, 0.5]]], type: :f32)), ig_ts)
      candidates = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[[6.0, 8.0]], [[0.0, 0.0]]], type: :f32)), cand_ts)
      h0 = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[2.0, 4.0], [10.0, 10.0]], type: :f32)), h0_ts)

      assert [result = %DeviceBuffer{}] =
               run_one([forget_gates, input_gates, candidates, h0], [], [out_ts],
                 fn _builder, fg, ig, c, h ->
                   [Value.fused_minlstm_scan(fg, ig, c, h, out_ts)]
                 end)

      result_tensor = Nx.from_binary(DeviceBuffer.read(result), :f32) |> Nx.reshape({2, 1, 2})
      expected = Nx.tensor([[[4.0, 6.0]], [[5.0, 5.0]]])
      assert Nx.all_close(result_tensor, expected, atol: 1.0e-4) == Nx.tensor(1, type: :u8)
    end
  end

  describe "fused_liquid_scan CUDA custom call" do
    test "exact LTC solver single step" do
      # h = activation + (h0 - activation) * exp(-1/tau)
      # tau=1.0, activation=10.0, h0=0.0
      # h = 10 + (0 - 10) * exp(-1) = 10 - 10*0.3679 = 6.321
      out_ts = f32_3d_typespec(1, 1, 2)
      tau_ts = f32_3d_typespec(1, 1, 2)
      act_ts = f32_3d_typespec(1, 1, 2)
      h0_ts = f32_2d_typespec(1, 2)

      tau = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[[1.0, 2.0]]], type: :f32)), tau_ts)
      activation = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[[10.0, 10.0]]], type: :f32)), act_ts)
      h0 = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[0.0, 0.0]], type: :f32)), h0_ts)

      assert [result = %DeviceBuffer{}] =
               run_one([tau, activation, h0], [], [out_ts], fn _builder, t, a, h ->
                 [Value.fused_liquid_scan(t, a, h, out_ts)]
               end)

      result_tensor = Nx.from_binary(DeviceBuffer.read(result), :f32)
      # tau=1: 10 + (0-10)*exp(-1) = 10 - 3.6788 = 6.3212
      # tau=2: 10 + (0-10)*exp(-0.5) = 10 - 6.0653 = 3.9347
      expected = Nx.tensor([6.3212, 3.9347])
      assert Nx.all_close(result_tensor, expected, atol: 1.0e-3) == Nx.tensor(1, type: :u8)
    end

    test "multi-step convergence" do
      # With constant activation=5.0, tau=0.5, h0=0.0
      # Each step: h = 5 + (h - 5)*exp(-2) = 5 + (h-5)*0.1353
      # Step 0: 5 + (0-5)*0.1353 = 5 - 0.6767 = 4.3233
      # Step 1: 5 + (4.3233-5)*0.1353 = 5 - 0.0916 = 4.9084
      out_ts = f32_3d_typespec(1, 2, 1)
      tau_ts = f32_3d_typespec(1, 2, 1)
      act_ts = f32_3d_typespec(1, 2, 1)
      h0_ts = f32_2d_typespec(1, 1)

      tau = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[[0.5], [0.5]]], type: :f32)), tau_ts)
      activation = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[[5.0], [5.0]]], type: :f32)), act_ts)
      h0 = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[0.0]], type: :f32)), h0_ts)

      assert [result = %DeviceBuffer{}] =
               run_one([tau, activation, h0], [], [out_ts], fn _builder, t, a, h ->
                 [Value.fused_liquid_scan(t, a, h, out_ts)]
               end)

      result_tensor = Nx.from_binary(DeviceBuffer.read(result), :f32) |> Nx.reshape({1, 2, 1})
      expected = Nx.tensor([[[4.3233], [4.9084]]])
      assert Nx.all_close(result_tensor, expected, atol: 1.0e-3) == Nx.tensor(1, type: :u8)
    end
  end

  describe "fused_elu_gru_scan CUDA custom call" do
    test "single step ELU-GRU update" do
      # z = sigmoid(gate), c = 1 + elu(cand), h = (1-z)*h0 + z*c
      # gate=0 → z=0.5, cand=0 → c=1+elu(0)=1+0=1
      # h = 0.5*h0 + 0.5*1
      # h0=[2,4] → h = [0.5*2+0.5, 0.5*4+0.5] = [1.5, 2.5]
      out_ts = f32_3d_typespec(1, 1, 2)
      gates_ts = f32_3d_typespec(1, 1, 2)
      cand_ts = f32_3d_typespec(1, 1, 2)
      h0_ts = f32_2d_typespec(1, 2)

      gates = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[[0.0, 0.0]]], type: :f32)), gates_ts)
      candidates = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[[0.0, 0.0]]], type: :f32)), cand_ts)
      h0 = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[2.0, 4.0]], type: :f32)), h0_ts)

      assert [result = %DeviceBuffer{}] =
               run_one([gates, candidates, h0], [], [out_ts], fn _builder, g, c, h ->
                 [Value.fused_elu_gru_scan(g, c, h, out_ts)]
               end)

      result_tensor = Nx.from_binary(DeviceBuffer.read(result), :f32)
      expected = Nx.tensor([1.5, 2.5])
      assert Nx.all_close(result_tensor, expected, atol: 1.0e-4) == Nx.tensor(1, type: :u8)
    end

    test "multi-step with positive candidates (elu passthrough)" do
      # gate=large → z≈1 → h ≈ 1+elu(cand) = 1+cand (for cand>0)
      # gate=100 → z≈1.0
      # Step 0: cand=2 → c=3, h≈3
      # Step 1: cand=5 → c=6, h≈6
      out_ts = f32_3d_typespec(1, 2, 1)
      gates_ts = f32_3d_typespec(1, 2, 1)
      cand_ts = f32_3d_typespec(1, 2, 1)
      h0_ts = f32_2d_typespec(1, 1)

      gates = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[[100.0], [100.0]]], type: :f32)), gates_ts)
      candidates = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[[2.0], [5.0]]], type: :f32)), cand_ts)
      h0 = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[0.0]], type: :f32)), h0_ts)

      assert [result = %DeviceBuffer{}] =
               run_one([gates, candidates, h0], [], [out_ts], fn _builder, g, c, h ->
                 [Value.fused_elu_gru_scan(g, c, h, out_ts)]
               end)

      result_tensor = Nx.from_binary(DeviceBuffer.read(result), :f32) |> Nx.reshape({1, 2, 1})
      expected = Nx.tensor([[[3.0], [6.0]]])
      assert Nx.all_close(result_tensor, expected, atol: 1.0e-3) == Nx.tensor(1, type: :u8)
    end
  end

  describe "fused_real_gru_scan CUDA custom call" do
    test "sigmoid applied in-kernel to raw gates" do
      # gate=0 → z=sigmoid(0)=0.5, h = 0.5*h0 + 0.5*cand
      # h0=[10], cand=[20] → h = 0.5*10 + 0.5*20 = 15
      out_ts = f32_3d_typespec(1, 1, 1)
      gates_ts = f32_3d_typespec(1, 1, 1)
      cand_ts = f32_3d_typespec(1, 1, 1)
      h0_ts = f32_2d_typespec(1, 1)

      gates = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[[0.0]]], type: :f32)), gates_ts)
      candidates = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[[20.0]]], type: :f32)), cand_ts)
      h0 = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[10.0]], type: :f32)), h0_ts)

      assert [result = %DeviceBuffer{}] =
               run_one([gates, candidates, h0], [], [out_ts], fn _builder, g, c, h ->
                 [Value.fused_real_gru_scan(g, c, h, out_ts)]
               end)

      result_tensor = Nx.from_binary(DeviceBuffer.read(result), :f32)
      expected = Nx.tensor([15.0])
      assert Nx.all_close(result_tensor, expected, atol: 1.0e-4) == Nx.tensor(1, type: :u8)
    end

    test "large positive gate → full replacement" do
      # gate=100 → z≈1 → h ≈ candidate
      out_ts = f32_3d_typespec(1, 1, 2)
      gates_ts = f32_3d_typespec(1, 1, 2)
      cand_ts = f32_3d_typespec(1, 1, 2)
      h0_ts = f32_2d_typespec(1, 2)

      gates = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[[100.0, 100.0]]], type: :f32)), gates_ts)
      candidates = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[[7.0, 8.0]]], type: :f32)), cand_ts)
      h0 = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[999.0, 999.0]], type: :f32)), h0_ts)

      assert [result = %DeviceBuffer{}] =
               run_one([gates, candidates, h0], [], [out_ts], fn _builder, g, c, h ->
                 [Value.fused_real_gru_scan(g, c, h, out_ts)]
               end)

      result_tensor = Nx.from_binary(DeviceBuffer.read(result), :f32)
      expected = Nx.tensor([7.0, 8.0])
      assert Nx.all_close(result_tensor, expected, atol: 1.0e-3) == Nx.tensor(1, type: :u8)
    end
  end

  describe "fused_diag_linear_scan CUDA custom call" do
    test "h = sigmoid(a)*h + b with sigmoid in kernel" do
      # a=0 → sigmoid(0)=0.5, b=1 → h = 0.5*h0 + 1
      # h0=[0,0] → h = [1, 1]
      out_ts = f32_3d_typespec(1, 1, 2)
      a_ts = f32_3d_typespec(1, 1, 2)
      b_ts = f32_3d_typespec(1, 1, 2)
      h0_ts = f32_2d_typespec(1, 2)

      a_vals = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[[0.0, 0.0]]], type: :f32)), a_ts)
      b_vals = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[[1.0, 1.0]]], type: :f32)), b_ts)
      h0 = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[0.0, 0.0]], type: :f32)), h0_ts)

      assert [result = %DeviceBuffer{}] =
               run_one([a_vals, b_vals, h0], [], [out_ts], fn _builder, a, b, h ->
                 [Value.fused_diag_linear_scan(a, b, h, out_ts)]
               end)

      result_tensor = Nx.from_binary(DeviceBuffer.read(result), :f32)
      expected = Nx.tensor([1.0, 1.0])
      assert Nx.all_close(result_tensor, expected, atol: 1.0e-4) == Nx.tensor(1, type: :u8)
    end

    test "multi-step accumulation" do
      # a=large → sigmoid≈1, b=[1] at each step → h accumulates
      # Step 0: h = 1*0 + 1 = 1
      # Step 1: h = 1*1 + 1 = 2
      # Step 2: h = 1*2 + 1 = 3
      out_ts = f32_3d_typespec(1, 3, 1)
      a_ts = f32_3d_typespec(1, 3, 1)
      b_ts = f32_3d_typespec(1, 3, 1)
      h0_ts = f32_2d_typespec(1, 1)

      a_vals = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[[100.0], [100.0], [100.0]]], type: :f32)), a_ts)
      b_vals = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[[1.0], [1.0], [1.0]]], type: :f32)), b_ts)
      h0 = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[0.0]], type: :f32)), h0_ts)

      assert [result = %DeviceBuffer{}] =
               run_one([a_vals, b_vals, h0], [], [out_ts], fn _builder, a, b, h ->
                 [Value.fused_diag_linear_scan(a, b, h, out_ts)]
               end)

      result_tensor = Nx.from_binary(DeviceBuffer.read(result), :f32) |> Nx.reshape({1, 3, 1})
      expected = Nx.tensor([[[1.0], [2.0], [3.0]]])
      assert Nx.all_close(result_tensor, expected, atol: 1.0e-3) == Nx.tensor(1, type: :u8)
    end
  end

  describe "fused_linear_scan CUDA custom call" do
    test "h = a*h + b (no nonlinearities)" do
      # a=0.5, b=1, h0=0 → Step 0: h=0.5*0+1=1, Step 1: h=0.5*1+1=1.5
      out_ts = f32_3d_typespec(1, 2, 1)
      a_ts = f32_3d_typespec(1, 2, 1)
      b_ts = f32_3d_typespec(1, 2, 1)
      h0_ts = f32_2d_typespec(1, 1)

      a_vals = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[[0.5], [0.5]]], type: :f32)), a_ts)
      b_vals = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[[1.0], [1.0]]], type: :f32)), b_ts)
      h0 = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[0.0]], type: :f32)), h0_ts)

      assert [result = %DeviceBuffer{}] =
               run_one([a_vals, b_vals, h0], [], [out_ts], fn _builder, a, b, h ->
                 [Value.fused_linear_scan(a, b, h, out_ts)]
               end)

      result_tensor = Nx.from_binary(DeviceBuffer.read(result), :f32) |> Nx.reshape({1, 2, 1})
      expected = Nx.tensor([[[1.0], [1.5]]])
      assert Nx.all_close(result_tensor, expected, atol: 1.0e-5) == Nx.tensor(1, type: :u8)
    end

    test "exponential decay (a=0.9, b=0)" do
      # h0=100, a=0.9 each step → 90, 81, 72.9
      out_ts = f32_3d_typespec(1, 3, 1)
      a_ts = f32_3d_typespec(1, 3, 1)
      b_ts = f32_3d_typespec(1, 3, 1)
      h0_ts = f32_2d_typespec(1, 1)

      a_vals = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[[0.9], [0.9], [0.9]]], type: :f32)), a_ts)
      b_vals = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[[0.0], [0.0], [0.0]]], type: :f32)), b_ts)
      h0 = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[100.0]], type: :f32)), h0_ts)

      assert [result = %DeviceBuffer{}] =
               run_one([a_vals, b_vals, h0], [], [out_ts], fn _builder, a, b, h ->
                 [Value.fused_linear_scan(a, b, h, out_ts)]
               end)

      result_tensor = Nx.from_binary(DeviceBuffer.read(result), :f32) |> Nx.reshape({1, 3, 1})
      expected = Nx.tensor([[[90.0], [81.0], [72.9]]])
      assert Nx.all_close(result_tensor, expected, atol: 1.0e-3) == Nx.tensor(1, type: :u8)
    end
  end

  describe "fused_delta_net_scan CUDA custom call" do
    test "identity query retrieves stored value" do
      # batch=1, seq_len=1, heads=1, head_dim=2
      # S starts at 0. After delta update with beta=1:
      #   retrieval = S@k = 0
      #   error = v - 0 = v
      #   S = 0 + 1 * outer(v, k) = outer(v, k)
      #   output = S @ q
      # k=[1,0], v=[3,5], q=[1,0] → S=[[3,0],[5,0]], out=S@[1,0]=[3,5]
      out_ts = f32_4d_typespec(1, 1, 1, 2)
      qkvb_ts = f32_4d_typespec(1, 1, 1, 2)

      q = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[[[1.0, 0.0]]]], type: :f32)), qkvb_ts)
      k = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[[[1.0, 0.0]]]], type: :f32)), qkvb_ts)
      v = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[[[3.0, 5.0]]]], type: :f32)), qkvb_ts)
      beta = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[[[1.0, 1.0]]]], type: :f32)), qkvb_ts)

      assert [result = %DeviceBuffer{}] =
               run_one([q, k, v, beta], [], [out_ts], fn _builder, q_, k_, v_, b_ ->
                 [Value.fused_delta_net_scan(q_, k_, v_, b_, out_ts)]
               end)

      result_tensor = Nx.from_binary(DeviceBuffer.read(result), :f32)
      expected = Nx.tensor([3.0, 5.0])
      assert Nx.all_close(result_tensor, expected, atol: 1.0e-4) == Nx.tensor(1, type: :u8)
    end

    test "two-step associative memory" do
      # Step 0: Store v=[1,0] at k=[1,0] → S=[[1,0],[0,0]]
      # Step 1: Store v=[0,1] at k=[0,1], query q=[1,0]
      #   retrieval = S@[0,1] = [0,0], error=[0,1], S += outer([0,1],[0,1])
      #   S = [[1,0],[0,1]], out = S@[1,0] = [1,0]
      out_ts = f32_4d_typespec(1, 2, 1, 2)
      qkvb_ts = f32_4d_typespec(1, 2, 1, 2)

      q = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[[[1.0, 0.0]], [[1.0, 0.0]]]], type: :f32)), qkvb_ts)
      k = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]], type: :f32)), qkvb_ts)
      v = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]], type: :f32)), qkvb_ts)
      beta = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[[[1.0, 1.0]], [[1.0, 1.0]]]], type: :f32)), qkvb_ts)

      assert [result = %DeviceBuffer{}] =
               run_one([q, k, v, beta], [], [out_ts], fn _builder, q_, k_, v_, b_ ->
                 [Value.fused_delta_net_scan(q_, k_, v_, b_, out_ts)]
               end)

      result_tensor = Nx.from_binary(DeviceBuffer.read(result), :f32) |> Nx.reshape({1, 2, 1, 2})
      # Step 0: S=[[1,0],[0,0]], out=S@[1,0]=[1,0]
      # Step 1: S=[[1,0],[0,1]], out=S@[1,0]=[1,0]
      expected = Nx.tensor([[[[1.0, 0.0]], [[1.0, 0.0]]]])
      assert Nx.all_close(result_tensor, expected, atol: 1.0e-4) == Nx.tensor(1, type: :u8)
    end
  end

  describe "fused_gated_delta_net_scan CUDA custom call" do
    test "alpha decay erases state" do
      # Step 0: Store v=[10,20] at k=[1,0], beta=1
      #   S = [[10,0],[20,0]], out = S@q=[10,20] (q=[1,0])
      # Step 1: alpha=0 → S decays to 0, then store v=[1,1] at k=[0,1]
      #   S_decayed = 0, retrieval=0, error=[1,1], S=outer([1,1],[0,1])=[[0,1],[0,1]]
      #   out = S@[0,1] = [1,1]
      out_ts = f32_4d_typespec(1, 2, 1, 2)
      qkvb_ts = f32_4d_typespec(1, 2, 1, 2)
      alpha_ts = Typespec.tensor({:f, 32}, {1, 2, 1})

      q = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]], type: :f32)), qkvb_ts)
      k = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]], type: :f32)), qkvb_ts)
      v = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[[[10.0, 20.0]], [[1.0, 1.0]]]], type: :f32)), qkvb_ts)
      beta = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[[[1.0, 1.0]], [[1.0, 1.0]]]], type: :f32)), qkvb_ts)
      alpha = BinaryBuffer.from_binary(
        Nx.to_binary(Nx.tensor([[[1.0], [0.0]]], type: :f32)), alpha_ts)

      assert [result = %DeviceBuffer{}] =
               run_one([q, k, v, beta, alpha], [], [out_ts], fn _builder, q_, k_, v_, b_, a_ ->
                 [Value.fused_gated_delta_net_scan(q_, k_, v_, b_, a_, out_ts)]
               end)

      result_tensor = Nx.from_binary(DeviceBuffer.read(result), :f32) |> Nx.reshape({1, 2, 1, 2})
      # Step 0: alpha=1 (no decay), S=[[10,0],[20,0]], out=[10,20]
      # Step 1: alpha=0 (full decay), S_decayed=0, new S=[[0,1],[0,1]], out=[1,1]
      expected = Nx.tensor([[[[10.0, 20.0]], [[1.0, 1.0]]]])
      assert Nx.all_close(result_tensor, expected, atol: 1.0e-3) == Nx.tensor(1, type: :u8)
    end
  end
end
