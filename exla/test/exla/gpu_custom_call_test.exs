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
end
