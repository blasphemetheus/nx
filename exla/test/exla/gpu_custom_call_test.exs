defmodule EXLA.GPUCustomCallTest do
  use ExUnit.Case, async: false

  alias EXLA.BinaryBuffer
  alias EXLA.DeviceBuffer
  alias EXLA.Typespec
  alias EXLA.MLIR.Value
  import EXLAHelpers

  @moduletag platform: :cuda

  defp f32_vec_typespec(n), do: Typespec.tensor({:f, 32}, {n})

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
end
