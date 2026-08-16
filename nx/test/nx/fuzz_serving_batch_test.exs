defmodule Nx.FuzzServingBatchTest do
  @moduledoc """
  Metamorphic fuzz for `Nx.Serving` / `Nx.Batch` (FUZZ_ROADMAP T3.2).

  The contract under test: **results are invariant to batching topology**.
  However requests are stacked, split, merged, padded, or interleaved with
  other requests by the batching machinery, each caller must get exactly
  what a direct computation on their own input would produce.

  Zero fuzz coverage existed for either module.
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  import Nx.Testing

  defp double_serving do
    Nx.Serving.new(fn opts -> Nx.Defn.jit(&Nx.multiply(&1, 2), opts) end)
  end

  defp rows(count, width) do
    list_of(list_of(float(min: -100.0, max: 100.0), length: width), length: count)
  end

  describe "Nx.Batch structure" do
    property "stack size accounting and split sizes" do
      check all(count <- integer(1..8), width <- integer(1..4), max_runs: 15) do
        check all(vals <- rows(count, width), split_at <- integer(1..count), max_runs: 1) do
          tensors = Enum.map(vals, &Nx.tensor(&1, type: {:f, 32}))
          batch = Nx.Batch.stack(tensors)

          assert batch.size == count

          if split_at < count do
            {left, right} = Nx.Batch.split(batch, split_at)
            assert left.size == split_at
            assert right.size == count - split_at
          end
        end
      end
    end

    property "pad increases the padding, not the size" do
      check all(count <- integer(1..6), pad <- integer(1..4), max_runs: 15) do
        tensors = for i <- 1..count, do: Nx.tensor([i * 1.0])
        batch = Nx.Batch.stack(tensors) |> Nx.Batch.pad(pad)

        assert batch.size == count
        assert batch.pad == pad
      end
    end
  end

  describe "inline serving: batching topology invariance" do
    property "run(serving, batch) equals the direct computation" do
      check all(count <- integer(1..8), width <- integer(1..4), max_runs: 15) do
        check all(vals <- rows(count, width), max_runs: 1) do
          tensors = Enum.map(vals, &Nx.tensor(&1, type: {:f, 32}))
          batch = Nx.Batch.stack(tensors)

          result = Nx.Serving.run(double_serving(), batch)
          direct = Nx.multiply(Nx.stack(tensors), 2)

          assert_all_close(result, direct, atol: 0.0)
        end
      end
    end

    property "batch_size forcing splits does not change results" do
      check all(count <- integer(2..8), batch_size <- integer(1..4), max_runs: 15) do
        check all(vals <- rows(count, 3), max_runs: 1) do
          tensors = Enum.map(vals, &Nx.tensor(&1, type: {:f, 32}))
          batch = Nx.Batch.stack(tensors)

          serving = double_serving() |> Nx.Serving.batch_size(batch_size)

          result = Nx.Serving.run(serving, batch)
          direct = Nx.multiply(Nx.stack(tensors), 2)

          assert_all_close(result, direct, atol: 0.0)
        end
      end
    end

    property "running split halves separately equals running the whole" do
      check all(count <- integer(2..8), max_runs: 15) do
        check all(vals <- rows(count, 3), split_at <- integer(1..(count - 1)), max_runs: 1) do
          tensors = Enum.map(vals, &Nx.tensor(&1, type: {:f, 32}))
          batch = Nx.Batch.stack(tensors)
          serving = double_serving()

          whole = Nx.Serving.run(serving, batch)

          {left, right} = Nx.Batch.split(batch, split_at)
          left_result = Nx.Serving.run(serving, left)
          right_result = Nx.Serving.run(serving, right)
          reassembled = Nx.concatenate([left_result, right_result])

          assert_all_close(reassembled, whole, atol: 0.0)
        end
      end
    end

    property "padded batch yields the same visible rows" do
      check all(count <- integer(1..6), pad <- integer(1..4), max_runs: 15) do
        check all(vals <- rows(count, 3), max_runs: 1) do
          tensors = Enum.map(vals, &Nx.tensor(&1, type: {:f, 32}))

          plain = Nx.Serving.run(double_serving(), Nx.Batch.stack(tensors))

          padded_batch = Nx.Batch.stack(tensors) |> Nx.Batch.pad(pad)
          padded = Nx.Serving.run(double_serving(), padded_batch)

          assert_all_close(padded, plain, atol: 0.0)
        end
      end
    end
  end

  describe "process serving: concurrent requests stay isolated" do
    test "concurrent batched_run requests each get their own answer" do
      name = :fuzz_serving_concurrent

      start_supervised!(
        {Nx.Serving,
         name: name, serving: double_serving(), batch_size: 3, batch_timeout: 5, shutdown: 1000}
      )

      # 16 concurrent requests with distinct payloads; the server will
      # merge/split them across batches of 3 with a short flush timeout.
      # Isolation contract: every caller gets exactly 2x its own input.
      tasks =
        for i <- 1..16 do
          Task.async(fn ->
            input = Nx.tensor([[i * 1.0, i + 0.5, i - 0.25]])
            batch = Nx.Batch.concatenate([input])
            result = Nx.Serving.batched_run(name, batch)
            {input, result}
          end)
        end

      for {input, result} <- Task.await_many(tasks, 10_000) do
        assert_all_close(result, Nx.multiply(input, 2), atol: 0.0)
      end
    end

    test "mixed-size concurrent requests reassemble correctly" do
      name = :fuzz_serving_mixed_sizes

      start_supervised!(
        {Nx.Serving,
         name: name, serving: double_serving(), batch_size: 4, batch_timeout: 5, shutdown: 1000}
      )

      # request sizes 1..5 rows — some smaller than batch_size (merged),
      # some larger (split), exercising both boundary paths concurrently
      tasks =
        for size <- [1, 2, 3, 4, 5, 2, 1, 3] do
          Task.async(fn ->
            rows = for r <- 1..size, do: [r * 1.0, r * 10.0]
            input = Nx.tensor(rows)
            result = Nx.Serving.batched_run(name, Nx.Batch.concatenate([input]))
            {input, result}
          end)
        end

      for {input, result} <- Task.await_many(tasks, 10_000) do
        assert Nx.shape(result) == Nx.shape(input)
        assert_all_close(result, Nx.multiply(input, 2), atol: 0.0)
      end
    end
  end

  describe "input streams" do
    test "a stream of batches equals the concatenated direct computation" do
      serving = double_serving() |> Nx.Serving.batch_size(2)

      batches = [
        Nx.Batch.concatenate([Nx.tensor([[1.0, 2.0]])]),
        Nx.Batch.concatenate([Nx.tensor([[3.0, 4.0], [5.0, 6.0]])])
      ]

      result = Nx.Serving.run(serving, Stream.map(batches, & &1))

      direct = Nx.multiply(Nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), 2)
      assert_all_close(result, direct, atol: 0.0)
    end
  end
end
