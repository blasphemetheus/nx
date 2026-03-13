defmodule EXLA.Defn.OutfeedGuard do
  @moduledoc false

  # Detects concurrent outfeed access on the same device from different EXLA clients.
  #
  # XLA's outfeed queue (XfeedQueueManager) is global per device ordinal, not per client.
  # EXLA.Defn.Lock serializes outfeed-using executions per [client_ref | device_id],
  # which means different clients can interleave on the shared queue. When that happens,
  # buffer sizes mismatch and XLA aborts the process (SIGABRT).
  #
  # This guard uses a public ETS table keyed by device_id. If a second client tries to
  # start an outfeed on a device that already has one in progress, it raises a clear
  # error instead of letting the runtime crash.

  use GenServer

  @table __MODULE__

  def start_link(_opts) do
    GenServer.start_link(__MODULE__, :ok, name: __MODULE__)
  end

  @impl true
  def init(:ok) do
    :ets.new(@table, [:named_table, :public, :set])
    {:ok, nil}
  end

  @doc """
  Registers an outfeed operation on `device_id` for `client_name`.

  Raises if another client already has an active outfeed on the same device.
  """
  def acquire(device_id, client_name) do
    case :ets.insert_new(@table, {device_id, client_name, self()}) do
      true ->
        :ok

      false ->
        [{^device_id, other_client, other_pid}] = :ets.lookup(@table, device_id)

        if Process.alive?(other_pid) do
          raise RuntimeError,
                "Concurrent outfeed conflict on device #{device_id}: " <>
                  "client :#{other_client} (#{inspect(other_pid)}) already has an active outfeed. " <>
                  "Client :#{client_name} cannot start a concurrent outfeed on the same device. " <>
                  "XLA outfeed queues are global per device — concurrent access from different " <>
                  "clients will corrupt the queue and crash the runtime."
        else
          # Stale entry from a crashed process, clean up and retry
          :ets.delete(@table, device_id)
          acquire(device_id, client_name)
        end
    end
  end

  @doc """
  Releases the outfeed guard for `device_id`. Idempotent.
  """
  def release(device_id) do
    :ets.delete(@table, device_id)
    :ok
  end
end
