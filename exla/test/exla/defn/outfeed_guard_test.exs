defmodule EXLA.Defn.OutfeedGuardTest do
  use ExUnit.Case, async: true

  alias EXLA.Defn.OutfeedGuard, as: G

  test "acquire and release on the same device" do
    device = System.unique_integer()
    assert :ok = G.acquire(device, :test_client)
    assert :ok = G.release(device)
  end

  test "sequential acquire-release cycles on the same device" do
    device = System.unique_integer()
    assert :ok = G.acquire(device, :client_a)
    assert :ok = G.release(device)
    assert :ok = G.acquire(device, :client_b)
    assert :ok = G.release(device)
  end

  test "different devices can be acquired concurrently" do
    device_a = System.unique_integer()
    device_b = System.unique_integer()
    assert :ok = G.acquire(device_a, :client_a)
    assert :ok = G.acquire(device_b, :client_b)
    assert :ok = G.release(device_a)
    assert :ok = G.release(device_b)
  end

  test "same client can reacquire after release" do
    device = System.unique_integer()
    assert :ok = G.acquire(device, :host)
    assert :ok = G.release(device)
    assert :ok = G.acquire(device, :host)
    assert :ok = G.release(device)
  end

  test "raises on concurrent cross-client outfeed on the same device" do
    device = System.unique_integer()
    parent = self()

    task =
      Task.async(fn ->
        G.acquire(device, :host)
        send(parent, :acquired)
        assert_receive :release
        G.release(device)
      end)

    assert_receive :acquired

    assert_raise RuntimeError, ~r/Concurrent outfeed conflict on device/, fn ->
      G.acquire(device, :other_host)
    end

    send(task.pid, :release)
    Task.await(task)
  end

  test "error message includes both client names and device id" do
    device = System.unique_integer()
    parent = self()

    task =
      Task.async(fn ->
        G.acquire(device, :host)
        send(parent, :acquired)
        assert_receive :release
        G.release(device)
      end)

    assert_receive :acquired

    error =
      assert_raise RuntimeError, fn ->
        G.acquire(device, :other_host)
      end

    assert error.message =~ "client :host"
    assert error.message =~ "Client :other_host"
    assert error.message =~ "device #{device}"
    assert error.message =~ "XLA outfeed queues are global per device"

    send(task.pid, :release)
    Task.await(task)
  end

  test "cleans up stale entries from dead processes" do
    device = System.unique_integer()

    task =
      Task.async(fn ->
        G.acquire(device, :dead_client)
        # Exit without releasing — simulates a crash
      end)

    Task.await(task)

    # The entry is stale (process is dead). Acquire should clean it up.
    assert :ok = G.acquire(device, :new_client)
    assert :ok = G.release(device)
  end

  test "release is idempotent" do
    device = System.unique_integer()
    assert :ok = G.release(device)
    assert :ok = G.release(device)
  end
end
