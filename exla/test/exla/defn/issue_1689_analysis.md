# Issue #1689 Analysis: Vectorized Cond + Hooks Outfeed Race Condition

## Status: Fix Verified

## Bug Summary

When a `cond` with hooks (`send_value` / `Nx.Defn.Kernel.hook`) has vectorized
predicates on **different axes**, repeated execution intermittently crashes with:

```
** (RuntimeError) XLA runtime-managed outfeed buffer size 2 did not match
the outfeed operation parameter buffer size 4

[FATAL] xla/backends/cpu/runtime/xfeed_manager.cc:62
Check failed: current_buffer_ == nullptr
```

Exit code 134 (SIGABRT) — hard crash, kills the BEAM.

## Reproduction

Reproduced on main (not PR-specific) via 50-iteration repetition test on
Elixir 1.17.3 / OTP 27.3, Linux CI (GitHub Actions):

```elixir
test "hooked cross-axis cond under repetition (flakiness detector)" do
  for _ <- 1..50 do
    result =
      hooked_cond_different_axes(
        Nx.vectorize(~VEC[1 0], :a),
        Nx.vectorize(~VEC[0 1 0], :b)
      )
    assert_equal(result, expected)
  end
end
```

- Failed on 1.17.3/OTP 27.3, passed on 1.18.4/OTP 28.3 (same CI run)
- Non-deterministic: does not fail every run

## Outfeed Architecture

### Execution Lifecycle (defn.ex:263-281)

```
1. Runner GenServer starts (holds lock, waits for signal)
2. Outfeed Erlang task starts → calls from_outfeed(flag_typespec) [dirty IO NIF, blocks]
3. Lock.transfer(lock, send_to_runner, outfeed_pid)
   → runner receives lock → executes XLA
4. XLA writes outfeed: flag → data → flag → data → ... → flag=0
5. Outfeed task reads flags, dispatches hooks, loops
6. Outfeed task reads flag=0 → returns :ok → process exits
7. Lock releases (via :DOWN monitor in Lock GenServer)
8. Main process gets :DOWN → reads runner results
9. Next execution can now acquire lock
```

### Device Outfeed Queue (xfeed_manager.cc)

- **Global per device ordinal** — one `XfeedManager` per device, shared across ALL executions
- Backed by `std::deque<XfeedBuffer*>` (FIFO)
- `EnqueueBuffersAtomically()` — Erlang side enqueues buffers (via from_outfeed NIF)
- `BlockingDequeueBuffer()` — XLA side dequeues (blocks until buffer available)
- `ReleaseCurrentBuffer()` — XLA releases after memcpy
- Protected by `absl::Mutex mu_`
- Invariant: `CHECK(current_buffer_ == nullptr)` before dequeue (line 62)

### from_outfeed NIF (exla.cc:465-478)

```cpp
fine::Ok<> transfer_from_outfeed(..., std::vector<xla::Shape> shapes,
                                  ErlNifPid pid, fine::Term ref) {
  for (auto &shape : shapes) {
    auto msg = client->TransferFromOutfeed(device_id, shape);
    enif_send(env, &pid, msg_env, tuple(ref, msg));
  }
  return fine::Ok();
}
FINE_NIF(transfer_from_outfeed, ERL_NIF_DIRTY_JOB_IO_BOUND);
```

- Runs on dirty IO scheduler
- Blocks per-shape on device queue
- Sends results to Erlang PID via enif_send

### Vectorized Cond Compilation (expr.ex:136-189)

Vectorized cond with predicates on different axes compiles to:

```
then_selector = Nx.any(devec_pred)     # scalar: "any element true?"
else_selector = Nx.all(devec_pred)     # scalar: "all elements true?"
then_result = if(then_selector) { hook_true; expr }
else_result = if(!else_selector) { hook_false; cond(rest) }
final = Nx.select(pred, then_result, else_result)
```

**Both branches always execute** (any/all), both hooks always fire.
`Nx.select` picks per-element after. Outfeed sequence is deterministic.

## Outfeed Queue Protocol (Confirmed via XLA source)

### Buffer Flow Direction

The terminology is counterintuitive. The outfeed queue works like this:

1. **Consumer (Erlang)** calls `TransferFromOutfeed` → `EnqueueBuffersAtomically`
   - Enqueues an **empty buffer** of expected size onto the device queue
2. **Producer (XLA)** calls `OutfeedThunk::Execute` → `BlockingDequeueBuffer`
   - Dequeues the empty buffer, **checks size matches**, fills it via memcpy
3. **Producer (XLA)** calls `ReleaseCurrentBuffer` → triggers `Done()` callback
   - Consumer gets notification that data is ready

So the consumer **pre-allocates** buffers and XLA **fills** them. The size check
at step 2 verifies the pre-allocated buffer matches what XLA wants to write.

### NIF Scheduler Assignments

- `run_cpu` → `ERL_NIF_DIRTY_JOB_CPU_BOUND` (dirty CPU scheduler)
- `from_outfeed` → `ERL_NIF_DIRTY_JOB_IO_BOUND` (dirty IO scheduler)
- These run on **different** scheduler pools, concurrently

### XLA Execution Synchrony

`run_cpu` NIF is **synchronous** — blocks until XLA computation completes,
including all OutfeedThunk executions. XLA's OutfeedThunk blocks on
`BlockingDequeueBuffer` waiting for the consumer to enqueue a buffer.
So XLA execution and outfeed consumption are **co-dependent**:
- XLA blocks waiting for consumer to provide buffers
- Consumer blocks waiting for XLA to fill buffers

## Race Condition Analysis

### The Buffer Size Mismatch (2 vs 4)

- u16 flag = 2 bytes (pre-allocated by consumer for flag reads)
- s32 scalar hook result = 4 bytes (pre-allocated by consumer for data reads)
- **XLA dequeued a 2-byte flag buffer when it expected to fill a 4-byte data buffer**

### Confirmed Race Mechanism

The device outfeed queue (`enqueued_buffers_` deque) is **global per device**.
All executions sharing the same device_id share the same FIFO queue.

```
Execution N (in progress):
  - XLA is executing on dirty CPU scheduler
  - Outfeed task N is enqueuing buffers on dirty IO scheduler
  - XLA dequeues buffers, fills them, releases them
  - Outfeed task N reads flag=0 → exits
  - Lock releases (:DOWN in Lock GenServer)

Execution N+1 (starts immediately):
  - Outfeed task N+1 starts, calls from_outfeed(flag_typespec)
  - from_outfeed NIF runs on dirty IO scheduler
  - NIF calls EnqueueBuffersAtomically → puts 2-byte flag buffer on queue

  BUT: Execution N's run_cpu NIF is still running on dirty CPU scheduler!
  - N's OutfeedThunk calls BlockingDequeueBuffer()
  - Gets N+1's 2-byte flag buffer (instead of a buffer from N's consumer)
  - buffer->length() (2) != outfeed_buffer.slice.size() (4)
  - → RuntimeError: "buffer size 2 did not match buffer size 4"
  - → CHECK(current_buffer_ == nullptr) fails → SIGABRT
```

### Why The Lock Doesn't Prevent This

The lock is transferred to `outfeed_pid` (defn.ex:271). When `outfeed_pid`
exits, the lock releases via `:DOWN` handler in `Lock` GenServer. But:

1. `outfeed_pid` exits when it reads flag=0
2. The flag=0 outfeed write is the **last token-ordered** outfeed in the XLA graph
3. But `run_cpu` NIF hasn't returned yet — it's still executing on the dirty
   CPU scheduler (there may be post-outfeed computation like `Nx.select`)
4. More critically: XLA's `OutfeedThunk::Execute` for the flag=0 write must:
   a. Call `BlockingDequeueBuffer` (gets the consumer's 2-byte buffer)
   b. Fill the buffer with `0x0000`
   c. Call `ReleaseCurrentBuffer` (triggers Done callback → consumer gets data)
   d. **Return from Execute**

   The consumer reads the flag=0 and exits at step (c). But step (d) hasn't
   completed yet. If there are more thunks after the outfeed close, XLA is
   still running. Even if flag=0 is the last outfeed, `run_cpu` still needs
   to return from the NIF call.

5. The next execution's outfeed task starts and enqueues a buffer on the
   same global device queue
6. If any pending XLA operation from execution N tries to dequeue (shouldn't
   happen if flag=0 is truly last), it gets N+1's buffer

### Revised Theory: The Token Chain vs Thunk Execution Order

The real question is whether the XLA ThunkExecutor guarantees that outfeed
thunks execute in token-chain order. From the stack trace:

```
xla::cpu::ThunkExecutor::TracedExecute()
xla::cpu::ThunkExecutor::ExecuteSequential()
xla::cpu::ThunkExecutor::Execute()
```

`ExecuteSequential` suggests sequential execution, but `ThunkExecutor::Execute`
may choose parallel execution for independent thunks. If two outfeed thunks
(from if-true and if-false branches) are considered independent, they could
execute in parallel or out-of-order, causing the consumer's pre-allocated
buffers to be dequeued by the wrong outfeed thunk.

### Answered Questions

- [x] Can ThunkExecutor execute outfeed thunks out of token-chain order?
      **NO.** Stack trace shows `ThunkExecutor::ExecuteSequential()` — thunks run
      in order within a single execution. The bug is cross-execution, not intra.
- [x] Could the lock release → next execution start happen fast enough on
      1.17.3/OTP 27.3 but not on 1.18.4/OTP 28.3 (explaining version-specific
      failure)?
      **NO.** Reproduced on 1.18.4/OTP 28.3 when sufficient dirty scheduler
      pressure exists. Not version-specific — timing-dependent.

- [x] Is there computation after the flag=0 outfeed in the vectorized cond graph?
      **YES.** The compilation flow in `exla/lib/exla/defn.ex:236-238` is:
      ```
      {res, cache} = recur_flatten(expr, state, new_cache(outfeed))
      outfeed = cache |> get_outfeed() |> Outfeed.close(function)  # flag=0 here
      Value.func_return(function, res)  # Nx.select result returned here
      ```
      The `Nx.select` (from vectorized cond, `expr.ex:185`) is part of `res`,
      compiled before `close()` in Elixir code, BUT in the XLA graph the outfeed
      close is a token-chain operation while `Nx.select` is a data operation.
      Both are thunks in the final graph. The key point: `run_cpu` NIF does not
      return until ALL thunks complete, including any data operations after the
      close flag. This widens the race window.
- [x] Does the dirty IO scheduler guarantee FIFO ordering for consecutive
      from_outfeed NIF calls from different Erlang processes?
      **NO.** Dirty IO schedulers are a thread pool (default 10, configurable via
      `+SDio`). Processes are enqueued on a shared run queue, but:
      - Different normal schedulers may enqueue to dirty queue at different times
      - OS thread scheduling determines actual execution order
      - No cross-scheduler synchronization preserves caller ordering
      - OTP docs make no ordering guarantees for dirty NIF dispatch
      This means concurrent `from_outfeed` calls from different outfeed tasks
      can enqueue buffers in arbitrary order on the global device queue.

### All Questions Answered

The complete picture:
1. The bug is **cross-execution** buffer interleaving (not intra-execution thunk reordering)
2. **Post-outfeed computation** (Nx.select) keeps `run_cpu` busy after flag=0, widening the race
3. **No dirty IO ordering guarantees** means concurrent outfeed tasks can interleave freely
4. The **device lock** releases too early (on outfeed task exit, before `run_cpu` returns)
5. The fix should ensure the lock is not released until both outfeed AND runner complete

## Confirmed Reproduction (2026-03-12)

**Crash reproduced on Elixir 1.18.4 / OTP 28.3** — previously only seen on 1.17.3.

The crash occurred on the basic single-execution test (`test "hook inside cond with
different vectorization axes"` at line 373) while concurrent stress tests were
running in the same ExUnit session. This confirms:

1. The bug is **cross-execution** — concurrent stress tests create enough outfeed
   queue traffic that even a simple hooked cond execution hits interleaved buffers.
2. The bug is **not OTP-version-specific** — it triggers on any version when dirty
   scheduler pressure is sufficient.
3. The key trigger is **concurrent hooked defn executions on the same device** —
   the global per-device outfeed queue has no execution-scoped isolation.

### Crash Details

```
F0312 22:55:01.563633  xfeed_manager.cc:62  Check failed: current_buffer_ == nullptr
RuntimeError: XLA runtime-managed outfeed buffer size 2 did not match
the outfeed operation parameter buffer size 4

Stack trace:
  xla::cpu::XfeedQueueManager::BlockingDequeueBuffer()
  xla::cpu::OutfeedThunk::Execute()
  xla::cpu::ThunkExecutor::TracedExecute()
  xla::cpu::ThunkExecutor::ExecuteSequential()  ← sequential, not parallel
  xla::cpu::ThunkExecutor::Execute()
  xla::CpuPjRtRawLoadedExecutable::Execute()
  exla::ExlaExecutable::Run()
  exla::run()
```

The crash occurred in `run_cpu` NIF → `EXLA.Defn.Runner.handle_continue/2`.
GenServer PID `#PID<0.897.0>` was the runner that got the wrong buffer.

### Reproduction Strategy

The most effective trigger was **concurrent mixed workloads** (blast mode tests,
scheduler pressure tests) running simultaneously with basic hooked cond tests.
ExUnit's `async: true` runs all tests concurrently, so the stress tests created
background outfeed queue traffic that the basic test couldn't handle.

## Vulnerability Scope

| Code Path | Uses Outfeed Queue | Vulnerable | Notes |
|-----------|-------------------|------------|-------|
| Hooks (`send_value`, `hook`) | YES | **YES** | Same lock pattern |
| Lazy transfers (`:always`) | YES (infeed) | **YES** | Same global queue, same lock |
| `runtime_call` | NO | **NO** | Uses `custom_call` + `CallbackServer` bridge |
| Normal execution (no hooks) | NO | **NO** | Synchronous `Lock.unlock` in `after` block |

## Fix: Lock Chaining via `on_unlock`

### Approach

Chain the lock from `outfeed_pid` → `runner` so the lock holds until `run_cpu` NIF
returns. Uses `Lock.on_unlock` to register a `{:transfer, runner}` callback before
transferring the lock to `outfeed_pid`:

```elixir
# defn.ex — before the fix:
_ = EXLA.Defn.Lock.transfer(lock, fn -> send(runner, lock) end, outfeed_pid)

# defn.ex — after the fix:
_ = EXLA.Defn.Lock.on_unlock(lock, fn -> :ok end, fn -> {:transfer, runner} end)
_ = EXLA.Defn.Lock.transfer(lock, fn -> send(runner, lock) end, outfeed_pid)
```

`on_unlock` MUST be called before `transfer` — the `to_unlock` callback is preserved
through transfer (Lock GenServer does `{{to_unlock, _pid}, queue} -> {{to_unlock, pid}, queue}}`).
This avoids a race where outfeed exits before `on_unlock` is processed.

### Results

**Fix verified on CI (both Elixir versions) and locally.**

- All previously crashing seeds (12345, 99999, 22222, 55555) now pass
- Seed 55555: 5 consecutive runs, zero SIGABRT crashes
- Fork CI (PR #2): passes on both 1.17.3/OTP 27.3 and 1.18.4/OTP 28.3

Note: An earlier version of the fix placed `on_unlock` AFTER `transfer`, which
still crashed intermittently (seeds 22222, 55555). Moving `on_unlock` BEFORE
`transfer` eliminated the remaining race — the callback must be registered before
outfeed_pid is monitored.

## CI Results

| Commit | 1.17.3 | 1.18.4 | Notes |
|--------|--------|--------|-------|
| Before fix (stress tests) | **CRASH** | **CRASH** | Reproduced #1689 on both versions |
| With fix (on_unlock chain) | **PASS** | **PASS** | No SIGABRT crashes |

## Local Reproduction

| Seed | Without fix | With fix (on_unlock) |
|------|------------|---------------------|
| 12345 | CRASH (134) | pass |
| 99999 | CRASH (139) | pass |
| 22222 | CRASH (134) | pass |
| 55555 | CRASH (134) | pass (5/5 runs) |
| 990472 | pass | pass |
| 286354 | pass | pass |

## Files of Interest

- `exla/lib/exla/defn/outfeed.ex` — Outfeed task lifecycle, hook dispatch
- `exla/lib/exla/defn.ex:263-281` — Execution coordination (runner + outfeed + lock)
- `exla/lib/exla/defn/runner.ex` — Runner GenServer (executes XLA)
- `exla/lib/exla/defn/lock.ex` — Device lock (per device_id, released on :DOWN)
- `exla/c_src/exla/exla.cc:465-480` — from_outfeed NIF (dirty IO)
- `xla/backends/cpu/runtime/xfeed_manager.cc` — XLA outfeed queue (global per device)
- `xla/backends/cpu/runtime/outfeed_thunk.cc` — XLA outfeed execution (size check)
- `nx/lib/nx/defn/expr.ex:136-189` — Vectorized cond compilation (any/all/select)
