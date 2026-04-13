<original_task>
Contributing to elixir-nx/nx on two issues:

1. **Issue #1689** — Fatal XLA SIGABRT crash when vectorized cond with hooks has predicates on different axes. Root cause: cross-client outfeed queue corruption because the EXLA defn lock key included client_ref, allowing different clients to interleave on XLA's global per-device outfeed queue.

2. **Issue #1683 / PR #1683** — CallbackServer process leak on repeated JIT compilation. Every `__compile__` call starts a CallbackServer process that lives forever, exhausting the BEAM process limit in long-running workloads (~25K+ JIT calls).
</original_task>

<work_completed>
## Issue #1689 — COMPLETE, MERGED

**PR #1691** merged on 2026-03-13. Fix: change lock key from `[client_ref | device_id]` to platform-aware key.

Final code in `lib/exla/defn.ex`:
```elixir
defp run_key(%{client: %{platform: :host}, device_id: device_id}), do: [:host | device_id]
defp run_key(%{client: %{ref: ref}, device_id: device_id}), do: [ref | device_id]
```

## Issue #1683 — PID-as-tensor IMPLEMENTATION COMPLETE

**PR #1683** on branch `fix/lazy-callback-server`. Paulo/Jose's PID-as-tensor approach is fully implemented and tested.

### What was done:

**Elixir compilation side (`lib/exla/defn.ex`, `lib/exla/defn/outfeed.ex`, `lib/exla/mlir/value.ex`):**
- `runtime_call` ops detected during `used_inputs_and_hooks` pre-scan via `:__has_runtime_calls__` marker
- `has_runtime_calls` flag stored on Outfeed struct
- When `has_runtime_calls` is true, a `{:u, 8}` PID tensor typespec appended to `comp_typespecs` (computation input parameters)
- Last function argument extracted as `callback_pid_param` Value, stored on Outfeed struct
- PID Value threaded through while loops (`cached_recur_operator(:while, ...)` and `mlir_while_computation`) — prepended to initial state after token, extracted from region params, returned in body
- PID Value threaded through optional computations (`optional_computation`) — added to function arg/return typespecs, extracted from function args, returned in function results
- `merge_outfeed` updated to restore outer PID param (alongside outer token)
- `Value.runtime_call/3` changed to `/4` — PID appended as last operand to `stablehlo.custom_call`
- If branches (`to_mlir_if_branch`) don't need PID threading — they close over outer scope directly

**Elixir execution side (`lib/exla/defn.ex`, `lib/exla/defn/outfeed.ex`):**
- `Outfeed.start_child` returns `{:ok, outfeed_pid, callback_target}` — callback_target is either the task itself (callbacks-only) or a separate helper process (when hooks block the task in from_outfeed NIF)
- `maybe_outfeed` callbacks-only clause: serializes `callback_target` PID via `term_to_binary`, wraps as `BinaryBuffer`, appends to input buffers
- `maybe_outfeed` hooks+callbacks clause: same PID injection, but uses `callback_target` (the helper process PID)
- Outfeed `init` no longer registers with CallbackDispatcher — the C++ side sends directly to the PID from the tensor

**C++ side:**
- `runtime_callback.cc`: Last input arg extracted as PID tensor (excluded from callback args), PID data + size passed to bridge
- `runtime_callback_bridge.cc`: `InvokeRuntimeCallback` accepts `pid_data`/`pid_size`, decodes PID via `enif_binary_to_term` + `enif_get_local_pid`, sends message directly to decoded PID
- `runtime_callback_cuda.cc`: Same PID extraction with D→H copy for the PID buffer
- Global `BridgeState` singleton removed entirely
- `start_runtime_callback_bridge` and `clear_runtime_callback_bridge` functions removed from bridge, header, NIF registrations (`exla.cc`), and Elixir NIF stubs

**Cleanup:**
- `lib/exla/defn/callback_dispatcher.ex` — DELETED (was ETS-based global dispatcher from earlier approach)
- Removed `CallbackDispatcher` from supervision tree (`lib/exla/application.ex`)
- Removed NIF stubs for `start_runtime_callback_bridge` and `clear_runtime_callback_bridge` from `lib/exla/nif.ex`
- Removed NIF registrations from `c_src/exla/exla.cc`
- Test updated: replaced ETS cleanup test with process leak test

### Test results:
- `runtime_call_test.exs`: **39 pass, 0 fail**, 1 excluded (cuda)
- `api_test.exs`: **17 pass, 0 fail**
- `expr_test.exs`: **294 pass, 0 fail**
- `mix test --stale`: **1417 pass, 1 fail** (pre-existing SVD tolerance issue), 49 excluded
</work_completed>

<work_remaining>
## PR #1683 — Ready for review

The implementation is complete and pushed to `fork/fix/lazy-callback-server`. Remaining steps:

1. **CI validation** — Wait for GitHub Actions to run the full matrix (CPU, CUDA, multi-device)
2. **Address reviewer feedback** — Paulo/Jose may have comments on the implementation
3. **Potential cleanups reviewers might request:**
   - The `callback_pid_size/0` helper always returns 29 for local PIDs — reviewers might want a comment about remote PID limitation
   - The `@doc` ordering fix in outfeed.ex (was a pre-existing issue we fixed)
   - Whether the remaining `runtime_callback_reply` NIF should be renamed or documented

## Issue #1690 — Raw outfeed API (CLOSED, underlying issue remains)

Not actively being worked on. The defn path is protected by #1691. The raw `Client.from_outfeed` API has no lock. Only revisit if raw outfeed usage expands beyond tests. Branch `fix/outfeed-guard` has a ready-to-apply detection approach if needed.
</work_remaining>

<attempted_approaches>
## Approach 1: Lazy CallbackServer (initial PR, superseded)
- Made CallbackServer start lazily (only when runtime_call encountered)
- Problem: PID still baked into compiled graph at compile time → leaks on recompilation
- Status: Superseded

## Approach 2: Delete CallbackServer, use global ETS dispatcher (superseded)
- Deleted CallbackServer entirely
- Created ETS-based CallbackDispatcher for routing
- Problem: Paulo flagged ETS leak risk, Jose/Paulo prefer PID-as-tensor
- Status: Working but replaced by approach 3

## Approach 3: PID-as-tensor (IMPLEMENTED, current)
- Serialize callback server PID via `term_to_binary`, pass as u8 tensor argument
- C++ extracts PID from last input buffer, sends messages directly
- No global state, no ETS, no leaks
- Status: Complete, all tests passing

## Approach 4: OutfeedGuard (for #1689, kept as reference)
- ETS-based concurrent outfeed conflict detection
- Jose preferred fix within existing lock system
- Status: Branch `fix/outfeed-guard` on fork

## Key dead ends:
- `on_unlock` fix for #1689 was a timing coincidence, not correctness
- `device_id` alone as lock key too broad (CPU-0 blocks GPU-0)
- Stress tests in single module couldn't reproduce #1689 — needed cross-module async
</attempted_approaches>

<critical_context>
## Architecture: How runtime_call now works (PID-as-tensor)

1. **Compile time**: `used_hooks` pre-scan detects `:runtime_call` ops → `has_runtime_calls` flag on Outfeed struct → extra `{:u, 8}` parameter appended to computation inputs → PID Value threaded through while/optional regions alongside token → `stablehlo.custom_call` gets PID as last operand

2. **Execution time**: `Outfeed.start_child` spawns ephemeral callback server → returns `callback_target` PID → PID serialized via `term_to_binary` → wrapped as `BinaryBuffer` → appended to input buffers → XLA runs computation

3. **C++ FFI**: Handler extracts last input as PID buffer → `enif_binary_to_term` + `enif_get_local_pid` → sends `{:exla_runtime_call, callback_id, args, reply_tag}` directly to decoded PID → callback server executes Elixir function → replies via `runtime_callback_reply` NIF → C++ unblocks

4. **Cleanup**: Callback server receives `:done` when execution finishes → process exits → no leak

## PID threading pattern

The PID is threaded through control flow regions alongside the token. Order in while loop state: `[token, pid | user_values]`. Key invariants:
- `reset_token` + explicit PID set when entering inner scope
- `merge_outfeed` restores both outer token and outer PID when exiting inner scope
- If branches don't thread PID — they capture outer scope directly
- Pred regions receive PID as parameter but don't return it (only body returns it)

## PID binary size

`:erlang.term_to_binary(pid)` for local PIDs is always 29 bytes (verified by test). This is critical for cached computation consistency. Remote/distributed PIDs have variable size — acknowledged limitation, not addressed.

## Hooks + callbacks coexistence

When both outfeed hooks and runtime_calls are present:
- Outfeed task blocks in `from_outfeed` NIF (dirty IO) — can't receive messages
- Helper process spawned via `spawn_link` for callback handling
- `start_child` returns helper PID as `callback_target`
- That helper PID is what gets serialized into the tensor

## Issue #1690 notes

Raw `Client.from_outfeed` has no lock protection. Branch `fix/outfeed-guard` has ready-to-apply ETS-based detection. Practical risk is low — API is internal-only. Revisit if raw outfeed usage expands.
</critical_context>

<current_state>
## Git state
- **Current branch**: `fix/lazy-callback-server` (synced with `fork/fix/lazy-callback-server`)
- **Working tree**: Clean except for `whats-next.md` (untracked)
- **Latest commit**: `9648e1b3` (Replace global CallbackDispatcher with PID-as-tensor approach)

## Branch inventory

| Branch | Status | Purpose |
|--------|--------|---------|
| `fix/outfeed-device-lock` | MERGED as PR #1691 | #1689 fix (lock key) |
| `fix/outfeed-guard` | On fork, not pursued | #1689 alternative (detection) |
| `fix/lazy-callback-server` | Open PR #1683, ready for review | #1683 fix (PID-as-tensor) |
| `test/verify-1689-on-main` | On fork, investigation only | #1689 stress tests + analysis |

## PR #1683 status
- Open, implementation complete
- PID-as-tensor approach per Paulo/Jose feedback fully implemented
- 39 runtime_call tests pass, 1417 stale tests pass (1 pre-existing SVD failure)
- Pushed to fork, awaiting CI and reviewer feedback

## PR #1691 status
- **MERGED** on 2026-03-13
</current_state>
