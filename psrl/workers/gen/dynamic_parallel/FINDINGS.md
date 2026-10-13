# Findings

Measured behavior of dynamic heterogeneous rollout parallelism. Reproduce with
`psrl/bench/dynamic_parallel/oversubscribe_probe.py`.

Platform for every number below: 1 node, 8 x NVIDIA H20 (97871 MiB each),
Qwen3-8B, `gpu_memory_utilization=0.90`, bf16.

## Go / no-go: oversubscription is viable, with zero KV loss

The question Milestone 0 existed to answer was whether an awake replica still
gets its full KV budget when every other TP degree is resident but asleep. It
does, exactly.

Full `{1,2,4,8}` set on 8 GPUs, so 15 replica slots and **4 engines per GPU**,
against a control run of each degree alone:

| TP | slots | `num_gpu_blocks` oversubscribed | control | KV tokens per slot |
|---|---|---|---|---|
| 8 | 1 | 298296 | 298296 | 4772736 |
| 4 | 2 | 145421 | 145421 | 2326736 |
| 2 | 4 | 69256 | 69256 | 1108096 |
| 1 | 8 | 31143 | 31143 | 498288 |

Every value matches its control with no rounding, in both directions of the
staged order. Two full switch cycles over all four degrees ran afterwards without
a single block lost.

**All 15 engines asleep cost at most 8615 MiB per GPU**, around 8.7% of the card.
A single sleeping engine measured 739 to 983 MiB, roughly 45x less than awake,
so sleep level 2 genuinely releases the pages rather than just the KV blocks.

The memory objection to pre-materializing every TP degree therefore does not hold
at this model size on 97 GB cards. On smaller cards the resident floor is the same
per engine but the margin shrinks, so narrow `tp_degrees` there.

## Run slots with CUDA graphs. TMS is the only mechanism that currently works

**Set `psrl.tms.range: gen` and `psrl.tms.enable_cuda_graph: true`.** Graphs are
worth keeping, and TMS is what makes a slot both switchable and fast. Eager is
accepted as a fallback but costs decode throughput.

`gen` rather than `all`: `all` also wraps the train worker's allocations, and
Megatron offloads optimizer state with a plain device-to-host copy
(`megatron_utils.py:646 offload_tensor_to_cpu`) that fails with
`CUDA error: invalid argument` on a TMS-managed region. The train worker dies during
`init_model` before any rollout slot is reached. `validate_config` now rejects
`range='all'` together with optimizer offload and names `gen` as the fix.

Two engines sharing one GPU, graphs on, both at level 2:

| | graph + TMS | eager |
|---|---|---|
| wake latency | **0.03 s** | 0.30 s |
| `num_gpu_blocks` | 11725 | 11686 |
| resident while asleep | 983 MiB | 739 MiB |
| correct after weight reload | yes | yes |

Graphs cost about 240 MiB more per sleeping engine and wake 10x faster, with no
throughput penalty, so there is no tradeoff to make.

### Why graphs without TMS does not work

This is not a TMS design preference, it is where the two available code paths
currently stand.

**PSRL patches vLLM's sleep to destroy and rebuild graphs.**
`release_cudagraphs` and `recapture_cudagraphs` are **not upstream**. They are 151
lines added by `patch/vllm/v0.22.0.patch`, and stock `GPUWorker.sleep` only calls
`CuMemAllocator.sleep(offload_tags=...)` without touching graphs at all. On that
patched path, wake fails during rebuild:

```
RuntimeError: it->second->use_count > 0 INTERNAL ASSERT FAILED at
"/pytorch/c10/cuda/CUDACachingAllocator.cpp":2799
```

It reproduces with a **single** engine alone on a GPU, so oversubscription is not
the cause. Invalidating the memoized process-global capture pool
(`cuda_graph.py:200`, `platforms/interface.py:880-887`) does not help. The reset
reached 37 wrappers and wake still failed, so the surviving reference is elsewhere,
most likely in the vLLM 0.22 `CudaGraphManager` path.

**Reverting to the stock path fails earlier, which is why the patch exists.**
Neutering `release_cudagraphs` so graphs stay in cumem memory moves the failure
from wake to **sleep**:

```
cumem.py:204  cpu_backup_tensor = torch.empty(...)
torch.AcceleratorError: CUDA error: invalid argument
```

cumem cannot offload a region that a resident CUDA graph pool holds, which is
exactly what the patch's own docstring warns about. So both non-TMS options are
broken: keeping the graphs breaks sleep, and destroying them breaks wake.

**TMS avoids both** by capturing graphs into a
`torch_memory_saver.cuda_graph(..., tag="graph")` region
(`patch/vllm/vllm_plugin/vllm_patches/patches/cuda_graph.py:107`) and pausing and
resuming that region alongside weights and KV
(`patches/gpu_worker.py:38-41, 59-60`). The graphs are offloaded and onloaded as
memory rather than destroyed, so neither failure is reachable.
`PSRL_vLLMHttpServer.wake_up` already appends the `"graph"` tag when
`enable_cuda_graph` is set (`vllm_async_server.py:750-752`).

**This matters outside this feature.** `enforce_eager` defaults to False
(`third_party/verl/verl/workers/config/rollout.py:178`) and `psrl.tms.range`
defaults to `null`, so a default rollout engine has graphs and no TMS, which is
the combination that breaks on wake. `elastic_rm` scale-up issues `WAKE_UP` to
those engines. Either that path is never exercised in that configuration, or it is
already broken. **Worth checking independently of this work.** Fixing either
non-TMS path upstream would remove the TMS requirement here.

## A level-2 wake needs a weight pull before it is correct

This looks like corruption and is not. After a level-2 wake, generation is
garbage:

```
before sleep:  " Paris. The capital of France is Paris."
after wake:    "!!!!!!!!!!!!!!!!"
```

Level 2 discards weights by design, so the model is reading unmapped memory until
weights come back. Calling `reload_weights` restores byte-identical output in both
graph and eager mode. PSRL always re-pulls from the PS after a wake
(`SwitchMixin.switch_to` step 7, via `sync_with_ps`), so the production path is
correct, but **any future code that wakes an engine without pulling weights will
silently produce garbage rather than failing.** A level-1 wake under eager keeps
weights and needs no pull.

## Switch latency is negligible against a training step

Sleep plus wake for a whole degree, eager, over two cycles:

| target TP | sleep (s) | wake (s) | switch (s) |
|---|---|---|---|
| 8 | 0.00 to 0.51 | 0.27 to 0.29 | 0.27 to 0.79 |
| 4 | 1.08 | 0.18 to 0.19 | 1.26 to 1.27 |
| 2 | 1.52 to 1.55 | 0.22 to 0.26 | 1.77 to 1.78 |
| 1 | 2.07 to 2.15 | 0.33 to 0.36 | 2.42 to 2.47 |

Worst case 2.47 s against `T_train` on the order of 30 s, so under 10%. Cost
scales with the number of slots being put to sleep rather than with the TP degree,
and the sleep side dominates. With graphs via TMS the wake side drops roughly
10x, so these are an upper bound. Switching is not a reason to avoid re-planning
at every weight update.

Wake does not re-profile. vLLM freezes `cache_config.num_gpu_blocks` during init
(`patch/vllm/vllm_plugin/.../gpu_worker.py:102`) and `initialize_kv_cache` runs
only on the init path, so a woken engine restores the budget it was built with and
nothing more. That is why bootstrap has to be staged rather than letting every
degree profile against an empty GPU.

## Staged bootstrap: what the end-to-end run proved, and what it did not

Reproduce with `examples/dapo_trainer/qwen3_8b_dynamic_parallel.sh`. Trainer TP2 x
PP2 on 4 GPUs, rollout pool of 4 GPUs carrying TP-1 and TP-2 slots, so 6 slots and
2 engines per GPU.

**Confirmed working on real GPUs:**

- The slot pool builds as one oversubscribed pool (`world_size=4`, `store=[4]`) and
  each slot carves a contiguous bundle range out of it.
- Staged bootstrap runs in descending TP order and completes: `Staging TP-2: 2
  slots` then `TP-2 slots built and asleep`, then the same for TP-1, ending with
  `All 6 slots built and asleep`.
- The planner picks the version-0 awake set from the table, here the four TP-1
  slots.
- The NIXL barrier sizes itself correctly at **12 rollout workers**, which is
  4x1 + 2x2 summed over all six slots, confirming the `world_size` accounting.

**Five ordering and sizing bugs the unit tests could not catch**, each found only
by running:

1. `nixl_sleep` awaits `_is_init_nixl_client`, which the NIXL protocol sets *after*
   worker init. Calling it during staged bootstrap deadlocks. Staged sleep must use
   plain `sleep`, and there is nothing registered to deregister at that point anyway.
2. NIXL registration pins live weight pages, so a sleeping engine fails
   `agent.registerMem` with `NIXL_ERR_BACKEND`. **Every slot must be awake for the
   barrier**, and the unselected ones go back to sleep afterwards. Waking is safe
   there because profiling is already done and `num_gpu_blocks` is frozen.
3. **Every slot being awake at once bounds `gpu_memory_utilization`.** With two
   degrees at 0.9 the barrier wake needs 1.8x each GPU and the vLLM worker dies with
   `out of memory`. `validate_config` now rejects
   `gpu_memory_utilization * engines_per_gpu > 1.0` and names the ceiling. This is a
   real cost of the design, not a tuning detail: four degrees caps each slot at 0.25
   of a card, so a slot's KV budget shrinks roughly in proportion to how many degrees
   are materialized. The NIXL barrier is a single collective across all agents, so
   registering slots one at a time is not available as an alternative.
4. `initial_pull_from_ps` pulled on every registered replica and marked them all
   current. For a sleeping slot the pull targets unmapped weights, and marking it
   current would make the sync loop skip it once woken. It is now scoped to awake
   slots.
5. `RayWorkerGroup._init_with_subresource_pool` never set `self.resource_pool`,
   unlike `_init_with_resource_pool`. Anything reading the pool back off the group
   crashed, which `_get_node_ids_from_wg` does during NIXL setup. Fixed in the
   vendored verl copy. **This affects `elastic_rm` too**, since it also builds
   groups from `SubRayResourcePool`.

**The degree count now trades against KV per slot.** Milestone 0 measured that
*sleeping* engines are nearly free, which remains true. What the end-to-end run
added is that they cannot all be *asleep* during the NIXL barrier, and that is what
forces the utilization split. So the practical ceiling on `tp_degrees` is set by how
much KV a slot needs while serving, not by idle memory.

**Training works once validate colocation is off.** The NCCL failure above is real
but avoidable, and a three-way comparison on the same node isolates it:

| run | `dynamic_parallel` | `colocate_validate_and_train` | result |
|---|---|---|---|
| control | False | True | 3/3 steps, clean |
| dynamic | True | True | trainer NCCL error at step 0 |
| dynamic | True | **False** | **2/3+ steps, clean** |

So the conflict is between the oversubscribed slot pool and validate instances
colocated on trainer GPUs at 0.9/0.1 fractions. Three fractional claimants on one
GPU is what breaks the trainer's `reduce_scatter_tensor_coalesced`. **Dynamic
parallelism and `colocate_validate_and_train` are incompatible as configured**, and
`validate_config` rejects the pair with that reason. Both cases are verified.

**The switch works end to end on real GPUs.** Verified with
`examples/dapo_trainer/qwen3_8b_dynamic_parallel.sh`: trainer TP2 x PP2 on 4 GPUs,
rollout pool of 4 GPUs carrying 6 slots, `tp_schedule=[1,2]`.

| boundary | from | to | latency |
|---|---|---|---|
| version 1 | slots [2,3,4,5] at TP-1 | slots [0,1] at TP-2 | 8.97 s |
| version 2 | slots [0,1] at TP-2 | slots [2,3,4,5] at TP-1 | 5.08 s |
| version 3 | slots [2,3,4,5] at TP-1 | slots [0,1] at TP-2 | 4.87 s |

Three weight-update boundaries produced three switches and zero `replan failed`
lines, so every boundary in the run switched. Both directions complete with no
errors and no trajectory loss. The first switch is slower because it sleeps four
slots and wakes two, against two and four for the others, which matches the earlier
finding that cost scales with the number of slots being put to sleep. The awake set
alternates exactly as `tp_schedule` dictates, so the planner, the
`_pending_switch_version` handoff, and `switch_to` all work on the real path rather
than only against a fake coordinator.

Bootstrap on the same run, for reference: `Staging TP-2` then `TP-2 slots built and
asleep`, then the same for TP-1, then `All 6 slots built and asleep. Version 0 will
wake [2,3,4,5]`, then `Woke all 6 slots for the NIXL barrier`, then
`Initialized NIXL clients: replica_count=6` and
`Initializing NIXL server with 1 PS workers, 4 actor workers, 8 rollout workers`.
The 8 rollout workers are 4x1 + 2x2 summed over the six slots. The initial pull
covered 4 replicas rather than 6, confirming it is scoped to the awake set.

**Two bugs had to be fixed before any of this was observable.**

**Bug 1 — switch.py's logger had no file handler.** `_LOG_FILE_TO_MODULE_LOGGERS` in
`coordinator.py` routes each module logger to its log file via `DualOutputHandler`.
`dynamic_parallel.switch` was not in the table, so every record from `switch.py`
— including the `psrl_logger.exception(...)` that swallows failures inside
`replan_parallelism` — went to a handler-less logger and disappeared. Fixed by
importing the module and adding its logger to the `SyncAndMigStrategy` entry.

**Bug 2 — state-transition lines were logged at INFO.** "Switching awake slots",
"Switch complete", and the early-exit "already matches" line are the evidence that a
boundary did something, so they are now WARNING. `consume_pending_parallelism_switch`
and `replan_parallelism` also log their entry and their decision, which is what made
the working path verifiable rather than merely silent.

## Two TP degrees can serve at once, and SMG routes to both

The switch results above alternate between homogeneous awake sets, so they never
answer whether one TP-2 replica and two TP-1 replicas can serve the same traffic
concurrently. That is the configuration §9.2 actually asks for, and it needs a
planner that names slots rather than degrees. `StaticCohortPlanner` wakes every
slot of one degree, so `ScriptedCohortPlanner` takes an explicit per-boundary slot
list instead.

Reproduce with `examples/dapo_trainer/qwen3_8b_dynamic_parallel_mixed_tp.sh`, which
sets `planner=scripted` and `slot_schedule=[[0,4,5],[1,2,3]]`. On the 4-GPU pool
slot 0 is TP-2 on GPUs 0-1, slot 1 is TP-2 on GPUs 2-3, and slots 2 to 5 are TP-1
one GPU each, so both entries are one TP-2 replica beside two TP-1 replicas with
the TP-2 replica on the opposite half. Every boundary therefore moves all three
slots and no transition degenerates to empty.

| boundary | from | to | latency |
|---|---|---|---|
| version 1 | slots [0,4,5] | slots [1,2,3] | 5.49 s |
| version 2 | slots [1,2,3] | slots [0,4,5] | 4.81 s |
| version 3 | slots [0,4,5] | slots [1,2,3] | 5.08 s |

Three boundaries, three switches, zero `replan failed`, and
`Training completed successfully`. Latency sits in the same 5 s band as the
homogeneous switches, which is expected because three slots move either way.

**Both degrees carry real traffic at the same time.** Per-slot request counts,
summed over each model version as `num_running_reqs + num_waiting_reqs`:

| slot | TP | v0 | v1 | v2 |
|---|---|---|---|---|
| r0 | 2 | 5658, peak 43 | 0 | 269 |
| r4 | 1 | 10824, peak 43 | 0 | 698 |
| r5 | 1 | 10356, peak 42 | 0 | 490 |
| r1 | 2 | 0 | 2175, peak 22 | 0 |
| r2 | 1 | 0 | 2243, peak 21 | 0 |
| r3 | 1 | 0 | 2251, peak 21 | 0 |

In v0 the TP-2 slot and both TP-1 slots are busy together and all three reach the
same concurrency ceiling, and v1 shows the same for the opposite triple. Sleeping
slots report exactly 0 rather than going silent, which keeps `status_loop`'s
unknown-index invariant intact.

The TP-1 slots accumulate roughly twice the requests of the TP-2 slot while
peaking at the same concurrency. That is `request_num_balance` doing its job: a
TP-2 replica finishes a request sooner, so it turns over faster at equal
occupancy. Nothing in the router needed to know that the two degrees differ.

**A dirty cluster breaks the trainer before any of this is reached.** The first
attempt at this run died in `init_model` at
`megatron_utils.py:646 offload_tensor_to_cpu`, a `non_blocking=True` pinned copy,
with `CUDA error: invalid argument`. Leftover processes from an earlier run held
13.6 GiB on one pool GPU and stale contexts on the trainer GPUs. The identical
script and config succeeded immediately after a clean restart, so this is a
stale-state failure rather than anything to do with heterogeneous parallelism.
Clear the cluster before a run rather than reading this crash as a code fault.

## Build time is the real cost

| TP | slots | build (s) |
|---|---|---|
| 8 | 1 | 299 |
| 4 | 2 | 434 |
| 2 | 4 | 881 |
| 1 | 8 | 598 |

**Total bootstrap for the full set: 2211 s, about 37 minutes.** Staged bootstrap
builds slots one at a time by construction, so this grows with the slot count and
is the dominant practical objection to a wide degree set. It is a one-time cost
per job rather than per step, and it is what the `tp_degrees` default of `[1, 2]`
exists to contain. A `{1,2}` set is 12 slots against 15, and skipping TP-8 and
TP-4 removes the two cheapest stages, so expect roughly 25 minutes.

Per-slot build time is not monotonic in slot count (TP-2's 4 slots took longer
than TP-1's 8), so treat these as order-of-magnitude rather than a model.

## tau(t, b) and tput(t, b) lookup tables

Reproduce with `psrl/bench/dynamic_parallel/decode_cost_probe.py`. PAPER.md §9.1
asks for these as measured tables rather than a closed form. Per-replica decode of
128 tokens from a 512-token prompt, best of 3.

`tau(t, b)`, per-token decode latency in ms, graphs on:

| b | TP1 | TP2 | TP4 |
|---|---|---|---|
| 1 | 6.61 | 4.47 | 3.02 |
| 8 | 9.93 | 6.46 | 4.15 |
| 32 | 21.67 | 12.56 | 7.67 |
| 128 | 76.46 | 41.58 | 24.05 |

Per-replica throughput, tok/s, graphs on:

| b | TP1 | TP2 | TP4 |
|---|---|---|---|
| 1 | 151 | 224 | 331 |
| 8 | 806 | 1239 | 1927 |
| 32 | 1476 | 2548 | 4174 |
| 128 | 1674 | 3078 | 5322 |

`tau` falls monotonically as TP rises at every batch size, which is the premise
§9.2 rests on: high TP is the low-latency choice. Aggregate throughput on 8 GPUs
is per-replica times `8/t`, so TP1 gives 13392 tok/s at b=128 against TP4's
10644, confirming low TP is the throughput choice. **Both halves of the §9.2
asymmetry hold.**

For reference, the eager numbers are also archived under
`results/decode_tp*_eager.json`. Eager inverts the `tau` ordering below b=32,
making high TP worse on both axes, which is why slots should run with graphs. The
eager tables are useful only as a floor, not as the configuration to ship.
