# Findings

Verified behavior of the collection path, and the traps worth knowing before a
large run. Reproduce with `examples/sciaccel_rl/batch_rollout_qwen35_4b.sh`.

## Verified

| Backend | Setup | Result |
|---|---|---|
| `smg_local` | 2 Qwen3.5-4B replicas at tp=2, `dump_tokens=true` | 23 trajectories, 11434 response tokens, token payloads length-consistent, graded 1.0 |
| `openai_api` | `openai/gpt-oss-120b`, 2 nodes, 4 workers | 8 tasks, 8/8 graded 1.0 |

Containers were reclaimed after every run. On `smg_local` the token payload is
measured rather than reconstructed: TITO records what the engines processed, so
`prompt_ids`, `response_ids`, `response_mask`, and `logprobs` all come from the
serving path. The parameter server is genuinely optional, so the collection
config root carries 11 `psrl` keys against the RL root's 29.

## `trajectory_id_strategy` must be `auto`

Under the `manual` default an external harness never sends the trajectory
header, every turn misses the prefix lookup, and TITO keeps only the final turn.
A run recorded `num_turns=1` with 558 tokens for an episode that had really run
many turns, and it still graded 1.0, which is what makes the loss easy to miss.
With `auto` the same task yields 18 turns and 5650 tokens.

Note that `auto` forks one trajectory per turn rather than chaining them.
`psrl.agentic_rl.thinking_template` selects the retention policy over the forks.

## A hosted endpoint may answer in `reasoning_content`

Some reasoning models put the answer there and leave `content` empty. An agent
harness reads `content`, sees nothing, takes no action, and burns the episode
while the API reports 200 with hundreds of completion tokens.

No config prevents this. `thinking_template`'s knobs are SMG and vLLM
conventions, and a third-party gateway drops fields it does not recognise, so
the mode only selects the retention policy on that path. Rewriting the response
is worse: the same provider uses `reasoning_content` for both deliberation and
the answer, so any rewrite has to guess.

Check a transcript before committing a large run to a new endpoint:

```bash
python -c "
import json, glob
for f in sorted(glob.glob('<output_dir>/transcripts/*.json')):
    turns = json.load(open(f))['turns']
    msg = lambda t: ((t.get('response') or {}).get('choices') or [{}])[0].get('message', {})
    print(f, len(turns),
          sum(1 for t in turns if msg(t).get('content')),
          sum(1 for t in turns if msg(t).get('reasoning_content')))"
```

An empty `<output_dir>/logs/trajectories/v0/*.txt` (about 190 bytes, all token
counts zero) is the same symptom seen from the other end.

## All-zero scores with `finished` episodes usually means quota

A rate-limited endpoint still lets every episode finish and be graded, which is
exactly how exhaustion disguises itself as model failure. Count HTTP 200s per
transcript before concluding anything about the model:

```bash
python -c "
import json, glob
for f in sorted(glob.glob('<output_dir>/transcripts/*.json')):
    turns = json.load(open(f))['turns']
    print(f, len(turns), sum(1 for t in turns if t['status'] == 200))"
```

`examples/sciaccel_rl/novita_models.json` records which models one key could
reach, with prices. Probe with a 1-token request per model, because a catalog
lists far more than a key is entitled to.

## Concurrency is bounded by CPU, not disk

Container layers are shared read-only, so N containers of one image do not cost
N times the image. Measured on a 384-core node: 24 containers of a 176 MB image
consumed 10 MB of disk in total, about 0.4 MB each, because only writes land in
the writable layer.

Idle containers are close to free. 128 of them held start latency flat at about
0.4 s and `docker ps` at 0.02 s. CPU-bound ones are what cost: 64 running a busy
numpy loop drove load to 58 of 384 cores, still fully responsive.

So size `max_concurrent_episodes` against cores and the task's own `cpus`
declaration, not against disk. One episode holds one agent container, and then
one verifier container after it under `environment_mode = "separate"`.

## Resource limits fail in three different ways

| Limit | Exceeded | Consequence |
|---|---|---|
| `mem_limit` | Kernel OOM-kills the container, exit 137, `OOMKilled: true` | A solver that briefly peaks dies mid-run and the verifier sees no output |
| `cpus` | Throttled to the ceiling, never killed | The task just takes longer, so a tight agent timeout turns into a spurious failure |
| writable layer | Not capped at all | A runaway write fills the shared docker filesystem for every other container |

`--storage-opt size` is rejected under `fuse-overlayfs`, so a per-container disk
quota is unavailable on this driver. Bound writes through the task itself.

Memory is the dangerous one: it is a hard ceiling, so set it above the real peak
or leave `memory_mb_override` at 0 and trust the task declaration.
