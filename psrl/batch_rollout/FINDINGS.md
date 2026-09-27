# Findings

## First end-to-end run against a hosted API (2026-09-24)

Reproduce with `examples/sciaccel_rl/batch_rollout_qwen35_4b.sh`, after
`examples/sciaccel_rl/check_api.sh` confirms the endpoint answers.

| Setting | Value |
|---|---|
| Model | `cohere/north-mini-code:free` via OpenRouter |
| Dataset | `data/mitgcm-atmos/repair_easy/train/L1.parquet` |
| Workers | 2 nodes, 2 concurrent episodes each |

| Run | Tasks | `max_turns` | Terminations | Graded | Mean score |
|---|---|---|---|---|---|
| A | 2 | 12 | 2 `max_turns_exceeded` | 2/2 | 0.500 |
| B | 8 | 40 | 8 `finished` | 8/8 | 0.000 |

Run A proved the pipeline: one task reached `reward_repair=1.0` with
`equivalence_pass=1`, meaning the verifier compiled the patched source and reran
the decks. Containers were fully reclaimed after both runs.

A later run on a fresh daily quota, 40 turns allowed, scored **2/2 at 1.0** with
`equivalence_pass=1` on both (23 and 37 turns). So the pipeline and the model both
work on this task family, and the earlier zeros were purely the request cap.

**Run B's zeros measure the API quota, not the model.** The OpenRouter free tier
is 50 requests per day and run A had already spent most of it, so each episode got
only 2 to 7 real answers before the rest returned 429. Every episode still
finished and was graded, which is exactly how quota exhaustion disguises itself.
Diagnose by counting HTTP 200s per transcript:

```bash
python -c "
import json, glob
for f in sorted(glob.glob('<output_dir>/transcripts/*.json')):
    turns = json.load(open(f))['turns']
    print(f, len(turns), sum(1 for t in turns if t['status'] == 200))"
```

A meaningful accuracy number needs a paid tier or a local vLLM fleet. Run A's
0.500 is two samples and should not be read as a rate.

## Free-model availability

Of the 20 `:free` OpenRouter models listed on that day, 3 answered a bash-emission
probe. `qwen3-coder:free` and `deepseek-chat-v3-0324:free` are retired, and the
rest returned 429 or 503. Reasoning-first models emit a thinking preamble rather
than a parseable command, which makes them poor agent backends even when up.

| Model | Behavior |
|---|---|
| `cohere/north-mini-code:free` | Clean fenced bash, fast. Used above. |
| `nex-agi/nex-n2.5-mini:free` | Clean fenced bash. |
| `poolside/laguna-xs-2.1:free` | Fenced block, unlabeled. |

## `smg_local` verified (2026-09-24)

2 tasks, 2 replicas of Qwen3.5-4B at tp=2 on 4 GPUs, `dump_tokens=true`.

| Metric | Value |
|---|---|
| Records | 26 trajectories from 2 episodes, all `finished` |
| Tokens | 8604 response tokens, masks and logprobs length-consistent |
| Score | `reward_repair=1.0` on both tasks, `equivalence_pass=1` |
| Elapsed | 747 s including engine startup |

The token payload is measured rather than reconstructed: TITO records what the
engines processed, so `prompt_ids`, `response_ids`, `response_mask`, and
`logprobs` all come from the serving path.

**`trajectory_id_strategy` must be `auto`.** Under the `manual` default an
external harness never sends the trajectory header, every turn misses the prefix
lookup, and TITO keeps only the final turn. The first run recorded `num_turns=1`
with 558 tokens for an episode that had really run many turns, and it still
graded 1.0, which is what makes the loss easy to miss. With `auto` the same task
yields 18 turns and 5650 tokens.

Note that `auto` forks one trajectory per turn rather than chaining them into one.
`psrl.agentic_rl.thinking_template` selects the retention policy over those forks.

## Re-verified after making the RL subsystems optional (2026-09-24)

The first `smg_local` run needed four inert config groups (`psrl.tms`,
`psrl.nixl`, `psrl.lmcache`, `psrl.ps_mode` plus `psrl.ps_manager_ip`) purely so
attribute access would resolve. Those reads are now optional in
`vllm_async_server.py`, and the engine honors a `None` status endpoint as the
`GenInterface` docstring always claimed.

Same run with all five removed from the config root: 23 trajectories, 11434
response tokens, token payloads length-consistent, `reward_repair=1.0`, containers
reclaimed, 898 s. The collection config root is back to 11 `psrl` keys against the
RL root's 29.

## Novita and reasoning-only replies (2026-09-24)

`examples/sciaccel_rl/novita_models.json` records which models a key can reach,
with prices. Probe it with a 1-token request per model, because the catalog lists
far more than a key is entitled to: 197 listed, 7 reachable, the rest
`403 MODEL_ACCESS_DENIED`.

Prices are Novita's unit, 1e-6 USD per million tokens. The cheapest reachable
model with an agentic-sized window was `openai/gpt-oss-20b` (400 in, 1500 out,
131k context).

**Reasoning models return the answer in `reasoning_content` and leave `content`
empty.** Agent harnesses read `content`, so they see nothing, take no action, and
burn the episode while the API reports 200 with hundreds of completion tokens.
Measured over one 4-task run: `reasoning_content` present on all 60 answered
turns, `content` non-empty on only 18.

**No config fixes this, and `thinking_template` in particular does not.** Its
knobs are SMG and vLLM conventions. Probed against Novita on `gpt-oss-120b`,
sending `separate_reasoning: False` plus
`chat_template_kwargs.enable_thinking: False` changed nothing: reasoning came
back at 184, 112, and 105 characters for baseline, both knobs, and
`reasoning_effort: low` respectively, with `content` identical throughout. A
third-party gateway drops fields it does not recognise.

Rewriting the response is worse than leaving it alone. It would have to guess
whether a given `reasoning_content` is deliberation or the answer, and one
provider uses it for both: in the same run some turns carried a bare JSON
command there while others carried "We need to output a JSON with ...".
Concatenating both feeds deliberation to the command parser.

So this is a model-selection question. `gpt-oss-120b` filled `content` on every
turn and scored 8/8; `gpt-oss-20b` filled 2 of 3 and scored 0. Check a
transcript before committing a large run to a new endpoint.

Diagnose by counting the two fields per transcript:

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
