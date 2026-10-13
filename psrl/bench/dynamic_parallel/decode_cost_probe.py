"""Measure the decode cost of `enforce_eager`, which slots must pay to be switchable.

Oversubscription forces every slot to run eager, because waking a second engine on
a GPU dies in CUDA graph recapture (see FINDINGS.md). Eager decode is slower per
token, so the parallelism gain from switching TP degrees has to beat that loss.
This probe quantifies the loss.

It also measures `tau(t)` and `tput(t)` per TP degree, which is the lookup table
PAPER.md §9.1 asks for rather than a closed form. The numbers for a given TP
degree come from one engine, so aggregate throughput at that degree is the
per-replica figure times the replica count `G/t`.

Usage:

    # One (tp, mode) pair per process, because CUDA graph state is process-wide.
    PYTHONUNBUFFERED=1 python -m psrl.bench.dynamic_parallel.decode_cost_probe \\
        --model /path/to/Qwen3-8B --tp 2 --mode eager \\
        --batch-sizes 1,8,32,128 --output results/decode_tp2_eager.json

Compare `eager` against `graph` at the same TP and batch size for the penalty,
and across TP degrees at fixed mode for the latency-throughput tradeoff.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import time
from typing import Any

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=os.getenv("PSRL_LOGGING_LEVEL", "INFO"),
)
psrl_logger = logging.getLogger(__name__)


def _measure(llm, batch_size: int, prompt_len: int, output_len: int, repeats: int) -> dict[str, Any]:
    """Time a fixed decode workload at one batch size.

    Prompts are synthetic token ids rather than text so that every run decodes the
    same shape, and `ignore_eos` keeps the output length exact.
    """
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    sampling = SamplingParams(temperature=0.0, max_tokens=output_len, ignore_eos=True)
    prompts = [TokensPrompt(prompt_token_ids=[1] * prompt_len) for _ in range(batch_size)]

    # One untimed pass so lazy kernels and any graph capture land outside the measurement.
    llm.generate(prompts, sampling, use_tqdm=False)

    latencies: list[float] = []
    generated: list[int] = []
    for _ in range(repeats):
        started = time.perf_counter()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        latencies.append(time.perf_counter() - started)
        generated.append(sum(len(o.outputs[0].token_ids) for o in outputs))

    best = min(latencies)
    tokens = generated[0]
    return {
        "batch_size": batch_size,
        "prompt_len": prompt_len,
        "output_len": output_len,
        "repeats": repeats,
        "tokens_generated": tokens,
        "latency_seconds_min": round(best, 4),
        "latency_seconds_median": round(sorted(latencies)[len(latencies) // 2], 4),
        "latency_seconds_all": [round(x, 4) for x in latencies],
        # Per-replica decode throughput. Aggregate at TP t is this times G/t.
        "tokens_per_second": round(tokens / best, 1),
        # tau: wall-clock seconds per decode step, the quantity PAPER.md calls tau(t, b).
        "seconds_per_step": round(best / output_len, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure eager versus CUDA graph decode cost per TP degree.")
    parser.add_argument("--model", required=True, help="Path to the HuggingFace model.")
    parser.add_argument("--tp", type=int, required=True, help="Tensor parallel size.")
    parser.add_argument("--mode", choices=["eager", "graph"], required=True, help="Decode execution mode.")
    parser.add_argument("--batch-sizes", default="1,8,32,128", help="Comma-separated batch sizes to sweep.")
    parser.add_argument("--prompt-len", type=int, default=512, help="Synthetic prompt length in tokens.")
    parser.add_argument("--output-len", type=int, default=128, help="Tokens to decode per sequence.")
    parser.add_argument("--repeats", type=int, default=3, help="Timed repeats per batch size.")
    parser.add_argument("--gpu-util", type=float, default=0.90, help="gpu_memory_utilization.")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Engine max_model_len.")
    parser.add_argument("--dtype", default="bfloat16", help="Model dtype.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code when loading the model.")
    parser.add_argument("--output", default=None, help="Write the result JSON here instead of stdout.")
    args = parser.parse_args()

    from vllm import LLM

    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    enforce_eager = args.mode == "eager"

    psrl_logger.info(f"Building TP-{args.tp} engine in {args.mode} mode.")
    started = time.perf_counter()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_util,
        max_model_len=args.max_model_len,
        max_num_seqs=max(batch_sizes),
        enable_prefix_caching=False,
        enforce_eager=enforce_eager,
        trust_remote_code=args.trust_remote_code,
        seed=0,
    )
    build_seconds = time.perf_counter() - started

    measurements = []
    for batch_size in batch_sizes:
        record = _measure(llm, batch_size, args.prompt_len, args.output_len, args.repeats)
        measurements.append(record)
        psrl_logger.info(
            f"TP-{args.tp} {args.mode} b={batch_size}: {record['tokens_per_second']} tok/s, "
            f"tau={record['seconds_per_step'] * 1000:.2f} ms/step."
        )

    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": platform.node(),
        "model_path": args.model,
        "tensor_parallel_size": args.tp,
        "mode": args.mode,
        "enforce_eager": enforce_eager,
        "gpu_memory_utilization": args.gpu_util,
        "max_model_len": args.max_model_len,
        "dtype": args.dtype,
        "build_seconds": round(build_seconds, 2),
        "experiment": "decode_cost",
        "measurements": measurements,
    }
    payload = json.dumps(result, indent=2)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as handle:
            handle.write(payload)
        psrl_logger.info(f"Wrote decode cost result to {args.output}.")
    else:
        print(payload)


if __name__ == "__main__":
    main()
