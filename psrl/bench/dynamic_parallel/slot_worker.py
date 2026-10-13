"""One replica slot as a subprocess, driven over stdin and stdout.

Production runs each replica as its own Ray actor, so the probe runs each slot
as its own process rather than several engines in one interpreter. The parent
sets `CUDA_VISIBLE_DEVICES` to this slot's GPU range before exec, so the engine
sees its GPUs as 0 to tp-1.

Protocol: one JSON line per direction. The worker emits a ready line once the
engine is built, then replies to each `sleep`, `wake`, `stats`, or `exit`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

_EXTENSION_CLS = "psrl.bench.chunked_prefill.vllm_extension.ChunkedPrefillProbeExtension"


def _reply(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one replica slot for the oversubscription probe.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--gpu-util", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-seqs", type=int, default=256)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    try:
        from vllm import LLM

        started = time.perf_counter()
        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tp,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_util,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_seqs,
            enable_prefix_caching=False,
            enable_sleep_mode=True,
            # Mandatory. With CUDA graphs on, waking a second engine on the same
            # GPU dies in `capture_begin` with `use_count > 0 INTERNAL ASSERT
            # FAILED`, both in one process and across processes.
            enforce_eager=True,
            trust_remote_code=args.trust_remote_code,
            seed=0,
            worker_extension_cls=_EXTENSION_CLS,
        )
        build_seconds = time.perf_counter() - started

        def stats() -> dict[str, Any]:
            breakdown = llm.collective_rpc("get_memory_breakdown")[0]
            return {
                "num_gpu_blocks": breakdown.get("num_gpu_blocks", 0),
                "kv_token_capacity": breakdown.get("kv_token_capacity", 0),
                "weights_bytes": breakdown.get("weights_bytes", 0),
                "available_kv_bytes": breakdown.get("available_kv_bytes", 0),
            }

        _reply({"event": "ready", "pid": os.getpid(), "build_seconds": round(build_seconds, 2), **stats()})
    except Exception as exc:
        _reply({"error": f"{type(exc).__name__}: {exc}"})
        raise

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        command = json.loads(line).get("cmd")
        try:
            if command == "sleep":
                started = time.perf_counter()
                llm.sleep(level=2)
                _reply({"event": "asleep", "seconds": round(time.perf_counter() - started, 3)})
            elif command == "wake":
                started = time.perf_counter()
                llm.wake_up()
                elapsed = time.perf_counter() - started
                _reply({"event": "awake", "seconds": round(elapsed, 3), **stats()})
            elif command == "stats":
                _reply({"event": "stats", **stats()})
            elif command == "exit":
                _reply({"event": "exiting"})
                return
            else:
                _reply({"error": f"Unknown command {command!r}"})
        except Exception as exc:
            _reply({"error": f"{type(exc).__name__}: {exc}"})
            raise


if __name__ == "__main__":
    main()
