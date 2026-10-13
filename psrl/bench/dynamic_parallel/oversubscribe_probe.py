"""Milestone 0: measure what oversubscribing a GPU with sleeping engines costs.

Dynamic parallelism materializes every TP degree's replica slots at startup
because the NIXL shard-grid negotiation is a one-shot barrier. That puts
`len(tp_degrees)` vLLM engines on each GPU with all but one asleep at level 2.
Whether the awake engine still gets its full KV budget is not knowable from
reading code, so this probe measures it before the rest of the feature is built.

Bootstrap is staged in descending TP order: build one degree's engines, sleep
them, then the next. Each then profiles as if it were the sole occupant, because
vLLM freezes `num_gpu_blocks` during init and never re-profiles on wake.
Descending order is the realistic worst case, since a later TP-1 engine profiles
against whatever the larger engines left behind.

Each slot runs in its own subprocess, matching production where every replica is
a separate Ray actor. `enforce_eager` is mandatory: with CUDA graphs on, waking a
second engine on a GPU dies inside `capture_begin` with
`use_count > 0 INTERNAL ASSERT FAILED`, in the same process or across processes.
See FINDINGS.md.

Usage:

    # Control: one TP degree alone, for the baseline KV budget.
    PYTHONUNBUFFERED=1 python -m psrl.bench.dynamic_parallel.oversubscribe_probe \\
        --model /path/to/Qwen3-8B --tp-degrees 2 --gpus 8 \\
        --output results/control_tp2.json

    # Oversubscribed: every degree resident, one awake at a time.
    PYTHONUNBUFFERED=1 python -m psrl.bench.dynamic_parallel.oversubscribe_probe \\
        --model /path/to/Qwen3-8B --tp-degrees 1,2,4,8 --gpus 8 \\
        --switch-cycles 2 --output results/oversubscribed.json

Compare `num_gpu_blocks` per degree against its control run. The feature is
viable if the awake engine keeps its budget, and the degree set must narrow if
it does not.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import time
from typing import Any

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=os.getenv("PSRL_LOGGING_LEVEL", "INFO"),
)
psrl_logger = logging.getLogger(__name__)

_WORKER_MODULE = "psrl.bench.dynamic_parallel.slot_worker"


def gpu_used_mib() -> list[int]:
    """Per-GPU used memory straight from the driver.

    `torch.cuda.mem_get_info` only sees the calling process. A sleeping engine's
    cost lives in sibling processes and their CUDA contexts, so the driver's view
    is the one that matters.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        return [int(line.strip()) for line in out.stdout.splitlines() if line.strip()]
    except Exception as exc:
        psrl_logger.warning(f"Could not read nvidia-smi memory: {exc!r}.")
        return []


class SlotProcess:
    """One replica slot running as a subprocess, driven over stdin and stdout.

    The protocol is one JSON line per direction. Commands are `sleep`, `wake`,
    `stats`, and `exit`, which keeps the orchestrator free of CUDA state.
    """

    def __init__(self, tp: int, slot_index: int, devices: str, args) -> None:
        self.tp = tp
        self.slot_index = slot_index
        self.devices = devices
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = devices
        env["PYTHONUNBUFFERED"] = "1"
        self._proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                _WORKER_MODULE,
                "--model",
                args.model,
                "--tp",
                str(tp),
                "--gpu-util",
                str(args.gpu_util),
                "--max-model-len",
                str(args.max_model_len),
                "--max-seqs",
                str(args.max_seqs),
                "--dtype",
                args.dtype,
            ]
            + (["--trust-remote-code"] if args.trust_remote_code else []),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL if args.quiet_workers else None,
            env=env,
            text=True,
            bufsize=1,
        )

    @property
    def label(self) -> str:
        return f"tp{self.tp}/slot{self.slot_index}"

    def _read_until_reply(self) -> dict[str, Any]:
        """Read lines until one parses as the worker's JSON reply."""
        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError(f"Slot {self.label} exited before replying (rc={self._proc.poll()}).")
            line = line.strip()
            if not line.startswith("{"):
                continue
            reply = json.loads(line)
            if reply.get("error"):
                raise RuntimeError(f"Slot {self.label} failed: {reply['error']}.")
            return reply

    def wait_ready(self) -> dict[str, Any]:
        return self._read_until_reply()

    def command(self, name: str) -> dict[str, Any]:
        self._proc.stdin.write(json.dumps({"cmd": name}) + "\n")
        self._proc.stdin.flush()
        return self._read_until_reply()

    def close(self) -> None:
        try:
            self._proc.stdin.write(json.dumps({"cmd": "exit"}) + "\n")
            self._proc.stdin.flush()
            self._proc.wait(timeout=60)
        except Exception:
            self._proc.kill()


def _probe(args) -> dict[str, Any]:
    tp_degrees = sorted({int(t) for t in args.tp_degrees.split(",")}, reverse=True)
    for tp in tp_degrees:
        if tp > args.gpus:
            raise ValueError(f"TP degree {tp} exceeds the {args.gpus} GPUs given.")
        if args.gpus % tp != 0:
            raise ValueError(f"TP degree {tp} does not evenly tile {args.gpus} GPUs.")

    slots: dict[int, list[SlotProcess]] = {}
    stage_records: list[dict[str, Any]] = []

    try:
        # --- Staged bootstrap, descending TP ---
        for tp in tp_degrees:
            n_slots = args.gpus // tp
            started = time.perf_counter()
            built: list[SlotProcess] = []
            ready: list[dict[str, Any]] = []
            for slot_index in range(n_slots):
                devices = ",".join(str(slot_index * tp + i) for i in range(tp))
                proc = SlotProcess(tp, slot_index, devices, args)
                built.append(proc)
                ready.append(proc.wait_ready())
                psrl_logger.info(f"Built {proc.label} on GPUs {devices}: blocks={ready[-1]['num_gpu_blocks']}.")
            build_seconds = time.perf_counter() - started

            # Sleep this degree before building the next, so the next profiles
            # against freed physical pages rather than this degree's KV.
            sleep_started = time.perf_counter()
            for proc in built:
                proc.command("sleep")
            sleep_seconds = time.perf_counter() - sleep_started

            slots[tp] = built
            stage_records.append(
                {
                    "tp": tp,
                    "n_slots": n_slots,
                    "build_seconds": round(build_seconds, 2),
                    "sleep_seconds": round(sleep_seconds, 2),
                    "num_gpu_blocks_at_build": [r["num_gpu_blocks"] for r in ready],
                    "kv_token_capacity_at_build": [r["kv_token_capacity"] for r in ready],
                    "weights_bytes": ready[0]["weights_bytes"],
                    "gpu_used_mib_after_sleep": gpu_used_mib(),
                }
            )
            psrl_logger.info(
                f"Staged TP-{tp}: {n_slots} slots in {build_seconds:.1f}s, slept in {sleep_seconds:.1f}s, "
                f"blocks={stage_records[-1]['num_gpu_blocks_at_build']}."
            )

        resident_all_asleep = gpu_used_mib()
        total_engines = sum(len(v) for v in slots.values())
        psrl_logger.info(f"All {total_engines} engines asleep. GPU MiB: {resident_all_asleep}.")

        # --- Switch cycles ---
        switch_records: list[dict[str, Any]] = []
        awake_tp: int | None = None
        for cycle in range(args.switch_cycles):
            for tp in tp_degrees:
                started = time.perf_counter()
                if awake_tp is not None:
                    for proc in slots[awake_tp]:
                        proc.command("sleep")
                sleep_done = time.perf_counter()
                woke = [proc.command("wake") for proc in slots[tp]]
                wake_done = time.perf_counter()
                awake_tp = tp

                switch_records.append(
                    {
                        "cycle": cycle,
                        "tp": tp,
                        "sleep_seconds": round(sleep_done - started, 3),
                        "wake_seconds": round(wake_done - sleep_done, 3),
                        "switch_seconds": round(wake_done - started, 3),
                        "num_gpu_blocks_awake": [r["num_gpu_blocks"] for r in woke],
                        "gpu_used_mib_awake": gpu_used_mib(),
                    }
                )
                psrl_logger.info(
                    f"Cycle {cycle} TP-{tp}: switch {wake_done - started:.2f}s "
                    f"(sleep {sleep_done - started:.2f}s, wake {wake_done - sleep_done:.2f}s), "
                    f"blocks={switch_records[-1]['num_gpu_blocks_awake']}."
                )
    finally:
        for procs in slots.values():
            for proc in procs:
                proc.close()

    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": platform.node(),
        "gpu_count": args.gpus,
        "model_path": args.model,
        "tp_degrees": tp_degrees,
        "gpu_memory_utilization": args.gpu_util,
        "max_model_len": args.max_model_len,
        "dtype": args.dtype,
        "enforce_eager": True,
        "engines_per_gpu": len(tp_degrees),
        "total_engines": sum(args.gpus // tp for tp in tp_degrees),
        "experiment": "milestone0_oversubscribe",
        "stages": stage_records,
        "switches": switch_records,
        "gpu_used_mib_all_asleep": resident_all_asleep,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure the cost of oversubscribing GPUs with sleeping engines.")
    parser.add_argument("--model", required=True, help="Path to the HuggingFace model.")
    parser.add_argument(
        "--tp-degrees",
        default="1,2,4,8",
        help="Comma-separated TP degrees to materialize. Pass one degree for a control run.",
    )
    parser.add_argument("--gpus", type=int, default=8, help="GPUs in the rollout pool on this node.")
    parser.add_argument("--gpu-util", type=float, default=0.90, help="gpu_memory_utilization per engine.")
    parser.add_argument("--max-model-len", type=int, default=8192, help="Engine max_model_len.")
    parser.add_argument("--max-seqs", type=int, default=256, help="Engine max_num_seqs.")
    parser.add_argument("--dtype", default="bfloat16", help="Model dtype.")
    parser.add_argument(
        "--switch-cycles",
        type=int,
        default=2,
        help="Times to walk the whole degree set, waking one degree at a time.",
    )
    parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code when loading the model.")
    parser.add_argument("--quiet-workers", action="store_true", help="Discard worker stderr.")
    parser.add_argument("--output", default=None, help="Write the result JSON here instead of stdout.")
    args = parser.parse_args()

    result = _probe(args)
    payload = json.dumps(result, indent=2)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as handle:
            handle.write(payload)
        psrl_logger.info(f"Wrote probe result to {args.output}.")
    else:
        print(payload)


if __name__ == "__main__":
    main()
