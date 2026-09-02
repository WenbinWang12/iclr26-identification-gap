"""Non-preemptive GPU queue for the shared box.

Polls ``nvidia-smi`` and runs a fixed list of stages as soon as enough memory
is genuinely free, then keeps going through the list.  Design constraints, all
of which come from the box being shared with other people's jobs:

* It never signals, kills, or inspects another user's process.  The only
  decision input is free memory reported by ``nvidia-smi``.
* Free memory must stay above the threshold for ``--hold-seconds`` before a
  launch, so a checkpoint-write dip or a job that is still ramping up does not
  look like an opening.
* Memory is re-checked immediately before every stage, not once at the start,
  because another user can claim the card while an earlier stage runs.
* CPU-only stages declare ``gpu_gib=0`` and run without waiting.
* Every stage is logged with its exit code; a failing stage does not silently
  advance the ones that depend on it (``requires``).

Stages are declared in a JSON file so the queue can be re-pointed without
editing this module.  Each entry:

    {"name": "...", "cmd": "...", "gpu_gib": 12, "requires": ["earlier-name"],
     "timeout_seconds": 7200}
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    print("[%s] %s" % (stamp, message), flush=True)


def gpu_free_gib() -> list[float]:
    """Per-card free memory in GiB, or [] if nvidia-smi is unavailable."""

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        ).stdout
    except Exception as error:  # noqa: BLE001 - driver hiccup, treat as busy
        log("nvidia-smi failed: %s" % error)
        return []
    values = []
    for line in out.strip().splitlines():
        line = line.strip()
        if line:
            values.append(float(line) / 1024.0)
    return values


def pick_card(need_gib: float) -> int | None:
    """Index of the emptiest card with at least ``need_gib`` free, else None."""

    free = gpu_free_gib()
    if not free:
        return None
    best = max(range(len(free)), key=lambda i: free[i])
    return best if free[best] >= need_gib else None


def wait_for_card(need_gib: float, *, hold_seconds: int, poll_seconds: int,
                  deadline: float | None) -> int | None:
    """Block until one card holds ``need_gib`` free for ``hold_seconds``.

    Returns the card index, or None if ``deadline`` passes first.  The hold
    requirement is what makes this safe on a shared box: a transient dip in
    another job's allocation (checkpoint write, eval phase) will not be
    mistaken for the job having finished.
    """

    stable_since: float | None = None
    stable_card: int | None = None
    last_report = 0.0
    while True:
        if deadline is not None and time.time() > deadline:
            log("wait deadline exceeded for need=%.1f GiB" % need_gib)
            return None
        card = pick_card(need_gib)
        now = time.time()
        if card is None:
            stable_since, stable_card = None, None
        elif stable_card != card:
            stable_since, stable_card = now, card
        if card is not None and stable_since is not None:
            held = now - stable_since
            if held >= hold_seconds:
                log("card %d free for %.0fs (need %.1f GiB) -> claiming" % (card, held, need_gib))
                return card
        if now - last_report > 600:
            free = gpu_free_gib()
            log("waiting: need %.1f GiB, free per card = %s"
                % (need_gib, [round(v, 1) for v in free]))
            last_report = now
        time.sleep(poll_seconds)


def run_stage(stage: dict, log_dir: Path, card: int | None) -> int:
    """Run one stage synchronously, streaming its output to its own log."""

    name = stage["name"]
    log_path = log_dir / ("%s.log" % name)
    env = dict(os.environ)
    # huggingface.co is unroutable from this box; every stage needs the mirror
    # and the shared-disk cache, so set them here rather than per stage.
    env.setdefault("HF_HOME", "/mnt/data/wenbin/iclr26/models/hf")
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    if card is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(card)
    log("start %s (card=%s) -> %s" % (name, card, log_path))
    started = time.time()
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n===== %s start %s card=%s =====\n"
                     % (name, time.strftime("%Y-%m-%dT%H:%M:%S"), card))
        handle.flush()
        try:
            code = subprocess.run(
                stage["cmd"],
                shell=True,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=env,
                timeout=stage.get("timeout_seconds", 24 * 3600),
            ).returncode
        except subprocess.TimeoutExpired:
            handle.write("\n[queue] stage timed out\n")
            code = 124
    log("done  %s exit=%d elapsed=%.1f min" % (name, code, (time.time() - started) / 60.0))
    return code


def write_state(state_path: Path, results: dict[str, int], record: dict[str, dict]) -> None:
    """Persist every known success, not just the stages attempted this run.

    ``record`` holds only this run's attempts.  Writing that alone would erase
    the record of stages skipped because they had already succeeded, so a later
    restart would redo them.  Successes carried in ``results`` are merged back.
    """

    merged = dict(record)
    for name, code in results.items():
        if code == 0 and name not in merged:
            merged[name] = {"exit": 0, "carried": True}
    state_path.write_text(
        json.dumps({"stages": merged, "updated": time.strftime("%Y-%m-%dT%H:%M:%S")},
                   indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", required=True, help="JSON file describing stages")
    parser.add_argument("--log-dir", default="/mnt/data/wenbin/iclr26/logs/queue")
    parser.add_argument("--state", default="/mnt/data/wenbin/iclr26/logs/queue/state.json")
    parser.add_argument("--hold-seconds", type=int, default=180)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-wait-hours", type=float, default=24.0)
    args = parser.parse_args()

    stages = json.loads(Path(args.stages).read_text(encoding="utf-8"))
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    state_path = Path(args.state)

    results: dict[str, int] = {}
    if state_path.exists():
        try:
            results = {
                k: int(v["exit"])
                for k, v in json.loads(state_path.read_text(encoding="utf-8")).get("stages", {}).items()
                if v.get("exit") == 0
            }
            if results:
                log("resuming, already-succeeded stages: %s" % sorted(results))
        except Exception as error:  # noqa: BLE001 - corrupt state, start clean
            log("could not read state (%s), starting clean" % error)

    record: dict[str, dict] = {}
    log("queue up, %d stages, hold=%ds poll=%ds"
        % (len(stages), args.hold_seconds, args.poll_seconds))

    for stage in stages:
        name = stage["name"]
        if results.get(name) == 0:
            log("skip %s (already succeeded)" % name)
            continue

        missing = [dep for dep in stage.get("requires", []) if results.get(dep) != 0]
        if missing:
            log("SKIP %s: unmet requirements %s" % (name, missing))
            record[name] = {"exit": -1, "reason": "unmet requirements %s" % missing}
            write_state(state_path, results, record)
            continue

        need = float(stage.get("gpu_gib", 0))
        card = None
        if need > 0:
            deadline = time.time() + args.max_wait_hours * 3600
            card = wait_for_card(
                need,
                hold_seconds=args.hold_seconds,
                poll_seconds=args.poll_seconds,
                deadline=deadline,
            )
            if card is None:
                log("SKIP %s: no card became free within %.1f h" % (name, args.max_wait_hours))
                record[name] = {"exit": -1, "reason": "no free card"}
                write_state(state_path, results, record)
                continue

        code = run_stage(stage, log_dir, card)
        results[name] = code
        record[name] = {"exit": code, "card": card,
                        "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
        write_state(state_path, results, record)

    ok = sum(1 for v in record.values() if v.get("exit") == 0)
    log("QUEUE_DONE ok=%d of %d attempted" % (ok, len(record)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
