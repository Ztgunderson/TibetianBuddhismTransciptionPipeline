"""tegrastats wrapper — attaches hardware telemetry (power, memory, GPU/CPU freq) to a run.

This is the edge-specific data the paper leans on: measured watts and MB per quant level, not
theoretical. Every full sweep is wrapped so each experiment_id has a telemetry log beside its
results. Also used as the GPU-execution sanity check — a run with all-zero GPU rows means the
model silently fell back to the CPU-only host torch (see the platform note in the plan).
"""

from __future__ import annotations

import contextlib
import subprocess
from pathlib import Path


@contextlib.contextmanager
def tegrastats(log_path: str | Path, interval_ms: int = 500):
    """Run tegrastats for the duration of the block, logging to `log_path`.

    No-ops gracefully (yields, logs a note) if tegrastats is unavailable — e.g. when developing
    the harness off-Jetson — so tests and dry runs don't require the binary.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = None
    try:
        proc = subprocess.Popen(
            ["tegrastats", "--interval", str(interval_ms), "--logfile", str(log_path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        log_path.write_text("# tegrastats unavailable on this host — no telemetry captured\n")
    try:
        yield log_path
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


def lock_clocks() -> bool:
    """Pin CPU/GPU clocks so RTF is comparable run-to-run. Returns True on success.

    Must be run before any Phase-1 benchmark session (needs sudo). Reset afterwards with
    `jetson_clocks --restore`. Returns False (with no exception) if unavailable, so the runner
    can warn rather than crash.
    """
    try:
        subprocess.run(["sudo", "jetson_clocks"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
