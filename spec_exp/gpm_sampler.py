from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass, field


GPM_METRICS = "1,2,10"


@dataclass
class GpmDmonSampler:
    gpu_id: int = 0
    interval_s: float = 1.0
    samples: list[dict[str, float]] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _proc: subprocess.Popen[str] | None = field(default=None, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _t0: float = 0.0

    def start(self) -> None:
        self.samples.clear()
        self._t0 = time.time()
        self._stop.clear()
        delay = max(1, int(round(self.interval_s)))
        cmd = [
            "nvidia-smi",
            "dmon",
            "-i",
            str(self.gpu_id),
            "-d",
            str(delay),
            "-s",
            "u",
            "--gpm-metrics",
            GPM_METRICS,
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            row = parse_dmon_line(line)
            if row:
                row["t_abs"] = time.time()
                row["t_sec"] = row["t_abs"] - self._t0
                self.samples.append(row)

    def stop(self) -> None:
        self._stop.set()
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread:
            self._thread.join(timeout=2)

    def phase_stats(self) -> dict[str, float | None]:
        keys = ("sm_activity_gpm_pct", "dram_activity_gpm_pct")
        out: dict[str, float | None] = {"sample_count": float(len(self.samples))}
        for k in keys:
            vals = [s[k] for s in self.samples if k in s]
            if vals:
                out[f"{k}_mean"] = sum(vals) / len(vals)
                out[f"{k}_peak"] = max(vals)
        return out

    def bucket_per_second(self) -> list[dict]:
        """One row per elapsed second: mean SM / DRAM activity."""
        if not self.samples:
            return []
        max_sec = int(max(s["t_sec"] for s in self.samples))
        rows: list[dict] = []
        for sec in range(max_sec + 1):
            bucket = [s for s in self.samples if int(s["t_sec"]) == sec]
            if not bucket:
                continue
            sm = sum(s["sm_activity_gpm_pct"] for s in bucket) / len(bucket)
            dram = sum(s["dram_activity_gpm_pct"] for s in bucket) / len(bucket)
            rows.append(
                {
                    "t_sec": float(sec),
                    "sm_activity_mean_pct": sm,
                    "dram_activity_mean_pct": dram,
                }
            )
        return rows


def parse_dmon_line(line: str) -> dict[str, float] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split()
    if len(parts) < 10 or not parts[0].isdigit():
        return None
    try:
        gract, smutil, dram = parts[-3], parts[-2], parts[-1]
        if smutil == "-":
            return None
        return {
            "gract_gpm_pct": float(gract),
            "sm_activity_gpm_pct": float(smutil),
            "dram_activity_gpm_pct": float(dram),
            "gpu_util_pct": float(parts[1]),
            "mem_util_pct": float(parts[2]),
        }
    except (ValueError, IndexError):
        return None


def normalize_stats(stats: dict) -> dict:
    """Unified column names for CSV/plots."""
    return {
        "sm_activity_mean_pct": stats.get("sm_activity_gpm_pct_mean"),
        "dram_activity_mean_pct": stats.get("dram_activity_gpm_pct_mean"),
        "sm_activity_peak_pct": stats.get("sm_activity_gpm_pct_peak"),
        "dram_activity_peak_pct": stats.get("dram_activity_gpm_pct_peak"),
        "sample_count": stats.get("sample_count"),
    }
