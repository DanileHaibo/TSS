from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass, field


@dataclass
class GpuSample:
    timestamp: float
    utilization_gpu_pct: float
    memory_used_mib: float
    memory_total_mib: float


@dataclass
class GpuMonitor:
    gpu_id: int = 0
    interval_s: float = 0.2
    samples: list[GpuSample] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, float | None]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        return self.summary()

    def _run(self) -> None:
        query = "utilization.gpu,memory.used,memory.total"
        cmd = [
            "nvidia-smi",
            f"--id={self.gpu_id}",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
                first = out.splitlines()[0]
                util, used, total = [float(x.strip()) for x in first.split(",")]
                self.samples.append(
                    GpuSample(
                        timestamp=time.time(),
                        utilization_gpu_pct=util,
                        memory_used_mib=used,
                        memory_total_mib=total,
                    )
                )
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def summary(self) -> dict[str, float | None]:
        if not self.samples:
            return {
                "gpu_utilization_mean_pct": None,
                "gpu_utilization_peak_pct": None,
                "gpu_memory_peak_mib": None,
                "gpu_memory_total_mib": None,
            }
        utils = [s.utilization_gpu_pct for s in self.samples]
        mem = [s.memory_used_mib for s in self.samples]
        return {
            "gpu_utilization_mean_pct": sum(utils) / len(utils),
            "gpu_utilization_peak_pct": max(utils),
            "gpu_memory_peak_mib": max(mem),
            "gpu_memory_total_mib": self.samples[-1].memory_total_mib,
        }

