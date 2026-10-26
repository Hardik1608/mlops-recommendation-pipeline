"""Small helpers shared by benchmark and load-test reports."""

from __future__ import annotations

import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path


def _cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _ram_gb() -> float | None:
    try:
        import psutil

        return round(psutil.virtual_memory().total / 1024**3, 1)
    except Exception:
        return None


def hardware_info() -> dict[str, object]:
    return {
        "cpu": _cpu_name(),
        "logical_cpus": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python": platform.python_version(),
        "recorded_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
    }
