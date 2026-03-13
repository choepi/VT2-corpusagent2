from __future__ import annotations

import os
import platform
import random
import subprocess
from typing import Any

import numpy as np


_ALLOWED_DEVICES = {"auto", "cpu", "cuda", "mps"}
_ALLOWED_RUN_MODES = {"debug", "full"}


def resolve_run_mode(default: str = "debug") -> str:
    requested = os.getenv("CORPUSAGENT2_MODE", "").strip().lower()
    if requested in _ALLOWED_RUN_MODES:
        return requested

    normalized_default = (default or "debug").strip().lower()
    if normalized_default in _ALLOWED_RUN_MODES:
        return normalized_default
    return "debug"


def set_global_seed(seed: int) -> None:
    """Set deterministic seeds across common Python ML libraries."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        # Torch is optional in some scripts.
        pass


def _torch_runtime_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "torch_import_ok": False,
        "torch_version": None,
        "torch_cuda_version": None,
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_devices": [],
        "mps_available": False,
        "torch_error": None,
    }

    try:
        import torch

        report["torch_import_ok"] = True
        report["torch_version"] = getattr(torch, "__version__", None)
        report["torch_cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)

        cuda_available = bool(torch.cuda.is_available())
        report["cuda_available"] = cuda_available

        if cuda_available:
            device_count = int(torch.cuda.device_count())
            report["cuda_device_count"] = device_count
            devices: list[str] = []
            for idx in range(device_count):
                try:
                    devices.append(str(torch.cuda.get_device_name(idx)))
                except Exception:
                    devices.append(f"cuda:{idx}")
            report["cuda_devices"] = devices

        try:
            report["mps_available"] = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
        except Exception:
            report["mps_available"] = False
    except Exception as exc:
        report["torch_error"] = str(exc)

    return report


def _nvidia_smi_runtime_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "nvidia_smi_found": False,
        "nvidia_smi_ok": False,
        "nvidia_smi_output": None,
    }

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
        report["nvidia_smi_found"] = True
        report["nvidia_smi_ok"] = result.returncode == 0
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        report["nvidia_smi_output"] = stdout if stdout else stderr
    except Exception:
        report["nvidia_smi_found"] = False
        report["nvidia_smi_ok"] = False
        report["nvidia_smi_output"] = None

    return report


def resolve_device(preferred: str | None = None, precomputed_report: dict[str, Any] | None = None) -> str:
    requested = (preferred or "").strip().lower()
    env_override = os.getenv("CORPUSAGENT2_DEVICE", "").strip().lower()

    if env_override:
        requested = env_override

    if not requested:
        requested = "auto"

    if requested not in _ALLOWED_DEVICES:
        requested = "auto"

    report = precomputed_report or _torch_runtime_report()
    cuda_available = bool(report.get("cuda_available", False))
    mps_available = bool(report.get("mps_available", False))

    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        return "cuda" if cuda_available else "cpu"
    if requested == "mps":
        return "mps" if mps_available else "cpu"

    if cuda_available:
        return "cuda"
    if mps_available:
        return "mps"
    return "cpu"


def hf_pipeline_device_arg(device: str) -> int | str:
    normalized = (device or "cpu").strip().lower()
    if normalized == "cuda":
        return 0
    if normalized == "mps":
        return "mps"
    return -1


def runtime_device_report() -> dict[str, Any]:
    torch_report = _torch_runtime_report()
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "env_override": os.getenv("CORPUSAGENT2_DEVICE", "").strip().lower(),
        **torch_report,
        **_nvidia_smi_runtime_report(),
    }
    report["recommended_device"] = resolve_device("auto", precomputed_report=torch_report)
    return report
