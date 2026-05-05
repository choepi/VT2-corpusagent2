from __future__ import annotations

from dataclasses import dataclass
import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import sys


GIB = 1024**3
CPU_LARGE_HOST_THRESHOLD = 16
MEMORY_LARGE_HOST_THRESHOLD_BYTES = 96 * 1000**3
SMALL_HOST_FRACTION = 0.80
LARGE_HOST_FRACTION = 0.60

SERVICE_WEIGHTS = {
    "MCP": 0.45,
    "API": 0.20,
    "OPENSEARCH": 0.25,
    "POSTGRES": 0.10,
}

MIN_CPU_BY_SERVICE = {
    "MCP": 1.0,
    "API": 1.0,
    "OPENSEARCH": 1.0,
    "POSTGRES": 0.5,
}

MIN_MEMORY_BYTES_BY_SERVICE = {
    "MCP": 1 * GIB,
    "API": 1 * GIB,
    "OPENSEARCH": 2 * GIB,
    "POSTGRES": 1 * GIB,
}


@dataclass(frozen=True, slots=True)
class HostHardware:
    logical_cpus: int
    total_memory_bytes: int
    gpu_available: bool = False


@dataclass(frozen=True, slots=True)
class DockerResourcePlan:
    host: HostHardware
    cpu_fraction: float
    memory_fraction: float
    cpu_budget: float
    memory_budget_bytes: int
    use_gpu: bool
    env: dict[str, str]


def _allocation_fraction(value: float, threshold: float) -> float:
    return LARGE_HOST_FRACTION if value >= threshold else SMALL_HOST_FRACTION


def _format_cpu(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _format_memory_limit(value_bytes: float) -> str:
    mib = max(128, int(value_bytes / (1024**2)))
    return f"{mib}m"


def _weighted_split(
    budget: float,
    *,
    minimums: dict[str, float],
) -> dict[str, float]:
    if budget <= 0:
        return {key: 0.0 for key in SERVICE_WEIGHTS}

    min_total = sum(minimums.values())
    if budget < min_total:
        return {key: budget * weight for key, weight in SERVICE_WEIGHTS.items()}

    allocations = {
        key: max(budget * SERVICE_WEIGHTS[key], minimums[key])
        for key in SERVICE_WEIGHTS
    }
    overflow = sum(allocations.values()) - budget
    if overflow <= 0:
        return allocations

    reducible = {
        key: max(0.0, allocations[key] - minimums[key])
        for key in allocations
    }
    reducible_total = sum(reducible.values())
    if reducible_total <= 0:
        return allocations

    for key, value in reducible.items():
        allocations[key] -= overflow * (value / reducible_total)
    return allocations


def _detect_total_memory_bytes() -> int:
    if sys.platform.startswith("linux"):
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            for line in meminfo.read_text(encoding="utf-8").splitlines():
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024

    if sys.platform == "win32":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            return int(status.ullTotalPhys)

    page_size = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 0
    page_count = os.sysconf("SC_PHYS_PAGES") if hasattr(os, "sysconf") else 0
    if page_size and page_count:
        return int(page_size * page_count)
    return 8 * GIB


def detect_gpu_available() -> bool:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False
    try:
        completed = subprocess.run(
            [nvidia_smi, "-L"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
            check=False,
        )
    except Exception:
        return False
    return completed.returncode == 0 and bool(completed.stdout.strip())


def detect_host_hardware() -> HostHardware:
    return HostHardware(
        logical_cpus=max(1, os.cpu_count() or 1),
        total_memory_bytes=max(1, _detect_total_memory_bytes()),
        gpu_available=detect_gpu_available(),
    )


def compute_docker_resource_plan(
    host: HostHardware,
    *,
    gpu_mode: str = "auto",
) -> DockerResourcePlan:
    normalized_gpu_mode = gpu_mode.strip().lower()
    if normalized_gpu_mode not in {"auto", "on", "off"}:
        raise ValueError("gpu_mode must be one of: auto, on, off")

    cpu_fraction = _allocation_fraction(host.logical_cpus, CPU_LARGE_HOST_THRESHOLD)
    memory_gib = host.total_memory_bytes / GIB
    memory_fraction = _allocation_fraction(host.total_memory_bytes, MEMORY_LARGE_HOST_THRESHOLD_BYTES)
    cpu_budget = host.logical_cpus * cpu_fraction
    memory_budget_bytes = int(host.total_memory_bytes * memory_fraction)

    cpu_allocations = _weighted_split(cpu_budget, minimums=MIN_CPU_BY_SERVICE)
    memory_allocations = _weighted_split(
        float(memory_budget_bytes),
        minimums={key: float(value) for key, value in MIN_MEMORY_BYTES_BY_SERVICE.items()},
    )
    opensearch_mem = memory_allocations["OPENSEARCH"]
    heap_bytes = min(opensearch_mem * 0.50, 31 * GIB)
    heap_bytes = max(512 * 1024**2, min(heap_bytes, max(512 * 1024**2, opensearch_mem - 512 * 1024**2)))
    python_runner_cpus = min(2.0, max(0.5, cpu_allocations["MCP"] / 4))
    python_runner_memory = min(2 * GIB, max(512 * 1024**2, memory_allocations["MCP"] / 4))

    use_gpu = (
        True
        if normalized_gpu_mode == "on"
        else False
        if normalized_gpu_mode == "off"
        else host.gpu_available
    )

    env = {
        "CORPUSAGENT2_DOCKER_HW_LOGICAL_CPUS": str(host.logical_cpus),
        "CORPUSAGENT2_DOCKER_HW_MEMORY_GIB": f"{memory_gib:.2f}",
        "CORPUSAGENT2_DOCKER_HW_GPU_AVAILABLE": "true" if host.gpu_available else "false",
        "CORPUSAGENT2_DOCKER_USE_GPU": "true" if use_gpu else "false",
        "CORPUSAGENT2_DOCKER_CPU_FRACTION": _format_cpu(cpu_fraction),
        "CORPUSAGENT2_DOCKER_MEMORY_FRACTION": _format_cpu(memory_fraction),
        "CORPUSAGENT2_API_CPUS": _format_cpu(cpu_allocations["API"]),
        "CORPUSAGENT2_MCP_CPUS": _format_cpu(cpu_allocations["MCP"]),
        "CORPUSAGENT2_OPENSEARCH_CPUS": _format_cpu(cpu_allocations["OPENSEARCH"]),
        "CORPUSAGENT2_POSTGRES_CPUS": _format_cpu(cpu_allocations["POSTGRES"]),
        "CORPUSAGENT2_API_MEM_LIMIT": _format_memory_limit(memory_allocations["API"]),
        "CORPUSAGENT2_MCP_MEM_LIMIT": _format_memory_limit(memory_allocations["MCP"]),
        "CORPUSAGENT2_OPENSEARCH_MEM_LIMIT": _format_memory_limit(opensearch_mem),
        "CORPUSAGENT2_POSTGRES_MEM_LIMIT": _format_memory_limit(memory_allocations["POSTGRES"]),
        "CORPUSAGENT2_OPENSEARCH_RECOMMENDED_JAVA_OPTS": (
            f"-Xms{_format_memory_limit(heap_bytes)} -Xmx{_format_memory_limit(heap_bytes)}"
        ),
        "CORPUSAGENT2_PYTHON_RUNNER_CPUS": _format_cpu(python_runner_cpus),
        "CORPUSAGENT2_PYTHON_RUNNER_MEMORY": _format_memory_limit(python_runner_memory),
    }
    return DockerResourcePlan(
        host=host,
        cpu_fraction=cpu_fraction,
        memory_fraction=memory_fraction,
        cpu_budget=cpu_budget,
        memory_budget_bytes=memory_budget_bytes,
        use_gpu=use_gpu,
        env=env,
    )


def compose_files_for_stack(*, use_gpu: bool) -> list[str]:
    files = ["docker-compose.yml", "docker-compose.mcp.yml"]
    if use_gpu:
        files.append("docker-compose.mcp.gpu.yml")
    return files


def data_service_resource_updates(plan: DockerResourcePlan) -> dict[str, dict[str, str]]:
    return {
        "corpus_postgres": {
            "cpus": plan.env["CORPUSAGENT2_POSTGRES_CPUS"],
            "memory": plan.env["CORPUSAGENT2_POSTGRES_MEM_LIMIT"],
        },
        "os_news": {
            "cpus": plan.env["CORPUSAGENT2_OPENSEARCH_CPUS"],
            "memory": plan.env["CORPUSAGENT2_OPENSEARCH_MEM_LIMIT"],
        },
    }
