from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARTITIONS = ("gpu", "gpu_ia", "gpu_top")
DEFAULT_SBATCH_SCRIPT = REPO_ROOT / "slurm" / "run_prebuilt_bundle.sbatch"


@dataclass(slots=True)
class JobProfile:
    name: str
    gpus: int
    preferred_cpus: int
    minimum_cpus: int
    preferred_mem_mb: int
    minimum_mem_mb: int
    preferred_time_minutes: int
    aggressive_time_minutes: int
    preferred_partitions: tuple[str, ...]
    default_account: str


@dataclass(slots=True)
class NodeInfo:
    name: str
    partitions: tuple[str, ...]
    gpu_model: str
    total_gpus: int
    allocated_gpus: int
    free_gpus: int
    cpu_total: int
    cpu_alloc: int
    free_cpus: int
    real_mem_mb: int
    alloc_mem_mb: int
    free_mem_mb: int


@dataclass(slots=True)
class Recommendation:
    partition: str
    node: str | None
    gpus: int
    cpus_per_task: int
    mem_mb: int
    time_minutes: int
    reason: str


PROFILES: dict[str, JobProfile] = {
    "prebuilt_bundle": JobProfile(
        name="prebuilt_bundle",
        gpus=1,
        preferred_cpus=8,
        minimum_cpus=4,
        preferred_mem_mb=32 * 1024,
        minimum_mem_mb=24 * 1024,
        preferred_time_minutes=4 * 60,
        aggressive_time_minutes=2 * 60,
        preferred_partitions=DEFAULT_PARTITIONS,
        default_account="cai_nlp",
    )
}


def _run(command: list[str]) -> str:
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    return completed.stdout


def _normalize_partition(value: str) -> str:
    return value.strip().rstrip("*")


def _format_mem_gb(mem_mb: int) -> str:
    return f"{max(mem_mb // 1024, 1)}G"


def _format_time(minutes: int) -> str:
    hours, mins = divmod(max(minutes, 1), 60)
    return f"{hours:02d}:{mins:02d}:00"


def _parse_mem_mb(value: str) -> int:
    stripped = value.strip()
    match = re.fullmatch(r"(?P<num>\d+)(?P<unit>[KMGTP]?)", stripped)
    if match is None:
        return 0
    number = int(match.group("num"))
    unit = match.group("unit")
    scale = {
        "": 1,
        "K": 1 // 1024,
        "M": 1,
        "G": 1024,
        "T": 1024 * 1024,
        "P": 1024 * 1024 * 1024,
    }
    if unit == "K":
        return max(number // 1024, 0)
    return number * scale[unit]


def _parse_sinfo(stdout: str) -> dict[str, set[str]]:
    node_partitions: dict[str, set[str]] = {}
    for raw_line in stdout.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        node_name, _, partition_name = stripped.partition("|")
        if not partition_name:
            continue
        node_name = node_name.strip()
        partition = _normalize_partition(partition_name)
        if not node_name or not partition:
            continue
        node_partitions.setdefault(node_name, set()).add(partition)
    return node_partitions


def _extract_tres_int(blob: str, prefix: str) -> int:
    pattern = re.compile(rf"(?:^|,){re.escape(prefix)}=(\d+)")
    match = pattern.search(blob)
    return int(match.group(1)) if match else 0


def _parse_node_details(name: str, partitions: Iterable[str], stdout: str) -> NodeInfo:
    cpu_alloc_match = re.search(r"CPUAlloc=(\d+)", stdout)
    cpu_total_match = re.search(r"CPUTot=(\d+)", stdout)
    real_mem_match = re.search(r"RealMemory=(\d+)", stdout)
    alloc_mem_match = re.search(r"AllocMem=(\d+)", stdout)
    free_mem_match = re.search(r"FreeMem=(\d+)", stdout)
    gres_match = re.search(r"Gres=gpu(?::([^:\s]+))?:(\d+)", stdout)
    cfg_tres_match = re.search(r"CfgTRES=([^\s]+)", stdout)
    alloc_tres_match = re.search(r"AllocTRES=([^\s]+)", stdout)

    gpu_model = ""
    total_gpus = 0
    if gres_match is not None:
        gpu_model = gres_match.group(1) or ""
        total_gpus = int(gres_match.group(2))
    elif cfg_tres_match is not None:
        total_gpus = _extract_tres_int(cfg_tres_match.group(1), "gres/gpu")

    allocated_gpus = 0
    if alloc_tres_match is not None:
        allocated_gpus = _extract_tres_int(alloc_tres_match.group(1), "gres/gpu")

    cpu_total = int(cpu_total_match.group(1)) if cpu_total_match else 0
    cpu_alloc = int(cpu_alloc_match.group(1)) if cpu_alloc_match else 0
    real_mem_mb = int(real_mem_match.group(1)) if real_mem_match else 0
    alloc_mem_mb = int(alloc_mem_match.group(1)) if alloc_mem_match else 0
    free_mem_mb = int(free_mem_match.group(1)) if free_mem_match else max(real_mem_mb - alloc_mem_mb, 0)

    return NodeInfo(
        name=name,
        partitions=tuple(sorted(set(partitions))),
        gpu_model=gpu_model,
        total_gpus=total_gpus,
        allocated_gpus=allocated_gpus,
        free_gpus=max(total_gpus - allocated_gpus, 0),
        cpu_total=cpu_total,
        cpu_alloc=cpu_alloc,
        free_cpus=max(cpu_total - cpu_alloc, 0),
        real_mem_mb=real_mem_mb,
        alloc_mem_mb=alloc_mem_mb,
        free_mem_mb=max(free_mem_mb, 0),
    )


def _load_nodes(partitions: tuple[str, ...]) -> list[NodeInfo]:
    sinfo_stdout = _run(["sinfo", "-h", "-p", ",".join(partitions), "-N", "-o", "%N|%P"])
    node_partitions = _parse_sinfo(sinfo_stdout)
    nodes: list[NodeInfo] = []
    for node_name, node_parts in sorted(node_partitions.items()):
        details = _run(["scontrol", "show", "node", node_name])
        nodes.append(_parse_node_details(node_name, node_parts, details))
    return nodes


def _load_fairshare(user: str, account: str) -> float | None:
    try:
        stdout = _run(["sshare", "-u", user, "-A", account])
    except Exception:
        return None

    fairshare_values: list[float] = []
    for raw_line in stdout.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("Account") or stripped.startswith("-"):
            continue
        columns = stripped.split()
        if len(columns) < 7:
            continue
        row_account = columns[0]
        row_user = columns[1] if len(columns) > 1 else ""
        fairshare_raw = columns[-1]
        if row_account != account or row_user != user:
            continue
        if fairshare_raw in {"", "inf", "nan"}:
            continue
        try:
            fairshare_values.append(float(fairshare_raw))
        except ValueError:
            continue
    if not fairshare_values:
        return None
    return min(fairshare_values)


def _profile_for(name: str) -> JobProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown profile: {name}") from exc


def _partition_rank(partition: str, profile: JobProfile) -> int:
    try:
        return len(profile.preferred_partitions) - profile.preferred_partitions.index(partition)
    except ValueError:
        return 0


def _node_model_rank(model: str) -> int:
    lowered = model.lower()
    if "h200" in lowered:
        return 6
    if "h100" in lowered:
        return 5
    if "a100" in lowered:
        return 4
    if "l40" in lowered:
        return 3
    if "v100" in lowered:
        return 2
    if model:
        return 1
    return 0


def _candidate_resources(node: NodeInfo, profile: JobProfile) -> tuple[int, int] | None:
    if node.free_gpus < profile.gpus:
        return None
    cpus = min(profile.preferred_cpus, node.free_cpus)
    mem_mb = min(profile.preferred_mem_mb, node.free_mem_mb)
    if cpus < profile.minimum_cpus or mem_mb < profile.minimum_mem_mb:
        return None
    return cpus, mem_mb


def _best_partition(nodes: list[NodeInfo], profile: JobProfile) -> str | None:
    partition_scores: dict[str, tuple[int, int, int]] = {}
    for node in nodes:
        candidate = _candidate_resources(node, profile)
        if candidate is None:
            continue
        for partition in node.partitions:
            eligible, gpu_free, cpu_free = partition_scores.get(partition, (0, 0, 0))
            partition_scores[partition] = (
                eligible + 1,
                gpu_free + node.free_gpus,
                cpu_free + node.free_cpus,
            )
    if not partition_scores:
        return None
    return max(
        partition_scores,
        key=lambda partition: (
            partition_scores[partition][0],
            partition_scores[partition][1],
            partition_scores[partition][2],
            _partition_rank(partition, profile),
        ),
    )


def _best_node(nodes: list[NodeInfo], partition: str, profile: JobProfile) -> NodeInfo | None:
    eligible_nodes = [
        node
        for node in nodes
        if partition in node.partitions and _candidate_resources(node, profile) is not None
    ]
    if not eligible_nodes:
        return None
    return max(
        eligible_nodes,
        key=lambda node: (
            node.free_gpus,
            node.free_mem_mb,
            node.free_cpus,
            _node_model_rank(node.gpu_model),
            -len(node.name),
        ),
    )


def _recommendations(
    *,
    nodes: list[NodeInfo],
    profile: JobProfile,
    fairshare: float | None,
) -> list[Recommendation]:
    partition = _best_partition(nodes, profile)
    if partition is None:
        return []
    node = _best_node(nodes, partition, profile)
    if node is None:
        return []

    cpus, mem_mb = _candidate_resources(node, profile) or (profile.minimum_cpus, profile.minimum_mem_mb)
    aggressive_time = profile.aggressive_time_minutes
    balanced_time = profile.preferred_time_minutes
    if fairshare is not None and fairshare >= 0.2:
        aggressive_time = min(profile.preferred_time_minutes, max(profile.aggressive_time_minutes, 3 * 60))

    recommendations = [
        Recommendation(
            partition=partition,
            node=None,
            gpus=profile.gpus,
            cpus_per_task=cpus,
            mem_mb=mem_mb,
            time_minutes=aggressive_time,
            reason="Best backfill chance: broad partition request without node pinning.",
        ),
        Recommendation(
            partition=partition,
            node=node.name,
            gpus=profile.gpus,
            cpus_per_task=cpus,
            mem_mb=mem_mb,
            time_minutes=balanced_time,
            reason=f"Best pinned-node option on {node.name} ({node.gpu_model or 'gpu'} with {node.free_gpus} GPU free).",
        ),
    ]
    return recommendations


def _command_for(
    recommendation: Recommendation,
    *,
    account: str,
    sbatch_script: Path,
) -> str:
    command = [
        "sbatch",
        "-A",
        account,
        "-p",
        recommendation.partition,
        "--gres",
        f"gpu:{recommendation.gpus}",
        "--cpus-per-task",
        str(recommendation.cpus_per_task),
        "--mem",
        _format_mem_gb(recommendation.mem_mb),
        "--time",
        _format_time(recommendation.time_minutes),
    ]
    if recommendation.node:
        command.extend(["-w", recommendation.node])
    command.append(str(sbatch_script))
    return " ".join(shlex.quote(part) for part in command)


def _print_node_table(nodes: list[NodeInfo]) -> None:
    print("")
    print("GPU nodes")
    print("name           partitions         model       gpu_free  cpu_free  mem_free_gb")
    for node in sorted(nodes, key=lambda item: (item.partitions, item.name)):
        print(
            f"{node.name:<14} {'/'.join(node.partitions):<18} "
            f"{(node.gpu_model or '-'): <10} {node.free_gpus:>8} {node.free_cpus:>9} {node.free_mem_mb // 1024:>12}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect GPU Slurm nodes and print best-effort sbatch parameters for a fast start."
    )
    parser.add_argument("--profile", default="prebuilt_bundle", choices=sorted(PROFILES), help="Workload profile to tune for.")
    parser.add_argument("--partitions", default=",".join(DEFAULT_PARTITIONS), help="Comma-separated partitions to inspect.")
    parser.add_argument("--account", default="", help="Slurm account to charge. Defaults to the profile default.")
    parser.add_argument("--user", default=os.getenv("USER", ""), help="User for fairshare inspection. Defaults to $USER.")
    parser.add_argument(
        "--sbatch-script",
        default=str(DEFAULT_SBATCH_SCRIPT),
        help="Path to the sbatch wrapper that should be submitted.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    profile = _profile_for(args.profile)
    partitions = tuple(part.strip() for part in args.partitions.split(",") if part.strip())
    account = args.account or profile.default_account
    user = args.user or os.getenv("USER", "")
    sbatch_script = Path(args.sbatch_script).expanduser().resolve()

    try:
        nodes = _load_nodes(partitions)
    except subprocess.CalledProcessError as exc:
        print(exc.stderr.strip() or str(exc), file=sys.stderr)
        return int(exc.returncode or 1)

    fairshare = _load_fairshare(user, account) if user else None
    recommendations = _recommendations(nodes=nodes, profile=profile, fairshare=fairshare)

    print(f"Profile: {profile.name}")
    print(f"Partitions checked: {', '.join(partitions)}")
    print(f"Account: {account}")
    if fairshare is None:
        print("Fairshare: unavailable")
    else:
        print(f"Fairshare: {fairshare:.6f}")
        if fairshare < 0.2:
            print("Warning: fairshare is low, so an immediate start is unlikely regardless of visible free GPUs.")

    _print_node_table(nodes)
    print("")

    if not recommendations:
        print("No node currently satisfies the minimum GPU/CPU/memory thresholds for this profile.")
        return 2

    print("Best-effort submit commands")
    for index, recommendation in enumerate(recommendations, start=1):
        print(f"{index}. {recommendation.reason}")
        print(f"   {_command_for(recommendation, account=account, sbatch_script=sbatch_script)}")

    print("")
    print("Notes:")
    print("- These commands improve backfill odds but cannot guarantee an instant start when Slurm says Reason=Priority.")
    print("- The unpinned command is usually the fastest option because it gives the scheduler more placement freedom.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
