from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_advise_slurm_submit():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "29_advise_slurm_submit.py"
    spec = importlib.util.spec_from_file_location("advise_slurm_submit", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_sinfo_aggregates_partitions_per_node() -> None:
    module = _load_advise_slurm_submit()

    payload = "losangeles|gpu\nlosangeles|gpu_ia\nsacramento|gpu\n"

    parsed = module._parse_sinfo(payload)

    assert parsed == {
        "losangeles": {"gpu", "gpu_ia"},
        "sacramento": {"gpu"},
    }


def test_parse_node_details_extracts_allocated_and_free_resources() -> None:
    module = _load_advise_slurm_submit()

    payload = """
NodeName=sacramento Arch=x86_64 CoresPerSocket=64
CPUAlloc=144 CPUEfctv=256 CPUTot=256 CPULoad=16.05
Gres=gpu:a100sxm:8
RealMemory=950000 AllocMem=147456 FreeMem=609925 Sockets=2 Boards=1
CfgTRES=cpu=256,mem=950000M,billing=256,gres/gpu=8
AllocTRES=cpu=144,mem=147456M,gres/gpu=3
""".strip()

    node = module._parse_node_details("sacramento", ("gpu",), payload)

    assert node.gpu_model == "a100sxm"
    assert node.total_gpus == 8
    assert node.allocated_gpus == 3
    assert node.free_gpus == 5
    assert node.free_cpus == 112
    assert node.free_mem_mb == 609925


def test_recommendations_prefer_unpinned_partition_then_best_node() -> None:
    module = _load_advise_slurm_submit()
    profile = module.PROFILES["prebuilt_bundle"]

    nodes = [
        module.NodeInfo(
            name="losangeles",
            partitions=("gpu", "gpu_ia"),
            gpu_model="v100sxm",
            total_gpus=8,
            allocated_gpus=5,
            free_gpus=3,
            cpu_total=256,
            cpu_alloc=200,
            free_cpus=56,
            real_mem_mb=950000,
            alloc_mem_mb=500000,
            free_mem_mb=450000,
        ),
        module.NodeInfo(
            name="sacramento",
            partitions=("gpu",),
            gpu_model="a100sxm",
            total_gpus=8,
            allocated_gpus=3,
            free_gpus=5,
            cpu_total=256,
            cpu_alloc=144,
            free_cpus=112,
            real_mem_mb=950000,
            alloc_mem_mb=147456,
            free_mem_mb=609925,
        ),
    ]

    recommendations = module._recommendations(nodes=nodes, profile=profile, fairshare=0.1)

    assert recommendations[0].partition == "gpu"
    assert recommendations[0].node is None
    assert recommendations[0].time_minutes == profile.aggressive_time_minutes
    assert recommendations[1].node == "sacramento"
    assert recommendations[1].cpus_per_task == profile.preferred_cpus
