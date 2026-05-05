from __future__ import annotations

from corpusagent2.deploy_resources import (
    GIB,
    HostHardware,
    compose_files_for_stack,
    compute_docker_resource_plan,
    data_service_resource_updates,
)


def _memory_mib(value: str) -> int:
    assert value.endswith("m")
    return int(value[:-1])


def test_small_host_uses_larger_fraction_with_service_budget_split() -> None:
    plan = compute_docker_resource_plan(
        HostHardware(logical_cpus=8, total_memory_bytes=16 * GIB, gpu_available=False)
    )

    assert plan.cpu_fraction == 0.80
    assert plan.memory_fraction == 0.80
    assert plan.use_gpu is False
    service_cpus = sum(
        float(plan.env[key])
        for key in (
            "CORPUSAGENT2_API_CPUS",
            "CORPUSAGENT2_MCP_CPUS",
            "CORPUSAGENT2_OPENSEARCH_CPUS",
            "CORPUSAGENT2_POSTGRES_CPUS",
        )
    )
    assert service_cpus <= 8 * 0.80 + 0.05
    service_mem_mib = sum(
        _memory_mib(plan.env[key])
        for key in (
            "CORPUSAGENT2_API_MEM_LIMIT",
            "CORPUSAGENT2_MCP_MEM_LIMIT",
            "CORPUSAGENT2_OPENSEARCH_MEM_LIMIT",
            "CORPUSAGENT2_POSTGRES_MEM_LIMIT",
        )
    )
    assert service_mem_mib <= int(16 * 1024 * 0.80)


def test_large_host_uses_smaller_fraction_and_can_enable_gpu() -> None:
    plan = compute_docker_resource_plan(
        HostHardware(logical_cpus=16, total_memory_bytes=96 * GIB, gpu_available=True)
    )

    assert plan.cpu_fraction == 0.60
    assert plan.memory_fraction == 0.60
    assert plan.use_gpu is True
    assert plan.env["CORPUSAGENT2_DOCKER_USE_GPU"] == "true"
    assert float(plan.env["CORPUSAGENT2_MCP_CPUS"]) > float(plan.env["CORPUSAGENT2_API_CPUS"])
    assert _memory_mib(plan.env["CORPUSAGENT2_MCP_MEM_LIMIT"]) > _memory_mib(plan.env["CORPUSAGENT2_API_MEM_LIMIT"])
    assert plan.env["CORPUSAGENT2_OPENSEARCH_RECOMMENDED_JAVA_OPTS"].startswith("-Xms")


def test_memory_threshold_uses_decimal_server_gb_not_gib() -> None:
    plan = compute_docker_resource_plan(
        HostHardware(logical_cpus=16, total_memory_bytes=96 * 1000**3, gpu_available=False)
    )

    assert plan.memory_fraction == 0.60


def test_compose_files_include_gpu_override_only_when_requested() -> None:
    assert compose_files_for_stack(use_gpu=False) == ["docker-compose.yml", "docker-compose.mcp.yml"]
    assert compose_files_for_stack(use_gpu=True) == [
        "docker-compose.yml",
        "docker-compose.mcp.yml",
        "docker-compose.mcp.gpu.yml",
    ]


def test_data_service_resource_updates_are_in_place_container_limits() -> None:
    plan = compute_docker_resource_plan(
        HostHardware(logical_cpus=8, total_memory_bytes=16 * GIB, gpu_available=False)
    )

    updates = data_service_resource_updates(plan)

    assert updates["corpus_postgres"]["cpus"] == plan.env["CORPUSAGENT2_POSTGRES_CPUS"]
    assert updates["corpus_postgres"]["memory"] == plan.env["CORPUSAGENT2_POSTGRES_MEM_LIMIT"]
    assert updates["os_news"]["cpus"] == plan.env["CORPUSAGENT2_OPENSEARCH_CPUS"]
    assert updates["os_news"]["memory"] == plan.env["CORPUSAGENT2_OPENSEARCH_MEM_LIMIT"]
