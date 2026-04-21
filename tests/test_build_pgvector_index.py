from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_build_pgvector_index():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "11_build_pgvector_index.py"
    spec = importlib.util.spec_from_file_location("build_pgvector_index", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_pgvector_index_build_defaults_match_current_cluster_oriented_tuning() -> None:
    module = _load_build_pgvector_index()

    assert module.DEFAULT_HNSW_EF_CONSTRUCTION == 128
    assert module.DEFAULT_MAX_PARALLEL_MAINTENANCE_WORKERS == 6
