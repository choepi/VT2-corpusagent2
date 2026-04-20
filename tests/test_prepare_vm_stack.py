from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_prepare_vm_stack():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "22_prepare_vm_stack.py"
    spec = importlib.util.spec_from_file_location("prepare_vm_stack", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_hybrid_profile_prefers_pgvector_backfill_over_local_dense_assets() -> None:
    module = _load_prepare_vm_stack()

    env = module._retrieval_profile_env("hybrid")

    assert env["CORPUSAGENT2_BUILD_LEXICAL_ASSETS"] == "true"
    assert env["CORPUSAGENT2_BUILD_DENSE_ASSETS"] == "false"
    assert env["CORPUSAGENT2_PG_INCLUDE_EMBEDDINGS"] == "false"
    assert env["CORPUSAGENT2_ENABLE_DENSE_RETRIEVAL"] == "true"
    assert env["CORPUSAGENT2_RETRIEVAL_BACKEND"] == "pgvector"
    assert env["CORPUSAGENT2_PG_BACKFILL_ENCODE_BATCH_SIZE"] == "16"
    assert env["CORPUSAGENT2_PG_BUILD_HNSW"] == "false"


def test_parse_args_defaults_to_hybrid_without_env_override(monkeypatch) -> None:
    module = _load_prepare_vm_stack()
    monkeypatch.delenv("CORPUSAGENT2_VM_RETRIEVAL_PROFILE", raising=False)
    module.DOTENV_VALUES = {}

    args = module.parse_args([])

    assert args.retrieval_profile == "hybrid"


def test_expected_pgvector_index_names_follow_vm_defaults() -> None:
    module = _load_prepare_vm_stack()

    env = module._retrieval_profile_env("hybrid")
    names = module._expected_pgvector_index_names(env, table_name="article_corpus")

    assert names == {"idx_article_corpus_embedding_ivfflat"}


def test_pgvector_backfill_complete_requires_full_row_coverage() -> None:
    module = _load_prepare_vm_stack()

    assert module._pgvector_backfill_complete(
        expected_documents=10,
        current_postgres_count=10,
        current_embedding_count=10,
    )
    assert not module._pgvector_backfill_complete(
        expected_documents=10,
        current_postgres_count=10,
        current_embedding_count=9,
    )
    assert not module._pgvector_backfill_complete(
        expected_documents=10,
        current_postgres_count=9,
        current_embedding_count=9,
    )
