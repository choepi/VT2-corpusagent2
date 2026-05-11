from __future__ import annotations

import pytest

from corpusagent2.agent_backends import InMemoryWorkingSetStore
from corpusagent2.agent_runtime import resolve_working_store


def test_memory_backend_skips_postgres(monkeypatch) -> None:
    monkeypatch.setenv("CORPUSAGENT2_WORKINGSET_BACKEND", "memory")
    monkeypatch.delenv("CORPUSAGENT2_PG_DSN", raising=False)
    store = resolve_working_store(doc_lookup={"d1": {"title": "T"}}, require_backend_services=True)
    assert isinstance(store, InMemoryWorkingSetStore)
    assert store.document_lookup["d1"]["title"] == "T"


def test_invalid_backend_raises(monkeypatch) -> None:
    monkeypatch.setenv("CORPUSAGENT2_WORKINGSET_BACKEND", "redis")
    monkeypatch.delenv("CORPUSAGENT2_PG_DSN", raising=False)
    with pytest.raises(ValueError, match="CORPUSAGENT2_WORKINGSET_BACKEND"):
        resolve_working_store(require_backend_services=False)


def test_postgres_backend_without_dsn_raises(monkeypatch) -> None:
    monkeypatch.setenv("CORPUSAGENT2_WORKINGSET_BACKEND", "postgres")
    monkeypatch.delenv("CORPUSAGENT2_PG_DSN", raising=False)
    with pytest.raises(RuntimeError, match="CORPUSAGENT2_PG_DSN"):
        resolve_working_store(require_backend_services=False)


def test_auto_backend_falls_back_to_memory_when_no_dsn(monkeypatch) -> None:
    monkeypatch.setenv("CORPUSAGENT2_WORKINGSET_BACKEND", "auto")
    monkeypatch.delenv("CORPUSAGENT2_PG_DSN", raising=False)
    store = resolve_working_store(require_backend_services=True)
    assert isinstance(store, InMemoryWorkingSetStore)


def test_unset_backend_keeps_historical_behaviour_without_dsn(monkeypatch) -> None:
    monkeypatch.delenv("CORPUSAGENT2_WORKINGSET_BACKEND", raising=False)
    monkeypatch.delenv("CORPUSAGENT2_PG_DSN", raising=False)
    store = resolve_working_store(require_backend_services=False)
    assert isinstance(store, InMemoryWorkingSetStore)


def test_unset_backend_with_required_services_and_no_dsn_raises(monkeypatch) -> None:
    monkeypatch.delenv("CORPUSAGENT2_WORKINGSET_BACKEND", raising=False)
    monkeypatch.delenv("CORPUSAGENT2_PG_DSN", raising=False)
    with pytest.raises(RuntimeError, match="Postgres working store"):
        resolve_working_store(require_backend_services=True)
