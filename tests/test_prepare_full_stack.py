from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_prepare_full_stack():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "33_prepare_full_stack.py"
    spec = importlib.util.spec_from_file_location("prepare_full_stack", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_restage_policy_uses_existing_non_empty_db_unless_rows_are_requested() -> None:
    module = _load_prepare_full_stack()

    assert module.should_restage_database(requested_rows=None, postgres_count=10, force_restage=False) is False
    assert module.should_restage_database(requested_rows=None, postgres_count=0, force_restage=False) is True
    assert module.should_restage_database(requested_rows=None, postgres_count=None, force_restage=False) is True
    assert module.should_restage_database(requested_rows=10, postgres_count=100, force_restage=False) is True
    assert module.should_restage_database(requested_rows=None, postgres_count=100, force_restage=True) is True


def test_parse_args_defaults_to_auto_gpu_and_no_row_limit() -> None:
    module = _load_prepare_full_stack()

    args = module.parse_args([])

    assert args.gpu == "auto"
    assert args.rows is None
    assert args.source_file == ""
    assert args.force_restage is False


def test_dense_model_path_can_be_explicit_local_directory(tmp_path: Path) -> None:
    module = _load_prepare_full_stack()
    model_dir = tmp_path / "e5-base-v2"
    model_dir.mkdir()
    (model_dir / "modules.json").write_text("[]", encoding="utf-8")

    resolved = module._resolve_dense_model_host_path(str(model_dir))

    assert resolved == model_dir.resolve()
