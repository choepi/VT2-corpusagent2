from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_prebuilt_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "27_build_prebuilt_bundle.py"
    spec = importlib.util.spec_from_file_location("build_prebuilt_bundle", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_prepare_dataset_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "01_prepare_dataset.py"
    spec = importlib.util.spec_from_file_location("prepare_dataset", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_streaming_sample_fraction_is_deterministic_and_applied_before_max_rows() -> None:
    module = _load_prebuilt_module()
    rows = [
        {
            "id": f"doc-{index}",
            "title": f"title {index}",
            "text": f"body {index}",
            "year": 2021,
            "source": "example.org",
        }
        for index in range(200)
    ]

    first = list(module._iter_normalized_records(rows, sample_fraction=0.10, sample_seed=7, max_rows=8))
    second = list(module._iter_normalized_records(rows, sample_fraction=0.10, sample_seed=7, max_rows=8))
    different_seed = list(module._iter_normalized_records(rows, sample_fraction=0.10, sample_seed=11, max_rows=8))

    assert first == second
    assert 0 < len(first) <= 8
    assert first != different_seed


def test_invalid_sample_fraction_is_rejected() -> None:
    module = _load_prebuilt_module()
    rows = [{"id": "doc-1", "title": "title", "text": "body"}]

    try:
        list(module._iter_normalized_records(rows, sample_fraction=0.0))
    except ValueError as exc:
        assert "sample_fraction" in str(exc)
    else:
        raise AssertionError("Expected invalid sample_fraction to raise ValueError")


def test_skip_dense_option_is_available_for_offline_smoke_builds() -> None:
    module = _load_prebuilt_module()

    args = module.parse_args(["--source-file", "example.jsonl", "--skip-dense"])

    assert args.skip_dense is True


def test_dense_model_id_option_accepts_local_model_path() -> None:
    module = _load_prebuilt_module()

    args = module.parse_args(["--source-file", "example.jsonl", "--dense-model-id", "C:/models/e5-base-v2"])

    assert args.dense_model_id == "C:/models/e5-base-v2"


def test_jsonl_source_accepts_utf8_bom_from_windows_powershell() -> None:
    module = _load_prebuilt_module()
    source_path = Path(__file__).with_name("_bom_source_tmp.jsonl")
    try:
        source_path.write_bytes(
            b'\xef\xbb\xbf{"id":"doc-1","title":"Smoke","text":"Body","published_at":"2021","source":"local"}\n'
        )

        rows = list(module._records_from_tabular_file(source_path))

        assert rows == [
            {
                "id": "doc-1",
                "title": "Smoke",
                "text": "Body",
                "published_at": "2021",
                "source": "local",
            }
        ]
    finally:
        source_path.unlink(missing_ok=True)


def test_prepare_dataset_accepts_utf8_bom_from_staged_jsonl() -> None:
    module = _load_prepare_dataset_module()
    source_path = Path(__file__).with_name("_prepare_bom_source_tmp.jsonl")
    try:
        source_path.write_bytes(
            b'\xef\xbb\xbf{"id":"doc-1","title":"Smoke","text":"Body","published_at":"2021","source":"local"}\n'
        )

        rows = list(module.iter_records(source_path))

        assert rows == [
            {
                "id": "doc-1",
                "title": "Smoke",
                "text": "Body",
                "published_at": "2021",
                "source": "local",
            }
        ]
    finally:
        source_path.unlink(missing_ok=True)
