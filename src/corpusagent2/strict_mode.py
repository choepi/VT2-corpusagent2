from __future__ import annotations

import os

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    return default


def is_strict_mode() -> bool:
    return _env_bool("CORPUSAGENT2_ANALYSIS_STRICT_MODE", False)


def allow_silent_fallbacks() -> bool:
    return _env_bool("CORPUSAGENT2_ALLOW_SILENT_FALLBACKS", not is_strict_mode())


def allow_provider_fallback() -> bool:
    return _env_bool("CORPUSAGENT2_ALLOW_PROVIDER_FALLBACK", not is_strict_mode())


def fail_on_required_node_empty() -> bool:
    return _env_bool("CORPUSAGENT2_FAIL_ON_REQUIRED_NODE_EMPTY", is_strict_mode())


def fail_on_metric_source_empty() -> bool:
    return _env_bool("CORPUSAGENT2_FAIL_ON_METRIC_SOURCE_EMPTY", False)


def plot_require_valid_y() -> bool:
    return _env_bool("CORPUSAGENT2_PLOT_REQUIRE_VALID_Y", is_strict_mode())


def require_series_assignment() -> bool:
    return _env_bool("CORPUSAGENT2_REQUIRE_SERIES_ASSIGNMENT", is_strict_mode())


def snapshot() -> dict[str, bool]:
    """Return the effective flag values — useful for run manifests and tests."""
    return {
        "strict_mode": is_strict_mode(),
        "allow_silent_fallbacks": allow_silent_fallbacks(),
        "allow_provider_fallback": allow_provider_fallback(),
        "fail_on_required_node_empty": fail_on_required_node_empty(),
        "fail_on_metric_source_empty": fail_on_metric_source_empty(),
        "plot_require_valid_y": plot_require_valid_y(),
        "require_series_assignment": require_series_assignment(),
    }
