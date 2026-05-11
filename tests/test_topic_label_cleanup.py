from __future__ import annotations

from corpusagent2.agent_capabilities import (
    TOPIC_LABEL_STOPWORDS,
    clean_topic_terms,
)


def test_clean_topic_terms_filters_function_words() -> None:
    raw = ["in", "of", "league", "is", "messi", "ronaldo"]
    cleaned = clean_topic_terms(raw, max_count=4)
    assert "in" not in cleaned
    assert "of" not in cleaned
    assert "is" not in cleaned
    assert cleaned[:3] == ["league", "messi", "ronaldo"]


def test_clean_topic_terms_caps_at_max_count() -> None:
    raw = ["alpha", "beta", "gamma", "delta", "epsilon"]
    assert clean_topic_terms(raw, max_count=3) == ["alpha", "beta", "gamma"]


def test_clean_topic_terms_drops_short_tokens() -> None:
    raw = ["ai", "ml", "model", "team", "goal"]
    cleaned = clean_topic_terms(raw, max_count=4, min_length=3)
    assert "ai" not in cleaned
    assert "ml" not in cleaned
    assert "model" in cleaned
    assert "team" in cleaned
    assert "goal" in cleaned


def test_clean_topic_terms_is_order_preserving_and_dedupes() -> None:
    raw = ["Messi", "messi", "Ronaldo", "ronaldo", "goal"]
    cleaned = clean_topic_terms(raw, max_count=5)
    assert cleaned == ["Messi", "Ronaldo", "goal"]


def test_clean_topic_terms_returns_empty_for_all_stopwords() -> None:
    raw = ["the", "and", "of", "is", "in", "on"]
    assert clean_topic_terms(raw, max_count=4) == []


def test_topic_label_stopwords_covers_user_reported_leak() -> None:
    # The user reported topic labels like "in of leage is" / "in of on" / "in messi kane on".
    # The stopword set must drop every leaked function word from that example.
    for token in ("in", "of", "is", "on", "to"):
        assert token in TOPIC_LABEL_STOPWORDS
