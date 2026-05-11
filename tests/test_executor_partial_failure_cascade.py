"""Cascade semantics: only transitive descendants of a failed node are skipped.

Before this fix, any non-optional node failure called `mark_pending_skipped`
which marked EVERY remaining pending node skipped — including independent
branches that had no dependency relationship to the failed node. The user's
real run showed n18 (doc_embeddings) failing and dragging down sentiment,
topic_model, keyterms, time_series_aggregate branches that did not depend
on n18 at all.

Retrieval-bailout failures (db_search / sql_query_search producing nothing)
still halt the whole run because there's no corpus to analyze.
"""

from __future__ import annotations

from corpusagent2.agent_executor import _transitive_dependents


class _Node:
    def __init__(self, node_id: str, depends_on: list[str], capability: str = "tool"):
        self.node_id = node_id
        self.depends_on = depends_on
        self.capability = capability


def _node_map(*nodes: _Node) -> dict[str, _Node]:
    return {n.node_id: n for n in nodes}


def test_failure_only_blocks_direct_descendants():
    # n1 -> n2 -> n3,  n1 -> n4  (n4 is independent of n2/n3)
    nodes = _node_map(
        _Node("n1", []),
        _Node("n2", ["n1"]),
        _Node("n3", ["n2"]),
        _Node("n4", ["n1"]),
    )
    blocked = _transitive_dependents({"n2"}, nodes)
    assert blocked == {"n3"}
    assert "n4" not in blocked
    assert "n1" not in blocked


def test_failure_does_not_block_unrelated_branches():
    # Mirrors the observed run: doc_embeddings (n18) fails. similarity_index
    # (n19) depends on n18 -> blocked. sentiment (n12), topic_model,
    # keyterms — independent branches off n6/n7 — must NOT be blocked.
    nodes = _node_map(
        _Node("n6", []),
        _Node("n7", []),
        _Node("n18", ["n7"], "doc_embeddings"),
        _Node("n19", ["n17", "n18"], "similarity_index"),
        _Node("n11", ["n6"], "sentence_split"),
        _Node("n12", ["n11"], "sentiment"),
        _Node("keyterms", ["n6"], "extract_keyterms"),
        _Node("topics", ["n6"], "topic_model"),
    )
    blocked = _transitive_dependents({"n18"}, nodes)
    assert "n19" in blocked
    assert "n12" not in blocked
    assert "n11" not in blocked
    assert "keyterms" not in blocked
    assert "topics" not in blocked


def test_multi_failure_blocks_combined_descendants():
    nodes = _node_map(
        _Node("a", []),
        _Node("b", []),
        _Node("c", ["a"]),
        _Node("d", ["b"]),
        _Node("e", ["c", "d"]),
    )
    blocked = _transitive_dependents({"a", "b"}, nodes)
    assert blocked == {"c", "d", "e"}


def test_no_failures_means_no_blocks():
    nodes = _node_map(_Node("a", []), _Node("b", ["a"]))
    assert _transitive_dependents(set(), nodes) == set()
