"""
Tests for path extraction and constraint engine.
"""
import pytest
from backend.graph.graph_builder import CERAPGraphBuilder
from backend.reasoning.path_extractor import PathExtractor
from backend.reasoning.constraint_engine import ConstraintEngine, KGTrie


@pytest.fixture(scope="module")
def builder():
    b = CERAPGraphBuilder()
    b.build_seed_graph()
    return b


@pytest.fixture(scope="module")
def extractor(builder):
    return PathExtractor(builder)


@pytest.fixture(scope="module")
def constraint_engine(builder):
    return ConstraintEngine(builder.graph)


# ── PathExtractor tests ────────────────────────────────────────────────────────

def test_extract_known_gene(extractor):
    result = extractor.extract(["TP53"])
    assert result["status"] == "success"
    assert len(result["paths"]) >= 1


def test_extract_gene_alias(extractor):
    """Alias input (p53) should normalise and return success."""
    result = extractor.extract(["p53"])
    assert result["status"] == "success"


def test_extract_unknown_gene(extractor):
    """Unknown gene should return no_match or success with 0 paths."""
    result = extractor.extract(["COMPLETELYFAKEGENE999"])
    assert result["status"] in ("no_match", "success")


def test_extract_multiple_genes(extractor):
    result = extractor.extract(["TP53", "BRCA1"])
    assert result["status"] == "success"
    assert result["total_paths_found"] >= 1


def test_extract_returns_graph(extractor):
    """Result should include Cytoscape graph for visualisation."""
    result = extractor.extract(["PIK3CA"])
    if result["status"] == "success":
        assert "graph" in result
        assert "nodes" in result["graph"]
        assert "edges" in result["graph"]


def test_extract_top_k(extractor):
    """top_k parameter constrains the number of returned paths."""
    result = extractor.extract(["TP53"], top_k=3)
    if result["status"] == "success":
        assert len(result["paths"]) <= 3


# ── ConstraintEngine tests ─────────────────────────────────────────────────────

def test_trie_contains_known_entity(constraint_engine):
    assert constraint_engine.is_valid_entity("TP53") is True
    assert constraint_engine.is_valid_entity("BRCA1") is True


def test_trie_does_not_contain_unknown(constraint_engine):
    assert constraint_engine.is_valid_entity("FAKEGENE_XYZ") is False


def test_trie_case_insensitive(constraint_engine):
    # The trie uses .upper() internally
    assert constraint_engine.is_valid_entity("tp53") is True


def test_valid_edge(constraint_engine, builder):
    """An edge that exists in seed data should be valid."""
    # TP53 → MDM2 exists
    assert constraint_engine.is_valid_edge("TP53", "MDM2") is True


def test_invalid_edge(constraint_engine):
    """An edge that does not exist should return False."""
    assert constraint_engine.is_valid_edge("TP53", "FAKEGENE_XYZ") is False


def test_extract_entities_from_text(constraint_engine):
    text = "TP53 is a tumour suppressor that activates APOPTOSIS via MDM2."
    found = constraint_engine.extract_entities_from_text(text)
    assert "TP53" in found
    assert "MDM2" in found


def test_build_prompt_context(constraint_engine, extractor):
    result = extractor.extract(["TP53"])
    if result["status"] == "success":
        paths = result["paths"]
        ctx = constraint_engine.build_prompt_context(paths)
        assert "KNOWLEDGE GRAPH CONSTRAINTS" in ctx
        assert "TP53" in ctx
