"""
Tests for knowledge graph construction and path traversal.
"""
import pytest
from backend.graph.graph_builder import CERAPGraphBuilder
from backend.graph.graph_query import GraphQueryEngine


@pytest.fixture(scope="module")
def builder():
    b = CERAPGraphBuilder()
    b.build_seed_graph()
    return b


@pytest.fixture(scope="module")
def query_engine(builder):
    return GraphQueryEngine(builder.graph)


def test_seed_graph_node_count(builder):
    """Seed graph must have at least the expected number of nodes."""
    assert builder.graph.number_of_nodes() >= 20


def test_seed_graph_edge_count(builder):
    """Seed graph must have at least the expected number of edges."""
    assert builder.graph.number_of_edges() >= 25


def test_key_nodes_exist(builder):
    """Core biological entities must be present."""
    for gene in ["TP53", "BRCA1", "BRCA2", "PIK3CA", "ERBB2"]:
        assert gene in builder.graph, f"Expected node {gene} in graph"


def test_alias_resolution(builder):
    """Aliases should resolve to canonical IDs."""
    assert builder.resolve_alias("HER2") == "ERBB2"
    assert builder.resolve_alias("p53") == "TP53"
    assert builder.resolve_alias("PI3K") == "PIK3CA"


def test_edges_have_expected_types(builder):
    """Every edge should have an edge_type attribute."""
    for u, v, data in builder.graph.edges(data=True):
        assert "edge_type" in data, f"Edge {u}->{v} missing edge_type"


def test_dynamic_node_insertion(builder):
    """Dynamic gene insertion should extend the graph."""
    initial_count = builder.graph.number_of_nodes()
    builder.add_gene_node("TEST_GENE_XYZ", aliases=["TGX"])
    assert builder.graph.number_of_nodes() == initial_count + 1
    assert builder.resolve_alias("TGX") == "TEST_GENE_XYZ"


def test_path_extraction_tp53(builder, query_engine):
    """Path extraction from TP53 should return at least one path."""
    paths = query_engine.extract_paths("TP53", max_hops=3, top_k=5)
    assert len(paths) >= 1


def test_path_extraction_unknown_node(query_engine):
    """Path extraction on a non-existent node should return empty list."""
    paths = query_engine.extract_paths("NONEXISTENT_GENE_ABC123", max_hops=3)
    assert paths == []


def test_cytoscape_export(builder):
    """to_cytoscape_json should produce nodes and edges lists."""
    cyto = builder.to_cytoscape_json()
    assert "nodes" in cyto
    assert "edges" in cyto
    assert len(cyto["nodes"]) == builder.graph.number_of_nodes()
