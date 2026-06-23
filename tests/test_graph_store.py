"""
Phase 7 — Test suite for the GraphStore abstraction (NetworkXStore implementation).

Covers add/has/get for nodes and edges, directional neighbour queries, ego-network
subgraph extraction, category filtering, pathfinding, entity alias resolution,
save/load cache roundtrip, and Biolink-typed statistics.
"""
import json
import os
import pytest
from pathlib import Path

from backend.graph.graph_store import NetworkXStore, create_graph_store
from backend.graph.schema import (
    NodeSchema, EdgeSchema, BiolinkCategory, BiolinkPredicate, Provenance,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


def _gene_node(gene_id: str, aliases: list | None = None) -> NodeSchema:
    return NodeSchema(
        id=gene_id,
        category=BiolinkCategory.GENE,
        label=gene_id,
        aliases=aliases or [],
    )


def _pathway_node(pid: str) -> NodeSchema:
    return NodeSchema(id=pid, category=BiolinkCategory.PATHWAY, label=pid)


def _disease_node(did: str) -> NodeSchema:
    return NodeSchema(id=did, category=BiolinkCategory.DISEASE, label=did)


def _edge(src: str, tgt: str,
          predicate: BiolinkPredicate = BiolinkPredicate.POSITIVELY_REGULATES,
          weight: float = 1.0) -> EdgeSchema:
    return EdgeSchema(source=src, target=tgt, predicate=predicate, weight=weight)


@pytest.fixture
def empty_store() -> NetworkXStore:
    return NetworkXStore()


@pytest.fixture
def small_store() -> NetworkXStore:
    """A small linear graph: A → B → C → D with one branch B → E."""
    store = NetworkXStore()
    for nid in ["A", "B", "C", "D", "E"]:
        store.add_node(_gene_node(nid))
    store.add_edge(_edge("A", "B"))
    store.add_edge(_edge("B", "C"))
    store.add_edge(_edge("C", "D"))
    store.add_edge(_edge("B", "E"))
    return store


@pytest.fixture
def typed_store() -> NetworkXStore:
    """A store with mixed Biolink categories and predicates."""
    store = NetworkXStore()
    store.add_node(_gene_node("TP53", aliases=["p53", "TRP53"]))
    store.add_node(_gene_node("MDM2"))
    store.add_node(_pathway_node("APOPTOSIS"))
    store.add_node(_disease_node("BRCA"))
    store.add_edge(_edge("TP53", "MDM2", BiolinkPredicate.POSITIVELY_REGULATES, 0.9))
    store.add_edge(_edge("MDM2", "TP53", BiolinkPredicate.NEGATIVELY_REGULATES, 0.9))
    store.add_edge(_edge("TP53", "APOPTOSIS", BiolinkPredicate.POSITIVELY_REGULATES, 1.0))
    store.add_edge(_edge("TP53", "BRCA", BiolinkPredicate.GENE_ASSOCIATED_WITH_CONDITION, 1.0))
    return store


# ═══════════════════════════════════════════════════════════════════════════════
# add_node / add_edge / has_node / has_edge
# ═══════════════════════════════════════════════════════════════════════════════


class TestNetworkXStoreBasicCRUD:
    """Test basic node and edge insertion and existence checks."""

    def test_add_node_and_has_node(self, empty_store: NetworkXStore):
        store = empty_store
        store.add_node(_gene_node("TP53"))
        assert store.has_node("TP53")
        assert not store.has_node("NONEXISTENT")

    def test_add_edge_and_has_edge(self, empty_store: NetworkXStore):
        store = empty_store
        store.add_node(_gene_node("A"))
        store.add_node(_gene_node("B"))
        store.add_edge(_edge("A", "B"))
        assert store.has_edge("A", "B")
        assert not store.has_edge("B", "A")  # directed graph

    def test_add_edge_skips_missing_nodes(self, empty_store: NetworkXStore):
        """Edges referencing nodes not in the graph should be silently skipped."""
        store = empty_store
        store.add_edge(_edge("MISSING_SRC", "MISSING_TGT"))
        assert store.edge_count() == 0

    def test_node_count(self, small_store: NetworkXStore):
        assert small_store.node_count() == 5

    def test_edge_count(self, small_store: NetworkXStore):
        assert small_store.edge_count() == 4

    def test_get_node_returns_dict(self, typed_store: NetworkXStore):
        data = typed_store.get_node("TP53")
        assert data is not None
        assert data["category"] == "biolink:Gene"

    def test_get_node_returns_none_for_missing(self, empty_store: NetworkXStore):
        assert empty_store.get_node("NOPE") is None

    def test_get_edge_returns_dict(self, typed_store: NetworkXStore):
        data = typed_store.get_edge("TP53", "MDM2")
        assert data is not None
        assert data["predicate"] == "biolink:positively_regulates"

    def test_get_edge_returns_none_for_missing(self, typed_store: NetworkXStore):
        assert typed_store.get_edge("MDM2", "APOPTOSIS") is None

    def test_get_all_nodes(self, small_store: NetworkXStore):
        all_nodes = small_store.get_all_nodes()
        assert set(all_nodes) == {"A", "B", "C", "D", "E"}

    def test_get_all_edges(self, small_store: NetworkXStore):
        all_edges = small_store.get_all_edges()
        assert ("A", "B") in all_edges
        assert ("B", "C") in all_edges
        assert len(all_edges) == 4


# ═══════════════════════════════════════════════════════════════════════════════
# get_neighbors (out, in, both)
# ═══════════════════════════════════════════════════════════════════════════════


class TestGetNeighbors:
    """Test directional neighbour queries."""

    def test_get_neighbors_out(self, small_store: NetworkXStore):
        """Outgoing neighbours of B should be C and E."""
        out = small_store.get_neighbors("B", direction="out")
        assert set(out) == {"C", "E"}

    def test_get_neighbors_in(self, small_store: NetworkXStore):
        """Incoming neighbour of B should be A."""
        inc = small_store.get_neighbors("B", direction="in")
        assert set(inc) == {"A"}

    def test_get_neighbors_both(self, small_store: NetworkXStore):
        """Both-direction neighbours of B should be A, C, E."""
        both = small_store.get_neighbors("B", direction="both")
        assert set(both) == {"A", "C", "E"}

    def test_get_neighbors_missing_node(self, small_store: NetworkXStore):
        """Querying neighbours of a non-existent node returns empty list."""
        assert small_store.get_neighbors("MISSING") == []


# ═══════════════════════════════════════════════════════════════════════════════
# get_subgraph
# ═══════════════════════════════════════════════════════════════════════════════


class TestGetSubgraph:
    """Test ego-network subgraph extraction."""

    def test_subgraph_radius_1(self, small_store: NetworkXStore):
        """Radius-1 subgraph around B should contain A, B, C, E."""
        sub = small_store.get_subgraph(["B"], radius=1)
        assert sub.has_node("B")
        assert sub.has_node("A")
        assert sub.has_node("C")
        assert sub.has_node("E")
        # D is 2 hops from B, should be excluded
        assert not sub.has_node("D")

    def test_subgraph_radius_2(self, small_store: NetworkXStore):
        """Radius-2 subgraph around B should contain all 5 nodes."""
        sub = small_store.get_subgraph(["B"], radius=2)
        assert sub.node_count() == 5

    def test_subgraph_invalid_seed(self, small_store: NetworkXStore):
        """Subgraph with non-existent seeds returns empty store."""
        sub = small_store.get_subgraph(["DOES_NOT_EXIST"])
        assert sub.node_count() == 0

    def test_subgraph_is_networkx_store(self, small_store: NetworkXStore):
        sub = small_store.get_subgraph(["A"], radius=1)
        assert isinstance(sub, NetworkXStore)


# ═══════════════════════════════════════════════════════════════════════════════
# get_nodes_by_category
# ═══════════════════════════════════════════════════════════════════════════════


class TestGetNodesByCategory:
    """Test filtering nodes by Biolink category."""

    def test_get_genes(self, typed_store: NetworkXStore):
        genes = typed_store.get_nodes_by_category(BiolinkCategory.GENE)
        ids = {g["id"] for g in genes}
        assert "TP53" in ids
        assert "MDM2" in ids

    def test_get_pathways(self, typed_store: NetworkXStore):
        pathways = typed_store.get_nodes_by_category(BiolinkCategory.PATHWAY)
        ids = {p["id"] for p in pathways}
        assert "APOPTOSIS" in ids

    def test_get_diseases(self, typed_store: NetworkXStore):
        diseases = typed_store.get_nodes_by_category(BiolinkCategory.DISEASE)
        ids = {d["id"] for d in diseases}
        assert "BRCA" in ids

    def test_get_category_no_matches(self, typed_store: NetworkXStore):
        drugs = typed_store.get_nodes_by_category(BiolinkCategory.DRUG)
        assert drugs == []


# ═══════════════════════════════════════════════════════════════════════════════
# find_paths
# ═══════════════════════════════════════════════════════════════════════════════


class TestFindPaths:
    """Test path-finding with and without a target."""

    def test_find_paths_with_target(self, small_store: NetworkXStore):
        """A path from A to D should exist: A→B→C→D."""
        paths = small_store.find_paths("A", "D")
        assert len(paths) >= 1
        # Check the path goes through B and C
        first_path = paths[0]
        assert first_path["nodes"] == ["A", "B", "C", "D"]
        assert first_path["length"] == 3

    def test_find_paths_without_target(self, small_store: NetworkXStore):
        """Without a target, find_paths should return paths to all reachable nodes."""
        paths = small_store.find_paths("A")
        assert len(paths) >= 1
        # Should find paths to B, C, D, E
        all_targets = set()
        for p in paths:
            all_targets.add(p["nodes"][-1])
        assert "B" in all_targets or "C" in all_targets or "D" in all_targets

    def test_find_paths_nonexistent_source(self, small_store: NetworkXStore):
        """Source not in graph should return empty list."""
        assert small_store.find_paths("NOWHERE") == []

    def test_find_paths_score_field(self, small_store: NetworkXStore):
        """Each path result must have a 'score' field."""
        paths = small_store.find_paths("A", "D")
        for p in paths:
            assert "score" in p

    def test_find_paths_readable_field(self, small_store: NetworkXStore):
        """Each path result must have a 'readable' field."""
        paths = small_store.find_paths("A", "D")
        for p in paths:
            assert "readable" in p
            assert "--[" in p["readable"]  # contains predicate formatting


# ═══════════════════════════════════════════════════════════════════════════════
# resolve_alias
# ═══════════════════════════════════════════════════════════════════════════════


class TestResolveAlias:
    """Test entity alias resolution via the entity index."""

    def test_resolve_alias_by_canonical_id(self, typed_store: NetworkXStore):
        """Canonical IDs should resolve to themselves."""
        assert typed_store.resolve_alias("TP53") == "TP53"

    def test_resolve_alias_by_alias(self, typed_store: NetworkXStore):
        """Aliases should resolve to the canonical ID."""
        assert typed_store.resolve_alias("p53") == "TP53"
        assert typed_store.resolve_alias("TRP53") == "TP53"

    def test_resolve_alias_case_insensitive(self, typed_store: NetworkXStore):
        """Alias resolution should be case-insensitive."""
        assert typed_store.resolve_alias("tp53") == "TP53"
        assert typed_store.resolve_alias("P53") == "TP53"

    def test_resolve_alias_unknown(self, typed_store: NetworkXStore):
        """Unknown names should return None."""
        assert typed_store.resolve_alias("UNKNOWN_GENE_XYZ") is None


# ═══════════════════════════════════════════════════════════════════════════════
# save_cache / load_cache roundtrip
# ═══════════════════════════════════════════════════════════════════════════════


class TestCacheRoundtrip:
    """Test save_cache and load_cache for data integrity."""

    def test_save_and_load_cache(self, typed_store: NetworkXStore, tmp_path: Path):
        cache_file = str(tmp_path / "test_graph.json")
        typed_store.save_cache(cache_file)

        # Verify the file was created
        assert os.path.exists(cache_file)

        # Load into a fresh store
        loaded_store = NetworkXStore()
        result = loaded_store.load_cache(cache_file)
        assert result is True

        # Verify data integrity
        assert loaded_store.node_count() == typed_store.node_count()
        assert loaded_store.edge_count() == typed_store.edge_count()
        assert loaded_store.has_node("TP53")
        assert loaded_store.has_edge("TP53", "MDM2")

    def test_load_cache_missing_file(self, empty_store: NetworkXStore):
        """Loading a non-existent cache should return False."""
        result = empty_store.load_cache("/nonexistent/path/cache.json")
        assert result is False

    def test_cache_preserves_node_attributes(self, typed_store: NetworkXStore, tmp_path: Path):
        cache_file = str(tmp_path / "attrs_test.json")
        typed_store.save_cache(cache_file)

        loaded = NetworkXStore()
        loaded.load_cache(cache_file)
        tp53_data = loaded.get_node("TP53")
        assert tp53_data is not None
        assert tp53_data["category"] == "biolink:Gene"

    def test_cache_preserves_edge_attributes(self, typed_store: NetworkXStore, tmp_path: Path):
        cache_file = str(tmp_path / "edge_attrs_test.json")
        typed_store.save_cache(cache_file)

        loaded = NetworkXStore()
        loaded.load_cache(cache_file)
        edge_data = loaded.get_edge("TP53", "MDM2")
        assert edge_data is not None
        assert edge_data["predicate"] == "biolink:positively_regulates"

    def test_cache_rebuilds_entity_index(self, typed_store: NetworkXStore, tmp_path: Path):
        """After loading from cache, alias resolution should still work."""
        cache_file = str(tmp_path / "alias_test.json")
        typed_store.save_cache(cache_file)

        loaded = NetworkXStore()
        loaded.load_cache(cache_file)
        # The entity index is rebuilt on load from stored aliases
        assert loaded.resolve_alias("TP53") == "TP53"


# ═══════════════════════════════════════════════════════════════════════════════
# get_statistics
# ═══════════════════════════════════════════════════════════════════════════════


class TestGetStatistics:
    """Test statistics generation with category/predicate breakdowns."""

    def test_statistics_totals(self, typed_store: NetworkXStore):
        stats = typed_store.get_statistics()
        assert stats["total_nodes"] == 4
        assert stats["total_edges"] == 4

    def test_statistics_nodes_by_category(self, typed_store: NetworkXStore):
        stats = typed_store.get_statistics()
        assert "nodes_by_category" in stats
        cat_counts = stats["nodes_by_category"]
        assert cat_counts.get("biolink:Gene", 0) == 2
        assert cat_counts.get("biolink:Pathway", 0) == 1
        assert cat_counts.get("biolink:Disease", 0) == 1

    def test_statistics_edges_by_predicate(self, typed_store: NetworkXStore):
        stats = typed_store.get_statistics()
        assert "edges_by_predicate" in stats
        pred_counts = stats["edges_by_predicate"]
        assert pred_counts.get("biolink:positively_regulates", 0) == 2
        assert pred_counts.get("biolink:negatively_regulates", 0) == 1
        assert pred_counts.get("biolink:gene_associated_with_condition", 0) == 1

    def test_statistics_empty_store(self, empty_store: NetworkXStore):
        stats = empty_store.get_statistics()
        assert stats["total_nodes"] == 0
        assert stats["total_edges"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# create_graph_store factory
# ═══════════════════════════════════════════════════════════════════════════════


class TestCreateGraphStore:
    """Test the factory function."""

    def test_create_networkx_store(self):
        store = create_graph_store(mode="networkx")
        assert isinstance(store, NetworkXStore)

    def test_create_default_store(self):
        store = create_graph_store()
        assert isinstance(store, NetworkXStore)
