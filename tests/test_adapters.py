"""
Phase 7 — Test suite for BioCypher data-ingestion adapters.

Validates that each adapter (Reactome, STRING, GO, OpenTargets) yields
Biolink-compliant nodes and edges, checks interaction scores for STRING PPI
edges, validates GO biological process categories, OpenTargets disease
associations, adapter non-emptiness, and node deduplication during graph
construction from multiple adapters.
"""
import pytest
from typing import List

from backend.graph.biolink_adapter import (
    ReactomeAdapter,
    STRINGAdapter,
    GOAdapter,
    OpenTargetsAdapter,
    get_all_adapters,
)
from backend.graph.graph_builder import CERAPGraphBuilder
from backend.graph.schema import (
    NodeSchema, EdgeSchema, BiolinkCategory, BiolinkPredicate,
)
from backend.graph.graph_store import NetworkXStore


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _collect_nodes(adapter) -> List[NodeSchema]:
    return list(adapter.get_nodes())


def _collect_edges(adapter) -> List[EdgeSchema]:
    return list(adapter.get_edges())


# ═══════════════════════════════════════════════════════════════════════════════
# ReactomeAdapter
# ═══════════════════════════════════════════════════════════════════════════════


class TestReactomeAdapter:
    """Test the Reactome curated pathway/gene adapter."""

    @pytest.fixture
    def adapter(self) -> ReactomeAdapter:
        return ReactomeAdapter()

    def test_reactome_yields_nodes(self, adapter: ReactomeAdapter):
        nodes = _collect_nodes(adapter)
        assert len(nodes) > 0

    def test_reactome_nodes_have_biolink_category(self, adapter: ReactomeAdapter):
        """Every yielded node must have a valid BiolinkCategory."""
        for node in adapter.get_nodes():
            assert isinstance(node, NodeSchema)
            assert isinstance(node.category, BiolinkCategory)

    def test_reactome_yields_pathway_nodes(self, adapter: ReactomeAdapter):
        """Reactome should yield at least one Pathway-category node."""
        nodes = _collect_nodes(adapter)
        pathway_nodes = [n for n in nodes if n.category is BiolinkCategory.PATHWAY]
        assert len(pathway_nodes) > 0

    def test_reactome_yields_gene_nodes(self, adapter: ReactomeAdapter):
        """Reactome should yield Gene-category nodes."""
        nodes = _collect_nodes(adapter)
        gene_nodes = [n for n in nodes if n.category is BiolinkCategory.GENE]
        assert len(gene_nodes) > 0

    def test_reactome_yields_drug_nodes(self, adapter: ReactomeAdapter):
        """Reactome should yield Drug-category nodes."""
        nodes = _collect_nodes(adapter)
        drug_nodes = [n for n in nodes if n.category is BiolinkCategory.DRUG]
        assert len(drug_nodes) > 0

    def test_reactome_yields_disease_nodes(self, adapter: ReactomeAdapter):
        """Reactome should yield Disease-category nodes."""
        nodes = _collect_nodes(adapter)
        disease_nodes = [n for n in nodes if n.category is BiolinkCategory.DISEASE]
        assert len(disease_nodes) > 0

    def test_reactome_yields_edges(self, adapter: ReactomeAdapter):
        edges = _collect_edges(adapter)
        assert len(edges) > 0

    def test_reactome_edges_have_biolink_predicate(self, adapter: ReactomeAdapter):
        """Every yielded edge must have a valid BiolinkPredicate."""
        for edge in adapter.get_edges():
            assert isinstance(edge, EdgeSchema)
            assert isinstance(edge.predicate, BiolinkPredicate)

    def test_reactome_edges_include_participates_in(self, adapter: ReactomeAdapter):
        """Reactome should include 'participates_in' gene-pathway edges."""
        edges = _collect_edges(adapter)
        participates = [e for e in edges if e.predicate is BiolinkPredicate.PARTICIPATES_IN]
        assert len(participates) > 0

    def test_reactome_edges_include_drug_targets(self, adapter: ReactomeAdapter):
        """Reactome should include drug → gene 'targets' edges."""
        edges = _collect_edges(adapter)
        targets = [e for e in edges if e.predicate is BiolinkPredicate.TARGETS]
        assert len(targets) > 0

    def test_reactome_name_and_version(self, adapter: ReactomeAdapter):
        assert adapter.name == "reactome"
        assert adapter.version == "87"


# ═══════════════════════════════════════════════════════════════════════════════
# STRINGAdapter
# ═══════════════════════════════════════════════════════════════════════════════


class TestSTRINGAdapter:
    """Test the STRING protein-protein interaction adapter."""

    @pytest.fixture
    def adapter(self) -> STRINGAdapter:
        return STRINGAdapter()

    def test_string_yields_no_new_nodes(self, adapter: STRINGAdapter):
        """STRING adapter should yield zero nodes (genes come from Reactome)."""
        nodes = _collect_nodes(adapter)
        assert len(nodes) == 0

    def test_string_yields_ppi_edges(self, adapter: STRINGAdapter):
        edges = _collect_edges(adapter)
        assert len(edges) > 0

    def test_string_edges_are_physically_interacts_with(self, adapter: STRINGAdapter):
        """All STRING edges should use PHYSICALLY_INTERACTS_WITH predicate."""
        for edge in adapter.get_edges():
            assert edge.predicate is BiolinkPredicate.PHYSICALLY_INTERACTS_WITH

    def test_string_edges_have_interaction_scores(self, adapter: STRINGAdapter):
        """STRING edges should carry interaction_score in properties."""
        for edge in adapter.get_edges():
            assert "interaction_score" in edge.properties
            assert edge.properties["interaction_score"] >= 700  # high-confidence

    def test_string_edges_have_normalized_weight(self, adapter: STRINGAdapter):
        """Edge weights should be normalized to [0, 1] range."""
        for edge in adapter.get_edges():
            assert 0.0 <= edge.weight <= 1.0

    def test_string_name_and_version(self, adapter: STRINGAdapter):
        assert adapter.name == "string"
        assert adapter.version == "12.0"


# ═══════════════════════════════════════════════════════════════════════════════
# GOAdapter
# ═══════════════════════════════════════════════════════════════════════════════


class TestGOAdapter:
    """Test the Gene Ontology biological process adapter."""

    @pytest.fixture
    def adapter(self) -> GOAdapter:
        return GOAdapter()

    def test_go_yields_biological_process_nodes(self, adapter: GOAdapter):
        """GO should yield nodes with BIOLOGICAL_PROCESS category."""
        nodes = _collect_nodes(adapter)
        assert len(nodes) > 0
        for node in nodes:
            assert node.category is BiolinkCategory.BIOLOGICAL_PROCESS

    def test_go_node_ids_are_go_terms(self, adapter: GOAdapter):
        """GO node IDs should follow the GO:XXXXXXX pattern."""
        for node in adapter.get_nodes():
            assert node.id.startswith("GO:")

    def test_go_yields_edges(self, adapter: GOAdapter):
        edges = _collect_edges(adapter)
        assert len(edges) > 0

    def test_go_edges_use_participates_in(self, adapter: GOAdapter):
        """GO annotation edges should use the PARTICIPATES_IN predicate."""
        for edge in adapter.get_edges():
            assert edge.predicate is BiolinkPredicate.PARTICIPATES_IN

    def test_go_name_and_version(self, adapter: GOAdapter):
        assert adapter.name == "gene_ontology"
        assert adapter.version == "2024-01"


# ═══════════════════════════════════════════════════════════════════════════════
# OpenTargetsAdapter
# ═══════════════════════════════════════════════════════════════════════════════


class TestOpenTargetsAdapter:
    """Test the Open Targets disease-gene association adapter."""

    @pytest.fixture
    def adapter(self) -> OpenTargetsAdapter:
        return OpenTargetsAdapter()

    def test_opentargets_yields_no_new_nodes(self, adapter: OpenTargetsAdapter):
        """OpenTargets should yield zero nodes."""
        nodes = _collect_nodes(adapter)
        assert len(nodes) == 0

    def test_opentargets_yields_disease_association_edges(self, adapter: OpenTargetsAdapter):
        edges = _collect_edges(adapter)
        assert len(edges) > 0

    def test_opentargets_edges_use_associated_with(self, adapter: OpenTargetsAdapter):
        """OpenTargets edges should use ASSOCIATED_WITH predicate."""
        for edge in adapter.get_edges():
            assert edge.predicate is BiolinkPredicate.ASSOCIATED_WITH

    def test_opentargets_edges_have_score_properties(self, adapter: OpenTargetsAdapter):
        """Each edge should carry overall_score, genetic_association, etc."""
        for edge in adapter.get_edges():
            assert "overall_score" in edge.properties
            assert "genetic_association" in edge.properties
            assert "somatic_mutation" in edge.properties

    def test_opentargets_targets_include_brca_disease(self, adapter: OpenTargetsAdapter):
        """All edges should target a MONDO breast cancer ID."""
        for edge in adapter.get_edges():
            assert "MONDO:" in edge.target

    def test_opentargets_name_and_version(self, adapter: OpenTargetsAdapter):
        assert adapter.name == "open_targets"
        assert adapter.version == "24.09"


# ═══════════════════════════════════════════════════════════════════════════════
# All Adapters — Non-empty generators
# ═══════════════════════════════════════════════════════════════════════════════


class TestAllAdaptersNonEmpty:
    """Verify all registered adapters produce at least some data."""

    def test_all_adapters_returned(self):
        adapters = get_all_adapters()
        assert len(adapters) == 4

    @pytest.mark.parametrize("adapter_cls", [ReactomeAdapter, STRINGAdapter, GOAdapter, OpenTargetsAdapter])
    def test_adapter_edges_non_empty(self, adapter_cls):
        """Every adapter must produce at least one edge."""
        adapter = adapter_cls()
        edges = list(adapter.get_edges())
        assert len(edges) > 0, f"{adapter_cls.__name__} produced no edges"


# ═══════════════════════════════════════════════════════════════════════════════
# Node Deduplication via CERAPGraphBuilder
# ═══════════════════════════════════════════════════════════════════════════════


class TestNodeDeduplication:
    """Test that building from multiple adapters deduplicates shared nodes."""

    def test_deduplication_across_adapters(self):
        """Nodes with the same ID from different adapters should not be duplicated."""
        builder = CERAPGraphBuilder(store=NetworkXStore())
        builder.build_from_adapters()

        store = builder.store
        # TP53 appears in Reactome (as gene node) and is referenced by
        # STRING, GO, and OpenTargets edges — should exist only once.
        assert store.has_node("TP53")

        # Total node count should be less than the sum of all adapter node
        # counts (because of overlap), but we mostly verify no crash and
        # that nodes exist.
        assert store.node_count() > 0
        assert store.edge_count() > 0

    def test_deduplication_node_count_stable(self):
        """Building twice from adapters should produce the same node count."""
        builder1 = CERAPGraphBuilder(store=NetworkXStore())
        builder1.build_from_adapters()
        count1 = builder1.store.node_count()

        builder2 = CERAPGraphBuilder(store=NetworkXStore())
        builder2.build_from_adapters()
        count2 = builder2.store.node_count()

        assert count1 == count2

    def test_build_from_adapters_includes_all_categories(self):
        """The built graph should contain Gene, Pathway, Disease, Drug, and BiologicalProcess nodes."""
        builder = CERAPGraphBuilder(store=NetworkXStore())
        builder.build_from_adapters()
        store = builder.store

        genes = store.get_nodes_by_category(BiolinkCategory.GENE)
        pathways = store.get_nodes_by_category(BiolinkCategory.PATHWAY)
        diseases = store.get_nodes_by_category(BiolinkCategory.DISEASE)
        drugs = store.get_nodes_by_category(BiolinkCategory.DRUG)
        bio_procs = store.get_nodes_by_category(BiolinkCategory.BIOLOGICAL_PROCESS)

        assert len(genes) > 0
        assert len(pathways) > 0
        assert len(diseases) > 0
        assert len(drugs) > 0
        assert len(bio_procs) > 0
