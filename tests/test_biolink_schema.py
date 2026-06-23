"""
Phase 7 — Test suite for Biolink-compliant graph schema.

Validates BiolinkCategory/BiolinkPredicate enums, backward-compatible
legacy mappings, NodeSchema/EdgeSchema serialization, Provenance,
EdgeQualifiers, and all Association classes (.to_edge() methods).
"""
import pytest

from backend.graph.schema import (
    BiolinkCategory,
    BiolinkPredicate,
    DirectionQualifier,
    CausalMechanism,
    EvidenceLevel,
    Provenance,
    EdgeQualifiers,
    NodeSchema,
    EdgeSchema,
    GeneToDiseaseAssociation,
    GeneToPathwayAssociation,
    DrugToGeneAssociation,
    ProteinToProteinAssociation,
    NodeType,
    EdgeType,
)


# ═══════════════════════════════════════════════════════════════════════════════
# BiolinkCategory enum
# ═══════════════════════════════════════════════════════════════════════════════


class TestBiolinkCategory:
    """Validate every member of the BiolinkCategory enum."""

    EXPECTED_MEMBERS = [
        "GENE", "PROTEIN", "SEQUENCE_VARIANT", "PATHWAY",
        "BIOLOGICAL_PROCESS", "MOLECULAR_ACTIVITY", "CELLULAR_COMPONENT",
        "DISEASE", "PHENOTYPIC_FEATURE", "DRUG", "CHEMICAL_ENTITY",
        "GENE_FAMILY", "ANATOMICAL_ENTITY", "CLINICAL_FINDING",
    ]

    @pytest.mark.parametrize("member", EXPECTED_MEMBERS)
    def test_biolink_category_member_exists(self, member: str):
        """Each expected Biolink category must exist in the enum."""
        assert hasattr(BiolinkCategory, member), f"Missing BiolinkCategory.{member}"

    def test_biolink_category_values_prefixed(self):
        """All enum values must carry the 'biolink:' prefix."""
        for cat in BiolinkCategory:
            assert cat.value.startswith("biolink:"), f"{cat.name} value missing 'biolink:' prefix"

    def test_biolink_category_is_str_enum(self):
        """BiolinkCategory inherits from str, so members compare equal to strings."""
        assert BiolinkCategory.GENE == "biolink:Gene"

    def test_biolink_category_from_legacy_known(self):
        """from_legacy() maps well-known legacy NodeType strings."""
        assert BiolinkCategory.from_legacy("Gene") is BiolinkCategory.GENE
        assert BiolinkCategory.from_legacy("Protein") is BiolinkCategory.PROTEIN
        assert BiolinkCategory.from_legacy("Pathway") is BiolinkCategory.PATHWAY
        assert BiolinkCategory.from_legacy("Drug") is BiolinkCategory.DRUG
        assert BiolinkCategory.from_legacy("Disease") is BiolinkCategory.DISEASE

    def test_biolink_category_from_legacy_unknown_defaults_to_gene(self):
        """Unknown legacy strings should fall back to GENE."""
        assert BiolinkCategory.from_legacy("UnknownThing") is BiolinkCategory.GENE

    def test_node_type_alias(self):
        """NodeType is a convenience alias for BiolinkCategory."""
        assert NodeType is BiolinkCategory


# ═══════════════════════════════════════════════════════════════════════════════
# BiolinkPredicate enum
# ═══════════════════════════════════════════════════════════════════════════════


class TestBiolinkPredicate:
    """Validate every member of the BiolinkPredicate enum."""

    EXPECTED_MEMBERS = [
        "POSITIVELY_REGULATES", "NEGATIVELY_REGULATES", "REGULATES",
        "PHYSICALLY_INTERACTS_WITH", "INTERACTS_WITH", "BINDS",
        "AFFECTS", "CONTRIBUTES_TO", "PARTICIPATES_IN", "ENABLES",
        "CATALYZES", "PHOSPHORYLATES",
        "CAUSES", "PREDISPOSES", "PREVENTS",
        "GENE_ASSOCIATED_WITH_CONDITION", "VARIANT_ASSOCIATED_WITH_CONDITION",
        "HAS_PHENOTYPE", "TREATS", "TARGETS",
        "EXPRESSED_IN", "LOCATED_IN", "COLOCALIZES_WITH",
        "IS_SEQUENCE_VARIANT_OF", "HAS_GENE_PRODUCT",
        "ASSOCIATED_WITH", "CORRELATED_WITH", "RELATED_TO",
    ]

    @pytest.mark.parametrize("member", EXPECTED_MEMBERS)
    def test_biolink_predicate_member_exists(self, member: str):
        """Each expected Biolink predicate must exist in the enum."""
        assert hasattr(BiolinkPredicate, member), f"Missing BiolinkPredicate.{member}"

    def test_biolink_predicate_values_prefixed(self):
        """All predicate values must carry the 'biolink:' prefix."""
        for pred in BiolinkPredicate:
            assert pred.value.startswith("biolink:"), f"{pred.name} missing 'biolink:' prefix"

    def test_biolink_predicate_from_legacy_known(self):
        """from_legacy() maps well-known legacy EdgeType strings."""
        assert BiolinkPredicate.from_legacy("ACTIVATES") is BiolinkPredicate.POSITIVELY_REGULATES
        assert BiolinkPredicate.from_legacy("INHIBITS") is BiolinkPredicate.NEGATIVELY_REGULATES
        assert BiolinkPredicate.from_legacy("TARGETS") is BiolinkPredicate.TARGETS
        assert BiolinkPredicate.from_legacy("INTERACTS_WITH") is BiolinkPredicate.PHYSICALLY_INTERACTS_WITH
        assert BiolinkPredicate.from_legacy("PHOSPHORYLATES") is BiolinkPredicate.PHOSPHORYLATES

    def test_biolink_predicate_from_legacy_unknown_defaults_to_related_to(self):
        """Unknown legacy edge types should fall back to RELATED_TO."""
        assert BiolinkPredicate.from_legacy("MYSTERY_EDGE") is BiolinkPredicate.RELATED_TO

    def test_edge_type_alias(self):
        """EdgeType is a convenience alias for BiolinkPredicate."""
        assert EdgeType is BiolinkPredicate


# ═══════════════════════════════════════════════════════════════════════════════
# Provenance
# ═══════════════════════════════════════════════════════════════════════════════


class TestProvenance:
    """Test Provenance dataclass creation and serialization."""

    def test_provenance_defaults(self):
        """Default Provenance should have sensible defaults."""
        p = Provenance()
        assert p.source_database == "seed_brca"
        assert p.pmids == []
        assert p.evidence_level == EvidenceLevel.CURATED
        assert p.retrieval_source == ""

    def test_provenance_to_dict(self):
        """to_dict() must include all provenance fields."""
        p = Provenance(
            source_database="Reactome",
            pmids=["12345678", "87654321"],
            evidence_level=EvidenceLevel.EXPERIMENTAL,
            publications=["DOI:10.1234/test"],
            retrieval_source="reactome_adapter",
            last_updated="2025-01-01",
        )
        d = p.to_dict()
        assert d["source_database"] == "Reactome"
        assert d["pmids"] == ["12345678", "87654321"]
        assert d["evidence_level"] == "experimental"
        assert d["publications"] == ["DOI:10.1234/test"]
        assert d["retrieval_source"] == "reactome_adapter"
        assert d["last_updated"] == "2025-01-01"

    def test_provenance_to_dict_evidence_level_is_string(self):
        """evidence_level in the dict must be the string value, not the enum."""
        p = Provenance(evidence_level=EvidenceLevel.CLINICAL)
        d = p.to_dict()
        assert isinstance(d["evidence_level"], str)
        assert d["evidence_level"] == "clinical"


# ═══════════════════════════════════════════════════════════════════════════════
# EdgeQualifiers
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeQualifiers:
    """Test EdgeQualifiers dataclass serialization and direction qualifiers."""

    def test_edge_qualifiers_defaults(self):
        """Default EdgeQualifiers should set species to 'Homo sapiens'."""
        eq = EdgeQualifiers()
        assert eq.species == "Homo sapiens"
        assert eq.direction is None
        assert eq.causal_mechanism is None

    def test_edge_qualifiers_with_direction(self):
        """Direction qualifier should serialize into to_dict() output."""
        eq = EdgeQualifiers(direction=DirectionQualifier.UPREGULATED)
        d = eq.to_dict()
        assert d["direction"] == "UP_REGULATED"
        assert "species" in d

    def test_edge_qualifiers_with_causal_mechanism(self):
        """Causal mechanism qualifier should appear in dict when set."""
        eq = EdgeQualifiers(causal_mechanism=CausalMechanism.GAIN_OF_FUNCTION)
        d = eq.to_dict()
        assert d["causal_mechanism"] == "gain_of_function"

    def test_edge_qualifiers_with_tissue_and_disease_context(self):
        """Tissue and disease context should appear when provided."""
        eq = EdgeQualifiers(tissue_context="breast", disease_context="BRCA")
        d = eq.to_dict()
        assert d["tissue_context"] == "breast"
        assert d["disease_context"] == "BRCA"

    def test_edge_qualifiers_omits_none_fields(self):
        """Fields with None values should not appear in the dict."""
        eq = EdgeQualifiers()
        d = eq.to_dict()
        assert "direction" not in d
        assert "causal_mechanism" not in d
        assert "tissue_context" not in d
        assert "disease_context" not in d


# ═══════════════════════════════════════════════════════════════════════════════
# NodeSchema
# ═══════════════════════════════════════════════════════════════════════════════


class TestNodeSchema:
    """Test NodeSchema creation, backward compat, and serialization."""

    def test_node_schema_basic(self):
        """NodeSchema stores category, id, label, aliases."""
        node = NodeSchema(
            id="TP53",
            category=BiolinkCategory.GENE,
            label="TP53",
            aliases=["p53"],
            description="Tumour protein p53",
        )
        assert node.id == "TP53"
        assert node.category is BiolinkCategory.GENE
        assert node.label == "TP53"
        assert "p53" in node.aliases

    def test_node_schema_node_type_backward_compat(self):
        """node_type property must return the same as category."""
        node = NodeSchema(id="X", category=BiolinkCategory.DRUG, label="X")
        assert node.node_type is BiolinkCategory.DRUG
        assert node.node_type is node.category

    def test_node_schema_to_dict_contains_category(self):
        """to_dict() must include 'category' key."""
        node = NodeSchema(id="BRCA1", category=BiolinkCategory.GENE, label="BRCA1")
        d = node.to_dict()
        assert "category" in d
        assert d["category"] == "biolink:Gene"

    def test_node_schema_to_dict_contains_node_type_compat(self):
        """to_dict() must include backward-compat 'node_type' key equal to category."""
        node = NodeSchema(id="BRCA1", category=BiolinkCategory.GENE, label="BRCA1")
        d = node.to_dict()
        assert "node_type" in d
        assert d["node_type"] == d["category"]

    def test_node_schema_to_dict_all_fields(self):
        """All NodeSchema fields appear in the serialized dict."""
        node = NodeSchema(
            id="PIK3CA",
            category=BiolinkCategory.GENE,
            label="PIK3CA",
            aliases=["PI3K"],
            description="PI3K catalytic subunit alpha",
            xrefs={"ensembl": "ENSG00000121879"},
            source="test",
            properties={"score": 0.9},
        )
        d = node.to_dict()
        assert d["id"] == "PIK3CA"
        assert d["aliases"] == ["PI3K"]
        assert d["description"] == "PI3K catalytic subunit alpha"
        assert d["xrefs"]["ensembl"] == "ENSG00000121879"
        assert d["source"] == "test"
        assert d["properties"]["score"] == 0.9


# ═══════════════════════════════════════════════════════════════════════════════
# EdgeSchema
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeSchema:
    """Test EdgeSchema creation, backward compat, and serialization."""

    def _make_edge(self, **overrides) -> EdgeSchema:
        defaults = dict(
            source="TP53",
            target="MDM2",
            predicate=BiolinkPredicate.POSITIVELY_REGULATES,
            weight=0.9,
            provenance=Provenance(pmids=["10499594"]),
        )
        defaults.update(overrides)
        return EdgeSchema(**defaults)

    def test_edge_schema_to_dict_contains_predicate(self):
        """to_dict() must include the 'predicate' key."""
        e = self._make_edge()
        d = e.to_dict()
        assert "predicate" in d
        assert d["predicate"] == "biolink:positively_regulates"

    def test_edge_schema_to_dict_contains_qualifiers(self):
        """to_dict() must include 'qualifiers' as a dict."""
        e = self._make_edge()
        d = e.to_dict()
        assert "qualifiers" in d
        assert isinstance(d["qualifiers"], dict)

    def test_edge_schema_to_dict_contains_provenance(self):
        """to_dict() must include 'provenance' with source_database and pmids."""
        e = self._make_edge()
        d = e.to_dict()
        assert "provenance" in d
        assert d["provenance"]["pmids"] == ["10499594"]

    def test_edge_schema_backward_compat_edge_type(self):
        """edge_type property must return the predicate value string."""
        e = self._make_edge()
        assert e.edge_type == "biolink:positively_regulates"

    def test_edge_schema_backward_compat_pmid(self):
        """pmid property returns the first PMID or None."""
        e = self._make_edge()
        assert e.pmid == "10499594"

    def test_edge_schema_backward_compat_pmid_none(self):
        """pmid property returns None when no PMIDs are set."""
        e = self._make_edge(provenance=Provenance(pmids=[]))
        assert e.pmid is None

    def test_edge_schema_backward_compat_source_db(self):
        """source_db property returns the provenance source_database."""
        e = self._make_edge()
        assert e.source_db == "seed_brca"

    def test_edge_schema_to_dict_legacy_flat_fields(self):
        """to_dict() must include flat legacy fields: evidence, pmid, source_db."""
        e = self._make_edge()
        d = e.to_dict()
        assert "evidence" in d
        assert "pmid" in d
        assert "source_db" in d
        assert d["edge_type"] == d["predicate"]


# ═══════════════════════════════════════════════════════════════════════════════
# Association Classes → to_edge()
# ═══════════════════════════════════════════════════════════════════════════════


class TestGeneToDiseaseAssociation:
    """Test GeneToDiseaseAssociation.to_edge()."""

    def test_to_edge_produces_valid_edge_schema(self):
        assoc = GeneToDiseaseAssociation(
            gene_id="BRCA1",
            disease_id="MONDO:0007254",
            clinical_significance="pathogenic",
        )
        edge = assoc.to_edge()
        assert isinstance(edge, EdgeSchema)
        assert edge.source == "BRCA1"
        assert edge.target == "MONDO:0007254"
        assert edge.predicate is BiolinkPredicate.GENE_ASSOCIATED_WITH_CONDITION

    def test_to_edge_includes_clinical_significance(self):
        assoc = GeneToDiseaseAssociation(
            gene_id="TP53",
            disease_id="MONDO:0007254",
            clinical_significance="pathogenic",
            inheritance_pattern="autosomal_dominant",
        )
        edge = assoc.to_edge()
        assert edge.properties["clinical_significance"] == "pathogenic"
        assert edge.properties["inheritance_pattern"] == "autosomal_dominant"

    def test_to_edge_omits_none_properties(self):
        assoc = GeneToDiseaseAssociation(gene_id="BRCA2", disease_id="D")
        edge = assoc.to_edge()
        assert "clinical_significance" not in edge.properties


class TestDrugToGeneAssociation:
    """Test DrugToGeneAssociation.to_edge()."""

    def test_to_edge_produces_valid_edge_schema(self):
        assoc = DrugToGeneAssociation(
            drug_id="CHEMBL521",
            gene_id="BRCA1",
            mechanism_of_action="PARP inhibition",
            clinical_phase=3,
        )
        edge = assoc.to_edge()
        assert isinstance(edge, EdgeSchema)
        assert edge.source == "CHEMBL521"
        assert edge.target == "BRCA1"
        assert edge.predicate is BiolinkPredicate.TARGETS

    def test_to_edge_includes_mechanism_and_phase(self):
        assoc = DrugToGeneAssociation(
            drug_id="D", gene_id="G",
            mechanism_of_action="inhibition",
            clinical_phase=2,
        )
        edge = assoc.to_edge()
        assert edge.properties["mechanism_of_action"] == "inhibition"
        assert edge.properties["clinical_phase"] == 2


class TestProteinToProteinAssociation:
    """Test ProteinToProteinAssociation.to_edge()."""

    def test_to_edge_produces_valid_edge_schema(self):
        assoc = ProteinToProteinAssociation(
            protein_a="TP53",
            protein_b="MDM2",
            interaction_score=999,
            detection_method="combined",
        )
        edge = assoc.to_edge()
        assert isinstance(edge, EdgeSchema)
        assert edge.source == "TP53"
        assert edge.target == "MDM2"
        assert edge.predicate is BiolinkPredicate.PHYSICALLY_INTERACTS_WITH

    def test_to_edge_normalizes_weight(self):
        """Scores > 1 are divided by 1000 to normalize to [0,1]."""
        assoc = ProteinToProteinAssociation(protein_a="A", protein_b="B", interaction_score=900)
        edge = assoc.to_edge()
        assert edge.weight == pytest.approx(0.9, abs=0.001)

    def test_to_edge_weight_already_normalized(self):
        """Scores ≤ 1 are kept as-is."""
        assoc = ProteinToProteinAssociation(protein_a="A", protein_b="B", interaction_score=0.85)
        edge = assoc.to_edge()
        assert edge.weight == pytest.approx(0.85, abs=0.001)

    def test_to_edge_includes_interaction_score_property(self):
        assoc = ProteinToProteinAssociation(protein_a="A", protein_b="B", interaction_score=950)
        edge = assoc.to_edge()
        assert edge.properties["interaction_score"] == 950

    def test_to_edge_includes_detection_method(self):
        assoc = ProteinToProteinAssociation(
            protein_a="A", protein_b="B",
            interaction_score=900,
            detection_method="yeast_two_hybrid",
        )
        edge = assoc.to_edge()
        assert edge.properties["detection_method"] == "yeast_two_hybrid"
