"""
Graph Schema — Biolink Model-compliant node/edge types for the CEREP Knowledge Graph.

Implements the Biolink Model ontology (https://biolink.github.io/biolink-model/)
with hierarchical entity classification, directional predicates, edge qualifiers,
association classes, and mandatory provenance metadata.

All nodes and edges in the CEREP KG must conform to this schema.
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


# ══════════════════════════════════════════════════════════════════════════════
# Biolink Node Categories
# ══════════════════════════════════════════════════════════════════════════════

class BiolinkCategory(str, Enum):
    """Biolink Model hierarchical entity categories."""
    GENE = "biolink:Gene"
    PROTEIN = "biolink:Protein"
    SEQUENCE_VARIANT = "biolink:SequenceVariant"
    PATHWAY = "biolink:Pathway"
    BIOLOGICAL_PROCESS = "biolink:BiologicalProcess"
    MOLECULAR_ACTIVITY = "biolink:MolecularActivity"
    CELLULAR_COMPONENT = "biolink:CellularComponent"
    DISEASE = "biolink:Disease"
    PHENOTYPIC_FEATURE = "biolink:PhenotypicFeature"
    DRUG = "biolink:Drug"
    CHEMICAL_ENTITY = "biolink:ChemicalEntity"
    GENE_FAMILY = "biolink:GeneFamily"
    ANATOMICAL_ENTITY = "biolink:AnatomicalEntity"
    CLINICAL_FINDING = "biolink:ClinicalFinding"

    # ── Backward-compatible aliases ──────────────────────────────────────────
    @classmethod
    def from_legacy(cls, legacy: str) -> "BiolinkCategory":
        """Map legacy NodeType values to Biolink categories."""
        _map = {
            "Gene": cls.GENE,
            "Protein": cls.PROTEIN,
            "Pathway": cls.PATHWAY,
            "Drug": cls.DRUG,
            "Disease": cls.DISEASE,
        }
        return _map.get(legacy, cls.GENE)


# Convenience aliases for existing code that uses NodeType
NodeType = BiolinkCategory


# ══════════════════════════════════════════════════════════════════════════════
# Biolink Predicates (Edge Types)
# ══════════════════════════════════════════════════════════════════════════════

class BiolinkPredicate(str, Enum):
    """Biolink Model directional predicates for edges.
    All inherit from biolink:related_to root.
    """
    # Regulatory
    POSITIVELY_REGULATES = "biolink:positively_regulates"
    NEGATIVELY_REGULATES = "biolink:negatively_regulates"
    REGULATES = "biolink:regulates"

    # Physical interaction
    PHYSICALLY_INTERACTS_WITH = "biolink:physically_interacts_with"
    INTERACTS_WITH = "biolink:interacts_with"
    BINDS = "biolink:binds"

    # Functional
    AFFECTS = "biolink:affects"
    CONTRIBUTES_TO = "biolink:contributes_to"
    PARTICIPATES_IN = "biolink:participates_in"
    ENABLES = "biolink:enables"
    CATALYZES = "biolink:catalyzes"
    PHOSPHORYLATES = "biolink:phosphorylates"

    # Causal
    CAUSES = "biolink:causes"
    PREDISPOSES = "biolink:predisposes"
    PREVENTS = "biolink:prevents"

    # Disease & Clinical
    GENE_ASSOCIATED_WITH_CONDITION = "biolink:gene_associated_with_condition"
    VARIANT_ASSOCIATED_WITH_CONDITION = "biolink:condition_associated_with_gene"
    HAS_PHENOTYPE = "biolink:has_phenotype"
    TREATS = "biolink:treats"
    TARGETS = "biolink:targets"

    # Expression & Location
    EXPRESSED_IN = "biolink:expressed_in"
    LOCATED_IN = "biolink:located_in"
    COLOCALIZES_WITH = "biolink:colocalizes_with"

    # Sequence
    IS_SEQUENCE_VARIANT_OF = "biolink:is_sequence_variant_of"
    HAS_GENE_PRODUCT = "biolink:has_gene_product"

    # Association
    ASSOCIATED_WITH = "biolink:associated_with"
    CORRELATED_WITH = "biolink:correlated_with"
    RELATED_TO = "biolink:related_to"

    # ── Backward-compatible mapping ──────────────────────────────────────────
    @classmethod
    def from_legacy(cls, legacy: str) -> "BiolinkPredicate":
        """Map legacy EdgeType values to Biolink predicates."""
        _map = {
            "ACTIVATES": cls.POSITIVELY_REGULATES,
            "INHIBITS": cls.NEGATIVELY_REGULATES,
            "ASSOCIATED_WITH": cls.ASSOCIATED_WITH,
            "TARGETS": cls.TARGETS,
            "MUTATED_IN": cls.GENE_ASSOCIATED_WITH_CONDITION,
            "EXPRESSED_IN": cls.EXPRESSED_IN,
            "PHOSPHORYLATES": cls.PHOSPHORYLATES,
            "REGULATES": cls.REGULATES,
            "INTERACTS_WITH": cls.PHYSICALLY_INTERACTS_WITH,
        }
        return _map.get(legacy, cls.RELATED_TO)


# Convenience alias for existing code that uses EdgeType
EdgeType = BiolinkPredicate


# ══════════════════════════════════════════════════════════════════════════════
# Edge Qualifiers
# ══════════════════════════════════════════════════════════════════════════════

class DirectionQualifier(str, Enum):
    """Biolink DirectionQualifierEnum — contextual modifiers for edge triples."""
    UPREGULATED = "UP_REGULATED"
    DOWNREGULATED = "DOWN_REGULATED"
    ACTIVATED = "ACTIVATED"
    INHIBITED = "INHIBITED"
    INCREASED = "INCREASED"
    DECREASED = "DECREASED"
    UNCHANGED = "UNCHANGED"


class CausalMechanism(str, Enum):
    """Biolink-inspired qualifiers for causal mechanism type."""
    GAIN_OF_FUNCTION = "gain_of_function"
    LOSS_OF_FUNCTION = "loss_of_function"
    DOMINANT_NEGATIVE = "dominant_negative"
    HAPLOINSUFFICIENCY = "haploinsufficiency"
    OVEREXPRESSION = "overexpression"
    AMPLIFICATION = "amplification"
    DELETION = "deletion"
    FUSION = "fusion"
    UNKNOWN = "unknown"


class EvidenceLevel(str, Enum):
    """Evidence level classification for provenance."""
    EXPERIMENTAL = "experimental"     # direct experimental validation
    CURATED = "curated"               # expert-curated database entry
    LITERATURE = "literature"         # published literature reference
    COMPUTATIONAL = "computational"   # computationally predicted
    INFERRED = "inferred"             # transitively inferred
    CLINICAL = "clinical"             # clinical trial or case report


# ══════════════════════════════════════════════════════════════════════════════
# Association Classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Provenance:
    """Biolink has_evidence + publications provenance metadata.
    Every edge in the CEREP KG must carry provenance.
    """
    source_database: str = "seed_brca"
    pmids: List[str] = field(default_factory=list)
    evidence_level: EvidenceLevel = EvidenceLevel.CURATED
    publications: List[str] = field(default_factory=list)
    retrieval_source: str = ""        # e.g., "reactome_adapter", "string_adapter"
    last_updated: str = ""

    def to_dict(self) -> dict:
        return {
            "source_database": self.source_database,
            "pmids": self.pmids,
            "evidence_level": self.evidence_level.value,
            "publications": self.publications,
            "retrieval_source": self.retrieval_source,
            "last_updated": self.last_updated,
        }


@dataclass
class EdgeQualifiers:
    """Contextual modifiers attached to an edge triple."""
    direction: Optional[DirectionQualifier] = None
    causal_mechanism: Optional[CausalMechanism] = None
    species: str = "Homo sapiens"
    tissue_context: Optional[str] = None  # e.g., "breast", "lung"
    disease_context: Optional[str] = None  # e.g., "BRCA"

    def to_dict(self) -> dict:
        result: Dict[str, Any] = {"species": self.species}
        if self.direction:
            result["direction"] = self.direction.value
        if self.causal_mechanism:
            result["causal_mechanism"] = self.causal_mechanism.value
        if self.tissue_context:
            result["tissue_context"] = self.tissue_context
        if self.disease_context:
            result["disease_context"] = self.disease_context
        return result


# ══════════════════════════════════════════════════════════════════════════════
# Core Data Classes — Biolink-compliant Nodes and Edges
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class NodeSchema:
    """Biolink-compliant node representation.

    Every entity in the CEREP Knowledge Graph is a NodeSchema instance with
    a Biolink category, provenance source, and optional cross-references.
    """
    id: str                                      # canonical identifier (HUGO symbol, ChEMBL ID, etc.)
    category: BiolinkCategory                    # biolink:Gene, biolink:Protein, etc.
    label: str                                   # human-readable display name
    aliases: List[str] = field(default_factory=list)
    description: Optional[str] = None
    xrefs: Dict[str, str] = field(default_factory=dict)  # cross-references: {"ensembl": "ENSG...", "uniprot": "P04637"}
    source: str = "seed_brca"
    properties: Dict[str, Any] = field(default_factory=dict)  # arbitrary extra metadata

    # ── Backward compatibility ───────────────────────────────────────────────
    @property
    def node_type(self) -> BiolinkCategory:
        """Legacy accessor — returns category."""
        return self.category

    @property
    def evidence_level(self) -> str:
        return self.properties.get("evidence_level", "curated")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "category": self.category.value,
            "node_type": self.category.value,       # backward compat for frontend
            "label": self.label,
            "aliases": self.aliases,
            "description": self.description,
            "xrefs": self.xrefs,
            "source": self.source,
            "properties": self.properties,
        }


@dataclass
class EdgeSchema:
    """Biolink-compliant edge representation with mandatory provenance.

    Implements the Biolink core triple (subject, predicate, object) with
    edge qualifiers and provenance metadata.
    """
    source: str                                  # subject node ID
    target: str                                  # object node ID
    predicate: BiolinkPredicate                  # biolink:positively_regulates, etc.
    weight: float = 1.0                          # confidence / interaction score
    qualifiers: EdgeQualifiers = field(default_factory=EdgeQualifiers)
    provenance: Provenance = field(default_factory=Provenance)
    properties: Dict[str, Any] = field(default_factory=dict)

    # ── Backward compatibility ───────────────────────────────────────────────
    @property
    def edge_type(self) -> str:
        """Legacy accessor — returns predicate value."""
        return self.predicate.value

    @property
    def pmid(self) -> Optional[str]:
        """Legacy accessor — returns first PMID."""
        return self.provenance.pmids[0] if self.provenance.pmids else None

    @property
    def source_db(self) -> str:
        """Legacy accessor."""
        return self.provenance.source_database

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "predicate": self.predicate.value,
            "edge_type": self.predicate.value,      # backward compat for frontend
            "weight": self.weight,
            "qualifiers": self.qualifiers.to_dict(),
            "provenance": self.provenance.to_dict(),
            "properties": self.properties,
            # Legacy flat fields for backward compat
            "evidence": self.provenance.evidence_level.value,
            "pmid": self.pmid,
            "source_db": self.provenance.source_database,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Biolink Association Classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GeneToDiseaseAssociation:
    """Biolink GeneToDiseaseAssociation — links a gene/variant to a disease."""
    gene_id: str
    disease_id: str
    predicate: BiolinkPredicate = BiolinkPredicate.GENE_ASSOCIATED_WITH_CONDITION
    qualifiers: EdgeQualifiers = field(default_factory=EdgeQualifiers)
    provenance: Provenance = field(default_factory=Provenance)
    clinical_significance: Optional[str] = None   # pathogenic, benign, VUS
    inheritance_pattern: Optional[str] = None

    def to_edge(self) -> EdgeSchema:
        props = {}
        if self.clinical_significance:
            props["clinical_significance"] = self.clinical_significance
        if self.inheritance_pattern:
            props["inheritance_pattern"] = self.inheritance_pattern
        return EdgeSchema(
            source=self.gene_id,
            target=self.disease_id,
            predicate=self.predicate,
            qualifiers=self.qualifiers,
            provenance=self.provenance,
            properties=props,
        )


@dataclass
class GeneToPathwayAssociation:
    """Biolink GeneToPathwayAssociation — links a gene to a biological pathway."""
    gene_id: str
    pathway_id: str
    predicate: BiolinkPredicate = BiolinkPredicate.PARTICIPATES_IN
    qualifiers: EdgeQualifiers = field(default_factory=EdgeQualifiers)
    provenance: Provenance = field(default_factory=Provenance)

    def to_edge(self) -> EdgeSchema:
        return EdgeSchema(
            source=self.gene_id,
            target=self.pathway_id,
            predicate=self.predicate,
            qualifiers=self.qualifiers,
            provenance=self.provenance,
        )


@dataclass
class DrugToGeneAssociation:
    """Links a drug/chemical entity to its gene target."""
    drug_id: str
    gene_id: str
    predicate: BiolinkPredicate = BiolinkPredicate.TARGETS
    qualifiers: EdgeQualifiers = field(default_factory=EdgeQualifiers)
    provenance: Provenance = field(default_factory=Provenance)
    mechanism_of_action: Optional[str] = None
    clinical_phase: Optional[int] = None

    def to_edge(self) -> EdgeSchema:
        props: Dict[str, Any] = {}
        if self.mechanism_of_action:
            props["mechanism_of_action"] = self.mechanism_of_action
        if self.clinical_phase is not None:
            props["clinical_phase"] = self.clinical_phase
        return EdgeSchema(
            source=self.drug_id,
            target=self.gene_id,
            predicate=self.predicate,
            qualifiers=self.qualifiers,
            provenance=self.provenance,
            properties=props,
        )


@dataclass
class ProteinToProteinAssociation:
    """Biolink PairwiseMolecularInteraction for PPI edges."""
    protein_a: str
    protein_b: str
    predicate: BiolinkPredicate = BiolinkPredicate.PHYSICALLY_INTERACTS_WITH
    interaction_score: float = 0.0   # e.g., STRING combined_score
    detection_method: Optional[str] = None
    qualifiers: EdgeQualifiers = field(default_factory=EdgeQualifiers)
    provenance: Provenance = field(default_factory=Provenance)

    def to_edge(self) -> EdgeSchema:
        props: Dict[str, Any] = {"interaction_score": self.interaction_score}
        if self.detection_method:
            props["detection_method"] = self.detection_method
        return EdgeSchema(
            source=self.protein_a,
            target=self.protein_b,
            predicate=self.predicate,
            weight=self.interaction_score / 1000.0 if self.interaction_score > 1 else self.interaction_score,
            qualifiers=self.qualifiers,
            provenance=self.provenance,
            properties=props,
        )

