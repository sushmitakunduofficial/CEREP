"""
Biolink Adapters — BioCypher-style data ingestion adapters for the CEREP Knowledge Graph.

Each adapter streams Biolink-compliant nodes and edges from a specific biological
database. Adapters are modular, reusable, and decoupled from the graph store.

Architecture:
    BioCypherAdapter (ABC)
    ├── ReactomeAdapter      — pathway nodes + reaction edges
    ├── STRINGAdapter        — protein-protein interaction edges
    ├── GOAdapter            — Gene Ontology terms + annotations
    └── OpenTargetsAdapter   — disease-gene-drug associations
"""
from abc import ABC, abstractmethod
from typing import List, Generator, Tuple, Dict, Any, Optional
import json
from pathlib import Path

from backend.graph.schema import (
    NodeSchema, EdgeSchema, BiolinkCategory, BiolinkPredicate,
    Provenance, EdgeQualifiers, EvidenceLevel,
    DirectionQualifier, CausalMechanism,
    GeneToDiseaseAssociation, GeneToPathwayAssociation,
    DrugToGeneAssociation, ProteinToProteinAssociation,
)
from backend.core.logging import get_logger

logger = get_logger("graph.adapters")


# ══════════════════════════════════════════════════════════════════════════════
# Abstract Adapter
# ══════════════════════════════════════════════════════════════════════════════

class BioCypherAdapter(ABC):
    """Base adapter interface — streams nodes and edges from a data source."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Adapter identifier (e.g., 'reactome', 'string')."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Data source version."""

    @abstractmethod
    def get_nodes(self) -> Generator[NodeSchema, None, None]:
        """Yield Biolink-compliant nodes from the data source."""

    @abstractmethod
    def get_edges(self) -> Generator[EdgeSchema, None, None]:
        """Yield Biolink-compliant edges from the data source."""

    def get_statistics(self) -> Dict[str, int]:
        """Count nodes and edges (consumes generators — call last)."""
        nodes = sum(1 for _ in self.get_nodes())
        edges = sum(1 for _ in self.get_edges())
        return {"nodes": nodes, "edges": edges}


# ══════════════════════════════════════════════════════════════════════════════
# Reactome Adapter — Curated BRCA-Relevant Pathway Data
# ══════════════════════════════════════════════════════════════════════════════

class ReactomeAdapter(BioCypherAdapter):
    """Streams curated Reactome pathway nodes and gene-pathway edges.

    For production: extend to query Reactome Content Service API
    (https://reactome.org/ContentService/).
    Current: curated BRCA-relevant pathways with literature-backed edges.
    """

    @property
    def name(self) -> str:
        return "reactome"

    @property
    def version(self) -> str:
        return "87"

    def _provenance(self, pmids: Optional[List[str]] = None) -> Provenance:
        return Provenance(
            source_database="Reactome",
            pmids=pmids or [],
            evidence_level=EvidenceLevel.CURATED,
            retrieval_source="reactome_adapter",
        )

    def get_nodes(self) -> Generator[NodeSchema, None, None]:
        """Yield Reactome pathway nodes relevant to BRCA oncology."""
        pathways = [
            # Core signaling pathways
            ("R-HSA-2219528", "PI3K/AKT Signaling", "Phosphoinositide 3-kinase cascade driving cell survival and growth"),
            ("R-HSA-5674135", "MAP Kinase Cascade", "RAS-RAF-MEK-ERK signaling cascade"),
            ("R-HSA-69278", "Cell Cycle Checkpoints", "G1/S and G2/M checkpoint regulation"),
            ("R-HSA-109582", "Hemostasis", "Wound healing and coagulation cascade"),
            # DNA repair
            ("R-HSA-5693538", "Homologous Recombination Repair", "DSB repair via sister chromatid template"),
            ("R-HSA-5693567", "HDR through Homologous Recombination", "Homology-directed repair of DSBs"),
            ("R-HSA-73894", "DNA Repair", "Global DNA damage response and repair"),
            ("R-HSA-5693607", "Processing of DNA Interstrand Crosslinks", "Fanconi anemia pathway ICL repair"),
            # Apoptosis & cell death
            ("R-HSA-109581", "Apoptosis", "Programmed cell death cascade"),
            ("R-HSA-5357801", "Programmed Cell Death", "Regulated cell death mechanisms"),
            ("R-HSA-3700989", "Transcriptional Regulation by TP53", "p53-mediated transcriptional programs"),
            # Growth factor signaling
            ("R-HSA-1227990", "ERBB Signaling", "ErbB/HER family receptor signaling"),
            ("R-HSA-177929", "Signaling by EGFR", "Epidermal growth factor receptor cascade"),
            ("R-HSA-1236394", "Signaling by ERBB2", "HER2/ErbB2 oncogenic signaling"),
            ("R-HSA-6811558", "PI5P/PI3P/PI3,4P2 Regulation", "Phosphoinositide metabolism"),
            # Hormone receptor
            ("R-HSA-9009391", "ESR-mediated Signaling", "Estrogen receptor alpha signaling"),
            ("R-HSA-2644603", "Signaling by BMP", "Bone morphogenetic protein signaling"),
            # Immune
            ("R-HSA-1280218", "Adaptive Immune System", "T-cell and B-cell mediated immunity"),
            ("R-HSA-168256", "Immune System", "Global immune response pathways"),
            ("R-HSA-1280215", "Cytokine Signaling", "Interleukin and interferon cascades"),
            # Metabolism
            ("R-HSA-1430728", "Metabolism", "Central metabolic pathways"),
            ("R-HSA-556833", "Metabolism of Lipids", "Fatty acid and lipid metabolism"),
            # Chromatin & epigenetics
            ("R-HSA-3214841", "Chromatin Modifying Enzymes", "Histone modification and chromatin remodeling"),
            ("R-HSA-212165", "Epigenetic Gene Regulation", "DNA methylation and histone modification"),
        ]
        for pid, label, desc in pathways:
            yield NodeSchema(
                id=pid,
                category=BiolinkCategory.PATHWAY,
                label=label,
                description=desc,
                xrefs={"reactome": pid},
                source="reactome",
            )

        # Also yield key genes as Reactome annotates them
        genes = [
            ("TP53", "Tumor protein p53", ["p53", "TRP53"], {"ensembl": "ENSG00000141510", "uniprot": "P04637"}),
            ("BRCA1", "BRCA1 DNA repair", ["BRCA-1", "RNF53"], {"ensembl": "ENSG00000012048", "uniprot": "P38398"}),
            ("BRCA2", "BRCA2 DNA repair", ["BRCA-2", "FANCD1"], {"ensembl": "ENSG00000139618", "uniprot": "P51587"}),
            ("PIK3CA", "PI3K catalytic subunit alpha", ["PI3K", "p110alpha"], {"ensembl": "ENSG00000121879", "uniprot": "P42336"}),
            ("AKT1", "AKT serine/threonine kinase 1", ["PKB"], {"ensembl": "ENSG00000142208", "uniprot": "P31749"}),
            ("MTOR", "Mechanistic target of rapamycin", ["FRAP1"], {"ensembl": "ENSG00000198793", "uniprot": "P42345"}),
            ("ERBB2", "Receptor tyrosine-protein kinase erbB-2", ["HER2", "NEU", "CD340"], {"ensembl": "ENSG00000141736", "uniprot": "P04626"}),
            ("PTEN", "Phosphatase and tensin homolog", ["MMAC1", "TEP1"], {"ensembl": "ENSG00000171862", "uniprot": "P60484"}),
            ("RB1", "RB transcriptional corepressor 1", ["ppRB", "RB"], {"ensembl": "ENSG00000139687", "uniprot": "P06400"}),
            ("KRAS", "KRAS proto-oncogene", ["Ki-RAS", "KRAS2"], {"ensembl": "ENSG00000133703", "uniprot": "P01116"}),
            ("MYC", "MYC proto-oncogene", ["c-MYC", "bHLHe39"], {"ensembl": "ENSG00000136997", "uniprot": "P01106"}),
            ("CDH1", "Cadherin 1", ["E-Cadherin", "ECAD"], {"ensembl": "ENSG00000039068", "uniprot": "P12830"}),
            ("ESR1", "Estrogen receptor alpha", ["ER-alpha", "NR3A1"], {"ensembl": "ENSG00000091831", "uniprot": "P03372"}),
            ("PGR", "Progesterone receptor", ["PR", "NR3C3"], {"ensembl": "ENSG00000082175", "uniprot": "P06401"}),
            ("PALB2", "Partner and localizer of BRCA2", ["FANCN"], {"ensembl": "ENSG00000083093", "uniprot": "Q86YC2"}),
            ("ATM", "ATM serine/threonine kinase", ["ATA", "ATC"], {"ensembl": "ENSG00000149311", "uniprot": "Q13315"}),
            ("CHEK2", "Checkpoint kinase 2", ["CHK2", "RAD53"], {"ensembl": "ENSG00000183765", "uniprot": "O96017"}),
            ("RAD51", "RAD51 recombinase", ["RECA"], {"ensembl": "ENSG00000051180", "uniprot": "Q06609"}),
            ("MDM2", "MDM2 proto-oncogene", ["HDM2"], {"ensembl": "ENSG00000135679", "uniprot": "Q00987"}),
            ("BARD1", "BRCA1-associated RING domain protein 1", [], {"ensembl": "ENSG00000138376", "uniprot": "Q99728"}),
            ("GATA3", "GATA binding protein 3", ["HDR"], {"ensembl": "ENSG00000107485", "uniprot": "P23771"}),
            ("FOXA1", "Forkhead box A1", ["HNF3A"], {"ensembl": "ENSG00000129514", "uniprot": "P55317"}),
            ("MAP2K1", "MAP kinase kinase 1", ["MEK1", "MKK1"], {"ensembl": "ENSG00000169032", "uniprot": "Q02750"}),
            ("MAP2K2", "MAP kinase kinase 2", ["MEK2", "MKK2"], {"ensembl": "ENSG00000126934", "uniprot": "P36507"}),
            ("MAPK1", "Mitogen-activated protein kinase 1", ["ERK2"], {"ensembl": "ENSG00000100030", "uniprot": "P28482"}),
            ("MAPK3", "Mitogen-activated protein kinase 3", ["ERK1"], {"ensembl": "ENSG00000102882", "uniprot": "P27361"}),
            ("BRAF", "B-Raf proto-oncogene", ["RAFB1"], {"ensembl": "ENSG00000157764", "uniprot": "P15056"}),
            ("RAF1", "Raf-1 proto-oncogene", ["CRAF"], {"ensembl": "ENSG00000132155", "uniprot": "P04049"}),
            ("EGFR", "Epidermal growth factor receptor", ["ERBB1", "HER1"], {"ensembl": "ENSG00000146648", "uniprot": "P00533"}),
            ("JAK2", "Janus kinase 2", [], {"ensembl": "ENSG00000096968", "uniprot": "O60674"}),
            ("STAT3", "Signal transducer and activator of transcription 3", [], {"ensembl": "ENSG00000168610", "uniprot": "P40763"}),
            ("BCL2", "BCL2 apoptosis regulator", [], {"ensembl": "ENSG00000171791", "uniprot": "P10415"}),
            ("BAX", "BCL2-associated X protein", [], {"ensembl": "ENSG00000087088", "uniprot": "Q07812"}),
            ("CASP3", "Caspase-3", ["CPP32"], {"ensembl": "ENSG00000164305", "uniprot": "P42574"}),
            ("CASP9", "Caspase-9", ["APAF3"], {"ensembl": "ENSG00000132906", "uniprot": "P55211"}),
            ("CDKN1A", "Cyclin-dependent kinase inhibitor 1A", ["p21", "WAF1", "CIP1"], {"ensembl": "ENSG00000124762", "uniprot": "P38936"}),
            ("CDKN2A", "Cyclin-dependent kinase inhibitor 2A", ["p16", "INK4a", "ARF"], {"ensembl": "ENSG00000147889", "uniprot": "P42771"}),
            ("CDK4", "Cyclin-dependent kinase 4", [], {"ensembl": "ENSG00000135446", "uniprot": "P11802"}),
            ("CDK6", "Cyclin-dependent kinase 6", [], {"ensembl": "ENSG00000105810", "uniprot": "Q00534"}),
            ("CCND1", "Cyclin D1", ["BCL1", "PRAD1"], {"ensembl": "ENSG00000110092", "uniprot": "P24385"}),
            ("VEGFA", "Vascular endothelial growth factor A", ["VEGF"], {"ensembl": "ENSG00000112715", "uniprot": "P15692"}),
            ("KDR", "Kinase insert domain receptor", ["VEGFR2", "FLK1"], {"ensembl": "ENSG00000128052", "uniprot": "P35968"}),
            ("NOTCH1", "Notch receptor 1", ["hN1", "TAN1"], {"ensembl": "ENSG00000148400", "uniprot": "P46531"}),
            ("WNT1", "Wnt family member 1", ["INT1"], {"ensembl": "ENSG00000125084", "uniprot": "P04628"}),
            ("CTNNB1", "Catenin beta-1", ["beta-catenin"], {"ensembl": "ENSG00000168036", "uniprot": "P35222"}),
            ("APC", "APC regulator of WNT signaling", [], {"ensembl": "ENSG00000134982", "uniprot": "P25054"}),
        ]
        for gid, desc, aliases, xrefs in genes:
            yield NodeSchema(
                id=gid,
                category=BiolinkCategory.GENE,
                label=gid,
                aliases=aliases,
                description=desc,
                xrefs=xrefs,
                source="reactome",
            )

        # Proteins
        proteins = [
            ("MDM2_PROTEIN", "MDM2 protein", "p53-binding ubiquitin ligase"),
            ("AKT1_PROTEIN", "AKT1 protein", "Serine/threonine-protein kinase AKT1"),
            ("MTOR_PROTEIN", "mTOR protein", "Mechanistic target of rapamycin kinase"),
        ]
        for pid, label, desc in proteins:
            yield NodeSchema(
                id=pid,
                category=BiolinkCategory.PROTEIN,
                label=label,
                description=desc,
                source="reactome",
            )

        # Diseases
        diseases = [
            ("MONDO:0007254", "Breast cancer", "Invasive breast carcinoma", ["BRCA"]),
            ("MONDO:0008903", "Lung adenocarcinoma", "Non-small cell lung cancer adenocarcinoma subtype", ["LUAD"]),
            ("MONDO:0005575", "Colorectal cancer", "Adenocarcinoma of the colon or rectum", ["COAD"]),
            ("MONDO:0005015", "Triple-negative breast cancer", "ER-/PR-/HER2- breast carcinoma", ["TNBC"]),
        ]
        for did, label, desc, aliases in diseases:
            yield NodeSchema(
                id=did,
                category=BiolinkCategory.DISEASE,
                label=label,
                aliases=aliases,
                description=desc,
                xrefs={"mondo": did},
                source="reactome",
            )

        # Drugs
        drugs = [
            ("CHEMBL521", "Olaparib", "PARP inhibitor for BRCA-mutant cancers", {"chembl": "CHEMBL521", "drugbank": "DB09074"}),
            ("CHEMBL1201585", "Trastuzumab", "HER2-targeted monoclonal antibody", {"chembl": "CHEMBL1201585", "drugbank": "DB00072"}),
            ("CHEMBL83", "Tamoxifen", "Selective estrogen receptor modulator", {"chembl": "CHEMBL83", "drugbank": "DB00675"}),
            ("CHEMBL3545110", "Alpelisib", "PI3K alpha-selective inhibitor", {"chembl": "CHEMBL3545110", "drugbank": "DB12015"}),
            ("CHEMBL1201631", "Everolimus", "mTOR inhibitor", {"chembl": "CHEMBL1201631", "drugbank": "DB01590"}),
            ("CHEMBL3301610", "Palbociclib", "CDK4/6 inhibitor", {"chembl": "CHEMBL3301610", "drugbank": "DB09073"}),
            ("CHEMBL3545396", "Ribociclib", "CDK4/6 inhibitor", {"chembl": "CHEMBL3545396", "drugbank": "DB12001"}),
            ("CHEMBL3137309", "Abemaciclib", "CDK4/6 inhibitor", {"chembl": "CHEMBL3137309", "drugbank": "DB12001"}),
            ("CHEMBL1743", "Pertuzumab", "HER2 dimerization inhibitor", {"chembl": "CHEMBL1743", "drugbank": "DB06366"}),
            ("CHEMBL1237028", "T-DM1", "Ado-trastuzumab emtansine", {"chembl": "CHEMBL1237028"}),
            ("CHEMBL4297436", "Talazoparib", "PARP inhibitor", {"chembl": "CHEMBL4297436", "drugbank": "DB11963"}),
            ("CHEMBL1873475", "Bevacizumab", "VEGF-targeted monoclonal antibody", {"chembl": "CHEMBL1873475", "drugbank": "DB00112"}),
        ]
        for cid, label, desc, xrefs in drugs:
            yield NodeSchema(
                id=cid,
                category=BiolinkCategory.DRUG,
                label=label,
                description=desc,
                xrefs=xrefs,
                source="reactome",
            )

    def get_edges(self) -> Generator[EdgeSchema, None, None]:
        """Yield curated gene-pathway and gene-gene edges from Reactome."""
        # Gene → Pathway participations (curated from Reactome pathway annotations)
        gene_pathway_edges = [
            # PI3K/AKT pathway
            ("PIK3CA", "R-HSA-2219528", BiolinkPredicate.PARTICIPATES_IN, 0.95, ["12829596"]),
            ("AKT1", "R-HSA-2219528", BiolinkPredicate.PARTICIPATES_IN, 0.95, ["17604717"]),
            ("MTOR", "R-HSA-2219528", BiolinkPredicate.PARTICIPATES_IN, 0.90, ["17604717"]),
            ("PTEN", "R-HSA-2219528", BiolinkPredicate.NEGATIVELY_REGULATES, 0.85, ["10866302"]),
            # MAPK cascade
            ("KRAS", "R-HSA-5674135", BiolinkPredicate.PARTICIPATES_IN, 0.90, ["17496910"]),
            ("BRAF", "R-HSA-5674135", BiolinkPredicate.PARTICIPATES_IN, 0.90, ["12068308"]),
            ("RAF1", "R-HSA-5674135", BiolinkPredicate.PARTICIPATES_IN, 0.85, ["15520807"]),
            ("MAP2K1", "R-HSA-5674135", BiolinkPredicate.PARTICIPATES_IN, 0.90, ["17496910"]),
            ("MAP2K2", "R-HSA-5674135", BiolinkPredicate.PARTICIPATES_IN, 0.85, []),
            ("MAPK1", "R-HSA-5674135", BiolinkPredicate.PARTICIPATES_IN, 0.90, ["17496910"]),
            ("MAPK3", "R-HSA-5674135", BiolinkPredicate.PARTICIPATES_IN, 0.85, []),
            # DNA repair
            ("BRCA1", "R-HSA-5693538", BiolinkPredicate.PARTICIPATES_IN, 1.0, ["20378540"]),
            ("BRCA2", "R-HSA-5693538", BiolinkPredicate.PARTICIPATES_IN, 1.0, ["20378540"]),
            ("PALB2", "R-HSA-5693538", BiolinkPredicate.PARTICIPATES_IN, 0.95, ["17200672"]),
            ("RAD51", "R-HSA-5693538", BiolinkPredicate.PARTICIPATES_IN, 0.95, ["20378540"]),
            ("ATM", "R-HSA-73894", BiolinkPredicate.PARTICIPATES_IN, 0.90, ["15064730"]),
            ("CHEK2", "R-HSA-73894", BiolinkPredicate.PARTICIPATES_IN, 0.85, ["12556884"]),
            # p53 axis
            ("TP53", "R-HSA-3700989", BiolinkPredicate.PARTICIPATES_IN, 1.0, ["22869723"]),
            ("MDM2", "R-HSA-3700989", BiolinkPredicate.PARTICIPATES_IN, 0.90, ["10499594"]),
            ("CDKN1A", "R-HSA-3700989", BiolinkPredicate.PARTICIPATES_IN, 0.85, ["7523952"]),
            # Apoptosis
            ("TP53", "R-HSA-109581", BiolinkPredicate.POSITIVELY_REGULATES, 1.0, ["22869723"]),
            ("BCL2", "R-HSA-109581", BiolinkPredicate.NEGATIVELY_REGULATES, 0.90, ["9407023"]),
            ("BAX", "R-HSA-109581", BiolinkPredicate.POSITIVELY_REGULATES, 0.90, ["12189386"]),
            ("CASP3", "R-HSA-109581", BiolinkPredicate.PARTICIPATES_IN, 0.90, ["9517222"]),
            ("CASP9", "R-HSA-109581", BiolinkPredicate.PARTICIPATES_IN, 0.85, ["9517222"]),
            # Cell cycle
            ("RB1", "R-HSA-69278", BiolinkPredicate.PARTICIPATES_IN, 0.90, ["26461249"]),
            ("CDK4", "R-HSA-69278", BiolinkPredicate.PARTICIPATES_IN, 0.85, ["26461249"]),
            ("CDK6", "R-HSA-69278", BiolinkPredicate.PARTICIPATES_IN, 0.85, []),
            ("CCND1", "R-HSA-69278", BiolinkPredicate.PARTICIPATES_IN, 0.85, ["26461249"]),
            ("CDKN2A", "R-HSA-69278", BiolinkPredicate.NEGATIVELY_REGULATES, 0.85, ["26461249"]),
            # ERBB signaling
            ("ERBB2", "R-HSA-1236394", BiolinkPredicate.PARTICIPATES_IN, 1.0, ["17496910"]),
            ("EGFR", "R-HSA-177929", BiolinkPredicate.PARTICIPATES_IN, 0.95, ["17496910"]),
            # Hormone receptor
            ("ESR1", "R-HSA-9009391", BiolinkPredicate.PARTICIPATES_IN, 0.95, ["16630834"]),
        ]
        for gene, pathway, pred, weight, pmids in gene_pathway_edges:
            yield EdgeSchema(
                source=gene,
                target=pathway,
                predicate=pred,
                weight=weight,
                provenance=self._provenance(pmids),
            )

        # Gene-gene regulatory edges (curated from Reactome reactions)
        gene_gene_edges = [
            # TP53 network
            ("TP53", "MDM2", BiolinkPredicate.POSITIVELY_REGULATES, 0.9, DirectionQualifier.ACTIVATED, ["10499594"]),
            ("MDM2", "TP53", BiolinkPredicate.NEGATIVELY_REGULATES, 0.9, DirectionQualifier.INHIBITED, ["10499594"]),
            ("TP53", "CDKN1A", BiolinkPredicate.POSITIVELY_REGULATES, 0.85, DirectionQualifier.UPREGULATED, ["7523952"]),
            ("TP53", "BAX", BiolinkPredicate.POSITIVELY_REGULATES, 0.85, DirectionQualifier.UPREGULATED, ["12189386"]),
            ("TP53", "BCL2", BiolinkPredicate.NEGATIVELY_REGULATES, 0.80, DirectionQualifier.DOWNREGULATED, ["9407023"]),
            # PI3K cascade
            ("PIK3CA", "AKT1", BiolinkPredicate.POSITIVELY_REGULATES, 0.95, DirectionQualifier.ACTIVATED, ["12829596"]),
            ("AKT1", "MTOR", BiolinkPredicate.POSITIVELY_REGULATES, 0.90, DirectionQualifier.ACTIVATED, ["17604717"]),
            ("PTEN", "PIK3CA", BiolinkPredicate.NEGATIVELY_REGULATES, 0.85, DirectionQualifier.INHIBITED, ["10866302"]),
            ("AKT1", "BAX", BiolinkPredicate.NEGATIVELY_REGULATES, 0.75, DirectionQualifier.INHIBITED, []),
            ("AKT1", "MDM2", BiolinkPredicate.POSITIVELY_REGULATES, 0.70, DirectionQualifier.ACTIVATED, []),
            # MAPK cascade
            ("KRAS", "BRAF", BiolinkPredicate.POSITIVELY_REGULATES, 0.90, DirectionQualifier.ACTIVATED, ["12068308"]),
            ("BRAF", "MAP2K1", BiolinkPredicate.POSITIVELY_REGULATES, 0.90, DirectionQualifier.ACTIVATED, ["12068308"]),
            ("MAP2K1", "MAPK1", BiolinkPredicate.POSITIVELY_REGULATES, 0.90, DirectionQualifier.ACTIVATED, ["17496910"]),
            ("MAP2K1", "MAPK3", BiolinkPredicate.POSITIVELY_REGULATES, 0.85, DirectionQualifier.ACTIVATED, []),
            ("KRAS", "PIK3CA", BiolinkPredicate.POSITIVELY_REGULATES, 0.75, DirectionQualifier.ACTIVATED, []),
            # ERBB/HER2
            ("ERBB2", "PIK3CA", BiolinkPredicate.POSITIVELY_REGULATES, 0.85, DirectionQualifier.ACTIVATED, ["17496910"]),
            ("ERBB2", "KRAS", BiolinkPredicate.POSITIVELY_REGULATES, 0.75, DirectionQualifier.ACTIVATED, []),
            ("EGFR", "KRAS", BiolinkPredicate.POSITIVELY_REGULATES, 0.85, DirectionQualifier.ACTIVATED, ["17496910"]),
            ("EGFR", "PIK3CA", BiolinkPredicate.POSITIVELY_REGULATES, 0.80, DirectionQualifier.ACTIVATED, []),
            # BRCA DNA repair
            ("BRCA1", "RAD51", BiolinkPredicate.POSITIVELY_REGULATES, 0.90, DirectionQualifier.ACTIVATED, ["20378540"]),
            ("BRCA2", "RAD51", BiolinkPredicate.POSITIVELY_REGULATES, 0.90, DirectionQualifier.ACTIVATED, ["20378540"]),
            ("BRCA1", "BARD1", BiolinkPredicate.PHYSICALLY_INTERACTS_WITH, 0.95, None, ["10499594"]),
            ("PALB2", "BRCA2", BiolinkPredicate.PHYSICALLY_INTERACTS_WITH, 0.90, None, ["17200672"]),
            ("ATM", "CHEK2", BiolinkPredicate.POSITIVELY_REGULATES, 0.85, DirectionQualifier.ACTIVATED, ["15064730"]),
            ("ATM", "BRCA1", BiolinkPredicate.POSITIVELY_REGULATES, 0.85, DirectionQualifier.ACTIVATED, ["15064730"]),
            ("ATM", "TP53", BiolinkPredicate.POSITIVELY_REGULATES, 0.85, DirectionQualifier.ACTIVATED, ["15064730"]),
            ("CHEK2", "TP53", BiolinkPredicate.POSITIVELY_REGULATES, 0.80, DirectionQualifier.ACTIVATED, ["12556884"]),
            # Cell cycle
            ("RB1", "CDK4", BiolinkPredicate.NEGATIVELY_REGULATES, 0.80, DirectionQualifier.INHIBITED, ["26461249"]),
            ("CDKN2A", "CDK4", BiolinkPredicate.NEGATIVELY_REGULATES, 0.85, DirectionQualifier.INHIBITED, ["26461249"]),
            ("CDKN2A", "CDK6", BiolinkPredicate.NEGATIVELY_REGULATES, 0.85, DirectionQualifier.INHIBITED, []),
            ("CCND1", "CDK4", BiolinkPredicate.POSITIVELY_REGULATES, 0.85, DirectionQualifier.ACTIVATED, ["26461249"]),
            ("MYC", "CCND1", BiolinkPredicate.POSITIVELY_REGULATES, 0.75, DirectionQualifier.UPREGULATED, []),
            # Hormone receptor cross-talk
            ("ESR1", "PIK3CA", BiolinkPredicate.POSITIVELY_REGULATES, 0.70, DirectionQualifier.ACTIVATED, ["16630834"]),
            ("ESR1", "MYC", BiolinkPredicate.POSITIVELY_REGULATES, 0.70, DirectionQualifier.UPREGULATED, []),
            # Angiogenesis
            ("VEGFA", "KDR", BiolinkPredicate.POSITIVELY_REGULATES, 0.90, DirectionQualifier.ACTIVATED, ["15698582"]),
            # JAK/STAT
            ("JAK2", "STAT3", BiolinkPredicate.POSITIVELY_REGULATES, 0.85, DirectionQualifier.ACTIVATED, []),
            # Wnt/beta-catenin
            ("WNT1", "CTNNB1", BiolinkPredicate.POSITIVELY_REGULATES, 0.85, DirectionQualifier.ACTIVATED, []),
            ("APC", "CTNNB1", BiolinkPredicate.NEGATIVELY_REGULATES, 0.85, DirectionQualifier.INHIBITED, []),
        ]
        for src, tgt, pred, weight, direction, pmids in gene_gene_edges:
            quals = EdgeQualifiers(
                direction=direction,
                tissue_context="breast",
                disease_context="BRCA",
            )
            yield EdgeSchema(
                source=src,
                target=tgt,
                predicate=pred,
                weight=weight,
                qualifiers=quals,
                provenance=self._provenance(pmids),
            )

        # Drug → Gene target edges
        drug_targets = [
            ("CHEMBL521", "BRCA1", BiolinkPredicate.TARGETS, 0.90, "PARP inhibition (synthetic lethality)", ["23220880"]),
            ("CHEMBL521", "BRCA2", BiolinkPredicate.TARGETS, 0.90, "PARP inhibition (synthetic lethality)", ["23220880"]),
            ("CHEMBL4297436", "BRCA1", BiolinkPredicate.TARGETS, 0.90, "PARP trapping", []),
            ("CHEMBL1201585", "ERBB2", BiolinkPredicate.TARGETS, 0.95, "HER2 extracellular domain binding", ["17496910"]),
            ("CHEMBL1743", "ERBB2", BiolinkPredicate.TARGETS, 0.90, "HER2 dimerization inhibitor", []),
            ("CHEMBL83", "ESR1", BiolinkPredicate.TARGETS, 0.90, "SERM — ER antagonist in breast", ["16630834"]),
            ("CHEMBL3545110", "PIK3CA", BiolinkPredicate.TARGETS, 0.95, "PI3K alpha-selective inhibitor", ["29146937"]),
            ("CHEMBL1201631", "MTOR", BiolinkPredicate.TARGETS, 0.90, "mTOR kinase inhibitor", ["17604717"]),
            ("CHEMBL3301610", "CDK4", BiolinkPredicate.TARGETS, 0.90, "CDK4/6 selective inhibitor", ["26461249"]),
            ("CHEMBL3301610", "CDK6", BiolinkPredicate.TARGETS, 0.90, "CDK4/6 selective inhibitor", ["26461249"]),
            ("CHEMBL3545396", "CDK4", BiolinkPredicate.TARGETS, 0.90, "CDK4/6 inhibitor", []),
            ("CHEMBL3545396", "CDK6", BiolinkPredicate.TARGETS, 0.85, "CDK4/6 inhibitor", []),
            ("CHEMBL3137309", "CDK4", BiolinkPredicate.TARGETS, 0.85, "CDK4/6 inhibitor", []),
            ("CHEMBL1873475", "VEGFA", BiolinkPredicate.TARGETS, 0.90, "Anti-VEGF monoclonal antibody", ["15698582"]),
        ]
        for drug, gene, pred, weight, moa, pmids in drug_targets:
            yield EdgeSchema(
                source=drug,
                target=gene,
                predicate=pred,
                weight=weight,
                qualifiers=EdgeQualifiers(tissue_context="breast"),
                provenance=self._provenance(pmids),
                properties={"mechanism_of_action": moa},
            )

        # Gene → Disease associations
        gene_disease = [
            ("TP53", "MONDO:0007254", 1.0, "pathogenic", ["22869723"]),
            ("BRCA1", "MONDO:0007254", 1.0, "pathogenic", ["20378540"]),
            ("BRCA2", "MONDO:0007254", 1.0, "pathogenic", ["20378540"]),
            ("PIK3CA", "MONDO:0007254", 0.90, "pathogenic", ["12829596"]),
            ("ERBB2", "MONDO:0007254", 0.90, "pathogenic", ["17496910"]),
            ("CDH1", "MONDO:0007254", 0.80, "pathogenic", []),
            ("PTEN", "MONDO:0007254", 0.85, "pathogenic", ["10866302"]),
            ("ESR1", "MONDO:0007254", 0.85, "driver", ["16630834"]),
            ("GATA3", "MONDO:0007254", 0.75, "driver", []),
            ("ATM", "MONDO:0007254", 0.75, "risk_factor", ["15064730"]),
            ("CHEK2", "MONDO:0007254", 0.70, "risk_factor", ["12556884"]),
            ("PALB2", "MONDO:0007254", 0.80, "pathogenic", ["17200672"]),
        ]
        for gene, disease, weight, clin_sig, pmids in gene_disease:
            yield GeneToDiseaseAssociation(
                gene_id=gene,
                disease_id=disease,
                qualifiers=EdgeQualifiers(tissue_context="breast"),
                provenance=self._provenance(pmids),
                clinical_significance=clin_sig,
            ).to_edge()


# ══════════════════════════════════════════════════════════════════════════════
# STRING Adapter — Protein-Protein Interactions
# ══════════════════════════════════════════════════════════════════════════════

class STRINGAdapter(BioCypherAdapter):
    """Streams protein-protein interaction edges from STRING database.

    Curated high-confidence (combined_score ≥ 700) interactions for
    BRCA-relevant proteins. For production: extend to query STRING API.
    """

    @property
    def name(self) -> str:
        return "string"

    @property
    def version(self) -> str:
        return "12.0"

    def _provenance(self, score: int) -> Provenance:
        return Provenance(
            source_database="STRING",
            evidence_level=EvidenceLevel.COMPUTATIONAL if score < 900 else EvidenceLevel.EXPERIMENTAL,
            retrieval_source="string_adapter",
        )

    def get_nodes(self) -> Generator[NodeSchema, None, None]:
        """STRING adapter contributes no new nodes — genes already from Reactome."""
        return
        yield  # make it a generator

    def get_edges(self) -> Generator[EdgeSchema, None, None]:
        """Yield high-confidence PPI edges (combined_score ≥ 700)."""
        # Curated STRING interactions for BRCA-relevant proteins
        # Format: (protein_a, protein_b, combined_score, experimental_score)
        interactions = [
            ("TP53", "MDM2", 999, 920),
            ("BRCA1", "BARD1", 999, 980),
            ("BRCA1", "BRCA2", 970, 800),
            ("BRCA1", "RAD51", 977, 850),
            ("BRCA2", "RAD51", 981, 900),
            ("BRCA2", "PALB2", 994, 950),
            ("BRCA1", "TP53", 905, 700),
            ("BRCA1", "ATM", 917, 750),
            ("ATM", "CHEK2", 980, 900),
            ("ATM", "TP53", 959, 800),
            ("CHEK2", "TP53", 910, 700),
            ("PIK3CA", "AKT1", 987, 850),
            ("PIK3CA", "PTEN", 960, 800),
            ("AKT1", "MTOR", 973, 850),
            ("AKT1", "MDM2", 900, 700),
            ("AKT1", "BAX", 850, 700),
            ("ERBB2", "EGFR", 977, 900),
            ("ERBB2", "PIK3CA", 910, 750),
            ("EGFR", "KRAS", 950, 800),
            ("KRAS", "BRAF", 986, 900),
            ("BRAF", "MAP2K1", 990, 950),
            ("MAP2K1", "MAPK1", 995, 970),
            ("MAP2K1", "MAPK3", 990, 950),
            ("MAP2K2", "MAPK1", 980, 900),
            ("RB1", "CDK4", 972, 850),
            ("RB1", "CDK6", 950, 800),
            ("CCND1", "CDK4", 985, 920),
            ("CCND1", "CDK6", 970, 850),
            ("CDKN2A", "CDK4", 975, 900),
            ("CDKN2A", "CDK6", 960, 850),
            ("CDK4", "RB1", 972, 850),
            ("MYC", "CCND1", 850, 700),
            ("TP53", "CDKN1A", 990, 920),
            ("TP53", "BAX", 960, 800),
            ("TP53", "BCL2", 880, 700),
            ("BCL2", "BAX", 985, 950),
            ("CASP9", "CASP3", 975, 900),
            ("BAX", "CASP9", 900, 750),
            ("ESR1", "FOXA1", 900, 750),
            ("ESR1", "GATA3", 850, 700),
            ("JAK2", "STAT3", 960, 850),
            ("VEGFA", "KDR", 990, 950),
            ("WNT1", "CTNNB1", 950, 800),
            ("APC", "CTNNB1", 990, 950),
            ("NOTCH1", "CTNNB1", 800, 700),
        ]
        for prot_a, prot_b, combined, experimental in interactions:
            yield ProteinToProteinAssociation(
                protein_a=prot_a,
                protein_b=prot_b,
                interaction_score=combined,
                detection_method="combined" if experimental >= 900 else "predicted",
                provenance=self._provenance(combined),
            ).to_edge()


# ══════════════════════════════════════════════════════════════════════════════
# Gene Ontology Adapter
# ══════════════════════════════════════════════════════════════════════════════

class GOAdapter(BioCypherAdapter):
    """Streams Gene Ontology biological process annotations.

    Curated GO terms for BRCA-relevant genes.
    For production: extend to query QuickGO API.
    """

    @property
    def name(self) -> str:
        return "gene_ontology"

    @property
    def version(self) -> str:
        return "2024-01"

    def _provenance(self) -> Provenance:
        return Provenance(
            source_database="Gene Ontology",
            evidence_level=EvidenceLevel.CURATED,
            retrieval_source="go_adapter",
        )

    def get_nodes(self) -> Generator[NodeSchema, None, None]:
        """Yield GO biological process terms."""
        go_terms = [
            ("GO:0006915", "Apoptotic process", "A form of programmed cell death"),
            ("GO:0006281", "DNA repair", "Restoration of DNA after damage"),
            ("GO:0000724", "Double-strand break repair via HR", "Repair of DSBs by homologous recombination"),
            ("GO:0007049", "Cell cycle", "Progression through phases of the cell cycle"),
            ("GO:0008283", "Cell population proliferation", "Increase in cell number"),
            ("GO:0007165", "Signal transduction", "Initiation of a change in cell state"),
            ("GO:0016310", "Phosphorylation", "Addition of phosphate group to a molecule"),
            ("GO:0006468", "Protein phosphorylation", "Phosphorylation of a protein amino acid residue"),
            ("GO:0043066", "Negative regulation of apoptotic process", "Prevention of apoptosis"),
            ("GO:0043065", "Positive regulation of apoptotic process", "Promotion of apoptosis"),
            ("GO:0008284", "Positive regulation of cell proliferation", "Activation of cell growth"),
            ("GO:0008285", "Negative regulation of cell proliferation", "Inhibition of cell growth"),
            ("GO:0006468", "Protein phosphorylation", "Phosphorylation of protein"),
            ("GO:0016477", "Cell migration", "Movement of a cell"),
            ("GO:0006955", "Immune response", "Response to foreign agent"),
            ("GO:0001525", "Angiogenesis", "Formation of new blood vessels"),
            ("GO:0035556", "Intracellular signal transduction", "Signal relay within cell"),
            ("GO:0000086", "G2/M transition of mitotic cell cycle", "Transition from G2 to M phase"),
            ("GO:0051301", "Cell division", "Process resulting in cell division"),
            ("GO:0045893", "Positive regulation of transcription", "Activation of gene expression"),
        ]
        for goid, label, desc in go_terms:
            yield NodeSchema(
                id=goid,
                category=BiolinkCategory.BIOLOGICAL_PROCESS,
                label=label,
                description=desc,
                xrefs={"go": goid},
                source="gene_ontology",
            )

    def get_edges(self) -> Generator[EdgeSchema, None, None]:
        """Yield gene → GO term annotation edges."""
        annotations = [
            # DNA repair genes
            ("BRCA1", "GO:0000724", 1.0),
            ("BRCA2", "GO:0000724", 1.0),
            ("RAD51", "GO:0000724", 1.0),
            ("PALB2", "GO:0000724", 0.95),
            ("ATM", "GO:0006281", 0.95),
            ("CHEK2", "GO:0006281", 0.90),
            # Apoptosis genes
            ("TP53", "GO:0043065", 1.0),
            ("BAX", "GO:0043065", 0.95),
            ("BCL2", "GO:0043066", 0.95),
            ("CASP3", "GO:0006915", 0.95),
            ("CASP9", "GO:0006915", 0.90),
            # Cell cycle
            ("RB1", "GO:0007049", 0.95),
            ("CDK4", "GO:0007049", 0.90),
            ("CDK6", "GO:0007049", 0.90),
            ("CCND1", "GO:0007049", 0.90),
            ("CDKN1A", "GO:0007049", 0.85),
            ("CDKN2A", "GO:0007049", 0.85),
            ("MYC", "GO:0008283", 0.90),
            # Signaling
            ("PIK3CA", "GO:0007165", 0.95),
            ("AKT1", "GO:0007165", 0.95),
            ("MTOR", "GO:0007165", 0.90),
            ("KRAS", "GO:0007165", 0.95),
            ("BRAF", "GO:0007165", 0.90),
            ("ERBB2", "GO:0007165", 0.95),
            ("EGFR", "GO:0007165", 0.95),
            # Proliferation
            ("PIK3CA", "GO:0008284", 0.85),
            ("AKT1", "GO:0008284", 0.85),
            ("MYC", "GO:0008284", 0.90),
            ("ERBB2", "GO:0008284", 0.90),
            ("PTEN", "GO:0008285", 0.90),
            ("RB1", "GO:0008285", 0.85),
            ("TP53", "GO:0008285", 0.90),
            # Phosphorylation
            ("ATM", "GO:0006468", 0.95),
            ("CHEK2", "GO:0006468", 0.90),
            ("AKT1", "GO:0006468", 0.90),
            ("JAK2", "GO:0006468", 0.90),
            # Angiogenesis
            ("VEGFA", "GO:0001525", 0.95),
            ("KDR", "GO:0001525", 0.90),
        ]
        for gene, go_term, weight in annotations:
            yield EdgeSchema(
                source=gene,
                target=go_term,
                predicate=BiolinkPredicate.PARTICIPATES_IN,
                weight=weight,
                provenance=self._provenance(),
            )


# ══════════════════════════════════════════════════════════════════════════════
# Open Targets Adapter
# ══════════════════════════════════════════════════════════════════════════════

class OpenTargetsAdapter(BioCypherAdapter):
    """Streams disease-gene association and drug-target data from Open Targets.

    Curated BRCA-relevant associations.
    For production: extend to query Open Targets GraphQL API.
    """

    @property
    def name(self) -> str:
        return "open_targets"

    @property
    def version(self) -> str:
        return "24.09"

    def _provenance(self) -> Provenance:
        return Provenance(
            source_database="Open Targets",
            evidence_level=EvidenceLevel.CURATED,
            retrieval_source="opentargets_adapter",
        )

    def get_nodes(self) -> Generator[NodeSchema, None, None]:
        """Open Targets contributes no new nodes."""
        return
        yield

    def get_edges(self) -> Generator[EdgeSchema, None, None]:
        """Yield disease-gene association scores from Open Targets."""
        # (gene, disease, overall_score, genetic_association, somatic_mutation, known_drug)
        associations = [
            ("TP53", "MONDO:0007254", 0.95, 0.90, 0.98, 0.30),
            ("BRCA1", "MONDO:0007254", 0.98, 0.99, 0.85, 0.80),
            ("BRCA2", "MONDO:0007254", 0.97, 0.99, 0.80, 0.75),
            ("PIK3CA", "MONDO:0007254", 0.92, 0.85, 0.95, 0.90),
            ("ERBB2", "MONDO:0007254", 0.95, 0.80, 0.90, 0.95),
            ("PTEN", "MONDO:0007254", 0.88, 0.80, 0.85, 0.50),
            ("ESR1", "MONDO:0007254", 0.90, 0.75, 0.70, 0.95),
            ("CDH1", "MONDO:0007254", 0.82, 0.85, 0.70, 0.20),
            ("GATA3", "MONDO:0007254", 0.78, 0.70, 0.75, 0.10),
            ("ATM", "MONDO:0007254", 0.75, 0.80, 0.60, 0.40),
            ("CHEK2", "MONDO:0007254", 0.72, 0.80, 0.50, 0.30),
            ("PALB2", "MONDO:0007254", 0.80, 0.85, 0.65, 0.60),
            ("CDK4", "MONDO:0007254", 0.70, 0.50, 0.60, 0.85),
            ("MYC", "MONDO:0007254", 0.75, 0.60, 0.80, 0.20),
            ("RB1", "MONDO:0007254", 0.68, 0.55, 0.65, 0.30),
        ]
        for gene, disease, overall, genetic, somatic, drug in associations:
            yield EdgeSchema(
                source=gene,
                target=disease,
                predicate=BiolinkPredicate.ASSOCIATED_WITH,
                weight=overall,
                provenance=self._provenance(),
                properties={
                    "genetic_association": genetic,
                    "somatic_mutation": somatic,
                    "known_drug": drug,
                    "overall_score": overall,
                },
            )


# ══════════════════════════════════════════════════════════════════════════════
# Adapter Registry
# ══════════════════════════════════════════════════════════════════════════════

ALL_ADAPTERS: List[BioCypherAdapter] = [
    ReactomeAdapter(),
    STRINGAdapter(),
    GOAdapter(),
    OpenTargetsAdapter(),
]


def get_adapter(name: str) -> Optional[BioCypherAdapter]:
    """Look up adapter by name."""
    for adapter in ALL_ADAPTERS:
        if adapter.name == name:
            return adapter
    return None


def get_all_adapters() -> List[BioCypherAdapter]:
    """Return all registered adapters."""
    return ALL_ADAPTERS
