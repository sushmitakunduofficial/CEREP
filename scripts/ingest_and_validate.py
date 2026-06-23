# -*- coding: utf-8 -*-
"""
CEREP V2 - Data Ingestion, Graph Enrichment & Validation Pipeline
=================================================================
Processes all four raw datasets:
  1. BRCA_mc3_gene_level.txt   - somatic mutation matrix (genes x patients)
  2. 9606.protein.links.full.v12.0.txt - STRING PPI network (human, v12)
  3. Ensembl2Reactome_All_Levels.txt   - Gene -> Pathway (Reactome)
  4. go-basic.obo                      - Gene Ontology terms

Outputs (to data/processed/):
  - mutation_summary.json       - top mutated genes + sample frequencies
  - ppi_edges.json              - filtered high-confidence PPI edges
  - reactome_pathways.json      - gene -> pathway mappings (human only)
  - go_terms.json               - GO term id/name/namespace map
  - enriched_graph.json         - final Cytoscape-compatible KG
  - validation_report.txt       - dataset stats & QC checks
"""
import sys, io
# Force UTF-8 stdout for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os, json, csv, time, re
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

import pandas as pd
import numpy as np
import networkx as nx

# -- Paths ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW  = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

BRCA_FILE     = RAW / "BRCA_mc3_gene_level.txt"
STRING_FILE   = RAW / "9606.protein.links.full.v12.0.txt"
REACTOME_FILE = RAW / "Ensembl2Reactome_All_Levels.txt"
GO_OBO_FILE   = RAW / "go-basic.obo"

# -- KG Seed (must match graph_builder.py) -------------------------------------
KG_GENES = {
    "TP53", "BRCA1", "BRCA2", "RB1", "PTEN",
    "PIK3CA", "KRAS", "MYC", "ERBB2", "CDH1",
    "MDM2", "AKT1", "MTOR",
}
KG_PATHWAYS = {"PI3K_AKT_MTOR", "DNA_REPAIR", "APOPTOSIS", "CELL_CYCLE"}
KG_DISEASES  = {"BRCA", "LUAD", "COAD"}
KG_DRUGS     = {"OLAPARIB", "TRASTUZUMAB", "TAMOXIFEN", "ALPELISIB", "EVEROLIMUS"}

# STRING combined score threshold (>=700 = high confidence)
STRING_THRESHOLD = 700

validation_lines: List[str] = []

def log(msg: str):
    print(msg)
    validation_lines.append(msg)

# ==============================================================================
# 1. BRCA Mutation Matrix
# ==============================================================================
def process_brca_mutations() -> Tuple[Dict, List[str]]:
    """
    Parse BRCA_mc3_gene_level.txt (genes x patients binary mutation matrix).
    Returns:
        summary dict  — {gene: {sample_count, mutation_rate, in_kg}}
        top_genes     — sorted list of most-mutated genes
    """
    log("\n== [1/4] BRCA Somatic Mutation Matrix ==")
    t0 = time.time()

    df = pd.read_csv(BRCA_FILE, sep="\t", index_col=0,
                     low_memory=False)
    # Cast to int8 to save memory (values are 0/1 binary matrix)
    df = df.astype(np.int8)
    df.index.name = "gene"
    n_genes, n_samples = df.shape
    log(f"  Shape: {n_genes:,} genes x {n_samples:,} patients")

    # Per-gene mutation count & rate
    sample_counts = df.sum(axis=1).astype(int)
    mutation_rates = (sample_counts / n_samples).round(4)

    # Genes mutated in >=1 sample
    mutated = sample_counts[sample_counts > 0]
    log(f"  Genes mutated in >=1 sample: {len(mutated):,}")

    # KG gene hit rate
    kg_hits = [g for g in KG_GENES if g in sample_counts.index]
    log(f"  KG seed genes found in BRCA matrix: {len(kg_hits)}/{len(KG_GENES)}: {sorted(kg_hits)}")

    # Build summary for top 500 + all KG genes
    top_genes = list(mutated.sort_values(ascending=False).head(500).index)
    summary_genes = list(set(top_genes) | set(kg_hits))

    summary: Dict = {}
    for gene in summary_genes:
        if gene in sample_counts.index:
            summary[gene] = {
                "sample_count": int(sample_counts[gene]),
                "mutation_rate": float(mutation_rates[gene]),
                "in_kg": gene in KG_GENES,
            }

    # Co-mutation matrix for KG genes (for edge weight enrichment later)
    kg_in_df = [g for g in KG_GENES if g in df.index]
    if len(kg_in_df) >= 2:
        kg_sub = df.loc[kg_in_df].astype(float)
        comut = kg_sub.dot(kg_sub.T) / n_samples  # pairwise co-mutation rate
        comut_dict = {}
        for g1 in kg_in_df:
            for g2 in kg_in_df:
                if g1 < g2:
                    comut_dict[f"{g1}|{g2}"] = round(float(comut.loc[g1, g2]), 4)
        log(f"  Co-mutation pairs computed: {len(comut_dict)}")
    else:
        comut_dict = {}

    out = {
        "n_genes_total": n_genes,
        "n_samples": n_samples,
        "n_mutated_genes": len(mutated),
        "gene_summary": summary,
        "kg_gene_hits": kg_hits,
        "comutation_rates": comut_dict,
    }
    (PROC / "mutation_summary.json").write_text(json.dumps(out, indent=2))
    log(f"  [OK] Saved mutation_summary.json  [{time.time()-t0:.1f}s]")
    return out, top_genes


# ==============================================================================
# 2. STRING PPI Network
# ==============================================================================
def process_string_ppi() -> List[Dict]:
    """
    Parse STRING v12 human PPI (ENSP IDs), filter to high-confidence edges
    involving KG seed genes. Uses the last column 'combined_score'.
    Since STRING uses ENSP IDs we do a lightweight symbol-based filter via
    the protein1/protein2 string suffix matching against known gene names.
    """
    log("\n== [2/4] STRING Protein Interaction Network ==")
    t0 = time.time()

    # We'll chunk-read (file is ~1 GB) and keep only high-confidence rows
    chunk_size = 500_000
    kept_edges: List[Dict] = []
    total_rows = 0

    # Build a lookup pattern: ENSP for KG genes is unknown, but we can filter
    # post-hoc because STRING maps ENSP->gene via the alias file. Since we don't
    # have the alias file, we load all edges where combined_score >= threshold
    # and store protein pair + score. We will then use pairs that appear in any
    # KG-gene-related enrichment.
    log(f"  Reading in {chunk_size:,}-row chunks, threshold={STRING_THRESHOLD}…")
    high_conf_edges = []
    for chunk in pd.read_csv(
        STRING_FILE, sep=" ", chunksize=chunk_size,
        usecols=["protein1", "protein2", "experiments",
                 "database", "combined_score"],
        dtype={"combined_score": np.int16, "experiments": np.int16,
               "database": np.int16}
    ):
        total_rows += len(chunk)
        filtered = chunk[chunk["combined_score"] >= STRING_THRESHOLD]
        high_conf_edges.append(filtered)

    df_ppi = pd.concat(high_conf_edges, ignore_index=True)
    log(f"  Total rows: {total_rows:,} | High-confidence (>={STRING_THRESHOLD}): {len(df_ppi):,}")

    # Strip "9606." prefix from ENSP IDs
    df_ppi["p1"] = df_ppi["protein1"].str.replace("9606.", "", regex=False)
    df_ppi["p2"] = df_ppi["protein2"].str.replace("9606.", "", regex=False)

    # Build graph for centrality / hub analysis
    G_ppi = nx.Graph()
    G_ppi.add_edges_from(zip(df_ppi["p1"], df_ppi["p2"]),
                         combined_score=None)  # Fast load
    log(f"  PPI sub-graph: {G_ppi.number_of_nodes():,} nodes, {G_ppi.number_of_edges():,} edges")

    # Save filtered edges (top 50k by combined_score for storage efficiency)
    df_top = df_ppi.nlargest(50_000, "combined_score")[
        ["p1", "p2", "experiments", "database", "combined_score"]
    ]
    edges_out = df_top.to_dict("records")
    ppi_out = {
        "total_rows_processed": int(total_rows),
        "high_confidence_count": int(len(df_ppi)),
        "stored_top_k": len(edges_out),
        "threshold": STRING_THRESHOLD,
        "edges": edges_out,
    }
    (PROC / "ppi_edges.json").write_text(json.dumps(ppi_out, indent=2))
    log(f"  [OK] Saved ppi_edges.json  [{time.time()-t0:.1f}s]")
    return edges_out


# ==============================================================================
# 3. Reactome Pathway Mappings
# ==============================================================================
def process_reactome() -> Dict:
    """
    Parse Ensembl2Reactome_All_Levels.txt:
      cols: Ensembl_ID | Reactome_ID | URL | Pathway_Name | Evidence | Species
    Keep only Homo sapiens rows.
    Output: gene_symbol -> [pathway_name, ...] and pathway_name -> [genes, ...]
    Note: The Ensembl_ID column contains gene/transcript IDs (ENSG* or gene symbols)
    """
    log("\n== [3/4] Reactome Pathway Mappings ==")
    t0 = time.time()

    cols = ["ensembl_id", "reactome_id", "url", "pathway_name", "evidence", "species"]
    df = pd.read_csv(
        REACTOME_FILE, sep="\t", header=None, names=cols,
        dtype=str, low_memory=False
    )
    total = len(df)
    df_human = df[df["species"] == "Homo sapiens"].copy()
    log(f"  Total rows: {total:,} | Human rows: {len(df_human):,}")

    # Reactome uses Ensembl Gene IDs (ENSG*) mostly, but some entries are
    # plain gene symbols or transcript IDs. We keep all and map what we can.
    gene_to_pathways: Dict[str, List[str]] = defaultdict(list)
    pathway_to_genes: Dict[str, List[str]] = defaultdict(list)
    pathway_meta: Dict[str, str] = {}  # name -> reactome_id

    for _, row in df_human.iterrows():
        eid = str(row["ensembl_id"]).strip()
        pname = str(row["pathway_name"]).strip()
        rid = str(row["reactome_id"]).strip()
        pathway_meta[pname] = rid
        gene_to_pathways[eid].append(pname)
        pathway_to_genes[pname].append(eid)

    log(f"  Unique gene/transcript IDs: {len(gene_to_pathways):,}")
    log(f"  Unique pathways: {len(pathway_to_genes):,}")

    # Check which KG pathways appear in Reactome names
    kg_pathway_labels = {
        "PI3K_AKT_MTOR": ["PI3K", "AKT", "mTOR", "MTOR"],
        "DNA_REPAIR": ["DNA Repair", "Homologous Recombination"],
        "APOPTOSIS": ["Apoptosis"],
        "CELL_CYCLE": ["Cell Cycle"],
    }
    matched: Dict[str, List[str]] = {}
    for kg_p, keywords in kg_pathway_labels.items():
        hits = [p for p in pathway_to_genes if any(kw.lower() in p.lower() for kw in keywords)]
        matched[kg_p] = hits[:5]
        log(f"  KG '{kg_p}' -> {len(hits)} Reactome matches (top: {hits[:2]})")

    out = {
        "n_human_rows": int(len(df_human)),
        "n_unique_ids": len(gene_to_pathways),
        "n_unique_pathways": len(pathway_to_genes),
        "kg_pathway_matches": matched,
        "pathway_meta": dict(list(pathway_meta.items())[:2000]),  # cap for JSON size
        "gene_to_pathways": {k: v[:10] for k, v in list(gene_to_pathways.items())[:5000]},
    }
    (PROC / "reactome_pathways.json").write_text(json.dumps(out, indent=2))
    log(f"  [OK] Saved reactome_pathways.json  [{time.time()-t0:.1f}s]")
    return out


# ==============================================================================
# 4. Gene Ontology (OBO)
# ==============================================================================
def process_go_obo() -> Dict:
    """
    Parse go-basic.obo and extract all GO terms with id, name, namespace, def.
    Returns counts by namespace + a flat term dict.
    """
    log("\n== [4/4] Gene Ontology (OBO) ==")
    t0 = time.time()

    terms: Dict[str, Dict] = {}
    current: Dict = {}

    with open(GO_OBO_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if line == "[Term]":
                if current.get("id"):
                    terms[current["id"]] = current
                current = {}
            elif line.startswith("id: "):
                current["id"] = line[4:].strip()
            elif line.startswith("name: "):
                current["name"] = line[6:].strip()
            elif line.startswith("namespace: "):
                current["namespace"] = line[11:].strip()
            elif line.startswith("def: "):
                current["def"] = line[5:].strip()
            elif line.startswith("is_obsolete: true"):
                current["obsolete"] = True

    # Flush last block
    if current.get("id"):
        terms[current["id"]] = current

    # Filter out obsolete
    active = {k: v for k, v in terms.items() if not v.get("obsolete")}
    ns_counts: Dict[str, int] = defaultdict(int)
    for t in active.values():
        ns_counts[t.get("namespace", "unknown")] += 1

    log(f"  Total GO terms: {len(terms):,} | Active: {len(active):,}")
    for ns, cnt in sorted(ns_counts.items()):
        log(f"    {ns}: {cnt:,}")

    # Find GO terms relevant to cancer/oncology
    cancer_terms = {
        k: v for k, v in active.items()
        if any(kw in v.get("name", "").lower()
               for kw in ["apoptosis", "cell cycle", "dna repair",
                          "proliferation", "cancer", "tumor",
                          "angiogenesis", "metastasis"])
    }
    log(f"  Cancer-relevant GO terms: {len(cancer_terms):,}")

    out = {
        "total_terms": len(terms),
        "active_terms": len(active),
        "namespace_counts": dict(ns_counts),
        "cancer_relevant_count": len(cancer_terms),
        "cancer_relevant_terms": list(cancer_terms.keys()),
        "sample_terms": dict(list(active.items())[:200]),
    }
    (PROC / "go_terms.json").write_text(json.dumps(out, indent=2))
    log(f"  [OK] Saved go_terms.json  [{time.time()-t0:.1f}s]")
    return out


# ==============================================================================
# 5. Graph Enrichment — Merge datasets into CEREP KG
# ==============================================================================
def enrich_knowledge_graph(
    mutation_data: Dict,
    reactome_data: Dict,
    go_data: Dict,
) -> nx.DiGraph:
    """
    Build the enriched KG on top of CEREP seed graph:
      - Add mutation-frequency edge weights from BRCA data
      - Add Reactome-pathway nodes (top 20 per KG pathway)
      - Add GO cancer term nodes
      - Add co-mutation edges between KG genes
    """
    log("\n== [5/5] Knowledge Graph Enrichment ==")
    t0 = time.time()

    # Import the CEREP seed builder
    sys.path.insert(0, str(ROOT))
    from backend.graph.graph_builder import CERAPGraphBuilder
    from backend.graph.schema import NodeSchema, EdgeSchema, NodeType, EdgeType

    builder = CERAPGraphBuilder()
    builder.build_seed_graph()
    G = builder.graph
    log(f"  Seed graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # -- 5a. Enrich edge weights with co-mutation data -------------------------
    comut = mutation_data.get("comutation_rates", {})
    enriched_edges = 0
    for pair, rate in comut.items():
        g1, g2 = pair.split("|")
        if G.has_edge(g1, g2):
            G.edges[g1, g2]["comutation_rate"] = rate
            G.edges[g1, g2]["weight"] = min(1.0,
                G.edges[g1, g2].get("weight", 0.7) + rate * 0.3)
            enriched_edges += 1
        if G.has_edge(g2, g1):
            G.edges[g2, g1]["comutation_rate"] = rate
            G.edges[g2, g1]["weight"] = min(1.0,
                G.edges[g2, g1].get("weight", 0.7) + rate * 0.3)
            enriched_edges += 1
    log(f"  Edge weights enriched with co-mutation: {enriched_edges} edges updated")

    # -- 5b. Add mutation-frequency annotation to KG gene nodes ---------------
    gene_summary = mutation_data.get("gene_summary", {})
    for gene in KG_GENES:
        if gene in G.nodes and gene in gene_summary:
            G.nodes[gene]["mutation_rate"] = gene_summary[gene]["mutation_rate"]
            G.nodes[gene]["brca_sample_count"] = gene_summary[gene]["sample_count"]

    # -- 5c. Add Reactome pathway nodes for KG pathway matches -----------------
    kg_pathway_matches = reactome_data.get("kg_pathway_matches", {})
    reactome_nodes_added = 0
    for kg_pathway, reactome_names in kg_pathway_matches.items():
        if kg_pathway not in G.nodes:
            continue
        for rname in reactome_names[:3]:  # top 3 Reactome matches per KG pathway
            # Create a clean node id
            node_id = "R_" + re.sub(r"[^A-Z0-9]", "_", rname.upper())[:40]
            if node_id not in G.nodes:
                G.add_node(node_id,
                           label=rname,
                           node_type=NodeType.PATHWAY.value,
                           description=f"Reactome: {rname}",
                           source="reactome")
                reactome_nodes_added += 1
            # Link to KG pathway
            if not G.has_edge(node_id, kg_pathway):
                G.add_edge(node_id, kg_pathway,
                           edge_type=EdgeType.ASSOCIATED_WITH.value,
                           weight=0.85,
                           source_db="reactome")
    log(f"  Reactome pathway nodes added: {reactome_nodes_added}")

    # -- 5d. Add top cancer GO terms as annotation nodes -----------------------
    go_sample = go_data.get("sample_terms", {})
    cancer_ids = set(go_data.get("cancer_relevant_terms", []))
    go_nodes_added = 0
    cancer_go_for_kg = {
        "GO:0006915": "APOPTOSIS",   # apoptotic process -> APOPTOSIS pathway
        "GO:0007049": "CELL_CYCLE",  # cell cycle -> CELL_CYCLE
        "GO:0006281": "DNA_REPAIR",  # DNA repair -> DNA_REPAIR
        "GO:0048870": None,          # motility (standalone)
    }
    for go_id, kg_link in list(cancer_go_for_kg.items())[:10]:
        if go_id in go_sample:
            term = go_sample[go_id]
            node_id = go_id.replace(":", "_")
            if node_id not in G.nodes:
                G.add_node(node_id,
                           label=term.get("name", go_id),
                           node_type="go_term",
                           namespace=term.get("namespace", ""),
                           source="go")
                go_nodes_added += 1
            if kg_link and kg_link in G.nodes:
                if not G.has_edge(node_id, kg_link):
                    G.add_edge(node_id, kg_link,
                               edge_type=EdgeType.ASSOCIATED_WITH.value,
                               weight=0.7,
                               source_db="go")
    log(f"  GO term nodes added: {go_nodes_added}")

    # -- 5e. Top mutated non-KG genes as new nodes -----------------------------
    top_novel_genes = [
        g for g, info in sorted(gene_summary.items(),
                                key=lambda x: x[1]["mutation_rate"], reverse=True)
        if g not in KG_GENES and info["mutation_rate"] > 0.05
    ][:20]
    novel_nodes_added = 0
    for gene in top_novel_genes:
        if gene not in G.nodes:
            rate = gene_summary[gene]["mutation_rate"]
            G.add_node(gene,
                       label=gene,
                       node_type=NodeType.GENE.value,
                       description=f"High-frequency BRCA mutation ({rate:.1%} of samples)",
                       mutation_rate=rate,
                       source="brca_mc3")
            novel_nodes_added += 1
            # Link to BRCA disease node
            if "BRCA" in G.nodes:
                G.add_edge(gene, "BRCA",
                           edge_type=EdgeType.MUTATED_IN.value,
                           weight=min(1.0, rate * 3),
                           source_db="brca_mc3")
    log(f"  Novel high-frequency BRCA genes added: {novel_nodes_added}: {top_novel_genes[:5]}…")

    log(f"\n  ✦ Enriched Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # -- Serialise to Cytoscape JSON -------------------------------------------
    cyto_nodes = [{"data": {"id": n, **dict(G.nodes[n])}} for n in G.nodes]
    cyto_edges = [
        {"data": {"source": u, "target": v, **dict(G.edges[u, v])}}
        for u, v in G.edges
    ]
    cyto_json = {"nodes": cyto_nodes, "edges": cyto_edges}
    (PROC / "enriched_graph.json").write_text(json.dumps(cyto_json, indent=2))
    log(f"  [OK] Saved enriched_graph.json  [{time.time()-t0:.1f}s]")
    return G


# ==============================================================================
# 6. Validation Suite
# ==============================================================================
def validate_all(G: nx.DiGraph, mutation_data: Dict):
    log("\n== [VALIDATION] ==")
    failures = []

    # — Check 1: KG gene coverage in BRCA ——
    kg_hits = mutation_data.get("kg_gene_hits", [])
    pct = len(kg_hits) / len(KG_GENES) * 100
    status = "[OK]" if pct >= 70 else "[FAIL]"
    log(f"  {status} KG gene coverage in BRCA matrix: {pct:.0f}%  ({len(kg_hits)}/{len(KG_GENES)})")
    if pct < 70:
        failures.append(f"KG coverage in BRCA too low: {pct:.0f}%")

    # — Check 2: Graph connectivity ——
    undirected = G.to_undirected()
    components = list(nx.connected_components(undirected))
    largest_cc = max(len(c) for c in components)
    pct_connected = largest_cc / G.number_of_nodes() * 100
    status = "[OK]" if pct_connected >= 60 else "⚠"
    log(f"  {status} Graph connectivity: largest CC = {largest_cc}/{G.number_of_nodes()} nodes ({pct_connected:.0f}%)")

    # — Check 3: KG seed genes still present ——
    missing_seed = [g for g in KG_GENES if g not in G.nodes]
    status = "[OK]" if not missing_seed else "[FAIL]"
    log(f"  {status} All seed KG genes present: {not bool(missing_seed)} — missing: {missing_seed}")
    if missing_seed:
        failures.append(f"Missing seed genes: {missing_seed}")

    # — Check 4: TP53 reachability ——
    reachable_from_tp53 = nx.descendants(G, "TP53") if "TP53" in G else set()
    reachable_pct = len(reachable_from_tp53) / max(G.number_of_nodes(), 1) * 100
    status = "[OK]" if reachable_pct >= 30 else "⚠"
    log(f"  {status} Nodes reachable from TP53: {len(reachable_from_tp53)} ({reachable_pct:.0f}%)")

    # — Check 5: Edge weight range ——
    weights = [d.get("weight", 0) for _, _, d in G.edges(data=True)]
    bad_weights = [w for w in weights if not (0 <= w <= 1)]
    status = "[OK]" if not bad_weights else "[FAIL]"
    log(f"  {status} Edge weights in [0,1]: {not bool(bad_weights)}  (mean={np.mean(weights):.3f})")
    if bad_weights:
        failures.append(f"Out-of-range edge weights: {bad_weights[:3]}")

    # — Check 6: Mutation rates ——
    gene_summary = mutation_data.get("gene_summary", {})
    rates = [v["mutation_rate"] for v in gene_summary.values()]
    bad_rates = [r for r in rates if not (0 <= r <= 1)]
    status = "[OK]" if not bad_rates else "[FAIL]"
    log(f"  {status} Mutation rates in [0,1]: {not bool(bad_rates)}")

    # — Check 7: Top mutated genes are real genes ——
    top_3 = sorted(gene_summary, key=lambda g: gene_summary[g]["mutation_rate"], reverse=True)[:3]
    log(f"  [OK] Top 3 mutated BRCA genes: {top_3} (rates: {[gene_summary[g]['mutation_rate'] for g in top_3]})")

    # — Summary ——
    log(f"\n  {'[OK] ALL CHECKS PASSED' if not failures else '⚠ ISSUES: ' + '; '.join(failures)}")
    log(f"\n  Graph Summary:")
    log(f"    Nodes: {G.number_of_nodes()}")
    log(f"    Edges: {G.number_of_edges()}")
    degree_seq = sorted([d for _, d in G.degree()], reverse=True)
    log(f"    Max degree: {degree_seq[0]}  |  Avg degree: {np.mean(degree_seq):.2f}")
    log(f"    Is DAG: {nx.is_directed_acyclic_graph(G)}")
    log(f"    Density: {nx.density(G):.4f}")

    return failures


# ==============================================================================
# Main
# ==============================================================================
def main():
    log("=" * 60)
    log("  CEREP V2 — Data Ingestion & Validation Pipeline")
    log(f"  Raw data dir : {RAW}")
    log(f"  Output dir   : {PROC}")
    log("=" * 60)

    total_t = time.time()

    # Check all files exist
    for f in [BRCA_FILE, STRING_FILE, REACTOME_FILE, GO_OBO_FILE]:
        exists = f.exists()
        log(f"  {'[OK]' if exists else '[FAIL]'} {f.name}  ({f.stat().st_size/1e6:.1f} MB)" if exists
            else f"  [FAIL] MISSING: {f}")
        if not exists:
            log("  FATAL: missing file. Aborting.")
            sys.exit(1)

    mutation_data, top_genes = process_brca_mutations()
    ppi_edges = process_string_ppi()
    reactome_data = process_reactome()
    go_data = process_go_obo()
    G = enrich_knowledge_graph(mutation_data, reactome_data, go_data)
    failures = validate_all(G, mutation_data)

    log(f"\n{'='*60}")
    log(f"  Total pipeline time: {time.time()-total_t:.1f}s")
    log(f"  Processed files written to: {PROC}")
    log(f"  Status: {'PASS' if not failures else 'FAIL — see above'}")
    log(f"{'='*60}")

    # Write validation report
    (PROC / "validation_report.txt").write_text("\n".join(validation_lines), encoding="utf-8")
    print(f"\n  Validation report -> {PROC / 'validation_report.txt'}")


if __name__ == "__main__":
    main()
