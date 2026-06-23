"""
Path Extractor — orchestrates multi-hop KG path extraction for a gene query.
Normalizes gene names, resolves to KG IDs, and returns ranked paths.
"""
from typing import List, Dict, Any, Optional

from backend.graph.graph_builder import CERAPGraphBuilder
from backend.graph.graph_query import GraphQueryEngine, GraphPath
from backend.pipelines.gene_normalizer import GeneNormalizer
from backend.core.config import get_settings
from backend.core.logging import get_logger

logger = get_logger("reasoning.path_extractor")
settings = get_settings()


class PathExtractor:
    """Extracts and ranks valid KG paths for a set of query genes."""

    def __init__(self, builder: CERAPGraphBuilder) -> None:
        self.builder = builder
        self.query_engine = GraphQueryEngine(builder.graph)
        self.normalizer = GeneNormalizer(builder.get_entity_index())

    def extract(
        self,
        gene_query: str | List[str],
        target_gene: Optional[str] = None,
        max_hops: int = 4,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """
        Full path extraction pipeline:
        1. Parse and normalize gene names
        2. Resolve to KG canonical IDs
        3. Extract multi-hop paths
        4. Score and rank
        5. Return structured result
        """
        # Parse input
        if isinstance(gene_query, str):
            raw_genes = [g.strip() for g in gene_query.replace(",", " ").split() if g.strip()]
        else:
            raw_genes = gene_query

        # Normalize
        norm_report = self.normalizer.normalize_with_report(raw_genes)
        canonical_genes = list(set(norm_report.values()))

        # Filter to KG members
        kg_genes = [g for g in canonical_genes if g in self.builder.graph]
        unknown = [r for r, c in norm_report.items() if c not in self.builder.graph]

        if not kg_genes:
            return {
                "status": "no_match",
                "message": f"None of the input genes found in KG: {raw_genes}",
                "normalization_map": norm_report,
                "paths": [],
            }

        # Extract paths
        all_paths: List[GraphPath] = []
        for gene in kg_genes:
            tgt = self.normalizer.normalize(target_gene) if target_gene else None
            paths = self.query_engine.extract_paths(
                source=gene, target=tgt,
                max_hops=max_hops, top_k=top_k
            )
            all_paths.extend(paths)

        # Deduplicate and re-rank
        seen_readable = set()
        unique_paths = []
        for p in sorted(all_paths, key=lambda x: x.score, reverse=True):
            if p._to_readable() not in seen_readable:
                seen_readable.add(p._to_readable())
                unique_paths.append(p)

        top_paths = unique_paths[:top_k]

        # Subgraph for visualisation
        all_nodes = set()
        for p in top_paths:
            all_nodes.update(p.nodes)
        subgraph = self.builder.graph.subgraph(all_nodes)
        cyto = self.query_engine.subgraph_to_cytoscape(subgraph)

        return {
            "status": "success",
            "query_genes": raw_genes,
            "normalization_map": norm_report,
            "kg_matched_genes": kg_genes,
            "unknown_genes": unknown,
            "paths": [p.to_dict() for p in top_paths],
            "graph": cyto,
            "total_paths_found": len(top_paths),
        }
