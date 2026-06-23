"""
Graph Query Engine — multi-hop path traversal, subgraph extraction, path scoring.

Updated for Biolink Model compliance:
    - Paths include Biolink predicates, qualifiers, and provenance
    - Evidence scoring accounts for provenance chain quality
    - Cytoscape export includes full Biolink metadata
"""
from typing import List, Dict, Any, Optional, Tuple, Set
import networkx as nx

from backend.graph.schema import BiolinkCategory, BiolinkPredicate, EvidenceLevel
from backend.graph.graph_store import GraphStore, NetworkXStore
from backend.core.logging import get_logger

logger = get_logger("graph.query")


class GraphPath:
    """Represents a single extracted path through the knowledge graph.

    Enhanced with Biolink predicates, edge qualifiers, and provenance metadata.
    """

    def __init__(self, nodes: List[str], edges: List[Dict],
                 score: float = 1.0, provenance_chain: Optional[List[Dict]] = None):
        self.nodes = nodes
        self.edges = edges
        self.score = score
        self.provenance_chain = provenance_chain or []

    def to_dict(self) -> dict:
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "score": round(self.score, 4),
            "length": len(self.nodes) - 1,
            "readable": self._to_readable(),
            "provenance_chain": self.provenance_chain,
        }

    def _to_readable(self) -> str:
        """Build a human-readable path string using Biolink predicates."""
        parts = [self.nodes[0]]
        for i, edge in enumerate(self.edges):
            predicate = edge.get("predicate", edge.get("edge_type", "?"))
            # Use short form for readability
            short_pred = predicate.replace("biolink:", "")
            parts.append(f"--[{short_pred}]-->")
            parts.append(self.nodes[i + 1])
        return " ".join(parts)

    def get_pmids(self) -> List[str]:
        """Extract all unique PMIDs from the provenance chain."""
        pmids: Set[str] = set()
        for prov in self.provenance_chain:
            pmids.update(prov.get("pmids", []))
        for edge in self.edges:
            # Legacy flat field
            if edge.get("pmid"):
                pmids.add(edge["pmid"])
            # Biolink provenance
            prov = edge.get("provenance", {})
            if isinstance(prov, dict):
                pmids.update(prov.get("pmids", []))
        return sorted(pmids)

    def get_source_databases(self) -> List[str]:
        """Extract unique source databases from the provenance chain."""
        sources: Set[str] = set()
        for prov in self.provenance_chain:
            if prov.get("source_database"):
                sources.add(prov["source_database"])
        for edge in self.edges:
            prov = edge.get("provenance", {})
            if isinstance(prov, dict) and prov.get("source_database"):
                sources.add(prov["source_database"])
            if edge.get("source_db"):
                sources.add(edge["source_db"])
        return sorted(sources)


class GraphQueryEngine:
    """Executes structured queries against the CEREP Knowledge Graph.

    Works with both GraphStore and raw NetworkX DiGraph for backward compat.
    """

    def __init__(self, graph: Any) -> None:
        """Accept either a GraphStore or a raw nx.DiGraph."""
        if isinstance(graph, GraphStore):
            self._store = graph
            self.graph = graph.to_networkx()
        elif isinstance(graph, nx.DiGraph):
            self._store = NetworkXStore(graph=graph)
            self.graph = graph
        else:
            raise TypeError(f"Expected GraphStore or nx.DiGraph, got {type(graph)}")

    # ── Subgraph extraction ───────────────────────────────────────────────────

    def get_subgraph(self, gene_ids: List[str], radius: int = 2) -> nx.DiGraph:
        """Extract ego-network around given genes up to `radius` hops."""
        seed_nodes = [g for g in gene_ids if g in self.graph]
        if not seed_nodes:
            return nx.DiGraph()
        neighbors: set = set()
        for node in seed_nodes:
            nb = nx.ego_graph(self.graph, node, radius=radius,
                              undirected=True).nodes
            neighbors.update(nb)
        return self.graph.subgraph(neighbors).copy()

    def get_neighbors(self, node: str, depth: int = 1) -> Dict[str, Any]:
        """Return immediate successors and predecessors of a node."""
        if node not in self.graph:
            return {}
        successors = [
            {"node": n, **self.graph.edges[node, n]}
            for n in self.graph.successors(node)
        ]
        predecessors = [
            {"node": n, **self.graph.edges[n, node]}
            for n in self.graph.predecessors(node)
        ]
        return {"successors": successors, "predecessors": predecessors}

    # ── Path traversal ────────────────────────────────────────────────────────

    def extract_paths(
        self,
        source: str,
        target: Optional[str] = None,
        max_hops: int = 4,
        top_k: int = 10,
    ) -> List[GraphPath]:
        """
        Extract all simple paths from source to target (or any reachable node).
        Returns top_k paths ranked by cumulative edge weight.
        Includes Biolink provenance in each path.
        """
        if source not in self.graph:
            logger.warning(f"Source node '{source}' not in graph")
            return []

        paths: List[GraphPath] = []

        if target and target in self.graph:
            try:
                raw_paths = list(
                    nx.all_simple_paths(self.graph, source, target,
                                        cutoff=max_hops)
                )
            except nx.NetworkXNoPath:
                raw_paths = []
        else:
            raw_paths = []
            for tgt in self.graph.nodes:
                if tgt == source:
                    continue
                try:
                    for p in nx.all_simple_paths(self.graph, source, tgt,
                                                  cutoff=max_hops):
                        raw_paths.append(p)
                        if len(raw_paths) >= top_k * 5:
                            break
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
                if len(raw_paths) >= top_k * 5:
                    break

        for node_seq in raw_paths:
            edge_data = []
            provenance_chain = []
            score = 1.0
            for i in range(len(node_seq) - 1):
                u, v = node_seq[i], node_seq[i + 1]
                edata = dict(self.graph.edges[u, v])
                edge_data.append(edata)
                score *= edata.get("weight", 1.0)
                # Extract provenance for chain
                prov = edata.get("provenance", {})
                if isinstance(prov, dict):
                    provenance_chain.append({
                        "step": i,
                        "source": u,
                        "target": v,
                        "predicate": edata.get("predicate", edata.get("edge_type", "unknown")),
                        "source_database": prov.get("source_database", edata.get("source_db", "")),
                        "pmids": prov.get("pmids", [edata["pmid"]] if edata.get("pmid") else []),
                        "evidence_level": prov.get("evidence_level", edata.get("evidence", "")),
                    })
                else:
                    provenance_chain.append({
                        "step": i,
                        "source": u,
                        "target": v,
                        "predicate": edata.get("predicate", edata.get("edge_type", "unknown")),
                        "source_database": edata.get("source_db", ""),
                        "pmids": [edata["pmid"]] if edata.get("pmid") else [],
                        "evidence_level": edata.get("evidence", ""),
                    })
            paths.append(GraphPath(node_seq, edge_data, score, provenance_chain))

        paths.sort(key=lambda p: p.score, reverse=True)
        return paths[:top_k]

    # ── Node info ─────────────────────────────────────────────────────────────

    def get_node_info(self, node_id: str) -> Optional[Dict]:
        if node_id not in self.graph:
            return None
        return dict(self.graph.nodes[node_id])

    def get_all_entities(self) -> List[str]:
        return list(self.graph.nodes())

    def get_entity_by_type(self, node_type: str) -> List[Dict]:
        """Get entities by category (supports both Biolink and legacy type strings)."""
        return [
            {"id": n, **self.graph.nodes[n]}
            for n in self.graph.nodes
            if self.graph.nodes[n].get("category") == node_type
            or self.graph.nodes[n].get("node_type") == node_type
        ]

    def get_entity_by_category(self, category: BiolinkCategory) -> List[Dict]:
        """Get entities by Biolink category enum."""
        return self.get_entity_by_type(category.value)

    # ── Subgraph serialization for frontend ───────────────────────────────────

    def subgraph_to_cytoscape(self, subgraph: nx.DiGraph) -> dict:
        """Export subgraph as Cytoscape.js-compatible JSON with Biolink metadata."""
        nodes = []
        for n in subgraph.nodes:
            node_data = dict(subgraph.nodes[n])
            nodes.append({"data": {"id": n, **node_data}})

        edges = []
        for u, v in subgraph.edges:
            edge_data = dict(subgraph.edges[u, v])
            edges.append({"data": {"source": u, "target": v, **edge_data}})

        return {"nodes": nodes, "edges": edges}

    # ── Evidence scoring ─────────────────────────────────────────────────────

    def score_path_evidence(self, path: GraphPath) -> float:
        """Aggregate evidence quality score for a path (0–1).

        Uses Biolink evidence levels when available, falls back to legacy.
        """
        if not path.edges:
            return 0.0

        evidence_weights = {
            # Biolink EvidenceLevel values
            "experimental": 1.0,
            "clinical": 0.95,
            "curated": 0.9,
            "literature": 0.8,
            "computational": 0.5,
            "inferred": 0.6,
        }
        scores = []
        for edge in path.edges:
            # Try Biolink provenance first
            prov = edge.get("provenance", {})
            if isinstance(prov, dict):
                ev = prov.get("evidence_level", "")
            else:
                ev = ""
            # Fall back to legacy
            if not ev:
                ev = edge.get("evidence", "inferred")
            scores.append(evidence_weights.get(ev, 0.6))

        # Bonus for paths with PMIDs
        pmid_count = len(path.get_pmids())
        pmid_bonus = min(pmid_count * 0.02, 0.1)  # max 0.1 bonus

        base_score = sum(scores) / len(scores)
        return round(min(base_score + pmid_bonus, 1.0), 4)

    def get_path_provenance_summary(self, path: GraphPath) -> Dict[str, Any]:
        """Build a structured provenance summary for a path."""
        return {
            "pmids": path.get_pmids(),
            "source_databases": path.get_source_databases(),
            "provenance_chain": path.provenance_chain,
            "evidence_score": self.score_path_evidence(path),
            "total_hops": len(path.edges),
        }
