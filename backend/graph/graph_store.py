"""
Graph Store — Abstract storage layer for the CEREP Knowledge Graph.

Implements the Strategy pattern:
    GraphStore (ABC)
    ├── NetworkXStore  ← development (in-memory, instant startup)
    └── Neo4jStore     ← production (persistent, Cypher queries, scalable)

Usage:
    # Development
    store = NetworkXStore()

    # Production
    store = Neo4jStore(uri="bolt://localhost:7687", user="neo4j", password="...")
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set, Tuple
import json
from pathlib import Path

import networkx as nx

from backend.graph.schema import (
    NodeSchema, EdgeSchema, BiolinkCategory, BiolinkPredicate,
    Provenance, EdgeQualifiers,
)
from backend.core.logging import get_logger

logger = get_logger("graph.store")


# ══════════════════════════════════════════════════════════════════════════════
# Abstract Graph Store
# ══════════════════════════════════════════════════════════════════════════════

class GraphStore(ABC):
    """Abstract interface for KG persistence."""

    @abstractmethod
    def add_node(self, node: NodeSchema) -> None:
        """Insert a node into the graph."""

    @abstractmethod
    def add_edge(self, edge: EdgeSchema) -> None:
        """Insert a directed edge into the graph."""

    @abstractmethod
    def has_node(self, node_id: str) -> bool:
        """Check if a node exists."""

    @abstractmethod
    def has_edge(self, source: str, target: str) -> bool:
        """Check if an edge exists between source and target."""

    @abstractmethod
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve node attributes."""

    @abstractmethod
    def get_edge(self, source: str, target: str) -> Optional[Dict[str, Any]]:
        """Retrieve edge attributes."""

    @abstractmethod
    def get_neighbors(self, node_id: str, direction: str = "both") -> List[str]:
        """Get neighbor node IDs. direction: 'out', 'in', or 'both'."""

    @abstractmethod
    def get_all_nodes(self) -> List[str]:
        """Return all node IDs."""

    @abstractmethod
    def get_all_edges(self) -> List[Tuple[str, str]]:
        """Return all (source, target) pairs."""

    @abstractmethod
    def node_count(self) -> int:
        """Total number of nodes."""

    @abstractmethod
    def edge_count(self) -> int:
        """Total number of edges."""

    @abstractmethod
    def get_subgraph(self, seed_nodes: List[str], radius: int = 2) -> "GraphStore":
        """Extract an ego-network subgraph around seed nodes."""

    @abstractmethod
    def get_nodes_by_category(self, category: BiolinkCategory) -> List[Dict[str, Any]]:
        """Return all nodes of a given Biolink category."""

    @abstractmethod
    def find_paths(
        self, source: str, target: Optional[str] = None,
        max_hops: int = 4, top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Find and rank paths from source to target (or all reachable nodes)."""

    @abstractmethod
    def to_networkx(self) -> nx.DiGraph:
        """Export the store contents as a NetworkX DiGraph."""

    @abstractmethod
    def to_cytoscape_json(self) -> Dict[str, Any]:
        """Export as Cytoscape.js-compatible JSON."""

    @abstractmethod
    def save_cache(self, path: str) -> None:
        """Persist graph to disk cache."""

    @abstractmethod
    def load_cache(self, path: str) -> bool:
        """Load graph from disk cache. Returns True if loaded successfully."""

    def get_statistics(self) -> Dict[str, Any]:
        """Return graph statistics broken down by Biolink category/predicate."""
        return {
            "total_nodes": self.node_count(),
            "total_edges": self.edge_count(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# NetworkX Store — Development / Default
# ══════════════════════════════════════════════════════════════════════════════

class NetworkXStore(GraphStore):
    """In-memory graph store backed by NetworkX DiGraph.

    Best for:
        - Development and testing
        - Graphs up to ~500K nodes
        - Instant startup, zero dependencies
    """

    def __init__(self, graph: Optional[nx.DiGraph] = None) -> None:
        self.graph: nx.DiGraph = graph or nx.DiGraph()
        self._entity_index: Dict[str, str] = {}  # alias → canonical ID

    def add_node(self, node: NodeSchema) -> None:
        self.graph.add_node(node.id, **node.to_dict())
        # Register aliases for resolution
        self._entity_index[node.id.upper()] = node.id
        for alias in node.aliases:
            self._entity_index[alias.upper()] = node.id

    def add_edge(self, edge: EdgeSchema) -> None:
        if edge.source not in self.graph or edge.target not in self.graph:
            logger.warning(
                f"Skipping edge {edge.source}→{edge.target}: node(s) not in graph"
            )
            return
        self.graph.add_edge(edge.source, edge.target, **edge.to_dict())

    def has_node(self, node_id: str) -> bool:
        return node_id in self.graph

    def has_edge(self, source: str, target: str) -> bool:
        return self.graph.has_edge(source, target)

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        if node_id not in self.graph:
            return None
        return dict(self.graph.nodes[node_id])

    def get_edge(self, source: str, target: str) -> Optional[Dict[str, Any]]:
        if not self.graph.has_edge(source, target):
            return None
        return dict(self.graph.edges[source, target])

    def get_neighbors(self, node_id: str, direction: str = "both") -> List[str]:
        if node_id not in self.graph:
            return []
        result: Set[str] = set()
        if direction in ("out", "both"):
            result.update(self.graph.successors(node_id))
        if direction in ("in", "both"):
            result.update(self.graph.predecessors(node_id))
        return list(result)

    def get_all_nodes(self) -> List[str]:
        return list(self.graph.nodes())

    def get_all_edges(self) -> List[Tuple[str, str]]:
        return list(self.graph.edges())

    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    def edge_count(self) -> int:
        return self.graph.number_of_edges()

    def get_subgraph(self, seed_nodes: List[str], radius: int = 2) -> "NetworkXStore":
        valid_seeds = [n for n in seed_nodes if n in self.graph]
        if not valid_seeds:
            return NetworkXStore()
        neighbors: Set[str] = set()
        for node in valid_seeds:
            ego = nx.ego_graph(self.graph, node, radius=radius, undirected=True)
            neighbors.update(ego.nodes)
        sub = self.graph.subgraph(neighbors).copy()
        store = NetworkXStore(graph=sub)
        # Rebuild entity index for the subgraph
        for n in sub.nodes:
            store._entity_index[n.upper()] = n
            for alias in sub.nodes[n].get("aliases", []):
                store._entity_index[alias.upper()] = n
        return store

    def get_nodes_by_category(self, category: BiolinkCategory) -> List[Dict[str, Any]]:
        cat_value = category.value
        return [
            {"id": n, **self.graph.nodes[n]}
            for n in self.graph.nodes
            if self.graph.nodes[n].get("category") == cat_value
            or self.graph.nodes[n].get("node_type") == cat_value
        ]

    def find_paths(
        self, source: str, target: Optional[str] = None,
        max_hops: int = 4, top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        if source not in self.graph:
            return []

        raw_paths: List[List[str]] = []

        if target and target in self.graph:
            try:
                raw_paths = list(
                    nx.all_simple_paths(self.graph, source, target, cutoff=max_hops)
                )
            except nx.NetworkXNoPath:
                raw_paths = []
        else:
            for tgt in self.graph.nodes:
                if tgt == source:
                    continue
                try:
                    for p in nx.all_simple_paths(self.graph, source, tgt, cutoff=max_hops):
                        raw_paths.append(p)
                        if len(raw_paths) >= top_k * 5:
                            break
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
                if len(raw_paths) >= top_k * 5:
                    break

        # Score and rank paths
        scored_paths = []
        for node_seq in raw_paths:
            edge_data = []
            score = 1.0
            for i in range(len(node_seq) - 1):
                u, v = node_seq[i], node_seq[i + 1]
                edata = dict(self.graph.edges[u, v])
                edge_data.append(edata)
                score *= edata.get("weight", 1.0)

            # Build readable path string
            parts = [node_seq[0]]
            for i, ed in enumerate(edge_data):
                pred = ed.get("predicate", ed.get("edge_type", "?"))
                parts.append(f"--[{pred}]-->")
                parts.append(node_seq[i + 1])

            scored_paths.append({
                "nodes": node_seq,
                "edges": edge_data,
                "score": round(score, 4),
                "length": len(node_seq) - 1,
                "readable": " ".join(parts),
            })

        scored_paths.sort(key=lambda p: p["score"], reverse=True)
        return scored_paths[:top_k]

    def to_networkx(self) -> nx.DiGraph:
        return self.graph

    def to_cytoscape_json(self) -> Dict[str, Any]:
        nodes = [
            {"data": {"id": n, **self.graph.nodes[n]}}
            for n in self.graph.nodes
        ]
        edges = [
            {"data": {"source": u, "target": v, **self.graph.edges[u, v]}}
            for u, v in self.graph.edges
        ]
        return {"nodes": nodes, "edges": edges}

    def save_cache(self, path: str) -> None:
        cache_path = Path(path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "nodes": [
                {"id": n, **self.graph.nodes[n]}
                for n in self.graph.nodes
            ],
            "edges": [
                {**self.graph.edges[u, v], "source": u, "target": v}
                for u, v in self.graph.edges
            ],
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(
            "Graph cache saved",
            extra={"extra": {"path": str(cache_path),
                             "nodes": self.node_count(),
                             "edges": self.edge_count()}}
        )

    def load_cache(self, path: str) -> bool:
        cache_path = Path(path)
        if not cache_path.exists():
            return False
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.graph = nx.DiGraph()
            self._entity_index = {}
            for node_data in data["nodes"]:
                node_id = node_data.pop("id")
                self.graph.add_node(node_id, **node_data)
                self._entity_index[node_id.upper()] = node_id
                for alias in node_data.get("aliases", []):
                    self._entity_index[alias.upper()] = node_id
            for edge_data in data["edges"]:
                src = edge_data.pop("source")
                tgt = edge_data.pop("target")
                self.graph.add_edge(src, tgt, **edge_data)
            logger.info(
                "Graph loaded from cache",
                extra={"extra": {"path": str(cache_path),
                                 "nodes": self.node_count(),
                                 "edges": self.edge_count()}}
            )
            return True
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning(f"Cache load failed ({exc}), will rebuild.")
            return False

    # ── Entity resolution ────────────────────────────────────────────────────

    def resolve_alias(self, name: str) -> Optional[str]:
        """Return canonical ID for a name/alias, or None if unknown."""
        return self._entity_index.get(name.upper())

    def get_entity_index(self) -> Dict[str, str]:
        """Return the full alias → canonical ID index."""
        return self._entity_index

    def get_statistics(self) -> Dict[str, Any]:
        """Detailed statistics broken down by Biolink category and predicate."""
        cat_counts: Dict[str, int] = {}
        for n in self.graph.nodes:
            cat = self.graph.nodes[n].get("category",
                  self.graph.nodes[n].get("node_type", "unknown"))
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        pred_counts: Dict[str, int] = {}
        for u, v in self.graph.edges:
            pred = self.graph.edges[u, v].get("predicate",
                   self.graph.edges[u, v].get("edge_type", "unknown"))
            pred_counts[pred] = pred_counts.get(pred, 0) + 1

        return {
            "total_nodes": self.node_count(),
            "total_edges": self.edge_count(),
            "nodes_by_category": cat_counts,
            "edges_by_predicate": pred_counts,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Neo4j Store — Production
# ══════════════════════════════════════════════════════════════════════════════

class Neo4jStore(GraphStore):
    """Persistent graph store backed by Neo4j.

    Best for:
        - Production deployments
        - Graphs with >100K nodes
        - Cypher-based multi-hop queries
        - Future BioCypher compatibility
    """

    def __init__(self, uri: str = "bolt://localhost:7687",
                 user: str = "neo4j", password: str = "neo4j") -> None:
        self._uri = uri
        self._user = user
        self._password = password
        self._driver = None
        self._connect()

    def _connect(self) -> None:
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                self._uri, auth=(self._user, self._password)
            )
            self._driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {self._uri}")
        except ImportError:
            logger.error("neo4j package not installed. Install with: pip install neo4j")
            raise
        except Exception as exc:
            logger.error(f"Failed to connect to Neo4j: {exc}")
            raise

    def _run(self, query: str, **params) -> List[Dict]:
        """Execute a Cypher query and return results as dicts."""
        with self._driver.session() as session:
            result = session.run(query, **params)
            return [dict(record) for record in result]

    def add_node(self, node: NodeSchema) -> None:
        query = """
        MERGE (n {id: $id})
        SET n += $props
        SET n:BiolinkEntity
        """
        self._run(query, id=node.id, props=node.to_dict())

    def add_edge(self, edge: EdgeSchema) -> None:
        query = """
        MATCH (a {id: $src}), (b {id: $tgt})
        MERGE (a)-[r:BIOLINK_EDGE {predicate: $pred}]->(b)
        SET r += $props
        """
        self._run(
            query,
            src=edge.source,
            tgt=edge.target,
            pred=edge.predicate.value,
            props=edge.to_dict(),
        )

    def has_node(self, node_id: str) -> bool:
        result = self._run("MATCH (n {id: $id}) RETURN count(n) as cnt", id=node_id)
        return result[0]["cnt"] > 0 if result else False

    def has_edge(self, source: str, target: str) -> bool:
        result = self._run(
            "MATCH ({id: $src})-[r]->({id: $tgt}) RETURN count(r) as cnt",
            src=source, tgt=target,
        )
        return result[0]["cnt"] > 0 if result else False

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        result = self._run("MATCH (n {id: $id}) RETURN properties(n) as props", id=node_id)
        return result[0]["props"] if result else None

    def get_edge(self, source: str, target: str) -> Optional[Dict[str, Any]]:
        result = self._run(
            "MATCH ({id: $src})-[r]->({id: $tgt}) RETURN properties(r) as props",
            src=source, tgt=target,
        )
        return result[0]["props"] if result else None

    def get_neighbors(self, node_id: str, direction: str = "both") -> List[str]:
        if direction == "out":
            q = "MATCH ({id: $id})-[]->(n) RETURN n.id as nid"
        elif direction == "in":
            q = "MATCH ({id: $id})<-[]-(n) RETURN n.id as nid"
        else:
            q = "MATCH ({id: $id})-[]-(n) RETURN DISTINCT n.id as nid"
        return [r["nid"] for r in self._run(q, id=node_id)]

    def get_all_nodes(self) -> List[str]:
        return [r["nid"] for r in self._run("MATCH (n) RETURN n.id as nid")]

    def get_all_edges(self) -> List[Tuple[str, str]]:
        results = self._run(
            "MATCH (a)-[r]->(b) RETURN a.id as src, b.id as tgt"
        )
        return [(r["src"], r["tgt"]) for r in results]

    def node_count(self) -> int:
        result = self._run("MATCH (n) RETURN count(n) as cnt")
        return result[0]["cnt"] if result else 0

    def edge_count(self) -> int:
        result = self._run("MATCH ()-[r]->() RETURN count(r) as cnt")
        return result[0]["cnt"] if result else 0

    def get_subgraph(self, seed_nodes: List[str], radius: int = 2) -> "NetworkXStore":
        """Extract ego-network via Cypher variable-length path, return as NetworkXStore."""
        query = f"""
        MATCH (seed {{id: $seed_id}})
        CALL apoc.path.subgraphNodes(seed, {{maxLevel: $radius}}) YIELD node
        RETURN properties(node) as props
        """
        all_node_ids: Set[str] = set()
        nx_graph = nx.DiGraph()

        for seed_id in seed_nodes:
            try:
                nodes = self._run(query, seed_id=seed_id, radius=radius)
                for n in nodes:
                    props = n["props"]
                    nid = props.get("id", "")
                    nx_graph.add_node(nid, **props)
                    all_node_ids.add(nid)
            except Exception as exc:
                logger.warning(f"Subgraph extraction failed for {seed_id}: {exc}")
                # Fallback: simple 1-hop
                neighbors = self.get_neighbors(seed_id)
                for nb in neighbors:
                    node_data = self.get_node(nb)
                    if node_data:
                        nx_graph.add_node(nb, **node_data)
                        all_node_ids.add(nb)

        # Get edges between the collected nodes
        if all_node_ids:
            id_list = list(all_node_ids)
            edge_query = """
            MATCH (a)-[r]->(b)
            WHERE a.id IN $ids AND b.id IN $ids
            RETURN a.id as src, b.id as tgt, properties(r) as props
            """
            edges = self._run(edge_query, ids=id_list)
            for e in edges:
                nx_graph.add_edge(e["src"], e["tgt"], **e["props"])

        return NetworkXStore(graph=nx_graph)

    def get_nodes_by_category(self, category: BiolinkCategory) -> List[Dict[str, Any]]:
        results = self._run(
            "MATCH (n) WHERE n.category = $cat RETURN properties(n) as props",
            cat=category.value,
        )
        return [r["props"] for r in results]

    def find_paths(
        self, source: str, target: Optional[str] = None,
        max_hops: int = 4, top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Find paths via Cypher variable-length pattern matching."""
        if target:
            query = f"""
            MATCH path = (a {{id: $src}})-[*1..{max_hops}]->(b {{id: $tgt}})
            RETURN [n IN nodes(path) | n.id] as node_ids,
                   [r IN relationships(path) | properties(r)] as edge_props
            LIMIT $limit
            """
            results = self._run(query, src=source, tgt=target, limit=top_k * 5)
        else:
            query = f"""
            MATCH path = (a {{id: $src}})-[*1..{max_hops}]->(b)
            WHERE a <> b
            RETURN [n IN nodes(path) | n.id] as node_ids,
                   [r IN relationships(path) | properties(r)] as edge_props
            LIMIT $limit
            """
            results = self._run(query, src=source, limit=top_k * 5)

        paths = []
        for r in results:
            node_ids = r["node_ids"]
            edge_props = r["edge_props"]
            score = 1.0
            for ep in edge_props:
                score *= ep.get("weight", 1.0)

            parts = [node_ids[0]]
            for i, ep in enumerate(edge_props):
                pred = ep.get("predicate", ep.get("edge_type", "?"))
                parts.append(f"--[{pred}]-->")
                parts.append(node_ids[i + 1])

            paths.append({
                "nodes": node_ids,
                "edges": edge_props,
                "score": round(score, 4),
                "length": len(node_ids) - 1,
                "readable": " ".join(parts),
            })

        paths.sort(key=lambda p: p["score"], reverse=True)
        return paths[:top_k]

    def to_networkx(self) -> nx.DiGraph:
        """Export entire Neo4j graph to NetworkX (use cautiously for large graphs)."""
        g = nx.DiGraph()
        for nid in self.get_all_nodes():
            props = self.get_node(nid) or {}
            g.add_node(nid, **props)
        for src, tgt in self.get_all_edges():
            props = self.get_edge(src, tgt) or {}
            g.add_edge(src, tgt, **props)
        return g

    def to_cytoscape_json(self) -> Dict[str, Any]:
        nodes = []
        for nid in self.get_all_nodes():
            props = self.get_node(nid) or {}
            nodes.append({"data": {"id": nid, **props}})
        edges = []
        for src, tgt in self.get_all_edges():
            props = self.get_edge(src, tgt) or {}
            edges.append({"data": {"source": src, "target": tgt, **props}})
        return {"nodes": nodes, "edges": edges}

    def save_cache(self, path: str) -> None:
        """Export Neo4j to JSON cache (for offline development)."""
        nx_graph = self.to_networkx()
        temp_store = NetworkXStore(graph=nx_graph)
        temp_store.save_cache(path)

    def load_cache(self, path: str) -> bool:
        """Load JSON cache into Neo4j (for bootstrapping)."""
        temp_store = NetworkXStore()
        if not temp_store.load_cache(path):
            return False
        # Batch import into Neo4j
        for nid in temp_store.get_all_nodes():
            node_data = temp_store.get_node(nid)
            if node_data:
                node = NodeSchema(
                    id=nid,
                    category=BiolinkCategory(node_data.get("category", "biolink:Gene")),
                    label=node_data.get("label", nid),
                    aliases=node_data.get("aliases", []),
                    description=node_data.get("description"),
                    source=node_data.get("source", "cache"),
                )
                self.add_node(node)
        for src, tgt in temp_store.get_all_edges():
            edge_data = temp_store.get_edge(src, tgt)
            if edge_data:
                edge = EdgeSchema(
                    source=src,
                    target=tgt,
                    predicate=BiolinkPredicate(
                        edge_data.get("predicate", "biolink:related_to")
                    ),
                    weight=edge_data.get("weight", 1.0),
                )
                self.add_edge(edge)
        logger.info(f"Loaded cache into Neo4j: {self.node_count()} nodes, {self.edge_count()} edges")
        return True

    def close(self) -> None:
        if self._driver:
            self._driver.close()

    def __del__(self) -> None:
        self.close()


# ══════════════════════════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════════════════════════

def create_graph_store(mode: str = "networkx", **kwargs) -> GraphStore:
    """Factory function to create the appropriate GraphStore.

    Args:
        mode: "networkx" or "neo4j"
        **kwargs: Passed to the store constructor (e.g., uri, user, password for Neo4j)
    """
    if mode == "neo4j":
        return Neo4jStore(**kwargs)
    return NetworkXStore()
