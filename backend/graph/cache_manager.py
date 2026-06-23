"""
Cache Manager — persists the CEREP Knowledge Graph to/from disk as JSON.
This allows instant graph loading without rebuilding from seed data each startup.
"""
import json
import os
from pathlib import Path
from typing import Optional

import networkx as nx

from backend.core.config import get_settings
from backend.core.logging import get_logger

logger = get_logger("graph.cache")
settings = get_settings()


class GraphCacheManager:
    """Handles serialization and deserialization of the KG to JSON cache."""

    def __init__(self, cache_path: Optional[str] = None) -> None:
        self.cache_path = Path(cache_path or settings.kg_cache_path)

    def exists(self) -> bool:
        return self.cache_path.exists()

    def save(self, graph: nx.DiGraph) -> None:
        """Persist graph to JSON cache file."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "nodes": [
                {"id": n, **graph.nodes[n]}
                for n in graph.nodes
            ],
            "edges": [
                {**graph.edges[u, v], "source": u, "target": v}
                for u, v in graph.edges
            ],
        }
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(
            "Graph cache saved",
            extra={"extra": {"path": str(self.cache_path),
                              "nodes": graph.number_of_nodes(),
                              "edges": graph.number_of_edges()}}
        )

    def load(self) -> Optional[nx.DiGraph]:
        """Load graph from JSON cache file. Returns None if cache is absent."""
        if not self.exists():
            return None
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            graph = nx.DiGraph()
            for node in data["nodes"]:
                node_id = node.pop("id")
                graph.add_node(node_id, **node)
            for edge in data["edges"]:
                src = edge.pop("source")
                tgt = edge.pop("target")
                graph.add_edge(src, tgt, **edge)
            logger.info(
                "Graph loaded from cache",
                extra={"extra": {"path": str(self.cache_path),
                                  "nodes": graph.number_of_nodes(),
                                  "edges": graph.number_of_edges()}}
            )
            return graph
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning(f"Cache load failed ({exc}), will rebuild.")
            return None

    def invalidate(self) -> None:
        """Remove the cache file to force rebuild on next startup."""
        if self.cache_path.exists():
            os.remove(self.cache_path)
            logger.info("Graph cache invalidated")

    def get_cache_info(self) -> dict:
        if not self.exists():
            return {"exists": False}
        stat = self.cache_path.stat()
        return {
            "exists": True,
            "path": str(self.cache_path),
            "size_kb": round(stat.st_size / 1024, 2),
        }
