"""
CEREP Graph Module — Biolink-compliant knowledge graph construction and querying.

Public API:
    CERAPGraphBuilder    — builds KG from adapters or seed data
    GraphQueryEngine     — multi-hop path extraction and subgraph queries
    NetworkXStore        — default graph storage backend
    create_graph_store   — factory for storage backends
"""
from backend.graph.graph_builder import CERAPGraphBuilder
from backend.graph.graph_query import GraphQueryEngine, GraphPath
from backend.graph.graph_store import NetworkXStore, create_graph_store

__all__ = [
    "CERAPGraphBuilder",
    "GraphQueryEngine",
    "GraphPath",
    "NetworkXStore",
    "create_graph_store",
]
