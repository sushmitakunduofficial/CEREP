"""
CEREP FastAPI Application Entry Point
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.middleware import TimingMiddleware

from backend.core.config import get_settings
from backend.core.logging import get_logger
from backend.models.database import init_db
from backend.graph.graph_builder import CERAPGraphBuilder
from backend.graph.graph_store import create_graph_store
from backend.api import auth, workspace, analysis, reasoning, evaluation

settings = get_settings()
logger = get_logger("main")

# ── Singleton KG builder shared across requests ────────────────────────────────
_kg_builder: CERAPGraphBuilder | None = None


def get_kg_builder() -> CERAPGraphBuilder:
    global _kg_builder
    if _kg_builder is None:
        raise RuntimeError("KG not initialised — lifespan not complete")
    return _kg_builder


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("CEREP startup — initialising database and knowledge graph")
    # 1. Init SQLite DB
    init_db()
    # 2. Build KG singleton
    global _kg_builder

    # Create the appropriate graph store
    store = create_graph_store(
        mode=settings.kg_mode,
        **({"uri": settings.neo4j_uri, "user": settings.neo4j_user,
            "password": settings.neo4j_password} if settings.kg_mode == "neo4j" else {})
    )
    _kg_builder = CERAPGraphBuilder(store=store)

    # Build graph using configured mode
    if settings.kg_build_mode == "adapters":
        _kg_builder.build_from_adapters(include_seed=False)
    else:
        _kg_builder.build_seed_graph()

    # Expose via app state so dependencies can access it
    app.state.kg_builder = _kg_builder
    stats = _kg_builder.get_statistics()
    logger.info(
        "Startup complete",
        extra={"extra": {
            "kg_nodes": stats.get("total_nodes", 0),
            "kg_edges": stats.get("total_edges", 0),
            "kg_mode": settings.kg_mode,
            "kg_build_mode": settings.kg_build_mode,
        }}
    )
    yield
    logger.info("CEREP shutting down")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CEREP API",
    description=(
        "Neuro-symbolic AI system for precision oncology — "
        "knowledge-graph-constrained mechanistic explanations"
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# CORS — allow local frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request timing + structured logging
app.add_middleware(TimingMiddleware)

# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(auth.router,       prefix="/auth",       tags=["auth"])
app.include_router(workspace.router,  prefix="/workspaces", tags=["workspaces"])
app.include_router(analysis.router,   prefix="/analysis",   tags=["analysis"])
app.include_router(reasoning.router,  prefix="/reasoning",  tags=["reasoning"])
app.include_router(evaluation.router, prefix="/evaluation", tags=["evaluation"])


@app.get("/health", tags=["health"])
async def health() -> dict:
    kg = app.state.kg_builder
    stats = kg.get_statistics()
    return {
        "status": "ok",
        "app": settings.app_name,
        "env": settings.app_env,
        "kg_mode": settings.kg_mode,
        "kg_build_mode": settings.kg_build_mode,
        "kg_nodes": stats.get("total_nodes", 0),
        "kg_edges": stats.get("total_edges", 0),
        "nodes_by_category": stats.get("nodes_by_category", {}),
    }
