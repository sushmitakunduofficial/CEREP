"""
Analysis API — POST /analysis/run   GET /analysis/{job_id}
"""
import json
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.api.deps import get_current_user, get_kg
from backend.models.database import get_db
from backend.models.user import User
from backend.models.job import Job, JobStatus, JobType
from backend.graph.graph_builder import CERAPGraphBuilder
from worker.job_runner import JobRunner
from backend.evaluation.ablation import parse_genes_from_upload
from backend.core.logging import get_logger

router = APIRouter()
logger = get_logger("api.analysis")


# ── Schemas ────────────────────────────────────────────────────────────────────
class AnalysisRequest(BaseModel):
    workspace_id: str
    genes: List[str]
    max_hops: int = 4
    top_k: int = 10


class JobStatusOut(BaseModel):
    job_id: str
    status: str
    job_type: str
    result: Optional[dict] = None
    error: Optional[str] = None


# ── Routes ─────────────────────────────────────────────────────────────────────
@router.post("/run", response_model=JobStatusOut, status_code=202)
async def run_analysis(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    kg: CERAPGraphBuilder = Depends(get_kg),
):
    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type:
        form = await request.form()
        workspace_id = str(form.get("workspace_id"))
        max_hops = int(form.get("max_hops", 4))
        top_k = int(form.get("top_k", 10))
        upload: Optional[UploadFile] = form.get("file")  # type: ignore

        if upload:
            raw = await upload.read()
            genes = parse_genes_from_upload(raw, filename=upload.filename or "")
        else:
            genes_raw = str(form.get("genes", ""))
            genes = [g.strip() for g in genes_raw.split(",") if g.strip()]
    else:
        body = await request.json()
        payload = AnalysisRequest(**body)
        workspace_id = payload.workspace_id
        genes = payload.genes
        max_hops = payload.max_hops
        top_k = payload.top_k

    if not genes:
        raise HTTPException(status_code=400, detail="No genes provided for analysis")

    job = Job(
        id=str(uuid.uuid4()),
        workspace_id=workspace_id,
        job_type=JobType.ANALYSIS,
        status=JobStatus.PENDING,
        gene_query=",".join(genes),
        input_data=json.dumps({
            "workspace_id": workspace_id,
            "genes": genes,
            "max_hops": max_hops,
            "top_k": top_k
        }),
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Run in-process (async)
    runner = JobRunner(db=db, kg_builder=kg)
    await runner.run_analysis_job(job, genes, max_hops, top_k)
    db.refresh(job)

    result = json.loads(job.result_data) if job.result_data else None
    return JobStatusOut(
        job_id=job.id,
        status=job.status.value,
        job_type=job.job_type.value,
        result=result,
        error=job.error_message,
    )


@router.get("/{job_id}", response_model=JobStatusOut)
def get_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    result = json.loads(job.result_data) if job.result_data else None
    return JobStatusOut(
        job_id=job.id,
        status=job.status.value,
        job_type=job.job_type.value,
        result=result,
        error=job.error_message,
    )


@router.get("/kg/stats")
def kg_stats(
    current_user: User = Depends(get_current_user),
    kg: CERAPGraphBuilder = Depends(get_kg),
):
    """Return detailed KG statistics including Biolink category/predicate breakdown."""
    return kg.get_statistics()


@router.get("/kg/search")
def kg_search(
    query: str,
    current_user: User = Depends(get_current_user),
    kg: CERAPGraphBuilder = Depends(get_kg),
):
    """Search KG entities by name or alias. Returns matching nodes."""
    canonical = kg.resolve_alias(query.strip().upper())
    if not canonical:
        # Fuzzy search across entity index
        index = kg.get_entity_index()
        matches = [
            {"alias": alias, "canonical": cid}
            for alias, cid in index.items()
            if query.upper() in alias
        ][:20]
        return {"exact_match": None, "fuzzy_matches": matches}

    node_data = kg.store.get_node(canonical) if canonical else None
    return {
        "exact_match": {"id": canonical, **(node_data or {})},
        "fuzzy_matches": [],
    }
