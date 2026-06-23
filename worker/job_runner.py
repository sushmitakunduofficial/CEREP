"""
Job Runner — in-process asyncio job runner.
Accepts a Job ORM object, dispatches to the right task, and updates DB status.
"""
import json
import traceback
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy.orm import Session

from backend.models.job import Job, JobStatus
from backend.graph.graph_builder import CERAPGraphBuilder
from worker.tasks import analysis_task, reasoning_task, evaluation_task
from backend.core.logging import get_logger

logger = get_logger("worker.job_runner")


class JobRunner:
    """In-process asyncio-based job dispatcher."""

    def __init__(self, db: Session, kg_builder: CERAPGraphBuilder) -> None:
        self.db = db
        self.kg = kg_builder

    async def run_analysis_job(
        self,
        job: Job,
        genes: List[str],
        max_hops: int = 4,
        top_k: int = 10,
    ) -> None:
        await self._execute(job, analysis_task(self.kg, genes, max_hops, top_k))

    async def run_reasoning_job(
        self,
        job: Job,
        genes: List[str],
        max_hops: int = 4,
        top_k: int = 10,
        temperature: Optional[float] = None,
    ) -> None:
        await self._execute(
            job, reasoning_task(self.kg, genes, max_hops, top_k, temperature)
        )

    async def run_evaluation_job(
        self,
        job: Job,
        genes: List[str],
        max_hops: int = 4,
        top_k: int = 10,
    ) -> None:
        await self._execute(job, evaluation_task(self.kg, genes, max_hops, top_k))

    async def _execute(self, job: Job, coro) -> None:
        """
        Generic coroutine executor:
        - Sets job status to RUNNING
        - Awaits the task coroutine
        - Persists result or error to DB
        """
        self._set_status(job, JobStatus.RUNNING)
        try:
            result = await coro
            job.result_data = json.dumps(result, default=str)
            job.completed_at = datetime.now(timezone.utc)
            if "confidence_score" in result:
                job.confidence_score = result.get("confidence_score")
            if "hallucination_check" in result:
                hc = result["hallucination_check"]
                if isinstance(hc, dict):
                    job.hallucination_rate = hc.get("hallucination_rate")
            self._set_status(job, JobStatus.COMPLETED)
            logger.info(
                "Job completed",
                extra={"extra": {"job_id": job.id, "type": job.job_type.value}}
            )
        except Exception as exc:
            tb = traceback.format_exc()
            job.error_message = f"{type(exc).__name__}: {exc}\n{tb}"
            self._set_status(job, JobStatus.FAILED)
            logger.error(
                "Job failed",
                extra={"extra": {"job_id": job.id, "error": str(exc)}}
            )

    def _set_status(self, job: Job, status: JobStatus) -> None:
        job.status = status
        self.db.commit()
