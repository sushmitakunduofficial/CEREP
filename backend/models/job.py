from sqlalchemy import Column, String, Text, DateTime, Float, ForeignKey, func, Enum
from sqlalchemy.orm import relationship
import enum
from backend.models.base import Base


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobType(str, enum.Enum):
    ANALYSIS = "analysis"
    REASONING = "reasoning"
    EVALUATION = "evaluation"


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, index=True)
    workspace_id = Column(String, ForeignKey("workspaces.id"), nullable=False)
    job_type = Column(Enum(JobType), nullable=False)
    status = Column(Enum(JobStatus), default=JobStatus.PENDING)
    gene_query = Column(String, nullable=True)
    input_data = Column(Text, nullable=True)   # JSON-serialized input params
    result_data = Column(Text, nullable=True)   # JSON-serialized output
    error_message = Column(Text, nullable=True)
    hallucination_rate = Column(Float, nullable=True)
    confidence_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    completed_at = Column(DateTime, nullable=True)

    workspace = relationship("Workspace", back_populates="jobs")
