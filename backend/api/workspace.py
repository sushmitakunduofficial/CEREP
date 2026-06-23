"""
Workspace API — GET/POST/DELETE /workspaces
"""
import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.api.deps import get_current_user
from backend.models.database import get_db
from backend.models.user import User
from backend.models.workspace import Workspace

router = APIRouter()


# ── Schemas ────────────────────────────────────────────────────────────────────
class WorkspaceCreate(BaseModel):
    name: str
    description: str = ""


class WorkspaceOut(BaseModel):
    id: str
    name: str
    description: str
    owner_id: str

    model_config = {"from_attributes": True}


# ── Routes ─────────────────────────────────────────────────────────────────────
@router.get("", response_model=List[WorkspaceOut])
def list_workspaces(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return db.query(Workspace).filter(Workspace.owner_id == current_user.id).all()


@router.post("", response_model=WorkspaceOut, status_code=status.HTTP_201_CREATED)
def create_workspace(
    payload: WorkspaceCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    ws = Workspace(
        id=str(uuid.uuid4()),
        name=payload.name,
        description=payload.description,
        owner_id=current_user.id,
    )
    db.add(ws)
    db.commit()
    db.refresh(ws)
    return ws


@router.delete("/{workspace_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_workspace(
    workspace_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    ws = db.query(Workspace).filter(
        Workspace.id == workspace_id,
        Workspace.owner_id == current_user.id,
    ).first()
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
    db.delete(ws)
    db.commit()
