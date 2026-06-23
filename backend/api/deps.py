"""
Shared API dependencies — JWT auth, KG builder injection.
"""
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from backend.core.security import decode_token as decode_access_token
from backend.models.database import get_db
from backend.models.user import User
from backend.graph.graph_builder import CERAPGraphBuilder

_bearer = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
    db: Session = Depends(get_db),
) -> User:
    token = credentials.credentials
    
    # Support for Quick Demo Access flow
    if token == "demo_mode_token":
        user = db.query(User).filter(User.id == "demo").first()
        if not user:
             raise HTTPException(status_code=404, detail="Demo user not seeded")
        return user

    payload = decode_access_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user_id: str = payload.get("sub")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


def get_kg(request: Request) -> CERAPGraphBuilder:
    """Inject the singleton KG builder from app state."""
    return request.app.state.kg_builder
