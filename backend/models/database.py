from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.core.config import get_settings
from backend.models.base import Base
from backend.models.user import User       # noqa: F401 – register models
from backend.models.workspace import Workspace  # noqa: F401
from backend.models.job import Job         # noqa: F401

settings = get_settings()

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    
    # Seed Demo Data for Quick Access
    from backend.models.user import User
    from backend.models.workspace import Workspace
    from sqlalchemy.orm import Session
    
    db = Session(bind=engine)
    try:
        # 1. Ensure Demo User exists
        demo_user = db.query(User).filter(User.id == "demo").first()
        if not demo_user:
            demo_user = User(
                id="demo",
                username="DemoUser",
                email="demo@cerep.ai",
                hashed_password="demo_not_used_directly",
            )
            db.add(demo_user)
            db.flush()
            
        # 2. Ensure Demo Workspace exists
        demo_ws = db.query(Workspace).filter(Workspace.id == "demo-brca-1").first()
        if not demo_ws:
            demo_ws = Workspace(
                id="demo-brca-1",
                name="BRCA Mechanistic Demo",
                description="Sample workspace for breast cancer mechanistic reasoning demo.",
                owner_id=demo_user.id,
            )
            db.add(demo_ws)
        
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[ERROR] Failed to seed demo data: {e}")
    finally:
        db.close()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
