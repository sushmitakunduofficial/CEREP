"""
CEREP Backend Launcher
Run this from the project root: python start_backend.py
"""
import sys, os

# Ensure the project root is on sys.path so 'backend.*' imports resolve
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Create .env from .env.example if missing
env_path = os.path.join(ROOT, ".env")
example_path = os.path.join(ROOT, ".env.example")
if not os.path.exists(env_path) and os.path.exists(example_path):
    import shutil
    shutil.copy(example_path, env_path)
    print("[INFO] Created .env from .env.example — edit SECRET_KEY before production use!")

import uvicorn

if __name__ == "__main__":
    print("=" * 55)
    print("  CEREP Backend  ->  http://localhost:8000")
    print("  API docs       ->  http://localhost:8000/docs")
    print("  Health check   ->  http://localhost:8000/health")
    print("=" * 55)
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[ROOT],
    )
