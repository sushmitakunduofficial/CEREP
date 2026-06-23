"""
FastAPI endpoint tests using TestClient.
Covers: auth (register/login), workspace CRUD, analysis job.
"""
import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture(scope="module")
def client():
    """Provide a TestClient with lifespan (DB init + KG preload)."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def auth_headers(client):
    """Register a test user and return auth headers."""
    reg = client.post("/auth/register", json={
        "username": "testuser_cerep",
        "email": "test@cerep.ai",
        "password": "Test1234!",
    })
    # If user already exists from a prior run, login instead
    if reg.status_code == 409:
        reg = client.post("/auth/login", json={
            "username": "testuser_cerep",
            "password": "Test1234!",
        })
    assert reg.status_code in (200, 201), reg.text
    token = reg.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


# ── /health ────────────────────────────────────────────────────────────────────
def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["kg_nodes"] >= 20


# ── /auth ──────────────────────────────────────────────────────────────────────
def test_register_duplicate_returns_409(client):
    # First registration done in auth_headers fixture
    resp = client.post("/auth/register", json={
        "username": "testuser_cerep",
        "email": "test@cerep.ai",
        "password": "Test1234!",
    })
    assert resp.status_code == 409


def test_login_wrong_password(client):
    resp = client.post("/auth/login", json={
        "username": "testuser_cerep",
        "password": "wrongpassword",
    })
    assert resp.status_code == 401


def test_login_returns_token(client):
    resp = client.post("/auth/login", json={
        "username": "testuser_cerep",
        "password": "Test1234!",
    })
    assert resp.status_code == 200
    assert "access_token" in resp.json()


# ── /workspaces ────────────────────────────────────────────────────────────────
def test_create_workspace(client, auth_headers):
    resp = client.post("/workspaces", json={
        "name": "Test Workspace",
        "description": "Created by pytest",
    }, headers=auth_headers)
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "Test Workspace"
    assert "id" in data


def test_list_workspaces(client, auth_headers):
    resp = client.get("/workspaces", headers=auth_headers)
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
    assert len(resp.json()) >= 1


def test_workspaces_require_auth(client):
    resp = client.get("/workspaces")
    assert resp.status_code == 403


# ── /analysis ──────────────────────────────────────────────────────────────────
def test_analysis_run(client, auth_headers):
    # First get a workspace id
    ws_resp = client.get("/workspaces", headers=auth_headers)
    workspace_id = ws_resp.json()[0]["id"]

    resp = client.post("/analysis/run", json={
        "workspace_id": workspace_id,
        "genes": ["TP53"],
        "max_hops": 3,
        "top_k": 5,
    }, headers=auth_headers)
    assert resp.status_code == 202
    data = resp.json()
    assert data["status"] in ("completed", "failed")
    assert "job_id" in data


def test_get_job(client, auth_headers):
    ws_resp = client.get("/workspaces", headers=auth_headers)
    workspace_id = ws_resp.json()[0]["id"]

    run_resp = client.post("/analysis/run", json={
        "workspace_id": workspace_id,
        "genes": ["BRCA1"],
        "max_hops": 2,
        "top_k": 3,
    }, headers=auth_headers)
    job_id = run_resp.json()["job_id"]

    get_resp = client.get(f"/analysis/{job_id}", headers=auth_headers)
    assert get_resp.status_code == 200
    assert get_resp.json()["job_id"] == job_id


# ── /reasoning/paths ──────────────────────────────────────────────────────────
def test_reasoning_paths(client, auth_headers):
    resp = client.get("/reasoning/paths?genes=TP53&max_hops=3&top_k=5",
                      headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    # Should return status success or no_match
    assert "status" in data
