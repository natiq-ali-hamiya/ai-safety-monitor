import pytest
from fastapi.testclient import TestClient
import os

# Set test environment variables before importing app
os.environ["SQLITE_DB_PATH"] = "/tmp/test_safety_monitor.db"
os.environ["USE_SUPABASE"] = "false"

from main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_login_success():
    response = client.post(
        "/api/auth/login",
        json={"email": "admin@aisafety.pk", "password": "secret"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["user"]["email"] == "admin@aisafety.pk"
    assert data["user"]["role"] == "admin"

def test_login_trailing_slash():
    response = client.post(
        "/api/auth/login/",
        json={"email": "operator@aisafety.pk", "password": "secret"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["user"]["email"] == "operator@aisafety.pk"

def test_login_invalid_password():
    response = client.post(
        "/api/auth/login",
        json={"email": "admin@aisafety.pk", "password": "wrongpassword"}
    )
    assert response.status_code == 401
    assert "Invalid email or password" in response.json()["detail"]

def test_login_nonexistent_user():
    response = client.post(
        "/api/auth/login",
        json={"email": "nonexistent@aisafety.pk", "password": "secret"}
    )
    assert response.status_code == 401

def test_auth_me_endpoint():
    # First login to get token
    login_res = client.post(
        "/api/auth/login",
        json={"email": "admin@aisafety.pk", "password": "secret"}
    )
    token = login_res.json()["access_token"]

    # Test GET /api/auth/me
    me_res = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert me_res.status_code == 200
    assert me_res.json()["email"] == "admin@aisafety.pk"
