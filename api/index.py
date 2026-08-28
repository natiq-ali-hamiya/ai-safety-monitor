"""
AI Safety Monitoring System — FastAPI Backend
Hardened for Serverless (Vercel / AWS Lambda) & Traditional Hosting (Docker / Render)
"""

from fastapi import FastAPI, APIRouter, HTTPException, Depends, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, field_validator
from typing import Optional, List, Dict, Any
import httpx
import hashlib
import hmac
import jwt
import re
import uuid
import sqlite3
import json
from datetime import datetime, timedelta
import os
from pathlib import Path
from dotenv import load_dotenv
from collections import defaultdict
import time
import traceback

load_dotenv()

# ── Base Directory & Config ───────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

SUPABASE_URL     = os.getenv("SUPABASE_URL", "https://lgsvbbzaocprdingtgtc.supabase.co")
SUPABASE_KEY     = os.getenv("SUPABASE_KEY")
JWT_SECRET       = os.getenv("JWT_SECRET", "super-secret-ai-safety-command-center-jwt-key-2026-production")
JWT_EXPIRE_HOURS = 24

# Writable ephemeral database path for Serverless (/tmp) vs Local
if os.getenv("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
    SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "/tmp/safety_monitor.db")
else:
    SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", str(BASE_DIR / "safety_monitor.db"))

USE_SUPABASE = bool(SUPABASE_KEY and SUPABASE_KEY.strip() and not SUPABASE_KEY.startswith("your_"))

# ── Password Hashing (Lightweight & Pure-Python Safe) ─────
def hash_password(password: str) -> str:
    """Generate SHA-256 salted hash (immune to C-extension crashes in Lambda)."""
    salt = "ai_safety_secure_salt_2026"
    return hashlib.sha256(f"{salt}{password}".encode("utf-8")).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify plain password against hashed password."""
    return hmac.compare_digest(hash_password(plain_password), hashed_password)

# ── SQLite Database Setup & Initialization ────────────────
_db_initialized = False

def get_db_connection():
    global _db_initialized
    conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False, timeout=10.0)
    conn.row_factory = sqlite3.Row
    if not _db_initialized:
        init_sqlite_db(conn)
        _db_initialized = True
    return conn

def init_sqlite_db(conn):
    cursor = conn.cursor()

    # Users
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        full_name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'operator',
        is_active INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL
    )
    """)

    # Incidents
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS incidents (
        id TEXT PRIMARY KEY,
        camera_id TEXT,
        incident_type TEXT NOT NULL,
        severity TEXT NOT NULL DEFAULT 'medium',
        status TEXT NOT NULL DEFAULT 'pending',
        location_description TEXT,
        latitude REAL,
        longitude REAL,
        detected_persons TEXT,
        notes TEXT,
        reviewed_by TEXT,
        reviewed_at TEXT,
        created_at TEXT NOT NULL
    )
    """)

    # Cameras
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS cameras (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        location TEXT NOT NULL,
        latitude REAL,
        longitude REAL,
        is_active INTEGER NOT NULL DEFAULT 1,
        added_by TEXT,
        created_at TEXT NOT NULL
    )
    """)

    # Alert Logs
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS alert_logs (
        id TEXT PRIMARY KEY,
        incident_id TEXT NOT NULL,
        alert_type TEXT NOT NULL,
        channel TEXT NOT NULL,
        recipient TEXT NOT NULL,
        message TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'sent',
        sent_by TEXT,
        created_at TEXT NOT NULL
    )
    """)

    # Evidence
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS evidence (
        id TEXT PRIMARY KEY,
        incident_id TEXT NOT NULL,
        evidence_type TEXT NOT NULL DEFAULT 'video_clip',
        file_url TEXT,
        thumbnail_url TEXT,
        duration_seconds INTEGER,
        created_at TEXT NOT NULL
    )
    """)

    conn.commit()

    # Seed Default Admin & Sample Data if empty
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        admin_id = str(uuid.uuid4())
        pw_hash = hash_password("secret")
        now = datetime.utcnow().isoformat()
        
        cursor.execute("""
        INSERT INTO users (id, full_name, email, password_hash, role, is_active, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (admin_id, "System Admin", "admin@aisafety.pk", pw_hash, "admin", 1, now))

        operator_id = str(uuid.uuid4())
        cursor.execute("""
        INSERT INTO users (id, full_name, email, password_hash, role, is_active, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (operator_id, "Chief Operator", "operator@aisafety.pk", pw_hash, "operator", 1, now))

        # Seed Sample Cameras
        cam1_id = str(uuid.uuid4())
        cam2_id = str(uuid.uuid4())
        cam3_id = str(uuid.uuid4())
        cursor.execute("""
        INSERT INTO cameras (id, name, location, latitude, longitude, is_active, added_by, created_at)
        VALUES 
        (?, 'Main Entrance CCTV 01', 'School Main Gate', 31.5204, 74.3587, 1, ?, ?),
        (?, 'Playground Perimeter 02', 'North Play Area', 31.5209, 74.3592, 1, ?, ?),
        (?, 'Parking / Rear Exit 03', 'South Parking Lot', 31.5198, 74.3579, 1, ?, ?)
        """, (cam1_id, admin_id, now, cam2_id, admin_id, now, cam3_id, admin_id, now))

        # Seed Sample Incidents for Demo
        cursor.execute("""
        INSERT INTO incidents (id, camera_id, incident_type, severity, status, location_description, latitude, longitude, detected_persons, notes, created_at)
        VALUES 
        (?, ?, 'CHILD_IN_DANGER', 'critical', 'pending', 'Main Entrance - Restricted Zone', 31.5204, 74.3587, '["Child (Age ~5)", "Vehicle #LEA-4521"]', 'Unsupervised child approaching road near entrance gate.', ?),
        (?, ?, 'WEAPON_DETECTED', 'critical', 'pending', 'South Parking Lot', 31.5198, 74.3579, '["Suspect in dark jacket"]', 'YOLOv8 detected knife object near rear gate.', ?),
        (?, ?, 'FIGHT_DETECTED', 'high', 'confirmed', 'Playground North', 31.5209, 74.3592, '["2 Persons"]', 'Physical altercation flagged by pose detector.', ?),
        (?, ?, 'PERSON_LYING_DOWN', 'high', 'resolved', 'East Hallway', 31.5201, 74.3582, '["1 Student"]', 'Individual fell; resolved by 1122 First Aid team.', ?)
        """, (
            str(uuid.uuid4()), cam1_id, (datetime.utcnow() - timedelta(minutes=5)).isoformat(),
            str(uuid.uuid4()), cam3_id, (datetime.utcnow() - timedelta(minutes=25)).isoformat(),
            str(uuid.uuid4()), cam2_id, (datetime.utcnow() - timedelta(hours=2)).isoformat(),
            str(uuid.uuid4()), cam1_id, (datetime.utcnow() - timedelta(hours=5)).isoformat()
        ))

        conn.commit()

# ── App & Router ──────────────────────────────────────────
app = FastAPI(
    title="AI Safety Monitoring System",
    description="AI-powered safety monitoring API with multi-database support",
    version="2.0.0"
)

# Global exception handler prevents unhandled crash from killing Lambda
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"[Error] Unhandled error: {exc}")
    traceback.print_exc()
    resp = JSONResponse(status_code=500, content={"detail": str(exc), "type": type(exc).__name__})
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "*"
    return resp

# Path normalization & CORS guarantee middleware
@app.middleware("http")
async def vercel_path_and_cors_middleware(request: Request, call_next):
    if request.method == "OPTIONS":
        resp = JSONResponse(content={"status": "ok"}, status_code=200)
    else:
        path = request.scope.get("path", "")
        if path.startswith("/api/index.py"):
            request.scope["path"] = path.replace("/api/index.py", "", 1) or "/"
        resp = await call_next(request)
    
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "*"
    return resp

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer(auto_error=False)
router = APIRouter()

# ── Database Layer (Dual Supabase / SQLite) ───────────────
async def db_query(table: str, method: str = "GET", data: dict = None, filters: str = "", single: bool = False):
    global USE_SUPABASE

    if USE_SUPABASE:
        try:
            url = f"{SUPABASE_URL}/rest/v1/{table}{filters}"
            headers = {
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=representation"
            }
            async with httpx.AsyncClient(timeout=10.0) as client:
                if method == "GET":
                    r = await client.get(url, headers=headers)
                elif method == "POST":
                    r = await client.post(url, headers=headers, json=data)
                elif method == "PATCH":
                    r = await client.patch(url, headers=headers, json=data)
                elif method == "DELETE":
                    r = await client.delete(url, headers=headers)
                r.raise_for_status()
                res = r.json()
                return res[0] if single and isinstance(res, list) and res else res
        except Exception as e:
            print(f"[Database] Supabase error ({e}). Falling back to local SQLite.")

    # SQLite Fallback Engine
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        if method == "GET":
            query = f"SELECT * FROM {table}"
            params = []
            where_clauses = []

            if filters:
                clean_filters = filters.lstrip("?")
                for part in clean_filters.split("&"):
                    if "=" in part:
                        k, v = part.split("=", 1)
                        if v.startswith("eq."):
                            val = v[3:]
                            where_clauses.append(f"{k} = ?")
                            params.append(val)

            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)
            
            if "order=created_at.desc" in filters or table == "incidents":
                query += " ORDER BY created_at DESC"

            if "limit=" in filters:
                match = re.search(r"limit=(\d+)", filters)
                if match:
                    query += f" LIMIT {match.group(1)}"

            cursor.execute(query, params)
            rows = [dict(row) for row in cursor.fetchall()]
            for r in rows:
                if "detected_persons" in r and isinstance(r["detected_persons"], str):
                    try:
                        r["detected_persons"] = json.loads(r["detected_persons"])
                    except:
                        pass
            return rows[0] if single and rows else rows

        elif method == "POST":
            data = data or {}
            if "id" not in data:
                data["id"] = str(uuid.uuid4())
            if "created_at" not in data:
                data["created_at"] = datetime.utcnow().isoformat()

            clean_data = {}
            for k, v in data.items():
                if isinstance(v, (list, dict)):
                    clean_data[k] = json.dumps(v)
                else:
                    clean_data[k] = v

            keys = list(clean_data.keys())
            placeholders = ", ".join(["?"] * len(keys))
            columns = ", ".join(keys)
            query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
            cursor.execute(query, list(clean_data.values()))
            conn.commit()
            return [data]

        elif method == "PATCH":
            data = data or {}
            clean_data = {}
            for k, v in data.items():
                if isinstance(v, (list, dict)):
                    clean_data[k] = json.dumps(v)
                else:
                    clean_data[k] = v

            set_clauses = [f"{k} = ?" for k in clean_data.keys()]
            params = list(clean_data.values())
            
            where_clauses = []
            if filters:
                clean_filters = filters.lstrip("?")
                for part in clean_filters.split("&"):
                    if "=" in part:
                        k, v = part.split("=", 1)
                        if v.startswith("eq."):
                            val = v[3:]
                            where_clauses.append(f"{k} = ?")
                            params.append(val)

            where_str = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
            query = f"UPDATE {table} SET {', '.join(set_clauses)}{where_str}"
            cursor.execute(query, params)
            conn.commit()

            cursor.execute(f"SELECT * FROM {table}{where_str}", [val for k, v in [(p.split("=")[0], p.split("=")[1][3:]) for p in clean_filters.split("&") if "=" in p and p.split("=")[1].startswith("eq.")]])
            rows = [dict(r) for r in cursor.fetchall()]
            return rows if rows else [data]

        elif method == "DELETE":
            where_clauses = []
            params = []
            if filters:
                clean_filters = filters.lstrip("?")
                for part in clean_filters.split("&"):
                    if "=" in part:
                        k, v = part.split("=", 1)
                        if v.startswith("eq."):
                            val = v[3:]
                            where_clauses.append(f"{k} = ?")
                            params.append(val)
            where_str = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
            cursor.execute(f"DELETE FROM {table}{where_str}", params)
            conn.commit()
            return []

    finally:
        conn.close()

# ── JWT helpers ───────────────────────────────────────────
def create_token(user_id: str, email: str, role: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "role": role,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRE_HOURS)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> dict:
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired — please log in again")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

def require_admin(token: dict = Depends(verify_token)):
    if token.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return token

def sanitize_string(value: str, max_length: int = 500) -> str:
    if not value:
        return value
    value = re.sub(r"['\";\\]", "", str(value))
    return value[:max_length].strip()

# ── Request Models ────────────────────────────────────────
class RegisterRequest(BaseModel):
    full_name: str
    email: str
    password: str
    role: Optional[str] = "operator"

    @field_validator("full_name")
    @classmethod
    def validate_name(cls, v):
        if len(v) < 2:
            raise ValueError("Name too short")
        return sanitize_string(v, 100)

    @field_validator("password")
    @classmethod
    def validate_password(cls, v):
        if len(v) < 4:
            raise ValueError("Password must be at least 4 characters")
        return v

class LoginRequest(BaseModel):
    email: str
    password: str

class IncidentReport(BaseModel):
    camera_id: Optional[str] = None
    incident_type: str
    severity: str = "medium"
    location_description: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    detected_persons: Optional[List[Any]] = []

class IncidentReview(BaseModel):
    status: str
    notes: Optional[str] = None

class AlertRequest(BaseModel):
    incident_id: str
    alert_type: str
    channel: str
    recipient: str
    message: str

class EvidenceRecord(BaseModel):
    incident_id: str
    evidence_type: str = "video_clip"
    file_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    duration_seconds: Optional[int] = None

# ── ROUTER ENDPOINTS ──────────────────────────────────────

@router.get("/")
async def root_status():
    return {
        "system": "AI Safety Monitoring System",
        "version": "2.0.0",
        "database": "Supabase" if USE_SUPABASE else "SQLite",
        "status": "online"
    }

@router.get("/health")
async def health():
    return {
        "status": "healthy",
        "database": "Supabase" if USE_SUPABASE else "SQLite",
        "timestamp": datetime.utcnow().isoformat()
    }

@router.post("/auth/register")
async def register(req: RegisterRequest, token: dict = Depends(require_admin)):
    clean_email = req.email.strip().lower()
    existing = await db_query("users", "GET", filters=f"?email=eq.{clean_email}")
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    pw_hash = hash_password(req.password)
    user = await db_query("users", "POST", {
        "full_name":     req.full_name,
        "email":         clean_email,
        "password_hash": pw_hash,
        "role":          req.role or "operator",
        "is_active":     1
    })
    u = user[0]
    return {"message": "User created", "user": {
        "id": u["id"], "email": u["email"], "role": u["role"]}}

@router.post("/auth/login")
@router.post("/auth/login/")
async def login(req: LoginRequest, request: Request):
    clean_email = req.email.strip().lower()
    users = await db_query("users", "GET", filters=f"?email=eq.{clean_email}")
    if not users:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    user = users[0]
    if not user.get("is_active", 1):
        raise HTTPException(status_code=403, detail="Account deactivated")

    if not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_token(user["id"], user["email"], user["role"])
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id":        user["id"],
            "full_name": user["full_name"],
            "email":     user["email"],
            "role":      user["role"]
        }
    }

@router.get("/auth/me")
@router.get("/auth/me/")
async def get_current_user(token: dict = Depends(verify_token)):
    users = await db_query("users", "GET", filters=f"?id=eq.{token['sub']}")
    if not users:
        raise HTTPException(status_code=404, detail="User not found")
    u = users[0]
    return {"id": u["id"], "full_name": u["full_name"],
            "email": u["email"], "role": u["role"]}

@router.post("/incidents")
@router.post("/incidents/")
async def report_incident(req: IncidentReport, token: dict = Depends(verify_token)):
    incident = await db_query("incidents", "POST", {
        "camera_id":            req.camera_id,
        "incident_type":        req.incident_type,
        "severity":             req.severity,
        "status":               "pending",
        "location_description": req.location_description or "CCTV Camera Feed",
        "latitude":             req.latitude,
        "longitude":            req.longitude,
        "detected_persons":     req.detected_persons or []
    })
    return {"message": "Incident recorded", "incident": incident[0]}

@router.get("/incidents")
@router.get("/incidents/")
async def list_incidents(status: Optional[str] = None, limit: Optional[int] = 50, token: dict = Depends(verify_token)):
    filters = f"?order=created_at.desc&limit={limit}"
    if status:
        allowed = ["pending", "reviewed", "confirmed", "false_alarm", "resolved"]
        if status not in allowed:
            raise HTTPException(status_code=400, detail="Invalid status filter")
        filters += f"&status=eq.{status}"
    return await db_query("incidents", "GET", filters=filters)

@router.get("/incidents/{incident_id}")
async def get_incident(incident_id: str, token: dict = Depends(verify_token)):
    result = await db_query("incidents", "GET", filters=f"?id=eq.{incident_id}")
    if not result:
        raise HTTPException(status_code=404, detail="Incident not found")
    return result[0]

@router.patch("/incidents/{incident_id}/review")
async def review_incident(incident_id: str, req: IncidentReview, token: dict = Depends(verify_token)):
    updated = await db_query("incidents", "PATCH", {
        "status":      req.status,
        "notes":       req.notes,
        "reviewed_by": token["sub"],
        "reviewed_at": datetime.utcnow().isoformat()
    }, filters=f"?id=eq.{incident_id}")
    return {"message": f"Incident marked as {req.status}", "incident": updated[0]}

@router.post("/alerts/send")
@router.post("/alerts/send/")
async def send_alert(req: AlertRequest, token: dict = Depends(verify_token)):
    incidents = await db_query("incidents", "GET", filters=f"?id=eq.{req.incident_id}")
    if not incidents:
        raise HTTPException(status_code=404, detail="Incident not found")

    if incidents[0]["status"] != "confirmed":
        raise HTTPException(status_code=400, detail="Incident must be confirmed before sending alert")

    log_entry = await db_query("alert_logs", "POST", {
        "incident_id": req.incident_id,
        "alert_type":  req.alert_type,
        "channel":     req.channel,
        "recipient":   req.recipient,
        "message":     req.message,
        "status":      "sent",
        "sent_by":     token["sub"]
    })
    return {"message": f"Alert sent via {req.channel}", "alert_log": log_entry[0]}

@router.get("/alerts/{incident_id}")
@router.get("/alerts/{incident_id}/")
async def get_alert_logs(incident_id: str, token: dict = Depends(verify_token)):
    return await db_query("alert_logs", "GET", filters=f"?incident_id=eq.{incident_id}")

@router.post("/evidence")
@router.post("/evidence/")
async def save_evidence(req: EvidenceRecord, token: dict = Depends(verify_token)):
    result = await db_query("evidence", "POST", {
        "incident_id":      req.incident_id,
        "evidence_type":    req.evidence_type,
        "file_url":         req.file_url,
        "thumbnail_url":    req.thumbnail_url,
        "duration_seconds": req.duration_seconds,
    })
    return {"message": "Evidence saved", "evidence": result[0]}

@router.get("/evidence/{incident_id}")
@router.get("/evidence/{incident_id}/")
async def get_evidence(incident_id: str, token: dict = Depends(verify_token)):
    return await db_query("evidence", "GET", filters=f"?incident_id=eq.{incident_id}")

@router.get("/cameras")
@router.get("/cameras/")
async def list_cameras(token: dict = Depends(verify_token)):
    return await db_query("cameras", "GET", filters="?is_active=eq.1")

@router.post("/cameras")
@router.post("/cameras/")
async def add_camera(camera: dict, token: dict = Depends(require_admin)):
    camera["added_by"] = token["sub"]
    if "name" in camera:
        camera["name"] = sanitize_string(camera["name"], 100)
    if "location" in camera:
        camera["location"] = sanitize_string(camera["location"], 200)
    result = await db_query("cameras", "POST", camera)
    return {"message": "Camera registered", "camera": result[0]}

@router.get("/dashboard/stats")
@router.get("/dashboard/stats/")
async def dashboard_stats(token: dict = Depends(verify_token)):
    all_incidents = await db_query("incidents", "GET")
    pending      = sum(1 for i in all_incidents if i.get("status") == "pending")
    confirmed    = sum(1 for i in all_incidents if i.get("status") == "confirmed")
    false_alarms = sum(1 for i in all_incidents if i.get("status") == "false_alarm")
    resolved     = sum(1 for i in all_incidents if i.get("status") == "resolved")
    return {
        "total_incidents": len(all_incidents),
        "pending_review":  pending,
        "confirmed":       confirmed,
        "false_alarms":    false_alarms,
        "resolved":        resolved
    }

@router.post("/demo/simulate-incident")
@router.post("/demo/simulate-incident/")
async def simulate_demo_incident(incident_type: Optional[str] = "CHILD_IN_DANGER", token: dict = Depends(verify_token)):
    types = {
        "CHILD_IN_DANGER": ("critical", "Main Gate Playground", ["Child (Age ~4)"]),
        "WEAPON_DETECTED": ("critical", "South Gate Perimeter", ["Person holding knife"]),
        "FIGHT_DETECTED": ("high", "Courtyard Area", ["2 Active Combatants"]),
        "PERSON_LYING_DOWN": ("high", "Corridor B Entrance", ["1 Unresponsive Individual"]),
        "HIT_AND_RUN": ("critical", "East Parking Lot", ["Vehicle Plate LEA-9988", "1 Pedestrian"])
    }
    sev, loc, persons = types.get(incident_type, ("medium", "CCTV Zone 1", ["Person"]))
    
    incident = await db_query("incidents", "POST", {
        "incident_type":        incident_type,
        "severity":             sev,
        "status":               "pending",
        "location_description": loc,
        "latitude":             31.5204,
        "longitude":            74.3587,
        "detected_persons":     persons
    })
    return {"message": f"Simulated {incident_type} generated!", "incident": incident[0]}

# ── Mount router to both root and /api prefixes ───────────
app.include_router(router)
app.include_router(router, prefix="/api")

# ── Fallback Web UI route directly on app ─────────────────
@app.get("/app", response_class=HTMLResponse)
@app.get("/index.html", response_class=HTMLResponse)
async def serve_frontend_app():
    html_file = BASE_DIR / "index.html"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text(encoding="utf-8"), status_code=200)
    return HTMLResponse("<h1>AI Safety Monitoring System</h1>")
