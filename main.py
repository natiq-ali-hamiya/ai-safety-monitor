"""
AI Safety Monitoring System — FastAPI Backend
Hardened for Serverless (Vercel) & Traditional Hosting (Docker / local)

=== FIX LOG (2026-08-30) ===
BUG 1 (CRITICAL): USE_SUPABASE defaulted to False — on Vercel, every cold
  start got a fresh empty /tmp SQLite with no users, so login always returned
  "Invalid email or password". Fixed: on Vercel, Supabase is REQUIRED.
BUG 2: Silent SQLite fallback — Supabase errors were silently caught and the
  code fell through to an empty ephemeral SQLite. Fixed: on Vercel, if Supabase
  fails, login returns a loud 503 error instead of a silent wrong-database miss.
BUG 3: httpx timeout was 1.5s — Supabase cold starts take 2–5s, causing every
  first request to time out and fall back to empty SQLite. Fixed: 8s timeout.
BUG 4: No debug logging — impossible to tell from Vercel logs which backend was
  used. Fixed: login response now includes _auth_backend field in debug mode,
  and every login attempt prints which backend was used to stdout.
BUG 5: OPTIONS middleware had a missing 'return' — fell through to call_next
  which returned 405. Fixed in previous commit, preserved here.
"""

from fastapi import FastAPI, APIRouter, HTTPException, Depends, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, field_validator
from typing import Optional, List, Any
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
import traceback

load_dotenv()

# ── Environment & Config ───────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

SUPABASE_URL     = os.getenv("SUPABASE_URL", "https://lgsvbbzaocprdingtgtc.supabase.co")
SUPABASE_KEY     = os.getenv("SUPABASE_KEY", "")          # NO hardcoded fallback
JWT_SECRET       = os.getenv("JWT_SECRET", "super-secret-ai-safety-command-center-jwt-key-2026-production")
JWT_EXPIRE_HOURS = 24

# Detect serverless environment
IS_VERCEL  = bool(os.getenv("VERCEL"))
IS_LAMBDA  = bool(os.getenv("AWS_LAMBDA_FUNCTION_NAME"))
IS_SERVERLESS = IS_VERCEL or IS_LAMBDA

# SQLite path — /tmp on serverless (ephemeral!), local file otherwise
if IS_SERVERLESS:
    SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "/tmp/safety_monitor.db")
else:
    SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", str(BASE_DIR / "safety_monitor.db"))

# ── Auth-Backend Selection ─────────────────────────────────
# On Vercel: Supabase is REQUIRED (SQLite is ephemeral/empty on every cold start).
# Locally:   Default to SQLite for zero-latency dev experience.
#
# To enable Supabase locally, set USE_SUPABASE=true in your .env file.
# To enable Supabase on Vercel, just set SUPABASE_KEY in Vercel env vars
#   (IS_SERVERLESS automatically forces Supabase when the key is present).

_supabase_key_valid = bool(SUPABASE_KEY and len(SUPABASE_KEY.strip()) > 20)

if IS_SERVERLESS:
    # On Vercel: REQUIRE Supabase. If key is missing, we'll fail loudly at login.
    USE_SUPABASE = _supabase_key_valid
    if not USE_SUPABASE:
        print("[STARTUP WARNING] Running on Vercel without a valid SUPABASE_KEY! "
              "Login will fail because SQLite has no persistent users. "
              "Set SUPABASE_KEY in Vercel environment variables.")
else:
    # Local: USE_SUPABASE env var controls it, default False (fast SQLite dev)
    USE_SUPABASE = bool(
        os.getenv("USE_SUPABASE", "false").lower() == "true"
        and _supabase_key_valid
    )

print(f"[STARTUP] Environment: {'Vercel' if IS_VERCEL else 'Lambda' if IS_LAMBDA else 'Local'} | "
      f"Auth backend: {'Supabase' if USE_SUPABASE else 'SQLite'} | "
      f"SQLite path: {SQLITE_DB_PATH}")

# ── Password Hashing ───────────────────────────────────────
HASH_SALT = "ai_safety_secure_salt_2026"

def hash_password(password: str) -> str:
    """SHA-256 + static salt. Pure-Python, no C extensions needed on Lambda."""
    return hashlib.sha256(f"{HASH_SALT}{password}".encode("utf-8")).hexdigest()

def verify_password(plain: str, hashed: str) -> bool:
    return hmac.compare_digest(hash_password(plain), hashed)

# ── SQLite Database Setup ──────────────────────────────────
_db_initialized = False

def get_db_connection():
    global _db_initialized
    db_path = Path(SQLITE_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=10.0)
    conn.row_factory = sqlite3.Row
    if not _db_initialized:
        _init_sqlite_db(conn)
        _db_initialized = True
    return conn

def _init_sqlite_db(conn):
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        full_name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'operator',
        is_active INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL
    )""")

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
    )""")

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
    )""")

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
    )""")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS evidence (
        id TEXT PRIMARY KEY,
        incident_id TEXT NOT NULL,
        evidence_type TEXT NOT NULL DEFAULT 'video_clip',
        file_url TEXT,
        thumbnail_url TEXT,
        duration_seconds INTEGER,
        created_at TEXT NOT NULL
    )""")

    conn.commit()

    # Seed default users (local dev only — Supabase has its own persistent data)
    pw_hash = hash_password("secret")
    now = datetime.utcnow().isoformat()

    cursor.execute("SELECT COUNT(*) FROM users WHERE email = ?", ("admin@aisafety.pk",))
    if cursor.fetchone()[0] == 0:
        cursor.execute("""
        INSERT INTO users (id, full_name, email, password_hash, role, is_active, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (str(uuid.uuid4()), "System Admin", "admin@aisafety.pk", pw_hash, "admin", 1, now))

    cursor.execute("SELECT COUNT(*) FROM users WHERE email = ?", ("operator@aisafety.pk",))
    if cursor.fetchone()[0] == 0:
        cursor.execute("""
        INSERT INTO users (id, full_name, email, password_hash, role, is_active, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (str(uuid.uuid4()), "Chief Operator", "operator@aisafety.pk", pw_hash, "operator", 1, now))

    # Seed cameras and sample incidents if empty
    cursor.execute("SELECT COUNT(*) FROM cameras")
    if cursor.fetchone()[0] == 0:
        placeholder_admin = str(uuid.uuid4())
        cam1_id, cam2_id, cam3_id = str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())
        cursor.execute("""
        INSERT INTO cameras (id, name, location, latitude, longitude, is_active, added_by, created_at)
        VALUES
        (?, 'Main Entrance CCTV 01', 'School Main Gate', 31.5204, 74.3587, 1, ?, ?),
        (?, 'Playground Perimeter 02', 'North Play Area', 31.5209, 74.3592, 1, ?, ?),
        (?, 'Parking / Rear Exit 03', 'South Parking Lot', 31.5198, 74.3579, 1, ?, ?)
        """, (cam1_id, placeholder_admin, now,
              cam2_id, placeholder_admin, now,
              cam3_id, placeholder_admin, now))

        cursor.execute("""
        INSERT INTO incidents (id, camera_id, incident_type, severity, status,
            location_description, latitude, longitude, detected_persons, notes, created_at)
        VALUES
        (?, ?, 'CHILD_IN_DANGER',   'critical', 'pending',   'Main Entrance - Restricted Zone', 31.5204, 74.3587, '["Child (Age ~5)", "Vehicle #LEA-4521"]', 'Unsupervised child approaching road near entrance gate.', ?),
        (?, ?, 'WEAPON_DETECTED',   'critical', 'pending',   'South Parking Lot', 31.5198, 74.3579, '["Suspect in dark jacket"]', 'YOLOv8 detected knife object near rear gate.', ?),
        (?, ?, 'FIGHT_DETECTED',    'high',     'confirmed', 'Playground North',  31.5209, 74.3592, '["2 Persons"]', 'Physical altercation flagged by pose detector.', ?),
        (?, ?, 'PERSON_LYING_DOWN', 'high',     'resolved',  'East Hallway',      31.5201, 74.3582, '["1 Student"]', 'Individual fell; resolved by 1122 First Aid team.', ?)
        """, (
            str(uuid.uuid4()), cam1_id, (datetime.utcnow() - timedelta(minutes=5)).isoformat(),
            str(uuid.uuid4()), cam3_id, (datetime.utcnow() - timedelta(minutes=25)).isoformat(),
            str(uuid.uuid4()), cam2_id, (datetime.utcnow() - timedelta(hours=2)).isoformat(),
            str(uuid.uuid4()), cam1_id, (datetime.utcnow() - timedelta(hours=5)).isoformat(),
        ))

    conn.commit()


# ── FastAPI App ────────────────────────────────────────────
app = FastAPI(
    title="AI Safety Monitoring System",
    description="AI-powered safety monitoring API",
    version="2.0.0"
)

# ── Global exception handler ───────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"[UNHANDLED ERROR] {type(exc).__name__}: {exc}")
    traceback.print_exc()
    resp = JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": type(exc).__name__}
    )
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "*"
    return resp

# ── CORS + OPTIONS preflight middleware ────────────────────
@app.middleware("http")
async def cors_and_path_middleware(request: Request, call_next):
    # Return 200 immediately for all OPTIONS preflight requests
    if request.method == "OPTIONS":
        resp = JSONResponse(content={"status": "ok"}, status_code=200)
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "*"
        resp.headers["Access-Control-Max-Age"] = "86400"
        return resp  # ← CRITICAL: must return here, not fall through

    # Strip /api/index.py if Vercel injects it into the ASGI path scope
    path = request.scope.get("path", "")
    if "/api/index.py" in path:
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


# ── Database Layer ─────────────────────────────────────────
async def _supabase_query(table: str, method: str, data: dict = None, filters: str = ""):
    """Query Supabase REST API. Raises on any error — no silent fallback."""
    url = f"{SUPABASE_URL}/rest/v1/{table}{filters}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }
    # 8s timeout — Supabase cold starts can take 3-5s
    async with httpx.AsyncClient(timeout=8.0) as client:
        if method == "GET":
            r = await client.get(url, headers=headers)
        elif method == "POST":
            r = await client.post(url, headers=headers, json=data)
        elif method == "PATCH":
            r = await client.patch(url, headers=headers, json=data)
        elif method == "DELETE":
            r = await client.delete(url, headers=headers)
        else:
            raise ValueError(f"Unknown method: {method}")
        r.raise_for_status()
        return r.json()


def _sqlite_query(table: str, method: str, data: dict = None, filters: str = ""):
    """Query local SQLite. Returns a list of dicts."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        if method == "GET":
            query = f"SELECT * FROM {table}"
            params = []
            where_clauses = []

            if filters:
                for part in filters.lstrip("?").split("&"):
                    if "=" in part:
                        k, v = part.split("=", 1)
                        if v.startswith("eq."):
                            where_clauses.append(f"{k} = ?")
                            params.append(v[3:])

            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)

            if "order=created_at.desc" in filters or table == "incidents":
                query += " ORDER BY created_at DESC"

            if "limit=" in filters:
                m = re.search(r"limit=(\d+)", filters)
                if m:
                    query += f" LIMIT {m.group(1)}"

            cursor.execute(query, params)
            rows = [dict(row) for row in cursor.fetchall()]
            for row in rows:
                if "detected_persons" in row and isinstance(row["detected_persons"], str):
                    try:
                        row["detected_persons"] = json.loads(row["detected_persons"])
                    except Exception:
                        pass
            return rows

        elif method == "POST":
            data = data or {}
            if "id" not in data:
                data["id"] = str(uuid.uuid4())
            if "created_at" not in data:
                data["created_at"] = datetime.utcnow().isoformat()

            clean = {k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
                     for k, v in data.items()}
            cols = ", ".join(clean.keys())
            placeholders = ", ".join(["?"] * len(clean))
            cursor.execute(f"INSERT INTO {table} ({cols}) VALUES ({placeholders})",
                           list(clean.values()))
            conn.commit()
            return [data]

        elif method == "PATCH":
            data = data or {}
            clean = {k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
                     for k, v in data.items()}
            set_clause = ", ".join(f"{k} = ?" for k in clean)
            params = list(clean.values())

            where_clauses = []
            where_params = []
            if filters:
                for part in filters.lstrip("?").split("&"):
                    if "=" in part:
                        k, v = part.split("=", 1)
                        if v.startswith("eq."):
                            where_clauses.append(f"{k} = ?")
                            where_params.append(v[3:])

            where_str = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
            cursor.execute(f"UPDATE {table} SET {set_clause}{where_str}",
                           params + where_params)
            conn.commit()
            cursor.execute(f"SELECT * FROM {table}{where_str}", where_params)
            return [dict(r) for r in cursor.fetchall()] or [data]

        elif method == "DELETE":
            where_clauses = []
            params = []
            if filters:
                for part in filters.lstrip("?").split("&"):
                    if "=" in part:
                        k, v = part.split("=", 1)
                        if v.startswith("eq."):
                            where_clauses.append(f"{k} = ?")
                            params.append(v[3:])
            where_str = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
            cursor.execute(f"DELETE FROM {table}{where_str}", params)
            conn.commit()
            return []

    finally:
        conn.close()


async def db_query(table: str, method: str = "GET", data: dict = None,
                   filters: str = "", single: bool = False):
    """
    Unified DB query. On Vercel (IS_SERVERLESS=True):
      - Uses Supabase. If SUPABASE_KEY is missing, raises 503 immediately.
      - Does NOT silently fall back to empty SQLite.
    Locally:
      - Uses SQLite by default, or Supabase if USE_SUPABASE=true.
    """
    if USE_SUPABASE:
        try:
            result = await _supabase_query(table, method, data, filters)
            return result[0] if single and isinstance(result, list) and result else result
        except httpx.TimeoutException as e:
            print(f"[DB ERROR] Supabase timeout on {method} {table}: {e}")
            if IS_SERVERLESS:
                # On Vercel: do NOT fall to empty SQLite — fail loudly
                raise HTTPException(
                    status_code=503,
                    detail="Database temporarily unavailable (Supabase timeout). Please retry."
                )
            # Local: fall through to SQLite backup
            print("[DB] Falling back to local SQLite (non-serverless)")
        except httpx.HTTPStatusError as e:
            print(f"[DB ERROR] Supabase HTTP {e.response.status_code} on {method} {table}: {e}")
            if IS_SERVERLESS:
                raise HTTPException(
                    status_code=502,
                    detail=f"Database error: Supabase returned {e.response.status_code}. "
                           "Check SUPABASE_KEY and table permissions."
                )
            print("[DB] Falling back to local SQLite (non-serverless)")
        except Exception as e:
            print(f"[DB ERROR] Supabase unexpected error on {method} {table}: {e}")
            if IS_SERVERLESS:
                raise HTTPException(
                    status_code=503,
                    detail="Database connection failed. Check Vercel environment variables."
                )
            print("[DB] Falling back to local SQLite (non-serverless)")
    elif IS_SERVERLESS and not _supabase_key_valid:
        # Serverless + no valid key = immediate loud error
        raise HTTPException(
            status_code=503,
            detail="Server misconfiguration: SUPABASE_KEY is not set in Vercel environment "
                   "variables. SQLite cannot be used on Vercel (ephemeral filesystem). "
                   "Add SUPABASE_KEY to your Vercel project settings."
        )

    # SQLite path (local dev, or local Supabase fallback)
    result = _sqlite_query(table, method, data, filters)
    return result[0] if single and isinstance(result, list) and result else result


# ── JWT Helpers ────────────────────────────────────────────
def create_token(user_id: str, email: str, role: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "role": role,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRE_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def verify_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> dict:
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        return jwt.decode(credentials.credentials, JWT_SECRET, algorithms=["HS256"])
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
    value = re.sub(r"[\"'\\;]", "", str(value))
    return value[:max_length].strip()


# ── Request Models ─────────────────────────────────────────
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


# ── API Endpoints ──────────────────────────────────────────

@router.get("/")
async def root_status():
    return {
        "system": "AI Safety Monitoring System",
        "version": "2.0.0",
        "database": "Supabase" if USE_SUPABASE else "SQLite",
        "environment": "Vercel" if IS_VERCEL else "Lambda" if IS_LAMBDA else "Local",
        "status": "online",
        "supabase_key_present": _supabase_key_valid,
    }


@router.get("/health")
async def health():
    return {
        "status": "healthy",
        "database": "Supabase" if USE_SUPABASE else "SQLite",
        "is_serverless": IS_SERVERLESS,
        "supabase_key_present": _supabase_key_valid,
        "timestamp": datetime.utcnow().isoformat(),
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
        "is_active":     1,
    })
    u = user[0]
    return {"message": "User created", "user": {
        "id": u["id"], "email": u["email"], "role": u["role"]}}


@router.post("/auth/login")
@router.post("/auth/login/")
async def login(req: LoginRequest, request: Request):
    """
    Login endpoint. Logs which backend (Supabase vs SQLite) was used.
    Response includes _auth_backend for Vercel log debugging.
    """
    clean_email = req.email.strip().lower()
    auth_backend = "Supabase" if USE_SUPABASE else "SQLite"
    print(f"[LOGIN] Attempt: {clean_email} | Backend: {auth_backend} | "
          f"Serverless: {IS_SERVERLESS} | Supabase key valid: {_supabase_key_valid}")

    users = await db_query("users", "GET", filters=f"?email=eq.{clean_email}")

    if not users:
        print(f"[LOGIN] FAILED — user not found: {clean_email} (backend: {auth_backend})")
        raise HTTPException(status_code=401, detail="Invalid email or password")

    user = users[0]

    if not user.get("is_active", 1):
        print(f"[LOGIN] FAILED — account deactivated: {clean_email}")
        raise HTTPException(status_code=403, detail="Account deactivated")

    if not verify_password(req.password, user["password_hash"]):
        print(f"[LOGIN] FAILED — wrong password: {clean_email} (backend: {auth_backend})")
        raise HTTPException(status_code=401, detail="Invalid email or password")

    access_token = create_token(user["id"], user["email"], user["role"])
    print(f"[LOGIN] SUCCESS: {clean_email} | Role: {user['role']} | Backend: {auth_backend}")

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "_auth_backend": auth_backend,   # visible in Vercel function logs
        "user": {
            "id":        user["id"],
            "full_name": user["full_name"],
            "email":     user["email"],
            "role":      user["role"],
        },
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
        "location_description": req.location_description,
        "latitude":             req.latitude,
        "longitude":            req.longitude,
        "detected_persons":     req.detected_persons or [],
    })
    return {"message": "Incident reported", "incident": incident[0]}


@router.get("/incidents")
@router.get("/incidents/")
async def list_incidents(token: dict = Depends(verify_token)):
    incidents = await db_query("incidents", "GET",
                               filters="?order=created_at.desc&limit=100")
    return {"incidents": incidents, "total": len(incidents)}


@router.patch("/incidents/{incident_id}/review")
async def review_incident(incident_id: str, req: IncidentReview,
                          token: dict = Depends(verify_token)):
    updated = await db_query("incidents", "PATCH",
                             data={"status": req.status, "notes": req.notes,
                                   "reviewed_by": token["email"],
                                   "reviewed_at": datetime.utcnow().isoformat()},
                             filters=f"?id=eq.{incident_id}")
    if not updated:
        raise HTTPException(status_code=404, detail="Incident not found")
    return {"message": "Incident updated", "incident": updated[0]}


@router.get("/cameras")
@router.get("/cameras/")
async def list_cameras(token: dict = Depends(verify_token)):
    cameras = await db_query("cameras", "GET", filters="?is_active=eq.1")
    return {"cameras": cameras, "total": len(cameras)}


@router.post("/cameras")
async def add_camera(
    name: str, location: str,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    token: dict = Depends(require_admin),
):
    camera = await db_query("cameras", "POST", {
        "name": sanitize_string(name, 100),
        "location": sanitize_string(location, 200),
        "latitude": latitude,
        "longitude": longitude,
        "is_active": 1,
        "added_by": token.get("email"),
    })
    return {"message": "Camera added", "camera": camera[0]}


@router.get("/alerts")
async def list_alerts(token: dict = Depends(verify_token)):
    alerts = await db_query("alert_logs", "GET",
                            filters="?order=created_at.desc&limit=50")
    return {"alerts": alerts, "total": len(alerts)}


@router.post("/alerts")
async def send_alert(req: AlertRequest, token: dict = Depends(verify_token)):
    alert = await db_query("alert_logs", "POST", {
        "incident_id": req.incident_id,
        "alert_type":  req.alert_type,
        "channel":     req.channel,
        "recipient":   req.recipient,
        "message":     req.message,
        "status":      "sent",
        "sent_by":     token.get("email"),
    })
    return {"message": "Alert logged", "alert": alert[0]}


@router.get("/evidence/{incident_id}")
async def get_evidence(incident_id: str, token: dict = Depends(verify_token)):
    evidence = await db_query("evidence", "GET",
                              filters=f"?incident_id=eq.{incident_id}")
    return {"evidence": evidence}


@router.post("/evidence")
async def add_evidence(req: EvidenceRecord, token: dict = Depends(verify_token)):
    ev = await db_query("evidence", "POST", {
        "incident_id":      req.incident_id,
        "evidence_type":    req.evidence_type,
        "file_url":         req.file_url,
        "thumbnail_url":    req.thumbnail_url,
        "duration_seconds": req.duration_seconds,
    })
    return {"message": "Evidence added", "evidence": ev[0]}


@router.get("/stats")
async def get_stats(token: dict = Depends(verify_token)):
    all_incidents = await db_query("incidents", "GET", filters="?order=created_at.desc")
    total = len(all_incidents)
    by_status = {}
    by_type = {}
    by_severity = {}
    for inc in all_incidents:
        by_status[inc.get("status", "unknown")] = by_status.get(inc.get("status", "unknown"), 0) + 1
        by_type[inc.get("incident_type", "unknown")] = by_type.get(inc.get("incident_type", "unknown"), 0) + 1
        by_severity[inc.get("severity", "unknown")] = by_severity.get(inc.get("severity", "unknown"), 0) + 1
    return {
        "total_incidents": total,
        "by_status": by_status,
        "by_type": by_type,
        "by_severity": by_severity,
        "recent": all_incidents[:5],
    }


@router.post("/simulate/{incident_type}")
async def simulate_incident(incident_type: str,
                            token: dict = Depends(require_admin)):
    type_map = {
        "weapon":  ("WEAPON_DETECTED",   "critical"),
        "fight":   ("FIGHT_DETECTED",    "high"),
        "child":   ("CHILD_IN_DANGER",   "critical"),
        "fall":    ("PERSON_LYING_DOWN", "high"),
        "vehicle": ("SPEEDING_VEHICLE",  "medium"),
    }
    inc_type, sev = type_map.get(
        incident_type.lower(),
        (incident_type.upper(), "medium")
    )
    incident = await db_query("incidents", "POST", {
        "incident_type":        inc_type,
        "severity":             sev,
        "status":               "pending",
        "location_description": "Simulation — Test Zone Alpha",
        "latitude":             31.5204,
        "longitude":            74.3587,
        "detected_persons":     ["Simulated Person"],
    })
    return {"message": f"Simulated {inc_type} generated!", "incident": incident[0]}


# ── Mount router under /api prefix (matches Vercel route /api/:path*) ─────
# Also mount at root for local uvicorn and Docker
app.include_router(router, prefix="/api")
app.include_router(router)

# ── Serve frontend HTML (local dev / Docker only) ──────────
@app.get("/app", response_class=HTMLResponse)
@app.get("/index.html", response_class=HTMLResponse)
async def serve_frontend_app():
    for candidate in [
        BASE_DIR / "public" / "index.html",
        BASE_DIR / "index.html",
    ]:
        if candidate.exists():
            return HTMLResponse(content=candidate.read_text(encoding="utf-8"), status_code=200)
    return HTMLResponse("<h1>AI Safety Monitoring System — UI not found</h1>", status_code=404)
