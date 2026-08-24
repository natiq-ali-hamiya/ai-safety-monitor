"""
AI Safety Monitoring System — FastAPI Backend
Phase 6: Full Security Hardening Applied
"""

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, field_validator
from typing import Optional
import httpx
import bcrypt
import jwt
import re
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from collections import defaultdict
import time

load_dotenv()

# ── Config ────────────────────────────────────────────────
SUPABASE_URL     = os.getenv("SUPABASE_URL", "https://lgsvbbzaocprdingtgtc.supabase.co")
SUPABASE_KEY     = os.getenv("SUPABASE_KEY")
JWT_SECRET       = os.getenv("JWT_SECRET", "change-this-in-production")
JWT_EXPIRE_HOURS = 24

# ── Rate limiter ──────────────────────────────────────────
_login_attempts: dict = defaultdict(list)
MAX_ATTEMPTS  = 5
WINDOW_SECS   = 300

def check_rate_limit(ip: str):
    now = time.time()
    _login_attempts[ip] = [t for t in _login_attempts[ip] if now - t < WINDOW_SECS]
    if len(_login_attempts[ip]) >= MAX_ATTEMPTS:
        raise HTTPException(status_code=429,
                            detail="Too many attempts. Try again in 5 minutes.")
    _login_attempts[ip].append(now)

# ── App ───────────────────────────────────────────────────
app = FastAPI(
    title="AI Safety Monitoring System",
    description="AI-powered safety monitoring API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "http://127.0.0.1:5500",
        "null",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

security = HTTPBearer()

# ── Supabase helper ───────────────────────────────────────
async def supabase_request(method: str, table: str,
                           data: dict = None, filters: str = "") -> dict:
    url = f"{SUPABASE_URL}/rest/v1/{table}{filters}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }
    async with httpx.AsyncClient() as client:
        if method == "GET":
            r = await client.get(url, headers=headers)
        elif method == "POST":
            r = await client.post(url, headers=headers, json=data)
        elif method == "PATCH":
            r = await client.patch(url, headers=headers, json=data)
        elif method == "DELETE":
            r = await client.delete(url, headers=headers)
        r.raise_for_status()
        return r.json()

# ── JWT helpers ───────────────────────────────────────────
def create_token(user_id: str, email: str, role: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "role": role,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRE_HOURS)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired — please log in again")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def require_admin(token: dict = Depends(verify_token)):
    if token.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return token

# ── Input validation helpers ──────────────────────────────
def sanitize_string(value: str, max_length: int = 200) -> str:
    """Remove dangerous characters and limit length."""
    if not value:
        return value
    # Remove SQL injection patterns
    value = re.sub(r"['\";\\]", "", value)
    # Limit length
    return value[:max_length].strip()

def validate_uuid(value: str) -> bool:
    """Check if string is a valid UUID."""
    pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    return bool(re.match(pattern, value.lower())) if value else False

# ── Request models with validation ───────────────────────
class RegisterRequest(BaseModel):
    full_name: str
    email: EmailStr
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
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v

    @field_validator("role")
    @classmethod
    def validate_role(cls, v):
        allowed = ["admin", "operator", "viewer"]
        if v not in allowed:
            raise ValueError(f"Role must be one of: {allowed}")
        return v

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

    @field_validator("password")
    @classmethod
    def validate_password(cls, v):
        if len(v) > 200:
            raise ValueError("Password too long")
        return v

class IncidentReport(BaseModel):
    camera_id: Optional[str] = None
    incident_type: str
    severity: str = "medium"
    location_description: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    detected_persons: Optional[list] = []

    @field_validator("incident_type")
    @classmethod
    def validate_type(cls, v):
        return sanitize_string(v, 100)

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v):
        allowed = ["low", "medium", "high", "critical"]
        if v not in allowed:
            return "medium"
        return v

    @field_validator("latitude", "longitude")
    @classmethod
    def validate_coords(cls, v):
        if v is not None and (v < -180 or v > 180):
            raise ValueError("Invalid coordinate")
        return v

class IncidentReview(BaseModel):
    status: str
    notes: Optional[str] = None

    @field_validator("status")
    @classmethod
    def validate_status(cls, v):
        allowed = ["confirmed", "false_alarm", "resolved"]
        if v not in allowed:
            raise ValueError(f"Status must be one of: {allowed}")
        return v

    @field_validator("notes")
    @classmethod
    def validate_notes(cls, v):
        if v:
            return sanitize_string(v, 500)
        return v

class AlertRequest(BaseModel):
    incident_id: str
    alert_type: str
    channel: str
    recipient: str
    message: str

    @field_validator("alert_type")
    @classmethod
    def validate_alert_type(cls, v):
        allowed = ["police", "ambulance", "parent", "admin"]
        if v not in allowed:
            raise ValueError(f"Alert type must be one of: {allowed}")
        return v

    @field_validator("channel")
    @classmethod
    def validate_channel(cls, v):
        allowed = ["sms", "whatsapp", "email"]
        if v not in allowed:
            raise ValueError(f"Channel must be one of: {allowed}")
        return v

    @field_validator("message")
    @classmethod
    def validate_message(cls, v):
        return sanitize_string(v, 1000)

class EvidenceRecord(BaseModel):
    incident_id: str
    evidence_type: str = "video_clip"
    file_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    duration_seconds: Optional[int] = None

    @field_validator("evidence_type")
    @classmethod
    def validate_evidence_type(cls, v):
        allowed = ["video_clip", "snapshot", "identity_match"]
        if v not in allowed:
            return "video_clip"
        return v

# ── Routes ────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "system": "AI Safety Monitoring System",
        "version": "1.0.0",
        "status": "online"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

# ── AUTH ──────────────────────────────────────────────────

@app.post("/auth/register")
async def register(req: RegisterRequest, token: dict = Depends(require_admin)):
    existing = await supabase_request("GET", "users",
                                      filters=f"?email=eq.{req.email}")
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    pw_hash = bcrypt.hashpw(req.password.encode(), bcrypt.gensalt()).decode()
    user = await supabase_request("POST", "users", {
        "full_name":     req.full_name,
        "email":         req.email,
        "password_hash": pw_hash,
        "role":          req.role
    })
    u = user[0]
    return {"message": "User created", "user": {
        "id": u["id"], "email": u["email"], "role": u["role"]}}


@app.post("/auth/login")
async def login(req: LoginRequest, request: Request):
    # Rate limit by IP
    client_ip = request.client.host
    check_rate_limit(client_ip)

    users = await supabase_request("GET", "users",
                                   filters=f"?email=eq.{req.email}")
    if not users:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    user = users[0]
    if not user.get("is_active"):
        raise HTTPException(status_code=403, detail="Account deactivated")

    if not bcrypt.checkpw(req.password.encode(), user["password_hash"].encode()):
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


@app.get("/auth/me")
async def get_current_user(token: dict = Depends(verify_token)):
    users = await supabase_request("GET", "users",
                                   filters=f"?id=eq.{token['sub']}")
    if not users:
        raise HTTPException(status_code=404, detail="User not found")
    u = users[0]
    return {"id": u["id"], "full_name": u["full_name"],
            "email": u["email"], "role": u["role"]}


# ── INCIDENTS ─────────────────────────────────────────────

@app.post("/incidents")
async def report_incident(req: IncidentReport,
                          token: dict = Depends(verify_token)):
    incident = await supabase_request("POST", "incidents", {
        "camera_id":           req.camera_id,
        "incident_type":       req.incident_type,
        "severity":            req.severity,
        "status":              "pending",
        "location_description": req.location_description,
        "latitude":            req.latitude,
        "longitude":           req.longitude,
        "detected_persons":    req.detected_persons
    })
    return {"message": "Incident recorded", "incident": incident[0]}


@app.get("/incidents")
async def list_incidents(status: Optional[str] = None,
                         token: dict = Depends(verify_token)):
    filters = "?order=created_at.desc&limit=50"
    if status:
        allowed = ["pending", "reviewed", "confirmed", "false_alarm", "resolved"]
        if status not in allowed:
            raise HTTPException(status_code=400, detail="Invalid status filter")
        filters += f"&status=eq.{status}"
    return await supabase_request("GET", "incidents", filters=filters)


@app.get("/incidents/{incident_id}")
async def get_incident(incident_id: str,
                       token: dict = Depends(verify_token)):
    if not validate_uuid(incident_id):
        raise HTTPException(status_code=400, detail="Invalid incident ID")
    result = await supabase_request("GET", "incidents",
                                    filters=f"?id=eq.{incident_id}")
    if not result:
        raise HTTPException(status_code=404, detail="Incident not found")
    return result[0]


@app.patch("/incidents/{incident_id}/review")
async def review_incident(incident_id: str, req: IncidentReview,
                          token: dict = Depends(verify_token)):
    if not validate_uuid(incident_id):
        raise HTTPException(status_code=400, detail="Invalid incident ID")
    updated = await supabase_request("PATCH", "incidents", {
        "status":      req.status,
        "notes":       req.notes,
        "reviewed_by": token["sub"],
        "reviewed_at": datetime.utcnow().isoformat()
    }, filters=f"?id=eq.{incident_id}")
    return {"message": f"Incident marked as {req.status}", "incident": updated[0]}


# ── ALERTS ────────────────────────────────────────────────

@app.post("/alerts/send")
async def send_alert(req: AlertRequest,
                     token: dict = Depends(verify_token)):
    if not validate_uuid(req.incident_id):
        raise HTTPException(status_code=400, detail="Invalid incident ID")

    incidents = await supabase_request("GET", "incidents",
                                       filters=f"?id=eq.{req.incident_id}")
    if not incidents:
        raise HTTPException(status_code=404, detail="Incident not found")

    if incidents[0]["status"] != "confirmed":
        raise HTTPException(status_code=400,
                            detail="Incident must be confirmed before sending alert")

    log = await supabase_request("POST", "alert_logs", {
        "incident_id": req.incident_id,
        "alert_type":  req.alert_type,
        "channel":     req.channel,
        "recipient":   req.recipient,
        "message":     req.message,
        "status":      "sent",
        "sent_by":     token["sub"]
    })
    return {"message": f"Alert sent via {req.channel}", "alert_log": log[0]}


@app.get("/alerts/{incident_id}")
async def get_alert_logs(incident_id: str,
                         token: dict = Depends(verify_token)):
    if not validate_uuid(incident_id):
        raise HTTPException(status_code=400, detail="Invalid incident ID")
    return await supabase_request("GET", "alert_logs",
                                  filters=f"?incident_id=eq.{incident_id}")


# ── EVIDENCE ──────────────────────────────────────────────

@app.post("/evidence")
async def save_evidence(req: EvidenceRecord,
                        token: dict = Depends(verify_token)):
    result = await supabase_request("POST", "evidence", {
        "incident_id":    req.incident_id,
        "evidence_type":  req.evidence_type,
        "file_url":       req.file_url,
        "thumbnail_url":  req.thumbnail_url,
        "duration_seconds": req.duration_seconds,
    })
    return {"message": "Evidence saved", "evidence": result[0]}


@app.get("/evidence/{incident_id}")
async def get_evidence(incident_id: str,
                       token: dict = Depends(verify_token)):
    if not validate_uuid(incident_id):
        raise HTTPException(status_code=400, detail="Invalid incident ID")
    return await supabase_request("GET", "evidence",
                                  filters=f"?incident_id=eq.{incident_id}")


# ── CAMERAS ───────────────────────────────────────────────

@app.get("/cameras")
async def list_cameras(token: dict = Depends(verify_token)):
    return await supabase_request("GET", "cameras",
                                  filters="?is_active=eq.true")


@app.post("/cameras")
async def add_camera(camera: dict, token: dict = Depends(require_admin)):
    camera["added_by"] = token["sub"]
    # Sanitize input
    if "name" in camera:
        camera["name"] = sanitize_string(camera["name"], 100)
    if "location" in camera:
        camera["location"] = sanitize_string(camera["location"], 200)
    result = await supabase_request("POST", "cameras", camera)
    return {"message": "Camera registered", "camera": result[0]}


# ── DASHBOARD STATS ───────────────────────────────────────

@app.get("/dashboard/stats")
async def dashboard_stats(token: dict = Depends(verify_token)):
    all_incidents = await supabase_request("GET", "incidents",
                                           filters="?select=status")
    pending     = sum(1 for i in all_incidents if i["status"] == "pending")
    confirmed   = sum(1 for i in all_incidents if i["status"] == "confirmed")
    false_alarms = sum(1 for i in all_incidents if i["status"] == "false_alarm")
    resolved    = sum(1 for i in all_incidents if i["status"] == "resolved")
    return {
        "total_incidents": len(all_incidents),
        "pending_review":  pending,
        "confirmed":       confirmed,
        "false_alarms":    false_alarms,
        "resolved":        resolved
    }
