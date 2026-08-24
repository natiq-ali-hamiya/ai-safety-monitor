"""
cloud_reporter.py
Sends incidents from the local detection device to the cloud backend.
Runs in a background thread so it never slows down the camera loop.
"""

import threading
import queue
import requests
import logging
from datetime import datetime

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
API_URL  = "http://127.0.0.1:8000"   # change to your deployed URL later
TOKEN    = None                        # set after login
_queue   = queue.Queue(maxsize=100)
_running = False
_thread  = None

# ── Severity map ──────────────────────────────────────────────────────────────
SEVERITY_MAP = {
    "WEAPON":             "critical",
    "FIGHT":              "critical",
    "SNATCHING":          "critical",
    "CHILD_IN_DANGER":    "critical",
    "ACCIDENT":           "high",
    "LYING_DOWN":         "high",
    "PERSON_LYING_DOWN":  "high",
    "PERSON_IN_ZONE":     "medium",
    "ZONE_BREACH":        "medium",
    "LICENSE_PLATE":      "low",
}

def _get_severity(incident_type: str) -> str:
    inc_upper = incident_type.upper()
    for key, sev in SEVERITY_MAP.items():
        if key in inc_upper:
            return sev
    return "medium"

# ── Auth ──────────────────────────────────────────────────────────────────────
def login(email: str, password: str) -> bool:
    """Login to the cloud backend and store the JWT token."""
    global TOKEN
    try:
        res = requests.post(f"{API_URL}/auth/login",
                            json={"email": email, "password": password},
                            timeout=10)
        if res.status_code == 200:
            TOKEN = res.json()["access_token"]
            log.info("[Cloud] Logged in successfully.")
            return True
        else:
            log.warning(f"[Cloud] Login failed: {res.text}")
            return False
    except Exception as e:
        log.warning(f"[Cloud] Login error: {e}")
        return False

# ── Queue an incident ─────────────────────────────────────────────────────────
def report(incident_type: str, confidence: float = 0.9,
           location: str = None, persons: list = None,
           latitude: float = None, longitude: float = None):
    """
    Call this from main_v3.py whenever an incident is detected.
    Non-blocking — queues the incident and returns immediately.
    """
    if not TOKEN:
        return  # not logged in, skip silently

    payload = {
        "incident_type": incident_type,
        "severity":      _get_severity(incident_type),
        "location_description": location or "Camera Feed",
        "detected_persons": persons or [],
        "latitude":  latitude,
        "longitude": longitude,
    }
    try:
        _queue.put_nowait(payload)
    except queue.Full:
        log.warning("[Cloud] Queue full — incident dropped.")

# ── Background sender thread ──────────────────────────────────────────────────
def _sender_loop():
    """Runs in background, drains the queue and POSTs to the API."""
    headers = {"Authorization": f"Bearer {TOKEN}",
               "Content-Type": "application/json"}
    while _running:
        try:
            payload = _queue.get(timeout=2)
            try:
                res = requests.post(f"{API_URL}/incidents",
                                    json=payload,
                                    headers={"Authorization": f"Bearer {TOKEN}",
                                             "Content-Type": "application/json"},
                                    timeout=8)
                if res.status_code == 200:
                    log.info(f"[Cloud] ✓ Incident sent: {payload['incident_type']}")
                else:
                    log.warning(f"[Cloud] ✗ Failed to send: {res.status_code} {res.text}")
            except requests.exceptions.ConnectionError:
                log.warning("[Cloud] Backend offline — incident not sent.")
            except Exception as e:
                log.warning(f"[Cloud] Send error: {e}")
            finally:
                _queue.task_done()
        except queue.Empty:
            continue

def start(email: str = "admin@aisafety.pk", password: str = "secret"):
    """Start the background sender. Call once at app startup."""
    global _running, _thread
    if not login(email, password):
        log.warning("[Cloud] Could not connect to backend — running offline.")
        return
    _running = True
    _thread  = threading.Thread(target=_sender_loop, daemon=True)
    _thread.start()
    log.info("[Cloud] Reporter started.")

def stop():
    """Stop the background sender. Call on app shutdown."""
    global _running
    _running = False
    if _thread:
        _thread.join(timeout=5)
    log.info("[Cloud] Reporter stopped.")
