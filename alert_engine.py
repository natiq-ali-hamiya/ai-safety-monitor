"""
alert_engine.py
Complete alert system for AI Safety Monitoring System.
Handles all 4 modules:
  1. Crime Detection  → Police alert with evidence
  2. Child Safety     → Parents + Police alert
  3. Hit & Run        → Police alert with plate + vehicle data
  4. Health Emergency → 1122 ambulance alert
"""

import smtplib
import threading
import queue
import logging
import os
import requests
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

# ── Email Config ──────────────────────────────────────────
SENDER_EMAIL    = os.getenv("SENDER_EMAIL",    "teko2847@gmail.com")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD", "pvgy zvcq mhwg wjki")
SMTP_SERVER     = "smtp.gmail.com"
SMTP_PORT       = 587

# ── Alert Recipients ──────────────────────────────────────
POLICE_EMAIL    = os.getenv("POLICE_EMAIL",    "natiqalihamiya84@gmail.com")
FAMILY_EMAIL    = os.getenv("FAMILY_EMAIL",    "natiqalihamiya84@gmail.com")
RESCUE_EMAIL    = os.getenv("RESCUE_EMAIL",    "natiqalihamiya84@gmail.com")

# ── Twilio Config (optional) ──────────────────────────────
TWILIO_ENABLED     = os.getenv("TWILIO_ENABLED", "false").lower() == "true"
TWILIO_SID         = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_TOKEN       = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM        = os.getenv("TWILIO_FROM_NUMBER", "")
POLICE_PHONE       = os.getenv("POLICE_PHONE",  "+923001234567")
FAMILY_PHONE       = os.getenv("FAMILY_PHONE",  "+923001234567")
RESCUE_PHONE       = os.getenv("RESCUE_PHONE",  "+923001234567")

# ── Alert Queue ───────────────────────────────────────────
_queue   = queue.Queue(maxsize=50)
_running = False
_thread  = None


def start():
    global _running, _thread
    _running = True
    _thread  = threading.Thread(target=_alert_loop, daemon=True)
    _thread.start()
    log.info("[AlertEngine] Started.")


def stop():
    global _running
    _running = False
    if _thread:
        _thread.join(timeout=5)


# ── Alert thresholds — prevents wrong/spam alerts ─────────
MIN_CONFIDENCE_FOR_ALERT = 0.50   # only send email if confidence >= 50%
_last_alert_time: dict = {}       # cooldown tracker per alert type
ALERT_COOLDOWN_SECS = 30          # minimum seconds between same alert type


def _cooldown_ok(alert_type: str) -> bool:
    """Returns True if enough time has passed since last alert of this type."""
    import time
    now = time.time()
    last = _last_alert_time.get(alert_type, 0)
    if now - last >= ALERT_COOLDOWN_SECS:
        _last_alert_time[alert_type] = now
        return True
    return False


def _alert_loop():
    while _running:
        try:
            job = _queue.get(timeout=2)
            _process_alert(job)
            _queue.task_done()
        except queue.Empty:
            continue


def _process_alert(job: dict):
    alert_type = job.get("alert_type")

    if alert_type == "crime":
        _send_crime_alert(job)
    elif alert_type == "child":
        _send_child_alert(job)
    elif alert_type == "hitrun":
        _send_hitrun_alert(job)
    elif alert_type == "health":
        _send_health_alert(job)


# ── MODULE 1: Crime Alert → Police ────────────────────────
def report_crime(incident_type: str, criminal_info: dict,
                 victim_info: dict, evidence_url: str,
                 location: str, snapshot_path: str = None):
    """
    Call this when a crime is detected.
    Only sends alert if confidence >= MIN_CONFIDENCE_FOR_ALERT
    and cooldown period has passed.
    """
    confidence = criminal_info.get("confidence", 0)

    # Skip low confidence detections — reduces wrong alerts
    if confidence < MIN_CONFIDENCE_FOR_ALERT:
        log.info(f"[AlertEngine] Crime alert skipped — confidence too low: {confidence:.0%}")
        return

    # Skip if same alert type sent recently — reduces spam
    if not _cooldown_ok(f"crime_{incident_type}"):
        log.info(f"[AlertEngine] Crime alert skipped — cooldown active for {incident_type}")
        return

    try:
        _queue.put_nowait({
            "alert_type":    "crime",
            "incident_type": incident_type,
            "criminal_info": criminal_info,
            "victim_info":   victim_info,
            "evidence_url":  evidence_url,
            "location":      location,
            "snapshot_path": snapshot_path,
            "timestamp":     datetime.now().strftime("%d %b %Y, %I:%M %p"),
        })
    except queue.Full:
        log.warning("[AlertEngine] Queue full — crime alert dropped.")


def _send_crime_alert(job: dict):
    criminal  = job.get("criminal_info", {})
    victim    = job.get("victim_info", {})
    inc_type  = job.get("incident_type", "CRIME")
    location  = job.get("location", "Unknown Location")
    evidence  = job.get("evidence_url", "No evidence URL")
    timestamp = job.get("timestamp")

    criminal_name = criminal.get("name", "Unknown Person")
    criminal_known = criminal.get("known", False)
    confidence = criminal.get("confidence", 0)

    subject = f"🚨 CRIME ALERT: {inc_type} at {location}"

    body = f"""
AI SAFETY MONITORING SYSTEM — CRIME ALERT
==========================================
Time:           {timestamp}
Location:       {location}
Incident Type:  {inc_type}

CRIMINAL INFORMATION:
---------------------
Name:       {criminal_name}
Status:     {"IDENTIFIED — Known Person" if criminal_known else "UNKNOWN PERSON"}
Confidence: {confidence*100:.0f}%

VICTIM INFORMATION:
-------------------
Name: {victim.get("name", "Unknown") if victim else "Unknown"}

EVIDENCE:
---------
Video Clip: {evidence}

ACTION REQUIRED:
----------------
Please review the evidence and take immediate action.
This alert was generated automatically by the AI Safety Monitor.
    """

    # Send email to police
    _send_email(
        to=POLICE_EMAIL,
        subject=subject,
        body=body,
        snapshot_path=job.get("snapshot_path")
    )

    # Send SMS if Twilio enabled
    if TWILIO_ENABLED:
        sms = f"CRIME ALERT: {inc_type} at {location}. Suspect: {criminal_name}. Evidence: {evidence}"
        _send_sms(POLICE_PHONE, sms)

    log.info(f"[AlertEngine] Crime alert sent for {inc_type}")


# ── MODULE 2: Child Safety Alert → Parents + Police ───────
def report_child_danger(child_info: dict, zone_name: str,
                         location: str, evidence_url: str,
                         snapshot_path: str = None):
    if not _cooldown_ok("child_danger"):
        return
    try:
        _queue.put_nowait({
            "alert_type":  "child",
            "child_info":  child_info,
            "zone_name":   zone_name,
            "location":    location,
            "evidence_url": evidence_url,
            "snapshot_path": snapshot_path,
            "timestamp":   datetime.now().strftime("%d %b %Y, %I:%M %p"),
        })
    except queue.Full:
        log.warning("[AlertEngine] Queue full — child alert dropped.")


def _send_child_alert(job: dict):
    child     = job.get("child_info", {})
    zone      = job.get("zone_name", "Danger Zone")
    location  = job.get("location", "Unknown")
    evidence  = job.get("evidence_url", "No evidence URL")
    timestamp = job.get("timestamp")

    child_name = child.get("name", "Unknown Child")
    child_age  = child.get("age", "Unknown")
    known      = child.get("known", False)

    # Alert to parents
    parent_subject = f"⚠️ CHILD SAFETY ALERT — {child_name} in Danger Zone"
    parent_body = f"""
AI SAFETY MONITORING SYSTEM — CHILD SAFETY ALERT
=================================================
Time:     {timestamp}
Location: {location}
Zone:     {zone}

YOUR CHILD HAS BEEN DETECTED IN A DANGER ZONE.

Child Name: {child_name}
Age:        {child_age}
Status:     {"Identified" if known else "Unidentified child"}

Please go to the location immediately or contact authorities.

Evidence: {evidence}

This is an automated alert from the AI Safety Monitor.
    """

    _send_email(
        to=FAMILY_EMAIL,
        subject=parent_subject,
        body=parent_body,
        snapshot_path=job.get("snapshot_path")
    )

    # Alert to police
    police_subject = f"🚨 CHILD IN DANGER ZONE — {location}"
    police_body = f"""
AI SAFETY MONITORING SYSTEM — CHILD SAFETY ALERT
=================================================
Time:      {timestamp}
Location:  {location}
Zone:      {zone}
Child:     {child_name} (Age: {child_age})
Evidence:  {evidence}

Please send assistance to the location immediately.
    """

    _send_email(
        to=POLICE_EMAIL,
        subject=police_subject,
        body=police_body,
        snapshot_path=job.get("snapshot_path")
    )

    # WhatsApp to parents if Twilio enabled
    if TWILIO_ENABLED:
        msg = (f"CHILD SAFETY ALERT: {child_name} detected in danger zone "
               f"at {location}. Please go immediately. Evidence: {evidence}")
        _send_whatsapp(FAMILY_PHONE, msg)

    log.info(f"[AlertEngine] Child safety alert sent for {child_name}")


# ── MODULE 3: Hit & Run Alert → Police ────────────────────
def report_hit_and_run(plate_number: str, vehicle_type: str,
                        owner_info: dict, location: str,
                        evidence_url: str, snapshot_path: str = None):
    """
    Call when a hit and run vehicle is detected.
    owner_info: {"name": str, "cnic": str, "address": str} or {}
    """
    try:
        _queue.put_nowait({
            "alert_type":   "hitrun",
            "plate_number": plate_number,
            "vehicle_type": vehicle_type,
            "owner_info":   owner_info,
            "location":     location,
            "evidence_url": evidence_url,
            "snapshot_path": snapshot_path,
            "timestamp":    datetime.now().strftime("%d %b %Y, %I:%M %p"),
        })
    except queue.Full:
        log.warning("[AlertEngine] Queue full — hit & run alert dropped.")


def _send_hitrun_alert(job: dict):
    plate     = job.get("plate_number", "Unknown")
    v_type    = job.get("vehicle_type", "Vehicle")
    owner     = job.get("owner_info", {})
    location  = job.get("location", "Unknown")
    evidence  = job.get("evidence_url", "No evidence URL")
    timestamp = job.get("timestamp")

    owner_name = owner.get("name", "Unknown Owner") if owner else "Unknown Owner"
    owner_cnic = owner.get("cnic", "Not in database") if owner else "Not in database"

    subject = f"🚨 HIT & RUN ALERT — Plate: {plate} at {location}"
    body = f"""
AI SAFETY MONITORING SYSTEM — HIT & RUN ALERT
==============================================
Time:          {timestamp}
Location:      {location}
Incident:      Hit & Run — Vehicle fled the scene

VEHICLE INFORMATION:
--------------------
License Plate: {plate}
Vehicle Type:  {v_type}

OWNER INFORMATION:
------------------
Name:  {owner_name}
CNIC:  {owner_cnic}
Note:  {"Owner identified from database" if owner else "Owner NOT in database — plate traced only"}

EVIDENCE:
---------
Video Clip: {evidence}

ACTION REQUIRED:
----------------
Trace the vehicle using the license plate number.
Evidence clip attached for identification.
    """

    _send_email(
        to=POLICE_EMAIL,
        subject=subject,
        body=body,
        snapshot_path=job.get("snapshot_path")
    )

    if TWILIO_ENABLED:
        sms = (f"HIT & RUN ALERT: Plate {plate} ({v_type}) at {location}. "
               f"Owner: {owner_name}. Evidence: {evidence}")
        _send_sms(POLICE_PHONE, sms)

    log.info(f"[AlertEngine] Hit & run alert sent for plate {plate}")


# ── MODULE 4: Health Emergency Alert → 1122 ───────────────
def report_health_emergency(person_info: dict, location: str,
                              seconds_down: int, evidence_url: str,
                              snapshot_path: str = None):
    if not _cooldown_ok("health_emergency"):
        return
    try:
        _queue.put_nowait({
            "alert_type":   "health",
            "person_info":  person_info,
            "location":     location,
            "seconds_down": seconds_down,
            "evidence_url": evidence_url,
            "snapshot_path": snapshot_path,
            "timestamp":    datetime.now().strftime("%d %b %Y, %I:%M %p"),
        })
    except queue.Full:
        log.warning("[AlertEngine] Queue full — health alert dropped.")


def _send_health_alert(job: dict):
    person       = job.get("person_info", {})
    location     = job.get("location", "Unknown")
    seconds_down = job.get("seconds_down", 0)
    evidence     = job.get("evidence_url", "No evidence URL")
    timestamp    = job.get("timestamp")

    person_name = person.get("name", "Unknown Person") if person else "Unknown Person"
    minutes     = seconds_down // 60
    seconds     = seconds_down % 60

    subject = f"🚑 MEDICAL EMERGENCY — Person Unconscious at {location}"
    body = f"""
AI SAFETY MONITORING SYSTEM — MEDICAL EMERGENCY
================================================
Time:        {timestamp}
Location:    {location}
Emergency:   Person found unconscious / not moving

PERSON INFORMATION:
-------------------
Name:         {person_name}
Down for:     {minutes} min {seconds} sec
Condition:    Unconscious / Not responding

IMMEDIATE ACTION REQUIRED:
---------------------------
Please send an ambulance to the location immediately.
The person has been motionless and may need urgent medical attention.

Evidence: {evidence}

This is an automated alert from the AI Safety Monitor.
    """

    # Send to rescue/ambulance
    _send_email(
        to=RESCUE_EMAIL,
        subject=subject,
        body=body,
        snapshot_path=job.get("snapshot_path")
    )

    # Also notify police
    _send_email(
        to=POLICE_EMAIL,
        subject=subject,
        body=body,
        snapshot_path=job.get("snapshot_path")
    )

    if TWILIO_ENABLED:
        sms = (f"MEDICAL EMERGENCY: Person unconscious at {location} "
               f"for {minutes}m {seconds}s. Send ambulance immediately.")
        _send_sms(RESCUE_PHONE, sms)
        _send_sms(POLICE_PHONE, sms)

    log.info(f"[AlertEngine] Health emergency alert sent for {person_name}")


# ── Email sender ──────────────────────────────────────────
def _send_email(to: str, subject: str, body: str,
                snapshot_path: str = None):
    try:
        msg = MIMEMultipart()
        msg["From"]    = SENDER_EMAIL
        msg["To"]      = to
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        # Attach snapshot if available
        if snapshot_path and os.path.exists(snapshot_path):
            with open(snapshot_path, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header(
                    "Content-Disposition",
                    f"attachment; filename=evidence.jpg"
                )
                msg.attach(part)

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)

        log.info(f"[AlertEngine] Email sent to {to}")

    except Exception as e:
        log.error(f"[AlertEngine] Email failed to {to}: {e}")


# ── SMS sender (Twilio) ───────────────────────────────────
def _send_sms(to: str, message: str):
    if not TWILIO_ENABLED:
        return
    try:
        from twilio.rest import Client
        client = Client(TWILIO_SID, TWILIO_TOKEN)
        client.messages.create(body=message, from_=TWILIO_FROM, to=to)
        log.info(f"[AlertEngine] SMS sent to {to}")
    except Exception as e:
        log.error(f"[AlertEngine] SMS failed to {to}: {e}")


# ── WhatsApp sender (Twilio) ──────────────────────────────
def _send_whatsapp(to: str, message: str):
    if not TWILIO_ENABLED:
        return
    try:
        from twilio.rest import Client
        client = Client(TWILIO_SID, TWILIO_TOKEN)
        client.messages.create(
            body=message,
            from_=f"whatsapp:{TWILIO_FROM}",
            to=f"whatsapp:{to}"
        )
        log.info(f"[AlertEngine] WhatsApp sent to {to}")
    except Exception as e:
        log.error(f"[AlertEngine] WhatsApp failed to {to}: {e}")