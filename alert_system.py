# ============================================================
#  Child Safety Monitoring System v2.0 — Alert System
# ============================================================
"""
Three-channel alert system:
  1. Visual  — flashing red banner on screen
  2. Sound   — system beep
  3. Email   — with snapshot photo attached
  4. WhatsApp — via Twilio API
  5. SMS      — via Twilio API

Three recipient groups:
  • Family  — child enters danger zone
  • Police  — any zone breach / suspicious activity
  • 1122    — person lying down emergency
"""

import time
import threading
import smtplib
import logging
import platform
import os
from email.mime.text        import MIMEText
from email.mime.multipart   import MIMEMultipart
from email.mime.image       import MIMEImage
from datetime               import datetime

import cv2
import numpy as np
import config

logger = logging.getLogger(__name__)

# Attempt Twilio import
try:
    from twilio.rest import Client as TwilioClient
    _TWILIO_AVAILABLE = True
except ImportError:
    _TWILIO_AVAILABLE = False


class AlertSystem:
    def __init__(self):
        self._last_sound_time  = 0.0
        self._last_email_times = {}   # recipient → last send time
        self._last_sms_times   = {}
        self._alert_active     = False
        self._flash_state      = False
        self._flash_counter    = 0
        self._twilio           = None

        logging.basicConfig(
            filename=config.LOG_FILE,
            level=logging.INFO,
            format="%(asctime)s  %(levelname)s  %(message)s",
        )

        if config.TWILIO_ENABLED and _TWILIO_AVAILABLE:
            try:
                self._twilio = TwilioClient(
                    config.TWILIO_ACCOUNT_SID,
                    config.TWILIO_AUTH_TOKEN
                )
                print("[Alert] Twilio connected — WhatsApp + SMS enabled.")
            except Exception as e:
                print(f"[Alert] Twilio init failed: {e}")

    # ----------------------------------------------------------
    # HIGH-LEVEL TRIGGERS
    # ----------------------------------------------------------

    def trigger_child_in_zone(self, zone_names: list[str],
                               snapshot_path: str = None,
                               age_info: dict = None):
        """Child (under 15) detected in danger zone → Family + Police."""
        self._alert_active = True
        now = time.time()
        age_label = age_info.get("label", "Child") if age_info else "Child"

        subject = f"🚨 CHILD SAFETY ALERT — Child in Danger Zone"
        body = f"""
CHILD SAFETY ALERT
==================
A child has been detected entering a restricted danger zone.

Timestamp  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Zone(s)    : {', '.join(zone_names)}
Age Info   : {age_label}
Priority   : HIGH — Immediate response required

Please check the area immediately.

-- Child Safety Monitoring System (Automated Alert)
"""
        self._fire_all_channels(
            recipients   = [config.FAMILY_EMAIL, config.POLICE_EMAIL],
            wa_recipients= [config.FAMILY_WHATSAPP, config.POLICE_WHATSAPP],
            sms_recipients=[config.FAMILY_PHONE, config.POLICE_PHONE],
            subject      = subject,
            body         = body,
            snapshot_path= snapshot_path,
            sound        = True
        )
        logger.info(f"CHILD ZONE ALERT — zones: {zone_names}, age: {age_label}")

    def trigger_person_in_zone(self, zone_names: list[str],
                                snapshot_path: str = None,
                                subject_override: str = None):
        """Adult or unknown person in danger zone → Police."""
        self._alert_active = True
        subject = subject_override or "⚠️ SAFETY ALERT — Person in Restricted Zone"
        body = f"""
SAFETY ALERT
============
A person has been detected in a restricted zone / incident detected.

Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Zone(s)   : {', '.join(zone_names)}
Priority  : HIGH

Please review the incident snapshot attached.

-- AI Safety Monitoring System (Automated Alert)
"""
        self._fire_all_channels(
            recipients   = [config.POLICE_EMAIL],
            wa_recipients= [config.POLICE_WHATSAPP],
            sms_recipients=[config.POLICE_PHONE],
            subject      = subject,
            body         = body,
            snapshot_path= snapshot_path,
            sound        = True
        )
        logger.info(f"PERSON ZONE ALERT — zones: {zone_names}")

    def trigger_lying_down(self, seconds_down: float,
                            snapshot_path: str = None,
                            age_info: dict = None,
                            subject_override: str = None):
        """Person lying down for too long → 1122 Rescue + Family."""
        self._alert_active = True
        age_label = age_info.get("label", "Unknown") if age_info else "Unknown"
        subject = subject_override or "🚑 EMERGENCY — Person Lying Down / Possible Injury"
        body = f"""
EMERGENCY MEDICAL ALERT
========================
A person has been detected lying motionless / an emergency incident was detected.

Timestamp    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Duration     : {seconds_down:.0f} seconds
Age Info     : {age_label}
Priority     : CRITICAL — Possible medical emergency

Please dispatch assistance immediately.
Emergency Services: 1122

-- AI Safety Monitoring System (Automated Alert)
"""
        self._fire_all_channels(
            recipients   = [config.RESCUE_EMAIL, config.FAMILY_EMAIL],
            wa_recipients= [config.RESCUE_WHATSAPP, config.FAMILY_WHATSAPP],
            sms_recipients=[config.RESCUE_PHONE, config.FAMILY_PHONE],
            subject      = subject,
            body         = body,
            snapshot_path= snapshot_path,
            sound        = True
        )
        logger.info(f"1122 LYING DOWN ALERT — {seconds_down:.0f}s, age: {age_label}")

    def clear(self):
        self._alert_active = False

    # ----------------------------------------------------------
    # CHANNEL DISPATCHER
    # ----------------------------------------------------------

    def _fire_all_channels(self, recipients, wa_recipients,
                           sms_recipients, subject, body,
                           snapshot_path=None, sound=True):
        now = time.time()

        # Sound
        if sound and now - self._last_sound_time >= config.ALERT_COOLDOWN_SECONDS:
            self._last_sound_time = now
            threading.Thread(target=self._play_beep, daemon=True).start()

        # Email
        for email in recipients:
            last = self._last_email_times.get(email, 0)
            if now - last >= config.EMAIL_COOLDOWN_SECS:
                self._last_email_times[email] = now
                threading.Thread(
                    target=self._send_email,
                    args=(email, subject, body, snapshot_path),
                    daemon=True
                ).start()

        # WhatsApp + SMS via Twilio
        if self._twilio and config.TWILIO_ENABLED:
            short_msg = f"{subject}\n{datetime.now().strftime('%H:%M:%S')}\n{body[:200]}"
            for wa in wa_recipients:
                last = self._last_sms_times.get(wa, 0)
                if now - last >= config.EMAIL_COOLDOWN_SECS:
                    self._last_sms_times[wa] = now
                    threading.Thread(
                        target=self._send_whatsapp,
                        args=(wa, short_msg),
                        daemon=True
                    ).start()
            for sms in sms_recipients:
                last = self._last_sms_times.get(sms, 0)
                if now - last >= config.EMAIL_COOLDOWN_SECS:
                    self._last_sms_times[sms] = now
                    threading.Thread(
                        target=self._send_sms,
                        args=(sms, short_msg),
                        daemon=True
                    ).start()

    # ----------------------------------------------------------
    # VISUAL BANNER
    # ----------------------------------------------------------

    def draw_alert_banner(self, frame: np.ndarray,
                          alert_type: str = "DANGER") -> np.ndarray:
        if not self._alert_active:
            return frame

        self._flash_counter += 1
        if self._flash_counter % 8 == 0:
            self._flash_state = not self._flash_state

        if self._flash_state:
            h, w     = frame.shape[:2]
            banner_h = 64
            color    = (0, 0, 200) if alert_type == "DANGER" else (0, 100, 200)
            overlay  = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, banner_h), color, -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

            messages = {
                "CHILD_ZONE":  "🚨  ALERT — CHILD IN DANGER ZONE  |  Family & Police Notified  🚨",
                "PERSON_ZONE": "⚠️   ALERT — PERSON IN RESTRICTED ZONE  |  Police Notified  ⚠️",
                "LYING":       "🚑  EMERGENCY — PERSON DOWN  |  1122 & Family Notified  🚑",
                "DANGER":      "⚠️   DANGER DETECTED  |  Alerts Sent  ⚠️",
            }
            text = messages.get(alert_type, messages["DANGER"])

            font  = cv2.FONT_HERSHEY_DUPLEX
            scale = 0.75
            thick = 2
            (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
            tx = max((w - tw) // 2, 8)
            ty = (banner_h + th) // 2

            cv2.putText(frame, text, (tx + 2, ty + 2),
                        font, scale, (0, 0, 0), thick + 1, cv2.LINE_AA)
            cv2.putText(frame, text, (tx, ty),
                        font, scale, (255, 255, 255), thick, cv2.LINE_AA)

        return frame

    # ----------------------------------------------------------
    # SOUND
    # ----------------------------------------------------------

    @staticmethod
    def _play_beep():
        system = platform.system()
        try:
            if system == "Windows":
                import winsound
                winsound.Beep(config.BEEP_FREQUENCY, config.BEEP_DURATION)
            else:
                import subprocess
                subprocess.run(
                    ["speaker-test", "-t", "sine",
                     "-f", str(config.BEEP_FREQUENCY), "-l", "1"],
                    timeout=3, stderr=subprocess.DEVNULL
                )
        except Exception as e:
            logger.warning(f"Beep failed: {e}")

    # ----------------------------------------------------------
    # EMAIL WITH ATTACHMENT
    # ----------------------------------------------------------

    @staticmethod
    def _send_email(to_email: str, subject: str,
                    body: str, snapshot_path: str = None):
        if not config.EMAIL_ENABLED:
            return
        try:
            msg             = MIMEMultipart()
            msg["From"]     = config.SENDER_EMAIL
            msg["To"]       = to_email
            msg["Subject"]  = subject
            msg.attach(MIMEText(body, "plain"))

            # Attach snapshot if available
            if snapshot_path and os.path.exists(snapshot_path):
                with open(snapshot_path, "rb") as f:
                    img = MIMEImage(f.read(), name=os.path.basename(snapshot_path))
                    img.add_header("Content-Disposition", "attachment",
                                   filename=os.path.basename(snapshot_path))
                    msg.attach(img)

            with smtplib.SMTP(config.SMTP_SERVER, config.SMTP_PORT, timeout=10) as s:
                s.ehlo()
                s.starttls()
                s.login(config.SENDER_EMAIL, config.SENDER_PASSWORD)
                s.sendmail(config.SENDER_EMAIL, to_email, msg.as_string())

            logger.info(f"Email sent → {to_email}")
            print(f"[Alert] Email sent → {to_email}")

        except Exception as e:
            logger.error(f"Email failed → {to_email}: {e}")
            print(f"[Alert] Email error → {to_email}: {e}")

    # ----------------------------------------------------------
    # WHATSAPP
    # ----------------------------------------------------------

    def _send_whatsapp(self, to: str, body: str):
        try:
            self._twilio.messages.create(
                from_=f"whatsapp:{config.TWILIO_FROM_NUMBER}",
                to=to,
                body=body
            )
            logger.info(f"WhatsApp sent → {to}")
            print(f"[Alert] WhatsApp sent → {to}")
        except Exception as e:
            logger.error(f"WhatsApp failed → {to}: {e}")
            print(f"[Alert] WhatsApp error: {e}")

    # ----------------------------------------------------------
    # SMS
    # ----------------------------------------------------------

    def _send_sms(self, to: str, body: str):
        try:
            self._twilio.messages.create(
                from_=config.TWILIO_FROM_NUMBER,
                to=to,
                body=body[:160]   # SMS limit
            )
            logger.info(f"SMS sent → {to}")
            print(f"[Alert] SMS sent → {to}")
        except Exception as e:
            logger.error(f"SMS failed → {to}: {e}")
            print(f"[Alert] SMS error: {e}")
