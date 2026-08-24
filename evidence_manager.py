"""
evidence_manager.py
Captures video clips when incidents occur and uploads them to Cloudinary.
Runs in background thread — never slows down the camera loop.
"""

import threading
import queue
import os
import time
import logging
import requests
import hashlib
import hmac
import base64
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

# ── Cloudinary Config ─────────────────────────────────────
CLOUD_NAME  = "blldvboo"
API_KEY     = "171985456889429"
API_SECRET  = "dSC2WQdjdiUpD9MMyl_KD7p25Qw"
UPLOAD_URL  = f"https://api.cloudinary.com/v1_1/{CLOUD_NAME}/video/upload"
IMG_UPLOAD  = f"https://api.cloudinary.com/v1_1/{CLOUD_NAME}/image/upload"

# ── Local buffer config ───────────────────────────────────
CLIP_DIR         = "evidence_clips"   # local folder for temp clips
CLIP_PRE_SECONDS  = 3                 # seconds before incident
CLIP_POST_SECONDS = 5                 # seconds after incident
FPS              = 30
BUFFER_SECONDS   = CLIP_PRE_SECONDS + 2   # how many seconds to keep in buffer

os.makedirs(CLIP_DIR, exist_ok=True)


class EvidenceManager:
    def __init__(self, backend_url: str, token_getter):
        """
        backend_url  : your FastAPI URL e.g. http://127.0.0.1:8000
        token_getter : callable that returns the current JWT token string
        """
        self._backend_url   = backend_url
        self._token_getter  = token_getter
        self._frame_buffer  = []        # rolling buffer of recent frames
        self._buffer_lock   = threading.Lock()
        self._upload_queue  = queue.Queue(maxsize=20)
        self._running       = False
        self._thread        = None
        self._frame_size    = None      # (width, height)

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._upload_loop, daemon=True)
        self._thread.start()
        log.info("[Evidence] Manager started.")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        log.info("[Evidence] Manager stopped.")

    # ── Called every frame from main_v3.py ───────────────
    def push_frame(self, frame):
        """Buffer the latest frame. Call this every frame in the camera loop."""
        import cv2
        now = time.time()
        if self._frame_size is None:
            h, w = frame.shape[:2]
            self._frame_size = (w, h)

        with self._buffer_lock:
            self._frame_buffer.append((now, frame.copy()))
            # Keep only last BUFFER_SECONDS worth of frames
            cutoff = now - BUFFER_SECONDS
            self._frame_buffer = [(t, f) for (t, f) in self._frame_buffer
                                  if t >= cutoff]

    # ── Called when incident is detected ─────────────────
    def capture_incident(self, incident_id: str, incident_type: str,
                         snapshot_path: str = None):
        """
        Queue a clip capture for this incident.
        Non-blocking — returns immediately.
        """
        try:
            with self._buffer_lock:
                frames_copy = list(self._frame_buffer)

            self._upload_queue.put_nowait({
                "incident_id":   incident_id,
                "incident_type": incident_type,
                "frames":        frames_copy,
                "snapshot_path": snapshot_path,
                "captured_at":   time.time(),
            })
        except queue.Full:
            log.warning("[Evidence] Upload queue full — clip dropped.")

    # ── Background upload loop ────────────────────────────
    def _upload_loop(self):
        while self._running:
            try:
                job = self._upload_queue.get(timeout=2)
                self._process_job(job)
                self._upload_queue.task_done()
            except queue.Empty:
                continue

    def _process_job(self, job: dict):
        incident_id   = job["incident_id"]
        incident_type = job["incident_type"]
        frames        = job["frames"]
        snapshot_path = job.get("snapshot_path")

        clip_url      = None
        thumbnail_url = None

        # ── 1. Save video clip locally ────────────────────
        if frames and self._frame_size:
            clip_path = self._save_clip(frames, incident_type)
            if clip_path:
                clip_url = self._upload_video(clip_path, incident_id)
                try:
                    os.remove(clip_path)   # clean up local file
                except Exception:
                    pass

        # ── 2. Upload snapshot as thumbnail ───────────────
        if snapshot_path and os.path.exists(snapshot_path):
            thumbnail_url = self._upload_image(snapshot_path, incident_id)

        # ── 3. Post evidence record to backend ────────────
        if clip_url or thumbnail_url:
            self._post_evidence(incident_id, clip_url, thumbnail_url)

    def _save_clip(self, frames: list, incident_type: str) -> str | None:
        """Write buffered frames to a local MP4 file."""
        try:
            import cv2
            ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename  = f"{CLIP_DIR}/clip_{incident_type}_{ts}.mp4"
            w, h      = self._frame_size
            fourcc    = cv2.VideoWriter_fourcc(*'mp4v')
            writer    = cv2.VideoWriter(filename, fourcc, FPS, (w, h))

            for (_, frame) in frames:
                writer.write(frame)
            writer.release()

            log.info(f"[Evidence] Clip saved: {filename}")
            return filename
        except Exception as e:
            log.warning(f"[Evidence] Failed to save clip: {e}")
            return None

    def _upload_video(self, clip_path: str, incident_id: str) -> str | None:
        """Upload MP4 clip to Cloudinary and return the URL."""
        try:
            timestamp  = str(int(time.time()))
            public_id  = f"incidents/{incident_id}/clip"
            sig_str    = f"public_id={public_id}&timestamp={timestamp}{API_SECRET}"
            signature  = hashlib.sha256(sig_str.encode()).hexdigest()

            with open(clip_path, "rb") as f:
                res = requests.post(UPLOAD_URL, data={
                    "api_key":   API_KEY,
                    "timestamp": timestamp,
                    "public_id": public_id,
                    "signature": signature,
                    "resource_type": "video",
                }, files={"file": f}, timeout=60)

            if res.status_code == 200:
                url = res.json().get("secure_url")
                log.info(f"[Evidence] Video uploaded: {url}")
                return url
            else:
                log.warning(f"[Evidence] Video upload failed: {res.text}")
                return None
        except Exception as e:
            log.warning(f"[Evidence] Upload error: {e}")
            return None

    def _upload_image(self, image_path: str, incident_id: str) -> str | None:
        """Upload snapshot image to Cloudinary and return URL."""
        try:
            timestamp  = str(int(time.time()))
            public_id  = f"incidents/{incident_id}/thumbnail"
            sig_str    = f"public_id={public_id}&timestamp={timestamp}{API_SECRET}"
            signature  = hashlib.sha256(sig_str.encode()).hexdigest()

            with open(image_path, "rb") as f:
                res = requests.post(IMG_UPLOAD, data={
                    "api_key":   API_KEY,
                    "timestamp": timestamp,
                    "public_id": public_id,
                    "signature": signature,
                }, files={"file": f}, timeout=30)

            if res.status_code == 200:
                url = res.json().get("secure_url")
                log.info(f"[Evidence] Thumbnail uploaded: {url}")
                return url
            else:
                log.warning(f"[Evidence] Image upload failed: {res.text}")
                return None
        except Exception as e:
            log.warning(f"[Evidence] Image upload error: {e}")
            return None

    def _post_evidence(self, incident_id: str,
                       clip_url: str, thumbnail_url: str):
        """Post the evidence record to the FastAPI backend."""
        token = self._token_getter()
        if not token:
            return
        try:
            res = requests.post(
                f"{self._backend_url}/evidence",
                json={
                    "incident_id":   incident_id,
                    "evidence_type": "video_clip",
                    "file_url":      clip_url,
                    "thumbnail_url": thumbnail_url,
                },
                headers={"Authorization": f"Bearer {token}",
                         "Content-Type": "application/json"},
                timeout=10
            )
            if res.status_code == 200:
                log.info(f"[Evidence] Record saved to backend.")
            else:
                log.warning(f"[Evidence] Backend post failed: {res.text}")
        except Exception as e:
            log.warning(f"[Evidence] Backend post error: {e}")
