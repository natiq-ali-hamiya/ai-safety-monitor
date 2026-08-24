# ============================================================
#  Child Safety Monitoring System v2.0 — Pose Detector
# ============================================================
"""
Detects whether a person is lying down using MediaPipe pose
estimation. If a person is horizontal for LYING_DOWN_SECONDS,
a 1122 emergency alert is triggered.

Falls back to bounding-box aspect ratio if MediaPipe is
unavailable — a very wide, short box usually means lying down.
"""

import time
import cv2
import numpy as np
import logging
import config

logger = logging.getLogger(__name__)

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
    print("[PoseDetector] MediaPipe loaded successfully.")
except ImportError:
    _MP_AVAILABLE = False
    print("[PoseDetector] MediaPipe not installed — using bbox fallback.")
    print("  Run: pip install mediapipe")


class PoseDetector:
    def __init__(self):
        self.available = _MP_AVAILABLE
        self._pose     = None
        self._lying_since: dict[int, float] = {}   # idx → timestamp when lying started
        self._alerted:     set[int]          = set()

        if self.available:
            mp_pose    = mp.solutions.pose
            self._pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=0,          # fastest
                enable_segmentation=False,
                min_detection_confidence=0.5
            )

    # ----------------------------------------------------------
    # Main analysis
    # ----------------------------------------------------------

    def analyze(self, frame: np.ndarray,
                detections: list[tuple]) -> dict[int, dict]:
        """
        Returns { idx: { lying, seconds_down, needs_1122 } }
        """
        results   = {}
        now       = time.time()
        h, w      = frame.shape[:2]
        active_ids = set(range(len(detections)))

        # Clean up trackers for persons who left frame
        stale = [k for k in self._lying_since if k not in active_ids]
        for k in stale:
            del self._lying_since[k]
            self._alerted.discard(k)

        for idx, (x1, y1, x2, y2, _) in enumerate(detections):
            lying = self._is_lying(frame, x1, y1, x2, y2)

            if lying:
                if idx not in self._lying_since:
                    self._lying_since[idx] = now
                seconds_down = now - self._lying_since[idx]
            else:
                self._lying_since.pop(idx, None)
                self._alerted.discard(idx)
                seconds_down = 0.0

            needs_1122 = (
                lying and
                seconds_down >= config.LYING_DOWN_SECONDS and
                idx not in self._alerted
            )

            if needs_1122:
                self._alerted.add(idx)

            results[idx] = {
                "lying":       lying,
                "seconds_down": round(seconds_down, 1),
                "needs_1122":  needs_1122
            }

        return results

    # ----------------------------------------------------------
    # Lying detection logic
    # ----------------------------------------------------------

    def _is_lying(self, frame, x1, y1, x2, y2) -> bool:
        if self.available and self._pose:
            return self._mp_lying(frame, x1, y1, x2, y2)
        return self._bbox_lying(x1, y1, x2, y2)

    def _mp_lying(self, frame, x1, y1, x2, y2) -> bool:
        """Use MediaPipe shoulder-hip angle to detect lying."""
        try:
            crop = frame[max(0,y1):y2, max(0,x1):x2]
            if crop.size == 0:
                return False
            rgb     = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            result  = self._pose.process(rgb)
            if not result.pose_landmarks:
                return self._bbox_lying(x1, y1, x2, y2)

            lm = result.pose_landmarks.landmark
            mp_pose = mp.solutions.pose.PoseLandmark

            # Get shoulder and hip Y positions (normalised 0-1)
            ls = lm[mp_pose.LEFT_SHOULDER].y
            rs = lm[mp_pose.RIGHT_SHOULDER].y
            lh = lm[mp_pose.LEFT_HIP].y
            rh = lm[mp_pose.RIGHT_HIP].y

            shoulder_y = (ls + rs) / 2
            hip_y      = (lh + rh) / 2

            # If shoulder and hip are at similar Y → lying down
            vertical_diff = abs(shoulder_y - hip_y)
            return vertical_diff < 0.15

        except Exception as e:
            logger.debug(f"MediaPipe pose failed: {e}")
            return self._bbox_lying(x1, y1, x2, y2)

    @staticmethod
    def _bbox_lying(x1, y1, x2, y2) -> bool:
        """Fallback: wide bounding box = lying person."""
        bw = x2 - x1
        bh = y2 - y1
        if bh == 0:
            return False
        aspect = bw / bh
        return aspect > 1.8   # wider than tall = probably lying

    # ----------------------------------------------------------
    # Rendering
    # ----------------------------------------------------------

    def draw_pose_status(self, frame: np.ndarray,
                         detections: list[tuple],
                         pose_data: dict) -> np.ndarray:
        """Draw lying-down warning indicators."""
        for idx, (x1, y1, x2, y2, _) in enumerate(detections):
            if idx not in pose_data:
                continue
            info = pose_data[idx]
            if not info["lying"]:
                continue

            secs  = info["seconds_down"]
            color = config.BBOX_COLOR_LYING

            # Override bounding box colour
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            label = f"LYING DOWN  {secs:.0f}s"
            font  = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.6
            thick = 2
            (tw, th), _ = cv2.getTextSize(label, font, scale, thick)

            cv2.rectangle(frame,
                          (x1, y1 - th - 10),
                          (x1 + tw + 8, y1),
                          color, -1)
            cv2.putText(frame, label,
                        (x1 + 4, y1 - 5),
                        font, scale,
                        (255, 255, 255), thick, cv2.LINE_AA)

            # Progress bar showing time until 1122 alert
            bar_w  = x2 - x1
            filled = min(int(bar_w * secs / config.LYING_DOWN_SECONDS), bar_w)
            cv2.rectangle(frame, (x1, y2 + 2), (x2, y2 + 8), (80, 80, 80), -1)
            cv2.rectangle(frame, (x1, y2 + 2), (x1 + filled, y2 + 8), color, -1)

        return frame
