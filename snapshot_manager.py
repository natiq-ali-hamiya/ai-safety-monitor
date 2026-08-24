# ============================================================
#  Child Safety Monitoring System v2.0 — Snapshot Manager
# ============================================================
"""
Captures and saves incident snapshots with:
 - Timestamp burned into the image
 - Incident type label
 - Structured filename for easy sorting
 - Returns the saved filepath for email attachment
"""

import os
import cv2
import numpy as np
from datetime import datetime
import logging
import config

logger = logging.getLogger(__name__)


class SnapshotManager:
    def __init__(self):
        os.makedirs(config.SNAPSHOT_DIR, exist_ok=True)
        print(f"[Snapshot] Incident folder: {config.SNAPSHOT_DIR}/")

    # ----------------------------------------------------------
    # Capture
    # ----------------------------------------------------------

    def capture(self, frame: np.ndarray,
                incident_type: str,
                person_idx: int = 0,
                age_info: dict = None) -> str | None:
        """
        Saves a snapshot of the current frame.
        Returns the saved filepath, or None if failed.

        incident_type examples:
          "CHILD_IN_DANGER_ZONE"
          "PERSON_LYING_DOWN"
          "UNKNOWN_IN_ZONE"
        """
        try:
            snapshot = frame.copy()
            snapshot = self._burn_overlay(snapshot, incident_type, age_info)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename  = f"{incident_type}_{timestamp}_person{person_idx}.jpg"
            filepath  = os.path.join(config.SNAPSHOT_DIR, filename)

            cv2.imwrite(filepath,
                        snapshot,
                        [cv2.IMWRITE_JPEG_QUALITY, config.SNAPSHOT_QUALITY])

            logger.info(f"Snapshot saved: {filepath}")
            print(f"[Snapshot] Saved: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Snapshot failed: {e}")
            return None

    # ----------------------------------------------------------
    # Overlay burned into the snapshot
    # ----------------------------------------------------------

    @staticmethod
    def _burn_overlay(frame: np.ndarray,
                      incident_type: str,
                      age_info: dict = None) -> np.ndarray:
        """Burns timestamp, incident type and age info onto the image."""
        h, w = frame.shape[:2]
        font  = cv2.FONT_HERSHEY_DUPLEX
        now   = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

        # Bottom info bar
        bar_h   = 70
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Incident type (left)
        incident_label = incident_type.replace("_", " ")
        cv2.putText(frame, incident_label,
                    (16, h - bar_h + 28),
                    font, 0.75, (0, 80, 255), 2, cv2.LINE_AA)

        # Age info (centre)
        if age_info:
            age_label = age_info.get("label", "")
            color     = (0, 165, 255) if age_info.get("is_child") else (200, 200, 200)
            cv2.putText(frame, age_label,
                        (w // 2 - 80, h - bar_h + 28),
                        font, 0.65, color, 1, cv2.LINE_AA)

        # Timestamp (right)
        (tw, _), _ = cv2.getTextSize(now, font, 0.60, 1)
        cv2.putText(frame, now,
                    (w - tw - 16, h - bar_h + 28),
                    font, 0.60, (200, 200, 200), 1, cv2.LINE_AA)

        # Top alert stripe
        cv2.rectangle(frame, (0, 0), (w, 6), (0, 0, 220), -1)

        # "INCIDENT CAPTURED" watermark
        cv2.putText(frame, "CHILD SAFETY MONITOR  |  INCIDENT CAPTURED",
                    (16, h - bar_h + 56),
                    font, 0.50, (150, 150, 150), 1, cv2.LINE_AA)

        return frame
