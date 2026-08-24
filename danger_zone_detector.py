# ============================================================
#  Danger Zone Detector — Custom Dataset Integration
#  AI Safety Monitoring System — Add-on Module
#
#  Detects:
#    • Sewerage holes / open manholes
#    • Construction / working areas
#    • Electrical hazards
#    • Barrier violations
#    • Human proximity to each danger type
#
#  HOW TO USE:
#    1. Train YOLOv8 on your custom dataset (see TRAINING_GUIDE.md)
#    2. Place the trained model as: danger_zone_model.pt
#    3. Import this module in main_v3.py (see instructions at bottom)
# ============================================================

import cv2
import numpy as np

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

# ── Class Names ─────────────────────────────────────────────
# These must match the order in your dataset's data.yaml
# Edit this list to match YOUR labels exactly
DANGER_CLASS_NAMES = {
    0:  "manhole_open",       # Open sewerage holes
    1:  "manhole_covered",    # Covered manholes (less dangerous)
    2:  "construction_zone",  # Active working/construction area
    3:  "electrical_hazard",  # Exposed wires, panels
    4:  "barrier",            # Safety barriers / cones
    5:  "excavation",         # Dug-up ground / trenches
    6:  "water_hazard",       # Flooded areas
    7:  "fire_hazard",        # Flames, smoke sources
}

# ── Danger levels per class (1=low, 2=medium, 3=critical) ──
DANGER_LEVEL = {
    0: 3,   # open manhole → CRITICAL
    1: 1,   # covered → low
    2: 2,   # construction → medium
    3: 3,   # electrical → CRITICAL
    4: 1,   # barrier → informational
    5: 2,   # excavation → medium
    6: 2,   # water → medium
    7: 3,   # fire → CRITICAL
}

# ── Visual colors by danger level ──────────────────────────
LEVEL_COLORS = {
    1: (0, 200, 100),    # Green  — low
    2: (0, 165, 255),    # Orange — medium
    3: (0, 0, 255),      # Red    — critical
}

PROXIMITY_THRESHOLD_PX = 80   # how many pixels from hazard = "too close"
MODEL_PATH = "danger_zone_model.pt"
CONFIDENCE_THRESH = 0.45


class DangerZoneDetector:
    """
    Detects physical danger zones (manholes, construction sites, etc.)
    and checks if any person is dangerously close to them.
    """

    def __init__(self, model_path: str = MODEL_PATH,
                 conf_thresh: float = CONFIDENCE_THRESH):
        self.model = None
        self.conf_thresh = conf_thresh
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        if not _YOLO_AVAILABLE:
            print("[DangerZone] ultralytics not installed. pip install ultralytics")
            return
        try:
            self.model = YOLO(self.model_path)
            print(f"[DangerZone] Custom danger model loaded: {self.model_path}")
        except Exception as e:
            print(f"[DangerZone] Could not load model ({e})")
            print(f"[DangerZone] Train your model first — see TRAINING_GUIDE.md")

    def detect_hazards(self, frame: np.ndarray) -> list:
        """
        Run inference on frame.
        Returns list of dicts:
          { x1, y1, x2, y2, class_id, class_name, conf, danger_level }
        """
        if self.model is None:
            return []

        results = self.model(frame, conf=self.conf_thresh, verbose=False)
        hazards = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls  = int(box.cls[0])
                hazards.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "class_id":    cls,
                    "class_name":  DANGER_CLASS_NAMES.get(cls, f"hazard_{cls}"),
                    "conf":        conf,
                    "danger_level": DANGER_LEVEL.get(cls, 1),
                })
        return hazards

    def check_person_proximity(self, person_boxes: list,
                               hazards: list) -> dict:
        """
        For each person box (x1,y1,x2,y2,...), check if they are
        dangerously close to any hazard.

        Returns:
          { person_index: [list of hazard dicts they are near] }
        """
        alerts = {}
        for p_idx, pbox in enumerate(person_boxes):
            px1, py1, px2, py2 = pbox[0], pbox[1], pbox[2], pbox[3]
            p_cx = (px1 + px2) // 2
            p_cy = (py1 + py2) // 2

            near_hazards = []
            for hazard in hazards:
                hx1, hy1, hx2, hy2 = (hazard["x1"], hazard["y1"],
                                       hazard["x2"], hazard["y2"])

                # Check if person bounding box overlaps hazard box (with buffer)
                buffer = PROXIMITY_THRESHOLD_PX
                if (px1 < hx2 + buffer and px2 > hx1 - buffer and
                        py1 < hy2 + buffer and py2 > hy1 - buffer):
                    near_hazards.append(hazard)

                # Also check center-point distance for point hazards
                else:
                    h_cx = (hx1 + hx2) // 2
                    h_cy = (hy1 + hy2) // 2
                    dist = ((p_cx - h_cx)**2 + (p_cy - h_cy)**2) ** 0.5
                    if dist < PROXIMITY_THRESHOLD_PX * 2:
                        near_hazards.append(hazard)

            if near_hazards:
                alerts[p_idx] = near_hazards
        return alerts

    def draw_hazards(self, frame: np.ndarray, hazards: list,
                     proximity_alerts: dict = None) -> np.ndarray:
        """
        Draw all detected hazards and any proximity alerts on the frame.
        """
        overlay = frame.copy()
        proximity_alerts = proximity_alerts or {}

        # Collect all alerted hazard names for banner
        all_alerted = set()
        for near_list in proximity_alerts.values():
            for h in near_list:
                all_alerted.add(h["class_name"])

        for hazard in hazards:
            x1, y1, x2, y2 = hazard["x1"], hazard["y1"], hazard["x2"], hazard["y2"]
            lvl   = hazard["danger_level"]
            color = LEVEL_COLORS.get(lvl, (200, 200, 0))
            name  = hazard["class_name"].replace("_", " ").upper()
            conf  = hazard["conf"]

            is_alerted = hazard["class_name"] in all_alerted

            # Fill if critical
            if lvl == 3 or is_alerted:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

            # Border
            thickness = 3 if is_alerted else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Label
            label = f"{'⚠ ' if lvl >= 2 else ''}{name} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # Blend overlay
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

        # Proximity warning banners
        if proximity_alerts:
            hazard_names = ", ".join(all_alerted)
            banner = f"DANGER PROXIMITY: {hazard_names}"
            bw, bh = frame.shape[1], 48
            cv2.rectangle(frame, (0, 0), (bw, bh), (0, 0, 220), -1)
            cv2.putText(frame, banner, (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        return frame

    def get_alert_message(self, proximity_alerts: dict,
                          hazards: list) -> str:
        """
        Returns a human-readable alert message for email/WhatsApp.
        """
        if not proximity_alerts:
            return ""

        lines = ["⚠️ DANGER ZONE ALERT ⚠️", ""]
        lines.append(f"Persons near hazards: {len(proximity_alerts)}")
        lines.append("")

        for p_idx, near_list in proximity_alerts.items():
            for h in near_list:
                lvl_name = {1: "LOW", 2: "MEDIUM", 3: "CRITICAL"}[h["danger_level"]]
                lines.append(
                    f"  Person #{p_idx + 1} near "
                    f"{h['class_name'].replace('_', ' ').upper()} "
                    f"[{lvl_name}] — conf {h['conf']:.0%}"
                )
        return "\n".join(lines)


# ============================================================
#  HOW TO INTEGRATE INTO main_v3.py
# ============================================================
# 
#  1. At the top of main_v3.py, add:
#       from danger_zone_detector import DangerZoneDetector
#
#  2. In SafetyApp.__init__(), add:
#       self.danger_det = DangerZoneDetector()
#
#  3. In your main frame loop (inside run()), add after detect_all():
#
#       # ── Danger Zone Detection ──────────────────────────
#       hazards = self.danger_det.detect_hazards(frame)
#       person_boxes = [(d[0],d[1],d[2],d[3]) for d in detections
#                       if d[4] == 0]  # class 0 = person
#       prox_alerts = self.danger_det.check_person_proximity(
#                           person_boxes, hazards)
#       frame = self.danger_det.draw_hazards(frame, hazards, prox_alerts)
#
#       if prox_alerts:
#           msg = self.danger_det.get_alert_message(prox_alerts, hazards)
#           self.alerter.send_email("Danger Zone Alert", msg)
#
# ============================================================
