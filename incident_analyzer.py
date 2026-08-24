#!/usr/bin/env python3
# ============================================================
#  Incident Analyzer v1.1 — FIXED weapon detection
# ============================================================
import cv2
import numpy as np
import time
from collections import defaultdict, deque
from datetime import datetime

# ── Constants ─────────────────────────────────────────────
VELOCITY_HISTORY      = 8
SNATCH_SPEED_THRESH   = 60
SNATCH_DIST_THRESH    = 80
FIGHT_OVERLAP_THRESH  = 0.15
FIGHT_WRIST_DIST      = 120
ACCIDENT_SIZE_CHANGE  = 0.35
WEAPON_CONF_THRESH    = 0.30   # FIXED: lowered from 0.45 → catches more knives

# COCO class IDs
VEHICLE_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
WEAPON_CLASSES  = {43: "Knife", 76: "Scissors"}

# Weapon model classes (offset by 100 in detector.py)
WEAPON_MODEL_CLASSES = {
    100: "Gun",
    101: "Knife",
    102: "Rifle",
    103: "Weapon",
}

# FIXED: combined lookup — covers COCO knives AND weapon model guns
ALL_WEAPON_CLASSES = {**WEAPON_CLASSES, **WEAPON_MODEL_CLASSES}

INCIDENT_FIGHT    = "FIGHT"
INCIDENT_SNATCH   = "SNATCHING"
INCIDENT_WEAPON   = "WEAPON_DISPLAYED"
INCIDENT_ACCIDENT = "ACCIDENT"


class IncidentAnalyzer:
    def __init__(self):
        self._positions: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=VELOCITY_HISTORY)
        )
        self._vehicle_areas: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=5)
        )
        self._last_incident_time: dict[str, float] = {}
        self._incident_cooldown = 8.0
        self.active_incidents: list[dict] = []
        print("[Incident] Analyzer initialised (v1.1 — weapon fix applied).")

    def analyze(self,
                person_detections: list,
                all_detections: list,
                pose_data: dict) -> list[dict]:
        self.active_incidents = []
        now = time.time()

        vehicles = [(x1,y1,x2,y2,cid) for (x1,y1,x2,y2,cid,conf)
                    in all_detections if cid in VEHICLE_CLASSES]

        # FIXED: was "cid in WEAPON_CLASSES" — now uses ALL_WEAPON_CLASSES
        # This catches: COCO knives (43), scissors (76), guns (100), rifles (102)
        weapons = [(x1,y1,x2,y2,cid,conf) for (x1,y1,x2,y2,cid,conf)
                   in all_detections
                   if cid in ALL_WEAPON_CLASSES and conf >= WEAPON_CONF_THRESH]

        for i, (x1,y1,x2,y2) in enumerate(person_detections):
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            self._positions[i].append((cx, cy))

        for i, (x1,y1,x2,y2,_) in enumerate(vehicles):
            area = (x2-x1) * (y2-y1)
            self._vehicle_areas[i].append(area)

        if len(person_detections) >= 2:
            fight = self._detect_fight(person_detections, pose_data)
            if fight and self._cooldown_ok(INCIDENT_FIGHT, now):
                self.active_incidents.append(fight)
                self._last_incident_time[INCIDENT_FIGHT] = now

            snatch = self._detect_snatch(person_detections)
            if snatch and self._cooldown_ok(INCIDENT_SNATCH, now):
                self.active_incidents.append(snatch)
                self._last_incident_time[INCIDENT_SNATCH] = now

        if weapons and self._cooldown_ok(INCIDENT_WEAPON, now):
            inc = self._detect_weapon(weapons, person_detections)
            if inc:
                self.active_incidents.append(inc)
                self._last_incident_time[INCIDENT_WEAPON] = now

        if vehicles and self._cooldown_ok(INCIDENT_ACCIDENT, now):
            acc = self._detect_accident(vehicles, person_detections)
            if acc:
                self.active_incidents.append(acc)
                self._last_incident_time[INCIDENT_ACCIDENT] = now

        return self.active_incidents

    def _detect_fight(self, persons: list, pose_data: dict) -> dict | None:
        for i in range(len(persons)):
            for j in range(i+1, len(persons)):
                iou = _bbox_iou(persons[i], persons[j])
                vi  = _velocity(self._positions[i])
                vj  = _velocity(self._positions[j])
                if iou > FIGHT_OVERLAP_THRESH and (vi > 20 or vj > 20):
                    confidence = 0.5 + min(iou, 0.3) + min((vi+vj)/200, 0.2)
                    if pose_data:
                        if self._wrist_near_other(i, j, persons, pose_data):
                            confidence = min(confidence + 0.3, 1.0)
                    if confidence >= 0.55:
                        return {
                            "type"      : INCIDENT_FIGHT,
                            "persons"   : [i, j],
                            "bbox"      : _merge_bboxes(persons[i], persons[j]),
                            "confidence": round(confidence, 2),
                            "label"     : "FIGHT DETECTED",
                            "color"     : (0, 0, 220),
                            "timestamp" : datetime.now().isoformat(),
                        }
        return None

    def _wrist_near_other(self, i, j, persons, pose_data) -> bool:
        pi = pose_data.get(i, {})
        pj = pose_data.get(j, {})
        for (attacker_pose, victim_bbox) in [(pi, persons[j]), (pj, persons[i])]:
            lms = attacker_pose.get("landmarks", {})
            for wrist_key in ["left_wrist", "right_wrist"]:
                w = lms.get(wrist_key)
                if w:
                    wx, wy = w
                    bx1, by1, bx2, by2 = victim_bbox
                    if (bx1-30 < wx < bx2+30) and (by1 < wy < by2):
                        return True
        return False

    def _detect_snatch(self, persons: list) -> dict | None:
        for i in range(len(persons)):
            for j in range(i+1, len(persons)):
                hist_i = self._positions[i]
                hist_j = self._positions[j]
                if len(hist_i) < VELOCITY_HISTORY or len(hist_j) < VELOCITY_HISTORY:
                    continue
                mid = VELOCITY_HISTORY // 2
                mid_dist = np.hypot(
                    hist_i[mid][0]-hist_j[mid][0],
                    hist_i[mid][1]-hist_j[mid][1]
                )
                vi_recent = _velocity_recent(hist_i, 3)
                vj_recent = _velocity_recent(hist_j, 3)
                if (mid_dist < SNATCH_DIST_THRESH and
                        (vi_recent > SNATCH_SPEED_THRESH or
                         vj_recent > SNATCH_SPEED_THRESH)):
                    faster = i if vi_recent > vj_recent else j
                    return {
                        "type"      : INCIDENT_SNATCH,
                        "persons"   : [faster],
                        "bbox"      : _merge_bboxes(persons[i], persons[j]),
                        "confidence": 0.70,
                        "label"     : "SNATCHING ALERT",
                        "color"     : (0, 128, 255),
                        "timestamp" : datetime.now().isoformat(),
                    }
        return None

    def _detect_weapon(self, weapons: list, persons: list) -> dict | None:
        for (wx1, wy1, wx2, wy2, cid, conf) in weapons:
            # FIXED: was WEAPON_CLASSES.get() — missed gun/rifle classes
            weapon_name = ALL_WEAPON_CLASSES.get(cid, "Weapon")

            # If no persons in frame, still flag the weapon
            if not persons:
                return {
                    "type"      : INCIDENT_WEAPON,
                    "persons"   : [],
                    "bbox"      : (wx1, wy1, wx2, wy2),
                    "confidence": round(float(conf), 2),
                    "label"     : f"{weapon_name} DETECTED",
                    "color"     : (0, 0, 255),
                    "timestamp" : datetime.now().isoformat(),
                }

            # Find nearest person
            for i, (px1, py1, px2, py2) in enumerate(persons):
                if _bboxes_overlap_or_near(
                    (wx1,wy1,wx2,wy2), (px1,py1,px2,py2), margin=80
                ):
                    return {
                        "type"      : INCIDENT_WEAPON,
                        "persons"   : [i],
                        "bbox"      : (wx1, wy1, wx2, wy2),
                        "confidence": round(float(conf), 2),
                        "label"     : f"{weapon_name} DETECTED",
                        "color"     : (0, 0, 255),
                        "timestamp" : datetime.now().isoformat(),
                    }

            # Weapon detected but no person nearby — still flag it
            return {
                "type"      : INCIDENT_WEAPON,
                "persons"   : [],
                "bbox"      : (wx1, wy1, wx2, wy2),
                "confidence": round(float(conf), 2),
                "label"     : f"{weapon_name} DETECTED (unattended)",
                "color"     : (0, 0, 255),
                "timestamp" : datetime.now().isoformat(),
            }
        return None

    def _detect_accident(self, vehicles: list, persons: list) -> dict | None:
        for i, (px1, py1, px2, py2) in enumerate(persons):
            for (vx1, vy1, vx2, vy2, cid) in vehicles:
                iou = _bbox_iou((px1,py1,px2,py2), (vx1,vy1,vx2,vy2))
                if iou > 0.10:
                    return {
                        "type"      : INCIDENT_ACCIDENT,
                        "persons"   : [i],
                        "bbox"      : _merge_bboxes(
                            (px1,py1,px2,py2), (vx1,vy1,vx2,vy2)),
                        "confidence": min(0.5 + iou * 2, 0.95),
                        "label"     : f"ACCIDENT — {VEHICLE_CLASSES.get(cid,'Vehicle')}",
                        "color"     : (0, 80, 255),
                        "timestamp" : datetime.now().isoformat(),
                    }
        for vi, area_hist in self._vehicle_areas.items():
            if len(area_hist) >= 3:
                change = abs(area_hist[-1] - area_hist[0]) / max(area_hist[0], 1)
                if change > ACCIDENT_SIZE_CHANGE:
                    vx1, vy1, vx2, vy2, cid = vehicles[min(vi, len(vehicles)-1)]
                    return {
                        "type"      : INCIDENT_ACCIDENT,
                        "persons"   : [],
                        "bbox"      : (vx1, vy1, vx2, vy2),
                        "confidence": 0.65,
                        "label"     : "ACCIDENT — Sudden vehicle change",
                        "color"     : (0, 80, 255),
                        "timestamp" : datetime.now().isoformat(),
                    }
        return None

    def draw_incidents(self, frame: np.ndarray) -> np.ndarray:
        for inc in self.active_incidents:
            x1, y1, x2, y2 = inc["bbox"]
            color = inc["color"]
            label = inc["label"]
            conf  = inc["confidence"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            text = f"{label}  {conf*100:.0f}%"
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_DUPLEX, 0.65, 2)
            cv2.rectangle(frame,
                          (x1, y1 - th - 12),
                          (x1 + tw + 8, y1), color, -1)
            cv2.putText(frame, text, (x1 + 4, y1 - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65,
                        (255, 255, 255), 2, cv2.LINE_AA)
        return frame

    def _cooldown_ok(self, incident_type: str, now: float) -> bool:
        last = self._last_incident_time.get(incident_type, 0)
        return (now - last) >= self._incident_cooldown


# ── Pure utility functions ─────────────────────────────────

def _bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = ((ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter)
    return inter / max(union, 1)

def _merge_bboxes(a, b):
    return (min(a[0],b[0]), min(a[1],b[1]),
            max(a[2],b[2]), max(a[3],b[3]))

def _velocity(pos_deque) -> float:
    if len(pos_deque) < 2: return 0.0
    pts = list(pos_deque)
    total = sum(np.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1])
                for i in range(1, len(pts)))
    return total / max(len(pts) - 1, 1)

def _velocity_recent(pos_deque, n: int = 3) -> float:
    if len(pos_deque) < 2: return 0.0
    pts = list(pos_deque)[-n:]
    total = sum(np.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1])
                for i in range(1, len(pts)))
    return total / max(len(pts) - 1, 1)

def _bboxes_overlap_or_near(a, b, margin: int = 0) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return (ax1-margin < bx2 and ax2+margin > bx1 and
            ay1-margin < by2 and ay2+margin > by1)