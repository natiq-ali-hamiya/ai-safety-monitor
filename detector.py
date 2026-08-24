# ============================================================
#  Person + Weapon Detector — v3.1
#  Uses two YOLO models:
#    1. yolov8n.pt       → people, vehicles
#    2. yolov8n-weapons.pt → guns, knives
# ============================================================
import cv2
import numpy as np
import config

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False


class PersonDetector:
    def __init__(self):
        self.backend        = "none"
        self._model         = None
        self._weapon_model  = None
        self._hog           = None
        # Temporal smoothing — remember weapon detections for N frames
        # so a knife seen for 1-2 frames still triggers an alert
        self._weapon_buffer = []
        self._weapon_buffer_size = 8
        self._load_model()

    def _load_model(self):
        if _YOLO_AVAILABLE:
            try:
                self._model  = YOLO(config.YOLO_MODEL)
                self.backend = "yolo"
                print(f"[Detector] YOLOv8 loaded — model: {config.YOLO_MODEL}")
            except Exception as e:
                print(f"[Detector] YOLO load failed ({e}), falling back to HOG.")
                self._hog = cv2.HOGDescriptor()
                self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
                self.backend = "hog"
                return

            # NEW — load weapon model separately
            try:
                self._weapon_model = YOLO(config.WEAPON_MODEL)
                print(f"[Detector] Weapon model loaded — {config.WEAPON_MODEL}")
            except Exception as e:
                print(f"[Detector] Weapon model not loaded ({e}). Gun detection disabled.")
                self._weapon_model = None
        else:
            self._hog = cv2.HOGDescriptor()
            self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self.backend = "hog"
            print("[Detector] Using OpenCV HOG person detector (fallback).")

    def detect(self, frame):
        """Original v2.0 person-only detection"""
        if self.backend == "yolo":
            return self._detect_yolo(frame)
        return self._detect_hog(frame)

    def detect_all(self, frame):
        """
        v3.1 detection:
          Model 1 (yolov8n)         → People, Vehicles, Knives (COCO class 43)
          Model 2 (yolov8n-weapons) → Guns, Rifles, Knives (weapon-trained)
        """
        if self.backend != "yolo":
            return [(x1,y1,x2,y2, 0, conf)
                    for (x1,y1,x2,y2, conf) in self.detect(frame)]

        all_detections = []

        # ── Model 1: People + Vehicles + Knife (COCO) ──────────
        target_classes = [0, 2, 3, 5, 7, 43]
        results = self._model(frame, conf=config.CONFIDENCE_THRESH,
                              classes=target_classes, verbose=False)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls  = int(box.cls[0])
                all_detections.append((x1, y1, x2, y2, cls, conf))

        # ── Model 2: Weapons with temporal smoothing ───────────
        if self._weapon_model is not None:
            current_frame_weapons = []
            w_results = self._weapon_model(
                frame,
                conf=config.WEAPON_CONFIDENCE_THRESH,
                verbose=False
            )
            for result in w_results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf      = float(box.conf[0])
                    cls       = int(box.cls[0])
                    weapon_cls = 100 + cls
                    current_frame_weapons.append((x1, y1, x2, y2, weapon_cls, conf))

            # Add current frame to rolling buffer
            self._weapon_buffer.append(current_frame_weapons)
            if len(self._weapon_buffer) > self._weapon_buffer_size:
                self._weapon_buffer.pop(0)

            # Use best detection from recent buffer — if weapon seen
            # in ANY of last N frames, keep showing it at highest confidence
            best_weapons = {}
            for frame_dets in self._weapon_buffer:
                for det in frame_dets:
                    cls_id = det[4]
                    if cls_id not in best_weapons or det[5] > best_weapons[cls_id][5]:
                        best_weapons[cls_id] = det

            all_detections.extend(best_weapons.values())

        return all_detections

    def _detect_yolo(self, frame):
        results = self._model(frame, conf=config.CONFIDENCE_THRESH,
                              classes=[config.PERSON_CLASS_ID], verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                detections.append((x1, y1, x2, y2, conf))
        return detections

    def _detect_hog(self, frame):
        scale  = 0.5
        small  = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        rects, weights = self._hog.detectMultiScale(
            small, winStride=(8,8), padding=(4,4), scale=1.05)
        detections = []
        for (x, y, w, h), weight in zip(rects, weights):
            detections.append((
                int(x/scale), int(y/scale),
                int((x+w)/scale), int((y+h)/scale),
                float(weight[0]) if hasattr(weight, '__len__') else float(weight)
            ))
        return detections

    def draw_detections(self, frame, detections, alert_ids):
        for idx, (x1, y1, x2, y2, conf) in enumerate(detections):
            in_danger = idx in alert_ids
            color     = config.BBOX_COLOR_ALERT if in_danger else config.BBOX_COLOR_NORMAL
            thickness = 3 if in_danger else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            label = f"{'DANGER' if in_danger else 'Person'} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX,
                config.FONT_SCALE, config.FONT_THICKNESS)
            cv2.rectangle(frame, (x1, y1-th-10), (x1+tw+8, y1), color, -1)
            cv2.putText(frame, label, (x1+4, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE,
                        (255,255,255), config.FONT_THICKNESS, cv2.LINE_AA)
        return frame
    