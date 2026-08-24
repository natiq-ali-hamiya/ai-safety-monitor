#!/usr/bin/env python3
# ============================================================
#  License Plate Reader v1.0
#  Detects vehicles → crops plate region → runs EasyOCR
#  Works on standard CCTV footage (720p/1080p)
# ============================================================
"""
PIPELINE:
  1. YOLOv8 detects vehicles (Car, Motorcycle, Bus, Truck).
  2. For each vehicle, the lower-centre 40% of the bbox is
     cropped (where plates typically appear).
  3. The crop is preprocessed (grayscale → threshold → upscale)
     to improve OCR accuracy on low-res footage.
  4. EasyOCR reads the plate text.
  5. A regex filter strips noise — keeps only strings that look
     like real plates (letters + digits, 4-10 chars).
  6. Results are stored in the database and overlaid on frame.

INSTALL:
  pip install easyocr
  (downloads ~300MB model on first run — needs internet once)

CONFIG:
  PLATE_READ_INTERVAL — only run OCR every N frames (CPU saving)
  PLATE_CONF_THRESH   — minimum YOLO confidence to attempt OCR
  PLATE_MIN_CHARS     — reject OCR output shorter than this
"""

import re
import cv2
import numpy as np
import time
from datetime import datetime
from collections import defaultdict

try:
    import easyocr
    _EASYOCR_AVAILABLE = True
except ImportError:
    _EASYOCR_AVAILABLE = False
    print("[PlateReader] WARNING: easyocr not installed. "
          "Run: pip install easyocr")

# ── Config ─────────────────────────────────────────────────
PLATE_READ_INTERVAL = 10      # OCR every N frames per vehicle
PLATE_CONF_THRESH   = 0.45
PLATE_MIN_CHARS     = 4
PLATE_MAX_CHARS     = 10
PLATE_CROP_TOP      = 0.55    # crop from this fraction of bbox height down
PLATE_UPSCALE       = 3.0     # upscale factor before OCR
VEHICLE_CLASSES     = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

# Plate regex: alphanumeric, 4-10 chars, at least 1 letter + 1 digit
_PLATE_RE = re.compile(r'^[A-Z0-9\-]{4,10}$')


class PlateReader:
    def __init__(self):
        self._reader        = None
        self._frame_counter: dict[int, int] = defaultdict(int)
        self._plate_cache:   dict[int, str] = {}   # vehicle_idx → last plate
        self._plate_times:   dict[int, str] = {}   # vehicle_idx → timestamp
        self._all_plates:    list[dict]     = []   # full log

        if _EASYOCR_AVAILABLE:
            print("[PlateReader] Loading EasyOCR model (first run takes ~30s)...")
            try:
                self._reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                print("[PlateReader] EasyOCR ready.")
            except Exception as e:
                print(f"[PlateReader] EasyOCR load failed: {e}")
        else:
            print("[PlateReader] EasyOCR unavailable — plate reading disabled.")

    # ----------------------------------------------------------
    # Main update (call every frame)
    # ----------------------------------------------------------

    def process(self,
                frame: np.ndarray,
                all_detections: list) -> list[dict]:
        """
        all_detections: list of (x1,y1,x2,y2,class_id,conf)
        Returns: list of plate result dicts for this frame.
        """
        if not self._reader:
            return []

        results = []
        h, w    = frame.shape[:2]

        vehicle_dets = [
            (x1,y1,x2,y2,cid,conf)
            for (x1,y1,x2,y2,cid,conf) in all_detections
            if cid in VEHICLE_CLASSES and conf >= PLATE_CONF_THRESH
        ]

        for vi, (vx1, vy1, vx2, vy2, cid, conf) in enumerate(vehicle_dets):
            self._frame_counter[vi] += 1

            # Show cached result every frame
            if vi in self._plate_cache:
                results.append({
                    "vehicle_idx" : vi,
                    "vehicle_type": VEHICLE_CLASSES[cid],
                    "plate"       : self._plate_cache[vi],
                    "bbox"        : (vx1, vy1, vx2, vy2),
                    "timestamp"   : self._plate_times.get(vi, ""),
                    "fresh"       : False,
                })

            # Run OCR every N frames
            if self._frame_counter[vi] % PLATE_READ_INTERVAL != 0:
                continue

            plate_text = self._read_plate(frame, vx1, vy1, vx2, vy2, h, w)
            if plate_text:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self._plate_cache[vi]  = plate_text
                self._plate_times[vi]  = ts
                record = {
                    "vehicle_idx" : vi,
                    "vehicle_type": VEHICLE_CLASSES[cid],
                    "plate"       : plate_text,
                    "bbox"        : (vx1, vy1, vx2, vy2),
                    "timestamp"   : ts,
                    "confidence"  : conf,
                    "fresh"       : True,
                }
                self._all_plates.append(record)
                results.append(record)
                print(f"[PlateReader] {VEHICLE_CLASSES[cid]} → {plate_text} @ {ts}")

        return results

    # ----------------------------------------------------------
    # OCR pipeline for a single vehicle crop
    # ----------------------------------------------------------

    def _read_plate(self, frame, vx1, vy1, vx2, vy2, h, w) -> str | None:
        # Compute plate crop region (lower portion of vehicle bbox)
        crop_y1 = int(vy1 + (vy2 - vy1) * PLATE_CROP_TOP)
        crop_y2 = vy2
        crop_x1 = max(0, vx1 - 10)
        crop_x2 = min(w, vx2 + 10)

        if crop_y2 <= crop_y1 or crop_x2 <= crop_x1:
            return None

        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size == 0:
            return None

        # Pre-process for better OCR
        processed = self._preprocess(crop)

        # Run OCR
        try:
            ocr_results = self._reader.readtext(
                processed,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-',
                detail=1,
                paragraph=False,
            )
        except Exception as e:
            print(f"[PlateReader] OCR error: {e}")
            return None

        # Filter and pick best result
        candidates = []
        for (bbox_pts, text, prob) in ocr_results:
            text_clean = text.upper().replace(' ', '').replace('.', '')
            if (PLATE_MIN_CHARS <= len(text_clean) <= PLATE_MAX_CHARS and
                    _PLATE_RE.match(text_clean) and
                    prob >= 0.25):
                candidates.append((prob, text_clean))

        if not candidates:
            return None

        # Return highest confidence
        candidates.sort(reverse=True)
        return candidates[0][1]

    # ----------------------------------------------------------
    # Pre-processing for better plate OCR
    # ----------------------------------------------------------

    @staticmethod
    def _preprocess(crop: np.ndarray) -> np.ndarray:
        # Upscale
        h, w = crop.shape[:2]
        up   = cv2.resize(crop,
                          (int(w * PLATE_UPSCALE), int(h * PLATE_UPSCALE)),
                          interpolation=cv2.INTER_CUBIC)
        # Grayscale
        gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
        # CLAHE for contrast
        clahe  = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray   = clahe.apply(gray)
        # Threshold
        _, thresh = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Slight denoise
        denoised  = cv2.fastNlMeansDenoising(thresh, h=10)
        return denoised

    # ----------------------------------------------------------
    # Draw overlays on frame
    # ----------------------------------------------------------

    def draw_plates(self, frame: np.ndarray, plate_results: list) -> np.ndarray:
        for pr in plate_results:
            x1, y1, x2, y2 = pr["bbox"]
            plate = pr["plate"]
            vtype = pr["vehicle_type"]
            fresh = pr.get("fresh", False)

            # Vehicle box
            color = (0, 220, 255) if fresh else (0, 160, 200)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Plate label strip
            label  = f"{vtype}: {plate}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_DUPLEX, 0.65, 2
            )
            cv2.rectangle(frame,
                          (x1, y2),
                          (x1 + tw + 10, y2 + th + 10),
                          (20, 20, 20), -1)
            cv2.putText(frame, label,
                        (x1 + 5, y2 + th + 4),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65,
                        (0, 255, 180), 2, cv2.LINE_AA)

            # Highlight plate crop area
            crop_y1 = int(y1 + (y2 - y1) * PLATE_CROP_TOP)
            cv2.rectangle(frame,
                          (x1, crop_y1), (x2, y2),
                          (0, 220, 255), 1)

        return frame

    # ----------------------------------------------------------
    # Data access
    # ----------------------------------------------------------

    def get_all_plate_records(self) -> list[dict]:
        """Returns full log of all detected plates (for DB/export)."""
        return self._all_plates.copy()

    def get_recent_plates(self, n: int = 10) -> list[dict]:
        return self._all_plates[-n:]
