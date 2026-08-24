# ============================================================
#  Face Recognizer — Simple ML (OpenCV LBPH)
#  NO DeepFace needed. Works instantly. 100% offline.
# ============================================================
"""
HOW IT WORKS (Simple ML):
  1. On startup, reads every photo from person_db/ folders
  2. Detects the face in each photo using Haar Cascade (OpenCV)
  3. Trains an LBPH (Local Binary Pattern Histogram) model
     — this is classical Machine Learning, very fast
  4. Every frame: detects face in webcam, asks LBPH "who is this?"
  5. If confidence is good enough → shows person info card
  6. If no match → shows Unknown card

WHY LBPH INSTEAD OF DEEPFACE:
  - No heavy install (no tensorflow, no GPU)
  - Works on any Python version including 3.14
  - Trains in under 1 second
  - Recognition happens in milliseconds per frame
  - Works well with 1-5 photos per person

FOLDER FORMAT:
  person_db/
"""

import cv2
import numpy as np
import os
import logging
import config

logger = logging.getLogger(__name__)

# ── Confidence threshold ───────────────────────────────────
# Lower = stricter matching. 70 is a good starting point.
# If it shows wrong names, lower this. If it shows Unknown too
# often, raise it.
LBPH_CONFIDENCE_THRESHOLD = 100.0

# How often to run recognition (every N frames)
RECOGNITION_INTERVAL = 10


def _parse_folder_name(folder_name: str) -> dict:
    """
    'Natiq Ali-Age-21_Role-Student'
    → { name:'Natiq Ali', age:'21', role:'Student' }
    """
    info = {"name": folder_name, "age": "?", "role": "Person", "known": True}
    try:
        parts = folder_name.split("_")
        for p in parts:
            if p.startswith("Role-"):
                info["role"] = p.replace("Role-", "").replace("-", " ")
        name_age = parts[0]
        if "-Age-" in name_age:
            name_part, age_part = name_age.split("-Age-", 1)
            info["name"] = name_part.strip()
            info["age"]  = age_part.strip()
        else:
            info["name"] = name_age.strip()
    except Exception as e:
        logger.debug(f"Parse failed: {e}")
    return info


def _load_cascade_classifier():
    """
    Attempts to load Haar Cascade with multiple fallback paths.
    Returns the loaded cascade or None if all attempts fail.
    """
    cascade_paths = [
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
        cv2.data.haarcascades + "haarcascade_frontalface_alt.xml",
        cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml",
        "haarcascade_frontalface_default.xml",  # Current directory
    ]
    
    for path in cascade_paths:
        try:
            cascade = cv2.CascadeClassifier(path)
            if not cascade.empty():
                print(f"[FaceRecog] Cascade loaded from: {path}")
                return cascade
        except Exception as e:
            logger.debug(f"Failed to load cascade from {path}: {e}")
    
    print("[FaceRecog] WARNING: Could not load face cascade classifier!")
    print("  Face detection will be skipped. Add photos directly as face crops.")
    return None


class AgeEstimator:
    """
    Face recognition using OpenCV LBPH — simple, fast, reliable.
    Named AgeEstimator to stay compatible with the rest of the project.
    """

    def __init__(self):
        self.db_path      = "person_db"
        self._cache       = {}        # idx → info dict
        self._identity_cache = {}     # persistent: keeps last known match
        self._frame_count = 0

        # LBPH recognizer (built into OpenCV — no extra install needed)
        self._recognizer  = cv2.face.LBPHFaceRecognizer_create()
        self._trained     = False
        self._label_map   = {}        # int label → person info dict
        self._persons     = []        # list of person info dicts

        # Haar cascade for face detection inside crops (with error handling)
        self._face_cascade = _load_cascade_classifier()

        # Train on startup
        self._train()

    # ----------------------------------------------------------
    # Training
    # ----------------------------------------------------------

    def _train(self):
        """
        Reads all photos from person_db/, detects faces,
        and trains the LBPH model. Runs once at startup.
        """
        if not os.path.exists(self.db_path):
            print(f"[FaceRecog] person_db/ folder not found at '{self.db_path}'")
            print("  Create it and add person folders inside.")
            return

        faces  = []
        labels = []
        label  = 0

        for folder_name in sorted(os.listdir(self.db_path)):
            folder_path = os.path.join(self.db_path, folder_name)
            if not os.path.isdir(folder_path):
                continue

            info = _parse_folder_name(folder_name)
            photos_loaded = 0

            for filename in os.listdir(folder_path):
                if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    continue

                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"[FaceRecog] Could not read: {img_path}")
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Try to detect a face in the photo
                if self._face_cascade is not None:
                    try:
                        detected = self._face_cascade.detectMultiScale(
                            gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
                        )
                    except cv2.error as e:
                        print(f"[FaceRecog] Cascade detection error: {e}")
                        detected = []
                else:
                    detected = []

                if len(detected) > 0:
                    # Use the largest detected face
                    x, y, w, h = max(detected, key=lambda r: r[2]*r[3])
                    face_gray = gray[y:y+h, x:x+w]
                else:
                    # No face detected — use the whole image resized
                    # (works when the photo is already a face crop)
                    face_gray = gray

                # Resize to standard size for LBPH
                face_resized = cv2.resize(face_gray, (100, 100))
                faces.append(face_resized)
                labels.append(label)
                photos_loaded += 1

            if photos_loaded > 0:
                self._label_map[label] = info
                self._persons.append(info)
                label += 1
                print(f"[FaceRecog] Loaded '{info['name']}' — {photos_loaded} photo(s)")

        if len(faces) == 0:
            print("[FaceRecog] No faces loaded — add photos to person_db/ folders")
            return

        # Train LBPH
        self._recognizer.train(faces, np.array(labels))
        self._trained = True
        print(f"[FaceRecog] Training complete — {len(self._persons)} person(s) registered")

    # ----------------------------------------------------------
    # Recognition
    # ----------------------------------------------------------

    def _recognize_face(self, face_gray: np.ndarray) -> dict | None:
        """
        Runs LBPH recognition on a grayscale face crop.
        Returns person info dict if confident enough, else None.
        """
        if not self._trained:
            return None

        face_resized = cv2.resize(face_gray, (100, 100))
        label, confidence = self._recognizer.predict(face_resized)

        # Lower confidence value = better match in LBPH
        if confidence < LBPH_CONFIDENCE_THRESHOLD:
            return self._label_map.get(label)
        return None

    def _extract_face(self, crop: np.ndarray):
        """
        Detects face inside a person bounding box crop.
        Returns grayscale face, or grayscale full crop if no face found.
        """
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        if self._face_cascade is None:
            return gray  # Cascade not available, return full crop
        
        try:
            detected = self._face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
            )
        except cv2.error:
            return gray  # Error in detection, return full crop
        
        if len(detected) > 0:
            x, y, w, h = max(detected, key=lambda r: r[2]*r[3])
            return gray[y:y+h, x:x+w]
        return gray   # fallback: use full crop

    # ----------------------------------------------------------
    # estimate() — called every frame by main_v3.py
    # ----------------------------------------------------------

    def estimate(self, frame: np.ndarray, detections: list,
                 track_ids: list = None) -> dict:
        """
        Returns { tid: { name, age, role, is_child, known, label } }

        KEY FIX: Keys are now Track IDs (TIDs), not list indices.
        This locks identity data to the correct person box even when
        2+ people are in frame and YOLO returns them in different orders.

        track_ids — list of TIDs from PersonTracker, same length as detections.
                    Falls back to index-based if not provided (backward compat).
        """
        self._frame_count += 1

        # Use track IDs if provided, else fall back to index
        ids = track_ids if track_ids and len(track_ids) == len(detections) \
              else list(range(len(detections)))

        # Between recognition frames, return cached result
        if self._frame_count % RECOGNITION_INTERVAL != 0:
            merged = {}
            for tid in ids:
                if tid in self._identity_cache:
                    merged[tid] = self._identity_cache[tid]
                elif tid in self._cache:
                    merged[tid] = self._cache[tid]
            return merged

        h, w = frame.shape[:2]

        for list_idx, det in enumerate(detections):
            tid = ids[list_idx]   # permanent track ID for this person

            x1 = max(0, int(det[0]))
            y1 = max(0, int(det[1]))
            x2 = min(w, int(det[2]))
            y2 = min(h, int(det[3]))

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 30 or crop.shape[1] < 30:
                continue

            try:
                face_gray = self._extract_face(crop)
                db_info   = self._recognize_face(face_gray)

                if db_info:
                    is_child = (
                        int(db_info["age"]) < config.AGE_CHILD_THRESHOLD
                        if db_info["age"].isdigit()
                        else False
                    )
                    entry = {
                        "name":     db_info["name"],
                        "age":      db_info["age"],
                        "role":     db_info["role"],
                        "est_age":  db_info["age"],
                        "is_child": is_child,
                        "known":    True,
                        "label":    f"{db_info['name']} | Age-{db_info['age']} | {db_info['role']}",
                        "tid":      tid,
                    }
                    # Lock to TID — not list index
                    self._cache[tid]          = entry
                    self._identity_cache[tid] = entry

                else:
                    # Unknown — keep previous identity if we had one for this TID
                    if tid in self._identity_cache:
                        self._cache[tid] = self._identity_cache[tid]
                    else:
                        self._cache[tid] = {
                            "name":     "Unknown Subject",
                            "age":      "?",
                            "role":     "Unknown",
                            "est_age":  "?",
                            "is_child": False,
                            "known":    False,
                            "label":    "Unknown Subject",
                            "tid":      tid,
                        }

            except Exception as e:
                logger.debug(f"Recognition error tid {tid}: {e}")

        # Clean stale TIDs that are no longer tracked
        active_tids = set(ids)
        stale = [k for k in self._cache if k not in active_tids]
        for k in stale:
            del self._cache[k]
            # NOTE: Do NOT delete from _identity_cache — keep it so
            # we can restore identity if the person comes back

        return self._cache.copy()

    # ----------------------------------------------------------
    # get_person_details() — called by main_v3.py _get_identity()
    # ----------------------------------------------------------

    def get_person_details(self, face_img: np.ndarray) -> str:
        """Returns 'Name | Age | Role' string."""
        if not self._trained or face_img.size == 0:
            return "Unknown Subject"
        try:
            face_gray = self._extract_face(face_img)
            info = self._recognize_face(face_gray)
            if info:
                return f"{info['name']} | Age-{info['age']} | {info['role']}"
        except Exception as e:
            logger.debug(f"get_person_details error: {e}")
        return "Unknown Subject"

    # ----------------------------------------------------------
    # draw_age_labels() — draws info cards on frame
    # ----------------------------------------------------------

    def draw_age_labels(self, frame: np.ndarray,
                        detections: list,
                        age_data: dict,
                        track_ids: list = None) -> np.ndarray:
        """
        Draws identity cards on frame.
        track_ids — list of TIDs matching detections (from PersonTracker).
        Falls back to index-based if not provided.
        """
        ids = track_ids if track_ids and len(track_ids) == len(detections) \
              else list(range(len(detections)))

        for list_idx, det in enumerate(detections):
            tid = ids[list_idx]
            if tid not in age_data:
                continue
            x1 = int(det[0]); y1 = int(det[1])
            x2 = int(det[2]); y2 = int(det[3])
            info = age_data[tid]
            if info.get("known"):
                self._draw_known_card(frame, x1, y1, x2, y2, info)
            else:
                self._draw_unknown_card(frame, x1, y1, x2, y2, info)
        return frame

    # ----------------------------------------------------------
    # KNOWN person card — green
    # ----------------------------------------------------------

    def _draw_known_card(self, frame, x1, y1, x2, y2, info):
        is_child   = info.get("is_child", False)
        card_color = (0, 180, 60) if not is_child else (200, 120, 0)
        font       = cv2.FONT_HERSHEY_SIMPLEX

        lines = [
            ("KNOWN PERSON",              0.52, True),
            (f"Name : {info['name']}",   0.48, False),
            (f"Age  : {info['age']}",    0.48, False),
            (f"Role : {info['role']}",   0.48, False),
            ("Status: IDENTIFIED",        0.44, False),
        ]

        line_h  = 22
        padding = 10
        max_w   = max(cv2.getTextSize(t, font, s, 2 if b else 1)[0][0] for t, s, b in lines)
        card_w  = max_w + padding * 2 + 8
        card_h  = len(lines) * line_h + padding * 2

        fw = frame.shape[1]
        cx = x2 + 8 if x2 + card_w + 8 < fw else max(0, x1 - card_w - 8)
        cy = max(0, y1)

        # Background + accent bar + border
        cv2.rectangle(frame, (cx, cy), (cx + card_w, cy + card_h), (20, 20, 20), -1)
        cv2.rectangle(frame, (cx, cy), (cx + 5, cy + card_h), card_color, -1)
        cv2.rectangle(frame, (cx, cy), (cx + card_w, cy + card_h), card_color, 2)

        ty = cy + padding + 14
        for text, scale, bold in lines:
            cv2.putText(frame, text, (cx + padding + 6, ty), font, scale,
                        card_color if bold else (255, 255, 255),
                        2 if bold else 1, cv2.LINE_AA)
            ty += line_h

        # Connector line
        mid_y = cy + card_h // 2
        conn_x = cx if x2 + card_w + 8 < fw else cx + card_w
        conn_bx = x2 if x2 + card_w + 8 < fw else x1
        cv2.line(frame, (conn_bx, (y1 + y2) // 2), (conn_x, mid_y), card_color, 1)

    # ----------------------------------------------------------
    # UNKNOWN person card — orange/red
    # ----------------------------------------------------------

    def _draw_unknown_card(self, frame, x1, y1, x2, y2, info):
        card_color = (0, 100, 220)
        font       = cv2.FONT_HERSHEY_SIMPLEX

        lines = [
            ("UNKNOWN PERSON",            0.50, True),
            ("Not in database",           0.44, False),
            ("Status: UNIDENTIFIED",      0.44, False),
        ]

        line_h  = 20
        padding = 8
        max_w   = max(cv2.getTextSize(t, font, s, 1)[0][0] for t, s, b in lines)
        card_w  = max_w + padding * 2 + 8
        card_h  = len(lines) * line_h + padding * 2

        fw = frame.shape[1]
        cx = x2 + 8 if x2 + card_w + 8 < fw else max(0, x1 - card_w - 8)
        cy = max(0, y1)

        cv2.rectangle(frame, (cx, cy), (cx + card_w, cy + card_h), (20, 20, 20), -1)
        cv2.rectangle(frame, (cx, cy), (cx + 5, cy + card_h), card_color, -1)
        cv2.rectangle(frame, (cx, cy), (cx + card_w, cy + card_h), card_color, 1)

        ty = cy + padding + 13
        for text, scale, bold in lines:
            cv2.putText(frame, text, (cx + padding + 6, ty), font, scale,
                        card_color if bold else (200, 200, 200),
                        2 if bold else 1, cv2.LINE_AA)
            ty += line_h

        mid_y  = cy + card_h // 2
        conn_x = cx if x2 + card_w + 8 < fw else cx + card_w
        conn_bx= x2 if x2 + card_w + 8 < fw else x1
        cv2.line(frame, (conn_bx, (y1 + y2) // 2), (conn_x, mid_y), card_color, 1)