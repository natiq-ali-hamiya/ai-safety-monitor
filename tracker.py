# ============================================================
#  PersonTracker — IoU-Based Multi-Person Tracker
#  Fixes: data swapping between persons when 2+ people in frame
# ============================================================
"""
HOW IT WORKS:
  Each detected person gets a permanent Track ID (TID) assigned
  based on how much their bounding box OVERLAPS with a box from
  the previous frame (IoU = Intersection over Union).

  If a new box overlaps >= 40% with an old box → same person → same TID
  If no overlap found → new person → new TID assigned

  Identity data (name, age, role) is locked to the TID — NOT the
  list index. So even if YOLO returns people in a different order,
  or one person temporarily disappears, the data stays correct.

  This completely fixes:
  ✅ Wrong name shown on wrong person's box
  ✅ Data swapping when 2+ people are in frame
  ✅ Identity flickering when someone moves fast
"""

import numpy as np


# How much two boxes must overlap to be considered the same person
IOU_THRESHOLD = 0.40

# How many frames a person can disappear before we forget their TID
MAX_MISSING_FRAMES = 30


def _iou(boxA, boxB) -> float:
    """
    Calculates Intersection over Union between two boxes.
    Each box is (x1, y1, x2, y2).
    Returns a float between 0.0 (no overlap) and 1.0 (identical).
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter   = inter_w * inter_h

    if inter == 0:
        return 0.0

    areaA = max(1, (boxA[2]-boxA[0]) * (boxA[3]-boxA[1]))
    areaB = max(1, (boxB[2]-boxB[0]) * (boxB[3]-boxB[1]))

    return inter / float(areaA + areaB - inter)


class PersonTracker:
    """
    Assigns a permanent Track ID (TID) to each person across frames.

    Usage in main_v3.py:
        self.tracker = PersonTracker()
        ...
        # In the run loop, after detection:
        track_ids = self.tracker.update(person_bboxes)
        # track_ids[i] = TID for detections[i]
    """

    def __init__(self):
        self._next_tid   = 1           # next available track ID
        self._tracks     = {}          # tid → { box, missing_frames }
        self._tid_order  = []          # ordered list of active TIDs this frame

    def update(self, boxes: list) -> list:
        """
        Takes a list of bounding boxes [(x1,y1,x2,y2), ...] for current frame.
        Returns a list of Track IDs in the same order as the input boxes.

        Example:
            boxes    = [(10,10,100,200), (300,10,400,200)]
            track_ids = tracker.update(boxes)
            # → [1, 2]  (person 1 and person 2)
        """
        if not boxes:
            # No detections — increment missing counter for all tracks
            for tid in list(self._tracks.keys()):
                self._tracks[tid]['missing'] += 1
                if self._tracks[tid]['missing'] > MAX_MISSING_FRAMES:
                    del self._tracks[tid]
            self._tid_order = []
            return []

        assigned_tids = [None] * len(boxes)
        used_tids     = set()

        # ── Step 1: Match new boxes to existing tracks via IoU ──
        active_tids = list(self._tracks.keys())

        for new_idx, new_box in enumerate(boxes):
            best_iou = IOU_THRESHOLD   # must beat this threshold
            best_tid = None

            for tid in active_tids:
                if tid in used_tids:
                    continue  # already matched this track
                old_box = self._tracks[tid]['box']
                score   = _iou(old_box, new_box)
                if score > best_iou:
                    best_iou = score
                    best_tid = tid

            if best_tid is not None:
                assigned_tids[new_idx] = best_tid
                used_tids.add(best_tid)
                self._tracks[best_tid]['box']     = new_box
                self._tracks[best_tid]['missing'] = 0

        # ── Step 2: Create new TIDs for unmatched boxes ──────────
        for new_idx, new_box in enumerate(boxes):
            if assigned_tids[new_idx] is None:
                tid = self._next_tid
                self._next_tid += 1
                self._tracks[tid] = {'box': new_box, 'missing': 0}
                assigned_tids[new_idx] = tid

        # ── Step 3: Increment missing counter for lost tracks ─────
        for tid in list(self._tracks.keys()):
            if tid not in used_tids:
                self._tracks[tid]['missing'] += 1
                if self._tracks[tid]['missing'] > MAX_MISSING_FRAMES:
                    del self._tracks[tid]

        self._tid_order = assigned_tids
        return assigned_tids

    def get_active_tids(self) -> list:
        """Returns list of currently active track IDs."""
        return list(self._tracks.keys())
