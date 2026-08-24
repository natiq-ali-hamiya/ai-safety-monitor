#!/usr/bin/env python3
# ============================================================
#  Auto Danger Zone Detector
#  Analyzes movement heatmap and marks high-density areas
#  as danger zones automatically.
# ============================================================
"""
HOW IT WORKS:
  1. Every frame, the bottom-center of each detected person bbox
     is marked on a density grid (same size as the frame).
  2. After HEATMAP_SAMPLE_FRAMES frames, a Gaussian blur is applied
     to the grid to create a smooth heatmap.
  3. Cells above the AUTO_ZONE_THRESHOLD percentile are flagged as
     "danger" and rectangular zones are fitted around them.
  4. These zones are automatically added to ZoneManager.

CONTROLS (add to your main loop):
  A — toggle auto-zone analysis mode
  R — re-analyze and redraw auto zones now
"""

import cv2
import numpy as np
import config

# ── Tuneable constants ─────────────────────────────────────
HEATMAP_SAMPLE_FRAMES  = 150   # frames to collect before first analysis
AUTO_ZONE_THRESHOLD    = 85    # percentile above which a region is "danger"
HEATMAP_BLUR_KERNEL    = 51    # Gaussian blur kernel size (must be odd)
MIN_ZONE_AREA_PX       = 2000  # ignore tiny noise blobs below this area
MAX_AUTO_ZONES         = 6     # cap so the whole frame isn't flagged


class AutoZoneDetector:
    def __init__(self, frame_w: int, frame_h: int):
        self.frame_w = frame_w
        self.frame_h = frame_h

        # Accumulator grid — float32 for smooth blending
        self._grid = np.zeros((frame_h, frame_w), dtype=np.float32)
        self._frame_count   = 0
        self._auto_zones    = []   # list of dicts matching ZoneManager format
        self._heatmap_rgb   = None # for overlay rendering
        self._enabled       = True
        print("[AutoZone] Initialised — collecting heatmap data...")

    # ----------------------------------------------------------
    # Update (call every frame with current detections)
    # ----------------------------------------------------------

    def update(self, detections: list) -> None:
        """
        detections: list of (x1, y1, x2, y2) bboxes from PersonDetector.
        Marks the foot-point of each person on the density grid.
        """
        if not self._enabled:
            return

        for (x1, y1, x2, y2) in detections:
            # Use the centre-bottom point (foot position)
            cx = int((x1 + x2) / 2)
            cy = int(y2)
            cx = max(0, min(cx, self.frame_w - 1))
            cy = max(0, min(cy, self.frame_h - 1))
            # Splat a small gaussian around the foot point
            cv2.circle(self._grid, (cx, cy), 20, 1.0, -1)

        self._frame_count += 1

        # Auto-analyze after enough samples, then every N frames
        if (self._frame_count == HEATMAP_SAMPLE_FRAMES or
                self._frame_count % HEATMAP_SAMPLE_FRAMES == 0):
            self._analyze()

    # ----------------------------------------------------------
    # Analysis — convert heatmap to zones
    # ----------------------------------------------------------

    def _analyze(self) -> None:
        """Convert current density grid into rectangular auto-zones."""
        if self._grid.max() == 0:
            return  # no data yet

        # Smooth the grid
        blurred = cv2.GaussianBlur(
            self._grid,
            (HEATMAP_BLUR_KERNEL, HEATMAP_BLUR_KERNEL),
            0
        )

        # Threshold at the chosen percentile
        threshold = np.percentile(blurred[blurred > 0], AUTO_ZONE_THRESHOLD)
        binary    = (blurred >= threshold).astype(np.uint8) * 255

        # Morphological cleanup — fill gaps, remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))

        # Find contours and fit bounding rects
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Sort by area descending, cap at MAX_AUTO_ZONES
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        contours = contours[:MAX_AUTO_ZONES]

        new_zones = []
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < MIN_ZONE_AREA_PX:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            # Add a small margin
            margin = 10
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(self.frame_w, x + w + margin)
            y2 = min(self.frame_h, y + h + margin)
            new_zones.append({
                "name"  : f"AutoZone {i+1}",
                "x1"    : x1,  "y1": y1,
                "x2"    : x2,  "y2": y2,
                "active": True,
                "auto"  : True,   # flag so we know we generated this
            })

        self._auto_zones = new_zones
        self._build_heatmap_overlay(blurred)
        print(f"[AutoZone] Analysis complete — {len(new_zones)} danger zone(s) detected.")

    # ----------------------------------------------------------
    # Push auto-zones into ZoneManager
    # ----------------------------------------------------------

    def apply_to_zone_manager(self, zone_mgr) -> None:
        """
        Replaces all auto-generated zones in ZoneManager with
        the latest analysis results.
        """
        # Remove old auto-zones, keep manually drawn ones
        zone_mgr.zones = [z for z in zone_mgr.zones if not z.get("auto")]
        zone_mgr.zones.extend(self._auto_zones)
        zone_mgr._zone_counter = len(zone_mgr.zones) + 1
        print(f"[AutoZone] {len(self._auto_zones)} auto-zone(s) applied to ZoneManager.")

    # ----------------------------------------------------------
    # Heatmap overlay for rendering
    # ----------------------------------------------------------

    def _build_heatmap_overlay(self, blurred: np.ndarray) -> None:
        # Normalise to 0-255
        norm = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        self._heatmap_rgb = colored

    def draw_heatmap(self, frame: np.ndarray, alpha: float = 0.35) -> np.ndarray:
        """Blend the heatmap overlay onto the frame."""
        if self._heatmap_rgb is None:
            return frame
        overlay = frame.copy()
        # Resize in case frame size changed
        h, w = frame.shape[:2]
        hm = cv2.resize(self._heatmap_rgb, (w, h))
        cv2.addWeighted(hm, alpha, overlay, 1 - alpha, 0, overlay)
        return overlay

    def draw_auto_zones(self, frame: np.ndarray) -> np.ndarray:
        """Draw auto-zone rectangles with a distinct dotted border."""
        for zone in self._auto_zones:
            # Dashed rectangle effect
            pts = [
                ((zone["x1"], zone["y1"]), (zone["x2"], zone["y1"])),
                ((zone["x2"], zone["y1"]), (zone["x2"], zone["y2"])),
                ((zone["x2"], zone["y2"]), (zone["x1"], zone["y2"])),
                ((zone["x1"], zone["y2"]), (zone["x1"], zone["y1"])),
            ]
            for (p1, p2) in pts:
                _draw_dashed_line(frame, p1, p2, (0, 140, 255), 2, 12)

            # Label
            cv2.putText(
                frame, zone["name"],
                (zone["x1"] + 4, zone["y1"] + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 140, 255), 2, cv2.LINE_AA
            )
        return frame

    # ----------------------------------------------------------
    # Status info for HUD
    # ----------------------------------------------------------

    def get_status(self) -> str:
        progress = min(self._frame_count, HEATMAP_SAMPLE_FRAMES)
        pct      = int(progress / HEATMAP_SAMPLE_FRAMES * 100)
        if self._frame_count < HEATMAP_SAMPLE_FRAMES:
            return f"AutoZone: collecting {pct}%"
        return f"AutoZone: {len(self._auto_zones)} zone(s) active"

    def reset(self) -> None:
        self._grid        = np.zeros((self.frame_h, self.frame_w), dtype=np.float32)
        self._frame_count = 0
        self._auto_zones  = []
        self._heatmap_rgb = None
        print("[AutoZone] Heatmap reset.")

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, val: bool) -> None:
        self._enabled = val
        print(f"[AutoZone] {'Enabled' if val else 'Disabled'}")


# ── Helper ─────────────────────────────────────────────────

def _draw_dashed_line(img, pt1, pt2, color, thickness, dash_len):
    """Draw a dashed line between two points."""
    x1, y1 = pt1
    x2, y2 = pt2
    dist    = np.hypot(x2 - x1, y2 - y1)
    dx      = (x2 - x1) / dist
    dy      = (y2 - y1) / dist
    pos     = 0.0
    draw    = True
    while pos < dist:
        seg_end = min(pos + dash_len, dist)
        if draw:
            sx = int(x1 + dx * pos)
            sy = int(y1 + dy * pos)
            ex = int(x1 + dx * seg_end)
            ey = int(y1 + dy * seg_end)
            cv2.line(img, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)
        pos  += dash_len
        draw  = not draw
