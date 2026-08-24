# ============================================================
#  Child Safety Monitoring System — Utilities
# ============================================================
import time
import cv2
import numpy as np
import config


class FPSCounter:
    def __init__(self, window=30):
        self._times = []
        self._window = window

    def update(self):
        now = time.time()
        self._times.append(now)
        if len(self._times) > self._window:
            self._times.pop(0)

    @property
    def fps(self):
        if len(self._times) < 2: return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times)-1)/elapsed if elapsed > 0 else 0.0


def draw_hud(frame, fps, num_persons, num_zones, alert_active, mode, backend):
    h, w = frame.shape[:2]
    panel_w, panel_h = 320, 150
    x0, y0 = 10, h - panel_h - 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0+panel_w, y0+panel_h), (20,20,20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    border_color = (0,0,200) if alert_active else (0,180,0)
    cv2.rectangle(frame, (x0, y0), (x0+panel_w, y0+panel_h), border_color, 2)
    font, scale, thick, lh = cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1, 22
    tx, ty = x0+12, y0+24
    status_color = (0,80,255) if alert_active else (80,255,80)
    lines = [
        (f"Status : {'ALERT!' if alert_active else 'Safe'}", status_color),
        (f"FPS    : {fps:5.1f}", (200,200,200)),
        (f"Persons: {num_persons}", (200,200,200)),
        (f"Zones  : {num_zones}", (200,200,200)),
        (f"Model  : {backend.upper()}", (160,160,160)),
        (f"Mode   : {mode}", (160,160,160)),
    ]
    for text, color in lines:
        cv2.putText(frame, text, (tx, ty), font, scale, color, thick, cv2.LINE_AA)
        ty += lh
    return frame


def draw_controls_hint(frame):
    h, w = frame.shape[:2]
    hints = ["D=Draw zone", "C=Undo zone", "X=Clear all", "S=Snapshot", "Q=Quit"]
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1
    line_h = 20
    max_w  = max(cv2.getTextSize(t, font, scale, thick)[0][0] for t in hints)
    panel_w, panel_h = max_w+24, len(hints)*line_h+16
    x0, y0 = w-panel_w-10, 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0,y0), (x0+panel_w, y0+panel_h), (20,20,20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (x0,y0), (x0+panel_w, y0+panel_h), (100,100,100), 1)
    ty = y0+line_h
    for hint in hints:
        cv2.putText(frame, hint, (x0+12, ty), font, scale, (200,200,200), thick, cv2.LINE_AA)
        ty += line_h
    return frame
