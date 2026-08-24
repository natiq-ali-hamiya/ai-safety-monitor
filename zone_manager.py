# ============================================================
#  Child Safety Monitoring System — Zone Manager
# ============================================================
import cv2
import numpy as np
import config


class ZoneManager:
    def __init__(self):
        self.zones = []
        self._drawing = False
        self._start_pt = None
        self._temp_end = None
        self._zone_counter = 1

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drawing = True
            self._start_pt = (x, y)
            self._temp_end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self._drawing:
            self._temp_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self._drawing:
            self._drawing = False
            x1, y1 = self._start_pt
            x2, y2 = x, y
            if abs(x2-x1) > 20 and abs(y2-y1) > 20:
                self.zones.append({"name": f"Zone {self._zone_counter}",
                                   "x1": min(x1,x2), "y1": min(y1,y2),
                                   "x2": max(x1,x2), "y2": max(y1,y2), "active": True})
                self._zone_counter += 1
            self._start_pt = None
            self._temp_end = None

    def delete_last_zone(self):
        if self.zones:
            removed = self.zones.pop()
            self._zone_counter -= 1
            return removed["name"]
        return None

    def clear_all_zones(self):
        self.zones.clear()
        self._zone_counter = 1

    def get_zone_count(self):
        return len(self.zones)

    def check_intrusion(self, bbox):
        bx1, by1, bx2, by2, *_ = bbox  # This line is now fixed for v3.0
        breached = []
        for zone in self.zones:
            if not zone["active"]: 
                continue
            if bx1 < zone["x2"] and bx2 > zone["x1"] and by1 < zone["y2"] and by2 > zone["y1"]:
                breached.append(zone["name"])
        return breached

    def draw_zones(self, frame, breached_zones):
        overlay = frame.copy()
        for zone in self.zones:
            in_breach = zone["name"] in breached_zones
            color = config.ZONE_COLOR_DANGER if in_breach else config.ZONE_COLOR_SAFE
            cv2.rectangle(overlay, (zone["x1"], zone["y1"]), (zone["x2"], zone["y2"]), color, -1)
        cv2.addWeighted(overlay, config.ZONE_FILL_ALPHA, frame, 1-config.ZONE_FILL_ALPHA, 0, frame)
        for zone in self.zones:
            in_breach = zone["name"] in breached_zones
            color = config.ZONE_COLOR_DANGER if in_breach else config.ZONE_COLOR_SAFE
            cv2.rectangle(frame, (zone["x1"], zone["y1"]), (zone["x2"], zone["y2"]), color, 3 if in_breach else 2)
            cv2.putText(frame, zone["name"], (zone["x1"]+4, max(zone["y1"]-8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, color, config.FONT_THICKNESS, cv2.LINE_AA)
        if self._drawing and self._start_pt and self._temp_end:
            cv2.rectangle(frame, self._start_pt, self._temp_end, (0,255,255), 2)
        return frame
