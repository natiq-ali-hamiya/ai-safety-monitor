#!/usr/bin/env python3
# ============================================================
#  AI Safety Monitoring System v4.0  — FULLY FIXED + UPGRADED
# ============================================================

import sys, cv2, numpy as np
from datetime import datetime
import config
from zone_manager       import ZoneManager
from detector           import PersonDetector
from alert_system       import AlertSystem
from age_estimator      import AgeEstimator
from pose_detector      import PoseDetector
from snapshot_manager   import SnapshotManager
from utils              import FPSCounter, draw_hud, draw_controls_hint
from auto_zone_detector import AutoZoneDetector
from incident_analyzer  import IncidentAnalyzer
from plate_reader       import PlateReader
from data_store         import DataStore
from tracker            import PersonTracker
import cloud_reporter
import alert_engine
from evidence_manager   import EvidenceManager


class SafetyApp:
    def __init__(self):
        print("\n[System] Initialising AI Safety Monitor v4.0 ...")
        self.zone_mgr   = ZoneManager()
        self.detector   = PersonDetector()
        self.alerter    = AlertSystem()
        self.age_est    = AgeEstimator()
        self.pose_det   = PoseDetector()
        self.snapshot   = SnapshotManager()
        self.fps_ctr    = FPSCounter()
        self.auto_zones = None
        self.incidents  = IncidentAnalyzer()
        self.plates     = PlateReader()
        self.store      = DataStore()
        self.tracker    = PersonTracker()
        cloud_reporter.start(email="admin@aisafety.pk", password="secret")
        self.evidence = EvidenceManager(
            backend_url="http://127.0.0.1:8000",
            token_getter=lambda: cloud_reporter.TOKEN
        )
        self.evidence.start()
        alert_engine.start()
        self.draw_mode = False
        self.show_heatmap = False
        self.frame_in_danger = {}
        self.cap = None
        self._current_banner = "SAFE"
        self._alert_active = False
        self._last_snapshot = {}
        self._snapshot_cooldown = 15.0
        import time; self._time = time

    def _open_camera(self):
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
        if not self.cap.isOpened():
            print(f"[ERROR] Cannot open camera {config.CAMERA_INDEX}"); sys.exit(1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS,          config.TARGET_FPS)
        w, h = int(self.cap.get(3)), int(self.cap.get(4))
        print(f"[Camera] {w}x{h} opened.")
        self.auto_zones = AutoZoneDetector(w, h)

    def _try_snapshot(self, frame, inc_type, person_idx, age_info=None):
        now = self._time.time()
        if now - self._last_snapshot.get(person_idx, 0) < self._snapshot_cooldown:
            return None
        path = self.snapshot.capture(frame, inc_type, person_idx, age_info)
        if path: self._last_snapshot[person_idx] = now
        return path

    def _send_incident_alert(self, incident_type: str, label: str,
                              snap: str, age_info: dict, name: str):
        itype = incident_type.upper()
        if "FIGHT" in itype or "SNATCH" in itype or "WEAPON" in itype:
            self.alerter.trigger_person_in_zone(
                [label], snap,
                subject_override=f"ALERT: {incident_type} detected — {name}"
            )
        elif "ACCIDENT" in itype:
            self.alerter.trigger_lying_down(
                0, snap, age_info,
                subject_override=f"ALERT: ACCIDENT detected — {name}"
            )
            self.alerter.trigger_person_in_zone(
                [label], snap,
                subject_override=f"ALERT: ACCIDENT detected — {name}"
            )
        elif "CHILD" in itype:
            self.alerter.trigger_child_in_zone([label], snap, age_info)
        elif "LYING" in itype:
            seconds = age_info.get("seconds_down", 0) if age_info else 0
            self.alerter.trigger_lying_down(seconds, snap, age_info)
        else:
            self.alerter.trigger_person_in_zone([label], snap)

    def run(self):
        self._open_camera()
        win = "AI Safety Monitor v4.0"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, config.FRAME_WIDTH, config.FRAME_HEIGHT)
        cv2.setMouseCallback(win, self._mouse_cb)
        print("[System] Running! Web dashboard → http://localhost:5000\n")

        while True:
            ret, frame = self.cap.read()
            if not ret: continue

            self.fps_ctr.update()
            self.evidence.push_frame(frame)
            banner_type    = "SAFE"
            breached_zones = set()
            alert_ids      = set()
            any_zone_alert = False

            # 1. Detect
            detections    = self.detector.detect(frame)
            person_bboxes = [(int(d[0]),int(d[1]),int(d[2]),int(d[3])) for d in detections]
            all_dets      = self.detector.detect_all(frame)

            # 1b. Track IDs
            track_ids = self.tracker.update(person_bboxes)

            # 2. Heatmap
            self.auto_zones.update(person_bboxes)

            # 3. Age + identity
            age_data  = self.age_est.estimate(frame, detections, track_ids)
            pose_data = self.pose_det.analyze(frame, detections)

            # 4. Incident analysis
            active_incidents = self.incidents.analyze(person_bboxes, all_dets, pose_data)
            for inc in active_incidents:
                snap   = self._try_snapshot(frame, inc["type"], 0)
                inc_id = self.store.log_incident(inc["type"], inc["confidence"], snap,
                                                 person_count=len(inc["persons"]),
                                                 extra={"label": inc["label"]})
                if snap: self.store.log_snapshot(snap, inc["type"], inc_id)

                # Old alert system
                self._send_incident_alert(
                    incident_type=inc["type"],
                    label=inc["label"],
                    snap=snap,
                    age_info={},
                    name="Unknown"
                )

                banner_type = inc["type"]

                # Cloud report
                cloud_reporter.report(
                    incident_type=inc["type"],
                    confidence=inc.get("confidence", 0.9),
                    location=getattr(config, "CAMERA_LOCATION", "Main Camera"),
                    persons=[{"label": inc.get("label", "Unknown")}]
                )

                # Evidence capture
                self.evidence.capture_incident(
                    incident_id=str(inc_id) if inc_id else "unknown",
                    incident_type=inc["type"],
                    snapshot_path=snap
                )

                # ── NEW: Real email alert for crimes ─────────────
                itype = inc["type"].upper()
                if "LYING" in itype:
                    alert_engine.report_health_emergency(
                        person_info={"name": "Unknown Person", "known": False},
                        location=getattr(config, "CAMERA_LOCATION", "Main Camera"),
                        seconds_down=0,
                        evidence_url=snap or "No evidence",
                        snapshot_path=snap
                    )
                else:
                    alert_engine.report_crime(
                        incident_type=inc["type"],
                        criminal_info={
                            "name": "Unknown Person",
                            "known": False,
                            "confidence": inc.get("confidence", 0.9)
                        },
                        victim_info={},
                        evidence_url=snap or "No evidence",
                        location=getattr(config, "CAMERA_LOCATION", "Main Camera"),
                        snapshot_path=snap
                    )

            # 5. Per-person: identity + zone + pose
            for list_idx, bbox in enumerate(detections):
                tid = track_ids[list_idx]
                x1, y1, x2, y2 = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])

                age_info = age_data.get(tid, {
                    "is_child": False, "age": 25,
                    "label": "", "known": False, "name": "Unknown"
                })

                if age_info.get("known"):
                    identity_label = age_info.get("label", "Unknown Subject")
                    id_color = (0, 255, 0)
                else:
                    identity_label = age_info.get("label", "Unknown Subject") or "Unknown Subject"
                    id_color = (0, 165, 255)

                name_only = age_info.get("name", "Unknown")

                cv2.putText(frame, f"TID:{tid} {identity_label}", (x1, max(y1-15, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, id_color, 2, cv2.LINE_AA)

                # Zone check
                zones_hit = self.zone_mgr.check_intrusion((x1, y1, x2, y2))
                if zones_hit:
                    self.frame_in_danger[tid] = self.frame_in_danger.get(tid, 0) + 1
                    if self.frame_in_danger[tid] >= config.ALERT_FRAME_THRESHOLD:
                        alert_ids.add(list_idx)
                        breached_zones.update(zones_hit)
                        any_zone_alert = True

                        base_type = "CHILD_IN_DANGER_ZONE" if age_info.get("is_child") else "PERSON_IN_ZONE"
                        inc_type  = f"{base_type} ({name_only})"
                        snap      = self._try_snapshot(frame, inc_type, tid, age_info)
                        inc_id    = self.store.log_incident(inc_type, 0.9, snap,
                                                            zone_names=list(zones_hit), person_count=1)
                        if snap: self.store.log_snapshot(snap, inc_type, inc_id)

                        self._send_incident_alert(
                            incident_type=inc_type,
                            label=f"Zone breach: {list(zones_hit)}",
                            snap=snap,
                            age_info=age_info,
                            name=name_only
                        )

                        banner_type = "CHILD_ZONE" if age_info.get("is_child") else "PERSON_ZONE"

                        cloud_reporter.report(
                            incident_type=inc_type,
                            confidence=0.9,
                            location=getattr(config, "CAMERA_LOCATION", "Main Camera"),
                            persons=[{"name": name_only, "is_child": age_info.get("is_child", False)}]
                        )

                        # ── NEW: Child safety email alert ────────
                        if age_info.get("is_child"):
                            alert_engine.report_child_danger(
                                child_info={
                                    "name": name_only,
                                    "age": age_info.get("age", "Unknown"),
                                    "known": age_info.get("known", False)
                                },
                                zone_name=str(list(zones_hit)),
                                location=getattr(config, "CAMERA_LOCATION", "Main Camera"),
                                evidence_url=snap or "No evidence",
                                snapshot_path=snap
                            )
                        else:
                            alert_engine.report_crime(
                                incident_type=inc_type,
                                criminal_info={
                                    "name": name_only,
                                    "known": age_info.get("known", False),
                                    "confidence": 0.9
                                },
                                victim_info={},
                                evidence_url=snap or "No evidence",
                                location=getattr(config, "CAMERA_LOCATION", "Main Camera"),
                                snapshot_path=snap
                            )
                else:
                    self.frame_in_danger.pop(tid, None)

                # Pose / lying down
                p_info = pose_data.get(list_idx, {})
                if p_info.get("needs_1122"):
                    banner_type = "LYING"
                    lit    = f"PERSON_LYING_DOWN ({name_only})"
                    snap   = self._try_snapshot(frame, lit, tid, age_info)
                    inc_id = self.store.log_incident(lit, 0.85, snap, person_count=1)
                    if snap: self.store.log_snapshot(snap, lit, inc_id)

                    lying_info = dict(age_info)
                    lying_info["seconds_down"] = p_info.get("seconds_down", 0)

                    self._send_incident_alert(
                        incident_type="LYING_DOWN",
                        label=lit,
                        snap=snap,
                        age_info=lying_info,
                        name=name_only
                    )

                    # ── NEW: Health emergency email alert ────────
                    alert_engine.report_health_emergency(
                        person_info={"name": name_only, "known": age_info.get("known", False)},
                        location=getattr(config, "CAMERA_LOCATION", "Main Camera"),
                        seconds_down=p_info.get("seconds_down", 0),
                        evidence_url=snap or "No evidence",
                        snapshot_path=snap
                    )

            # Remove stale danger counters
            active_set = set(track_ids)
            stale = [k for k in self.frame_in_danger if k not in active_set]
            for k in stale: del self.frame_in_danger[k]

            # 6. License plates
            plate_results = self.plates.process(frame, all_dets)
            for pr in plate_results:
                if pr.get("fresh"):
                    snap = self._try_snapshot(frame, "LICENSE_PLATE", 99)
                    pid  = self.store.log_plate(pr["plate"], pr["vehicle_type"],
                                                pr.get("confidence", 0.0), snap)
                    if snap: self.store.log_snapshot(snap, "LICENSE_PLATE", pid)

                    # ── NEW: Hit & run email alert ───────────────
                    alert_engine.report_hit_and_run(
                        plate_number=pr["plate"],
                        vehicle_type=pr.get("vehicle_type", "Vehicle"),
                        owner_info={},
                        location=getattr(config, "CAMERA_LOCATION", "Main Camera"),
                        evidence_url=snap or "No evidence",
                        snapshot_path=snap
                    )

            # 7. Render
            if self.show_heatmap:
                frame = self.auto_zones.draw_heatmap(frame)
            frame = self.auto_zones.draw_auto_zones(frame)
            frame = self.zone_mgr.draw_zones(frame, breached_zones)
            frame = self.detector.draw_detections(frame, detections, alert_ids)
            frame = self.age_est.draw_age_labels(frame, detections, age_data, track_ids)
            frame = self.pose_det.draw_pose_status(frame, detections, pose_data)
            frame = self.incidents.draw_incidents(frame)
            frame = self.plates.draw_plates(frame, plate_results)

            self._alert_active = any_zone_alert or bool(active_incidents)
            if self._alert_active:
                self._current_banner = banner_type
                self.alerter._alert_active = True
            else:
                self.alerter.clear()
            frame = self.alerter.draw_alert_banner(frame, self._current_banner)

            mode_label = ("DRAW MODE" if self.draw_mode
                          else f"Monitoring | {self.auto_zones.get_status()}")
            frame = draw_hud(frame, self.fps_ctr.fps, len(detections),
                             self.zone_mgr.get_zone_count(),
                             self._alert_active, mode_label, self.detector.backend)
            frame = draw_controls_hint(frame)

            if self.draw_mode:
                fh, fw = frame.shape[:2]
                cv2.rectangle(frame, (0, fh-5), (fw, fh), (0, 220, 255), -1)

            cv2.imshow(win, frame)

            # 8. Keyboard
            key = cv2.waitKey(1) & 0xFF
            if   key == ord('q'): print("[System] Quit."); break
            elif key == ord('d'):
                self.draw_mode = not self.draw_mode
                print(f"[Zone] Draw mode {'ON' if self.draw_mode else 'OFF'}")
            elif key == ord('c'):
                r = self.zone_mgr.delete_last_zone()
                print(f"[Zone] Deleted {r}" if r else "[Zone] Nothing to delete.")
            elif key == ord('x'):
                self.zone_mgr.clear_all_zones()
                print("[Zone] Cleared all.")
            elif key == ord('s'):
                p = self.snapshot.capture(frame, "MANUAL_SNAPSHOT")
                if p: self.store.log_snapshot(p, "MANUAL_SNAPSHOT"); print(f"[Snap] {p}")
            elif key == ord('h'):
                self.show_heatmap = not self.show_heatmap
                print(f"[Heatmap] {'ON' if self.show_heatmap else 'OFF'}")
            elif key == ord('a'):
                self.auto_zones.enabled = not self.auto_zones.enabled
            elif key == ord('r'):
                self.auto_zones._analyze()
                self.auto_zones.apply_to_zone_manager(self.zone_mgr)
                print("[AutoZone] Re-analyzed.")

        self._cleanup()

    def _mouse_cb(self, event, x, y, flags, param):
        if self.draw_mode:
            self.zone_mgr.mouse_callback(event, x, y, flags, param)

    def _cleanup(self):
        if self.cap: self.cap.release()
        cv2.destroyAllWindows()
        self.store.close()
        if hasattr(self.pose_det, '_pose') and self.pose_det._pose:
            self.pose_det._pose.close()
        cloud_reporter.stop()
        self.evidence.stop()
        alert_engine.stop()
        print("[System] Shutdown complete.")


if __name__ == "__main__":
    SafetyApp().run()