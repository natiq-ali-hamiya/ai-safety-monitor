#!/usr/bin/env python3
# ============================================================
#  Data Store v1.0
#  SQLite database for all incidents, plates, snapshots
#  + CSV export + image gallery index
# ============================================================
"""
TABLES:
  incidents   — all detected incidents (fight, snatch, weapon, accident, zone)
  plate_reads — all license plate detections
  snapshots   — all captured snapshot images with metadata
  heatmap_log — periodic heatmap density summaries

USAGE:
  store = DataStore()
  store.log_incident(type, confidence, snapshot_path, extra_data)
  store.log_plate(plate_text, vehicle_type, confidence, snapshot_path)
  store.export_csv("output.csv")
  store.get_stats()
"""

import os
import csv
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path


DB_PATH       = "safety_data.db"
EXPORT_DIR    = "exports"
GALLERY_DIR   = "incidents"   # must match config.SNAPSHOT_DIR


class DataStore:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(EXPORT_DIR, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row   # dict-like rows
        self._create_tables()
        print(f"[DataStore] Database: {db_path}")

    # ----------------------------------------------------------
    # Schema
    # ----------------------------------------------------------

    def _create_tables(self):
        cur = self._conn.cursor()
        cur.executescript("""
        CREATE TABLE IF NOT EXISTS incidents (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT    NOT NULL,
            type            TEXT    NOT NULL,
            confidence      REAL,
            zone_names      TEXT,
            person_count    INTEGER DEFAULT 0,
            snapshot_path   TEXT,
            extra_json      TEXT,
            alerted         INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS plate_reads (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT    NOT NULL,
            plate_text      TEXT    NOT NULL,
            vehicle_type    TEXT,
            confidence      REAL,
            snapshot_path   TEXT,
            incident_id     INTEGER REFERENCES incidents(id)
        );

        CREATE TABLE IF NOT EXISTS snapshots (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT    NOT NULL,
            filename        TEXT    NOT NULL,
            filepath        TEXT    NOT NULL,
            incident_type   TEXT,
            incident_id     INTEGER REFERENCES incidents(id),
            file_size_kb    REAL
        );

        CREATE TABLE IF NOT EXISTS heatmap_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT    NOT NULL,
            zones_detected  INTEGER,
            max_density     REAL,
            heatmap_path    TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_incidents_ts   ON incidents(timestamp);
        CREATE INDEX IF NOT EXISTS idx_plates_ts      ON plate_reads(timestamp);
        CREATE INDEX IF NOT EXISTS idx_plates_text    ON plate_reads(plate_text);
        """)
        self._conn.commit()

    # ----------------------------------------------------------
    # Logging methods
    # ----------------------------------------------------------

    def log_incident(self,
                     incident_type: str,
                     confidence: float = 0.0,
                     snapshot_path: str = None,
                     zone_names: list = None,
                     person_count: int = 0,
                     extra: dict = None) -> int:
        """
        Log any detected incident. Returns the new row ID.
        """
        ts    = datetime.now().isoformat(sep=' ', timespec='seconds')
        zones = json.dumps(zone_names) if zone_names else None
        extra_json = json.dumps(extra) if extra else None

        cur = self._conn.execute(
            """INSERT INTO incidents
               (timestamp, type, confidence, zone_names,
                person_count, snapshot_path, extra_json)
               VALUES (?,?,?,?,?,?,?)""",
            (ts, incident_type, confidence, zones,
             person_count, snapshot_path, extra_json)
        )
        self._conn.commit()
        row_id = cur.lastrowid
        print(f"[DataStore] Incident logged: {incident_type} id={row_id}")
        return row_id

    def log_plate(self,
                  plate_text: str,
                  vehicle_type: str = "Unknown",
                  confidence: float = 0.0,
                  snapshot_path: str = None,
                  incident_id: int = None) -> int:
        """Log a license plate read. Returns new row ID."""
        ts = datetime.now().isoformat(sep=' ', timespec='seconds')
        cur = self._conn.execute(
            """INSERT INTO plate_reads
               (timestamp, plate_text, vehicle_type, confidence,
                snapshot_path, incident_id)
               VALUES (?,?,?,?,?,?)""",
            (ts, plate_text, vehicle_type, confidence,
             snapshot_path, incident_id)
        )
        self._conn.commit()
        print(f"[DataStore] Plate logged: {plate_text} ({vehicle_type})")
        return cur.lastrowid

    def log_snapshot(self,
                     filepath: str,
                     incident_type: str,
                     incident_id: int = None) -> int:
        """Log a saved snapshot. Returns new row ID."""
        ts  = datetime.now().isoformat(sep=' ', timespec='seconds')
        fn  = os.path.basename(filepath)
        try:
            size_kb = os.path.getsize(filepath) / 1024
        except OSError:
            size_kb = 0.0
        cur = self._conn.execute(
            """INSERT INTO snapshots
               (timestamp, filename, filepath, incident_type,
                incident_id, file_size_kb)
               VALUES (?,?,?,?,?,?)""",
            (ts, fn, filepath, incident_type, incident_id, round(size_kb, 1))
        )
        self._conn.commit()
        return cur.lastrowid

    def log_heatmap(self,
                    zones_detected: int,
                    max_density: float,
                    heatmap_path: str = None):
        ts = datetime.now().isoformat(sep=' ', timespec='seconds')
        self._conn.execute(
            """INSERT INTO heatmap_log
               (timestamp, zones_detected, max_density, heatmap_path)
               VALUES (?,?,?,?)""",
            (ts, zones_detected, max_density, heatmap_path)
        )
        self._conn.commit()

    # ----------------------------------------------------------
    # Query helpers
    # ----------------------------------------------------------

    def get_recent_incidents(self, n: int = 50) -> list[dict]:
        cur = self._conn.execute(
            "SELECT * FROM incidents ORDER BY id DESC LIMIT ?", (n,)
        )
        return [dict(row) for row in cur.fetchall()]

    def get_incidents_last_hours(self, hours: int = 24) -> list[dict]:
        since = (datetime.now() - timedelta(hours=hours)).isoformat(
            sep=' ', timespec='seconds'
        )
        cur = self._conn.execute(
            "SELECT * FROM incidents WHERE timestamp >= ? ORDER BY id DESC",
            (since,)
        )
        return [dict(row) for row in cur.fetchall()]

    def get_recent_plates(self, n: int = 20) -> list[dict]:
        cur = self._conn.execute(
            "SELECT * FROM plate_reads ORDER BY id DESC LIMIT ?", (n,)
        )
        return [dict(row) for row in cur.fetchall()]

    def search_plate(self, plate_text: str) -> list[dict]:
        cur = self._conn.execute(
            "SELECT * FROM plate_reads WHERE plate_text LIKE ? ORDER BY id DESC",
            (f"%{plate_text.upper()}%",)
        )
        return [dict(row) for row in cur.fetchall()]

    def get_stats(self) -> dict:
        """Summary statistics for the dashboard."""
        stats = {}
        for itype in ["FIGHT", "SNATCHING", "WEAPON_DISPLAYED",
                      "ACCIDENT", "CHILD_IN_DANGER_ZONE",
                      "PERSON_IN_ZONE", "PERSON_LYING_DOWN"]:
            cur = self._conn.execute(
                "SELECT COUNT(*) FROM incidents WHERE type=?", (itype,)
            )
            stats[itype] = cur.fetchone()[0]

        cur = self._conn.execute("SELECT COUNT(*) FROM plate_reads")
        stats["PLATES_READ"] = cur.fetchone()[0]

        cur = self._conn.execute("SELECT COUNT(*) FROM snapshots")
        stats["SNAPSHOTS"] = cur.fetchone()[0]

        cur = self._conn.execute(
            "SELECT COUNT(*) FROM incidents WHERE timestamp >= ?",
            ((datetime.now() - timedelta(hours=24)).isoformat(
                sep=' ', timespec='seconds'),)
        )
        stats["INCIDENTS_24H"] = cur.fetchone()[0]

        return stats

    # ----------------------------------------------------------
    # CSV Export
    # ----------------------------------------------------------

    def export_incidents_csv(self, filename: str = None) -> str:
        """Export all incidents to CSV. Returns the file path."""
        if filename is None:
            ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"incidents_{ts}.csv"
        filepath = os.path.join(EXPORT_DIR, filename)

        rows = self._conn.execute(
            "SELECT * FROM incidents ORDER BY timestamp"
        ).fetchall()

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "ID", "Timestamp", "Type", "Confidence",
                "Zones", "Persons", "Snapshot", "Extra", "Alerted"
            ])
            for row in rows:
                writer.writerow(list(row))

        print(f"[DataStore] Incidents exported to {filepath}")
        return filepath

    def export_plates_csv(self, filename: str = None) -> str:
        """Export all plate reads to CSV."""
        if filename is None:
            ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"plates_{ts}.csv"
        filepath = os.path.join(EXPORT_DIR, filename)

        rows = self._conn.execute(
            "SELECT * FROM plate_reads ORDER BY timestamp"
        ).fetchall()

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "ID", "Timestamp", "Plate", "Vehicle Type",
                "Confidence", "Snapshot", "Incident ID"
            ])
            for row in rows:
                writer.writerow(list(row))

        print(f"[DataStore] Plates exported to {filepath}")
        return filepath

    # ----------------------------------------------------------
    # Image gallery index (for the web dashboard)
    # ----------------------------------------------------------

    def build_gallery_index(self) -> list[dict]:
        """
        Scans the GALLERY_DIR for snapshots and returns a sorted
        list of {filename, filepath, timestamp, incident_type}.
        """
        gallery = []
        if not os.path.isdir(GALLERY_DIR):
            return gallery

        for fn in sorted(os.listdir(GALLERY_DIR), reverse=True):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            # Parse incident type from filename (e.g. FIGHT_20250417_...)
            parts     = fn.split("_")
            inc_type  = parts[0] if parts else "UNKNOWN"
            filepath  = os.path.join(GALLERY_DIR, fn)
            gallery.append({
                "filename"     : fn,
                "filepath"     : filepath,
                "incident_type": inc_type,
                "timestamp"    : _mtime_str(filepath),
            })
        return gallery

    # ----------------------------------------------------------
    # Cleanup
    # ----------------------------------------------------------

    def close(self):
        self._conn.close()
        print("[DataStore] Database closed.")

    def __del__(self):
        try:
            self._conn.close()
        except Exception:
            pass


def _mtime_str(filepath: str) -> str:
    try:
        ts = os.path.getmtime(filepath)
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except OSError:
        return ""
