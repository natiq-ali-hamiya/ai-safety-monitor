# 🛡️ AI Safety & Surveillance Command Center

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF6F00.svg)](https://docs.ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![Deploy on Render](https://img.shields.io/badge/Deploy-Render-46E3B7.svg?logo=render&logoColor=black)](https://render.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)

An AI-powered multi-camera surveillance and incident response system designed for school campuses, public areas, and smart city infrastructure. It integrates real-time computer vision threat detection (weapons, fighting, child danger zones, falls) with a unified FastAPI backend, SQLite/Supabase persistence, and a real-time web command center.

---

## 🌟 Key Capabilities & Modules

| Module | Core Technology | Threat Detection & Actions |
| :--- | :--- | :--- |
| 🚸 **Child Safety & Virtual Fence** | YOLOv8 + Polygon Zone Mapping | Detects unsupervised children entering danger zones (roads, pools, gates); triggers parent/guard sirens. |
| 🔪 **Crime & Weapon Detection** | YOLOv8 + Custom Weapon Weights | Detects firearms, knives, physical fights, and snatching; flags critical incidents for operator verification. |
| 🚗 **Hit & Run / Vehicle Tracker** | ByteTrack + OCR Plate Reader | Tracks vehicle intrusions, captures license plates, and logs timestamps for forensic evidence. |
| 🚑 **Medical / Fall Emergency** | Pose Estimation (Keypoints) | Detects unresponsive individuals lying down $>30\text{s}$; triggers 1122 emergency ambulance alerts. |
| 🖥️ **Command Center & Alerts** | FastAPI + Web UI + SMTP/Twilio | Real-time incident verification, emergency dispatch (SMS/Email), camera management, and analytics. |

---

## 📐 System Architecture

```mermaid
graph TD
    A[CCTV / Video Streams] --> B[AI Computer Vision Engine]
    B -->|YOLOv8 Detection| C{Threat Analyzer}
    
    C -->|Weapon / Fight| D[Crime Module]
    C -->|Child in Perimeter| E[Child Safety Module]
    C -->|Fall / Inactive| F[Medical 1122 Module]
    C -->|Vehicle / Speed| G[Plate Reader Module]
    
    D & E & F & G --> H[Cloud Reporter / Evidence Manager]
    H -->|REST API + JWT| I[FastAPI Cloud Server]
    
    I -->|Dual-Engine| J[(SQLite Local DB)]
    I -->|Dual-Engine| K[(Supabase PostgreSQL Cloud)]
    
    I --> L[Web Command Center UI]
    I --> M[Alert Dispatcher: Email SMTP / Twilio SMS]


## 📄 License
MIT License. Developed for University Academic Research 
