# Intelligent Face Tracker — Auto-Registration & Visitor Counting

> **This project is a part of a hackathon run by https://katomaran.com**

---

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Feature List](#feature-list)
4. [Compute Estimates](#compute-estimates)
5. [Setup — Windows + PostgreSQL](#setup--windows--postgresql)
6. [Configuration Reference](#configuration-reference)
7. [Running the System](#running-the-system)
8. [Dashboard](#dashboard)
9. [Output Samples](#output-samples)
10. [Assumptions](#assumptions)
11. [Demo Video](#demo-video)

---

## Overview

A real-time AI surveillance pipeline that:
- Detects faces with **YOLOv8**
- Generates 512-d face embeddings via **InsightFace (ArcFace / buffalo_l)**
- Tracks faces across frames with an **IoU multi-object tracker**
- Auto-registers new faces and re-identifies returning ones via **cosine similarity**
- Logs every **entry** and **exit** with a cropped image + timestamp
- Counts **unique visitors** accurately (no double-counting across frames or restarts)
- Stores all data in **PostgreSQL** with a thread-safe connection pool
- Provides a **live web dashboard** at `http://localhost:5000`

---

## Architecture

```
Video File / RTSP Stream
          │
          ▼
┌──────────────────────────────────────────────────────┐
│                    Core Pipeline                     │
│                                                      │
│  ┌──────────────────┐   ┌────────────────────────┐   │
│  │  YOLOv8 Detector │──▶│  InsightFace Embedder  │   │
│  │  (every N frames)│   │  (ArcFace 512-d)       │   │
│  └──────────────────┘   └────────────────────────┘   │
│           │                         │                │
│           ▼                         ▼                │
│  ┌──────────────────┐   ┌────────────────────────┐   │
│  │   IoU Tracker    │──▶│    Face Registry       │   │
│  │  (cross-frame)   │   │  (cosine similarity)   │   │
│  └──────────────────┘   └────────────────────────┘   │
│                    │                                  │
│                    ▼                                  │
│           Entry / Exit Detector                       │
└──────────────────────────────────────────────────────┘
        │              │                │
        ▼              ▼                ▼
  PostgreSQL DB    events.log    logs/entries/YYYY-MM-DD/
        │
        ▼
  Flask Dashboard (http://localhost:5000)
```

---

## Feature List

| # | Feature | Module |
|---|---------|--------|
| 1 | YOLOv8 real-time face detection | `detector.py` |
| 2 | Configurable frame-skip (N frames between detections) | `config.json` |
| 3 | ArcFace 512-d embedding generation | `embedder.py` |
| 4 | Cosine similarity face matching | `registry.py` |
| 5 | Auto-registration of new faces with UUID | `registry.py` |
| 6 | Persistent identity across restarts (loaded from DB) | `registry.py` |
| 7 | IoU-based cross-frame multi-object tracking | `tracker.py` |
| 8 | One entry event per unique visit | `event_handler.py` |
| 9 | One exit event when track disappears | `event_handler.py` |
| 10 | Cropped face image saved per event | `logger.py` |
| 11 | Timestamped `events.log` text log | `logger.py` |
| 12 | PostgreSQL storage (thread-safe pool) | `database.py` |
| 13 | Unique visitor count with no double-counting | `registry.py` |
| 14 | Live Flask dashboard — MJPEG + stats API | `dashboard/` |
| 15 | Surveillance-aesthetic UI with event stream + chart | `dashboard/templates/index.html` |
| 16 | RTSP stream support | `main.py` |
| 17 | Windows-compatible (no SIGTERM, `psycopg2-binary`) | `main.py`, `requirements.txt` |

---

## Compute Estimates

### CPU-only (Windows laptop — no GPU)

| Stage | CPU Usage | Approx. throughput |
|-------|-----------|-------------------|
| YOLOv8n-face inference | 30–45% | ~10–15 FPS |
| InsightFace ArcFace (ONNX CPU) | 20–30% | ~8–12 FPS |
| IoU Tracker | <1% | negligible |
| PostgreSQL writes | <1% | negligible |
| Flask dashboard | ~2% | background thread |
| **Total (CPU)** | **~55–75%** | Comfortable on 8-core i7 |

### GPU-accelerated (NVIDIA CUDA)

| Stage | VRAM | Throughput |
|-------|------|------------|
| YOLOv8n-face | ~400 MB | 60–120 FPS |
| InsightFace (ONNX-GPU) | ~500 MB | 40–80 FPS |
| **Total** | **~1 GB** | Real-time at 1080p |

To enable GPU: replace `onnxruntime` with `onnxruntime-gpu` in `requirements.txt`.

---

## Setup — Windows + PostgreSQL

### Prerequisites
- Windows 10/11
- Python 3.9+ ([python.org](https://www.python.org/downloads/))
- PostgreSQL 14+ ([postgresql.org](https://www.postgresql.org/download/windows/))

### 1. Clone the repo
```cmd
git clone <your-repo-url>
cd face_tracker
```

### 2. Run the setup script (automated)
```cmd
setup_windows.bat
```
This creates a `venv`, installs all packages, and downloads the YOLO weights.

### 3. Create the PostgreSQL database
Open **pgAdmin** or **psql** and run:
```sql
CREATE DATABASE face_tracker;
```

### 4. Update config.json
Edit the `database.dsn` field:
```json
"database": {
  "dsn": "host=localhost port=5432 dbname=face_tracker user=postgres password=YOUR_PASSWORD"
}
```

### 5. Place your video
```cmd
copy C:\path\to\your\video.mp4 sample_video.mp4
```

### 6. Activate venv and run
```cmd
venv\Scripts\activate.bat
python main.py
```

---

## Configuration Reference

```jsonc
{
  "video_source": "sample_video.mp4",  // local video file
  "rtsp_url":     "rtsp://...",        // live camera stream
  "use_rtsp":     false,               // true → prefer rtsp_url

  "detection": {
    "model":                "yolov8n-face.pt",
    "confidence_threshold": 0.5,    // 0–1; lower = more detections
    "frame_skip":           3,      // run YOLO every 3 frames
    "input_size":           640     // inference resolution
  },

  "recognition": {
    "model_name":           "buffalo_l",  // InsightFace model pack
    "similarity_threshold": 0.45,         // cosine sim cutoff for match
    "embedding_size":       512
  },

  "tracking": {
    "max_disappeared_frames": 30,   // frames before track pruned (~1 s @ 30 fps)
    "iou_threshold":          0.3,  // min IoU to link bbox between frames
    "use_bytetrack":          true
  },

  "logging": {
    "log_file":      "logs/events.log",
    "image_base_dir":"logs/entries",
    "image_format":  "jpg",
    "log_level":     "INFO"
  },

  "database": {
    "dsn":             "host=localhost port=5432 dbname=face_tracker user=postgres password=...",
    "min_connections": 2,
    "max_connections": 10
  },

  "display":   { "show_video": true, "draw_bboxes": true, "draw_ids": true },

  "dashboard": { "enabled": true, "host": "0.0.0.0", "port": 5000 }
}
```

---

## Running the System

```cmd
REM Activate virtualenv first
venv\Scripts\activate.bat

REM Default (video file)
python main.py

REM Override video source
python main.py --source C:\videos\crowd.mp4

REM RTSP live camera
python main.py --source rtsp://192.168.1.100:554/stream

REM No OpenCV window (headless)
python main.py --no-display

REM Custom config file
python main.py --config my_config.json
```

Press **Q** in the video window, or **Ctrl+C** in the terminal, to stop.

---

## Dashboard

Open **http://localhost:5000** in any browser while the tracker is running.

Features:
- **Live MJPEG feed** with corner-bracket overlay
- **Unique Visitor counter** — total distinct faces seen
- **Active count** — currently visible faces
- **Entry / Exit totals** for today
- **Real-time event stream** — every entry and exit with face ID + timestamp
- **7-day bar chart** of daily unique visitors

---

## Output Samples

### logs/events.log
```
2025-03-21 10:14:03 | INFO     | Face Tracker starting up (Windows / PostgreSQL)
2025-03-21 10:14:05 | INFO     | FaceDetector ready | model=yolov8n-face.pt conf=0.50 frame_skip=3
2025-03-21 10:14:06 | INFO     | FaceEmbedder ready | model=buffalo_l
2025-03-21 10:14:06 | INFO     | FaceRegistry ready | 0 known face(s) loaded | threshold=0.45
2025-03-21 10:14:07 | INFO     | New face registered as face_3a7f1c2d
2025-03-21 10:14:07 | INFO     | ENTRY | face_id=face_3a7f1c2d track_id=0 new=True
2025-03-21 10:14:09 | INFO     | Recognised face face_3a7f1c2d (sim=0.821)
2025-03-21 10:14:22 | INFO     | EXIT  | face_id=face_3a7f1c2d
```

### Image folder
```
logs/
└── entries/
    └── 2025-03-21/
        ├── entry_face_3a7f1c2d_101407_123.jpg
        ├── entry_face_a91b4d0e_101412_456.jpg
        └── entry_face_c234f8b1_101418_789.jpg
```

### PostgreSQL — faces table
```
 id              | first_seen           | last_seen            | visit_count
-----------------+----------------------+----------------------+------------
 face_3a7f1c2d   | 2025-03-21 10:14:07  | 2025-03-21 10:14:09  |     2
 face_a91b4d0e   | 2025-03-21 10:14:12  | 2025-03-21 10:14:12  |     1
```

### PostgreSQL — events table
```
 id | face_id        | event_type | timestamp            | image_path
----+----------------+------------+----------------------+------------------
  1 | face_3a7f1c2d  | entry      | 2025-03-21 10:14:07  | logs/entries/…
  2 | face_3a7f1c2d  | exit       | 2025-03-21 10:14:22  | null
```

---

## Assumptions

1. **Similarity threshold 0.45** — empirically tuned. Lower → same person may be re-registered. Higher → different people merged. Adjust in `config.json`.
2. **Entry fires on first confirmed embedding** — partial/edge-of-frame detections that yield no embedding do not generate events.
3. **Exit fires after `max_disappeared_frames`** — approximately 1 second at 30 FPS. No reliable "walked out" signal exists in a monocular camera setup.
4. **psycopg2-binary** used instead of psycopg2 — avoids needing a C compiler on Windows.
5. **Embeddings stored in PostgreSQL as BYTEA** — allows registry to survive restarts and re-identify returning visitors.
6. **Single GPU not assumed** — defaults to CPU ONNX. Swap to `onnxruntime-gpu` for acceleration.

---

## Demo Video

🎥 **[Watch demo on YouTube / Loom]** — _add link before submission_

---

*This project is a part of a hackathon run by https://katomaran.com*
