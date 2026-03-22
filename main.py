import argparse
import json
import signal
import sys
import threading
import time
import cv2

from database import Database
from detector import FaceDetector
from embedder import FaceEmbedder
from event_handler import EventHandler
from logger import setup_logger
from registry import FaceRegistry
from tracker import IoUTracker
from video_quality_analyzer import VideoQualityAnalyzer
import dashboard.app  # Import as module to access video_source

_shutdown = False


def _handle_signal(sig, frame):
    global _shutdown
    _shutdown = True
    print("\n[main] Shutdown requested – finishing current frame…")


def load_config(path: str = "config.json") -> dict:
    with open(path, "r") as f:
        return json.load(f)


def safe_visitor_count(event_handler):
    try:
        return event_handler.unique_visitor_count()
    except:
        return 0


def apply_dynamic_parameters(detector, tracker, registry, params):
    """
    Apply dynamically calculated parameters to components.
    
    Args:
        detector: FaceDetector instance
        tracker: IoUTracker instance
        registry: FaceRegistry instance
        params: Dict of recommended parameters
    """
    detector.conf = params["confidence_threshold"]
    detector.frame_skip = params["frame_skip"]
    detector.input_size = params["input_size"]
    tracker.iou_threshold = params["iou_threshold"]
    registry.threshold = params["similarity_threshold"]
    
    print(f"\n{'=' * 50}")
    print("🎬 DYNAMIC PARAMETERS APPLIED")
    print(f"{'=' * 50}")
    print(f"Quality Grade            : {params['quality_grade'].upper()}")
    print(f"Confidence Threshold     : {params['confidence_threshold']:.2f}")
    print(f"Frame Skip               : {params['frame_skip']}")
    print(f"Input Size               : {params['input_size']}")
    print(f"IoU Threshold            : {params['iou_threshold']:.2f}")
    print(f"Similarity Threshold     : {params['similarity_threshold']:.2f}")
    print(f"{'=' * 50}\n")


def draw_overlays(frame, tracks, active_faces: dict, unique_count: int):
    for track in tracks:
        x1, y1, x2, y2 = track.bbox
        color = (0, 200, 80) if track.confirmed else (0, 165, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = track.face_id if track.face_id else f"track_{track.track_id}"
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

    cv2.putText(
        frame,
        f"Unique visitors: {unique_count} | In frame: {len(active_faces)}",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    return frame


def main():
    global _shutdown

    parser = argparse.ArgumentParser(description="Face Tracker")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--source", default=None)
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)

    log_cfg = cfg["logging"]
    app_logger = setup_logger(log_cfg["log_file"], log_cfg.get("log_level", "INFO"))

    app_logger.info("Face Tracker starting...")

    # ── DATABASE ───────────────────────────────
    db_cfg = cfg["database"]
    db = Database(
        dsn=db_cfg["dsn"],
        min_conn=db_cfg.get("min_connections", 2),
        max_conn=db_cfg.get("max_connections", 10),
    )

    # ── COMPONENTS ─────────────────────────────
    det_cfg = cfg["detection"]
    detector = FaceDetector(
        model_path=det_cfg["model"],
        conf=det_cfg["confidence_threshold"],
        input_size=det_cfg["input_size"],
        frame_skip=det_cfg["frame_skip"],
    )

    embedder = FaceEmbedder(model_name=cfg["recognition"]["model_name"])

    registry = FaceRegistry(
        db=db,
        similarity_threshold=cfg["recognition"]["similarity_threshold"],
    )

    trk_cfg = cfg["tracking"]
    tracker = IoUTracker(
        max_disappeared=trk_cfg["max_disappeared_frames"],
        iou_threshold=trk_cfg["iou_threshold"],
    )

    # ── VIDEO QUALITY ANALYZER ─────────────────
    quality_analyzer = VideoQualityAnalyzer()

    event_handler = EventHandler(
        tracker,
        embedder,
        registry,
        db,
        log_cfg["image_base_dir"],
        log_cfg.get("image_format", "jpg"),
    )

    # ── FLASK DASHBOARD ───────────────────────
    if cfg.get("dashboard", {}).get("enabled"):
        from dashboard.app import create_app
        import dashboard.app as dashboard_app

        app = create_app(db)
        
        # 🔥 Expose tracker and event_handler to dashboard for stats
        dashboard_app.tracker_instance = tracker
        dashboard_app.event_handler_instance = event_handler
        
        threading.Thread(
            target=app.run,
            kwargs={
                "host": "0.0.0.0",
                "port": 5000,
                "debug": False,
                "use_reloader": False,
            },
            daemon=True,
        ).start()

    # ── VIDEO SOURCE ──────────────────────────
    # Wait for video upload instead of using default
    cap = None
    current_source = None
    display = not args.no_display

    signal.signal(signal.SIGINT, _handle_signal)

    print("[INFO] Waiting for video upload on dashboard...")

    # ── MAIN LOOP ─────────────────────────────
    while not _shutdown:

        # 🔥 WAIT FOR VIDEO UPLOAD
        if not current_source or (dashboard.app.video_source and dashboard.app.video_source != current_source):
            if dashboard.app.video_source:
                print(f"\n[INFO] Loading video: {dashboard.app.video_source}")
                if cap:
                    cap.release()
                cap = cv2.VideoCapture(dashboard.app.video_source)
                
                if not cap.isOpened():
                    print(f"[ERROR] Cannot open video: {dashboard.app.video_source}")
                    dashboard.app.video_source = None
                    time.sleep(1)
                    continue
                
                current_source = dashboard.app.video_source
                
                # ──── ANALYZE VIDEO QUALITY & APPLY DYNAMIC PARAMETERS ────
                print("\n[INFO] Analyzing video quality...")
                quality_params = quality_analyzer.analyze_video(cap, sample_frames=5)
                apply_dynamic_parameters(detector, tracker, registry, quality_params)
                
                # ──── RESET STATE FOR NEW VIDEO ────
                detector._frame_count = 0
                detector._last_results = []
                tracker.tracks.clear()
                tracker._next_id = 0
                event_handler._active.clear()
                event_handler._prev_track_ids.clear()
                event_handler._recently_identified.clear()
                print("[INFO] Detector and tracker reset for new video\n")
            else:
                time.sleep(0.5)
                continue
        
        ret, frame = cap.read()

        if not ret:
            print("[INFO] End of video stream")
            cap.release()
            cap = None
            current_source = None
            dashboard.app.video_source = None
            print("[INFO] Waiting for next video upload...")
            time.sleep(1)
            continue

        # Detection
        detections = detector.detect(frame)
        tracks = tracker.update(detections)

        # Recognition + events
        event_handler.process_frame(frame, tracks)

        # Visitor count
        count = safe_visitor_count(event_handler)

        # Draw overlays
        annotated = draw_overlays(
            frame.copy(),
            tracks,
            event_handler._active,
            count,
        )

        # 🔥 SEND FRAME TO DASHBOARD
        dashboard.app.set_latest_frame(annotated)

        # Optional OpenCV window
        if display:
            cv2.imshow("Face Tracker", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # ── CLEANUP ───────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    db.close()


if __name__ == "__main__":
    main()