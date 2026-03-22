import threading
import time
import os
import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from flask_cors import CORS

_latest_frame: np.ndarray = None
_frame_lock = threading.Lock()

UPLOAD_PATH = "uploads"
os.makedirs(UPLOAD_PATH, exist_ok=True)

video_source = None  # 🔥 dynamic source


def set_latest_frame(frame: np.ndarray):
    global _latest_frame
    with _frame_lock:
        _latest_frame = frame.copy()


def create_app(db):
    app = Flask(__name__, template_folder="templates", static_folder="static")
    CORS(app)

    def _mjpeg_generator():
        while True:
            with _frame_lock:
                frame = _latest_frame

            if frame is None:
                time.sleep(0.05)
                continue

            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" +
                   buffer.tobytes() + b"\r\n")

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/video_feed")
    def video_feed():
        return Response(_mjpeg_generator(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/frame")
    def frame():
        """Return the latest frame as a JPEG image."""
        with _frame_lock:
            frame = _latest_frame

        if frame is None:
            # Return a blank image if no frame available
            blank = np.zeros((600, 800, 3), dtype=np.uint8)
            ret, buffer = cv2.imencode(".jpg", blank)
            return Response(buffer.tobytes(), mimetype="image/jpeg")

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            return Response(b"", mimetype="image/jpeg")

        return Response(buffer.tobytes(), mimetype="image/jpeg")

    # 🔥 VIDEO UPLOAD
    @app.route("/upload_video", methods=["POST"])
    def upload_video():
        global video_source

        file = request.files["video"]
        path = os.path.join(UPLOAD_PATH, file.filename)
        file.save(path)

        video_source = path
        print("New video source:", path)

        return jsonify({"status": "ok"})

    @app.route("/api/stats")
    def stats():
        try:
            unique = db.get_unique_visitor_count()
        except:
            unique = 0

        events = []
        try:
            for r in db.get_recent_events(20):
                events.append({
                    "face_id": r["face_id"],
                    "event_type": r["event_type"],
                    "timestamp": str(r["timestamp"]),
                })
        except:
            pass

        return jsonify({
            "unique_visitors": unique,
            "recent_events": events
        })

    return app