# app.py
from flask import Flask, render_template, Response, request, jsonify, send_from_directory
import os
from gesture_control import controller

app = Flask(__name__, static_folder="static", template_folder="templates")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/gallery")
def gallery():
    shots_dir = controller.config["screenshot_dir"]
    files = []
    if os.path.isdir(shots_dir):
        files = sorted([f for f in os.listdir(shots_dir) if f.lower().endswith(".png")], reverse=True)
    return render_template("gallery.html", files=files)

# ---- Controls ----
@app.route("/api/start", methods=["POST"])
def api_start():
    controller.start()
    return jsonify(controller.status())

@app.route("/api/stop", methods=["POST"])
def api_stop():
    controller.stop()
    return jsonify(controller.status())

@app.route("/api/status")
def api_status():
    return jsonify(controller.status())

@app.route("/api/config", methods=["GET", "POST"])
def api_config():
    if request.method == "POST":
        data = request.json or {}
        controller.update_config(**data)
    return jsonify(controller.config)

# ---- Stream camera frames to browser ----
@app.route("/video")
def video():
    def gen():
        while True:
            frame = controller.get_jpeg_frame()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

# Serve screenshots (already under /static), helper if needed
@app.route("/screenshots/<path:fname>")
def screenshots(fname):
    return send_from_directory(controller.config["screenshot_dir"], fname)

if __name__ == "__main__":
    # Expose on LAN too; remove host param if you only want localhost
    app.run(host="0.0.0.0", port=5000, debug=True)
