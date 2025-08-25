from flask import Flask, render_template, Response
from gesture_control import GestureController

app = Flask(__name__)
gesture_controller = GestureController()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gesture_controller.generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
