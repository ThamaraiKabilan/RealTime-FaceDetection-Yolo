from flask import Flask, render_template, Response, request, jsonify
import cv2
import os
from face_utils import FaceRecognitionSystem
import numpy as np
import traceback

app = Flask(__name__)
face_system = FaceRecognitionSystem()

# Global variables
camera = None
detection_active = False
detection_history = []

def generate_frames():
    """Captures frames, runs recognition, and streams the video."""
    global camera, detection_active
    while True:
        if not (camera and detection_active):
            import time
            time.sleep(0.1)
            continue

        success, frame = camera.read()
        if not success or frame is None:
            print("❌ Could not read frame from camera.")
            break

        try:
            processed_frame, face_info = face_system.recognize_faces(frame)
            if face_info:
                update_detection_history(face_info)
        except Exception as e:
            print(f"🔥🔥🔥 ERROR during face recognition processing: {e}")
            traceback.print_exc()
            processed_frame = frame
            cv2.putText(processed_frame, "PROCESSING ERROR", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def update_detection_history(face_info_list):
    """Adds new detections to the global history list for the UI."""
    global detection_history
    from datetime import datetime
    for info in face_info_list:
        is_duplicate = any(
            old['name'] == info['name'] and
            (datetime.now() - datetime.strptime(old['time'], "%Y-%m-%d %H:%M:%S")).total_seconds() < 5
            for old in reversed(detection_history)
        )
        if not is_duplicate:
            detection_history.append(info)
            if len(detection_history) > 100:
                detection_history.pop(0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera')
def start_camera():
    global camera, detection_active
    if camera is None:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            camera = None
            return jsonify({"status": "error", "message": "Cannot access camera."}), 500
    detection_active = True
    return jsonify({"status": "camera started"})

@app.route('/stop_camera')
def stop_camera():
    global camera, detection_active
    detection_active = False
    if camera:
        camera.release()
        camera = None
    return jsonify({"status": "camera stopped"})

# --- THIS IS THE MODIFIED ROUTE ---
@app.route('/add_face', methods=['POST'])
def add_face():
    """Adds a new face to the system from an uploaded file."""
    if 'image' not in request.files or 'name' not in request.form:
        return jsonify({"status": "error", "message": "Missing name or image file."}), 400

    name = request.form['name'].strip()
    file = request.files['image']

    if not name or file.filename == '':
        return jsonify({"status": "error", "message": "Name or file cannot be empty."}), 400

    try:
        # Read the image file stream and convert it to a CV2 image
        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"status": "error", "message": "Could not decode image file."}), 400

        success = face_system.add_new_face(image, name)
        if success:
            return jsonify({"status": "success", "message": f"Face '{name}' added. Embeddings are being recalculated."})
        else:
            return jsonify({"status": "error", "message": "Failed to save face image."}), 500

    except Exception as e:
        print(f"🔥🔥🔥 ERROR in add_face: {e}")
        return jsonify({"status": "error", "message": f"An error occurred: {e}"}), 500

# --- HISTORY AND STATS ROUTES ---
@app.route('/get_detections')
def get_detections():
    return jsonify(detection_history)

@app.route('/detection_history')
def detection_history_page():
    return render_template('history.html')

@app.route('/api/detection_history')
def api_detection_history():
    history = face_system.database.get_detection_history(limit=200)
    return jsonify(history)

@app.route('/api/statistics')
def api_statistics():
    stats = face_system.database.get_statistics()
    return jsonify(stats)

@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    success = face_system.database.clear_history()
    if success:
        return jsonify({"status": "success", "message": "History cleared!"})
    else:
        return jsonify({"status": "error", "message": "Failed to clear history."}), 500

@app.route('/get_known_people')
def get_known_people():
    people = face_system.get_known_people()
    return jsonify(people)

if __name__ == '__main__':
    print("🌐 Starting Flask web server at http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)