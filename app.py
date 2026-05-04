from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import mediapipe as mp
import numpy as np
import time
import os
import webbrowser
from threading import Timer

app = Flask(__name__)

# ---------------- Folder for snapshots ----------------
ALERT_FOLDER = os.path.join("static", "alerts")
os.makedirs(ALERT_FOLDER, exist_ok=True)

messages = {
    "English": "You look sleepy. Please take rest for a few minutes.",
    "Hindi": "आपको नींद आ रही है। कृपया कुछ मिनट आराम करें।",
    "Telugu": "మీకు నిద్ర వస్తోంది. దయచేసి కొన్ని నిమిషాలు విశ్రాంతి తీసుకోండి.",
    "Kannada": "ನೀವು ನಿದ್ರಾವಸ್ಥೆಯಲ್ಲಿ ಇದ್ದೀರಿ. ದಯವಿಟ್ಟು ಕೆಲವು ನಿಮಿಷ ವಿಶ್ರಾಂತಿ ತೆಗೆದುಕೊಳ್ಳಿ."
}

selected_language = "English"

# ---------------- MediaPipe Face Mesh ----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])

    if C == 0:
        return 0.0

    return (A + B) / (2.0 * C)

# ---------------- Global Status ----------------
status_data = {
    "status": "Starting...",
    "language": selected_language,
    "alerts": 0,
    "last_snapshot": ""
}

# ---------------- Camera ----------------
cap = cv2.VideoCapture(0)

def generate_frames():
    global selected_language

    EAR_THRESHOLD = 0.18          # eye closed threshold
    DROWSY_SECONDS = 2.0          # must stay closed for this many seconds
    ALERT_COOLDOWN = 3.0          # avoid repeated alerts quickly

    eyes_closed_start = None
    last_alert_time = 0

    while True:
        success, frame = cap.read()
        if not success:
            status_data["status"] = "Camera Error"
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            h, w, _ = frame.shape

            mesh_points = np.array([
                [int(lm.x * w), int(lm.y * h)]
                for lm in face_landmarks.landmark
            ])

            left_eye = mesh_points[LEFT_EYE]
            right_eye = mesh_points[RIGHT_EYE]

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            current_time = time.time()

            if ear < EAR_THRESHOLD:
                if eyes_closed_start is None:
                    eyes_closed_start = current_time

                closed_duration = current_time - eyes_closed_start

                if closed_duration >= DROWSY_SECONDS:
                    status_data["status"] = "Sleepy"

                    cv2.putText(
                        frame,
                        "DROWSINESS ALERT!",
                        (40, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3
                    )

                    if current_time - last_alert_time > ALERT_COOLDOWN:
                        timestamp = int(current_time)
                        filename = f"driver_alert_{timestamp}.jpg"
                        filepath = os.path.join(ALERT_FOLDER, filename)
                        cv2.imwrite(filepath, frame)

                        status_data["alerts"] += 1
                        status_data["last_snapshot"] = filename
                        last_alert_time = current_time
                else:
                    status_data["status"] = "Eyes Closed"
                    cv2.putText(
                        frame,
                        "Eyes Closed",
                        (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        3
                    )
            else:
                eyes_closed_start = None
                status_data["status"] = "Awake"

                cv2.putText(
                    frame,
                    "Driver Awake",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    3
                )
        else:
            eyes_closed_start = None
            status_data["status"] = "No Face Detected"

            cv2.putText(
                frame,
                "No Face Detected",
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

        status_data["language"] = selected_language

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

@app.route("/")
def index():
    return render_template(
        "index.html",
        data=status_data,
        languages=list(messages.keys()),
        selected_language=selected_language
    )

@app.route("/set_language", methods=["POST"])
def set_language():
    global selected_language

    language = request.form.get("language")
    if language in messages:
        selected_language = language
        status_data["language"] = selected_language

    return redirect(url_for("index"))

@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run(debug=True, use_reloader=False)
    