from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import mediapipe as mp
import joblib
from tensorflow.keras.models import load_model
from collections import deque
import math
import threading
import time

app = Flask(__name__)

pause_event = threading.Event()
pause_event.set()

status_lock = threading.Lock()

detection_status = {
    "left_eye": "N/A", "right_eye": "N/A", "yawn_status": "PAUSED", "mar": 0.0,
    "pitch": 0.0, "yaw": 0.0, "roll": 0.0, "alert": False, "alert_reason": "",
    "is_paused": True
}

HEAD_DOWN_FRAMES_THRESHOLD = 20
EYES_CLOSED_FRAMES_THRESHOLD = 15
PITCH_THRESHOLD_DEG = 30
try:
    print("⏳ [Startup] Memuat model...")
    HEAD_POSE_MODELS = {'pitch': joblib.load('model/headposeModel/xgb_pitch_model.joblib'), 'yaw': joblib.load('model/headposeModel/xgb_yaw_model.joblib'), 'roll': joblib.load('model/headposeModel/xgb_roll_model.joblib')}
    EYE_STATUS_MODEL = load_model('model/eyeModel/eye_status_model.h5')
    YAWN_SVM_MODEL = joblib.load('model/yawnModel/svm_yawn_detector.joblib')
    YAWN_SCALER = joblib.load('model/yawnModel/scaler.joblib')
    print("[Startup] Semua model berhasil dimuat.")
except Exception as e:
    print(f"[Startup] Fatal: Error saat memuat model: {e}")
    exit()

EYE_IMG_SIZE = 24
EYE_CLASS_LABELS = ['Tertutup', 'Terbuka']
LEFT_EYE_IDXS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_IDXS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
MOUTH_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
MAR_WINDOW_SIZE = 20
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

print("[Startup] Melakukan warm-up pada model untuk mempercepat loading awal...")
try:
    dummy_head_features = np.zeros((1, 1401))
    HEAD_POSE_MODELS['pitch'].predict(dummy_head_features)
    print("Head pose model siap.")
    dummy_eye_image = np.zeros((1, EYE_IMG_SIZE, EYE_IMG_SIZE, 1))
    EYE_STATUS_MODEL.predict(dummy_eye_image, verbose=0)
    print("Eye status model siap.")
    dummy_yawn_features = np.zeros((1, 3))
    scaled_dummy_yawn = YAWN_SCALER.transform(dummy_yawn_features)
    YAWN_SVM_MODEL.predict(scaled_dummy_yawn)
    print("Yawn model siap.")
    print("[Startup] Aplikasi siap digunakan!")
except Exception as e:
    print(f"[Startup] Peringatan: Error saat warm-up model: {e}")

def create_feature_vector(face_landmarks):
    anchor_point = face_landmarks.landmark[1]
    p_left, p_right = face_landmarks.landmark[359], face_landmarks.landmark[130]
    scale_distance = np.linalg.norm([p_left.x - p_right.x, p_left.y - p_right.y])
    if scale_distance < 1e-6: return None
    feature_vector = []
    for i in range(468):
        if i == 1: continue
        landmark = face_landmarks.landmark[i]
        feature_vector.extend([(landmark.x - anchor_point.x) / scale_distance, (landmark.y - anchor_point.y) / scale_distance, (landmark.z - anchor_point.z) / scale_distance])
    return np.array(feature_vector)

def draw_axes(img, pitch, yaw, roll, nose_2d, size=100):
    if nose_2d is None: return img
    pitch_rad, yaw_rad, roll_rad = pitch * np.pi / 180, -(yaw * np.pi / 180), roll * np.pi / 180
    Rx = np.array([[1, 0, 0], [0, math.cos(pitch_rad), -math.sin(pitch_rad)], [0, math.sin(pitch_rad), math.cos(pitch_rad)]])
    Ry = np.array([[math.cos(yaw_rad), 0, math.sin(yaw_rad)], [0, 1, 0], [-math.sin(yaw_rad), 0, math.cos(yaw_rad)]])
    Rz = np.array([[math.cos(roll_rad), -math.sin(roll_rad), 0], [math.sin(roll_rad), math.cos(roll_rad), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    axis = R @ np.array([[size, 0, 0], [0, size, 0], [0, 0, size]]).T
    p1 = (int(nose_2d[0]), int(nose_2d[1]))
    cv2.line(img, p1, (int(p1[0] + axis[0,0]), int(p1[1] + axis[1,0])), (255, 0, 0), 3)
    cv2.line(img, p1, (int(p1[0] + axis[0,1]), int(p1[1] + axis[1,1])), (0, 255, 0), 3)
    cv2.line(img, p1, (int(p1[0] + axis[0,2]), int(p1[1] + axis[1,2])), (0, 0, 255), 3)
    return img

def preprocess_eye_for_predict(eye_img):
    if eye_img is None or eye_img.size == 0: return None
    gray_eye = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    resized_eye = cv2.resize(gray_eye, (EYE_IMG_SIZE, EYE_IMG_SIZE))
    normalized_eye = resized_eye / 255.0
    return np.expand_dims(normalized_eye, axis=-1)

def calculate_mar(face_landmarks):
    p_v1, p_v2 = face_landmarks.landmark[13], face_landmarks.landmark[14]
    p_h1, p_h2 = face_landmarks.landmark[78], face_landmarks.landmark[308]
    v_dist = np.linalg.norm([p_v1.x - p_v2.x, p_v1.y - p_v2.y])
    h_dist = np.linalg.norm([p_h1.x - p_h2.x, p_h1.y - p_h2.y])
    return v_dist / h_dist if h_dist > 0 else 0.0


def create_placeholder_frame(error_message):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    (text_width, text_height), _ = cv2.getTextSize(error_message, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    text_x = (frame.shape[1] - text_width) // 2
    text_y = (frame.shape[0] + text_height) // 2
    cv2.putText(frame, error_message, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    ret, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()

def process_frames():
    global detection_status
    
    print("ℹ[Stream Thread] Mencoba membuka kamera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[Stream Thread] FATAL: Kamera tidak dapat diakses.")
        error_frame = create_placeholder_frame("Kamera tidak ditemukan")
        while True:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + error_frame + b'\r\n')
            time.sleep(1)

    print("[Stream Thread] Kamera berhasil dibuka.")
    head_down_counter, eyes_closed_counter = 0, 0
    mar_buffer = deque(maxlen=MAR_WINDOW_SIZE)
    PITCH_THRESHOLD_RAD = -PITCH_THRESHOLD_DEG * np.pi / 180
    frame_counter, PROCESSING_INTERVAL = 0, 3
    last_known_visuals = {"face_box": None, "nose_2d": None, "pitch": 0, "yaw": 0, "roll": 0, "eye_draw_info": [], "mouth_box": None, "alert": False, "alert_reason": ""}

    while True:
        success, frame = cap.read()
        if not success:
            print("[Stream Thread] Gagal membaca frame dari kamera.")
            error_frame = create_placeholder_frame("Error membaca frame")
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + error_frame + b'\r\n')
            time.sleep(0.5)
            continue
        
        frame = cv2.resize(frame, (640, 480))
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        try:
            if not pause_event.is_set():
                frame_counter += 1
                if frame_counter % PROCESSING_INTERVAL == 0:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(rgb_frame)
                    current_visuals = last_known_visuals.copy()
                    current_visuals["eye_draw_info"] = []
                    drowsiness_alert, alert_reason = False, ""
                    if results.multi_face_landmarks:
                        face_landmarks = results.multi_face_landmarks[0]
                        feature_vector = create_feature_vector(face_landmarks)
                        if feature_vector is not None:
                            pitch_rad = HEAD_POSE_MODELS['pitch'].predict(feature_vector.reshape(1, -1))[0]
                            current_visuals["pitch"] = pitch_rad * 180 / np.pi
                            current_visuals["yaw"] = HEAD_POSE_MODELS['yaw'].predict(feature_vector.reshape(1, -1))[0] * 180 / np.pi
                            current_visuals["roll"] = HEAD_POSE_MODELS['roll'].predict(feature_vector.reshape(1, -1))[0] * 180 / np.pi
                            head_down_counter = (head_down_counter + PROCESSING_INTERVAL) if pitch_rad < PITCH_THRESHOLD_RAD else 0
                            if head_down_counter >= HEAD_DOWN_FRAMES_THRESHOLD: drowsiness_alert, alert_reason = True, "Kepala Terkulai"
                        current_visuals["nose_2d"] = (face_landmarks.landmark[1].x * w, face_landmarks.landmark[1].y * h)
                        eye_crops, eye_statuses = [], []
                        padding = 5
                        for idxs in [LEFT_EYE_IDXS, RIGHT_EYE_IDXS]:
                            points = np.array([[lm.x * w, lm.y * h] for lm in [face_landmarks.landmark[i] for i in idxs]]).astype(int)
                            x, y, pw, ph = cv2.boundingRect(points)
                            eye_crop = frame[y-padding:y+ph+padding, x-padding:x+pw+padding]
                            processed = preprocess_eye_for_predict(eye_crop)
                            if processed is not None: eye_crops.append(processed)
                            current_visuals["eye_draw_info"].append({'box': (x-padding, y-padding, x+pw+padding, y+ph+padding)})
                        if eye_crops:
                            predictions = EYE_STATUS_MODEL.predict(np.array(eye_crops), verbose=0)
                            for i, pred in enumerate(predictions):
                                status = EYE_CLASS_LABELS[1] if pred[0] > 0.5 else EYE_CLASS_LABELS[0]
                                eye_statuses.append(status)
                                current_visuals["eye_draw_info"][i]['color'] = (0, 180, 0) if status == 'Terbuka' else (255, 0, 0)
                        if any(s == 'Tertutup' for s in eye_statuses): eyes_closed_counter += PROCESSING_INTERVAL
                        else: eyes_closed_counter = 0
                        if eyes_closed_counter >= EYES_CLOSED_FRAMES_THRESHOLD: drowsiness_alert, alert_reason = True, "Mata Tertutup"
                        current_mar = calculate_mar(face_landmarks)
                        mar_buffer.append(current_mar)
                        yawn_status = "NORMAL"
                        if len(mar_buffer) == MAR_WINDOW_SIZE:
                            features = np.array([[np.mean(mar_buffer), np.max(mar_buffer), np.std(mar_buffer)]])
                            prediction = YAWN_SVM_MODEL.predict(YAWN_SCALER.transform(features))[0]
                            yawn_status = "MENGUAP" if prediction == 1 else "NORMAL"
                            if yawn_status == "MENGUAP": drowsiness_alert, alert_reason = True, "Menguap"
                        all_x, all_y = [lm.x * w for lm in face_landmarks.landmark], [lm.y * h for lm in face_landmarks.landmark]
                        current_visuals["face_box"] = (int(min(all_x)), int(min(all_y)), int(max(all_x)), int(max(all_y)))
                        mouth_points = np.array([[face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h] for i in MOUTH_INDICES]).astype(int)
                        mx, my, mw, mh = cv2.boundingRect(mouth_points)
                        current_visuals["mouth_box"] = (mx, my, mx+mw, my+mh)
                        with status_lock:
                            detection_status.update({
                                "left_eye": eye_statuses[0] if len(eye_statuses) > 0 else "N/A",
                                "right_eye": eye_statuses[1] if len(eye_statuses) > 1 else "N/A",
                                "yawn_status": yawn_status,
                                "mar": float(current_mar),
                                "pitch": float(current_visuals["pitch"]),
                                "yaw": float(current_visuals["yaw"]),
                                "roll": float(current_visuals["roll"]),
                                "alert": drowsiness_alert,
                                "alert_reason": alert_reason,
                                "is_paused": False
                            })
                    else:
                        head_down_counter, eyes_closed_counter = 0, 0
                        with status_lock: detection_status.update({"left_eye": "N/A", "right_eye": "N/A", "alert": False})
                    current_visuals["alert"], current_visuals["alert_reason"] = drowsiness_alert, alert_reason
                    last_known_visuals = current_visuals
                
                if last_known_visuals["face_box"]: cv2.rectangle(frame, (last_known_visuals["face_box"][0], last_known_visuals["face_box"][1]), (last_known_visuals["face_box"][2], last_known_visuals["face_box"][3]), (0, 255, 0), 2)
                if last_known_visuals["nose_2d"]: draw_axes(frame, last_known_visuals["pitch"], last_known_visuals["yaw"], last_known_visuals["roll"], last_known_visuals["nose_2d"])
                for info in last_known_visuals["eye_draw_info"]: cv2.rectangle(frame, (info['box'][0], info['box'][1]), (info['box'][2], info['box'][3]), info.get('color', (255,255,255)), 2)
                if last_known_visuals["mouth_box"]: cv2.rectangle(frame, (last_known_visuals["mouth_box"][0], last_known_visuals["mouth_box"][1]), (last_known_visuals["mouth_box"][2], last_known_visuals["mouth_box"][3]), (0, 255, 255), 1)

        except Exception as e:
            print(f"[Stream Thread] Terjadi error pada proses deteksi: {e}")
            cv2.putText(frame, "ERROR PADA PROSES DETEKSI", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def home():
    return render_template('home.html')

 
@app.route('/cara-kerja')
def how_it_works():
    return render_template('how_it_works.html')

@app.route('/tentang-kami')
def about():
    return render_template('about.html')

@app.route('/models')
def models():
    return render_template('models.html')

@app.route('/detector')
def detector():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(process_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    with status_lock:
        response = jsonify(detection_status)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.route('/start_detection', methods=['POST'])
def start_detection():
    pause_event.clear() 
    with status_lock:
        detection_status["is_paused"] = False
    print("▶️ [Main Thread] Deteksi DIMULAI.")
    return jsonify(success=True)

@app.route('/pause_detection', methods=['POST'])
def pause_detection():
    pause_event.set() 
    with status_lock:
        detection_status.update({
            "left_eye": "N/A", "right_eye": "N/A", "yawn_status": "PAUSED", "mar": 0.0,
            "pitch": 0.0, "yaw": 0.0, "roll": 0.0, "alert": False, "alert_reason": "",
            "is_paused": True
        })
    print("⏸️ [Main Thread] Deteksi DIJEDA.")
    return jsonify(success=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)