import cv2
import torch
import numpy as np
from collections import deque
from model import DrowsinessTransformer
from utils import eye_aspect_ratio, mouth_aspect_ratio, head_pitch_angle, LEFT_EYE, RIGHT_EYE
import mediapipe as mp
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="SymbolDatabase.GetPrototype")

# ---------------- PARAMETERS ----------------
SEQ_LEN = 30
SMOOTHING_WINDOW = 7
EYE_CLOSED_EAR = 0.18        # EAR threshold for eye closed
CLOSED_EYES_FRAME_THRESHOLD = 8  # frames to trigger drowsy

# Normalization constants (set according to model training)
EAR_MIN, EAR_MAX = 0.15, 0.35
MAR_MIN, MAR_MAX = 0.0, 0.3
HEAD_MIN, HEAD_MAX = 0.0, 180.0

# ---------------- LOAD MODEL ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DrowsinessTransformer()
model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=device))
model.eval()
model.to(device)

# ---------------- DEQUE FOR SMOOTHING ----------------
pred_queue = deque(maxlen=SMOOTHING_WINDOW)

# ---------------- MEDIAPIPE INIT ----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# ---------------- VIDEO CAPTURE ----------------
cap = cv2.VideoCapture(0)

# Initialize sequence with neutral "alert" values
seq_features = [[0.5, 0.0, 0.5]] * SEQ_LEN

closed_frames = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0]
        landmarks = {i: (lmk.x * w, lmk.y * h) for i, lmk in enumerate(lm.landmark)}
        ear = (eye_aspect_ratio(landmarks, LEFT_EYE) + eye_aspect_ratio(landmarks, RIGHT_EYE)) / 2
        mar = mouth_aspect_ratio(landmarks)
        head_angle = head_pitch_angle(landmarks)
    else:
        ear, mar, head_angle = 0.0, 0.0, 0.0

    # ---------------- NORMALIZE FEATURES ----------------
    ear_norm = np.clip((ear - EAR_MIN) / (EAR_MAX - EAR_MIN), 0, 1)
    mar_norm = np.clip((mar - MAR_MIN) / (MAR_MAX - MAR_MIN), 0, 1)
    head_norm = np.clip((head_angle - HEAD_MIN) / (HEAD_MAX - HEAD_MIN), 0, 1)

    seq_features.append([ear_norm, mar_norm, head_norm])
    if len(seq_features) > SEQ_LEN:
        seq_features.pop(0)

    pred_label = "Alert"

    # ---------------- PREDICTION ----------------
    if len(seq_features) == SEQ_LEN:
        x = torch.tensor([seq_features], dtype=torch.float32).to(device)  # batch=1, seq_len, features
        with torch.no_grad():
            out = model(x)
            out = torch.softmax(out, dim=-1)
            pred = torch.argmax(out, dim=-1).item()
            pred_queue.append(pred)

        # Smoothed prediction
        pred_smoothed = int(np.round(np.mean(pred_queue)))

        # Safety check: EAR threshold
        if ear < EYE_CLOSED_EAR:
            closed_frames += 1
        else:
            closed_frames = 0

        if closed_frames >= CLOSED_EYES_FRAME_THRESHOLD or pred_smoothed == 1:
            pred_label = "Drowsy 😴"
        else:
            pred_label = "Alert"

    # ---------------- DRAW INFO ----------------
    cv2.putText(frame, f"EAR:{ear:.2f} MAR:{mar:.2f} Head:{head_angle:.1f} | Pred:{pred_label}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0,0,255) if pred_label=="Drowsy 😴" else (0,255,0), 2)

    cv2.imshow("Driver Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()
