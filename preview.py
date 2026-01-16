import cv2
import numpy as np
import mediapipe as mp
from utils import eye_aspect_ratio, mouth_aspect_ratio, head_pitch_angle, LEFT_EYE, RIGHT_EYE


EYE_CLOSED_EAR = 0.18  


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    ear, mar, head_angle = 0.0, 0.0, 0.0

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0]
        landmarks = {i: (lmk.x * w, lmk.y * h) for i, lmk in enumerate(lm.landmark)}
        ear_left = eye_aspect_ratio(landmarks, LEFT_EYE)
        ear_right = eye_aspect_ratio(landmarks, RIGHT_EYE)
        ear = (ear_left + ear_right) / 2
        mar = mouth_aspect_ratio(landmarks)
        head_angle = head_pitch_angle(landmarks)

    cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"MAR: {mar:.2f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"Head Angle: {head_angle:.1f}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Driver Drowsiness Feature Display", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()
