import cv2
import os
import pandas as pd
import mediapipe as mp
from utils import eye_aspect_ratio, mouth_aspect_ratio, head_pitch_angle, LEFT_EYE, RIGHT_EYE

mp_face_mesh = mp.solutions.face_mesh

def extract_features_from_images(image_folder, output_csv, label):
    image_files = sorted([os.path.join(image_folder,f) for f in os.listdir(image_folder) if f.endswith(".jpg")])
    rows = []
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as mesh:
        for frame_no, img_file in enumerate(image_files):
            frame = cv2.imread(img_file)
            frame = cv2.resize(frame, (480, 360))  
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = mesh.process(rgb)
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0]
                landmarks = {i: (lmk.x*w, lmk.y*h) for i,lmk in enumerate(lm.landmark)}
                ear = (eye_aspect_ratio(landmarks, LEFT_EYE)+eye_aspect_ratio(landmarks, RIGHT_EYE))/2
                mar = mouth_aspect_ratio(landmarks)
                head_angle = head_pitch_angle(landmarks)
            else:
                ear, mar, head_angle = 0.0,0.0,0.0
            rows.append({'frame': frame_no, 'ear': ear, 'mar': mar, 'head_angle': head_angle, 'label': label})

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Features saved: {output_csv}, frames: {len(df)}")
