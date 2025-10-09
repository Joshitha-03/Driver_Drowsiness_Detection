import numpy as np
import math

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
UPPER_LIP = 13
LOWER_LIP = 14

def euclidean(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def eye_aspect_ratio(landmarks, eye_idx_list):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_idx_list]
    return (euclidean(p2, p6) + euclidean(p3, p5)) / (2.0 * euclidean(p1, p4))

def mouth_aspect_ratio(landmarks):
    top = landmarks[UPPER_LIP]
    bottom = landmarks[LOWER_LIP]
    return euclidean(top, bottom) / 50.0

def head_pitch_angle(landmarks):
    nose = landmarks.get(1, (0,0))
    left_eye = np.mean([landmarks[i] for i in LEFT_EYE if i in landmarks], axis=0)
    right_eye = np.mean([landmarks[i] for i in RIGHT_EYE if i in landmarks], axis=0)
    eye_mid = (left_eye + right_eye)/2
    dx = nose[0] - eye_mid[0]
    dy = nose[1] - eye_mid[1]
    return math.degrees(math.atan2(dy, dx+1e-6))
