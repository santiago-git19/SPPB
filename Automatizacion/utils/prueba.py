import cv2
import numpy as np
import mediapipe as mp
import mediapipe.tasks.python as mp_tasks
import mediapipe.tasks.python.vision as mp_vision
import mediapipe.tasks as mp_tasks_root
import os

# --- Configuración ---
MODEL_PATH = '../models/pose_landmarker_lite.task'
IMAGE_PATH = 'persona.PNG'
OUTPUT_PATH = 'pose_output_task.png'

# --- Cargar imagen ---
img = cv2.imread(IMAGE_PATH)
if img is None:
    print("❌ Error: No se pudo cargar la imagen. Verifica la ruta del archivo.")
    exit(1)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# --- Crear pose landmarker ---
BaseOptions = mp_tasks.BaseOptions
VisionRunningMode = mp_vision.VisionRunningMode
PoseLandmarker = mp_vision.PoseLandmarker
PoseLandmarkerResult = mp_vision.PoseLandmarkerResult

options = mp_vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_poses=1
)

with PoseLandmarker.create_from_options(options) as landmarker:
    mp_image = mp_vision.MpImage(image_format=mp_vision.ImageFormat.SRGB, data=img_rgb)
    result = landmarker.detect(mp_image)

    if not result.pose_landmarks:
        print("❌ No se detectaron keypoints.")
        exit(1)

    # Dibujar keypoints y conexiones
    annotated = img.copy()
    for pose in result.pose_landmarks:
        for x, y, z in pose:
            px = int(x * img.shape[1])
            py = int(y * img.shape[0])
            cv2.circle(annotated, (px, py), 4, (255,255,255), -1)
        # Conexiones
        for pt1, pt2 in mp.solutions.pose.POSE_CONNECTIONS:
            x1, y1, _ = pose[pt1]
            x2, y2, _ = pose[pt2]
            px1 = int(x1 * img.shape[1])
            py1 = int(y1 * img.shape[0])
            px2 = int(x2 * img.shape[1])
            py2 = int(y2 * img.shape[0])
            cv2.line(annotated, (px1, py1), (px2, py2), (0,0,255), 2)

    cv2.imwrite(OUTPUT_PATH, annotated)
    print(f"✅ Imagen procesada guardada en: {OUTPUT_PATH}")