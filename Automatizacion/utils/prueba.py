import cv2
import numpy as np
from utils.mediapipe_pose_proc import MediaPipePoseProcessor

# Cargar imagen
img = cv2.imread('persona.PNG')
if img is None:
    print("❌ Error: No se pudo cargar la imagen. Verifica la ruta del archivo.")
    exit(1)
height, width = img.shape[:2]

# Inicializar el procesador TensorRT
processor = MediaPipePoseProcessor(
    model_path="../models/pose_landmark_lite_fp16.engine",
    input_width=256,
    input_height=256,
    confidence_threshold=0.0  # Mostrar todos los puntos
)

# Procesar la imagen para obtener keypoints
keypoints = processor.process_frame(img)
if keypoints is None:
    print("❌ No se detectaron keypoints.")
    exit(1)

# Dibujar keypoints y conexiones
for i, (x, y, conf) in enumerate(keypoints):
    if conf > 0:
        cv2.circle(img, (int(x), int(y)), 4, (255, 255, 255), -1)

for pt1, pt2 in processor.POSE_CONNECTIONS:
    x1, y1, c1 = keypoints[pt1]
    x2, y2, c2 = keypoints[pt2]
    if c1 > 0 and c2 > 0:
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

# Guardar la imagen procesada
output_path = 'pose_output_trt.png'
cv2.imwrite(output_path, img)
print(f"✅ Imagen procesada guardada en: {output_path}")

# Liberar recursos
processor.cleanup()