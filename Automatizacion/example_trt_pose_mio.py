import json                                           # Para cargar definición de keypoints y esqueleto
import trt_pose.coco                                  # Utilidad para convertir formato COCO a topología de trt_pose

with open('/home/Documentos/Trabajo/trt_pose/tasks/human_pose/human_pose.json', 'r') as f:
    human_pose = json.load(f)                         # Carga la topología de keypoints y conexiones

topology = trt_pose.coco.coco_category_to_topology(human_pose)  # Convierte COCO a formato de trt_pose

import trt_pose.models                               # Importa arquitecturas de red para pose estimation

num_parts = len(human_pose['keypoints'])             # Número de puntos clave (COCO)
num_links = len(human_pose['skeleton'])               # Número de conexiones/eslabones

model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()  # Inicializa modelo PyTorch

import torch                                         # PyTorch para cargar pesos

# Ruta al checkpoint original de PyTorch
MODEL_WEIGHTS = '/home/Documentos/Trabajo/trt_pose/models/resnet18_baseline_att_224x224_A_epoch_249.pth'
model.load_state_dict(torch.load(MODEL_WEIGHTS))     # Carga pesos entrenados

WIDTH, HEIGHT = 224, 224                            # Dimensiones de entrada requeridas por el modelo

data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()     # Tensor de prueba para la conversión a TensorRT

import torch2trt                                      # Utilidad para convertir modelo PyTorch a TensorRT

# Convertir el modelo PyTorch a TensorRT (modo fp16)
model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)

OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'  # Archivo de salida con pesos optimizados

torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)  # Guarda el state_dict TensorRT para recarga rápida

from torch2trt import TRTModule                        # Para cargar el state_dict TensorRT

# Recargar el modelo optimizado (TensorRT) en TRTModule
model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

# --- Procesamiento de video con exoesqueleto ---
import cv2                                            # OpenCV para I/O de video y dibujo
import time
import sys

# Inicializar procesador que envuelve el engine TensorRT
from utils.trt_pose_proc import TRTPoseProcessor
processor = TRTPoseProcessor(model_path=OPTIMIZED_MODEL,
                              topology_path='/home/Documentos/Trabajo/trt_pose/tasks/human_pose/human_pose.json')

# Rutas de video de entrada y salida
input_video = 'Automatizacion/WIN_20250702_12_09_08_Pro.mp4'  # Video SPPB
output_video = 'video_con_exoesqueleto.mp4'  # Salida con exoesqueleto

# Abrir captura de video
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print(f"ERROR: No se pudo abrir el video: {input_video}")
    sys.exit(1)

# Configurar escritor de video con el mismo tamaño y FPS
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS) or 15
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

print(f"Procesando video: {input_video}")
frame_idx = 0

# Bucle de procesamiento
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Extraer keypoints del frame
    keypoints = processor.process_frame(frame)
    # Dibujar exoesqueleto si hay keypoints
    if keypoints is not None:
        frame = processor.visualize_keypoints(frame, keypoints, draw_skeleton=True)
    # Escribir frame en el video de salida
    out.write(frame)
    # Mostrar en pantalla
    cv2.imshow('Exoesqueleto', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_idx += 1
    if frame_idx % 50 == 0:
        print(f"Procesados {frame_idx} frames...")

# Liberar recursos
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Video procesado guardado en: {output_video}")