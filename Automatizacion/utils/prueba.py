import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import os

# --- Configuración ---
ENGINE_PATH = "../models/pose_landmark_lite_fp16.engine"
INPUT_WIDTH = 256
INPUT_HEIGHT = 256

# --- Cargar imagen y preprocesar ---
img = cv2.imread('persona.PNG')
if img is None:
    print("❌ Error: No se pudo cargar la imagen. Verifica la ruta del archivo.")
    exit(1)
orig_h, orig_w = img.shape[:2]
resized = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT))
rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
input_data = rgb.astype(np.float32) / 255.0
input_data = np.expand_dims(input_data, axis=0)  # (1,256,256,3)
input_data = np.ascontiguousarray(input_data, dtype=np.float32)

# --- Cargar engine TensorRT ---
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with open(ENGINE_PATH, 'rb') as f:
    engine_data = f.read()
runtime = trt.Runtime(TRT_LOGGER)
engine = runtime.deserialize_cuda_engine(engine_data)
context = engine.create_execution_context()

# --- Preparar bindings ---
input_binding_idx = None
output_binding_idx = None
for i in range(engine.num_bindings):
    if engine.binding_is_input(i):
        input_binding_idx = i
        input_shape = engine.get_binding_shape(i)
        input_size = trt.volume(input_shape)
    else:
        output_binding_idx = i
        output_shape = engine.get_binding_shape(i)
        output_size = trt.volume(output_shape)

# --- Reservar memoria GPU ---
d_input = cuda.mem_alloc(input_data.nbytes)
d_output = cuda.mem_alloc(output_size * np.dtype(np.float32).itemsize)
stream = cuda.Stream()

# --- Copiar datos a GPU ---
cuda.memcpy_htod_async(d_input, input_data, stream)

# --- Ejecutar inferencia ---
bindings = [int(d_input) if i == input_binding_idx else int(d_output) for i in range(engine.num_bindings)]
context.execute_async_v2(bindings, stream.handle)

# --- Copiar resultado a CPU ---
h_output = np.empty(output_shape, dtype=np.float32)
cuda.memcpy_dtoh_async(h_output, d_output, stream)
stream.synchronize()

# --- Postprocesar y visualizar keypoints ---
# Suponemos salida (1,195) o (1,117) o similar. Usar la de mayor tamaño si hay varias.
output_flat = h_output.flatten()
if output_flat.size == 195:
    # 117 xyz (39 puntos) + 39 vis + 39 pres
    xyz = output_flat[:117].reshape(39,3)
    keypoints = xyz[:33,:2]  # Solo los 33 del cuerpo
elif output_flat.size == 117:
    xyz = output_flat[:117].reshape(39,3)
    keypoints = xyz[:33,:2]
else:
    print(f"❌ Salida inesperada: {output_flat.shape}")
    exit(1)

# Los valores suelen estar en [-1,1], normalizar a [0,1]
keypoints = (keypoints + 1) / 2
# Escalar a tamaño original
keypoints[:,0] = keypoints[:,0] * orig_w
keypoints[:,1] = keypoints[:,1] * orig_h

# Dibujar keypoints
for x, y in keypoints:
    cv2.circle(img, (int(x), int(y)), 4, (255,255,255), -1)

# Guardar imagen
output_path = 'pose_output_trt_direct.png'
cv2.imwrite(output_path, img)
print(f"✅ Imagen procesada guardada en: {output_path}")

# Liberar recursos
context = None
engine = None
runtime = None