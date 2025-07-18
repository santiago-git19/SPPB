# Guía de Uso: TRT Pose + Clasificación de Poses

## Descripción General

Este sistema integra **trt_pose** (detección de keypoints) con **PoseClassificationNet** de NVIDIA TAO (clasificación de poses) para crear un pipeline completo de análisis de poses humanas en tiempo real.

## Arquitectura del Sistema

```
Video/Imagen → TRTPoseProcessor → Keypoints → TRTPoseClassifier → Clasificación
                (existente)                     (nueva clase)
```

### Componentes:

1. **TRTPoseProcessor** (existente): Detecta keypoints usando modelo TensorRT optimizado
2. **TRTPoseClassifier** (nueva): Clasifica poses usando PoseClassificationNet de NVIDIA TAO
3. **TRTPoseWithClassifier** (ejemplo): Integra ambos componentes

## Clases de Poses Soportadas

El modelo PoseClassificationNet puede clasificar 6 tipos de poses:

- 🪑 **sitting_down**: Persona sentándose
- 🏃 **getting_up**: Persona levantándose  
- 💺 **sitting**: Persona sentada
- 🧍 **standing**: Persona de pie
- 🚶 **walking**: Persona caminando
- 🦘 **jumping**: Persona saltando

## Formatos de Keypoints Soportados

La clase `TRTPoseClassifier` puede procesar múltiples formatos:

- **COCO** (17 keypoints) - Formato estándar de trt_pose
- **OpenPose** (18 keypoints)
- **NVIDIA** (34 keypoints) - Formato nativo del modelo
- **Human3.6M** (17 keypoints)
- **NTU RGB+D** (25 keypoints)

## Instalación y Configuración

### 1. Requisitos

```bash
# Dependencias principales
pip install torch torchvision
pip install opencv-python
pip install numpy
pip install onnxruntime-gpu  # o onnxruntime para CPU
pip install torch2trt
pip install trt_pose
```

### 2. Obtener Modelos

#### Modelo TRT Pose (ya convertido)
```bash
# Usar el script de conversión existente
python convert_model_to_tensorrt.py
```

#### Modelo PoseClassificationNet
```bash
# Descargar desde NGC (requiere cuenta NVIDIA)
ngc registry model download-version nvidia/tao/poseclassificationnet:deployable_onnx_v1.0

# O usar wget si tienes acceso directo
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/poseclassificationnet/versions/deployable_onnx_v1.0/files/st-gcn_3dbp_nvidia.onnx
```

### 3. Estructura de Archivos

```
proyecto/
├── utils/
│   ├── trt_pose_proc.py              # Procesador existente
│   ├── trt_pose_classifier.py        # Nueva clase clasificadora
│   └── jetson_utils.py               # Utilidades Jetson
├── Examples/
│   └── trt_pose_with_classification_example.py
├── models/
│   ├── resnet18_baseline_att_224x224_A_epoch_249_trt.pth
│   ├── st-gcn_3dbp_nvidia.onnx
│   └── human_pose.json
└── convert_model_to_tensorrt.py
```

## Uso Básico

### 1. Clasificador Individual

```python
from utils.trt_pose_classifier import create_pose_classifier
import numpy as np

# Crear clasificador
classifier = create_pose_classifier(
    model_path="models/st-gcn_3dbp_nvidia.onnx",
    keypoint_format='coco',
    sequence_length=30,
    confidence_threshold=0.3
)

# Procesar keypoints (formato COCO: 17 keypoints con x,y,confidence)
keypoints = np.random.rand(17, 3)  # Ejemplo
result = classifier.process_keypoints(keypoints)

if result and not result.get('error'):
    print(f"Pose detectada: {result['predicted_class']}")
    print(f"Confianza: {result['confidence']:.2f}")
```

### 2. Sistema Completo

```python
from Examples.trt_pose_with_classification_example import TRTPoseWithClassifier

# Configuración
system = TRTPoseWithClassifier(
    trt_pose_model_path="models/resnet18_baseline_att_224x224_A_epoch_249_trt.pth",
    pose_topology_path="models/human_pose.json",
    pose_classifier_model_path="models/st-gcn_3dbp_nvidia.onnx"
)

# Procesar video desde cámara
system.process_video(video_source=0)

# Procesar archivo de video
system.process_video(video_source="input_video.mp4", output_path="output_with_poses.mp4")
```

### 3. Procesamiento por Frames

```python
import cv2

# Inicializar sistema
system = TRTPoseWithClassifier(...)

# Procesar frame individual
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

if ret:
    result = system.process_frame_with_classification(frame)
    
    print(f"Personas detectadas: {result['people_detected']}")
    for classification in result['pose_classifications']:
        print(f"Persona {classification['person_id']}: {classification['pose_class']}")
```

## Configuración de Parámetros

### TRTPoseClassifier

```python
classifier = TRTPoseClassifier(
    model_path="path/to/model.onnx",
    keypoint_format='coco',           # Formato de entrada
    sequence_length=30,               # Frames para clasificación temporal
    confidence_threshold=0.3,         # Umbral de confianza para keypoints
    max_persons=1                     # Máximo número de personas
)
```

### Parámetros Importantes

- **sequence_length**: Número de frames consecutivos necesarios para clasificación
  - Más frames = mayor precisión pero mayor latencia
  - Recomendado: 30 frames (1 segundo a 30fps)

- **confidence_threshold**: Umbral para filtrar keypoints de baja calidad
  - Más alto = más selectivo pero puede perder información
  - Recomendado: 0.3

- **keypoint_format**: Debe coincidir con el formato de salida de trt_pose
  - Para trt_pose estándar: 'coco'

## Optimización para Jetson Nano

### 1. Configuración de Memoria

```python
# Usar configuración conservativa
classifier = create_pose_classifier(
    model_path="model.onnx",
    sequence_length=15,  # Reducir para ahorrar memoria
    confidence_threshold=0.4  # Más selectivo
)
```

### 2. Reducir Resolución

```python
system = TRTPoseWithClassifier(
    ...,
    width=224,   # Resolución mínima para trt_pose
    height=224
)
```

### 3. Procesamiento por Lotes

```python
# Procesar cada N frames en lugar de todos
frame_counter = 0
process_every = 3  # Procesar 1 de cada 3 frames

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    if frame_counter % process_every == 0:
        result = system.process_frame_with_classification(frame)
        # Usar último resultado para frames intermedios
    
    frame_counter += 1
```

## Monitoreo y Estadísticas

### Estadísticas del Clasificador

```python
# Obtener estadísticas
stats = classifier.get_statistics()
print(f"Predicciones totales: {stats['total_predictions']}")
print(f"Tasa de confianza: {stats['confidence_rate']:.2f}")
print(f"Clase más común: {stats['most_common_class']}")

# Guardar estadísticas
classifier.save_statistics("classification_stats.json")
```

### Estadísticas del Sistema

```python
# El sistema automáticamente reporta al final del procesamiento
system.process_video(...)
# Muestra estadísticas finales incluyendo FPS, detecciones, etc.
```

## Solución de Problemas

### Error: Modelo ONNX no encontrado
```bash
# Verificar descarga del modelo
ls -la models/st-gcn_3dbp_nvidia.onnx

# Si no existe, descargar desde NGC
ngc registry model download-version nvidia/tao/poseclassificationnet:deployable_onnx_v1.0
```

### Error: CUDA Out of Memory
```python
# Usar CPU para clasificación
providers = ['CPUExecutionProvider']
session = ort.InferenceSession(model_path, providers=providers)

# O reducir sequence_length
classifier = create_pose_classifier(..., sequence_length=10)
```

### Clasificaciones Inconsistentes
```python
# Aumentar sequence_length para más estabilidad
classifier = create_pose_classifier(..., sequence_length=45)

# O aumentar confidence_threshold
classifier = create_pose_classifier(..., confidence_threshold=0.5)
```

### Rendimiento Lento
```python
# Procesar menos frames
process_every = 2  # Solo cada 2 frames

# Usar resolución menor
system = TRTPoseWithClassifier(..., width=224, height=224)

# Limitar número de personas
classifier = create_pose_classifier(..., max_persons=1)
```

## Ejemplos de Aplicaciones

### 1. Monitoreo de Ejercicio
```python
# Detectar secuencias de ejercicio
exercise_sequence = ['standing', 'getting_down', 'sitting', 'getting_up']
```

### 2. Análisis de Comportamiento
```python
# Contar transiciones entre poses
transitions = {}
last_pose = None

for result in pose_results:
    current_pose = result['predicted_class']
    if last_pose and last_pose != current_pose:
        transition = f"{last_pose}_to_{current_pose}"
        transitions[transition] = transitions.get(transition, 0) + 1
    last_pose = current_pose
```

### 3. Seguridad y Vigilancia
```python
# Detectar comportamientos anómalos
unusual_poses = ['jumping', 'getting_up']
if result['predicted_class'] in unusual_poses:
    alert_security_system()
```

## Rendimiento Esperado

### Jetson Nano
- **FPS**: 10-15 fps con clasificación completa
- **Latencia**: ~100-150ms por frame
- **Memoria**: ~1.5GB RAM total

### Jetson AGX Orin
- **FPS**: 25-30 fps con clasificación completa  
- **Latencia**: ~30-50ms por frame
- **Memoria**: ~2GB RAM total

### PC con GPU
- **FPS**: 30+ fps con clasificación completa
- **Latencia**: ~15-30ms por frame
- **Memoria**: Variable según GPU

## Próximos Pasos

1. **Entrenamiento Personalizado**: Usar TAO Toolkit para entrenar con datos específicos
2. **Optimización**: Convertir modelo ONNX a TensorRT para mejor rendimiento
3. **Multiples Personas**: Extender para tracking de múltiples personas simultáneamente
4. **Integración**: Conectar con sistemas de bases de datos o alertas

---

Para más información, consulta:
- [NVIDIA TAO Toolkit Documentation](https://docs.nvidia.com/tao/tao-toolkit/)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [trt_pose GitHub](https://github.com/NVIDIA-AI-IOT/trt_pose)
