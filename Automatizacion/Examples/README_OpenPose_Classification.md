# OpenPose + Pose Classification Example

Este ejemplo integra **OpenPose** con el **clasificador de poses PoseClassificationNet** de NVIDIA TAO para detección y clasificación de poses en tiempo real.

## Características

- ✅ **Detección precisa**: Usa OpenPose para detectar 18 keypoints por persona
- 🎭 **Clasificación temporal**: Clasifica poses usando secuencias temporales
- 🎯 **6 clases de poses**: sitting_down, getting_up, sitting, standing, walking, jumping
- 🎨 **Visualización completa**: Esqueleto coloreado + clasificación en tiempo real
- ⚡ **Tiempo real**: Procesamiento optimizado para video en vivo

## Diferencias con trt_pose

| Aspecto | trt_pose | OpenPose |
|---------|----------|----------|
| **Keypoints** | 17 (COCO) | 18 (OpenPose) |
| **Precisión** | Buena | Excelente |
| **Velocidad** | Muy rápida | Rápida |
| **Robustez** | Media | Alta |
| **Esqueleto** | Básico | Completo con colores |

## Requisitos Previos

### 1. Instalación de OpenPose
```bash
# Seguir la guía oficial de OpenPose
# https://github.com/CMU-Perceptual-Computing-Lab/openpose
```

### 2. Modelo del Clasificador
- Descargar `PoseClassificationNet` desde NVIDIA NGC
- Convertir a TensorRT engine (usar `convert_model_to_tensorrt.py`)

## Configuración

### 1. Editar rutas en el código
```python
# En openpose_with_classification_example.py línea ~350
OPENPOSE_MODEL_FOLDER = "/ruta/a/openpose/models/"
POSE_CLASSIFIER_MODEL = "/ruta/a/pose_classification_net.engine"
```

### 2. Configurar parámetros
```python
SEQUENCE_LENGTH = 100          # Frames para análisis temporal
CONFIDENCE_THRESHOLD = 0.3     # Umbral de confianza keypoints
VIDEO_SOURCE = 0               # 0=webcam, "video.mp4"=archivo
OUTPUT_PATH = None             # Opcional: guardar video
```

## Uso

### Ejecución básica
```bash
cd Automatizacion/Examples
python openpose_with_classification_example.py
```

### Controles durante ejecución
- **`q`**: Salir del programa
- **`s`**: Mostrar estadísticas en consola

## Estructura del Código

### Clase Principal: `OpenPoseWithClassifier`

```python
system = OpenPoseWithClassifier(
    openpose_model_folder="/path/to/openpose/models/",
    pose_classifier_model_path="/path/to/classifier.engine",
    sequence_length=100,
    confidence_threshold=0.3
)
```

### Flujo de Procesamiento

1. **Detección**: OpenPose extrae 18 keypoints
2. **Conversión**: Formato OpenPose → NVIDIA (34 keypoints con padding)
3. **Clasificación**: TRTPoseClassifier analiza secuencia temporal
4. **Visualización**: Esqueleto coloreado + etiqueta de clase

### Visualización Mejorada

- **Esqueleto coloreado**:
  - 🔴 Rojo: Cabeza (nariz, ojos, orejas)
  - 🟢 Verde: Brazos (hombros, codos, muñecas)
  - 🔵 Azul: Torso (conexiones centrales)
  - 🟡 Amarillo: Piernas (caderas, rodillas, tobillos)

- **Keypoints especiales**:
  - ⚪ Blanco: Nariz (punto de referencia)
  - 🟣 Magenta: Ojos (orientación facial)
  - 🟢 Verde: Otros keypoints

## Comparación de Resultados

### Con trt_pose (17 keypoints)
```
Keypoints detectados: 17
Formato: COCO → NVIDIA mapping
Precisión: Media-Alta
Velocidad: Muy rápida
```

### Con OpenPose (18 keypoints)
```
Keypoints detectados: 18 (incluye neck)
Formato: OpenPose → NVIDIA mapping  
Precisión: Alta-Excelente
Velocidad: Rápida
```

## Solución de Problemas

### Error: "Import openpose could not be resolved"
```bash
# Verificar instalación de OpenPose
python -c "from openpose import pyopenpose as op; print('OK')"

# Añadir al PATH si es necesario
export PYTHONPATH=$PYTHONPATH:/path/to/openpose/build/python
```

### Error: "Model folder not found"
```python
# Verificar que existe la carpeta de modelos
OPENPOSE_MODEL_FOLDER = "/ruta/correcta/openpose/models/"
```

### Baja precisión en clasificación
```python
# Aumentar longitud de secuencia
SEQUENCE_LENGTH = 200  # Más contexto temporal

# Ajustar umbral de confianza
CONFIDENCE_THRESHOLD = 0.4  # Keypoints más confiables
```

## Rendimiento Esperado

- **Webcam 720p**: ~15-25 FPS
- **Video 1080p**: ~10-20 FPS  
- **CPU**: Funcional pero más lento
- **GPU**: Rendimiento óptimo

## Archivos Relacionados

- `openpose_with_classification_example.py`: Ejemplo principal
- `openpose_proc.py`: Wrapper de OpenPose
- `trt_pose_classifier.py`: Clasificador (soporta formato 'openpose')
- `convert_model_to_tensorrt.py`: Conversión de modelos
