# Sistema SPPB con TRT Pose - Guía Completa

## Introducción

Este sistema implementa el Short Physical Performance Battery (SPPB) utilizando TRT Pose para detección de poses y clasificación automática, con soporte para dos cámaras (frontal y lateral).

## Características Principales

- ✅ **TRT Pose** en lugar de OpenPose para mejor rendimiento en Jetson Nano
- ✅ **Dos cámaras** para medición precisa de distancia y alineación
- ✅ **Clasificación automática** de poses (walking, standing, sitting, etc.)
- ✅ **Cálculo automático** de distancia recorrida en Gait Speed
- ✅ **Optimizado** para Jetson Nano y PC de escritorio

## Arquitectura del Sistema

```
SPPB System
├── main.py                 # Punto de entrada principal
├── sppb_test.py           # Coordinador de pruebas (inicializa TRT Pose)
├── config.py              # Configuración del sistema
├── phases/
│   ├── balance.py         # Prueba de equilibrio
│   ├── gait_speed.py      # Prueba de velocidad de marcha
│   └── chair_rise.py      # Prueba de levantarse de la silla
├── utils/
│   ├── trt_pose_proc.py   # Procesador TRT Pose (keypoints)
│   └── trt_pose_classifier.py # Clasificador de poses
└── models/
    ├── resnet18_baseline_att_224x224_A_epoch_249.pth  # Modelo TRT Pose
    ├── human_pose.json                                # Topología COCO
    └── pose_classification/
        └── st-gcn_3dbp_nvidia.engine                  # Clasificador
```

## Instalación

### 1. Requisitos del Sistema

#### Para Jetson Nano:
```bash
# TensorRT (ya incluido en JetPack)
# PyTorch para Jetson
pip3 install torch torchvision torchaudio

# TRT Pose
git clone https://github.com/NVIDIA-AI-IOT/trt_pose
cd trt_pose
python3 setup.py install

# Dependencias adicionales
pip3 install opencv-python numpy onnxruntime-gpu
```

#### Para PC con GPU NVIDIA:
```bash
# PyTorch con CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# TRT Pose
git clone https://github.com/NVIDIA-AI-IOT/trt_pose
cd trt_pose
python setup.py install

# ONNX Runtime con GPU
pip install onnxruntime-gpu opencv-python numpy
```

### 2. Descargar Modelos

```bash
# Crear directorio de modelos
mkdir -p models/pose_classification

# Descargar modelo TRT Pose (ejemplo)
wget -O models/resnet18_baseline_att_224x224_A_epoch_249.pth \
  https://nvidia.box.com/shared/static/d696d0bt613UQO6wvus0en2a8hhkwwBh.pth

# Descargar topología COCO
wget -O models/human_pose.json \
  https://raw.githubusercontent.com/NVIDIA-AI-IOT/trt_pose/master/tasks/human_pose/human_pose.json

# Descargar clasificador de poses (ajustar URL según disponibilidad)
# wget -O models/pose_classification/st-gcn_3dbp_nvidia.engine [URL_DEL_MODELO]
```

## Configuración

### 1. Configuración Básica

Edita `config.py` con las rutas correctas:

```python
class Config:
    def __init__(self):
        # === RUTAS DE MODELOS ===
        self.trt_pose_model = "models/resnet18_baseline_att_224x224_A_epoch_249.pth"
        self.pose_topology = "models/human_pose.json"
        self.pose_classifier_model = "models/pose_classification/st-gcn_3dbp_nvidia.engine"
        
        # === CONFIGURACIÓN DE CÁMARAS ===
        self.frontal_camera = 0  # Índice de cámara USB o ruta de video
        self.lateral_camera = 1  # Índice de cámara USB o ruta de video
        
        # === CALIBRACIÓN ===
        self.pixel_to_meter_ratio = 0.01  # ¡DEBE CALIBRARSE!
```

### 2. Calibración del Sistema

#### Calibrar pixel_to_meter_ratio:

1. **Método de objeto de referencia:**
   ```python
   # Coloca una regla de 1 metro en el plano de movimiento
   # Mide cuántos píxeles ocupa en la imagen
   # Si 1 metro = 100 píxeles:
   pixel_to_meter_ratio = 1.0 / 100.0  # = 0.01
   ```

2. **Método de distancia conocida:**
   ```python
   # Marca 4 metros en el suelo
   # Mide la distancia en píxeles en la imagen
   # Si 4 metros = 400 píxeles:
   pixel_to_meter_ratio = 4.0 / 400.0  # = 0.01
   ```

### 3. Configuración de Cámaras

#### Cámaras USB:
```python
# Verificar índices disponibles
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Cámara {i}: Disponible")
        cap.release()
```

#### Videos pregrabados:
```python
config.frontal_camera = "videos/frontal_view.mp4"
config.lateral_camera = "videos/lateral_view.mp4"
```

## Uso del Sistema

### 1. Ejecución Básica

```bash
# Ejecutar con configuración por defecto
python3 main.py

# O usar configuración específica
python3 -c "
from config_examples import get_config
from sppb_test import SPPBTest

config = get_config('jetson')  # o 'desktop', 'development'
sppb = SPPBTest(config)
result = sppb.run()
print(result.to_dict())
"
```

### 2. Flujo de Pruebas

El sistema ejecuta automáticamente:

1. **Balance Test**: Medición de equilibrio con diferentes posturas
2. **Gait Speed Test**: Medición de velocidad de marcha (4 metros)
3. **Chair Rise Test**: Prueba de levantarse de la silla

### 3. Interpretación de Resultados

```python
{
    'balance': {'score': 4, 'time': 30.0, 'completed': True},
    'gait_speed': {'score': 3, 'best_time': 5.2, 'all_times': [5.2, 5.8]},
    'chair_rise': {'score': 2, 'time': 12.1, 'repetitions': 5},
    'total_score': 9  # Suma de las tres pruebas (0-12)
}
```

## Algoritmo de Cálculo de Distancia

### Proceso de Medición:

1. **Detección de Keypoints**: TRT Pose detecta 17 keypoints COCO
2. **Extracción de Cadera**: Se usan Left Hip (11) y Right Hip (12)
3. **Cálculo de Posición**: `hip_x = (left_hip_x + right_hip_x) / 2`
4. **Cálculo de Distancia**: `distance = |current_x - previous_x| * ratio`
5. **Acumulación**: Se suma la distancia total recorrida

### Consideraciones Importantes:

- **Cámara Lateral**: Debe estar perpendicular al movimiento
- **Altura Constante**: Mantener cámara a altura fija
- **Iluminación**: Buena iluminación para detección óptima
- **Calibración**: El `pixel_to_meter_ratio` es crítico para precisión

## Troubleshooting

### Problemas Comunes

1. **Error: "Modelo no encontrado"**
   ```bash
   # Verificar rutas en config.py
   ls -la models/
   ```

2. **Error: "Cámara no disponible"**
   ```bash
   # Verificar cámaras conectadas
   ls /dev/video*
   ```

3. **Distancias incorrectas**
   - Verificar calibración `pixel_to_meter_ratio`
   - Comprobar posición de cámara lateral
   - Verificar detección de keypoints de cadera

4. **Rendimiento lento en Jetson Nano**
   ```python
   # Reducir sequence_length en config
   config.sequence_length = 10
   config.confidence_threshold = 0.25
   ```

### Logs de Debug

Activar logs detallados:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Verificación del Sistema

```python
# Script de verificación
from utils.trt_pose_proc import TRTPoseProcessor
from utils.trt_pose_classifier import TRTPoseClassifier

# Verificar TRT Pose
processor = TRTPoseProcessor("models/resnet18_baseline_att_224x224_A_epoch_249.pth", 
                           "models/human_pose.json")
print("✅ TRT Pose inicializado")

# Verificar clasificador
classifier = TRTPoseClassifier("models/pose_classification/st-gcn_3dbp_nvidia.engine")
print("✅ Clasificador inicializado")
```

## Optimizaciones de Rendimiento

### Para Jetson Nano:
- `sequence_length = 10-15`
- `confidence_threshold = 0.25-0.3`
- Usar solo cámara lateral si es necesario
- Reducir resolución de cámaras si es posible

### Para PC de Escritorio:
- `sequence_length = 30`
- `confidence_threshold = 0.3-0.4`
- Usar ambas cámaras para máxima precisión
- Resolución completa de cámaras

## Desarrollo y Extensión

### Añadir Nueva Fase:

1. Crear archivo en `phases/nueva_fase.py`
2. Heredar de `PhaseBase`
3. Implementar métodos requeridos
4. Añadir a `sppb_test.py`

### Personalizar Clasificación:

1. Entrenar nuevo modelo con clases específicas
2. Actualizar `POSE_CLASSES` en `trt_pose_classifier.py`
3. Ajustar lógica de detección en las fases

## Soporte y Contribución

Para reportar problemas o contribuir:

1. Verificar que la configuración sea correcta
2. Incluir logs de error completos
3. Especificar hardware utilizado (Jetson Nano, PC, etc.)
4. Incluir configuración de cámaras

## Referencias

- [TRT Pose GitHub](https://github.com/NVIDIA-AI-IOT/trt_pose)
- [NVIDIA TAO Toolkit](https://developer.nvidia.com/tao-toolkit)
- [Short Physical Performance Battery](https://www.nia.nih.gov/research/labs/leps/short-physical-performance-battery-sppb)
