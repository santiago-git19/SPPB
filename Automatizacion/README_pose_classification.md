# Clasificación de Poses con TensorRT Pose + PoseClassificationNet

Este módulo integra TensorRT Pose para extracción de keypoints con PoseClassificationNet para clasificación de poses en tiempo real.

## Estructura del Proyecto

```
Automatizacion/
├── utils/
│   ├── pose_classifier.py          # Clase principal para clasificación
│   ├── trt_pose_proc.py            # Procesador TensorRT Pose
│   └── openpose_proc.py            # Procesador OpenPose (alternativo)
├── models/                         # Directorio para modelos
│   ├── resnet18_baseline_att_224x224_A_epoch_249.pth
│   ├── human_pose.json
│   └── pose_classification.engine
├── example_pose_classification.py   # Script completo de ejemplo
├── simple_pose_classification.py    # Script simplificado
├── convert_pose_model.py           # Convertidor de modelos
└── README_pose_classification.md    # Este archivo
```

## Requisitos

### Dependencias Básicas
```bash
pip install opencv-python numpy torch torchvision
```

### Dependencias Avanzadas (para TensorRT)
```bash
# TensorRT (requiere CUDA)
pip install tensorrt pycuda

# torch2trt (para conversión de modelos)
pip install torch2trt
```

### Verificar Dependencias
```bash
python simple_pose_classification.py --check
```

## Modelos Necesarios

### 1. Modelo TensorRT Pose
- **Archivo**: `models/resnet18_baseline_att_224x224_A_epoch_249.pth`
- **Descarga**: [NVIDIA TensorRT Pose](https://github.com/NVIDIA-AI-IOT/trt_pose)
- **Topología**: `models/human_pose.json`

### 2. Engine de Clasificación
- **Archivo**: `models/pose_classification.engine`
- **Generación**: Usar `convert_pose_model.py` para convertir modelo PyTorch entrenado

## Uso Básico

### 1. Crear Video de Demostración
```bash
python simple_pose_classification.py --demo
```

### 2. Procesar Video
```bash
python simple_pose_classification.py video.mp4
```

### 3. Procesar con Salida
```bash
python simple_pose_classification.py video.mp4 --output procesado.mp4
```

### 4. Uso Avanzado
```bash
python example_pose_classification.py video.mp4 \
  --output procesado.mp4 \
  --stats \
  --export-results resultados.json \
  --fps-limit 30
```

## Clases de Poses Soportadas

La clasificación actual soporta las siguientes poses:

| ID | Clase | Descripción |
|----|-------|-------------|
| 0  | de_pie | Persona de pie |
| 1  | sentado | Persona sentada |
| 2  | levantandose | Transición de sentado a de pie |
| 3  | caminando | Persona caminando |
| 4  | equilibrio | Pose de equilibrio |
| 5  | desconocido | Pose no clasificada |

## Ejemplo de Código

```python
from utils.pose_classifier import PoseClassifier

# Inicializar clasificador
classifier = PoseClassifier(
    pose_model_path="models/resnet18_baseline_att_224x224_A_epoch_249.pth",
    topology_path="models/human_pose.json",
    classification_engine_path="models/pose_classification.engine"
)

# Procesar video
results = classifier.process_video(
    video_path="mi_video.mp4",
    output_path="procesado.mp4",
    show_video=True,
    fps_limit=15
)

# Obtener estadísticas
stats = classifier.get_pose_statistics()
print(f"Pose más común: {stats['most_common_pose']}")

# Procesar frame individual
frame_result = classifier.process_single_frame(frame)
pose_class, confidence, keypoints = frame_result
```

## Configuración Avanzada

### Personalizar Clases de Poses

Edita el mapeo de clases en `utils/pose_classifier.py`:

```python
self.pose_classes = {
    0: "pose_personalizada_1",
    1: "pose_personalizada_2",
    2: "pose_personalizada_3",
    # ...
}
```

### Ajustar Topología

Para usar diferente número de keypoints, modifica `human_pose.json`:

```json
{
    "keypoints": [
        "nose", "left_eye", "right_eye", 
        // ... más keypoints
    ],
    "skeleton": [
        [1, 2], [2, 3], 
        // ... conexiones
    ]
}
```

## Entrenar Modelo Personalizado

### 1. Recolectar Datos

```python
# Ejemplo de estructura de datos
training_data = {
    'keypoints': [
        [x1, y1, x2, y2, ..., x17, y17],  # 17 keypoints × 2 coordenadas
        # ... más muestras
    ],
    'labels': [0, 1, 2, ...],  # Etiquetas correspondientes
}
```

### 2. Entrenar Red Neural

```python
import torch
import torch.nn as nn

class PoseClassificationNet(nn.Module):
    def __init__(self, num_keypoints=17, num_classes=6):
        super().__init__()
        input_size = num_keypoints * 2
        
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)
```

### 3. Convertir a TensorRT

```bash
python convert_pose_model.py
```

## Solución de Problemas

### Error: "Modelo no encontrado"
- Verifica que los modelos estén en el directorio `models/`
- Descarga los modelos necesarios desde los enlaces oficiales

### Error: "TensorRT no disponible"
- Instala CUDA y TensorRT
- Verifica compatibilidad de versiones
- Usa CPU como alternativa (más lento)

### Error: "Keypoints no detectados"
- Verifica que la persona esté visible en el video
- Ajusta iluminación y calidad del video
- Considera usar OpenPose como alternativa

### Rendimiento Lento
- Reduce `fps_limit` en el procesamiento
- Usa resolución más baja
- Verifica que esté usando GPU

## Integración con SPPB

Para usar la clasificación de poses en el test SPPB:

```python
from utils.pose_classifier import PoseClassifier
from phases.balance import BalancePhase

# En la fase de equilibrio
classifier = PoseClassifier(...)
balance_phase = BalancePhase()

# Procesar video durante la prueba
results = classifier.process_video(balance_video_path)

# Analizar poses para scoring automático
balance_poses = [r for r in results if r['pose_class'] == 'equilibrio']
balance_score = len(balance_poses) / len(results)
```

## Archivos de Salida

### Video Procesado
- Keypoints visualizados en cada frame
- Clasificación de pose en tiempo real
- Barra de confianza

### Archivo JSON de Resultados
```json
{
    "metadata": {
        "video_path": "video.mp4",
        "processing_date": "2024-01-01T12:00:00",
        "total_frames": 1000
    },
    "statistics": {
        "pose_counts": {"de_pie": 500, "sentado": 300, "caminando": 200},
        "pose_percentages": {"de_pie": 50.0, "sentado": 30.0, "caminando": 20.0}
    },
    "results": [
        {
            "frame": 0,
            "pose_class": "de_pie",
            "confidence": 0.95,
            "keypoints_detected": true
        }
    ]
}
```

## Limitaciones Actuales

1. **Modelo de Clasificación**: El engine incluido es de ejemplo. Para uso real, entrena tu propio modelo.
2. **Topología**: Configurado para COCO (17 keypoints). Ajusta según tus necesidades.
3. **Rendimiento**: Optimizado para precisión, no velocidad máxima.
4. **Detección Multi-persona**: Actualmente procesa una persona por frame.

## Próximos Pasos

1. Entrenar modelo con datos específicos del dominio médico
2. Implementar detección multi-persona
3. Añadir análisis temporal de secuencias
4. Integrar con pipeline completo de SPPB
5. Optimizar para inferencia en tiempo real

## Soporte

Para problemas específicos:
1. Verifica los logs de error
2. Usa `--check` para verificar dependencias
3. Prueba con el video demo primero
4. Consulta documentación de TensorRT Pose

## Referencias

- [TensorRT Pose](https://github.com/NVIDIA-AI-IOT/trt_pose)
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)
- [COCO Keypoints](https://cocodataset.org/#keypoints-2017)
- [PyTorch](https://pytorch.org/)
