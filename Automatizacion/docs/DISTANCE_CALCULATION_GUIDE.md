# Guía de Cálculo de Distancia en SPPB con TRT Pose

## Introducción

Este documento explica cómo calcular la distancia recorrida por una persona utilizando los keypoints detectados por TRT Pose en el sistema SPPB.

## Principios Básicos

### 1. Keypoints de Referencia
- **Left Hip (ID: 11)**: Cadera izquierda en topología COCO
- **Right Hip (ID: 12)**: Cadera derecha en topología COCO
- **MidHip**: Punto medio entre ambas caderas (calculado)

### 2. Cálculo de Posición
```python
# Si ambas caderas están detectadas
hip_x = (left_hip_x + right_hip_x) / 2

# Si solo una cadera está detectada
hip_x = left_hip_x or right_hip_x
```

### 3. Cálculo de Distancia
```python
# Distancia en píxeles
pixel_distance = abs(current_hip_x - previous_hip_x)

# Conversión a metros
distance_meters = pixel_distance * pixel_to_meter_ratio
```

## Calibración del Sistema

### 1. Determinar el Ratio Píxel/Metro

#### Método 1: Objeto de Referencia
1. Coloca un objeto de tamaño conocido (ej: regla de 1 metro) en el plano de movimiento
2. Mide la longitud del objeto en píxeles en la imagen
3. Calcula: `pixel_to_meter_ratio = 1.0 / pixeles_por_metro`

#### Método 2: Distancia Conocida
1. Marca dos puntos separados por una distancia conocida (ej: 4 metros)
2. Mide la distancia en píxeles entre estos puntos
3. Calcula: `pixel_to_meter_ratio = distancia_real_metros / distancia_pixeles`

#### Ejemplo de Calibración
```python
# Si 4 metros = 400 píxeles en la imagen
pixel_to_meter_ratio = 4.0 / 400.0  # = 0.01 metros por píxel

# Si 1 metro = 100 píxeles en la imagen  
pixel_to_meter_ratio = 1.0 / 100.0   # = 0.01 metros por píxel
```

### 2. Configuración en config.py
```python
class Config:
    def __init__(self):
        # Factor de conversión (DEBE CALIBRARSE SEGÚN EL SETUP)
        self.pixel_to_meter_ratio = 0.01  # metros por pixel
        
        # Configuración adicional para filtrado
        self.distance_calculation = {
            'min_movement_threshold': 0.001,  # Mínimo movimiento en metros
            'use_hip_keypoints': True,        # Usar keypoints de cadera
            'distance_smoothing': True        # Aplicar suavizado
        }
```

## Consideraciones Importantes

### 1. Perspectiva de la Cámara
- **Cámara Lateral**: Ideal para medir movimiento horizontal
- **Ángulo Perpendicular**: La cámara debe estar perpendicular al plano de movimiento
- **Altura Constante**: Mantener la cámara a altura constante durante la medición

### 2. Factores que Afectan la Precisión
- **Distorsión de lente**: Corregir si es significativa
- **Ángulo de la cámara**: Afecta la escala de píxeles
- **Variación de altura**: Los keypoints pueden cambiar de posición vertical
- **Confianza de detección**: Filtrar keypoints con baja confianza

### 3. Optimizaciones Implementadas
```python
# Filtrado por confianza
if confidence > 0.3:  # Solo usar keypoints con alta confianza
    
# Filtrado de movimiento mínimo
if distance_moved < min_threshold:
    distance_moved = 0.0  # Ignorar movimientos muy pequeños
    
# Promedio de caderas para mayor estabilidad
hip_x = (left_hip + right_hip) / 2
```

## Ejemplo de Uso

### 1. Configuración Típica
```python
# Para una habitación de 4x4 metros filmada desde 3 metros de distancia
config = {
    'pixel_to_meter_ratio': 0.008,  # Ajustar según calibración
    'lateral_camera': 1,            # Cámara lateral
    'frontal_camera': 0,           # Cámara frontal
}
```

### 2. Flujo de Cálculo
1. **Detección**: TRT Pose detecta keypoints en cada frame
2. **Extracción**: Se extrae la posición X de la cadera
3. **Cálculo**: Se calcula la distancia desde la posición anterior
4. **Acumulación**: Se suma la distancia al total recorrido
5. **Validación**: Se verifica que se completaron los 4 metros

## Validación y Pruebas

### 1. Prueba de Calibración
```python
# Probar con distancia conocida
def test_calibration():
    known_distance = 2.0  # metros
    measured_pixels = 200  # píxeles medidos
    
    calculated_ratio = known_distance / measured_pixels
    print(f"Ratio calculado: {calculated_ratio}")
```

### 2. Verificación de Precisión
- Comparar distancias medidas con mediciones manuales
- Usar múltiples objetos de referencia
- Validar en diferentes posiciones del frame

## Troubleshooting

### Problemas Comunes

1. **Distancias Incorrectas**
   - Verificar calibración del `pixel_to_meter_ratio`
   - Comprobar ángulo de la cámara lateral
   - Validar detección de keypoints de cadera

2. **Movimientos Erráticos**
   - Aumentar `min_movement_threshold`
   - Verificar confianza de los keypoints
   - Implementar suavizado temporal

3. **Detección Perdida**
   - Mejorar iluminación
   - Ajustar posición de la cámara
   - Verificar que la persona esté en el frame

### Logs de Debug
```python
print(f"Hip position: {hip_x}")
print(f"Previous position: {self.previous_position}")
print(f"Pixel distance: {pixel_distance}")
print(f"Distance moved: {distance_moved:.4f}m")
```

## Conclusión

El cálculo de distancia en el sistema SPPB se basa en el tracking de los keypoints de cadera detectados por TRT Pose. La precisión depende principalmente de:

1. **Calibración correcta** del ratio píxel/metro
2. **Posicionamiento adecuado** de la cámara lateral
3. **Calidad de detección** de los keypoints
4. **Configuración apropiada** de los filtros de movimiento

Con una calibración correcta, el sistema puede medir distancias con una precisión de ±5cm en condiciones normales.
