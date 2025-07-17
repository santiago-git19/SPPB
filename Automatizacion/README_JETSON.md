# TensorRT Pose Estimation - Jetson Nano Optimizado

## Descripción

Sistema completo de estimación de poses para Jetson Nano, optimizado para:
- **Monitoreo automático de recursos** (CPU, RAM, GPU, temperatura)
- **Manejo automático de swap** para evitar crashes por memoria
- **Limitación inteligente de CPU** para mantener temperatura controlada
- **Reportes de progreso detallados** con estadísticas en tiempo real
- **Gestión robusta de errores** por falta de memoria

## Archivos del Sistema

### Scripts Principales
- `example_trt_pose_final.py` - Script principal optimizado
- `convert_model_to_tensorrt.py` - Convertidor PyTorch → TensorRT con monitoreo completo
- `model_manager.py` - Gestor de modelos con verificación y conversión automática
- `download_models_v2.py` - Descargador automático de dependencias
- `utils/jetson_utils.py` - Utilidades específicas para Jetson Nano
- `utils/trt_pose_proc.py` - Procesador TensorRT Pose

### Archivos de Configuración
- `trt_pose_config.json` - Configuración automática generada
- `models/` - Directorio para modelos pre-entrenados
- `configs/` - Directorio para archivos de configuración

## Conversión de Modelos PyTorch → TensorRT

### 🔄 Conversión Automática (Recomendado)
```bash
# Conversión automática con monitoreo completo
python convert_model_to_tensorrt.py
```

Este proceso:
- ✅ **Configura swap de 4GB** automáticamente
- ✅ **Limita CPU a 1 core** para evitar sobrecalentamiento
- ✅ **Monitorea recursos** cada 15 segundos durante conversión
- ✅ **Reporta progreso** con temperatura y memoria
- ✅ **Pausas automáticas** si temperatura > 70°C o memoria > 80%
- ✅ **Verifica modelo convertido** con benchmark de rendimiento
- ✅ **Tiempo estimado**: 5-15 minutos en Jetson Nano

### 📊 Gestión de Modelos
```bash
# Verificar estado de modelos
python model_manager.py --check

# Conversión solo si es necesario
python model_manager.py --auto

# Validar modelo TensorRT existente
python model_manager.py --validate

# Reporte completo del sistema
python model_manager.py --status
```

### ⚙️ Configuración Manual de Swap (si falla automático)
```bash
# Crear swap de 4GB para conversión
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Verificar swap activo
swapon --show
free -h
```

### 📈 Monitoreo Durante Conversión
El script muestra logs como:
```
2025-01-XX 10:15:30 - INFO - ⚡ Iniciando conversión PyTorch → TensorRT...
2025-01-XX 10:15:45 - INFO - ✅ Modelo PyTorch: 45.2 MB
2025-01-XX 10:16:00 - INFO - 🔄 Ejecutando torch2trt...
2025-01-XX 10:16:00 - INFO -    Esto puede tomar 5-15 minutos en Jetson Nano...
2025-01-XX 10:16:30 - INFO - ⏱️ Conversión en progreso (0.5 min) - Memoria: 72.3% - Temp: 58.2°C
2025-01-XX 10:17:00 - INFO - ⏱️ Conversión en progreso (1.0 min) - Memoria: 75.1% - Temp: 61.4°C
2025-01-XX 10:18:00 - WARNING - 🌡️ ALERTA TEMPERATURA: 71.2°C - Pausando para enfriar...
2025-01-XX 10:20:45 - INFO - ✅ Conversión completada en 4.8 minutos
2025-01-XX 10:20:50 - INFO - ✅ Modelo TensorRT guardado: resnet18_baseline_att_224x224_A_epoch_249_trt.pth (38.7 MB)
2025-01-XX 10:21:00 - INFO - 📊 Resultados de rendimiento:
2025-01-XX 10:21:00 - INFO -    PyTorch: 145.2 ms por inferencia
2025-01-XX 10:21:00 - INFO -    TensorRT: 32.7 ms por inferencia
2025-01-XX 10:21:00 - INFO -    Aceleración: 4.4x
```

## Instalación Automática

### 1. Descargar Dependencias
```bash
# Ejecutar descargador automático
python download_models_v2.py
```

Este script automáticamente:
- ✅ Verifica requisitos del sistema
- ✅ Instala dependencias de Python
- ✅ Descarga modelos pre-entrenados
- ✅ Configura directorios necesarios
- ✅ Crea archivo de configuración

### 2. Instalación Manual de TensorRT Pose (si es necesario)
```bash
# Clonar repositorio TensorRT Pose
git clone https://github.com/NVIDIA-AI-IOT/trt_pose
cd trt_pose
sudo python setup.py install

# Clonar torch2trt
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
sudo python setup.py install
```

## Uso del Sistema

### 🚀 Flujo Completo Recomendado
```bash
# 1. Descargar dependencias
python download_models_v2.py

# 2. Convertir modelo a TensorRT (primera vez)
python convert_model_to_tensorrt.py

# 3. Procesar video con modelo optimizado
python example_trt_pose_final.py
```

### ⚡ Conversión Rápida
```bash
# Conversión automática solo si es necesario
python model_manager.py --auto
python example_trt_pose_final.py
```

### 🔧 Uso Avanzado
```bash
# Ejecutar procesamiento con monitoreo automático
python example_trt_pose_final.py
```

### Configuración de Swap (automático)
El sistema automáticamente:
- Detecta si hay swap activo
- Crea y activa 2GB de swap si es necesario
- Configura permisos apropiados

### Monitoreo Manual de Recursos
```bash
# Monitorear recursos por 5 minutos
python utils/jetson_utils.py 5
```

### Configuración Personalizada
Editar `trt_pose_config.json`:
```json
{
  "model_paths": {
    "topology": "configs/human_pose.json",
    "pytorch_model": "models/resnet18_baseline_att_224x224_A_epoch_249.pth",
    "tensorrt_model": "models/resnet18_baseline_att_224x224_A_epoch_249_trt.pth"
  },
  "jetson_config": {
    "enable_swap": true,
    "swap_size_gb": 2,
    "max_cpu_cores": 2,
    "memory_limit_percent": 85,
    "temperature_limit": 75
  }
}
```

## Características del Sistema

### 🔍 Monitoreo de Recursos
- **CPU**: Uso porcentual y frecuencia
- **Memoria**: RAM y swap con alertas automáticas
- **Temperatura**: Monitoreo continuo con alertas
- **GPU**: Estadísticas usando `tegrastats`
- **Red**: Estadísticas de tráfico

### 🚨 Sistema de Alertas
- **Memoria > 85%**: Libera automáticamente recursos
- **Temperatura > 75°C**: Reduce carga de procesamiento
- **CPU > 90%**: Optimiza uso de cores

### ⚙️ Optimizaciones Automáticas
- **Variables de entorno**: Configuración CUDA optimizada
- **Afinidad de CPU**: Limita a 2 cores para control térmico
- **Cache CUDA**: Deshabilitado para ahorrar memoria
- **Garbage Collection**: Automático bajo presión de memoria

### 📊 Reportes Detallados
```
=== RECURSOS JETSON (t=145.3s) ===
CPU: 67.2%
RAM: 78.5% (3.1GB/4.0GB)
SWAP: 15.2% (0.3GB/2.0GB)
Temperatura: 52.3°C

🎯 Progreso: 1250/3000 (41.7%) - FPS: 8.6 - ETA: 203.4s
```

## Flujo de Procesamiento

### 1. Inicialización
- ✅ Configurar swap automático
- ✅ Limitar cores de CPU
- ✅ Iniciar monitoreo de recursos
- ✅ Cargar modelo TensorRT optimizado

### 2. Procesamiento de Video
- 🎬 Abrir video de entrada
- 🔄 Procesar frame por frame con keypoints
- 🎨 Dibujar esqueleto en tiempo real
- 💾 Guardar video procesado
- 📈 Reportar progreso cada 10 segundos

### 3. Manejo de Recursos
- 🚨 Detectar presión de memoria
- 🧹 Liberar recursos automáticamente
- 🌡️ Monitorear temperatura
- ⏸️ Pausas automáticas para enfriamiento

### 4. Finalización
- 📊 Estadísticas completas
- 🧹 Limpieza de recursos
- 🔄 Restaurar configuración CPU

## Comandos de Monitoreo Manual

### Verificar Estado del Sistema
```bash
# Temperatura
cat /sys/devices/virtual/thermal/thermal_zone0/temp

# Memoria
free -h

# Swap
swapon --show

# GPU (si tegrastats está disponible)
tegrastats --interval 1000
```

### Configurar Swap Manualmente
```bash
# Crear swap de 2GB
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Verificar
swapon --show
```

### Limitar CPU Manualmente
```bash
# Limitar proceso a cores 0 y 1
taskset -cp 0,1 [PID]

# Verificar afinidad
taskset -p [PID]
```

## Solución de Problemas

### Error: "Out of Memory"
```bash
# Verificar memoria disponible
free -h

# Activar swap si no está activo
sudo swapon /swapfile

# Reducir resolución de video o procesar menos frames
```

### Error: "Model not found"
```bash
# Ejecutar descargador automático
python download_models_v2.py

# Verificar rutas en configuración
cat trt_pose_config.json
```

### Temperatura alta (>80°C)
```bash
# El sistema automáticamente:
# - Pausa procesamiento
# - Reduce cores de CPU
# - Limpia memoria

# Verificar ventilación del Jetson Nano
```

### Rendimiento bajo
```bash
# Verificar uso de TensorRT
# Ver logs del script para confirmar modelo optimizado

# Reducir resolución de entrada
# Procesar menos frames por segundo
```

## Logs y Depuración

### Habilitar Logs Detallados
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Archivos de Log
- Salida estándar: Progreso y estadísticas
- Alertas: Eventos de recursos y temperaturas
- Errores: Problemas de procesamiento

### Ejemplo de Log Completo
```
2025-01-XX 10:15:23 - INFO - 🚀 Configurando entorno para Jetson Nano...
2025-01-XX 10:15:24 - INFO - ✅ CUDA_CACHE_DISABLE=1
2025-01-XX 10:15:25 - INFO - ✅ Swap configurado y activado exitosamente
2025-01-XX 10:15:26 - INFO - ✅ CPU limitada a cores: [0, 1]
2025-01-XX 10:15:27 - INFO - 🔍 Monitor de recursos iniciado
2025-01-XX 10:15:28 - INFO - 📊 Configurando modelo de pose estimation...
2025-01-XX 10:15:30 - INFO - ✅ Modelo configurado exitosamente
2025-01-XX 10:15:31 - INFO - 🎬 Iniciando procesamiento de video...
2025-01-XX 10:15:45 - INFO - 🎯 Progreso: 150/3000 (5.0%) - FPS: 10.2 - ETA: 279.4s
2025-01-XX 10:15:55 - INFO - 🎯 Progreso: 250/3000 (8.3%) - FPS: 9.8 - ETA: 280.6s
2025-01-XX 10:16:05 - WARNING - ⚠️  Presión de memoria: 87.3%, liberando recursos...
```

## Rendimiento Esperado

### Jetson Nano (4GB)
- **FPS**: 8-12 fps (video 1080p)
- **Memoria**: 3-3.5GB uso pico
- **Temperatura**: 45-65°C (con ventilación)
- **Tiempo**: ~5-10 min para video de 1 minuto

### Optimizaciones Recomendadas
- Usar modelo MobileNetV2 para mayor velocidad
- Reducir resolución de entrada a 720p
- Procesar cada 2do o 3er frame
- Habilitar swap de 4GB para videos largos

---

## Próximos Pasos

1. **Ejecutar descargador**: `python download_models_v2.py`
2. **Verificar instalación**: Revisar logs de descarga
3. **Probar sistema**: `python example_trt_pose_final.py`
4. **Monitorear recursos**: Observar logs durante ejecución
5. **Optimizar configuración**: Ajustar según rendimiento
