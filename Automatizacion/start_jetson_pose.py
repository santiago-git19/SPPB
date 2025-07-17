#!/bin/bash
"""
Script de Inicio Rápido - TensorRT Pose en Jetson Nano
======================================================

Script para configurar y ejecutar automáticamente el sistema completo.
Incluye verificaciones, instalación automática y ejecución optimizada.

Uso:
    chmod +x start_jetson_pose.py
    python start_jetson_pose.py [opciones]
    
Opciones:
    --setup-only    Solo configurar, no procesar video
    --monitor-only  Solo monitorear recursos por 5 minutos
    --video INPUT   Especificar video de entrada
    --frames N      Procesar solo N frames
    
Autor: Sistema de IA
Fecha: 2025
"""

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Verifica dependencias básicas"""
    logger.info("🔍 Verificando dependencias...")
    
    required_packages = ['torch', 'cv2', 'numpy', 'psutil']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('cv2', 'cv2'))
            logger.info("✅ %s", package)
        except ImportError:
            missing_packages.append(package)
            logger.error("❌ %s no encontrado", package)
    
    return len(missing_packages) == 0

def run_setup():
    """Ejecuta el setup automático"""
    logger.info("⚙️ Ejecutando setup automático...")
    
    try:
        result = subprocess.run([
            sys.executable, 'download_models_v2.py'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ Setup completado exitosamente")
            return True
        else:
            logger.error("❌ Error en setup: %s", result.stderr)
            return False
            
    except Exception as e:
        logger.error("❌ Error ejecutando setup: %s", str(e))
        return False

def monitor_resources(duration_minutes=5):
    """Ejecuta solo el monitor de recursos"""
    logger.info("📊 Iniciando monitoreo de recursos por %d minutos...", duration_minutes)
    
    try:
        result = subprocess.run([
            sys.executable, 'utils/jetson_utils.py', str(duration_minutes)
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ Monitoreo completado")
            return True
        else:
            logger.error("❌ Error en monitoreo: %s", result.stderr)
            return False
            
    except Exception as e:
        logger.error("❌ Error ejecutando monitoreo: %s", str(e))
        return False

def run_pose_estimation(video_path=None, max_frames=None):
    """Ejecuta el procesamiento de pose estimation"""
    logger.info("🤖 Iniciando procesamiento de pose estimation...")
    
    # Configurar comando
    cmd = [sys.executable, 'example_trt_pose_final.py']
    
    # Si se especifica un video diferente, modificar el script temporalmente
    if video_path:
        logger.info("📹 Usando video: %s", video_path)
        # Aquí se podría modificar el script o usar variables de entorno
        os.environ['INPUT_VIDEO'] = video_path
        
    if max_frames:
        logger.info("🎞️ Limitando a %d frames", max_frames)
        os.environ['MAX_FRAMES'] = str(max_frames)
    
    try:
        # Ejecutar en tiempo real para ver logs
        result = subprocess.run(cmd, text=True)
        
        if result.returncode == 0:
            logger.info("✅ Procesamiento completado exitosamente")
            return True
        else:
            logger.error("❌ Error en procesamiento")
            return False
            
    except KeyboardInterrupt:
        logger.info("⏹️ Procesamiento interrumpido por usuario")
        return False
    except Exception as e:
        logger.error("❌ Error ejecutando procesamiento: %s", str(e))
        return False

def print_system_info():
    """Muestra información del sistema"""
    logger.info("💻 Información del Sistema Jetson Nano")
    logger.info("=" * 50)
    
    # Información básica
    try:
        # CPU info
        with open('/proc/cpuinfo', 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'model name' in line:
                    logger.info("CPU: %s", line.split(':')[1].strip())
                    break
                    
        # Memoria
        import psutil
        memory = psutil.virtual_memory()
        logger.info("RAM: %.1fGB total, %.1fGB disponible", 
                   memory.total / (1024**3), memory.available / (1024**3))
        
        # Swap
        swap = psutil.swap_memory()
        if swap.total > 0:
            logger.info("SWAP: %.1fGB total, %.1fGB libre", 
                       swap.total / (1024**3), swap.free / (1024**3))
        else:
            logger.info("SWAP: No configurado")
            
        # Temperatura
        temp_paths = [
            '/sys/devices/virtual/thermal/thermal_zone0/temp',
            '/sys/devices/virtual/thermal/thermal_zone1/temp'
        ]
        
        for path in temp_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    temp = int(f.read().strip()) / 1000.0
                    logger.info("Temperatura: %.1f°C", temp)
                break
                
        # CUDA
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("CUDA: Disponible - %s", torch.cuda.get_device_name(0))
            else:
                logger.info("CUDA: No disponible")
        except ImportError:
            logger.info("CUDA: PyTorch no instalado")
            
    except Exception as e:
        logger.error("Error obteniendo información del sistema: %s", str(e))

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description='Script de inicio rápido para TensorRT Pose en Jetson Nano',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--setup-only', action='store_true',
                       help='Solo ejecutar setup, no procesar video')
    parser.add_argument('--monitor-only', action='store_true',
                       help='Solo monitorear recursos por 5 minutos')
    parser.add_argument('--video', type=str,
                       help='Especificar video de entrada')
    parser.add_argument('--frames', type=int,
                       help='Número máximo de frames a procesar')
    parser.add_argument('--info', action='store_true',
                       help='Mostrar información del sistema')
    
    args = parser.parse_args()
    
    # Banner
    print("🤖 TensorRT Pose Estimation - Jetson Nano")
    print("=" * 50)
    
    # Mostrar información del sistema si se solicita
    if args.info:
        print_system_info()
        return 0
    
    # Solo monitoreo
    if args.monitor_only:
        success = monitor_resources(5)
        return 0 if success else 1
    
    # Verificar directorio de trabajo
    if not Path('example_trt_pose_final.py').exists():
        logger.error("❌ No se encontró el script principal")
        logger.error("   Asegúrate de estar en el directorio correcto")
        return 1
    
    # Verificar dependencias básicas
    if not check_dependencies():
        logger.info("⚙️ Ejecutando setup para instalar dependencias...")
        if not run_setup():
            logger.error("❌ Error en setup automático")
            return 1
    
    # Solo setup
    if args.setup_only:
        logger.info("✅ Setup completado. Listo para procesar videos.")
        return 0
    
    # Verificar video de entrada
    video_path = args.video or 'Automatizacion/WIN_20250702_12_09_08_Pro.mp4'
    if not os.path.exists(video_path):
        logger.error("❌ Video de entrada no encontrado: %s", video_path)
        logger.info("💡 Especifica un video con --video RUTA")
        return 1
    
    # Ejecutar procesamiento principal
    logger.info("🚀 Iniciando procesamiento completo...")
    success = run_pose_estimation(args.video, args.frames)
    
    if success:
        logger.info("🎉 ¡Procesamiento completado exitosamente!")
        logger.info("📁 Busca el video procesado en el directorio actual")
        return 0
    else:
        logger.error("❌ Error durante el procesamiento")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("⏹️ Script interrumpido por usuario")
        sys.exit(1)
    except Exception as e:
        logger.error("❌ Error crítico: %s", str(e))
        sys.exit(1)
