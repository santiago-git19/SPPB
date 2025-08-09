#!/usr/bin/env python3
"""
Script de Diagnóstico TensorRT Pose Processor
=============================================

Script para verificar que el sistema TensorRT funciona correctamente
antes de procesar videos o usar en producción.

Ejecutar con:
    python test_tensorrt_pose.py

Autor: Sistema de IA
Fecha: 2025
"""

import os
import sys
import time
import numpy as np

# Agregar el directorio actual al path para importar
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from mediapipe_pose_proc import MediaPipePoseProcessor, TRT_AVAILABLE
except ImportError as e:
    print(f"❌ Error importando MediaPipePoseProcessor: {e}")
    print("💡 Asegúrese de que mediapipe_pose_proc.py esté en el mismo directorio")
    exit(1)

def main():
    """Función principal de diagnóstico"""
    print("🔧 SCRIPT DE DIAGNÓSTICO TENSORRT POSE PROCESSOR")
    print("=" * 60)
    
    # Verificar disponibilidad de TensorRT
    if not TRT_AVAILABLE:
        print("❌ TensorRT no está disponible")
        print("💡 Instale TensorRT y PyCUDA para usar esta clase")
        print("\n📋 Instalación requerida:")
        print("   1. TensorRT: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/")
        print("   2. PyCUDA: pip install pycuda")
        exit(1)
    
    print("✅ TensorRT y PyCUDA disponibles")
    
    # Buscar modelo TensorRT
    print("\n🔍 Buscando modelo TensorRT...")
    
    # Rutas posibles del modelo
    possible_paths = [
        "pose_landmark_lite_fp16.engine",
        "../models/pose_landmark_lite_fp16.engine", 
        "models/pose_landmark_lite_fp16.engine",
        "Documentos/Trabajo/SPPB/Automatizacion/models/pose_landmark_lite_fp16.engine"
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            print(f"✅ Modelo encontrado: {path}")
            break
    
    if model_path is None:
        print("❌ No se encontró el modelo pose_landmark_lite_fp16.engine")
        print("\n🔍 Buscando cualquier modelo .engine...")
        
        # Buscar cualquier archivo .engine
        for root, dirs, files in os.walk("."):
            for file in files:
                if file.endswith('.engine'):
                    full_path = os.path.join(root, file)
                    print(f"📁 Encontrado: {full_path}")
                    if model_path is None:
                        model_path = full_path
        
        if model_path is None:
            print("❌ No se encontraron archivos .engine")
            print("💡 Asegúrese de tener el modelo pose_landmark_lite_fp16.engine")
            exit(1)
        else:
            print(f"🎯 Usando: {model_path}")
    
    # Crear procesador y ejecutar diagnósticos
    print(f"\n🚀 Inicializando procesador con modelo: {os.path.basename(model_path)}")
    
    try:
        processor = MediaPipePoseProcessor(
            model_path=model_path,
            input_width=256,
            input_height=256,
            confidence_threshold=0.5
        )
        
        # Ejecutar diagnósticos completos
        success = processor.run_diagnostics()
        
        # Resultados finales
        print("\n" + "🎯 RESULTADOS FINALES" + " " + "🎯")
        print("=" * 60)
        
        if success:
            print("🎉 ¡EXCELENTE! El sistema está funcionando perfectamente")
            print("\n✅ Puede proceder a:")
            print("   • Procesar videos con mediapipe_pose_proc.py")
            print("   • Integrar en sus aplicaciones")
            print("   • Usar en producción")
            
            print("\n📊 Información del sistema:")
            print(f"   • Modelo: {os.path.basename(model_path)}")
            print(f"   • Entrada: {processor.input_width}x{processor.input_height}")
            print(f"   • Confianza: {processor.confidence_threshold}")
            print(f"   • Keypoints: 33 (MediaPipe BlazePose)")
            
        else:
            print("❌ Se detectaron problemas en el sistema")
            print("\n🔧 Acciones recomendadas:")
            print("   1. Verificar que el modelo .engine sea compatible")
            print("   2. Regenerar el modelo en esta máquina:")
            print("      trtexec --onnx=pose_landmark_lite.onnx --saveEngine=pose_landmark_lite_fp16.engine --fp16")
            print("   3. Verificar instalación de TensorRT y PyCUDA")
            print("   4. Comprobar drivers NVIDIA y CUDA")
            
        # Limpiar recursos
        processor.cleanup()
        
    except Exception as e:
        print(f"\n❌ ERROR CRÍTICO: {e}")
        print("\n🔧 Soluciones posibles:")
        print("   1. Verificar que CUDA esté instalado y funcionando")
        print("   2. Verificar que los drivers NVIDIA estén actualizados")
        print("   3. Regenerar el modelo .engine en esta máquina")
        print("   4. Verificar permisos de archivos")
        return False
    
    return success

def test_model_compatibility(model_path: str):
    """
    Prueba la compatibilidad del modelo con trtexec
    
    Args:
        model_path: Ruta al modelo .engine
    """
    print(f"\n🧪 Probando compatibilidad del modelo con trtexec...")
    
    try:
        import subprocess
        
        # Ejecutar trtexec para probar el modelo
        cmd = f"trtexec --loadEngine={model_path} --iterations=1 --avgRuns=1"
        
        print(f"   📝 Ejecutando: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("   ✅ Modelo compatible con TensorRT")
            
            # Extraer información del modelo
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if 'Created input binding' in line:
                    print(f"   📐 {line.strip()}")
                elif 'Created output binding' in line:
                    print(f"   📊 {line.strip()}")
                elif 'Average on' in line and 'GPU latency' in line:
                    print(f"   ⏱️ {line.strip()}")
                    
        else:
            print("   ❌ Modelo incompatible o error en TensorRT")
            print(f"   📝 Error: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("   ⏰ Timeout - El modelo tarda mucho en cargar")
    except FileNotFoundError:
        print("   ❌ trtexec no encontrado - TensorRT no instalado correctamente")
    except Exception as e:
        print(f"   ❌ Error probando modelo: {e}")

if __name__ == "__main__":
    print("🚀 Iniciando diagnósticos del sistema TensorRT...")
    print(f"📅 Fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🖥️ Directorio: {os.getcwd()}")
    
    success = main()
    
    print(f"\n⏰ Diagnósticos completados: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success:
        exit(0)  # Éxito
    else:
        exit(1)  # Error
