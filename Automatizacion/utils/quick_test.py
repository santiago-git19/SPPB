#!/usr/bin/env python3
"""
Prueba Rápida TensorRT Pose Processor
=====================================

Script para una verificación rápida del sistema sin diagnósticos completos.

Ejecutar con:
    python quick_test.py

Autor: Sistema de IA
Fecha: 2025
"""

import os
import sys
import time
import numpy as np
import cv2

# Agregar el directorio actual al path para importar
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def quick_test():
    """Prueba rápida del sistema"""
    print("⚡ PRUEBA RÁPIDA TENSORRT POSE PROCESSOR")
    print("=" * 50)
    
    try:
        from mediapipe_pose_proc import MediaPipePoseProcessor, TRT_AVAILABLE
        
        if not TRT_AVAILABLE:
            print("❌ TensorRT no disponible")
            return False
        
        # Buscar modelo
        model_paths = [
            "pose_landmark_lite_fp16.engine",
            "../models/pose_landmark_lite_fp16.engine",
            "models/pose_landmark_lite_fp16.engine"
        ]
        
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if not model_path:
            print("❌ Modelo no encontrado")
            return False
        
        print(f"✅ Modelo: {os.path.basename(model_path)}")
        
        # Crear procesador
        print("🔄 Inicializando...")
        processor = MediaPipePoseProcessor(model_path, confidence_threshold=0.3)
        
        # Crear imagen de prueba simple
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Una inferencia de prueba
        print("🧪 Ejecutando inferencia...")
        start = time.time()
        keypoints = processor.process_frame(test_image)
        elapsed = (time.time() - start) * 1000
        
        if keypoints is not None:
            print(f"✅ Éxito en {elapsed:.1f}ms")
            print(f"🎯 Keypoints: {len(keypoints)}")
            
            # Contar keypoints con confianza > 0.1
            valid_kp = np.sum(keypoints[:, 2] > 0.1)
            print(f"📊 Válidos: {valid_kp}/33")
            
            # Prueba de visualización
            try:
                visualized = processor.visualize_keypoints(test_image, keypoints)
                print("✅ Visualización OK")
            except:
                print("⚠️ Error en visualización")
            
            processor.cleanup()
            return True
        else:
            print("❌ Inferencia falló")
            processor.cleanup()
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    start_time = time.time()
    success = quick_test()
    total_time = time.time() - start_time
    
    print(f"\n⏱️ Tiempo total: {total_time:.2f}s")
    
    if success:
        print("🎉 SISTEMA OK - Listo para usar")
    else:
        print("❌ PROBLEMAS DETECTADOS - Ejecutar test_tensorrt_pose.py para más detalles")
