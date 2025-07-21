#!/usr/bin/env python3
"""
Script de Verificación del Sistema SPPB con TRT Pose
====================================================

Este script verifica que todos los componentes del sistema estén
correctamente configurados antes de ejecutar las pruebas SPPB.
"""

import os
import sys
import cv2
from pathlib import Path

def check_file_exists(file_path, description):
    """Verifica si un archivo existe"""
    if os.path.exists(file_path):
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description}: {file_path} (NO ENCONTRADO)")
        return False

def check_camera(camera_index, description):
    """Verifica si una cámara está disponible"""
    try:
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"✅ {description}: Cámara {camera_index} ({frame.shape})")
                cap.release()
                return True
            else:
                print(f"❌ {description}: Cámara {camera_index} (NO PUEDE LEER FRAMES)")
                cap.release()
                return False
        else:
            print(f"❌ {description}: Cámara {camera_index} (NO SE PUEDE ABRIR)")
            return False
    except Exception as e:
        print(f"❌ {description}: Error verificando cámara {camera_index}: {e}")
        return False

def check_video_file(video_path, description):
    """Verifica si un archivo de video es válido"""
    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                print(f"✅ {description}: {video_path}")
                print(f"   📊 Resolución: {frame.shape}, FPS: {fps:.1f}, Duración: {duration:.1f}s")
                cap.release()
                return True
            else:
                print(f"❌ {description}: {video_path} (NO PUEDE LEER FRAMES)")
                cap.release()
                return False
        else:
            print(f"❌ {description}: {video_path} (NO SE PUEDE ABRIR)")
            return False
    except Exception as e:
        print(f"❌ {description}: Error verificando video {video_path}: {e}")
        return False

def test_imports():
    """Verifica que todas las librerías necesarias estén disponibles"""
    print("\n🔧 VERIFICANDO IMPORTS...")
    
    required_imports = [
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("torch", "PyTorch"),
    ]
    
    optional_imports = [
        ("onnxruntime", "ONNX Runtime"),
        ("trt_pose", "TRT Pose"),
    ]
    
    success = True
    
    # Verificar imports requeridos
    for module, description in required_imports:
        try:
            __import__(module)
            print(f"✅ {description}")
        except ImportError as e:
            print(f"❌ {description}: {e}")
            success = False
    
    # Verificar imports opcionales
    for module, description in optional_imports:
        try:
            __import__(module)
            print(f"✅ {description}")
        except ImportError:
            print(f"⚠️ {description}: No disponible (opcional)")
    
    return success

def test_models():
    """Verifica que los modelos estén disponibles"""
    print("\n📁 VERIFICANDO MODELOS...")
    
    # Importar configuración
    try:
        from utils.config import Config
        config = Config()
        
        models_ok = True
        models_ok &= check_file_exists(config.trt_pose_model, "Modelo TRT Pose")
        models_ok &= check_file_exists(config.pose_topology, "Topología de poses")
        models_ok &= check_file_exists(config.pose_classifier_model, "Clasificador de poses")
        
        return models_ok
        
    except Exception as e:
        print(f"❌ Error importando configuración: {e}")
        return False

def test_cameras():
    """Verifica que las cámaras estén disponibles"""
    print("\n📹 VERIFICANDO CÁMARAS...")
    
    try:
        from utils.config import Config
        config = Config()
        
        cameras_ok = True
        
        # Verificar cámara frontal
        if isinstance(config.frontal_camera, int):
            cameras_ok &= check_camera(config.frontal_camera, "Cámara frontal")
        elif isinstance(config.frontal_camera, str):
            cameras_ok &= check_video_file(config.frontal_camera, "Video frontal")
        
        # Verificar cámara lateral
        if isinstance(config.lateral_camera, int):
            cameras_ok &= check_camera(config.lateral_camera, "Cámara lateral")
        elif isinstance(config.lateral_camera, str):
            cameras_ok &= check_video_file(config.lateral_camera, "Video lateral")
        
        return cameras_ok
        
    except Exception as e:
        print(f"❌ Error verificando cámaras: {e}")
        return False

def test_trt_pose_initialization():
    """Verifica que TRT Pose se pueda inicializar"""
    print("\n🤖 VERIFICANDO INICIALIZACIÓN TRT POSE...")
    
    try:
        from utils.config import Config
        from utils.trt_pose_proc import TRTPoseProcessor
        from utils.trt_pose_classifier import create_pose_classifier
        
        config = Config()
        
        # Inicializar procesador
        print("🔧 Inicializando TRTPoseProcessor...")
        processor = TRTPoseProcessor(
            model_path=config.trt_pose_model,
            topology_path=config.pose_topology
        )
        print("✅ TRTPoseProcessor inicializado")
        
        # Inicializar clasificador
        print("🔧 Inicializando TRTPoseClassifier...")
        classifier = create_pose_classifier(
            model_path=config.pose_classifier_model,
            keypoint_format=config.keypoint_format,
            sequence_length=config.sequence_length,
            confidence_threshold=config.confidence_threshold
        )
        print("✅ TRTPoseClassifier inicializado")
        
        return True
        
    except Exception as e:
        print(f"❌ Error inicializando TRT Pose: {e}")
        return False

def test_calibration():
    """Verifica la configuración de calibración"""
    print("\n📏 VERIFICANDO CALIBRACIÓN...")
    
    try:
        from utils.config import Config
        config = Config()
        
        ratio = config.pixel_to_meter_ratio
        print(f"📐 Ratio píxel/metro: {ratio}")
        
        if ratio <= 0:
            print("❌ Ratio píxel/metro debe ser positivo")
            return False
        elif ratio > 0.1:
            print("⚠️ Ratio píxel/metro parece muy alto (>0.1)")
            print("   Verificar calibración")
        elif ratio < 0.001:
            print("⚠️ Ratio píxel/metro parece muy bajo (<0.001)")
            print("   Verificar calibración")
        else:
            print("✅ Ratio píxel/metro parece razonable")
        
        # Mostrar ejemplos de conversión
        print(f"📊 Ejemplos de conversión:")
        print(f"   100 píxeles = {100 * ratio:.3f} metros")
        print(f"   1 metro = {1 / ratio:.0f} píxeles")
        
        return True
        
    except Exception as e:
        print(f"❌ Error verificando calibración: {e}")
        return False

def test_system_performance():
    """Prueba básica de rendimiento"""
    print("\n⚡ PRUEBA DE RENDIMIENTO...")
    
    try:
        import time
        import numpy as np
        from utils.config import Config
        
        config = Config()
        
        # Simular procesamiento de imagen
        start_time = time.time()
        
        # Crear imagen de prueba
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Simular múltiples operaciones
        for i in range(10):
            # Simular procesamiento
            processed = cv2.resize(test_image, (224, 224))
            _ = np.mean(processed)
        
        elapsed = time.time() - start_time
        fps_estimate = 10 / elapsed
        
        print(f"📈 Tiempo para 10 frames: {elapsed:.3f}s")
        print(f"📊 FPS estimado: {fps_estimate:.1f}")
        
        if fps_estimate < 5:
            print("⚠️ Rendimiento bajo - considerar optimizaciones")
        elif fps_estimate < 15:
            print("✅ Rendimiento aceptable")
        else:
            print("✅ Buen rendimiento")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en prueba de rendimiento: {e}")
        return False

def main():
    """Función principal de verificación"""
    print("🏥 VERIFICACIÓN DEL SISTEMA SPPB CON TRT POSE")
    print("=" * 60)
    
    # Lista de verificaciones
    checks = [
        ("Imports", test_imports),
        ("Modelos", test_models),
        ("Cámaras", test_cameras),
        ("Inicialización TRT Pose", test_trt_pose_initialization),
        ("Calibración", test_calibration),
        ("Rendimiento", test_system_performance),
    ]
    
    results = {}
    
    # Ejecutar verificaciones
    for check_name, check_function in checks:
        try:
            result = check_function()
            results[check_name] = result
        except Exception as e:
            print(f"❌ Error en verificación {check_name}: {e}")
            results[check_name] = False
    
    # Resumen final
    print("\n" + "=" * 60)
    print("📋 RESUMEN DE VERIFICACIÓN")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for check_name, result in results.items():
        status = "✅ PASÓ" if result else "❌ FALLÓ"
        print(f"{check_name:.<30} {status}")
        if result:
            passed += 1
    
    print(f"\n📊 Resultado: {passed}/{total} verificaciones pasaron")
    
    if passed == total:
        print("🎉 ¡Sistema completamente verificado!")
        print("   Puede proceder a ejecutar las pruebas SPPB")
    elif passed >= total * 0.8:
        print("⚠️ Sistema mayormente funcional")
        print("   Revisar las verificaciones que fallaron")
    else:
        print("🚨 Sistema requiere configuración adicional")
        print("   Resolver los problemas antes de continuar")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
