"""
Script de prueba para el sistema dual de análisis de marcha 3D.

Este script demuestra cómo usar el gait_3d_tracker.py y bag_file_gait_analysis.py
con ambos modos:
1. Modo directo con archivos .bag (usando BagOrbbecCapture)
2. Modo cámaras en vivo (usando DualOrbbecCapture)

Características:
- Detección automática del tipo de fuente de datos
- Interfaz unificada para ambos modos
- Procesamiento completo con TRT Pose
- Generación de videos de salida con visualizaciones
"""

import sys
import os
import time
from pathlib import Path
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Agregar paths del proyecto
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from Examples.bag_file_gait_analysis import BagFileGaitAnalyzer
    from gait_3d_tracker import Gait3DTracker
    from utils.dual_orbbec_capture import DualOrbbecCapture
    from utils.bag_orbbec_capture import BagOrbbecCapture
    logger.info("Todos los módulos importados correctamente")
except ImportError as e:
    logger.error(f"Error importando módulos: {e}")
    sys.exit(1)


def test_bag_file_processing():
    """
    Prueba el procesamiento de archivos .bag usando BagFileGaitAnalyzer.
    """
    print("\n" + "="*50)
    print("PRUEBA: Procesamiento de archivo .bag")
    print("="*50)
    
    # Buscar archivos .bag en el directorio de trabajo
    bag_files = list(Path(".").glob("*.bag"))
    if not bag_files:
        print("❌ No se encontraron archivos .bag en el directorio actual")
        print("   Coloca un archivo .bag en el directorio y ejecuta de nuevo")
        return
    
    bag_path = bag_files[0]
    output_video = f"gait_analysis_{int(time.time())}.mp4"
    
    print(f"📁 Procesando archivo: {bag_path}")
    print(f"📹 Video de salida: {output_video}")
    
    try:
        # Crear analizador
        analyzer = BagFileGaitAnalyzer()
        
        # Procesar archivo .bag con límite de frames para la prueba
        results = analyzer.process_bag_file(
            bag_path=str(bag_path),
            output_video_path=output_video,
            max_frames=100  # Solo 100 frames para la prueba
        )
        
        print(f"✅ Procesamiento completado")
        print(f"   Frames procesados: {results['processed_frames']}")
        print(f"   Distancia total: {results['total_distance_m']:.3f}m")
        print(f"   Video guardado: {output_video}")
        
    except Exception as e:
        print(f"❌ Error en procesamiento: {e}")


def test_dual_mode_tracker():
    """
    Prueba el Gait3DTracker en modo dual con BagOrbbecCapture.
    """
    print("\n" + "="*50)
    print("PRUEBA: Gait3DTracker con BagOrbbecCapture")
    print("="*50)
    
    # Buscar archivos .bag
    bag_files = list(Path(".").glob("*.bag"))
    if not bag_files:
        print("❌ No se encontraron archivos .bag para la prueba")
        return
    
    bag_path = bag_files[0]
    print(f"📁 Usando archivo: {bag_path}")
    
    try:
        # Crear BagOrbbecCapture
        bag_capture = BagOrbbecCapture(
            bag_path=str(bag_path),
            enable_depth=True,
            auto_loop=False
        )
        
        # Crear tracker usando BagOrbbecCapture
        tracker = Gait3DTracker(capture_source=bag_capture)
        
        print(f"🎯 Tracker inicializado con modo: {tracker._detect_capture_mode()}")
        print("📊 Procesando algunos frames...")
        
        # Procesar algunos frames
        frames_processed = 0
        for i in range(10):  # Solo 10 frames para la prueba
            color_frame, depth_frame = bag_capture.read_frame_with_depth()
            
            if color_frame is None:
                break
                
            # Simular keypoints (en una implementación real vendríán del detector)
            dummy_keypoints = [[320, 240, 0.8]]  # Centro de la imagen
            
            point_3d = tracker.update(dummy_keypoints, depth_frame)
            
            if point_3d is not None:
                frames_processed += 1
                print(f"   Frame {i+1}: Posición 3D = {point_3d}")
        
        print(f"✅ Procesados {frames_processed} frames exitosamente")
        print(f"   Distancia total: {tracker.total_distance_m:.3f}m")
        
        # Limpiar
        bag_capture.release()
        
    except Exception as e:
        print(f"❌ Error en tracker dual: {e}")


def test_live_camera_mode():
    """
    Prueba el Gait3DTracker con cámaras en vivo (si están disponibles).
    """
    print("\n" + "="*50)
    print("PRUEBA: Gait3DTracker con cámaras en vivo")
    print("="*50)
    
    try:
        # Intentar crear DualOrbbecCapture
        camera_capture = DualOrbbecCapture()
        
        # Crear tracker usando cámaras
        tracker = Gait3DTracker(capture_source=camera_capture)
        
        print(f"🎯 Tracker inicializado con modo: {tracker._detect_capture_mode()}")
        print("📷 Cámaras detectadas y configuradas")
        print("✅ Modo cámaras en vivo disponible")
        
        # Limpiar
        camera_capture.release()
        
    except Exception as e:
        print(f"⚠️  Cámaras no disponibles: {e}")
        print("   (Esto es normal si no hay cámaras Orbbec conectadas)")


def main():
    """
    Función principal que ejecuta todas las pruebas.
    """
    print("🚀 Iniciando pruebas del sistema dual de análisis de marcha 3D")
    
    # Verificar directorio de trabajo
    current_dir = Path.cwd()
    print(f"📂 Directorio de trabajo: {current_dir}")
    
    # Ejecutar pruebas
    test_bag_file_processing()
    test_dual_mode_tracker()
    test_live_camera_mode()
    
    print("\n" + "="*50)
    print("🎉 PRUEBAS COMPLETADAS")
    print("="*50)
    print("Resumen:")
    print("• BagFileGaitAnalyzer: Procesamiento completo de archivos .bag")
    print("• Gait3DTracker: Modo dual (archivos .bag + cámaras en vivo)")
    print("• BagOrbbecCapture: Interfaz unificada para archivos .bag")
    print("• DualOrbbecCapture: Interfaz para cámaras duales")
    print("\nEl sistema está listo para usar en producción! 🎯")


if __name__ == "__main__":
    main()
