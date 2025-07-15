#!/usr/bin/env python3
"""
Script simplificado para usar PoseClassifier con un video.
Procesa un video y clasifica poses usando TensorRT Pose + PoseClassificationNet.
"""

import os
import sys
import json
import argparse
from datetime import datetime

# Añadir el directorio del proyecto al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_requirements():
    """
    Verifica que las dependencias necesarias estén instaladas.
    """
    try:
        import cv2
        import numpy as np
        import torch
        print("✓ OpenCV, NumPy y PyTorch disponibles")
    except ImportError as e:
        print(f"✗ Dependencia faltante: {e}")
        return False
    
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        print("✓ TensorRT y PyCUDA disponibles")
    except ImportError as e:
        print(f"⚠ TensorRT/PyCUDA no disponible: {e}")
        print("  Esto es necesario para PoseClassificationNet")
        return False
    
    return True

def run_pose_classification(video_path, output_path=None, stats=True):
    """
    Ejecuta la clasificación de poses en un video.
    """
    try:
        from utils.pose_classifier import PoseClassifier
        
        # Verificar que el video existe
        if not os.path.exists(video_path):
            print(f"ERROR: Video no encontrado: {video_path}")
            return None
        
        # Rutas de modelos por defecto
        pose_model = "models/resnet18_baseline_att_224x224_A_epoch_249.pth"
        topology = "models/human_pose.json"
        classification_engine = "models/pose_classification.engine"
        
        # Verificar modelos
        missing_models = []
        for model_path in [pose_model, topology, classification_engine]:
            if not os.path.exists(model_path):
                missing_models.append(model_path)
        
        if missing_models:
            print("ERROR: Modelos faltantes:")
            for model in missing_models:
                print(f"  - {model}")
            print("Consulta la documentación para obtener los modelos necesarios.")
            return None
        
        print("=== Iniciando Clasificación de Poses ===")
        print(f"Video: {video_path}")
        print(f"Salida: {output_path or 'Solo visualización'}")
        
        # Inicializar clasificador
        print("Inicializando clasificador...")
        classifier = PoseClassifier(
            pose_model_path=pose_model,
            topology_path=topology,
            classification_engine_path=classification_engine
        )
        
        # Procesar video
        results = classifier.process_video(
            video_path=video_path,
            output_path=output_path,
            show_video=True,
            fps_limit=15
        )
        
        if not results:
            print("No se obtuvieron resultados.")
            return None
        
        print(f"Procesamiento completado: {len(results)} frames")
        
        # Mostrar estadísticas
        if stats:
            stats_data = classifier.get_pose_statistics()
            print("\n=== Estadísticas de Poses ===")
            print(f"Total de frames: {stats_data['total_frames']}")
            print(f"Pose más común: {stats_data['most_common_pose']}")
            print("\nDistribución:")
            for pose, percentage in stats_data['pose_percentages'].items():
                count = stats_data['pose_counts'][pose]
                print(f"  {pose}: {count} frames ({percentage:.1f}%)")
        
        return results
        
    except ImportError as e:
        print(f"ERROR: No se pudo importar PoseClassifier: {e}")
        print("Verifica que todos los modelos estén disponibles.")
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None

def create_demo_video():
    """
    Crea un video de demostración simple.
    """
    try:
        import cv2
        import numpy as np
        
        print("Creando video de demostración...")
        
        # Configuración del video
        width, height = 640, 480
        fps = 30
        duration = 10  # segundos
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('demo_poses.mp4', fourcc, fps, (width, height))
        
        total_frames = fps * duration
        
        for i in range(total_frames):
            # Crear frame negro
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Simular diferentes poses
            pose_type = (i // 60) % 4  # Cambiar pose cada 2 segundos
            
            center_x = width // 2
            center_y = height // 2
            
            # Dibujar figura básica
            if pose_type == 0:  # De pie
                # Cabeza
                cv2.circle(frame, (center_x, center_y - 150), 25, (0, 255, 0), -1)
                # Cuerpo
                cv2.rectangle(frame, (center_x - 15, center_y - 120), 
                             (center_x + 15, center_y + 50), (0, 255, 0), -1)
                # Brazos
                cv2.line(frame, (center_x - 15, center_y - 80), 
                        (center_x - 40, center_y - 40), (0, 255, 0), 5)
                cv2.line(frame, (center_x + 15, center_y - 80), 
                        (center_x + 40, center_y - 40), (0, 255, 0), 5)
                # Piernas
                cv2.line(frame, (center_x - 8, center_y + 50), 
                        (center_x - 8, center_y + 120), (0, 255, 0), 5)
                cv2.line(frame, (center_x + 8, center_y + 50), 
                        (center_x + 8, center_y + 120), (0, 255, 0), 5)
                pose_text = "DE PIE"
                
            elif pose_type == 1:  # Sentado
                # Cabeza
                cv2.circle(frame, (center_x, center_y - 100), 25, (0, 0, 255), -1)
                # Cuerpo
                cv2.rectangle(frame, (center_x - 15, center_y - 70), 
                             (center_x + 15, center_y + 20), (0, 0, 255), -1)
                # Brazos
                cv2.line(frame, (center_x - 15, center_y - 50), 
                        (center_x - 40, center_y - 10), (0, 0, 255), 5)
                cv2.line(frame, (center_x + 15, center_y - 50), 
                        (center_x + 40, center_y - 10), (0, 0, 255), 5)
                # Piernas (dobladas)
                cv2.line(frame, (center_x - 8, center_y + 20), 
                        (center_x - 8, center_y + 60), (0, 0, 255), 5)
                cv2.line(frame, (center_x + 8, center_y + 20), 
                        (center_x + 8, center_y + 60), (0, 0, 255), 5)
                # Asiento
                cv2.rectangle(frame, (center_x - 40, center_y + 20), 
                             (center_x + 40, center_y + 30), (128, 128, 128), -1)
                pose_text = "SENTADO"
                
            elif pose_type == 2:  # Caminando
                # Animación de caminar
                step = (i % 30) / 30.0
                leg_offset = int(20 * np.sin(step * np.pi))
                
                # Cabeza
                cv2.circle(frame, (center_x, center_y - 150), 25, (255, 0, 0), -1)
                # Cuerpo
                cv2.rectangle(frame, (center_x - 15, center_y - 120), 
                             (center_x + 15, center_y + 50), (255, 0, 0), -1)
                # Brazos (balanceándose)
                cv2.line(frame, (center_x - 15, center_y - 80), 
                        (center_x - 40 + leg_offset, center_y - 40), (255, 0, 0), 5)
                cv2.line(frame, (center_x + 15, center_y - 80), 
                        (center_x + 40 - leg_offset, center_y - 40), (255, 0, 0), 5)
                # Piernas (alternando)
                cv2.line(frame, (center_x - 8, center_y + 50), 
                        (center_x - 8 + leg_offset, center_y + 120), (255, 0, 0), 5)
                cv2.line(frame, (center_x + 8, center_y + 50), 
                        (center_x + 8 - leg_offset, center_y + 120), (255, 0, 0), 5)
                pose_text = "CAMINANDO"
                
            else:  # Equilibrio
                # Cabeza
                cv2.circle(frame, (center_x, center_y - 150), 25, (255, 255, 0), -1)
                # Cuerpo (inclinado)
                cv2.rectangle(frame, (center_x - 15, center_y - 120), 
                             (center_x + 15, center_y + 50), (255, 255, 0), -1)
                # Brazos extendidos
                cv2.line(frame, (center_x - 15, center_y - 80), 
                        (center_x - 80, center_y - 80), (255, 255, 0), 5)
                cv2.line(frame, (center_x + 15, center_y - 80), 
                        (center_x + 80, center_y - 80), (255, 255, 0), 5)
                # Piernas (una levantada)
                cv2.line(frame, (center_x - 8, center_y + 50), 
                        (center_x - 8, center_y + 120), (255, 255, 0), 5)
                cv2.line(frame, (center_x + 8, center_y + 50), 
                        (center_x + 30, center_y + 80), (255, 255, 0), 5)
                pose_text = "EQUILIBRIO"
            
            # Añadir texto
            cv2.putText(frame, pose_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f'Frame {i+1}/{total_frames}', 
                       (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            out.write(frame)
        
        out.release()
        print("Video de demostración creado: demo_poses.mp4")
        print("Usa este video para probar la clasificación de poses.")
        
    except ImportError:
        print("ERROR: OpenCV no está disponible para crear el video demo.")
    except Exception as e:
        print(f"ERROR creando video demo: {e}")

def main():
    """
    Función principal.
    """
    parser = argparse.ArgumentParser(description='Clasificación de poses en video')
    parser.add_argument('video_path', nargs='?', help='Ruta al video')
    parser.add_argument('--output', '-o', help='Video de salida')
    parser.add_argument('--no-stats', action='store_true', help='No mostrar estadísticas')
    parser.add_argument('--demo', action='store_true', help='Crear video demo')
    parser.add_argument('--check', action='store_true', help='Verificar dependencias')
    
    args = parser.parse_args()
    
    # Verificar dependencias
    if args.check:
        print("=== Verificando Dependencias ===")
        if check_requirements():
            print("✓ Todas las dependencias están disponibles")
        else:
            print("✗ Faltan dependencias")
        return
    
    # Crear video demo
    if args.demo:
        create_demo_video()
        return
    
    # Verificar argumentos
    if not args.video_path:
        print("=== Clasificador de Poses ===")
        print("Uso:")
        print("  python simple_pose_classification.py video.mp4")
        print("  python simple_pose_classification.py --demo")
        print("  python simple_pose_classification.py --check")
        return
    
    # Ejecutar clasificación
    results = run_pose_classification(
        video_path=args.video_path,
        output_path=args.output,
        stats=not args.no_stats
    )
    
    if results:
        print("\n¡Clasificación completada exitosamente!")
    else:
        print("\nLa clasificación falló.")

if __name__ == "__main__":
    main()
