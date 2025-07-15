#!/usr/bin/env python3
"""
Script de ejemplo para usar PoseClassifier con videos.
Procesa un video y clasifica poses frame por frame usando TensorRT Pose + PoseClassificationNet.
"""

import os
import sys
import cv2
import json
import argparse
from datetime import datetime

# Añadir el directorio del proyecto al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.pose_classifier import PoseClassifier

def main():
    parser = argparse.ArgumentParser(description='Clasificador de poses en video')
    parser.add_argument('video_path', help='Ruta al video de entrada')
    parser.add_argument('--output', '-o', help='Ruta al video de salida (opcional)')
    parser.add_argument('--pose-model', default='models/resnet18_baseline_att_224x224_A_epoch_249.pth',
                       help='Ruta al modelo TensorRT Pose')
    parser.add_argument('--topology', default='models/human_pose.json',
                       help='Ruta al archivo de topología')
    parser.add_argument('--classification-engine', default='models/pose_classification.engine',
                       help='Ruta al engine de clasificación TensorRT')
    parser.add_argument('--fps-limit', type=int, default=15,
                       help='Límite de FPS para procesamiento')
    parser.add_argument('--no-display', action='store_true',
                       help='No mostrar video en tiempo real')
    parser.add_argument('--export-results', help='Exportar resultados a archivo JSON')
    parser.add_argument('--stats', action='store_true',
                       help='Mostrar estadísticas al final')
    
    args = parser.parse_args()
    
    # Verificar que el video existe
    if not os.path.exists(args.video_path):
        print(f"ERROR: Video no encontrado: {args.video_path}")
        return 1
    
    # Verificar modelos
    models_missing = []
    if not os.path.exists(args.pose_model):
        models_missing.append(args.pose_model)
    if not os.path.exists(args.topology):
        models_missing.append(args.topology)
    if not os.path.exists(args.classification_engine):
        models_missing.append(args.classification_engine)
    
    if models_missing:
        print("ERROR: Modelos faltantes:")
        for model in models_missing:
            print(f"  - {model}")
        print("Por favor, asegúrate de tener todos los modelos necesarios.")
        print("Consulta la documentación para descargar los modelos.")
        return 1
    
    print("=== Clasificador de Poses en Video ===")
    print(f"Video: {args.video_path}")
    print(f"Modelo TensorRT Pose: {args.pose_model}")
    print(f"Engine de clasificación: {args.classification_engine}")
    print(f"FPS límite: {args.fps_limit}")
    print()
    
    try:
        # Inicializar clasificador
        print("Inicializando clasificador...")
        classifier = PoseClassifier(
            pose_model_path=args.pose_model,
            topology_path=args.topology,
            classification_engine_path=args.classification_engine
        )
        
        # Procesar video
        print("Procesando video...")
        results = classifier.process_video(
            video_path=args.video_path,
            output_path=args.output,
            show_video=not args.no_display,
            fps_limit=args.fps_limit
        )
        
        print("\n=== Procesamiento Completado ===")
        print(f"Total de frames procesados: {len(results)}")
        
        # Mostrar estadísticas
        if args.stats:
            stats = classifier.get_pose_statistics()
            print("\n=== Estadísticas de Poses ===")
            print(f"Total de frames: {stats['total_frames']}")
            print(f"Pose más común: {stats['most_common_pose']}")
            print("\nDistribución de poses:")
            for pose, percentage in stats['pose_percentages'].items():
                count = stats['pose_counts'][pose]
                print(f"  {pose}: {count} frames ({percentage:.1f}%)")
        
        # Exportar resultados
        if args.export_results:
            print(f"\nExportando resultados a: {args.export_results}")
            export_data = {
                'metadata': {
                    'video_path': args.video_path,
                    'processing_date': datetime.now().isoformat(),
                    'pose_model': args.pose_model,
                    'classification_engine': args.classification_engine,
                    'fps_limit': args.fps_limit
                },
                'statistics': classifier.get_pose_statistics(),
                'results': results
            }
            
            with open(args.export_results, 'w') as f:
                json.dump(export_data, f, indent=2)
        
        print("\n¡Procesamiento completado exitosamente!")
        
    except KeyboardInterrupt:
        print("\nProcesamiento interrumpido por el usuario.")
        return 1
    except Exception as e:
        print(f"\nERROR: {e}")
        return 1
    
    return 0

def create_demo_video():
    """
    Crea un video de demostración simple para pruebas.
    """
    print("Creando video de demostración...")
    
    # Crear un video simple con formas geométricas
    width, height = 640, 480
    fps = 30
    duration = 5  # segundos
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('demo_video.mp4', fourcc, fps, (width, height))
    
    total_frames = fps * duration
    
    for i in range(total_frames):
        # Crear frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Dibujar figura simple que simule una persona
        # (esto es solo para demostración, no representa poses reales)
        center_x = width // 2
        center_y = height // 2
        
        # Cabeza
        cv2.circle(frame, (center_x, center_y - 100), 30, (0, 255, 0), -1)
        
        # Cuerpo
        cv2.rectangle(frame, (center_x - 20, center_y - 70), 
                     (center_x + 20, center_y + 50), (0, 255, 0), -1)
        
        # Brazos (animados)
        arm_angle = (i / total_frames) * 360
        arm_x = int(30 * np.cos(np.radians(arm_angle)))
        arm_y = int(30 * np.sin(np.radians(arm_angle)))
        
        cv2.line(frame, (center_x - 20, center_y - 30), 
                (center_x - 20 + arm_x, center_y - 30 + arm_y), (0, 255, 0), 5)
        cv2.line(frame, (center_x + 20, center_y - 30), 
                (center_x + 20 - arm_x, center_y - 30 + arm_y), (0, 255, 0), 5)
        
        # Piernas
        cv2.line(frame, (center_x - 10, center_y + 50), 
                (center_x - 10, center_y + 120), (0, 255, 0), 5)
        cv2.line(frame, (center_x + 10, center_y + 50), 
                (center_x + 10, center_y + 120), (0, 255, 0), 5)
        
        # Texto
        cv2.putText(frame, f'Frame {i+1}/{total_frames}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print("Video de demostración creado: demo_video.mp4")

if __name__ == "__main__":
    import numpy as np
    
    # Si se ejecuta sin argumentos, mostrar ayuda
    if len(sys.argv) == 1:
        print("=== Clasificador de Poses en Video ===")
        print("Uso:")
        print("  python example_pose_classification.py video.mp4")
        print("  python example_pose_classification.py video.mp4 --output procesado.mp4")
        print("  python example_pose_classification.py video.mp4 --stats --export-results results.json")
        print()
        print("Opciones:")
        print("  --demo    Crear video de demostración")
        print("  --help    Mostrar ayuda completa")
        print()
        
        # Opción para crear demo
        if '--demo' in sys.argv:
            create_demo_video()
        
        sys.exit(0)
    
    sys.exit(main())
