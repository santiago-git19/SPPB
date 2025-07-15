from utils.pose_classifier import PoseClassifier
import cv2
import time
import os

def main():
    print("=== Ejemplo de Clasificación de Poses ===")
    
    # Configuración de rutas (ajusta según tu setup)
    pose_model_path = "models/resnet18_baseline_att_224x224_A_epoch_249.pth"
    topology_path = "models/human_pose.json"
    classification_engine_path = "models/pose_classification.engine"
    
    # Verificar que los modelos existen
    required_files = [pose_model_path, topology_path, classification_engine_path]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("ERROR: Faltan archivos de modelos:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nEjecuta: python download_models.py")
        return
    
    # Ruta del video a procesar
    video_path = "WIN_20250702_12_09_08_Pro.mp4"  # Ajusta esta ruta
    
    # Verificar que el video existe
    if not os.path.exists(video_path):
        print(f"ERROR: Video no encontrado: {video_path}")
        print("Crea un video de demostración: python simple_pose_classification.py --demo")
        return
    
    try:
        # Inicializar el clasificador de poses
        print("Inicializando clasificador de poses...")
        classifier = PoseClassifier(
            pose_model_path=pose_model_path,
            topology_path=topology_path,
            classification_engine_path=classification_engine_path
        )
        
        # Procesar video con clasificación de poses
        print(f"Procesando video: {video_path}")
        output_path = "video_con_poses_clasificadas.mp4"
        
        results = classifier.process_video(
            video_path=video_path,
            output_path=output_path,
            show_video=True,
            fps_limit=15
        )
        
        if results:
            print("\n=== Resultados ===")
            print(f"Frames procesados: {len(results)}")
            
            # Mostrar estadísticas
            stats = classifier.get_pose_statistics()
            print(f"Pose más común: {stats['most_common_pose']}")
            print(f"Total de frames: {stats['total_frames']}")
            
            print("\nDistribución de poses:")
            for pose, percentage in stats['pose_percentages'].items():
                count = stats['pose_counts'][pose]
                print(f"  {pose}: {count} frames ({percentage:.1f}%)")
            
            print(f"\nVideo procesado guardado en: {output_path}")
        else:
            print("No se obtuvieron resultados.")
            
    except Exception as e:
        print(f"ERROR: {e}")
        print("Verifica que todos los modelos estén correctamente instalados.")

def test_single_frame():
    """
    Ejemplo de procesamiento de un solo frame.
    """
    print("\n=== Procesamiento de Frame Individual ===")
    
    try:
        # Inicializar clasificador
        classifier = PoseClassifier(
            pose_model_path="models/resnet18_baseline_att_224x224_A_epoch_249.pth",
            topology_path="models/human_pose.json",
            classification_engine_path="models/pose_classification.engine"
        )
        
        # Crear un frame de prueba (imagen negra)
        import numpy as np
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Procesar frame
        pose_class, confidence, keypoints = classifier.process_single_frame(test_frame)
        
        print(f"Pose detectada: {pose_class}")
        print(f"Confianza: {confidence:.2f}")
        print(f"Keypoints detectados: {keypoints is not None}")
        
    except Exception as e:
        print(f"Error en procesamiento individual: {e}")

if __name__ == "__main__":
    main()
    
    # Ejemplo adicional de procesamiento de frame individual
    # test_single_frame()