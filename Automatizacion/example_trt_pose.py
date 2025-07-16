from utils.trt_pose_proc import TRTPoseProcessor
import cv2
import time
import os

def main():
    print("=== Procesamiento de Keypoints y Exoesqueleto ===")
    
    # Configuración de rutas (ajusta según tu setup)
    pose_model_path = "home/mobilenet/Documentos/Trabajo/models/resnet18_baseline_att_224x224_A_epoch_249.pth"
    topology_path = "home/mobilenet/Documentos/Trabajo/models/human_pose.json"
    
    # Verificar que los modelos existen
    required_files = [pose_model_path, topology_path]
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
        print("Coloca tu video en el directorio del proyecto")
        return
    
    try:
        # Inicializar el procesador TensorRT Pose
        print("Inicializando TRTPoseProcessor...")
        processor = TRTPoseProcessor(pose_model_path, topology_path)
        
        # Abrir video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"ERROR: No se pudo abrir el video: {video_path}")
            return
        
        # Configurar video de salida
        output_path = "video_con_exoesqueleto.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 15  # Limitar FPS
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Procesando video: {video_path}")
        print(f"Resolución: {width}x{height}")
        print(f"FPS: {fps}")
        
        # Variables para control de FPS
        frame_count = 0
        prev_time = 0
        frame_interval = 1.0 / fps
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Fin del video.")
                break
            
            # Controlar FPS
            current_time = time.time()
            if current_time - prev_time < frame_interval:
                continue
            prev_time = current_time
            
            # Procesar frame para extraer keypoints
            keypoints = processor.process_frame(frame)
            
            # Visualizar keypoints y exoesqueleto
            if keypoints is not None:
                frame = processor.visualize_keypoints(frame, keypoints, draw_skeleton=True)
                print(f"Frame {frame_count}: {len(keypoints)} keypoints detectados")
            else:
                print(f"Frame {frame_count}: No se detectaron keypoints")
            
            # Añadir información al frame
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Keypoints: {len(keypoints) if keypoints is not None else 0}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Guardar frame procesado
            out.write(frame)
            
            # Mostrar frame
            cv2.imshow('TensorRT Pose - Exoesqueleto', frame)
            
            # Salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
            
            # Mostrar progreso cada 30 frames
            if frame_count % 30 == 0:
                print(f"Procesados {frame_count} frames...")
        
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print("\n=== Procesamiento Completado ===")
        print(f"Frames procesados: {frame_count}")
        print(f"Video con exoesqueleto guardado en: {output_path}")
            
    except Exception as e:
        print(f"ERROR: {e}")
        print("Verifica que el modelo TensorRT Pose esté correctamente instalado.")

def test_single_frame():
    """
    Ejemplo de procesamiento de un solo frame.
    """
    print("\n=== Procesamiento de Frame Individual ===")
    
    try:
        # Inicializar procesador
        processor = TRTPoseProcessor(
            model_path="models/resnet18_baseline_att_224x224_A_epoch_249.pth",
            topology_path="models/human_pose.json"
        )
        
        # Crear un frame de prueba (imagen negra)
        import numpy as np
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Procesar frame
        keypoints = processor.process_frame(test_frame)
        
        print(f"Keypoints detectados: {keypoints is not None}")
        if keypoints is not None:
            print(f"Número de keypoints: {len(keypoints)}")
        
    except Exception as e:
        print(f"Error en procesamiento individual: {e}")

if __name__ == "__main__":
    main()
    
    # Ejemplo adicional de procesamiento de frame individual
    # test_single_frame()