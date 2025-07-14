#!/usr/bin/env python3
"""
Ejemplo de uso de TRTPoseProcessor
"""

from utils.trt_pose_proc import TRTPoseProcessor
import cv2

def main():
    # Opción 1: Descargar modelo automáticamente
    print("=== Descargando modelo preentrenado ===")
    model_path, topology_path = TRTPoseProcessor.download_pretrained_model('resnet18', 'models')
    
    # Opción 2: Usar rutas específicas
    # model_path = "models/trt_pose_resnet18.pth"
    # topology_path = "models/topology.json"
    
    # Opción 3: Usar rutas por defecto (busca automáticamente en models/)
    # processor = TRTPoseProcessor()
    
    # Inicializar procesador
    print("=== Inicializando TRTPoseProcessor ===")
    processor = TRTPoseProcessor(model_path, topology_path)
    
    # Ejemplo con webcam
    print("=== Iniciando captura de video ===")
    cap = cv2.VideoCapture(0)  # 0 para webcam, o ruta de archivo de video
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Procesar frame
            keypoints = processor.process_frame(frame)
            
            # Visualizar resultados
            if keypoints is not None:
                frame = processor.visualize_keypoints(frame, keypoints, draw_skeleton=True)
                print(f"Detectados {len(keypoints)} keypoints")
            
            # Mostrar frame
            cv2.imshow('TensorRT Pose', frame)
            
            # Salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nDeteniendo...")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
