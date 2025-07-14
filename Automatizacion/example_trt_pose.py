#!/usr/bin/env python3
"""
Ejemplo de uso de TRTPoseProcessor con un video
"""

from utils.trt_pose_proc import TRTPoseProcessor
import cv2

def main():
    # Rutas de los modelos y topología
    model_path = "models/trt_pose_resnet18.pth"  # Cambia según tu modelo
    topology_path = "models/topology.json"

    # Inicializar procesador
    print("=== Inicializando TRTPoseProcessor ===")
    processor = TRTPoseProcessor(model_path, topology_path)

    # Ruta del video a procesar
    video_path = "ruta_a_tu_video.mp4"  # Cambia esta ruta al archivo de video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video en {video_path}")
        return

    print("=== Procesando video ===")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Fin del video.")
                break

            # Procesar frame
            keypoints = processor.process_frame(frame)

            # Visualizar resultados
            if keypoints is not None:
                frame = processor.visualize_keypoints(frame, keypoints, draw_skeleton=True)
                print(f"Detectados {len(keypoints)} keypoints")

            # Mostrar frame
            cv2.imshow('TensorRT Pose - Video', frame)

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