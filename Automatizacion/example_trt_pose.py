#!/usr/bin/env python3
"""
Ejemplo de uso de TRTPoseProcessor con un video
"""

from utils.trt_pose_proc import TRTPoseProcessor
import cv2

def main():
    # Rutas de los modelos y topología
    model_path = "home/mobilenet/Documentos/Trabajo/trt_pose/models/trt_pose_resnet18.pth"  # Cambia según tu modelo
    topology_path = "home/mobilenet/Documentos/Trabajo/trt_pose/models/topology.json"

    # Inicializar procesador
    print("=== Inicializando TRTPoseProcessor ===")
    processor = TRTPoseProcessor(model_path, topology_path)

    # Ruta del video a procesar
    video_path = "C:\Users\USUARIO\Desktop\Trabajo\Codigo\Automatizacion\WIN_20250702_12_09_08_Pro.mp4"  # Cambia esta ruta al archivo de video
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

            # Salir con 'q'            #!/usr/bin/env python3
            """
            Ejemplo de uso de TRTPoseProcessor con un video, guardado del resultado y limitación a 15 FPS
            """
            
            from utils.trt_pose_proc import TRTPoseProcessor
            import cv2
            import time
            
            def main():
                # Rutas de los modelos y topología
                model_path = "home/mobilenet/Documentos/Trabajo/trt_pose/models/trt_pose_resnet18.pth"  # Cambia según tu modelo
                topology_path = "home/mobilenet/Documentos/Trabajo/trt_pose/models/topology.json"
            
                # Inicializar procesador
                print("=== Inicializando TRTPoseProcessor ===")
                processor = TRTPoseProcessor(model_path, topology_path)
            
                # Ruta del video a procesar
                video_path = "home/mobilenet/Documentos/Trabajo/SPPB/Automatizacion/WIN_20250702_12_09_08_Pro.mp4"  # Cambia esta ruta al archivo de video
                cap = cv2.VideoCapture(video_path)
            
                if not cap.isOpened():
                    print(f"Error: No se pudo abrir el video en {video_path}")
                    return
            
                # Configurar el escritor de video
                output_path = "home/mobilenet/Documentos/Trabajo/SPPB/Automatizacion/video_procesado.mp4"  # Ruta del video procesado
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para el video
                fps = 15  # Limitar a 15 FPS
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Ancho del video
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Alto del video
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
                print("=== Procesando video ===")
                try:
                    # Temporizador para limitar FPS
                    prev_time = 0
                    frame_interval = 1 / fps  # Intervalo entre frames (en segundos)
            
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            print("Fin del video.")
                            break
            
                        # Limitar a 15 FPS
                        current_time = time.time()
                        if current_time - prev_time < frame_interval:
                            continue
                        prev_time = current_time
            
                        # Procesar frame
                        keypoints = processor.process_frame(frame)
            
                        # Visualizar resultados
                        if keypoints is not None:
                            frame = processor.visualize_keypoints(frame, keypoints, draw_skeleton=True)
                            print(f"Detectados {len(keypoints)} keypoints")
            
                        # Guardar frame procesado en el video de salida
                        out.write(frame)
            
                        # Mostrar frame
                        cv2.imshow('TensorRT Pose - Video', frame)
            
                        # Salir con 'q'
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
            
                except KeyboardInterrupt:
                    print("\nDeteniendo...")
                finally:
                    cap.release()
                    out.release()  # Liberar el escritor de video
                    cv2.destroyAllWindows()
                    print(f"Video procesado guardado en: {output_path}")
            
            if __name__ == "__main__":
                main()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nDeteniendo...")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()