import time
import cv2
import sys
sys.path.append('/home/work/openpose/build/python')
from openpose import pyopenpose as op

def process_video(input_path, output_path, codec='XVID'):
    # Inicializa temporizador global
    start_time = time.time()
    
    # 1. Parámetros de configuración de OpenPose
    params = {
        "model_folder": "/home/work/openpose/models/",
        "hand": False,                  # no detectar manos
        "face": False,                  # opcional: no cara
        "net_resolution": "-1x368",
        "disable_blending": False,
        "render_threshold": 0.1
    }
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # 2. Abre el vídeo de entrada
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: no se pudo abrir el vídeo {input_path}")
        sys.exit(1)

    # 3. Lee propiedades del vídeo
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Propiedades del vídeo de entrada: {w}x{h} @ {fps:.2f} FPS")

    # 4. Prepara el VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        print(f"Error: no se pudo crear el vídeo de salida {output_path} con codec {codec}")
        cap.release()
        sys.exit(1)
    print(f"Usando codec {codec}. Salida en: {output_path}")

    # 5. Procesa frame a frame con debug
    frame_idx = 0
    written_frames = 0
    processing_start = time.time()
    while True:
        ret, frame = cap.read() # Saca un frame del video
        if not ret:
            break  # fin del vídeo

        # Prepara el Datum
        datum = op.Datum() # Representa el frame a procesar
        datum.cvInputData = frame
        vectorDatum = op.VectorDatum()
        vectorDatum.append(datum)

        # Procesa
        success = opWrapper.emplaceAndPop(vectorDatum) # Procesa y devuelve el resultado dentro del mismo datum
        if not success:
            print(f"[Warning] OpenPose falló en el frame {frame_idx}")
        output_frame = datum.cvOutputData # Frame con los esqueletos dibujados


        # 1) Obtenemos el array de keypoints
        keypoints = datum.poseKeypoints  # Devuelve: (n_personas, n_keypoints, 3), con 3 = (x, y, confidence)
                                         # (confidence = precisión del punto)
        
        if keypoints is not None:
            # Por ejemplo, imprimimos la matriz para la primera persona:
            print("Persona 0, matriz de keypoints (25×3):")
            print(keypoints[0])
            # shape completo:
            print("Shape:", keypoints.shape)

        

        # Comprueba que OpenPose devolvió algo
        if output_frame is None:
            print(f"[Warning] output_frame es None en el frame {frame_idx}")
        else:
            # Escribe el frame al vídeo
            writer.write(output_frame)
            written_frames += 1

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Frames leídos: {frame_idx}, escritos: {written_frames}")

    # 6. Libera recursos
    cap.release()
    writer.release()
    total_time = time.time() - start_time
    print(f"\n✅ Vídeo procesado en {total_time:.2f} segundos.")
    print(f"⏱️ Promedio por frame: {total_time/frame_idx:.3f} segundos")
    print(f"Procesamiento completado. Total leídos: {frame_idx}, escritos: {written_frames}")

if __name__ == "__main__":
    if len(sys.argv) not in (3,4):
        print("Uso: python3 process_video.py <input.mp4> <output.avi> [codec]")
        sys.exit(1)
    input_video  = sys.argv[1]
    output_video = sys.argv[2]
    codec        = sys.argv[3] if len(sys.argv) == 4 else 'XVID'
    process_video(input_video, output_video, codec)
