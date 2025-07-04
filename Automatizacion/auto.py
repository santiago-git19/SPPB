import sys
sys.path.append('/home/work/openpose/build/python')
import os
import time
import cv2
import numpy as np
from openpose import pyopenpose as op

# Rutas por defecto
models_path = "/home/work/openpose/models/"
output_base_default = "/home/work/gonartrosis/data/processed/SPPB_keypoints"

# Duraciones estándar SPPB (en segundos)
BALANCE_TIME   = 10.0  # tiempo para cada posición de equilibrio
WALK_DISTANCE  = 4.0   # metros
CHAIR_RISES   = 5      # número de levantadas de silla

class SPPBProcessor:
    """
    Automatiza el test SPPB usando OpenPose.
    Procesa un vídeo continuo con las fases en orden:
      1) Equilibrio (3 posturas, 10s c/u)
      2) Marcha de 4m (cronometra tiempo de desplazamiento)
      3) Levantadas de silla (5 repeticiones, mide tiempo entre arrancos)
    Guarda keypoints de cada fase y devuelve resultados.
    """
    def __init__(self, model_folder: str = models_path,
                 output_base: str = output_base_default,
                 target_fps: float = 5.0):
        # Inicializar wrapper OpenPose
        params = {"model_folder": model_folder}
        self.op_wrapper = op.WrapperPython()
        self.op_wrapper.configure(params)
        self.op_wrapper.start()

        # Carpeta de salida
        self.output_base = output_base
        os.makedirs(self.output_base, exist_ok=True)

        # Muestreo de frames
        self.target_interval = 1.0 / target_fps

    def _process_frame(self, frame):
        """Procesa un frame y devuelve los keypoints de la primera persona (o None)."""
        datum = op.Datum()
        datum.cvInputData = frame
        vec = op.VectorDatum()
        vec.append(datum)
        self.op_wrapper.emplaceAndPop(vec)
        kps = datum.poseKeypoints
        return kps[0] if kps is not None and len(kps)>0 else None

    def run_test(self, video_path: str, camera_id: int):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"No se pudo abrir {video_path}")

        input_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / input_fps
        interval = self.target_interval

        print(f"Iniciando SPPB: duración vídeo {duration:.1f}s, muestreo cada {interval:.2f}s")

        # Resultados
        results = { 'balance': [], 'walk_time': None, 'chair_times': [] }

        # FASE 1: TEST DE EQUILIBRIO
        for posture in ['side-by-side','semi-tandem','tandem']:
            start = time.time()
            posture_kps = []
            print(f"Equilibrio: {posture} por {BALANCE_TIME}s...")
            while (time.time() - start) < BALANCE_TIME:
                ret, frame = cap.read()
                if not ret: break
                current = time.time()
                if not hasattr(self, '_last') or (current - self._last) >= interval:
                    kps = self._process_frame(frame)
                    posture_kps.append(kps)
                    self._last = current
            results['balance'].append({ 'posture': posture, 'keypoints': posture_kps })

        # FASE 2: MARCHA DE 4 METROS
        print("Marcha de 4m: cronometra desplazamiento...")
        # Esperamos señal de inicio (por ejemplo, línea de salida visible)
        # Aquí simplificado: medimos el tiempo desde el siguiente frame
        walk_start = None
        walk_end = None
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            kps = self._process_frame(frame)
            # Implementar detección de línea de salida/llegada con kps
            # Placeholder: primera detección de persona inicia cronómetro
            if walk_start is None and kps is not None:
                walk_start = time.time()
            # Placeholder: condición de fin (p.ej. persona desaparece tras 4m)
            if walk_start and walk_end is None and kps is None:
                walk_end = time.time()
                break
        if walk_start and walk_end:
            results['walk_time'] = walk_end - walk_start

        # FASE 3: LEVANTADAS DE SILLA
        print(f"Levántate y siéntate x{CHAIR_RISES}: tiempo de cada repetición...")
        chair_count = 0
        last_state = None  # sentado o de pie
        rep_start = None
        while cap.isOpened() and chair_count < CHAIR_RISES:
            ret, frame = cap.read()
            if not ret: break
            current = time.time()
            if not hasattr(self, '_last') or (current - self._last) >= interval:
                kps = self._process_frame(frame)
                # Implementar detección de estado sentado/de pie con kps
                # Placeholder lógica:
                is_standing = kps is not None and kps[8][1] >  kps[0][1]  # MidHip y Nose
                if last_state is None:
                    last_state = is_standing
                elif last_state and not is_standing:
                    # paso de de pie a sentado: final de repetición
                    if rep_start:
                        results['chair_times'].append(time.time() - rep_start)
                        chair_count += 1
                    last_state = is_standing
                elif not last_state and is_standing:
                    # paso de sentado a pie: inicio de repetición
                    rep_start = time.time()
                    last_state = is_standing
                self._last = current
        cap.release()

        # Guardar resultados globales
        np.save(os.path.join(self.output_base, f"results_camara{camera_id}.npy"), results)
        print("SPPB completado:", results)
        return results

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Uso: python3 sppb.py <video> <camera_id> <fps>")
        sys.exit(1)

    video_path = sys.argv[1]
    camera_id = int(sys.argv[2])
    fps = float(sys.argv[3])

    processor = SPPBProcessor(model_folder=models_path,
                              output_base=output_base_default,
                              target_fps=fps)
    processor.run_test(video_path, camera_id)
