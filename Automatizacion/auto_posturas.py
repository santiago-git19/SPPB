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

class BalanceTest:
    """
    Ejecuta la Fase 1 (Test de Equilibrio) del SPPB.
    El usuario inicia cada postura pulsando ENTER, y tiene 'duration' segundos para mantenerla.
    Se muestrea al ritmo 'fps' y se utiliza 'cumple_prueba' para validar la postura.
    """
    def __init__(self, model_folder: str = models_path,
                 output_base: str = output_base_default,
                 fps: float = 5.0):
        # Inicializar OpenPose
        params = {"model_folder": model_folder}
        self.op_wrapper = op.WrapperPython()
        self.op_wrapper.configure(params)
        self.op_wrapper.start()

        # Configuración de muestreo
        self.interval = 1.0 / fps

        # Carpeta de salida
        self.output_base = output_base
        os.makedirs(self.output_base, exist_ok=True)

    def _process_frame(self, frame):
        datum = op.Datum()
        datum.cvInputData = frame
        vec = op.VectorDatum()
        vec.append(datum)
        self.op_wrapper.emplaceAndPop(vec)
        kps = datum.poseKeypoints
        return kps[0] if kps is not None and len(kps) > 0 else None

    def run(self, video_path: str, camera_id: int,
            duration: float,
            cumple_prueba: callable):
        """
        :param video_path: Ruta al vídeo de la prueba.
        :param camera_id: Identificador de la cámara.
        :param duration: Tiempo en segundos para cada postura.
        :param cumple_prueba: Función cumple_prueba(kps) -> bool.
        :return: Dict con keypoints y estado de cada postura.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"No se pudo abrir {video_path}")

        posturas = ["side-by-side", "semi-tandem", "tandem"]
        results = []
        
        for idx, posture in enumerate(posturas, start=1):
            input(f"Presiona ENTER para iniciar '{posture}' (tienes {duration}s)...")
            # CÓDIGO DE ESPERA DEL INICIO
            print(f"Iniciando '{posture}'...")

            start_test_time = time.time()
            last_sample = start_test_time
            tiempo_continuo = 0.0
            posture_kps = []
            aprobado = False

            while not aprobado:
                now = time.time()
                elapsed = now - start_test_time
                if elapsed >= duration:
                    break

                # muestreo a 'fps'
                if now - last_sample >= self.interval:
                    ret, frame = cap.read()
                    if not ret:
                        print("Vídeo terminó prematuramente.")
                        break
                    kps = self._process_frame(frame)
                    posture_kps.append(kps)

                    # evaluar continuidad de la prueba
                    if cumple_prueba(posture, kps): #Comprueba si está haciendo bien la postura
                        tiempo_continuo += now - last_sample
                        if tiempo_continuo >= duration:
                            aprobado = True
                            print(f"Postura '{posture}' aprobada antes de tiempo.")
                            break
                    else:
                        tiempo_continuo = 0.0

                    last_sample = now

                # chequear si abortar
                # por simplicidad, el usuario puede interrumpir con Ctrl+C

            # fin de la postura
            # guardar datos
            salida = {
                'posture': posture,
                'aprobado': aprobado or elapsed >= duration
            }
            # serializar .npy
            filename = f"cam{camera_id}_{posture}.npy"
            np.save(os.path.join(self.output_base, filename), salida)
            results.append(salida)
            print(f"'{posture}' completada. Resultado: {salida['aprobado']}")
            if not aprobado: break
        
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
        return results

# Ejemplo de uso
if __name__ == "__main__":
    def dummy_cumple(kps):
        # placeholder: siempre True si hay keypoints
        return kps is not None

    video = sys.argv[1] if len(sys.argv) > 1 else "/ruta/video.mp4"
    cam_id = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    dur = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
    fps = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0

    tester = BalanceTest(models_path, output_base_default, fps)
    results = tester.run(video, cam_id, dur, dummy_cumple)
    print("Resultados Fase 1:", results)
