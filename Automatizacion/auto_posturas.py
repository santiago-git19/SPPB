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
        results = []
        # Fase 1
        results += self.fase_equilibrio(cap, camera_id, duration, cumple_prueba)
        # Fase 2: TEST DE LA VELOCIDAD DE LA MARCAH (4 METROS, 2 INTENTOS)
        results.append(self.fase_marcha(cap))
        # Fase 3: TEST DE LEVANTARSE DE LA SILLA
        results.append(self.fase_silla(cap))
        cap.release()
        return results

    def fase_equilibrio(self, cap, camera_id, duration, cumple_prueba):
        print("Iniciando Test de Equilibrio...")
        posturas = ["side-by-side", "semi-tandem", "tandem"]
        results = []
        for idx, posture in enumerate(posturas, start=1):
            #INICIO DE LA POSTURA
            input(f"Presiona ENTER para iniciar '{posture}' (tienes {duration}s)...")
            # CÓDIGO DE ESPERA DEL INICIO
            print(f"Iniciando '{posture}'...")

            # reiniciar captura de vídeo
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reiniciar video si es necesario
            start_test_time = time.time() # reiniciar tiempo de prueba
            last_sample = start_test_time
            tiempo_continuo = 0.0 # tiempo acumulado en postura
            #posture_kps = []
            aprobado = False # True si la postura se mantiene el tiempo requerido

            while not aprobado:
                now = time.time()
                elapsed = now - start_test_time

                # chequear si se ha alcanzado el tiempo máximo de la prueba
                if elapsed >= duration:
                    break

                # muestra cada 'interval' segundos
                if now - last_sample >= self.interval:
                    ret, frame = cap.read()
                    if not ret:
                        print("Vídeo terminó prematuramente.")
                        break
                    kps = self._process_frame(frame)
                    #posture_kps.append(kps)

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
        return results

    def fase_marcha(self, cap):
        print("Test de velocidad de la marcha (4m): se realizarán 2 intentos, se guarda el mejor tiempo.")
        walk_times = []
        for intento in range(2):
            input(f"Presiona ENTER para iniciar el intento {intento+1} de la marcha de 4m...")
            walk_start = None
            walk_end = None
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reiniciar video si es necesario
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                kps = self._process_frame(frame)
                # Detectar inicio: cuando se detecta movimiento hacia adelante (puedes mejorar con lógica de kps)
                if walk_start is None and kps is not None:
                    walk_start = time.time()
                # Detectar llegada: cuando la cadera (MidHip) supera cierto umbral en el eje X (simula línea de meta)
                if walk_start and walk_end is None and kps is not None:
                    # Suponemos que la cámara está lateral y el eje X crece hacia la meta
                    if kps[8][0] > frame.shape[1] * 0.85:  # 85% del ancho del frame
                        walk_end = time.time()
                        break
                # Si la persona desaparece, termina intento
                if walk_start and walk_end is None and kps is None:
                    walk_end = time.time()
                    break
            if walk_start and walk_end:
                walk_time = walk_end - walk_start
                walk_times.append(walk_time)
                print(f"Intento {intento+1}: {walk_time:.2f} segundos")
            else:
                print(f"Intento {intento+1}: No se pudo medir correctamente.")
        if walk_times:
            best_walk_time = min(walk_times)
            return {'test': 'walk', 'best_time': best_walk_time, 'all_times': walk_times}
        else:
            return {'test': 'walk', 'best_time': None, 'all_times': walk_times}

    def fase_silla(self, cap):
        print("Test de levantarse de la silla: Pre-test (cruzar brazos y levantarse una vez)")
        puede_levantar = input("¿El sujeto puede levantarse de la silla sin ayuda? (s/n): ").strip().lower() == 's'
        chair_result = {'test': 'chair', 'pretest': puede_levantar, 'times': [], 'total_time': None, 'score': 0}
        if not puede_levantar:
            print("No puede realizarlo. Puntuación: 0 puntos.")
            return chair_result
        print("Ahora, levántese y siéntese 5 veces lo más rápido posible (sin usar brazos). Se medirá el tiempo total.")
        CHAIR_RISES = 5
        chair_count = 0
        last_state = None  # sentado o de pie
        rep_start = None
        chair_times = []
        total_start = None
        interval = self.interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reiniciar video si es necesario
        while cap.isOpened() and chair_count < CHAIR_RISES:
            ret, frame = cap.read()
            if not ret: break
            current = time.time()
            if not hasattr(self, '_last') or (current - self._last) >= interval:
                kps = self._process_frame(frame)
                # Detectar estado: sentado si MidHip (8) está bajo, de pie si está alto
                is_standing = kps is not None and kps[8][1] < frame.shape[0] * 0.5  # de pie si cadera arriba
                if last_state is None:
                    last_state = is_standing
                    if is_standing:
                        total_start = current
                elif not last_state and is_standing:
                    # paso de sentado a de pie: inicio de repetición
                    rep_start = current
                    last_state = is_standing
                elif last_state and not is_standing:
                    # paso de de pie a sentado: fin de repetición
                    if rep_start:
                        chair_times.append(current - rep_start)
                        chair_count += 1
                        print(f"Repetición {chair_count}: {current - rep_start:.2f} segundos")
                        rep_start = None
                    last_state = is_standing
                self._last = current
        total_time = None
        if total_start and chair_count == CHAIR_RISES:
            total_time = time.time() - total_start
            print(f"Tiempo total para 5 repeticiones: {total_time:.2f} segundos")
        chair_result['times'] = chair_times
        chair_result['total_time'] = total_time
        # Asignar puntuación según protocolo SPPB
        if total_time is not None:
            if total_time <= 11.19:
                chair_result['score'] = 4
            elif total_time <= 13.69:
                chair_result['score'] = 3
            elif total_time <= 16.69:
                chair_result['score'] = 2
            elif total_time <= 60:
                chair_result['score'] = 1
            else:
                chair_result['score'] = 0
        return chair_result

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
