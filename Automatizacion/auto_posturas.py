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
        # Fase 1: TEST DE EQUILIBRIO (3 POSTURAS, 30 SEGUNDOS CADA UNA)
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
        aprobado = [False, False, False]
        tandem_time = 0.0
        for idx, posture in enumerate(posturas, start=1):
            # Inicio de la postura
            input(f"Presiona ENTER para iniciar '{posture}' (tienes {duration}s)...")
            # CÓDIGO DE ESPERA DEL INICIO
            print(f"Iniciando '{posture}'...")

            # reiniciar captura de vídeo
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reiniciar video si es necesario
            start_test_time = time.time() # reiniciar tiempo de prueba
            last_sample = start_test_time
            tiempo_continuo = 0.0 # tiempo acumulado en postura
            #posture_kps = []
            posture_aprobado = False # True si la postura se mantiene el tiempo requerido

            while not posture_aprobado and elapsed < duration:
                now = time.time()
                elapsed = now - start_test_time

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
                            posture_aprobado = True
                            print(f"Postura '{posture}' aprobada antes de tiempo.")
                            break
                    else:
                        tiempo_continuo = 0.0

                    last_sample = now
            # Guardar resultado de la postura
            aprobado[idx-1] = posture_aprobado
            # Guardar el tiempo mantenido en tandem
            if posture == "tandem":
                tandem_time = tiempo_continuo
            salida = {
                'posture': posture,
                'aprobado': aprobado[idx-1],
                'tiempo_mantenido': tiempo_continuo if posture == "tandem" else None
            }
            # serializar .npy --> POR MIRAR SI VOY A GUARDAR LOS KEYPOINTS
            filename = f"cam{camera_id}_{posture}.npy"
            np.save(os.path.join(self.output_base, filename), salida)
            results.append(salida)
            print(f"'{posture}' completada. Resultado: {salida['aprobado']}")
            if not aprobado[idx-1]:
                break  # Si falla una postura, termina la fase

        # Asignar puntuación SPPB según el flujo del test
        # Si no supera la primera: 0 puntos
        # Si supera la primera pero no la segunda: 1 punto
        # Si supera la segunda pero no la tercera: 2 puntos
        # Si supera la tercera:
        #   - 10s: 2 puntos
        #   - 3-9.99s: 1 punto
        #   - <3s: 0 puntos
        score = 0
        if aprobado[0]:
            score = 1
            if aprobado[1]:
                score += 1
                if aprobado[2]:
                    if tandem_time >= 10:
                        score += 2
                    elif 3 <= tandem_time < 10:
                        score += 1
                    else:
                        score += 0
        results.append({'test': 'balance', 'score': score, 'tandem_time': tandem_time})
        return results

    def fase_marcha(self, cap): # CREO QUE NO HARÁ FALTA UN MODELO DE DETECCIÓN DE CAMINATA, SÓLO TIENE QUE DETECTAR EL INICIO Y EL FIN
        print("Test de velocidad de la marcha (4m): se realizarán 2 intentos, se guarda el mejor tiempo.")
        walk_times = []
        for intento in range(2):
            input(f"Presiona ENTER para iniciar el intento {intento+1} de la marcha de 4m...")
            # CÓDIGO DE ESPERA DEL INICIO

            walk_start = None # Tiempo de inicio del intento
            walk_end = None # Tiempo de finalización del intento
            distanciaTotal = 4.0  # metros
            distanciaRecorrida = 0.0  # metros

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reiniciar video si es necesario

            while cap.isOpened() and distanciaRecorrida < distanciaTotal:
                ret, frame = cap.read()
                if not ret: break
                kps = self._process_frame(frame)
                # LÓGICA DE DETECCIÓN DE CAMINATA
                # NO INICIAR HASTA QUE SE DETECTE QUE SE HA INICIADO LA CAMINATA
                # SE PODRÍA CALCULAR DISTANCIA RECORRIDA POR LA CADERA (MidHip) O POR LOS PIES (Ankle), USANDO CÁMARA LATERAL
                # TENER EN QUE CUENTA LA PERSONA PUEDE DESAPARECER DEL FRAME
            
            
            if walk_start and walk_end: # Si se detectó el inicio y fin de la caminata
                walk_time = walk_end - walk_start
                walk_times.append(walk_time)
                print(f"Intento {intento+1}: {walk_time:.2f} segundos")
            else:
                print(f"Intento {intento+1}: No se pudo medir correctamente.")
        # determinar el mejor tiempo y asignar puntuación SPPB
        result = {'test': 'walk', 'best_time': None, 'all_times': walk_times, 'score': 0}
        if walk_times:
            best_walk_time = min(walk_times)
            result['best_time'] = best_walk_time
            # Asignar puntuación según protocolo SPPB
            if best_walk_time < 4.82:
                result['score'] = 4
            elif best_walk_time < 6.21:
                result['score'] = 3
            elif best_walk_time < 8.71:
                result['score'] = 2
            elif best_walk_time >= 8.71:
                result['score'] = 1
        else:
            result['score'] = 0  # No puede realizarlo
        return result

    def fase_silla(self, cap):
        print("Test de levantarse de la silla: Pre-test (cruzar brazos y levantarse una vez)")

        puede_levantar = input("¿El sujeto puede levantarse de la silla sin ayuda? (s/n): ").strip().lower() == 's'
        chair_result = {'test': 'chair', 'pretest': puede_levantar, 'times': [], 'total_time': None, 'score': 0}
        if not puede_levantar:
            print("No puede realizarlo. Puntuación: 0 puntos.")
            return chair_result
        

        print("Ahora, levántese y siéntese 5 veces lo más rápido posible, con los brazos cruzados (sin usar brazos). Se medirá el tiempo total.")

        CHAIR_RISES = 5 # Número de repeticiones a realizar
        chair_count = 0 # Contador de repeticiones
        last_state = None  # sentado o de pie
        rep_start = None # Tiempo de inicio de la repetición actual
        chair_times = [] # Lista para almacenar los tiempos de cada repetición
        total_start = None # Tiempo de inicio del total de repeticiones
        interval = self.interval # Intervalo de muestreo para procesar frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reiniciar video si es necesario

        while cap.isOpened() and chair_count < CHAIR_RISES:
            ret, frame = cap.read()
            if not ret: break
            current = time.time()

            # Procesa el fram si el intervalo ha pasado o es el primer frame
            if not hasattr(self, '_last') or (current - self._last) >= interval:
                kps = self._process_frame(frame)

                # LÓGICA DE DETECCIÓN DE LEVANTARSE Y SENTARSE EN LA SILLA
                # HAY QUE TENER EN CUENTA QUE TIENE QUE TENER LOS BRAZOS CRUZADOS
                # SE PODRÍA CALCULAR USANDO LA CADERA (MidHip)
                # TENER EN CUENTA QUE LA PERSONA PUEDE DESAPARECER DEL FRAME
                
                # FALTARÍA DETECTAR SI TIENE LOS BRAZOS CRUZADOS
                # Detectar estado: sentado si MidHip (8) está bajo, de pie si está alto
                is_standing = kps is not None and kps[8][1] < frame.shape[0] * 0.5  # de pie si cadera arriba
                
                if last_state is None: # primer frame
                    # Inicializar el estado
                    last_state = is_standing
                    if is_standing: # si está de pie, iniciar el total
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
        
        # Finalizar el test
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
    
    # POSIBLE IMPLEMETACIÓN DE BRAZOS CRUZADOS
    def brazos_cruzados(kps, frame):
        # Keypoints: 4 = muñeca derecha, 7 = muñeca izquierda, 1 = pecho
        if kps is None:
            return False
        wrist_right = kps[4]
        wrist_left = kps[7]
        chest = kps[1]
        # Distancia entre muñecas y pecho
        dist_right = np.linalg.norm(wrist_right[:2] - chest[:2])
        dist_left = np.linalg.norm(wrist_left[:2] - chest[:2])
        # Distancia entre muñecas
        dist_wrists = np.linalg.norm(wrist_right[:2] - wrist_left[:2])
        # Umbrales heurísticos (ajustar según resolución)
        threshold_chest = frame.shape[0] * 0.15
        threshold_wrists = frame.shape[0] * 0.20
        return (dist_right < threshold_chest and
                dist_left < threshold_chest and
                dist_wrists < threshold_wrists)

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
