import time
import cv2

class ChairRisePhase:
    def __init__(self, openpose, config):
        self.openpose = openpose
        self.config = config
        self.interval = getattr(config, 'fps', 5.0)
        if self.interval:
            self.interval = 1.0 / self.interval
        else:
            self.interval = 0.2

    def run(self, cap, camera_id):
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

            # Procesa el frame si el intervalo ha pasado o es el primer frame
            if not hasattr(self, '_last') or (current - self._last) >= interval:
                kps = self.openpose.process_frame(frame)

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
        chair_result['score'] = self.score(total_time)
        return chair_result

    @staticmethod
    def score(total_time):
        """
        Calcula la puntuación SPPB para la fase de levantarse de la silla.
        """
        puntuacion = 0

        if total_time is None:
            puntuacion = 0
        if total_time <= 11.19:
            puntuacion = 4
        elif total_time <= 13.69:
            puntuacion = 3
        elif total_time <= 16.69:
            puntuacion = 2
        elif total_time <= 60:
            puntuacion = 1
        
        return puntuacion
