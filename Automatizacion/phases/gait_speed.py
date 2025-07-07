class GaitSpeedPhase:
    def __init__(self, openpose, config):
        self.openpose = openpose
        self.config = config
        self.interval = getattr(config, 'fps', 5.0)
        if self.interval:
            self.interval = 1.0 / self.interval
        else:
            self.interval = 0.2

    def run(self, cap, camera_id):
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
                kps = self.openpose.process_frame(frame)
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
            result['score'] = self.score(best_walk_time)
        else:
            result['score'] = 0  # No puede realizarlo
        return result

    @staticmethod
    def score(best_walk_time):
        """
        Calcula la puntuación SPPB para la fase de marcha.
        """
        puntuacion = 0

        if best_walk_time is None:
            puntuacion = 0
        if best_walk_time < 4.82:
            puntuacion = 4
        elif best_walk_time < 6.21:
            puntuacion = 3
        elif best_walk_time < 8.71:
            puntuacion = 2
        elif best_walk_time >= 8.71:
            puntuacion = 1

        return puntuacion
