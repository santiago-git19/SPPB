import cv2
from ..utils.phase_base import PhaseBase, FullRestartRequested

class GaitSpeedPhase(PhaseBase):
    def __init__(self, openpose, config):
        super().__init__(openpose, config)

    def reset_test(self):
        """
        Reinicia el estado específico de la prueba Gait Speed.
        """
        super().reset_test()
        # Reiniciar cualquier estado específico de gait speed
        print("Estado de Gait Speed reiniciado.")

    def run(self, cap, camera_id):
        """
        Método principal que ejecuta la prueba con capacidad de reinicio y reinicio global.
        """
        return self.run_test_with_global_restart(cap, camera_id)

    def _run_phase(self, cap, camera_id):
        """
        Implementación específica de la fase Gait Speed.
        """
        self.print_instructions(
            "Test de Velocidad de la Marcha",
            [
                "Se realizarán 2 intentos de marcha de 4 metros",
                "Camine a su ritmo normal, como cuando va por la calle",
                "Se guardará el mejor tiempo de los dos intentos"
            ]
        )
        
        walk_times = []
        for intento in range(2):
            self.print_instructions(
                f"Intento {intento+1} de 2",
                [
                    "Colóquese en la posición inicial de la marca de 4 metros",
                    "Espere la señal para comenzar"
                ]
            )
            
            action = self.wait_for_ready_with_restart(f"Presione ENTER cuando esté listo para comenzar el intento {intento+1}...")
            
            if action == 'restart':
                raise Exception("Reinicio solicitado por el usuario")
            elif action == 'full_restart':
                raise FullRestartRequested("Reinicio completo solicitado por el usuario")
            elif action == 'skip':
                return self.create_skipped_result('walk', 'user_choice')
            elif action == 'exit' or action == 'emergency_            git push -u origin mainstop':
                return None
            
            # Ejecutar el intento con monitoreo de interrupciones
            try:
                walk_start = None # Tiempo de inicio del intento
                walk_end = None # Tiempo de finalización del intento
                distanciaTotal = 4.0  # metros
                distanciaRecorrida = 0.0  # metros
                emergency_shown = False  # Para mostrar mensaje de emergencia solo una vez

                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reiniciar video si es necesario

                while cap.isOpened() and distanciaRecorrida < distanciaTotal:
                    ret, frame = cap.read()
                    if not ret: break
                    
                    # Mostrar información de emergencia solo una vez
                    if not emergency_shown:
                        self.monitor_emergency_stop("Durante la marcha, presione Ctrl+C si necesita parar por emergencia")
                        emergency_shown = True
                    
                    kps = self.openpose.process_frame(frame)
                    # LÓGICA DE DETECCIÓN DE CAMINATA
                    # NO INICIAR HASTA QUE SE DETECTE QUE SE HA INICIADO LA CAMINATA
                    # SE PODRÍA CALCULAR DISTANCIA RECORRIDA POR LA CADERA (MidHip) O POR LOS PIES (Ankle), USANDO CÁMARA LATERAL
                    # TENER EN QUE CUENTA LA PERSONA PUEDE DESAPARECER DEL FRAME
                    
            except KeyboardInterrupt:
                print("\n🚨 PARADA DE EMERGENCIA durante Gait Speed")
                raise  # Re-lanzar para que sea manejada por run_with_restart
            
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
