import time
import cv2
import numpy as np
from ..utils.phase_base import PhaseBase, FullRestartRequested

class ChairRisePhase(PhaseBase):
    def __init__(self, pose_processor, pose_classifier, config):
        """
        Inicializa la fase de Chair Rise con procesadores TRT Pose centralizados
        
        Args:
            pose_processor: Instancia de TRTPoseProcessor (reutilizada)
            pose_classifier: Instancia de TRTPoseClassifier (reutilizada)
            config: Configuración del sistema
        """
        super().__init__(None, config)  # No usar openpose
        
        # === PROCESADORES CENTRALIZADOS ===
        self.pose_processor = pose_processor
        self.pose_classifier = pose_classifier
        
        print("✅ ChairRisePhase inicializada con procesadores TRT Pose centralizados")

    def reset_test(self):
        """
        Reinicia el estado específico de la prueba Chair Rise.
        """
        super().reset_test()
        # Reiniciar cualquier estado específico de chair rise
        print("Estado de Chair Rise reiniciado.")

    def run(self, cap_frontal, cap_lateral):
        """
        Método principal que ejecuta la prueba con capacidad de reinicio y reinicio global.
        
        Args:
            cap_frontal: Captura de video de la cámara frontal
            cap_lateral: Captura de video de la cámara lateral (puede ser None)
        """
        """
        return self.run_test_with_global_restart(cap)

    def _run_phase(self, cap):
        """
        Implementación específica de la fase Chair Rise con manejo de emergencias.
        """
        # Parte 1: Pre-test
        puede_levantar = self.pre_test(cap)
        
        if puede_levantar is None:  # Saltada o cancelada
            return self.create_skipped_result('chair', 'pretest_skipped')
        
        chair_result = {
            'test': 'chair',
            'pretest': puede_levantar,
            'total_time': None,
            'score': 0
        }
        
        if not puede_levantar:
            print("No puede realizarlo. Puntuación: 0 puntos.")
            return chair_result
            
        # Parte 2: Test principal (5 repeticiones)
        main_result = self.test_principal(cap)
        
        if main_result is None:  # Saltada o cancelada
            return self.create_skipped_result('chair', 'main_test_skipped')
        
        return main_result
    
    def pre_test(self, cap):
        """
        Primera parte: Pre-test
        Pedir que cruce los brazos sobre el pecho e intente levantarse de la silla
        """
        self.print_instructions(
            "Pre-test de Chair Rise",
            [
                "Siéntese en la silla",
                "Cruce los brazos sobre el pecho",
                "Cuando esté listo, deberá levantarse una vez"
            ]
        )
        
        action = self.wait_for_ready_with_restart("Presione ENTER cuando esté en posición y listo para intentar levantarse...")
        
        if action == 'restart':
            raise Exception("Reinicio solicitado por el usuario")
        elif action == 'full_restart':
            raise FullRestartRequested("Reinicio completo solicitado por el usuario")
        elif action == 'skip':
            return None  # Indicar que se saltó la prueba
        elif action == 'exit' or action == 'emergency_stop':
            return None
        
        # LÓGICA DE DETECCIÓN DE LEVANTARSE Y SENTARSE EN LA SILLA
        puede_levantar = input("¿El sujeto puede levantarse de la silla sin ayuda? (s/n): ").strip().lower() == 's'
        return puede_levantar

    def test_principal(self, cap):
        """
        Segunda parte: Test principal
        5 repeticiones midiendo el tiempo total
        """
        self.print_instructions(
            "Test Principal de Chair Rise",
            [
                "Siéntese en la silla",
                "Mantenga los brazos cruzados sobre el pecho",
                "Cuando se le indique, deberá levantarse y sentarse 5 veces lo más rápido posible",
                "Se medirá el tiempo total para las 5 repeticiones"
            ]
        )
        
        action = self.wait_for_ready_with_restart("Presione ENTER cuando esté en posición inicial (sentado con brazos cruzados)...")
        
        if action == 'restart':
            raise Exception("Reinicio solicitado por el usuario")
        elif action == 'full_restart':
            raise FullRestartRequested("Reinicio completo solicitado por el usuario")
        elif action == 'skip':
            return None  # Indicar que se saltó la prueba
        elif action == 'exit' or action == 'emergency_stop':
            return None
        
        chair_result = {'test': 'chair', 'pretest': True, 'times': [], 'total_time': None, 'score': 0}

        CHAIR_RISES = 5
        chair_count = 0
        last_state = None
        chair_times = []
        total_start = None
        rep_start = None  # Inicializar rep_start
        emergency_shown = False  # Para mostrar mensaje de emergencia solo una vez
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        try:
            while cap.isOpened() and chair_count < CHAIR_RISES:
                ret, frame = cap.read()
                if not ret: break
                current = time.time()

                # Mostrar información de emergencia solo una vez
                if not emergency_shown:
                    self.monitor_emergency_stop("Durante la prueba, presione Ctrl+C si necesita parar por emergencia")
                    emergency_shown = True

                # Procesa el frame si el intervalo ha pasado o es el primer frame
                if not hasattr(self, '_last') or (current - self._last) >= self.interval:  # Usar self.interval
                    kps = self.openpose.process_frame(frame)

                    # LÓGICA DE DETECCIÓN DE LEVANTARSE Y SENTARSE EN LA SILLA
                    # HAY QUE TENER EN CUENTA QUE TIENE QUE TENER LOS BRAZOS CRUZADOS
                    # SE PODRÍA CALCULAR USANDO LA CADERA (MidHip)
                    # TENER EN CUENTA QUE LA PERSONA PUEDE DESAPARECER DEL FRAME

                # Validar que tenemos keypoints válidos
                if kps is not None and len(kps) >= 9:  # Asegurar que tenemos al menos hasta MidHip
                    # Detectar estado: sentado si MidHip (8) está bajo, de pie si está alto
                    is_standing = kps[8][1] < frame.shape[0] * 0.5  # de pie si cadera arriba
                    brazos_ok = self.verificar_brazos_cruzados(kps, frame)
                    
                    if last_state is None:  # primer frame
                        last_state = is_standing
                        if is_standing:
                            total_start = current
                            
                    elif not last_state and is_standing and brazos_ok:  # Transición a de pie
                        rep_start = current
                        last_state = is_standing
                        
                    elif last_state and not is_standing and brazos_ok:  # Transición a sentado
                        if rep_start is not None:  # Validar que tenemos un inicio válido
                            chair_times.append(current - rep_start)
                            chair_count += 1
                            print(f"Repetición {chair_count}: {current - rep_start:.2f} segundos")
                            rep_start = None
                        last_state = is_standing
                        
                    self._last = current

        except KeyboardInterrupt:
            print("\n🚨 PARADA DE EMERGENCIA durante Chair Rise")
            raise  # Re-lanzar para que sea manejada por run_with_restart

        # Finalizar el test
        total_time = None
        if total_start and chair_count == CHAIR_RISES:
            total_time = time.time() - total_start
            print(f"Tiempo total para 5 repeticiones: {total_time:.2f} segundos")
        chair_result['times'] = chair_times
        chair_result['total_time'] = total_time
        chair_result['score'] = self.score(total_time)
        return chair_result
            

    def verificar_brazos_cruzados(self, kps, frame):
        """Verifica si los brazos están cruzados sobre el pecho"""
        if kps is None:
            return False
        
        # Keypoints relevantes
        wrist_right = kps[4]  # muñeca derecha
        wrist_left = kps[7]   # muñeca izquierda
        chest = kps[1]        # pecho
        
        # Calcular distancias
        dist_right = np.linalg.norm(wrist_right[:2] - chest[:2])
        dist_left = np.linalg.norm(wrist_left[:2] - chest[:2])
        dist_wrists = np.linalg.norm(wrist_right[:2] - wrist_left[:2])
        
        # Umbrales (ajustar según resolución)
        threshold_chest = frame.shape[0] * 0.15
        threshold_wrists = frame.shape[0] * 0.20
        
        return (dist_right < threshold_chest and
                dist_left < threshold_chest and
                dist_wrists < threshold_wrists)

    @staticmethod
    def score(total_time):
        """
        Calcula la puntuación según el tiempo total:
        ≤11.19 seg: 4 puntos
        11.20-13.69 seg: 3 puntos
        13.70-16.69 seg: 2 puntos
        ≥16.70 seg: 1 punto
        >60 seg o no puede: 0 puntos
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
