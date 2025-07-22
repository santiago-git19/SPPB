import time
import cv2
from ..utils.phase_base import PhaseBase, FullRestartRequested

class BalancePhase(PhaseBase):
    def __init__(self, pose_processor, pose_classifier, config):
        """
        Inicializa la fase de Balance con procesadores TRT Pose centralizados
        
        Args:
            pose_processor: Instancia de TRTPoseProcessor (reutilizada)
            pose_classifier: Instancia de TRTPoseClassifier (reutilizada)
            config: Configuración del sistema
        """
        super().__init__(pose_processor, pose_classifier, config)  # No usar openpose
        
        print("✅ BalancePhase inicializada con procesadores TRT Pose centralizados")

    def reset_test(self):
        """
        Reinicia el estado específico de la prueba Balance.
        """
        super().reset_test()
        # Reiniciar cualquier estado específico de balance
        print("Estado de Balance reiniciado.")

    def run(self, cap_frontal, cap_lateral, duration):
        """
        Método principal que ejecuta la prueba con capacidad de reinicio y reinicio global.
        
        Args:
            cap_frontal: Captura de video de la cámara frontal
            cap_lateral: Captura de video de la cámara lateral (puede ser None)
            duration: Duración de la prueba en segundos
        """
        return self.run_test_with_global_restart(cap, camera_id, duration)

    def _run_phase(self, cap, camera_id, duration):
        """
        Implementación específica de la fase Balance.
        """
        self.print_instructions(
            "Test de Equilibrio",
            [
                "Se evaluarán tres posturas diferentes:",
                "1. Side-by-side (pies juntos)",
                "2. Semi-tandem (un pie medio adelantado)",
                "3. Tandem (un pie delante del otro)",
                f"Cada postura debe mantenerse durante 10 segundos, tendrá un máximo de {duration} segundos para realizar cada prueba.",
            ]
        )
        
        posturas = ["side-by-side", "semi-tandem", "tandem"]
        results = []
        aprobado = [False, False, False]
        tandem_time = 0.0
        
        for idx, posture in enumerate(posturas, start=1):
            self.print_instructions(
                f"Postura {idx}: {posture}",
                [
                    "Colóquese en la posición indicada",
                    "Mantenga la postura durante el tiempo requerido"
                ]
            )
            
            action = self.wait_for_ready_with_restart(f"Presione ENTER cuando esté en posición para la postura '{posture}'...")
            
            self.reset_pose_processors()

            if action == 'restart':
                raise Exception("Reinicio solicitado por el usuario")
            elif action == 'full_restart':
                raise FullRestartRequested("Reinicio completo solicitado por el usuario")
            elif action == 'skip':
                return self.create_skipped_result('balance', 'user_choice')
            elif action == 'exit' or action == 'emergency_stop':
                return None
            
            # Ejecutar la postura con monitoreo de interrupciones
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reiniciar video si es necesario
                start_test_time = time.time() # reiniciar tiempo de prueba
                last_sample = start_test_time
                tiempo_continuo = 0.0 # tiempo acumulado en postura
                posture_aprobado = False # True si la postura se mantiene el tiempo requerido
                emergency_shown = False  # Para mostrar mensaje de emergencia solo una vez
                mejor_tiempo_continuo = 0.0  # Inicializar el mejor tiempo continuo

                while not posture_aprobado and (time.time() - start_test_time) < duration:
                    now = time.time()

                    # Mostrar información de emergencia solo una vez
                    if not emergency_shown:
                        self.monitor_emergency_stop(f"Durante la postura {posture}, presione Ctrl+C si necesita parar por emergencia")
                        emergency_shown = True

                    # muestra cada 'interval' segundos
                    if now - last_sample >= self.interval:
                        ret, frame = cap.read()
                        if not ret:
                            print("Vídeo terminó prematuramente.")
                            break
                        kps = self.openpose._process_frame(frame)

                        # evaluar continuidad de la prueba
                        if self.cumple_prueba(posture, kps): # Comprueba si está haciendo bien la postura
                            tiempo_continuo += now - last_sample
                            mejor_tiempo_continuo = max(mejor_tiempo_continuo, tiempo_continuo)  # Actualizar el mejor tiempo
                            if tiempo_continuo >= 10:
                                posture_aprobado = True
                                print(f"Postura '{posture}' aprobada antes de tiempo.")
                                break
                        else:
                            tiempo_continuo = 0.0

                        last_sample = now
                        
            except KeyboardInterrupt:
                print(f"\n🚨 PARADA DE EMERGENCIA durante la postura {posture}")
                raise  # Re-lanzar para que sea manejada por run_with_restart
            # Guardar resultado de la postura
            aprobado[idx-1] = posture_aprobado
            # Guardar el mejor tiempo mantenido en tandem
            if posture == "tandem":
                tandem_time = mejor_tiempo_continuo
            salida = {
                'posture': posture,
                'aprobado': aprobado[idx-1],
                'tiempo_mantenido': tiempo_continuo if posture == "tandem" else None
            }
            
            # Añadir resultado a la lista
            results.append(salida)
            print(f"'{posture}' completada. Resultado: {salida['aprobado']}")
            if not aprobado[idx-1]:
                break

        score = self.puntuacion_equilibrio(aprobado, tandem_time)
        results.append({'test': 'balance', 'score': score, 'tandem_time': tandem_time})
        return results

    @staticmethod
    def score(aprobado, tandem_time):
        """
        Calcula la puntuación SPPB para la fase de equilibrio según el flujo oficial.
        """
        puntuacion = 0

        if aprobado[0]: puntuacion = 1
        if aprobado[1]: puntuacion += 1
        if aprobado[2]:
            # Si supera la tandem, depende del tiempo mantenido
            if tandem_time >= 10:
                puntuacion += 2
            elif 3 <= tandem_time < 10:
                puntuacion += 1
            else:
                puntuacion += 0
        
        return puntuacion
    
    def cumple_prueba(self, posture, kps):
        """
        Comprueba si la postura se mantiene correctamente.
        """
        if posture == "side-by-side":
            # Lógica para comprobar side-by-side
            return True
