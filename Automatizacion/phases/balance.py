import time
import cv2

class BalancePhase:
    def __init__(self, openpose, config):
        self.openpose = openpose
        self.config = config
        self.interval = getattr(config, 'fps', 5.0)
        if self.interval:
            self.interval = 1.0 / self.interval
        else:
            self.interval = 0.2

    def run(self, cap, camera_id, duration):
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
                    kps = self.openpose._process_frame(frame) # REVISAR SI FUNCIONA

                    # evaluar continuidad de la prueba
                    if self.cumple_prueba(posture, kps): #Comprueba si está haciendo bien la postura
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
