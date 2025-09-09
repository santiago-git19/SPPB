import cv2
import time
import numpy as np
from pathlib import Path
import sys

# Añadir rutas para importar utils
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ..utils.phase_base import PhaseBase, FullRestartRequested
from Automatizacion.utils.pose_detection.trt_pose_proc import TRTPoseProcessor
from Codigo.Automatizacion.utils.action_and_movement_detection.action_classifier import TRTPoseClassifier, create_pose_classifier

class GaitSpeedPhase(PhaseBase):
    def __init__(self, pose_processor, pose_classifier, config):
        """
        Inicializa la fase de Gait Speed con procesadores TRT Pose centralizados
        
        Args:
            pose_processor: Instancia de TRTPoseProcessor (reutilizada)
            pose_classifier: Instancia de TRTPoseClassifier (reutilizada)
            config: Configuración del sistema
        """
        # Llamar al constructor padre con None para openpose (ya no se usa)
        super().__init__(pose_processor, pose_classifier, config)
        
        # === CONFIGURACIÓN ESPECÍFICA DE GAIT SPEED ===
        # Variables para el cálculo de distancia
        self.pixel_to_meter_ratio = config.pixel_to_meter_ratio
        self.previous_position = None
        self.walking_started = False
        
        # Configuración de distancia del config
        self.distance_config = config.distance_calculation
        
        print("✅ GaitSpeedPhase inicializada con procesadores TRT Pose centralizados")
        
    def reset_test(self):
        """
        Reinicia el estado específico de la prueba Gait Speed.
        """
        super().reset_test()
        # Reiniciar cualquier estado específico de gait speed
        self.previous_position = None
        self.walking_started = False
        print("Estado de Gait Speed reiniciado.")

    def run(self, cap_frontal, cap_lateral, duration=None):
        """
        Método principal que ejecuta la prueba con soporte para dos cámaras
        
        Args:
            cap_frontal: Captura de video de la cámara frontal
            cap_lateral: Captura de video de la cámara lateral (puede ser None)
            duration: Duración no utilizada en gait_speed
        """
        # Pasar ambas cámaras al método de ejecución
        return self.run_test_with_global_restart(cap_frontal, cap_lateral)

    def _process_dual_camera_frame(self, frame_frontal, frame_lateral):
        """
        Procesa frames de ambas cámaras simultáneamente
        
        Args:
            frame_frontal: Frame de la cámara frontal
            frame_lateral: Frame de la cámara lateral
            
        Returns:
            dict: Resultados del procesamiento con keypoints y clasificación
        """
        result = {
            'frontal_keypoints': None,
            'lateral_keypoints': None,
            'pose_classification': None,
            'person_centered': False,
            'person_detected': False,
            'distance_moved': 0.0
        }
        
        try:
            # Procesar frame frontal para verificar alineación
            if frame_frontal is not None and self.pose_processor:
                frontal_keypoints = self.pose_processor.process_frame(frame_frontal)
                result['frontal_keypoints'] = frontal_keypoints
                
                if frontal_keypoints:
                    result['person_detected'] = True
                    result['person_centered'] = self._is_person_centered(frontal_keypoints, frame_frontal.shape)
            
            # Procesar frame lateral para cálculo de distancia
            if frame_lateral is not None and self.pose_processor:
                # Usar el procesador existente para obtener keypoints
                lateral_keypoints = self.pose_processor.process_frame(frame_lateral)
                result['lateral_keypoints'] = lateral_keypoints

                
                if lateral_keypoints:
                     # Clasificar pose usando keypoints laterales
                    if self.pose_classifier:
                        for person_idx, keypoints in enumerate(lateral_keypoints):
                            classification_result = self.pose_classifier.process_keypoints(keypoints)
                            result['pose_classification'] = classification_result

                    
                    # Calcular distancia recorrida
                    result['distance_moved'] = self._calculate_distance_moved(lateral_keypoints)
        
        except Exception as e:
            print(f"⚠️ Error procesando frames: {e}")
        
        return result
    
    def _is_person_centered(self, keypoints, frame_shape):
        """
        Verifica si la persona está centrada en el frame frontal
        
        Args:
            keypoints: Lista de keypoints detectados
            frame_shape: Dimensiones del frame (height, width, channels)
            
        Returns:
            bool: True si la persona está centrada
        """
        if not keypoints:
            return False
        
        frame_height, frame_width = frame_shape[:2]
        center_x = frame_width // 2
        center_tolerance = frame_width * 0.2  # 20% de tolerancia
        
        # Buscar keypoints de la cabeza/cuello para determinar posición central
        head_keypoints = []
        for kp in keypoints:
            x, y, confidence, part_id = kp
            # IDs típicos para cabeza/cuello en topología COCO: 0=nose, 1=neck, etc.
            if part_id in [0, 1, 2, 5] and confidence > 0.3:
                head_keypoints.append(x)
        
        if head_keypoints:
            avg_head_x = np.mean(head_keypoints)
            return abs(avg_head_x - center_x) < center_tolerance
        
        return False
    
    
    def _calculate_distance_moved(self, keypoints):
        """
        Calcula la distancia recorrida basada en el movimiento de la cadera
        Mejorado con suavizado y filtrado según configuración
        
        Args:
            keypoints: Lista de keypoints detectados
            
        Returns:
            float: Distancia recorrida en metros
        """
        if not keypoints:
            return 0.0
        
        # === EXTRACCIÓN DE KEYPOINTS DE CADERA ===
        hip_x = self._extract_hip_position(keypoints)
        
        if hip_x is None:
            return 0.0
        
        # === CÁLCULO DE DISTANCIA ===
        distance_moved = 0.0
        if self.previous_position is not None:
            pixel_distance = abs(hip_x - self.previous_position)
            distance_moved = pixel_distance * self.pixel_to_meter_ratio
            
            # Aplicar filtrado de movimiento mínimo
            min_threshold = self.distance_config.get('min_movement_threshold', 0.001)
            if distance_moved < min_threshold:
                distance_moved = 0.0
        
        # Actualizar posición anterior
        self.previous_position = hip_x
        
        return distance_moved
    
    def _extract_hip_position(self, keypoints):
        """
        Extrae la posición X de la cadera de los keypoints detectados
        
        Args:
            keypoints: Lista de keypoints
            
        Returns:
            float: Posición X de la cadera, o None si no se detecta
        """
        left_hip = None
        right_hip = None
        
        for kp in keypoints:
            x, _, confidence, part_id = kp
            if confidence > 0.3:
                if part_id == 11:  # Left Hip en topología COCO
                    left_hip = x
                elif part_id == 12:  # Right Hip en topología COCO
                    right_hip = x
        
        # Calcular posición de la cadera
        if left_hip is not None and right_hip is not None:
            return (left_hip + right_hip) / 2
        elif left_hip is not None:
            return left_hip
        elif right_hip is not None:
            return right_hip
        
        return None
    
    def _is_walking_pose(self, classification_result):
        """
        Determina si la pose clasificada corresponde a caminar
        
        Args:
            classification_result: Resultado de la clasificación de pose
            
        Returns:
            bool: True si la persona está caminando
        """
        if not classification_result or classification_result.get('error', False):
            return False
        
        predicted_class = classification_result.get('predicted_class', '')
        confidence = classification_result.get('confidence', 0.0)
        
        # Considerar como caminata si la confianza es alta y la clase es 'walking'
        return predicted_class == 'walking' and confidence > 0.5

    def _run_phase(self, cap_frontal, cap_lateral):
        """
        Implementación específica de la fase Gait Speed con procesamiento dual de cámaras.
        
        Args:
            cap_frontal: Captura de video de la cámara frontal
            cap_lateral: Captura de video de la cámara lateral (puede ser None)
        """
        self.print_instructions(
            "Test de Velocidad de la Marcha",
            [
                "Se realizarán 2 intentos de marcha de 4 metros",
                "Camine a su ritmo normal, como cuando va por la calle",
                "Se guardará el mejor tiempo de los dos intentos",
                "Sistema usando TRT Pose con detección avanzada"
            ]
        )
        
        # Verificar estado de las cámaras
        print(f"📹 Cámara frontal: {'✅ Configurada' if cap_frontal else '❌ No disponible'}")
        print(f"📹 Cámara lateral: {'✅ Configurada' if cap_lateral else '❌ No disponible'}")
        
        if cap_lateral is None:
            print("⚠️ Funcionando solo con cámara frontal - cálculo de distancia limitado")
        
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
            
            self.reset_pose_processors()

            if action == 'restart':
                raise Exception("Reinicio solicitado por el usuario")
            elif action == 'full_restart':
                raise FullRestartRequested("Reinicio completo solicitado por el usuario")
            elif action == 'skip':
                return self.create_skipped_result('walk', 'user_choice')
            elif action == 'exit' or action == 'emergency_            git push -u origin mainstop':
                return None
            
            # Ejecutar el intento
            walk_time = self._execute_walking_attempt(cap_frontal, cap_lateral, intento + 1)
            
            if walk_time is not None:
                walk_times.append(walk_time)
                print(f"✅ Intento {intento+1}: {walk_time:.2f} segundos")
            else:
                print(f"❌ Intento {intento+1}: No se pudo medir correctamente.")
        
        # Determinar el mejor tiempo y asignar puntuación SPPB
        result = {'test': 'walk', 'best_time': None, 'all_times': walk_times, 'score': 0}
        if walk_times:
            best_walk_time = min(walk_times)
            result['best_time'] = best_walk_time
            result['score'] = self.score(best_walk_time)
            print(f"🏆 Mejor tiempo: {best_walk_time:.2f} segundos (Puntuación SPPB: {result['score']})")
        else:
            result['score'] = 0
            print("❌ No se pudieron completar las mediciones")
        
        return result
    
    def _execute_walking_attempt(self, cap_frontal, cap_lateral, attempt_number):
        """
        Ejecuta un intento de caminata con monitoreo de poses y cálculo de distancia
        
        Args:
            cap_frontal: Captura de video de la cámara frontal
            cap_lateral: Captura de video de la cámara lateral (puede ser None)
            attempt_number: Número del intento actual
            
        Returns:
            float: Tiempo de caminata en segundos, o None si falló
        """
        print(f"🎬 Iniciando intento {attempt_number}")
        
        # Variables de control
        walk_start_time = None
        walk_end_time = None
        distance_total = 4.0  # metros
        distance_covered = 0.0  # metros
        emergency_shown = False
        frames_without_detection = 0
        max_frames_without_detection = 30  # ~2 segundos a 15fps

        # Reiniciar estado
        self.previous_position = None
        self.walking_started = False
        
        try:
            # Reiniciar posición de videos
            cap_frontal.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if cap_lateral:
                cap_lateral.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            print("⏱️ Esperando que comience a caminar...")
            
            while distance_covered < distance_total:
                # Leer frames
                ret_frontal, frame_frontal = cap_frontal.read()
                frame_lateral = None
                
                if cap_lateral:
                    ret_lateral, frame_lateral = cap_lateral.read()
                    if not ret_lateral:
                        frame_lateral = None
                
                if not ret_frontal:
                    print("❌ Error leyendo video frontal")
                    break
                
                # Mostrar mensaje de emergencia solo una vez
                if not emergency_shown:
                    self.monitor_emergency_stop("Durante la marcha, presione Ctrl+C si necesita parar por emergencia")
                    emergency_shown = True
                
                # Procesar frames con ambas cámaras
                processing_result = self._process_dual_camera_frame(frame_frontal, frame_lateral)
                
                # Verificar si hay detección de persona
                if processing_result['person_detected']:
                    frames_without_detection = 0
                    
                    # Verificar si está caminando
                    is_walking = self._is_walking_pose(processing_result['pose_classification'])
                    
                    if is_walking and not self.walking_started:
                        # Iniciar caminata
                        walk_start_time = time.time()
                        self.walking_started = True
                        print("� ¡Caminata iniciada!")
                        
                    elif self.walking_started:
                        # Acumular distancia recorrida
                        distance_moved = processing_result['distance_moved']
                        distance_covered += distance_moved
                        
                        # Mostrar progreso cada cierto tiempo
                        if int(distance_covered * 10) % 5 == 0:  # Cada 0.5 metros
                            progress = min((distance_covered / distance_total) * 100, 100)
                            print(f"📏 Progreso: {distance_covered:.1f}m / {distance_total}m ({progress:.0f}%)")
                        
                        # Verificar si completó la distancia
                        if distance_covered >= distance_total:
                            walk_end_time = time.time()
                            print("🏁 ¡Caminata completada!")
                            break
                
                else:
                    # Incrementar contador de frames sin detección
                    frames_without_detection += 1
                    
                    # Si hay demasiados frames sin detección, pausar
                    if frames_without_detection > max_frames_without_detection and self.walking_started:
                        print("⏸️ Persona perdida del frame, pausando medición...")
                        # Aquí podrías pausar el cronómetro si lo deseas
                
                # Pequeña pausa para no saturar el procesador
                time.sleep(0.01)
            
            # Calcular tiempo final
            if walk_start_time and walk_end_time:
                total_time = walk_end_time - walk_start_time
                print(f"📊 Distancia recorrida: {distance_covered:.2f}m en {total_time:.2f}s")
                return total_time
            else:
                print("❌ No se pudo completar la medición")
                return None
                
        except KeyboardInterrupt:
            print("\n🚨 PARADA DE EMERGENCIA durante intento de caminata")
            raise
        

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
