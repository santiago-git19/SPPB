#!/usr/bin/env python3
"""
MediaPipe Tasks Pose Processor - Detección de poses usando MediaPipe Tasks
===========================================================================

Clase para procesar frames de imágenes y detectar keypoints de poses humanas
usando el modelo pose_landmarker_lite.task con MediaPipe Tasks API.

MediaPipe BlazePose detecta 33 keypoints del cuerpo humano en tiempo real
con alta precisión usando el modelo .task optimizado.

Instalación de dependencias:
    pip install opencv-python numpy mediapipe

Autor: Sistema de IA
Fecha: 2025
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
import logging
import os
import time

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importar MediaPipe Tasks
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from mediapipe import solutions
    from mediapipe.framework.formats import landmark_pb2
    MP_AVAILABLE = True
    logger.info("✅ MediaPipe Tasks importado correctamente")
except ImportError as e:
    MP_AVAILABLE = False
    logger.warning(f"⚠️ MediaPipe no disponible: {e}")
    logger.warning("💡 Para usar esta clase, instale MediaPipe: pip install mediapipe")

class MediaPipeTasksPoseProcessor:
    """
    Procesador de poses usando MediaPipe Tasks con modelo .task
    
    Utiliza el modelo pose_landmarker_lite.task para detectar
    33 keypoints del cuerpo humano según la topología de MediaPipe BlazePose:
    0: nose, 1: left_eye_inner, 2: left_eye, 3: left_eye_outer,
    4: right_eye_inner, 5: right_eye, 6: right_eye_outer,
    7: left_ear, 8: right_ear, 9: mouth_left, 10: mouth_right,
    11: left_shoulder, 12: right_shoulder, 13: left_elbow, 14: right_elbow,
    15: left_wrist, 16: right_wrist, 17: left_pinky, 18: right_pinky,
    19: left_index, 20: right_index, 21: left_thumb, 22: right_thumb,
    23: left_hip, 24: right_hip, 25: left_knee, 26: right_knee,
    27: left_ankle, 28: right_ankle, 29: left_heel, 30: right_heel,
    31: left_foot_index, 32: right_foot_index
    """
    
    # Nombres de los keypoints de MediaPipe BlazePose (33 keypoints)
    KEYPOINT_NAMES = [
        'nose',                 # 0
        'left_eye_inner',       # 1
        'left_eye',             # 2
        'left_eye_outer',       # 3
        'right_eye_inner',      # 4
        'right_eye',            # 5
        'right_eye_outer',      # 6
        'left_ear',             # 7
        'right_ear',            # 8
        'mouth_left',           # 9
        'mouth_right',          # 10
        'left_shoulder',        # 11
        'right_shoulder',       # 12
        'left_elbow',           # 13
        'right_elbow',          # 14
        'left_wrist',           # 15
        'right_wrist',          # 16
        'left_pinky',           # 17
        'right_pinky',          # 18
        'left_index',           # 19
        'right_index',          # 20
        'left_thumb',           # 21
        'right_thumb',          # 22
        'left_hip',             # 23
        'right_hip',            # 24
        'left_knee',            # 25
        'right_knee',           # 26
        'left_ankle',           # 27
        'right_ankle',          # 28
        'left_heel',            # 29
        'right_heel',           # 30
        'left_foot_index',      # 31
        'right_foot_index'      # 32
    ]
    
    # Conexiones del esqueleto para visualización
    POSE_CONNECTIONS = [
        # Face connections
        (0, 1), (1, 2), (2, 3),    # left eye line
        (0, 4), (4, 5), (5, 6),    # right eye line
        (0, 9), (0, 10), (9, 10),  # mouth connections
        (2, 7), (5, 8),            # eyes to ears
        
        # Arms - Left arm
        (11, 13), (13, 15),        # left shoulder -> elbow -> wrist
        (15, 17), (15, 19), (15, 21),  # left wrist to hand points
        
        # Arms - Right arm
        (12, 14), (14, 16),        # right shoulder -> elbow -> wrist
        (16, 18), (16, 20), (16, 22),  # right wrist to hand points
        
        # Body core
        (11, 12),                  # shoulders
        (11, 23), (12, 24),        # shoulders to hips
        (23, 24),                  # hips
        
        # Legs - Left leg
        (23, 25), (25, 27),        # left hip -> knee -> ankle
        (27, 29), (27, 31),        # left ankle to foot points
        
        # Legs - Right leg
        (24, 26), (26, 28),        # right hip -> knee -> ankle
        (28, 30), (28, 32)         # right ankle to foot points
    ]
    
    def __init__(self, 
                 model_path: str = "../models/pose_landmarker_lite.task",
                 confidence_threshold: float = 0.5,
                 output_segmentation_masks: bool = True,
                 debug: bool = False):
        """
        Inicializa el procesador de poses MediaPipe Tasks
        
        Args:
            model_path: Ruta al modelo pose_landmarker_lite.task
            confidence_threshold: Umbral de confianza para los keypoints
            output_segmentation_masks: Si generar máscaras de segmentación
            debug: Activar modo depuración
        """
        if not MP_AVAILABLE:
            raise ImportError("MediaPipe Tasks es requerido. Instale con: pip install mediapipe")
        
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.output_segmentation_masks = output_segmentation_masks
        self.debug = debug
        
        # Variables MediaPipe Tasks
        self.detector = None
        
        # Cargar modelo MediaPipe Tasks
        self._load_mediapipe_model()
        
        logger.info("✅ MediaPipe Tasks Pose Processor inicializado correctamente")
        logger.info(f"   📁 Modelo: {os.path.basename(model_path)}")
        logger.info(f"   🎯 Confianza: {confidence_threshold}")
        
    def _load_mediapipe_model(self):
        """Carga el modelo MediaPipe Tasks .task"""
        try:
            # Verificar que el archivo existe
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Modelo no encontrado: {self.model_path}")
            
            # Crear detector MediaPipe Tasks
            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                output_segmentation_masks=self.output_segmentation_masks
            )
            self.detector = vision.PoseLandmarker.create_from_options(options)
            
            logger.info(f"✅ Modelo MediaPipe Tasks cargado: {os.path.basename(self.model_path)}")
            
        except Exception as e:
            logger.error(f"❌ Error cargando modelo MediaPipe Tasks: {e}")
            raise
    
    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Procesa un frame y retorna los keypoints detectados usando MediaPipe Tasks
        
        Args:
            frame: Frame de imagen en formato BGR (numpy array)
            
        Returns:
            keypoints: Array de keypoints [33, 3] donde cada fila es (x, y, confidence)
                      o None si ocurre un error
        """
        if frame is None or frame.size == 0:
            logger.warning("⚠️ Frame vacío o None recibido")
            return None
        
        try:
            # Convertir frame a formato MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Detectar poses
            detection_result = self.detector.detect(mp_image)
            
            if not detection_result.pose_landmarks:
                if self.debug:
                    logger.debug("🚫 No se detectaron poses en el frame")
                return None
            
            # Convertir landmarks a formato numpy [33, 3]
            pose_landmarks = detection_result.pose_landmarks[0]  # Tomar la primera pose detectada
            keypoints = np.zeros((33, 3), dtype=np.float32)
            
            height, width = frame.shape[:2]
            
            for i, landmark in enumerate(pose_landmarks):
                # Convertir coordenadas normalizadas a píxeles
                x = landmark.x * width
                y = landmark.y * height
                confidence = landmark.visibility  # MediaPipe usa visibility como confidence
                
                keypoints[i] = [x, y, confidence]
            
            if self.debug:
                logger.debug(f"✅ Detectados 33 keypoints con MediaPipe Tasks")
                
            return keypoints
            
        except Exception as e:
            logger.error(f"❌ Error procesando frame con MediaPipe Tasks: {e}")
            return None
    
    def visualize_keypoints(self, frame: np.ndarray, 
                          keypoints: Optional[np.ndarray] = None,
                          draw_landmarks: bool = True,
                          draw_connections: bool = True,
                          draw_labels: bool = False,
                          confidence_threshold: float = 0.1) -> np.ndarray:
        """
        Visualiza los keypoints en el frame
        
        Args:
            frame: Frame original
            keypoints: Array de keypoints [33, 3] (opcional, si None usa process_frame)
            draw_landmarks: Si dibujar los landmarks
            draw_connections: Si dibujar las conexiones del esqueleto
            draw_labels: Si dibujar etiquetas de los keypoints
            confidence_threshold: Umbral de confianza para mostrar keypoints
            
        Returns:
            frame: Frame con keypoints visualizados
        """
        if frame is None or frame.size == 0:
            logger.warning("⚠️ Frame vacío para visualización")
            return frame
        
        # Si no se proporcionan keypoints, procesarlos
        if keypoints is None:
            keypoints = self.process_frame(frame)
            if keypoints is None:
                return frame
        
        # Crear una copia del frame para no modificar el original
        output_frame = frame.copy()
        
        # Colores para diferentes partes del cuerpo
        colors = {
            'face': (255, 255, 255),      # Blanco
            'right_arm': (0, 0, 255),     # Azul
            'left_arm': (0, 255, 0),      # Verde
            'torso': (255, 255, 0),       # Amarillo
            'right_leg': (0, 255, 255),   # Cian
            'left_leg': (255, 0, 255),    # Magenta
        }
        
        # Grupos de keypoints por parte del cuerpo
        body_parts = {
            'face': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],     # Face and ears
            'left_arm': [11, 13, 15, 17, 19, 21],            # Left arm and hand
            'right_arm': [12, 14, 16, 18, 20, 22],           # Right arm and hand
            'torso': [11, 12, 23, 24],                       # Shoulders and hips
            'left_leg': [23, 25, 27, 29, 31],                # Left leg and foot
            'right_leg': [24, 26, 28, 30, 32]                # Right leg and foot
        }
        
        # Dibujar landmarks
        if draw_landmarks:
            for i, (x, y, confidence) in enumerate(keypoints):
                if confidence > confidence_threshold:
                    # Determinar color según la parte del cuerpo
                    color = (128, 128, 128)  # Gris por defecto
                    for part_name, indices in body_parts.items():
                        if i in indices:
                            color = colors[part_name]
                            break
                    
                    # Dibujar punto
                    cv2.circle(output_frame, (int(x), int(y)), 4, color, -1)
                    cv2.circle(output_frame, (int(x), int(y)), 6, (255, 255, 255), 1)
                    
                    # Dibujar etiqueta si se solicita
                    if draw_labels and i < len(self.KEYPOINT_NAMES):
                        label = f"{i}:{self.KEYPOINT_NAMES[i]}"
                        cv2.putText(output_frame, label, (int(x) + 10, int(y) - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Dibujar conexiones del esqueleto
        if draw_connections:
            for connection in self.POSE_CONNECTIONS:
                point1_idx, point2_idx = connection
                
                if (point1_idx < len(keypoints) and point2_idx < len(keypoints) and
                    keypoints[point1_idx][2] > confidence_threshold and
                    keypoints[point2_idx][2] > confidence_threshold):
                    
                    point1 = (int(keypoints[point1_idx][0]), int(keypoints[point1_idx][1]))
                    point2 = (int(keypoints[point2_idx][0]), int(keypoints[point2_idx][1]))
                    
                    # Color de la línea según la parte del cuerpo
                    line_color = (200, 200, 200)  # Gris claro por defecto
                    for part_name, indices in body_parts.items():
                        if point1_idx in indices and point2_idx in indices:
                            line_color = colors[part_name]
                            break
                    
                    cv2.line(output_frame, point1, point2, line_color, 2)
        
        return output_frame
    
    def get_pose_angles(self, keypoints: np.ndarray) -> dict:
        """
        Calcula ángulos importantes de la pose
        
        Args:
            keypoints: Array de keypoints [33, 3]
            
        Returns:
            dict: Diccionario con ángulos calculados
        """
        angles = {}
        
        def calculate_angle(p1, p2, p3):
            """Calcula el ángulo entre tres puntos"""
            v1 = p1 - p2
            v2 = p3 - p2
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            
            return np.degrees(angle)
        
        try:
            # Ángulos de los brazos
            if all(keypoints[[11, 13, 15], 2] > 0.1):  # left arm (shoulder, elbow, wrist)
                angles['left_elbow'] = calculate_angle(
                    keypoints[11][:2], keypoints[13][:2], keypoints[15][:2]
                )
            
            if all(keypoints[[12, 14, 16], 2] > 0.1):  # right arm (shoulder, elbow, wrist)
                angles['right_elbow'] = calculate_angle(
                    keypoints[12][:2], keypoints[14][:2], keypoints[16][:2]
                )
            
            # Ángulos de las piernas
            if all(keypoints[[23, 25, 27], 2] > 0.1):  # left leg (hip, knee, ankle)
                angles['left_knee'] = calculate_angle(
                    keypoints[23][:2], keypoints[25][:2], keypoints[27][:2]
                )
            
            if all(keypoints[[24, 26, 28], 2] > 0.1):  # right leg (hip, knee, ankle)
                angles['right_knee'] = calculate_angle(
                    keypoints[24][:2], keypoints[26][:2], keypoints[28][:2]
                )
            
            # Ángulo del torso (inclinación)
            if all(keypoints[[11, 12, 23, 24], 2] > 0.1):
                shoulder_center = (keypoints[11][:2] + keypoints[12][:2]) / 2  # left + right shoulder
                hip_center = (keypoints[23][:2] + keypoints[24][:2]) / 2      # left + right hip
                
                # Ángulo con respecto a la vertical
                torso_vector = shoulder_center - hip_center
                vertical_vector = np.array([0, -1])
                
                cos_angle = np.dot(torso_vector, vertical_vector) / np.linalg.norm(torso_vector)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angles['torso_lean'] = np.degrees(np.arccos(cos_angle))
        
        except Exception as e:
            logger.warning(f"⚠️ Error calculando ángulos: {e}")
        
        return angles
    
    def get_pose_landmarks_world(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Obtiene landmarks en coordenadas del mundo (3D)
        
        Args:
            frame: Frame de imagen
            
        Returns:
            world_landmarks: Array de landmarks en coordenadas del mundo o None
        """
        if frame is None or frame.size == 0:
            logger.warning("⚠️ Frame vacío o None recibido")
            return None
        
        try:
            # Convertir frame a formato MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Detectar poses
            detection_result = self.detector.detect(mp_image)
            
            if not detection_result.pose_world_landmarks:
                if self.debug:
                    logger.debug("🚫 No se detectaron poses en el frame (world landmarks)")
                return None
            
            # Convertir world landmarks a formato numpy [33, 3]
            pose_world_landmarks = detection_result.pose_world_landmarks[0]
            world_landmarks = np.zeros((33, 3), dtype=np.float32)
            
            for i, landmark in enumerate(pose_world_landmarks):
                world_landmarks[i] = [landmark.x, landmark.y, landmark.z]
            
            if self.debug:
                logger.debug(f"✅ Detectados 33 world landmarks con MediaPipe Tasks")
                
            return world_landmarks
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo world landmarks: {e}")
            return None
    
    def draw_landmarks_on_image(self, rgb_image, detection_result):
        """
        Función auxiliar para dibujar landmarks usando MediaPipe drawing utils
        """
        pose_landmarks_list = detection_result.pose_landmarks
        # Convert RGB to BGR for drawing_utils
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        annotated_image = np.copy(bgr_image)

        # Loop through the detected poses to visualize.
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

            # Draw the pose landmarks.
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                pose_landmarks_proto,
                solutions.pose.POSE_CONNECTIONS,
                solutions.drawing_styles.get_default_pose_landmarks_style())
        return annotated_image
    
    def cleanup(self):
        """Libera recursos de MediaPipe"""
        try:
            if hasattr(self, 'detector') and self.detector is not None:
                self.detector = None
                
            logger.info("✅ Recursos MediaPipe liberados correctamente")
            
        except Exception as e:
            logger.warning(f"⚠️ Error durante limpieza: {e}")
    
    def __del__(self):
        """Destructor que asegura la limpieza de recursos"""
        self.cleanup()
    
    def __str__(self) -> str:
        """Representación string del procesador"""
        return (f"MediaPipeTasksPoseProcessor(Tasks, "
                f"model={os.path.basename(self.model_path)}, "
                f"confidence={self.confidence_threshold})")
    
    def __repr__(self) -> str:
        return self.__str__()


# Ejemplo de uso
if __name__ == "__main__":
    print("🎭 MediaPipe Tasks Pose Processor - Ejemplo de uso")
    print("=" * 50)
    
    # Verificar disponibilidad de MediaPipe
    if not MP_AVAILABLE:
        print("❌ MediaPipe no está disponible")
        print("💡 Instale MediaPipe: pip install mediapipe")
        exit(1)
    
    # Crear procesador con modelo MediaPipe Tasks
    model_path = "../models/pose_landmarker_lite.task"
    
    if not os.path.exists(model_path):
        print(f"❌ Modelo no encontrado: {model_path}")
        exit(1)
    
    try:
        processor = MediaPipeTasksPoseProcessor(model_path=model_path, debug=True)
        
        # Probar con imagen
        test_image_path = "persona.PNG"
        if os.path.exists(test_image_path):
            frame = cv2.imread(test_image_path)
            keypoints = processor.process_frame(frame)
            
            if keypoints is not None:
                # Calcular ángulos
                angles = processor.get_pose_angles(keypoints)
                print(f"✅ Ángulos detectados: {angles}")
                
                # Visualizar keypoints
                visualized = processor.visualize_keypoints(frame, keypoints)
                cv2.imwrite('pose_output_task_class.png', visualized)
                print("✅ Imagen procesada guardada en: pose_output_task_class.png")
            else:
                print("🚫 No se detectaron poses en la imagen")
        
        # Limpiar recursos
        processor.cleanup()
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n✅ Ejemplo completado")