#!/usr/bin/env python3
"""
MediaPipe Pose Processor - Detección de poses usando MediaPipe BlazePose
========================================================================

Clase para procesar frames de imágenes y detectar keypoints de poses humanas
usando el modelo BlazePose de MediaPipe.

MediaPipe BlazePose detecta 33 keypoints del cuerpo humano en tiempo real
con alta precisión y eficiencia computacional.

Instalación de dependencias:
    pip install mediapipe opencv-python numpy

Autor: Sistema de IA
Fecha: 2025
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, List
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MediaPipePoseProcessor:
    """
    Procesador de poses usando MediaPipe BlazePose
    
    MediaPipe BlazePose detecta 33 keypoints del cuerpo humano según la topología oficial:
    0: nose, 1: right_eye_inner, 2: right_eye, 3: right_eye_outer,
    4: left_eye_inner, 5: left_eye, 6: left_eye_outer,
    7: right_ear, 8: left_ear, 9: mouth_right, 10: mouth_left,
    11: right_shoulder, 12: left_shoulder, 13: right_elbow, 14: left_elbow,
    15: right_wrist, 16: left_wrist, 17: right_pinky_knuckle, 18: left_pinky_knuckle,
    19: right_index_knuckle, 20: left_index_knuckle, 21: right_thumb_knuckle, 22: left_thumb_knuckle,
    23: right_hip, 24: left_hip, 25: right_knee, 26: left_knee,
    27: right_ankle, 28: left_ankle, 29: right_heel, 30: left_heel,
    31: right_foot_index, 32: left_foot_index
    """
    
    # Nombres de los keypoints de MediaPipe BlazePose (33 keypoints) - Topología oficial
    KEYPOINT_NAMES = [
        'nose',                 # 0
        'right_eye_inner',      # 1
        'right_eye',            # 2
        'right_eye_outer',      # 3
        'left_eye_inner',       # 4
        'left_eye',             # 5
        'left_eye_outer',       # 6
        'right_ear',            # 7
        'left_ear',             # 8
        'mouth_right',          # 9
        'mouth_left',           # 10
        'right_shoulder',       # 11
        'left_shoulder',        # 12
        'right_elbow',          # 13
        'left_elbow',           # 14
        'right_wrist',          # 15
        'left_wrist',           # 16
        'right_pinky_knuckle',  # 17
        'left_pinky_knuckle',   # 18
        'right_index_knuckle',  # 19
        'left_index_knuckle',   # 20
        'right_thumb_knuckle',  # 21
        'left_thumb_knuckle',   # 22
        'right_hip',            # 23
        'left_hip',             # 24
        'right_knee',           # 25
        'left_knee',            # 26
        'right_ankle',          # 27
        'left_ankle',           # 28
        'right_heel',           # 29
        'left_heel',            # 30
        'right_foot_index',     # 31
        'left_foot_index'       # 32
    ]
    
    # Conexiones del esqueleto para visualización - Topología oficial MediaPipe BlazePose
    POSE_CONNECTIONS = [
        # Face connections
        (0, 1), (1, 2), (2, 3),    # right eye line
        (0, 4), (4, 5), (5, 6),    # left eye line
        (0, 9), (0, 10), (9, 10),  # mouth connections
        (2, 7), (5, 8),            # eyes to ears
        
        # Arms - Right arm
        (11, 13), (13, 15),        # right shoulder -> elbow -> wrist
        (15, 17), (15, 19), (15, 21),  # right wrist to hand points
        
        # Arms - Left arm
        (12, 14), (14, 16),        # left shoulder -> elbow -> wrist
        (16, 18), (16, 20), (16, 22),  # left wrist to hand points
        
        # Body core
        (11, 12),                  # shoulders
        (11, 23), (12, 24),        # shoulders to hips
        (23, 24),                  # hips
        
        # Legs - Right leg
        (23, 25), (25, 27),        # right hip -> knee -> ankle
        (27, 29), (27, 31),        # right ankle to foot points
        
        # Legs - Left leg
        (24, 26), (26, 28),        # left hip -> knee -> ankle
        (28, 30), (28, 32)         # left ankle to foot points
    ]
    
    def __init__(self, 
                 static_image_mode: bool = False,
                 model_complexity: int = 1,
                 smooth_landmarks: bool = True,
                 enable_segmentation: bool = False,
                 smooth_segmentation: bool = True,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Inicializa el procesador de poses MediaPipe
        
        Args:
            static_image_mode: Si True, trata cada imagen como independiente.
                              Si False, usa tracking para video.
            model_complexity: Complejidad del modelo (0, 1, 2). Mayor = más preciso pero más lento.
            smooth_landmarks: Si aplicar suavizado a los landmarks entre frames.
            enable_segmentation: Si habilitar la segmentación de la persona.
            smooth_segmentation: Si aplicar suavizado a la segmentación.
            min_detection_confidence: Confianza mínima para la detección.
            min_tracking_confidence: Confianza mínima para el tracking.
        """
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Inicializar MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Crear el modelo de pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=self.static_image_mode,
            model_complexity=self.model_complexity,
            smooth_landmarks=self.smooth_landmarks,
            enable_segmentation=self.enable_segmentation,
            smooth_segmentation=self.smooth_segmentation,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        
        logger.info("✅ MediaPipe BlazePose inicializado correctamente")
        logger.info(f"   📐 Modelo complejidad: {model_complexity}")
        logger.info(f"   🎯 Confianza detección: {min_detection_confidence}")
        logger.info(f"   🔄 Confianza tracking: {min_tracking_confidence}")
        logger.info(f"   🎬 Modo estático: {static_image_mode}")
        
    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Procesa un frame y retorna los keypoints detectados
        
        Args:
            frame: Frame de imagen en formato BGR (numpy array)
            
        Returns:
            keypoints: Array de keypoints [33, 3] donde cada fila es (x, y, confidence)
                      o None si no se detectó ninguna pose
        """
        if frame is None or frame.size == 0:
            logger.warning("⚠️ Frame vacío o None recibido")
            return None
        
        try:
            # Obtener dimensiones de la imagen
            height, width = frame.shape[:2]
            
            # Convertir BGR a RGB (MediaPipe usa RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Procesar la imagen con MediaPipe
            results = self.pose.process(rgb_frame)
            
            # Verificar si se detectaron landmarks
            if results.pose_landmarks is None:
                logger.debug("🚫 No se detectaron poses en el frame")
                return None
            
            # Extraer keypoints
            keypoints = self._extract_keypoints(results.pose_landmarks, width, height)
            
            logger.debug(f"✅ Detectados {len(keypoints)} keypoints")
            return keypoints
            
        except Exception as e:
            logger.error(f"❌ Error procesando frame: {e}")
            return None
    
    def _extract_keypoints(self, landmarks, width: int, height: int) -> np.ndarray:
        """
        Extrae keypoints de los landmarks de MediaPipe
        
        Args:
            landmarks: Landmarks de MediaPipe
            width: Ancho de la imagen
            height: Alto de la imagen
            
        Returns:
            keypoints: Array [33, 3] con keypoints (x, y, confidence)
        """
        keypoints = np.zeros((33, 3), dtype=np.float32)
        
        for i, landmark in enumerate(landmarks.landmark):
            # Convertir coordenadas normalizadas a píxeles
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            confidence = landmark.visibility  # MediaPipe usa 'visibility' como confianza
            
            keypoints[i] = [x, y, confidence]
        
        return keypoints
    
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
        
        # Colores para diferentes partes del cuerpo (actualizados para topología correcta)
        colors = {
            'face': (255, 255, 255),      # Blanco
            'right_arm': (0, 0, 255),     # Azul
            'left_arm': (0, 255, 0),      # Verde
            'torso': (255, 255, 0),       # Amarillo
            'right_leg': (0, 255, 255),   # Cian
            'left_leg': (255, 0, 255),    # Magenta
        }
        
        # Grupos de keypoints por parte del cuerpo (según topología oficial)
        body_parts = {
            'face': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],     # Face and ears
            'right_arm': [11, 13, 15, 17, 19, 21],           # Right arm and hand
            'left_arm': [12, 14, 16, 18, 20, 22],            # Left arm and hand
            'torso': [11, 12, 23, 24],                       # Shoulders and hips
            'right_leg': [23, 25, 27, 29, 31],               # Right leg and foot
            'left_leg': [24, 26, 28, 30, 32]                 # Left leg and foot
        }
        
        # Dibujar landmarks
        if draw_landmarks:
            for i, (x, y, confidence) in enumerate(keypoints):
                if confidence > confidence_threshold:
                    # Determinar color según la parte del cuerpo
                    color = (128, 128, 128)  # Gris por defecto
                    for part, indices in body_parts.items():
                        if i in indices:
                            color = colors[part]
                            break
                    
                    # Dibujar círculo
                    cv2.circle(output_frame, (int(x), int(y)), 4, color, -1)
                    cv2.circle(output_frame, (int(x), int(y)), 6, (255, 255, 255), 1)
                    
                    # Dibujar etiqueta si se solicita
                    if draw_labels and i < len(self.KEYPOINT_NAMES):
                        label = f"{self.KEYPOINT_NAMES[i]}:{confidence:.2f}"
                        cv2.putText(output_frame, label,
                                   (int(x) + 5, int(y) - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Dibujar conexiones
        if draw_connections:
            for connection in self.POSE_CONNECTIONS:
                pt1_idx, pt2_idx = connection
                
                if (pt1_idx < len(keypoints) and pt2_idx < len(keypoints)):
                    x1, y1, conf1 = keypoints[pt1_idx]
                    x2, y2, conf2 = keypoints[pt2_idx]
                    
                    # Solo dibujar si ambos puntos tienen buena confianza
                    if conf1 > confidence_threshold and conf2 > confidence_threshold:
                        cv2.line(output_frame, 
                                (int(x1), int(y1)), 
                                (int(x2), int(y2)), 
                                (0, 255, 0), 2)
        
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
            # Ángulos de los brazos (corregidos según topología oficial)
            if all(keypoints[[12, 14, 16], 2] > 0.1):  # left arm (shoulder, elbow, wrist)
                angles['left_elbow'] = calculate_angle(
                    keypoints[12][:2], keypoints[14][:2], keypoints[16][:2]
                )
            
            if all(keypoints[[11, 13, 15], 2] > 0.1):  # right arm (shoulder, elbow, wrist)
                angles['right_elbow'] = calculate_angle(
                    keypoints[11][:2], keypoints[13][:2], keypoints[15][:2]
                )
            
            # Ángulos de las piernas (corregidos según topología oficial)
            if all(keypoints[[24, 26, 28], 2] > 0.1):  # left leg (hip, knee, ankle)
                angles['left_knee'] = calculate_angle(
                    keypoints[24][:2], keypoints[26][:2], keypoints[28][:2]
                )
            
            if all(keypoints[[23, 25, 27], 2] > 0.1):  # right leg (hip, knee, ankle)
                angles['right_knee'] = calculate_angle(
                    keypoints[23][:2], keypoints[25][:2], keypoints[27][:2]
                )
            
            # Ángulo del torso (inclinación) - corregido según topología oficial
            if all(keypoints[[11, 12, 23, 24], 2] > 0.1):
                shoulder_center = (keypoints[11][:2] + keypoints[12][:2]) / 2  # right + left shoulder
                hip_center = (keypoints[23][:2] + keypoints[24][:2]) / 2      # right + left hip
                
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
            world_landmarks: Array [33, 3] con coordenadas del mundo (x, y, z)
        """
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_world_landmarks is None:
                return None
            
            world_landmarks = np.zeros((33, 3), dtype=np.float32)
            
            for i, landmark in enumerate(results.pose_world_landmarks.landmark):
                world_landmarks[i] = [landmark.x, landmark.y, landmark.z]
            
            return world_landmarks
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo landmarks del mundo: {e}")
            return None
    
    def cleanup(self):
        """Libera recursos de MediaPipe"""
        try:
            if hasattr(self, 'pose') and self.pose is not None:
                self.pose.close()
                self.pose = None
            logger.info("✅ Recursos de MediaPipe liberados")
        except Exception as e:
            logger.warning(f"⚠️ Error durante limpieza: {e}")
    
    def __del__(self):
        """Destructor que asegura la limpieza de recursos"""
        self.cleanup()
    
    def __str__(self) -> str:
        """Representación string del procesador"""
        return (f"MediaPipePoseProcessor("
                f"complexity={self.model_complexity}, "
                f"detection_conf={self.min_detection_confidence}, "
                f"tracking_conf={self.min_tracking_confidence})")
    
    def __repr__(self) -> str:
        return self.__str__()


# Ejemplo de uso
if __name__ == "__main__":
    import time
    
    print("🎭 MediaPipe Pose Processor - Ejemplo de uso")
    print("=" * 50)
    
    # Crear procesador
    processor = MediaPipePoseProcessor(
        static_image_mode=False,  # Para video en tiempo real
        model_complexity=1,       # Complejidad media
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Opción 1: Procesar desde cámara web
    print("\n📷 Iniciando captura desde cámara web...")
    print("Presiona 'q' para salir")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara web")
        print("💡 Intenta con un archivo de video o imagen")
        
        # Opción 2: Procesar imagen de ejemplo
        print("\n🖼️ Creando imagen de ejemplo...")
        
        # Crear una imagen de ejemplo (negro con texto)
        example_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(example_frame, "Coloca una persona aqui", 
                   (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Procesar imagen de ejemplo
        keypoints = processor.process_frame(example_frame)
        
        if keypoints is not None:
            print(f"✅ Detectados {len(keypoints)} keypoints")
            
            # Visualizar
            visualized = processor.visualize_keypoints(
                example_frame, keypoints, 
                draw_landmarks=True, 
                draw_connections=True,
                draw_labels=True
            )
            
            cv2.imshow("MediaPipe Pose - Ejemplo", visualized)
            cv2.waitKey(5000)  # Mostrar por 5 segundos
        else:
            print("🚫 No se detectaron poses en la imagen de ejemplo")
        
        cv2.destroyAllWindows()
    
    else:
        # Procesar desde cámara web
        fps_counter = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Error leyendo frame de la cámara")
                break
            
            # Procesar frame
            frame_start = time.time()
            keypoints = processor.process_frame(frame)
            process_time = time.time() - frame_start
            
            # Visualizar resultados
            if keypoints is not None:
                # Calcular ángulos
                angles = processor.get_pose_angles(keypoints)
                
                # Visualizar keypoints
                visualized = processor.visualize_keypoints(
                    frame, keypoints,
                    draw_landmarks=True,
                    draw_connections=True,
                    draw_labels=False
                )
                
                # Mostrar información en pantalla
                info_text = [
                    f"Keypoints: {len(keypoints)}",
                    f"Process time: {process_time*1000:.1f}ms",
                    f"FPS: {1/process_time:.1f}"
                ]
                
                # Añadir ángulos a la información
                for angle_name, angle_value in angles.items():
                    info_text.append(f"{angle_name}: {angle_value:.1f}°")
                
                # Dibujar información
                for i, text in enumerate(info_text):
                    cv2.putText(visualized, text, (10, 30 + i*25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.imshow("MediaPipe BlazePose - Tiempo Real", visualized)
            else:
                # No se detectaron poses
                cv2.putText(frame, "No pose detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("MediaPipe BlazePose - Tiempo Real", frame)
            
            # Calcular FPS promedio
            fps_counter += 1
            if fps_counter % 30 == 0:
                elapsed = time.time() - start_time
                avg_fps = fps_counter / elapsed
                print(f"📊 FPS promedio: {avg_fps:.1f}")
            
            # Salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    # Limpiar recursos
    processor.cleanup()
    print("\n✅ Ejemplo completado exitosamente")
    print("\n📋 Información de keypoints de MediaPipe BlazePose:")
    print("   • Total: 33 keypoints")
    print("   • Cara: 11 keypoints (0-10)")
    print("   • Brazos: 12 keypoints (11-22)")
    print("   • Torso: 4 keypoints (11, 12, 23, 24)")
    print("   • Piernas: 10 keypoints (23-32)")
    print("\n💡 Para integrar con otras clases:")
    print("   from utils.mediapipe_pose_proc import MediaPipePoseProcessor")
    print("   processor = MediaPipePoseProcessor()")
    print("   keypoints = processor.process_frame(frame)  # [33, 3] array")
