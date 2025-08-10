#!/usr/bin/env python3
"""
MediaPipe Pose Processor 2 - Detección de poses usando MediaPipe oficial
=============================================        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        
        # Variables MediaPipe
        self.pose = None
        
        # Cargar modelo MediaPipe
        self._load_mediapipe_model()
        
        logger.info("✅ MediaPipe Pose Processor 2 inicializado correctamente")
        logger.info(f"   🎯 Confianza detección: {min_detection_confidence}")
        logger.info(f"   🎯 Confianza tracking: {min_tracking_confidence}")
        logger.info(f"   🖼️ Modo imagen estática: {static_image_mode}")
        logger.info(f"   🧠 Complejidad modelo: {model_complexity}")=========

Clase para procesar frames de imágenes y detectar keypoints de poses humanas
usando la librería oficial de MediaPipe PoseLandmarker.

MediaPipe BlazePose detecta 33 keypoints del cuerpo humano en tiempo real
con alta precisión y eficiencia computacional.

Instalación de dependencias:
    pip install mediapipe opencv-python numpy

Referencia oficial:
    https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python

Autor: Sistema de IA
Fecha: 2025
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Union
import logging
import os
import time

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importar MediaPipe
try:
    import mediapipe as mp
    # Usar mediapipe.solutions para compatibilidad con versiones anteriores
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    MP_AVAILABLE = True
    logger.info("✅ MediaPipe importado correctamente")
except ImportError as e:
    MP_AVAILABLE = False
    logger.warning(f"⚠️ MediaPipe no disponible: {e}")
    logger.warning("💡 Para usar esta clase, instale MediaPipe: pip install mediapipe")

class MediaPipePoseProcessor2:
    """
    Procesador de poses usando MediaPipe oficial PoseLandmarker
    
    Utiliza el modelo PoseLandmarker de MediaPipe para detectar 33 keypoints 
    del cuerpo humano según la topología oficial:
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
    
    # Nombres de los keypoints de MediaPipe BlazePose (33 keypoints) - Topología oficial
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
    
    # Conexiones del esqueleto para visualización - Topología oficial MediaPipe BlazePose
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
        (12, 14), (14, 16),        # right sshoulder -> elbow -> wrist
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
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 static_image_mode: bool = False,
                 model_complexity: int = 1,
                 smooth_landmarks: bool = True):
        """
        Inicializa el procesador de poses MediaPipe
        
        Args:
            min_detection_confidence: Confianza mínima para detección (0.0-1.0)
            min_tracking_confidence: Confianza mínima para tracking (0.0-1.0)
            static_image_mode: Si tratar cada imagen independientemente
            model_complexity: Complejidad del modelo (0, 1, 2)
            smooth_landmarks: Si suavizar landmarks entre frames
        """
        if not MP_AVAILABLE:
            raise ImportError("MediaPipe es requerido. Instale con: pip install mediapipe")
        
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        
        # Variables MediaPipe
        self.pose = None
        
        # Cargar modelo MediaPipe
        self._load_mediapipe_model()
        
        logger.info("✅ MediaPipe Pose Processor 2 inicializado correctamente")
        logger.info(f"   🎯 Confianza detección: {min_detection_confidence}")
        logger.info(f"   🎯 Confianza tracking: {min_tracking_confidence}")
        logger.info(f"   �️ Modo imagen estática: {static_image_mode}")
        logger.info(f"   🧠 Complejidad modelo: {model_complexity}")
        
    def _load_mediapipe_model(self):
        """Carga el modelo MediaPipe Pose usando mediapipe.solutions"""
        try:
            # Inicializar MediaPipe Pose usando solutions
            self.pose = mp_pose.Pose(
                static_image_mode=self.static_image_mode,
                model_complexity=self.model_complexity,
                smooth_landmarks=self.smooth_landmarks,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
            
            logger.info("✅ Modelo MediaPipe Pose cargado exitosamente")
            
        except Exception as e:
            error_msg = f"❌ Error cargando modelo MediaPipe: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Procesa un frame y retorna los keypoints detectados usando MediaPipe
        
        Args:
            frame: Frame de imagen en formato BGR (numpy array)
            
        Returns:
            keypoints: Array de keypoints [33, 3] donde cada fila es (x, y, confidence)
                      o None si ocurre un error o no se detectan poses
        """
        if frame is None or frame.size == 0:
            logger.warning("⚠️ Frame vacío o None recibido")
            return None
        
        try:
            # Convertir BGR a RGB para MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Procesar con MediaPipe usando solutions.pose
            results = self.pose.process(rgb_frame)
            
            # Extraer keypoints si se detectaron poses
            if results.pose_landmarks:
                # Convertir a formato [33, 3]
                keypoints = np.zeros((33, 3), dtype=np.float32)
                
                height, width = frame.shape[:2]
                
                for i, landmark in enumerate(results.pose_landmarks.landmark):
                    # Convertir coordenadas normalizadas a píxeles
                    x = landmark.x * width
                    y = landmark.y * height
                    confidence = landmark.visibility  # Usar visibility como confidence
                    
                    keypoints[i] = [x, y, confidence]
                
                logger.debug(f"✅ Detectados {len(keypoints)} keypoints con MediaPipe")
                return keypoints
            else:
                logger.debug("⚠️ No se detectaron poses en el frame")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error procesando frame con MediaPipe: {e}")
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
            'left_arm': (0, 255, 0),      # Verde
            'right_arm': (0, 0, 255),     # Azul
            'torso': (255, 255, 0),       # Amarillo
            'left_leg': (255, 0, 255),    # Magenta
            'right_leg': (0, 255, 255),   # Cian
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
            world_landmarks: Array de landmarks 3D [33, 3] (x, y, z) o None
        """
        if frame is None or frame.size == 0:
            logger.warning("⚠️ Frame vacío o None recibido")
            return None
        
        try:
            # Convertir BGR a RGB para MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Procesar con MediaPipe usando solutions.pose
            results = self.pose.process(rgb_frame)
            
            # Extraer world landmarks si están disponibles
            if results.pose_world_landmarks:
                # Convertir a formato [33, 3]
                world_keypoints = np.zeros((33, 3), dtype=np.float32)
                
                for i, landmark in enumerate(results.pose_world_landmarks.landmark):
                    world_keypoints[i] = [landmark.x, landmark.y, landmark.z]
                
                return world_keypoints
            else:
                return None
                
        except Exception as e:
            logger.error(f"❌ Error obteniendo world landmarks: {e}")
            return None
    
    def cleanup(self):
        """Libera recursos de MediaPipe"""
        try:
            if self.pose is not None:
                self.pose.close()
                self.pose = None
                
            logger.info("✅ Recursos MediaPipe liberados correctamente")
            
        except Exception as e:
            logger.warning(f"⚠️ Error durante limpieza: {e}")
    
    def __del__(self):
        """Destructor que asegura la limpieza de recursos"""
        self.cleanup()
    
    def __str__(self) -> str:
        """Representación string del procesador"""
        return (f"MediaPipePoseProcessor2(MediaPipe, "
                f"detection_conf={self.min_detection_confidence}, "
                f"tracking_conf={self.min_tracking_confidence}, "
                f"complexity={self.model_complexity})")
    
    def __repr__(self) -> str:
        return self.__str__()


# Ejemplo de uso
if __name__ == "__main__":
    print("🎭 MediaPipe Pose Processor 2 - Ejemplo de uso")
    print("=" * 55)
    
    # Verificar disponibilidad de MediaPipe
    if not MP_AVAILABLE:
        print("❌ MediaPipe no está disponible")
        print("💡 Instale MediaPipe con: pip install mediapipe")
        exit(1)
    
    # Crear procesador con MediaPipe
    try:
        processor = MediaPipePoseProcessor2(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=False,
            model_complexity=1
        )
    except Exception as e:
        print(f"❌ Error inicializando procesador: {e}")
        exit(1)
    
    # Ejemplo 1: Procesar imagen de ejemplo
    print("\n🖼️ Ejemplo 1: Procesando imagen sintética...")
    
    # Crear una imagen de ejemplo
    example_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Crear un patrón simple para simular una persona
    # Cabeza
    cv2.circle(example_frame, (320, 120), 40, (255, 255, 255), -1)
    
    # Torso
    cv2.rectangle(example_frame, (280, 160), (360, 300), (200, 200, 200), -1)
    
    # Brazos
    cv2.rectangle(example_frame, (240, 180), (280, 220), (150, 150, 150), -1)  # Brazo izq
    cv2.rectangle(example_frame, (360, 180), (400, 220), (150, 150, 150), -1)  # Brazo der
    
    # Piernas
    cv2.rectangle(example_frame, (290, 300), (320, 420), (100, 100, 100), -1)  # Pierna izq
    cv2.rectangle(example_frame, (320, 300), (350, 420), (100, 100, 100), -1)  # Pierna der
    
    # Texto
    cv2.putText(example_frame, "MediaPipe BlazePose Test", 
               (180, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Procesar frame
    start_time = time.time()
    keypoints = processor.process_frame(example_frame)
    process_time = (time.time() - start_time) * 1000
    
    if keypoints is not None:
        print(f"✅ Detectados {len(keypoints)} keypoints en {process_time:.2f}ms")
        
        # Calcular ángulos
        angles = processor.get_pose_angles(keypoints)
        
        # Visualizar keypoints
        visualized = processor.visualize_keypoints(
            example_frame, keypoints,
            draw_landmarks=True,
            draw_connections=True,
            draw_labels=True
        )
        
        # Mostrar información
        print("📊 Información de keypoints:")
        valid_keypoints = np.sum(keypoints[:, 2] > 0.1)
        print(f"   • Keypoints válidos: {valid_keypoints}/33")
        print(f"   • Confianza promedio: {np.mean(keypoints[:, 2]):.3f}")
        
        if angles:
            print("📐 Ángulos calculados:")
            for angle_name, angle_value in angles.items():
                print(f"   • {angle_name}: {angle_value:.1f}°")
        else:
            print("⚠️ No se calcularon ángulos (confianza baja)")
        
        # Guardar imagen de ejemplo
        output_path = "mediapipe_pose_example.jpg"
        cv2.imwrite(output_path, visualized)
        print(f"💾 Imagen guardada: {output_path}")
        
    else:
        print("🚫 No se detectaron poses en la imagen de ejemplo")
    
    # Ejemplo 2: Procesar desde cámara web (opcional)
    print("\n📷 Ejemplo 2: Procesamiento desde cámara web")
    print("💡 Descomente el siguiente código para probar con cámara web:")
    print("""
    cap = cv2.VideoCapture(0)
    
    if cap.isOpened():
        print("📷 Iniciando captura desde cámara web...")
        print("Presiona 'q' para salir")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Procesar frame
            keypoints = processor.process_frame(frame)
            
            if keypoints is not None:
                # Visualizar
                visualized = processor.visualize_keypoints(frame, keypoints)
                
                # Calcular ángulos
                angles = processor.get_pose_angles(keypoints)
                
                # Mostrar información en pantalla
                info_text = [
                    "MediaPipe BlazePose",
                    f"Keypoints: {len(keypoints)}",
                    f"Válidos: {np.sum(keypoints[:, 2] > 0.1)}/33"
                ]
                
                for angle_name, angle_value in angles.items():
                    info_text.append(f"{angle_name}: {angle_value:.1f}°")
                
                # Dibujar información
                for i, text in enumerate(info_text):
                    color = (0, 255, 255) if i == 0 else (0, 255, 0)
                    cv2.putText(visualized, text, (10, 30 + i*25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                cv2.imshow("MediaPipe Pose - Tiempo Real", visualized)
            else:
                cv2.putText(frame, "No pose detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("MediaPipe Pose - Tiempo Real", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("❌ No se pudo abrir la cámara web")
    """)
    
    # Limpiar recursos
    processor.cleanup()
    print("\n✅ Ejemplo completado exitosamente")
    print("\n📋 Información de MediaPipe BlazePose:")
    print("   • Librería: MediaPipe oficial")
    print("   • Total: 33 keypoints")
    print("   • Soporte: Imágenes, videos, streams")
    print("   • Coordenadas 3D: Disponibles")
    print("\n💡 Para integrar con otras clases:")
    print("   from utils.mediapipe_pose_proc_2 import MediaPipePoseProcessor2")
    print("   processor = MediaPipePoseProcessor2()")
    print("   keypoints = processor.process_frame(frame)  # [33, 3] array")
    print("\n🔧 Dependencias necesarias:")
    print("   • MediaPipe: pip install mediapipe")
    print("   • OpenCV: pip install opencv-python")
    print("   • NumPy: pip install numpy")
