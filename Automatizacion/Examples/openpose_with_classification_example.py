#!/usr/bin/env python3
"""
Ejemplo de Uso: OpenPose + Clasificación de Poses
=================================================

Este script demuestra cómo integrar OpenPose con el clasificador de poses
PoseClassificationNet de NVIDIA TAO para detección y clasificación en tiempo real.

Flujo de trabajo:
1. OpenPose detecta keypoints de personas en video
2. TRTPoseClassifier procesa y clasifica las poses usando formato OpenPose
3. Se muestra el resultado en tiempo real

Diferencias con trt_pose:
- Usa OpenPose (18 keypoints) en lugar de trt_pose
- Formato de keypoints 'openpose' en el clasificador
- Mejor precisión en detección de keypoints

Autor: Sistema de IA
Fecha: 2025
"""

import cv2
import numpy as np
import torch
import json
import time
import logging
from pathlib import Path

import sys
from pathlib import Path
# Añadir el directorio 'Automatizacion' al sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Importar procesador de OpenPose y nuestro clasificador
from utils.openpose_proc import OpenPoseProcessor
from Automatizacion.utils.action_classifier import TRTPoseClassifier, create_pose_classifier


# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenPoseWithClassifier:
    """
    Integra OpenPose con clasificación de poses en tiempo real
    Usa OpenPoseProcessor y TRTPoseClassifier con formato 'openpose'
    """
    
    def __init__(self, 
                 openpose_model_folder: str,
                 pose_classifier_model_path: str,
                 sequence_length: int = 100,
                 confidence_threshold: float = 0.3):
        """
        Inicializa el sistema completo OpenPose + Clasificador
        
        Args:
            openpose_model_folder: Ruta a la carpeta de modelos de OpenPose
            pose_classifier_model_path: Ruta al modelo engine del clasificador
            sequence_length: Longitud de secuencia temporal (máximo 300 frames)
            confidence_threshold: Umbral mínimo de confianza para keypoints
        """
        self.openpose_model_folder = openpose_model_folder
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        
        # Crear procesador de OpenPose
        logger.info("🔧 Inicializando OpenPose...")
        self.openpose_processor = OpenPoseProcessor(model_folder=openpose_model_folder)
        
        # Crear clasificador de poses usando formato OpenPose
        logger.info("🎭 Inicializando clasificador de poses...")
        self.pose_classifier = create_pose_classifier(
            model_path=pose_classifier_model_path,
            keypoint_format='openpose',  # ✅ Usar formato OpenPose (18 keypoints)
            sequence_length=sequence_length,      
            confidence_threshold=confidence_threshold
        )
        
        # Estadísticas
        self.stats = {
            'frames_processed': 0,
            'poses_detected': 0,
            'poses_classified': 0,
            'start_time': time.time()
        }
        
        logger.info("✅ Sistema OpenPose + Clasificador inicializado")
        logger.info("   🔧 Usando OpenPoseProcessor")
        logger.info("   🎭 Clasificador con formato 'openpose' (18 keypoints)")
        logger.info(f"   ⏱️ Secuencia temporal: {sequence_length} frames")
        
    def process_frame_with_classification(self, image: np.ndarray) -> dict:
        """
        Procesa un frame completo: detección OpenPose + clasificación
        """
        frame_result = {
            'people_detected': 0,
            'pose_classifications': [],
            'processing_time_ms': 0,
            'keypoints_extracted': None
        }
        
        start_time = time.time()
        
        try:
            # Usar OpenPose para obtener keypoints
            keypoints = self.openpose_processor.process_frame(image)
            
            # Verificar que los keypoints sean válidos
            if keypoints is not None and isinstance(keypoints, np.ndarray):
                # OpenPose devuelve keypoints en formato (18, 3) para una persona
                if keypoints.shape[0] == 18 and keypoints.shape[1] == 3:
                    frame_result['people_detected'] = 1
                    frame_result['keypoints_extracted'] = keypoints
                    
                    # Clasificar poses usando el formato OpenPose
                    classification_result = self.pose_classifier.process_keypoints(keypoints)
                    
                    if classification_result and not classification_result.get('error', False):
                        frame_result['pose_classifications'].append({
                            'person_id': 0,
                            'pose_class': classification_result['predicted_class'],
                            'confidence': classification_result['confidence'],
                            'probabilities': classification_result['probabilities']
                        })
                        
                        self.stats['poses_classified'] += 1
                    
                    self.stats['poses_detected'] += 1
                else:
                    logger.warning(f"⚠️ Formato de keypoints inesperado: {keypoints.shape}")
            
        except Exception as e:
            logger.error(f"❌ Error procesando frame: {e}")
        
        finally:
            frame_result['processing_time_ms'] = (time.time() - start_time) * 1000
            self.stats['frames_processed'] += 1
        
        return frame_result
    
    def draw_results(self, image: np.ndarray, frame_result: dict) -> np.ndarray:
        """
        Dibuja resultados sobre la imagen
        
        Args:
            image: Imagen original
            frame_result: Resultados del procesamiento
            
        Returns:
            Imagen con resultados dibujados
        """
        result_image = image.copy()
        
        keypoints = frame_result['keypoints_extracted']
        
        if keypoints is not None and len(keypoints) > 0:
            # Encontrar clasificación (solo hay una persona)
            classification = None
            if frame_result['pose_classifications']:
                classification = frame_result['pose_classifications'][0]
            
            # Dibujar keypoints de OpenPose
            self._draw_openpose_skeleton(result_image, keypoints)
            
            # Dibujar clasificación si está disponible
            if classification:
                pose_class = classification['pose_class']
                confidence = classification['confidence']
                
                # Encontrar posición para texto (centroide de keypoints válidos)
                valid_kps = keypoints[keypoints[:, 2] > self.confidence_threshold]
                if len(valid_kps) > 0:
                    center_x = int(np.mean(valid_kps[:, 0]))
                    center_y = int(np.mean(valid_kps[:, 1])) - 30
                    
                    # Texto con fondo
                    text = f"{pose_class}: {confidence:.2f}"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    
                    cv2.rectangle(result_image, 
                                (center_x - 5, center_y - text_size[1] - 5),
                                (center_x + text_size[0] + 5, center_y + 5),
                                (0, 0, 0), -1)
                    
                    cv2.putText(result_image, text,
                              (center_x, center_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Información general
        info_text = f"Personas: {frame_result['people_detected']} | Tiempo: {frame_result['processing_time_ms']:.1f}ms"
        cv2.putText(result_image, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return result_image
    
    def _draw_openpose_skeleton(self, image: np.ndarray, keypoints: np.ndarray):
        """
        Dibuja el esqueleto completo de OpenPose con conexiones
        
        Args:
            image: Imagen donde dibujar
            keypoints: Array de keypoints (18, 3)
        """
        # Definir conexiones del esqueleto OpenPose (18 keypoints)
        openpose_connections = [
            # Cabeza
            (0, 1), (1, 2), (2, 3), (3, 4),    # nose -> ojos -> orejas
            (1, 5), (2, 6),                     # ojos -> hombros
            # Torso
            (5, 6), (5, 7), (6, 8),            # hombros y brazos superiores
            (7, 9), (8, 10),                   # brazos
            (5, 11), (6, 12), (11, 12),        # caderas
            # Piernas
            (11, 13), (12, 14),                # muslos
            (13, 15), (14, 16),                # piernas inferiores
        ]
        
        # Colores para diferentes partes del cuerpo
        colors = {
            'head': (255, 0, 0),      # Azul para cabeza
            'arms': (0, 255, 0),      # Verde para brazos
            'torso': (0, 0, 255),     # Rojo para torso
            'legs': (255, 255, 0),    # Cian para piernas
        }
        
        # Dibujar conexiones
        for connection in openpose_connections:
            kp1_idx, kp2_idx = connection
            kp1 = keypoints[kp1_idx]
            kp2 = keypoints[kp2_idx]
            
            # Solo dibujar si ambos keypoints tienen buena confianza
            if kp1[2] > self.confidence_threshold and kp2[2] > self.confidence_threshold:
                x1, y1 = int(kp1[0]), int(kp1[1])
                x2, y2 = int(kp2[0]), int(kp2[1])
                
                # Elegir color según la parte del cuerpo
                if kp1_idx <= 4 or kp2_idx <= 4:
                    color = colors['head']
                elif kp1_idx in [5, 6, 7, 8, 9, 10] or kp2_idx in [5, 6, 7, 8, 9, 10]:
                    color = colors['arms']
                elif kp1_idx in [11, 12] or kp2_idx in [11, 12]:
                    color = colors['torso']
                else:
                    color = colors['legs']
                
                cv2.line(image, (x1, y1), (x2, y2), color, 2)
        
        # Dibujar keypoints individuales
        for i, kp in enumerate(keypoints):
            x, y, conf = kp
            if conf > self.confidence_threshold:
                # Color diferente para keypoints importantes
                if i == 0:  # Nariz
                    color = (255, 255, 255)
                    radius = 4
                elif i in [1, 2]:  # Ojos
                    color = (255, 0, 255)
                    radius = 3
                else:
                    color = (0, 255, 0)
                    radius = 3
                
                cv2.circle(image, (int(x), int(y)), radius, color, -1)
    
    def process_video(self, video_source: str = 0, output_path: str = None):
        """
        Procesa video en tiempo real con OpenPose + clasificación
        
        Args:
            video_source: Fuente de video (0 para webcam, ruta de archivo para video)
            output_path: Ruta para guardar video de salida (opcional)
        """
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            logger.error(f"❌ No se pudo abrir la fuente de video: {video_source}")
            return
        
        # Configurar video writer si se especifica output
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        logger.info("🎥 Iniciando procesamiento de video...")
        logger.info("   Presiona 'q' para salir")
        logger.info("   Presiona 's' para mostrar estadísticas")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Procesar frame
                result = self.process_frame_with_classification(frame)
                
                # Dibujar resultados
                output_frame = self.draw_results(frame, result)
                
                # Guardar si es necesario
                if writer:
                    writer.write(output_frame)
                
                # Mostrar
                cv2.imshow('OpenPose + Clasificación de Poses', output_frame)
                
                # Controles de teclado
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self._print_stats()
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
        self._print_final_stats()
    
    def _print_stats(self):
        """Imprime estadísticas actuales"""
        elapsed = time.time() - self.stats['start_time']
        fps = self.stats['frames_processed'] / elapsed if elapsed > 0 else 0
        
        logger.info("📊 Estadísticas actuales:")
        logger.info(f"   ⏱️ Tiempo transcurrido: {elapsed:.1f}s")
        logger.info(f"   🎬 Frames procesados: {self.stats['frames_processed']}")
        logger.info(f"   👥 Poses detectadas: {self.stats['poses_detected']}")
        logger.info(f"   🎭 Poses clasificadas: {self.stats['poses_classified']}")
        logger.info(f"   📈 FPS promedio: {fps:.1f}")
    
    def _print_final_stats(self):
        """Imprime estadísticas finales"""
        logger.info("🏁 Procesamiento completado")
        self._print_stats()


def main():
    """
    Función principal para ejecutar el ejemplo
    """
    # Configuración de rutas (ajustar según tu instalación)
    OPENPOSE_MODEL_FOLDER = "/home/work/openpose/models/"  # Ajustar según instalación
    POSE_CLASSIFIER_MODEL = "/path/to/pose_classification_net.engine"  # Ajustar ruta
    
    # Configuración de parámetros
    SEQUENCE_LENGTH = 100          # Frames para clasificación temporal
    CONFIDENCE_THRESHOLD = 0.3     # Umbral de confianza para keypoints
    
    # Fuente de video
    VIDEO_SOURCE = 0  # 0 para webcam, o ruta a archivo de video
    OUTPUT_PATH = None  # Opcional: "output_video.mp4" para guardar
    
    try:
        # Crear sistema integrado
        logger.info("🚀 Iniciando sistema OpenPose + Clasificación...")
        
        system = OpenPoseWithClassifier(
            openpose_model_folder=OPENPOSE_MODEL_FOLDER,
            pose_classifier_model_path=POSE_CLASSIFIER_MODEL,
            sequence_length=SEQUENCE_LENGTH,
            confidence_threshold=CONFIDENCE_THRESHOLD
        )
        
        # Procesar video
        system.process_video(
            video_source=VIDEO_SOURCE,
            output_path=OUTPUT_PATH
        )
        
    except KeyboardInterrupt:
        logger.info("⏹️ Detenido por el usuario")
    except Exception as e:
        logger.error(f"❌ Error en ejecución: {e}")
        raise


if __name__ == "__main__":
    main()
