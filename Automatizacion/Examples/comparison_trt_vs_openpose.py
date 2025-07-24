#!/usr/bin/env python3
"""
Comparación: trt_pose vs OpenPose para Clasificación de Poses
===========================================================

Script de demostración que compara ambos enfoques:
1. trt_pose + clasificación (17 keypoints COCO → NVIDIA)
2. OpenPose + clasificación (18 keypoints OpenPose → NVIDIA)

Muestra las diferencias en precisión, velocidad y robustez.

Autor: Sistema de IA
Fecha: 2025
"""

import cv2
import numpy as np
import time
import logging
from pathlib import Path
import sys

# Añadir rutas
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.trt_pose_proc import TRTPoseProcessor
from utils.openpose_proc import OpenPoseProcessor
from utils.trt_pose_classifier import create_pose_classifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PoseDetectionComparison:
    """
    Compara trt_pose vs OpenPose para clasificación de poses
    """
    
    def __init__(self, 
                 trt_pose_model_path: str,
                 trt_pose_topology_path: str,
                 openpose_model_folder: str,
                 pose_classifier_model_path: str):
        """
        Inicializa ambos sistemas para comparación
        """
        self.systems = {}
        
        # Sistema 1: trt_pose
        logger.info("🔧 Inicializando trt_pose...")
        self.systems['trt_pose'] = {
            'detector': TRTPoseProcessor(
                model_path=trt_pose_model_path,
                topology_path=trt_pose_topology_path,
                use_tensorrt=False
            ),
            'classifier': create_pose_classifier(
                model_path=pose_classifier_model_path,
                keypoint_format='nvidia',  # trt_pose usa formato COCO → NVIDIA
                sequence_length=100,
                confidence_threshold=0.3
            ),
            'stats': {'total_time': 0, 'frames': 0, 'detections': 0}
        }
        
        # Sistema 2: OpenPose
        logger.info("🔧 Inicializando OpenPose...")
        self.systems['openpose'] = {
            'detector': OpenPoseProcessor(model_folder=openpose_model_folder),
            'classifier': create_pose_classifier(
                model_path=pose_classifier_model_path,
                keypoint_format='openpose',  # OpenPose usa su propio formato
                sequence_length=100,
                confidence_threshold=0.3
            ),
            'stats': {'total_time': 0, 'frames': 0, 'detections': 0}
        }
        
        logger.info("✅ Ambos sistemas inicializados")
    
    def process_frame_comparison(self, image: np.ndarray) -> dict:
        """
        Procesa un frame con ambos sistemas y compara resultados
        """
        results = {}
        
        for system_name, system in self.systems.items():
            start_time = time.time()
            
            try:
                # Detectar keypoints
                keypoints = system['detector'].process_frame(image)
                
                result = {
                    'keypoints': keypoints,
                    'classification': None,
                    'processing_time': 0,
                    'detected': False
                }
                
                if keypoints is not None:
                    result['detected'] = True
                    system['stats']['detections'] += 1
                    
                    # Clasificar pose
                    classification = system['classifier'].process_keypoints(keypoints)
                    if classification and not classification.get('error', False):
                        result['classification'] = classification
                
                # Estadísticas de tiempo
                processing_time = time.time() - start_time
                result['processing_time'] = processing_time * 1000  # ms
                system['stats']['total_time'] += processing_time
                system['stats']['frames'] += 1
                
                results[system_name] = result
                
            except Exception as e:
                logger.error(f"❌ Error en {system_name}: {e}")
                results[system_name] = {
                    'keypoints': None,
                    'classification': None,
                    'processing_time': 0,
                    'detected': False,
                    'error': str(e)
                }
        
        return results
    
    def draw_comparison(self, image: np.ndarray, results: dict) -> np.ndarray:
        """
        Dibuja una comparación lado a lado de ambos sistemas
        """
        h, w = image.shape[:2]
        
        # Crear imagen de comparación (lado a lado)
        comparison_image = np.zeros((h, w * 2, 3), dtype=np.uint8)
        
        # Lado izquierdo: trt_pose
        left_image = image.copy()
        if 'trt_pose' in results and results['trt_pose']['detected']:
            self._draw_trt_pose_results(left_image, results['trt_pose'])
        
        # Título y estadísticas para trt_pose
        cv2.putText(left_image, "trt_pose (COCO 17)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if 'trt_pose' in results:
            time_text = f"Tiempo: {results['trt_pose']['processing_time']:.1f}ms"
            cv2.putText(left_image, time_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if results['trt_pose']['classification']:
                class_text = f"Clase: {results['trt_pose']['classification']['predicted_class']}"
                conf_text = f"Conf: {results['trt_pose']['classification']['confidence']:.2f}"
                cv2.putText(left_image, class_text, (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(left_image, conf_text, (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Lado derecho: OpenPose
        right_image = image.copy()
        if 'openpose' in results and results['openpose']['detected']:
            self._draw_openpose_results(right_image, results['openpose'])
        
        # Título y estadísticas para OpenPose
        cv2.putText(right_image, "OpenPose (18)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        if 'openpose' in results:
            time_text = f"Tiempo: {results['openpose']['processing_time']:.1f}ms"
            cv2.putText(right_image, time_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if results['openpose']['classification']:
                class_text = f"Clase: {results['openpose']['classification']['predicted_class']}"
                conf_text = f"Conf: {results['openpose']['classification']['confidence']:.2f}"
                cv2.putText(right_image, class_text, (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(right_image, conf_text, (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Combinar imágenes
        comparison_image[:, :w] = left_image
        comparison_image[:, w:] = right_image
        
        # Línea divisoria
        cv2.line(comparison_image, (w, 0), (w, h), (255, 255, 255), 2)
        
        return comparison_image
    
    def _draw_trt_pose_results(self, image: np.ndarray, result: dict):
        """Dibuja resultados de trt_pose"""
        keypoints = result['keypoints']
        if keypoints is not None:
            # Dibujar keypoints simples para trt_pose
            for kp in keypoints:
                x, y, conf = kp
                if conf > 0.3:
                    cv2.circle(image, (int(x), int(y)), 3, (0, 255, 255), -1)
    
    def _draw_openpose_results(self, image: np.ndarray, result: dict):
        """Dibuja resultados de OpenPose con esqueleto completo"""
        keypoints = result['keypoints']
        if keypoints is not None:
            # Conexiones OpenPose
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (2, 6),
                (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
                (5, 11), (6, 12), (11, 12), (11, 13), (12, 14),
                (13, 15), (14, 16)
            ]
            
            # Dibujar conexiones
            for connection in connections:
                kp1_idx, kp2_idx = connection
                if kp1_idx < len(keypoints) and kp2_idx < len(keypoints):
                    kp1 = keypoints[kp1_idx]
                    kp2 = keypoints[kp2_idx]
                    
                    if kp1[2] > 0.3 and kp2[2] > 0.3:
                        cv2.line(image, 
                               (int(kp1[0]), int(kp1[1])),
                               (int(kp2[0]), int(kp2[1])),
                               (255, 0, 255), 2)
            
            # Dibujar keypoints
            for kp in keypoints:
                x, y, conf = kp
                if conf > 0.3:
                    cv2.circle(image, (int(x), int(y)), 4, (255, 0, 255), -1)
    
    def run_comparison(self, video_source: str = 0):
        """
        Ejecuta la comparación en tiempo real
        """
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            logger.error(f"❌ No se pudo abrir: {video_source}")
            return
        
        logger.info("🎥 Iniciando comparación...")
        logger.info("   Presiona 'q' para salir")
        logger.info("   Presiona 's' para estadísticas")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Procesar con ambos sistemas
                results = self.process_frame_comparison(frame)
                
                # Dibujar comparación
                comparison_frame = self.draw_comparison(frame, results)
                
                # Mostrar
                cv2.imshow('Comparación: trt_pose vs OpenPose', comparison_frame)
                
                # Controles
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self._print_comparison_stats()
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self._print_final_comparison()
    
    def _print_comparison_stats(self):
        """Imprime estadísticas de comparación"""
        logger.info("📊 Estadísticas de Comparación:")
        
        for system_name, system in self.systems.items():
            stats = system['stats']
            avg_time = (stats['total_time'] / stats['frames']) * 1000 if stats['frames'] > 0 else 0
            fps = stats['frames'] / stats['total_time'] if stats['total_time'] > 0 else 0
            detection_rate = (stats['detections'] / stats['frames']) * 100 if stats['frames'] > 0 else 0
            
            logger.info(f"   {system_name.upper()}:")
            logger.info(f"     ⏱️ Tiempo promedio: {avg_time:.1f}ms")
            logger.info(f"     📈 FPS: {fps:.1f}")
            logger.info(f"     🎯 Tasa detección: {detection_rate:.1f}%")
    
    def _print_final_comparison(self):
        """Imprime resumen final"""
        logger.info("🏁 Comparación completada")
        self._print_comparison_stats()
        
        # Recomendaciones
        trt_stats = self.systems['trt_pose']['stats']
        op_stats = self.systems['openpose']['stats']
        
        trt_fps = trt_stats['frames'] / trt_stats['total_time'] if trt_stats['total_time'] > 0 else 0
        op_fps = op_stats['frames'] / op_stats['total_time'] if op_stats['total_time'] > 0 else 0
        
        logger.info("🎯 Recomendaciones:")
        if trt_fps > op_fps:
            logger.info("   ⚡ trt_pose es más rápido - mejor para tiempo real")
        if op_stats['detections'] > trt_stats['detections']:
            logger.info("   🎯 OpenPose detecta más poses - mejor precisión")


def main():
    """Función principal"""
    # Configuración (ajustar rutas)
    TRT_POSE_MODEL = "/path/to/trt_pose_model.pth"
    TRT_POSE_TOPOLOGY = "/path/to/topology.json"
    OPENPOSE_MODELS = "/path/to/openpose/models/"
    POSE_CLASSIFIER = "/path/to/pose_classification_net.engine"
    
    VIDEO_SOURCE = 0  # 0 para webcam
    
    try:
        comparison = PoseDetectionComparison(
            trt_pose_model_path=TRT_POSE_MODEL,
            trt_pose_topology_path=TRT_POSE_TOPOLOGY,
            openpose_model_folder=OPENPOSE_MODELS,
            pose_classifier_model_path=POSE_CLASSIFIER
        )
        
        comparison.run_comparison(video_source=VIDEO_SOURCE)
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
