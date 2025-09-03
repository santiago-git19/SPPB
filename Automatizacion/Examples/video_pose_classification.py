#!/usr/bin/env python3
"""
Ejemplo de Detección de Poses + Clasificación de Postura
========================================================

Este script procesa un video de entrada y genera un video de salida con:
- Keypoints visualizados
- Predicción de postura (de pie/sentado/indeterminado)
- Nivel de confianza de la predicción

El script usa MediaPipe para detectar poses y el manual_action_detector 
para clasificar si la persona está de pie, sentada o indeterminado.

Uso:
    python video_pose_classification.py --video input.mp4 --output output.mp4
    python video_pose_classification.py --video input.mp4  # Solo mostrar, no guardar

Características:
- Detección de keypoints con MediaPipe
- Clasificación de postura (de pie/sentado/indeterminado)
- Visualización de keypoints y texto de predicción
- Guardado opcional del video procesado
- Estadísticas de procesamiento

Autor: Sistema de IA
Fecha: 2025
"""

import argparse
import cv2
import logging
import numpy as np
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from collections import Counter

# Añadir el directorio padre al sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Importar módulos necesarios
try:
    from utils.manual_action_detector import PostureClassifier, KEYPOINT_NAMES
    from utils.pose_detection.trt_pose_proc import TRTPoseProcessor
except ImportError as e:
    print(f"Error importando módulos: {e}")
    print("Asegúrate de ejecutar desde el directorio Examples/")
    sys.exit(1)

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Colores para visualización
COLORS = {
    'de pie': (0, 255, 0),      # Verde
    'sentado': (0, 165, 255),   # Naranja
    'indeterminado': (128, 128, 128)  # Gris
}

# Conexiones entre keypoints para dibujar el esqueleto
SKELETON_CONNECTIONS = [
    # Cara
    (0, 1), (0, 2), (1, 3), (2, 4),  # nose to eyes, eyes to ears
    # Torso
    (5, 6), (5, 11), (6, 12), (11, 12),  # shoulders to hips
    # Brazo izquierdo
    (5, 7), (7, 9),  # left shoulder -> elbow -> wrist
    # Brazo derecho  
    (6, 8), (8, 10),  # right shoulder -> elbow -> wrist
    # Pierna izquierda
    (11, 13), (13, 15),  # left hip -> knee -> ankle
    # Pierna derecha
    (12, 14), (14, 16),  # right hip -> knee -> ankle
    # Cuello (si existe)
    (17, 5), (17, 6),  # neck to shoulders
]

class VideoPoseClassifier:
    """
    Procesador de video que combina detección de poses con clasificación de postura.
    """
    
    def __init__(self, confidence_threshold=0.01, model_path=None, topology_path=None):
        """
        Inicializar el procesador.
        
        Args:
            confidence_threshold: Umbral de confianza para keypoints
            model_path: Ruta al modelo TRT Pose (.pth)
            topology_path: Ruta al archivo de topología JSON
        """
        self.confidence_threshold = confidence_threshold
        
        # Paths por defecto si no se especifican
        if model_path is None:
            model_path = "../models/resnet18_baseline_att_224x224_A_epoch_249.pth"  # Cambiar según tu modelo
        if topology_path is None:
            topology_path = "../models/human_pose.json"
        
        # Inicializar detector de poses TRT
        try:
            self.pose_detector = TRTPoseProcessor(
                model_path=model_path,
                topology_path=topology_path,
                use_tensorrt=True
            )
            logger.info("Detector TRT Pose inicializado")
        except Exception as e:
            logger.error(f"Error inicializando TRT Pose: {e}")
            logger.info("Intentando con PyTorch normal...")
            try:
                self.pose_detector = TRTPoseProcessor(
                    model_path=model_path,
                    topology_path=topology_path,
                    use_tensorrt=False
                )
                logger.info("Detector PyTorch Pose inicializado")
            except Exception as e2:
                logger.error(f"Error inicializando PyTorch Pose: {e2}")
                raise ValueError("No se pudo inicializar ningún detector de poses")
        
        # Inicializar clasificador de postura
        self.posture_classifier = PostureClassifier(confidence_threshold)
        logger.info("Clasificador de postura inicializado")
        
        # Estadísticas
        self.frame_count = 0
        self.posture_stats = Counter()
        
    def process_video(self, video_path: str, output_path: Optional[str] = None, 
                     show_preview: bool = True) -> Dict:
        """
        Procesar video completo.
        
        Args:
            video_path: Ruta del video de entrada
            output_path: Ruta del video de salida (opcional)
            show_preview: Mostrar preview en tiempo real
            
        Returns:
            Dict: Estadísticas del procesamiento
        """
        # Abrir video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video_path}")
            
        # Propiedades del video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Configurar writer si se especifica salida
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            logger.info(f"Guardando resultado en: {output_path}")
        
        # Estadísticas
        start_time = time.time()
        processed_frames = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Procesar frame
                processed_frame = self._process_frame(frame)
                processed_frames += 1
                
                # Mostrar progreso
                if processed_frames % 30 == 0:
                    progress = (processed_frames / total_frames) * 100 if total_frames > 0 else 0
                    logger.info(f"Procesado: {processed_frames}/{total_frames} frames ({progress:.1f}%)")
                
                # Guardar frame si hay writer
                if writer:
                    writer.write(processed_frame)
                    
                # Mostrar preview
                # if show_preview:
                #     cv2.imshow('Pose Classification', processed_frame)
                #     key = cv2.waitKey(1) & 0xFF
                #     if key == ord('q'):
                #         logger.info("Detenido por usuario")
                #         break
                        
        finally:
            # Limpiar recursos
            cap.release()
            if writer:
                writer.release()
            if show_preview:
                cv2.destroyAllWindows()
        
        # Calcular estadísticas finales
        end_time = time.time()
        processing_time = end_time - start_time
        fps_processed = processed_frames / processing_time if processing_time > 0 else 0
        
        stats = {
            'processed_frames': processed_frames,
            'total_frames': total_frames,
            'processing_time': processing_time,
            'fps_processed': fps_processed,
            'posture_distribution': dict(self.posture_stats),
            'output_saved': output_path is not None
        }
        
        logger.info(f"Procesamiento completado: {processed_frames} frames en {processing_time:.1f}s")
        logger.info(f"FPS de procesamiento: {fps_processed:.1f}")
        logger.info(f"Distribución de posturas: {dict(self.posture_stats)}")
        
        return stats
        
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Procesar un frame individual.
        
        Args:
            frame: Frame de entrada
            
        Returns:
            np.ndarray: Frame procesado con keypoints y predicción
        """
        self.frame_count += 1
        
        # Detectar poses usando TRT Pose
        keypoints = self.pose_detector.process_frame(frame)
        
        if len(keypoints) > 0:
            # Dibujar esqueleto
            frame = self.pose_detector.visualize_keypoints(frame, keypoints, draw_skeleton=True)
            
            # Clasificar postura (usamos los mismos keypoints para ambas "cámaras")
            # En un caso real, tendrías keypoints de cámaras frontal y lateral
            posture_result = self.posture_classifier.classify_posture(
                keypoints_frontal=[],
                keypoints_lateral=keypoints  # En este ejemplo usamos los mismos
            )
            
            # Actualizar estadísticas
            self.posture_stats[posture_result['posture']] += 1
            
            # Dibujar predicción adicional
            self._draw_prediction(frame, posture_result, keypoints)
        else:
            # No se detectaron poses
            self.posture_stats['sin_detección'] += 1
            
        return frame
        
    def _draw_skeleton(self, frame: np.ndarray, keypoints: List[Tuple]) -> None:
        """
        Dibujar el esqueleto de la pose.
        
        Args:
            frame: Frame donde dibujar
            keypoints: Lista de keypoints (x, y, confidence, idx)
        """
        # Convertir keypoints a diccionario para fácil acceso
        kp_dict = {}
        for x, y, conf, idx in keypoints:
            if conf >= self.confidence_threshold and 0 <= idx < len(KEYPOINT_NAMES):
                kp_dict[idx] = (int(x), int(y), conf)
        
        # Dibujar conexiones
        for start_idx, end_idx in SKELETON_CONNECTIONS:
            if start_idx in kp_dict and end_idx in kp_dict:
                start_point = kp_dict[start_idx][:2]
                end_point = kp_dict[end_idx][:2]
                
                # Color basado en confianza promedio
                avg_conf = (kp_dict[start_idx][2] + kp_dict[end_idx][2]) / 2
                color_intensity = int(255 * avg_conf)
                color = (0, color_intensity, 255 - color_intensity)  # De rojo a verde
                
                cv2.line(frame, start_point, end_point, color, 2)
        
        # Dibujar keypoints
        for idx, (x, y, conf) in kp_dict.items():
            if conf >= self.confidence_threshold:
                # Color basado en confianza
                color_intensity = int(255 * conf)
                color = (255 - color_intensity, color_intensity, 0)  # De rojo a verde
                
                cv2.circle(frame, (x, y), 4, color, -1)
                cv2.circle(frame, (x, y), 6, (255, 255, 255), 2)  # Borde blanco
    
    def _draw_prediction(self, frame: np.ndarray, posture_result: Dict, 
                        keypoints: List[Tuple]) -> None:
        """
        Dibujar la predicción de postura en el frame.
        
        Args:
            frame: Frame donde dibujar
            posture_result: Resultado de clasificación de postura
            keypoints: Keypoints de la pose
        """
        posture = posture_result['posture']
        confidence = posture_result['confidence']
        
        # Color según postura
        color = COLORS.get(posture, (255, 255, 255))
        
        # Encontrar posición para el texto (cerca del centro de la pose)
        valid_keypoints = [(x, y) for x, y, conf, idx in keypoints 
                          if conf >= self.confidence_threshold]
        
        if valid_keypoints:
            # Calcular centro de la pose
            center_x = int(np.mean([x for x, y in valid_keypoints]))
            center_y = int(np.mean([y for x, y in valid_keypoints])) - 50  # Arriba del centro
        else:
            center_x, center_y = 50, 50
        
        # Texto de la predicción
        text = f"{posture.upper()}"
        confidence_text = f"Confianza: {confidence:.2f}"
        
        # Configuración del texto
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        # Calcular tamaño del texto
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        (conf_w, conf_h), _ = cv2.getTextSize(confidence_text, font, font_scale * 0.6, thickness - 1)
        
        # Dibujar fondo para el texto
        bg_width = max(text_w, conf_w) + 20
        bg_height = text_h + conf_h + 30
        
        cv2.rectangle(frame, 
                     (center_x - bg_width//2, center_y - bg_height//2),
                     (center_x + bg_width//2, center_y + bg_height//2),
                     (0, 0, 0), -1)
        
        cv2.rectangle(frame, 
                     (center_x - bg_width//2, center_y - bg_height//2),
                     (center_x + bg_width//2, center_y + bg_height//2),
                     color, 2)
        
        # Dibujar texto
        cv2.putText(frame, text,
                   (center_x - text_w//2, center_y - 5),
                   font, font_scale, color, thickness)
        
        cv2.putText(frame, confidence_text,
                   (center_x - conf_w//2, center_y + 20),
                   font, font_scale * 0.6, (255, 255, 255), thickness - 1)


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description="Procesar video con detección de poses y clasificación de postura usando TRT Pose"
    )
    parser.add_argument('--video', type=str, required=True,
                       help='Ruta del video de entrada')
    parser.add_argument('--output', type=str, default=None,
                       help='Ruta del video de salida (opcional)')
    parser.add_argument('--confidence', type=float, default=0.01,
                       help='Umbral de confianza para keypoints (default: 0.5)')
    parser.add_argument('--model', type=str, default=None,
                       help='Ruta al modelo TRT Pose (.pth)')
    parser.add_argument('--topology', type=str, default=None,
                       help='Ruta al archivo de topología JSON')
    parser.add_argument('--no-preview', action='store_true',
                       help='No mostrar preview en tiempo real')
    
    args = parser.parse_args()
    
    # Verificar que el archivo de video existe
    if not Path(args.video).exists():
        logger.error(f"El archivo de video no existe: {args.video}")
        return
    
    try:
        # Crear procesador
        processor = VideoPoseClassifier(
            confidence_threshold=args.confidence,
            model_path=args.model,
            topology_path=args.topology
        )
        
        # Procesar video
        logger.info(f"Iniciando procesamiento de: {args.video}")
        stats = processor.process_video(
            video_path=args.video,
            output_path=args.output,
            show_preview=not args.no_preview
        )
        
        # Mostrar estadísticas finales
        print("\\n" + "="*60)
        print("ESTADÍSTICAS DE PROCESAMIENTO")
        print("="*60)
        print(f"Frames procesados: {stats['processed_frames']}")
        print(f"Tiempo total: {stats['processing_time']:.1f}s")
        print(f"FPS de procesamiento: {stats['fps_processed']:.1f}")
        print("\\nDistribución de posturas:")
        for posture, count in stats['posture_distribution'].items():
            percentage = (count / stats['processed_frames']) * 100
            print(f"  {posture}: {count} fra    mes ({percentage:.1f}%)")
        
        if args.output and stats['output_saved']:
            print(f"\\nVideo guardado en: {args.output}")
        
    except KeyboardInterrupt:
        logger.info("Procesamiento interrumpido por el usuario")
    except Exception as e:
        logger.error(f"Error durante el procesamiento: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
