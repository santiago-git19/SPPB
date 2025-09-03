#!/usr/bin/env python3
"""
Análisis de la Marcha con Archivo .bag
=====================================

Este script procesa un archivo .bag generado por OrbbecViewer para realizar 
análisis de la marcha 3D. Extrae los frames de color y profundidad del archivo
.bag y los procesa para:

- Detectar poses usando TRT Pose
- Calcular la trayectoria 3D del centro de la cadera
- Medir la distancia total recorrida
- Generar estadísticas de la marcha

El script usa:
- pyorbbecsdk para leer archivos .bag
- TRTPoseProcessor para detección de poses
- Gait3DTracker para análisis de marcha 3D

Uso:
    python bag_file_gait_analysis.py --bag archivo.bag --output resultados.json
    python bag_file_gait_analysis.py --bag archivo.bag --visualize

Características:
- Procesamiento de archivos .bag de Orbbec
- Detección de poses en frames de color
- Reconstrucción 3D de la trayectoria de la cadera
- Cálculo de distancia recorrida
- Visualización opcional en tiempo real
- Exportación de resultados a JSON

Autor: Sistema de IA
Fecha: 2025
"""

import argparse
import cv2
import json
import logging
import numpy as np
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from collections import defaultdict

# Añadir el directorio padre al sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Importar módulos necesarios
try:
# Imports del SDK de Orbbec: Ya no necesitamos Pipeline, Config, Frame
# porque usamos BagOrbbecCapture que los abstrae
    from utils.action_and_movement_detection.gait_3d_tracker import Gait3DTracker
    from utils.pose_detection.trt_pose_proc import TRTPoseProcessor
    from utils.bag_orbbec_capture import BagOrbbecCapture
except ImportError as e:
    print(f"Error importando módulos: {e}")
    print("Asegúrate de que pyorbbecsdk esté instalado y los módulos estén disponibles")
    sys.exit(1)

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BagFileGaitAnalyzer:
    """
    Analizador de marcha que procesa archivos .bag de Orbbec.
    """
    
    def __init__(self, model_path=None, topology_path=None):
        """
        Inicializar el analizador.
        
        Args:
            model_path: Ruta al modelo TRT Pose (.pth o .engine)
            topology_path: Ruta al archivo de topología JSON
        """
        # Paths por defecto si no se especifican
        if model_path is None:
            model_path = "../models/resnet18_baseline_att_224x224_A_epoch_249.pth"
        if topology_path is None:
            topology_path = "../models/human_pose.json"
        
        # Inicializar detector de poses TRT
        try:
            self.pose_processor = TRTPoseProcessor(
                model_path=model_path,
                topology_path=topology_path,
                use_tensorrt=False  # Usar PyTorch por defecto
            )
            logger.info("Detector TRT Pose (PyTorch) inicializado")
        except Exception as e:
            logger.error(f"Error inicializando TRT Pose: {e}")
            raise ValueError("No se pudo inicializar el detector de poses")
        
        # El tracker se inicializará en process_bag_file con el archivo .bag real
        self.gait_tracker = None
        
        # Estadísticas
        self.frame_count = 0
        self.processed_frames = 0
        self.trajectory_3d = []
        self.processing_times = []
        
    def process_bag_file(self, bag_path: str, output_video_path: str, 
                        max_frames: Optional[int] = None) -> Dict:
        """
        Procesar archivo .bag completo y generar video de salida.
        
        Args:
            bag_path: Ruta del archivo .bag
            output_video_path: Ruta del video de salida
            max_frames: Número máximo de frames a procesar (None = todos)
            
        Returns:
            Dict: Resultados del análisis de marcha
        """
        if not Path(bag_path).exists():
            raise ValueError(f"El archivo .bag no existe: {bag_path}")
        
        # Inicializar BagOrbbecCapture
        bag_capture = BagOrbbecCapture(
            bag_path=bag_path,
            enable_depth=True,
            auto_loop=False
        )
        
        # Actualizar el tracker con la nueva fuente
        self.gait_tracker = Gait3DTracker(capture_source=bag_capture)
        
        # Variables para el video de salida
        video_writer = None
        frame_width = 640
        frame_height = 480
        
        # Estadísticas
        start_time = time.time()
        
        try:
            logger.info(f"Procesando archivo: {bag_path}")
            
            while True:
                # Verificar límite de frames
                if max_frames and self.processed_frames >= max_frames:
                    logger.info(f"Alcanzado límite de frames: {max_frames}")
                    break
                
                frame_start_time = time.time()
                
                # Leer frames usando BagOrbbecCapture
                color_frame, depth_frame = bag_capture.read_frame_with_depth()
                
                if color_frame is None:
                    logger.info("No hay más frames disponibles")
                    break
                    
                self.frame_count += 1
                
                if depth_frame is None:
                    logger.warning(f"Frame {self.frame_count}: Falta frame de profundidad")
                    continue
                
                # Procesar frame
                success, vis_frame = self._process_frame_pair(color_frame, depth_frame)
                
                if success:
                    self.processed_frames += 1
                    
                    # Inicializar video writer si es el primer frame exitoso
                    if video_writer is None and vis_frame is not None:
                        frame_height, frame_width = vis_frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(
                            output_video_path, fourcc, 30.0, (frame_width, frame_height)
                        )
                        logger.info(f"Video writer inicializado: {frame_width}x{frame_height}")
                    
                    # Escribir frame al video
                    if video_writer is not None and vis_frame is not None:
                        video_writer.write(vis_frame)
                    
                    # Calcular tiempo de procesamiento
                    frame_time = time.time() - frame_start_time
                    self.processing_times.append(frame_time)
                    
                    # Mostrar progreso cada 30 frames
                    if self.processed_frames % 30 == 0:
                        avg_time = np.mean(self.processing_times[-30:])
                        logger.info(f"Procesados: {self.processed_frames} frames "
                                   f"(tiempo promedio: {avg_time:.3f}s/frame)")
                        
        except KeyboardInterrupt:
            logger.info("Procesamiento interrumpido por el usuario")
            
        finally:
            # Limpiar recursos
            if video_writer is not None:
                video_writer.release()
                logger.info(f"Video guardado en: {output_video_path}")
            bag_capture.release()
        
        # Calcular estadísticas finales
        end_time = time.time()
        total_time = end_time - start_time
        
        results = self._generate_results(total_time)
        
        logger.info(f"Procesamiento completado: {self.processed_frames} frames en {total_time:.1f}s")
        logger.info(f"Distancia total recorrida: {results['total_distance_m']:.3f} metros")
        
        return results
        
    def _process_frame_pair(self, color_frame: np.ndarray, depth_frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Procesar un par de frames (color + profundidad) y generar frame visualizado.
        
        Args:
            color_frame: Frame de color como numpy array BGR
            depth_frame: Frame de profundidad como numpy array uint16
            
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (éxito, frame_visualizado)
        """
        try:
            # El color_frame ya viene en formato BGR desde BagOrbbecCapture
            color_image = color_frame
            
            # El depth_frame ya viene como numpy array uint16 desde BagOrbbecCapture
            depth_image = depth_frame
            
            # Detectar poses usando TRT Pose
            keypoints = self.pose_processor.process_frame(color_image)
            
            if len(keypoints) > 0:
                # Actualizar tracker de marcha 3D
                point_3d = self.gait_tracker.update(keypoints, depth_image)
                
                if point_3d is not None:
                    # Agregar punto 3D a la trayectoria
                    self.trajectory_3d.append({
                        'frame': self.processed_frames,
                        'timestamp': time.time(),
                        'point_3d': point_3d.tolist(),
                        'keypoints_count': len(keypoints)
                    })
                
                # Generar frame visualizado para el video
                vis_frame = self._create_visualization_frame(color_image, keypoints, point_3d)
                
                return True, vis_frame
            else:
                logger.debug(f"Frame {self.frame_count}: No se detectaron poses")
                return False, None
                
        except Exception as e:
            logger.error(f"Error procesando frame {self.frame_count}: {e}")
            return False, None
            
    def _create_visualization_frame(self, frame: np.ndarray, keypoints: List, 
                                  point_3d: Optional[np.ndarray]) -> np.ndarray:
        """
        Crear frame visualizado con keypoints y información 3D para el video.
        
        Args:
            frame: Frame de imagen
            keypoints: Lista de keypoints detectados
            point_3d: Punto 3D del centro de cadera (si está disponible)
            
        Returns:
            np.ndarray: Frame visualizado para el video
        """
        # Dibujar keypoints
        vis_frame = self.pose_processor.visualize_keypoints(
            frame, keypoints, draw_skeleton=True
        )
        
        # Agregar información de marcha 3D
        if point_3d is not None:
            info_text = f"Posicion 3D: ({point_3d[0]:.2f}, {point_3d[1]:.2f}, {point_3d[2]:.2f})"
            cv2.putText(vis_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Información general
        distance_text = f"Distancia total: {self.gait_tracker.total_distance_m:.3f}m"
        cv2.putText(vis_frame, distance_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        frame_text = f"Frame: {self.processed_frames}"
        cv2.putText(vis_frame, frame_text, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return vis_frame
        
    def _generate_results(self, total_time: float) -> Dict:
        """
        Generar resultados finales del análisis.
        
        Args:
            total_time: Tiempo total de procesamiento
            
        Returns:
            Dict: Resultados del análisis
        """
        # Calcular estadísticas de velocidad
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        fps_processed = self.processed_frames / total_time if total_time > 0 else 0
        
        # Calcular estadísticas de trayectoria
        trajectory_stats = self._calculate_trajectory_stats()
        
        results = {
            'processing_info': {
                'total_frames': self.frame_count,
                'processed_frames': self.processed_frames,
                'processing_time': total_time,
                'avg_frame_time': avg_processing_time,
                'fps_processed': fps_processed
            },
            'gait_analysis': {
                'total_distance_m': self.gait_tracker.total_distance_m,
                'trajectory_points': len(self.trajectory_3d),
                **trajectory_stats
            },
            'trajectory_3d': self.trajectory_3d  # Trayectoria completa
        }
        
        return results
    
    def _calculate_trajectory_stats(self) -> Dict:
        """
        Calcular estadísticas de la trayectoria 3D.
        
        Returns:
            Dict: Estadísticas de la trayectoria
        """
        if not self.trajectory_3d:
            return {'avg_speed_ms': 0, 'max_speed_ms': 0, 'speed_variance': 0}
        
        speeds = []
        for i in range(1, len(self.trajectory_3d)):
            prev_point = np.array(self.trajectory_3d[i-1]['point_3d'])
            curr_point = np.array(self.trajectory_3d[i]['point_3d'])
            
            distance = np.linalg.norm(curr_point - prev_point)
            time_diff = self.trajectory_3d[i]['timestamp'] - self.trajectory_3d[i-1]['timestamp']
            
            if time_diff > 0:
                speed = distance / time_diff  # m/s
                speeds.append(speed)
        
        if speeds:
            return {
                'avg_speed_ms': float(np.mean(speeds)),
                'max_speed_ms': float(np.max(speeds)),
                'speed_variance': float(np.var(speeds))
            }
        else:
            return {'avg_speed_ms': 0, 'max_speed_ms': 0, 'speed_variance': 0}


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description="Análisis de marcha 3D usando archivos .bag de Orbbec"
    )
    parser.add_argument('--bag', type=str, required=True,
                       help='Ruta del archivo .bag de entrada')
    parser.add_argument('--output', type=str, required=True,
                       help='Ruta del archivo de video de salida (.mp4)')
    parser.add_argument('--model', type=str, default=None,
                       help='Ruta al modelo TRT Pose (.pth o .engine)')
    parser.add_argument('--topology', type=str, default=None,
                       help='Ruta al archivo de topología JSON')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Número máximo de frames a procesar (para pruebas)')
    parser.add_argument('--json-output', type=str, default=None,
                       help='Ruta del archivo JSON de estadísticas (opcional)')
    
    args = parser.parse_args()
    
    # Verificar que el archivo .bag existe
    if not Path(args.bag).exists():
        logger.error(f"El archivo .bag no existe: {args.bag}")
        return
    
    try:
        # Crear analizador
        analyzer = BagFileGaitAnalyzer(
            model_path=args.model,
            topology_path=args.topology
        )
        
        # Procesar archivo .bag
        logger.info(f"Iniciando análisis de: {args.bag}")
        results = analyzer.process_bag_file(
            bag_path=args.bag,
            output_video_path=args.output,
            max_frames=args.max_frames
        )
        
        # Mostrar estadísticas finales
        print("\n" + "="*60)
        print("RESULTADOS DEL ANÁLISIS DE MARCHA")
        print("="*60)
        print(f"Frames procesados: {results['processing_info']['processed_frames']}")
        print(f"Tiempo total: {results['processing_info']['processing_time']:.1f}s")
        print(f"FPS de procesamiento: {results['processing_info']['fps_processed']:.1f}")
        print(f"Tiempo promedio por frame: {results['processing_info']['avg_frame_time']:.3f}s")
        print(f"\nDistancia total recorrida: {results['gait_analysis']['total_distance_m']:.3f} metros")
        print(f"Puntos de trayectoria: {results['gait_analysis']['trajectory_points']}")
        print(f"Velocidad promedio: {results['gait_analysis']['avg_speed_ms']:.3f} m/s")
        print(f"Velocidad máxima: {results['gait_analysis']['max_speed_ms']:.3f} m/s")
        print(f"\nVideo guardado en: {args.output}")
        
        # Guardar estadísticas en JSON si se especifica
        if args.json_output:
            with open(args.json_output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Estadísticas guardadas en: {args.json_output}")
        
    except KeyboardInterrupt:
        logger.info("Análisis interrumpido por el usuario")
    except Exception as e:
        logger.error(f"Error durante el análisis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
