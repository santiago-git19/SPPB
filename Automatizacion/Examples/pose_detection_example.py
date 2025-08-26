#!/usr/bin/env python3
"""
Ejemplo de Detección de Poses con Interfaz Unificada
====================================================

Este script demuestra el uso de la interfaz PoseDetection unificada para detectar
keypoints de poses usando diferentes backends (MediaPipe Tasks o TensorRT Pose).

Características:
- Interfaz unificada para cambiar entre diferentes modelos fácilmente
- Procesamiento en lotes (batch processing) configurable
- Control de FPS objetivo para procesamiento
- Soporte para video o cámara en tiempo real
- Visualización de keypoints en tiempo real
- Guardado opcional del video procesado

Uso:
    # MediaPipe con cámara
    python pose_detection_example.py --model mediapipe --batch_size 4 --fps 15

    # TensorRT con video
    python pose_detection_example.py --model trt_pose --batch_size 8 --fps 30 --video input.mp4 --output output.mp4

    # MediaPipe con video y guardado
    python pose_detection_example.py --model mediapipe --video input.mp4 --output result.mp4 --batch_size 2 --fps 10

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
from typing import List, Optional, Tuple
from collections import deque

# Añadir el directorio padre al sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Importar la interfaz y los procesadores
from utils.pose_detection.pose_detection import PoseDetection

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PoseDetectionRunner:
    """
    Clase principal para ejecutar la detección de poses con diferentes backends
    """
    
    def __init__(self, model_type: str, batch_size: int = 1, target_fps: float = 30.0,
                 video_path: Optional[str] = None, output_path: Optional[str] = None,
                 camera_id: int = 0, debug: bool = False):
        """
        Inicializa el runner de detección de poses
        
        Args:
            model_type: Tipo de modelo ('mediapipe' o 'trt_pose')
            batch_size: Número de frames a procesar a la vez
            target_fps: FPS objetivo para procesamiento
            video_path: Ruta del video a procesar (None para cámara)
            output_path: Ruta para guardar video procesado (opcional)
            camera_id: ID de la cámara a usar
            debug: Activar modo debug
        """
        self.model_type = model_type
        self.batch_size = batch_size
        self.target_fps = target_fps
        self.video_path = video_path
        self.output_path = output_path
        self.camera_id = camera_id
        self.debug = debug
        
        # Variables de control
        self.frame_buffer = deque(maxlen=batch_size)
        self.fps_counter = deque(maxlen=30)  # Para calcular FPS promedio
        self.frame_count = 0
        self.start_time = time.time()
        
        # Inicializar procesador de poses
        self.pose_processor = self._initialize_pose_processor()
        
        # Configurar captura de video
        self.cap = self._setup_video_capture()
        
        # Configurar writer de video si es necesario
        self.video_writer = self._setup_video_writer()
        
        logger.info(f"✅ PoseDetectionRunner inicializado")
        logger.info(f"   🔧 Modelo: {model_type}")
        logger.info(f"   📦 Batch size: {batch_size}")
        logger.info(f"   🎯 Target FPS: {target_fps}")
        logger.info(f"   📹 Fuente: {'Cámara' if video_path is None else video_path}")
    
    def _initialize_pose_processor(self) -> PoseDetection:
        """Inicializa el procesador de poses según el tipo especificado"""
        if self.model_type.lower() == 'mediapipe':
            from utils.pose_detection.mediapipe_pose_proc import MediaPipeTasksPoseProcessor
            # Configurar rutas para MediaPipe
            model_path = Path(__file__).parent.parent / "models" / "pose_landmarker_lite.task"
            return MediaPipeTasksPoseProcessor(
                model_path=str(model_path),
                confidence_threshold=0.5,
                debug=self.debug
            )
        
        elif self.model_type.lower() == 'trt_pose':
            from utils.pose_detection.trt_pose_proc import TRTPoseProcessor
            # Configurar rutas para TensorRT Pose
            model_path = Path(__file__).parent.parent / "models" / "resnet18_baseline_att_224x224_A_epoch_249.pth"
            topology_path = Path(__file__).parent.parent / "models" / "human_pose.json"
            return TRTPoseProcessor(
                model_path=str(model_path),
                topology_path=str(topology_path),
                confidence_threshold=0.3,
                debug=self.debug
            )
        
        else:
            raise ValueError(f"Modelo no soportado: {self.model_type}. Use 'mediapipe' o 'trt_pose'")
    
    def _setup_video_capture(self) -> cv2.VideoCapture:
        """Configura la captura de video desde archivo o cámara"""
        if self.video_path:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise ValueError(f"No se pudo abrir el video: {self.video_path}")
            logger.info(f"📹 Video cargado: {self.video_path}")
        else:
            cap = cv2.VideoCapture(self.camera_id)
            if not cap.isOpened():
                raise ValueError(f"No se pudo abrir la cámara: {self.camera_id}")
            logger.info(f"📷 Cámara iniciada: {self.camera_id}")
        
        return cap
    
    def _setup_video_writer(self) -> Optional[cv2.VideoWriter]:
        """Configura el writer de video si se especifica output_path"""
        if not self.output_path:
            return None
        
        # Obtener propiedades del video de entrada
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Configurar codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        writer = cv2.VideoWriter(self.output_path, fourcc, self.target_fps, (width, height))
        
        if writer.isOpened():
            logger.info(f"💾 Video de salida configurado: {self.output_path}")
        else:
            logger.warning(f"⚠️ No se pudo configurar el video de salida: {self.output_path}")
            return None
        
        return writer
    
    def _process_batch(self, frames: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """
        Procesa un lote de frames usando el procesador de poses
        
        Args:
            frames: Lista de frames a procesar
            
        Returns:
            Lista de keypoints detectados para cada frame
        """
        start_time = time.time()
        
        # Procesar frames usando la interfaz unificada
        results = self.pose_processor.process_frames(frames)
        
        process_time = time.time() - start_time
        
        if self.debug:
            logger.debug(f"🔄 Batch de {len(frames)} frames procesado en {process_time:.3f}s")
        
        return results
    
    def _visualize_frame(self, frame: np.ndarray, keypoints: Optional[np.ndarray]) -> np.ndarray:
        """
        Visualiza los keypoints en el frame
        
        Args:
            frame: Frame original
            keypoints: Keypoints detectados
            
        Returns:
            Frame con keypoints visualizados
        """
        if keypoints is None:
            return frame.copy()
        
        # Usar el método de visualización de la interfaz
        return self.pose_processor.visualize_keypoints(frame, keypoints)
    
    def _update_fps_counter(self):
        """Actualiza el contador de FPS"""
        current_time = time.time()
        self.fps_counter.append(current_time)
        
        # Calcular FPS promedio de los últimos frames
        if len(self.fps_counter) >= 2:
            fps = (len(self.fps_counter) - 1) / (self.fps_counter[-1] - self.fps_counter[0])
            return fps
        return 0.0
    
    def _draw_info_overlay(self, frame: np.ndarray, fps: float, batch_info: str) -> np.ndarray:
        """Dibuja información de overlay en el frame"""
        height, width = frame.shape[:2]
        
        # Información a mostrar
        info_lines = [
            f"Modelo: {self.model_type.upper()}",
            f"FPS: {fps:.1f}",
            f"Batch: {batch_info}",
            f"Target FPS: {self.target_fps}",
            f"Frame: {self.frame_count}"
        ]
        
        # Dibujar fondo semi-transparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 140), (0, 0, 0), -1)
        cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)
        
        # Dibujar texto
        y_offset = 30
        for line in info_lines:
            cv2.putText(frame, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
        
        return frame
    
    def run(self):
        """Ejecuta el bucle principal de procesamiento"""
        logger.info("🚀 Iniciando procesamiento...")
        
        try:
            while True:
                # Leer frame
                ret, frame = self.cap.read()
                if not ret:
                    if self.video_path:
                        logger.info("📹 Video terminado")
                        break
                    else:
                        logger.warning("⚠️ No se pudo leer frame de cámara")
                        continue
                
                # Añadir frame al buffer
                self.frame_buffer.append(frame.copy())
                self.frame_count += 1
                
                # Procesar cuando el buffer esté lleno o sea el último frame
                if len(self.frame_buffer) == self.batch_size:
                    # Procesar lote
                    batch_frames = list(self.frame_buffer)
                    results = self._process_batch(batch_frames)
                    
                    # Visualizar y mostrar cada frame del lote
                    for i, (batch_frame, keypoints) in enumerate(zip(batch_frames, results)):
                        # Visualizar keypoints
                        vis_frame = self._visualize_frame(batch_frame, keypoints)
                        
                        # Actualizar FPS y dibujar overlay
                        fps = self._update_fps_counter()
                        batch_info = f"{i+1}/{len(batch_frames)}"
                        vis_frame = self._draw_info_overlay(vis_frame, fps, batch_info)
                        
                        # Mostrar frame
                        cv2.imshow('Pose Detection', vis_frame)
                        
                        # Guardar frame si está configurado
                        if self.video_writer:
                            self.video_writer.write(vis_frame)
                        
                        # Control de FPS
                        if self.target_fps > 0:
                            time.sleep(1.0 / self.target_fps)
                        
                        # Verificar tecla de salida
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == 27:  # 'q' o ESC
                            logger.info("🛑 Detenido por el usuario")
                            return
                    
                    # Limpiar buffer
                    self.frame_buffer.clear()
        
        except KeyboardInterrupt:
            logger.info("🛑 Interrumpido por el usuario")
        
        except Exception as e:
            logger.error(f"❌ Error durante el procesamiento: {e}")
            raise
        
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Limpia recursos"""
        logger.info("🧹 Limpiando recursos...")
        
        if self.cap:
            self.cap.release()
        
        if self.video_writer:
            self.video_writer.release()
        
        cv2.destroyAllWindows()
        
        # Estadísticas finales
        total_time = time.time() - self.start_time
        avg_fps = self.frame_count / total_time if total_time > 0 else 0
        
        logger.info("📊 Estadísticas finales:")
        logger.info(f"   ⏱️ Tiempo total: {total_time:.2f}s")
        logger.info(f"   🖼️ Frames procesados: {self.frame_count}")
        logger.info(f"   📈 FPS promedio: {avg_fps:.2f}")


def parse_arguments() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(
        description="Detección de poses con interfaz unificada",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # MediaPipe con cámara
  python pose_detection_example.py --model mediapipe --batch_size 4 --fps 15

  # TensorRT con video
  python pose_detection_example.py --model trt_pose --batch_size 8 --fps 30 --video input.mp4 --output output.mp4

  # MediaPipe con video y guardado
  python pose_detection_example.py --model mediapipe --video input.mp4 --output result.mp4 --batch_size 2 --fps 10
        """
    )
    
    # Argumentos principales
    parser.add_argument('--model', type=str, choices=['mediapipe', 'trt_pose'], 
                       default='mediapipe', help='Tipo de modelo a usar (default: mediapipe)')
    
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Número de frames a procesar a la vez (default: 1)')
    
    parser.add_argument('--fps', type=float, default=30.0,
                       help='FPS objetivo para procesamiento (default: 30.0)')
    
    # Entrada y salida
    parser.add_argument('--video', type=str, default=None,
                       help='Ruta del video a procesar (usa cámara si no se especifica)')
    
    parser.add_argument('--output', type=str, default=None,
                       help='Ruta para guardar video procesado (opcional)')
    
    parser.add_argument('--camera', type=int, default=0,
                       help='ID de la cámara a usar (default: 0)')
    
    # Opciones adicionales
    parser.add_argument('--debug', action='store_true',
                       help='Activar modo debug con logging detallado')
    
    return parser.parse_args()


def main():
    """Función principal"""
    # Parsear argumentos
    args = parse_arguments()
    
    # Configurar nivel de logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validar argumentos
    if args.batch_size < 1:
        logger.error("❌ batch_size debe ser mayor que 0")
        return
    
    if args.fps <= 0:
        logger.error("❌ fps debe ser mayor que 0")
        return
    
    # Verificar que existe el video si se especifica
    if args.video and not Path(args.video).exists():
        logger.error(f"❌ El archivo de video no existe: {args.video}")
        return
    
    # Crear directorio de salida si es necesario
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Crear y ejecutar runner
        runner = PoseDetectionRunner(
            model_type=args.model,
            batch_size=args.batch_size,
            target_fps=args.fps,
            video_path=args.video,
            output_path=args.output,
            camera_id=args.camera,
            debug=args.debug
        )
        
        runner.run()
    
    except Exception as e:
        logger.error(f"❌ Error en la ejecución: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1
    
    logger.info("✅ Procesamiento completado exitosamente")
    return 0


if __name__ == "__main__":
    exit(main())
