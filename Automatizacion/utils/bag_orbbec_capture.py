"""
BagOrbbecCapture - Reproductor de archivos .bag de Orbbec
=========================================================

Esta clase reproduce archivos .bag generados por OrbbecViewer,
proporcionando una interfaz similar a DualOrbbecCapture pero
para datos pregrabados en lugar de cámaras en vivo.

Funcionalidades:
- Reproducción de frames de color y profundidad desde archivos .bag
- Control de reproducción (play/pause/seek)
- Conversión a formato OpenCV/NumPy
- Medición de distancia y coordenadas 3D
- Estadísticas de reproducción
- Interfaz compatible con DualOrbbecCapture

Uso:
    bag_capture = BagOrbbecCapture("archivo.bag")
    
    # Leer frame de color
    color_frame = bag_capture.read_frame()
    
    # Leer frame de color y profundidad
    color_frame, depth_frame = bag_capture.read_frame_with_depth()
    
    # Medición de distancia
    distance = bag_capture.get_distance_at_point(depth_frame, x, y)
"""

import logging
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import numpy as np

try:
    from pyorbbecsdk import Pipeline, Config, FrameSet, Frame
    ORBBEC_AVAILABLE = True
except ImportError:
    ORBBEC_AVAILABLE = False
    Pipeline = None
    Config = None
    FrameSet = None
    Frame = None

logger = logging.getLogger(__name__)


class BagOrbbecCapture:
    """
    Reproductor de archivos .bag de Orbbec que proporciona una interfaz
    similar a DualOrbbecCapture para datos pregrabados.
    """
    
    def __init__(
        self,
        bag_path: str,
        auto_loop: bool = False,
        enable_depth: bool = True,
        target_fps: float = 30.0
    ):
        """
        Inicializar el reproductor de archivos .bag.
        
        Args:
            bag_path: Ruta del archivo .bag
            auto_loop: Si True, reinicia automáticamente al llegar al final
            enable_depth: Habilitar captura de frames de profundidad
            target_fps: FPS objetivo para reproducción
        """
        if not ORBBEC_AVAILABLE:
            raise ImportError("pyorbbecsdk no está disponible")
        
        self.bag_path = Path(bag_path)
        if not self.bag_path.exists():
            raise FileNotFoundError(f"Archivo .bag no encontrado: {bag_path}")
        
        self.auto_loop = auto_loop
        self.enable_depth = enable_depth
        self.target_fps = target_fps
        
        # Pipeline de Orbbec
        self.pipeline = None
        self.config = None
        
        # Estado de reproducción
        self.is_playing = False
        self.current_frame_number = 0
        self.total_frames = 0
        
        # Estadísticas
        self.stats = {
            'frames_read': 0,
            'depth_frames_read': 0,
            'failed_reads': 0,
            'start_time': None,
            'last_frame_time': None,
            'fps_actual': 0.0
        }
        
        # Parámetros de calibración (valores típicos para Orbbec Gemini 335Le)
        self.camera_params = {
            'fx': 570.3,  # Focal length X
            'fy': 570.3,  # Focal length Y  
            'cx': 320.0,  # Centro óptico X
            'cy': 240.0,  # Centro óptico Y
            'depth_scale': 1.0  # Escala de profundidad (mm)
        }
        
        # Inicializar pipeline
        self._initialize_pipeline()
        
        logger.info(f"BagOrbbecCapture inicializado: {self.bag_path}")
    
    def _initialize_pipeline(self) -> bool:
        """
        Inicializar el pipeline de Orbbec para reproducir el archivo .bag.
        
        Returns:
            bool: True si la inicialización fue exitosa
        """
        try:
            self.pipeline = Pipeline()
            self.config = Config()
            
            # Configurar para leer desde archivo .bag
            self.config.enable_device_from_file(str(self.bag_path))
            
            # Habilitar stream de color
            self.config.enable_stream_profile(
                stream_type="color",
                format="RGB888",
                width=640,
                height=480,
                fps=30
            )
            
            # Habilitar stream de profundidad si está habilitado
            if self.enable_depth:
                self.config.enable_stream_profile(
                    stream_type="depth",
                    format="Y16",
                    width=640,
                    height=480,
                    fps=30
                )
            
            # Iniciar el pipeline
            self.pipeline.start(self.config)
            self.is_playing = True
            self.stats['start_time'] = time.time()
            
            logger.info("Pipeline inicializado correctamente")
            return True
            
        except Exception as e:
            logger.error(f"Error inicializando pipeline: {e}")
            return False
    
    def read_frame(self, timeout_ms: int = 1000) -> Optional[np.ndarray]:
        """
        Leer el siguiente frame de color.
        
        Args:
            timeout_ms: Timeout en milisegundos para esperar el frame
            
        Returns:
            np.ndarray: Frame de color en formato BGR (OpenCV), o None si falla
        """
        if not self.is_playing:
            return None
        
        try:
            # Esperar por el siguiente frameset
            frameset = self.pipeline.wait_for_frames(timeout_ms)
            if not frameset:
                if self.auto_loop:
                    self._restart_playback()
                    return self.read_frame(timeout_ms)
                return None
            
            # Obtener frame de color
            color_frame = frameset.get_color_frame()
            if not color_frame:
                self.stats['failed_reads'] += 1
                return None
            
            # Convertir a formato OpenCV
            color_array = self._frame_to_opencv(color_frame)
            
            # Actualizar estadísticas
            self._update_stats()
            
            return color_array
            
        except Exception as e:
            logger.error(f"Error leyendo frame: {e}")
            self.stats['failed_reads'] += 1
            return None
    
    def read_frame_with_depth(self, timeout_ms: int = 1000) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Leer el siguiente frame de color y profundidad.
        
        Args:
            timeout_ms: Timeout en milisegundos para esperar los frames
            
        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]: (color_frame, depth_frame)
        """
        if not self.is_playing:
            return None, None
        
        try:
            # Esperar por el siguiente frameset
            frameset = self.pipeline.wait_for_frames(timeout_ms)
            if not frameset:
                if self.auto_loop:
                    self._restart_playback()
                    return self.read_frame_with_depth(timeout_ms)
                return None, None
            
            # Obtener frames
            color_frame = frameset.get_color_frame()
            depth_frame = frameset.get_depth_frame() if self.enable_depth else None
            
            color_array = None
            depth_array = None
            
            # Convertir frame de color
            if color_frame:
                color_array = self._frame_to_opencv(color_frame)
            
            # Convertir frame de profundidad
            if depth_frame and self.enable_depth:
                depth_array = self._depth_frame_to_array(depth_frame)
                self.stats['depth_frames_read'] += 1
            
            if color_array is None and depth_array is None:
                self.stats['failed_reads'] += 1
                return None, None
            
            # Actualizar estadísticas
            self._update_stats()
            
            return color_array, depth_array
            
        except Exception as e:
            logger.error(f"Error leyendo frames: {e}")
            self.stats['failed_reads'] += 1
            return None, None
    
    def _frame_to_opencv(self, frame: Frame) -> Optional[np.ndarray]:
        """
        Convertir frame de Orbbec a formato OpenCV.
        
        Args:
            frame: Frame de Orbbec
            
        Returns:
            np.ndarray: Array en formato BGR para OpenCV
        """
        try:
            # Obtener datos del frame
            data = frame.get_data()
            width = frame.get_width()
            height = frame.get_height()
            
            # Convertir a array numpy
            frame_array = np.frombuffer(data, dtype=np.uint8)
            frame_array = frame_array.reshape((height, width, 3))
            
            # Convertir de RGB a BGR para OpenCV
            frame_bgr = frame_array[:, :, ::-1].copy()
            
            return frame_bgr
            
        except Exception as e:
            logger.error(f"Error convirtiendo frame de color: {e}")
            return None
    
    def _depth_frame_to_array(self, depth_frame: Frame) -> Optional[np.ndarray]:
        """
        Convertir frame de profundidad de Orbbec a array numpy.
        
        Args:
            depth_frame: Frame de profundidad de Orbbec
            
        Returns:
            np.ndarray: Array de profundidad en mm
        """
        try:
            # Obtener datos del frame
            data = depth_frame.get_data()
            width = depth_frame.get_width()
            height = depth_frame.get_height()
            
            # Convertir a array numpy (16-bit depth)
            depth_array = np.frombuffer(data, dtype=np.uint16)
            depth_array = depth_array.reshape((height, width))
            
            return depth_array
            
        except Exception as e:
            logger.error(f"Error convirtiendo frame de profundidad: {e}")
            return None
    
    def get_distance_at_point(self, depth_frame: np.ndarray, x: int, y: int) -> Optional[float]:
        """
        Obtener la distancia en milímetros en un punto específico.
        
        Args:
            depth_frame: Frame de profundidad
            x: Coordenada X del pixel
            y: Coordenada Y del pixel
            
        Returns:
            float: Distancia en milímetros, o None si no es válida
        """
        try:
            if depth_frame is None:
                return None
            
            height, width = depth_frame.shape
            if not (0 <= x < width and 0 <= y < height):
                return None
            
            depth_value = depth_frame[y, x]
            if depth_value == 0:
                return None
            
            # Convertir a milímetros (asumiendo que ya está en mm)
            distance_mm = float(depth_value * self.camera_params['depth_scale'])
            
            return distance_mm
            
        except Exception as e:
            logger.error(f"Error obteniendo distancia: {e}")
            return None
    
    def get_3d_coordinates(self, depth_frame: np.ndarray, x: int, y: int) -> Optional[Tuple[float, float, float]]:
        """
        Convertir coordenadas 2D + profundidad a coordenadas 3D reales.
        
        Args:
            depth_frame: Frame de profundidad
            x: Coordenada X del pixel
            y: Coordenada Y del pixel
            
        Returns:
            Tuple[float, float, float]: Coordenadas 3D (X, Y, Z) en mm, o None si no es válido
        """
        try:
            # Obtener distancia en Z
            z_mm = self.get_distance_at_point(depth_frame, x, y)
            if z_mm is None or z_mm <= 0:
                return None
            
            # Convertir a coordenadas 3D usando parámetros intrínsecos
            fx = self.camera_params['fx']
            fy = self.camera_params['fy']
            cx = self.camera_params['cx']
            cy = self.camera_params['cy']
            
            x_3d = (x - cx) * z_mm / fx
            y_3d = (y - cy) * z_mm / fy
            z_3d = z_mm
            
            return (x_3d, y_3d, z_3d)
            
        except Exception as e:
            logger.error(f"Error calculando coordenadas 3D: {e}")
            return None
    
    def get_depth_statistics_in_region(
        self, 
        depth_frame: np.ndarray, 
        x: int, 
        y: int, 
        width: int, 
        height: int
    ) -> Optional[Dict[str, float]]:
        """
        Obtener estadísticas de profundidad en una región rectangular.
        
        Args:
            depth_frame: Frame de profundidad
            x: Coordenada X de la esquina superior izquierda
            y: Coordenada Y de la esquina superior izquierda
            width: Ancho de la región
            height: Alto de la región
            
        Returns:
            Dict: Estadísticas de la región (mean, median, std, min, max, valid_pixels_percent)
        """
        try:
            if depth_frame is None:
                return None
            
            frame_height, frame_width = depth_frame.shape
            
            # Asegurar que la región esté dentro del frame
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(frame_width, x + width)
            y2 = min(frame_height, y + height)
            
            if x1 >= x2 or y1 >= y2:
                return None
            
            # Extraer región
            region = depth_frame[y1:y2, x1:x2]
            valid_pixels = region[region > 0]
            
            if len(valid_pixels) == 0:
                return {
                    'mean': 0.0, 'median': 0.0, 'std': 0.0,
                    'min': 0.0, 'max': 0.0, 'valid_pixels_percent': 0.0
                }
            
            total_pixels = region.size
            valid_percent = (len(valid_pixels) / total_pixels) * 100.0
            
            stats = {
                'mean': float(np.mean(valid_pixels)),
                'median': float(np.median(valid_pixels)),
                'std': float(np.std(valid_pixels)),
                'min': float(np.min(valid_pixels)),
                'max': float(np.max(valid_pixels)),
                'valid_pixels_percent': valid_percent
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculando estadísticas de región: {e}")
            return None
    
    def _update_stats(self):
        """Actualizar estadísticas de reproducción."""
        current_time = time.time()
        self.stats['frames_read'] += 1
        self.current_frame_number += 1
        
        if self.stats['last_frame_time'] is not None:
            frame_interval = current_time - self.stats['last_frame_time']
            if frame_interval > 0:
                instant_fps = 1.0 / frame_interval
                # FPS suavizado
                if self.stats['fps_actual'] == 0:
                    self.stats['fps_actual'] = instant_fps
                else:
                    alpha = 0.1
                    self.stats['fps_actual'] = alpha * instant_fps + (1 - alpha) * self.stats['fps_actual']
        
        self.stats['last_frame_time'] = current_time
    
    def _restart_playback(self):
        """Reiniciar la reproducción desde el principio."""
        try:
            if self.pipeline:
                self.pipeline.stop()
            
            self.current_frame_number = 0
            self._initialize_pipeline()
            
            logger.info("Reproducción reiniciada")
            
        except Exception as e:
            logger.error(f"Error reiniciando reproducción: {e}")
    
    def pause(self):
        """Pausar la reproducción."""
        self.is_playing = False
        logger.info("Reproducción pausada")
    
    def resume(self):
        """Reanudar la reproducción."""
        self.is_playing = True
        logger.info("Reproducción reanudada")
    
    def stop(self):
        """Detener la reproducción."""
        self.is_playing = False
        if self.pipeline:
            try:
                self.pipeline.stop()
            except:
                pass
        logger.info("Reproducción detenida")
    
    def set_camera_parameters(self, fx: float, fy: float, cx: float, cy: float, depth_scale: float = 1.0):
        """
        Configurar parámetros intrínsecos de la cámara.
        
        Args:
            fx: Focal length en X
            fy: Focal length en Y
            cx: Centro óptico en X
            cy: Centro óptico en Y
            depth_scale: Factor de escala de profundidad
        """
        self.camera_params.update({
            'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy, 'depth_scale': depth_scale
        })
        logger.info(f"Parámetros de cámara actualizados: {self.camera_params}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Obtener estadísticas de reproducción.
        
        Returns:
            Dict: Estadísticas de reproducción
        """
        current_time = time.time()
        elapsed_time = current_time - self.stats['start_time'] if self.stats['start_time'] else 0
        
        stats = self.stats.copy()
        stats.update({
            'current_frame': self.current_frame_number,
            'elapsed_time': elapsed_time,
            'is_playing': self.is_playing,
            'bag_path': str(self.bag_path),
            'enable_depth': self.enable_depth,
            'success_rate': (self.stats['frames_read'] / max(1, self.stats['frames_read'] + self.stats['failed_reads'])) * 100
        })
        
        return stats
    
    def release(self):
        """Liberar recursos."""
        try:
            self.stop()
            self.pipeline = None
            self.config = None
            logger.info("Recursos liberados")
            
        except Exception as e:
            logger.error(f"Error liberando recursos: {e}")
    
    def __enter__(self):
        """Soporte para context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Soporte para context manager."""
        self.release()


# Función de conveniencia para crear una instancia
def create_bag_capture(bag_path: str, **kwargs) -> BagOrbbecCapture:
    """
    Crear una instancia de BagOrbbecCapture con parámetros por defecto.
    
    Args:
        bag_path: Ruta del archivo .bag
        **kwargs: Argumentos adicionales para BagOrbbecCapture
        
    Returns:
        BagOrbbecCapture: Instancia configurada
    """
    return BagOrbbecCapture(bag_path, **kwargs)


if __name__ == "__main__":
    # Ejemplo de uso
    bag_path = "../Videos/Entrada/ejemplo.bag"
    
    try:
        with BagOrbbecCapture(bag_path, auto_loop=True, enable_depth=True) as bag_capture:
            # Leer algunos frames
            for i in range(10):
                color_frame, depth_frame = bag_capture.read_frame_with_depth()
                
                if color_frame is not None:
                    print(f"Frame {i+1}: Color {color_frame.shape}")
                    
                    if depth_frame is not None:
                        # Medir distancia en el centro
                        center_x, center_y = color_frame.shape[1]//2, color_frame.shape[0]//2
                        distance = bag_capture.get_distance_at_point(depth_frame, center_x, center_y)
                        print(f"  Distancia en centro: {distance} mm")
                        
                        # Obtener coordenadas 3D
                        coords_3d = bag_capture.get_3d_coordinates(depth_frame, center_x, center_y)
                        if coords_3d:
                            print(f"  Coordenadas 3D: {coords_3d}")
                else:
                    print(f"Frame {i+1}: Error leyendo")
                
                time.sleep(1/30)  # Simular 30 FPS
            
            # Mostrar estadísticas
            stats = bag_capture.get_statistics()
            print(f"\nEstadísticas finales:")
            print(f"  Frames leídos: {stats['frames_read']}")
            print(f"  FPS actual: {stats['fps_actual']:.1f}")
            print(f"  Tasa de éxito: {stats['success_rate']:.1f}%")
            
    except Exception as e:
        print(f"Error: {e}")
