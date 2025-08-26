#!/usr/bin/env python3
"""
Dual Orbbec Camera Manager
==========================

Clase para gestionar dos cámaras Orbbec Gemini 335Le usando el SDK oficial
de Orbbec (pyorbbecsdk), proporcionando captura sincronizada de frames.

Características:
- Detección automática de dos cámaras Gemini 335Le usando SDK nativo
- Captura sincronizada frame a frame con color y depth
- Manejo automático de desconexiones y reconexiones
- Integración con OpenCV y el pipeline existente de trt_pose

Autor: Sistema de IA
Fecha: 2025
"""

import cv2
import numpy as np
import time
import logging
import threading
from typing import Tuple, Optional, List, Dict
from pathlib import Path
import sys

# Configurar logging
logger = logging.getLogger(__name__)

# Importar SDK de Orbbec
try:
    import pyorbbecsdk as ob
    SDK_AVAILABLE = True
    logger.info("✅ SDK de Orbbec importado correctamente")
except ImportError as e:
    SDK_AVAILABLE = False
    logger.error("❌ SDK de Orbbec no disponible")
    logger.error("💡 Instalar con: pip install pyorbbecsdk")
    logger.error("💡 O compilar desde: https://github.com/orbbec/OrbbecSDK")

class DualOrbbecCapture:
    """
    Gestor de dos cámaras Orbbec Gemini 335Le para captura sincronizada usando SDK nativo
    
    Esta clase maneja la inicialización, captura sincronizada y reconexión
    automática de dos cámaras Orbbec usando el SDK oficial pyorbbecsdk.
    """
    
    def __init__(self, 
                 resolution: Tuple[int, int] = (640, 480),
                 fps: int = 30,
                 auto_reconnect: bool = True,
                 max_reconnect_attempts: int = 5,
                 reconnect_delay: float = 2.0,
                 enable_depth: bool = True):
        """
        Inicializa el gestor de cámaras duales
        
        Args:
            resolution: Resolución de captura (width, height)
            fps: Frames por segundo objetivo
            auto_reconnect: Activar reconexión automática
            max_reconnect_attempts: Máximo número de intentos de reconexión
            reconnect_delay: Tiempo entre intentos de reconexión (segundos)
            enable_depth: Habilitar captura de depth además de color
        """
        if not SDK_AVAILABLE:
            raise RuntimeError("SDK de Orbbec no disponible. Instalar pyorbbecsdk")
        
        self.resolution = resolution
        self.fps = fps
        self.auto_reconnect = auto_reconnect
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.enable_depth = enable_depth
        
        # Estado de las cámaras usando SDK de Orbbec
        self.context: Optional[ob.Context] = None
        self.left_pipeline: Optional[ob.Pipeline] = None
        self.right_pipeline: Optional[ob.Pipeline] = None
        self.left_device: Optional[ob.Device] = None
        self.right_device: Optional[ob.Device] = None
        
        # Control de sincronización
        self._sync_lock = threading.Lock()
        self._is_capturing = False
        
        # Estadísticas
        self.stats = {
            'frames_captured': 0,
            'sync_failures': 0,
            'reconnections': 0,
            'start_time': time.time()
        }
        
        logger.info("🎥 Inicializando DualOrbbecCapture con SDK...")
        self._initialize_cameras()
    
    def _discover_orbbec_cameras(self) -> List[ob.Device]:
        """Detecta cámaras Orbbec usando el SDK oficial"""
        try:
            logger.info("🎯 Buscando cámaras Orbbec usando SDK...")
            
            # Crear contexto Orbbec
            if not self.context:
                self.context = ob.Context()
            
            # Obtener lista de dispositivos
            device_list = self.context.query_devices()
            device_count = device_list.device_count()
            
            logger.info(f"📱 Dispositivos Orbbec encontrados: {device_count}")
            
            if device_count == 0:
                logger.error("❌ No se encontraron dispositivos Orbbec")
                return []
            
            devices = []
            for i in range(device_count):
                try:
                    device = device_list.get_device(i)
                    device_info = device.get_device_info()
                    
                    logger.info(f"   📷 Dispositivo {i}:")
                    logger.info(f"      Nombre: {device_info.name()}")
                    logger.info(f"      Serial: {device_info.serial_number()}")
                    logger.info(f"      PID: {device_info.pid()}")
                    logger.info(f"      VID: {device_info.vid()}")
                    
                    devices.append(device)
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error accediendo a dispositivo {i}: {e}")
                    continue
            
            if len(devices) < 2:
                logger.warning(f"⚠️ Solo se encontraron {len(devices)} cámaras, se necesitan 2")
            
            return devices
            
        except Exception as e:
            logger.error(f"❌ Error detectando cámaras Orbbec: {e}")
            return []
    
    def _initialize_cameras(self) -> bool:
        """
        Inicializa ambas cámaras usando el SDK de Orbbec
        
        Returns:
            True si ambas cámaras se inicializaron correctamente
        """
        try:
            # Descubrir cámaras disponibles
            available_devices = self._discover_orbbec_cameras()
            
            if len(available_devices) < 2:
                logger.error(f"❌ Se necesitan 2 cámaras Orbbec, encontradas: {len(available_devices)}")
                logger.error("💡 Verificar:")
                logger.error("   - Ambas cámaras están conectadas por USB")
                logger.error("   - Los drivers de Orbbec están instalados")
                logger.error("   - El SDK pyorbbecsdk está correctamente instalado")
                return False
            
            # Asignar dispositivos
            self.left_device = available_devices[0]
            self.right_device = available_devices[1]
            
            logger.info("📷 Asignando cámaras:")
            left_info = self.left_device.get_device_info()
            right_info = self.right_device.get_device_info()
            logger.info(f"   🔷 Cámara izquierda: {left_info.serial_number()}")
            logger.info(f"   🔶 Cámara derecha: {right_info.serial_number()}")
            
            # Inicializar pipelines
            if not self._initialize_pipeline('left'):
                return False
            
            if not self._initialize_pipeline('right'):
                self._release_single_camera('left')
                return False
            
            logger.info("✅ Ambas cámaras inicializadas correctamente")
            self._is_capturing = True
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error inicializando cámaras: {e}")
            return False
    
    def _initialize_pipeline(self, camera_side: str) -> bool:
        """Inicializa el pipeline de una cámara específica"""
        try:
            device = self.left_device if camera_side == 'left' else self.right_device
            
            logger.info(f"🔧 Inicializando pipeline de cámara {camera_side}...")
            
            # Crear pipeline
            pipeline = ob.Pipeline(device)
            config = ob.Config()
            
            # Configurar stream de color
            if not self._configure_color_stream(pipeline, config):
                return False
            
            # Configurar stream de depth si está habilitado
            if self.enable_depth:
                self._configure_depth_stream(pipeline, config)
            
            # Iniciar pipeline
            pipeline.start(config)
            
            # Asignar pipeline
            if camera_side == 'left':
                self.left_pipeline = pipeline
            else:
                self.right_pipeline = pipeline
            
            logger.info(f"   ✅ Pipeline de cámara {camera_side} iniciado")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error inicializando pipeline {camera_side}: {e}")
            return False
    
    def _configure_color_stream(self, pipeline, config) -> bool:
        """Configura el stream de color para el pipeline"""
        try:
            color_profiles = pipeline.get_stream_profile_list(ob.SENSOR_COLOR)
            if color_profiles.count() > 0:
                # Buscar perfil que coincida con la resolución deseada
                color_profile = self._find_best_profile(color_profiles, 'color')
                if color_profile:
                    config.enable_stream(color_profile)
                    logger.info(f"   ✅ Color stream: {color_profile.width()}x{color_profile.height()}@{color_profile.fps()}fps")
                    return True
                else:
                    logger.warning("   ⚠️ No se encontró perfil de color adecuado")
                    return False
            else:
                logger.error("   ❌ No hay perfiles de color disponibles")
                return False
        except Exception as e:
            logger.error(f"   ❌ Error configurando color stream: {e}")
            return False
    
    def _configure_depth_stream(self, pipeline, config):
        """Configura el stream de depth para el pipeline"""
        try:
            depth_profiles = pipeline.get_stream_profile_list(ob.SENSOR_DEPTH)
            if depth_profiles.count() > 0:
                depth_profile = self._find_best_profile(depth_profiles, 'depth')
                if depth_profile:
                    config.enable_stream(depth_profile)
                    logger.info(f"   ✅ Depth stream: {depth_profile.width()}x{depth_profile.height()}@{depth_profile.fps()}fps")
                else:
                    logger.warning("   ⚠️ No se encontró perfil de depth adecuado")
            else:
                logger.warning("   ⚠️ No hay perfiles de depth disponibles")
        except Exception as e:
            logger.warning(f"   ⚠️ Error configurando depth stream: {e}")
    
    def _find_best_profile(self, profiles, stream_type: str):
        """Encuentra el mejor perfil de stream basado en la resolución deseada"""
        try:
            width, height = self.resolution
            best_profile = None
            best_score = float('inf')
            
            for i in range(profiles.count()):
                try:
                    profile = profiles.get_video_stream_profile(i)
                    
                    # Calcular diferencia con resolución deseada
                    width_diff = abs(profile.width() - width)
                    height_diff = abs(profile.height() - height)
                    fps_diff = abs(profile.fps() - self.fps)
                    
                    # Score basado en diferencia (menor es mejor)
                    score = width_diff + height_diff + fps_diff * 0.1
                    
                    logger.debug(f"      Perfil {i}: {profile.width()}x{profile.height()}@{profile.fps()}fps (score: {score:.1f})")
                    
                    if score < best_score:
                        best_score = score
                        best_profile = profile
                        
                except Exception as e:
                    logger.debug(f"      Error evaluando perfil {i}: {e}")
                    continue
            
            if best_profile:
                logger.info(f"   🎯 Mejor perfil {stream_type}: {best_profile.width()}x{best_profile.height()}@{best_profile.fps()}fps")
            
            return best_profile
            
        except Exception as e:
            logger.error(f"Error buscando mejor perfil: {e}")
            return None
    
    def read_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Captura un par de frames sincronizados de ambas cámaras
        
        Returns:
            Tupla (frame_left, frame_right) o (None, None) si falla
        """
        if not self._is_capturing or not self.left_pipeline or not self.right_pipeline:
            if self.auto_reconnect and self._attempt_reconnection():
                return self.read_frames()
            return None, None
        
        with self._sync_lock:
            try:
                # Capturar framesets de ambas cámaras
                framesets = self._capture_framesets()
                if not framesets:
                    return self._handle_capture_failure()
                
                left_frameset, right_frameset = framesets
                
                # Extraer y convertir frames de color
                frames = self._extract_color_frames(left_frameset, right_frameset)
                if not frames:
                    return self._handle_capture_failure()
                
                left_image, right_image = frames
                
                # Actualizar estadísticas y retornar
                self.stats['frames_captured'] += 1
                logger.debug(f"📷 Frames capturados: L{left_image.shape} R{right_image.shape}")
                
                return left_image, right_image
                
            except Exception as e:
                logger.error(f"❌ Error en captura sincronizada: {e}")
                return self._handle_capture_failure()
    
    def read_frames_with_depth(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Captura frames de color y depth de ambas cámaras
        
        Returns:
            Tupla (left_color, right_color, left_depth, right_depth)
        """
        if not self.enable_depth:
            logger.warning("⚠️ Depth no está habilitado. Usa enable_depth=True")
            return None, None, None, None
        
        if not self._is_capturing or not self.left_pipeline or not self.right_pipeline:
            if self.auto_reconnect and self._attempt_reconnection():
                return self.read_frames_with_depth()
            return None, None, None, None
        
        with self._sync_lock:
            try:
                # Capturar framesets de ambas cámaras
                framesets = self._capture_framesets()
                if not framesets:
                    return None, None, None, None
                
                left_frameset, right_frameset = framesets
                
                # Extraer frames de color y depth
                left_color = self._frameset_to_numpy(left_frameset.get_color_frame())
                right_color = self._frameset_to_numpy(right_frameset.get_color_frame())
                left_depth = self._frameset_to_numpy(left_frameset.get_depth_frame())
                right_depth = self._frameset_to_numpy(right_frameset.get_depth_frame())
                
                if left_color is None or right_color is None or left_depth is None or right_depth is None:
                    return self._handle_capture_failure()
                
                # Actualizar estadísticas
                self.stats['frames_captured'] += 1
                logger.debug(f"📷 Frames con depth capturados: LC{left_color.shape} RC{right_color.shape} LD{left_depth.shape} RD{right_depth.shape}")
                
                return left_color, right_color, left_depth, right_depth
                
            except Exception as e:
                logger.error(f"❌ Error en captura con depth: {e}")
                return None, None, None, None
    
    def get_distance_at_point(self, depth_frame: np.ndarray, x: int, y: int) -> Optional[float]:
        """
        Obtiene la distancia en milímetros de un punto específico
        
        Args:
            depth_frame: Frame de profundidad
            x, y: Coordenadas del pixel
            
        Returns:
            Distancia en milímetros o None si no es válida
        """
        if depth_frame is None:
            return None
        
        h, w = depth_frame.shape
        if 0 <= x < w and 0 <= y < h:
            depth_value = depth_frame[y, x]
            # Valores de depth están en milímetros, 0 indica sin datos
            return float(depth_value) if depth_value > 0 else None
        
        return None
    
    def get_distance_in_region(self, depth_frame: np.ndarray, x: int, y: int, width: int, height: int) -> Optional[float]:
        """
        Calcula la distancia promedio en una región rectangular
        
        Args:
            depth_frame: Frame de profundidad
            x, y: Coordenadas de la esquina superior izquierda
            width, height: Dimensiones de la región
            
        Returns:
            Distancia promedio en milímetros o None si no hay datos válidos
        """
        if depth_frame is None:
            return None
        
        h, w = depth_frame.shape
        
        # Asegurar que la región esté dentro de los límites
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x + width)
        y2 = min(h, y + height)
        
        if x1 >= x2 or y1 >= y2:
            return None
        
        # Extraer región y filtrar valores válidos (> 0)
        region = depth_frame[y1:y2, x1:x2]
        valid_depths = region[region > 0]
        
        if len(valid_depths) > 0:
            return float(np.mean(valid_depths))
        
        return None
    
    def get_3d_coordinates(self, depth_frame: np.ndarray, x: int, y: int, camera_side: str = 'left') -> Optional[Tuple[float, float, float]]:
        """
        Convierte coordenadas 2D a coordenadas 3D reales usando parámetros intrínsecos
        
        Args:
            depth_frame: Frame de profundidad
            x, y: Coordenadas 2D del pixel
            camera_side: 'left' o 'right'
            
        Returns:
            Tupla (X, Y, Z) en milímetros o None si no es válido
        """
        distance = self.get_distance_at_point(depth_frame, x, y)
        if distance is None:
            return None
        
        try:
            # Obtener parámetros intrínsecos de la cámara correspondiente
            pipeline = self.left_pipeline if camera_side == 'left' else self.right_pipeline
            if pipeline is None:
                return None
            
            # Obtener perfil de depth stream
            depth_profile = pipeline.get_stream_profile_list(ob.SENSOR_DEPTH).get_profile(0)
            intrinsics = depth_profile.get_intrinsic()
            
            # Convertir coordenadas de imagen a coordenadas 3D reales
            # Fórmulas estándar de proyección de cámara
            Z = distance  # Profundidad en mm
            X = (x - intrinsics.cx) * Z / intrinsics.fx
            Y = (y - intrinsics.cy) * Z / intrinsics.fy
            
            return float(X), float(Y), float(Z)
            
        except Exception as e:
            logger.error(f"❌ Error calculando coordenadas 3D: {e}")
            return None
    
    def calculate_real_distance_between_points(self, depth_frame: np.ndarray, 
                                               x1: int, y1: int, x2: int, y2: int, 
                                               camera_side: str = 'left') -> Optional[float]:
        """
        Calcula la distancia real en 3D entre dos puntos
        
        Args:
            depth_frame: Frame de profundidad
            x1, y1: Primer punto
            x2, y2: Segundo punto
            camera_side: 'left' o 'right'
            
        Returns:
            Distancia en milímetros o None si no es válida
        """
        point1_3d = self.get_3d_coordinates(depth_frame, x1, y1, camera_side)
        point2_3d = self.get_3d_coordinates(depth_frame, x2, y2, camera_side)
        
        if point1_3d is None or point2_3d is None:
            return None
        
        # Calcular distancia euclidiana 3D
        dx = point1_3d[0] - point2_3d[0]
        dy = point1_3d[1] - point2_3d[1]
        dz = point1_3d[2] - point2_3d[2]
        
        return float(np.sqrt(dx*dx + dy*dy + dz*dz))
    
    def get_depth_statistics_in_region(self, depth_frame: np.ndarray, x: int, y: int, 
                                       width: int, height: int) -> Optional[dict]:
        """
        Obtiene estadísticas completas de profundidad en una región
        
        Args:
            depth_frame: Frame de profundidad
            x, y: Coordenadas de la región
            width, height: Dimensiones de la región
            
        Returns:
            Dict con estadísticas: mean, median, std, min, max, valid_pixels_percent
        """
        if depth_frame is None:
            return None
        
        h, w = depth_frame.shape
        
        # Asegurar región dentro de límites
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x + width)
        y2 = min(h, y + height)
        
        if x1 >= x2 or y1 >= y2:
            return None
        
        # Extraer región y calcular estadísticas
        region = depth_frame[y1:y2, x1:x2]
        valid_depths = region[region > 0]
        total_pixels = region.size
        
        if len(valid_depths) == 0:
            return {
                'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0,
                'valid_pixels_percent': 0, 'total_pixels': total_pixels
            }
        
        return {
            'mean': float(np.mean(valid_depths)),
            'median': float(np.median(valid_depths)),
            'std': float(np.std(valid_depths)),
            'min': float(np.min(valid_depths)),
            'max': float(np.max(valid_depths)),
            'valid_pixels_percent': (len(valid_depths) / total_pixels) * 100,
            'total_pixels': total_pixels,
            'valid_pixels': len(valid_depths)
        }
    
    def _capture_framesets(self):
        """Captura framesets de ambas cámaras con timeout"""
        start_time = time.time()
        
        # Capturar frameset de cámara izquierda
        left_frameset = self.left_pipeline.wait_for_frames(1000)  # 1 segundo timeout
        
        # Capturar frameset de cámara derecha
        right_frameset = self.right_pipeline.wait_for_frames(1000)
        
        capture_time = time.time() - start_time
        
        if not left_frameset or not right_frameset:
            logger.warning("⚠️ No se recibieron framesets de una o ambas cámaras")
            return None
        
        # Verificar tiempo de captura
        if capture_time > 0.1:  # 100ms es demasiado
            logger.debug(f"⚠️ Captura lenta: {capture_time*1000:.1f}ms")
        
        return left_frameset, right_frameset
    
    def _extract_color_frames(self, left_frameset, right_frameset):
        """Extrae y convierte frames de color de los framesets"""
        # Extraer frames de color
        left_color_frame = left_frameset.color_frame()
        right_color_frame = right_frameset.color_frame()
        
        if not left_color_frame or not right_color_frame:
            logger.warning("⚠️ No se recibieron frames de color de una o ambas cámaras")
            return None
        
        # Convertir a arrays de NumPy
        left_image = self._frame_to_numpy(left_color_frame)
        right_image = self._frame_to_numpy(right_color_frame)
        
        if left_image is None or right_image is None:
            logger.warning("⚠️ Error convirtiendo frames a NumPy")
            return None
        
        return left_image, right_image
    
    def _handle_capture_failure(self):
        """Maneja fallos de captura con posible reconexión"""
        self.stats['sync_failures'] += 1
        
        if self.auto_reconnect and self._attempt_reconnection():
            return self.read_frames()
        
        return None, None
    
    def _frame_to_numpy(self, frame) -> Optional[np.ndarray]:
        """Convierte un frame de Orbbec a array de NumPy"""
        try:
            # Obtener datos del frame
            frame_data = frame.data()
            width = frame.width()
            height = frame.height()
            
            # Convertir según el formato
            if frame.format() == ob.FORMAT_RGB:
                # RGB format
                img_array = np.frombuffer(frame_data, dtype=np.uint8)
                img_array = img_array.reshape((height, width, 3))
                # Convertir RGB a BGR para OpenCV
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            elif frame.format() == ob.FORMAT_BGR:
                # BGR format (ya compatible con OpenCV)
                img_array = np.frombuffer(frame_data, dtype=np.uint8)
                img_array = img_array.reshape((height, width, 3))
            elif frame.format() == ob.FORMAT_YUYV:
                # YUYV format
                img_array = np.frombuffer(frame_data, dtype=np.uint8)
                img_array = img_array.reshape((height, width, 2))
                # Convertir YUYV a BGR
                img_array = cv2.cvtColor(img_array, cv2.COLOR_YUV2BGR_YUYV)
            else:
                logger.warning(f"⚠️ Formato de frame no soportado: {frame.format()}")
                return None
            
            return img_array
            
        except Exception as e:
            logger.error(f"❌ Error convirtiendo frame: {e}")
            return None
    
    def _attempt_reconnection(self) -> bool:
        """
        Intenta reconectar las cámaras automáticamente
        
        Returns:
            True si la reconexión fue exitosa
        """
        if not self.auto_reconnect:
            return False
        
        logger.info("🔄 Intentando reconexión automática...")
        
        for attempt in range(self.max_reconnect_attempts):
            logger.info(f"   Intento {attempt + 1}/{self.max_reconnect_attempts}")
            
            # Liberar pipelines actuales
            self._release_cameras()
            
            # Esperar antes de reintentar
            time.sleep(self.reconnect_delay)
            
            # Intentar reinicializar
            if self._initialize_cameras():
                self.stats['reconnections'] += 1
                logger.info("✅ Reconexión exitosa")
                return True
            
            logger.warning(f"❌ Falló intento {attempt + 1}")
        
        logger.error("❌ Falló reconexión después de todos los intentos")
        self._is_capturing = False
        return False
    
    def _release_single_camera(self, camera_side: str):
        """Libera una cámara individual"""
        try:
            if camera_side == 'left' and self.left_pipeline:
                self.left_pipeline.stop()
                self.left_pipeline = None
            elif camera_side == 'right' and self.right_pipeline:
                self.right_pipeline.stop()
                self.right_pipeline = None
        except Exception as e:
            logger.debug(f"Error liberando cámara {camera_side}: {e}")
    
    def _release_cameras(self):
        """Libera ambas cámaras y el contexto"""
        self._is_capturing = False
        
        try:
            if self.left_pipeline:
                self.left_pipeline.stop()
                self.left_pipeline = None
            
            if self.right_pipeline:
                self.right_pipeline.stop()
                self.right_pipeline = None
                
            # No liberar el contexto aquí para permitir reconexión
            # self.context se libera solo en release() final
            
        except Exception as e:
            logger.debug(f"Error liberando cámaras: {e}")
    
    def is_opened(self) -> bool:
        """
        Verifica si ambas cámaras están abiertas y funcionando
        
        Returns:
            True si ambas cámaras están operativas
        """
        return (self._is_capturing and
                self.left_pipeline is not None and 
                self.right_pipeline is not None)
    
    def get_statistics(self) -> Dict:
        """
        Obtiene estadísticas de funcionamiento
        
        Returns:
            Diccionario con estadísticas de captura
        """
        elapsed_time = time.time() - self.stats['start_time']
        fps = self.stats['frames_captured'] / elapsed_time if elapsed_time > 0 else 0
        
        return {
            'frames_captured': self.stats['frames_captured'],
            'sync_failures': self.stats['sync_failures'],
            'reconnections': self.stats['reconnections'],
            'success_rate': (self.stats['frames_captured'] / 
                           (self.stats['frames_captured'] + self.stats['sync_failures'])
                           if (self.stats['frames_captured'] + self.stats['sync_failures']) > 0 else 0),
            'fps': fps,
            'elapsed_time': elapsed_time,
            'is_capturing': self._is_capturing
        }
    
    def release(self):
        """
        Libera todas las cámaras y recursos
        """
        logger.info("🔒 Liberando cámaras...")
        
        self._release_cameras()
        
        # Liberar contexto
        if self.context:
            self.context = None
        
        # Mostrar estadísticas finales
        stats = self.get_statistics()
        logger.info("📊 Estadísticas finales:")
        logger.info(f"   📷 Frames capturados: {stats['frames_captured']}")
        logger.info(f"   ❌ Fallos de sincronización: {stats['sync_failures']}")
        logger.info(f"   🔄 Reconexiones: {stats['reconnections']}")
        logger.info(f"   ✅ Tasa de éxito: {stats['success_rate']:.2%}")
        logger.info(f"   📈 FPS promedio: {stats['fps']:.1f}")
        
        logger.info("✅ Cámaras liberadas correctamente")
    
    def __enter__(self):
        """Soporte para context manager (with statement)"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Liberación automática al salir del context manager"""
        self.release()
    
    def __del__(self):
        """Destructor que asegura la liberación de recursos"""
        try:
            self.release()
        except Exception:
            pass  # Evitar errores durante la destrucción


# Función de utilidad para testing rápido
def test_dual_cameras():
    """
    Función de prueba para verificar el funcionamiento de las cámaras duales
    """
    logger.info("🧪 Iniciando prueba de cámaras duales...")
    
    try:
        with DualOrbbecCapture(resolution=(640, 480), fps=30) as dual_cam:
            if not dual_cam.is_opened():
                logger.error("❌ No se pudieron inicializar las cámaras")
                return False
            
            logger.info("✅ Cámaras inicializadas, iniciando captura de prueba...")
            
            for i in range(30):  # Capturar 30 frames de prueba
                left, right = dual_cam.read_frames()
                
                if left is not None and right is not None:
                    logger.info(f"Frame {i+1}: L{left.shape} R{right.shape}")
                    
                    # Opcional: mostrar frames (comentar si no hay display)
                    # cv2.imshow('Left Camera', left)
                    # cv2.imshow('Right Camera', right)
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     break
                else:
                    logger.warning(f"Frame {i+1}: Falló captura")
                
                time.sleep(0.1)  # 10 FPS para prueba
            
            # Mostrar estadísticas
            stats = dual_cam.get_statistics()
            logger.info("📊 Estadísticas de prueba:")
            for key, value in stats.items():
                logger.info(f"   {key}: {value}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Error en prueba: {e}")
        return False


def test_distance_measurement():
    """
    Función de prueba para demostrar las capacidades de medición de distancia
    """
    logger.info("📏 Iniciando prueba de medición de distancia...")
    
    try:
        # Inicializar con depth habilitado
        with DualOrbbecCapture(resolution=(640, 480), fps=30, enable_depth=True) as dual_cam:
            if not dual_cam.is_opened():
                logger.error("❌ No se pudieron inicializar las cámaras")
                return False
            
            logger.info("✅ Cámaras inicializadas con depth, capturando frames...")
            
            for i in range(10):
                left_color, right_color, left_depth, right_depth = dual_cam.read_frames_with_depth()
                
                if all(frame is not None for frame in [left_color, right_color, left_depth, right_depth]):
                    logger.info(f"📷 Frame {i+1} capturado con depth")
                    
                    # Medir distancia en el centro de la imagen
                    center_x, center_y = left_color.shape[1] // 2, left_color.shape[0] // 2
                    distance = dual_cam.get_distance_at_point(left_depth, center_x, center_y)
                    
                    if distance:
                        logger.info(f"   📏 Distancia en centro: {distance:.1f}mm ({distance/10:.1f}cm)")
                        
                        # Calcular coordenadas 3D del punto central
                        coords_3d = dual_cam.get_3d_coordinates(left_depth, center_x, center_y, 'left')
                        if coords_3d:
                            logger.info(f"   🎯 Coordenadas 3D: X={coords_3d[0]:.1f}mm, Y={coords_3d[1]:.1f}mm, Z={coords_3d[2]:.1f}mm")
                        
                        # Estadísticas en región central (100x100 pixels)
                        stats = dual_cam.get_depth_statistics_in_region(left_depth, center_x-50, center_y-50, 100, 100)
                        if stats:
                            logger.info(f"   📊 Región central (100x100):")
                            logger.info(f"      Distancia promedio: {stats['mean']:.1f}mm")
                            logger.info(f"      Rango: {stats['min']:.1f}-{stats['max']:.1f}mm")
                            logger.info(f"      Pixels válidos: {stats['valid_pixels_percent']:.1f}%")
                        
                        # Calcular distancia entre dos puntos
                        corner_x, corner_y = center_x + 100, center_y + 100
                        if corner_x < left_color.shape[1] and corner_y < left_color.shape[0]:
                            real_distance = dual_cam.calculate_real_distance_between_points(
                                left_depth, center_x, center_y, corner_x, corner_y, 'left')
                            if real_distance:
                                logger.info(f"   📐 Distancia real entre puntos: {real_distance:.1f}mm")
                    
                    else:
                        logger.warning(f"   ⚠️ No se pudo medir distancia en centro (sin datos depth)")
                
                else:
                    logger.warning(f"Frame {i+1}: Falló captura con depth")
                
                time.sleep(0.2)
            
            logger.info("✅ Prueba de distancia completada")
            return True
            
    except Exception as e:
        logger.error(f"❌ Error en prueba de medición de distancia: {e}")
        return False


if __name__ == "__main__":
    # Configurar logging para pruebas
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Ejecutar pruebas
    logger.info("🚀 Iniciando pruebas de DualOrbbecCapture...")
    
    # Prueba básica de captura dual
    if test_dual_cameras():
        logger.info("✅ Prueba de captura dual exitosa")
    else:
        logger.error("❌ Prueba de captura dual falló")
    
    # Prueba de medición de distancia
    if test_distance_measurement():
        logger.info("✅ Prueba de medición de distancia exitosa")
    else:
        logger.error("❌ Prueba de medición de distancia falló")
