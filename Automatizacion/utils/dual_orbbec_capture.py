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


if __name__ == "__main__":
    # Configurar logging para pruebas
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Ejecutar prueba
    success = test_dual_cameras()
    print(f"Prueba {'exitosa' if success else 'fallida'}")
