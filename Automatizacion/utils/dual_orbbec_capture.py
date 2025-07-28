#!/usr/bin/env python3
"""
Dual Orbbec Camera Manager
==========================

Clase para gestionar dos cámaras Orbbec Gemini 335Le conectadas a Jetson Nano
a través de un switch USB, proporcionando captura sincronizada de frames.

Características:
- Detección automática de dos cámaras Gemini 335Le
- Captura sincronizada frame a frame
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

class DualOrbbecCapture:
    """
    Gestor de dos cámaras Orbbec Gemini 335Le para captura sincronizada
    
    Esta clase maneja la inicialización, captura sincronizada y reconexión
    automática de dos cámaras Orbbec conectadas vía switch USB.
    """
    
    def __init__(self, 
                 resolution: Tuple[int, int] = (640, 480),
                 fps: int = 30,
                 auto_reconnect: bool = True,
                 max_reconnect_attempts: int = 5,
                 reconnect_delay: float = 2.0):
        """
        Inicializa el gestor de cámaras duales
        
        Args:
            resolution: Resolución de captura (width, height)
            fps: Frames por segundo objetivo
            auto_reconnect: Activar reconexión automática
            max_reconnect_attempts: Máximo número de intentos de reconexión
            reconnect_delay: Tiempo entre intentos de reconexión (segundos)
        """
        self.resolution = resolution
        self.fps = fps
        self.auto_reconnect = auto_reconnect
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay
        
        # Estado de las cámaras
        self.left_camera: Optional[cv2.VideoCapture] = None
        self.right_camera: Optional[cv2.VideoCapture] = None
        self.left_device_id: Optional[int] = None
        self.right_device_id: Optional[int] = None
        
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
        
        logger.info("🎥 Inicializando DualOrbbecCapture...")
        self._initialize_cameras()
    
    def _discover_orbbec_cameras(self) -> List[int]:
        """Detección usando SDK oficial de Orbbec"""
        try:
            # Importar SDK de Orbbec si está disponible
            from pyorbbecsdk import Context, DeviceList
            
            orbbec_devices = []
            
            # Crear contexto Orbbec
            ctx = Context()
            device_list = ctx.query_devices()
            
            logger.info(f"🔍 SDK Orbbec encontró {device_list.device_count()} dispositivos")
            
            for i in range(device_list.device_count()):
                device = device_list.get_device(i)
                device_info = device.get_device_info()
                
                logger.info(f"   📷 Dispositivo {i}: {device_info.name} "
                        f"(Serial: {device_info.serial_number})")
                
                # Mapear dispositivo SDK a /dev/video*
                video_device_id = self._map_orbbec_to_video_device(device_info)
                if video_device_id is not None:
                    orbbec_devices.append(video_device_id)
            
            return orbbec_devices
        except Exception as e:
            logger.warning(f"⚠️ Error con SDK Orbbec: {e}")
            
    
    def _initialize_cameras(self) -> bool:
        """
        Inicializa ambas cámaras con configuración optimizada
        
        Returns:
            True si ambas cámaras se inicializaron correctamente
        """
        # Descubrir cámaras disponibles
        available_cameras = self._discover_orbbec_cameras()
        
        if len(available_cameras) < 2:
            logger.error(f"❌ Se necesitan 2 cámaras Orbbec, encontradas: {len(available_cameras)}")
            logger.error("💡 Verificar:")
            logger.error("   - Ambas cámaras están conectadas al switch USB")
            logger.error("   - El switch USB tiene alimentación suficiente")
            logger.error("   - Los drivers de Orbbec están instalados")
            return False
        
        # Asignar dispositivos (el primero como izquierda, segundo como derecha)
        self.left_device_id = available_cameras[0]
        self.right_device_id = available_cameras[1]
        
        logger.info(f"📷 Asignando cámaras:")
        logger.info(f"   🔷 Cámara izquierda: dispositivo {self.left_device_id}")
        logger.info(f"   🔶 Cámara derecha: dispositivo {self.right_device_id}")
        
        # Inicializar cámara izquierda
        success = self._initialize_single_camera('left')
        if not success:
            return False
        
        # Inicializar cámara derecha
        success = self._initialize_single_camera('right')
        if not success:
            self._release_single_camera('left')
            return False
        
        logger.info("✅ Ambas cámaras inicializadas correctamente")
        self._is_capturing = True
        
        return True
    
    def _initialize_single_camera(self, camera_side: str) -> bool:
        """
        Inicializa una cámara individual con configuración optimizada
        
        Args:
            camera_side: 'left' o 'right'
            
        Returns:
            True si la cámara se inicializó correctamente
        """
        device_id = self.left_device_id if camera_side == 'left' else self.right_device_id
        
        try:
            logger.info(f"🔧 Inicializando cámara {camera_side} (dispositivo {device_id})...")
            
            # Crear captura con backend preferido para Jetson
            cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)  # V4L2 es mejor en Linux/Jetson
            
            if not cap.isOpened():
                logger.error(f"❌ No se pudo abrir cámara {camera_side}")
                return False
            
            # Configurar propiedades de captura
            width, height = self.resolution
            
            # Configuración optimizada para Orbbec Gemini 335Le
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Configuraciones adicionales para mejor rendimiento
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer mínimo para reducir latencia
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Auto exposición
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Auto enfoque si está disponible
            
            # Verificar configuración aplicada
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"   📐 Resolución aplicada: {actual_width}x{actual_height}")
            logger.info(f"   📈 FPS aplicado: {actual_fps:.1f}")
            
            # Probar captura
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.error(f"❌ No se pudo capturar frame de prueba de cámara {camera_side}")
                cap.release()
                return False
            
            # Asignar cámara
            if camera_side == 'left':
                self.left_camera = cap
            else:
                self.right_camera = cap
            
            logger.info(f"✅ Cámara {camera_side} inicializada: {frame.shape}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error inicializando cámara {camera_side}: {e}")
            return False
    
    def _release_single_camera(self, camera_side: str):
        """Libera una cámara individual"""
        if camera_side == 'left' and self.left_camera:
            self.left_camera.release()
            self.left_camera = None
        elif camera_side == 'right' and self.right_camera:
            self.right_camera.release()
            self.right_camera = None
    
    def read_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Captura un par de frames sincronizados de ambas cámaras
        
        Returns:
            Tupla (frame_left, frame_right) o (None, None) si falla
        """
        if not self._is_capturing or not self.left_camera or not self.right_camera:
            if self.auto_reconnect:
                logger.warning("⚠️ Cámaras no disponibles, intentando reconectar...")
                if self._attempt_reconnection():
                    return self.read_frames()  # Reintentar después de reconectar
            return None, None
        
        with self._sync_lock:
            try:
                # Captura sincronizada - leer ambas cámaras lo más simultáneamente posible
                start_time = time.time()
                
                # Leer frame izquierdo
                ret_left, frame_left = self.left_camera.read()
                
                # Leer frame derecho inmediatamente después
                ret_right, frame_right = self.right_camera.read()
                
                capture_time = time.time() - start_time
                
                # Verificar que ambas capturas fueron exitosas
                if not ret_left or not ret_right or frame_left is None or frame_right is None:
                    logger.warning("⚠️ Falló captura de uno o ambos frames")
                    self.stats['sync_failures'] += 1
                    
                    if self.auto_reconnect:
                        if self._attempt_reconnection():
                            return self.read_frames()
                    
                    return None, None
                
                # Verificar sincronización (tiempo de captura razonable)
                if capture_time > 0.1:  # 100ms es demasiado para captura sincronizada
                    logger.debug(f"⚠️ Captura lenta: {capture_time*1000:.1f}ms")
                
                self.stats['frames_captured'] += 1
                
                logger.debug(f"📷 Frames capturados: L{frame_left.shape} R{frame_right.shape} "
                           f"({capture_time*1000:.1f}ms)")
                
                return frame_left, frame_right
                
            except Exception as e:
                logger.error(f"❌ Error en captura sincronizada: {e}")
                self.stats['sync_failures'] += 1
                
                if self.auto_reconnect:
                    if self._attempt_reconnection():
                        return self.read_frames()
                
                return None, None
    
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
            
            # Liberar cámaras actuales
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
    
    def _release_cameras(self):
        """Libera ambas cámaras"""
        self._is_capturing = False
        
        if self.left_camera:
            self.left_camera.release()
            self.left_camera = None
        
        if self.right_camera:
            self.right_camera.release()
            self.right_camera = None
    
    def is_opened(self) -> bool:
        """
        Verifica si ambas cámaras están abiertas y funcionando
        
        Returns:
            True si ambas cámaras están operativas
        """
        return (self._is_capturing and
                self.left_camera is not None and 
                self.right_camera is not None and
                self.left_camera.isOpened() and 
                self.right_camera.isOpened())
    
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
        except:
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
