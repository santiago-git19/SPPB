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
import socket
import subprocess
import ipaddress
from concurrent.futures import ThreadPoolExecutor
import requests

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
    
    def _discover_orbbec_cameras(self) -> List[str]:
        """Detección de cámaras Orbbec con IPs conocidas"""
        try:
            # IPs específicas de tus cámaras Orbbec
            camera_ips = ["192.168.1.10", "192.168.1.11"]
            
            logger.info("🎯 Buscando cámaras Orbbec en IPs conocidas...")
            
            # Método 1: Verificación directa de IPs conocidas
            cameras = self._discover_cameras_known_ips(camera_ips)
            if len(cameras) >= 2:
                return cameras
            
            # Método 2: Escaneo del rango de red si no se encuentran las IPs conocidas
            cameras = self._discover_cameras_network_range()
            if cameras:
                return cameras
                
            # Método 3: Detección por ping y puerto
            cameras = self._discover_cameras_ping_test(camera_ips)
            return cameras
            
        except Exception as e:
            logger.error(f"❌ Error en detección de cámaras: {e}")
            return []

    def _discover_cameras_known_ips(self, camera_ips: List[str]) -> List[str]:
        """Verifica cámaras en IPs específicas conocidas"""
        logger.info("🔍 Verificando IPs conocidas de cámaras...")
        
        cameras = []
        
        for ip in camera_ips:
            logger.info(f"   📡 Probando cámara en: {ip}")
            
            # Probar puertos comunes de cámaras Orbbec
            camera_ports = [554, 8554, 80, 8080, 1935]  # RTSP, HTTP, RTMP
            
            for port in camera_ports:
                if self._test_camera_connection(ip, port):
                    camera_url = self._build_camera_url(ip, port)
                    if self._verify_camera_stream(camera_url):
                        cameras.append(camera_url)
                        logger.info(f"   ✅ Cámara encontrada: {camera_url}")
                        break
            else:
                logger.warning(f"   ❌ No se pudo conectar a cámara en {ip}")
        
        return cameras

    def _test_camera_connection(self, ip: str, port: int) -> bool:
        """Prueba conexión TCP a una IP y puerto específicos"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)  # 2 segundos timeout
            
            result = sock.connect_ex((ip, port))
            sock.close()
            
            if result == 0:
                logger.debug(f"   ✅ Puerto {port} abierto en {ip}")
                return True
            else:
                logger.debug(f"   ❌ Puerto {port} cerrado en {ip}")
                return False
                
        except Exception as e:
            logger.debug(f"   ❌ Error probando {ip}:{port} - {e}")
            return False

    def _build_camera_url(self, ip: str, port: int) -> str:
        """Construye URL de cámara basada en IP y puerto"""
        if port in [554, 8554]:
            # Intentar diferentes rutas RTSP comunes para Orbbec
            rtsp_paths = [
                f"rtsp://{ip}:{port}/live",
                f"rtsp://{ip}:{port}/stream1", 
                f"rtsp://{ip}:{port}/color",
                f"rtsp://{ip}:{port}/rgb",
                f"rtsp://{ip}:{port}/"
            ]
            
            # Probar cada ruta RTSP
            for url in rtsp_paths:
                if self._test_rtsp_url(url):
                    return url
            
            # Si ninguna funciona, devolver la primera
            return rtsp_paths[0]
            
        elif port in [80, 8080]:
            return f"http://{ip}:{port}/video"
        else:
            return f"rtsp://{ip}:{port}/"

    def _test_rtsp_url(self, url: str) -> bool:
        """Prueba si una URL RTSP funciona"""
        try:
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    logger.debug(f"   📹 URL RTSP válida: {url}")
                    return True
            return False
        except Exception:
            return False

    def _verify_camera_stream(self, camera_url: str) -> bool:
        """Verifica que el stream de la cámara funcione correctamente"""
        try:
            logger.debug(f"   🔍 Verificando stream: {camera_url}")
            
            # Configurar captura con timeout
            if camera_url.startswith('rtsp://'):
                cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
                # Configuraciones para RTSP
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_TIMEOUT, 5000)  # 5 segundos timeout
            else:
                cap = cv2.VideoCapture(camera_url)
            
            if not cap.isOpened():
                logger.debug("   ❌ No se pudo abrir stream")
                return False
            
            # Intentar capturar varios frames para asegurar estabilidad
            successful_frames = 0
            for attempt in range(5):
                ret, frame = cap.read()
                if ret and frame is not None:
                    successful_frames += 1
                    logger.debug(f"   📷 Frame {attempt+1}: {frame.shape}")
                else:
                    logger.debug(f"   ❌ Falló frame {attempt+1}")
                
                time.sleep(0.1)  # Pausa entre frames
            
            cap.release()
            
            # Considerar válido si al menos 3 de 5 frames fueron exitosos
            is_valid = successful_frames >= 3
            logger.debug(f"   {'✅' if is_valid else '❌'} Stream válido: {successful_frames}/5 frames")
            
            return is_valid
            
        except Exception as e:
            logger.debug(f"   ❌ Error verificando stream: {e}")
            return False

    def _discover_cameras_network_range(self) -> List[str]:
        """Escanea el rango de red 192.168.1.x para encontrar más cámaras"""
        logger.info("🌐 Escaneando rango 192.168.1.x...")
        
        cameras = []
        base_ip = "192.168.1."
        
        # Escanear rango común para cámaras IP (normalmente 10-50)
        ip_range = list(range(10, 51))  # 192.168.1.10 a 192.168.1.50
        
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            
            for ip_suffix in ip_range:
                ip = f"{base_ip}{ip_suffix}"
                future = executor.submit(self._check_ip_for_camera, ip)
                futures.append(future)
            
            for future in futures:
                result = future.result()
                if result:
                    cameras.append(result)
                    logger.info(f"   ✅ Cámara adicional encontrada: {result}")
        
        return cameras

    def _check_ip_for_camera(self, ip: str) -> Optional[str]:
        """Verifica si hay una cámara en una IP específica"""
        camera_ports = [554, 8554, 80, 8080]
        
        for port in camera_ports:
            if self._test_camera_connection(ip, port):
                camera_url = self._build_camera_url(ip, port)
                if self._verify_camera_stream(camera_url):
                    return camera_url
        
        return None

    def _discover_cameras_ping_test(self, camera_ips: List[str]) -> List[str]:
        """Método de respaldo usando ping para verificar conectividad"""
        logger.info("🏓 Verificando conectividad con ping...")
        
        cameras = []
        
        for ip in camera_ips:
            if self._ping_host(ip):
                logger.info(f"   ✅ {ip} responde a ping")
                
                # Si responde a ping, asumir que es una cámara y construir URL por defecto
                default_url = f"rtsp://{ip}:554/live"
                cameras.append(default_url)
                logger.info(f"   📷 Usando URL por defecto: {default_url}")
            else:
                logger.warning(f"   ❌ {ip} no responde a ping")
        
        return cameras

    def _ping_host(self, ip: str) -> bool:
        """Ejecuta ping para verificar conectividad"""
        try:
            result = subprocess.run(
                ['ping', '-c', '2', '-W', '2', ip],
                capture_output=True, 
                text=True, 
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    # Modificar el método de inicialización para manejar URLs de red
    def _initialize_single_camera(self, camera_side: str) -> bool:
        """Inicializa una cámara de red con configuración optimizada"""
        camera_url = self.left_device_id if camera_side == 'left' else self.right_device_id
        
        try:
            logger.info(f"🌐 Inicializando cámara {camera_side} de red: {camera_url}")
            
            # Configuración específica para streams de red
            if camera_url.startswith('rtsp://'):
                cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
                
                # Configuraciones optimizadas para RTSP
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, self.fps)
                cap.set(cv2.CAP_PROP_TIMEOUT, 10000)  # 10 segundos timeout
                
                # Configurar resolución si es posible
                width, height = self.resolution
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                
            elif camera_url.startswith('http://'):
                cap = cv2.VideoCapture(camera_url)
                
            else:
                logger.error(f"❌ Protocolo no soportado en URL: {camera_url}")
                return False
            
            if not cap.isOpened():
                logger.error(f"❌ No se pudo abrir stream de cámara {camera_side}")
                return False
            
            # Verificar captura inicial
            logger.info(f"   🔍 Verificando captura inicial...")
            for attempt in range(5):
                ret, frame = cap.read()
                if ret and frame is not None:
                    logger.info(f"   ✅ Frame capturado: {frame.shape}")
                    break
                else:
                    logger.warning(f"   ⚠️ Intento {attempt+1} falló, reintentando...")
                    time.sleep(0.5)
            else:
                logger.error(f"❌ No se pudo capturar frame de cámara {camera_side}")
                cap.release()
                return False
            
            # Asignar cámara
            if camera_side == 'left':
                self.left_camera = cap
            else:
                self.right_camera = cap
            
            logger.info(f"✅ Cámara {camera_side} de red inicializada correctamente")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error inicializando cámara {camera_side} de red: {e}")
            return False

    # También necesitamos eliminar las líneas que no aplican para red
    def _get_local_networks(self) -> List[str]:
        """Para tu configuración específica, solo necesitamos la red 192.168.1.0/24"""
        return ['192.168.1.0/24']

    def _get_local_ip(self) -> str:
        """Obtiene la IP local de la Jetson en la red 192.168.1.x"""
        try:
            # Buscar interfaz de red que esté en la red 192.168.1.x
            result = subprocess.run(['ip', 'addr', 'show'], capture_output=True, text=True)
            
            for line in result.stdout.split('\n'):
                if 'inet 192.168.1.' in line and '192.168.1.255' not in line:
                    # Extraer IP del formato: "inet 192.168.1.100/24 brd 192.168.1.255 scope global eth0"
                    ip = line.strip().split()[1].split('/')[0]
                    logger.info(f"📍 IP local detectada: {ip}")
                    return ip
            
            # Si no encuentra IP en 192.168.1.x, usar método genérico
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("192.168.1.1", 80))
                return s.getsockname()[0]
                
        except Exception as e:
            logger.warning(f"⚠️ Error obteniendo IP local: {e}")
            return "192.168.1.100"  # IP por defecto asumida
    
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
