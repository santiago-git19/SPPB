#!/usr/bin/env python3
"""
Script de prueba corregido para Python < 3.7
"""

import subprocess
import socket
import time
import logging
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_cameras_python36():
    """Prueba las cámaras usando sintaxis compatible con Python 3.6"""
    logger.info("📷 === PRUEBA DE CÁMARAS (Python 3.6 compatible) ===")
    
    camera_configs = {
        "192.168.1.10": "54:14:fd:23:09:47",
        "192.168.1.11": "54:14:fd:23:09:b0"
    }
    
    working_urls = []
    
    for ip, mac in camera_configs.items():
        logger.info(f"\n🔍 Probando cámara {ip} (MAC: {mac}):")
        
        # 1. Ping compatible con Python 3.6
        if ping_python36(ip):
            logger.info("   ✅ Ping: Responde")
            
            # 2. Probar puertos
            open_ports = test_camera_ports_python36(ip)
            
            # 3. Probar streams
            camera_url = find_working_stream_python36(ip, open_ports)
            if camera_url:
                working_urls.append(camera_url)
                logger.info(f"   🎥 Stream funcional: {camera_url}")
        else:
            logger.error(f"   ❌ Ping: No responde")
    
    # 4. Probar captura dual
    if len(working_urls) >= 2:
        test_dual_capture_python36(working_urls[:2])
    
    return working_urls

def ping_python36(ip):
    """Función de ping compatible con Python 3.6"""
    try:
        # Método 1: subprocess.run sin capture_output
        result = subprocess.run(['ping', '-c', '3', ip], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE,
                              timeout=10)
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        logger.debug(f"Timeout en ping a {ip}")
        return False
    except Exception as e:
        # Método 2: Usar os.system como fallback
        try:
            import os
            result = os.system(f"ping -c 1 {ip} > /dev/null 2>&1")
            return result == 0
        except:
            logger.debug(f"Error en ping a {ip}: {e}")
            return False

def test_camera_ports_python36(ip):
    """Probar puertos usando socket (compatible con cualquier Python)"""
    common_ports = [22, 23, 80, 554, 8080, 8554, 1935]
    open_ports = []
    
    logger.info("   🔌 Puertos:")
    for port in common_ports:
        if test_port_python36(ip, port):
            open_ports.append(port)
            logger.info(f"      ✅ {port}: ABIERTO")
        else:
            logger.info(f"      ❌ {port}: cerrado")
    
    return open_ports

def test_port_python36(ip, port, timeout=3.0):
    """Probar puerto usando socket"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((ip, port))
        sock.close()
        return result == 0
    except:
        return False

def find_working_stream_python36(ip, open_ports):
    """Encontrar stream funcional"""
    logger.info("   📹 Probando streams:")
    
    # URLs a probar basadas en puertos abiertos
    test_urls = []
    
    if 554 in open_ports:
        rtsp_paths = ["/live", "/stream1", "/color", "/rgb", "/main", "/"]
        for path in rtsp_paths:
            test_urls.append(f"rtsp://{ip}:554{path}")
    
    if 8554 in open_ports:
        test_urls.append(f"rtsp://{ip}:8554/live")
    
    if 80 in open_ports:
        test_urls.append(f"http://{ip}:80/video")
    
    if 8080 in open_ports:
        test_urls.append(f"http://{ip}:8080/video")
    
    # Si no hay puertos abiertos, probar URLs por defecto
    if not test_urls:
        test_urls = [
            f"rtsp://{ip}:554/live",
            f"rtsp://{ip}:554/",
            f"http://{ip}:80/video"
        ]
    
    # Probar cada URL
    for url in test_urls:
        logger.info(f"      🔍 Probando: {url}")
        if test_stream_url_python36(url):
            logger.info(f"      ✅ FUNCIONA: {url}")
            return url
        else:
            logger.info(f"      ❌ No funciona")
    
    return None

def test_stream_url_python36(url):
    """Probar URL de stream"""
    try:
        if url.startswith('rtsp://'):
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            cap = cv2.VideoCapture(url)
        
        if not cap.isOpened():
            return False
        
        # Probar captura
        success_count = 0
        for i in range(3):
            ret, frame = cap.read()
            if ret and frame is not None:
                success_count += 1
                logger.info(f"         📷 Frame {i+1}: {frame.shape}")
            time.sleep(0.2)
        
        cap.release()
        return success_count >= 2
        
    except Exception as e:
        logger.debug(f"Error probando {url}: {e}")
        return False

def test_dual_capture_python36(camera_urls):
    """Probar captura dual"""
    logger.info(f"\n🎥 === PRUEBA DE CAPTURA DUAL ===")
    
    left_url = camera_urls[0]
    right_url = camera_urls[1]
    
    logger.info(f"📷 Cámara izquierda: {left_url}")
    logger.info(f"📷 Cámara derecha: {right_url}")
    
    try:
        # Inicializar cámaras
        left_cap = cv2.VideoCapture(left_url, cv2.CAP_FFMPEG if left_url.startswith('rtsp') else cv2.CAP_ANY)
        right_cap = cv2.VideoCapture(right_url, cv2.CAP_FFMPEG if right_url.startswith('rtsp') else cv2.CAP_ANY)
        
        # Configurar cámaras
        for cap in [left_cap, right_cap]:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not (left_cap.isOpened() and right_cap.isOpened()):
            logger.error("❌ No se pudieron abrir ambas cámaras")
            return False
        
        logger.info("✅ Ambas cámaras inicializadas")
        
        # Capturar frames
        successful_frames = 0
        for i in range(10):
            ret_left, frame_left = left_cap.read()
            ret_right, frame_right = right_cap.read()
            
            if ret_left and ret_right and frame_left is not None and frame_right is not None:
                successful_frames += 1
                logger.info(f"   Frame {i+1}: L{frame_left.shape} R{frame_right.shape} ✅")
            else:
                logger.warning(f"   Frame {i+1}: Error en captura")
            
            time.sleep(0.1)
        
        left_cap.release()
        right_cap.release()
        
        success_rate = successful_frames / 10 * 100
        logger.info(f"📊 Tasa de éxito: {success_rate}%")
        
        return success_rate >= 70
        
    except Exception as e:
        logger.error(f"❌ Error en captura dual: {e}")
        return False

if __name__ == "__main__":
    working_urls = test_cameras_python36()
    
    if working_urls:
        logger.info(f"\n✅ ¡ÉXITO! Encontradas {len(working_urls)} cámaras funcionales:")
        for i, url in enumerate(working_urls, 1):
            logger.info(f"   Cámara {i}: {url}")
    else:
        logger.error("\n❌ No se encontraron cámaras funcionales")