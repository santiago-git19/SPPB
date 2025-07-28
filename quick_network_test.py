#!/usr/bin/env python3
"""
Prueba las cámaras Orbbec encontradas en 192.168.1.10 y 192.168.1.11
"""

import cv2
import socket
import time
import logging
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_found_cameras():
    logger.info("📷 === PRUEBA DE CÁMARAS ENCONTRADAS ===")
    
    camera_ips = {
        "192.168.1.10": "54:14:fd:23:09:47",
        "192.168.1.11": "54:14:fd:23:09:b0"
    }
    
    working_urls = []
    
    for ip, mac in camera_ips.items():
        logger.info(f"\n🔍 Probando cámara {ip} (MAC: {mac}):")
        
        # 1. Verificar conectividad básica
        if test_ping(ip):
            logger.info("   ✅ Ping: Responde")
            
            # 2. Probar puertos
            open_ports = test_camera_ports(ip)
            
            # 3. Probar streams RTSP/HTTP
            camera_url = test_camera_streams(ip, open_ports)
            if camera_url:
                working_urls.append(camera_url)
                logger.info(f"   🎥 Stream funcional: {camera_url}")
            else:
                logger.warning(f"   ❌ No se encontró stream funcional")
        else:
            logger.error(f"   ❌ Ping: No responde")
    
    # 4. Probar captura dual si hay 2 cámaras
    if len(working_urls) >= 2:
        test_dual_capture(working_urls[:2])
    else:
        logger.warning(f"⚠️ Solo {len(working_urls)} cámaras funcionales, se necesitan 2")
    
    return working_urls

def test_ping(ip):
    """Probar ping a IP"""
    try:
        result = subprocess.run(['ping', '-c', '3', '-W', '2', ip],
                              capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except:
        return False

def test_camera_ports(ip):
    """Probar puertos comunes de cámaras"""
    common_ports = [22, 23, 80, 554, 8080, 8554, 1935]
    open_ports = []
    
    logger.info("   🔌 Puertos:")
    for port in common_ports:
        if test_port(ip, port):
            open_ports.append(port)
            logger.info(f"      ✅ {port}: ABIERTO")
        else:
            logger.info(f"      ❌ {port}: cerrado")
    
    return open_ports

def test_port(ip, port, timeout=3.0):
    """Probar conexión a puerto"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((ip, port))
        sock.close()
        return result == 0
    except:
        return False

def test_camera_streams(ip, open_ports):
    """Probar diferentes tipos de streams"""
    logger.info("   📹 Probando streams:")
    
    # Definir URLs a probar basadas en puertos abiertos
    test_urls = []
    
    # RTSP URLs (puerto 554/8554)
    if 554 in open_ports:
        rtsp_paths = ["/live", "/stream1", "/color", "/rgb", "/main", "/"]
        for path in rtsp_paths:
            test_urls.append(f"rtsp://{ip}:554{path}")
    
    if 8554 in open_ports:
        rtsp_paths = ["/live", "/stream1", "/color", "/rgb", "/main", "/"]
        for path in rtsp_paths:
            test_urls.append(f"rtsp://{ip}:8554{path}")
    
    # HTTP URLs (puerto 80/8080)
    if 80 in open_ports:
        http_paths = ["/video", "/mjpeg", "/stream", "/live"]
        for path in http_paths:
            test_urls.append(f"http://{ip}:80{path}")
    
    if 8080 in open_ports:
        http_paths = ["/video", "/mjpeg", "/stream", "/live"]
        for path in http_paths:
            test_urls.append(f"http://{ip}:8080{path}")
    
    # Si no hay puertos conocidos abiertos, probar URLs por defecto
    if not test_urls:
        test_urls = [
            f"rtsp://{ip}:554/live",
            f"rtsp://{ip}:554/",
            f"http://{ip}:80/video"
        ]
    
    # Probar cada URL
    for url in test_urls:
        logger.info(f"      🔍 Probando: {url}")
        if test_stream_url(url):
            logger.info(f"      ✅ FUNCIONA: {url}")
            return url
        else:
            logger.info(f"      ❌ No funciona: {url}")
    
    return None

def test_stream_url(url):
    """Probar una URL de stream específica"""
    try:
        # Configurar captura según el protocolo
        if url.startswith('rtsp://'):
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_TIMEOUT, 5000)  # 5 segundos
        else:
            cap = cv2.VideoCapture(url)
        
        if not cap.isOpened():
            return False
        
        # Intentar capturar múltiples frames
        successful_frames = 0
        for i in range(3):
            ret, frame = cap.read()
            if ret and frame is not None:
                successful_frames += 1
                logger.info(f"         📷 Frame {i+1}: {frame.shape}")
            time.sleep(0.2)
        
        cap.release()
        
        # Considerar exitoso si al menos 2 de 3 frames funcionaron
        return successful_frames >= 2
        
    except Exception as e:
        logger.debug(f"         Error: {e}")
        return False

def test_dual_capture(camera_urls):
    """Probar captura dual"""
    logger.info(f"\n🎥 === PRUEBA DE CAPTURA DUAL ===")
    
    left_url = camera_urls[0]
    right_url = camera_urls[1]
    
    logger.info(f"📷 Cámara izquierda: {left_url}")
    logger.info(f"📷 Cámara derecha: {right_url}")
    
    try:
        # Inicializar cámaras
        logger.info("🔄 Inicializando cámaras...")
        
        left_cap = cv2.VideoCapture(left_url, cv2.CAP_FFMPEG if left_url.startswith('rtsp') else cv2.CAP_ANY)
        right_cap = cv2.VideoCapture(right_url, cv2.CAP_FFMPEG if right_url.startswith('rtsp') else cv2.CAP_ANY)
        
        # Configurar cámaras
        for cap in [left_cap, right_cap]:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_TIMEOUT, 5000)
        
        if not (left_cap.isOpened() and right_cap.isOpened()):
            logger.error("❌ No se pudieron abrir ambas cámaras")
            return False
        
        logger.info("✅ Ambas cámaras inicializadas")
        
        # Capturar frames de prueba
        logger.info("📹 Capturando frames de prueba...")
        
        successful_frames = 0
        for i in range(10):
            ret_left, frame_left = left_cap.read()
            ret_right, frame_right = right_cap.read()
            
            if ret_left and ret_right and frame_left is not None and frame_right is not None:
                successful_frames += 1
                logger.info(f"   Frame {i+1}: L{frame_left.shape} R{frame_right.shape} ✅")
            else:
                logger.warning(f"   Frame {i+1}: Error - L:{ret_left} R:{ret_right}")
            
            time.sleep(0.1)
        
        left_cap.release()
        right_cap.release()
        
        success_rate = successful_frames / 10 * 100
        logger.info(f"📊 Tasa de éxito: {success_rate}% ({successful_frames}/10 frames)")
        
        if success_rate >= 70:
            logger.info("🎉 ¡CAPTURA DUAL EXITOSA!")
            return True
        else:
            logger.warning("⚠️ Captura dual inestable")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error en captura dual: {e}")
        return False

if __name__ == "__main__":
    working_urls = test_found_cameras()
    
    if working_urls:
        logger.info(f"\n✅ === RESUMEN ===")
        logger.info(f"Cámaras funcionales encontradas: {len(working_urls)}")
        for i, url in enumerate(working_urls, 1):
            logger.info(f"   Cámara {i}: {url}")
    else:
        logger.error("\n❌ No se encontraron cámaras funcionales")