#!/usr/bin/env python3
"""
Verificar si las cámaras Orbbec aparecen como dispositivos UVC
"""

import subprocess
import cv2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_uvc_devices():
    logger.info("� === VERIFICANDO DISPOSITIVOS UVC ===")
    
    # Método 1: lsusb para ver dispositivos USB
    logger.info("\n1. 📱 Dispositivos USB:")
    try:
        result = subprocess.run(['lsusb'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = result.stdout.decode('utf-8')
        
        for line in output.split('\n'):
            if line.strip():
                logger.info(f"   {line}")
                # Buscar Orbbec específicamente
                if 'orbbec' in line.lower() or '2bc5' in line.lower():
                    logger.info(f"   � ¡ORBBEC ENCONTRADO!: {line}")
                    
    except Exception as e:
        logger.error(f"Error ejecutando lsusb: {e}")
    
    # Método 2: v4l2-ctl para ver dispositivos de video
    logger.info("\n2. 📹 Dispositivos de video V4L2:")
    try:
        result = subprocess.run(['v4l2-ctl', '--list-devices'], 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = result.stdout.decode('utf-8')
        logger.info(output)
        
    except Exception as e:
        logger.info(f"v4l2-ctl no disponible o error: {e}")
    
    # Método 3: Listar /dev/video*
    logger.info("\n3. 🎥 Dispositivos /dev/video*:")
    try:
        import glob
        video_devices = glob.glob('/dev/video*')
        
        if video_devices:
            for device in sorted(video_devices):
                logger.info(f"   Encontrado: {device}")
                test_video_device(device)
        else:
            logger.info("   No se encontraron dispositivos /dev/video*")
            
    except Exception as e:
        logger.error(f"Error listando dispositivos de video: {e}")
    
    # Método 4: Probar OpenCV con índices de cámara
    logger.info("\n4. � Probando cámaras con OpenCV:")
    test_opencv_cameras()

def test_video_device(device):
    """Probar un dispositivo de video específico"""
    try:
        # Obtener información del dispositivo
        result = subprocess.run(['v4l2-ctl', '-d', device, '--all'], 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = result.stdout.decode('utf-8')
        
        # Buscar información relevante
        for line in output.split('\n'):
            if 'Card type' in line or 'Driver name' in line:
                logger.info(f"      {line.strip()}")
                
    except Exception as e:
        logger.debug(f"      Error obteniendo info de {device}: {e}")

def test_opencv_cameras():
    """Probar cámaras usando OpenCV con diferentes índices"""
    
    working_cameras = []
    
    # Probar índices 0-10
    for i in range(11):
        logger.info(f"   Probando cámara índice {i}:")
        
        try:
            cap = cv2.VideoCapture(i)
            
            if cap.isOpened():
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    logger.info(f"      ✅ Cámara {i}: Funcional - {frame.shape}")
                    working_cameras.append(i)
                    
                    # Obtener propiedades de la cámara
                    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    logger.info(f"         Resolución: {int(width)}x{int(height)}")
                    logger.info(f"         FPS: {fps}")
                else:
                    logger.info(f"      ❌ Cámara {i}: No puede capturar frames")
            else:
                logger.info(f"      ❌ Cámara {i}: No se puede abrir")
                
            cap.release()
            
        except Exception as e:
            logger.debug(f"      Error probando cámara {i}: {e}")
    
    return working_cameras

if __name__ == "__main__":
    working_cameras = check_uvc_devices()
    
    if working_cameras:
        logger.info(f"\n✅ ¡ÉXITO! Encontradas {len(working_cameras)} cámaras UVC:")
        for cam_id in working_cameras:
            logger.info(f"   Cámara índice: {cam_id}")
    else:
        logger.info("\n❌ No se encontraron cámaras UVC funcionales")