#!/usr/bin/env python3
"""
Probar cámaras usando el SDK nativo de Orbbec
"""

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_orbbec_sdk():
    logger.info("🔍 === PROBANDO SDK NATIVO DE ORBBEC ===")
    
    try:
        # Intentar importar pyorbbec
        import pyorbbec as ob
        logger.info("✅ pyorbbec importado correctamente")
        
        # Crear contexto
        ctx = ob.Context()
        logger.info("✅ Contexto Orbbec creado")
        
        # Obtener lista de dispositivos
        device_list = ctx.query_devices()
        device_count = device_list.device_count()
        
        logger.info(f"📱 Dispositivos encontrados: {device_count}")
        
        if device_count == 0:
            logger.error("❌ No se encontraron dispositivos Orbbec")
            return False
        
        # Probar cada dispositivo
        for i in range(device_count):
            test_orbbec_device(device_list, i)
        
        return True
        
    except ImportError:
        logger.error("❌ pyorbbec no está instalado")
        logger.info("💡 Instalar con: pip3 install pyorbbec")
        return False
    except Exception as e:
        logger.error(f"❌ Error con SDK Orbbec: {e}")
        return False

def test_orbbec_device(device_list, index):
    """Probar un dispositivo Orbbec específico"""
    try:
        import pyorbbec as ob
        
        device = device_list.get_device(index)
        device_info = device.get_device_info()
        
        logger.info(f"\n📷 Dispositivo {index}:")
        logger.info(f"   Nombre: {device_info.name()}")
        logger.info(f"   PID: {device_info.pid()}")
        logger.info(f"   VID: {device_info.vid()}")
        logger.info(f"   Serial: {device_info.serial_number()}")
        
        # Obtener perfiles de stream
        profiles = device.get_sensor_list()
        
        for i in range(profiles.count()):
            sensor = profiles.get_sensor(i)
            sensor_type = sensor.get_type()
            logger.info(f"   Sensor {i}: {sensor_type}")
        
        # Intentar crear pipeline
        pipeline = ob.Pipeline(device)
        config = ob.Config()
        
        # Configurar streams
        try:
            # Color stream
            color_profiles = pipeline.get_stream_profile_list(ob.SENSOR_COLOR)
            if color_profiles.count() > 0:
                color_profile = color_profiles.get_video_stream_profile(0)
                config.enable_stream(color_profile)
                logger.info(f"   ✅ Color stream: {color_profile.width()}x{color_profile.height()}")
            
            # Depth stream 
            depth_profiles = pipeline.get_stream_profile_list(ob.SENSOR_DEPTH)
            if depth_profiles.count() > 0:
                depth_profile = depth_profiles.get_video_stream_profile(0)
                config.enable_stream(depth_profile)
                logger.info(f"   ✅ Depth stream: {depth_profile.width()}x{depth_profile.height()}")
            
            # Iniciar pipeline
            pipeline.start(config)
            logger.info("   ✅ Pipeline iniciado")
            
            # Capturar algunos frames
            for frame_num in range(5):
                frameset = pipeline.wait_for_frames(1000)  # 1 segundo timeout
                
                if frameset:
                    color_frame = frameset.color_frame()
                    depth_frame = frameset.depth_frame()
                    
                    logger.info(f"   Frame {frame_num + 1}: Color={color_frame is not None}, Depth={depth_frame is not None}")
                else:
                    logger.warning(f"   Frame {frame_num + 1}: Sin datos")
            
            pipeline.stop()
            logger.info("   ✅ Pipeline detenido")
            
        except Exception as e:
            logger.error(f"   ❌ Error con pipeline: {e}")
        
    except Exception as e:
        logger.error(f"❌ Error probando dispositivo {index}: {e}")

if __name__ == "__main__":
    if not test_orbbec_sdk():
        logger.info("\n💡 === PASOS SIGUIENTES ===")
        logger.info("1. Instalar SDK: https://github.com/orbbec/OrbbecSDK")
        logger.info("2. pip3 install pyorbbec")
        logger.info("3. Verificar que las cámaras estén conectadas por USB (no solo ethernet)")