#!/usr/bin/env python3
"""
Diagnóstico rápido de red para cámaras Orbbec
"""

import subprocess
import socket
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quick_network_diagnostic():
    logger.info("🔍 === DIAGNÓSTICO RÁPIDO DE RED ===")
    
    # 1. Confirmar configuración
    logger.info("\n1. 📊 Red configurada correctamente:")
    logger.info("   ✅ Jetson IP: 192.168.1.2/24")
    logger.info("   ✅ Interfaz eth0: UP")
    logger.info("   ✅ Red objetivo: 192.168.1.0/24")
    
    # 2. Ping a gateway/router
    logger.info("\n2. 🏓 Ping a gateway:")
    gateway_ips = ["192.168.1.1", "192.168.1.254"]
    
    for gateway in gateway_ips:
        try:
            result = subprocess.run(['ping', '-c', '2', '-W', '1', gateway],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                logger.info(f"   ✅ Gateway {gateway}: Responde")
            else:
                logger.info(f"   ❌ Gateway {gateway}: No responde")
        except:
            logger.info(f"   ❌ Gateway {gateway}: Error")
    
    # 3. Ping a cámaras específicas
    logger.info("\n3. 📷 Ping a cámaras:")
    camera_ips = ["192.168.1.10", "192.168.1.11"]
    
    for ip in camera_ips:
        logger.info(f"   Probando {ip}:")
        
        # Ping normal
        try:
            result = subprocess.run(['ping', '-c', '3', '-W', '2', ip],
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info(f"      ✅ Ping: Responde")
                
                # Si responde, probar puertos
                test_camera_ports(ip)
            else:
                logger.info(f"      ❌ Ping: No responde")
                logger.info(f"         Output: {result.stdout.strip()}")
        except Exception as e:
            logger.info(f"      ❌ Ping: Error - {e}")
    
    # 4. Escaneo de red completo
    logger.info("\n4. 🌐 Escaneo de red 192.168.1.x:")
    scan_network_range()

def test_camera_ports(ip):
    """Prueba puertos específicos de cámara"""
    camera_ports = [22, 23, 80, 554, 8080, 8554]
    
    for port in camera_ports:
        if test_port(ip, port):
            logger.info(f"         ✅ Puerto {port}: ABIERTO")
        else:
            logger.info(f"         ❌ Puerto {port}: cerrado")

def test_port(ip, port, timeout=2.0):
    """Prueba conexión a puerto"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((ip, port))
        sock.close()
        return result == 0
    except:
        return False

def scan_network_range():
    """Escanea toda la red 192.168.1.x"""
    active_devices = []
    
    for i in range(1, 255):
        ip = f"192.168.1.{i}"
        
        # Skip IP propia
        if ip == "192.168.1.2":
            continue
            
        try:
            result = subprocess.run(['ping', '-c', '1', '-W', '1', ip],
                                  capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                active_devices.append(ip)
                logger.info(f"   ✅ Dispositivo activo: {ip}")
        except:
            continue
    
    logger.info(f"\n📊 Total dispositivos activos: {len(active_devices)}")
    
    if len(active_devices) == 0:
        logger.warning("⚠️ No se encontraron dispositivos activos")
        logger.info("💡 Posibles causas:")
        logger.info("   - Switch apagado o sin alimentación")
        logger.info("   - Cámaras apagadas o sin PoE")
        logger.info("   - Cables ethernet desconectados")
        logger.info("   - Cámaras en IPs diferentes")

def suggest_next_steps():
    logger.info("\n💡 === PRÓXIMOS PASOS ===")
    
    logger.info("1. 🔌 Verificaciones físicas:")
    logger.info("   • ¿El switch TL-SG1005P tiene LEDs encendidos?")
    logger.info("   • ¿Las cámaras tienen LEDs indicadores encendidos?")
    logger.info("   • ¿Los cables ethernet están bien conectados?")
    
    logger.info("\n2. ⚡ Verificaciones de alimentación:")
    logger.info("   • ¿El switch tiene alimentación externa para PoE?")
    logger.info("   • ¿Las cámaras reciben PoE del switch?")
    
    logger.info("\n3. 🔧 Configuración de cámaras:")
    logger.info("   • Las cámaras podrían tener IPs por defecto diferentes")
    logger.info("   • Usar herramienta de Orbbec para configurar IPs")
    logger.info("   • Verificar manual de las cámaras Gemini 335Le")
    
    logger.info("\n4. 🧪 Pruebas adicionales:")
    logger.info("   • Conectar una cámara directamente (sin switch)")
    logger.info("   • Usar herramienta de escaneo de red más agresiva")
    logger.info("   • Verificar con Wireshark si hay tráfico de red")

if __name__ == "__main__":
    quick_network_diagnostic()
    suggest_next_steps()