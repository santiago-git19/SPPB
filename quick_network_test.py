#!/usr/bin/env python3
"""
Probar protocolos específicos de Orbbec
"""

import socket
import struct
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_orbbec_protocols():
    logger.info("🔍 === PROBANDO PROTOCOLOS ORBBEC ===")
    
    camera_ips = ["192.168.1.10", "192.168.1.11"]
    
    for ip in camera_ips:
        logger.info(f"\n� Probando protocolos para {ip}:")
        
        # Probar diferentes puertos y protocolos
        test_orbbec_discovery_protocol(ip)
        test_raw_socket_scan(ip)
        test_icmp_responses(ip)

def test_orbbec_discovery_protocol(ip):
    """Probar protocolo de descubrimiento de Orbbec"""
    logger.info("   1. 🔍 Protocolo de descubrimiento Orbbec:")
    
    # Puertos comunes para descubrimiento de dispositivos
    discovery_ports = [1900, 5353, 8089, 8090, 9999, 10001]
    
    for port in discovery_ports:
        try:
            # UDP Socket para descubrimiento
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(2.0)
            
            # Mensaje de descubrimiento genérico
            discovery_msg = b"DISCOVER\r\n"
            
            sock.sendto(discovery_msg, (ip, port))
            
            try:
                data, addr = sock.recvfrom(1024)
                logger.info(f"      ✅ Puerto UDP {port}: Respuesta - {data[:50]}")
            except socket.timeout:
                logger.debug(f"      ❌ Puerto UDP {port}: Sin respuesta")
            
            sock.close()
            
        except Exception as e:
            logger.debug(f"      Error UDP puerto {port}: {e}")

def test_raw_socket_scan(ip):
    """Escaneo más exhaustivo de puertos"""
    logger.info("   2. 🔎 Escaneo exhaustivo de puertos:")
    
    # Rangos de puertos a probar
    port_ranges = [
        range(1, 100),      # Puertos bien conocidos
        range(8000, 8100),  # Puertos HTTP alternativos
        range(9000, 9100),  # Puertos de aplicación
        range(10000, 10100) # Puertos altos
    ]
    
    open_ports = []
    
    for port_range in port_ranges:
        for port in port_range:
            if test_port_with_timeout(ip, port, 0.5):
                open_ports.append(port)
                logger.info(f"      ✅ Puerto TCP {port}: ABIERTO")
    
    if not open_ports:
        logger.info("      ❌ No se encontraron puertos TCP abiertos")
    
    return open_ports

def test_port_with_timeout(ip, port, timeout):
    """Probar puerto con timeout muy corto"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((ip, port))
        sock.close()
        return result == 0
    except:
        return False

def test_icmp_responses(ip):
    """Probar diferentes tipos de ping/ICMP"""
    logger.info("   3. 🏓 Análisis de respuestas ICMP:")
    
    # Diferentes tamaños de ping
    ping_sizes = [56, 1024, 1472]  # Normal, grande, MTU máximo
    
    for size in ping_sizes:
        try:
            result = subprocess.run(['ping', '-c', '1', '-s', str(size), ip],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  timeout=5)
            
            if result.returncode == 0:
                # Analizar tiempo de respuesta
                output = result.stdout.decode('utf-8')
                for line in output.split('\n'):
                    if 'time=' in line:
                        time_part = line.split('time=')[1].split()[0]
                        logger.info(f"      ✅ Ping {size} bytes: {time_part}")
                        break
            else:
                logger.info(f"      ❌ Ping {size} bytes: Sin respuesta")
                
        except Exception as e:
            logger.debug(f"      Error ping {size} bytes: {e}")

if __name__ == "__main__":
    test_orbbec_protocols()