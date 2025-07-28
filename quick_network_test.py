#!/usr/bin/env python3
"""
Debug del problema de ping entre línea de comandos y script
"""

import subprocess
import os
import sys
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_ping_issue():
    logger.info("� === DEBUG DE PROBLEMA DE PING ===")
    
    camera_ips = ["192.168.1.10", "192.168.1.11"]
    
    for ip in camera_ips:
        logger.info(f"\n� Debugging ping a {ip}:")
        
        # Método 1: Replicar exactamente lo que funciona en línea de comandos
        test_direct_command(ip)
        
        # Método 2: Probar diferentes formas de subprocess
        test_subprocess_variants(ip)
        
        # Método 3: Verificar variables de entorno
        test_environment_issues(ip)

def test_direct_command(ip):
    """Probar comando directo como en línea de comandos"""
    logger.info("   1. 🎯 Comando directo (como línea de comandos):")
    
    try:
        # Exactamente el mismo comando que funciona manualmente
        cmd = f"ping -c 3 {ip}"
        logger.info(f"      Ejecutando: {cmd}")
        
        result = os.system(cmd)
        if result == 0:
            logger.info(f"      ✅ os.system(): Ping exitoso")
        else:
            logger.info(f"      ❌ os.system(): Ping falló (código: {result})")
            
    except Exception as e:
        logger.error(f"      ❌ Error con os.system(): {e}")

def test_subprocess_variants(ip):
    """Probar diferentes variantes de subprocess"""
    logger.info("   2. 🔧 Variantes de subprocess:")
    
    # Variante 1: Como en el script original
    try:
        logger.info("      a) Método original del script:")
        result = subprocess.run(['ping', '-c', '3', '-W', '2', ip],
                              capture_output=True, text=True, timeout=10)
        logger.info(f"         Return code: {result.returncode}")
        logger.info(f"         STDOUT: {result.stdout[:100]}...")
        logger.info(f"         STDERR: {result.stderr[:100]}...")
        
        if result.returncode == 0:
            logger.info("         ✅ Exitoso")
        else:
            logger.info("         ❌ Falló")
            
    except subprocess.TimeoutExpired:
        logger.error("         ❌ Timeout en subprocess")
    except Exception as e:
        logger.error(f"         ❌ Error: {e}")
    
    # Variante 2: Sin capture_output
    try:
        logger.info("      b) Sin capture_output:")
        result = subprocess.run(['ping', '-c', '3', ip], timeout=10)
        logger.info(f"         Return code: {result.returncode}")
        
        if result.returncode == 0:
            logger.info("         ✅ Exitoso")
        else:
            logger.info("         ❌ Falló")
            
    except Exception as e:
        logger.error(f"         ❌ Error: {e}")
    
    # Variante 3: Con shell=True
    try:
        logger.info("      c) Con shell=True:")
        cmd = f"ping -c 3 {ip}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        logger.info(f"         Return code: {result.returncode}")
        
        if result.returncode == 0:
            logger.info("         ✅ Exitoso")
            logger.info(f"         Output: {result.stdout[:100]}...")
        else:
            logger.info("         ❌ Falló")
            logger.info(f"         Error: {result.stderr[:100]}...")
            
    except Exception as e:
        logger.error(f"         ❌ Error: {e}")
    
    # Variante 4: Con Popen
    try:
        logger.info("      d) Con Popen:")
        process = subprocess.Popen(['ping', '-c', '3', ip], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE, 
                                 text=True)
        stdout, stderr = process.communicate(timeout=10)
        
        logger.info(f"         Return code: {process.returncode}")
        if process.returncode == 0:
            logger.info("         ✅ Exitoso")
        else:
            logger.info("         ❌ Falló")
            logger.info(f"         Error: {stderr[:100]}...")
            
    except Exception as e:
        logger.error(f"         ❌ Error: {e}")

def test_environment_issues(ip):
    """Verificar problemas de entorno"""
    logger.info("   3. 🌍 Verificaciones de entorno:")
    
    # Verificar PATH
    logger.info(f"      PATH: {os.environ.get('PATH', 'No definido')[:100]}...")
    
    # Verificar ubicación de ping
    try:
        which_result = subprocess.run(['which', 'ping'], capture_output=True, text=True)
        logger.info(f"      Ubicación de ping: {which_result.stdout.strip()}")
    except:
        logger.info("      ❌ No se pudo encontrar ubicación de ping")
    
    # Verificar permisos
    try:
        ping_path = '/bin/ping'
        if os.path.exists(ping_path):
            stat_info = os.stat(ping_path)
            logger.info(f"      Permisos de /bin/ping: {oct(stat_info.st_mode)}")
        else:
            logger.info("      /bin/ping no existe")
    except Exception as e:
        logger.error(f"      Error verificando permisos: {e}")
    
    # Verificar red desde Python
    test_network_from_python(ip)

def test_network_from_python(ip):
    """Probar conectividad de red desde Python directamente"""
    logger.info("   4. 🐍 Conectividad desde Python:")
    
    try:
        import socket
        
        # Test 1: Resolver DNS
        try:
            socket.gethostbyname(ip)
            logger.info(f"      ✅ Resolución DNS de {ip}: OK")
        except Exception as e:
            logger.info(f"      ❌ Resolución DNS: {e}")
        
        # Test 2: Socket directo
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3.0)
            
            # Probar puerto 80 (si está abierto)
            result = sock.connect_ex((ip, 80))
            sock.close()
            
            if result == 0:
                logger.info(f"      ✅ Socket TCP a {ip}:80: Conecta")
            else:
                logger.info(f"      ⚠️ Socket TCP a {ip}:80: No conecta (normal si puerto cerrado)")
                
        except Exception as e:
            logger.info(f"      ❌ Socket test: {e}")
        
        # Test 3: Ping usando socket ICMP (requiere root)
        try:
            # Solo intentar si somos root
            if os.geteuid() == 0:
                test_icmp_ping(ip)
            else:
                logger.info("      ⚠️ ICMP ping: Requiere permisos root")
                
        except Exception as e:
            logger.info(f"      ❌ ICMP test: {e}")
            
    except Exception as e:
        logger.error(f"   ❌ Error en pruebas de Python: {e}")

def test_icmp_ping(ip):
    """Prueba ping ICMP directo (requiere root)"""
    import socket
    import struct
    import time
    
    try:
        # Crear socket ICMP
        sock = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_ICMP)
        sock.settimeout(3.0)
        
        # Crear paquete ICMP Echo Request
        checksum = 0
        icmp_header = struct.pack('!BBHHH', 8, 0, checksum, 1, 1)
        
        # Calcular checksum
        checksum = calculate_checksum(icmp_header)
        icmp_header = struct.pack('!BBHHH', 8, 0, checksum, 1, 1)
        
        # Enviar ping
        sock.sendto(icmp_header, (ip, 0))
        
        # Recibir respuesta
        data, addr = sock.recvfrom(1024)
        sock.close()
        
        logger.info(f"      ✅ ICMP ping directo: Respuesta de {addr[0]}")
        
    except Exception as e:
        logger.info(f"      ❌ ICMP ping directo: {e}")

def calculate_checksum(data):
    """Calcular checksum para ICMP"""
    checksum = 0
    for i in range(0, len(data), 2):
        if i + 1 < len(data):
            checksum += (data[i] << 8) + data[i + 1]
        else:
            checksum += data[i] << 8
    
    checksum = (checksum >> 16) + (checksum & 0xFFFF)
    checksum += (checksum >> 16)
    return (~checksum) & 0xFFFF

def create_working_ping_function():
    """Crear función de ping que funcione basada en las pruebas"""
    logger.info("\n🔧 === CREANDO FUNCIÓN DE PING FUNCIONAL ===")
    
    def working_ping(ip):
        """Función de ping que debería funcionar"""
        try:
            # Usar el método que funcione mejor según las pruebas
            cmd = f"ping -c 3 {ip}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
            
            success = result.returncode == 0
            logger.info(f"working_ping({ip}): {'✅ SUCCESS' if success else '❌ FAILED'}")
            
            if not success:
                logger.info(f"   Error output: {result.stderr}")
            
            return success
            
        except Exception as e:
            logger.error(f"working_ping({ip}): Exception - {e}")
            return False
    
    # Probar la función
    camera_ips = ["192.168.1.10", "192.168.1.11"]
    for ip in camera_ips:
        working_ping(ip)

if __name__ == "__main__":
    debug_ping_issue()
    create_working_ping_function()