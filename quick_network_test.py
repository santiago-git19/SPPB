#!/usr/bin/env python3
"""
Explorar el puerto 8090 de las cámaras Orbbec
"""

import socket
import requests
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def explore_orbbec_management():
    logger.info("🔍 === EXPLORANDO PUERTO 8090 DE ORBBEC ===")
    
    camera_ips = ["192.168.1.10", "192.168.1.11"]
    
    for ip in camera_ips:
        logger.info(f"\n� Explorando {ip}:8090:")
        
        # 1. Probar conexión HTTP
        test_http_connection(ip)
        
        # 2. Probar conexión TCP raw
        test_tcp_connection(ip)
        
        # 3. Probar diferentes URLs HTTP
        test_http_endpoints(ip)

def test_http_connection(ip):
    """Probar conexión HTTP básica"""
    logger.info("   1. 🌐 Conexión HTTP:")
    
    try:
        # Probar HTTP básico
        response = requests.get(f"http://{ip}:8090", timeout=5)
        logger.info(f"      ✅ HTTP Status: {response.status_code}")
        logger.info(f"      Headers: {dict(response.headers)}")
        
        if response.text:
            logger.info(f"      Content: {response.text[:200]}...")
            
        return True
        
    except requests.exceptions.ConnectTimeout:
        logger.info("      ❌ HTTP: Timeout")
    except requests.exceptions.ConnectionError as e:
        logger.info(f"      ❌ HTTP: Connection error - {e}")
    except Exception as e:
        logger.info(f"      ❌ HTTP: Error - {e}")
    
    return False

def test_tcp_connection(ip):
    """Probar conexión TCP raw"""
    logger.info("   2. 🔌 Conexión TCP raw:")
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        
        result = sock.connect_ex((ip, 8090))
        if result == 0:
            logger.info("      ✅ TCP: Conectado")
            
            # Enviar algunos comandos de prueba
            test_commands = [
                b"GET / HTTP/1.1\r\nHost: " + ip.encode() + b"\r\n\r\n",
                b"ORBBEC_DISCOVER\r\n",
                b"GET_INFO\r\n",
                b"STATUS\r\n"
            ]
            
            for cmd in test_commands:
                try:
                    sock.send(cmd)
                    time.sleep(0.5)
                    
                    sock.settimeout(2.0)
                    response = sock.recv(1024)
                    
                    if response:
                        logger.info(f"         Comando: {cmd[:20]}...")
                        logger.info(f"         Respuesta: {response[:100]}...")
                        break
                        
                except socket.timeout:
                    continue
                except Exception as e:
                    logger.debug(f"         Error con comando: {e}")
                    continue
        else:
            logger.info(f"      ❌ TCP: No se pudo conectar (code: {result})")
        
        sock.close()
        
    except Exception as e:
        logger.info(f"      ❌ TCP: Error - {e}")

def test_http_endpoints(ip):
    """Probar diferentes endpoints HTTP"""
    logger.info("   3. 🎯 Endpoints HTTP:")
    
    # Endpoints comunes para cámaras IP/dispositivos de gestión
    endpoints = [
        "/",
        "/info",
        "/status", 
        "/config",
        "/stream",
        "/video",
        "/api",
        "/api/info",
        "/api/status",
        "/api/stream",
        "/cgi-bin/info",
        "/device/info",
        "/orbbec/info",
        "/camera/info"
    ]
    
    working_endpoints = []
    
    for endpoint in endpoints:
        try:
            url = f"http://{ip}:8090{endpoint}"
            response = requests.get(url, timeout=3)
            
            if response.status_code == 200:
                logger.info(f"      ✅ {endpoint}: {response.status_code}")
                logger.info(f"         Content-Type: {response.headers.get('content-type', 'N/A')}")
                
                if response.text:
                    content_preview = response.text[:100].replace('\n', ' ')
                    logger.info(f"         Content: {content_preview}...")
                
                working_endpoints.append(endpoint)
                
            elif response.status_code != 404:
                logger.info(f"      ⚠️ {endpoint}: {response.status_code}")
                
        except requests.exceptions.Timeout:
            continue
        except requests.exceptions.ConnectionError:
            continue
        except Exception as e:
            logger.debug(f"      Error {endpoint}: {e}")
            continue
    
    return working_endpoints

def try_orbbec_specific_protocols(ip):
    """Probar protocolos específicos de Orbbec"""
    logger.info("   4. 🔧 Protocolos específicos Orbbec:")
    
    try:
        # Protocolo de configuración Orbbec
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect((ip, 8090))
        
        # Comandos específicos de Orbbec (basados en documentación)
        orbbec_commands = [
            b"GetDeviceInfo\r\n",
            b"GetStreamProfile\r\n", 
            b"StartStream\r\n",
            b"GetCameraParams\r\n"
        ]
        
        for cmd in orbbec_commands:
            try:
                sock.send(cmd)
                time.sleep(0.5)
                
                response = sock.recv(2048)
                if response:
                    logger.info(f"      ✅ {cmd.decode().strip()}: {response[:100]}...")
                    
            except Exception as e:
                logger.debug(f"      Error con {cmd}: {e}")
        
        sock.close()
        
    except Exception as e:
        logger.info(f"      ❌ Error protocolos Orbbec: {e}")

if __name__ == "__main__":
    explore_orbbec_management()