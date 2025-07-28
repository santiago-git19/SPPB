#!/usr/bin/env python3
"""
Escaneo agresivo para encontrar cámaras Orbbec
"""

import subprocess
import time
import logging
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def aggressive_network_scan():
    logger.info("� === ESCANEO AGRESIVO DE RED ===")
    
    # 1. Usar nmap si está disponible
    try_nmap_scan()
    
    # 2. Escaneo ARP
    arp_scan()
    
    # 3. Escaneo paralelo con múltiples pings
    parallel_ping_scan()
    
    # 4. Escaneo de diferentes redes
    scan_multiple_networks()

def try_nmap_scan():
    logger.info("\n1. 🎯 Intentando escaneo con nmap:")
    
    try:
        # Instalar nmap si no está disponible
        result = subprocess.run(['which', 'nmap'], capture_output=True)
        if result.returncode != 0:
            logger.info("   Instalando nmap...")
            subprocess.run(['sudo', 'apt-get', 'update'], capture_output=True)
            subprocess.run(['sudo', 'apt-get', 'install', '-y', 'nmap'], capture_output=True)
        
        # Escaneo de red con nmap
        logger.info("   Escaneando 192.168.1.0/24 con nmap...")
        result = subprocess.run(['nmap', '-sn', '192.168.1.0/24'], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            logger.info("   Resultado nmap:")
            for line in result.stdout.split('\n'):
                if 'Nmap scan report for' in line:
                    logger.info(f"      {line}")
        
    except Exception as e:
        logger.info(f"   ❌ Error con nmap: {e}")

def arp_scan():
    logger.info("\n2. � Escaneo ARP:")
    
    try:
        # Generar tráfico ARP enviando pings
        logger.info("   Generando tráfico ARP...")
        
        def ping_for_arp(ip_suffix):
            ip = f"192.168.1.{ip_suffix}"
            try:
                subprocess.run(['ping', '-c', '1', '-W', '1', ip],
                             capture_output=True, timeout=2)
            except:
                pass
        
        # Ping a toda la red para generar entradas ARP
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(ping_for_arp, i) for i in range(1, 255)]
            
            # Esperar un poco
            time.sleep(2)
        
        # Leer tabla ARP
        logger.info("   Tabla ARP actual:")
        result = subprocess.run(['arp', '-a'], capture_output=True, text=True)
        
        for line in result.stdout.split('\n'):
            if '192.168.1.' in line and 'incomplete' not in line:
                logger.info(f"      {line.strip()}")
                
    except Exception as e:
        logger.info(f"   ❌ Error en escaneo ARP: {e}")

def parallel_ping_scan():
    logger.info("\n3. ⚡ Escaneo paralelo intensivo:")
    
    def intensive_ping(ip_suffix):
        ip = f"192.168.1.{ip_suffix}"
        
        # Múltiples pings con diferentes parámetros
        ping_variants = [
            ['ping', '-c', '3', '-W', '1', ip],
            ['ping', '-c', '5', '-W', '2', ip],
            ['ping', '-c', '1', '-s', '1024', ip],  # Ping con payload grande
        ]
        
        for ping_cmd in ping_variants:
            try:
                result = subprocess.run(ping_cmd, capture_output=True, 
                                      text=True, timeout=10)
                if result.returncode == 0:
                    return ip
            except:
                continue
        
        return None
    
    logger.info("   Escaneando con múltiples métodos...")
    
    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = [executor.submit(intensive_ping, i) for i in range(1, 255)]
        
        active_ips = []
        for future in futures:
            result = future.result()
            if result:
                active_ips.append(result)
                logger.info(f"      ✅ IP activa: {result}")
    
    logger.info(f"   Total IPs encontradas: {len(active_ips)}")

def scan_multiple_networks():
    logger.info("\n4. 🌐 Escaneando múltiples redes:")
    
    # Las cámaras podrían estar en diferentes redes
    networks = [
        "192.168.0.0/24",
        "192.168.1.0/24", 
        "10.0.0.0/24",
        "172.16.0.0/24"
    ]
    
    for network in networks:
        logger.info(f"   Escaneando {network}...")
        
        base_ip = network.split('/')[0].rsplit('.', 1)[0]
        
        # Escaneo rápido de IPs comunes para cámaras
        common_camera_ips = [1, 10, 11, 100, 101, 168, 200, 254]
        
        for suffix in common_camera_ips:
            ip = f"{base_ip}.{suffix}"
            
            try:
                result = subprocess.run(['ping', '-c', '1', '-W', '1', ip],
                                      capture_output=True, timeout=3)
                if result.returncode == 0:
                    logger.info(f"      ✅ {ip} responde")
            except:
                continue

if __name__ == "__main__":
    aggressive_network_scan()