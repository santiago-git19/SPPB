#!/usr/bin/env python3
"""
Verificar SDK y herramientas de Orbbec
"""

import subprocess
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_orbbec_environment():
    logger.info("🔍 === VERIFICANDO ENTORNO ORBBEC ===")
    
    # 1. Buscar bibliotecas de Orbbec
    logger.info("\n1. 📚 Bibliotecas de Orbbec:")
    orbbec_libs = [
        '/usr/local/lib/liborbbec.so*',
        '/usr/lib/liborbbec.so*',
        '/opt/orbbec/*'
    ]
    
    found_libs = []
    for lib_pattern in orbbec_libs:
        try:
            result = subprocess.run(['find', '/', '-name', lib_pattern.split('/')[-1]], 
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
            if result.stdout:
                found_libs.extend(result.stdout.decode().strip().split('\n'))
        except:
            continue
    
    if found_libs:
        for lib in found_libs:
            if lib.strip():
                logger.info(f"   ✅ Encontrado: {lib}")
    else:
        logger.info("   ❌ No se encontraron bibliotecas de Orbbec")
    
    # 2. Verificar Python packages
    logger.info("\n2. 🐍 Paquetes Python de Orbbec:")
    try:
        import pyorbbec
        logger.info("   ✅ pyorbbec: Instalado")
    except ImportError:
        logger.info("   ❌ pyorbbec: No instalado")
    
    # 3. Sugerir instalación si es necesario
    if not found_libs:
        suggest_orbbec_installation()

def suggest_orbbec_installation():
    logger.info("\n💡 === SUGERENCIAS DE INSTALACIÓN ===")
    
    logger.info("Para instalar el SDK de Orbbec:")
    logger.info("1. Descargar OrbbecSDK desde: https://github.com/orbbec/OrbbecSDK")
    logger.info("2. Instalar dependencias:")
    logger.info("   sudo apt-get install libusb-1.0-0-dev")
    logger.info("3. Compilar e instalar el SDK")
    logger.info("4. Instalar Python bindings:")
    logger.info("   pip install pyorbbec")

if __name__ == "__main__":
    check_orbbec_environment()