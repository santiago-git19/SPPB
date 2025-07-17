#!/usr/bin/env python3
"""
Descargador Automático de Dependencias - TensorRT Pose
======================================================

Script para descargar y verificar automáticamente todas las dependencias
necesarias para el sistema de pose estimation en Jetson Nano.

Incluye:
- Modelos pre-entrenados
- Archivos de configuración
- Verificación de dependencias de Python
- Configuración automática de directorios

Autor: Sistema de IA
Fecha: 2025
"""

import os
import sys
import urllib.request
import json
import subprocess
import logging
from pathlib import Path
import hashlib
import zipfile
import tarfile

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DependencyManager:
    """Gestor automático de dependencias para TensorRT Pose"""
    
    def __init__(self):
        # Configuración de directorios
        self.base_dir = Path.cwd()
        self.models_dir = self.base_dir / "models"
        self.configs_dir = self.base_dir / "configs"
        self.utils_dir = self.base_dir / "utils"
        
        # Crear directorios necesarios
        self.models_dir.mkdir(exist_ok=True)
        self.configs_dir.mkdir(exist_ok=True)
        self.utils_dir.mkdir(exist_ok=True)
        
        # Definir archivos necesarios
        self.required_files = {
            # Configuraciones
            "human_pose.json": {
                "url": "https://raw.githubusercontent.com/NVIDIA-AI-IOT/trt_pose/master/tasks/human_pose/human_pose.json",
                "path": self.configs_dir / "human_pose.json",
                "description": "Configuración de topología para pose estimation",
                "required": True
            },
            
            # Modelos pre-entrenados (nota: estos links son ejemplos, ajustar según disponibilidad)
            "resnet18_baseline_att_224x224_A_epoch_249.pth": {
                "url": "https://drive.google.com/uc?id=1XYDdCUdiF2xxx4rznmLb62SdOUZuoNbd",
                "path": self.models_dir / "resnet18_baseline_att_224x224_A_epoch_249.pth",
                "description": "Modelo ResNet18 pre-entrenado para pose estimation",
                "required": True,
                "size_mb": 45  # Tamaño aproximado en MB
            },
            
            # Modelo ligero alternativo para Jetson Nano
            "mobilenetv2_baseline_att_224x224_B_epoch_200.pth": {
                "url": "https://drive.google.com/uc?id=1-EXAMPLE_ID",  # Cambiar por URL real
                "path": self.models_dir / "mobilenetv2_baseline_att_224x224_B_epoch_200.pth",
                "description": "Modelo MobileNetV2 (más ligero para Jetson Nano)",
                "required": False,
                "size_mb": 15
            }
        }
        
        # Dependencias de Python
        self.python_dependencies = [
            "torch>=1.6.0",
            "torchvision>=0.7.0",
            "opencv-python>=4.2.0",
            "numpy>=1.18.0",
            "scipy>=1.4.0",
            "psutil>=5.7.0",
            "Pillow>=7.0.0",
            "tqdm>=4.0.0"
        ]
        
        # Dependencias opcionales para Jetson
        self.jetson_dependencies = [
            "torch2trt",  # Para optimización TensorRT
            "trt_pose",   # Biblioteca principal
        ]
        
    def check_system_requirements(self):
        """Verifica los requisitos del sistema"""
        logger.info("🔍 Verificando requisitos del sistema...")
        
        # Verificar Python
        python_version = sys.version_info
        if python_version < (3, 6):
            logger.error("❌ Python 3.6+ requerido")
            return False
        logger.info(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Verificar CUDA
        try:
            import torch
            if torch.cuda.is_available():
                cuda_version = torch.version.cuda
                logger.info(f"✅ CUDA {cuda_version} disponible")
                logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
            else:
                logger.warning("⚠️  CUDA no disponible, usando CPU")
        except ImportError:
            logger.warning("⚠️  PyTorch no instalado")
            
        # Verificar espacio en disco
        disk_usage = self._get_disk_space()
        if disk_usage < 2:  # Menos de 2GB libres
            logger.warning(f"⚠️  Poco espacio en disco: {disk_usage:.1f}GB libres")
        else:
            logger.info(f"✅ Espacio en disco: {disk_usage:.1f}GB libres")
            
        return True
        
    def _get_disk_space(self):
        """Obtiene el espacio libre en disco en GB"""
        try:
            import shutil
            total, used, free = shutil.disk_usage(self.base_dir)
            return free / (1024**3)  # Convertir a GB
        except:
            return float('inf')
            
    def install_python_dependencies(self):
        """Instala las dependencias de Python"""
        logger.info("📦 Instalando dependencias de Python...")
        
        for dep in self.python_dependencies:
            if not self._is_package_installed(dep.split('>=')[0]):
                logger.info(f"Instalando {dep}...")
                if not self._install_package(dep):
                    logger.error(f"❌ Error instalando {dep}")
                    return False
            else:
                logger.info(f"✅ {dep.split('>=')[0]} ya instalado")
                
        return True
        
    def install_jetson_dependencies(self):
        """Instala dependencias específicas de Jetson (opcional)"""
        logger.info("🚀 Instalando dependencias específicas de Jetson...")
        
        for dep in self.jetson_dependencies:
            if not self._is_package_installed(dep):
                logger.info(f"Intentando instalar {dep}...")
                if not self._install_jetson_package(dep):
                    logger.warning(f"⚠️  No se pudo instalar {dep} (puede requerir instalación manual)")
            else:
                logger.info(f"✅ {dep} ya instalado")
                
    def _is_package_installed(self, package_name):
        """Verifica si un paquete está instalado"""
        try:
            __import__(package_name.replace('-', '_'))
            return True
        except ImportError:
            return False
            
    def _install_package(self, package):
        """Instala un paquete usando pip"""
        try:
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', package
            ], capture_output=True, text=True, check=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error instalando {package}: {e.stderr}")
            return False
            
    def _install_jetson_package(self, package):
        """Instala paquetes específicos de Jetson"""
        # Instrucciones especiales para algunos paquetes
        if package == "torch2trt":
            logger.info("Para torch2trt, usar: git clone https://github.com/NVIDIA-AI-IOT/torch2trt")
            return False  # Requiere instalación manual
        elif package == "trt_pose":
            logger.info("Para trt_pose, usar: git clone https://github.com/NVIDIA-AI-IOT/trt_pose")
            return False  # Requiere instalación manual
        else:
            return self._install_package(package)
            
    def download_file_with_progress(self, url, local_path, description=""):
        """Descarga un archivo con barra de progreso"""
        try:
            logger.info(f"📥 Descargando: {description}")
            logger.info(f"    URL: {url}")
            logger.info(f"    Destino: {local_path}")
            
            def show_progress(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min(100, (block_num * block_size * 100) / total_size)
                    sys.stdout.write(f"\r    Progreso: {percent:.1f}%")
                    sys.stdout.flush()
                    
            urllib.request.urlretrieve(url, local_path, show_progress)
            print()  # Nueva línea después del progreso
            
            # Verificar que el archivo se descargó correctamente
            if local_path.exists() and local_path.stat().st_size > 0:
                size_mb = local_path.stat().st_size / (1024*1024)
                logger.info(f"✅ Descarga completada: {size_mb:.1f}MB")
                return True
            else:
                logger.error("❌ Error: archivo vacío o no existe")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error descargando: {e}")
            return False
            
    def download_required_files(self):
        """Descarga todos los archivos necesarios"""
        logger.info("📂 Descargando archivos necesarios...")
        
        for filename, file_info in self.required_files.items():
            local_path = file_info["path"]
            
            # Verificar si el archivo ya existe
            if local_path.exists():
                size_mb = local_path.stat().st_size / (1024*1024)
                logger.info(f"✅ {filename} ya existe ({size_mb:.1f}MB)")
                continue
                
            # Descargar archivo requerido
            if file_info["required"]:
                logger.info(f"📥 Descargando archivo requerido: {filename}")
                if not self.download_file_with_progress(
                    file_info["url"], 
                    local_path, 
                    file_info["description"]
                ):
                    logger.error(f"❌ Error descargando archivo requerido: {filename}")
                    return False
            else:
                # Archivo opcional
                logger.info(f"📥 Descargando archivo opcional: {filename}")
                if not self.download_file_with_progress(
                    file_info["url"], 
                    local_path, 
                    file_info["description"]
                ):
                    logger.warning(f"⚠️  No se pudo descargar archivo opcional: {filename}")
                    
        return True
        
    def verify_installation(self):
        """Verifica que todo esté correctamente instalado"""
        logger.info("🔍 Verificando instalación...")
        
        all_good = True
        
        # Verificar archivos requeridos
        for filename, file_info in self.required_files.items():
            if file_info["required"]:
                if file_info["path"].exists():
                    logger.info(f"✅ {filename}")
                else:
                    logger.error(f"❌ Falta archivo requerido: {filename}")
                    all_good = False
                    
        # Verificar dependencias críticas
        critical_deps = ["torch", "cv2", "numpy"]
        for dep in critical_deps:
            if self._is_package_installed(dep):
                logger.info(f"✅ {dep}")
            else:
                logger.error(f"❌ Falta dependencia crítica: {dep}")
                all_good = False
                
        return all_good
        
    def create_config_file(self):
        """Crea archivo de configuración con rutas"""
        config = {
            "model_paths": {
                "topology": str(self.configs_dir / "human_pose.json"),
                "pytorch_model": str(self.models_dir / "resnet18_baseline_att_224x224_A_epoch_249.pth"),
                "tensorrt_model": str(self.models_dir / "resnet18_baseline_att_224x224_A_epoch_249_trt.pth"),
                "mobilenet_model": str(self.models_dir / "mobilenetv2_baseline_att_224x224_B_epoch_200.pth")
            },
            "jetson_config": {
                "enable_swap": True,
                "swap_size_gb": 2,
                "max_cpu_cores": 2,
                "memory_limit_percent": 85,
                "temperature_limit": 75
            }
        }
        
        config_file = self.base_dir / "trt_pose_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
            
        logger.info(f"✅ Archivo de configuración creado: {config_file}")
        
    def run_setup(self):
        """Ejecuta el setup completo"""
        logger.info("🚀 Iniciando setup automático de TensorRT Pose")
        logger.info("=" * 60)
        
        # Verificar sistema
        if not self.check_system_requirements():
            logger.error("❌ Requisitos del sistema no cumplidos")
            return False
            
        # Instalar dependencias Python
        if not self.install_python_dependencies():
            logger.error("❌ Error instalando dependencias de Python")
            return False
            
        # Intentar instalar dependencias de Jetson
        self.install_jetson_dependencies()
        
        # Descargar archivos
        if not self.download_required_files():
            logger.error("❌ Error descargando archivos necesarios")
            return False
            
        # Crear configuración
        self.create_config_file()
        
        # Verificar instalación
        if self.verify_installation():
            logger.info("🎉 Setup completado exitosamente!")
            logger.info("\nPróximos pasos:")
            logger.info("1. Verificar que torch2trt y trt_pose estén instalados")
            logger.info("2. Ejecutar: python example_trt_pose_optimized.py")
            return True
        else:
            logger.error("❌ Setup incompleto, revisar errores arriba")
            return False

def main():
    """Función principal"""
    print("🤖 TensorRT Pose - Descargador Automático de Dependencias")
    print("=" * 60)
    
    manager = DependencyManager()
    
    try:
        success = manager.run_setup()
        return 0 if success else 1
    except KeyboardInterrupt:
        logger.info("⏹️  Setup interrumpido por usuario")
        return 1
    except Exception as e:
        logger.error(f"❌ Error crítico durante setup: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
