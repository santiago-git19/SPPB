#!/usr/bin/env python3
"""
Script para descargar automáticamente las dependencias necesarias para example_trt_pose_mio.py.
Esto incluye modelos TensorRT Pose, topología COCO, y verificación de dependencias Python.
"""

import os
import sys
import urllib.request
import urllib.error
import json
import hashlib
import subprocess
from pathlib import Path

def create_models_directory():
    """Crea el directorio de modelos si no existe."""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    return models_dir

def download_file(url, filepath, expected_size=None):
    """
    Descarga un archivo con barra de progreso.
    """
    print(f"Descargando: {url}")
    print(f"Guardando en: {filepath}")
    
    try:
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) // total_size)
                sys.stdout.write(f"\rProgreso: {percent}% ")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(url, filepath, progress_hook)
        print(f"\n✓ Descarga completada: {filepath}")
        
        # Verificar tamaño si se proporciona
        if expected_size:
            actual_size = os.path.getsize(filepath)
            if actual_size != expected_size:
                print(f"⚠ Advertencia: Tamaño esperado {expected_size}, actual {actual_size}")
        
        return True
        
    except urllib.error.URLError as e:
        print(f"\n✗ Error descargando {url}: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Error inesperado: {e}")
        return False

def verify_python_dependencies():
    """
    Verifica que las dependencias Python necesarias estén instaladas.
    """
    print("=== Verificando Dependencias Python ===")
    
    required_packages = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("cv2", "OpenCV"),
        ("json", "JSON (built-in)"),
        ("numpy", "NumPy"),
        ("trt_pose", "TensorRT Pose"),
        ("torch2trt", "Torch2TRT")
    ]
    
    missing_packages = []
    
    for package, description in required_packages:
        try:
            if package == "cv2":
                import cv2
                print(f"✓ {description}: {cv2.__version__}")
            elif package == "torch":
                import torch
                print(f"✓ {description}: {torch.__version__}")
            elif package == "torchvision":
                import torchvision
                print(f"✓ {description}: {torchvision.__version__}")
            elif package == "numpy":
                import numpy
                print(f"✓ {description}: {numpy.__version__}")
            elif package == "json":
                import json
                print(f"✓ {description}: Built-in")
            elif package == "trt_pose":
                import trt_pose
                print(f"✓ {description}: Disponible")
            elif package == "torch2trt":
                import torch2trt
                print(f"✓ {description}: Disponible")
        except ImportError:
            print(f"✗ {description}: NO DISPONIBLE")
            missing_packages.append((package, description))
    
    if missing_packages:
        print(f"\n⚠ Faltan {len(missing_packages)} dependencias:")
        for package, description in missing_packages:
            print(f"  - {description} ({package})")
        print("\nPara instalar dependencias faltantes:")
        print("  pip3 install torch torchvision opencv-python numpy")
        print("  # Para trt_pose y torch2trt, consulta la documentación de NVIDIA")
        return False
    
    print("\n✓ Todas las dependencias Python están disponibles")
    return True

def verify_file_integrity(filepath, expected_hash=None):
    """
    Verifica la integridad de un archivo usando hash SHA256.
    """
    if not os.path.exists(filepath):
        return False
    
    if expected_hash:
        with open(filepath, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        
        if file_hash != expected_hash:
            print(f"⚠ Hash incorrecto para {filepath}")
            print(f"  Esperado: {expected_hash}")
            print(f"  Actual:   {file_hash}")
            return False
    
    return True

def download_trt_pose_model():
    """
    Descarga el modelo TensorRT Pose ResNet18 desde NVIDIA.
    """
    models_dir = create_models_directory()
    
    # Configuración del modelo
    model_url = "https://github.com/NVIDIA-AI-IOT/trt_pose/releases/download/v0.0.1/resnet18_baseline_att_224x224_A_epoch_249.pth"
    model_path = models_dir / "resnet18_baseline_att_224x224_A_epoch_249.pth"
    
    # Verificar si ya existe
    if model_path.exists():
        print(f"✓ Modelo TensorRT Pose ya existe: {model_path}")
        return True
    
    # Descargar
    print("=== Descargando Modelo TensorRT Pose ResNet18 ===")
    success = download_file(model_url, model_path)
    
    if success:
        print(f"✓ Modelo TensorRT Pose descargado: {model_path}")
        return True
    else:
        print("✗ Error descargando modelo TensorRT Pose")
        return False

def download_topology_file():
    """
    Descarga el archivo de topología COCO desde el repositorio oficial.
    """
    models_dir = create_models_directory()
    
    # Configuración de topología
    topology_url = "https://raw.githubusercontent.com/NVIDIA-AI-IOT/trt_pose/master/tasks/human_pose/human_pose.json"
    topology_path = models_dir / "human_pose.json"
    
    # Verificar si ya existe
    if topology_path.exists():
        print(f"✓ Topología ya existe: {topology_path}")
        return True
    
    # Descargar
    print("=== Descargando Archivo de Topología COCO ===")
    success = download_file(topology_url, topology_path)
    
    if success:
        # Verificar que es un JSON válido
        try:
            with open(topology_path, 'r') as f:
                topology_data = json.load(f)
            
            # Verificar estructura esperada
            if 'keypoints' in topology_data and 'skeleton' in topology_data:
                keypoints_count = len(topology_data['keypoints'])
                skeleton_count = len(topology_data['skeleton'])
                print(f"✓ Topología descargada y validada: {topology_path}")
                print(f"  - Keypoints: {keypoints_count}")
                print(f"  - Conexiones: {skeleton_count}")
                return True
            else:
                print("✗ Error: Estructura de topología inválida")
                return False
                
        except json.JSONDecodeError:
            print("✗ Error: Archivo de topología corrupto")
            os.remove(topology_path)
            return False
    else:
        print("✗ Error descargando topología")
        return False

def create_sample_classification_engine():
    """
    Crea un engine de clasificación de ejemplo para pruebas.
    NOTA: Este es solo un placeholder. Para uso real, necesitas entrenar tu propio modelo.
    """
    models_dir = create_models_directory()
    engine_path = models_dir / "pose_classification.engine"
    
    if engine_path.exists():
        print(f"✓ Engine de clasificación ya existe: {engine_path}")
        return True
    
    print("=== Creando Engine de Clasificación de Ejemplo ===")
    print("⚠ ADVERTENCIA: Esto es solo un placeholder para pruebas")
    print("  Para uso real, necesitas entrenar tu propio modelo PoseClassificationNet")
    
    # Crear archivo dummy
    dummy_engine_data = b"DUMMY_TENSORRT_ENGINE_FOR_TESTING_ONLY"
    
    try:
        with open(engine_path, 'wb') as f:
            f.write(dummy_engine_data)
        
        print(f"✓ Engine de ejemplo creado: {engine_path}")
        print("  Este engine NO funcionará para clasificación real")
        print("  Usa convert_pose_model.py para crear un engine real")
        return True
        
    except Exception as e:
        print(f"✗ Error creando engine de ejemplo: {e}")
        return False

def download_additional_models():
    """
    Descarga modelos adicionales opcionales.
    """
    models_dir = create_models_directory()
    
    # Modelo DenseNet (alternativo)
    densenet_url = "https://github.com/NVIDIA-AI-IOT/trt_pose/releases/download/v0.0.1/densenet121_baseline_att_256x256_B_epoch_160.pth"
    densenet_path = models_dir / "densenet121_baseline_att_256x256_B_epoch_160.pth"
    
    if not densenet_path.exists():
        print("=== Descargando Modelo DenseNet (Opcional) ===")
        success = download_file(densenet_url, densenet_path)
        if success:
            print(f"✓ Modelo DenseNet descargado: {densenet_path}")
        else:
            print("✗ Error descargando DenseNet (opcional)")

def verify_video_file():
    """
    Verifica que el archivo de video de prueba exista.
    """
    video_path = Path("Automatizacion/WIN_20250702_12_09_08_Pro.mp4")
    
    if video_path.exists():
        size = os.path.getsize(video_path)
        print(f"✓ Video de prueba encontrado: {video_path} ({size:,} bytes)")
        return True
    else:
        print(f"⚠ Video de prueba no encontrado: {video_path}")
        print("  Asegúrate de tener el video en la ubicación correcta")
        return False

def check_cuda_availability():
    """
    Verifica que CUDA esté disponible para TensorRT.
    """
    print("=== Verificando CUDA ===")
    
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            
            print(f"✓ CUDA disponible: {cuda_version}")
            print(f"  - Dispositivos GPU: {device_count}")
            print(f"  - GPU principal: {device_name}")
            return True
        else:
            print("✗ CUDA no disponible")
            print("  TensorRT Pose requiere CUDA para funcionar")
            return False
    except ImportError:
        print("✗ PyTorch no disponible para verificar CUDA")
        return False

def install_missing_dependencies():
    """
    Intenta instalar dependencias faltantes automáticamente.
    """
    print("=== Instalando Dependencias Faltantes ===")
    
    try:
        # Instalar dependencias básicas
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "opencv-python", "numpy"
        ], check=True)
        
        print("✓ Dependencias básicas instaladas")
        
        # Mostrar instrucciones para trt_pose y torch2trt
        print("\n⚠ Para completar la instalación:")
        print("  1. Instala trt_pose:")
        print("     git clone https://github.com/NVIDIA-AI-IOT/trt_pose")
        print("     cd trt_pose && python setup.py install")
        print("  2. Instala torch2trt:")
        print("     git clone https://github.com/NVIDIA-AI-IOT/torch2trt")
        print("     cd torch2trt && python setup.py install")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Error instalando dependencias: {e}")
        return False

def verify_installation():
    """
    Verifica que todos los componentes necesarios estén disponibles.
    """
    models_dir = Path("models")
    
    required_files = [
        "resnet18_baseline_att_224x224_A_epoch_249.pth",
        "human_pose.json"
    ]
    
    print("=== Verificando Instalación Completa ===")
    
    all_present = True
    
    # Verificar archivos de modelo
    for filename in required_files:
        filepath = models_dir / filename
        if filepath.exists():
            size = os.path.getsize(filepath)
            print(f"✓ {filename} ({size:,} bytes)")
        else:
            print(f"✗ {filename} - FALTANTE")
            all_present = False
    
    # Verificar dependencias Python
    deps_ok = verify_python_dependencies()
    
    # Verificar CUDA
    cuda_ok = check_cuda_availability()
    
    # Verificar video
    video_ok = verify_video_file()
    
    if all_present and deps_ok and cuda_ok:
        print("\n✓ Todos los componentes están listos")
        print("Puedes ejecutar:")
        print("  python3 example_trt_pose_mio.py")
        
        if not video_ok:
            print("  (Recuerda colocar el video en la ubicación correcta)")
    else:
        print("\n✗ Algunos componentes están faltantes")
        print("Ejecuta este script para instalarlos")
    
    return all_present and deps_ok and cuda_ok

def main():
    """
    Función principal para descargar modelos.
    """
    print("=== Descargador de Modelos para Clasificación de Poses ===")
    
    # Verificar estado actual
    if verify_installation():
        print("\nTodos los modelos ya están disponibles.")
        response = input("¿Descargar modelos adicionales? (y/n): ").lower()
        if response == 'y':
            download_additional_models()
        return
    
    print("\nDescargando modelos necesarios...")
    
    # Descargar modelos requeridos
    success_count = 0
    total_count = 3
    
    if download_trt_pose_model():
        success_count += 1
    
    if download_topology_file():
        success_count += 1
    
    if create_sample_classification_engine():
        success_count += 1
    
    # Resumen
    print("\n=== Resumen ===")
    print(f"Descargados: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("✓ Todos los modelos básicos están disponibles")
        
        # Ofrecer descargar modelos adicionales
        response = input("\n¿Descargar modelos adicionales? (y/n): ").lower()
        if response == 'y':
            download_additional_models()
        
        print("\n¡Instalación completada!")
        print("Siguiente paso: python simple_pose_classification.py --demo")
        
    else:
        print("✗ Algunos modelos fallaron al descargar")
        print("Verifica tu conexión a internet y vuelve a intentar")

if __name__ == "__main__":
    main()
