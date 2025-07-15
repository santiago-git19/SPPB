#!/usr/bin/env python3
"""
Script para descargar automáticamente los modelos necesarios para clasificación de poses.
"""

import os
import sys
import urllib.request
import urllib.error
import json
import hashlib
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
    Descarga el modelo TensorRT Pose ResNet18.
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
    print("=== Descargando Modelo TensorRT Pose ===")
    success = download_file(model_url, model_path)
    
    if success:
        print(f"✓ Modelo TensorRT Pose descargado: {model_path}")
        return True
    else:
        print("✗ Error descargando modelo TensorRT Pose")
        return False

def download_topology_file():
    """
    Descarga el archivo de topología COCO.
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
    print("=== Descargando Archivo de Topología ===")
    success = download_file(topology_url, topology_path)
    
    if success:
        # Verificar que es un JSON válido
        try:
            with open(topology_path, 'r') as f:
                json.load(f)
            print(f"✓ Topología descargada y validada: {topology_path}")
            return True
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

def verify_installation():
    """
    Verifica que todos los modelos necesarios estén disponibles.
    """
    models_dir = Path("models")
    
    required_files = [
        "resnet18_baseline_att_224x224_A_epoch_249.pth",
        "human_pose.json",
        "pose_classification.engine"
    ]
    
    print("=== Verificando Instalación ===")
    
    all_present = True
    for filename in required_files:
        filepath = models_dir / filename
        if filepath.exists():
            size = os.path.getsize(filepath)
            print(f"✓ {filename} ({size:,} bytes)")
        else:
            print(f"✗ {filename} - FALTANTE")
            all_present = False
    
    if all_present:
        print("\n✓ Todos los modelos necesarios están disponibles")
        print("Puedes ejecutar la clasificación de poses:")
        print("  python simple_pose_classification.py --demo")
        print("  python simple_pose_classification.py video.mp4")
    else:
        print("\n✗ Algunos modelos están faltantes")
        print("Ejecuta este script para descargarlos")
    
    return all_present

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
