#!/usr/bin/env python3
"""
Gestor de Modelos TensorRT - Jetson Nano
========================================

Script para gestionar modelos PyTorch y TensorRT:
- Verificar modelos existentes
- Convertir PyTorch a TensorRT automáticamente
- Monitorear conversión con recursos
- Validar modelos convertidos

Uso:
    python model_manager.py --check          # Verificar modelos
    python model_manager.py --convert        # Convertir PyTorch → TensorRT
    python model_manager.py --validate       # Validar modelo TensorRT
    python model_manager.py --auto           # Conversión automática si es necesario

Autor: Sistema de IA
Fecha: 2025
"""

import os
import sys
import argparse
import json
import time
import logging
import subprocess
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelManager:
    """Gestor de modelos para TensorRT Pose"""
    
    def __init__(self):
        self.config_file = Path("trt_pose_config.json")
        self._load_model_paths()
        
    def _load_model_paths(self):
        """Carga las rutas de modelos desde configuración"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                self.model_paths = config.get("model_paths", {})
        else:
            # Rutas por defecto
            self.model_paths = {
                'topology': '/home/mobilenet/Documentos/Trabajo/trt_pose/tasks/human_pose/human_pose.json',
                'pytorch_model': '/home/mobilenet/Documentos/Trabajo/trt_pose/models/resnet18_baseline_att_224x224_A_epoch_249.pth',
                'tensorrt_model': 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
            }
            
    def check_models(self):
        """Verifica el estado de todos los modelos"""
        logger.info("🔍 Verificando estado de modelos...")
        logger.info("=" * 50)
        
        all_ok = True
        
        for model_type, path in self.model_paths.items():
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024**2)
                logger.info("✅ %s: %s (%.1f MB)", model_type, path, size_mb)
                
                # Verificaciones adicionales
                if model_type == 'pytorch_model':
                    self._verify_pytorch_model(path)
                elif model_type == 'tensorrt_model':
                    self._verify_tensorrt_model(path)
                    
            else:
                logger.error("❌ %s: %s (NO ENCONTRADO)", model_type, path)
                all_ok = False
                
        return all_ok
        
    def _verify_pytorch_model(self, model_path):
        """Verifica que el modelo PyTorch sea válido"""
        try:
            import torch
            
            # Intentar cargar el modelo
            checkpoint = torch.load(model_path, map_location='cpu')
            
            if isinstance(checkpoint, dict):
                logger.info("   📋 Checkpoint con claves: %s", list(checkpoint.keys())[:5])
            else:
                logger.info("   📋 Modelo directo (no checkpoint)")
                
            logger.info("   ✅ Modelo PyTorch válido")
            
        except Exception as e:
            logger.error("   ❌ Error verificando PyTorch: %s", str(e))
            
    def _verify_tensorrt_model(self, model_path):
        """Verifica que el modelo TensorRT sea válido"""
        try:
            import torch
            from torch2trt import TRTModule
            
            # Intentar cargar el modelo TensorRT
            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(model_path, map_location='cpu'))
            
            logger.info("   ✅ Modelo TensorRT válido")
            
        except Exception as e:
            logger.error("   ❌ Error verificando TensorRT: %s", str(e))
            
    def convert_model(self):
        """Ejecuta la conversión PyTorch a TensorRT"""
        logger.info("⚡ Iniciando conversión PyTorch → TensorRT...")
        
        # Verificar que existe el modelo PyTorch
        pytorch_path = self.model_paths.get('pytorch_model', '')
        if not os.path.exists(pytorch_path):
            logger.error("❌ Modelo PyTorch no encontrado: %s", pytorch_path)
            return False
            
        # Verificar espacio en disco
        import shutil
        free_space = shutil.disk_usage('.').free / (1024**3)
        if free_space < 2:
            logger.error("❌ Espacio insuficiente en disco (%.1f GB)", free_space)
            return False
            
        logger.info("✅ Iniciando conversión...")
        
        try:
            # Ejecutar script de conversión
            result = subprocess.run([
                sys.executable, 'convert_model_to_tensorrt.py'
            ], text=True)
            
            if result.returncode == 0:
                logger.info("✅ Conversión completada exitosamente")
                
                # Verificar archivo resultante
                tensorrt_path = self.model_paths.get('tensorrt_model', '')
                if os.path.exists(tensorrt_path):
                    size_mb = os.path.getsize(tensorrt_path) / (1024**2)
                    logger.info("📁 Modelo TensorRT: %s (%.1f MB)", tensorrt_path, size_mb)
                    return True
                else:
                    logger.error("❌ Archivo TensorRT no se creó")
                    return False
            else:
                logger.error("❌ Error en conversión")
                return False
                
        except Exception as e:
            logger.error("❌ Error ejecutando conversión: %s", str(e))
            return False
            
    def validate_tensorrt_model(self):
        """Valida el modelo TensorRT con una prueba de inferencia"""
        logger.info("🧪 Validando modelo TensorRT...")
        
        tensorrt_path = self.model_paths.get('tensorrt_model', '')
        if not os.path.exists(tensorrt_path):
            logger.error("❌ Modelo TensorRT no encontrado: %s", tensorrt_path)
            return False
            
        try:
            import torch
            from torch2trt import TRTModule
            
            # Cargar modelo
            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(tensorrt_path, map_location='cuda:0'))
            
            # Crear entrada de prueba
            test_input = torch.zeros((1, 3, 224, 224), dtype=torch.float32, device='cuda:0')
            
            # Prueba de inferencia
            start_time = time.time()
            with torch.no_grad():
                output = model_trt(test_input)
            inference_time = (time.time() - start_time) * 1000
            
            logger.info("✅ Modelo TensorRT válido")
            logger.info("   Tiempo de inferencia: %.1f ms", inference_time)
            logger.info("   Salida: %s", [list(o.shape) for o in output])
            
            return True
            
        except Exception as e:
            logger.error("❌ Error validando TensorRT: %s", str(e))
            return False
            
    def auto_setup(self):
        """Configuración automática: convierte solo si es necesario"""
        logger.info("🤖 Configuración automática de modelos...")
        
        tensorrt_path = self.model_paths.get('tensorrt_model', '')
        pytorch_path = self.model_paths.get('pytorch_model', '')
        
        # Si ya existe TensorRT y es válido, no hacer nada
        if os.path.exists(tensorrt_path):
            logger.info("✅ Modelo TensorRT ya existe")
            if self.validate_tensorrt_model():
                logger.info("✅ Modelo TensorRT válido, no se requiere conversión")
                return True
            else:
                logger.warning("⚠️ Modelo TensorRT inválido, reconvirtiendo...")
                
        # Verificar modelo PyTorch
        if not os.path.exists(pytorch_path):
            logger.error("❌ Modelo PyTorch no encontrado para conversión")
            return False
            
        # Realizar conversión
        logger.info("⚡ Realizando conversión automática...")
        if self.convert_model():
            return self.validate_tensorrt_model()
        else:
            return False
            
    def get_model_info(self):
        """Obtiene información detallada de modelos"""
        info = {
            'models_found': {},
            'recommendations': []
        }
        
        for model_type, path in self.model_paths.items():
            if os.path.exists(path):
                stat = os.stat(path)
                info['models_found'][model_type] = {
                    'path': path,
                    'size_mb': stat.st_size / (1024**2),
                    'modified': time.ctime(stat.st_mtime)
                }
                
        # Generar recomendaciones
        if 'tensorrt_model' not in info['models_found']:
            if 'pytorch_model' in info['models_found']:
                info['recommendations'].append("Convertir modelo PyTorch a TensorRT para mejor rendimiento")
            else:
                info['recommendations'].append("Descargar modelo PyTorch primero")
                
        return info
        
    def print_status_report(self):
        """Imprime reporte de estado completo"""
        logger.info("📊 Reporte de Estado de Modelos")
        logger.info("=" * 60)
        
        info = self.get_model_info()
        
        # Modelos encontrados
        if info['models_found']:
            logger.info("📁 Modelos encontrados:")
            for model_type, details in info['models_found'].items():
                logger.info("   %s:", model_type)
                logger.info("     Archivo: %s", details['path'])
                logger.info("     Tamaño: %.1f MB", details['size_mb'])
                logger.info("     Modificado: %s", details['modified'])
        else:
            logger.warning("⚠️ No se encontraron modelos")
            
        # Recomendaciones
        if info['recommendations']:
            logger.info("\n💡 Recomendaciones:")
            for i, rec in enumerate(info['recommendations'], 1):
                logger.info("   %d. %s", i, rec)
                
        # Estado del sistema
        logger.info("\n🖥️ Estado del sistema:")
        try:
            import psutil
            memory = psutil.virtual_memory()
            logger.info("   RAM: %.1f GB disponible / %.1f GB total", 
                       memory.available / (1024**3), memory.total / (1024**3))
            
            import shutil
            disk = shutil.disk_usage('.')
            logger.info("   Disco: %.1f GB libres", disk.free / (1024**3))
            
        except ImportError:
            logger.info("   (psutil no disponible para estadísticas)")

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description='Gestor de modelos TensorRT para Jetson Nano',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--check', action='store_true',
                       help='Verificar estado de modelos')
    parser.add_argument('--convert', action='store_true',
                       help='Convertir PyTorch a TensorRT')
    parser.add_argument('--validate', action='store_true',
                       help='Validar modelo TensorRT')
    parser.add_argument('--auto', action='store_true',
                       help='Configuración automática')
    parser.add_argument('--status', action='store_true',
                       help='Reporte de estado completo')
    
    args = parser.parse_args()
    
    # Si no se especifica acción, mostrar estado
    if not any([args.check, args.convert, args.validate, args.auto]):
        args.status = True
        
    manager = ModelManager()
    
    try:
        if args.status:
            manager.print_status_report()
            return 0
            
        elif args.check:
            success = manager.check_models()
            return 0 if success else 1
            
        elif args.convert:
            success = manager.convert_model()
            return 0 if success else 1
            
        elif args.validate:
            success = manager.validate_tensorrt_model()
            return 0 if success else 1
            
        elif args.auto:
            success = manager.auto_setup()
            if success:
                logger.info("🎉 Configuración automática completada")
                return 0
            else:
                logger.error("❌ Error en configuración automática")
                return 1
                
    except KeyboardInterrupt:
        logger.info("⏹️ Operación interrumpida por usuario")
        return 1
    except Exception as e:
        logger.error("❌ Error crítico: %s", str(e))
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
