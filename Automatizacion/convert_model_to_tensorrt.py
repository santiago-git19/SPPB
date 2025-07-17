#!/usr/bin/env python3
"""
Convertidor PyTorch a TensorRT - Jetson Nano Optimizado
=======================================================

Script para convertir modelos PyTorch a TensorRT con:
- Monitoreo completo de recursos durante la conversión
- Configuración automática de swap
- Limitación de CPU para evitar sobrecalentamiento
- Reportes de progreso detallados
- Manejo robusto de memoria

Autor: Sistema de IA
Fecha: 2025
"""

import json
import torch
import time
import sys
import os
import gc
import psutil
import subprocess
import threading
from pathlib import Path
import logging
from datetime import datetime
import trt_pose.models
import trt_pose.coco
import torch2trt

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Importar utilidades de Jetson
from utils.jetson_utils import (
    JetsonResourceMonitor, 
    JetsonSwapManager, 
    JetsonCPULimiter, 
    setup_jetson_optimizations
)

class TensorRTModelConverter:
    """Convertidor optimizado de PyTorch a TensorRT para Jetson Nano"""
    
    def __init__(self):
        # Configurar monitores de recursos
        self.resource_monitor = JetsonResourceMonitor(
            log_interval=15,  # Más frecuente durante conversión
            memory_threshold=80.0,  # Más conservador
            temperature_threshold=70.0  # Más conservador
        )
        self.swap_manager = JetsonSwapManager(swap_size_gb=4)  # Más swap para conversión
        self.cpu_limiter = JetsonCPULimiter()
        
        # Configuración de rutas
        self.model_config = {
            'pytorch_model': '/home/mobilenet/Documentos/Trabajo/trt_pose/models/resnet18_baseline_att_224x224_A_epoch_249.pth',
            'topology_file': '/home/mobilenet/Documentos/Trabajo/trt_pose/tasks/human_pose/human_pose.json',
            'output_model': 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth',
            'backup_model': 'resnet18_baseline_att_224x224_A_epoch_249_trt_backup.pth'
        }
        
        # Configuración de conversión
        self.conversion_config = {
            'width': 224,
            'height': 224,
            'batch_size': 1,
            'fp16_mode': True,
            'max_workspace_size': 1 << 24,  # 16MB (conservador para Jetson)
            'strict_type_constraints': True
        }
        
        self._setup_alert_callbacks()
        
    def _setup_alert_callbacks(self):
        """Configura callbacks para alertas durante conversión"""
        def memory_alert(percent):
            logger.warning("🚨 ALERTA MEMORIA: %.1f%% - Pausando conversión...", percent)
            self._emergency_memory_cleanup()
            time.sleep(5)  # Pausa más larga durante conversión
            
        def temperature_alert(temp):
            logger.warning("🌡️ ALERTA TEMPERATURA: %.1f°C - Pausando para enfriar...", temp)
            time.sleep(10)  # Pausa para enfriamiento
            
        def cpu_alert(percent):
            logger.warning("💻 ALERTA CPU: %.1f%% - Reduciendo carga...", percent)
            
        self.resource_monitor.add_callback('memory_alert', memory_alert)
        self.resource_monitor.add_callback('temperature_alert', temperature_alert)
        self.resource_monitor.add_callback('cpu_alert', cpu_alert)
        
    def _emergency_memory_cleanup(self):
        """Limpieza agresiva de memoria durante conversión"""
        logger.info("🧹 Liberando memoria...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
    def setup_environment(self):
        """Configura el entorno optimizado para conversión"""
        logger.info("🚀 Configurando entorno para conversión TensorRT...")
        
        # Configuraciones específicas para conversión
        setup_jetson_optimizations()
        
        # Variables adicionales para conversión
        conversion_env = {
            'CUDA_VISIBLE_DEVICES': '0',
            'TRT_LOGGER_VERBOSITY': '1',  # Más verboso para conversión
            'CUDA_LAUNCH_BLOCKING': '1',  # Sincronización para debugging
            'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:32'  # Limitar fragmentación
        }
        
        for key, value in conversion_env.items():
            os.environ[key] = value
            logger.info("✅ %s=%s", key, value)
            
        # Configurar swap ampliado para conversión
        if not self.swap_manager.setup_swap():
            logger.warning("⚠️ No se pudo configurar swap automáticamente")
            logger.info("💡 Para configurar manualmente:")
            logger.info("   sudo fallocate -l 4G /swapfile")
            logger.info("   sudo chmod 600 /swapfile") 
            logger.info("   sudo mkswap /swapfile")
            logger.info("   sudo swapon /swapfile")
            
        # Limitar a 1 core durante conversión para evitar sobrecalentamiento
        if not self.cpu_limiter.limit_cpu_cores([0]):
            logger.warning("⚠️ No se pudo limitar CPU")
            
        # Iniciar monitoreo intensivo
        self.resource_monitor.start_monitoring()
        
        logger.info("✅ Entorno configurado para conversión")
        
    def verify_model_files(self):
        """Verifica que los archivos necesarios existan"""
        logger.info("🔍 Verificando archivos del modelo...")
        
        pytorch_model = self.model_config['pytorch_model']
        topology_file = self.model_config['topology_file']
        
        if not os.path.exists(pytorch_model):
            logger.error("❌ Modelo PyTorch no encontrado: %s", pytorch_model)
            return False
            
        if not os.path.exists(topology_file):
            logger.error("❌ Archivo de topología no encontrado: %s", topology_file)
            return False
            
        # Verificar tamaño del modelo
        model_size = os.path.getsize(pytorch_model) / (1024**2)
        logger.info("✅ Modelo PyTorch: %.1f MB", model_size)
        logger.info("✅ Archivo de topología encontrado")
        
        return True
        
    def load_model_configuration(self):
        """Carga la configuración del modelo"""
        logger.info("📋 Cargando configuración del modelo...")
        
        try:
            # Cargar topología
            with open(self.model_config['topology_file'], 'r') as f:
                self.human_pose = json.load(f)
                
            self.topology = trt_pose.coco.coco_category_to_topology(self.human_pose)
            self.num_parts = len(self.human_pose['keypoints'])
            self.num_links = len(self.human_pose['skeleton'])
            
            logger.info("✅ Partes del cuerpo: %d", self.num_parts)
            logger.info("✅ Conexiones: %d", self.num_links)
            
            return True
            
        except Exception as e:
            logger.error("❌ Error cargando configuración: %s", str(e))
            return False
            
    def create_pytorch_model(self):
        """Crea y carga el modelo PyTorch"""
        logger.info("🏗️ Creando modelo PyTorch...")
        
        try:
            # Verificar memoria antes de cargar modelo
            memory = psutil.virtual_memory()
            logger.info("Memoria disponible: %.1f GB", memory.available / (1024**3))
            
            if memory.available < 1024**3:  # Menos de 1GB
                logger.warning("⚠️ Poca memoria disponible, liberando recursos...")
                self._emergency_memory_cleanup()
                
            # Crear modelo
            self.model = trt_pose.models.resnet18_baseline_att(
                self.num_parts, 2 * self.num_links
            ).cuda().eval()
            
            logger.info("✅ Modelo creado en GPU")
            
            # Cargar pesos
            logger.info("📥 Cargando pesos del modelo...")
            checkpoint = torch.load(self.model_config['pytorch_model'], map_location='cuda:0')
            self.model.load_state_dict(checkpoint)
            
            logger.info("✅ Pesos cargados exitosamente")
            
            # Verificar memoria después de cargar
            memory = psutil.virtual_memory()
            logger.info("Memoria tras cargar modelo: %.1f GB disponible", memory.available / (1024**3))
            
            return True
            
        except Exception as e:
            logger.error("❌ Error creando modelo PyTorch: %s", str(e))
            return False
            
    def create_test_input(self):
        """Crea tensor de prueba para conversión"""
        logger.info("🎯 Creando tensor de entrada para conversión...")
        
        try:
            width = self.conversion_config['width']
            height = self.conversion_config['height']
            batch_size = self.conversion_config['batch_size']
            
            # Crear tensor de prueba
            self.test_input = torch.zeros(
                (batch_size, 3, height, width), 
                dtype=torch.float32,
                device='cuda:0'
            )
            
            logger.info("✅ Tensor de entrada: %s", list(self.test_input.shape))
            
            # Verificar que el modelo funciona con la entrada
            logger.info("🧪 Probando inferencia PyTorch...")
            with torch.no_grad():
                output = self.model(self.test_input)
                logger.info("✅ Inferencia PyTorch exitosa")
                logger.info("   Salida: %s", [list(o.shape) for o in output])
                
            return True
            
        except Exception as e:
            logger.error("❌ Error creando tensor de prueba: %s", str(e))
            return False
            
    def convert_to_tensorrt(self):
        """Realiza la conversión a TensorRT con monitoreo"""
        logger.info("⚡ Iniciando conversión PyTorch → TensorRT...")
        
        try:
            # Verificar recursos antes de conversión
            stats = self.resource_monitor.get_current_stats()
            memory_percent = stats.get('memory', {}).get('percent', 0)
            temp = stats.get('temperature', 0)
            
            logger.info("Estado inicial:")
            logger.info("  Memoria: %.1f%%", memory_percent)
            logger.info("  Temperatura: %.1f°C", temp or 0)
            
            if memory_percent > 75:
                logger.warning("⚠️ Memoria alta antes de conversión, liberando...")
                self._emergency_memory_cleanup()
                
            # Configurar parámetros de conversión
            conversion_params = {
                'fp16_mode': self.conversion_config['fp16_mode'],
                'max_workspace_size': self.conversion_config['max_workspace_size'],
                'strict_type_constraints': self.conversion_config['strict_type_constraints']
            }
            
            logger.info("Parámetros de conversión:")
            for key, value in conversion_params.items():
                logger.info("  %s: %s", key, value)
                
            # Realizar conversión con monitoreo
            start_time = time.time()
            
            logger.info("🔄 Ejecutando torch2trt...")
            logger.info("   Esto puede tomar 5-15 minutos en Jetson Nano...")
            
            # Thread para monitoreo durante conversión
            conversion_active = True
            
            def monitor_conversion():
                while conversion_active:
                    stats = self.resource_monitor.get_current_stats()
                    elapsed = time.time() - start_time
                    logger.info("⏱️ Conversión en progreso (%.1f min) - Memoria: %.1f%% - Temp: %.1f°C",
                              elapsed/60, 
                              stats.get('memory', {}).get('percent', 0),
                              stats.get('temperature', 0) or 0)
                    time.sleep(30)  # Log cada 30 segundos
                    
            monitor_thread = threading.Thread(target=monitor_conversion, daemon=True)
            monitor_thread.start()
            
            try:
                # Conversión principal
                self.model_trt = torch2trt.torch2trt(
                    self.model,
                    [self.test_input],
                    **conversion_params
                )
                conversion_active = False
                
                elapsed = time.time() - start_time
                logger.info("✅ Conversión completada en %.1f minutos", elapsed/60)
                
            except Exception as conversion_error:
                conversion_active = False
                raise conversion_error
                
            # Verificar el modelo convertido
            logger.info("🧪 Probando modelo TensorRT...")
            with torch.no_grad():
                trt_output = self.model_trt(self.test_input)
                logger.info("✅ Inferencia TensorRT exitosa")
                
            return True
            
        except Exception as e:
            logger.error("❌ Error durante conversión: %s", str(e))
            return False
            
    def save_tensorrt_model(self):
        """Guarda el modelo TensorRT convertido"""
        logger.info("💾 Guardando modelo TensorRT...")
        
        try:
            output_path = self.model_config['output_model']
            backup_path = self.model_config['backup_model']
            
            # Crear backup si ya existe
            if os.path.exists(output_path):
                logger.info("📋 Creando backup del modelo anterior...")
                import shutil
                shutil.copy2(output_path, backup_path)
                
            # Guardar nuevo modelo
            torch.save(self.model_trt.state_dict(), output_path)
            
            # Verificar archivo guardado
            if os.path.exists(output_path):
                model_size = os.path.getsize(output_path) / (1024**2)
                logger.info("✅ Modelo TensorRT guardado: %s (%.1f MB)", output_path, model_size)
                return True
            else:
                logger.error("❌ Error: archivo no se guardó correctamente")
                return False
                
        except Exception as e:
            logger.error("❌ Error guardando modelo: %s", str(e))
            return False
            
    def verify_converted_model(self):
        """Verifica que el modelo convertido funciona correctamente"""
        logger.info("🔍 Verificando modelo convertido...")
        
        try:
            from torch2trt import TRTModule
            
            # Cargar modelo guardado
            model_verify = TRTModule()
            model_verify.load_state_dict(torch.load(self.model_config['output_model']))
            
            # Probar inferencia
            with torch.no_grad():
                test_output = model_verify(self.test_input)
                
            logger.info("✅ Modelo TensorRT verificado correctamente")
            
            # Comparar rendimiento
            logger.info("⚡ Comparando rendimiento...")
            
            # Benchmark PyTorch
            start_time = time.time()
            for _ in range(10):
                with torch.no_grad():
                    _ = self.model(self.test_input)
            pytorch_time = (time.time() - start_time) / 10
            
            # Benchmark TensorRT
            start_time = time.time()
            for _ in range(10):
                with torch.no_grad():
                    _ = model_verify(self.test_input)
            tensorrt_time = (time.time() - start_time) / 10
            
            speedup = pytorch_time / tensorrt_time if tensorrt_time > 0 else 1
            
            logger.info("📊 Resultados de rendimiento:")
            logger.info("   PyTorch: %.1f ms por inferencia", pytorch_time * 1000)
            logger.info("   TensorRT: %.1f ms por inferencia", tensorrt_time * 1000)
            logger.info("   Aceleración: %.1fx", speedup)
            
            return True
            
        except Exception as e:
            logger.error("❌ Error verificando modelo: %s", str(e))
            return False
            
    def cleanup_resources(self):
        """Limpieza final de recursos"""
        logger.info("🧹 Limpiando recursos...")
        
        try:
            # Limpiar modelos de memoria
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'model_trt'):
                del self.model_trt
            if hasattr(self, 'test_input'):
                del self.test_input
                
            # Limpieza agresiva de memoria
            self._emergency_memory_cleanup()
            
            # Detener monitoreo
            self.resource_monitor.stop_monitoring()
            
            # Restaurar CPU
            self.cpu_limiter.restore_cpu_affinity()
            
            logger.info("✅ Limpieza completada")
            
        except Exception as e:
            logger.error("Error durante limpieza: %s", str(e))
            
    def run_conversion(self):
        """Ejecuta el proceso completo de conversión"""
        logger.info("🚀 Iniciando conversión completa PyTorch → TensorRT")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # 1. Configurar entorno
            if not self.setup_environment():
                return False
                
            # 2. Verificar archivos
            if not self.verify_model_files():
                return False
                
            # 3. Cargar configuración
            if not self.load_model_configuration():
                return False
                
            # 4. Crear modelo PyTorch
            if not self.create_pytorch_model():
                return False
                
            # 5. Crear entrada de prueba
            if not self.create_test_input():
                return False
                
            # 6. Conversión a TensorRT
            if not self.convert_to_tensorrt():
                return False
                
            # 7. Guardar modelo
            if not self.save_tensorrt_model():
                return False
                
            # 8. Verificar modelo
            if not self.verify_converted_model():
                return False
                
            # Estadísticas finales
            total_time = time.time() - start_time
            final_stats = self.resource_monitor.get_stats_summary()
            
            logger.info("🎉 ¡Conversión completada exitosamente!")
            logger.info("=" * 60)
            logger.info("⏱️ Tiempo total: %.1f minutos", total_time / 60)
            logger.info("📁 Modelo TensorRT: %s", self.model_config['output_model'])
            
            if final_stats:
                logger.info("📊 Estadísticas de recursos:")
                logger.info("   CPU promedio: %.1f%%", final_stats['cpu']['average'])
                logger.info("   Memoria máxima: %.1f%%", final_stats['memory']['max'])
                if final_stats['temperature']['max']:
                    logger.info("   Temperatura máxima: %.1f°C", final_stats['temperature']['max'])
                    
            return True
            
        except KeyboardInterrupt:
            logger.info("⏹️ Conversión interrumpida por usuario")
            return False
        except Exception as e:
            logger.error("❌ Error crítico durante conversión: %s", str(e))
            return False
        finally:
            self.cleanup_resources()

def main():
    """Función principal"""
    print("⚡ Convertidor PyTorch → TensorRT - Jetson Nano")
    print("=" * 60)
    
    converter = TensorRTModelConverter()
    
    try:
        success = converter.run_conversion()
        return 0 if success else 1
    except Exception as e:
        logger.error("❌ Error crítico: %s", str(e))
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
