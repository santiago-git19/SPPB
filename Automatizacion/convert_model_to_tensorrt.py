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
        
        # Configuración de conversión optimizada para memoria limitada
        self.conversion_config = {
            'width': 224,
            'height': 224,
            'batch_size': 1,
            'fp16_mode': True,
            'max_workspace_size': 1 << 20,  # 1MB (ultra conservador para 930MB RAM)
            'strict_type_constraints': True,
            'int8_mode': False,  # FP16 es suficiente para Jetson Nano
            'minimum_segment_size': 3,  # Fusionar solo segmentos grandes
            'max_batch_size': 1,  # Máximo batch size
            'optimize_for_memory': True  # Priorizar memoria sobre velocidad
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
        """Realiza la conversión a TensorRT con fallback automático CPU"""
        logger.info("⚡ Iniciando conversión PyTorch → TensorRT...")
        
        # Diagnóstico inicial de memoria
        self._diagnose_memory_limitations()
        
        try:
            # Intentar conversión GPU primero
            logger.info("🎯 Intentando conversión GPU...")
            return self._convert_gpu()
            
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower():
                logger.warning("💾 OOM en GPU detectado, usando fallback CPU...")
                logger.warning("   Esto usará swap efectivamente pero será más lento...")
                return self._convert_cpu_with_swap()
            else:
                logger.error("❌ Error no relacionado con memoria: %s", str(e))
                raise e
        except Exception as e:
            logger.error("❌ Error inesperado durante conversión: %s", str(e))
            return False
    
    def _diagnose_memory_limitations(self):
        """Diagnóstica limitaciones específicas de memoria"""
        logger.info("🔍 DIAGNÓSTICO DE MEMORIA JETSON NANO:")
        
        # Memoria total del sistema
        mem_info = psutil.virtual_memory()
        total_mem_gb = mem_info.total / (1024**3)
        available_mem_gb = mem_info.available / (1024**3)
        logger.info(f"   💾 RAM Total: {total_mem_gb:.1f} GB")
        logger.info(f"   💾 RAM Disponible: {available_mem_gb:.1f} GB")
        
        # Memoria CUDA disponible
        if torch.cuda.is_available():
            total_cuda_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            allocated_cuda_gb = torch.cuda.memory_allocated(0) / (1024**3)
            cached_cuda_gb = torch.cuda.memory_reserved(0) / (1024**3)
            free_cuda_gb = total_cuda_gb - cached_cuda_gb
            
            logger.info(f"   🎮 CUDA Total: {total_cuda_gb:.1f} GB")
            logger.info(f"   🎮 CUDA Asignada: {allocated_cuda_gb:.1f} GB")
            logger.info(f"   🎮 CUDA Cached: {cached_cuda_gb:.1f} GB")
            logger.info(f"   🎮 CUDA Libre: {free_cuda_gb:.1f} GB")
        
        # Estado del swap
        swap_info = psutil.swap_memory()
        if swap_info.total > 0:
            swap_total_gb = swap_info.total / (1024**3)
            swap_used_gb = swap_info.used / (1024**3)
            swap_free_gb = (swap_info.total - swap_info.used) / (1024**3)
            logger.info(f"   🔄 Swap Total: {swap_total_gb:.1f} GB")
            logger.info(f"   🔄 Swap Usado: {swap_used_gb:.1f} GB")
            logger.info(f"   🔄 Swap Libre: {swap_free_gb:.1f} GB")
        else:
            logger.warning("   ⚠️ Sin swap configurado")
        
        # Recomendar estrategia
        if available_mem_gb < 1.5:
            logger.warning("💡 MEMORIA BAJA: Recomendado usar conversión CPU con swap")
        elif free_cuda_gb < 0.5:
            logger.warning("💡 GPU MEMORY BAJA: Conversión puede fallar, CPU fallback disponible")
        else:
            logger.info("💡 Memoria aparenta ser suficiente para conversión GPU")
    
    def _convert_gpu(self):
        """Conversión estándar usando GPU"""
        try:
            # Verificar recursos antes de conversión
            stats = self.resource_monitor.get_current_stats()
            memory_percent = stats.get('memory', {}).get('percent', 0)
            temp = stats.get('temperature', 0)
            
            logger.info("Estado inicial GPU:")
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
            
            logger.info("Parámetros de conversión GPU:")
            for key, value in conversion_params.items():
                logger.info("  %s: %s", key, value)
                
            # Realizar conversión con monitoreo
            start_time = time.time()
            
            logger.info("🔄 Ejecutando torch2trt en GPU...")
            logger.info("   Esto puede tomar 5-15 minutos en Jetson Nano...")
            
            # Thread para monitoreo durante conversión
            conversion_active = True
            
            def monitor_conversion():
                while conversion_active:
                    stats = self.resource_monitor.get_current_stats()
                    swap_info = psutil.swap_memory()
                    elapsed = time.time() - start_time
                    
                    logger.info("⏱️ Conversión GPU (%.1f min) - Memoria: %.1f%% - Temp: %.1f°C - Swap: %.1f%%",
                              elapsed/60, 
                              stats.get('memory', {}).get('percent', 0),
                              stats.get('temperature', 0) or 0,
                              swap_info.percent if swap_info.total > 0 else 0)
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
                logger.info("✅ Conversión GPU completada en %.1f minutos", elapsed/60)
                
            except Exception as conversion_error:
                conversion_active = False
                raise conversion_error
                
            # Verificar el modelo convertido
            logger.info("🧪 Probando modelo TensorRT...")
            with torch.no_grad():
                trt_output = self.model_trt(self.test_input)
                logger.info("✅ Inferencia TensorRT exitosa")
                
            return True
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error("💥 GPU Out of Memory: %s", str(e))
            raise e
        except Exception as e:
            logger.error("❌ Error durante conversión GPU: %s", str(e))
            raise e
    
    def _convert_cpu_with_swap(self):
        """Conversión en CPU que SÍ usa swap efectivamente"""
        logger.info("🔄 INICIANDO CONVERSIÓN CPU (usa swap)...")
        logger.warning("   ⏰ Esto será más lento (15-30 min) pero más estable")
        
        try:
            # Limpiar GPU memory primero
            self._emergency_memory_cleanup()
            
            # Mover modelo y datos a CPU
            logger.info("📤 Moviendo modelo a CPU...")
            self.model = self.model.cpu()
            self.test_input = self.test_input.cpu()
            
            # Más limpieza después de mover a CPU
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            
            # Configurar parámetros para CPU
            conversion_params = {
                'fp16_mode': False,  # CPU no soporta FP16
                'max_workspace_size': self.conversion_config['max_workspace_size'] * 2,  # Más workspace en CPU
                'strict_type_constraints': self.conversion_config['strict_type_constraints']
            }
            
            logger.info("Parámetros de conversión CPU:")
            for key, value in conversion_params.items():
                logger.info("  %s: %s", key, value)
            
            # Monitoreo específico para conversión CPU
            start_time = time.time()
            conversion_active = True
            
            def monitor_cpu_conversion():
                initial_swap = psutil.swap_memory().used
                while conversion_active:
                    stats = self.resource_monitor.get_current_stats()
                    swap_info = psutil.swap_memory()
                    elapsed = time.time() - start_time
                    swap_increase = (swap_info.used - initial_swap) / (1024**2)  # MB
                    
                    logger.info("⏱️ Conversión CPU (%.1f min) - RAM: %.1f%% - Swap: %.1f%% (+%.0f MB) - Temp: %.1f°C",
                              elapsed/60,
                              stats.get('memory', {}).get('percent', 0),
                              swap_info.percent if swap_info.total > 0 else 0,
                              swap_increase,
                              stats.get('temperature', 0) or 0)
                    
                    if swap_increase > 100:  # Si está usando swap significativamente
                        logger.info("✅ Swap siendo usado efectivamente")
                    
                    time.sleep(45)  # Más frecuente para CPU
                    
            monitor_thread = threading.Thread(target=monitor_cpu_conversion, daemon=True)
            monitor_thread.start()
            
            # Conversión en CPU
            logger.info("🔄 Ejecutando torch2trt en CPU...")
            try:
                self.model_trt = torch2trt.torch2trt(
                    self.model,
                    [self.test_input],
                    **conversion_params
                )
                conversion_active = False
                
                elapsed = time.time() - start_time
                logger.info("✅ Conversión CPU completada en %.1f minutos", elapsed/60)
                
            except Exception as conversion_error:
                conversion_active = False
                raise conversion_error
            
            # Verificar modelo en CPU
            logger.info("🧪 Probando modelo TensorRT en CPU...")
            with torch.no_grad():
                trt_output = self.model_trt(self.test_input)
                logger.info("✅ Inferencia TensorRT CPU exitosa")
            
            # Mover modelo final de vuelta a GPU para guardado
            logger.info("📥 Moviendo modelo convertido a GPU para guardado...")
            try:
                self.model_trt = self.model_trt.cuda()
                logger.info("✅ Modelo movido a GPU exitosamente")
            except Exception as e:
                logger.warning("⚠️ No se pudo mover a GPU, guardando en CPU: %s", str(e))
            
            return True
            
        except Exception as e:
            logger.error("❌ Error durante conversión CPU: %s", str(e))
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
    
    def _setup_aggressive_memory_management(self):
        """Configuración agresiva de memoria para Jetson Nano con poca RAM"""
        logger.info("🛠️ Configurando gestión agresiva de memoria...")
        
        # Variables de entorno para optimización de memoria
        memory_env = {
            'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:16,garbage_collection_threshold:0.6',
            'CUDA_LAUNCH_BLOCKING': '1',  # Sincronización para evitar acumulación
            'TF_GPU_ALLOCATOR': 'cuda_malloc_async',  # Allocator optimizado
            'TRT_LOGGER_VERBOSITY': '0',  # Reducir verbosidad para ahorrar memoria
            'CUBLAS_WORKSPACE_CONFIG': ':16:8',  # Limitar workspace de cuBLAS
        }
        
        for key, value in memory_env.items():
            os.environ[key] = value
            logger.info(f"✅ {key}={value}")
            
        # Configurar PyTorch para uso mínimo de memoria
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False  # Evitar caching de kernels
            torch.backends.cudnn.deterministic = True
            
            # Limitar memoria CUDA disponible
            available_ram_mb = psutil.virtual_memory().available / (1024**2)
            # Usar máximo 70% de RAM disponible para CUDA
            cuda_memory_limit = min(int(available_ram_mb * 0.7), 1024)  # Max 1GB
            
            try:
                torch.cuda.set_per_process_memory_fraction(cuda_memory_limit / 1024.0)
                logger.info(f"✅ Límite CUDA: {cuda_memory_limit} MB")
            except Exception as e:
                logger.warning(f"⚠️ No se pudo limitar memoria CUDA: {e}")
                
    def _pre_conversion_memory_optimization(self):
        """Optimizaciones de memoria antes de iniciar conversión"""
        logger.info("🧹 Optimización de memoria pre-conversión...")
        
        # Limpiar cachés del sistema
        try:
            subprocess.run(['sudo', 'sync'], check=False)
            subprocess.run(['sudo', 'sh', '-c', 'echo 3 > /proc/sys/vm/drop_caches'], check=False)
            logger.info("✅ Cachés del sistema liberados")
        except Exception as e:
            logger.warning(f"⚠️ No se pudieron liberar cachés del sistema: {e}")
            
        # Garbage collection agresivo
        for _ in range(3):
            gc.collect()
            
        # Limpieza CUDA agresiva
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            
        # Verificar memoria después de limpieza
        memory = psutil.virtual_memory()
        logger.info(f"💾 Memoria disponible tras limpieza: {memory.available / (1024**2):.0f} MB")
        
    def _adaptive_workspace_size(self):
        """Calcula el workspace size óptimo basado en memoria disponible"""
        available_ram_mb = psutil.virtual_memory().available / (1024**2)
        
        # Usar máximo 1% de RAM disponible para workspace, con límites
        workspace_mb = max(1, min(4, int(available_ram_mb * 0.01)))
        workspace_bytes = workspace_mb * 1024 * 1024
        
        logger.info(f"🎯 Workspace calculado: {workspace_mb} MB ({available_ram_mb:.0f} MB disponibles)")
        
        # Actualizar configuración
        self.conversion_config['max_workspace_size'] = workspace_bytes
        return workspace_bytes
        
    def _validate_memory_requirements(self):
        """Valida si hay suficiente memoria para la conversión"""
        memory = psutil.virtual_memory()
        available_mb = memory.available / (1024**2)
        
        # Estimar memoria necesaria
        model_size_mb = 50  # ResNet18 base ~50MB
        conversion_overhead = 4  # Factor de overhead durante conversión
        minimum_required_mb = model_size_mb * conversion_overhead
        
        logger.info(f"📊 Análisis de memoria:")
        logger.info(f"   Disponible: {available_mb:.0f} MB")
        logger.info(f"   Requerido estimado: {minimum_required_mb:.0f} MB")
        
        if available_mb < minimum_required_mb:
            logger.error(f"❌ Memoria insuficiente: {available_mb:.0f} < {minimum_required_mb:.0f} MB")
            return False
            
        # Verificar swap si está disponible
        swap = psutil.swap_memory()
        if swap.total > 0:
            total_virtual_mb = available_mb + (swap.free / (1024**2))
            logger.info(f"   Total virtual (RAM+Swap): {total_virtual_mb:.0f} MB")
            
        return True
        
    def _create_model_with_memory_monitoring(self):
        """Crea modelo con monitoreo intensivo de memoria"""
        logger.info("🏗️ Creando modelo con monitoreo de memoria...")
        
        # Monitorear memoria antes
        initial_memory = psutil.virtual_memory()
        logger.info(f"💾 Memoria inicial: {initial_memory.available / (1024**2):.0f} MB disponible")
        
        try:
            # Crear modelo paso a paso para detectar problemas de memoria
            logger.info("📐 Creando arquitectura del modelo...")
            self.model = trt_pose.models.resnet18_baseline_att(
                self.num_parts, 2 * self.num_links
            )
            
            # Verificar memoria después de crear arquitectura
            arch_memory = psutil.virtual_memory()
            arch_used = (initial_memory.available - arch_memory.available) / (1024**2)
            logger.info(f"💾 Memoria usada por arquitectura: {arch_used:.1f} MB")
            
            # Mover a GPU con monitoreo
            logger.info("🎮 Moviendo modelo a GPU...")
            self.model = self.model.cuda()
            
            gpu_memory = psutil.virtual_memory()
            gpu_used = (arch_memory.available - gpu_memory.available) / (1024**2)
            logger.info(f"💾 Memoria usada por GPU transfer: {gpu_used:.1f} MB")
            
            # Poner en modo evaluación
            self.model.eval()
            
            # Cargar pesos con monitoreo
            logger.info("📥 Cargando pesos...")
            checkpoint = torch.load(
                self.model_config['pytorch_model'], 
                map_location='cuda:0',
                weights_only=True  # Más seguro y eficiente
            )
            
            self.model.load_state_dict(checkpoint)
            del checkpoint  # Liberar inmediatamente
            gc.collect()
            
            # Verificar memoria final
            final_memory = psutil.virtual_memory()
            total_used = (initial_memory.available - final_memory.available) / (1024**2)
            logger.info(f"💾 Memoria total usada por modelo: {total_used:.1f} MB")
            logger.info(f"💾 Memoria restante: {final_memory.available / (1024**2):.0f} MB")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creando modelo: {e}")
            # Limpieza en caso de error
            if hasattr(self, 'model'):
                del self.model
            torch.cuda.empty_cache()
            gc.collect()
            return False
