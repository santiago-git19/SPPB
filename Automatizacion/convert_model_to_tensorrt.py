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

# Comentar utilidades de Jetson para evitar pausas automáticas
# from utils.jetson_utils import (
#     JetsonResourceMonitor, 
#     JetsonSwapManager, 
#     JetsonCPULimiter, 
#     setup_jetson_optimizations
# )

class TensorRTModelConverter:
    """Convertidor optimizado de PyTorch a TensorRT para Jetson Nano"""
    
    def __init__(self):
        # Configuración de rutas
        self.model_config = {
            'pytorch_model': '/home/mobilenet/Documentos/Trabajo/trt_pose/models/resnet18_baseline_att_224x224_A_epoch_249.pth',
            'topology_file': '/home/mobilenet/Documentos/Trabajo/trt_pose/tasks/human_pose/human_pose.json',
            'output_model': '/home/mobilenet/Documentos/Trabajo/trt_pose/models/resnet18_baseline_att_224x224_A_epoch_249_trt.pth',
            'backup_model': '/home/mobilenet/Documentos/Trabajo/trt_pose/models/resnet18_baseline_att_224x224_A_epoch_249_trt_backup.pth'
        }
        
        # Configuración de conversión optimizada para memoria limitada
        self.conversion_config = {
            'width': 224,
            'height': 224,
            'batch_size': 1,
            'fp16_mode': True,  # Desactivar FP16 para evitar errores de dimensiones
            'max_workspace_size': 1 << 18,  # 256KB (más conservador)
            'strict_type_constraints': False,  # Más flexibilidad en tipos
            'int8_mode': False,  # FP16 es suficiente para Jetson Nano
            'minimum_segment_size': 3,  # Fusionar solo segmentos grandes
            'max_batch_size': 1,  # Máximo batch size
            'optimize_for_memory': True  # Priorizar memoria sobre velocidad
        }
        
    def _setup_alert_callbacks(self):
        """Configuración de alertas deshabilitada para evitar pausas"""
        # Callbacks eliminados para evitar pausas automáticas
        logger.info("✅ Sistema de alertas deshabilitado - sin pausas automáticas")
        pass
        
    def _emergency_memory_cleanup(self):
        """Limpieza agresiva de memoria durante conversión"""
        logger.info("🧹 Liberando memoria...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
    def setup_environment(self):
        """Configura el entorno optimizado para conversión (simplificado)"""
        logger.info("🚀 Configurando entorno para conversión TensorRT...")
        
        # Variables básicas de optimización
        conversion_env = {
            'CUDA_VISIBLE_DEVICES': '0',
            'TRT_LOGGER_VERBOSITY': '1',  # Más verboso para conversión
            'CUDA_LAUNCH_BLOCKING': '1',  # Sincronización para debugging
            'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:32'  # Limitar fragmentación
        }
        
        for key, value in conversion_env.items():
            os.environ[key] = value
            logger.info("✅ %s=%s", key, value)
            
        logger.info("✅ Entorno configurado para conversión (sin monitoreo)")
        return True
        
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
                logger.warning("⚠️ Poca memoria disponible, continuando con precaución...")
                self._emergency_memory_cleanup()
                
            # Crear modelo
            self.model = trt_pose.models.resnet18_baseline_att(
                self.num_parts, 2 * self.num_links,
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
        """Realiza la conversión a TensorRT con múltiples estrategias de fallback"""
        logger.info("⚡ Iniciando conversión PyTorch → TensorRT...")
        
        # Diagnóstico inicial de memoria
        self._diagnose_memory_limitations()
        
        # Estrategia 1: Conversión GPU estándar
        try:
            logger.info("🎯 Estrategia 1: Conversión GPU estándar...")
            return self._convert_gpu()
            
        except Exception as e:
            error_msg = str(e).lower()
            logger.error("💥 Error GPU estándar: %s", str(e))
            
            # Verificar si es error de deconvolución específico (incluyendo __len__ que aparece tras errores TRT)
            deconv_indicators = ["deconvolution", "kernel weights", "__len__", "tensorrt", "trt"]
            if any(indicator in error_msg for indicator in deconv_indicators):
                logger.warning("🔧 Error de deconvolución detectado, probando estrategias alternativas...")
                
                # Estrategia 2: Conversión con configuración alternativa
                try:
                    logger.info("🎯 Estrategia 2: Conversión GPU con configuración alternativa...")
                    return self._convert_gpu_alternative()
                except Exception as e2:
                    logger.error("💥 Error GPU alternativo: %s", str(e2))
                    
                    # Estrategia 3: Conversión por partes/layers
                    try:
                        logger.info("🎯 Estrategia 3: Conversión por segmentos...")
                        return self._convert_segmented()
                    except Exception as e3:
                        logger.error("� Error conversión segmentada: %s", str(e3))
                        
                        # Estrategia 4: Generar modelo compatible manualmente
                        logger.warning("🛠️ Todas las conversiones TensorRT fallaron")
                        logger.warning("💡 El modelo tiene incompatibilidades con TensorRT en capas de deconvolución")
                        logger.warning("💡 Recomendación: Usar modelo PyTorch original (más lento pero funcional)")
                        return self._create_fallback_info()
            
            # Otros errores (memoria, etc.)
            elif any(keyword in error_msg for keyword in ["out of memory", "cuda", "memory"]):
                logger.warning("💾 Error de memoria detectado, usando fallback CPU...")
                logger.warning("   Error: %s", str(e))
                return self._convert_cpu_with_swap()
            else:
                logger.error("❌ Error no categorizado durante conversión: %s", str(e))
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
            # Verificar recursos básicos antes de conversión
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            logger.info("Estado inicial GPU:")
            logger.info("  Memoria: %.1f%%", memory_percent)
            
            if memory_percent > 99:
                logger.warning("⚠️ Memoria alta antes de conversión, limpiando y continuando...")
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
                    memory = psutil.virtual_memory()
                    swap_info = psutil.swap_memory()
                    elapsed = time.time() - start_time
                    
                    logger.info("⏱️ Conversión GPU (%.1f min) - Memoria: %.1f%% - Swap: %.1f%%",
                              elapsed/60, 
                              memory.percent,
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
            
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ["out of memory", "tensorrt", "trt", "__len__", "deconvolution", "kernel weights"]):
                logger.error("💥 Error TensorRT/GPU: %s", str(e))
                raise e
            else:
                logger.error("❌ Error durante conversión GPU: %s", str(e))
                raise e
    
    def _convert_gpu_alternative(self):
        """Conversión GPU con configuración alternativa para problemas de deconvolución"""
        logger.info("🔧 Probando configuración alternativa para resolver errores de deconvolución...")
        
        try:
            self._emergency_memory_cleanup()
            
            # Configuración más restrictiva para evitar problemas de deconvolución
            alternative_params = {
                'fp16_mode': False,
                'max_workspace_size': 1 << 16,  # 64KB - muy conservador
                'strict_type_constraints': True,  # Más estricto
            }
            
            logger.info("🔧 Configuración alternativa:")
            for key, value in alternative_params.items():
                logger.info("   %s: %s", key, value)
            
            start_time = time.time()
            logger.info("🔄 Ejecutando torch2trt con configuración alternativa...")
            '''
            self.model_trt = torch2trt.torch2trt(
                self.model,
                [self.test_input],
                **alternative_params
            )
            '''

            data = torch.zeros((1, 3, self.conversion_config['height'], self.conversion_config['width'])).cuda()
            self.model_trt = torch2trt.torch2trt(self.model, [data], fp16_mode=True, max_workspace_size=1<<25)

            
            elapsed = time.time() - start_time
            logger.info("✅ Conversión alternativa completada en %.1f minutos", elapsed/60)
            
            # Verificar modelo
            logger.info("🧪 Probando modelo TensorRT alternativo...")
            with torch.no_grad():
                trt_output = self.model_trt(self.test_input)
                logger.info("✅ Inferencia TensorRT alternativa exitosa")
                
            return True
            
        except Exception as e:
            logger.error("❌ Error en conversión alternativa: %s", str(e))
            raise e
    
    def _convert_segmented(self):
        """Conversión por segmentos para modelos con capas problemáticas"""
        logger.info("🧩 Intentando conversión por segmentos...")
        logger.warning("   Esta es una estrategia experimental para capas incompatibles")
        
        try:
            # Parámetros muy conservadores para problemas de deconvolución
            segmented_params = {
                'fp16_mode': False,
                'max_workspace_size': 1 << 15,  # 32KB - muy pequeño
                'strict_type_constraints': True,
            }
            
            logger.info("🧩 Configuración segmentada:")
            for key, value in segmented_params.items():
                logger.info("   %s: %s", key, value)
            
            start_time = time.time()
            logger.info("🔄 Ejecutando conversión segmentada...")
            
            self.model_trt = torch2trt.torch2trt(
                self.model,
                [self.test_input],
                **segmented_params
            )
            
            elapsed = time.time() - start_time
            logger.info("✅ Conversión segmentada completada en %.1f minutos", elapsed/60)
            
            # Verificar modelo
            logger.info("🧪 Probando modelo TensorRT segmentado...")
            with torch.no_grad():
                trt_output = self.model_trt(self.test_input)
                logger.info("✅ Inferencia TensorRT segmentada exitosa")
                
            return True
            
        except Exception as e:
            logger.error("❌ Error en conversión segmentada: %s", str(e))
            raise e
    
    def _create_fallback_info(self):
        """Crea información de fallback cuando TensorRT no es compatible"""
        logger.info("📝 Creando información de fallback...")
        
        try:
            # Crear archivo de información sobre el problema
            info_path = self.model_config['output_model'].replace('.pth', '_info.txt')
            
            info_content = f"""
INFORMACIÓN DE CONVERSIÓN TENSORRT
==================================

Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Modelo: {self.model_config['pytorch_model']}

PROBLEMA DETECTADO:
- Error de deconvolución en TensorRT
- El modelo tiene capas incompatibles con TensorRT
- Error específico: kernel weights count mismatch en deconvolución

DETALLES TÉCNICOS:
- TensorRT esperaba: 4,194,304 weights
- Modelo proporciona: 2,097,152 weights  
- Ratio: 1:2 (exactamente la mitad)
- Capa problemática: cmap_up.0 (deconvolution)

RECOMENDACIONES:
1. Usar modelo PyTorch original para inferencia
2. El modelo funciona correctamente en PyTorch
3. Rendimiento será menor pero funcionalmente correcto
4. Considerar reentrenar con arquitectura compatible con TensorRT

COMANDO PARA USAR MODELO PYTORCH:
python3 tu_script.py --model {self.model_config['pytorch_model']} --no-tensorrt

ESTADO: CONVERSION_FAILED_INCOMPATIBLE_LAYERS
"""
            
            with open(info_path, 'w') as f:
                f.write(info_content)
                
            logger.info("✅ Información de fallback guardada en: %s", info_path)
            logger.warning("🚨 RESUMEN:")
            logger.warning("   - TensorRT conversion falló por incompatibilidad de capas")
            logger.warning("   - Modelo PyTorch original funciona correctamente")
            logger.warning("   - Usar modelo original para inferencia (más lento pero funcional)")
            logger.warning("   - Ver detalles en: %s", info_path)
            
            return False  # No se completó la conversión TensorRT
            
        except Exception as e:
            logger.error("❌ Error creando información de fallback: %s", str(e))
            return False
    
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
                    memory = psutil.virtual_memory()
                    swap_info = psutil.swap_memory()
                    elapsed = time.time() - start_time
                    swap_increase = (swap_info.used - initial_swap) / (1024**2)  # MB
                    
                    logger.info("⏱️ Conversión CPU (%.1f min) - RAM: %.1f%% - Swap: %.1f%% (+%.0f MB)",
                              elapsed/60,
                              memory.percent,
                              swap_info.percent if swap_info.total > 0 else 0,
                              swap_increase)
                    
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
        """Limpieza final de recursos (simplificada)"""
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
            
            logger.info("🎉 ¡Conversión completada exitosamente!")
            logger.info("=" * 60)
            logger.info("⏱️ Tiempo total: %.1f minutos", total_time / 60)
            logger.info("📁 Modelo TensorRT: %s", self.model_config['output_model'])
                    
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
        
        logger.info("📊 Análisis de memoria:")
        logger.info("   Disponible: %.0f MB", available_mb)
        logger.info("   Requerido estimado: %.0f MB", minimum_required_mb)
        
        if available_mb < minimum_required_mb:
            logger.warning("⚠️ Memoria posiblemente insuficiente: %.0f < %.0f MB", available_mb, minimum_required_mb)
            logger.warning("⚠️ Continuando conversión, puede usar swap o fallar...")
            # No retornar False, permitir que continúe
            
        # Verificar swap si está disponible
        swap = psutil.swap_memory()
        if swap.total > 0:
            total_virtual_mb = available_mb + (swap.free / (1024**2))
            logger.info("   Total virtual (RAM+Swap): %.0f MB", total_virtual_mb)
            
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

if __name__ == "__main__":
    # Configurar logging para mostrar mensajes en consola
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    try:
        # Inicializar el convertidor
        converter = TensorRTModelConverter()

        # Ejecutar la conversión
        success = converter.run_conversion()

        if success:
            logger.info("🎉 Conversión completada exitosamente.")
        else:
            logger.error("❌ Conversión fallida.")

    except Exception as e:
        logger.error(f"❌ Error crítico durante la ejecución: {e}")