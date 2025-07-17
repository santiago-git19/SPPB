#!/usr/bin/env python3
"""
TensorRT Pose Estimation Optimizado para Jetson Nano
====================================================

Script optimizado para procesamiento de video con pose estimation en Jetson Nano,
incluyendo:
- Monitoreo automático de recursos (CPU, RAM, GPU, temperatura)
- Manejo automático de swap
- Limitación de uso de CPU y memoria
- Reporte de progreso detallado
- Manejo robusto de errores por falta de memoria

Autor: Sistema de IA
Fecha: 2025
"""

import json
import torch
import cv2
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

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from utils.jetson_utils import (
    JetsonResourceMonitor, 
    JetsonSwapManager, 
    JetsonCPULimiter, 
    setup_jetson_optimizations
)

class OptimizedTRTPoseProcessor:
    """Procesador TRT Pose optimizado para Jetson Nano"""
    
    def __init__(self):
        # Configurar monitores usando las utilidades de Jetson
        self.resource_monitor = JetsonResourceMonitor(
            log_interval=30,
            memory_threshold=85.0,
            temperature_threshold=75.0
        )
        self.swap_manager = JetsonSwapManager(swap_size_gb=2)
        self.cpu_limiter = JetsonCPULimiter()
        self.processor = None
        
        # Configuración de rutas (ajustar según instalación)
        self.config_file = Path("trt_pose_config.json")
        
        # Cargar configuración si existe
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                self.model_paths = config.get("model_paths", {})
                self.jetson_config = config.get("jetson_config", {})
        else:
            # Configuración por defecto
            self.model_paths = {
                'topology': '/home/mobilenet/Documentos/Trabajo/trt_pose/tasks/human_pose/human_pose.json',
                'pytorch_model': '/home/mobilenet/Documentos/Trabajo/trt_pose/models/resnet18_baseline_att_224x224_A_epoch_249.pth',
                'tensorrt_model': 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
            }
            self.jetson_config = {
                'enable_swap': True,
                'swap_size_gb': 2,
                'max_cpu_cores': 2,
                'memory_limit_percent': 85,
                'temperature_limit': 75
            }
        
        # Configurar callbacks para alertas
        self._setup_alert_callbacks()
        
    def _setup_alert_callbacks(self):
        """Configura callbacks para alertas de recursos"""
        def memory_alert(percent):
            logger.warning("🚨 ALERTA MEMORIA: %.1f%% - Liberando recursos...", percent)
            self._emergency_memory_cleanup()
            
        def temperature_alert(temp):
            logger.warning("🌡️  ALERTA TEMPERATURA: %.1f°C - Reduciendo carga...", temp)
            self._reduce_processing_load()
            
        def cpu_alert(percent):
            logger.warning("💻 ALERTA CPU: %.1f%% - Optimizando...", percent)
            
        self.resource_monitor.add_callback('memory_alert', memory_alert)
        self.resource_monitor.add_callback('temperature_alert', temperature_alert)
        self.resource_monitor.add_callback('cpu_alert', cpu_alert)
        
    def _emergency_memory_cleanup(self):
        """Limpieza de emergencia de memoria"""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def _reduce_processing_load(self):
        """Reduce la carga de procesamiento temporalmente"""
        # Podría implementar skip de frames o reducción de resolución
        logger.info("Implementando reducción de carga...")
        time.sleep(2)  # Pausa breve para enfriar
        
    def setup_environment(self):
        """Configura el entorno para Jetson Nano"""
        logger.info("🚀 Configurando entorno para Jetson Nano...")
        
        # Usar utilidades de configuración optimizada
        setup_jetson_optimizations()
        
        # Configurar swap si está habilitado
        if self.jetson_config.get('enable_swap', True):
            swap_size = self.jetson_config.get('swap_size_gb', 2)
            self.swap_manager = JetsonSwapManager(swap_size_gb=swap_size)
            if not self.swap_manager.setup_swap():
                logger.warning("No se pudo configurar swap automáticamente")
                
        # Limitar cores de CPU si está configurado
        max_cores = self.jetson_config.get('max_cpu_cores', 2)
        if max_cores > 0:
            cores = list(range(min(max_cores, os.cpu_count())))
            if not self.cpu_limiter.limit_cpu_cores(cores):
                logger.warning("No se pudo limitar cores de CPU")
            
        # Iniciar monitoreo
        self.resource_monitor.start_monitoring()
        
    def setup_model(self):
        """Configura el modelo TensorRT"""
        logger.info("📊 Configurando modelo de pose estimation...")
        
        try:
            # Verificar si existe modelo TensorRT optimizado
            tensorrt_path = self.model_paths.get('tensorrt_model', '')
            pytorch_path = self.model_paths.get('pytorch_model', '')
            topology_path = self.model_paths.get('topology', '')
            
            if os.path.exists(tensorrt_path):
                logger.info("Usando modelo TensorRT preoptimizado")
                use_tensorrt = True
                model_path = tensorrt_path
            elif os.path.exists(pytorch_path):
                logger.info("Modelo TensorRT no encontrado, usando PyTorch")
                use_tensorrt = False
                model_path = pytorch_path
            else:
                logger.error("No se encontró ningún modelo válido")
                return False
                
            # Verificar archivo de topología
            if not os.path.exists(topology_path):
                logger.error(f"Archivo de topología no encontrado: {topology_path}")
                return False
                
            # Importar y inicializar procesador
            from utils.trt_pose_proc import TRTPoseProcessor
            
            self.processor = TRTPoseProcessor(
                model_path=model_path,
                topology_path=topology_path,
                use_tensorrt=use_tensorrt
            )
            
            logger.info("✅ Modelo configurado exitosamente")
            return True
            
        except Exception as e:
            logger.error("❌ Error configurando modelo: %s", str(e))
            return False
            
    def _setup_video_capture(self, input_path):
        """Configura la captura de video"""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logger.error("❌ No se pudo abrir el video: %s", input_path)
            return None, None
            
        # Obtener propiedades del video
        video_info = {
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': cap.get(cv2.CAP_PROP_FPS) or 15,
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        }
        
        logger.info("Video: %dx%d @ %.1ffps, %d frames", 
                   video_info['width'], video_info['height'], 
                   video_info['fps'], video_info['total_frames'])
        
        return cap, video_info
        
    def _setup_video_writer(self, output_path, video_info):
        """Configura el escritor de video"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(
            output_path, fourcc, video_info['fps'], 
            (video_info['width'], video_info['height'])
        )
        
    def _process_single_frame(self, frame, frame_idx):
        """Procesa un frame individual"""
        try:
            # Verificar presión de memoria cada 50 frames
            if frame_idx % 50 == 0:
                stats = self.resource_monitor.get_current_stats()
                memory_percent = stats.get('memory', {}).get('percent', 0)
                if memory_percent > 85:
                    logger.warning("⚠️  Presión de memoria: %.1f%%, liberando recursos...", memory_percent)
                    self._emergency_memory_cleanup()
                    
            # Procesar frame
            keypoints = self.processor.process_frame(frame)
            if keypoints is not None:
                frame = self.processor.visualize_keypoints(frame, keypoints, draw_skeleton=True)
                
            return frame, True
            
        except Exception as e:
            logger.error("Error procesando frame %d: %s", frame_idx, str(e))
            return frame, False
            
    def _log_progress(self, frame_idx, total_frames, start_time, last_progress_time):
        """Registra el progreso del procesamiento"""
        current_time = time.time()
        
        if current_time - last_progress_time >= 10:  # Cada 10 segundos
            elapsed = current_time - start_time
            fps_actual = frame_idx / elapsed if elapsed > 0 else 0
            progress = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
            eta = (total_frames - frame_idx) / fps_actual if fps_actual > 0 else 0
            
            logger.info("🎯 Progreso: %d/%d (%.1f%%) - FPS: %.1f - ETA: %.1fs",
                       frame_idx, total_frames, progress, fps_actual, eta)
            return current_time
            
        return last_progress_time
        
    def process_video(self, input_path, output_path, max_frames=None):
        """
        Procesa video con monitoreo de recursos
        
        Args:
            input_path: Ruta del video de entrada
            output_path: Ruta del video de salida
            max_frames: Máximo número de frames a procesar (None = todos)
        """
        logger.info("🎬 Iniciando procesamiento de video: %s", input_path)
        
        # Configurar captura de video
        cap, video_info = self._setup_video_capture(input_path)
        if cap is None:
            return False
            
        # Ajustar total de frames si hay límite
        total_frames = video_info['total_frames']
        if max_frames:
            total_frames = min(total_frames, max_frames)
            
        # Configurar video de salida
        out = self._setup_video_writer(output_path, video_info)
        
        # Variables de control
        frame_idx = 0
        start_time = time.time()
        last_progress_time = start_time
        
        try:
            while frame_idx < total_frames:
                # Leer frame
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Procesar frame
                processed_frame, success = self._process_single_frame(frame, frame_idx)
                
                # Escribir frame (procesado o original)
                out.write(processed_frame)
                
                # Actualizar progreso
                frame_idx += 1
                last_progress_time = self._log_progress(
                    frame_idx, total_frames, start_time, last_progress_time
                )
                
                # Verificar límite
                if max_frames and frame_idx >= max_frames:
                    break
                    
        except KeyboardInterrupt:
            logger.info("⏹️  Procesamiento interrumpido por usuario")
        except Exception as e:
            logger.error("❌ Error durante procesamiento: %s", str(e))
        finally:
            # Liberar recursos
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            
        # Estadísticas finales
        self._log_final_stats(frame_idx, start_time, output_path)
        return True
        
    def _log_final_stats(self, frame_idx, start_time, output_path):
        """Registra estadísticas finales"""
        total_time = time.time() - start_time
        avg_fps = frame_idx / total_time if total_time > 0 else 0
        
        logger.info("✅ Procesamiento completado:")
        logger.info("   Frames procesados: %d", frame_idx)
        logger.info("   Tiempo total: %.1fs", total_time)
        logger.info("   FPS promedio: %.1f", avg_fps)
        logger.info("   Video guardado: %s", output_path)
        """
        Procesa video con monitoreo de recursos
        
        Args:
            input_path: Ruta del video de entrada
            output_path: Ruta del video de salida
            max_frames: Máximo número de frames a procesar (None = todos)
        """
        logger.info(f"🎬 Iniciando procesamiento de video: {input_path}")
        
        # Abrir video de entrada
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logger.error(f"❌ No se pudo abrir el video: {input_path}")
            return False
            
        # Obtener propiedades del video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 15
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
            
        logger.info(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Configurar video de salida
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Variables de control
        frame_idx = 0
        start_time = time.time()
        last_progress_time = start_time
        
        try:
            while frame_idx < total_frames:
                # Verificar presión de memoria
                if self.resource_monitor.check_memory_pressure():
                    logger.warning("⚠️  Presión de memoria detectada, liberando recursos...")
                    self.resource_monitor.force_garbage_collection()
                    
                # Leer frame
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Procesar frame
                try:
                    keypoints = self.processor.process_frame(frame)
                    if keypoints is not None:
                        frame = self.processor.visualize_keypoints(frame, keypoints, draw_skeleton=True)
                except Exception as e:
                    logger.error(f"Error procesando frame {frame_idx}: {e}")
                    # Continuar con frame original si hay error
                    
                # Escribir frame
                out.write(frame)
                
                # Mostrar progreso
                frame_idx += 1
                current_time = time.time()
                
                if current_time - last_progress_time >= 10:  # Cada 10 segundos
                    elapsed = current_time - start_time
                    fps_actual = frame_idx / elapsed if elapsed > 0 else 0
                    progress = (frame_idx / total_frames) * 100
                    eta = (total_frames - frame_idx) / fps_actual if fps_actual > 0 else 0
                    
                    logger.info(f"🎯 Progreso: {frame_idx}/{total_frames} ({progress:.1f}%) "
                              f"- FPS: {fps_actual:.1f} - ETA: {eta:.1f}s")
                    last_progress_time = current_time
                    
                # Verificar si se alcanzó el límite
                if max_frames and frame_idx >= max_frames:
                    break
                    
        except KeyboardInterrupt:
            logger.info("⏹️  Procesamiento interrumpido por usuario")
        except Exception as e:
            logger.error(f"❌ Error durante procesamiento: {e}")
        finally:
            # Liberar recursos
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            
        # Estadísticas finales
        total_time = time.time() - start_time
        avg_fps = frame_idx / total_time if total_time > 0 else 0
        
        logger.info(f"✅ Procesamiento completado:")
        logger.info(f"   Frames procesados: {frame_idx}")
        logger.info(f"   Tiempo total: {total_time:.1f}s")
        logger.info(f"   FPS promedio: {avg_fps:.1f}")
        logger.info(f"   Video guardado: {output_path}")
        
        return True
        
    def cleanup(self):
        """Limpieza de recursos"""
        logger.info("🧹 Limpiando recursos...")
        self.resource_monitor.stop_monitoring()
        self.resource_monitor.force_garbage_collection()

def main():
    """Función principal"""
    logger.info("🤖 TensorRT Pose Estimation - Jetson Nano Optimizado")
    logger.info("=" * 60)
    
    # Configuración de archivos
    input_video = 'Automatizacion/WIN_20250702_12_09_08_Pro.mp4'
    output_video = 'video_con_exoesqueleto_optimizado.mp4'
    
    # Verificar archivo de entrada
    if not os.path.exists(input_video):
        logger.error(f"❌ Archivo de entrada no encontrado: {input_video}")
        return 1
        
    # Inicializar procesador
    processor = OptimizedTRTPoseProcessor()
    
    try:
        # Configurar entorno
        processor.setup_environment()
        
        # Configurar modelo
        if not processor.setup_model():
            logger.error("❌ No se pudo configurar el modelo")
            return 1
            
        # Procesar video
        success = processor.process_video(
            input_path=input_video,
            output_path=output_video,
            max_frames=None  # Procesar todo el video, cambiar a número específico si se desea
        )
        
        if success:
            logger.info("🎉 Procesamiento completado exitosamente!")
            return 0
        else:
            logger.error("❌ Error durante el procesamiento")
            return 1
            
    except KeyboardInterrupt:
        logger.info("⏹️  Programa interrumpido por usuario")
        return 1
    except Exception as e:
        logger.error(f"❌ Error crítico: {e}")
        return 1
    finally:
        processor.cleanup()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
