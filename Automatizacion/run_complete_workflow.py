#!/usr/bin/env python3
"""
Flujo Completo TensorRT Pose - Jetson Nano
==========================================

Script maestro que ejecuta todo el flujo de trabajo:
1. Verificar dependencias
2. Configurar swap automáticamente  
3. Convertir modelo PyTorch → TensorRT (si es necesario)
4. Procesar video con monitoreo completo de recursos
5. Generar reporte final

Uso:
    python run_complete_workflow.py [--video VIDEO] [--frames N] [--force-convert]
    
Opciones:
    --video VIDEO       Especificar video de entrada
    --frames N          Procesar solo N frames
    --force-convert     Forzar reconversión del modelo
    --setup-only        Solo configurar entorno y modelos
    
Autor: Sistema de IA
Fecha: 2025
"""

import os
import sys
import argparse
import subprocess
import logging
import time
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CompleteWorkflow:
    """Flujo de trabajo completo para TensorRT Pose en Jetson Nano"""
    
    def __init__(self):
        self.workflow_start_time = time.time()
        self.steps_completed = []
        self.errors_encountered = []
        
    def log_step(self, step_name, success=True, details=""):
        """Registra el resultado de cada paso"""
        if success:
            self.steps_completed.append(step_name)
            logger.info("✅ %s completado %s", step_name, details)
        else:
            self.errors_encountered.append((step_name, details))
            logger.error("❌ %s falló: %s", step_name, details)
            
    def run_subprocess(self, command, description, timeout=None):
        """Ejecuta subproceso con manejo de errores"""
        logger.info("🔄 %s...", description)
        
        try:
            result = subprocess.run(
                command, 
                capture_output=True, 
                text=True, 
                timeout=timeout
            )
            
            if result.returncode == 0:
                self.log_step(description, True)
                return True, result.stdout
            else:
                self.log_step(description, False, result.stderr)
                return False, result.stderr
                
        except subprocess.TimeoutExpired:
            self.log_step(description, False, f"Timeout ({timeout}s)")
            return False, f"Timeout después de {timeout} segundos"
        except Exception as e:
            self.log_step(description, False, str(e))
            return False, str(e)
            
    def step1_verify_dependencies(self):
        """Paso 1: Verificar y descargar dependencias"""
        logger.info("📦 PASO 1: Verificando dependencias...")
        
        # Verificar si existe el descargador
        if not Path('download_models_v2.py').exists():
            logger.error("❌ Script de descarga no encontrado")
            return False
            
        # Ejecutar descargador
        success, output = self.run_subprocess(
            [sys.executable, 'download_models_v2.py'],
            "Descarga de dependencias",
            timeout=600  # 10 minutos
        )
        
        return success
        
    def step2_check_models(self):
        """Paso 2: Verificar estado de modelos"""
        logger.info("🔍 PASO 2: Verificando modelos...")
        
        if not Path('model_manager.py').exists():
            logger.error("❌ Gestor de modelos no encontrado")
            return False
            
        success, output = self.run_subprocess(
            [sys.executable, 'model_manager.py', '--check'],
            "Verificación de modelos",
            timeout=60
        )
        
        return success
        
    def step3_convert_model(self, force_convert=False):
        """Paso 3: Convertir modelo si es necesario"""
        logger.info("⚡ PASO 3: Conversión de modelo...")
        
        # Verificar si ya existe modelo TensorRT
        tensorrt_model = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
        
        if os.path.exists(tensorrt_model) and not force_convert:
            logger.info("✅ Modelo TensorRT ya existe, omitiendo conversión")
            self.log_step("Conversión de modelo (omitida)", True, "- modelo existente")
            return True
            
        # Verificar script de conversión
        if not Path('convert_model_to_tensorrt.py').exists():
            logger.error("❌ Script de conversión no encontrado")
            return False
            
        # Ejecutar conversión
        logger.info("🔄 Iniciando conversión PyTorch → TensorRT...")
        logger.info("   ⏱️ Tiempo estimado: 5-15 minutos")
        logger.info("   🌡️ Monitoreo automático de temperatura activado")
        
        success, output = self.run_subprocess(
            [sys.executable, 'convert_model_to_tensorrt.py'],
            "Conversión PyTorch → TensorRT",
            timeout=1800  # 30 minutos
        )
        
        if success:
            # Verificar que el archivo se creó
            if os.path.exists(tensorrt_model):
                size_mb = os.path.getsize(tensorrt_model) / (1024**2)
                self.log_step("Conversión de modelo", True, f"- {size_mb:.1f} MB")
            else:
                self.log_step("Conversión de modelo", False, "archivo no creado")
                return False
                
        return success
        
    def step4_validate_model(self):
        """Paso 4: Validar modelo TensorRT"""
        logger.info("🧪 PASO 4: Validando modelo TensorRT...")
        
        success, output = self.run_subprocess(
            [sys.executable, 'model_manager.py', '--validate'],
            "Validación de modelo TensorRT",
            timeout=120
        )
        
        return success
        
    def step5_process_video(self, video_path=None, max_frames=None):
        """Paso 5: Procesar video con pose estimation"""
        logger.info("🎬 PASO 5: Procesando video...")
        
        # Verificar script principal
        if not Path('example_trt_pose_final.py').exists():
            logger.error("❌ Script principal no encontrado")
            return False
            
        # Determinar video de entrada
        if video_path:
            input_video = video_path
        else:
            input_video = 'WIN_20250702_12_09_08_Pro.mp4'
            
        # Verificar que existe el video
        if not os.path.exists(input_video):
            logger.error("❌ Video de entrada no encontrado: %s", input_video)
            return False
            
        # Configurar variables de entorno si es necesario
        env = os.environ.copy()
        if video_path:
            env['INPUT_VIDEO'] = video_path
        if max_frames:
            env['MAX_FRAMES'] = str(max_frames)
            
        # Ejecutar procesamiento
        logger.info("🔄 Iniciando procesamiento de video...")
        logger.info("   📹 Video: %s", input_video)
        if max_frames:
            logger.info("   🎞️ Frames: %d", max_frames)
        logger.info("   📊 Monitoreo de recursos activado")
        
        try:
            # Ejecutar sin timeout para permitir procesamiento completo
            result = subprocess.run(
                [sys.executable, 'example_trt_pose_final.py'],
                env=env,
                text=True
            )
            
            if result.returncode == 0:
                self.log_step("Procesamiento de video", True)
                return True
            else:
                self.log_step("Procesamiento de video", False, "error en procesamiento")
                return False
                
        except Exception as e:
            self.log_step("Procesamiento de video", False, str(e))
            return False
            
    def generate_final_report(self):
        """Genera reporte final del flujo de trabajo"""
        total_time = time.time() - self.workflow_start_time
        
        logger.info("=" * 60)
        logger.info("📊 REPORTE FINAL DEL FLUJO DE TRABAJO")
        logger.info("=" * 60)
        logger.info("⏱️ Tiempo total: %.1f minutos", total_time / 60)
        
        # Pasos completados
        if self.steps_completed:
            logger.info("✅ Pasos completados (%d):", len(self.steps_completed))
            for i, step in enumerate(self.steps_completed, 1):
                logger.info("   %d. %s", i, step)
        
        # Errores encontrados
        if self.errors_encountered:
            logger.info("❌ Errores encontrados (%d):", len(self.errors_encountered))
            for i, (step, error) in enumerate(self.errors_encountered, 1):
                logger.info("   %d. %s: %s", i, step, error)
        
        # Estado final
        if not self.errors_encountered:
            logger.info("🎉 ¡FLUJO COMPLETADO EXITOSAMENTE!")
            logger.info("📁 Busca el video procesado en el directorio actual")
            return True
        else:
            logger.info("⚠️ Flujo completado con errores")
            return False
            
    def run_complete_workflow(self, video_path=None, max_frames=None, force_convert=False, setup_only=False):
        """Ejecuta el flujo de trabajo completo"""
        logger.info("🚀 INICIANDO FLUJO COMPLETO TENSORRT POSE")
        logger.info("🤖 Jetson Nano - Optimizado para recursos limitados")
        logger.info("=" * 60)
        
        success = True
        
        try:
            # Paso 1: Dependencias
            if not self.step1_verify_dependencies():
                success = False
                
            # Paso 2: Verificar modelos
            if success and not self.step2_check_models():
                # No es crítico, continuar
                logger.warning("⚠️ Verificación de modelos falló, continuando...")
                
            # Paso 3: Conversión de modelo
            if success and not self.step3_convert_model(force_convert):
                success = False
                
            # Paso 4: Validar modelo
            if success and not self.step4_validate_model():
                # No es crítico si el modelo existe
                logger.warning("⚠️ Validación falló, intentando procesar anyway...")
                
            # Si solo configurar, parar aquí
            if setup_only:
                logger.info("✅ Configuración completada")
                return self.generate_final_report()
                
            # Paso 5: Procesar video
            if success and not self.step5_process_video(video_path, max_frames):
                success = False
                
        except KeyboardInterrupt:
            logger.info("⏹️ Flujo interrumpido por usuario")
            success = False
        except Exception as e:
            logger.error("❌ Error crítico en flujo: %s", str(e))
            success = False
            
        return self.generate_final_report()

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description='Flujo completo TensorRT Pose para Jetson Nano',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--video', type=str,
                       help='Video de entrada personalizado')
    parser.add_argument('--frames', type=int,
                       help='Número máximo de frames a procesar')
    parser.add_argument('--force-convert', action='store_true',
                       help='Forzar reconversión del modelo')
    parser.add_argument('--setup-only', action='store_true',
                       help='Solo configurar entorno y modelos')
    
    args = parser.parse_args()
    
    # Verificar directorio de trabajo
    if not Path('example_trt_pose_final.py').exists():
        logger.error("❌ Scripts no encontrados")
        logger.error("   Asegúrate de estar en el directorio correcto")
        return 1
        
    # Ejecutar flujo
    workflow = CompleteWorkflow()
    
    success = workflow.run_complete_workflow(
        video_path=args.video,
        max_frames=args.frames,
        force_convert=args.force_convert,
        setup_only=args.setup_only
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
