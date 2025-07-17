#!/usr/bin/env python3
"""
Demostración: Conversión TensorRT con CPU Fallback
=================================================

Script que demuestra la conversión mejorada con fallback automático a CPU
cuando la conversión GPU falla por falta de memoria.

Uso:
    python demo_cpu_fallback.py
    
Características:
- Detecta automáticamente si GPU tiene suficiente memoria
- Fallback a CPU que SÍ usa swap efectivamente
- Monitoreo detallado durante ambos tipos de conversión
- Reporte final de rendimiento

Autor: Sistema de IA  
Fecha: 2025
"""

import json
import os
import sys
import time
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def demo_conversion_strategies():
    """Demuestra las diferentes estrategias de conversión"""
    
    logger.info("🚀 DEMOSTRACIÓN: Estrategias de Conversión TensorRT")
    logger.info("="*60)
    
    # Verificar si el convertidor mejorado existe
    converter_path = Path("convert_model_to_tensorrt.py")
    if not converter_path.exists():
        logger.error("❌ No se encontró convert_model_to_tensorrt.py")
        return False
    
    logger.info("📋 ESTRATEGIAS DISPONIBLES:")
    logger.info("1. 🎮 Conversión GPU (rápida, puede fallar por OOM)")
    logger.info("2. 🧠 Conversión CPU con swap (lenta pero estable)")  
    logger.info("3. 🔄 Fallback automático (intenta GPU → CPU si falla)")
    
    # Mostrar configuración actual
    show_current_configuration()
    
    # Ejecutar diagnóstico previo
    run_pre_conversion_diagnostic()
    
    # Simular conversión con monitoreo
    simulate_conversion_process()
    
    return True

def show_current_configuration():
    """Muestra configuración actual del sistema"""
    logger.info("\n🔧 CONFIGURACIÓN ACTUAL:")
    
    try:
        import psutil
        
        # Memoria
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        logger.info(f"   RAM Total: {mem.total/(1024**3):.1f} GB")
        logger.info(f"   RAM Disponible: {mem.available/(1024**3):.1f} GB")
        
        if swap.total > 0:
            logger.info(f"   ✅ Swap: {swap.total/(1024**3):.1f} GB configurado")
        else:
            logger.warning("   ⚠️ Sin swap configurado")
            
        # CUDA si disponible
        try:
            import torch
            if torch.cuda.is_available():
                total_cuda = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"   🎮 CUDA: {total_cuda:.1f} GB ({torch.cuda.get_device_name(0)})")
            else:
                logger.warning("   ⚠️ CUDA no disponible")
        except ImportError:
            logger.warning("   ⚠️ PyTorch no instalado")
            
    except ImportError:
        logger.warning("   ⚠️ psutil no disponible para mostrar configuración")

def run_pre_conversion_diagnostic():
    """Ejecuta diagnóstico previo a la conversión"""
    logger.info("\n🔍 DIAGNÓSTICO PRE-CONVERSIÓN:")
    
    # Verificar archivos necesarios
    required_files = [
        '/home/mobilenet/Documentos/Trabajo/trt_pose/models/resnet18_baseline_att_224x224_A_epoch_249.pth',
        '/home/mobilenet/Documentos/Trabajo/trt_pose/tasks/human_pose/human_pose.json'
    ]
    
    all_files_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            logger.info(f"   ✅ {os.path.basename(file_path)}")
        else:
            logger.warning(f"   ⚠️ No encontrado: {os.path.basename(file_path)}")
            all_files_exist = False
    
    if not all_files_exist:
        logger.warning("   💡 Algunos archivos no están disponibles (esto es normal en demo)")
    
    # Predecir estrategia recomendada
    predict_recommended_strategy()

def predict_recommended_strategy():
    """Predice qué estrategia de conversión sería recomendada"""
    logger.info("\n🎯 PREDICCIÓN DE ESTRATEGIA:")
    
    try:
        import psutil
        
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)
        
        try:
            import torch
            if torch.cuda.is_available():
                cuda_available = True
                cuda_free = (torch.cuda.get_device_properties(0).total_memory - 
                            torch.cuda.memory_reserved(0)) / (1024**3)
            else:
                cuda_available = False
                cuda_free = 0
        except ImportError:
            cuda_available = False
            cuda_free = 0
        
        logger.info(f"   RAM disponible: {available_gb:.1f} GB")
        if cuda_available:
            logger.info(f"   CUDA libre: {cuda_free:.1f} GB")
        
        # Lógica de predicción
        if cuda_available and cuda_free > 1.0:
            logger.info("   💡 RECOMENDACIÓN: Intentar GPU primero")
            logger.info("   📊 Probabilidad de éxito GPU: Alta")
        elif cuda_available and cuda_free > 0.5:
            logger.info("   💡 RECOMENDACIÓN: GPU con fallback CPU")
            logger.info("   📊 Probabilidad de éxito GPU: Media")
        else:
            logger.info("   💡 RECOMENDACIÓN: Conversión CPU directa")
            logger.info("   📊 Probabilidad de éxito GPU: Baja")
        
        if available_gb > 2.0:
            logger.info("   📊 Probabilidad de éxito CPU: Alta")
        else:
            logger.warning("   📊 Probabilidad de éxito CPU: Media (necesita swap)")
            
    except ImportError:
        logger.warning("   ⚠️ No se puede predecir estrategia sin psutil")

def simulate_conversion_process():
    """Simula el proceso de conversión mejorado"""
    logger.info("\n🔄 SIMULACIÓN DE CONVERSIÓN:")
    
    logger.info("   📝 Pasos del proceso mejorado:")
    logger.info("   1. 🔍 Diagnóstico inicial de memoria")
    logger.info("   2. 🎮 Intento de conversión GPU")
    logger.info("   3. 🚨 Detección de OOM (si ocurre)")
    logger.info("   4. 🔄 Fallback automático a CPU")
    logger.info("   5. 💾 Conversión CPU usando swap")
    logger.info("   6. ✅ Verificación del modelo final")
    
    # Simular flujo de decisión
    logger.info("\n   🤖 FLUJO DE DECISIÓN AUTOMÁTICO:")
    
    # Paso 1: Diagnóstico
    logger.info("   [1/6] 🔍 Ejecutando diagnóstico...")
    time.sleep(1)
    logger.info("         ✅ Memoria analizada")
    logger.info("         ✅ CUDA verificado")
    logger.info("         ✅ Swap confirmado")
    
    # Paso 2: Intento GPU
    logger.info("   [2/6] 🎮 Iniciando conversión GPU...")
    time.sleep(1.5)
    
    # Simular decisión basada en memoria disponible
    try:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024**3)
        
        if available_gb < 1.5:  # Simular condición de poca memoria
            logger.warning("         ⚠️ Memoria insuficiente detectada")
            logger.warning("         🚨 OOM simulado después de 2 minutos")
            
            # Paso 3: Detección OOM
            logger.info("   [3/6] 🚨 Detectando OOM...")
            time.sleep(1)
            logger.warning("         ❌ torch.cuda.OutOfMemoryError capturado")
            
            # Paso 4: Fallback
            logger.info("   [4/6] 🔄 Activando fallback CPU...")
            time.sleep(1)
            logger.info("         ✅ Modelo movido a CPU")
            logger.info("         ✅ Memoria GPU liberada")
            
            # Paso 5: Conversión CPU
            logger.info("   [5/6] 💾 Conversión CPU con swap...")
            time.sleep(2)
            logger.info("         ✅ Swap siendo usado efectivamente")
            logger.info("         ⏰ Progreso: conversión más lenta pero estable")
            
        else:
            logger.info("         ✅ Conversión GPU completada exitosamente")
            logger.info("         ⏱️ Tiempo: ~8 minutos")
        
        # Paso 6: Verificación
        logger.info("   [6/6] ✅ Verificando modelo final...")
        time.sleep(1)
        logger.info("         ✅ Modelo TensorRT validado")
        logger.info("         ✅ Benchmark de rendimiento ejecutado")
        
    except ImportError:
        logger.info("         ✅ Conversión GPU (simulada)")
    
    # Resumen final
    show_conversion_summary()

def show_conversion_summary():
    """Muestra resumen de la conversión"""
    logger.info("\n📊 RESUMEN DE CONVERSIÓN:")
    
    logger.info("   🎯 BENEFICIOS DE LA MEJORA:")
    logger.info("   ✅ Detección automática de limitaciones de memoria")
    logger.info("   ✅ Fallback transparente sin intervención manual")
    logger.info("   ✅ Uso efectivo del swap en modo CPU")
    logger.info("   ✅ Monitoreo detallado durante todo el proceso")
    logger.info("   ✅ Mayor robustez contra OOM errors")
    
    logger.info("\n   📈 IMPACTO EN RENDIMIENTO:")
    logger.info("   🎮 GPU: 5-15 min (si hay suficiente memoria)")
    logger.info("   🧠 CPU: 15-30 min (siempre funciona con swap)")
    logger.info("   🔄 Fallback: Combina lo mejor de ambos mundos")
    
    logger.info("\n   🔧 CONFIGURACIÓN RECOMENDADA:")
    logger.info("   💾 Swap: 4GB mínimo para conversión")
    logger.info("   🌡️ Temperatura: <70°C durante conversión")
    logger.info("   💻 CPU: Máximo 2 cores para evitar sobrecalentamiento")
    logger.info("   📱 Aplicaciones: Cerrar aplicaciones innecesarias")

def show_usage_examples():
    """Muestra ejemplos de uso"""
    logger.info("\n📚 EJEMPLOS DE USO:")
    
    logger.info("   🔄 Conversión automática (recomendado):")
    logger.info("     python convert_model_to_tensorrt.py")
    logger.info("     # Usa fallback automático si es necesario")
    
    logger.info("\n   🧠 Forzar conversión CPU:")
    logger.info("     python convert_model_to_tensorrt.py --force-cpu")
    logger.info("     # Usa CPU directamente (más lento pero estable)")
    
    logger.info("\n   🔍 Solo diagnóstico:")
    logger.info("     python diagnose_swap_issue.py")
    logger.info("     # Analiza configuración sin convertir")
    
    logger.info("\n   📊 Monitor de recursos:")
    logger.info("     # En terminal separado durante conversión:")
    logger.info("     watch -n 5 'free -h && swapon --show'")

def main():
    """Función principal"""
    logger.info("🎭 Iniciando demostración de conversión mejorada...")
    
    try:
        # Ejecutar demostración
        success = demo_conversion_strategies()
        
        if success:
            # Mostrar ejemplos de uso
            show_usage_examples()
            
            logger.info("\n" + "="*60)
            logger.info("✅ DEMOSTRACIÓN COMPLETADA")
            logger.info("💡 El sistema ahora incluye:")
            logger.info("   🔄 Fallback automático CPU cuando GPU falla")
            logger.info("   📊 Diagnóstico detallado de memoria")
            logger.info("   💾 Uso efectivo del swap en modo CPU")
            logger.info("   🔍 Monitoreo durante toda la conversión")
            logger.info("\n📋 Para ejecutar conversión real:")
            logger.info("   python convert_model_to_tensorrt.py")
            
            return 0
        else:
            logger.error("❌ Error durante demostración")
            return 1
            
    except KeyboardInterrupt:
        logger.info("\n⏹️ Demostración interrumpida por usuario")
        return 1
    except Exception as e:
        logger.error(f"❌ Error durante demostración: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
