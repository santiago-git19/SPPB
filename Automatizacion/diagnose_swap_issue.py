#!/usr/bin/env python3
"""
Diagnóstico de Swap y Memoria para Jetson Nano
=============================================

Script para diagnosticar por qué el swap no se está usando durante
la conversión TensorRT y recomendar estrategias de solución.

Uso:
    python diagnose_swap_issue.py
    
Autor: Sistema de IA
Fecha: 2025
"""

import os
import sys
import time
import psutil
import subprocess
import json
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SwapDiagnosticTool:
    """Herramienta de diagnóstico para problemas de swap en Jetson"""
    
    def __init__(self):
        self.results = {}
        
    def run_full_diagnostic(self):
        """Ejecuta diagnóstico completo"""
        logger.info("🔍 DIAGNÓSTICO COMPLETO DE SWAP Y MEMORIA - JETSON NANO")
        logger.info("="*60)
        
        self.check_system_memory()
        self.check_swap_configuration()
        self.check_cuda_memory()
        self.test_swap_functionality()
        self.analyze_tensorrt_requirements()
        self.provide_recommendations()
        
        # Guardar resultados
        self.save_diagnostic_report()
        
    def check_system_memory(self):
        """Verifica memoria del sistema"""
        logger.info("\n🧠 1. ANÁLISIS DE MEMORIA DEL SISTEMA")
        
        mem = psutil.virtual_memory()
        self.results['system_memory'] = {
            'total_gb': mem.total / (1024**3),
            'available_gb': mem.available / (1024**3),
            'used_gb': mem.used / (1024**3),
            'percent_used': mem.percent,
            'free_gb': mem.free / (1024**3)
        }
        
        logger.info(f"   Total RAM: {self.results['system_memory']['total_gb']:.2f} GB")
        logger.info(f"   RAM Disponible: {self.results['system_memory']['available_gb']:.2f} GB")
        logger.info(f"   RAM Usada: {self.results['system_memory']['used_gb']:.2f} GB ({mem.percent:.1f}%)")
        logger.info(f"   RAM Libre: {self.results['system_memory']['free_gb']:.2f} GB")
        
        # Análisis
        if self.results['system_memory']['total_gb'] < 4.5:
            logger.warning("   ⚠️ PROBLEMA: Memoria total muy baja para conversión TensorRT")
        elif self.results['system_memory']['available_gb'] < 1.0:
            logger.warning("   ⚠️ PROBLEMA: Poca memoria disponible actualmente")
        else:
            logger.info("   ✅ Memoria del sistema aparenta estar OK")
            
    def check_swap_configuration(self):
        """Verifica configuración de swap"""
        logger.info("\n🔄 2. ANÁLISIS DE CONFIGURACIÓN DE SWAP")
        
        swap = psutil.swap_memory()
        self.results['swap'] = {
            'total_gb': swap.total / (1024**3),
            'used_gb': swap.used / (1024**3),
            'free_gb': (swap.total - swap.used) / (1024**3),
            'percent_used': swap.percent,
            'configured': swap.total > 0
        }
        
        if self.results['swap']['configured']:
            logger.info(f"   ✅ Swap configurado: {self.results['swap']['total_gb']:.2f} GB")
            logger.info(f"   Swap usado actualmente: {self.results['swap']['used_gb']:.2f} GB ({swap.percent:.1f}%)")
            logger.info(f"   Swap disponible: {self.results['swap']['free_gb']:.2f} GB")
            
            # Verificar detalles del swap
            self._check_swap_details()
        else:
            logger.error("   ❌ PROBLEMA: No hay swap configurado")
            
    def _check_swap_details(self):
        """Verifica detalles específicos del swap"""
        try:
            # Verificar archivos de swap activos
            result = subprocess.run(['swapon', '--show'], capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                logger.info("   Archivos de swap activos:")
                for line in result.stdout.strip().split('\n')[1:]:  # Skip header
                    logger.info(f"     {line}")
            
            # Verificar configuración en /proc/swaps
            if os.path.exists('/proc/swaps'):
                with open('/proc/swaps', 'r') as f:
                    content = f.read().strip()
                    if content:
                        logger.info("   Detalles de /proc/swaps:")
                        for line in content.split('\n'):
                            logger.info(f"     {line}")
                            
        except Exception as e:
            logger.warning(f"   ⚠️ No se pudo verificar detalles del swap: {e}")
            
    def check_cuda_memory(self):
        """Verifica memoria CUDA si está disponible"""
        logger.info("\n🎮 3. ANÁLISIS DE MEMORIA CUDA/GPU")
        
        try:
            import torch
            
            if torch.cuda.is_available():
                device_props = torch.cuda.get_device_properties(0)
                allocated = torch.cuda.memory_allocated(0)
                reserved = torch.cuda.memory_reserved(0)
                
                self.results['cuda_memory'] = {
                    'total_gb': device_props.total_memory / (1024**3),
                    'allocated_gb': allocated / (1024**3),
                    'reserved_gb': reserved / (1024**3),
                    'free_gb': (device_props.total_memory - reserved) / (1024**3),
                    'available': True
                }
                
                logger.info(f"   ✅ CUDA disponible: {torch.cuda.get_device_name(0)}")
                logger.info(f"   Memoria CUDA total: {self.results['cuda_memory']['total_gb']:.2f} GB")
                logger.info(f"   Memoria CUDA asignada: {self.results['cuda_memory']['allocated_gb']:.2f} GB")
                logger.info(f"   Memoria CUDA reservada: {self.results['cuda_memory']['reserved_gb']:.2f} GB")
                logger.info(f"   Memoria CUDA libre: {self.results['cuda_memory']['free_gb']:.2f} GB")
                
                # ¡PUNTO CLAVE!
                logger.warning("   🚨 IMPORTANTE: CUDA usa memoria unificada en Jetson")
                logger.warning("   🚨 El swap NO puede extender la memoria CUDA/GPU")
                
            else:
                logger.warning("   ⚠️ CUDA no disponible")
                self.results['cuda_memory'] = {'available': False}
                
        except ImportError:
            logger.warning("   ⚠️ PyTorch no instalado, no se puede verificar CUDA")
            self.results['cuda_memory'] = {'available': False}
            
    def test_swap_functionality(self):
        """Prueba si el swap funciona con operaciones de CPU"""
        logger.info("\n🧪 4. PRUEBA DE FUNCIONALIDAD DE SWAP")
        
        try:
            # Verificar swap inicial
            initial_swap = psutil.swap_memory().used
            
            logger.info("   Realizando prueba de asignación de memoria...")
            logger.info(f"   Swap inicial: {initial_swap / (1024**2):.1f} MB")
            
            # Crear lista grande para forzar uso de memoria (solo CPU)
            test_data = []
            allocated_mb = 0
            max_test_mb = 500  # Máximo 500MB de prueba
            
            try:
                for i in range(50):  # 50 iteraciones
                    # Crear 10MB de datos por iteración
                    chunk = [0] * (10 * 1024 * 1024 // 8)  # 10MB en enteros de 8 bytes
                    test_data.append(chunk)
                    allocated_mb += 10
                    
                    current_swap = psutil.swap_memory().used
                    swap_increase = (current_swap - initial_swap) / (1024**2)
                    
                    if swap_increase > 10:  # Si usa más de 10MB de swap
                        logger.info(f"   ✅ SWAP FUNCIONA: {swap_increase:.1f} MB adicionales usados")
                        break
                    
                    if allocated_mb >= max_test_mb:
                        break
                        
                    time.sleep(0.1)  # Pausa breve
                    
            finally:
                # Limpiar datos de prueba
                del test_data
                import gc
                gc.collect()
                
            final_swap = psutil.swap_memory().used
            total_swap_used = (final_swap - initial_swap) / (1024**2)
            
            if total_swap_used > 5:
                logger.info(f"   ✅ Swap funcionó correctamente (+{total_swap_used:.1f} MB)")
                self.results['swap_test'] = {'functional': True, 'increase_mb': total_swap_used}
            else:
                logger.warning(f"   ⚠️ Swap no se usó significativamente (+{total_swap_used:.1f} MB)")
                self.results['swap_test'] = {'functional': False, 'increase_mb': total_swap_used}
                
        except Exception as e:
            logger.error(f"   ❌ Error en prueba de swap: {e}")
            self.results['swap_test'] = {'functional': False, 'error': str(e)}
            
    def analyze_tensorrt_requirements(self):
        """Analiza requerimientos específicos de TensorRT"""
        logger.info("\n⚡ 5. ANÁLISIS DE REQUERIMIENTOS TENSORRT")
        
        # Estimar memoria necesaria para conversión
        estimated_pytorch_model_mb = 50   # Modelo ResNet18 ~50MB
        estimated_conversion_overhead = 3  # 3x overhead durante conversión
        estimated_total_mb = estimated_pytorch_model_mb * estimated_conversion_overhead
        
        logger.info(f"   Modelo PyTorch estimado: ~{estimated_pytorch_model_mb} MB")
        logger.info(f"   Overhead de conversión: ~{estimated_conversion_overhead}x")
        logger.info(f"   Memoria total estimada: ~{estimated_total_mb} MB")
        
        # Verificar si hay suficiente memoria disponible
        available_ram_mb = self.results['system_memory']['available_gb'] * 1024
        
        if 'cuda_memory' in self.results and self.results['cuda_memory'].get('available'):
            available_cuda_mb = self.results['cuda_memory']['free_gb'] * 1024
            logger.info(f"   Memoria CUDA disponible: {available_cuda_mb:.0f} MB")
            
            if available_cuda_mb < estimated_total_mb:
                logger.warning("   ⚠️ PROBLEMA: Posible insuficiencia de memoria CUDA")
                logger.warning("   💡 CUDA no usa swap - conversión puede fallar")
            else:
                logger.info("   ✅ Memoria CUDA aparenta ser suficiente")
        
        logger.info(f"   Memoria RAM disponible: {available_ram_mb:.0f} MB")
        
        # Análisis del problema principal
        logger.warning("\n   🎯 DIAGNÓSTICO CLAVE:")
        logger.warning("   1. TensorRT/CUDA opera en memoria unificada (GPU)")
        logger.warning("   2. Swap solo funciona para operaciones de CPU")
        logger.warning("   3. Durante torch2trt, los picos de memoria ocurren en GPU")
        logger.warning("   4. Por eso el swap no se usa aunque esté configurado")
        
    def provide_recommendations(self):
        """Proporciona recomendaciones específicas"""
        logger.info("\n💡 6. RECOMENDACIONES ESPECÍFICAS")
        
        ram_gb = self.results['system_memory']['total_gb']
        swap_configured = self.results['swap']['configured']
        cuda_available = self.results.get('cuda_memory', {}).get('available', False)
        
        logger.info("   ESTRATEGIAS RECOMENDADAS:")
        
        # Estrategia 1: CPU Fallback
        logger.info("   🔄 ESTRATEGIA 1: CPU Fallback (Recomendado)")
        logger.info("     - Intentar conversión GPU primero")
        logger.info("     - Si falla por OOM → usar CPU con swap")
        logger.info("     - CPU SÍ usa swap efectivamente")
        logger.info("     - Tiempo: 15-30 min vs 5-15 min GPU")
        
        # Estrategia 2: Optimización de parámetros
        logger.info("   ⚙️ ESTRATEGIA 2: Optimizar Parámetros")
        logger.info("     - Reducir max_workspace_size a 8-16MB")
        logger.info("     - Usar fp16_mode=True")
        logger.info("     - Cerrar aplicaciones innecesarias")
        
        # Estrategia 3: Conversión externa
        if ram_gb < 4.5:
            logger.info("   🌐 ESTRATEGIA 3: Conversión Externa")
            logger.info("     - Convertir en máquina con >6GB RAM")
            logger.info("     - Transferir modelo convertido a Jetson")
            logger.info("     - Más rápido y confiable")
        
        # Recomendación final
        if not swap_configured:
            logger.error("   ❌ ACCIÓN INMEDIATA: Configurar swap de 4GB")
        elif ram_gb < 4.5:
            logger.warning("   ⚠️ RECOMENDACIÓN: Usar conversión externa o CPU fallback")
        else:
            logger.info("   ✅ CONFIGURACIÓN: Implementar CPU fallback en convertidor")
            
    def save_diagnostic_report(self):
        """Guarda reporte de diagnóstico"""
        report = {
            'timestamp': time.time(),
            'system_info': {
                'os': os.uname().sysname if hasattr(os, 'uname') else 'Unknown',
                'python_version': sys.version
            },
            'diagnostic_results': self.results,
            'recommendations': self._get_structured_recommendations()
        }
        
        report_file = Path('swap_diagnostic_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"\n📋 Reporte guardado en: {report_file.absolute()}")
        
    def _get_structured_recommendations(self):
        """Obtiene recomendaciones estructuradas"""
        recommendations = []
        
        if not self.results['swap']['configured']:
            recommendations.append({
                'priority': 'HIGH',
                'action': 'Configure swap',
                'command': 'sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile'
            })
            
        if self.results['system_memory']['total_gb'] < 4.5:
            recommendations.append({
                'priority': 'MEDIUM',
                'action': 'Use external conversion',
                'description': 'Convert model on machine with >6GB RAM'
            })
            
        recommendations.append({
            'priority': 'HIGH',
            'action': 'Implement CPU fallback',
            'description': 'Modify converter to use CPU when GPU OOM occurs'
        })
        
        return recommendations

def main():
    """Función principal"""
    logger.info("🔧 Iniciando diagnóstico de swap para Jetson Nano...")
    
    try:
        diagnostic = SwapDiagnosticTool()
        diagnostic.run_full_diagnostic()
        
        logger.info("\n" + "="*60)
        logger.info("✅ Diagnóstico completado")
        logger.info("📋 Revisa el archivo 'swap_diagnostic_report.json' para detalles")
        logger.info("💡 Implementa las recomendaciones para solucionar el problema")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ Diagnóstico interrumpido por usuario")
        return 1
    except Exception as e:
        logger.error(f"❌ Error durante diagnóstico: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
