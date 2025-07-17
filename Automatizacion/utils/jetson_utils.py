#!/usr/bin/env python3
"""
Utilidades de Monitoreo para Jetson Nano
========================================

Herramientas específicas para monitorear y optimizar recursos en Jetson Nano:
- Monitor de recursos en tiempo real
- Gestor de swap automático
- Limitador de CPU
- Monitor de temperatura
- Liberación de memoria automática

Autor: Sistema de IA
Fecha: 2025
"""

import os
import sys
import time
import psutil
import subprocess
import threading
import logging
from pathlib import Path
import json
import signal
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class JetsonResourceMonitor:
    """Monitor de recursos optimizado para Jetson Nano"""
    
    def __init__(self, 
                 log_interval: int = 30,
                 memory_threshold: float = 85.0,
                 temperature_threshold: float = 75.0,
                 cpu_threshold: float = 90.0):
        """
        Inicializa el monitor de recursos
        
        Args:
            log_interval: Intervalo de logging en segundos
            memory_threshold: Umbral de memoria para alertas (%)
            temperature_threshold: Umbral de temperatura para alertas (°C)
            cpu_threshold: Umbral de CPU para alertas (%)
        """
        self.log_interval = log_interval
        self.memory_threshold = memory_threshold
        self.temperature_threshold = temperature_threshold
        self.cpu_threshold = cpu_threshold
        
        self.monitoring = False
        self.monitor_thread = None
        self.start_time = time.time()
        self.stats_history = []
        
        # Callbacks para alertas
        self.callbacks = {
            'memory_alert': [],
            'temperature_alert': [],
            'cpu_alert': []
        }
        
    def add_callback(self, event_type: str, callback):
        """Añade callback para alertas"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
            
    def start_monitoring(self):
        """Inicia el monitoreo en hilo separado"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("🔍 Monitor de recursos iniciado")
        
    def stop_monitoring(self):
        """Detiene el monitoreo"""
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        logger.info("⏹️  Monitor de recursos detenido")
        
    def _monitor_loop(self):
        """Bucle principal de monitoreo"""
        while self.monitoring:
            try:
                stats = self.get_current_stats()
                self._check_alerts(stats)
                self._log_stats(stats)
                
                # Guardar estadísticas
                self.stats_history.append(stats)
                # Mantener solo las últimas 100 mediciones
                if len(self.stats_history) > 100:
                    self.stats_history.pop(0)
                    
                time.sleep(self.log_interval)
                
            except Exception as e:
                logger.error(f"Error en monitor: {e}")
                time.sleep(5)
                
    def get_current_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas actuales del sistema"""
        stats = {
            'timestamp': time.time(),
            'uptime': time.time() - self.start_time,
            'cpu_percent': psutil.cpu_percent(interval=1),
            'cpu_freq': self._get_cpu_frequency(),
            'memory': self._get_memory_stats(),
            'temperature': self._get_temperature(),
            'gpu_stats': self._get_gpu_stats(),
            'disk_usage': self._get_disk_usage(),
            'network': self._get_network_stats()
        }
        return stats
        
    def _get_cpu_frequency(self) -> Optional[float]:
        """Obtiene frecuencia actual de CPU"""
        try:
            freq = psutil.cpu_freq()
            return freq.current if freq else None
        except:
            return None
            
    def _get_memory_stats(self) -> Dict[str, float]:
        """Obtiene estadísticas de memoria"""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percent': memory.percent,
            'swap_total_gb': swap.total / (1024**3),
            'swap_used_gb': swap.used / (1024**3),
            'swap_percent': swap.percent
        }
        
    def _get_temperature(self) -> Optional[float]:
        """Obtiene temperatura del sistema"""
        # Rutas comunes para temperatura en Jetson
        temp_paths = [
            '/sys/devices/virtual/thermal/thermal_zone0/temp',
            '/sys/devices/virtual/thermal/thermal_zone1/temp',
            '/sys/class/thermal/thermal_zone0/temp',
            '/sys/class/thermal/thermal_zone1/temp'
        ]
        
        for path in temp_paths:
            try:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        temp = int(f.read().strip()) / 1000.0
                        return temp
            except:
                continue
                
        return None
        
    def _get_gpu_stats(self) -> Optional[Dict[str, Any]]:
        """Obtiene estadísticas de GPU usando tegrastats"""
        try:
            # Ejecutar tegrastats por 1 segundo
            result = subprocess.run([
                'tegrastats', '--interval', '1000', '--logfile', '/dev/stdout'
            ], capture_output=True, text=True, timeout=3)
            
            if result.returncode == 0 and result.stdout:
                # Parsear la salida de tegrastats
                return self._parse_tegrastats(result.stdout)
                
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            pass
            
        return None
        
    def _parse_tegrastats(self, output: str) -> Dict[str, Any]:
        """Parsea la salida de tegrastats"""
        try:
            # Tomar la última línea
            line = output.strip().split('\n')[-1]
            
            stats = {}
            
            # Buscar patrones comunes en tegrastats
            if 'GR3D_FREQ' in line:
                # Extraer frecuencia de GPU
                import re
                freq_match = re.search(r'GR3D_FREQ (\d+)%@(\d+)', line)
                if freq_match:
                    stats['gpu_usage'] = int(freq_match.group(1))
                    stats['gpu_freq'] = int(freq_match.group(2))
                    
            if 'EMC_FREQ' in line:
                # Frecuencia de memoria
                emc_match = re.search(r'EMC_FREQ (\d+)%@(\d+)', line)
                if emc_match:
                    stats['memory_controller_usage'] = int(emc_match.group(1))
                    stats['memory_freq'] = int(emc_match.group(2))
                    
            return stats
            
        except Exception as e:
            logger.debug(f"Error parseando tegrastats: {e}")
            return {}
            
    def _get_disk_usage(self) -> Dict[str, float]:
        """Obtiene uso de disco"""
        try:
            usage = psutil.disk_usage('/')
            return {
                'total_gb': usage.total / (1024**3),
                'used_gb': usage.used / (1024**3),
                'free_gb': usage.free / (1024**3),
                'percent': (usage.used / usage.total) * 100
            }
        except:
            return {}
            
    def _get_network_stats(self) -> Dict[str, int]:
        """Obtiene estadísticas de red"""
        try:
            net = psutil.net_io_counters()
            return {
                'bytes_sent': net.bytes_sent,
                'bytes_recv': net.bytes_recv,
                'packets_sent': net.packets_sent,
                'packets_recv': net.packets_recv
            }
        except:
            return {}
            
    def _check_alerts(self, stats: Dict[str, Any]):
        """Verifica y dispara alertas según umbrales"""
        # Alerta de memoria
        memory_percent = stats.get('memory', {}).get('percent', 0)
        if memory_percent > self.memory_threshold:
            for callback in self.callbacks['memory_alert']:
                try:
                    callback(memory_percent)
                except Exception as e:
                    logger.error(f"Error en callback de memoria: {e}")
                    
        # Alerta de temperatura
        temperature = stats.get('temperature')
        if temperature and temperature > self.temperature_threshold:
            for callback in self.callbacks['temperature_alert']:
                try:
                    callback(temperature)
                except Exception as e:
                    logger.error(f"Error en callback de temperatura: {e}")
                    
        # Alerta de CPU
        cpu_percent = stats.get('cpu_percent', 0)
        if cpu_percent > self.cpu_threshold:
            for callback in self.callbacks['cpu_alert']:
                try:
                    callback(cpu_percent)
                except Exception as e:
                    logger.error(f"Error en callback de CPU: {e}")
                    
    def _log_stats(self, stats: Dict[str, Any]):
        """Registra estadísticas en log"""
        uptime = stats['uptime']
        cpu = stats['cpu_percent']
        memory = stats['memory']
        temp = stats.get('temperature', 'N/A')
        
        logger.info(f"=== RECURSOS JETSON (t={uptime:.1f}s) ===")
        logger.info(f"CPU: {cpu:.1f}%")
        logger.info(f"RAM: {memory['percent']:.1f}% "
                   f"({memory['used_gb']:.1f}GB/{memory['total_gb']:.1f}GB)")
        
        if memory['swap_total_gb'] > 0:
            logger.info(f"SWAP: {memory['swap_percent']:.1f}% "
                       f"({memory['swap_used_gb']:.1f}GB/{memory['swap_total_gb']:.1f}GB)")
                       
        logger.info(f"Temperatura: {temp}°C")
        
        # Alertas
        if memory['percent'] > self.memory_threshold:
            logger.warning("⚠️  USO DE RAM ALTO!")
        if temp != 'N/A' and temp > self.temperature_threshold:
            logger.warning("⚠️  TEMPERATURA ALTA!")
        if cpu > self.cpu_threshold:
            logger.warning("⚠️  USO DE CPU ALTO!")
            
    def get_stats_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de estadísticas"""
        if not self.stats_history:
            return {}
            
        # Calcular promedios
        cpu_values = [s['cpu_percent'] for s in self.stats_history]
        memory_values = [s['memory']['percent'] for s in self.stats_history]
        temp_values = [s['temperature'] for s in self.stats_history if s['temperature']]
        
        return {
            'uptime': self.stats_history[-1]['uptime'],
            'measurements': len(self.stats_history),
            'cpu': {
                'current': cpu_values[-1] if cpu_values else 0,
                'average': sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                'max': max(cpu_values) if cpu_values else 0
            },
            'memory': {
                'current': memory_values[-1] if memory_values else 0,
                'average': sum(memory_values) / len(memory_values) if memory_values else 0,
                'max': max(memory_values) if memory_values else 0
            },
            'temperature': {
                'current': temp_values[-1] if temp_values else None,
                'average': sum(temp_values) / len(temp_values) if temp_values else None,
                'max': max(temp_values) if temp_values else None
            }
        }

class JetsonSwapManager:
    """Gestor automático de swap para Jetson Nano"""
    
    def __init__(self, swap_size_gb: int = 2, swap_file: str = "/swapfile"):
        self.swap_size_gb = swap_size_gb
        self.swap_file = swap_file
        
    def is_swap_active(self) -> bool:
        """Verifica si hay swap activo"""
        try:
            result = subprocess.run(['swapon', '--show'], 
                                  capture_output=True, text=True)
            return len(result.stdout.strip()) > 0
        except:
            return False
            
    def setup_swap(self) -> bool:
        """Configura swap automáticamente"""
        try:
            if self.is_swap_active():
                logger.info("✅ Swap ya está activo")
                return True
                
            logger.info(f"⚙️  Configurando swap de {self.swap_size_gb}GB...")
            
            # Verificar si ya existe el archivo
            if os.path.exists(self.swap_file):
                logger.info("Archivo de swap ya existe, activando...")
                result = subprocess.run(['sudo', 'swapon', self.swap_file], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info("✅ Swap activado exitosamente")
                    return True
                    
            # Crear archivo de swap
            logger.info("Creando archivo de swap...")
            result = subprocess.run([
                'sudo', 'fallocate', '-l', f'{self.swap_size_gb}G', self.swap_file
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Error creando archivo swap: {result.stderr}")
                return False
                
            # Configurar permisos
            subprocess.run(['sudo', 'chmod', '600', self.swap_file])
            
            # Crear swap
            subprocess.run(['sudo', 'mkswap', self.swap_file])
            
            # Activar swap
            result = subprocess.run(['sudo', 'swapon', self.swap_file], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ Swap configurado y activado exitosamente")
                return True
            else:
                logger.error(f"Error activando swap: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error configurando swap: {e}")
            return False
            
    def disable_swap(self) -> bool:
        """Desactiva el swap"""
        try:
            result = subprocess.run(['sudo', 'swapoff', self.swap_file], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("✅ Swap desactivado")
                return True
            else:
                logger.error(f"Error desactivando swap: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Error desactivando swap: {e}")
            return False

class JetsonCPULimiter:
    """Limitador de CPU para Jetson Nano"""
    
    def __init__(self):
        self.original_affinity = None
        
    def limit_cpu_cores(self, cores: list) -> bool:
        """Limita el proceso a cores específicos"""
        try:
            pid = os.getpid()
            self.original_affinity = os.sched_getaffinity(pid)
            
            # Establecer afinidad de CPU
            os.sched_setaffinity(pid, cores)
            logger.info(f"✅ CPU limitada a cores: {cores}")
            return True
            
        except Exception as e:
            logger.error(f"Error limitando CPU: {e}")
            return False
            
    def restore_cpu_affinity(self) -> bool:
        """Restaura la afinidad original de CPU"""
        try:
            if self.original_affinity:
                pid = os.getpid()
                os.sched_setaffinity(pid, self.original_affinity)
                logger.info("✅ Afinidad de CPU restaurada")
                return True
        except Exception as e:
            logger.error(f"Error restaurando afinidad CPU: {e}")
            return False

def setup_jetson_optimizations():
    """Configura optimizaciones específicas para Jetson Nano"""
    logger.info("🚀 Configurando optimizaciones para Jetson Nano...")
    
    # Variables de entorno
    optimizations = {
        'CUDA_CACHE_DISABLE': '1',          # Deshabilitar cache CUDA
        'TRT_LOGGER_VERBOSITY': '2',        # Reducir verbosidad TensorRT  
        'OPENCV_FFMPEG_CAPTURE_OPTIONS': 'rtsp_transport;udp',  # Optimizar OpenCV
        'PYTHONUNBUFFERED': '1'             # Output inmediato
    }
    
    for key, value in optimizations.items():
        os.environ[key] = value
        logger.info(f"✅ {key}={value}")
        
    # Configurar prioridad del proceso
    try:
        os.nice(-5)  # Prioridad más alta
        logger.info("✅ Prioridad de proceso aumentada")
    except:
        logger.warning("⚠️  No se pudo aumentar prioridad del proceso")

def monitor_jetson_resources(duration_minutes: int = 5):
    """Función de utilidad para monitorear recursos por tiempo específico"""
    monitor = JetsonResourceMonitor(log_interval=10)
    
    def memory_alert(percent):
        logger.warning(f"🚨 ALERTA MEMORIA: {percent:.1f}%")
        
    def temperature_alert(temp):
        logger.warning(f"🌡️  ALERTA TEMPERATURA: {temp:.1f}°C")
        
    def cpu_alert(percent):
        logger.warning(f"💻 ALERTA CPU: {percent:.1f}%")
        
    monitor.add_callback('memory_alert', memory_alert)
    monitor.add_callback('temperature_alert', temperature_alert)
    monitor.add_callback('cpu_alert', cpu_alert)
    
    monitor.start_monitoring()
    
    try:
        logger.info(f"📊 Monitoreando por {duration_minutes} minutos...")
        time.sleep(duration_minutes * 60)
        
    except KeyboardInterrupt:
        logger.info("⏹️  Monitoreo interrumpido")
    finally:
        monitor.stop_monitoring()
        
        # Mostrar resumen
        summary = monitor.get_stats_summary()
        if summary:
            logger.info("📈 Resumen de monitoreo:")
            logger.info(f"   CPU promedio: {summary['cpu']['average']:.1f}%")
            logger.info(f"   Memoria promedio: {summary['memory']['average']:.1f}%")
            if summary['temperature']['average']:
                logger.info(f"   Temperatura promedio: {summary['temperature']['average']:.1f}°C")

if __name__ == "__main__":
    # Script de utilidad para monitoreo
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    if len(sys.argv) > 1:
        duration = int(sys.argv[1])
    else:
        duration = 5
        
    monitor_jetson_resources(duration)
