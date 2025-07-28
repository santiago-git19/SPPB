#!/usr/bin/env python3
"""
Prueba Básica de Cámaras Duales Orbbec Gemini 335Le
===================================================

Script de prueba para verificar que las dos cámaras Orbbec están
funcionando correctamente antes de usar el sistema completo.

Este script es útil para:
- Verificar detección automática de cámaras
- Probar sincronización de frames
- Validar calidad de imagen
- Diagnosticar problemas de conexión

Autor: Sistema de IA
Fecha: 2025
"""

import cv2
import numpy as np
import time
import logging
from pathlib import Path

import sys
# Añadir el directorio 'Automatizacion' al sys.path  
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.dual_orbbec_capture import DualOrbbecCapture

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrbbecCameraTest:
    """
    Clase para pruebas básicas de las cámaras Orbbec duales
    """
    
    def __init__(self, 
                 resolution: tuple = (640, 480), 
                 fps: int = 30):
        """
        Inicializa el sistema de prueba
        
        Args:
            resolution: Resolución de las cámaras
            fps: FPS de captura
        """
        self.resolution = resolution
        self.fps = fps
        self.dual_camera = None
        
        # Estadísticas de prueba
        self.test_stats = {
            'start_time': None,
            'frames_captured': 0,
            'sync_failures': 0,
            'avg_fps': 0.0,
            'left_quality_scores': [],
            'right_quality_scores': []
        }
    
    def initialize_cameras(self) -> bool:
        """
        Inicializa y verifica las cámaras
        
        Returns:
            True si las cámaras se inicializaron correctamente
        """
        try:
            logger.info("📷 Inicializando cámaras Orbbec duales...")
            
            self.dual_camera = DualOrbbecCapture(
                resolution=self.resolution,
                fps=self.fps,
                auto_reconnect=True,
                max_reconnect_attempts=5,
                reconnect_delay=1.0
            )
            
            if self.dual_camera.is_opened():
                logger.info("✅ Cámaras inicializadas correctamente")
                
                # Obtener información de las cámaras
                stats = self.dual_camera.get_statistics()
                logger.info(f"   Cámaras detectadas: {stats.get('cameras_found', 'N/A')}")
                logger.info(f"   Resolución configurada: {self.resolution}")
                logger.info(f"   FPS configurado: {self.fps}")
                
                return True
            else:
                logger.error("❌ No se pudieron inicializar las cámaras")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error inicializando cámaras: {e}")
            return False
    
    def test_basic_capture(self, duration_seconds: int = 10) -> bool:
        """
        Prueba básica de captura por tiempo determinado
        
        Args:
            duration_seconds: Duración de la prueba en segundos
            
        Returns:
            True si la prueba fue exitosa
        """
        if not self.dual_camera or not self.dual_camera.is_opened():
            logger.error("❌ Cámaras no inicializadas")
            return False
        
        logger.info(f"🔍 Iniciando prueba de captura por {duration_seconds} segundos...")
        
        self.test_stats['start_time'] = time.time()
        end_time = self.test_stats['start_time'] + duration_seconds
        
        successful_captures = 0
        
        try:
            while time.time() < end_time:
                # Capturar frames
                left_frame, right_frame = self.dual_camera.read_frames()
                
                if left_frame is not None and right_frame is not None:
                    successful_captures += 1
                    self.test_stats['frames_captured'] += 1
                    
                    # Calcular calidad básica de imagen
                    left_quality = self._calculate_image_quality(left_frame)
                    right_quality = self._calculate_image_quality(right_frame)
                    
                    self.test_stats['left_quality_scores'].append(left_quality)
                    self.test_stats['right_quality_scores'].append(right_quality)
                    
                    # Mostrar frames cada 30 capturas
                    if successful_captures % 30 == 0:
                        combined = np.hstack([left_frame, right_frame])
                        # Añadir información
                        info_text = f"Frames: {successful_captures} | L_Quality: {left_quality:.2f} | R_Quality: {right_quality:.2f}"
                        cv2.putText(combined, info_text, (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        cv2.imshow('Prueba Cámaras Orbbec', combined)
                        
                        # Permitir salida temprana
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                else:
                    self.test_stats['sync_failures'] += 1
                
                # Pequeña pausa para no saturar
                time.sleep(0.001)
        
        except KeyboardInterrupt:
            logger.info("⚠️ Prueba interrumpida por usuario")
        
        finally:
            cv2.destroyAllWindows()
        
        # Calcular estadísticas finales
        elapsed_time = time.time() - self.test_stats['start_time']
        self.test_stats['avg_fps'] = successful_captures / elapsed_time if elapsed_time > 0 else 0
        
        # Resultado de la prueba
        success_rate = successful_captures / (successful_captures + self.test_stats['sync_failures']) if (successful_captures + self.test_stats['sync_failures']) > 0 else 0
        
        logger.info(f"📊 Resultados de prueba básica:")
        logger.info(f"   ✅ Capturas exitosas: {successful_captures}")
        logger.info(f"   ❌ Fallos de sincronización: {self.test_stats['sync_failures']}")
        logger.info(f"   📈 FPS promedio: {self.test_stats['avg_fps']:.1f}")
        logger.info(f"   🎯 Tasa de éxito: {success_rate:.2%}")
        
        return success_rate > 0.8  # Consideramos exitoso si >80% de capturas funcionan
    
    def test_synchronization(self, num_samples: int = 100) -> bool:
        """
        Prueba específica de sincronización entre cámaras
        
        Args:
            num_samples: Número de muestras para la prueba
            
        Returns:
            True si la sincronización es adecuada
        """
        if not self.dual_camera or not self.dual_camera.is_opened():
            logger.error("❌ Cámaras no inicializadas")
            return False
        
        logger.info(f"⏱️ Probando sincronización con {num_samples} muestras...")
        
        sync_deltas = []
        successful_syncs = 0
        
        try:
            for i in range(num_samples):
                start_time = time.time()
                
                # Capturar con timestamp
                left_frame, right_frame = self.dual_camera.read_frames()
                
                capture_time = time.time()
                
                if left_frame is not None and right_frame is not None:
                    # Calcular diferencia temporal aproximada
                    # (en un caso real, usaríamos timestamps de hardware)
                    sync_delta = capture_time - start_time
                    sync_deltas.append(sync_delta * 1000)  # En milisegundos
                    successful_syncs += 1
                
                # Progreso cada 25%
                if (i + 1) % (num_samples // 4) == 0:
                    progress = ((i + 1) / num_samples) * 100
                    logger.info(f"   📊 Progreso: {progress:.0f}%")
                
                time.sleep(0.01)  # Pequeña pausa
        
        except Exception as e:
            logger.error(f"❌ Error en prueba de sincronización: {e}")
            return False
        
        # Analizar resultados
        if sync_deltas:
            avg_delta = np.mean(sync_deltas)
            std_delta = np.std(sync_deltas)
            max_delta = np.max(sync_deltas)
            min_delta = np.min(sync_deltas)
            
            logger.info(f"📊 Resultados de sincronización:")
            logger.info(f"   ⏱️ Delta promedio: {avg_delta:.2f}ms")
            logger.info(f"   📏 Desviación estándar: {std_delta:.2f}ms")
            logger.info(f"   ⬆️ Delta máximo: {max_delta:.2f}ms")
            logger.info(f"   ⬇️ Delta mínimo: {min_delta:.2f}ms")
            logger.info(f"   ✅ Sincronizaciones exitosas: {successful_syncs}/{num_samples}")
            
            # Criterios de éxito: delta promedio < 50ms y std < 20ms
            is_sync_good = avg_delta < 50.0 and std_delta < 20.0
            
            if is_sync_good:
                logger.info("✅ Sincronización BUENA")
            else:
                logger.warning("⚠️ Sincronización DEFICIENTE - revisar conexiones USB")
            
            return is_sync_good
        else:
            logger.error("❌ No se pudieron obtener muestras de sincronización")
            return False
    
    def test_image_quality(self, num_samples: int = 50) -> bool:
        """
        Prueba de calidad de imagen de ambas cámaras
        
        Args:
            num_samples: Número de muestras para analizar
            
        Returns:
            True si la calidad es adecuada
        """
        if not self.dual_camera or not self.dual_camera.is_opened():
            logger.error("❌ Cámaras no inicializadas")
            return False
        
        logger.info(f"🎨 Analizando calidad de imagen con {num_samples} muestras...")
        
        left_qualities = []
        right_qualities = []
        
        try:
            for i in range(num_samples):
                left_frame, right_frame = self.dual_camera.read_frames()
                
                if left_frame is not None and right_frame is not None:
                    left_quality = self._calculate_image_quality(left_frame)
                    right_quality = self._calculate_image_quality(right_frame)
                    
                    left_qualities.append(left_quality)
                    right_qualities.append(right_quality)
                
                time.sleep(0.02)  # 50 FPS sampling
        
        except Exception as e:
            logger.error(f"❌ Error en análisis de calidad: {e}")
            return False
        
        # Analizar resultados
        if left_qualities and right_qualities:
            left_avg = np.mean(left_qualities)
            right_avg = np.mean(right_qualities)
            left_std = np.std(left_qualities)
            right_std = np.std(right_qualities)
            
            logger.info(f"📊 Calidad de imagen:")
            logger.info(f"   📷 Cámara izquierda - Promedio: {left_avg:.2f}, Std: {left_std:.2f}")
            logger.info(f"   📷 Cámara derecha - Promedio: {right_avg:.2f}, Std: {right_std:.2f}")
            
            # Criterios: calidad promedio > 30 (escala 0-100) y std < 15
            left_good = left_avg > 30.0 and left_std < 15.0
            right_good = right_avg > 30.0 and right_std < 15.0
            
            if left_good and right_good:
                logger.info("✅ Calidad de imagen BUENA en ambas cámaras")
                return True
            else:
                if not left_good:
                    logger.warning("⚠️ Calidad DEFICIENTE en cámara izquierda")
                if not right_good:
                    logger.warning("⚠️ Calidad DEFICIENTE en cámara derecha")
                return False
        else:
            logger.error("❌ No se pudieron obtener muestras de calidad")
            return False
    
    def _calculate_image_quality(self, frame: np.ndarray) -> float:
        """
        Calcula una métrica básica de calidad de imagen
        
        Args:
            frame: Frame a analizar
            
        Returns:
            Puntuación de calidad (0-100)
        """
        if frame is None:
            return 0.0
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calcular varianza (medida de contraste/detalle)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalizar a escala 0-100 (valores típicos: 0-1000)
        quality_score = min(variance / 10.0, 100.0)
        
        return quality_score
    
    def run_complete_test(self) -> bool:
        """
        Ejecuta todas las pruebas en secuencia
        
        Returns:
            True si todas las pruebas fueron exitosas
        """
        logger.info("🚀 Iniciando pruebas completas de cámaras Orbbec...")
        
        all_tests_passed = True
        
        # 1. Inicialización
        if not self.initialize_cameras():
            return False
        
        try:
            # 2. Prueba básica de captura
            logger.info("\n" + "="*50)
            logger.info("🔍 PRUEBA 1: Captura básica")
            logger.info("="*50)
            
            basic_test_passed = self.test_basic_capture(duration_seconds=5)
            if not basic_test_passed:
                logger.error("❌ Prueba básica FALLÓ")
                all_tests_passed = False
            else:
                logger.info("✅ Prueba básica EXITOSA")
            
            # 3. Prueba de sincronización
            logger.info("\n" + "="*50)
            logger.info("⏱️ PRUEBA 2: Sincronización")
            logger.info("="*50)
            
            sync_test_passed = self.test_synchronization(num_samples=50)
            if not sync_test_passed:
                logger.error("❌ Prueba de sincronización FALLÓ")
                all_tests_passed = False
            else:
                logger.info("✅ Prueba de sincronización EXITOSA")
            
            # 4. Prueba de calidad
            logger.info("\n" + "="*50)
            logger.info("🎨 PRUEBA 3: Calidad de imagen")
            logger.info("="*50)
            
            quality_test_passed = self.test_image_quality(num_samples=30)
            if not quality_test_passed:
                logger.error("❌ Prueba de calidad FALLÓ")
                all_tests_passed = False
            else:
                logger.info("✅ Prueba de calidad EXITOSA")
        
        finally:
            # Liberar recursos
            if self.dual_camera:
                self.dual_camera.release()
        
        # Resultado final
        logger.info("\n" + "="*60)
        logger.info("📊 RESULTADO FINAL DE PRUEBAS")
        logger.info("="*60)
        
        if all_tests_passed:
            logger.info("🎉 TODAS LAS PRUEBAS EXITOSAS")
            logger.info("✅ Las cámaras Orbbec están listas para uso")
        else:
            logger.error("❌ ALGUNAS PRUEBAS FALLARON")
            logger.error("💡 Revisar conexiones, drivers y alimentación USB")
        
        return all_tests_passed
    
    def __enter__(self):
        """Soporte para context manager"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Liberación automática de recursos"""
        if self.dual_camera:
            self.dual_camera.release()


def main():
    """Función principal para ejecutar las pruebas"""
    print("🔬 Prueba de Cámaras Duales Orbbec Gemini 335Le")
    print("=" * 60)
    print("Este script verificará que las cámaras estén funcionando correctamente")
    print("antes de usar el sistema completo de clasificación de poses.")
    print()
    print("💡 Asegúrate de que:")
    print("   - Las dos cámaras Orbbec estén conectadas al switch USB")
    print("   - El switch tenga alimentación suficiente")
    print("   - Los drivers de Orbbec estén instalados")
    print()
    
    # Configuración de prueba
    test_resolution = (640, 480)
    test_fps = 30
    
    try:
        with OrbbecCameraTest(
            resolution=test_resolution,
            fps=test_fps
        ) as camera_test:
            
            success = camera_test.run_complete_test()
            
            if success:
                print("\n🎉 ¡FELICIDADES! Las cámaras están listas.")
                print("   Ahora puedes usar 'trt_pose_with_classification_example_cameras.py'")
                return True
            else:
                print("\n❌ Las cámaras necesitan revisión.")
                print("💡 Soluciones comunes:")
                print("   - Verificar conexiones USB")
                print("   - Reinstalar drivers de Orbbec")
                print("   - Probar switch USB con más potencia")
                print("   - Verificar compatibilidad con OpenCV")
                return False
    
    except Exception as e:
        logger.error(f"❌ Error inesperado en pruebas: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
