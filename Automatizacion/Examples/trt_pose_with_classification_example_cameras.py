#!/usr/bin/env python3
"""
Ejemplo de Uso: TRT Pose + Clasificación con Cámaras Duales Orbbec
==================================================================

Este script integra el sistema de detección y clasificación de poses con
dos cámaras Orbbec Gemini 335Le para captura estéreo sincronizada.

Flujo de trabajo:
1. DualOrbbecCapture maneja dos cámaras Gemini 335Le sincronizadas
2. trt_pose detecta keypoints en cada vista (izquierda/derecha)  
3. TRTPoseClassifier procesa y clasifica las poses
4. Se muestra el resultado dual en tiempo real

Autor: Sistema de IA
Fecha: 2025
"""

import cv2
import numpy as np
import torch
import json
import time
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict

import sys
from pathlib import Path
# Añadir el directorio 'Automatizacion' al sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Importar procesadores y cámaras
from utils.trt_pose_proc import TRTPoseProcessor
from utils.trt_pose_classifier import TRTPoseClassifier, create_pose_classifier
from utils.dual_orbbec_capture import DualOrbbecCapture

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TRTPoseWithDualCameras:
    """
    Integra trt_pose con clasificación de poses usando dos cámaras Orbbec sincronizadas
    """
    
    def __init__(self, 
                 trt_pose_model_path: str,
                 pose_topology_path: str,
                 pose_classifier_model_path: str,
                 camera_resolution: Tuple[int, int] = (640, 480),
                 camera_fps: int = 30,
                 width: int = 224,
                 height: int = 224):
        """
        Inicializa el sistema completo con cámaras duales
        
        Args:
            trt_pose_model_path: Ruta al modelo TensorRT de trt_pose
            pose_topology_path: Ruta al archivo JSON de topología
            pose_classifier_model_path: Ruta al modelo engine del clasificador
            camera_resolution: Resolución de captura de las cámaras
            camera_fps: FPS de captura de las cámaras
            width: Ancho de entrada para trt_pose
            height: Alto de entrada para trt_pose
        """
        self.width = width
        self.height = height
        self.camera_resolution = camera_resolution
        
        # Inicializar cámaras duales
        logger.info("📷 Inicializando sistema de cámaras duales...")
        self.dual_camera = DualOrbbecCapture(
            resolution=camera_resolution,
            fps=camera_fps,
            auto_reconnect=True,
            max_reconnect_attempts=3,
            reconnect_delay=2.0
        )
        
        if not self.dual_camera.is_opened():
            raise RuntimeError("❌ No se pudieron inicializar las cámaras Orbbec")
        
        # Crear procesador de trt_pose
        logger.info("🔧 Inicializando procesador TRT Pose...")
        self.trt_pose_processor = TRTPoseProcessor(
            model_path=trt_pose_model_path,
            topology_path=pose_topology_path,
            use_tensorrt=False
        )
        
        # Crear clasificador de poses
        logger.info("🎭 Inicializando clasificador de poses...")
        self.pose_classifier = create_pose_classifier(
            model_path=pose_classifier_model_path,
            keypoint_format='nvidia',  # Usar formato NVIDIA según documentación oficial
            sequence_length=15,      
            confidence_threshold=0.07
        )
        
        # Estadísticas
        self.stats = {
            'frames_processed': 0,
            'left_poses_detected': 0,
            'right_poses_detected': 0,
            'poses_classified': 0,
            'start_time': time.time()
        }
        
        logger.info("✅ Sistema TRT Pose + Cámaras Duales inicializado")
        logger.info(f"   📷 Resolución cámaras: {camera_resolution}")
        logger.info(f"   🎯 Resolución procesamiento: {width}x{height}")
    
    def process_dual_frames(self, 
                           left_frame: np.ndarray, 
                           right_frame: np.ndarray,
                           process_both: bool = True) -> Dict:
        """
        Procesa ambos frames con detección y clasificación de poses
        
        Args:
            left_frame: Frame de la cámara izquierda
            right_frame: Frame de la cámara derecha
            process_both: Si procesar ambas vistas o solo la izquierda
            
        Returns:
            Diccionario con resultados de procesamiento dual
        """
        dual_result = {
            'left_result': {'people_detected': 0, 'pose_classifications': [], 'keypoints_extracted': []},
            'right_result': {'people_detected': 0, 'pose_classifications': [], 'keypoints_extracted': []},
            'processing_time_ms': 0,
            'sync_quality': 'good'
        }
        
        start_time = time.time()
        
        try:
            # Procesar vista izquierda (siempre)
            left_keypoints = self.trt_pose_processor.process_frame(left_frame)
            if left_keypoints is not None and isinstance(left_keypoints, np.ndarray):
                dual_result['left_result']['people_detected'] = 1
                dual_result['left_result']['keypoints_extracted'] = left_keypoints
                self.stats['left_poses_detected'] += 1
                
                # Clasificar poses en vista izquierda
                classification_result = self.pose_classifier.process_keypoints(left_keypoints)
                if classification_result and not classification_result.get('error', False):
                    dual_result['left_result']['pose_classifications'].append({
                        'person_id': 0,
                        'pose_class': classification_result['predicted_class'],
                        'confidence': classification_result['confidence'],
                        'probabilities': classification_result['probabilities'],
                        'view': 'left'
                    })
                    self.stats['poses_classified'] += 1
            
            # Procesar vista derecha (opcional)
            if process_both:
                right_keypoints = self.trt_pose_processor.process_frame(right_frame)
                if right_keypoints is not None and isinstance(right_keypoints, np.ndarray):
                    dual_result['right_result']['people_detected'] = 1
                    dual_result['right_result']['keypoints_extracted'] = right_keypoints
                    self.stats['right_poses_detected'] += 1
                    
                    # Nota: Para clasificación, usamos solo una vista para evitar redundancia
                    # La vista derecha se usa principalmente para contexto visual
            
        except Exception as e:
            logger.error(f"❌ Error procesando frames duales: {e}")
            dual_result['sync_quality'] = 'error'
        
        finally:
            dual_result['processing_time_ms'] = (time.time() - start_time) * 1000
            self.stats['frames_processed'] += 1
        
        return dual_result
    
    def draw_dual_results(self, 
                         left_frame: np.ndarray, 
                         right_frame: np.ndarray,
                         dual_result: Dict) -> np.ndarray:
        """
        Dibuja resultados sobre ambas vistas y las combina
        
        Args:
            left_frame: Frame izquierdo original
            right_frame: Frame derecho original
            dual_result: Resultados del procesamiento dual
            
        Returns:
            Imagen combinada con resultados dibujados
        """
        # Procesar vista izquierda
        left_annotated = self._draw_single_view(
            left_frame, 
            dual_result['left_result'], 
            "IZQUIERDA"
        )
        
        # Procesar vista derecha
        right_annotated = self._draw_single_view(
            right_frame, 
            dual_result['right_result'], 
            "DERECHA"
        )
        
        # Combinar ambas vistas horizontalmente
        combined_frame = np.hstack([left_annotated, right_annotated])
        
        # Información global en la parte superior
        info_text = (f"Sync: {dual_result['sync_quality']} | "
                    f"Tiempo: {dual_result['processing_time_ms']:.1f}ms | "
                    f"L: {dual_result['left_result']['people_detected']} "
                    f"R: {dual_result['right_result']['people_detected']}")
        
        cv2.putText(combined_frame, info_text, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return combined_frame
    
    def _draw_single_view(self, 
                         frame: np.ndarray, 
                         view_result: Dict, 
                         view_label: str) -> np.ndarray:
        """
        Dibuja resultados en una vista individual
        
        Args:
            frame: Frame original
            view_result: Resultados de la vista
            view_label: Etiqueta de la vista ("IZQUIERDA"/"DERECHA")
            
        Returns:
            Frame con anotaciones
        """
        result_frame = frame.copy()
        
        # Etiqueta de vista
        cv2.putText(result_frame, view_label, (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Dibujar keypoints si hay detección
        keypoints = view_result['keypoints_extracted']
        if isinstance(keypoints, np.ndarray) and len(keypoints) > 0:
            # Dibujar keypoints
            for kp in keypoints:
                x, y, conf = kp
                if conf > self.pose_classifier.confidence_threshold:
                    # Color según confianza
                    color = (0, int(255 * conf), int(255 * (1 - conf)))
                    cv2.circle(result_frame, (int(x), int(y)), 3, color, -1)
            
            # Dibujar clasificación si está disponible
            if view_result['pose_classifications']:
                classification = view_result['pose_classifications'][0]
                pose_class = classification['pose_class']
                confidence = classification['confidence']
                
                # Encontrar posición para texto
                valid_kps = keypoints[keypoints[:, 2] > self.pose_classifier.confidence_threshold]
                if len(valid_kps) > 0:
                    center_x = int(np.mean(valid_kps[:, 0]))
                    center_y = int(np.mean(valid_kps[:, 1])) - 30
                    
                    # Texto con fondo
                    text = f"{pose_class}: {confidence:.2f}"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    
                    cv2.rectangle(result_frame, 
                                (center_x - 5, center_y - text_size[1] - 5),
                                (center_x + text_size[0] + 5, center_y + 5),
                                (0, 0, 0), -1)
                    
                    cv2.putText(result_frame, text,
                              (center_x, center_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return result_frame
    
    def run_dual_capture(self, 
                        output_path: Optional[str] = None,
                        process_both_views: bool = True):
        """
        Ejecuta captura y procesamiento dual en tiempo real
        
        Args:
            output_path: Ruta para guardar video procesado (opcional)
            process_both_views: Si procesar poses en ambas vistas
        """
        # Configurar escritura de video si se especifica
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 15  # FPS más bajo para el video de salida
            combined_width = self.camera_resolution[0] * 2
            combined_height = self.camera_resolution[1]
            out = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, combined_height))
        
        logger.info("🎥 Iniciando captura dual...")
        logger.info("   Controles:")
        logger.info("   - 'q': Salir")
        logger.info("   - 'r': Reiniciar secuencia de clasificación")
        logger.info("   - 's': Mostrar estadísticas")
        
        try:
            while True:
                # Capturar frames sincronizados
                left_frame, right_frame = self.dual_camera.read_frames()
                
                if left_frame is None or right_frame is None:
                    logger.warning("⚠️ No se pudieron capturar frames sincronizados")
                    time.sleep(0.1)
                    continue
                
                # Procesar ambos frames
                dual_result = self.process_dual_frames(
                    left_frame, right_frame, 
                    process_both=process_both_views
                )
                
                # Dibujar resultados combinados
                combined_frame = self.draw_dual_results(left_frame, right_frame, dual_result)
                
                # Mostrar resultado
                cv2.imshow('TRT Pose + Cámaras Duales Orbbec', combined_frame)
                
                # Guardar si se especifica
                if out:
                    out.write(combined_frame)
                
                # Controles de teclado
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.pose_classifier.reset_sequence()
                    logger.info("🔄 Secuencia de clasificación reiniciada")
                elif key == ord('s'):
                    self._print_current_statistics()
                
        except KeyboardInterrupt:
            logger.info("⚠️ Procesamiento interrumpido por usuario")
        
        finally:
            if out:
                out.release()
            cv2.destroyAllWindows()
            
            # Mostrar estadísticas finales
            self._print_final_statistics()
    
    def _print_current_statistics(self):
        """Imprime estadísticas actuales"""
        cam_stats = self.dual_camera.get_statistics()
        classifier_stats = self.pose_classifier.get_statistics()
        
        print("\n" + "="*60)
        print("📊 ESTADÍSTICAS ACTUALES")
        print("="*60)
        print(f"📷 Cámaras:")
        print(f"   Frames capturados: {cam_stats['frames_captured']}")
        print(f"   FPS: {cam_stats['fps']:.1f}")
        print(f"   Tasa de éxito: {cam_stats['success_rate']:.2%}")
        print(f"🎭 Clasificación:")
        print(f"   Poses detectadas (I/D): {self.stats['left_poses_detected']}/{self.stats['right_poses_detected']}")
        print(f"   Poses clasificadas: {self.stats['poses_classified']}")
        print(f"   Clase más común: {classifier_stats['most_common_class']}")
    
    def _print_final_statistics(self):
        """Imprime estadísticas finales del procesamiento"""
        elapsed_time = time.time() - self.stats['start_time']
        fps = self.stats['frames_processed'] / elapsed_time if elapsed_time > 0 else 0
        
        cam_stats = self.dual_camera.get_statistics()
        classifier_stats = self.pose_classifier.get_statistics()
        
        print("\n" + "="*60)
        print("📊 ESTADÍSTICAS FINALES")
        print("="*60)
        print(f"⏱️  Tiempo total: {elapsed_time:.1f} segundos")
        print(f"🎞️  Frames procesados: {self.stats['frames_processed']}")
        print(f"📈 FPS promedio procesamiento: {fps:.1f}")
        
        print(f"\n📷 Cámaras:")
        print(f"   Frames capturados: {cam_stats['frames_captured']}")
        print(f"   Fallos sincronización: {cam_stats['sync_failures']}")
        print(f"   Reconexiones: {cam_stats['reconnections']}")
        print(f"   FPS cámaras: {cam_stats['fps']:.1f}")
        
        print(f"\n🎭 Detección y Clasificación:")
        print(f"   Poses detectadas izquierda: {self.stats['left_poses_detected']}")
        print(f"   Poses detectadas derecha: {self.stats['right_poses_detected']}")
        print(f"   Poses clasificadas: {self.stats['poses_classified']}")
        print(f"   Tasa de confianza: {classifier_stats['confidence_rate']:.2%}")
        print(f"   Clase más común: {classifier_stats['most_common_class']}")
        
        print("\n📊 Distribución de clases:")
        for class_name, count in classifier_stats['class_distribution'].items():
            percentage = (count / classifier_stats['total_predictions']) * 100 if classifier_stats['total_predictions'] > 0 else 0
            print(f"   {class_name}: {count} ({percentage:.1f}%)")
    
    def release(self):
        """Libera todos los recursos"""
        logger.info("🔒 Liberando recursos del sistema dual...")
        self.dual_camera.release()
        logger.info("✅ Sistema liberado correctamente")
    
    def __enter__(self):
        """Soporte para context manager"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Liberación automática al salir del context manager"""
        self.release()


def main():
    """Función principal de ejemplo con cámaras duales"""
    # Configuración de rutas - AJUSTAR SEGÚN TU SISTEMA
    config = {
        'trt_pose_model': '/home/mobilenet/Documentos/Trabajo/trt_pose/models/resnet18_baseline_att_224x224_A_epoch_249.pth',
        'pose_topology': '/home/mobilenet/Documentos/Trabajo/trt_pose/tasks/human_pose/human_pose.json',
        'pose_classifier_model': '/home/mobilenet/Documentos/Trabajo/SPPB/Automatizacion/models/pose_classification/st-gcn_3dbp_nvidia.engine',
        'output_video': '/home/mobilenet/Documentos/Trabajo/SPPB/Automatizacion/Videos/Salida/dual_cameras_output.mp4',  # None para no guardar
        'camera_resolution': (640, 480),  # Resolución de las cámaras Orbbec
        'camera_fps': 30,  # FPS de captura
        'process_both_views': True  # Procesar poses en ambas vistas o solo izquierda
    }
    
    # Verificar archivos requeridos
    required_files = ['trt_pose_model', 'pose_topology', 'pose_classifier_model']
    for key in required_files:
        if not Path(config[key]).exists():
            print(f"❌ Archivo no encontrado: {config[key]}")
            print("💡 Para usar este ejemplo:")
            print("   1. Asegúrate de que el modelo TRT pose esté disponible")
            print("   2. Descarga PoseClassificationNet de NGC en formato ONNX")
            print("   3. Conecta dos cámaras Orbbec Gemini 335Le al switch USB")
            print("   4. Ajusta las rutas en la configuración")
            return False
    
    try:
        # Crear y ejecutar sistema con cámaras duales
        print("🚀 Inicializando sistema TRT Pose + Cámaras Duales Orbbec...")
        
        with TRTPoseWithDualCameras(
            trt_pose_model_path=config['trt_pose_model'],
            pose_topology_path=config['pose_topology'],
            pose_classifier_model_path=config['pose_classifier_model'],
            camera_resolution=config['camera_resolution'],
            camera_fps=config['camera_fps']
        ) as dual_system:
            
            # Ejecutar captura dual
            dual_system.run_dual_capture(
                output_path=config['output_video'],
                process_both_views=config['process_both_views']
            )
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en sistema dual: {e}")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        print("❌ Sistema no pudo ejecutarse correctamente")
        print("💡 Verificar:")
        print("   - Cámaras Orbbec conectadas y funcionando")
        print("   - Switch USB con alimentación suficiente")
        print("   - Drivers de Orbbec instalados")
        print("   - Modelos de IA disponibles")
    else:
        print("✅ Sistema ejecutado correctamente")
