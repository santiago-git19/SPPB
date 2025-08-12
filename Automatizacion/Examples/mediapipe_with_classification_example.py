#!/usr/bin/env python3
"""
Ejemplo de Uso: MediaPipe Tasks + Clasificación de Poses + Validación de Topología
=================================================================================

Este script demuestra cómo integrar MediaPipe Tasks con el clasificador de poses
PoseClassificationNet de NVIDIA TAO, incluyendo validación de la conversión de topología.

Flujo de trabajo:
1. MediaPipe Tasks detecta keypoints de personas en video (33 keypoints)
2. Se valida y convierte la topología de MediaPipe a NVIDIA (34 keypoints)
3. TRTPoseClassifier procesa y clasifica las poses
4. Se muestra el resultado en tiempo real con validación visual

Autor: Sistema de IA
Fecha: 2025
"""

import cv2
import numpy as np
import time
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import json

import sys
from pathlib import Path
# Añadir el directorio 'Automatizacion' al sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Importar nuestras clases
from utils.prueba import MediaPipeTasksPoseProcessor
from utils.action_classifier import TRTPoseClassifier, create_pose_classifier

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MediaPipeWithClassifier:
    """
    Integra MediaPipe Tasks con clasificación de poses y validación de topología
    """
    
    # Mapeo de nombres de keypoints para validación visual
    MEDIAPIPE_KEYPOINT_NAMES = [
        'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
        'right_eye_inner', 'right_eye', 'right_eye_outer',
        'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
        'left_index', 'right_index', 'left_thumb', 'right_thumb',
        'left_hip', 'right_hip', 'left_knee', 'right_knee',
        'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index'
    ]
    
    NVIDIA_KEYPOINT_NAMES = [
        'hips', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
        'neck', 'left_ankle', 'right_ankle', 'left_big_toe', 'right_big_toe',
        'head', 'left_little_toe', 'left_heel', 'right_heel', 'nose',
        'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder',
        'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
        'left_pinky_knuckle', 'right_pinky_knuckle', 'left_middle_knuckle', 'right_middle_knuckle',
        'left_index_knuckle', 'right_index_knuckle', 'left_thumb_tip', 'right_thumb_tip'
    ]
    
    def __init__(self, 
                 mediapipe_model_path: str,
                 pose_classifier_model_path: str,
                 validation_mode: bool = True):
        """
        Inicializa el sistema completo
        
        Args:
            mediapipe_model_path: Ruta al modelo .task de MediaPipe
            pose_classifier_model_path: Ruta al modelo engine del clasificador
            validation_mode: Si activar validación visual de topología
        """
        self.validation_mode = validation_mode
        
        # Crear procesador de MediaPipe Tasks
        self.mediapipe_processor = MediaPipeTasksPoseProcessor(
            model_path=mediapipe_model_path,
            confidence_threshold=0.3,
            debug=False
        )
        
        # Crear clasificador de poses
        self.pose_classifier = create_pose_classifier(
            model_path=pose_classifier_model_path,
            input_keypoint_format='mediapipe',  # ✅ Especificar entrada MediaPipe
            keypoint_format='nvidia',          # Formato del modelo
            sequence_length=30,                # Secuencia más corta para pruebas
            confidence_threshold=0.2           # Umbral más bajo para capturar más datos
        )
        
        # Estadísticas
        self.stats = {
            'frames_processed': 0,
            'poses_detected': 0,
            'poses_classified': 0,
            'topology_validations': 0,
            'mapping_errors': 0,
            'start_time': time.time()
        }
        
        # Datos para validación
        self.validation_data = {
            'mediapipe_keypoints_history': [],
            'nvidia_keypoints_history': [],
            'mapping_validation_results': []
        }
        
        logger.info("✅ Sistema MediaPipe + Clasificador + Validación inicializado")
        logger.info("   🎯 Procesador: MediaPipeTasksPoseProcessor")
        logger.info("   🎭 Clasificador: TRTPoseClassifier (entrada MediaPipe)")
        logger.info(f"   🔍 Validación de topología: {'Activada' if validation_mode else 'Desactivada'}")
        
    def validate_topology_mapping(self, mediapipe_keypoints: np.ndarray) -> dict:
        """
        Valida la conversión de topología MediaPipe a NVIDIA
        
        Args:
            mediapipe_keypoints: Keypoints originales de MediaPipe [33, 3]
            
        Returns:
            dict: Resultados de validación con estadísticas y errores detectados
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'mapped_keypoints_count': 0,
            'unmapped_keypoints_count': 0,
            'mediapipe_confidence_stats': {},
            'nvidia_confidence_stats': {},
            'mapping_details': []
        }
        
        try:
            # Obtener el mapeo desde el clasificador
            mapping = self.pose_classifier.MEDIAPIPE_TO_NVIDIA_MAPPING
            
            # Crear array NVIDIA inicializado a cero
            nvidia_keypoints = np.zeros((34, 3), dtype=np.float32)
            
            # Estadísticas de MediaPipe
            mediapipe_valid = mediapipe_keypoints[mediapipe_keypoints[:, 2] > 0.1]
            validation_result['mediapipe_confidence_stats'] = {
                'total_keypoints': len(mediapipe_keypoints),
                'valid_keypoints': len(mediapipe_valid),
                'mean_confidence': float(np.mean(mediapipe_valid[:, 2])) if len(mediapipe_valid) > 0 else 0.0,
                'max_confidence': float(np.max(mediapipe_valid[:, 2])) if len(mediapipe_valid) > 0 else 0.0,
                'min_confidence': float(np.min(mediapipe_valid[:, 2])) if len(mediapipe_valid) > 0 else 0.0
            }
            
            # Procesar cada keypoint de MediaPipe
            mapped_count = 0
            unmapped_count = 0
            
            for mp_idx in range(len(mediapipe_keypoints)):
                mp_kp = mediapipe_keypoints[mp_idx]
                mp_name = self.MEDIAPIPE_KEYPOINT_NAMES[mp_idx] if mp_idx < len(self.MEDIAPIPE_KEYPOINT_NAMES) else f"mp_{mp_idx}"
                
                if mp_idx in mapping:
                    nvidia_idx = mapping[mp_idx]
                    
                    if nvidia_idx is not None:
                        # Mapeo válido
                        nvidia_keypoints[nvidia_idx] = mp_kp
                        mapped_count += 1
                        
                        nvidia_name = self.NVIDIA_KEYPOINT_NAMES[nvidia_idx] if nvidia_idx < len(self.NVIDIA_KEYPOINT_NAMES) else f"nvidia_{nvidia_idx}"
                        
                        validation_result['mapping_details'].append({
                            'mediapipe_idx': mp_idx,
                            'mediapipe_name': mp_name,
                            'nvidia_idx': nvidia_idx,
                            'nvidia_name': nvidia_name,
                            'confidence': float(mp_kp[2]),
                            'coordinates': [float(mp_kp[0]), float(mp_kp[1])],
                            'is_valid': mp_kp[2] > 0.1
                        })
                        
                        # Validaciones específicas
                        if 'left' in mp_name.lower() and 'right' in nvidia_name.lower():
                            validation_result['errors'].append(f"Posible error L/R: {mp_name} -> {nvidia_name}")
                            validation_result['is_valid'] = False
                            
                        if 'right' in mp_name.lower() and 'left' in nvidia_name.lower():
                            validation_result['errors'].append(f"Posible error R/L: {mp_name} -> {nvidia_name}")
                            validation_result['is_valid'] = False
                    else:
                        # Keypoint sin mapeo directo
                        unmapped_count += 1
                        validation_result['warnings'].append(f"Sin mapeo: {mp_name} (idx {mp_idx})")
                else:
                    # Keypoint no considerado en el mapeo
                    unmapped_count += 1
                    validation_result['errors'].append(f"Keypoint no mapeado: {mp_name} (idx {mp_idx})")
                    validation_result['is_valid'] = False
            
            # Estadísticas de NVIDIA
            nvidia_valid = nvidia_keypoints[nvidia_keypoints[:, 2] > 0.1]
            validation_result['nvidia_confidence_stats'] = {
                'total_keypoints': len(nvidia_keypoints),
                'valid_keypoints': len(nvidia_valid),
                'mean_confidence': float(np.mean(nvidia_valid[:, 2])) if len(nvidia_valid) > 0 else 0.0,
                'max_confidence': float(np.max(nvidia_valid[:, 2])) if len(nvidia_valid) > 0 else 0.0,
                'min_confidence': float(np.min(nvidia_valid[:, 2])) if len(nvidia_valid) > 0 else 0.0
            }
            
            validation_result['mapped_keypoints_count'] = mapped_count
            validation_result['unmapped_keypoints_count'] = unmapped_count
            
            # Guardar para análisis posterior
            if self.validation_mode:
                self.validation_data['mediapipe_keypoints_history'].append(mediapipe_keypoints.copy())
                self.validation_data['nvidia_keypoints_history'].append(nvidia_keypoints.copy())
                self.validation_data['mapping_validation_results'].append(validation_result.copy())
            
            self.stats['topology_validations'] += 1
            if not validation_result['is_valid']:
                self.stats['mapping_errors'] += 1
                
        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Error en validación: {str(e)}")
            logger.error(f"❌ Error validando topología: {e}")
        
        return validation_result
    
    def process_frame_with_classification_and_validation(self, image: np.ndarray) -> dict:
        """
        Procesa un frame completo: detección + validación + clasificación
        """
        frame_result = {
            'people_detected': 0,
            'pose_classifications': [],
            'processing_time_ms': 0,
            'mediapipe_keypoints': None,
            'validation_result': None,
            'has_pose': False
        }
        
        start_time = time.time()
        
        try:
            # Usar MediaPipe Tasks para obtener keypoints
            mediapipe_keypoints = self.mediapipe_processor.process_frame(image)
            
            if mediapipe_keypoints is not None and isinstance(mediapipe_keypoints, np.ndarray):
                frame_result['people_detected'] = 1
                frame_result['mediapipe_keypoints'] = mediapipe_keypoints
                frame_result['has_pose'] = True
                
                # Validar conversión de topología si está activada
                if self.validation_mode:
                    validation_result = self.validate_topology_mapping(mediapipe_keypoints)
                    frame_result['validation_result'] = validation_result
                    
                    # Log errores críticos
                    if not validation_result['is_valid']:
                        logger.warning(f"⚠️ Errores de mapeo detectados: {len(validation_result['errors'])}")
                        for error in validation_result['errors'][:3]:  # Mostrar máximo 3 errores
                            logger.warning(f"   🔴 {error}")
                
                # Clasificar poses (el clasificador internamente convertirá MediaPipe -> NVIDIA)
                classification_result = self.pose_classifier.process_keypoints(mediapipe_keypoints)
                
                if classification_result and not classification_result.get('error', False):
                    frame_result['pose_classifications'].append({
                        'person_id': 0,
                        'pose_class': classification_result['predicted_class'],
                        'confidence': classification_result['confidence'],
                        'probabilities': classification_result['probabilities'],
                        'inference_time_ms': classification_result.get('inference_time_ms', 0)
                    })
                    
                    self.stats['poses_classified'] += 1
                
                self.stats['poses_detected'] += 1
            
        except Exception as e:
            logger.error(f"❌ Error procesando frame: {e}")
        
        finally:
            frame_result['processing_time_ms'] = (time.time() - start_time) * 1000
            self.stats['frames_processed'] += 1
        
        return frame_result
    
    def draw_results_with_validation(self, image: np.ndarray, frame_result: dict) -> np.ndarray:
        """
        Dibuja resultados sobre la imagen incluyendo información de validación
        """
        result_image = image.copy()
        
        mediapipe_keypoints = frame_result['mediapipe_keypoints']
        validation_result = frame_result['validation_result']
        
        if mediapipe_keypoints is not None:
            # Dibujar keypoints de MediaPipe con colores según validación
            for i, kp in enumerate(mediapipe_keypoints):
                x, y, conf = kp
                if conf > 0.1:  # Umbral mínimo para mostrar
                    # Color según estado de validación
                    if validation_result:
                        mapping = self.pose_classifier.MEDIAPIPE_TO_NVIDIA_MAPPING
                        if i in mapping and mapping[i] is not None:
                            color = (0, 255, 0)  # Verde: mapeado correctamente
                        else:
                            color = (0, 255, 255)  # Amarillo: no mapeado
                    else:
                        color = (255, 255, 255)  # Blanco: sin validación
                    
                    cv2.circle(result_image, (int(x), int(y)), 4, color, -1)
                    cv2.circle(result_image, (int(x), int(y)), 6, (0, 0, 0), 1)
                    
                    # Etiqueta con índice si es modo validación
                    if self.validation_mode:
                        cv2.putText(result_image, str(i), (int(x) + 8, int(y) - 8),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            # Dibujar clasificación si está disponible
            if frame_result['pose_classifications']:
                classification = frame_result['pose_classifications'][0]
                pose_class = classification['pose_class']
                confidence = classification['confidence']
                
                # Encontrar posición para texto
                valid_kps = mediapipe_keypoints[mediapipe_keypoints[:, 2] > 0.1]
                if len(valid_kps) > 0:
                    center_x = int(np.mean(valid_kps[:, 0]))
                    center_y = int(np.mean(valid_kps[:, 1])) - 50
                    
                    # Texto de clasificación con fondo
                    text = f"{pose_class}: {confidence:.2f}"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    
                    cv2.rectangle(result_image, 
                                (center_x - 5, center_y - text_size[1] - 10),
                                (center_x + text_size[0] + 10, center_y + 5),
                                (0, 0, 255), -1)
                    
                    cv2.putText(result_image, text,
                              (center_x, center_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Información de validación en la esquina
        if validation_result and self.validation_mode:
            y_offset = 30
            
            # Estado general
            status = "✅ OK" if validation_result['is_valid'] else "❌ ERROR"
            cv2.putText(result_image, f"Mapeo: {status}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if validation_result['is_valid'] else (0, 0, 255), 2)
            y_offset += 25
            
            # Estadísticas
            cv2.putText(result_image, f"Mapeados: {validation_result['mapped_keypoints_count']}/33", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
            
            cv2.putText(result_image, f"Errores: {len(validation_result['errors'])}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) if validation_result['errors'] else (255, 255, 255), 1)
        
        # Información general en la parte inferior
        info_text = f"MP Keypoints: {len(mediapipe_keypoints) if mediapipe_keypoints is not None else 0} | Tiempo: {frame_result['processing_time_ms']:.1f}ms"
        cv2.putText(result_image, info_text, (10, image.shape[0] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result_image
    
    def process_video(self, video_source: str = 0, output_path: str = None):
        """
        Procesa video en tiempo real con validación
        """
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            logger.error(f"❌ No se pudo abrir video: {video_source}")
            return
        
        # Configurar escritura de video si se especifica
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        logger.info("🎥 Iniciando procesamiento MediaPipe + Validación + Clasificación...")
        logger.info("   Presiona 'q' para salir")
        logger.info("   Presiona 'r' para reiniciar secuencia de clasificación")
        logger.info("   Presiona 'v' para alternar modo validación")
        logger.info("   Presiona 's' para guardar estadísticas de validación")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Procesar frame con validación y clasificación
                frame_result = self.process_frame_with_classification_and_validation(frame)
                
                # Dibujar resultados
                result_frame = self.draw_results_with_validation(frame, frame_result)
                
                # Mostrar
                cv2.imshow('MediaPipe + Validación + Clasificación', result_frame)
                
                # Guardar si se especifica
                if out:
                    out.write(result_frame)
                
                # Controles de teclado
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.pose_classifier.reset_sequence()
                    logger.info("🔄 Secuencia de clasificación reiniciada")
                elif key == ord('v'):
                    self.validation_mode = not self.validation_mode
                    logger.info(f"🔍 Validación {'activada' if self.validation_mode else 'desactivada'}")
                elif key == ord('s'):
                    self.save_validation_report()
                
        except KeyboardInterrupt:
            logger.info("⚠️ Procesamiento interrumpido")
        
        finally:
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            
            # Mostrar estadísticas finales
            self._print_final_statistics()
    
    def save_validation_report(self):
        """Guarda un reporte detallado de validación"""
        try:
            timestamp = int(time.time())
            report_path = f"validation_report_{timestamp}.json"
            
            report = {
                'timestamp': timestamp,
                'stats': self.stats.copy(),
                'validation_summary': {
                    'total_validations': len(self.validation_data['mapping_validation_results']),
                    'successful_mappings': sum(1 for r in self.validation_data['mapping_validation_results'] if r['is_valid']),
                    'mapping_errors': sum(len(r['errors']) for r in self.validation_data['mapping_validation_results']),
                    'mapping_warnings': sum(len(r['warnings']) for r in self.validation_data['mapping_validation_results'])
                },
                'mediapipe_to_nvidia_mapping': self.pose_classifier.MEDIAPIPE_TO_NVIDIA_MAPPING,
                'recent_validation_results': self.validation_data['mapping_validation_results'][-10:] if self.validation_data['mapping_validation_results'] else []
            }
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"📊 Reporte de validación guardado: {report_path}")
            
        except Exception as e:
            logger.error(f"❌ Error guardando reporte: {e}")
    
    def _print_final_statistics(self):
        """Imprime estadísticas finales del procesamiento"""
        elapsed_time = time.time() - self.stats['start_time']
        fps = self.stats['frames_processed'] / elapsed_time if elapsed_time > 0 else 0
        
        print("\n" + "="*60)
        print("📊 ESTADÍSTICAS FINALES - MediaPipe + Validación + Clasificación")
        print("="*60)
        print(f"⏱️  Tiempo total: {elapsed_time:.1f} segundos")
        print(f"🎞️  Frames procesados: {self.stats['frames_processed']}")
        print(f"📈 FPS promedio: {fps:.1f}")
        print(f"👥 Personas detectadas: {self.stats['poses_detected']}")
        print(f"🎭 Poses clasificadas: {self.stats['poses_classified']}")
        
        # Estadísticas de validación
        print(f"\n🔍 VALIDACIÓN DE TOPOLOGÍA:")
        print(f"   📊 Validaciones realizadas: {self.stats['topology_validations']}")
        print(f"   ❌ Errores de mapeo: {self.stats['mapping_errors']}")
        
        if self.stats['topology_validations'] > 0:
            error_rate = (self.stats['mapping_errors'] / self.stats['topology_validations']) * 100
            print(f"   📈 Tasa de errores: {error_rate:.1f}%")
            
            if error_rate < 5:
                print("   ✅ Mapeo de topología funcionando correctamente")
            elif error_rate < 20:
                print("   ⚠️ Algunos errores de mapeo detectados")
            else:
                print("   🔴 Múltiples errores de mapeo - revisar MEDIAPIPE_TO_NVIDIA_MAPPING")
        
        # Estadísticas del clasificador
        classifier_stats = self.pose_classifier.get_statistics()
        print(f"\n🎭 CLASIFICACIÓN:")
        print(f"   🎯 Tasa de confianza: {classifier_stats['confidence_rate']:.2f}")
        print(f"   🏆 Clase más común: {classifier_stats['most_common_class']}")
        
        print(f"\n📊 Distribución de clases:")
        for class_name, count in classifier_stats['class_distribution'].items():
            percentage = (count / classifier_stats['total_predictions']) * 100 if classifier_stats['total_predictions'] > 0 else 0
            print(f"   {class_name}: {count} ({percentage:.1f}%)")


def main():
    """Función principal de ejemplo"""
    # Configuración de rutas - AJUSTAR SEGÚN TU SISTEMA
    config = {
        'mediapipe_model': '../models/pose_landmarker_lite.task',
        'pose_classifier_model': '../models/pose_classification/st-gcn_3dbp_nvidia.engine',
        'video_source': '../Videos/Entrada/WIN_20250722_13_47_30_Pro.mp4',  # 0 para cámara
        'output_video': '../Videos/Salida/mediapipe_validation_output.mp4'  # None para no guardar
    }
    
    # Verificar archivos
    required_files = ['mediapipe_model', 'pose_classifier_model']
    for key in required_files:
        if not Path(config[key]).exists():
            print(f"❌ Archivo no encontrado: {config[key]}")
            print(f"💡 Para usar este ejemplo:")
            print(f"   1. Asegúrate de que el modelo MediaPipe .task esté disponible")
            print(f"   2. Descarga PoseClassificationNet de NGC en formato .engine")
            print(f"   3. Ajusta las rutas en la configuración")
            return False
    
    try:
        # Crear sistema integrado con validación
        print("🔧 Inicializando sistema MediaPipe + Validación + Clasificación...")
        system = MediaPipeWithClassifier(
            mediapipe_model_path=config['mediapipe_model'],
            pose_classifier_model_path=config['pose_classifier_model'],
            validation_mode=True  # ✅ Activar validación por defecto
        )
        
        # Procesar video
        system.process_video(
            video_source=config['video_source'],
            output_path=config['output_video']
        )
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en main: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        print("❌ Ejecución no exitosa")
    else:
        print("✅ Ejecución completada exitosamente")
