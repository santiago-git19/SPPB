#!/usr/bin/env python3
"""
Ejemplo de Uso: TRT Pose + Clasificación de Poses
=================================================

Este script demuestra cómo integrar trt_pose con el clasificador de poses
PoseClassificationNet de NVIDIA TAO para detección y clasificación en tiempo real.

Flujo de trabajo:
1. trt_pose detecta keypoints de personas en video
2. TRTPoseClassifier procesa y clasifica las poses
3. Se muestra el resultado en tiempo real

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

import sys
from pathlib import Path
# Añadir el directorio 'Automatizacion' al sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Importar procesador existente de trt_pose y nuestro clasificador
from utils.trt_pose_proc import TRTPoseProcessor
from Automatizacion.utils.action_classifier import TRTPoseClassifier, create_pose_classifier


# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TRTPoseWithClassifier:
    """
    Integra trt_pose con clasificación de poses en tiempo real
    Usa el procesador existente TRTPoseProcessor y añade clasificación
    """
    
    def __init__(self, 
                 trt_pose_model_path: str,
                 pose_topology_path: str,
                 pose_classifier_model_path: str,
                 width: int = 224,
                 height: int = 224):
        """
        Inicializa el sistema completo
        
        Args:
            trt_pose_model_path: Ruta al modelo TensorRT de trt_pose
            pose_topology_path: Ruta al archivo JSON de topología
            pose_classifier_model_path: Ruta al modelo engine del clasificador
            width: Ancho de entrada para trt_pose
            height: Alto de entrada para trt_pose
        """
        self.width = width
        self.height = height
        
        # Crear procesador de trt_pose usando la clase existente
        self.trt_pose_processor = TRTPoseProcessor(
            model_path=trt_pose_model_path,
            topology_path=pose_topology_path,
            use_tensorrt=False
        )
        
        # Crear clasificador de poses
        self.pose_classifier = create_pose_classifier(
            model_path=pose_classifier_model_path,
            keypoint_format='nvidia',  # Usar formato NVIDIA según documentación oficial
            sequence_length=15,      
            confidence_threshold=0.07
        )
        
        # Estadísticas
        self.stats = {
            'frames_processed': 0,
            'poses_detected': 0,
            'poses_classified': 0,
            'start_time': time.time()
        }
        
        logger.info("✅ Sistema TRT Pose + Clasificador inicializado")
        logger.info("   🔧 Usando TRTPoseProcessor existente")
        logger.info("   🎭 Clasificador de poses integrado")
        
    def process_frame_with_classification(self, image: np.ndarray) -> dict:
        """
        Procesa un frame completo: detección + clasificación
        """
        frame_result = {
            'people_detected': 0,
            'pose_classifications': [],
            'processing_time_ms': 0,
            'keypoints_extracted': []
        }
        
        start_time = time.time()
        
        try:
            # Usar el procesador existente para obtener keypoints
            keypoints = self.trt_pose_processor.process_frame(image)  # ✅ Un solo array (17, 3)
            
            # Verificar que los keypoints sean válidos
            if keypoints is not None and isinstance(keypoints, np.ndarray):
                # ✅ SOLUCIÓN: Es una sola persona, no una lista
                frame_result['people_detected'] = 1
                frame_result['keypoints_extracted'] = keypoints
                
                # Clasificar poses
                classification_result = self.pose_classifier.process_keypoints(keypoints)
                
                if classification_result and not classification_result.get('error', False):
                    frame_result['pose_classifications'].append({
                        'person_id': 0,
                        'pose_class': classification_result['predicted_class'],
                        'confidence': classification_result['confidence'],
                        'probabilities': classification_result['probabilities']
                    })
                    
                    self.stats['poses_classified'] += 1
                
                self.stats['poses_detected'] += 1
            
        except Exception as e:
            logger.error(f"❌ Error procesando frame: {e}")
        
        finally:
            frame_result['processing_time_ms'] = (time.time() - start_time) * 1000
            self.stats['frames_processed'] += 1
        
        return frame_result
    
    def draw_results(self, image: np.ndarray, frame_result: dict) -> np.ndarray:
        """
        Dibuja resultados sobre la imagen
        
        Args:
            image: Imagen original
            frame_result: Resultados del procesamiento
            
        Returns:
            Imagen con resultados dibujados
        """
        result_image = image.copy()
        
        # ✅ SOLUCIÓN: keypoints_extracted es un solo array (17, 3), no una lista
        keypoints = frame_result['keypoints_extracted']
        
        if keypoints is not None and len(keypoints) > 0:
            # Encontrar clasificación (solo hay una persona)
            classification = None
            if frame_result['pose_classifications']:
                classification = frame_result['pose_classifications'][0]
            
            # Dibujar keypoints
            for kp in keypoints:
                x, y, conf = kp
                if conf > self.pose_classifier.confidence_threshold:  # Solo keypoints con buena confianza
                    cv2.circle(result_image, (int(x), int(y)), 3, (0, 255, 0), -1)
            
            # Dibujar clasificación si está disponible
            if classification:
                pose_class = classification['pose_class']
                confidence = classification['confidence']
                
                # Encontrar posición para texto (centroide de keypoints válidos)
                valid_kps = keypoints[keypoints[:, 2] > self.pose_classifier.confidence_threshold]
                if len(valid_kps) > 0:
                    center_x = int(np.mean(valid_kps[:, 0]))
                    center_y = int(np.mean(valid_kps[:, 1])) - 30
                    
                    # Texto con fondo
                    text = f"{pose_class}: {confidence:.2f}"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    
                    cv2.rectangle(result_image, 
                                (center_x - 5, center_y - text_size[1] - 5),
                                (center_x + text_size[0] + 5, center_y + 5),
                                (0, 0, 0), -1)
                    
                    cv2.putText(result_image, text,
                              (center_x, center_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Información general
        info_text = f"Personas: {frame_result['people_detected']} | Tiempo: {frame_result['processing_time_ms']:.1f}ms"
        cv2.putText(result_image, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return result_image
    
    def process_video(self, video_source: str = 0, output_path: str = None):
        """
        Procesa video en tiempo real
        
        Args:
            video_source: Fuente de video (0 para cámara, ruta para archivo)
            output_path: Ruta para guardar video procesado (opcional)
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
        
        logger.info("🎥 Iniciando procesamiento de video...")
        logger.info("   Presiona 'q' para salir")
        logger.info("   Presiona 'r' para reiniciar secuencia de clasificación")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Procesar frame con clasificación
                frame_result = self.process_frame_with_classification(frame)
                
                # Dibujar resultados
                result_frame = self.draw_results(frame, frame_result)
                
                # Mostrar
                cv2.imshow('TRT Pose + Clasificación', result_frame)
                
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
                
        except KeyboardInterrupt:
            logger.info("⚠️ Procesamiento interrumpido")
        
        finally:
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            
            # Mostrar estadísticas finales
            self._print_final_statistics()
    
    def _print_final_statistics(self):
        """Imprime estadísticas finales del procesamiento"""
        elapsed_time = time.time() - self.stats['start_time']
        fps = self.stats['frames_processed'] / elapsed_time if elapsed_time > 0 else 0
        
        print("\n" + "="*50)
        print("📊 ESTADÍSTICAS FINALES")
        print("="*50)
        print(f"⏱️  Tiempo total: {elapsed_time:.1f} segundos")
        print(f"🎞️  Frames procesados: {self.stats['frames_processed']}")
        print(f"📈 FPS promedio: {fps:.1f}")
        print(f"👥 Personas detectadas: {self.stats['poses_detected']}")
        print(f"🎭 Poses clasificadas: {self.stats['poses_classified']}")
        
        # Estadísticas del clasificador
        classifier_stats = self.pose_classifier.get_statistics()
        print(f"🎯 Tasa de confianza: {classifier_stats['confidence_rate']:.2f}")
        print(f"🏆 Clase más común: {classifier_stats['most_common_class']}")
        
        print("\n📊 Distribución de clases:")
        for class_name, count in classifier_stats['class_distribution'].items():
            percentage = (count / classifier_stats['total_predictions']) * 100 if classifier_stats['total_predictions'] > 0 else 0
            print(f"   {class_name}: {count} ({percentage:.1f}%)")


def main():
    """Función principal de ejemplo"""
    # Configuración de rutas - AJUSTAR SEGÚN TU SISTEMA
    config = {
        'trt_pose_model': '/home/mobilenet/Documentos/Trabajo/trt_pose/models/resnet18_baseline_att_224x224_A_epoch_249.pth',
        'pose_topology': '/home/mobilenet/Documentos/Trabajo/trt_pose/tasks/human_pose/human_pose.json',
        'pose_classifier_model': '/home/mobilenet/Documentos/Trabajo/SPPB/Automatizacion/models/pose_classification/st-gcn_3dbp_nvidia.engine',
        'video_source': '/home/mobilenet/Documentos/Trabajo/SPPB/Automatizacion/Videos/Entrada/3052594-hd_1080_1920_24fps.mp4',  # Ruta para archivo de video
        'output_video': '/home/mobilenet/Documentos/Trabajo/SPPB/Automatizacion/Videos/Salida/caminando_salida.mp4'  # 'output_with_poses.mp4' para guardar
    }
    
    # Verificar archivos
    required_files = ['trt_pose_model', 'pose_topology', 'pose_classifier_model']
    for key in required_files:
        if not Path(config[key]).exists():
            print(f"❌ Archivo no encontrado: {config[key]}")
            print(f"💡 Para usar este ejemplo:")
            print(f"   1. Asegúrate de que el modelo TRT pose esté convertido")
            print(f"   2. Descarga PoseClassificationNet de NGC en formato ONNX")
            print(f"   3. Ajusta las rutas en la configuración")
            return False
    
    try:
        # Crear sistema integrado
        print("🔧 Inicializando sistema TRT Pose + Clasificación...")
        system = TRTPoseWithClassifier(
            trt_pose_model_path=config['trt_pose_model'],
            pose_topology_path=config['pose_topology'],
            pose_classifier_model_path=config['pose_classifier_model']
        )
        
        # Procesar video
        system.process_video(
            video_source=config['video_source'],
            output_path=config['output_video']
        )
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en main: {e}")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        print("Not successful")
