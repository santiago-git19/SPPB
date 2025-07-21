#!/usr/bin/env python3
"""
TRT Pose Classifier - Clasificación de Poses usando NVIDIA TAO PoseClassificationNet
===================================================================================

Clase para procesar keypoints de trt_pose y clasificar poses humanas usando
el modelo PoseClassificationNet de NVIDIA TAO Toolkit.

Soporta múltiples formatos de keypoints y clasificación en tiempo real.

Autor: Sistema de IA
Fecha: 2025
"""

import numpy as np
import torch
import logging
import os
import time
from collections import deque
from typing import List, Tuple, Dict, Optional, Union
import json
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# Configurar logging
logger = logging.getLogger(__name__)

class TRTPoseClassifier:
    """
    Clasificador de poses usando keypoints de trt_pose y PoseClassificationNet de NVIDIA TAO
    """
    
    # Mapeo de formatos de keypoints soportados
    KEYPOINT_FORMATS = {
        'coco': 17,          # COCO dataset format
        'openpose': 18,      # OpenPose format  
        'human3.6m': 17,     # Human3.6M dataset format
        'ntu-rgb+d': 25,     # NTU RGB+D dataset format
        'ntu_edge': 24,      # NTU Edge format
        'nvidia': 34         # NVIDIA 3D body pose format (target)
    }
    
    # Clases de poses que puede clasificar el modelo
    POSE_CLASSES = [
        'sitting_down',  # 0
        'getting_up',    # 1
        'sitting',       # 2
        'standing',      # 3
        'walking',       # 4
        'jumping'        # 5
    ]
    
    # Mapeo de keypoints COCO (17) a NVIDIA (34) - indices aproximados
    COCO_TO_NVIDIA_MAPPING = {
        0: 0,    # nose -> head_top
        1: 1,    # left_eye -> left_eye
        2: 2,    # right_eye -> right_eye
        3: 3,    # left_ear -> left_ear
        4: 4,    # right_ear -> right_ear
        5: 5,    # left_shoulder -> left_shoulder
        6: 6,    # right_shoulder -> right_shoulder
        7: 7,    # left_elbow -> left_elbow
        8: 8,    # right_elbow -> right_elbow
        9: 9,    # left_wrist -> left_wrist
        10: 10,  # right_wrist -> right_wrist
        11: 11,  # left_hip -> left_hip
        12: 12,  # right_hip -> right_hip
        13: 13,  # left_knee -> left_knee
        14: 14,  # right_knee -> right_knee
        15: 15,  # left_ankle -> left_ankle
        16: 16,  # right_ankle -> right_ankle
    }
    
    def __init__(self, 
                 model_path: str,
                 keypoint_format: str = 'coco',
                 sequence_length: int = 30,
                 confidence_threshold: float = 0.3,
                 max_persons: int = 1):
        """
        Inicializa el clasificador de poses
        
        Args:
            model_path: Ruta al modelo engine de PoseClassificationNet
            keypoint_format: Formato de keypoints de entrada ('coco', 'openpose', 'nvidia', etc.)
            sequence_length: Longitud de secuencia temporal para clasificación
            confidence_threshold: Umbral mínimo de confianza para keypoints
            max_persons: Número máximo de personas a procesar
        """
        self.model_path = model_path
        self.keypoint_format = keypoint_format.lower()
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        self.max_persons = max_persons
        
        # Validar formato
        if self.keypoint_format not in self.KEYPOINT_FORMATS:
            raise ValueError(f"Formato no soportado: {keypoint_format}. "
                           f"Soportados: {list(self.KEYPOINT_FORMATS.keys())}")
        
        self.input_keypoints = self.KEYPOINT_FORMATS[self.keypoint_format]
        self.target_keypoints = self.KEYPOINT_FORMATS['nvidia']  # 34 keypoints
        
        # Buffer para secuencias temporales
        self.sequence_buffer = deque(maxlen=sequence_length)
        
        # Variables TensorRT
        self.engine = None
        self.context = None
        self.runtime = None
        self.input_binding = None
        self.output_binding = None
        self.d_input = None  # Memoria GPU para entrada
        self.d_output = None  # Memoria GPU para salida
        self.input_shape = None
        self.output_shape = None
        self.input_size = None
        self.output_size = None
        
        print(f"Cargando modelo TensorRT {self.model_path}...")
        self._load_model()
        
        # Estadísticas
        self.stats = {
            'total_predictions': 0,
            'confident_predictions': 0,
            'class_predictions': {cls: 0 for cls in self.POSE_CLASSES}
        }
        
        logger.info(f"✅ TRTPoseClassifier inicializado:")
        logger.info(f"   📊 Formato entrada: {keypoint_format} ({self.input_keypoints} keypoints)")
        logger.info(f"   🎯 Formato destino: nvidia ({self.target_keypoints} keypoints)")
        logger.info(f"   ⏱️ Secuencia temporal: {sequence_length} frames")
        logger.info(f"   🎭 Clases: {self.POSE_CLASSES}")
        
    def _load_model(self):
        """Carga el modelo TensorRT .engine"""
        try:
            import pycuda.driver as cuda
            import tensorrt as trt
            
            # Verificar que el archivo existe
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Modelo no encontrado: {self.model_path}")
            
            # Inicializar CUDA
            cuda.init()
            
            # Cargar el archivo engine
            with open(self.model_path, 'rb') as f:
                engine_data = f.read()
            
            # Crear runtime TensorRT
            trt_logger = trt.Logger(trt.Logger.WARNING)
            self.runtime = trt.Runtime(trt_logger)
            
            # Deserializar el engine
            self.engine = self.runtime.deserialize_cuda_engine(engine_data)
            
            if self.engine is None:
                raise RuntimeError("Error al deserializar el engine TensorRT")
            
            # Crear contexto de ejecución
            self.context = self.engine.create_execution_context()
            
            # Obtener información de los bindings
            for i in range(self.engine.num_bindings):
                if self.engine.binding_is_input(i):
                    self.input_binding = i
                    self.input_shape = self.engine.get_binding_shape(i)
                    self.input_size = trt.volume(self.input_shape)
                else:
                    self.output_binding = i
                    self.output_shape = self.engine.get_binding_shape(i)
                    self.output_size = trt.volume(self.output_shape)
            
            # Alocar memoria GPU
            self.d_input = cuda.mem_alloc(self.input_size * np.dtype(np.float32).itemsize)
            self.d_output = cuda.mem_alloc(self.output_size * np.dtype(np.float32).itemsize)
            
            logger.info(f"✅ Modelo TensorRT cargado: {os.path.basename(self.model_path)}")
            logger.info(f"   📐 Forma entrada: {self.input_shape}")
            logger.info(f"   � Forma salida: {self.output_shape}")
            
        except ImportError as e:
            logger.error(f"❌ Error de importación: {e}")
            logger.error("💡 Asegúrese de que TensorRT y PyCUDA estén instalados correctamente")
            raise
        except Exception as e:
            logger.error(f"❌ Error cargando modelo TensorRT: {e}")
            logger.error("💡 Verifica que el archivo .engine sea válido y compatible")
            raise
    
    def _convert_keypoints_format(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Convierte keypoints del formato de entrada al formato NVIDIA (34 keypoints)
        
        Args:
            keypoints: Array de keypoints [num_keypoints, 3] (x, y, confidence)
            
        Returns:
            Array convertido al formato NVIDIA [34, 3]
        """
        # Crear array target con 34 keypoints inicializados a cero
        nvidia_keypoints = np.zeros((self.target_keypoints, 3), dtype=np.float32)
        
        if self.keypoint_format == 'nvidia':
            # Ya está en formato correcto
            return keypoints[:self.target_keypoints] if len(keypoints) >= 34 else nvidia_keypoints
        
        elif self.keypoint_format == 'coco':
            # Mapear COCO (17) a NVIDIA (34)
            for coco_idx, nvidia_idx in self.COCO_TO_NVIDIA_MAPPING.items():
                if coco_idx < len(keypoints) and nvidia_idx < self.target_keypoints:
                    nvidia_keypoints[nvidia_idx] = keypoints[coco_idx]
                    
        elif self.keypoint_format == 'openpose':
            # Mapeo similar a COCO pero con keypoint adicional
            # OpenPose tiene 18 puntos, COCO 17, mapeo aproximado
            openpose_to_nvidia = dict(self.COCO_TO_NVIDIA_MAPPING)
            openpose_to_nvidia[17] = 17  # neck adicional en OpenPose
            
            for op_idx, nvidia_idx in openpose_to_nvidia.items():
                if op_idx < len(keypoints) and nvidia_idx < self.target_keypoints:
                    nvidia_keypoints[nvidia_idx] = keypoints[op_idx]
                    
        else:
            # Para otros formatos, mapeo directo limitado
            max_copy = min(len(keypoints), self.target_keypoints)
            nvidia_keypoints[:max_copy] = keypoints[:max_copy]
        
        return nvidia_keypoints
    
    def _filter_low_confidence_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Filtra keypoints con confianza baja
        
        Args:
            keypoints: Array [num_keypoints, 3] donde la última dimensión es confianza
            
        Returns:
            Array filtrado con keypoints de baja confianza marcados como (0,0,0)
        """
        filtered = keypoints.copy()
        
        # Marcar keypoints con confianza baja
        low_confidence_mask = keypoints[:, 2] < self.confidence_threshold
        filtered[low_confidence_mask] = [0.0, 0.0, 0.0]
        
        return filtered
    
    def _normalize_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Normaliza las coordenadas de keypoints
        
        Args:
            keypoints: Array [num_keypoints, 3]
            
        Returns:
            Array normalizado
        """
        normalized = keypoints.copy()
        
        # Filtrar keypoints válidos (no cero)
        valid_mask = (keypoints[:, 0] != 0) | (keypoints[:, 1] != 0)
        
        if np.any(valid_mask):
            valid_keypoints = keypoints[valid_mask]
            
            # Normalizar coordenadas X e Y por separado
            if len(valid_keypoints) > 0:
                x_coords = valid_keypoints[:, 0]
                y_coords = valid_keypoints[:, 1]
                
                # Centrar y escalar (normalización básica)
                if np.max(x_coords) > np.min(x_coords):
                    x_center = np.mean(x_coords)
                    x_scale = np.max(x_coords) - np.min(x_coords)
                    normalized[valid_mask, 0] = (x_coords - x_center) / x_scale
                
                if np.max(y_coords) > np.min(y_coords):
                    y_center = np.mean(y_coords)
                    y_scale = np.max(y_coords) - np.min(y_coords)
                    normalized[valid_mask, 1] = (y_coords - y_center) / y_scale
        
        return normalized
    
    def _create_sequence_tensor(self) -> Optional[np.ndarray]:
        """
        Crea tensor de secuencia en formato (N, C, T, V, M) para el modelo
        
        Returns:
            Tensor de secuencia o None si no hay suficientes frames
        """
        if len(self.sequence_buffer) < self.sequence_length:
            return None
        
        # Convertir secuencia a array
        sequence_list = list(self.sequence_buffer)
        
        # Dimensiones: (T, V, C) -> (N, C, T, V, M)
        # T = tiempo (frames), V = keypoints, C = coordenadas, N = batch, M = personas
        
        # Stack temporal
        temporal_sequence = np.stack(sequence_list, axis=0)  # (T, V, C)
        
        # Reorganizar a formato modelo: (N, C, T, V, M)
        # N=1 (batch), C=3 (x,y,conf), T=sequence_length, V=34, M=1 (persona)
        model_input = temporal_sequence.transpose(2, 0, 1)  # (C, T, V)
        model_input = np.expand_dims(model_input, axis=0)   # (N, C, T, V)
        model_input = np.expand_dims(model_input, axis=-1)  # (N, C, T, V, M)
        
        return model_input.astype(np.float32)
    
    def process_keypoints(self, keypoints: Union[np.ndarray, List]) -> Optional[Dict]:
        """
        Procesa keypoints y los añade al buffer de secuencia
        
        Args:
            keypoints: Array o lista de keypoints [num_keypoints, 3] o [num_keypoints, 2]
            
        Returns:
            Diccionario con resultado de clasificación o None si no hay suficientes frames
        """
        try:
            # Convertir a numpy array si es necesario
            if isinstance(keypoints, list):
                keypoints = np.array(keypoints)
            
            # Asegurar formato correcto
            if keypoints.ndim != 2:
                logger.warning(f"⚠️ Formato de keypoints incorrecto: {keypoints.shape}")
                return None
            
            # Si solo tiene x,y, añadir columna de confianza
            if keypoints.shape[1] == 2:
                confidence_col = np.ones((keypoints.shape[0], 1))
                keypoints = np.hstack([keypoints, confidence_col])
            elif keypoints.shape[1] != 3:
                logger.warning(f"⚠️ Keypoints deben tener 2 o 3 columnas, recibidos: {keypoints.shape[1]}")
                return None
            
            # Filtrar keypoints de baja confianza
            filtered_keypoints = self._filter_low_confidence_keypoints(keypoints)
            
            # Convertir al formato NVIDIA
            nvidia_keypoints = self._convert_keypoints_format(filtered_keypoints)
            
            # Normalizar coordenadas
            normalized_keypoints = self._normalize_keypoints(nvidia_keypoints)
            
            # Añadir al buffer temporal
            self.sequence_buffer.append(normalized_keypoints)
            
            # Si tenemos suficientes frames, clasificar
            if len(self.sequence_buffer) >= self.sequence_length:
                return self._classify_sequence()
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error procesando keypoints: {e}")
            return None
    
    def _classify_sequence(self) -> Dict:
        """
        Clasifica la secuencia actual de keypoints usando TensorRT
        
        Returns:
            Diccionario con resultados de clasificación
        """
        try:
            import pycuda.driver as cuda
            
            # Crear tensor de entrada
            input_tensor = self._create_sequence_tensor()
            if input_tensor is None:
                return self._create_empty_result()
            
            # Ejecutar inferencia TensorRT
            start_time = time.time()
            
            # Copiar datos a GPU
            cuda.memcpy_htod(self.d_input, input_tensor.astype(np.float32))
            
            # Ejecutar inferencia
            bindings = [int(self.d_input), int(self.d_output)]
            self.context.execute_v2(bindings)
            
            # Copiar resultado de GPU a CPU
            h_output = np.empty(self.output_shape, dtype=np.float32)
            cuda.memcpy_dtoh(h_output, self.d_output)
            
            inference_time = time.time() - start_time
            
            # Procesar salida
            prediction_logits = h_output.flatten()  # Aplanar si es necesario
            probabilities = self._softmax(prediction_logits)
            
            # Obtener clase predicha
            predicted_class_idx = np.argmax(probabilities)
            predicted_class = self.POSE_CLASSES[predicted_class_idx]
            confidence = float(probabilities[predicted_class_idx])
            
            # Actualizar estadísticas
            self.stats['total_predictions'] += 1
            if confidence > 0.5:  # Umbral de confianza para estadísticas
                self.stats['confident_predictions'] += 1
            self.stats['class_predictions'][predicted_class] += 1
            
            # Crear resultado
            result = {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': {
                    cls: float(prob) for cls, prob in zip(self.POSE_CLASSES, probabilities)
                },
                'inference_time_ms': inference_time * 1000,
                'sequence_length': len(self.sequence_buffer),
                'keypoint_format': self.keypoint_format,
                'timestamp': time.time()
            }
            
            logger.debug(f"🎭 Clasificación TensorRT: {predicted_class} ({confidence:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error en clasificación TensorRT: {e}")
            return self._create_empty_result()
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Aplica función softmax a los logits"""
        exp_x = np.exp(x - np.max(x))  # Estabilidad numérica
        return exp_x / np.sum(exp_x)
    
    def _create_empty_result(self) -> Dict:
        """Crea resultado vacío en caso de error"""
        return {
            'predicted_class': 'unknown',
            'confidence': 0.0,
            'probabilities': {cls: 0.0 for cls in self.POSE_CLASSES},
            'inference_time_ms': 0.0,
            'sequence_length': len(self.sequence_buffer),
            'keypoint_format': self.keypoint_format,
            'timestamp': time.time(),
            'error': True
        }
    
    def reset_sequence(self):
        """Reinicia el buffer de secuencia temporal"""
        self.sequence_buffer.clear()
        logger.debug("🔄 Buffer de secuencia reiniciado")
    
    def get_statistics(self) -> Dict:
        """
        Obtiene estadísticas de uso del clasificador
        
        Returns:
            Diccionario con estadísticas
        """
        total = self.stats['total_predictions']
        confident = self.stats['confident_predictions']
        
        return {
            'total_predictions': total,
            'confident_predictions': confident,
            'confidence_rate': confident / total if total > 0 else 0.0,
            'class_distribution': self.stats['class_predictions'].copy(),
            'most_common_class': max(self.stats['class_predictions'], 
                                   key=self.stats['class_predictions'].get),
            'sequence_buffer_size': len(self.sequence_buffer),
            'sequence_buffer_max': self.sequence_buffer.maxlen
        }
    
    def save_statistics(self, filepath: str):
        """Guarda estadísticas en archivo JSON"""
        try:
            stats = self.get_statistics()
            stats['model_info'] = {
                'model_path': self.model_path,
                'keypoint_format': self.keypoint_format,
                'sequence_length': self.sequence_length,
                'confidence_threshold': self.confidence_threshold,
                'supported_classes': self.POSE_CLASSES
            }
            
            with open(filepath, 'w') as f:
                json.dump(stats, f, indent=2)
                
            logger.info(f"📊 Estadísticas guardadas en: {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Error guardando estadísticas: {e}")
    
    def __str__(self) -> str:
        """Representación string del clasificador"""
        stats = self.get_statistics()
        return (f"TRTPoseClassifier("
                f"format={self.keypoint_format}, "
                f"predictions={stats['total_predictions']}, "
                f"confidence_rate={stats['confidence_rate']:.2f})")
    
    def cleanup(self):
        """Libera la memoria GPU y limpia recursos TensorRT"""
        try:
            import pycuda.driver as cuda
            
            if hasattr(self, 'd_input') and self.d_input is not None:
                self.d_input.free()
                self.d_input = None
                
            if hasattr(self, 'd_output') and self.d_output is not None:
                self.d_output.free()
                self.d_output = None
                
            if hasattr(self, 'context') and self.context is not None:
                self.context = None
                
            if hasattr(self, 'engine') and self.engine is not None:
                self.engine = None
                
            if hasattr(self, 'runtime') and self.runtime is not None:
                self.runtime = None
                
            logger.info("✅ Memoria GPU liberada correctamente")
            
        except Exception as e:
            logger.warning(f"⚠️ Error durante limpieza: {e}")
    
    def __del__(self):
        """Destructor que asegura la limpieza de memoria"""
        self.cleanup()
    
    def __repr__(self) -> str:
        return self.__str__()


class TRTPoseClassifierManager:
    """
    Manager para manejar múltiples clasificadores de poses o diferentes configuraciones
    """
    
    def __init__(self):
        self.classifiers = {}
        self.active_classifier = None
        
    def add_classifier(self, name: str, classifier: TRTPoseClassifier):
        """Añade un clasificador al manager"""
        self.classifiers[name] = classifier
        if self.active_classifier is None:
            self.active_classifier = name
        logger.info(f"✅ Clasificador '{name}' añadido al manager")
    
    def set_active_classifier(self, name: str):
        """Cambia el clasificador activo"""
        if name in self.classifiers:
            self.active_classifier = name
            logger.info(f"🔄 Clasificador activo cambiado a: {name}")
        else:
            logger.warning(f"⚠️ Clasificador '{name}' no encontrado")
    
    def process_keypoints(self, keypoints, classifier_name: str = None) -> Optional[Dict]:
        """Procesa keypoints con el clasificador especificado o activo"""
        target_name = classifier_name or self.active_classifier
        
        if target_name and target_name in self.classifiers:
            return self.classifiers[target_name].process_keypoints(keypoints)
        else:
            logger.warning(f"⚠️ No hay clasificador disponible: {target_name}")
            return None
    
    def get_all_statistics(self) -> Dict:
        """Obtiene estadísticas de todos los clasificadores"""
        return {name: clf.get_statistics() 
                for name, clf in self.classifiers.items()}
    
    def reset_all_sequences(self):
        """Reinicia todas las secuencias"""
        for classifier in self.classifiers.values():
            classifier.reset_sequence()
        logger.info("🔄 Todas las secuencias reiniciadas")


# Función de utilidad para crear clasificador rápidamente
def create_pose_classifier(model_path: str, 
                          keypoint_format: str = 'coco',
                          **kwargs) -> TRTPoseClassifier:
    """
    Función de utilidad para crear un clasificador de poses
    
    Args:
        model_path: Ruta al modelo engine
        keypoint_format: Formato de keypoints ('coco', 'openpose', 'nvidia', etc.)
        **kwargs: Argumentos adicionales para TRTPoseClassifier
    
    Returns:
        Instancia de TRTPoseClassifier configurada
    """
    return TRTPoseClassifier(
        model_path=model_path,
        keypoint_format=keypoint_format,
        **kwargs
    )


# Ejemplo de uso
if __name__ == "__main__":
    # Configurar logging para ejemplo
    logging.basicConfig(level=logging.INFO)
    
    # Ejemplo de uso básico
    print("🎭 Ejemplo de uso de TRTPoseClassifier")
    
    # Configuración de ejemplo
    model_path = "models/pose_classification/st-gcn_3dbp_nvidia.engine"
    
    if os.path.exists(model_path):
        # Crear clasificador
        classifier = create_pose_classifier(
            model_path=model_path,
            keypoint_format='coco',
            sequence_length=30,
            confidence_threshold=0.3
        )
        
        # Simular keypoints de ejemplo (COCO format)
        example_keypoints = np.random.rand(17, 3)  # 17 keypoints COCO con x,y,confidence
        
        # Procesar varios frames
        for i in range(35):  # Más que sequence_length para obtener clasificación
            result = classifier.process_keypoints(example_keypoints + np.random.rand(17, 3) * 0.1)
            
            if result and not result.get('error', False):
                print(f"Frame {i}: {result['predicted_class']} (conf: {result['confidence']:.2f})")
        
        # Mostrar estadísticas
        stats = classifier.get_statistics()
        print(f"\n📊 Estadísticas: {stats}")
        
    else:
        print(f"⚠️ Modelo no encontrado: {model_path}")
        print("💡 Para usar este clasificador:")
        print("   1. Descarga el modelo PoseClassificationNet de NGC")
        print("   2. Convierte a formato ONNX si es necesario")
        print("   3. Actualiza la ruta del modelo")
