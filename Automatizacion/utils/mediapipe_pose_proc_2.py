#!/usr/bin/env python3
"""
TensorRT Pose Processor - Detección y estimación de poses usando TensorRT con MediaPipe BlazePose
=============================================================================================

Pipeline completo para detectar personas y estimar keypoints de poses humanas usando
pose_detection_fp16.engine y pose_landmark_lite_fp16.engine con TensorRT.

Optimiza FPS mediante:
- Uso de FP16 para ambos modelos
- Pinned memory para transferencias CPU-GPU
- Preprocesamiento eficiente con preservación de aspect ratio
- Soporte para múltiples personas
- Opcional: Procesamiento por lotes

Instalación de dependencias:
    pip install opencv-python numpy pycuda
    # Para TensorRT, seguir guía oficial de NVIDIA:
    # https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html

Autor: Sistema de IA
Fecha: 2025
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
import logging
import os
import time
import math

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importar TensorRT y PyCUDA
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
    logger.info("✅ TensorRT y PyCUDA importados correctamente")
except ImportError as e:
    TRT_AVAILABLE = False
    logger.error(f"❌ TensorRT/PyCUDA no disponible: {e}")
    logger.warning("💡 Instale TensorRT y PyCUDA para usar esta clase")

class PoseDetector:
    """
    Detector de personas usando pose_detection_fp16.engine con TensorRT
    """
    def __init__(self, model_path: str = "pose_detection_fp16.engine",
                 input_width: int = 224, input_height: int = 224,
                 min_score_thresh: float = 0.5):
        """
        Inicializa el detector de poses
        
        Args:
            model_path: Ruta al modelo pose_detection_fp16.engine
            input_width: Ancho de entrada (224)
            input_height: Alto de entrada (224)
            min_score_thresh: Umbral de confianza para detecciones
        """
        if not TRT_AVAILABLE:
            raise ImportError("TensorRT y PyCUDA son requeridos")
        
        self.model_path = model_path
        self.input_width = input_width
        self.input_height = input_height
        self.min_score_thresh = min_score_thresh
        
        self.engine = None
        self.context = None
        self.runtime = None
        self.input_binding = None
        self.output_bindings = []
        self.d_input = None
        self.d_outputs = []
        self.input_shape = None
        self.output_shapes = []
        self.output_sizes = []
        self.stream = None
        
        self.dtype_size_map = {
            trt.DataType.FLOAT: 4,
            trt.DataType.HALF: 2,
            trt.DataType.INT32: 4,
            trt.DataType.INT8: 1
        }
        
        self._load_tensorrt_model()
        
        # SSD anchors para pose_detection
        self.anchors = self._generate_ssd_anchors()
        
        logger.info("✅ PoseDetector inicializado")
        logger.info(f"   � Modelo: {os.path.basename(model_path)}")
        logger.info(f"   📐 Entrada: {input_width}x{input_height}")
        
    def _generate_ssd_anchors(self) -> np.ndarray:
        """
        Genera anchors SSD según mediapipe/modules/pose_detection/pose_detection_cpu.pbtxt
        """
        num_layers = 5
        min_scale = 0.1484375
        max_scale = 0.75
        input_size_height = 224
        input_size_width = 224
        anchor_offset_x = 0.5
        anchor_offset_y = 0.5
        strides = [8, 16, 32, 32, 32]
        aspect_ratios = [1.0]
        
        anchors = []
        for layer in range(num_layers):
            stride = strides[layer]
            num_grid_x = input_size_width // stride
            num_grid_y = input_size_height // stride
            scale = min_scale + (max_scale - min_scale) * layer / (num_layers - 1)
            
            for y in range(num_grid_y):
                for x in range(num_grid_x):
                    for aspect_ratio in aspect_ratios:
                        anchor_width = scale * np.sqrt(aspect_ratio)
                        anchor_height = scale / np.sqrt(aspect_ratio)
                        cx = (x + anchor_offset_x) * stride / input_size_width
                        cy = (y + anchor_offset_y) * stride / input_size_height
                        anchors.append([cx, cy, anchor_width, anchor_height])
        
        return np.array(anchors, dtype=np.float32)
    
    def _load_tensorrt_model(self):
        """Carga el modelo TensorRT .engine"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Modelo no encontrado: {self.model_path}")
            
            with open(self.model_path, 'rb') as f:
                engine_data = f.read()
            
            trt_logger = trt.Logger(trt.Logger.WARNING)
            self.runtime = trt.Runtime(trt_logger)
            self.engine = self.runtime.deserialize_cuda_engine(engine_data)
            
            if self.engine is None:
                raise RuntimeError("Error al deserializar el engine TensorRT")
            
            self.context = self.engine.create_execution_context()
            
            self.input_binding = None
            self.output_bindings = []
            self.output_shapes = []
            self.output_sizes = []
            self.d_outputs = []
            
            for i in range(self.engine.num_bindings):
                if self.engine.binding_is_input(i):
                    self.input_binding = i
                    self.input_shape = self.engine.get_binding_shape(i)
                    self.input_size = trt.volume(self.input_shape)
                    input_dtype = self.engine.get_binding_dtype(i)
                    input_itemsize = self.dtype_size_map.get(input_dtype, 4)
                    self.h_input = cuda.pagelocked_empty(self.input_shape, dtype=np.float32)
                    self.d_input = cuda.mem_alloc(self.input_size * input_itemsize)
                    logger.info(f"📥 Input binding {i}: shape={self.input_shape}, dtype={input_dtype}")
                else:
                    output_shape = self.engine.get_binding_shape(i)
                    output_size = trt.volume(output_shape)
                    output_dtype = self.engine.get_binding_dtype(i)
                    output_itemsize = self.dtype_size_map.get(output_dtype, 4)
                    self.output_bindings.append(i)
                    self.output_shapes.append(output_shape)
                    self.output_sizes.append(output_size)
                    self.d_outputs.append(cuda.mem_alloc(output_size * output_itemsize))
                    logger.info(f"📤 Output binding {i}: shape={output_shape}, dtype={output_dtype}")
            
            self.stream = cuda.Stream()
            
            logger.info(f"✅ Modelo TensorRT cargado: {os.path.basename(self.model_path)}")
            
        except Exception as e:
            logger.error(f"❌ Error cargando modelo TensorRT: {e}")
            raise
    
    def _preprocess_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        """
        Preprocesa el frame para el modelo de detección
        """
        if frame.shape[2] != 3:
            logger.error("❌ Frame debe ser BGR (3 canales)")
            return None, 1.0, 0, 0
        
        orig_h, orig_w = frame.shape[:2]
        scale = min(self.input_width / orig_w, self.input_height / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        
        resized = cv2.resize(frame, (new_w, new_h))
        pad_left = (self.input_width - new_w) // 2
        pad_top = (self.input_height - new_h) // 2
        
        padded = np.zeros((self.input_height, self.input_width, 3), dtype=np.float32)
        padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized.astype(np.float32)
        
        rgb_frame = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        normalized = rgb_frame / 255.0
        
        batched = np.expand_dims(normalized, axis=0)
        return batched, scale, pad_left, pad_top
    
    def _decode_detections(self, regressors: np.ndarray, classificators: np.ndarray,
                          scale: float, pad_left: int, pad_top: int,
                          original_width: int, original_height: int) -> Tuple[List[np.ndarray], List[float]]:
        """
        Decodifica las salidas del modelo de detección según TensorsToDetectionsCalculator
        """
        regressors = regressors[0]  # (896, 12)
        classificators = classificators[0]  # (896, 1)
        
        bboxes = []
        rotations = []
        
        for i in range(len(self.anchors)):
            score = 1 / (1 + np.exp(-classificators[i, 0]))  # Sigmoid
            if score < self.min_score_thresh:
                continue
            
            anchor = self.anchors[i]
            cx, cy, w, h = anchor
            dx, dy, dw, dh = regressors[i, :4]
            
            # Decodificar bounding box
            box_cx = cx + dx / self.input_width
            box_cy = cy + dy / self.input_height
            box_w = w * np.exp(dw / self.input_width)
            box_h = h * np.exp(dh / self.input_height)
            
            # Convertir a coordenadas absolutas
            x_min = (box_cx - box_w / 2) * self.input_width - pad_left
            y_min = (box_cy - box_h / 2) * self.input_height - pad_top
            x_max = (box_cx + box_w / 2) * self.input_width - pad_left
            y_max = (box_cy + box_h / 2) * self.input_height - pad_top
            
            x_min /= scale
            y_min /= scale
            x_max /= scale
            y_max /= scale
            
            x_min = max(0, min(x_min, original_width - 1))
            x_max = max(0, min(x_max, original_width - 1))
            y_min = max(0, min(y_min, original_height - 1))
            y_max = max(0, min(y_max, original_height - 1))
            
            # Calcular rotación (basado en keypoint 1: full-body rotation)
            kx, ky = regressors[i, 4:6]  # Primer keypoint (mid hip center)
            kx2, ky2 = regressors[i, 6:8]  # Segundo keypoint (size & rotation)
            rotation = np.arctan2(ky2 - ky, kx2 - kx)
            
            bboxes.append(np.array([x_min, y_min, x_max, y_max], dtype=np.float32))
            rotations.append(rotation)
        
        # Non-Max Suppression
        if bboxes:
            bboxes, rotations = self._non_max_suppression(bboxes, classificators, rotations)
        
        return bboxes, rotations
    
    def _non_max_suppression(self, bboxes: List[np.ndarray], scores: np.ndarray, rotations: List[float],
                            iou_threshold: float = 0.3) -> Tuple[List[np.ndarray], List[float]]:
        """
        Aplica Non-Max Suppression para eliminar detecciones redundantes
        """
        if not bboxes:
            return [], []
        
        areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in bboxes]
        order = np.argsort(scores[:, 0])[::-1]
        
        keep_bboxes = []
        keep_rotations = []
        while order.size > 0:
            i = order[0]
            keep_bboxes.append(bboxes[i])
            keep_rotations.append(rotations[i])
            
            ious = []
            for j in order[1:]:
                box1 = bboxes[i]
                box2 = bboxes[j]
                x1 = max(box1[0], box2[0])
                y1 = max(box1[1], box2[1])
                x2 = min(box1[2], box2[2])
                y2 = min(box1[3], box2[3])
                
                inter_area = max(0, x2 - x1) * max(0, y2 - y1)
                iou = inter_area / (areas[i] + areas[j] - inter_area + 1e-10)
                ious.append(iou)
            
            keep = np.where(np.array(ious) <= iou_threshold)[0]
            order = order[1:][keep]
        
        return keep_bboxes, keep_rotations
    
    def detect(self, frame: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        """
        Detecta personas en el frame y retorna bounding boxes y rotaciones
        
        Args:
            frame: Frame BGR
            
        Returns:
            bboxes: Lista de [x_min, y_min, x_max, y_max]
            rotations: Lista de ángulos de rotación
        """
        if frame is None or frame.shape[0] == 0:
            logger.warning("⚠️ Frame vacío")
            return [], []
        
        try:
            original_height, original_width = frame.shape[:2]
            input_data, scale, pad_left, pad_top = self._preprocess_frame(frame)
            if input_data is None:
                return [], []
            
            cuda.memcpy_htod_async(self.d_input, input_data, self.stream)
            
            bindings = [None] * self.engine.num_bindings
            bindings[self.input_binding] = int(self.d_input)
            for i, output_binding in enumerate(self.output_bindings):
                bindings[output_binding] = int(self.d_outputs[i])
            
            self.context.execute_async_v2(bindings, self.stream.handle)
            
            h_outputs = []
            for i, shape in enumerate(self.output_shapes):
                h_output = np.empty(shape, dtype=np.float32)
                cuda.memcpy_dtoh_async(h_output, self.d_outputs[i], self.stream)
                h_outputs.append(h_output)
            
            self.stream.synchronize()
            
            regressors = None
            classificators = None
            for i, shape in enumerate(self.output_shapes):
                if shape[-1] == 12:
                    regressors = h_outputs[i]
                elif shape[-1] == 1:
                    classificators = h_outputs[i]
            
            if regressors is None or classificators is None:
                logger.error("❌ Salidas del modelo no encontradas")
                return [], []
            
            bboxes, rotations = self._decode_detections(regressors, classificators, scale, pad_left, pad_top,
                                                      original_width, original_height)
            logger.debug(f"✅ Detectadas {len(bboxes)} personas")
            return bboxes, rotations
            
        except Exception as e:
            logger.error(f"❌ Error en detección: {e}")
            return [], []
    
    def cleanup(self):
        """Libera recursos de TensorRT y CUDA"""
        try:
            if self.d_input:
                self.d_input.free()
            for d_output in self.d_outputs:
                d_output.free()
            self.d_outputs = []
            self.stream = None
            self.context = None
            self.engine = None
            self.runtime = None
            logger.info("✅ Recursos PoseDetector liberados")
        except Exception as e:
            logger.warning(f"⚠️ Error durante limpieza: {e}")

class MediaPipePoseProcessor:
    """
    Procesador de poses usando TensorRT con pose_landmark_lite_fp16.engine
    """
    KEYPOINT_NAMES = [
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
    
    POSE_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
        (0, 9), (0, 10), (9, 10), (2, 7), (5, 8),
        (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
        (11, 12), (11, 23), (12, 24), (23, 24),
        (23, 25), (25, 27), (27, 29), (27, 31),
        (24, 26), (26, 28), (28, 30), (28, 32)
    ]
    
    def __init__(self,
                 detector_model_path: str = "models/pose_detection_fp16.engine",
                 landmark_model_path: str = "models/pose_landmark_lite_fp16.engine",
                 input_width: int = 256,
                 input_height: int = 256,
                 confidence_threshold: float = 0.5,
                 min_score_thresh: float = 0.5,
                 visualize_scale: float = 1.0):
        """
        Inicializa el procesador de poses con detección y landmarks
        
        Args:
            detector_model_path: Ruta al modelo de detección
            landmark_model_path: Ruta al modelo de landmarks
            input_width: Ancho de entrada para landmarks
            input_height: Alto de entrada para landmarks
            confidence_threshold: Umbral para landmarks
            min_score_thresh: Umbral para detecciones
            visualize_scale: Escala para visualización
        """
        if not TRT_AVAILABLE:
            raise ImportError("TensorRT y PyCUDA son requeridos")
        
        self.detector = PoseDetector(detector_model_path, 224, 224, min_score_thresh)
        self.model_path = landmark_model_path
        self.input_width = input_width
        self.input_height = input_height
        self.confidence_threshold = confidence_threshold
        self.visualize_scale = visualize_scale
        
        self.engine = None
        self.context = None
        self.runtime = None
        self.input_binding = None
        self.output_bindings = []
        self.d_input = None
        self.d_outputs = []
        self.input_shape = None
        self.output_shapes = []
        self.output_sizes = []
        self.stream = None
        
        self.dtype_size_map = {
            trt.DataType.FLOAT: 4,
            trt.DataType.HALF: 2,
            trt.DataType.INT32: 4,
            trt.DataType.INT8: 1
        }
        
        self._load_tensorrt_model()
        
        logger.info("✅ MediaPipePoseProcessor inicializado")
        logger.info(f"   � Landmark Model: {os.path.basename(landmark_model_path)}")
        logger.info(f"   � Detector Model: {os.path.basename(detector_model_path)}")
        
    def _load_tensorrt_model(self):
        """Carga el modelo de landmarks TensorRT"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Modelo no encontrado: {self.model_path}")
            
            with open(self.model_path, 'rb') as f:
                engine_data = f.read()
            
            trt_logger = trt.Logger(trt.Logger.WARNING)
            self.runtime = trt.Runtime(trt_logger)
            self.engine = self.runtime.deserialize_cuda_engine(engine_data)
            
            if self.engine is None:
                raise RuntimeError("Error al deserializar el engine TensorRT")
            
            self.context = self.engine.create_execution_context()
            
            self.input_binding = None
            self.output_bindings = []
            self.output_shapes = []
            self.output_sizes = []
            self.d_outputs = []
            
            for i in range(self.engine.num_bindings):
                if self.engine.binding_is_input(i):
                    self.input_binding = i
                    self.input_shape = self.engine.get_binding_shape(i)
                    self.input_size = trt.volume(self.input_shape)
                    input_dtype = self.engine.get_binding_dtype(i)
                    input_itemsize = self.dtype_size_map.get(input_dtype, 4)
                    self.h_input = cuda.pagelocked_empty(self.input_shape, dtype=np.float32)
                    self.d_input = cuda.mem_alloc(self.input_size * input_itemsize)
                    logger.info(f"📥 Input binding {i}: shape={self.input_shape}, dtype={input_dtype}")
                else:
                    output_shape = self.engine.get_binding_shape(i)
                    output_size = trt.volume(output_shape)
                    output_dtype = self.engine.get_binding_dtype(i)
                    output_itemsize = self.dtype_size_map.get(output_dtype, 4)
                    self.output_bindings.append(i)
                    self.output_shapes.append(output_shape)
                    self.output_sizes.append(output_size)
                    self.d_outputs.append(cuda.mem_alloc(output_size * output_itemsize))
                    logger.info(f"📤 Output binding {i}: shape={output_shape}, dtype={output_dtype}")
            
            self.stream = cuda.Stream()
            
            logger.info(f"✅ Modelo TensorRT cargado: {os.path.basename(self.model_path)}")
            
        except Exception as e:
            logger.error(f"❌ Error cargando modelo TensorRT: {e}")
            raise
    
    def _crop_and_rotate_roi(self, frame: np.ndarray, bbox: np.ndarray, rotation: float) -> np.ndarray:
        """
        Recorta y rota el ROI según el bounding box y ángulo
        
        Args:
            frame: Frame BGR
            bbox: [x_min, y_min, x_max, y_max]
            rotation: Ángulo en radianes
            
        Returns:
            roi: Imagen recortada y rotada
        """
        x_min, y_min, x_max, y_max = bbox
        x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
        
        # Asegurar límites
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(frame.shape[1] - 1, x_max)
        y_max = min(frame.shape[0] - 1, y_max)
        
        if x_max <= x_min or y_max <= y_min:
            logger.warning("⚠️ Bounding box inválido")
            return None
        
        # Recortar ROI
        roi = frame[y_min:y_max, x_min:x_max]
        if roi.size == 0:
            logger.warning("⚠️ ROI vacío")
            return None
        
        # Rotar ROI
        center = ((x_max - x_min) / 2, (y_max - y_min) / 2)
        M = cv2.getRotationMatrix2D(center, np.degrees(-rotation), 1.0)
        roi = cv2.warpAffine(roi, M, (x_max - x_min, y_max - y_min))
        
        return roi
    
    def _preprocess_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        """
        Preprocesa el frame para el modelo de landmarks
        """
        if frame.shape[2] != 3:
            logger.error("❌ Frame debe ser BGR (3 canales)")
            return None, 1.0, 0, 0
        
        orig_h, orig_w = frame.shape[:2]
        scale = min(self.input_width / orig_w, self.input_height / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        
        resized = cv2.resize(frame, (new_w, new_h))
        pad_left = (self.input_width - new_w) // 2
        pad_top = (self.input_height - new_h) // 2
        
        padded = np.zeros((self.input_height, self.input_width, 3), dtype=np.float32)
        padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized.astype(np.float32)
        
        rgb_frame = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        normalized = rgb_frame / 255.0
        
        batched = np.expand_dims(normalized, axis=0)
        return batched, scale, pad_left, pad_top
    
    def _postprocess_output(self, output_data: np.ndarray, scale: float, pad_left: int, pad_top: int,
                           original_width: int, original_height: int, bbox: np.ndarray, rotation: float) -> np.ndarray:
        """
        Postprocesa los keypoints, ajustando por escala, padding y rotación
        """
        landmarks_flat = output_data.flatten()
        if len(landmarks_flat) < 195:
            logger.warning("⚠️ Salida del modelo más pequeña de lo esperado")
            return np.zeros((33, 3), dtype=np.float32)
        
        xyz = landmarks_flat[:117].reshape(39, 3)
        visibility = landmarks_flat[117:156]
        num_landmarks = 33
        keypoints = np.column_stack((xyz[:num_landmarks, :2], visibility[:num_landmarks]))
        
        # Desnormalizar
        keypoints[:, 0] *= self.input_width
        keypoints[:, 1] *= self.input_height
        keypoints[:, 0] -= pad_left
        keypoints[:, 1] -= pad_top
        keypoints[:, 0] /= scale
        keypoints[:, 1] /= scale
        
        # Transformar al sistema de coordenadas original
        x_min, y_min, x_max, y_max = bbox
        center = ((x_max - x_min) / 2, (y_max - y_min) / 2)
        M = cv2.getRotationMatrix2D(center, np.degrees(rotation), 1.0)
        points = keypoints[:, :2]
        points = np.hstack([points, np.ones((points.shape[0], 1))])
        transformed = (M @ points.T).T
        keypoints[:, :2] = transformed + np.array([x_min, y_min])
        
        keypoints[keypoints[:, 2] < self.confidence_threshold] = [0, 0, 0]
        return keypoints.astype(np.float32)
    
    def process_frame(self, frame: np.ndarray, max_persons: int = 4) -> List[Dict[str, np.ndarray]]:
        """
        Procesa el frame, detecta personas y estima keypoints
        
        Args:
            frame: Frame BGR
            max_persons: Máximo número de personas a procesar
            
        Returns:
            results: Lista de diccionarios con 'bbox', 'rotation', 'keypoints'
        """
        if frame is None or frame.size == 0:
            logger.warning("⚠️ Frame vacío")
            return []
        
        try:
            bboxes, rotations = self.detector.detect(frame)
            if not bboxes:
                logger.debug("⚠️ No se detectaron personas")
                return []
            
            results = []
            original_height, original_width = frame.shape[:2]
            
            for i, (bbox, rotation) in enumerate(zip(bboxes, rotations)):
                if i >= max_persons:
                    break
                
                roi = self._crop_and_rotate_roi(frame, bbox, rotation)
                if roi is None:
                    continue
                
                input_data, scale, pad_left, pad_top = self._preprocess_frame(roi)
                if input_data is None:
                    continue
                
                cuda.memcpy_htod_async(self.d_input, input_data, self.stream)
                
                bindings = [None] * self.engine.num_bindings
                bindings[self.input_binding] = int(self.d_input)
                for j, output_binding in enumerate(self.output_bindings):
                    bindings[output_binding] = int(self.d_outputs[j])
                
                self.context.execute_async_v2(bindings, self.stream.handle)
                
                output_idx = 0
                for j, size in enumerate(self.output_sizes):
                    if size == 195:
                        output_idx = j
                        break
                
                h_output = np.empty(self.output_shapes[output_idx], dtype=np.float32)
                cuda.memcpy_dtoh_async(h_output, self.d_outputs[output_idx], self.stream)
                self.stream.synchronize()
                
                keypoints = self._postprocess_output(h_output, scale, pad_left, pad_top,
                                                   original_width, original_height, bbox, rotation)
                
                results.append({
                    'bbox': bbox,
                    'rotation': rotation,
                    'keypoints': keypoints
                })
            
            logger.debug(f"✅ Procesadas {len(results)} personas")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error procesando frame: {e}")
            return []
    
    def visualize_keypoints(self, frame: np.ndarray, results: Optional[List[Dict[str, np.ndarray]]] = None,
                          draw_landmarks: bool = True, draw_connections: bool = True,
                          draw_labels: bool = False, confidence_threshold: float = 0.1) -> np.ndarray:
        """
        Visualiza los keypoints y bounding boxes
        """
        if frame is None or frame.size == 0:
            logger.warning("⚠️ Frame vacío para visualización")
            return frame
        
        if results is None:
            results = self.process_frame(frame)
        
        output_frame = frame.copy()
        scale_factor = min(frame.shape[0], frame.shape[1]) / 1080.0 * self.visualize_scale
        circle_radius = int(4 * scale_factor)
        line_thickness = int(2 * scale_factor)
        font_scale = 0.3 * scale_factor
        
        colors = {
            'face': (255, 255, 255), 'right_arm': (0, 0, 255), 'left_arm': (0, 255, 0),
            'torso': (255, 255, 0), 'right_leg': (0, 255, 255), 'left_leg': (255, 0, 255)
        }
        body_parts = {
            'face': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'left_arm': [11, 13, 15, 17, 19, 21],
            'right_arm': [12, 14, 16, 18, 20, 22],
            'torso': [11, 12, 23, 24],
            'left_leg': [23, 25, 27, 29, 31],
            'right_leg': [24, 26, 28, 30, 32]
        }
        
        for result in results:
            bbox = result['bbox']
            keypoints = result['keypoints']
            
            # Dibujar bounding box
            x_min, y_min, x_max, y_max = map(int, bbox)
            cv2.rectangle(output_frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), line_thickness)
            
            if draw_landmarks:
                for i, (x, y, confidence) in enumerate(keypoints):
                    if confidence > confidence_threshold:
                        color = (128, 128, 128)
                        for part, indices in body_parts.items():
                            if i in indices:
                                color = colors[part]
                                break
                        
                        cv2.circle(output_frame, (int(x), int(y)), circle_radius, color, -1)
                        cv2.circle(output_frame, (int(x), int(y)), circle_radius + 2, (255, 255, 255), 1)
                        
                        if draw_labels and i < len(self.KEYPOINT_NAMES):
                            label = f"{self.KEYPOINT_NAMES[i]}:{confidence:.2f}"
                            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                            text_x, text_y = int(x) + 5, int(y) - 5
                            cv2.rectangle(output_frame, (text_x, text_y - text_size[1] - 5),
                                         (text_x + text_size[0], text_y + 5), (0, 0, 0), -1, cv2.LINE_AA)
                            cv2.putText(output_frame, label, (text_x, text_y),
                                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)
            
            if draw_connections:
                for pt1_idx, pt2_idx in self.POSE_CONNECTIONS:
                    if pt1_idx < len(keypoints) and pt2_idx < len(keypoints):
                        x1, y1, conf1 = keypoints[pt1_idx]
                        x2, y2, conf2 = keypoints[pt2_idx]
                        if conf1 > confidence_threshold and conf2 > confidence_threshold:
                            cv2.line(output_frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                    (0, 255, 0), line_thickness, cv2.LINE_AA)
        
        return output_frame
    
    def get_pose_angles(self, keypoints: np.ndarray) -> dict:
        """
        Calcula ángulos importantes de la pose
        """
        angles = {}
        
        def calculate_angle(p1, p2, p3):
            v1 = p1 - p2
            v2 = p3 - p2
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            return np.degrees(np.arccos(cos_angle))
        
        try:
            if all(keypoints[[11, 13, 15], 2] > 0.1):
                angles['left_elbow'] = calculate_angle(
                    keypoints[11][:2], keypoints[13][:2], keypoints[15][:2]
                )
            if all(keypoints[[12, 14, 16], 2] > 0.1):
                angles['right_elbow'] = calculate_angle(
                    keypoints[12][:2], keypoints[14][:2], keypoints[16][:2]
                )
            if all(keypoints[[23, 25, 27], 2] > 0.1):
                angles['left_knee'] = calculate_angle(
                    keypoints[23][:2], keypoints[25][:2], keypoints[27][:2]
                )
            if all(keypoints[[24, 26, 28], 2] > 0.1):
                angles['right_knee'] = calculate_angle(
                    keypoints[24][:2], keypoints[26][:2], keypoints[28][:2]
                )
            if all(keypoints[[11, 12, 23, 24], 2] > 0.1):
                shoulder_center = (keypoints[11][:2] + keypoints[12][:2]) / 2
                hip_center = (keypoints[23][:2] + keypoints[24][:2]) / 2
                torso_vector = shoulder_center - hip_center
                vertical_vector = np.array([0, -1])
                cos_angle = np.dot(torso_vector, vertical_vector) / np.linalg.norm(torso_vector)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angles['torso_lean'] = np.degrees(np.arccos(cos_angle))
        
        except Exception as e:
            logger.warning(f"⚠️ Error calculando ángulos: {e}")
        
        return angles
    
    def get_pose_landmarks_world(self, frame: np.ndarray) -> List[Dict[str, np.ndarray]]:
        """
        Obtiene landmarks en coordenadas 3D
        """
        results = self.process_frame(frame)
        for result in results:
            keypoints = result['keypoints']
            landmarks_flat = result['raw_output'].flatten()
            xyz = landmarks_flat[:117].reshape(39, 3)
            visibility = landmarks_flat[117:156]
            result['keypoints'] = np.column_stack((xyz[:33], visibility[:33]))
        return results
    
    def cleanup(self):
        """Libera recursos de TensorRT y CUDA"""
        self.detector.cleanup()
        try:
            if self.d_input:
                self.d_input.free()
            for d_output in self.d_outputs:
                d_output.free()
            self.d_outputs = []
            self.stream = None
            self.context = None
            self.engine = None
            self.runtime = None
            logger.info("✅ Recursos MediaPipePoseProcessor liberados")
        except Exception as e:
            logger.warning(f"⚠️ Error durante limpieza: {e}")

if __name__ == "__main__":
    print("🎭 TensorRT Pose Processor - Ejemplo de uso")
    print("=" * 50)
    
    if not TRT_AVAILABLE:
        print("❌ TensorRT no está disponible")
        exit(1)
    
    # Configurar rutas de modelos
    detector_model_path = "models/pose_detection_fp16.ftlite"
    landmark_model_path = "models/pose_landmark_lite_fp16.engine"
    
    # Configurar rutas de video
    video_path = "../Videos/Entrada/sentado.mp4"
    output_video_path = "../Videos/Salida/video_procesado.mp4"
    
    use_video_file = True
    save_output_video = True
    
    try:
        processor = MediaPipePoseProcessor(
            detector_model_path=detector_model_path,
            landmark_model_path=landmark_model_path,
            input_width=256,
            input_height=256,
            confidence_threshold=0.5,
            min_score_thresh=0.5,
            visualize_scale=1.0
        )
    except Exception as e:
        print(f"❌ Error inicializando procesador: {e}")
        exit(1)
    
    if use_video_file:
        if not os.path.exists(video_path):
            print(f"❌ Video no encontrado: {video_path}")
            print("🔄 Cambiando a modo cámara web...")
            use_video_file = False
        else:
            print(f"\n📹 Procesando video: {os.path.basename(video_path)}")
            if save_output_video:
                print(f"💾 Guardando resultado en: {os.path.basename(output_video_path)}")
            print("Presiona 'q' para salir o 'SPACE' para pausar/reanudar")
            cap = cv2.VideoCapture(video_path)
            
            out = None
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"📊 Info del video: {width}x{height}, {fps:.2f} FPS, {frame_count} frames")
                
                if save_output_video:
                    output_dir = os.path.dirname(output_video_path)
                    if output_dir and not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    fourcc = cv2.VideoWriter_fourcc(*'h264')
                    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                    if not out.isOpened():
                        print(f"❌ Error configurando video de salida")
                        save_output_video = False
    
    if not use_video_file:
        print("\n📷 Iniciando captura desde cámara web...")
        print("Presiona 'q' para salir")
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara web")
        exit(1)
    
    fps_counter = 0
    start_time = time.time()
    total_inference_time = 0.0
    paused = False
    current_frame = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if use_video_file else 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("✅ Video procesado" if use_video_file else "❌ Error leyendo frame")
                break
            
            current_frame += 1
            if use_video_file and current_frame % 30 == 0:
                progress = (current_frame / total_frames) * 100
                print(f"⏳ Progreso: {progress:.1f}% ({current_frame}/{total_frames} frames)")
            
            frame_start = time.time()
            results = processor.process_frame(frame)
            process_time = time.time() - frame_start
            total_inference_time += process_time
            
            visualized = processor.visualize_keypoints(frame, results, draw_landmarks=True,
                                                    draw_connections=True, draw_labels=False)
            
            info_text = [
                "TensorRT BlazePose",
                f"Personas: {len(results)}",
                f"Process time: {process_time*1000:.1f}ms"
            ]
            if use_video_file:
                progress = (current_frame / total_frames) * 100
                info_text.extend([f"Frame: {current_frame}/{total_frames}", f"Progreso: {progress:.1f}%"])
            else:
                info_text.append(f"FPS: {1/process_time:.1f}")
            
            for result in results:
                angles = processor.get_pose_angles(result['keypoints'])
                for angle_name, angle_value in angles.items():
                    info_text.append(f"{angle_name}: {angle_value:.1f}°")
            
            scale_factor = min(frame.shape[0], frame.shape[1]) / 1080.0 * processor.visualize_scale
            font_scale = 0.6 * scale_factor
            for i, text in enumerate(info_text):
                color = (0, 255, 255) if i == 0 else (0, 255, 0)
                cv2.putText(visualized, text, (10, 30 + i*25),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)
            
            if save_output_video and out is not None:
                out.write(visualized)
            
            cv2.imshow("TensorRT BlazePose", visualized)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            print(f"⏸️ Video {'pausado' if paused else 'reanudado'}")
        
        if not paused and fps_counter % 30 == 0:
            elapsed = time.time() - start_time
            avg_fps = fps_counter / elapsed
            avg_inference = (total_inference_time / fps_counter) * 1000
            print(f"📊 FPS promedio: {avg_fps:.1f} | Inferencia promedio: {avg_inference:.1f}ms")
        
        fps_counter += 1
    
    cap.release()
    if save_output_video and out is not None:
        out.release()
        if os.path.exists(output_video_path):
            file_size = os.path.getsize(output_video_path) / (1024 * 1024)
            print(f"✅ Video guardado: {output_video_path}, {file_size:.2f} MB, {current_frame} frames")
    
    cv2.destroyAllWindows()
    processor.cleanup()
    print("\n✅ Ejemplo completado")

