#!/usr/bin/env python3
"""
TensorRT Pose Processor - Detección de poses usando TensorRT con MediaPipe BlazePose
==================================================================================

Clase para procesar frames de imágenes y detectar keypoints de poses humanas
usando el modelo pose_landmark_lite_fp16.engine con TensorRT.

MediaPipe BlazePose detecta 33 keypoints del cuerpo humano en tiempo real
con alta precisión y eficiencia computacional usando aceleración TensorRT.

Instalación de dependencias:
    pip install opencv-python numpy
    # Para TensorRT, seguir guía oficial de NVIDIA:
    # https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html
    pip install pycuda

Autor: Sistema de IA
Fecha: 2025
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
import logging
import os
import time

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
    logger.warning(f"⚠️ TensorRT/PyCUDA no disponible: {e}")
    logger.warning("💡 Para usar esta clase, instale TensorRT y PyCUDA")

class MediaPipePoseProcessor:
    """
    Procesador de poses usando TensorRT con modelo MediaPipe BlazePose
    
    Utiliza el modelo pose_landmark_lite_fp16.engine con TensorRT para detectar
    33 keypoints del cuerpo humano según la topología oficial:
    0: nose, 1: right_eye_inner, 2: right_eye, 3: right_eye_outer,
    4: left_eye_inner, 5: left_eye, 6: left_eye_outer,
    7: right_ear, 8: left_ear, 9: mouth_right, 10: mouth_left,
    11: right_shoulder, 12: left_shoulder, 13: right_elbow, 14: left_elbow,
    15: right_wrist, 16: left_wrist, 17: right_pinky_knuckle, 18: left_pinky_knuckle,
    19: right_index_knuckle, 20: left_index_knuckle, 21: right_thumb_knuckle, 22: left_thumb_knuckle,
    23: right_hip, 24: left_hip, 25: right_knee, 26: left_knee,
    27: right_ankle, 28: left_ankle, 29: right_heel, 30: left_heel,
    31: right_foot_index, 32: left_foot_index
    """
    
    # Nombres de los keypoints de MediaPipe BlazePose (33 keypoints) - Topología oficial
    KEYPOINT_NAMES = [
        'nose',                 # 0
        'right_eye_inner',      # 1
        'right_eye',            # 2
        'right_eye_outer',      # 3
        'left_eye_inner',       # 4
        'left_eye',             # 5
        'left_eye_outer',       # 6
        'right_ear',            # 7
        'left_ear',             # 8
        'mouth_right',          # 9
        'mouth_left',           # 10
        'right_shoulder',       # 11
        'left_shoulder',        # 12
        'right_elbow',          # 13
        'left_elbow',           # 14
        'right_wrist',          # 15
        'left_wrist',           # 16
        'right_pinky_knuckle',  # 17
        'left_pinky_knuckle',   # 18
        'right_index_knuckle',  # 19
        'left_index_knuckle',   # 20
        'right_thumb_knuckle',  # 21
        'left_thumb_knuckle',   # 22
        'right_hip',            # 23
        'left_hip',             # 24
        'right_knee',           # 25
        'left_knee',            # 26
        'right_ankle',          # 27
        'left_ankle',           # 28
        'right_heel',           # 29
        'left_heel',            # 30
        'right_foot_index',     # 31
        'left_foot_index'       # 32
    ]
    
    # Conexiones del esqueleto para visualización - Topología oficial MediaPipe BlazePose
    POSE_CONNECTIONS = [
        # Face connections
        (0, 1), (1, 2), (2, 3),    # right eye line
        (0, 4), (4, 5), (5, 6),    # left eye line
        (0, 9), (0, 10), (9, 10),  # mouth connections
        (2, 7), (5, 8),            # eyes to ears
        
        # Arms - Right arm
        (11, 13), (13, 15),        # right shoulder -> elbow -> wrist
        (15, 17), (15, 19), (15, 21),  # right wrist to hand points
        
        # Arms - Left arm
        (12, 14), (14, 16),        # left shoulder -> elbow -> wrist
        (16, 18), (16, 20), (16, 22),  # left wrist to hand points
        
        # Body core
        (11, 12),                  # shoulders
        (11, 23), (12, 24),        # shoulders to hips
        (23, 24),                  # hips
        
        # Legs - Right leg
        (23, 25), (25, 27),        # right hip -> knee -> ankle
        (27, 29), (27, 31),        # right ankle to foot points
        
        # Legs - Left leg
        (24, 26), (26, 28),        # left hip -> knee -> ankle
        (28, 30), (28, 32)         # left ankle to foot points
    ]
    
    def __init__(self, 
                 model_path: str = "pose_landmark_lite_fp16.engine",
                 input_width: int = 256,
                 input_height: int = 256,
                 confidence_threshold: float = 0.5):
        """
        Inicializa el procesador de poses TensorRT
        
        Args:
            model_path: Ruta al modelo pose_landmark_lite_fp16.engine
            input_width: Ancho de entrada del modelo (256)
            input_height: Alto de entrada del modelo (256)
            confidence_threshold: Umbral de confianza para los keypoints
        """
        if not TRT_AVAILABLE:
            raise ImportError("TensorRT y PyCUDA son requeridos. Instale con: pip install pycuda")
        
        self.model_path = model_path
        self.input_width = input_width
        self.input_height = input_height
        self.confidence_threshold = confidence_threshold
        
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
        self.stream = None
        
        # Cargar modelo TensorRT
        self._load_tensorrt_model()
        
        logger.info("✅ TensorRT Pose Processor inicializado correctamente")
        logger.info(f"   � Modelo: {os.path.basename(model_path)}")
        logger.info(f"   📐 Entrada: {input_width}x{input_height}")
        logger.info(f"   🎯 Confianza: {confidence_threshold}")
        
    def _load_tensorrt_model(self):
        """Carga el modelo TensorRT .engine"""
        try:
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
            self.d_input = cuda.mem_alloc(self.input_size * np.dtype(np.float16).itemsize)
            self.d_output = cuda.mem_alloc(self.output_size * np.dtype(np.float16).itemsize)
            
            # Crear stream CUDA
            self.stream = cuda.Stream()
            
            logger.info(f"✅ Modelo TensorRT cargado: {os.path.basename(self.model_path)}")
            logger.info(f"   📐 Forma entrada: {self.input_shape}")
            logger.info(f"   📊 Forma salida: {self.output_shape}")
            
        except Exception as e:
            logger.error(f"❌ Error cargando modelo TensorRT: {e}")
            raise
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocesa el frame para el modelo TensorRT
        
        Args:
            frame: Frame en formato BGR
            
        Returns:
            input_data: Datos preprocesados para TensorRT
        """
        # Redimensionar al tamaño de entrada del modelo
        resized = cv2.resize(frame, (self.input_width, self.input_height))
        
        # Convertir BGR a RGB
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalizar a [0, 1] y convertir a float16
        normalized = rgb_frame.astype(np.float16) / 255.0
        
        # Reorganizar dimensiones de HWC a CHW
        transposed = normalized.transpose(2, 0, 1)
        
        # Añadir dimensión batch: (1, C, H, W)
        batched = np.expand_dims(transposed, axis=0)
        
        # Asegurar que sea contiguo
        input_data = np.ascontiguousarray(batched)
        
        return input_data
    
    def _postprocess_output(self, output_data: np.ndarray, original_width: int, original_height: int) -> np.ndarray:
        """
        Postprocesa los resultados del modelo TensorRT
        
        Args:
            output_data: Salida del modelo TensorRT
            original_width: Ancho original del frame
            original_height: Alto original del frame
            
        Returns:
            keypoints: Array [33, 3] con keypoints (x, y, confidence)
        """
        # Reshape a la forma esperada de keypoints
        # El modelo debería devolver algo como [1, 195] que representa [33 * 3 + extras]
        # Nos quedamos con los primeros 33*3 = 99 valores
        landmarks_flat = output_data.flatten()[:99]  # 33 keypoints * 3 coords
        
        # Reshape a [33, 3]
        keypoints = landmarks_flat.reshape(33, 3)
        
        # Escalar coordenadas del modelo al tamaño original
        scale_x = original_width / self.input_width
        scale_y = original_height / self.input_height
        
        # Aplicar escalado a coordenadas x e y
        keypoints[:, 0] *= scale_x  # x coordinates
        keypoints[:, 1] *= scale_y  # y coordinates
        # keypoints[:, 2] ya es la confianza, no necesita escalado
        
        # Filtrar keypoints con confianza baja
        keypoints[keypoints[:, 2] < self.confidence_threshold] = [0, 0, 0]
        
        return keypoints.astype(np.float32)
        
    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Procesa un frame y retorna los keypoints detectados usando TensorRT
        
        Args:
            frame: Frame de imagen en formato BGR (numpy array)
            
        Returns:
            keypoints: Array de keypoints [33, 3] donde cada fila es (x, y, confidence)
                      o None si ocurre un error
        """
        if frame is None or frame.size == 0:
            logger.warning("⚠️ Frame vacío o None recibido")
            return None
        
        try:
            # Obtener dimensiones originales
            original_height, original_width = frame.shape[:2]
            
            # Preprocesar frame
            input_data = self._preprocess_frame(frame)
            
            # Copiar datos a GPU
            cuda.memcpy_htod_async(self.d_input, input_data, self.stream)
            
            # Ejecutar inferencia
            bindings = [int(self.d_input), int(self.d_output)]
            self.context.execute_async_v2(bindings, self.stream.handle)
            
            # Copiar resultado de GPU a CPU
            h_output = np.empty(self.output_shape, dtype=np.float16)
            cuda.memcpy_dtoh_async(h_output, self.d_output, self.stream)
            self.stream.synchronize()
            
            # Postprocesar resultados
            keypoints = self._postprocess_output(h_output, original_width, original_height)
            
            logger.debug(f"✅ Detectados {len(keypoints)} keypoints con TensorRT")
            return keypoints
            
        except Exception as e:
            logger.error(f"❌ Error procesando frame con TensorRT: {e}")
            return None
    
    def visualize_keypoints(self, frame: np.ndarray, 
                          keypoints: Optional[np.ndarray] = None,
                          draw_landmarks: bool = True,
                          draw_connections: bool = True,
                          draw_labels: bool = False,
                          confidence_threshold: float = 0.1) -> np.ndarray:
        """
        Visualiza los keypoints en el frame
        
        Args:
            frame: Frame original
            keypoints: Array de keypoints [33, 3] (opcional, si None usa process_frame)
            draw_landmarks: Si dibujar los landmarks
            draw_connections: Si dibujar las conexiones del esqueleto
            draw_labels: Si dibujar etiquetas de los keypoints
            confidence_threshold: Umbral de confianza para mostrar keypoints
            
        Returns:
            frame: Frame con keypoints visualizados
        """
        if frame is None or frame.size == 0:
            logger.warning("⚠️ Frame vacío para visualización")
            return frame
        
        # Si no se proporcionan keypoints, procesarlos
        if keypoints is None:
            keypoints = self.process_frame(frame)
            if keypoints is None:
                return frame
        
        # Crear una copia del frame para no modificar el original
        output_frame = frame.copy()
        
        # Colores para diferentes partes del cuerpo (actualizados para topología correcta)
        colors = {
            'face': (255, 255, 255),      # Blanco
            'right_arm': (0, 0, 255),     # Azul
            'left_arm': (0, 255, 0),      # Verde
            'torso': (255, 255, 0),       # Amarillo
            'right_leg': (0, 255, 255),   # Cian
            'left_leg': (255, 0, 255),    # Magenta
        }
        
        # Grupos de keypoints por parte del cuerpo (según topología oficial)
        body_parts = {
            'face': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],     # Face and ears
            'right_arm': [11, 13, 15, 17, 19, 21],           # Right arm and hand
            'left_arm': [12, 14, 16, 18, 20, 22],            # Left arm and hand
            'torso': [11, 12, 23, 24],                       # Shoulders and hips
            'right_leg': [23, 25, 27, 29, 31],               # Right leg and foot
            'left_leg': [24, 26, 28, 30, 32]                 # Left leg and foot
        }
        
        # Dibujar landmarks
        if draw_landmarks:
            for i, (x, y, confidence) in enumerate(keypoints):
                if confidence > confidence_threshold:
                    # Determinar color según la parte del cuerpo
                    color = (128, 128, 128)  # Gris por defecto
                    for part, indices in body_parts.items():
                        if i in indices:
                            color = colors[part]
                            break
                    
                    # Dibujar círculo
                    cv2.circle(output_frame, (int(x), int(y)), 4, color, -1)
                    cv2.circle(output_frame, (int(x), int(y)), 6, (255, 255, 255), 1)
                    
                    # Dibujar etiqueta si se solicita
                    if draw_labels and i < len(self.KEYPOINT_NAMES):
                        label = f"{self.KEYPOINT_NAMES[i]}:{confidence:.2f}"
                        cv2.putText(output_frame, label,
                                   (int(x) + 5, int(y) - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Dibujar conexiones
        if draw_connections:
            for connection in self.POSE_CONNECTIONS:
                pt1_idx, pt2_idx = connection
                
                if (pt1_idx < len(keypoints) and pt2_idx < len(keypoints)):
                    x1, y1, conf1 = keypoints[pt1_idx]
                    x2, y2, conf2 = keypoints[pt2_idx]
                    
                    # Solo dibujar si ambos puntos tienen buena confianza
                    if conf1 > confidence_threshold and conf2 > confidence_threshold:
                        cv2.line(output_frame, 
                                (int(x1), int(y1)), 
                                (int(x2), int(y2)), 
                                (0, 255, 0), 2)
        
        return output_frame
    
    def get_pose_angles(self, keypoints: np.ndarray) -> dict:
        """
        Calcula ángulos importantes de la pose
        
        Args:
            keypoints: Array de keypoints [33, 3]
            
        Returns:
            dict: Diccionario con ángulos calculados
        """
        angles = {}
        
        def calculate_angle(p1, p2, p3):
            """Calcula el ángulo entre tres puntos"""
            v1 = p1 - p2
            v2 = p3 - p2
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            
            return np.degrees(angle)
        
        try:
            # Ángulos de los brazos (corregidos según topología oficial)
            if all(keypoints[[12, 14, 16], 2] > 0.1):  # left arm (shoulder, elbow, wrist)
                angles['left_elbow'] = calculate_angle(
                    keypoints[12][:2], keypoints[14][:2], keypoints[16][:2]
                )
            
            if all(keypoints[[11, 13, 15], 2] > 0.1):  # right arm (shoulder, elbow, wrist)
                angles['right_elbow'] = calculate_angle(
                    keypoints[11][:2], keypoints[13][:2], keypoints[15][:2]
                )
            
            # Ángulos de las piernas (corregidos según topología oficial)
            if all(keypoints[[24, 26, 28], 2] > 0.1):  # left leg (hip, knee, ankle)
                angles['left_knee'] = calculate_angle(
                    keypoints[24][:2], keypoints[26][:2], keypoints[28][:2]
                )
            
            if all(keypoints[[23, 25, 27], 2] > 0.1):  # right leg (hip, knee, ankle)
                angles['right_knee'] = calculate_angle(
                    keypoints[23][:2], keypoints[25][:2], keypoints[27][:2]
                )
            
            # Ángulo del torso (inclinación) - corregido según topología oficial
            if all(keypoints[[11, 12, 23, 24], 2] > 0.1):
                shoulder_center = (keypoints[11][:2] + keypoints[12][:2]) / 2  # right + left shoulder
                hip_center = (keypoints[23][:2] + keypoints[24][:2]) / 2      # right + left hip
                
                # Ángulo con respecto a la vertical
                torso_vector = shoulder_center - hip_center
                vertical_vector = np.array([0, -1])
                
                cos_angle = np.dot(torso_vector, vertical_vector) / np.linalg.norm(torso_vector)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angles['torso_lean'] = np.degrees(np.arccos(cos_angle))
        
        except Exception as e:
            logger.warning(f"⚠️ Error calculando ángulos: {e}")
        
        return angles
    
    
    def get_pose_landmarks_world(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Obtiene landmarks en coordenadas del mundo (3D) - No disponible con TensorRT
        
        Args:
            frame: Frame de imagen
            
        Returns:
            None: Esta funcionalidad no está disponible con el modelo TensorRT
        """
        logger.warning("⚠️ Coordenadas del mundo 3D no disponibles con modelo TensorRT")
        logger.info("💡 Para coordenadas 3D use MediaPipe BlazePose directamente")
        return None
    
    def cleanup(self):
        """Libera recursos de TensorRT y CUDA"""
        try:
            if hasattr(self, 'd_input') and self.d_input is not None:
                self.d_input.free()
                self.d_input = None
                
            if hasattr(self, 'd_output') and self.d_output is not None:
                self.d_output.free()
                self.d_output = None
                
            if hasattr(self, 'stream') and self.stream is not None:
                self.stream = None
                
            if hasattr(self, 'context') and self.context is not None:
                self.context = None
                
            if hasattr(self, 'engine') and self.engine is not None:
                self.engine = None
                
            if hasattr(self, 'runtime') and self.runtime is not None:
                self.runtime = None
                
            logger.info("✅ Recursos TensorRT liberados correctamente")
            
        except Exception as e:
            logger.warning(f"⚠️ Error durante limpieza: {e}")
    
    def __del__(self):
        """Destructor que asegura la limpieza de recursos"""
        self.cleanup()
    
    def __str__(self) -> str:
        """Representación string del procesador"""
        return (f"MediaPipePoseProcessor(TensorRT, "
                f"model={os.path.basename(self.model_path)}, "
                f"input_size={self.input_width}x{self.input_height}, "
                f"confidence={self.confidence_threshold})")
    
    def __repr__(self) -> str:
        return self.__str__()

    def run_diagnostics(self) -> bool:
        """
        Ejecuta pruebas de diagnóstico completas para verificar que todo funciona
        
        Returns:
            bool: True si todas las pruebas pasan, False si hay errores
        """
        print("\n🔍 DIAGNÓSTICOS DEL SISTEMA TENSORRT")
        print("=" * 60)
        
        all_tests_passed = True
        
        # Test 1: Verificar carga del modelo
        print("\n1️⃣ Test: Carga del modelo TensorRT")
        try:
            if self.engine is not None and self.context is not None:
                print("   ✅ Modelo cargado correctamente")
                print(f"   📐 Entrada: {self.input_shape}")
                print(f"   📊 Salida: {self.output_shape}")
            else:
                print("   ❌ Error: Modelo no cargado")
                all_tests_passed = False
        except Exception as e:
            print(f"   ❌ Error verificando modelo: {e}")
            all_tests_passed = False
        
        # Test 2: Verificar memoria GPU
        print("\n2️⃣ Test: Asignación de memoria GPU")
        try:
            if self.d_input is not None and self.d_output is not None:
                print("   ✅ Memoria GPU asignada correctamente")
                print(f"   💾 Entrada: {self.input_size * 2} bytes (FP16)")
                print(f"   💾 Salida: {self.output_size * 2} bytes (FP16)")
            else:
                print("   ❌ Error: Memoria GPU no asignada")
                all_tests_passed = False
        except Exception as e:
            print(f"   ❌ Error verificando memoria: {e}")
            all_tests_passed = False
        
        # Test 3: Crear imagen de prueba
        print("\n3️⃣ Test: Procesamiento de imagen sintética")
        try:
            # Crear imagen de prueba con gradiente
            test_image = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Crear un gradiente simple
            for i in range(480):
                for j in range(640):
                    test_image[i, j] = [i % 256, j % 256, (i + j) % 256]
            
            print("   🖼️ Imagen de prueba creada (480x640)")
            
            # Preprocesar
            preprocessed = self._preprocess_frame(test_image)
            expected_shape = (1, 3, self.input_width, self.input_height)
            
            if preprocessed.shape == expected_shape:
                print(f"   ✅ Preprocesamiento correcto: {preprocessed.shape}")
                print(f"   🎯 Rango de valores: [{preprocessed.min():.3f}, {preprocessed.max():.3f}]")
            else:
                print(f"   ❌ Error en preprocesamiento: {preprocessed.shape} != {expected_shape}")
                all_tests_passed = False
                
        except Exception as e:
            print(f"   ❌ Error en preprocesamiento: {e}")
            all_tests_passed = False
        
        # Test 4: Inferencia real
        print("\n4️⃣ Test: Inferencia con TensorRT")
        try:
            # Usar la imagen de prueba anterior
            start_time = time.time()
            keypoints = self.process_frame(test_image)
            inference_time = (time.time() - start_time) * 1000
            
            if keypoints is not None:
                print(f"   ✅ Inferencia exitosa en {inference_time:.2f}ms")
                print(f"   🎯 Keypoints detectados: {len(keypoints)}")
                
                # Verificar estructura de keypoints
                if keypoints.shape == (33, 3):
                    print(f"   ✅ Estructura correcta: {keypoints.shape}")
                    
                    # Verificar rangos de confianza
                    confidences = keypoints[:, 2]
                    valid_confidences = np.sum((confidences >= 0) & (confidences <= 1))
                    print(f"   📊 Confianzas válidas: {valid_confidences}/33")
                    print(f"   📈 Rango confianza: [{confidences.min():.3f}, {confidences.max():.3f}]")
                    
                    if valid_confidences == 33:
                        print("   ✅ Todas las confianzas en rango válido [0,1]")
                    else:
                        print("   ⚠️ Algunas confianzas fuera de rango")
                else:
                    print(f"   ❌ Estructura incorrecta: {keypoints.shape} != (33, 3)")
                    all_tests_passed = False
            else:
                print("   ❌ Error: Inferencia retornó None")
                all_tests_passed = False
                
        except Exception as e:
            print(f"   ❌ Error en inferencia: {e}")
            all_tests_passed = False
        
        # Test 5: Visualización
        print("\n5️⃣ Test: Generación de visualización")
        try:
            if keypoints is not None:
                visualized = self.visualize_keypoints(
                    test_image, keypoints,
                    draw_landmarks=True,
                    draw_connections=True,
                    draw_labels=False
                )
                
                if visualized is not None and visualized.shape == test_image.shape:
                    print("   ✅ Visualización generada correctamente")
                    print(f"   📐 Tamaño: {visualized.shape}")
                else:
                    print("   ❌ Error en generación de visualización")
                    all_tests_passed = False
            else:
                print("   ⚠️ Saltando test (no hay keypoints)")
                
        except Exception as e:
            print(f"   ❌ Error en visualización: {e}")
            all_tests_passed = False
        
        # Test 6: Cálculo de ángulos
        print("\n6️⃣ Test: Cálculo de ángulos de pose")
        try:
            if keypoints is not None:
                angles = self.get_pose_angles(keypoints)
                
                if len(angles) > 0:
                    print(f"   ✅ Ángulos calculados: {len(angles)}")
                    for angle_name, angle_value in angles.items():
                        print(f"   📐 {angle_name}: {angle_value:.1f}°")
                else:
                    print("   ⚠️ No se calcularon ángulos (confianza baja)")
            else:
                print("   ⚠️ Saltando test (no hay keypoints)")
                
        except Exception as e:
            print(f"   ❌ Error calculando ángulos: {e}")
            all_tests_passed = False
        
        # Test 7: Rendimiento
        print("\n7️⃣ Test: Rendimiento y estabilidad")
        try:
            print("   🚀 Ejecutando 10 inferencias para medir rendimiento...")
            
            times = []
            for i in range(10):
                start = time.time()
                result = self.process_frame(test_image)
                end = time.time()
                times.append((end - start) * 1000)
                
                if result is None:
                    print(f"   ❌ Inferencia {i+1} falló")
                    all_tests_passed = False
                    break
            
            if len(times) == 10:
                avg_time = np.mean(times)
                std_time = np.std(times)
                min_time = np.min(times)
                max_time = np.max(times)
                
                print(f"   ✅ Rendimiento estable:")
                print(f"   ⏱️ Tiempo promedio: {avg_time:.2f}ms")
                print(f"   📊 Desviación std: {std_time:.2f}ms")
                print(f"   ⚡ Rango: {min_time:.2f}ms - {max_time:.2f}ms")
                print(f"   🎯 FPS estimado: {1000/avg_time:.1f}")
                
                if std_time < avg_time * 0.1:  # Variación < 10%
                    print("   ✅ Rendimiento muy estable")
                elif std_time < avg_time * 0.2:  # Variación < 20%
                    print("   ✅ Rendimiento estable")
                else:
                    print("   ⚠️ Rendimiento variable")
            
        except Exception as e:
            print(f"   ❌ Error en test de rendimiento: {e}")
            all_tests_passed = False
        
        # Resumen final
        print("\n" + "=" * 60)
        if all_tests_passed:
            print("🎉 TODOS LOS TESTS PASARON - SISTEMA FUNCIONANDO CORRECTAMENTE")
            print("✅ El procesador TensorRT está listo para usar")
        else:
            print("❌ ALGUNOS TESTS FALLARON - REVISAR CONFIGURACIÓN")
            print("💡 Revise los errores anteriores para solucionar problemas")
        
        print("=" * 60)
        return all_tests_passed


# Ejemplo de uso
if __name__ == "__main__":
    import time
    
    print("🎭 TensorRT Pose Processor - Ejemplo de uso")
    print("=" * 50)
    
    # Verificar disponibilidad de TensorRT
    if not TRT_AVAILABLE:
        print("❌ TensorRT no está disponible")
        print("💡 Instale TensorRT y PyCUDA para usar esta clase")
        exit(1)
    
    # Crear procesador con modelo TensorRT
    model_path = "Documentos/Trabajo/SPPB/Automatizacion/models/pose_landmark_lite_fp16.engine"
    
    if not os.path.exists(model_path):
        print(f"❌ Modelo no encontrado: {model_path}")
        print("💡 Asegúrese de que el modelo esté en la carpeta actual")
        print("💡 O proporcione la ruta completa al modelo")
        
        # Intentar con ruta relativa
        model_path = "../models/pose_landmark_lite_fp16.engine"
        if not os.path.exists(model_path):
            print("❌ Modelo tampoco encontrado en ../models/")
            print("🔍 Buscando modelos .engine disponibles...")
            
            # Buscar modelos .engine en directorios comunes
            search_paths = [".", "../models", "models", "../"]
            found_models = []
            
            for path in search_paths:
                if os.path.exists(path):
                    for file in os.listdir(path):
                        if file.endswith('.engine'):
                            found_models.append(os.path.join(path, file))
            
            if found_models:
                print(f"📁 Modelos .engine encontrados:")
                for model in found_models:
                    print(f"   • {model}")
                model_path = found_models[0]
                print(f"🎯 Usando modelo: {model_path}")
            else:
                print("🚫 No se encontraron modelos .engine")
                exit(1)
    
    # Configurar ruta del video a procesar
    # ¡CAMBIA ESTA RUTA POR LA DE TU VIDEO!
    video_path = r"../Videos/Entrada/sentado.mp4"  # <-- MODIFICA ESTA LÍNEA
    
    # Configurar ruta del video de salida
    # ¡CAMBIA ESTA RUTA POR DONDE QUIERES GUARDAR EL RESULTADO!
    output_video_path = r"../Videos/Salida/video_procesado.mp4"  # <-- MODIFICA ESTA LÍNEA

    # Si no especificas un video, se usará la cámara web
    use_video_file = True  # Cambiar a False para usar cámara web
    save_output_video = True  # Cambiar a False para no guardar video
    
    try:
        processor = MediaPipePoseProcessor(
            model_path=model_path,
            input_width=256,
            input_height=256,
            confidence_threshold=0.5
        )
        
        # Ejecutar diagnósticos automáticamente
        print("\n🔧 Ejecutando diagnósticos del sistema...")
        diagnostics_passed = processor.run_diagnostics()
        
        if not diagnostics_passed:
            print("\n⚠️ Se detectaron problemas en los diagnósticos")
            print("💡 ¿Desea continuar de todos modos? (y/n)")
            
            # Para modo automático, continuar sin pedir input
            # En modo interactivo, descomente la siguiente línea:
            # user_input = input().lower().strip()
            user_input = 'y'  # Continuar automáticamente
            
            if user_input != 'y':
                print("🚪 Saliendo del programa")
                processor.cleanup()
                exit(1)
        else:
            print("\n✅ Todos los diagnósticos pasaron - Continuando...")
            
    except Exception as e:
        print(f"❌ Error inicializando procesador: {e}")
        exit(1)
    
    # Determinar fuente de video
    if use_video_file:
        # Opción 1: Procesar desde archivo de video
        if not os.path.exists(video_path):
            print(f"❌ Video no encontrado: {video_path}")
            print("💡 Verifique la ruta del video en la variable 'video_path'")
            print("💡 Formatos soportados: .mp4, .avi, .mov, .mkv, .wmv")
            
            # Buscar videos en el directorio actual
            print("\n🔍 Buscando videos en el directorio actual...")
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
            found_videos = []
            
            for file in os.listdir('.'):
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    found_videos.append(file)
            
            if found_videos:
                print("📁 Videos encontrados:")
                for video in found_videos:
                    print(f"   • {video}")
                video_path = found_videos[0]
                print(f"🎯 Usando video: {video_path}")
            else:
                print("🚫 No se encontraron videos")
                print("🔄 Cambiando a modo cámara web...")
                use_video_file = False
        
        if use_video_file:
            print(f"\n📹 Procesando video: {os.path.basename(video_path)}")
            if save_output_video:
                print(f"💾 Guardando resultado en: {os.path.basename(output_video_path)}")
            print("Presiona 'q' para salir o 'SPACE' para pausar/reanudar")
            cap = cv2.VideoCapture(video_path)
            
            # Variables para el video de salida
            out = None
            
            # Obtener información del video
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                print("📊 Info del video:")
                print(f"   • Resolución: {width}x{height}")
                print(f"   • FPS: {fps:.2f}")
                print(f"   • Frames: {frame_count}")
                print(f"   • Duración: {duration:.2f} segundos")
                
                # Configurar escritor de video si se va a guardar
                if save_output_video:
                    try:
                        # Crear directorio de salida si no existe
                        output_dir = os.path.dirname(output_video_path)
                        if output_dir and not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                            print(f"📁 Directorio creado: {output_dir}")
                        
                        # Configurar codec y escritor
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # o 'XVID' para .avi
                        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                        
                        if out.isOpened():
                            print(f"✅ Video de salida configurado: {output_video_path}")
                        else:
                            print(f"❌ Error configurando video de salida: {output_video_path}")
                            save_output_video = False
                            
                    except Exception as e:
                        print(f"❌ Error creando video de salida: {e}")
                        save_output_video = False
            else:
                print(f"❌ No se pudo abrir el video: {video_path}")
                use_video_file = False
    
    if not use_video_file:
        # Opción 2: Procesar desde cámara web
        print("\n📷 Iniciando captura desde cámara web...")
        print("Presiona 'q' para salir")
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara web")
        print("💡 Intenta con un archivo de video o imagen")
        
        # Opción 2: Procesar imagen de ejemplo
        print("\n🖼️ Creando imagen de ejemplo...")
        
        # Crear una imagen de ejemplo (negro con texto)
        example_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(example_frame, "Coloca una persona aqui", 
                   (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(example_frame, "TensorRT BlazePose", 
                   (200, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Procesar imagen de ejemplo
        keypoints = processor.process_frame(example_frame)
        
        if keypoints is not None:
            print(f"✅ Detectados {len(keypoints)} keypoints")
            
            # Visualizar
            visualized = processor.visualize_keypoints(
                example_frame, keypoints, 
                draw_landmarks=True, 
                draw_connections=True,
                draw_labels=True
            )
            
            #cv2.imshow("TensorRT Pose - Ejemplo", visualized)
            cv2.waitKey(5000)  # Mostrar por 5 segundos
        else:
            print("🚫 No se detectaron poses en la imagen de ejemplo")
        
        cv2.destroyAllWindows()
    
    else:
        # Procesar desde cámara web o video
        fps_counter = 0
        start_time = time.time()
        total_inference_time = 0.0
        paused = False
        current_frame = 0
        
        # Variables para el progreso del video
        if use_video_file:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"\n🎬 Iniciando procesamiento de {total_frames} frames...")
        
        while True:
            if not paused:
                ret, frame = cap.read()
                
                if not ret:
                    if use_video_file:
                        print("✅ Video procesado completamente")
                    else:
                        print("❌ Error leyendo frame de la cámara")
                    break
                
                current_frame += 1
                
                # Mostrar progreso para videos
                if use_video_file and current_frame % 30 == 0:
                    progress = (current_frame / total_frames) * 100
                    print(f"⏳ Progreso: {progress:.1f}% ({current_frame}/{total_frames} frames)")
                
                # Procesar frame
                frame_start = time.time()
                keypoints = processor.process_frame(frame)
                process_time = time.time() - frame_start
                total_inference_time += process_time
                
                # Crear frame visualizado
                if keypoints is not None:
                    # Calcular ángulos
                    angles = processor.get_pose_angles(keypoints)
                    
                    # Visualizar keypoints
                    visualized = processor.visualize_keypoints(
                        frame, keypoints,
                        draw_landmarks=True,
                        draw_connections=True,
                        draw_labels=False
                    )
                    
                    # Mostrar información en pantalla
                    info_text = [
                        "TensorRT BlazePose",
                        f"Keypoints: {len(keypoints)}",
                        f"Process time: {process_time*1000:.1f}ms"
                    ]
                    
                    # Añadir información específica según el modo
                    if use_video_file:
                        progress = (current_frame / total_frames) * 100
                        info_text.extend([
                            f"Frame: {current_frame}/{total_frames}",
                            f"Progreso: {progress:.1f}%"
                        ])
                    else:
                        info_text.append(f"FPS: {1/process_time:.1f}")
                    
                    # Añadir ángulos a la información
                    for angle_name, angle_value in angles.items():
                        info_text.append(f"{angle_name}: {angle_value:.1f}°")
                    
                    # Dibujar información
                    for i, text in enumerate(info_text):
                        color = (0, 255, 255) if i == 0 else (0, 255, 0)  # Amarillo para título
                        cv2.putText(visualized, text, (10, 30 + i*25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    display_frame = visualized
                else:
                    # No se detectaron poses
                    cv2.putText(frame, "No pose detected (TensorRT)", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    display_frame = frame
                
                # Guardar frame en video de salida
                if save_output_video and out is not None and out.isOpened():
                    out.write(display_frame)
                
                # Mostrar frame
                #cv2.imshow("TensorRT BlazePose - Tiempo Real", display_frame)
            
            # Controles de teclado
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Barra espaciadora para pausar/reanudar
                paused = not paused
                status = "pausado" if paused else "reanudado"
                print(f"⏸️ Video {status}")
            
            # Calcular FPS promedio (solo si no está pausado)
            if not paused:
                fps_counter += 1
                if fps_counter % 30 == 0:
                    elapsed = time.time() - start_time
                    avg_fps = fps_counter / elapsed
                    avg_inference = (total_inference_time / fps_counter) * 1000
                    print(f"📊 FPS promedio: {avg_fps:.1f} | Inferencia promedio: {avg_inference:.1f}ms")
        
        # Cerrar y limpiar recursos
        cap.release()
        
        # Finalizar video de salida
        if save_output_video and out is not None:
            out.release()
            if os.path.exists(output_video_path):
                file_size = os.path.getsize(output_video_path) / (1024 * 1024)  # MB
                print(f"✅ Video guardado exitosamente:")
                print(f"   📁 Ruta: {output_video_path}")
                print(f"   📊 Tamaño: {file_size:.2f} MB")
                print(f"   🎬 Frames procesados: {current_frame}")
            else:
                print(f"❌ Error: El video no se guardó correctamente")
        
        cv2.destroyAllWindows()
    
    # Limpiar recursos
    processor.cleanup()
    print("\n✅ Ejemplo completado exitosamente")
    print("\n📋 Información de TensorRT BlazePose:")
    print("   • Modelo: pose_landmark_lite_fp16.engine")
    print("   • Total: 33 keypoints")
    print("   • Aceleración: TensorRT (GPU)")
    print("   • Precisión: FP16 (half precision)")
    print("\n💡 Para integrar con otras clases:")
    print("   from utils.mediapipe_pose_proc import MediaPipePoseProcessor")
    print("   processor = MediaPipePoseProcessor('pose_landmark_lite_fp16.engine')")
    print("   keypoints = processor.process_frame(frame)  # [33, 3] array")
    print("\n🔧 Dependencias necesarias:")
    print("   • TensorRT (seguir guía oficial de NVIDIA)")
    print("   • PyCUDA: pip install pycuda")
    print("   • OpenCV: pip install opencv-python")
    print("   • NumPy: pip install numpy")
