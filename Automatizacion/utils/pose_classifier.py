import cv2
import numpy as np
import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from .trt_pose_proc import TRTPoseProcessor

class PoseClassifier:
    def __init__(self, pose_model_path, topology_path, classification_engine_path):
        """
        Inicializa el clasificador de poses.
        
        Args:
            pose_model_path (str): Ruta al modelo TensorRT Pose
            topology_path (str): Ruta al archivo de topología
            classification_engine_path (str): Ruta al motor TensorRT de PoseClassificationNet
        """
        # Inicializar el procesador de poses
        self.pose_processor = TRTPoseProcessor(pose_model_path, topology_path)
        
        # Cargar el motor de clasificación
        self.classification_engine = self._load_classification_engine(classification_engine_path)
        self.classification_context = self.classification_engine.create_execution_context()
        
        # Configurar memoria para inferencia
        self._setup_classification_memory()
        
        # Mapeo de clases (personalizable según tu modelo)
        self.pose_classes = {
            0: "de_pie",
            1: "sentado", 
            2: "levantandose",
            3: "caminando",
            4: "equilibrio",
            5: "desconocido"
        }
        
        # Estadísticas
        self.frame_count = 0
        self.pose_history = []
        
    def _load_classification_engine(self, engine_path):
        """
        Carga el motor TensorRT para clasificación de poses.
        """
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        try:
            with open(engine_path, "rb") as f:
                runtime = trt.Runtime(TRT_LOGGER)
                return runtime.deserialize_cuda_engine(f.read())
        except Exception as e:
            print(f"Error cargando motor de clasificación: {e}")
            return None
    
    def _setup_classification_memory(self):
        """
        Configura la memoria para la inferencia de clasificación.
        """
        if self.classification_engine is None:
            return
            
        # Obtener información de los bindings
        self.input_binding_idx = self.classification_engine.get_binding_index("input")
        self.output_binding_idx = self.classification_engine.get_binding_index("output")
        
        # Obtener formas de entrada y salida
        self.input_shape = self.classification_engine.get_binding_shape(self.input_binding_idx)
        self.output_shape = self.classification_engine.get_binding_shape(self.output_binding_idx)
        
        # Asignar memoria GPU
        input_size = np.prod(self.input_shape) * np.dtype(np.float32).itemsize
        output_size = np.prod(self.output_shape) * np.dtype(np.float32).itemsize
        
        self.input_memory = cuda.mem_alloc(input_size)
        self.output_memory = cuda.mem_alloc(output_size)
    
    def _preprocess_keypoints(self, keypoints):
        """
        Preprocesa los keypoints para la clasificación.
        
        Args:
            keypoints (numpy.ndarray): Keypoints detectados [N, 3] (x, y, confidence)
            
        Returns:
            numpy.ndarray: Keypoints preprocesados para clasificación
        """
        if keypoints is None:
            return np.zeros(self.input_shape, dtype=np.float32)
        
        # Extraer solo coordenadas x, y (ignorar confidence)
        coords = keypoints[:, :2]
        
        # Aplanar las coordenadas
        flattened = coords.flatten()
        
        # Normalizar coordenadas (asumiendo que están en píxeles)
        # Nota: Ajusta según el rango esperado por tu modelo
        flattened = flattened / 1000.0  # Normalización simple
        
        # Redimensionar para coincidir con la entrada del modelo
        if len(flattened) < np.prod(self.input_shape):
            # Rellenar con ceros si faltan keypoints
            padded = np.zeros(self.input_shape, dtype=np.float32)
            padded.flat[:len(flattened)] = flattened
            return padded
        elif len(flattened) > np.prod(self.input_shape):
            # Truncar si hay demasiados keypoints
            return flattened[:np.prod(self.input_shape)].reshape(self.input_shape)
        else:
            return flattened.reshape(self.input_shape)
    
    def _classify_pose(self, keypoints):
        """
        Clasifica la pose basándose en los keypoints.
        
        Args:
            keypoints (numpy.ndarray): Keypoints detectados
            
        Returns:
            tuple: (clase_id, confianza, nombre_clase)
        """
        if self.classification_engine is None:
            return -1, 0.0, "motor_no_disponible"
        
        # Preprocesar keypoints
        input_data = self._preprocess_keypoints(keypoints)
        
        try:
            # Copiar datos a memoria GPU
            cuda.memcpy_htod(self.input_memory, input_data)
            
            # Ejecutar inferencia
            self.classification_context.execute_v2([int(self.input_memory), int(self.output_memory)])
            
            # Obtener resultados
            output_data = np.empty(self.output_shape, dtype=np.float32)
            cuda.memcpy_dtoh(output_data, self.output_memory)
            
            # Procesar resultados
            if len(output_data.shape) > 1:
                output_data = output_data.flatten()
            
            # Aplicar softmax para obtener probabilidades
            exp_scores = np.exp(output_data - np.max(output_data))
            probabilities = exp_scores / np.sum(exp_scores)
            
            # Obtener clase con mayor probabilidad
            class_id = np.argmax(probabilities)
            confidence = probabilities[class_id]
            class_name = self.pose_classes.get(class_id, "desconocido")
            
            return class_id, confidence, class_name
            
        except Exception as e:
            print(f"Error en clasificación: {e}")
            return -1, 0.0, "error"
    
    def process_video(self, video_path, output_path=None, show_video=True, fps_limit=15):
        """
        Procesa un video completo y clasifica poses en tiempo real.
        
        Args:
            video_path (str): Ruta al video de entrada
            output_path (str): Ruta al video de salida (opcional)
            show_video (bool): Si mostrar el video en tiempo real
            fps_limit (int): Límite de FPS para procesamiento
            
        Returns:
            list: Lista de resultados de clasificación por frame
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: No se pudo abrir el video en {video_path}")
            return []
        
        # Configurar escritor de video si se especifica
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps_limit, (width, height))
        
        results = []
        frame_interval = 1.0 / fps_limit
        prev_time = 0
        
        print("=== Procesando video con clasificación de poses ===")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Fin del video.")
                    break
                
                # Controlar FPS
                current_time = time.time()
                if current_time - prev_time < frame_interval:
                    continue
                prev_time = current_time
                
                # Detectar keypoints
                keypoints = self.pose_processor.process_frame(frame)
                
                # Clasificar pose
                class_id, confidence, class_name = self._classify_pose(keypoints)
                
                # Guardar resultado
                result = {
                    'frame': self.frame_count,
                    'timestamp': current_time,
                    'pose_class': class_name,
                    'confidence': confidence,
                    'keypoints_detected': keypoints is not None
                }
                results.append(result)
                self.pose_history.append(class_name)
                
                # Visualizar resultados
                if keypoints is not None:
                    frame = self.pose_processor.visualize_keypoints(frame, keypoints, draw_skeleton=True)
                
                # Añadir información de clasificación al frame
                self._draw_classification_info(frame, class_name, confidence)
                
                # Guardar frame si se especifica
                if out:
                    out.write(frame)
                
                # Mostrar frame
                if show_video:
                    cv2.imshow('Clasificación de Poses', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Mostrar progreso
                if self.frame_count % 30 == 0:
                    print(f"Frame {self.frame_count}: {class_name} ({confidence:.2f})")
                
                self.frame_count += 1
                
        except KeyboardInterrupt:
            print("\nDeteniendo procesamiento...")
        finally:
            cap.release()
            if out:
                out.release()
            if show_video:
                cv2.destroyAllWindows()
        
        return results
    
    def _draw_classification_info(self, frame, class_name, confidence):
        """
        Dibuja información de clasificación en el frame.
        """
        # Configurar texto
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        # Texto principal
        text = f"Pose: {class_name}"
        confidence_text = f"Confianza: {confidence:.2f}"
        
        # Posición del texto
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        x = 10
        y = 30
        
        # Fondo del texto
        cv2.rectangle(frame, (x-5, y-25), (x + text_size[0] + 10, y + 40), (0, 0, 0), -1)
        
        # Dibujar texto
        cv2.putText(frame, text, (x, y), font, font_scale, (0, 255, 0), thickness)
        cv2.putText(frame, confidence_text, (x, y + 25), font, font_scale * 0.7, (255, 255, 255), thickness)
    
    def get_pose_statistics(self):
        """
        Obtiene estadísticas de las poses detectadas.
        
        Returns:
            dict: Estadísticas de poses
        """
        if not self.pose_history:
            return {}
        
        # Contar frecuencia de cada pose
        pose_counts = {}
        for pose in self.pose_history:
            pose_counts[pose] = pose_counts.get(pose, 0) + 1
        
        # Calcular porcentajes
        total_frames = len(self.pose_history)
        pose_percentages = {pose: (count / total_frames) * 100 
                          for pose, count in pose_counts.items()}
        
        return {
            'total_frames': total_frames,
            'pose_counts': pose_counts,
            'pose_percentages': pose_percentages,
            'most_common_pose': max(pose_counts, key=pose_counts.get)
        }
    
    def process_single_frame(self, frame):
        """
        Procesa un solo frame y devuelve la clasificación.
        
        Args:
            frame (numpy.ndarray): Frame de entrada
            
        Returns:
            tuple: (class_name, confidence, keypoints)
        """
        # Detectar keypoints
        keypoints = self.pose_processor.process_frame(frame)
        
        # Clasificar pose
        class_id, confidence, class_name = self._classify_pose(keypoints)
        
        return class_name, confidence, keypoints
