import torch
import torch2trt
from torch2trt import TRTModule
import trt_pose.coco
import trt_pose.models
import json
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from scipy.ndimage import maximum_filter, gaussian_filter
from typing import List, Optional, Tuple
from .pose_detection import PoseDetection

class TRTPoseProcessor(PoseDetection):
    def __init__(self, model_path, topology_path, use_tensorrt=True):
        """
        Inicializa el procesador de pose estimation
        
        Args:
            model_path: Ruta al modelo (.pth)
            topology_path: Ruta al archivo de topología JSON
            use_tensorrt: Si usar TensorRT (True) o PyTorch normal (False)
        """
        self.use_tensorrt = use_tensorrt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Cargar topología
        print("Cargando topología...")
        with open(topology_path, 'r') as f:
            self.human_pose = json.load(f)
        print("Topología cargada.")
        
        self.topology = trt_pose.coco.coco_category_to_topology(self.human_pose)
        self.num_parts = len(self.human_pose['keypoints'])
        self.num_links = len(self.human_pose['skeleton'])
        
        # Configurar transformaciones
        self.WIDTH = 224
        self.HEIGHT = 224
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).to(self.device)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).to(self.device)
        
        # Cargar modelo
        print(f"Cargando modelo {model_path}...")
        self.load_model(model_path)
        
        print(f"Modelo cargado exitosamente en: {self.device}")
        print(f"Usando TensorRT: {self.use_tensorrt}")
        
    def load_model(self, model_path):
        """Carga el modelo según el tipo especificado"""
        if self.use_tensorrt:
            try:
                # Intentar cargar como modelo TensorRT
                self.model = TRTModule()
                self.model.load_state_dict(torch.load(model_path))
                print(f"Modelo TensorRT cargado desde: {model_path}")
            except Exception as e:
                print(f"Error cargando modelo TensorRT: {e}")
                print("Intentando cargar como modelo PyTorch normal...")
                self.use_tensorrt = False
                self._load_pytorch_model(model_path)
        else:
            self._load_pytorch_model(model_path)
    
    def _load_pytorch_model(self, model_path):
        """Carga el modelo PyTorch normal"""
        # Crear modelo
        self.model = trt_pose.models.resnet18_baseline_att(
            self.num_parts, 2 * self.num_links
        ).to(self.device)
        
        # Cargar pesos
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"Modelo PyTorch cargado desde: {model_path}")
    
    def preprocess_image(self, image):
        """Preprocesa la imagen para el modelo"""
        # Redimensionar imagen
        image = cv2.resize(image, (self.WIDTH, self.HEIGHT))
        
        # Convertir BGR a RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convertir a tensor
        image = torch.from_numpy(image).float().to(self.device)
        image = image.permute(2, 0, 1)  # HWC a CHW
        
        # Normalizar
        image = image / 255.0
        image = (image - self.mean.view(-1, 1, 1)) / self.std.view(-1, 1, 1)
        
        # Añadir dimensión batch
        image = image.unsqueeze(0)
        
        return image
    
    def process_frame(self, frame):
        """
        Procesa un frame y retorna los keypoints detectados
        
        Args:
            frame: Frame de imagen (numpy array)
            
        Returns:
            keypoints: Lista de keypoints detectados
        """
        with torch.no_grad():
            original_shape = frame.shape[:2]

            # Preprocesar imagen
            input_tensor = self.preprocess_image(frame)
            
            # Realizar inferencia
            cmap, paf = self.model(input_tensor)
            
            # Postprocesar resultados
            keypoints = self.postprocess_results(cmap, paf, original_shape)

            return keypoints
    
    def postprocess_results(self, cmap, paf, original_shape):
        """
        Postprocesa los resultados del modelo para obtener keypoints
        
        Args:
            cmap: Confidence maps
            paf: Part Affinity Fields
            original_shape: Forma original de la imagen (height, width)
            
        Returns:
            keypoints: Lista de keypoints detectados
        """
        # Convertir a numpy y obtener el shape correcto
        cmap = cmap.squeeze().cpu().numpy()
        paf = paf.squeeze().cpu().numpy()
        
        # Obtener dimensiones
        height, width = original_shape
        cmap_height, cmap_width = cmap.shape[1], cmap.shape[2]
        
        # Calcular escalas
        scale_x = width / cmap_width
        scale_y = height / cmap_height
        
        # Inicializar keypoints con ceros
        #keypoints = np.zeros((self.num_parts, 3), dtype=np.float32)  # (num_keypoints, 3)
        keypoints = []
        
        # Umbral para detección de keypoints
        threshold = 0.1
        
        for i in range(self.num_parts):
            confidence_map = cmap[i]
            
            # Aplicar suavizado gaussiano para reducir ruido
            confidence_map = gaussian_filter(confidence_map, sigma=1.0)
            
            # Encontrar peaks
            peaks = self._find_peaks_improved(confidence_map, threshold)
            
            # Escalar coordenadas al tamaño original
            if len(peaks)>0:
                x, y, confidence = max(peaks, key=lambda p: p[2])
                # Escalar las coordenadas correctamente
                x_scaled = int(x * scale_x)
                y_scaled = int(y * scale_y)
                keypoints.append((x_scaled, y_scaled, confidence, i))

                #if i < len(keypoints):
                    #keypoints[i]= [x_scaled, y_scaled, confidence]

                
        
        return keypoints
    
    def _find_peaks_improved(self, confidence_map, threshold):
        """Encuentra peaks en el mapa de confianza con método mejorado"""
        peaks = []
        
        # Aplicar filtro de máximos locales con ventana más pequeña
        local_maxima = maximum_filter(confidence_map, size=3) == confidence_map
        
        # Aplicar umbral
        above_threshold = confidence_map > threshold
        
        # Combinar condiciones
        peak_mask = local_maxima & above_threshold
        
        # Obtener coordenadas de peaks
        y_coords, x_coords = np.where(peak_mask)
        
        # Ordenar por confianza y tomar solo el mejor peak por región
        peak_candidates = []
        for x, y in zip(x_coords, y_coords):
            confidence = confidence_map[y, x]
            peak_candidates.append((x, y, confidence))
        
        # Ordenar por confianza descendente
        peak_candidates.sort(key=lambda x: x[2], reverse=True)
        
        # Suprimir picos no máximos (NMS simple)
        final_peaks = []
        for candidate in peak_candidates:
            x, y, conf = candidate
            
            # Verificar si está muy cerca de un pico ya seleccionado
            too_close = False
            for selected_peak in final_peaks:
                sx, sy, _ = selected_peak
                dist = np.sqrt((x - sx)**2 + (y - sy)**2)
                if dist < 5:  # Distancia mínima entre peaks
                    too_close = True
                    break
            
            if not too_close:
                final_peaks.append(candidate)
                
            # Limitar número de peaks por parte del cuerpo
            if len(final_peaks) >= 3:
                break
        
        return final_peaks

    def visualize_keypoints(self, frame: np.ndarray, keypoints: Optional[np.ndarray], draw_skeleton: bool = True) -> np.ndarray:
        """
        Visualiza los keypoints en el frame
        
        Args:
            frame: Frame original
            keypoints: Lista de keypoints detectados
            draw_skeleton: Si dibujar el esqueleto
            
        Returns:
            frame: Frame con keypoints visualizados
        """
        if keypoints is None or len(keypoints) == 0:
            return frame
        
        # Colores para diferentes partes del cuerpo (COCO format)
        colors = [
            (255, 0, 0),    # 0: nose - Rojo
            (0, 255, 0),    # 1: left_eye - Verde
            (0, 0, 255),    # 2: right_eye - Azul
            (255, 255, 0),  # 3: left_ear - Amarillo
            (255, 0, 255),  # 4: right_ear - Magenta
            (0, 255, 255),  # 5: left_shoulder - Cian
            (128, 0, 128),  # 6: right_shoulder - Púrpura
            (255, 165, 0),  # 7: left_elbow - Naranja
            (255, 192, 203), # 8: right_elbow - Rosa
            (128, 128, 128), # 9: left_wrist - Gris
            (255, 255, 255), # 10: right_wrist - Blanco
            (0, 0, 0),      # 11: left_hip - Negro
            (255, 20, 147), # 12: right_hip - Deep Pink
            (0, 191, 255),  # 13: left_knee - Deep Sky Blue
            (34, 139, 34),  # 14: right_knee - Forest Green
            (255, 140, 0),  # 15: left_ankle - Dark Orange
            (220, 20, 60),  # 16: right_ankle - Crimson
        ]
        
        # Crear una copia del frame para no modificar el original
        output_frame = frame.copy()
        
        # Dibujar keypoints
        for keypoint in keypoints:
            x, y, confidence, part_id = keypoint
            
            if confidence > 0.1:  # Solo dibujar si la confianza es alta
                color = colors[part_id % len(colors)]
                
                # Dibujar círculo más grande para mejor visibilidad
                cv2.circle(output_frame, (int(x), int(y)), 2, color, -1)
                cv2.circle(output_frame, (int(x), int(y)), 4, (255, 255, 255), 2)
                
                # Dibujar nombre de la parte (opcional)
                if part_id < len(self.human_pose['keypoints']):
                    part_name = self.human_pose['keypoints'][part_id]
                    cv2.putText(output_frame, f"{part_name}:{confidence:.2f}", 
                               (int(x), int(y-15)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.2, color, 0.7)
        
        # Dibujar esqueleto si se solicita
        if draw_skeleton and len(keypoints) > 0:
            output_frame = self._draw_skeleton(output_frame, keypoints)
        
        return output_frame
    
    def _draw_skeleton(self, frame, keypoints):
        """
        Dibuja las conexiones del esqueleto en el frame.
    
        Args:
            frame: Frame donde se dibujará el esqueleto.
            keypoints: Lista de keypoints detectados con formato [(x, y, conf, part_id), ...].
    
        Returns:
            frame: Frame con las conexiones del esqueleto dibujadas.
        """
        # Crear un diccionario de keypoints válidos (confianza > 0.1)
        keypoint_dict = {}
        for keypoint in keypoints:
            x, y, confidence, part_id = keypoint
            if confidence > 0.1:  # Solo usar keypoints con confianza suficiente
                keypoint_dict[part_id] = (int(x), int(y))
    
        # Conexiones mejoradas del esqueleto
        skeleton = [
            # Pierna izquierda
            (15, 13),  # left_ankle -> left_knee
            (13, 11),  # left_knee -> left_hip
            # Pierna derecha
            (16, 14),  # right_ankle -> right_knee
            (14, 12),  # right_knee -> right_hip
            # Caderas
            (11, 12),  # left_hip -> right_hip
            # Brazo izquierdo
            (5, 7),    # left_shoulder -> left_elbow
            (7, 9),    # left_elbow -> left_wrist
            # Brazo derecho
            (6, 8),    # right_shoulder -> right_elbow
            (8, 10),   # right_elbow -> right_wrist
            # Cabeza
            (0, 1),    # nose -> left_eye
            (0, 2),    # nose -> right_eye
            (1, 3),    # left_eye -> left_ear
            (2, 4),    # right_eye -> right_ear
            # Cuello y torso
            (17, 0),   # neck -> nose
            (17, 5),   # neck -> left_shoulder
            (17, 6),   # neck -> right_shoulder
            (17, 11),  # neck -> left_hip
            (17, 12),  # neck -> right_hip
        ]
    
        # Dibujar las conexiones
        for part_a, part_b in skeleton:
            if part_a in keypoint_dict and part_b in keypoint_dict:
                point_a = keypoint_dict[part_a]
                point_b = keypoint_dict[part_b]
                cv2.line(frame, point_a, point_b, (0, 255, 0), 2)  # Color verde, grosor 2
    
        return frame

    def process_frames(self, frames: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """
        Procesa múltiples frames usando batch processing para mayor eficiencia
        
        Args:
            frames: Lista de frames a procesar
            
        Returns:
            List[Optional[np.ndarray]]: Lista de keypoints para cada frame
        """
        if not frames:
            return []
        
        with torch.no_grad():
            # Preprocesar todos los frames y crear un batch
            batch_tensors = []
            original_shapes = []
            
            for frame in frames:
                # Redimensionar imagen
                resized = cv2.resize(frame, (self.WIDTH, self.HEIGHT))
                
                # Convertir BGR a RGB
                rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                
                # Convertir a tensor
                tensor = torch.from_numpy(rgb_image).float().to(self.device)
                tensor = tensor.permute(2, 0, 1)  # HWC a CHW
                
                # Normalizar
                tensor = tensor / 255.0
                tensor = (tensor - self.mean.view(-1, 1, 1)) / self.std.view(-1, 1, 1)
                
                batch_tensors.append(tensor)
                original_shapes.append(frame.shape[:2])
            
            # Crear batch tensor (batch_size, channels, height, width)
            batch = torch.stack(batch_tensors, dim=0)
            
            print(f"Procesando batch de tamaño: {batch.shape[0]} frames")
            
            # Realizar inferencia en el batch completo
            cmap, paf = self.model(batch)
            
            # Postprocesar resultados para cada frame del batch
            results = []
            for i in range(batch.shape[0]):
                # Extraer mapas para el frame i
                frame_cmap = cmap[i:i+1]  # Mantener dimensión batch
                frame_paf = paf[i:i+1]    # Mantener dimensión batch
                
                # Postprocesar
                keypoints = self.postprocess_results(frame_cmap, frame_paf, original_shapes[i])
                results.append(keypoints)
            
            return results

    def topology(self) -> Tuple[List[str], List[Tuple[int, int]]]:
        """Devuelve los nombres de keypoints y conexiones del esqueleto"""
        keypoint_names = self.human_pose['keypoints']
        skeleton_connections = [(int(conn[0]), int(conn[1])) for conn in self.human_pose['skeleton']]
        return keypoint_names, skeleton_connections
    

'''
import torch
import torch2trt
from torch2trt import TRTModule
import trt_pose.coco
import trt_pose.models
import json
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

class TRTPoseProcessor:
    def __init__(self, model_path, topology_path, use_tensorrt=True):
        """
        Inicializa el procesador de pose estimation
        
        Args:
            model_path: Ruta al modelo (.pth)
            topology_path: Ruta al archivo de topología JSON
            use_tensorrt: Si usar TensorRT (True) o PyTorch normal (False)
        """
        self.use_tensorrt = use_tensorrt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Cargar topología
        with open(topology_path, 'r') as f:
            self.human_pose = json.load(f)
        
        self.topology = trt_pose.coco.coco_category_to_topology(self.human_pose)
        self.num_parts = len(self.human_pose['keypoints'])
        self.num_links = len(self.human_pose['skeleton'])
        
        # Configurar transformaciones
        self.WIDTH = 224
        self.HEIGHT = 224
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).to(self.device)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).to(self.device)
        
        # Cargar modelo
        self._load_model(model_path)
        
        print(f"Modelo cargado exitosamente en: {self.device}")
        print(f"Usando TensorRT: {self.use_tensorrt}")
        
    def _load_model(self, model_path):
        """Carga el modelo según el tipo especificado"""
        if self.use_tensorrt:
            try:
                # Intentar cargar como modelo TensorRT
                self.model = TRTModule()
                self.model.load_state_dict(torch.load(model_path))
                print(f"Modelo TensorRT cargado desde: {model_path}")
            except Exception as e:
                print(f"Error cargando modelo TensorRT: {e}")
                print("Intentando cargar como modelo PyTorch normal...")
                self.use_tensorrt = False
                self._load_pytorch_model(model_path)
        else:
            self._load_pytorch_model(model_path)
    
    def _load_pytorch_model(self, model_path):
        """Carga el modelo PyTorch normal"""
        # Crear modelo
        self.model = trt_pose.models.resnet18_baseline_att(
            self.num_parts, 2 * self.num_links
        ).to(self.device)
        
        # Cargar pesos
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"Modelo PyTorch cargado desde: {model_path}")
    
    def preprocess_image(self, image):
        """Preprocesa la imagen para el modelo"""
        # Redimensionar imagen
        image = cv2.resize(image, (self.WIDTH, self.HEIGHT))
        
        # Convertir BGR a RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convertir a tensor
        image = torch.from_numpy(image).float().to(self.device)
        image = image.permute(2, 0, 1)  # HWC a CHW
        
        # Normalizar
        image = image / 255.0
        image = (image - self.mean.view(-1, 1, 1)) / self.std.view(-1, 1, 1)
        
        # Añadir dimensión batch
        image = image.unsqueeze(0)
        
        return image
    
    def process_frame(self, frame):
        """
        Procesa un frame y retorna los keypoints detectados
        
        Args:
            frame: Frame de imagen (numpy array)
            
        Returns:
            keypoints: Lista de keypoints detectados
        """
        with torch.no_grad():
            # Preprocesar imagen
            input_tensor = self.preprocess_image(frame)
            
            # Realizar inferencia
            cmap, paf = self.model(input_tensor)
            
            # Postprocesar resultados
            keypoints = self.postprocess_results(cmap, paf, frame.shape[:2])
            
            return keypoints
    
    def postprocess_results(self, cmap, paf, original_shape):
        """
        Postprocesa los resultados del modelo para obtener keypoints
        
        Args:
            cmap: Confidence maps
            paf: Part Affinity Fields
            original_shape: Forma original de la imagen (height, width)
            
        Returns:
            keypoints: Lista de keypoints detectados
        """
        # Redimensionar mapas a tamaño original
        height, width = original_shape
        scale_x = width / self.WIDTH
        scale_y = height / self.HEIGHT
        
        # Convertir a numpy
        cmap = cmap.squeeze().cpu().numpy()
        paf = paf.squeeze().cpu().numpy()
        
        # Encontrar peaks en confidence maps
        keypoints = []
        
        # Umbral para detección de keypoints
        threshold = 0.1
        
        for i in range(self.num_parts):
            confidence_map = cmap[i]
            
            # Encontrar máximos locales
            peaks = self._find_peaks(confidence_map, threshold)
            
            # Escalar coordenadas al tamaño original
            scaled_peaks = []
            for peak in peaks:
                x, y, confidence = peak
                x = int(x * scale_x)
                y = int(y * scale_y)
                scaled_peaks.append((x, y, confidence, i))
            
            keypoints.extend(scaled_peaks)
        
        return keypoints
    
    def _find_peaks(self, confidence_map, threshold):
        """Encuentra peaks en el mapa de confianza"""
        peaks = []
        
        # Aplicar filtro de máximos locales
        from scipy.ndimage import maximum_filter
        
        # Encontrar máximos locales
        local_maxima = maximum_filter(confidence_map, size=3) == confidence_map
        
        # Aplicar umbral
        above_threshold = confidence_map > threshold
        
        # Combinar condiciones
        peak_mask = local_maxima & above_threshold
        
        # Obtener coordenadas de peaks
        y_coords, x_coords = np.where(peak_mask)
        
        for x, y in zip(x_coords, y_coords):
            confidence = confidence_map[y, x]
            peaks.append((x, y, confidence))
        
        return peaks
    
    def visualize_keypoints(self, frame, keypoints, draw_skeleton=True):
        """
        Visualiza los keypoints en el frame
        
        Args:
            frame: Frame original
            keypoints: Lista de keypoints detectados
            draw_skeleton: Si dibujar el esqueleto
            
        Returns:
            frame: Frame con keypoints visualizados
        """
        if keypoints is None or len(keypoints) == 0:
            return frame
        
        # Colores para diferentes partes del cuerpo
        colors = [
            (255, 0, 0),    # Rojo
            (0, 255, 0),    # Verde
            (0, 0, 255),    # Azul
            (255, 255, 0),  # Amarillo
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cian
            (128, 0, 128),  # Púrpura
            (255, 165, 0),  # Naranja
            (255, 192, 203), # Rosa
            (128, 128, 128), # Gris
            (255, 255, 255), # Blanco
            (0, 0, 0),      # Negro
            (255, 20, 147), # Deep Pink
            (0, 191, 255),  # Deep Sky Blue
            (34, 139, 34),  # Forest Green
            (255, 140, 0),  # Dark Orange
            (220, 20, 60),  # Crimson
        ]
        
        # Dibujar keypoints
        for keypoint in keypoints:
            x, y, confidence, part_id = keypoint
            
            if confidence > 0.1:  # Solo dibujar si la confianza es alta
                color = colors[part_id % len(colors)]
                cv2.circle(frame, (int(x), int(y)), 3, color, -1)
                
                # Dibujar nombre de la parte (opcional)
                if part_id < len(self.human_pose['keypoints']):
                    part_name = self.human_pose['keypoints'][part_id]
                    cv2.putText(frame, part_name, (int(x), int(y-10)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Dibujar esqueleto si se solicita
        if draw_skeleton and len(keypoints) > 0:
            self._draw_skeleton(frame, keypoints)
        
        return frame
    
    def _draw_skeleton(self, frame, keypoints):
        """Dibuja las conexiones del esqueleto"""
        # Crear diccionario de keypoints por parte
        keypoint_dict = {}
        for keypoint in keypoints:
            x, y, confidence, part_id = keypoint
            if confidence > 0.1:
                keypoint_dict[part_id] = (int(x), int(y))
        
        # Dibujar conexiones según el esqueleto definido
        for connection in self.human_pose['skeleton']:
            part_a, part_b = connection
            if part_a in keypoint_dict and part_b in keypoint_dict:
                point_a = keypoint_dict[part_a]
                point_b = keypoint_dict[part_b]
                cv2.line(frame, point_a, point_b, (0, 255, 0), 2)
        
        return frame

'''
