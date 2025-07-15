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