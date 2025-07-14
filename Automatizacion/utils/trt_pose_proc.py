import sys
import os
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import PIL.Image
from torch2trt import TRTModule

class TRTPoseProcessor:
    def __init__(self, model_path=None, topology_path=None):
        """
        Inicializa el procesador TensorRT Pose.
        
        Args:
            model_path (str): Ruta al modelo TensorRT (.pth o .engine)
            topology_path (str): Ruta al archivo de topología JSON (opcional)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Configurar rutas por defecto si no se proporcionan
        if model_path is None:
            # Buscar modelo por defecto en carpeta models/
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            model_path = os.path.join(project_root, 'models', 'trt_pose_resnet18.pth')
        
        if topology_path is None:
            # Buscar topología por defecto
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            topology_path = os.path.join(project_root, 'models', 'topology.json')
        
        # Verificar que el modelo existe
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo TensorRT no encontrado en: {model_path}")
        
        print(f"Cargando modelo TensorRT desde: {model_path}")
        
        # Cargar el modelo TensorRT
        self.model = TRTModule()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # Configurar transformaciones de imagen
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Dimensiones de entrada del modelo (ajustar según el modelo)
        self.input_width = 224
        self.input_height = 224
        
        # Topología de pose (COCO format por defecto)
        self.topology = self._load_topology(topology_path) if topology_path else self._get_default_topology()
        
    def _load_topology(self, topology_path):
        """
        Carga la topología desde un archivo JSON.
        """
        import json
        with open(topology_path, 'r') as f:
            topology = json.load(f)
        return topology
    
    def _get_default_topology(self):
        """
        Devuelve la topología COCO por defecto.
        """
        return {
            'keypoints': [
                'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
            ],
            'skeleton': [
                [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
                [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
                [2, 4], [3, 5], [4, 6], [5, 7]
            ]
        }
    
    def _preprocess_frame(self, frame):
        """
        Preprocesa el frame para la inferencia.
        """
        # Redimensionar frame
        frame_resized = cv2.resize(frame, (self.input_width, self.input_height))
        
        # Convertir BGR a RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # Convertir a PIL Image
        pil_image = PIL.Image.fromarray(frame_rgb)
        
        # Aplicar transformaciones
        tensor = self.transform(pil_image).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def _postprocess_output(self, output, original_shape):
        """
        Postprocesa la salida del modelo para obtener keypoints.
        """
        # Extraer mapas de calor y vectores de afinidad
        if isinstance(output, (tuple, list)):
            heatmaps = output[0]
            pafs = output[1] if len(output) > 1 else None
        else:
            heatmaps = output
            pafs = None
        
        # Procesar mapas de calor para obtener keypoints
        keypoints = self._extract_keypoints(heatmaps, original_shape)
        
        return keypoints
    
    def _extract_keypoints(self, heatmaps, original_shape):
        """
        Extrae keypoints de los mapas de calor.
        """
        keypoints = []
        
        # Obtener dimensiones originales
        orig_h, orig_w = original_shape[:2]
        
        # Procesar cada mapa de calor
        for i in range(heatmaps.shape[1]):  # Para cada keypoint
            heatmap = heatmaps[0, i].cpu().numpy()
            
            # Encontrar el punto máximo en el mapa de calor
            max_val = np.max(heatmap)
            if max_val > 0.1:  # Umbral de confianza
                max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                
                # Escalar coordenadas al tamaño original
                y = max_idx[0] * orig_h / heatmap.shape[0]
                x = max_idx[1] * orig_w / heatmap.shape[1]
                
                keypoints.append([x, y, max_val])
            else:
                keypoints.append([0, 0, 0])  # No detectado
        
        return np.array(keypoints)
    
    def process_frame(self, frame):
        """
        Procesa un frame y devuelve los keypoints detectados.
        
        Args:
            frame (numpy.ndarray): Frame de entrada en formato BGR
            
        Returns:
            numpy.ndarray: Array de keypoints [x, y, confidence] o None si no se detectan
        """
        try:
            # Guardar forma original
            original_shape = frame.shape
            
            # Preprocesar frame
            input_tensor = self._preprocess_frame(frame)
            
            # Inferencia
            with torch.no_grad():
                output = self.model(input_tensor)
            
            # Postprocesar salida
            keypoints = self._postprocess_output(output, original_shape)
            
            # Devolver keypoints si se detectaron personas
            if keypoints is not None and len(keypoints) > 0:
                return keypoints
            else:
                return None
                
        except Exception as e:
            print(f"Error procesando frame con TensorRT Pose: {e}")
            return None
    
    def _process_frame(self, frame):
        """
        Método alternativo para compatibilidad con la clase OpenPose.
        """
        return self.process_frame(frame)
    
    def get_keypoint_names(self):
        """
        Devuelve los nombres de los keypoints.
        """
        return self.topology['keypoints']
    
    def get_skeleton_connections(self):
        """
        Devuelve las conexiones del esqueleto.
        """
        return self.topology['skeleton']
    
    def visualize_keypoints(self, frame, keypoints, draw_skeleton=True):
        """
        Visualiza los keypoints en el frame.
        
        Args:
            frame (numpy.ndarray): Frame original
            keypoints (numpy.ndarray): Array de keypoints
            draw_skeleton (bool): Si dibujar las conexiones del esqueleto
            
        Returns:
            numpy.ndarray: Frame con keypoints dibujados
        """
        if keypoints is None:
            return frame
        
        result_frame = frame.copy()
        
        # Dibujar keypoints
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > 0.1:  # Solo dibujar si hay confianza
                cv2.circle(result_frame, (int(x), int(y)), 3, (0, 255, 0), -1)
                cv2.putText(result_frame, str(i), (int(x), int(y-10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Dibujar esqueleto
        if draw_skeleton:
            for connection in self.topology['skeleton']:
                idx1, idx2 = connection[0] - 1, connection[1] - 1  # Convertir a índices base 0
                if (0 <= idx1 < len(keypoints) and 0 <= idx2 < len(keypoints) and 
                    keypoints[idx1][2] > 0.1 and keypoints[idx2][2] > 0.1):
                    
                    pt1 = (int(keypoints[idx1][0]), int(keypoints[idx1][1]))
                    pt2 = (int(keypoints[idx2][0]), int(keypoints[idx2][1]))
                    cv2.line(result_frame, pt1, pt2, (0, 0, 255), 2)
        
        return result_frame
    
    @staticmethod
    def download_pretrained_model(model_name='resnet18', target_dir='models'):
        """
        Descarga un modelo preentrenado de TensorRT Pose.
        
        Args:
            model_name (str): Nombre del modelo ('resnet18' o 'densenet121')
            target_dir (str): Directorio donde guardar el modelo
        """
        import urllib.request
        import os
        
        # URLs de modelos preentrenados
        model_urls = {
            'resnet18': 'https://github.com/NVIDIA-AI-IOT/trt_pose/releases/download/v0.0.1/resnet18_baseline_att_224x224_A_epoch_249.pth',
            'densenet121': 'https://github.com/NVIDIA-AI-IOT/trt_pose/releases/download/v0.0.1/densenet121_baseline_att_256x256_B_epoch_160.pth'
        }
        
        topology_url = 'https://raw.githubusercontent.com/NVIDIA-AI-IOT/trt_pose/master/tasks/human_pose/human_pose.json'
        
        if model_name not in model_urls:
            raise ValueError(f"Modelo {model_name} no disponible. Opciones: {list(model_urls.keys())}")
        
        # Crear directorio si no existe
        os.makedirs(target_dir, exist_ok=True)
        
        # Descargar modelo
        model_filename = f"trt_pose_{model_name}.pth"
        model_path = os.path.join(target_dir, model_filename)
        
        if not os.path.exists(model_path):
            print(f"Descargando modelo {model_name}...")
            urllib.request.urlretrieve(model_urls[model_name], model_path)
            print(f"Modelo descargado en: {model_path}")
        else:
            print(f"Modelo ya existe en: {model_path}")
        
        # Descargar topología
        topology_path = os.path.join(target_dir, 'topology.json')
        if not os.path.exists(topology_path):
            print("Descargando topología...")
            urllib.request.urlretrieve(topology_url, topology_path)
            print(f"Topología descargada en: {topology_path}")
        else:
            print(f"Topología ya existe en: {topology_path}")
        
        return model_path, topology_path
