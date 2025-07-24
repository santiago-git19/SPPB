import os

class Config:
    def __init__(self):
        # === CONFIGURACIÓN DE RUTAS BASE ===
        # Ruta base del proyecto (directorio donde se encuentra este archivo)
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


        # === CONFIGURACIÓN TRT POSE ===
        # Rutas de modelos para TRT Pose y clasificación
        self.trt_pose_model = os.path.join(self.base_dir, "models", "resnet18_baseline_att_224x224_A_epoch_249.pth")
        self.pose_topology = os.path.join(self.base_dir, "models", "human_pose.json")
        self.pose_classifier_model = os.path.join(self.base_dir, "models", "pose_classification", "st-gcn_3dbp_nvidia.engine")
        
        # === CONFIGURACIÓN DE CÁMARAS ===
        # Cámaras: pueden ser índices (0, 1) o rutas de video
        self.frontal_camera = 0  # Cámara frontal para verificar alineación
        self.lateral_camera = 1  # Cámara lateral para calcular distancia
        
        # === CONFIGURACIÓN DE CALIBRACIÓN ===
        # Factor de conversión de píxeles a metros (debe calibrarse según el setup)
        self.pixel_to_meter_ratio = 0.01  # metros por pixel
        
        # === CONFIGURACIÓN GENERAL ===
        # Configuración heredada del sistema anterior
        self.model_folder = "/home/Documentos/Trabajo/openpose/models/"  # Mantener compatibilidad
        self.output_base = "/ruta/salida"
        self.fps = 5.0
        self.duration = 30
        
        # === CONFIGURACIÓN TRT POSE ESPECÍFICA ===
        # Optimizaciones para Jetson Nano
        self.sequence_length = 15  # Frames para clasificación (reducido para Jetson)
        self.confidence_threshold = 0.3  # Umbral de confianza para keypoints
        self.keypoint_format = 'coco'  # Formato de keypoints
        
        # === CONFIGURACIÓN DE CÁLCULO DE DISTANCIA ===
        # Parámetros para el cálculo de distancia en gait_speed
        self.distance_calculation = {
            'use_hip_keypoints': True,  # Usar keypoints de cadera para calcular distancia
            'max_frames_without_detection': 30,  # Máximo frames sin detección antes de pausar
            'distance_smoothing': True,  # Aplicar suavizado a los cálculos de distancia
            'min_movement_threshold': 0.001  # Mínimo movimiento en metros para considerar
        }
        