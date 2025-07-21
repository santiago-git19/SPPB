# Configuración de Ejemplo para SPPB con TRT Pose

"""
Este archivo contiene un ejemplo de configuración completa para el sistema SPPB
utilizando TRT Pose en lugar de OpenPose, con soporte para dos cámaras.
"""

from utils.config import Config

# === CONFIGURACIÓN PARA JETSON NANO ===
class JetsonNanoConfig(Config):
    def __init__(self):
        super().__init__()
        
        # === RUTAS DE MODELOS (AJUSTAR SEGÚN TU SISTEMA) ===
        self.trt_pose_model = "models/resnet18_baseline_att_224x224_A_epoch_249.pth"
        self.pose_topology = "models/human_pose.json"  
        self.pose_classifier_model = "models/pose_classification/st-gcn_3dbp_nvidia.engine"
        
        # === CONFIGURACIÓN DE CÁMARAS ===
        # Para cámaras USB conectadas
        self.frontal_camera = 0    # /dev/video0
        self.lateral_camera = 1    # /dev/video1
        
        # Para videos pregrabados (comentar las líneas de arriba)
        # self.frontal_camera = "videos/frontal_camera.mp4"
        # self.lateral_camera = "videos/lateral_camera.mp4"
        
        # === CALIBRACIÓN (DEBE AJUSTARSE SEGÚN EL SETUP) ===
        # Ejemplo: Si 1 metro = 100 píxeles en la imagen
        self.pixel_to_meter_ratio = 0.01  # metros por pixel
        
        # === OPTIMIZACIONES PARA JETSON NANO ===
        self.sequence_length = 10           # Reducido para mejor rendimiento
        self.confidence_threshold = 0.25    # Ajustado para Jetson
        
        # === CONFIGURACIÓN ESPECÍFICA DE CÁLCULO DE DISTANCIA ===
        self.distance_calculation = {
            'use_hip_keypoints': True,
            'max_frames_without_detection': 20,  # Reducido para Jetson
            'distance_smoothing': True,
            'min_movement_threshold': 0.005     # 5mm mínimo
        }

# === CONFIGURACIÓN PARA PC POTENTE ===
class DesktopConfig(Config):
    def __init__(self):
        super().__init__()
        
        # === RUTAS DE MODELOS ===
        self.trt_pose_model = "/home/user/models/resnet18_baseline_att_224x224_A_epoch_249.pth"
        self.pose_topology = "/home/user/models/human_pose.json"
        self.pose_classifier_model = "/home/user/models/st-gcn_3dbp_nvidia.engine"
        
        # === CONFIGURACIÓN DE CÁMARAS ===
        self.frontal_camera = 0
        self.lateral_camera = 1
        
        # === CALIBRACIÓN ===
        self.pixel_to_meter_ratio = 0.008   # Calibración para setup específico
        
        # === CONFIGURACIÓN OPTIMIZADA PARA PC ===
        self.sequence_length = 30           # Más frames para mejor precisión
        self.confidence_threshold = 0.3
        
        self.distance_calculation = {
            'use_hip_keypoints': True,
            'max_frames_without_detection': 30,
            'distance_smoothing': True,
            'min_movement_threshold': 0.001  # 1mm mínimo
        }

# === CONFIGURACIÓN PARA DESARROLLO/TESTING ===
class DevelopmentConfig(Config):
    def __init__(self):
        super().__init__()
        
        # === RUTAS DE MODELOS (DESARROLLO) ===
        self.trt_pose_model = "models/resnet18_baseline_att_224x224_A_epoch_249.pth"
        self.pose_topology = "models/human_pose.json"
        self.pose_classifier_model = "models/pose_classification/st-gcn_3dbp_nvidia.engine"
        
        # === CONFIGURACIÓN DE CÁMARAS (VIDEOS DE PRUEBA) ===
        self.frontal_camera = "test_videos/WIN_20250702_12_09_08_Pro.mp4"
        self.lateral_camera = "test_videos/lateral_view.mp4"
        
        # === CALIBRACIÓN (PARA VIDEOS DE PRUEBA) ===
        self.pixel_to_meter_ratio = 0.012   # Ajustar según videos
        
        # === CONFIGURACIÓN DE DESARROLLO ===
        self.sequence_length = 15
        self.confidence_threshold = 0.2     # Más permisivo para testing
        
        self.distance_calculation = {
            'use_hip_keypoints': True,
            'max_frames_without_detection': 60,  # Más tolerante
            'distance_smoothing': True,
            'min_movement_threshold': 0.002
        }

# === FUNCIÓN PARA SELECCIONAR CONFIGURACIÓN ===
def get_config(config_type="development"):
    """
    Devuelve la configuración apropiada según el tipo de sistema
    
    Args:
        config_type: Tipo de configuración ("jetson", "desktop", "development")
        
    Returns:
        Config: Instancia de configuración apropiada
    """
    configs = {
        "jetson": JetsonNanoConfig,
        "desktop": DesktopConfig, 
        "development": DevelopmentConfig
    }
    
    if config_type not in configs:
        print(f"⚠️ Tipo de configuración '{config_type}' no reconocido")
        print(f"📋 Tipos disponibles: {list(configs.keys())}")
        print("🔧 Usando configuración de desarrollo por defecto")
        config_type = "development"
    
    return configs[config_type]()

# === EJEMPLO DE USO ===
if __name__ == "__main__":
    # Seleccionar configuración según el entorno
    config = get_config("jetson")  # o "desktop" o "development"
    
    print("🔧 Configuración cargada:")
    print(f"   📁 Modelo TRT Pose: {config.trt_pose_model}")
    print(f"   📁 Clasificador: {config.pose_classifier_model}")
    print(f"   📹 Cámara frontal: {config.frontal_camera}")
    print(f"   📹 Cámara lateral: {config.lateral_camera}")
    print(f"   📏 Ratio píxel/metro: {config.pixel_to_meter_ratio}")
