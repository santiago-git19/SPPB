from phases.balance import BalancePhase
from phases.gait_speed import GaitSpeedPhase
from phases.chair_rise import ChairRisePhase
from utils.trt_pose_proc import TRTPoseProcessor
from utils.trt_pose_classifier import TRTPoseClassifier, create_pose_classifier
from results import SPPBResult
import cv2
import os

class SPPBTest:
    def __init__(self, config):
        self.config = config
        
        # === INICIALIZACIÓN DE PROCESADORES TRT POSE ===
        # Inicializar procesador de poses una sola vez para reutilizar en todas las fases
        self._initialize_pose_processors()
        
        # === INICIALIZACIÓN DE FASES ===
        # Pasar los procesadores a cada fase en lugar de openpose
        self.balance = BalancePhase(self.pose_processor, self.pose_classifier, config)
        self.gait = GaitSpeedPhase(self.pose_processor, self.pose_classifier, config)
        self.chair = ChairRisePhase(self.pose_processor, self.pose_classifier, config)
    
    def _initialize_pose_processors(self):
        """
        Inicializa los procesadores de TRT Pose una sola vez para todas las fases
        """
        try:
            print("🔧 Inicializando procesadores TRT Pose...")
            
            # Verificar que los archivos de modelos existen
            self._verify_model_files()
            
            # Inicializar procesador de keypoints
            self.pose_processor = TRTPoseProcessor(
                model_path=self.config.trt_pose_model,
                topology_path=self.config.pose_topology
            )
            
            # Inicializar clasificador de poses
            self.pose_classifier = create_pose_classifier(
                model_path=self.config.pose_classifier_model,
                keypoint_format=self.config.keypoint_format,
                sequence_length=self.config.sequence_length,
                confidence_threshold=self.config.confidence_threshold
            )
            
            print("✅ Procesadores TRT Pose inicializados correctamente")
            print(f"   📁 Modelo TRT Pose: {os.path.basename(self.config.trt_pose_model)}")
            print(f"   📁 Topología: {os.path.basename(self.config.pose_topology)}")
            print(f"   📁 Clasificador: {os.path.basename(self.config.pose_classifier_model)}")
            
        except Exception as e:
            print(f"❌ Error inicializando procesadores TRT Pose: {e}")
            print("💡 Verificar rutas de modelos en la configuración")
            raise
    
    def _verify_model_files(self):
        """
        Verifica que todos los archivos de modelos existan
        """
        required_files = [
            (self.config.trt_pose_model, "Modelo TRT Pose"),
            (self.config.pose_topology, "Topología de poses"),
            (self.config.pose_classifier_model, "Modelo clasificador")
        ]
        
        missing_files = []
        for file_path, description in required_files:
            if not os.path.exists(file_path):
                missing_files.append(f"  - {description}: {file_path}")
        
        if missing_files:
            error_msg = "❌ Archivos de modelos no encontrados:\n" + "\n".join(missing_files)
            error_msg += "\n💡 Verificar las rutas en config.py"
            raise FileNotFoundError(error_msg)

    def run(self, video_path=None, camera_id=None):
        """
        Ejecuta el test SPPB completo con soporte para dos cámaras
        
        Args:
            video_path: Ruta del video frontal (opcional si se usan cámaras)
            camera_id: ID de la cámara frontal (opcional si se usa video)
        """
        # === CONFIGURACIÓN DE CÁMARAS ===
        cap_frontal, cap_lateral = self._setup_cameras(video_path, camera_id)
        
        try:
            print("🎥 Iniciando test SPPB con TRT Pose...")
            print(f"   📹 Cámara frontal: {'Video' if video_path else f'Cámara {camera_id}'}")
            print(f"   📹 Cámara lateral: {'Configurada' if cap_lateral else 'No disponible'}")
            
            # === EJECUCIÓN DE LAS FASES ===
            # Cada fase recibe ambas cámaras
            balance_result = self.balance.run(cap_frontal, cap_lateral, self.config.duration)
            gait_result = self.gait.run(cap_frontal, cap_lateral)
            chair_result = self.chair.run(cap_frontal, cap_lateral)
            
            print("✅ Test SPPB completado")
            
            return SPPBResult(balance_result, gait_result, chair_result)
            
        except Exception as e:
            print(f"❌ Error durante el test SPPB: {e}")
            raise
        finally:
            # === LIMPIEZA DE RECURSOS ===
            self._cleanup_cameras(cap_frontal, cap_lateral)
    
    def _setup_cameras(self, video_path, camera_id):
        """
        Configura las cámaras frontal y lateral
        
        Args:
            video_path: Ruta del video frontal
            camera_id: ID de la cámara frontal
            
        Returns:
            tuple: (cap_frontal, cap_lateral)
        """
        # Configurar cámara frontal
        if video_path:
            cap_frontal = cv2.VideoCapture(video_path)
            if not cap_frontal.isOpened():
                raise FileNotFoundError(f"No se pudo abrir el video: {video_path}")
        elif camera_id is not None:
            cap_frontal = cv2.VideoCapture(camera_id)
            if not cap_frontal.isOpened():
                raise RuntimeError(f"No se pudo abrir la cámara: {camera_id}")
        else:
            # Usar configuración por defecto
            cap_frontal = cv2.VideoCapture(self.config.frontal_camera)
            if not cap_frontal.isOpened():
                raise RuntimeError(f"No se pudo abrir la cámara frontal: {self.config.frontal_camera}")
        
        # Configurar cámara lateral
        cap_lateral = None
        try:
            cap_lateral = cv2.VideoCapture(self.config.lateral_camera)
            if not cap_lateral.isOpened():
                print(f"⚠️ No se pudo abrir la cámara lateral: {self.config.lateral_camera}")
                cap_lateral = None
        except Exception as e:
            print(f"⚠️ Error configurando cámara lateral: {e}")
            cap_lateral = None
        
        return cap_frontal, cap_lateral
    
    def _cleanup_cameras(self, cap_frontal, cap_lateral):
        """
        Libera los recursos de las cámaras
        """
        if cap_frontal:
            cap_frontal.release()
        if cap_lateral:
            cap_lateral.release()
        cv2.destroyAllWindows()
