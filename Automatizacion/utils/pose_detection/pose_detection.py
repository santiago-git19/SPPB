from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import numpy as np

class PoseDetection(ABC):
    """
    Interfaz base para la detección de poses humanas.
    Define los métodos que deben implementar las clases derivadas.
    """

    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """
        Carga el modelo necesario para la detección de poses.
        """
        pass

    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Procesa un frame individual y devuelve los keypoints detectados.
        """
        pass

    @abstractmethod
    def process_frames(self, frames: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """
        Procesa múltiples frames y devuelve los keypoints detectados para cada uno.
        """
        pass

    @abstractmethod
    def visualize_keypoints(self, frame: np.ndarray, keypoints: Optional[np.ndarray], draw_skeleton: bool = True) -> np.ndarray:
        """
        Dibuja los keypoints y las conexiones en un frame.
        """
        pass

    @abstractmethod
    def topology(self) -> Tuple[List[str], List[Tuple[int, int]]]:
        """
        Devuelve la topología del modelo (nombres de los keypoints y conexiones del esqueleto).
        """
        pass