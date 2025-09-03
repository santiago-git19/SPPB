"""
Gait3DTracker
=============

Clase para reconstruir la trayectoria 3D del centro de cadera y
acumular la distancia recorrida a partir de:
- Keypoints 2D (salida de TRTPoseProcessor.process_frame)
- Frame de profundidad de Orbbec
- Utilidades de DualOrbbecCapture (para depth y proyección 3D)

Uso rápido:

from utils.dual_orbbec_capture import DualOrbbecCapture
from utils.gait_3d_tracker import Gait3DTracker

dual = DualOrbbecCapture(enable_depth=True)
tracker = Gait3DTracker(dual_cam=dual, camera_side='left')

# En cada frame:
keypoints = trt_pose.process_frame(color_frame)
point_3d = tracker.update(keypoints, depth_frame)
print(tracker.total_distance_m)

La trayectoria (en metros) queda en tracker.trajectory_m (lista de np.ndarray de shape (3,)).
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Any
import numpy as np

try:
    from .dual_orbbec_capture import DualOrbbecCapture
except Exception:  # pragma: no cover
    DualOrbbecCapture = None  # type: ignore


class Gait3DTracker:
    """
    Reconstruye la trayectoria 3D del punto medio de la cadera y acumula distancia.

    Parámetros:
    - dual_cam: instancia de DualOrbbecCapture con depth habilitado
    - camera_side: 'left' o 'right' para seleccionar la cámara usada para depth/intrínsecos
    - region: tamaño (en píxeles) del lado de la ventana cuadrada alrededor del hip para promediar depth
    - min_conf: confianza mínima de cada cadera (COCO 11 y 12) para considerar el punto
    - smoothing_alpha: factor de suavizado EMA (0..1), mayor = más suave
    - min_valid_depth_percent: porcentaje mínimo de píxeles válidos en la región para usar su mediana
    - max_jump_m: salto máximo permitido entre frames (en metros) para descartar outliers
    """

    COCO_LEFT_HIP = 11
    COCO_RIGHT_HIP = 12

    def __init__(
        self,
    dual_cam: Any,
        camera_side: str = 'left',
        region: int = 7,
        min_conf: float = 0.3,
        smoothing_alpha: float = 0.2,
        min_valid_depth_percent: float = 25.0,
        max_jump_m: float = 1.0,
    ) -> None:
        if dual_cam is None:
            raise ValueError("dual_cam no puede ser None y debe tener depth habilitado")
        
        # Permitir valor especial "bag_file" para procesamiento de archivos .bag
        if dual_cam != "bag_file" and not hasattr(dual_cam, 'enable_depth'):
            raise ValueError("dual_cam debe ser un objeto DualOrbbecCapture con depth habilitado o 'bag_file'")
        
        if dual_cam != "bag_file" and not dual_cam.enable_depth:
            raise ValueError("dual_cam debe tener depth habilitado")

        self.dual_cam = dual_cam
        self.camera_side = camera_side
        self.region = max(3, int(region) | 1)  # forzar impar >=3
        self.min_conf = float(min_conf)
        self.alpha = float(smoothing_alpha)
        self.min_valid_percent = float(min_valid_depth_percent)
        self.max_jump_m = float(max_jump_m)

        self.reset()

    def reset(self) -> None:
        self.trajectory_m: List[np.ndarray] = []
        self.total_distance_m: float = 0.0
        self._last_filtered: Optional[np.ndarray] = None

    @staticmethod
    def _extract_hip_center_2d(
        keypoints: Optional[List[Tuple[float, float, float, int]]],
        min_conf: float,
    ) -> Optional[Tuple[int, int, float]]:
        """
        Devuelve (x, y, conf) del centro de cadera usando COCO 11 (left_hip) y 12 (right_hip).
        keypoints: lista de tuplas (x, y, conf, part_id)
        """
        if not keypoints:
            return None

        left_hip = None
        right_hip = None
        for x, y, conf, pid in keypoints:
            if conf < min_conf:
                continue
            if pid == Gait3DTracker.COCO_LEFT_HIP:
                left_hip = (float(x), float(y), float(conf))
            elif pid == Gait3DTracker.COCO_RIGHT_HIP:
                right_hip = (float(x), float(y), float(conf))

        if left_hip and right_hip:
            x_c = int(round((left_hip[0] + right_hip[0]) / 2.0))
            y_c = int(round((left_hip[1] + right_hip[1]) / 2.0))
            conf_c = float(min(left_hip[2], right_hip[2]))
            return (x_c, y_c, conf_c)

        return None

    def _region_depth_mm(self, depth_frame: np.ndarray, x: int, y: int) -> Optional[float]:
        half = self.region // 2
        
        # Si estamos procesando un archivo .bag, usar métodos simples
        if self.dual_cam == "bag_file":
            return self._simple_region_depth(depth_frame, x, y)
        
        stats = self.dual_cam.get_depth_statistics_in_region(depth_frame, x - half, y - half, self.region, self.region)
        if stats and stats.get('valid_pixels_percent', 0.0) >= self.min_valid_percent:
            # Usamos la mediana para robustez frente a outliers
            return float(stats.get('median', 0.0)) if stats.get('median', 0.0) > 0 else None
        # Fallback a un único píxel si la región no es suficientemente válida
        return self.dual_cam.get_distance_at_point(depth_frame, x, y)
    
    def _simple_region_depth(self, depth_frame: np.ndarray, x: int, y: int) -> Optional[float]:
        """Método simple para obtener profundidad cuando se usa archivo .bag"""
        try:
            h, w = depth_frame.shape
            half = self.region // 2
            
            # Asegurar que la región esté dentro de la imagen
            x1 = max(0, x - half)
            y1 = max(0, y - half)
            x2 = min(w, x + half + 1)
            y2 = min(h, y + half + 1)
            
            region = depth_frame[y1:y2, x1:x2]
            valid_depths = region[region > 0]
            
            if len(valid_depths) >= len(region.flatten()) * 0.25:  # Al menos 25% de píxeles válidos
                return float(np.median(valid_depths))
            
            # Fallback a píxel central
            if 0 <= x < w and 0 <= y < h and depth_frame[y, x] > 0:
                return float(depth_frame[y, x])
                
            return None
        except Exception:
            return None
    
    def _simple_3d_coordinates(self, depth_frame: np.ndarray, x: int, y: int, z_mm: float) -> Optional[Tuple[float, float, float]]:
        """Método simple para convertir coordenadas 2D + profundidad a 3D cuando se usa archivo .bag"""
        try:
            # Parámetros intrínsecos aproximados para Orbbec Gemini 335Le
            # Estos valores son aproximados y deberían calibrarse para mayor precisión
            fx = 570.3  # Focal length en X
            fy = 570.3  # Focal length en Y
            cx = 320.0  # Centro óptico en X
            cy = 240.0  # Centro óptico en Y
            
            # Convertir de píxeles a coordenadas 3D en mm
            x_3d = (x - cx) * z_mm / fx
            y_3d = (y - cy) * z_mm / fy
            z_3d = z_mm
            
            return (x_3d, y_3d, z_3d)
            
        except Exception:
            return None

    def update(
        self,
        keypoints: Optional[List[Tuple[float, float, float, int]]],
        depth_frame: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """
        Actualiza el tracker con un nuevo frame y acumula distancia.

        Args:
            keypoints: lista (x, y, conf, part_id) de TRT Pose
            depth_frame: frame de profundidad (mm) correspondiente al color

        Returns:
            np.ndarray(3,) con el punto filtrado en metros (X,Y,Z) o None si inválido
        """
        if depth_frame is None or keypoints is None:
            return None

        # 1) Centro de cadera 2D
        hip = self._extract_hip_center_2d(keypoints, self.min_conf)
        if not hip:
            return None
        x, y, _ = hip

        # 2) Profundidad robusta (mm)
        z_mm_region = self._region_depth_mm(depth_frame, x, y)
        if z_mm_region is None:
            return None

        # 3) Coordenadas 3D en mm desde utilidades (basadas en el píxel)
        if self.dual_cam == "bag_file":
            # Para archivos .bag, usar conversión simple
            xyz_mm_pix = self._simple_3d_coordinates(depth_frame, x, y, z_mm_region)
        else:
            xyz_mm_pix = self.dual_cam.get_3d_coordinates(depth_frame, x, y, self.camera_side)
            
        if xyz_mm_pix is None:
            return None

        x_pix, y_pix, z_pix = xyz_mm_pix
        # Si la Z del píxel difiere de la mediana regional, reescala X/Y para usar la Z robusta
        if z_pix and z_pix > 0 and z_mm_region > 0 and abs(z_mm_region - z_pix) > 1e-3:
            scale = float(z_mm_region / z_pix)
            x_rescaled = float(x_pix * scale)
            y_rescaled = float(y_pix * scale)
            z_rescaled = float(z_mm_region)
        else:
            x_rescaled, y_rescaled, z_rescaled = float(x_pix), float(y_pix), float(z_pix)

        # 4) A metros
        point_m = np.array([x_rescaled, y_rescaled, z_rescaled], dtype=np.float32) / 1000.0

        # 5) Suavizado EMA
        if self._last_filtered is None:
            filtered = point_m
        else:
            filtered = self.alpha * point_m + (1.0 - self.alpha) * self._last_filtered

        # 6) Acumular distancia (y rechazo de outliers)
        if len(self.trajectory_m) > 0:
            step = float(np.linalg.norm(filtered - self.trajectory_m[-1]))
            if step <= self.max_jump_m:
                self.total_distance_m += step
            else:
                # Outlier: no sumar, pero podemos aún actualizar último filtrado
                pass

        self.trajectory_m.append(filtered)
        self._last_filtered = filtered
        return filtered

    # Utilidades
    def last_point(self) -> Optional[np.ndarray]:
        return self.trajectory_m[-1] if self.trajectory_m else None

    def stats(self) -> dict:
        return {
            'points': len(self.trajectory_m),
            'total_distance_m': float(self.total_distance_m),
            'last_point': self.last_point().tolist() if self.last_point() is not None else None,
            'region': self.region,
            'alpha': self.alpha,
            'min_conf': self.min_conf,
            'min_valid_percent': self.min_valid_percent,
        }
