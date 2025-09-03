"""
Manual Action Detector - Clasificación de postura (de pie/sentado)
Utiliza keypoints de TRT Pose para determinar si una persona está de pie o sentada
usando datos de cámaras frontal y lateral.
"""

import math
import logging
from typing import List, Tuple, Dict, Optional
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# Mapeo de nombres de keypoints según human_pose.json
# Los keypoints están en orden desde índice 0 hasta 17
KEYPOINT_NAMES = [
    "nose",           # 0
    "left_eye",       # 1
    "right_eye",      # 2
    "left_ear",       # 3
    "right_ear",      # 4
    "left_shoulder",  # 5
    "right_shoulder", # 6
    "left_elbow",     # 7
    "right_elbow",    # 8
    "left_wrist",     # 9
    "right_wrist",    # 10
    "left_hip",       # 11
    "right_hip",      # 12
    "left_knee",      # 13
    "right_knee",     # 14
    "left_ankle",     # 15
    "right_ankle",    # 16
    "neck"            # 17
]

# Índices de keypoints importantes
KEYPOINT_INDICES = {name: idx for idx, name in enumerate(KEYPOINT_NAMES)}

class PostureClassifier:
    """
    Clasificador de postura que determina si una persona está de pie o sentada
    usando keypoints de cámaras frontal y lateral.
    """
    
    def __init__(self, confidence_threshold=0.01):
        """
        Inicializar el clasificador de postura.
        
        Args:
            confidence_threshold (float): Umbral mínimo de confianza para considerar un keypoint válido
        """
        self.confidence_threshold = confidence_threshold
        self.required_keypoints_frontal = [
            'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        self.required_keypoints_lateral = [
            'left_shoulder', 'left_hip', 'left_knee', 'left_ankle',
            'neck'  # Para evaluar la posición de la espalda
        ]
        
        logger.info(f"PostureClassifier inicializado con umbral de confianza: {confidence_threshold}")
    
    def classify_posture(self, keypoints_frontal: List[Tuple], keypoints_lateral: List[Tuple]) -> Dict:
        """
        Clasifica si una persona está de pie o sentada usando keypoints de dos cámaras.
        
        Args:
            keypoints_frontal (List[Tuple]): Lista de keypoints de la cámara frontal [(x, y, confidence, keypoint_idx), ...]
            keypoints_lateral (List[Tuple]): Lista de keypoints de la cámara lateral [(x, y, confidence, keypoint_idx), ...]
        
        Returns:
            Dict: {
                'posture': 'de pie' | 'sentado' | 'indeterminado',
                'confidence': float,
                'frontal_result': dict,
                'lateral_result': dict,
                'details': dict
            }
        """
        logger.info("Iniciando clasificación de postura")
        
        # Debug: mostrar keypoints recibidos
        logger.debug(f"Keypoints frontales recibidos: {len(keypoints_frontal)} puntos")
        for x, y, conf, idx in keypoints_frontal:
            if idx < len(KEYPOINT_NAMES):
                logger.debug(f"  {KEYPOINT_NAMES[idx]} ({idx}): ({x:.1f}, {y:.1f}) conf={conf:.3f}")
        
        logger.debug(f"Keypoints laterales recibidos: {len(keypoints_lateral)} puntos")
        for x, y, conf, idx in keypoints_lateral:
            if idx < len(KEYPOINT_NAMES):
                logger.debug(f"  {KEYPOINT_NAMES[idx]} ({idx}): ({x:.1f}, {y:.1f}) conf={conf:.3f}")
        
        # Convertir keypoints a diccionarios para facilitar el acceso
        frontal_kps = self._keypoints_to_dict(keypoints_frontal)
        lateral_kps = self._keypoints_to_dict(keypoints_lateral)
        
        logger.debug(f"Keypoints frontales válidos: {list(frontal_kps.keys())}")
        logger.debug(f"Keypoints laterales válidos: {list(lateral_kps.keys())}")
        
        # Procesar keypoints de cada cámara
        frontal_result = self._process_frontal_keypoints(frontal_kps)
        lateral_result = self._process_lateral_keypoints(lateral_kps)
        
        # Combinar resultados
        final_result = self._combine_results(frontal_result, lateral_result)
        
        result = {
            'posture': final_result['posture'],
            'confidence': final_result['confidence'],
            'frontal_result': frontal_result,
            'lateral_result': lateral_result,
            'details': final_result['details']
        }
        
        logger.info(f"Clasificación completada: {result['posture']} (confianza: {result['confidence']:.2f})")
        return result
    
    def _keypoints_to_dict(self, keypoints: List[Tuple]) -> Dict[str, Tuple]:
        """
        Convierte lista de keypoints a diccionario para fácil acceso por nombre.
        
        Args:
            keypoints: Lista de tuplas (x, y, confidence, keypoint_idx)
            
        Returns:
            Dict[str, Tuple]: Diccionario {keypoint_name: (x, y, confidence)}
        """
        kp_dict = {}
        for x, y, confidence, idx in keypoints:
            #print(self.confidence_threshold)
            #print(confidence >= self.confidence_threshold)
            if confidence >= self.confidence_threshold and 0 <= idx < len(KEYPOINT_NAMES):
                print("++++++++++++++++++++++++++++++++++++++++++")
                keypoint_name = KEYPOINT_NAMES[idx]
                kp_dict[keypoint_name] = (x, y, confidence)
        
        # Si no tenemos el keypoint "neck" pero sí los hombros, calcularlo
        if 'neck' not in kp_dict and 'left_shoulder' in kp_dict and 'right_shoulder' in kp_dict:
            left_shoulder = kp_dict['left_shoulder']
            right_shoulder = kp_dict['right_shoulder']
            neck_x = (left_shoulder[0] + right_shoulder[0]) / 2
            neck_y = (left_shoulder[1] + right_shoulder[1]) / 2
            neck_conf = min(left_shoulder[2], right_shoulder[2])  # Usar la menor confianza
            kp_dict['neck'] = (neck_x, neck_y, neck_conf)
            logger.debug("Calculado keypoint 'neck' como punto medio entre hombros")
        
        return kp_dict
    
    def _process_frontal_keypoints(self, keypoints: Dict[str, Tuple]) -> Dict:
        """
        Procesa los keypoints de la cámara frontal para evaluar la postura.
        
        Args:
            keypoints: Diccionario de keypoints {name: (x, y, confidence)}
            
        Returns:
            Dict: Resultado del análisis frontal
        """
        result = {
            'posture': 'indeterminado',
            'confidence': 0.0,
            'metrics': {},
            'missing_keypoints': []
        }
        
        try:
            # Verificar keypoints requeridos
            missing = [kp for kp in self.required_keypoints_frontal if kp not in keypoints]
            result['missing_keypoints'] = missing
            
            if missing:
                logger.warning(f"Keypoints faltantes en cámara frontal: {missing}")
                return result
            
            # 1. Evaluar alineación vertical del torso
            left_shoulder = keypoints['left_shoulder']
            right_shoulder = keypoints['right_shoulder']
            left_hip = keypoints['left_hip']
            right_hip = keypoints['right_hip']
            
            shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
            hip_center_y = (left_hip[1] + right_hip[1]) / 2
            torso_height = abs(shoulder_center_y - hip_center_y)
            
            # 2. Evaluar separación entre rodillas
            left_knee = keypoints['left_knee']
            right_knee = keypoints['right_knee']
            knee_separation = abs(left_knee[0] - right_knee[0])
            
            # 3. Evaluar altura de las rodillas respecto a las caderas
            knee_center_y = (left_knee[1] + right_knee[1]) / 2
            knee_hip_diff = knee_center_y - hip_center_y  # Positivo si rodillas están más abajo
            
            # 4. Evaluar separación de los pies
            left_ankle = keypoints['left_ankle']
            right_ankle = keypoints['right_ankle']
            ankle_separation = abs(left_ankle[0] - right_ankle[0])
            
            result['metrics'] = {
                'torso_height': torso_height,
                'knee_separation': knee_separation,
                'knee_hip_diff': knee_hip_diff,
                'ankle_separation': ankle_separation
            }
            
            logger.info(f"Frontal - Altura torso: {torso_height:.1f}, Sep. rodillas: {knee_separation:.1f}, "
                       f"Dif. rodilla-cadera: {knee_hip_diff:.1f}, Sep. tobillos: {ankle_separation:.1f}")
            
            # Lógica de clasificación frontal
            sitting_indicators = 0
            standing_indicators = 0
            
            # Indicador 1: Separación de rodillas (sentado = rodillas más juntas)
            if knee_separation < torso_height * 0.8:  # Rodillas relativamente juntas
                sitting_indicators += 1
            else:
                standing_indicators += 1
            
            # Indicador 2: Posición de rodillas respecto a caderas (sentado = rodillas más bajas)
            if knee_hip_diff > torso_height * 0.3:  # Rodillas significativamente más bajas
                sitting_indicators += 1
            else:
                standing_indicators += 1
            
            # Indicador 3: Separación de tobillos (sentado = pies más cerca del cuerpo)
            if ankle_separation < knee_separation * 1.2:
                sitting_indicators += 1
            else:
                standing_indicators += 1
            
            # Determinar resultado frontal
            if sitting_indicators > standing_indicators:
                result['posture'] = 'sentado'
                result['confidence'] = sitting_indicators / 3.0
            elif standing_indicators > sitting_indicators:
                result['posture'] = 'de pie'
                result['confidence'] = standing_indicators / 3.0
            else:
                result['posture'] = 'indeterminado'
                result['confidence'] = 0.5
            
        except Exception as e:
            logger.error(f"Error procesando keypoints frontales: {e}")
            result['posture'] = 'indeterminado'
            result['confidence'] = 0.0
        
        return result
    
    def _process_lateral_keypoints(self, keypoints: Dict[str, Tuple]) -> Dict:
        """
        Procesa los keypoints de la cámara lateral para evaluar la postura.
        
        Args:
            keypoints: Diccionario de keypoints {name: (x, y, confidence)}
            
        Returns:
            Dict: Resultado del análisis lateral
        """
        result = {
            'posture': 'indeterminado',
            'confidence': 0.0,
            'metrics': {},
            'missing_keypoints': []
        }
        
        try:
            # Verificar keypoints requeridos
            print("-----------------------------------------------------------")
            print("hola" + str(keypoints))
            missing = [kp for kp in self.required_keypoints_lateral if kp not in keypoints]
            result['missing_keypoints'] = missing
            
            if missing:
                logger.warning(f"Keypoints faltantes en cámara lateral: {missing}")
                return result
            
            # Obtener keypoints
            shoulder = keypoints['left_shoulder']
            hip = keypoints['left_hip']
            knee = keypoints['left_knee']
            ankle = keypoints['left_ankle']
            neck = keypoints.get('neck', shoulder)  # Usar hombro si no hay cuello
            
            # 1. Calcular ángulo de la rodilla (cadera-rodilla-tobillo)
            knee_angle = self._calculate_angle(hip, knee, ankle)
            
            # 2. Calcular ángulo de la cadera (hombro-cadera-rodilla)
            hip_angle = self._calculate_angle(shoulder, hip, knee)
            
            # 3. Calcular ángulo de la espalda (cuello-cadera-línea vertical)
            # Crear punto vertical desde la cadera
            vertical_point = (hip[0], hip[1] - 100)  # Punto arriba de la cadera
            back_angle = self._calculate_angle(neck, hip, vertical_point)
            
            # 4. Evaluar altura relativa de rodilla respecto a cadera
            knee_height_ratio = (hip[1] - knee[1]) / abs(hip[1] - shoulder[1]) if abs(hip[1] - shoulder[1]) > 0 else 0
            
            result['metrics'] = {
                'knee_angle': knee_angle,
                'hip_angle': hip_angle,
                'back_angle': back_angle,
                'knee_height_ratio': knee_height_ratio
            }
            
            logger.info(f"Lateral - Ángulo rodilla: {knee_angle:.1f}°, Ángulo cadera: {hip_angle:.1f}°, "
                       f"Ángulo espalda: {back_angle:.1f}°, Ratio altura rodilla: {knee_height_ratio:.2f}")
            
            # Lógica de clasificación lateral
            sitting_indicators = 0
            standing_indicators = 0
            
            # Indicador 1: Ángulo de rodilla (sentado = rodilla doblada < 120°)
            if knee_angle < 120:
                sitting_indicators += 1
            else:
                standing_indicators += 1
            
            # Indicador 2: Ángulo de cadera (sentado = cadera doblada < 110°)
            if hip_angle < 110:
                sitting_indicators += 1
            else:
                standing_indicators += 1
            
            # Indicador 3: Posición de la espalda (sentado = más inclinada hacia adelante)
            if back_angle > 15:  # Espalda inclinada hacia adelante
                sitting_indicators += 1
            else:
                standing_indicators += 1
            
            # Indicador 4: Altura de rodilla (sentado = rodilla más alta)
            if knee_height_ratio < 0.5:  # Rodilla más alta que la mitad del torso
                sitting_indicators += 1
            else:
                standing_indicators += 1
            
            # Determinar resultado lateral
            total_indicators = sitting_indicators + standing_indicators
            if sitting_indicators > standing_indicators:
                result['posture'] = 'sentado'
                result['confidence'] = sitting_indicators / total_indicators
            elif standing_indicators > sitting_indicators:
                result['posture'] = 'de pie'
                result['confidence'] = standing_indicators / total_indicators
            else:
                result['posture'] = 'indeterminado'
                result['confidence'] = 0.5
            
        except Exception as e:
            logger.error(f"Error procesando keypoints laterales: {e}")
            result['posture'] = 'indeterminado'
            result['confidence'] = 0.0
        
        return result
    
    def _calculate_angle(self, point_a: Tuple, point_b: Tuple, point_c: Tuple) -> float:
        """
        Calcula el ángulo en el punto B formado por los puntos A-B-C.
        
        Args:
            point_a: Punto A (x, y)
            point_b: Punto B (vértice del ángulo) (x, y)
            point_c: Punto C (x, y)
            
        Returns:
            float: Ángulo en grados
        """
        try:
            # Vectores BA y BC
            ba = (point_a[0] - point_b[0], point_a[1] - point_b[1])
            bc = (point_c[0] - point_b[0], point_c[1] - point_b[1])
            
            # Producto punto
            dot_product = ba[0] * bc[0] + ba[1] * bc[1]
            
            # Magnitudes
            magnitude_ba = math.sqrt(ba[0]**2 + ba[1]**2)
            magnitude_bc = math.sqrt(bc[0]**2 + bc[1]**2)
            
            # Evitar división por cero
            if magnitude_ba == 0 or magnitude_bc == 0:
                return 0.0
            
            # Coseno del ángulo
            cosine_angle = dot_product / (magnitude_ba * magnitude_bc)
            
            # Asegurar que el coseno esté en el rango válido [-1, 1]
            cosine_angle = max(-1.0, min(1.0, cosine_angle))
            
            # Ángulo en radianes y luego en grados
            angle_radians = math.acos(cosine_angle)
            angle_degrees = math.degrees(angle_radians)
            
            return angle_degrees
        
        except Exception as e:
            logger.error(f"Error calculando ángulo: {e}")
            return 0.0
    
    def _combine_results(self, frontal_result: Dict, lateral_result: Dict) -> Dict:
        """
        Combina los resultados de ambas cámaras para tomar una decisión final.
        
        Args:
            frontal_result: Resultado del análisis frontal
            lateral_result: Resultado del análisis lateral
            
        Returns:
            Dict: Resultado final combinado
        """
        # Pesos para cada cámara (lateral tiene más peso para posturas)
        lateral_weight = 0.7
        frontal_weight = 0.3
        
        frontal_posture = frontal_result['posture']
        lateral_posture = lateral_result['posture']
        frontal_conf = frontal_result['confidence']
        lateral_conf = lateral_result['confidence']
        
        # Casos de decisión
        if frontal_posture == lateral_posture and frontal_posture != 'indeterminado':
            # Ambas cámaras coinciden
            final_posture = frontal_posture
            final_confidence = (frontal_conf * frontal_weight + lateral_conf * lateral_weight)
        elif lateral_posture != 'indeterminado' and lateral_conf > 0.6:
            # Confiar en la cámara lateral si tiene alta confianza
            final_posture = lateral_posture
            final_confidence = lateral_conf * 0.8  # Reducir ligeramente por no tener consenso
        elif frontal_posture != 'indeterminado' and frontal_conf > 0.6:
            # Confiar en la cámara frontal si lateral es indeterminado
            final_posture = frontal_posture
            final_confidence = frontal_conf * 0.6  # Reducir más por ser menos confiable para postura
        else:
            # Caso indeterminado
            final_posture = 'indeterminado'
            final_confidence = (frontal_conf * frontal_weight + lateral_conf * lateral_weight) * 0.5
        
        details = {
            'lateral_weight': lateral_weight,
            'frontal_weight': frontal_weight,
            'decision_logic': 'consensus' if frontal_posture == lateral_posture else 'weighted',
            'frontal_confidence': frontal_conf,
            'lateral_confidence': lateral_conf
        }
        
        logger.info(f"Combinación - Frontal: {frontal_posture}({frontal_conf:.2f}), "
                   f"Lateral: {lateral_posture}({lateral_conf:.2f}) → Final: {final_posture}({final_confidence:.2f})")
        
        return {
            'posture': final_posture,
            'confidence': final_confidence,
            'details': details
        }


def classify_posture_simple(keypoints_frontal: List[Tuple], keypoints_lateral: List[Tuple], 
                          confidence_threshold: float = 0.01, debug: bool = False) -> str:
    """
    Función simple para clasificar postura sin detalles adicionales.
    
    Args:
        keypoints_frontal: Lista de keypoints de la cámara frontal
        keypoints_lateral: Lista de keypoints de la cámara lateral  
        confidence_threshold: Umbral de confianza mínimo
        debug: Si True, muestra información detallada de debug
        
    Returns:
        str: 'de pie', 'sentado' o 'indeterminado'
    """
    # Temporalmente cambiar nivel de logging si debug está activado
    if debug:
        logger.setLevel(logging.DEBUG)
        logging.basicConfig(level=logging.DEBUG, 
                          format='%(levelname)s - %(message)s')
    
    classifier = PostureClassifier(confidence_threshold)
    result = classifier.classify_posture(keypoints_frontal, keypoints_lateral)
    
    # Restaurar nivel de logging
    if debug:
        logger.setLevel(logging.INFO)
    
    return result['posture']


# Ejemplo de uso
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Keypoints de ejemplo (formato: x, y, confidence, keypoint_idx)
    # Simulando una persona sentada
    sample_frontal = [
        (320, 100, 0.9, 5),   # left_shoulder
        (380, 100, 0.9, 6),   # right_shoulder  
        (300, 200, 0.8, 11),  # left_hip
        (400, 200, 0.8, 12),  # right_hip
        (290, 280, 0.7, 13),  # left_knee
        (410, 280, 0.7, 14),  # right_knee
        (285, 350, 0.6, 15),  # left_ankle
        (415, 350, 0.6, 16),  # right_ankle
    ]
    
    sample_lateral = [
        (200, 100, 0.9, 5),   # left_shoulder
        (180, 200, 0.8, 11),  # left_hip
        (220, 280, 0.7, 13),  # left_knee
        (250, 350, 0.6, 15),  # left_ankle
        (190, 80, 0.8, 17),   # neck
    ]
    
    classifier = PostureClassifier()
    result = classifier.classify_posture(sample_frontal, sample_lateral)
    
    print(f"Resultado: {result['posture']}")
    print(f"Confianza: {result['confidence']:.2f}")
    print(f"Detalles: {result['details']}")
