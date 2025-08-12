#!/usr/bin/env python3.9
"""
Ejemplo de Uso: MediaPipe Tasks + Clasificación de Poses + Validación de Topología
=================================================================================

Este script demuestra cómo integrar MediaPipe Tasks con el clasificador de poses
PoseClassificationNet de NVIDIA TAO, incluyendo validación de la conversión de topología.

IMPORTANTE: Este script maneja automáticamente las dependencias de Python:
- Python 3.9 para MediaPipe Tasks (self.mediapipe_processor)
- Python 3.6 para el clasificador TensorRT (self.pose_classifier via subprocess)

Flujo de trabajo:
1. MediaPipe Tasks detecta keypoints de personas en video (33 keypoints) - Python 3.9
2. Se valida y convierte la topología de MediaPipe a NVIDIA (34 keypoints)
3. TRTPoseClassifier procesa y clasifica las poses via subprocess - Python 3.6
4. Se muestra el resultado en tiempo real con validación visual

Autor: Sistema de IA
Fecha: 2025
"""

import cv2
import numpy as np
import time
import logging
from pathlib import Path
# Eliminado matplotlib (headless)
import json
import subprocess
import tempfile
import os
import sys
from pathlib import Path
import threading
import queue

# Añadir el directorio 'Automatizacion' al sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Importar solo MediaPipe Tasks (Python 3.9)
from utils.prueba import MediaPipeTasksPoseProcessor

# Configurar logging más detallado para debugging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constantes de verbosidad
LOG_EVERY_N_FRAMES = 10
LOG_EVERY_SECONDS = 5.0
SLOW_FRAME_THRESHOLD_MS = 150.0

class PoseClassifierPython36:
    """
    Wrapper para el clasificador de poses que se ejecuta en Python 3.6 via subprocess
    Optimizado: soporta modo persistente evitando spawn por frame y usando comunicación
    por stdin/stdout con JSON delimitado por líneas.
    """
    
    def __init__(self, model_path: str, sequence_length: int = 30, confidence_threshold: float = 0.2,
                 persistent: bool = True):
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        self._persistent = persistent
        
        # Crear directorio temporal para comunicación entre procesos
        self.temp_dir = tempfile.mkdtemp(prefix="pose_classifier_")
        self.input_file = os.path.join(self.temp_dir, "keypoints_input.json")
        self.output_file = os.path.join(self.temp_dir, "classification_output.json")
        
        # Crear script de clasificación para Python 3.6
        self.classifier_script = self._create_classifier_script()
        
        # Buffer de secuencias para el clasificador
        self.sequence_buffer = []
        
        # Estadísticas
        self.stats = {
            'total_predictions': 0,
            'confident_predictions': 0,
            'class_predictions': {
                'sitting_down': 0, 'getting_up': 0, 'sitting': 0,
                'standing': 0, 'walking': 0, 'jumping': 0
            }
        }
        
        logger.info("✅ PoseClassifierPython36 inicializado (Python 3.6 subprocess)")
        logger.info(f"   📁 Modelo: {os.path.basename(model_path)}")
        logger.info(f"   📂 Temp dir: {self.temp_dir}")
        
        # Verificar que Python 3.6 esté disponible
        self._test_python36_availability()
        self._worker = None
        self._worker_lock = threading.Lock()
        if self._persistent:
            self._start_persistent_process()
 
    def _test_python36_availability(self):
        """Verifica que Python 3.6 esté disponible y funcional"""
        try:
            logger.info("🔍 Verificando disponibilidad de Python 3.6...")
            result = subprocess.run(['python3.6', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info(f"✅ Python 3.6 disponible: {result.stdout.strip()}")
                
                # Prueba básica de importación
                test_script = f'''
import sys
import os
sys.path.append("{str(Path(__file__).resolve().parent.parent)}")
try:
    from utils.action_classifier import create_pose_classifier
    print("SUCCESS: action_classifier importado correctamente")
except ImportError as e:
    print(f"ERROR: No se pudo importar action_classifier: {{e}}")
    sys.exit(1)
'''
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(test_script)
                    test_file = f.name
                
                try:
                    result = subprocess.run(['python3.6', test_file], 
                                          capture_output=True, text=True, timeout=15)
                    if result.returncode == 0 and "SUCCESS" in result.stdout:
                        logger.info("✅ action_classifier disponible en Python 3.6")
                    else:
                        logger.error(f"❌ Error importando action_classifier: {result.stderr}")
                        logger.error(f"❌ Stdout: {result.stdout}")
                finally:
                    os.unlink(test_file)
            else:
                logger.error(f"❌ Python 3.6 no disponible: {result.stderr}")
        except Exception as e:
            logger.error(f"❌ Error verificando Python 3.6: {e}")
    
    def _create_classifier_script(self) -> str:
        """Crea el script de clasificación para Python 3.6"""
        script_path = os.path.join(self.temp_dir, "classifier_worker.py")
        # Usamos plantilla con placeholders para evitar conflictos con f-strings internos
        script_template = """#!/usr/bin/env python3.6
import json
import sys
import os
import numpy as np
import traceback

# Añadir path
sys.path.append("__PY_PATH__")

def debug_print(msg):
    # Debug helper
    print(f"[DEBUG CLASSIFIER] {msg}", file=sys.stderr)

try:
    debug_print("Iniciando importación del clasificador...")
    from utils.action_classifier import create_pose_classifier
    debug_print("Importación exitosa")
    
    # Crear clasificador
    debug_print("Creando clasificador...")
    classifier = create_pose_classifier(
        model_path="__MODEL_PATH__",
        input_keypoint_format='mediapipe',
        keypoint_format='nvidia',
        sequence_length=__SEQ_LEN__,
        confidence_threshold=__CONF_THRESH__
    )
    debug_print("Clasificador creado exitosamente")
    
    def classify_array(keypoints):
        try:
            result = classifier.process_keypoints(keypoints)
            return {'success': True, 'result': result if result else {'error': True, 'message': 'Classifier returned None'}}
        except Exception as e:
            return {'success': False, 'error': str(e), 'traceback': traceback.format_exc()}
    
    def loop_mode():
        debug_print("Entrando en loop persistente")
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            if line == 'QUIT':
                print(json.dumps({'quit': True}))
                sys.stdout.flush()
                break
            try:
                payload = json.loads(line)
                kps = np.array(payload['keypoints'], dtype=float)
                out = classify_array(kps)
                out['debug_info'] = {
                    'shape': list(kps.shape),
                    'valid_keypoints': int(np.sum(kps[:,2] > 0.1))
                }
                print(json.dumps(out))
                sys.stdout.flush()
            except Exception as e:
                print(json.dumps({'success': False, 'error': str(e), 'traceback': traceback.format_exc()}))
                sys.stdout.flush()
        debug_print("Loop persistente finalizado")
    
    def process_keypoints(input_file, output_file):
        try:
            debug_print(f"Procesando archivo: {input_file}")
            with open(input_file, 'r') as f:
                data = json.load(f)
            keypoints = np.array(data['keypoints'])
            debug_print(f"Keypoints shape: {keypoints.shape}")
            debug_print(f"Keypoints válidos: {np.sum(keypoints[:, 2] > 0.1)}")
            out = classify_array(keypoints)
            out['debug_info'] = {
                'keypoints_received': list(keypoints.shape),
                'valid_keypoints': int(np.sum(keypoints[:, 2] > 0.1))
            }
            with open(output_file, 'w') as f:
                json.dump(out, f)
        except Exception as e:
            with open(output_file, 'w') as f:
                json.dump({'success': False, 'error': str(e), 'traceback': traceback.format_exc()}, f)
    
    if __name__ == "__main__":
        if '--loop' in sys.argv:
            loop_mode()
        else:
            if len(sys.argv) != 3:
                debug_print("Argumentos incorrectos")
                print("Uso: python3.6 classifier_worker.py <input_file> <output_file>")
                sys.exit(1)
            process_keypoints(sys.argv[1], sys.argv[2])

except ImportError as e:
    debug_print(f"ImportError: {str(e)}")
    with open(sys.argv[2] if len(sys.argv) > 2 else "/tmp/error.json", 'w') as f:
        json.dump({'success': False, 'error': f"ImportError: {str(e)}", 'traceback': traceback.format_exc()}, f)
except Exception as e:
    debug_print(f"Error general: {str(e)}")
    with open(sys.argv[2] if len(sys.argv) > 2 else "/tmp/error.json", 'w') as f:
        json.dump({'success': False, 'error': str(e), 'traceback': traceback.format_exc()}, f)
"""
        script_content = (script_template
                          .replace("__PY_PATH__", str(Path(__file__).resolve().parent.parent))
                          .replace("__MODEL_PATH__", self.model_path.replace('"', '\"'))
                          .replace("__SEQ_LEN__", str(self.sequence_length))
                          .replace("__CONF_THRESH__", str(self.confidence_threshold)))
        with open(script_path, 'w') as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)
        return script_path
    
    def _start_persistent_process(self):
        """Inicia proceso persistente en modo loop"""
        if self._worker is not None:
            return
        try:
            logger.info("🚀 Iniciando clasificador persistente Python 3.6...")
            self._worker = subprocess.Popen(
                ['python3.6', self.classifier_script, '--loop'],
                stdin=subprocess.PIPE,
                stdout=sys.stdout,  # Redirigir stdout a la terminal principal
                #stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            # Hilo para leer stderr y loguear (evita deadlocks)
            def _stderr_reader(proc):
                for line in proc.stderr:
                    line = line.rstrip()
                    if line:
                        logger.debug(f"[CLF36] {line}")
            threading.Thread(target=_stderr_reader, args=(self._worker,), daemon=True).start()

            def _stdout_reader(proc):
                for line in proc.stdout:
                    line = line.rstrip()
                    if line:
                        logger.info(f"[CLF36 STDOUT] {line}")

            threading.Thread(target=_stdout_reader, args=(self._worker,), daemon=True).start()
            logger.info("✅ Clasificador persistente iniciado")
        except Exception as e:
            logger.error(f"❌ No se pudo iniciar proceso persistente: {e}")
            self._worker = None
 
    def process_keypoints(self, keypoints: np.ndarray) -> dict:
        """
        Procesa keypoints usando el clasificador en Python 3.6
        """
        # Ruta optimizada si proceso persistente activo
        if self._persistent and self._worker and self._worker.stdin and self._worker.poll() is None:
            try:
                payload = {'keypoints': keypoints.tolist()}
                line = json.dumps(payload)
                t0 = time.time()
                with self._worker_lock:
                    self._worker.stdin.write(line + '\n')
                    self._worker.stdin.flush()
                    # Leer respuesta (bloqueante). Se podría mejorar con timeout usando hilos.
                    response = self._worker.stdout.readline()
                rt_ms = (time.time() - t0) * 1000
                if not response:
                    return {'error': True, 'message': 'Empty response from persistent worker'}
                data = json.loads(response)
                if data.get('quit'):
                    return {'error': True, 'message': 'Worker terminating'}
                if not data.get('success', False):
                    return {'error': True, 'message': data.get('error', 'Unknown'), 'traceback': data.get('traceback')}
                classification_result = data.get('result')
                if classification_result and not classification_result.get('error', False):
                    self.stats['total_predictions'] += 1
                    if classification_result.get('confidence', 0) > 0.5:
                        self.stats['confident_predictions'] += 1
                    pose_class = classification_result.get('predicted_class', 'unknown')
                    if pose_class in self.stats['class_predictions']:
                        self.stats['class_predictions'][pose_class] += 1
                    classification_result['inference_time_ms'] = classification_result.get('inference_time_ms', rt_ms)
                return classification_result
            except Exception as e:
                logger.error(f"❌ Error en worker persistente: {e}")
                return {'error': True, 'message': str(e)}
        # Fallback a modo antiguo (no persistente)
        try:
            logger.debug("⚠️ Usando modo no persistente (fallback)")
            with open(self.input_file, 'w') as f:
                json.dump({'keypoints': keypoints.tolist()}, f)
            cmd = ['python3.6', self.classifier_script, self.input_file, self.output_file]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                return {'error': True, 'message': 'Subprocess failed', 'stderr': result.stderr}
            if os.path.exists(self.output_file):
                with open(self.output_file, 'r') as f:
                    data = json.load(f)
                if data.get('success'):
                    classification_result = data.get('result')
                    if classification_result and not classification_result.get('error', False):
                        self.stats['total_predictions'] += 1
                        if classification_result.get('confidence', 0) > 0.5:
                            self.stats['confident_predictions'] += 1
                        pose_class = classification_result.get('predicted_class', 'unknown')
                        if pose_class in self.stats['class_predictions']:
                            self.stats['class_predictions'][pose_class] += 1
                    return classification_result
                return {'error': True, 'message': data.get('error', 'Unknown')}
            return {'error': True, 'message': 'Output file not found'}
        except Exception as e:
            return {'error': True, 'message': str(e)}
    
    def reset_sequence(self):
        """Reinicia la secuencia del clasificador"""
        # Para reiniciar, podríamos llamar al script con una función especial
        # Por simplicidad, solo limpiamos el buffer local
        self.sequence_buffer.clear()
        logger.info("🔄 Secuencia de clasificador reiniciada")
    
    def get_statistics(self) -> dict:
        """Obtiene estadísticas del clasificador"""
        total = self.stats['total_predictions']
        confident = self.stats['confident_predictions']
        
        return {
            'total_predictions': total,
            'confident_predictions': confident,
            'confidence_rate': confident / total if total > 0 else 0.0,
            'class_distribution': self.stats['class_predictions'].copy(),
            'most_common_class': max(self.stats['class_predictions'], 
                                   key=self.stats['class_predictions'].get)
        }
    
    def cleanup(self):
        """Limpia archivos temporales (idempotente)"""
        if getattr(self, '_cleaned', False):
            return
        # Terminar proceso persistente si activo
        try:
            if self._persistent and self._worker and self._worker.poll() is None:
                try:
                    with self._worker_lock:
                        self._worker.stdin.write('QUIT\n')
                        self._worker.stdin.flush()
                except Exception:
                    pass
                self._worker.wait(timeout=5)
        except Exception as e:
            logger.warning(f"⚠️ Error terminando worker persistente: {e}")
        try:
            import shutil
            if os.path.isdir(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                logger.info("✅ Archivos temporales del clasificador limpiados")
        except Exception as e:
            logger.warning(f"⚠️ Error limpiando archivos temporales: {e}")
        self._cleaned = True
    
    def __del__(self):
        """Destructor"""
        self.cleanup()
    
    # Mapeo para compatibilidad con la validación
    MEDIAPIPE_TO_NVIDIA_MAPPING = {
        0: 15,   # nose -> nose (15)
        1: None, # left_eye_inner -> no equivalente directo
        2: 16,   # left_eye -> left_eye (16)
        3: None, # left_eye_outer -> no equivalente directo
        4: None, # right_eye_inner -> no equivalente directo
        5: 17,   # right_eye -> right_eye (17)
        6: None, # right_eye_outer -> no equivalente directo
        7: 18,   # left_ear -> left_ear (18)
        8: 19,   # right_ear -> right_ear (19)
        9: None, # mouth_left -> no equivalente directo
        10: None, # mouth_right -> no equivalente directo
        11: 20,  # left_shoulder -> left_shoulder (20)
        12: 21,  # right_shoulder -> right_shoulder (21)
        13: 22,  # left_elbow -> left_elbow (22)
        14: 23,  # right_elbow -> right_elbow (23)
        15: 24,  # left_wrist -> left_wrist (24)
        16: 25,  # right_wrist -> right_wrist (25)
        17: 26,  # left_pinky -> left_pinky_knuckle (26)
        18: 27,  # right_pinky -> right_pinky_knuckle (27)
        19: 30,  # left_index -> left_index_knuckle (30)
        20: 31,  # right_index -> right_index_knuckle (31)
        21: 32,  # left_thumb -> left_thumb_tip (32)
        22: 33,  # right_thumb -> right_thumb_tip (33)
        23: 1,   # left_hip -> left_hip (1)
        24: 2,   # right_hip -> right_hip (2)
        25: 4,   # left_knee -> left_knee (4)
        26: 5,   # right_knee -> right_knee (5)
        27: 7,   # left_ankle -> left_ankle (7)
        28: 8,   # right_ankle -> right_ankle (8)
        29: 13,  # left_heel -> left_heel (13)
        30: 14,  # right_heel -> right_heel (14)
        31: 9,   # left_foot_index -> left_big_toe (9)
        32: 10   # right_foot_index -> right_big_toe (10)
    }


class MediaPipeWithClassifier:
    """
    Integra MediaPipe Tasks con clasificación de poses y validación de topología
    """
    
    # Mapeo de nombres de keypoints para validación visual
    MEDIAPIPE_KEYPOINT_NAMES = [
        'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
        'right_eye_inner', 'right_eye', 'right_eye_outer',
        'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
        'left_index', 'right_index', 'left_thumb', 'right_thumb',
        'left_hip', 'right_hip', 'left_knee', 'right_knee',
        'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index'
    ]
    
    NVIDIA_KEYPOINT_NAMES = [
        'hips', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
        'neck', 'left_ankle', 'right_ankle', 'left_big_toe', 'right_big_toe',
        'head', 'left_little_toe', 'left_heel', 'right_heel', 'nose',
        'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder',
        'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
        'left_pinky_knuckle', 'right_pinky_knuckle', 'left_middle_knuckle', 'right_middle_knuckle',
        'left_index_knuckle', 'right_index_knuckle', 'left_thumb_tip', 'right_thumb_tip'
    ]
    
    def __init__(self, 
                 mediapipe_model_path: str,
                 pose_classifier_model_path: str,
                 validation_mode: bool = True,
                 fast_mode: bool = False,
                 parallel_classification: bool = True):
        """
        Inicializa el sistema completo
        
        Args:
            mediapipe_model_path: Ruta al modelo .task de MediaPipe
            pose_classifier_model_path: Ruta al modelo engine del clasificador
            validation_mode: Si activar validación visual de topología
            fast_mode: Si True, desactiva validación y reduce sobrecarga
            parallel_classification: Ejecuta clasificación en hilo separado
        """
        self.validation_mode = validation_mode and not fast_mode
        self.fast_mode = fast_mode
        self.parallel_classification = parallel_classification
         
        # Crear procesador de MediaPipe Tasks (Python 3.9)
        logger.info("🔧 Inicializando MediaPipe processor (Python 3.9)...")
        self.mediapipe_processor = MediaPipeTasksPoseProcessor(
            model_path=mediapipe_model_path,
            confidence_threshold=0.1,  # Reducido para capturar más keypoints
            debug=True  # Activar debug
        )
        
        # Crear clasificador de poses con wrapper Python 3.6
        logger.info("🔧 Inicializando clasificador de poses (Python 3.6 subprocess)...")
        self.pose_classifier = PoseClassifierPython36(
            model_path=pose_classifier_model_path,
            sequence_length=15,  # Secuencia más corta para pruebas
            confidence_threshold=0.05,  # Umbral más bajo
            persistent=True
        )
        
        # Estadísticas
        self.stats = {
            'frames_processed': 0,
            'poses_detected': 0,
            'poses_classified': 0,
            'topology_validations': 0,
            'mapping_errors': 0,
            'start_time': time.time()
        }
        
        # Datos para validación
        self.validation_data = {
            'mediapipe_keypoints_history': [],
            'nvidia_keypoints_history': [],
            'mapping_validation_results': []
        }
        
        # Progreso y métricas
        self._start_time = self.stats['start_time']
        self._last_log_time = self._start_time
        self._last_log_frame = 0
        self._slow_frames = 0
        self._classification_errors = 0
        self._classification_none = 0
        self._video_total_frames = None
        # Infraestructura para clasificación paralela
        self._classification_queue = None
        self._classification_thread = None
        self._classification_results = {}
        self._classification_results_lock = threading.Lock()
        if self.parallel_classification:
            self._classification_queue = queue.Queue(maxsize=5)
            self._classification_thread = threading.Thread(target=self._classification_worker, daemon=True)
            self._classification_thread.start()
         
        logger.info("✅ Sistema MediaPipe + Clasificador + Validación inicializado")
        logger.info("   🎯 Procesador: MediaPipeTasksPoseProcessor (Python 3.9)")
        logger.info("   🎭 Clasificador: PoseClassifierPython36 (subprocess Python 3.6)")
        logger.info(f"   🔍 Validación de topología: {'Activada' if self.validation_mode else 'Desactivada'}")
        logger.info(f"   ⚡ Modo rápido: {'Sí' if self.fast_mode else 'No'} | Paralelo: {'Sí' if self.parallel_classification else 'No'}")
    
    def _classification_worker(self):
        """Hilo consumidor para clasificación asíncrona"""
        while True:
            item = self._classification_queue.get()
            if item is None:
                break
            idx, keypoints = item
            result = self.pose_classifier.process_keypoints(keypoints)
            with self._classification_results_lock:
                self._classification_results[idx] = result
            self._classification_queue.task_done()

    def _enqueue_classification(self, frame_idx: int, keypoints: np.ndarray):
        if not self.parallel_classification:
            return None
        try:
            self._classification_queue.put_nowait((frame_idx, keypoints))
        except queue.Full:
            logger.warning("⚠️ Cola de clasificación llena, descartando frame")

    def _get_classification_result(self, frame_idx: int):
        if not self.parallel_classification:
            return None
        with self._classification_results_lock:
            result = self._classification_results.pop(frame_idx, None)
            logger.info(f"🔍 Resultado de clasificación para frame {frame_idx}: {result}")
            return result

    def validate_topology_mapping(self, mediapipe_keypoints: np.ndarray) -> dict:
        """
        Valida la conversión de topología MediaPipe a NVIDIA
        
        Args:
            mediapipe_keypoints: Keypoints originales de MediaPipe [33, 3]
            
        Returns:
            dict: Resultados de validación con estadísticas y errores detectados
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'mapped_keypoints_count': 0,
            'unmapped_keypoints_count': 0,
            'mediapipe_confidence_stats': {},
            'nvidia_confidence_stats': {},
            'mapping_details': []
        }
        
        try:
            # Obtener el mapeo desde el clasificador
            mapping = self.pose_classifier.MEDIAPIPE_TO_NVIDIA_MAPPING
            
            # Crear array NVIDIA inicializado a cero
            nvidia_keypoints = np.zeros((34, 3), dtype=np.float32)
            
            # Estadísticas de MediaPipe
            mediapipe_valid = mediapipe_keypoints[mediapipe_keypoints[:, 2] > 0.1]
            validation_result['mediapipe_confidence_stats'] = {
                'total_keypoints': len(mediapipe_keypoints),
                'valid_keypoints': len(mediapipe_valid),
                'mean_confidence': float(np.mean(mediapipe_valid[:, 2])) if len(mediapipe_valid) > 0 else 0.0,
                'max_confidence': float(np.max(mediapipe_valid[:, 2])) if len(mediapipe_valid) > 0 else 0.0,
                'min_confidence': float(np.min(mediapipe_valid[:, 2])) if len(mediapipe_valid) > 0 else 0.0
            }
            
            # Procesar cada keypoint de MediaPipe
            mapped_count = 0
            unmapped_count = 0
            
            for mp_idx in range(len(mediapipe_keypoints)):
                mp_kp = mediapipe_keypoints[mp_idx]
                mp_name = self.MEDIAPIPE_KEYPOINT_NAMES[mp_idx] if mp_idx < len(self.MEDIAPIPE_KEYPOINT_NAMES) else f"mp_{mp_idx}"
                
                if mp_idx in mapping:
                    nvidia_idx = mapping[mp_idx]
                    
                    if nvidia_idx is not None:
                        # Mapeo válido
                        nvidia_keypoints[nvidia_idx] = mp_kp
                        mapped_count += 1
                        
                        nvidia_name = self.NVIDIA_KEYPOINT_NAMES[nvidia_idx] if nvidia_idx < len(self.NVIDIA_KEYPOINT_NAMES) else f"nvidia_{nvidia_idx}"
                        
                        validation_result['mapping_details'].append({
                            'mediapipe_idx': mp_idx,
                            'mediapipe_name': mp_name,
                            'nvidia_idx': nvidia_idx,
                            'nvidia_name': nvidia_name,
                            'confidence': float(mp_kp[2]),
                            'coordinates': [float(mp_kp[0]), float(mp_kp[1])],
                            'is_valid': mp_kp[2] > 0.1
                        })
                        
                        # Validaciones específicas
                        if 'left' in mp_name.lower() and 'right' in nvidia_name.lower():
                            validation_result['errors'].append(f"Posible error L/R: {mp_name} -> {nvidia_name}")
                            validation_result['is_valid'] = False
                            
                        if 'right' in mp_name.lower() and 'left' in nvidia_name.lower():
                            validation_result['errors'].append(f"Posible error R/L: {mp_name} -> {nvidia_name}")
                            validation_result['is_valid'] = False
                    else:
                        # Keypoint sin mapeo directo
                        unmapped_count += 1
                        validation_result['warnings'].append(f"Sin mapeo: {mp_name} (idx {mp_idx})")
                else:
                    # Keypoint no considerado en el mapeo
                    unmapped_count += 1
                    validation_result['errors'].append(f"Keypoint no mapeado: {mp_name} (idx {mp_idx})")
                    validation_result['is_valid'] = False
            
            # Estadísticas de NVIDIA
            nvidia_valid = nvidia_keypoints[nvidia_keypoints[:, 2] > 0.1]
            validation_result['nvidia_confidence_stats'] = {
                'total_keypoints': len(nvidia_keypoints),
                'valid_keypoints': len(nvidia_valid),
                'mean_confidence': float(np.mean(nvidia_valid[:, 2])) if len(nvidia_valid) > 0 else 0.0,
                'max_confidence': float(np.max(nvidia_valid[:, 2])) if len(nvidia_valid) > 0 else 0.0,
                'min_confidence': float(np.min(nvidia_valid[:, 2])) if len(nvidia_valid) > 0 else 0.0
            }
            
            validation_result['mapped_keypoints_count'] = mapped_count
            validation_result['unmapped_keypoints_count'] = unmapped_count
            
            # Guardar para análisis posterior
            if self.validation_mode:
                self.validation_data['mediapipe_keypoints_history'].append(mediapipe_keypoints.copy())
                self.validation_data['nvidia_keypoints_history'].append(nvidia_keypoints.copy())
                self.validation_data['mapping_validation_results'].append(validation_result.copy())
            
            self.stats['topology_validations'] += 1
            if not validation_result['is_valid']:
                self.stats['mapping_errors'] += 1
                
        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Error en validación: {str(e)}")
            logger.error(f"❌ Error validando topología: {e}")
        
        return validation_result
    
    def _progress_metrics(self):
        frames = self.stats['frames_processed']
        elapsed = max(time.time() - self._start_time, 1e-6)
        avg_fps = frames / elapsed
        if self._video_total_frames:
            percent = (frames / self._video_total_frames) * 100.0
            remaining = max(self._video_total_frames - frames, 0)
            eta = remaining / avg_fps if avg_fps > 0 else -1
        else:
            percent, eta = None, -1
        return avg_fps, percent, eta

    def _maybe_log_progress(self, force=False):
        frames = self.stats['frames_processed']
        if frames == 0:
            return
        now = time.time()
        if not force:
            if frames - self._last_log_frame < LOG_EVERY_N_FRAMES and (now - self._last_log_time) < LOG_EVERY_SECONDS:
                return
        avg_fps, percent, eta = self._progress_metrics()
        msg = [f"Frames={frames}", f"AvgFPS={avg_fps:.2f}", f"Det={self.stats['poses_detected']}", f"Cls={self.stats['poses_classified']}", f"ClsErr={self._classification_errors}", f"ClsNone={self._classification_none}", f"Slow={self._slow_frames}"]
        if percent is not None:
            msg.append(f"{percent:.1f}%")
        if eta >= 0:
            msg.append(f"ETA={eta/60:.1f}m")
        logger.info("PROGRESO: " + " | ".join(msg))
        self._last_log_frame = frames
        self._last_log_time = now

    def process_frame_with_classification_and_validation(self, image: np.ndarray) -> dict:
        """
        Procesa un frame completo: detección + validación + clasificación
        """
        frame_result = {
            'people_detected': 0,
            'pose_classifications': [],
            'processing_time_ms': 0,
            'mediapipe_keypoints': None,
            'validation_result': None,
            'has_pose': False
        }
        
        start_time = time.time()
        frame_index = self.stats['frames_processed']  # índice antes de incrementar
         
        try:
            # Usar MediaPipe Tasks para obtener keypoints
            mediapipe_keypoints = self.mediapipe_processor.process_frame(image)
            logger.debug(f"🔍 MediaPipe keypoints detectados: {mediapipe_keypoints is not None}")
            if mediapipe_keypoints is not None and isinstance(mediapipe_keypoints, np.ndarray):
                logger.debug(f"🔍 Forma de keypoints: {mediapipe_keypoints.shape}")
                logger.debug(f"🔍 Confianza promedio: {np.mean(mediapipe_keypoints[:, 2]):.3f}")
                logger.debug(f"🔍 Keypoints válidos (>0.1): {np.sum(mediapipe_keypoints[:, 2] > 0.1)}/33")
                
                frame_result['people_detected'] = 1
                frame_result['mediapipe_keypoints'] = mediapipe_keypoints
                frame_result['has_pose'] = True
                
                # Validar conversión de topología si está activada
                if self.validation_mode:
                    validation_result = self.validate_topology_mapping(mediapipe_keypoints)
                    frame_result['validation_result'] = validation_result
                    logger.debug(f"🔍 Validación topología - válida: {validation_result['is_valid']}")
                    logger.debug(f"🔍 Keypoints mapeados: {validation_result['mapped_keypoints_count']}/33")
                    
                    # Log errores críticos
                    if not validation_result['is_valid']:
                        logger.warning(f"⚠️ Errores de mapeo detectados: {len(validation_result['errors'])}")
                        for error in validation_result['errors'][:3]:  # Mostrar máximo 3 errores
                            logger.warning(f"   🔴 {error}")
                
                # Clasificar poses (el clasificador internamente convertirá MediaPipe -> NVIDIA)
                logger.debug("🔍 Preparando clasificación...")
                if self.parallel_classification:
                    self._enqueue_classification(frame_index, mediapipe_keypoints)
                    classification_result = self._get_classification_result(frame_index)  # intentar obtener si ya listo
                else:
                    classification_result = self.pose_classifier.process_keypoints(mediapipe_keypoints)
                logger.debug(f"🔍 Resultado clasificador: {classification_result}")
                
                if classification_result and not classification_result.get('error', False):
                    logger.debug(f"✅ Clasificación exitosa: {classification_result.get('predicted_class')} (conf: {classification_result.get('confidence', 0):.3f})")
                    frame_result['pose_classifications'].append({
                        'person_id': 0,
                        'pose_class': classification_result['pose_class'] if 'pose_class' in classification_result else classification_result.get('predicted_class'),
                        'confidence': classification_result.get('confidence', 0),
                        'probabilities': classification_result.get('probabilities'),
                        'inference_time_ms': classification_result.get('inference_time_ms', 0)
                    })
                    
                    self.stats['poses_classified'] += 1
                else:
                    if classification_result and classification_result.get('error'):
                        self._classification_errors += 1
                        logger.warning(f"❌ Error en clasificación: {classification_result.get('message', 'Unknown')}")
                    else:
                        self._classification_none += 1
                        logger.warning("❌ Clasificador retornó None")
                
                self.stats['poses_detected'] += 1
            else:
                logger.debug("🔍 No se detectaron keypoints válidos en este frame")
            
        except Exception as e:
            logger.error(f"❌ Error procesando frame: {e}")
        
        finally:
            frame_result['processing_time_ms'] = (time.time() - start_time) * 1000
            self.stats['frames_processed'] += 1
            self._maybe_log_progress()
        
        return frame_result
    
    def draw_results_with_validation(self, image: np.ndarray, frame_result: dict) -> np.ndarray:
        """
        Dibuja resultados sobre la imagen incluyendo información de validación
        """
        result_image = image.copy()
        
        mediapipe_keypoints = frame_result['mediapipe_keypoints']
        validation_result = frame_result['validation_result']
        
        if mediapipe_keypoints is not None:
            # Dibujar keypoints de MediaPipe con colores según validación
            for i, kp in enumerate(mediapipe_keypoints):
                x, y, conf = kp
                if conf > 0.1:  # Umbral mínimo para mostrar
                    # Color según estado de validación
                    if validation_result:
                        mapping = self.pose_classifier.MEDIAPIPE_TO_NVIDIA_MAPPING
                        if i in mapping and mapping[i] is not None:
                            color = (0, 255, 0)  # Verde: mapeado correctamente
                        else:
                            color = (0, 255, 255)  # Amarillo: no mapeado
                    else:
                        color = (255, 255, 255)  # Blanco: sin validación
                    
                    cv2.circle(result_image, (int(x), int(y)), 4, color, -1)
                    cv2.circle(result_image, (int(x), int(y)), 6, (0, 0, 0), 1)
                    
                    # Etiqueta con índice si es modo validación
                    if self.validation_mode:
                        cv2.putText(result_image, str(i), (int(x) + 8, int(y) - 8),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            # Dibujar clasificación si está disponible
            if frame_result['pose_classifications']:
                classification = frame_result['pose_classifications'][0]
                pose_class = classification['pose_class']
                confidence = classification['confidence']
                
                # Encontrar posición para texto
                valid_kps = mediapipe_keypoints[mediapipe_keypoints[:, 2] > 0.1]
                if len(valid_kps) > 0:
                    center_x = int(np.mean(valid_kps[:, 0]))
                    center_y = int(np.mean(valid_kps[:, 1])) - 50
                    
                    # Texto de clasificación con fondo
                    text = f"{pose_class}: {confidence:.2f}"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    
                    cv2.rectangle(result_image, 
                                (center_x - 5, center_y - text_size[1] - 10),
                                (center_x + text_size[0] + 10, center_y + 5),
                                (0, 0, 255), -1)
                    
                    cv2.putText(result_image, text,
                              (center_x, center_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Información de validación en la esquina
        if validation_result and self.validation_mode:
            y_offset = 30
            
            # Estado general
            status = "✅ OK" if validation_result['is_valid'] else "❌ ERROR"
            cv2.putText(result_image, f"Mapeo: {status}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if validation_result['is_valid'] else (0, 0, 255), 2)
            y_offset += 25
            
            # Estadísticas
            cv2.putText(result_image, f"Mapeados: {validation_result['mapped_keypoints_count']}/33", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
            
            cv2.putText(result_image, f"Errores: {len(validation_result['errors'])}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) if validation_result['errors'] else (255, 255, 255), 1)
        
        # Información general en la parte inferior
        info_text = f"MP Keypoints: {len(mediapipe_keypoints) if mediapipe_keypoints is not None else 0} | Tiempo: {frame_result['processing_time_ms']:.1f}ms"
        cv2.putText(result_image, info_text, (10, image.shape[0] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result_image
    
    def process_video(self, video_source: str = 0, output_path: str = None):
        """
        Procesa video en tiempo real con validación
        """
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            logger.error(f"❌ No se pudo abrir video: {video_source}")
            return
        
        # Configurar escritura de video si se especifica
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Detectar total frames si es archivo
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0 and total_frames < 10_000_000:
            self._video_total_frames = total_frames
            logger.info(f"📄 Total frames detectados: {total_frames}")
        else:
            logger.info("📄 Total frames desconocido (stream o cámara)")
        
        logger.info("🎥 Iniciando procesamiento MediaPipe + Validación + Clasificación (headless)...")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.info("🔚 Fin del video o stream")
                    break
                
                t0 = time.time()
                frame_result = self.process_frame_with_classification_and_validation(frame)
                # Recuperar resultados atrasados (paralelo) para frames anteriores si disponibles
                if self.parallel_classification:
                    pending_idx = self.stats['frames_processed'] - 1
                    pending_result = self._get_classification_result(pending_idx)
                    if pending_result and not frame_result['pose_classifications']:
                        if not pending_result.get('error'):
                            frame_result['pose_classifications'].append({
                                'person_id': 0,
                                'pose_class': pending_result.get('predicted_class'),
                                'confidence': pending_result.get('confidence', 0),
                                'probabilities': pending_result.get('probabilities'),
                                'inference_time_ms': pending_result.get('inference_time_ms', 0)
                            })
                            self.stats['poses_classified'] += 1
                if output_path and out:
                    out.write(self.draw_results_with_validation(frame, frame_result))
                frame_time_ms = (time.time() - t0) * 1000
                if frame_time_ms > SLOW_FRAME_THRESHOLD_MS:
                    self._slow_frames += 1
                self._maybe_log_progress()
        except KeyboardInterrupt:
            logger.info("⚠️ Procesamiento interrumpido por usuario")
        finally:
            cap.release()
            if out:
                out.release()
            # Vaciar cola de clasificación y esperar hilo
            if self.parallel_classification and self._classification_queue:
                try:
                    self._classification_queue.put_nowait((self.stats['frames_processed'], np.zeros((33,3))))
                except Exception:
                    pass
                try:
                    self._classification_queue.put(None)
                except Exception:
                    pass
                if self._classification_thread:
                    self._classification_thread.join(timeout=3)
            self._maybe_log_progress(force=True)
            self._print_final_statistics()

    def save_validation_report(self):
        """Guarda un reporte detallado de validación"""
        try:
            timestamp = int(time.time())
            report_path = f"validation_report_{timestamp}.json"
            
            report = {
                'timestamp': timestamp,
                'stats': self.stats.copy(),
                'validation_summary': {
                    'total_validations': len(self.validation_data['mapping_validation_results']),
                    'successful_mappings': sum(1 for r in self.validation_data['mapping_validation_results'] if r['is_valid']),
                    'mapping_errors': sum(len(r['errors']) for r in self.validation_data['mapping_validation_results']),
                    'mapping_warnings': sum(len(r['warnings']) for r in self.validation_data['mapping_validation_results'])
                },
                'mediapipe_to_nvidia_mapping': self.pose_classifier.MEDIAPIPE_TO_NVIDIA_MAPPING,
                'recent_validation_results': self.validation_data['mapping_validation_results'][-10:] if self.validation_data['mapping_validation_results'] else []
            }
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"📊 Reporte de validación guardado: {report_path}")
            
        except Exception as e:
            logger.error(f"❌ Error guardando reporte: {e}")
    
    def _print_final_statistics(self):
        """Imprime estadísticas finales del procesamiento"""
        elapsed_time = time.time() - self.stats['start_time']
        fps = self.stats['frames_processed'] / elapsed_time if elapsed_time > 0 else 0
        
        print("\n" + "="*60)
        print("📊 ESTADÍSTICAS FINALES - MediaPipe + Validación + Clasificación")
        print("="*60)
        print(f"⏱️  Tiempo total: {elapsed_time:.1f} segundos")
        print(f"🎞️  Frames procesados: {self.stats['frames_processed']}")
        print(f"📈 FPS promedio: {fps:.1f}")
        print(f"👥 Personas detectadas: {self.stats['poses_detected']}")
        print(f"🎭 Poses clasificadas: {self.stats['poses_classified']}")
        
        # Estadísticas de validación
        print("\n🔍 VALIDACIÓN DE TOPOLOGÍA:")
        print(f"   📊 Validaciones realizadas: {self.stats['topology_validations']}")
        print(f"   ❌ Errores de mapeo: {self.stats['mapping_errors']}")
        
        if self.stats['topology_validations'] > 0:
            error_rate = (self.stats['mapping_errors'] / self.stats['topology_validations']) * 100
            print(f"   📈 Tasa de errores: {error_rate:.1f}%")
            
            if error_rate < 5:
                print("   ✅ Mapeo de topología funcionando correctamente")
            elif error_rate < 20:
                print("   ⚠️ Algunos errores de mapeo detectados")
            else:
                print("   🔴 Múltiples errores de mapeo - revisar MEDIAPIPE_TO_NVIDIA_MAPPING")
        
        # Estadísticas del clasificador
        classifier_stats = self.pose_classifier.get_statistics()
        print("\n🎭 CLASIFICACIÓN:")
        print(f"   🎯 Tasa de confianza: {classifier_stats['confidence_rate']:.2f}")
        print(f"   🏆 Clase más común: {classifier_stats['most_common_class']}")
        
        print("\n📊 Distribución de clases:")
        for class_name, count in classifier_stats['class_distribution'].items():
            percentage = (count / classifier_stats['total_predictions']) * 100 if classifier_stats['total_predictions'] > 0 else 0
            print(f"   {class_name}: {count} ({percentage:.1f}%)")

    def cleanup(self):
         if getattr(self, '_cleaned', False):
             return
         # Limpieza del clasificador Python 3.6
         try:
             if hasattr(self, 'pose_classifier') and self.pose_classifier:
                 self.pose_classifier.cleanup()
         except Exception as e:
             logger.warning(f"⚠️ Error limpiando clasificador: {e}")
         # Parar worker de clasificación paralelo
         try:
             if self.parallel_classification and self._classification_queue:
                 self._classification_queue.put(None)
         except Exception:
             pass
         # Limpieza de MediaPipe processor
         try:
             if hasattr(self, 'mediapipe_processor'):
                 closer = getattr(self.mediapipe_processor, 'close', None)
                 if callable(closer):
                     closer()
         except Exception as e:
             logger.warning(f"⚠️ Error liberando MediaPipe: {e}")
         self._cleaned = True
         logger.info("✅ Sistema completamente limpiado")
    
    def __del__(self):
        """Destructor"""
        self.cleanup()


def main():
    """Función principal de ejemplo"""
    # Configuración de rutas - AJUSTAR SEGÚN TU SISTEMA
    config = {
        'mediapipe_model': '../models/pose_landmarker_lite.task',
        'pose_classifier_model': '../models/pose_classification/st-gcn_3dbp_nvidia.engine',
        'video_source': '../Videos/Entrada/WIN_20250722_13_47_30_Pro.mp4',  # 0 para cámara
        'output_video': '../Videos/Salida/mediapipe_validation_output.mp4'  # None para no guardar
    }
    
    # Verificar archivos
    required_files = ['mediapipe_model', 'pose_classifier_model']
    for key in required_files:
        if not Path(config[key]).exists():
            print(f"❌ Archivo no encontrado: {config[key]}")
            print("💡 Para usar este ejemplo:")
            print("   1. Asegúrate de que el modelo MediaPipe .task esté disponible")
            print("   2. Descarga PoseClassificationNet de NGC en formato .engine")
            print("   3. Ajusta las rutas en la configuración")
            return False
    
    try:
        # Crear sistema integrado con validación
        print("🔧 Inicializando sistema MediaPipe + Validación + Clasificación...")
        system = MediaPipeWithClassifier(
            mediapipe_model_path=config['mediapipe_model'],
            pose_classifier_model_path=config['pose_classifier_model'],
            validation_mode=True,
            fast_mode=False,            # Cambiar a True para máxima velocidad (desactiva validación)
            parallel_classification=True  # Clasificación en hilo separado
        )
        
        # Procesar video
        system.process_video(
            video_source=config['video_source'],
            output_path=config['output_video']
        )
        
        # Limpiar recursos
        print("🧹 Limpiando recursos del sistema...")
        system.cleanup()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en main: {e}")
        import traceback
        traceback.print_exc()
        
        # Intentar limpiar recursos aún con error
        try:
            if 'system' in locals():
                system.cleanup()
        except Exception:
            pass
            
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        print("❌ Ejecución no exitosa")
    else:
        print("✅ Ejecución completada exitosamente")
