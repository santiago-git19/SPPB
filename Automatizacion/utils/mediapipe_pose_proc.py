#!/usr/bin/env python3
"""
MediaPipe BlazePose TensorRT Processor (corregido)
-------------------------------------------------

Clase para ejecutar un engine TensorRT (e.g. pose_landmark_lite_fp16.engine)
y obtener los 33 keypoints (x, y, confidence) por frame.

Principales correcciones respecto a versiones ingenuas:
 - Manejo de múltiples bindings (inputs/outputs)
 - Detección de layout (NHWC vs NCHW) y dtype (FP16/FP32)
 - Asignación de buffers por binding (GPU) y host outputs
 - Soporte para shapes dinámicas (context.set_binding_shape)
 - Selección robusta del output que contiene landmarks
 - Normalización / mapeo de coordenadas con advertencias sobre ROI
 - Limpieza y sincronización correcta de CUDA

Notas:
 - MediaPipe BlazePose puede utilizar ROI/crops: si tu .engine espera una ROI,
   deberás replicar la lógica del detector+crop para mapear correctamente las coordenadas
   a la imagen original. Aquí el código asume que el modelo devuelve coordenadas
   normalizadas [0,1] respecto a la imagen de entrada (o píxeles si así lo configuró).
"""

from typing import Optional, Tuple, List, Dict
import os
import time
import logging

import numpy as np
import cv2

# TensorRT + PyCUDA (Jetson)
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # inicializa CUDA automáticamente
    TRT_AVAILABLE = True
except Exception as e:
    TRT_AVAILABLE = False
    trt = None
    cuda = None

logger = logging.getLogger("MediaPipePoseTRT")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class MediaPipePoseProcessor:
    KEYPOINT_NAMES = [
        'nose','right_eye_inner','right_eye','right_eye_outer',
        'left_eye_inner','left_eye','left_eye_outer','right_ear','left_ear',
        'mouth_right','mouth_left','right_shoulder','left_shoulder','right_elbow',
        'left_elbow','right_wrist','left_wrist','right_pinky_knuckle','left_pinky_knuckle',
        'right_index_knuckle','left_index_knuckle','right_thumb_knuckle','left_thumb_knuckle',
        'right_hip','left_hip','right_knee','left_knee','right_ankle','left_ankle',
        'right_heel','left_heel','right_foot_index','left_foot_index'
    ]

    POSE_CONNECTIONS = [
        (0,1),(1,2),(2,3),(0,4),(4,5),(5,6),(0,9),(0,10),(9,10),(2,7),(5,8),
        (11,13),(13,15),(15,17),(15,19),(15,21),
        (12,14),(14,16),(16,18),(16,20),(16,22),
        (11,12),(11,23),(12,24),(23,24),
        (23,25),(25,27),(27,29),(27,31),
        (24,26),(26,28),(28,30),(28,32)
    ]

    def __init__(self,
                 model_path: str = "pose_landmark_lite_fp16.engine",
                 input_width: int = 256,
                 input_height: int = 256,
                 confidence_threshold: float = 0.3):
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT / PyCUDA no disponibles. Instale tensorrt y pycuda en el Jetson.")

        self.model_path = model_path
        self.input_width = input_width
        self.input_height = input_height
        self.confidence_threshold = confidence_threshold

        # TensorRT vars
        self.runtime: Optional[trt.Runtime] = None
        self.engine: Optional[trt.ICudaEngine] = None
        self.context: Optional[trt.IExecutionContext] = None

        # Binding metadata
        self.num_bindings: int = 0
        self.input_bindings: List[int] = []
        self.output_bindings: List[int] = []
        self.binding_shapes: Dict[int, Tuple[int, ...]] = {}
        self.binding_dtypes: Dict[int, np.dtype] = {}

        # Device and host buffers
        self.d_bindings: List[Optional[cuda.DeviceAllocation]] = []
        self.host_outputs: Dict[int, np.ndarray] = {}

        # stream
        self.stream: Optional[cuda.Stream] = None

        # Preprocessing layout detection
        self.input_layout: str = "UNKNOWN"  # 'NHWC' or 'NCHW'

        # load model and allocate buffers
        self._load_engine_and_alloc_buffers()

        logger.info(f"MediaPipePoseProcessor initialized with model={os.path.basename(self.model_path)}, "
                    f"input={self.input_width}x{self.input_height}, conf_thresh={self.confidence_threshold}")

    # --------------------------
    # Engine load & buffer alloc
    # --------------------------
    def _load_engine_and_alloc_buffers(self):
        # load engine
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Engine file not found: {self.model_path}")

        logger.info(f"Cargando engine TensorRT: {self.model_path}")
        trt_logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(trt_logger)

        with open(self.model_path, "rb") as f:
            engine_data = f.read()

        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError("Failed to deserialize engine")

        self.context = self.engine.create_execution_context()
        self.num_bindings = self.engine.num_bindings

        # collect binding info
        for i in range(self.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = tuple(self.engine.get_binding_shape(i))  # may include -1
            dtype_trt = self.engine.get_binding_dtype(i)  # trt.DataType
            try:
                nptype = trt.nptype(dtype_trt)
            except Exception:
                # fallback
                nptype = np.float32

            self.binding_shapes[i] = shape
            self.binding_dtypes[i] = np.dtype(nptype)

            if self.engine.binding_is_input(i):
                self.input_bindings.append(i)
            else:
                self.output_bindings.append(i)

            logger.debug(f"Binding {i}: name='{name}', shape={shape}, dtype={self.binding_dtypes[i]}, is_input={self.engine.binding_is_input(i)}")

        if len(self.input_bindings) == 0:
            raise RuntimeError("Engine does not have input bindings")

        # If any binding shape is dynamic (-1), set a concrete shape using input_width/height
        # We assume batch=1 for inference here.
        # Try to infer NHWC vs NCHW from pattern of shape
        # We'll choose first input binding as the primary input to set shape.
        input_idx = self.input_bindings[0]
        input_shape = list(self.binding_shapes[input_idx])

        # Decide input layout if possible
        # If there is a 3 in channel position check common patterns.
        if len(input_shape) == 4:
            # possible NCHW: (N, C, H, W) where C==3
            if input_shape[1] == 3 or input_shape[1] == -1:
                self.input_layout = "NCHW"
            # possible NHWC: (N, H, W, C) where C==3
            elif input_shape[-1] == 3 or input_shape[-1] == -1:
                self.input_layout = "NHWC"

        # If any -1, set binding shape for input accordingly
        need_set = any((d is None or (isinstance(d, int) and d < 0)) for d in input_shape)
        if need_set:
            # build a concrete shape with batch=1 and provided width/height
            if self.input_layout == "NHWC":
                concrete = (1, self.input_height, self.input_width, 3)
            elif self.input_layout == "NCHW":
                concrete = (1, 3, self.input_height, self.input_width)
            else:
                # fallback to NHWC
                concrete = (1, self.input_height, self.input_width, 3)
                self.input_layout = "NHWC"
            try:
                self.context.set_binding_shape(input_idx, concrete)
                # refresh shapes after set
                for i in range(self.num_bindings):
                    self.binding_shapes[i] = tuple(self.context.get_binding_shape(i))
                logger.info(f"Set concrete input binding shape {concrete} (input layout {self.input_layout})")
            except Exception as e:
                logger.warning(f"Could not set binding shape for input {input_idx}: {e}")

        # allocate device memory for each binding and host outputs for outputs
        self.d_bindings = [None] * self.num_bindings
        self.host_outputs = {}

        for i in range(self.num_bindings):
            shape = self.binding_shapes[i]
            nptype = self.binding_dtypes[i]
            if any((d is None or (isinstance(d, int) and d < 0)) for d in shape):
                raise RuntimeError(f"Binding {i} has unresolved dynamic shape: {shape}. You must set a concrete shape.")

            count = int(np.prod(shape))
            nbytes = int(count * np.dtype(nptype).itemsize)
            # allocate device memory
            self.d_bindings[i] = cuda.mem_alloc(nbytes)
            # allocate host buffer only for outputs
            if i in self.output_bindings:
                host_buf = np.empty(shape, dtype=nptype)
                self.host_outputs[i] = host_buf
            logger.debug(f"Allocated binding {i}: shape={shape}, dtype={nptype}, bytes={nbytes}")

        # create stream
        self.stream = cuda.Stream()

        # log summary
        logger.info(f"Engine loaded: num_bindings={self.num_bindings}, inputs={self.input_bindings}, outputs={self.output_bindings}, layout={self.input_layout}")

    # --------------------------
    # Pre/post processing
    # --------------------------
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for the selected input binding.
        Returns contiguous numpy array ready to copy to device memory.
        NOTE: Normalization here is simple [0,1]. If your converted model expects other
        normalization (e.g. mean/std or [-1,1]), change accordingly.
        """
        # resize then convert
        resized = cv2.resize(frame, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        input_idx = self.input_bindings[0]
        target_dtype = self.binding_dtypes[input_idx]

        arr = rgb.astype(target_dtype, copy=False) / (255.0 if np.issubdtype(target_dtype, np.floating) else 1.0)

        # layout handling
        if self.input_layout == "NCHW":
            arr = arr.transpose(2, 0, 1)  # HWC -> CHW

        batched = np.expand_dims(arr, axis=0).astype(target_dtype, copy=False)
        batched = np.ascontiguousarray(batched)
        return batched

    def _postprocess_keypoints(self, landmarks_arr: np.ndarray, original_width: int, original_height: int) -> np.ndarray:
        """
        landmarks_arr: 1D or flattened array that contains at least 33*3 floats
        Expectation: coordinates are normalized [0,1] for x,y relative to input image
        and confidence in [0,1]. If your model produces pixel coords, do not rescale.
        """
        flat = landmarks_arr.ravel().astype(np.float32)
        if flat.size < 33 * 3:
            raise ValueError(f"Landmarks array too small: {flat.size} < 99")

        lm99 = flat[:33 * 3]
        keypoints = lm99.reshape((33, 3))  # (x, y, score)

        # If x,y seem to be normalized (0..1) map to original image; if >1, assume already pixels
        if np.nanmin(keypoints[:, :2]) >= 0.0 and np.nanmax(keypoints[:, :2]) <= 1.0:
            keypoints[:, 0] = keypoints[:, 0] * original_width
            keypoints[:, 1] = keypoints[:, 1] * original_height
        else:
            # assume coordinates are already in pixels (no-op)
            pass

        # apply confidence threshold
        mask = keypoints[:, 2] < self.confidence_threshold
        keypoints[mask] = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        return keypoints.astype(np.float32)

    # --------------------------
    # Inference
    # --------------------------
    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Run inference on one BGR frame and return an array [33,3] of (x,y,conf) or None.
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty frame")
            return None

        original_h, original_w = frame.shape[:2]

        try:
            # preprocess
            input_data = self._preprocess_frame(frame)
            input_idx = self.input_bindings[0]
            # copy input to device
            cuda.memcpy_htod_async(self.d_bindings[input_idx], input_data, self.stream)

            # prepare ordered bindings list
            bindings_ptrs = [int(self.d_bindings[i]) for i in range(self.num_bindings)]

            # execute
            self.context.execute_async_v2(bindings_ptrs, self.stream.handle)

            # copy outputs back to host
            for out_idx in self.output_bindings:
                cuda.memcpy_dtoh_async(self.host_outputs[out_idx], self.d_bindings[out_idx], self.stream)

            self.stream.synchronize()

            # find landmarks binding heuristically
            landmarks_binding = self._select_landmarks_binding()

            if landmarks_binding is None:
                logger.error("Could not find landmarks binding among outputs")
                return None

            landmarks_host = self.host_outputs[landmarks_binding]
            keypoints = self._postprocess_keypoints(landmarks_host, original_w, original_h)

            return keypoints

        except Exception as e:
            logger.exception(f"Inference error: {e}")
            return None

    def _select_landmarks_binding(self) -> Optional[int]:
        """
        Heuristics to pick which output binding contains the flat landmarks.
        Prefer an output whose flattened size >= 99 and whose name contains
        'landmark' or 'lm' or matches typical sizes (e.g. 195).
        """
        candidate = None
        candidate_size = -1

        for out_idx in self.output_bindings:
            shape = self.binding_shapes[out_idx]
            size = int(np.prod(shape))
            name = self.engine.get_binding_name(out_idx).lower()
            # prefer name hints
            if ("landmark" in name) or ("lm" in name) or ("landmarks" in name):
                if size >= 99:
                    logger.debug(f"Selecting binding {out_idx} by name='{name}' size={size}")
                    return out_idx
            # fallback choose largest candidate with enough size
            if size >= 99 and size > candidate_size:
                candidate = out_idx
                candidate_size = size

        if candidate is not None:
            logger.debug(f"Selecting binding {candidate} by largest size {candidate_size}")
            return candidate

        # final fallback: if there's exactly one output, return it
        if len(self.output_bindings) == 1:
            return self.output_bindings[0]

        return None

    # --------------------------
    # Visualization helpers
    # --------------------------
    def visualize_keypoints(self,
                            frame: np.ndarray,
                            keypoints: Optional[np.ndarray] = None,
                            draw_landmarks: bool = True,
                            draw_connections: bool = True,
                            draw_labels: bool = False,
                            confidence_threshold: float = 0.1) -> np.ndarray:
        if frame is None or frame.size == 0:
            return frame

        if keypoints is None:
            keypoints = self.process_frame(frame)
            if keypoints is None:
                return frame

        out = frame.copy()

        # simple color mapping
        color_map = {
            'face': (255, 255, 255),
            'right_arm': (255, 0, 0),
            'left_arm': (0, 255, 0),
            'torso': (0, 255, 255),
            'right_leg': (255, 255, 0),
            'left_leg': (255, 0, 255)
        }

        # draw landmarks
        if draw_landmarks:
            for i, (x, y, conf) in enumerate(keypoints):
                if conf > confidence_threshold:
                    cv2.circle(out, (int(x), int(y)), 3, (0, 0, 255), -1)   # red filled
                    cv2.circle(out, (int(x), int(y)), 5, (255, 255, 255), 1) # white outline
                    if draw_labels and i < len(self.KEYPOINT_NAMES):
                        label = f"{self.KEYPOINT_NAMES[i]}:{conf:.2f}"
                        cv2.putText(out, label, (int(x) + 4, int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

        if draw_connections:
            for a, b in self.POSE_CONNECTIONS:
                if a < len(keypoints) and b < len(keypoints):
                    x1, y1, c1 = keypoints[a]
                    x2, y2, c2 = keypoints[b]
                    if c1 > confidence_threshold and c2 > confidence_threshold:
                        cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        return out

    # --------------------------
    # Utilities & cleanup
    # --------------------------
    def cleanup(self):
        # synchronize stream before freeing
        try:
            if self.stream is not None:
                try:
                    self.stream.synchronize()
                except Exception:
                    pass
            # free device memory
            for mem in getattr(self, "d_bindings", []) or []:
                try:
                    if mem is not None:
                        mem.free()
                except Exception:
                    pass
            self.d_bindings = []
            self.host_outputs = {}
            # drop context/engine/runtime refs
            self.context = None
            self.engine = None
            self.runtime = None
            logger.info("Resources freed")
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass

    # Diagnostics helper (lightweight)
    def inspect_bindings(self):
        """
        Print binding names, shapes and dtypes for debugging.
        """
        print("TensorRT Engine bindings:")
        for i in range(self.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = self.binding_shapes.get(i, None)
            dtype = self.binding_dtypes.get(i, None)
            role = "INPUT" if i in self.input_bindings else "OUTPUT"
            print(f"  idx={i} role={role} name='{name}' shape={shape} dtype={dtype}")

# --------------------------
# Example usage (main)
# --------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=str, default="pose_landmark_lite_fp16.engine", help="Path to .engine file")
    parser.add_argument("--video", type=str, default=None, help="Video file or camera index (0)")
    parser.add_argument("--out", type=str, default=None, help="Output video path")
    parser.add_argument("--show", action="store_true", help="Show live window")
    args = parser.parse_args()

    if not TRT_AVAILABLE:
        print("TensorRT/PyCUDA no disponibles. Ejecute en Jetson con TensorRT instalado.")
        exit(1)

    if not os.path.exists(args.engine):
        print(f"Engine not found: {args.engine}")
        exit(1)

    processor = MediaPipePoseProcessor(model_path=args.engine, input_width=256, input_height=256, confidence_threshold=0.3)
    processor.inspect_bindings()

    # capture source
    cap = None
    if args.video is None:
        cap = cv2.VideoCapture(0)
    else:
        # accept numeric camera index string
        try:
            cam_idx = int(args.video)
            cap = cv2.VideoCapture(cam_idx)
        except Exception:
            cap = cv2.VideoCapture(args.video)

    if not cap or not cap.isOpened():
        print("Cannot open video source")
        exit(1)

    writer = None
    if args.out:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out, fourcc, fps, (w, h))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            start = time.time()
            keypoints = processor.process_frame(frame)
            elapsed = (time.time() - start) * 1000.0
            if keypoints is not None:
                vis = processor.visualize_keypoints(frame, keypoints, draw_labels=False)
                cv2.putText(vis, f"{elapsed:.1f}ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                vis = frame

            if writer:
                writer.write(vis)
            if args.show:
                cv2.imshow("PoseTRT", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        processor.cleanup()
