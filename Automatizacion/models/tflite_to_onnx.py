import tf2onnx
import tensorflow as tf

# Rutas
tflite_model_path = "pose_detection.tflite"
onnx_model_path = "pose_detection.onnx"

# Convertir directamente desde TFLite
# (Esta API de tf2onnx es más reciente y evita 'from_function')
spec = tf2onnx.convert.from_tflite(
    tflite_model_path,
    output_path=onnx_model_path,
    opset=13
)

print(f"Modelo convertido y guardado en {onnx_model_path}")
