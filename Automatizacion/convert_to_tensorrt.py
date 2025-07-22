import torch
import trt_pose.models
from torch2trt import torch2trt

def convert_model_to_tensorrt(pytorch_model_path, tensorrt_model_path, num_parts, num_links, input_width=224, input_height=224):
    """
    Convierte un modelo PyTorch a TensorRT.

    Args:
        pytorch_model_path (str): Ruta al modelo PyTorch (.pth).
        tensorrt_model_path (str): Ruta para guardar el modelo TensorRT (.pth).
        num_parts (int): Número de keypoints (por ejemplo, 18 para COCO).
        num_links (int): Número de conexiones (esqueleto).
        input_width (int): Ancho de la entrada del modelo.
        input_height (int): Altura de la entrada del modelo.
    """
    try:
        print("📥 Cargando modelo PyTorch...")
        # Crear el modelo PyTorch
        model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()

        # Cargar los pesos del modelo
        checkpoint = torch.load(pytorch_model_path)
        model.load_state_dict(checkpoint)
        print("✅ Modelo PyTorch cargado exitosamente.")

        # Crear un tensor de entrada de prueba
        input_tensor = torch.zeros((1, 3, input_height, input_width)).cuda()

        print("⚡ Convirtiendo modelo a TensorRT...")
        # Convertir el modelo a TensorRT
        model_trt = torch2trt(
            model,
            [input_tensor],
            fp16_mode=True,  # Activar FP16 para mayor eficiencia
            max_workspace_size=1 << 22  # Tamaño máximo del workspace (4 MB)
        )
        print("✅ Conversión a TensorRT completada.")

        # Guardar el modelo convertido
        torch.save(model_trt.state_dict(), tensorrt_model_path)
        print(f"✅ Modelo TensorRT guardado en: {tensorrt_model_path}")

        # Verificar inferencia
        print("🧪 Probando inferencia con el modelo TensorRT...")
        output = model_trt(input_tensor)
        print("✅ Inferencia exitosa. Forma de la salida:", output.shape)

    except Exception as e:
        print(f"❌ Error durante la conversión: {e}")

if __name__ == "__main__":
    # Configuración de rutas
    pytorch_model_path = "/home/mobilenet/Documentos/Trabajo/trt_pose/models/resnet18_baseline_att_224x224_A_epoch_249.pth"
    tensorrt_model_path = "/home/mobilenet/Documentos/Trabajo/trt_pose/models/resnet18_baseline_att_224x224_A_epoch_249_trt.pth"


    # Configuración del modelo (COCO)
    num_parts = 21  # Número de keypoints actualizado para coincidir con el modelo
    num_links = 20  # Número de conexiones actualizado para coincidir con el modelo

    # Convertir el modelo
    convert_model_to_tensorrt(pytorch_model_path, tensorrt_model_path, num_parts, num_links)