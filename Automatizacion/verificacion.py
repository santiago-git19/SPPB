import os
import json
import torch
import trt_pose.coco
import trt_pose.models
from torch2trt import torch2trt, TRTModule

# Configuración de rutas
TOPOLOGY_FILE = '/home/mobilenet/Documentos/Trabajo/trt_pose/tasks/human_pose/human_pose.json'
MODEL_WEIGHTS = '/home/mobilenet/Documentos/Trabajo/trt_pose/models/resnet18_baseline_att_224x224_A_epoch_249.pth'
OPTIMIZED_MODEL = '/home/mobilenet/Documentos/Trabajo/trt_pose/models/resnet18_baseline_att_224x224_A_epoch_249_trt.pth'

WIDTH = 224
HEIGHT = 224

def verificar_topologia():
    """Verifica el archivo de topología"""
    assert os.path.exists(TOPOLOGY_FILE), "❌ Archivo de topología no encontrado"
    with open(TOPOLOGY_FILE, 'r') as f:
        human_pose = json.load(f)
    print(f"✅ Archivo de topología cargado correctamente")
    print(f"Keypoints: {human_pose['keypoints']}")
    print(f"Skeleton: {human_pose['skeleton']}")
    return human_pose

def verificar_pesos(model, model_weights):
    """Verifica los pesos del modelo"""
    assert os.path.exists(model_weights), "❌ Archivo de pesos no encontrado"
    try:
        model.load_state_dict(torch.load(model_weights))
        print("✅ Pesos cargados correctamente")
    except Exception as e:
        print(f"❌ Error al cargar los pesos: {e}")

def verificar_inferencia_pytorch(model):
    """Verifica la inferencia en PyTorch"""
    data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()
    try:
        output = model(data)
        print(f"✅ Inferencia en PyTorch exitosa. Salida: {output.shape}")
    except Exception as e:
        print(f"❌ Error en la inferencia de PyTorch: {e}")

def verificar_conversion_tensorrt(model):
    """Verifica la conversión a TensorRT"""
    data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()
    try:
        model_trt = torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
        print("✅ Conversión a TensorRT exitosa")
        return model_trt
    except Exception as e:
        print(f"❌ Error durante la conversión a TensorRT: {e}")
        return None

def verificar_inferencia_tensorrt(model_trt):
    """Verifica la inferencia en TensorRT"""
    data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()
    try:
        output_trt = model_trt(data)
        print(f"✅ Inferencia en TensorRT exitosa. Salida: {output_trt.shape}")
    except Exception as e:
        print(f"❌ Error en la inferencia de TensorRT: {e}")

def guardar_modelo_tensorrt(model_trt, optimized_model):
    """Guarda el modelo convertido a TensorRT"""
    try:
        torch.save(model_trt.state_dict(), optimized_model)
        print(f"✅ Modelo TensorRT guardado en: {optimized_model}")
    except Exception as e:
        print(f"❌ Error al guardar el modelo TensorRT: {e}")

def main():
    # Verificar topología
    human_pose = verificar_topologia()

    # Configurar modelo
    num_parts = len(human_pose['keypoints'])
    num_links = len(human_pose['skeleton'])
    model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()

    # Verificar pesos
    verificar_pesos(model, MODEL_WEIGHTS)

    # Verificar inferencia en PyTorch
    verificar_inferencia_pytorch(model)

    # Verificar conversión a TensorRT
    model_trt = verificar_conversion_tensorrt(model)

    if model_trt:
        # Verificar inferencia en TensorRT
        verificar_inferencia_tensorrt(model_trt)

        # Guardar modelo convertido
        guardar_modelo_tensorrt(model_trt, OPTIMIZED_MODEL)

if __name__ == "__main__":
    main()