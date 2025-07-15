import torch
import torch2trt
from torch2trt import torch2trt
import trt_pose.coco
import trt_pose.models
import json
import cv2
import numpy as np

def convert_model_to_tensorrt():
    """
    Convierte el modelo PyTorch a TensorRT
    """
    # Rutas
    model_path = "/home/mobilenet/Documentos/Trabajo/trt_pose/models/resnet18_baseline_att_224x224_A_epoch_249.pth"
    topology_path = "/home/mobilenet/Documentos/Trabajo/trt_pose/models/human_pose_18.json"
    output_path = "/home/mobilenet/Documentos/Trabajo/trt_pose/models/resnet18_baseline_att_224x224_A_epoch_249_trt.pth"
    
    # Cargar topología
    with open(topology_path, 'r') as f:
        human_pose = json.load(f)
    
    # Crear el modelo
    topology = trt_pose.coco.coco_category_to_topology(human_pose)
    num_parts = len(human_pose['keypoints'])
    num_links = len(human_pose['skeleton'])
    
    model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
    
    # Cargar los pesos del modelo PyTorch
    model.load_state_dict(torch.load(model_path))
    
    # Crear datos de entrada de ejemplo (tamaño de entrada del modelo)
    WIDTH = 224
    HEIGHT = 224
    data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()
    
    print("Convirtiendo modelo a TensorRT...")
    print("Esto puede tomar varios minutos...")
    
    # Convertir a TensorRT
    model_trt = torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
    
    # Guardar el modelo TensorRT
    torch.save(model_trt.state_dict(), output_path)
    
    print(f"Modelo TensorRT guardado en: {output_path}")
    
    # Verificar que el modelo funciona
    print("Verificando modelo TensorRT...")
    with torch.no_grad():
        # Prueba con el modelo original
        output_torch = model(data)
        
        # Prueba con el modelo TensorRT
        output_trt = model_trt(data)
        
        print(f"Diferencia máxima entre outputs: {torch.max(torch.abs(output_torch - output_trt))}")
        print("¡Conversión exitosa!")
    
    return output_path

if __name__ == "__main__":
    try:
        trt_model_path = convert_model_to_tensorrt()
        print(f"\nAhora puedes usar el modelo TensorRT: {trt_model_path}")
        print("Actualiza tu código para usar esta nueva ruta.")
    except Exception as e:
        print(f"Error durante la conversión: {e}")
        print("Verifica que tengas instalado torch2trt y que las rutas sean correctas.")