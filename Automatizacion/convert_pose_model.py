#!/usr/bin/env python3
"""
Script para convertir un modelo PoseClassificationNet de PyTorch a TensorRT.
Este script muestra cómo preparar el modelo para uso con PoseClassifier.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import tensorrt as trt
from torch2trt import torch2trt

class PoseClassificationNet(nn.Module):
    """
    Red neuronal de ejemplo para clasificación de poses.
    Esta es una implementación básica que debes adaptar según tu modelo real.
    """
    
    def __init__(self, num_keypoints=17, num_classes=6):
        super(PoseClassificationNet, self).__init__()
        
        # Entrada: keypoints (num_keypoints * 2 para x,y coordinates)
        input_size = num_keypoints * 2
        
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        return self.classifier(x)

def create_sample_model():
    """
    Crea un modelo de ejemplo para demostración.
    """
    print("Creando modelo de ejemplo...")
    
    # Crear modelo
    model = PoseClassificationNet(num_keypoints=17, num_classes=6)
    
    # Inicializar con pesos aleatorios (en la práctica, cargarías pesos entrenados)
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    return model

def pytorch_to_onnx(model, input_shape, onnx_path):
    """
    Convierte modelo PyTorch a ONNX.
    """
    print(f"Convirtiendo modelo PyTorch a ONNX: {onnx_path}")
    
    # Crear input dummy
    dummy_input = torch.randn(1, *input_shape)
    
    # Exportar a ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Modelo ONNX guardado en: {onnx_path}")

def onnx_to_tensorrt(onnx_path, engine_path, max_batch_size=1):
    """
    Convierte modelo ONNX a TensorRT engine.
    """
    print(f"Convirtiendo ONNX a TensorRT: {engine_path}")
    
    # Crear logger
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    # Crear builder
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    
    # Configurar memoria de trabajo
    config.max_workspace_size = 1 << 30  # 1GB
    
    # Crear red desde ONNX
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parsear ONNX
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Falló el parseo del modelo ONNX")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False
    
    # Construir engine
    print("Construyendo engine TensorRT...")
    engine = builder.build_engine(network, config)
    
    if engine is None:
        print("ERROR: No se pudo construir el engine")
        return False
    
    # Guardar engine
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    
    print(f"Engine TensorRT guardado en: {engine_path}")
    return True

def pytorch_to_tensorrt_direct(model, input_shape, engine_path):
    """
    Convierte directamente de PyTorch a TensorRT usando torch2trt.
    """
    print(f"Convirtiendo PyTorch a TensorRT (directo): {engine_path}")
    
    # Crear input dummy
    dummy_input = torch.randn(1, *input_shape).cuda()
    
    # Poner modelo en modo evaluación y mover a GPU
    model.eval()
    model.cuda()
    
    # Convertir a TensorRT
    model_trt = torch2trt(model, [dummy_input])
    
    # Guardar modelo TensorRT
    torch.save(model_trt.state_dict(), engine_path)
    
    print(f"Modelo TensorRT guardado en: {engine_path}")

def test_model_conversion(model_path, input_shape):
    """
    Prueba la conversión del modelo.
    """
    print("Probando conversión del modelo...")
    
    # Cargar modelo
    if os.path.exists(model_path):
        model = PoseClassificationNet()
        model.load_state_dict(torch.load(model_path))
    else:
        print("Modelo no encontrado, creando modelo de ejemplo...")
        model = create_sample_model()
    
    model.eval()
    
    # Crear datos de prueba
    test_input = torch.randn(1, *input_shape)
    
    # Probar inferencia PyTorch
    with torch.no_grad():
        pytorch_output = model(test_input)
    
    print(f"Salida PyTorch: {pytorch_output.shape}")
    print(f"Valores: {pytorch_output.squeeze().tolist()}")
    
    return model

def main():
    """
    Función principal para convertir modelos.
    """
    print("=== Convertidor de Modelos PoseClassificationNet ===")
    
    # Configuración
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Rutas de archivos
    pytorch_model_path = os.path.join(model_dir, "pose_classification_pytorch.pth")
    onnx_model_path = os.path.join(model_dir, "pose_classification.onnx")
    trt_engine_path = os.path.join(model_dir, "pose_classification.engine")
    
    # Configuración del modelo
    num_keypoints = 17  # COCO format
    input_shape = (num_keypoints * 2,)  # x,y coordinates
    
    try:
        # Paso 1: Crear/cargar modelo PyTorch
        print("\n1. Preparando modelo PyTorch...")
        model = test_model_conversion(pytorch_model_path, input_shape)
        
        # Guardar modelo PyTorch si no existe
        if not os.path.exists(pytorch_model_path):
            torch.save(model.state_dict(), pytorch_model_path)
            print(f"Modelo PyTorch guardado en: {pytorch_model_path}")
        
        # Paso 2: Convertir a ONNX
        print("\n2. Convirtiendo a ONNX...")
        pytorch_to_onnx(model, input_shape, onnx_model_path)
        
        # Paso 3: Convertir a TensorRT
        print("\n3. Convirtiendo a TensorRT...")
        
        # Opción 1: ONNX -> TensorRT (recomendado)
        if onnx_to_tensorrt(onnx_model_path, trt_engine_path):
            print("✓ Conversión ONNX -> TensorRT exitosa")
        else:
            print("✗ Conversión ONNX -> TensorRT falló")
            
            # Opción 2: PyTorch -> TensorRT directo (alternativa)
            print("Intentando conversión directa PyTorch -> TensorRT...")
            try:
                pytorch_to_tensorrt_direct(model, input_shape, trt_engine_path.replace('.engine', '.pth'))
                print("✓ Conversión directa PyTorch -> TensorRT exitosa")
            except Exception as e:
                print(f"✗ Conversión directa falló: {e}")
        
        print("\n=== Conversión Completada ===")
        print("Archivos generados:")
        print(f"  - Modelo PyTorch: {pytorch_model_path}")
        print(f"  - Modelo ONNX: {onnx_model_path}")
        print(f"  - Engine TensorRT: {trt_engine_path}")
        
        print("\nPara usar el modelo convertido:")
        print("  python example_pose_classification.py video.mp4 --classification-engine models/pose_classification.engine")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

def create_training_data_example():
    """
    Ejemplo de cómo crear datos de entrenamiento para PoseClassificationNet.
    """
    print("=== Ejemplo de Datos de Entrenamiento ===")
    
    # Ejemplo de estructura de datos
    example_data = {
        'keypoints': [
            # Formato: [x1, y1, x2, y2, ..., x17, y17] para 17 keypoints COCO
            [100, 50, 110, 45, 90, 45, 120, 40, 80, 40, 150, 80, 50, 80, 
             180, 120, 20, 120, 200, 160, 10, 160, 140, 200, 60, 200, 
             130, 250, 70, 250, 120, 300, 80, 300],
        ],
        'labels': [
            0,  # 0: de_pie, 1: sentado, 2: levantandose, 3: caminando, 4: equilibrio, 5: desconocido
        ],
        'classes': {
            0: "de_pie",
            1: "sentado", 
            2: "levantandose",
            3: "caminando",
            4: "equilibrio",
            5: "desconocido"
        }
    }
    
    print("Estructura de datos de ejemplo:")
    print(f"  - Keypoints: {len(example_data['keypoints'][0])} valores (17 keypoints × 2 coordenadas)")
    print(f"  - Labels: {example_data['labels']}")
    print(f"  - Classes: {example_data['classes']}")
    
    print("\\nPara entrenar tu modelo:")
    print("1. Recolecta videos con poses etiquetadas")
    print("2. Extrae keypoints usando TensorRT Pose")
    print("3. Etiqueta manualmente las poses")
    print("4. Entrena PoseClassificationNet")
    print("5. Convierte el modelo entrenado usando este script")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--training-example":
        create_training_data_example()
    else:
        sys.exit(main())
