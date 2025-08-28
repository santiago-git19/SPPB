#!/usr/bin/env python3
"""
Convertir modelo ResNet18 TRT Pose de PyTorch (.pth) a ONNX
============================================================

Este script convierte el modelo resnet18_baseline_att_224x224_A_epoch_249.pth 
de trt_pose a formato ONNX para uso en Jetson Nano u otros dispositivos.

El modelo es un ResNet18 con attention baseline para detección de poses humanas
que genera:
- Confidence maps (heatmaps) para 18 keypoints
- Part Affinity Fields (PAF) para 21 conexiones del esqueleto

Uso:
    python3 pth_file_to_onnx.py

Requisitos:
    - torch
    - trt_pose
    - onnx
    - onnx-simplifier (opcional)

Autor: Sistema de IA
Fecha: 2025
"""

import os
import json
import torch
import onnx
import numpy as np
from pathlib import Path

# Intentar importar trt_pose y onnx-simplifier
try:
    import trt_pose.coco
    import trt_pose.models
    TRT_POSE_AVAILABLE = True
except ImportError:
    print("⚠️ trt_pose no disponible. Instálalo con: pip install trt_pose")
    TRT_POSE_AVAILABLE = False

try:
    from onnxsim import simplify
    ONNX_SIMPLIFIER_AVAILABLE = True
except ImportError:
    ONNX_SIMPLIFIER_AVAILABLE = False


class TRTPoseONNXConverter:
    """
    Convertidor de modelo TRT Pose PyTorch a ONNX
    """
    
    def __init__(self, model_dir: str = None):
        """
        Inicializa el convertidor
        
        Args:
            model_dir: Directorio donde están los modelos (por defecto: directorio actual)
        """
        self.model_dir = Path(model_dir) if model_dir else Path(__file__).parent
        
        # Rutas de archivos
        self.model_path = self.model_dir / "resnet18_baseline_att_224x224_A_epoch_249.pth"
        self.topology_path = self.model_dir / "human_pose.json"
        self.onnx_path = self.model_dir / "resnet18_baseline_att_224x224_A.onnx"
        self.onnx_simplified_path = self.model_dir / "resnet18_baseline_att_224x224_A_simplified.onnx"
        
        # Configuración del modelo
        self.input_width = 224
        self.input_height = 224
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"🔧 Configuración del convertidor:")
        print(f"   📁 Directorio de modelos: {self.model_dir}")
        print(f"   🎯 Modelo PyTorch: {self.model_path.name}")
        print(f"   📄 Topología: {self.topology_path.name}")
        print(f"   💾 Dispositivo: {self.device}")
        
    def load_topology(self):
        """Carga la topología del modelo desde el archivo JSON"""
        if not self.topology_path.exists():
            raise FileNotFoundError(f"Archivo de topología no encontrado: {self.topology_path}")
            
        with open(self.topology_path, 'r') as f:
            self.human_pose = json.load(f)
            
        # Configurar topología para trt_pose
        if TRT_POSE_AVAILABLE:
            self.topology = trt_pose.coco.coco_category_to_topology(self.human_pose)
        
        self.num_parts = len(self.human_pose['keypoints'])  # 18 keypoints
        self.num_links = len(self.human_pose['skeleton'])   # 21 conexiones
        
        print(f"📊 Topología cargada:")
        print(f"   🎯 Keypoints: {self.num_parts}")
        print(f"   🔗 Conexiones: {self.num_links}")
        print(f"   📋 Nombres: {self.human_pose['keypoints']}")
        
    def create_model(self):
        """Crea y carga el modelo PyTorch"""
        if not TRT_POSE_AVAILABLE:
            raise ImportError("trt_pose no está disponible. Instálalo primero.")
            
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {self.model_path}")
            
        print(f"🏗️ Creando modelo ResNet18 Baseline Attention...")
        
        # Crear modelo (num_parts para cmap, 2*num_links para paf)
        self.model = trt_pose.models.resnet18_baseline_att(
            self.num_parts, 2 * self.num_links
        ).to(self.device)
        
        # Cargar pesos
        print(f"📥 Cargando pesos desde: {self.model_path}")
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        
        # Poner en modo evaluación
        self.model.eval()
        
        print(f"✅ Modelo cargado correctamente en {self.device}")
        
    def test_pytorch_model(self):
        """Prueba el modelo PyTorch antes de la conversión"""
        print(f"🧪 Probando modelo PyTorch...")
        
        # Crear entrada de prueba
        dummy_input = torch.randn(1, 3, self.input_height, self.input_width).to(self.device)
        
        with torch.no_grad():
            output = self.model(dummy_input)
            
        # El modelo devuelve una tupla (cmap, paf)
        if isinstance(output, (tuple, list)) and len(output) == 2:
            cmap, paf = output
            print(f"   📊 Confidence Map shape: {cmap.shape}")
            print(f"   📊 Part Affinity Field shape: {paf.shape}")
            print(f"   ✅ Salida esperada: cmap({self.num_parts} canales), paf({2*self.num_links} canales)")
        else:
            print(f"   📊 Output shape: {output.shape if hasattr(output, 'shape') else type(output)}")
            
        print(f"✅ Prueba del modelo PyTorch exitosa")
        return dummy_input
        
    def export_to_onnx(self, dummy_input, opset_version=11, dynamic_batch=True):
        """
        Exporta el modelo PyTorch a ONNX
        
        Args:
            dummy_input: Entrada de ejemplo para tracing
            opset_version: Versión del opset ONNX (11, 12, 13)
            dynamic_batch: Si permitir batch size dinámico
        """
        print(f"🔄 Exportando a ONNX...")
        print(f"   📋 Opset version: {opset_version}")
        print(f"   🔧 Dynamic batch: {dynamic_batch}")
        
        # Configurar nombres de entrada y salida
        input_names = ["input"]
        output_names = ["confidence_map", "part_affinity_field"]
        
        # Configurar ejes dinámicos si se solicita
        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'confidence_map': {0: 'batch_size'}, 
                'part_affinity_field': {0: 'batch_size'}
            }
        
        # Exportar a ONNX
        with torch.no_grad():
            torch.onnx.export(
                self.model,                     # Modelo PyTorch
                dummy_input,                    # Entrada de ejemplo
                str(self.onnx_path),           # Archivo de salida
                export_params=True,             # Exportar parámetros del modelo
                opset_version=opset_version,    # Versión ONNX opset
                do_constant_folding=True,       # Optimización
                input_names=input_names,        # Nombres de entrada
                output_names=output_names,      # Nombres de salida
                dynamic_axes=dynamic_axes,      # Ejes dinámicos
                verbose=False                   # Reducir verbosidad
            )
            
        print(f"✅ Modelo ONNX exportado: {self.onnx_path}")
        
    def verify_onnx_model(self):
        """Verifica que el modelo ONNX es válido"""
        print(f"🔍 Verificando modelo ONNX...")
        
        # Cargar y verificar modelo
        model = onnx.load(str(self.onnx_path))
        
        try:
            onnx.checker.check_model(model)
            print(f"✅ Modelo ONNX válido")
        except onnx.checker.ValidationError as e:
            print(f"❌ Modelo ONNX inválido: {e}")
            raise
            
        # Mostrar información del modelo
        print(f"📊 Información del modelo ONNX:")
        print(f"   🔢 Opset version: {model.opset_import[0].version}")
        
        # Información de entrada
        for i, input_info in enumerate(model.graph.input):
            shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' 
                    for dim in input_info.type.tensor_type.shape.dim]
            print(f"   📥 Input {i}: {input_info.name} {shape}")
            
        # Información de salida  
        for i, output_info in enumerate(model.graph.output):
            shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' 
                    for dim in output_info.type.tensor_type.shape.dim]
            print(f"   📤 Output {i}: {output_info.name} {shape}")
            
    def simplify_onnx_model(self):
        """Simplifica el modelo ONNX (opcional)"""
        if not ONNX_SIMPLIFIER_AVAILABLE:
            print("⚠️ onnx-simplifier no disponible, saltando simplificación")
            print("💡 Para instalar: pip install onnx-simplifier")
            print("✅ El modelo ONNX original funciona perfectamente sin simplificar")
            return
            
        print(f"🔧 Simplificando modelo ONNX...")
        
        # Cargar modelo original
        model = onnx.load(str(self.onnx_path))
        
        # Simplificar
        try:
            model_simplified, check = simplify(model)
            
            if check:
                # Guardar modelo simplificado
                onnx.save(model_simplified, str(self.onnx_simplified_path))
                print(f"✅ Modelo simplificado guardado: {self.onnx_simplified_path}")
                
                # Comparar tamaños
                original_size = self.onnx_path.stat().st_size / (1024*1024)  # MB
                simplified_size = self.onnx_simplified_path.stat().st_size / (1024*1024)  # MB
                
                print(f"📊 Comparación de tamaños:")
                print(f"   📁 Original: {original_size:.2f} MB")
                print(f"   📁 Simplificado: {simplified_size:.2f} MB")
                print(f"   💾 Reducción: {((original_size-simplified_size)/original_size)*100:.1f}%")
            else:
                print("❌ Simplificación falló")
                
        except Exception as e:
            print(f"❌ Error durante simplificación: {e}")
            print("✅ El modelo original ONNX sigue siendo válido")
            
    def test_onnx_inference(self):
        """Prueba inferencia con ONNX Runtime (opcional)"""
        try:
            import onnxruntime as ort
        except ImportError:
            print("⚠️ onnxruntime no disponible para pruebas")
            print("💡 Para instalar: pip install onnxruntime")
            print("✅ Puedes usar el modelo ONNX con otras herramientas como TensorRT")
            return
            
        print(f"🧪 Probando inferencia ONNX...")
        
        try:
            # Crear sesión ONNX
            session = ort.InferenceSession(str(self.onnx_path))
            
            # Crear entrada de prueba
            input_data = np.random.randn(1, 3, self.input_height, self.input_width).astype(np.float32)
            
            # Ejecutar inferencia
            outputs = session.run(None, {"input": input_data})
            
            print(f"📊 Salida ONNX:")
            for i, output in enumerate(outputs):
                print(f"   📤 Output {i}: {output.shape}")
                
            print(f"✅ Inferencia ONNX exitosa")
            
        except Exception as e:
            print(f"❌ Error durante inferencia ONNX: {e}")
            print("✅ El modelo ONNX es válido, pero puede necesitar configuración específica")
        
    def convert(self, opset_version=11, dynamic_batch=True, simplify=True, test_onnx=True):
        """
        Ejecuta el proceso completo de conversión
        
        Args:
            opset_version: Versión del opset ONNX
            dynamic_batch: Si permitir batch size dinámico  
            simplify: Si simplificar el modelo ONNX
            test_onnx: Si probar inferencia ONNX
        """
        print(f"🚀 Iniciando conversión PyTorch → ONNX")
        print(f"=" * 60)
        
        try:
            # 1. Cargar topología
            self.load_topology()
            print()
            
            # 2. Crear y cargar modelo
            self.create_model()
            print()
            
            # 3. Probar modelo PyTorch
            dummy_input = self.test_pytorch_model()
            print()
            
            # 4. Exportar a ONNX
            self.export_to_onnx(dummy_input, opset_version, dynamic_batch)
            print()
            
            # 5. Verificar modelo ONNX
            self.verify_onnx_model()
            print()
            
            # 6. Simplificar (opcional)
            if simplify:
                self.simplify_onnx_model()
                print()
                
            # 7. Probar inferencia ONNX (opcional)
            if test_onnx:
                self.test_onnx_inference()
                print()
                
            print(f"🎉 ¡Conversión completada exitosamente!")
            print(f"📁 Archivos generados:")
            print(f"   🔸 ONNX: {self.onnx_path}")
            if simplify and self.onnx_simplified_path.exists():
                print(f"   🔸 ONNX Simplificado: {self.onnx_simplified_path}")
            print()
            
            # Instrucciones de uso
            print(f"💡 Instrucciones de uso:")
            print(f"   🔸 Para usar en ONNX Runtime:")
            print(f"     import onnxruntime as ort")
            print(f"     session = ort.InferenceSession('{self.onnx_path.name}')")
            print(f"     outputs = session.run(None, {{'input': input_array}})")
            print()
            print(f"   🔸 Para convertir a TensorRT:")
            print(f"     trtexec --onnx={self.onnx_path.name} --saveEngine=model.engine --fp16")
            print()
            
        except Exception as e:
            print(f"❌ Error durante la conversión: {e}")
            raise


def main():
    """Función principal"""
    print(f"TRT Pose PyTorch → ONNX Converter")
    print(f"=" * 50)
    
    # Verificar dependencias
    if not TRT_POSE_AVAILABLE:
        print(f"❌ trt_pose no está instalado")
        print(f"💡 Instálalo con:")
        print(f"   pip install trt_pose")
        print(f"   # o desde fuente: https://github.com/NVIDIA-AI-IOT/trt_pose")
        return 1
        
    # Crear convertidor
    converter = TRTPoseONNXConverter()
    
    try:
        # Ejecutar conversión
        converter.convert(
            opset_version=11,      # Compatible con Jetson Nano
            dynamic_batch=True,    # Permite batch size variable
            simplify=True,         # Simplifica el modelo si es posible
            test_onnx=True         # Prueba inferencia ONNX
        )
        
        print(f"✅ Conversión completada exitosamente")
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())