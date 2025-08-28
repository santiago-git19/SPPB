import torch
from torch import nn

# Cargar el modelo PyTorch
model = torch.load("resnet18_baseline_att_224x224_A_epoch_249.pth", map_location="cpu")
model.eval()

# Crear entrada de prueba
dummy_input = torch.randn(1, 3, 224, 224)

# Exportar a ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)
print("Modelo convertido a ONNX: model.onnx")