import torch
from torch2trt import torch2trt
from trt_pose.models import resnet18_baseline_att

# Ruta del modelo PyTorch
model_path = "/home/mobilenet/Documentos/Trabajo/trt_pose/models/resnet18_baseline_att_224x224_A_epoch_249.pth"

# Cargar el modelo PyTorch con los valores correctos
num_parts = 18  # Número de keypoints
num_links = 19  # Número de conexiones entre keypoints
model = resnet18_baseline_att(num_parts, num_links).cuda().eval()
model.load_state_dict(torch.load(model_path))

# Convertir a TensorRT
data = torch.zeros((1, 3, 224, 224)).cuda()  # Entrada de ejemplo
model_trt = torch2trt(model, [data])

# Guardar el modelo TensorRT
engine_path = "/home/mobilenet/Documentos/Trabajo/trt_pose/models/resnet18_baseline_att_224x224_A_epoch_249.engine"
torch.save({'engine': model_trt.state_dict()}, engine_path)
print(f"Modelo convertido y guardado en: {engine_path}")