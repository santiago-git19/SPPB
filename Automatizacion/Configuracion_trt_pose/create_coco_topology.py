import json
import os

def create_coco_topology():
    """
    Crea la topología COCO estándar que podría ser compatible con tu modelo
    """
    
    # Topología COCO estándar (17 keypoints)
    coco_topology = {
        "supercategory": "person",
        "id": 1,
        "name": "person",
        "keypoints": [
            "nose",
            "left_eye",
            "right_eye", 
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle"
        ],
        "skeleton": [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
            [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
            [2, 4], [3, 5], [4, 6], [5, 7]
        ]
    }
    
    # Variante con 18 keypoints (añadiendo "neck")
    coco_18_topology = {
        "supercategory": "person",
        "id": 1,
        "name": "person",
        "keypoints": [
            "nose",
            "left_eye",
            "right_eye", 
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
            "neck"
        ],
        "skeleton": [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
            [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
            [2, 4], [3, 5], [4, 6], [5, 7], [18, 1], [18, 6], [18, 7]
        ]
    }
    
    # Guardar ambas topologías
    base_path = "/home/mobilenet/Documentos/Trabajo/trt_pose/models"
    
    # Crear directorio si no existe
    os.makedirs(base_path, exist_ok=True)
    
    # Guardar topología COCO 17
    coco_17_path = os.path.join(base_path, "human_pose_coco_17.json")
    with open(coco_17_path, 'w') as f:
        json.dump(coco_topology, f, indent=2)
    print(f"Topología COCO 17 guardada en: {coco_17_path}")
    
    # Guardar topología COCO 18
    coco_18_path = os.path.join(base_path, "human_pose_coco_18.json")
    with open(coco_18_path, 'w') as f:
        json.dump(coco_18_topology, f, indent=2)
    print(f"Topología COCO 18 guardada en: {coco_18_path}")
    
    return coco_17_path, coco_18_path

def test_topology_compatibility():
    """
    Prueba qué topología es compatible con el modelo
    """
    import torch
    import trt_pose.coco
    import trt_pose.models
    
    model_path = "/home/mobilenet/Documentos/Trabajo/trt_pose/models/resnet18_baseline_att_224x224_A_epoch_249.pth"
    
    coco_17_path, coco_18_path = create_coco_topology()
    
    print("\n=== Probando compatibilidad ===")
    
    for name, topology_path in [("COCO 17", coco_17_path), ("COCO 18", coco_18_path)]:
        try:
            print(f"\nProbando {name}...")
            
            # Cargar topología
            with open(topology_path, 'r') as f:
                human_pose = json.load(f)
            
            topology = trt_pose.coco.coco_category_to_topology(human_pose)
            num_parts = len(human_pose['keypoints'])
            num_links = len(human_pose['skeleton'])
            
            print(f"  Keypoints: {num_parts}, Links: {num_links}")
            
            # Crear modelo
            model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links)
            
            # Intentar cargar pesos
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint)
            
            print(f"  ✓ {name} es COMPATIBLE!")
            print(f"  Usa esta topología: {topology_path}")
            return topology_path
            
        except Exception as e:
            print(f"  ✗ {name} no es compatible: {e}")
    
    print("\n⚠️  Ninguna topología estándar es compatible.")
    print("Ejecuta investigate_model.py para más detalles.")
    return None

if __name__ == "__main__":
    compatible_topology = test_topology_compatibility()
    
    if compatible_topology:
        print(f"\n🎉 Topología compatible encontrada: {compatible_topology}")
        print("Actualiza tu código principal para usar esta topología.")
    else:
        print("\n❌ No se encontró una topología compatible.")
        print("Necesitas investigar más la estructura del modelo.")