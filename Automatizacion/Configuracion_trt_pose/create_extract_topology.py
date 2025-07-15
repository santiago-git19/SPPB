import json
import os
import torch
import trt_pose.coco
import trt_pose.models

def create_exact_topology():
    """
    Crea la topología exacta que necesita tu modelo: 18 keypoints y 21 conexiones
    """
    
    # Topología con exactamente 18 keypoints y 21 conexiones
    exact_topology = {
        "supercategory": "person",
        "id": 1,
        "name": "person",
        "keypoints": [
            "nose",           # 0
            "left_eye",       # 1
            "right_eye",      # 2
            "left_ear",       # 3
            "right_ear",      # 4
            "left_shoulder",  # 5
            "right_shoulder", # 6
            "left_elbow",     # 7
            "right_elbow",    # 8
            "left_wrist",     # 9
            "right_wrist",    # 10
            "left_hip",       # 11
            "right_hip",      # 12
            "left_knee",      # 13
            "right_knee",     # 14
            "left_ankle",     # 15
            "right_ankle",    # 16
            "neck"            # 17
        ],
        "skeleton": [
            [0, 1],   # nose -> left_eye
            [0, 2],   # nose -> right_eye
            [1, 3],   # left_eye -> left_ear
            [2, 4],   # right_eye -> right_ear
            [5, 6],   # left_shoulder -> right_shoulder
            [5, 7],   # left_shoulder -> left_elbow
            [6, 8],   # right_shoulder -> right_elbow
            [7, 9],   # left_elbow -> left_wrist
            [8, 10],  # right_elbow -> right_wrist
            [5, 11],  # left_shoulder -> left_hip
            [6, 12],  # right_shoulder -> right_hip
            [11, 12], # left_hip -> right_hip
            [11, 13], # left_hip -> left_knee
            [12, 14], # right_hip -> right_knee
            [13, 15], # left_knee -> left_ankle
            [14, 16], # right_knee -> right_ankle
            [17, 0],  # neck -> nose
            [17, 5],  # neck -> left_shoulder
            [17, 6],  # neck -> right_shoulder
            [17, 11], # neck -> left_hip
            [17, 12]  # neck -> right_hip
        ]
    }
    
    # Verificar que tenemos exactamente 21 conexiones
    print(f"Número de keypoints: {len(exact_topology['keypoints'])}")
    print(f"Número de conexiones: {len(exact_topology['skeleton'])}")
    
    if len(exact_topology['keypoints']) != 18:
        print("ERROR: No son exactamente 18 keypoints!")
        return None
    
    if len(exact_topology['skeleton']) != 21:
        print("ERROR: No son exactamente 21 conexiones!")
        return None
    
    # Guardar topología
    output_path = "/home/mobilenet/Documentos/Trabajo/trt_pose/models/human_pose_exact.json"
    
    try:
        with open(output_path, 'w') as f:
            json.dump(exact_topology, f, indent=2)
        print(f"✓ Topología exacta guardada en: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error guardando topología: {e}")
        return None

def test_exact_compatibility():
    """
    Prueba la compatibilidad exacta
    """
    model_path = "/home/mobilenet/Documentos/Trabajo/trt_pose/models/resnet18_baseline_att_224x224_A_epoch_249.pth"
    topology_path = create_exact_topology()
    
    if topology_path is None:
        return False
    
    print("\n=== Probando compatibilidad exacta ===")
    
    try:
        # Cargar topología
        with open(topology_path, 'r') as f:
            human_pose = json.load(f)
        
        topology = trt_pose.coco.coco_category_to_topology(human_pose)
        num_parts = len(human_pose['keypoints'])
        num_links = len(human_pose['skeleton'])
        
        print(f"Topología cargada: {num_parts} keypoints, {num_links} links")
        
        # Crear modelo
        model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links)
        
        print(f"Modelo creado para: {num_parts} keypoints, {2 * num_links} canales PAF")
        
        # Intentar cargar pesos
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint)
        
        print("✓ ¡ÉXITO! Modelo y topología son completamente compatibles!")
        print(f"Usa esta topología: {topology_path}")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def create_alternative_topologies():
    """
    Crea algunas topologías alternativas con 18 keypoints y 21 conexiones
    """
    print("\n=== Creando topologías alternativas ===")
    
    # Alternativa 1: Topología más simple
    alt1_topology = {
        "supercategory": "person",
        "id": 1,
        "name": "person",
        "keypoints": [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle", "neck"
        ],
        "skeleton": [
            [1, 2], [1, 3], [2, 4], [3, 5], [4, 6],  # cabeza
            [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], # brazos
            [5, 11], [6, 12], [11, 12],              # torso
            [11, 13], [12, 14], [13, 15], [14, 16],  # piernas
            [18, 1], [18, 5], [18, 6], [18, 11]     # neck connections
        ]
    }
    
    # Alternativa 2: Sin neck, pero con conexiones adicionales
    alt2_topology = {
        "supercategory": "person",
        "id": 1,
        "name": "person",
        "keypoints": [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle", "center"
        ],
        "skeleton": [
            [1, 2], [1, 3], [2, 4], [3, 5], [4, 6],  # cabeza
            [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], # brazos
            [5, 11], [6, 12], [11, 12],              # torso
            [11, 13], [12, 14], [13, 15], [14, 16],  # piernas
            [18, 1], [18, 5], [18, 6], [18, 11]     # center connections
        ]
    }
    
    # Probar ambas alternativas
    alternatives = [
        ("alternative1", alt1_topology),
        ("alternative2", alt2_topology)
    ]
    
    for name, topology in alternatives:
        output_path = f"/home/mobilenet/Documentos/Trabajo/trt_pose/models/human_pose_{name}.json"
        
        try:
            with open(output_path, 'w') as f:
                json.dump(topology, f, indent=2)
            print(f"✓ Topología {name} guardada en: {output_path}")
            
            # Probar compatibilidad
            if test_topology_compatibility(output_path):
                print(f"✓ {name} es compatible!")
                return output_path
            else:
                print(f"✗ {name} no es compatible")
                
        except Exception as e:
            print(f"Error con {name}: {e}")
    
    return None

def test_topology_compatibility(topology_path):
    """
    Prueba si una topología específica es compatible
    """
    model_path = "/home/mobilenet/Documentos/Trabajo/trt_pose/models/resnet18_baseline_att_224x224_A_epoch_249.pth"
    
    try:
        # Cargar topología
        with open(topology_path, 'r') as f:
            human_pose = json.load(f)
        
        topology = trt_pose.coco.coco_category_to_topology(human_pose)
        num_parts = len(human_pose['keypoints'])
        num_links = len(human_pose['skeleton'])
        
        # Crear modelo
        model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links)
        
        # Intentar cargar pesos
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint)
        
        return True
        
    except Exception as e:
        return False

if __name__ == "__main__":
    # Primero probar la topología exacta
    if test_exact_compatibility():
        print("\n🎉 ¡Topología exacta funciona!")
    else:
        print("\n⚠️  Topología exacta no funciona, probando alternativas...")
        alternative_path = create_alternative_topologies()
        
        if alternative_path:
            print(f"\n🎉 ¡Topología alternativa funciona!: {alternative_path}")
        else:
            print("\n❌ Ninguna topología funcionó. Hay algo más complejo...")