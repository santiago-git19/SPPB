import torch
import json
import os

def investigate_model_and_topology():
    """
    Investiga la estructura del modelo y la topología para encontrar la configuración correcta
    """
    
    model_path = "/home/mobilenet/Documentos/Trabajo/trt_pose/models/resnet18_baseline_att_224x224_A_epoch_249.pth"
    topology_path = "/home/mobilenet/Documentos/Trabajo/trt_pose/models/human_pose.json"
    
    print("=== Investigando modelo y topología ===")
    
    # 1. Cargar y examinar el modelo
    print("\n1. Examinando modelo...")
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Buscar las capas cmap_conv y paf_conv para determinar el número de keypoints
        cmap_weight_shape = None
        paf_weight_shape = None
        
        for key, value in checkpoint.items():
            if 'cmap_conv.weight' in key:
                cmap_weight_shape = value.shape
                print(f"  {key}: {value.shape}")
            elif 'paf_conv.weight' in key:
                paf_weight_shape = value.shape
                print(f"  {key}: {value.shape}")
        
        if cmap_weight_shape is not None:
            num_keypoints = cmap_weight_shape[0]
            print(f"  → Número de keypoints detectado: {num_keypoints}")
        
        if paf_weight_shape is not None:
            num_paf_channels = paf_weight_shape[0]
            num_connections = num_paf_channels // 2  # PAF tiene 2 canales por conexión
            print(f"  → Número de conexiones PAF detectado: {num_connections}")
            
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        return
    
    # 2. Examinar la topología actual
    print("\n2. Examinando topología actual...")
    try:
        with open(topology_path, 'r') as f:
            topology = json.load(f)
        
        keypoints = topology.get('keypoints', [])
        skeleton = topology.get('skeleton', [])
        
        print(f"  Keypoints en topología: {len(keypoints)}")
        print(f"  Conexiones en topología: {len(skeleton)}")
        
        print("  Keypoints:")
        for i, keypoint in enumerate(keypoints):
            print(f"    {i}: {keypoint}")
        
        print("  Skeleton:")
        for i, connection in enumerate(skeleton):
            print(f"    {i}: {connection}")
            
    except Exception as e:
        print(f"Error cargando topología: {e}")
        return
    
    # 3. Buscar topologías alternativas
    print("\n3. Buscando topologías alternativas...")
    
    # Buscar archivos JSON en el directorio del modelo
    model_dir = os.path.dirname(model_path)
    print(f"  Buscando en: {model_dir}")
    
    for file in os.listdir(model_dir):
        if file.endswith('.json'):
            json_path = os.path.join(model_dir, file)
            try:
                with open(json_path, 'r') as f:
                    alt_topology = json.load(f)
                
                if 'keypoints' in alt_topology and 'skeleton' in alt_topology:
                    alt_keypoints = len(alt_topology['keypoints'])
                    alt_skeleton = len(alt_topology['skeleton'])
                    
                    print(f"  {file}: {alt_keypoints} keypoints, {alt_skeleton} conexiones")
                    
                    # Verificar si coincide con el modelo
                    if alt_keypoints == num_keypoints and alt_skeleton == num_connections:
                        print(f"    ✓ COINCIDE CON EL MODELO!")
                        print(f"    Usa esta topología: {json_path}")
                        
            except Exception as e:
                print(f"    Error leyendo {file}: {e}")
    
    # 4. Sugerir topologías comunes
    print("\n4. Topologías comunes de trt_pose:")
    
    common_topologies = {
        "COCO (17 keypoints)": {
            "keypoints": 17,
            "skeleton": 19,
            "description": "Topología COCO estándar"
        },
        "MPII (16 keypoints)": {
            "keypoints": 16,
            "skeleton": 14,
            "description": "Topología MPII"
        },
        "Custom 18 keypoints": {
            "keypoints": 18,
            "skeleton": 21,
            "description": "Topología personalizada con 18 keypoints"
        }
    }
    
    for name, info in common_topologies.items():
        match_keypoints = info["keypoints"] == num_keypoints
        match_skeleton = info["skeleton"] == num_connections
        
        status = "✓" if match_keypoints and match_skeleton else "✗"
        print(f"  {status} {name}: {info['keypoints']} keypoints, {info['skeleton']} conexiones")
        
        if match_keypoints and match_skeleton:
            print(f"    → {info['description']} - POSIBLE COINCIDENCIA")

def create_correct_topology():
    """
    Crea una topología correcta basada en los hallazgos
    """
    print("\n=== Creando topología correcta ===")
    
    # Topología COCO modificada para 18 keypoints (ejemplo)
    # Puedes ajustar esto según tus necesidades
    topology_18_keypoints = {
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
            "neck"  # Keypoint adicional
        ],
        "skeleton": [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
            [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
            [2, 4], [3, 5], [4, 6], [5, 7], [18, 1], [18, 6], [18, 7]
        ]
    }
    
    # Guardar topología
    output_path = "/home/mobilenet/Documentos/Trabajo/trt_pose/models/human_pose_18.json"
    try:
        with open(output_path, 'w') as f:
            json.dump(topology_18_keypoints, f, indent=2)
        print(f"Topología de 18 keypoints guardada en: {output_path}")
        print("Prueba usar esta topología en tu código.")
    except Exception as e:
        print(f"Error guardando topología: {e}")

if __name__ == "__main__":
    investigate_model_and_topology()
    create_correct_topology()