import json

topology_path = "/home/mobilenet/Documentos/Trabajo/trt_pose/models/human_pose.json"

with open(topology_path, 'r') as f:
    topology = json.load(f)

num_keypoints = len(topology['keypoints'])
num_links = len(topology['skeleton'])

print(f"Topología: {num_keypoints} puntos clave, {num_links} conexiones")