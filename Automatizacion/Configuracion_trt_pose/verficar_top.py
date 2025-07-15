import json

#topology_path = "/home/mobilenet/Documentos/Trabajo/trt_pose/models/human_pose.json"

#with open(topology_path, 'r') as f:
#    topology = json.load(f)

topology = {
    "keypoints": [
        "nose", "neck", "shoulder_left", "shoulder_right", "elbow_left", "elbow_right",
        "wrist_left", "wrist_right", "hip_left", "hip_right", "knee_left", "knee_right",
        "ankle_left", "ankle_right", "eye_left", "eye_right", "ear_left", "ear_right"
    ],
    "skeleton": [
        [0, 1], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7],
        [1, 8], [1, 9], [8, 10], [9, 11], [10, 12], [11, 13],
        [0, 14], [0, 15], [14, 16], [15, 17], [16, 18], [17, 19]
    ]
}
num_keypoints = len(topology['keypoints'])
num_links = len(topology['skeleton'])

print(f"Topología: {num_keypoints} puntos clave, {num_links} conexiones")