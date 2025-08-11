import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np

mp_pose = mp.solutions.pose

interpreter = tf.lite.Interpreter("../models/pose_landmark_lite_fp16.tflite")
interpreter.allocate_tensors()

img = cv2.imread('persona.PNG')
rgb_img = cv2.resize(img, (256, 256))
rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
height, width = img.shape[:2]

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print('input_details: ', input_details)
print('output_details: ', output_details)

rgb_img = np.expand_dims(rgb_img, 0)
input = (rgb_img / 255).astype(np.float32)

interpreter.set_tensor(input_details[0]['index'], input)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
output = np.reshape(output, (1, 1, 39, 5))

kp = list()
for i in output[0][0]:
    kp.append(i.tolist())

for i in kp:
    x, y = int(i[0] / 256 * width), int(i[1] / 256 * height)
    cv2.circle(img, (x, y), 0, (255, 255, 255), 10)

kp = np.array(kp)
for _c in mp_pose.POSE_CONNECTIONS:
    x1, y1, x2, y2 = int(kp[_c[0], 0] / 256 * width), int(kp[_c[0], 1] / 256 * height), int(kp[_c[1], 0] / 256 * width), int(kp[_c[1], 1] / 256 * height)
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)

cv2.namedWindow("MediaPipe Pose", 0)
cv2.resizeWindow('MediaPipe Pose', 1200, 600)
#cv2.imshow('MediaPipe Pose', img)

# Guardar la imagen procesada en la carpeta 'utils'
output_path = 'pose_output.png'
cv2.imwrite(output_path, img)
print(f"✅ Imagen procesada guardada en: {output_path}")

cv2.waitKey(0)