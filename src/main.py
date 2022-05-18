import cv2
import numpy as np
import time
import os
from pathlib import Path
from lib.inference import OpInf, nPoints, posePairs, colors

device = 'gpu'

model_path = Path(os.path.dirname(os.path.realpath(__file__))) / '..' / 'model'

proto_file = str(model_path) + '/pose_deploy_linevec_faster_4_stages.prototxt'
weights_file = str(model_path) + '/pose_iter_160000.caffemodel'

net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)

image1 = cv2.imread('img/test_img/test-2.jpg')

if device == "cpu":
    net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
    print("Using CPU device")
elif device == "gpu":
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    print("Using GPU device")

t = time.time()

op = OpInf(net)
detected_keypoints, keypoints_list, personwiseKeypoints = op(image1)

frameClone = image1.copy()
for i in range(nPoints):
    for j in range(len(detected_keypoints[i])):
        cv2.circle(frameClone, detected_keypoints[i][j][0:2], 3, [0, 0, 255], -1, cv2.LINE_AA)

for i in range(14):
    for n in range(len(personwiseKeypoints)):
        index = personwiseKeypoints[n][np.array(posePairs[i])]
        if -1 in index:
            continue
        B = np.int32(keypoints_list[index.astype(int), 0])
        A = np.int32(keypoints_list[index.astype(int), 1])
        cv2.line(frameClone, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)

cv2.imshow('OpenPose Inference', frameClone)
cv2.waitKey(0)
