import sys

import cv2
import numpy as np
import time
import os
from pathlib import Path
from lib.inference import OpInf, nPoints, posePairs, colors
from lib.utils import coordPersonwiseKeypoints


def main():
    device = 'gpu'

    model_path = Path(os.path.dirname(os.path.realpath(__file__))) / '..' / 'model'

    proto_file = str(model_path) + '/pose_deploy_linevec_faster_4_stages.prototxt'
    weights_file = str(model_path) + '/pose_iter_160000.caffemodel'

    net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)
    op = OpInf(net, inHeight=400)

    if device == "cpu":
        net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
        print("Using CPU device")
    elif device == "gpu":
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print("Using GPU device")

    # image_dir = 'img/test_img/'
    # image_name = 'test-2.jpg'
    # image1 = cv2.imread(image_dir + image_name)

    video_name = 'test-3'
    cap = cv2.VideoCapture(f'video/test/{video_name}.mp4')
    if not cap.isOpened():
        sys.stderr.write('ERROR: Camera is not opened.')
        sys.exit()

    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    frame_skip_rate = 5

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(f'video/out/{video_name}.avi', fourcc, round(fps / 10), (w, h))

    skip = frame_skip_rate
    while True:
        t = time.time()

        ret, frame = cap.read()
        if not (skip == 0):
            skip = skip - 1
            continue
        if not ret:
            break

        skip = frame_skip_rate

        frame = runFrame(op, frame)

        # cv2.imshow('OpenPose Inference', frame)
        out.write(frame)

        timeTaken = time.time() - t
        print(f'FPS = {1 / timeTaken}, Time Taken = {timeTaken}')

    cv2.waitKey(0)


def runFrame(op, frame):
    detected_keypoints, keypoints_list, personwiseKeypoints = op(frame)
    # print('detected_keypoints : ', detected_keypoints)
    # print('keypoints_list : ', keypoints_list)
    # print('personwiseKeypoints : ', personwiseKeypoints)

    coordKeypoints = coordPersonwiseKeypoints(keypoints_list, personwiseKeypoints)
    # print('\ncoordKeypoints : ', coordKeypoints)

    frameClone = frame.copy()

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

    return frameClone


if __name__ == '__main__':
    main()
