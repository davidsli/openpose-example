import os
import cv2
import numpy as np
from .inference import OpInf, nPoints, posePairs, colors


def coordPersonwiseKeypoints(keypointsList, personwiseKeypoints):
    coordKeypoints = []
    for personKeypoints in personwiseKeypoints:
        coordPersonKeypoints = []
        for keypointsIdx in personKeypoints[:-1]:
            if keypointsIdx == -1:
                coordPersonKeypoints.append(None)
            else:
                coordPersonKeypoints.append(keypointsList[int(keypointsIdx)])
        coordKeypoints.append(coordPersonKeypoints)
    return np.array(coordKeypoints)


def runFrameWithInference(op: OpInf, frame, verbose=False):
    op(frame)
    return runFrame(op, frame, verbose)


def runFrame(op: OpInf, frame, verbose=False):
    detected_keypoints, keypoints_list, personwiseKeypoints = op.getResult()
    if verbose:
        print('detected_keypoints :')
        print(detected_keypoints)
        print('keypoints_list :')
        print(keypoints_list)
        print('personwiseKeypoints :')
        print(personwiseKeypoints)
        coordKeypoints = coordPersonwiseKeypoints(keypoints_list, personwiseKeypoints)
        print('\ncoordKeypoints :')
        print(coordKeypoints)
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


def extractFrame(video_lists, save_dir):
    ffmpeg_command = 'ffmpeg -ss 00:00:0 -i {0} -r 10 -f image2 {1}/{2}-%d.jpg'
    for i, video in enumerate(video_lists):
        os.system(ffmpeg_command.format(video, save_dir, i))


def removeRotate(video_lists, save_dir):
    ffmpeg_command = 'ffmpeg -i {0} -vf "rotate=0" {1}/{2}'
    for video in video_lists:
        os.system(ffmpeg_command.format(video, save_dir, os.path.basename(video)))


def keypointsDataFromImageFiles(op: OpInf, image_lists):
    i = 0
    keypointsData = np.zeros((0,30))
    for image_path in image_lists:
        i = i + 1
        print(i, image_path)
        image = cv2.imread(image_path)
        keypoints = keypointsDataFromImage(op, image)
        keypointsData = np.append(keypointsData, keypoints, axis=0)
        if i % 10 == 0:
            print(keypointsData)
    return keypointsData


def keypointsDataFromImage(op:OpInf, image):
    _, keypointsList, personwiseKeypoints = op(image)
    keypoints = coordPersonwiseKeypoints(keypointsList, personwiseKeypoints)
    keypoints = removePersonWithNone(keypoints)
    #  print(keypoints.shape)
    if keypoints.size != 0:
        keypoints = np.delete(keypoints, 2, axis=-1)
    keypoints = np.reshape(keypoints, (-1, 30))
    return keypoints


def removePersonWithNone(coordKeypoints):
    keypoints = []
    for i in range(coordKeypoints.shape[0]):
        #  print(coordKeypoints.shape[1:])
        if coordKeypoints.shape[1:] == (15,3):
            keypoints.append(coordKeypoints[i, ...])
    return np.array(keypoints)


if __name__ == '__main__':
    import os
    from pathlib import Path

    root_dir = str(Path(os.path.dirname(os.path.realpath(__file__))) / '..' / '..')
    device = 'gpu'
    model_path = root_dir + '/model'

    proto_file = model_path + '/pose_deploy_linevec_faster_4_stages.prototxt'
    weights_file = model_path + '/pose_iter_160000.caffemodel'

    net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)
    op = OpInf(net, inHeight=400)

    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    print("Using GPU device")

    image = cv2.imread(root_dir + '/img/test_img/test-2.jpg')

    image = runFrameWithInference(op, image, verbose=True)
    # cv2.imshow('Inferenced Image', image)
    # cv2.waitKey(0)
