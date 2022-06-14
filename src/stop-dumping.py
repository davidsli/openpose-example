import cv2
import time
import os
import sys
import argparse
import numpy as np
import playsound
from pathlib import Path
from keras.models import load_model
from lib.inference import OpInf
from lib.utils import keypointsDataFromImage, runFrame
from lib.video import BufferlessVideoCapture


def main():
    root_dir = str(Path(os.path.dirname(os.path.realpath(__file__))) / '..')

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', default='gpu', help='Device to use in OpenCV')
    parser.add_argument('-u', '--url', help='RTSP url as input')
    args = vars(parser.parse_args())

    device = args.get('device')

    model_path = root_dir + '/model'

    proto_file = model_path + '/pose_deploy_linevec_faster_4_stages.prototxt'
    weights_file = model_path + '/pose_iter_160000.caffemodel'

    net: cv2.dnn.Net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)
    op = OpInf(net, inHeight=400)

    model = load_model(root_dir + '/out/model/checkpoint.h5')

    if device == "cpu":
        net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
        print("Using CPU device")
    elif device == "gpu":
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print("Using GPU device")

    cam_url = args.get('url')
    capture = BufferlessVideoCapture(cam_url)
    if not capture.isOpened():
        sys.stderr.write('ERROR: Camera is not opened.')
        sys.exit()

    output_name = root_dir + '/output.avi'
    w = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = round(cap.get(cv2.CAP_PROP_FPS))
    fps = 10

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(output_name, fourcc, fps, (w, h))

    dumpLabel = 'Dumping: {0:.2f}'
    nodumpLabel = 'Person'

    buzzerTime = time.time()
    while True:
        t = time.time()

        # ret, frame = cap.read()
        frame = capture.read()
        # if not ret:
        #     break

        #  frame = runFrame(op, frame)
        keypoints = keypointsDataFromImage(op, frame).astype(np.uint16)
        if keypoints.size != 0:
            pred = model.predict(keypoints)
        else:
            pred = []
        print(pred)

        for i, prob in enumerate(pred):
            font_size = 1
            font_thickness = 2
            if prob[0] > 0.5:
                if time.time() - buzzerTime > 1:
                    buzzerTime = time.time()
                    playsound.playsound(root_dir + '/mp3/alarm.mp3', False)
                headPosition = keypoints[i,0:2]
                print('headPosition', headPosition)
                labelWithProb = dumpLabel.format(prob[0] * 100)
                (w, h), _ = cv2.getTextSize(labelWithProb, cv2.FONT_HERSHEY_SIMPLEX, font_size, font_thickness)
                frame = cv2.rectangle(frame, (headPosition[0], headPosition[1] - h),
                                      (headPosition[0] + w, headPosition[1]), (0,0,255), -1)
                frame = cv2.putText(frame, labelWithProb, (headPosition[0], headPosition[1] - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255), font_thickness)
            else:
                headPosition = keypoints[i,0:2]
                print('headPosition', headPosition)
                labelWithProb = nodumpLabel.format(prob[0] * 100)
                (w, h), _ = cv2.getTextSize(labelWithProb, cv2.FONT_HERSHEY_SIMPLEX, font_size, font_thickness)
                frame = cv2.rectangle(frame, (headPosition[0], headPosition[1] - h),
                                      (headPosition[0] + w, headPosition[1]), (47,157,39), -1)
                frame = cv2.putText(frame, labelWithProb, (headPosition[0], headPosition[1] - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255), font_thickness)

        frame1 = runFrame(op, frame)

        cv2.imshow('Real-time Video', frame)
        cv2.imshow('With skeleton', frame1)
        #  out.write(frame)

        timeTaken = time.time() - t
        print(f'FPS = {1 / timeTaken}, Time Taken = {timeTaken}')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()
    capture.release()


if __name__ == '__main__':
    main()
