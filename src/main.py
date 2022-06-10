import cv2
import time
import os
import sys
import argparse
from pathlib import Path
from lib.inference import OpInf
from lib.utils import runFrameWithInference
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

    # video_name = 'test-3'
    # cap = cv2.VideoCapture(f'video/test/{video_name}.mp4')
    # cam_url = 'rtsp://192.168.1.32:554/stream2'
    cam_url = args.get('url')
    cap = BufferlessVideoCapture(cam_url)
    if not cap.isOpened():
        sys.stderr.write('ERROR: Camera is not opened.')
        sys.exit()

    output_name = root_dir + '/output.avi'
    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = round(cap.get(cv2.CAP_PROP_FPS))
    fps = 10

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(output_name, fourcc, fps, (w, h))

    while True:
        t = time.time()

        # ret, frame = cap.read()
        frame = cap.read()
        # if not ret:
        #     break

        frame = runFrameWithInference(op, frame)

        cv2.imshow('OpenPose Inference', frame)
        out.write(frame)

        timeTaken = time.time() - t
        print(f'FPS = {1 / timeTaken}, Time Taken = {timeTaken}')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()
    cap.release()


if __name__ == '__main__':
    main()
