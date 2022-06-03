import cv2
import os
import argparse
import numpy as np
from pathlib import Path
from lib.inference import OpInf
from lib.utils import runFrame
import matplotlib.pyplot as plt


def main():
    root_dir = str(Path(os.path.dirname(os.path.realpath(__file__))) / '..')

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', default='gpu', help='Device to use in OpenCV')
    parser.add_argument('-i', '--image', help='Image file')
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

    image_name = args.get('image')
    image1 = cv2.imread(image_name)

    frame = runFrame(op, image1)

    output = op.getOutput()
    for i in range(16):
        heatmap = output[0,i,...]
        heatmap = cv2.resize(heatmap, (image1.shape[1], image1.shape[0]))

        heatmapnp = np.array(heatmap * 255, dtype=np.uint8)
        heatmap = cv2.applyColorMap(heatmapnp, cv2.COLORMAP_JET)
        overlayed = cv2.addWeighted(image1,0.4,heatmap,0.4,0)

        cv2.imwrite(root_dir + f'/out/skeleton/{i}.jpg', overlayed)

    x = np.linspace(0, image1.shape[1] - 1, int(image1.shape[1] / 20)).astype(np.uint16)
    y = np.linspace(0, image1.shape[0] - 1, int(image1.shape[0] / 20)).astype(np.uint16)
    x, y = np.meshgrid(x, y)
    for i in range(14):
        ix = i * 2 + 16
        iy = ix + 1
        vmapx = cv2.resize(output[0, ix, ...], (image1.shape[1], image1.shape[0]))
        vmapy = cv2.resize(output[0, iy, ...], (image1.shape[1], image1.shape[0]))
        u = vmapx[y, x]
        v = vmapy[y, x]

        n = -2
        color = np.sqrt(((u-n)/2)*2 + ((v-n)/2)*2)

        plt.figure()
        plt.imshow(image1)
        plt.quiver(x,y,u,v, color, scale=20)
        plt.axis('off')
        plt.savefig(root_dir + f'/out/skeleton/vmap-{i}.jpg')


if __name__ == '__main__':
    main()
