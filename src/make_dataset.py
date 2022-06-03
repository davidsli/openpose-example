import os
import cv2
from pathlib import Path
from glob import glob
import numpy
from lib.utils import extractFrame, keypointsDataFromImageFiles
from lib.inference import OpInf

def main():
    root_dir = str(Path(os.path.dirname(os.path.realpath(__file__))) / '..')

    model_path = root_dir + '/model'

    proto_file = model_path + '/pose_deploy_linevec_faster_4_stages.prototxt'
    weights_file = model_path + '/pose_iter_160000.caffemodel'

    net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)
    op = OpInf(net, inHeight=400)

    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    print("Using GPU device")

    extractFrame(glob(root_dir + '/video/pos/*.mp4'), root_dir + '/img/out/pos')
    keypointsDataPos = keypointsDataFromImageFiles(op, glob(root_dir + '/img/out/pos/*.jpg'))
    keypointsDataPos = keypointsDataPos.astype(numpy.uint8)
    numpy.savetxt(root_dir + '/out/pdata.csv', keypointsDataPos, fmt='%d', delimiter=',')

    extractFrame(glob(root_dir + '/video/neg/*.mp4') + glob(root_dir + '/video/neg/*.MOV'), root_dir + '/img/out/neg')
    keypointsDataNeg = keypointsDataFromImageFiles(op, glob(root_dir + '/img/out/neg/*.jpg'))
    keypointsDataNeg = keypointsDataNeg.astype(numpy.uint8)
    numpy.savetxt(root_dir + '/out/ndata.csv', keypointsDataNeg, fmt='%d', delimiter=',')

if __name__ == '__main__':
    main()
