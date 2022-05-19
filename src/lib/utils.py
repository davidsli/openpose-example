import numpy as np


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
