import cv2
import numpy as np

nPoints = 15

keypointsMapping = ["Head", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", "RHip", "RKnee",
                    "RAnkle", "LHip", "LKnee", "LAnkle", "Chest", "Background"]

posePairs = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14], [14, 8], [8, 9], [9, 10], [14, 11],
             [11, 12], [12, 13]]

mapIdx = [[16, 17], [18, 19], [20, 21], [22, 23], [24, 25], [26, 27], [28, 29], [30, 31], [32, 33], [34, 35], [36, 37],
          [38, 39], [40, 41], [42, 43]]

colors = [[0, 100, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255],
          [0, 255, 0], [255, 200, 100], [255, 0, 255], [0, 255, 0], [255, 200, 100], [255, 0, 255],
          [0, 0, 255], [255, 0, 0]]


class OpInf:
    def __init__(self, op_net, inHeight=368):
        self.net = op_net
        self.inHeight = inHeight
        self.output = None
        self.keypoints = None
        self.image = None
        self.detected_keypoints = None
        self.keypoints_list = None
        self.valid_pairs = None
        self.invalid_pairs = None
        self.personwiseKeypoints = None

    def __call__(self, image):
        self.netInference(image)
        self.probMapsToKeypoints()
        self.findValidPairs()
        self.findPersonwiseKeypoints()
        return self.detected_keypoints, self.keypoints_list, self.personwiseKeypoints

    def netInference(self, image):
        self.image = image
        frameHeight, frameWidth = image.shape[0:2]
        # Fix the input Height and get the width according to the Aspect Ratio
        inWidth = int((self.inHeight / frameHeight) * frameWidth)
        inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (inWidth, self.inHeight), (0, 0, 0), swapRB=False, crop=False)
        self.net.setInput(inpBlob)
        self.output = self.net.forward()

    def __getKeypoints(probMap, threshold=0.1):
        mapSmooth = cv2.GaussianBlur(probMap, (3, 3), 0, 0)
        mapMask = np.uint8(mapSmooth > threshold)
        keypoints = []
        # find the blobs
        contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # for each blob find the maxima
        for cnt in contours:
            blobMask = np.zeros(mapMask.shape)
            blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
            maskedProbMap = mapSmooth * blobMask
            _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
            keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))
        return keypoints

    def probMapsToKeypoints(self):
        self.detected_keypoints = []
        self.keypoints_list = np.zeros((0, 3))
        keypoint_id = 0
        threshold = 0.1
        for part in range(nPoints):
            probMap = self.output[0, part, :, :]
            probMap = cv2.resize(probMap, (self.image.shape[1], self.image.shape[0]))
            keypoints = OpInf.__getKeypoints(probMap, threshold)
            keypoints_with_id = []
            for i in range(len(keypoints)):
                keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                self.keypoints_list = np.vstack([self.keypoints_list, keypoints[i]])
                keypoint_id += 1
            self.detected_keypoints.append(keypoints_with_id)

    def findValidPairs(self):
        frameHeight, frameWidth = self.image.shape[0:2]
        self.valid_pairs = []
        self.invalid_pairs = []
        n_interp_samples = 10
        paf_score_th = 0.1
        conf_th = 0.7
        # loop for every posePairs
        for k in range(len(mapIdx)):
            # A->B constitute a limb
            pafA = self.output[0, mapIdx[k][0], :, :]
            pafB = self.output[0, mapIdx[k][1], :, :]
            pafA = cv2.resize(pafA, (frameWidth, frameHeight))
            pafB = cv2.resize(pafB, (frameWidth, frameHeight))
            # Find the keypoints for the first and second limb
            candA = self.detected_keypoints[posePairs[k][0]]
            candB = self.detected_keypoints[posePairs[k][1]]
            nA = len(candA)
            nB = len(candB)
            # If keypoints for the joint-pair is detected
            # check every joint in candA with every joint in candB
            # Calculate the distance vector between the two joints
            # Find the PAF values at a set of interpolated points between the joints
            # Use the above formula to compute a score to mark the connection valid
            if nA != 0 and nB != 0:
                valid_pair = np.zeros((0, 3))
                for i in range(nA):
                    max_j = -1
                    maxScore = -1
                    found = 0
                    for j in range(nB):
                        # Find d_ij
                        d_ij = np.subtract(candB[j][:2], candA[i][:2])
                        norm = np.linalg.norm(d_ij)
                        if norm:
                            d_ij = d_ij / norm
                        else:
                            continue
                        # Find p(u)
                        interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                                np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                        # Find L(p(u))
                        paf_interp = []
                        for k in range(len(interp_coord)):
                            paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                               pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))]])
                        # Find E
                        paf_scores = np.dot(paf_interp, d_ij)
                        avg_paf_score = sum(paf_scores) / len(paf_scores)
                        # Check if the connection is valid
                        # If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair
                        if (len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples) > conf_th:
                            if avg_paf_score > maxScore:
                                max_j = j
                                maxScore = avg_paf_score
                                found = 1
                    # Append the connection to the list
                    if found:
                        valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)
                # Append the detected connections to the global list
                self.valid_pairs.append(valid_pair)
            else:  # If no keypoints are detected
                # print("No Connection : k = {}".format(k))
                self.invalid_pairs.append(k)
                self.valid_pairs.append([])

    # This function creates a list of keypoints belonging to each person
    # For each detected valid pair, it assigns the joint(s) to a person
    # It finds the person and index at which the joint should be added. This can be done since we have an id for each joint
    def findPersonwiseKeypoints(self):
        # the last number in each row is the overall score
        self.personwiseKeypoints = -1 * np.ones((0, 16))
        for k in range(len(mapIdx)):
            if k not in self.invalid_pairs:
                partAs = self.valid_pairs[k][:, 0]
                partBs = self.valid_pairs[k][:, 1]
                indexA, indexB = np.array(posePairs[k])
                for i in range(len(self.valid_pairs[k])):
                    found = 0
                    person_idx = -1
                    for j in range(len(self.personwiseKeypoints)):
                        if self.personwiseKeypoints[j][indexA] == partAs[i]:
                            person_idx = j
                            found = 1
                            break
                    if found:
                        self.personwiseKeypoints[person_idx][indexB] = partBs[i]
                        self.personwiseKeypoints[person_idx][-1] += self.keypoints_list[partBs[i].astype(int), 2] + \
                                                                    self.valid_pairs[k][i][2]
                    # if find no partA in the subset, create a new subset
                    elif not found and k < 14:
                        row = -1 * np.ones(16)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        # add the keypoint_scores for the two keypoints and the paf_score
                        row[-1] = sum(self.keypoints_list[self.valid_pairs[k][i, :2].astype(int), 2]) + \
                                  self.valid_pairs[k][i][2]
                        self.personwiseKeypoints = np.vstack([self.personwiseKeypoints, row])
