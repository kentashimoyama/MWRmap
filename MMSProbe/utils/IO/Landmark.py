import math
import scipy.spatial as ss
import numpy as np

from MMSProbe.utils.Common import printLog
from MMSProbe.conf.Config import Config

FRONT_FAR = Config.landmark_front_far
SIDE_FAR = Config.landmark_side_far

class Landmark:
    def __init__(self, landmarks):
        self.landmarks = np.asarray(landmarks)

    @classmethod
    def read_points(cls, path: str):
        fr = open(path, "r")

        landmarks = []
        for data_line in fr:
            x, y, id = data_line.split(",")[:3]  # swap x, y
            landmark = np.asarray([x, y, id], np.float64)
            landmarks.append(landmark)
        fr.close()
        return cls(landmarks)

    def pop_nearby(self, x, y, angle):
        """
        select lines in front rect, by position and angle
        :param x: position x
        :param y: position y
        :param angle: of yaw
        :return:
            pts: Landmark position, center by given pos
        """
        if not len(self.landmarks) > 0: return None
        sub_points = self.landmarks[:,0:2] - (x, y)

        # selection 1st, get range rect
        valid1 = np.logical_and(sub_points[:, 0] > -FRONT_FAR, sub_points[:, 0] < FRONT_FAR)
        valid2 = np.logical_and(sub_points[:, 1] > -FRONT_FAR, sub_points[:, 1] < FRONT_FAR)
        valid_1st = np.logical_and(valid1, valid2)
        if not np.sum(valid_1st) > 0: return None
        group = self.landmarks[valid_1st, 2]
        sub_points = sub_points[valid_1st, :]

        # rotation
        cos = math.cos(angle)
        sin = math.sin(angle)
        T = np.asarray([[cos, -sin], [sin, cos]])
        sub_points = np.dot(sub_points, T)
        sub_points = np.hstack((sub_points, group[:,None]))

        return sub_points