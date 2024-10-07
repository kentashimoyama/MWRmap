import math
import scipy.spatial as ss
import numpy as np

from MMSProbe.utils.Common import printLog
from MMSProbe.conf.Config import Config

DEBUG = False  # debug switch

FRONT_FAR = Config.laneDet_front_far * 0.75
SIDE_FAR = Config.laneDet_side_far * 0.75


class MapLane:
    def __init__(self, lane, points):
        # storage para
        self.lane = lane
        self.points = points[:, :2]
        self.group = points[:, 3]
        length = len(lane.items())
        printLog("MapLane", f"got map lene total = {length}")
        if length != len(lane.items()) or length == 0:
            printLog("MapLane", f"line points count err ..")
            self.lane = None
            self.points = None

    @classmethod
    def read_points(cls, path: str):
        """
		format = (x, y, id, .. ) in JPRCS9
		:param path:
		:return:
		"""
        pts1 = np.zeros((0, 3))
        pts2 = np.zeros((0, 3))

        line_pts = []
        id_old = -1
        fr = open(path, "r")
        # for data_line in fr:
        # 	y, x, _, _, i = data_line.split(",")[:5]  # swap x, y
        # 	pt = np.asarray([x, y, i], np.float64)
        # 	i = int(i)
        # 	if id_old < 0: id_old = i
        # 	if id_old != i:
        # 		id_old = i
        # 		if len(line_pts) > 1:
        # 			pts1 = np.concatenate((pts1, line_pts[:-1]))
        # 			pts2 = np.concatenate((pts2, line_pts[1:]))
        # 		line_pts = []
        # 	else: line_pts.append(pt)
        # if len(line_pts) > 1:  # for last lane line
        # 	pts1 = np.concatenate((pts1, line_pts[:-1]))
        # 	pts2 = np.concatenate((pts2, line_pts[1:]))

        lane = {}  # groupをkeyにして読み込む
        for data_line in fr:
            y, x, _, _, i = data_line.split(",")[:5]  # swap x, y
            pt = np.asarray([x, y], np.float64)
            if i not in lane.keys():
                lane[i] = [pt]
            else:
                lane[i].append(pt)
        fr.close()

        from MMSProbe.utils.Visualization import PCModel
        from MMSProbe.utils.Common.debug_manager import folder_MapLane
        model = PCModel()
        # for p1, p2 in zip(pts1, pts2):
        # 	if p1[2] == p2[2]:
        # 		p1 = (p1[0], p1[1], 0)
        # 		p2 = (p2[0], p2[1], 0)
        for group_id in lane.keys():
            lane_tmp = lane[group_id].copy()
            KDTree = ss.cKDTree(lane_tmp)
            point_used = []
            point_used.append(0)
            loop = 0
            curr = 0
            while loop < len(lane_tmp) - 1:
                p1 = lane_tmp[curr]
                _, ids = KDTree.query(p1, k=len(lane_tmp))
                for idx in ids:
                    if idx not in point_used and (not np.array_equal(lane_tmp[idx], p1)):
                        p2 = lane_tmp[idx]
                        point_used.append(idx)
                        break
                curr = idx
                p1 = (p1[0], p1[1], 0, group_id)
                p2 = (p2[0], p2[1], 0, group_id)
                model.add_line(p1, p2, color=(200, 200, 200), density=10)
                loop += 1
        model.save_to_las(folder_MapLane + "/LaneLine.las")
        # ---------------------------
        return cls(lane, model.points)

    @classmethod
    def read_lines(cls, path: str):
        """
		format = (x1, y1, x2, y2, id, ...)
		:param path:
		:return:
		"""
        # todo:
        pass

    def save_to_csv(self, path: str):
        # todo:
        pass

    def pop_nearby(self, x, y, angle):
        """
		select lines in front rect, by position and angle
		:param x: position x
		:param y: position y
		:param angle: of yaw
		:return:
			pts1: 1st point position, center by given pos
			pts2: 2nd point position, center by given pos
		"""
        if not len(self.points) > 0: return None
        sub_points = self.points - (x, y)

        # selection 1st, get range rect
        valid1 = np.logical_and(sub_points[:, 0] > -FRONT_FAR, sub_points[:, 0] < FRONT_FAR)
        valid2 = np.logical_and(sub_points[:, 1] > -FRONT_FAR, sub_points[:, 1] < FRONT_FAR)
        valid_1st = np.logical_and(valid1, valid2)
        if not np.sum(valid_1st) > 0: return None
        sub_points = sub_points[valid_1st, :]
        sub_group = self.group[:, None][valid_1st, :]

        # rotation
        cos = math.cos(angle)
        sin = math.sin(angle)
        T = np.asarray([[cos, -sin], [sin, cos]])
        sub_points = np.dot(sub_points, T)

        result_points = np.c_[sub_points, sub_group]
        # 　若干複雑なHD Mapに対応するやつ
        set_group = set(sub_group[:, 0])  # なんのグループが入っているかを見る
        lanes_angle = []
        # if len(set_group) > 2:
        for i, group in enumerate(set_group):  # 区画線と自分の向いてる方向の間の角度を見る
            points = result_points[np.where(result_points[:, 2] == group)]
            lane_vector = points[0, :2] - points[-1, :2]  # 区画線のベクトル
            if lane_vector[0] < 0: lane_vector = -lane_vector
            lane_angle = math.atan2(lane_vector[1], lane_vector[0])
            lanes_angle.append([group, lane_angle])

        mask = np.zeros(result_points.shape[0])
        for g_lane_angle in lanes_angle:  # 進行方向に45°以上区画線を除去する
            if g_lane_angle[1] > 0.785398 or g_lane_angle[1] < -0.785398:  # 45deg
                lanes_angle.remove(g_lane_angle)
            else:
                mask = np.logical_or(result_points[:, 2] == g_lane_angle[0], mask)
        try:
            result_points = result_points[mask]
        except IndexError:
            print(mask)
        # valid1 = np.logical_and(sub_points[:, 0] > 0, sub_points[:, 0] < FRONT_FAR)
        # valid2 = np.logical_and(sub_points[:, 1] > -SIDE_FAR, sub_points[:, 1] < SIDE_FAR)
        # valid_2nd = np.logical_and(valid1, valid2)
        # if not np.sum(valid_2nd) > 0: return None
        # sub_pts1 = sub_points[valid_2nd, :]

        # # same process for 2nd point
        # sub_pts2 = self.pts2 - (x, y)
        # sub_pts2 = sub_pts2[valid_1st, :]
        # sub_pts2 = np.dot(sub_pts2, T)
        # sub_pts2 = sub_pts2[valid_2nd, :]

        return result_points
