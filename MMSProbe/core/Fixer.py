import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial

from MMSProbe.core import Camera, InitialPoseEstimator, Frame, LaneDetector
from MMSProbe.utils.Common import projection, perspective, trimming, printLog
from MMSProbe.utils.Common.debug_manager import folder_LaneFix
from MMSProbe.conf import Config

DEBUG = True  # debug switch

PIXEL_COE = Config.laneDet_pixel_coe
LANE_DIV_DENSITY = Config.laneDet_line_div_density
# about lane
LANE_LINE_WIDTH = Config.lane_line_width_real
LANE_WIDTH_BASIC = Config.lane_width_basic_real
# LANE_FIX_DIS_TH = LANE_WIDTH_BASIC / 2
LANE_FIX_DIS_TH = 10
# distance para
FRONT_FAR = Config.laneDet_front_far
SIDE_FAR = Config.laneDet_side_far
FRONT_FAR_HALF = FRONT_FAR / 2
CAMERA_GNSS_OFFSET = np.array([0.45, -0.37]) # GNSSアンテナを原点として，そこから見たRADARの位置
sigmoid = lambda x:1./(1.+np.exp(-x))

class Fixer:
    """
	warming, for easy using cv2, all float array dtype -> np.float32
	"""

    def __init__(self):
        # storage para
        self.size_g = (int(2 * SIDE_FAR * PIXEL_COE), 0)
        printLog("LaneFix", f"set init ground image size = {self.size_g}")
        self.pts_i = None
        self.pts_g = None
        self.pts_v = None

        self._flag = False
        self._T = None  # 2d transform mat


    @staticmethod
    def lane2points(lanes):
        """
        :param lanes: (pts1, pts2), two points of the lane lines
        :return:
            lane_pts_HD: unnecessary contains pts2
        """
        density = LANE_DIV_DENSITY
        pts1, pts2 = lanes
        lane_pts = np.zeros((0, 2), np.float32)
        for p1, p2 in zip(pts1, pts2):
            delta = p2 - p1
            num = int(np.linalg.norm(delta) * density)
            steps = np.arange(num) / num
            w = np.tile(steps, (2, 1)).transpose()
            sub_pts = delta * w + p1
            lane_pts = np.concatenate((lane_pts, sub_pts))
        return lane_pts.astype(np.float32)

    def setup_perspective(self, R_v2c, height, camera: Camera):
        img_w, img_h = camera.imgSize
        # calculate front near
        tmp_v = np.asarray([
            [FRONT_FAR, -height, -SIDE_FAR],
            [FRONT_FAR, -height, SIDE_FAR],
            [FRONT_FAR_HALF, -height, SIDE_FAR],
            [FRONT_FAR_HALF, -height, -SIDE_FAR],
        ], dtype=np.float32)
        tmp_c = np.asarray(np.dot(tmp_v, R_v2c), dtype=np.float32)
        tmp_i = projection(camera.mat, tmp_c) #地面を画像に投影する

        g_w = self.size_g[0]
        g_h = int(FRONT_FAR_HALF * PIXEL_COE)
        tmp_g = np.asarray([[0, 0], [g_w, 0], [g_w, g_h], [0, g_h]], dtype=np.float32)

        persMat_i2g = cv2.getPerspectiveTransform(tmp_i, tmp_g)
        persMat_g2i = np.linalg.inv(persMat_i2g)

        corners_i = np.asarray([[0, img_h], [img_w, img_h]])
        corners_g = perspective(persMat_i2g, corners_i)
        tmp_size_g = [self.size_g[0], self.size_g[1]]
        g_h = tmp_size_g[1] = int(min(corners_g[0, 1], corners_g[1, 1]))
        self.size_g = (tmp_size_g[0], tmp_size_g[1])

        self.pts_g = np.asarray([[0, 0], [g_w, 0], [g_w, g_h], [0, g_h]], dtype=np.float32)
        self.pts_i = perspective(persMat_g2i, self.pts_g)
        self.pts_v = np.zeros((4, 2), dtype=np.float32)
        self.pts_v[:, 0] = FRONT_FAR - self.pts_g[:, 1] / PIXEL_COE  # x-axis in vehicle
        self.pts_v[:, 1] = SIDE_FAR - self.pts_g[:, 0] / PIXEL_COE  # z-axis in vehicle
        pass

    #変更箇所
    # 直線のフィット関数
    def fit_line(self, points):
        x = points[:, 0]
        y = points[:, 1]
        A = np.vstack([x, np.ones(len(x))]).T
        k, b = np.linalg.lstsq(A, y, rcond=None)[0]
        return k, b

    # 点と直線の距離を計算
    def distance_point_to_line(self, point, line):
        k, b = line
        x, y = point
        d = abs(k * x - y + b) / np.sqrt(k ** 2 + 1)
        return d

    def ransac(self,points, iterations=100, threshold=0.2):
        best_model = None
        best_inliers = []
        for i in range(iterations):
            if len(points) <= 2:
                return 0, []

            if len(points) >= 4:
                maybe_inliers = np.random.choice(len(points), 3, replace=False)
            else:
                maybe_inliers = np.random.choice(len(points), 2, replace=False)
            maybe_model = self.fit_line(points[maybe_inliers])
            also_inliers = []
            for point in points:
                if point not in maybe_inliers:
                    if self.distance_point_to_line(point, maybe_model) < threshold:
                        also_inliers.append(point)
            if len(also_inliers) > len(best_inliers):
                best_model = self.fit_line(np.array(also_inliers))
                best_inliers = also_inliers

            print("inlier = ", best_inliers)
            # best_modelがNoneの場合の例外処理を追加
            if best_model is None:
                return 0, []

            return best_model, best_inliers

    #model, inliers = ransac(data, iterations=100, threshold=0.1)


    def matching(self, pts, tree: spatial.KDTree):
        loop_cd = 10
        pts_shifted = np.copy(pts)
        dis, idxes = tree.query(pts_shifted, k=1) # HDマップ付近の画像検出点
        while loop_cd > 0:
            # if all matched distance > threshold
            # if min(dis) > LANE_FIX_DIS_TH: return None, None
            valid = dis < LANE_FIX_DIS_TH
            if np.sum(valid) == 0: return None, None
            offset = np.average(tree.data[idxes[valid], 1] - pts_shifted[valid, 1], axis=0)
            pts_shifted[:, 1] += offset
            # if offset did not change
            if np.abs(offset).max() < 0.01: break
            dis, idxes = tree.query(pts_shifted, k=1)
            loop_cd -= 1
        return tree.data[idxes], dis < LANE_FIX_DIS_TH

    @staticmethod
    def solve_tR_t(tP, tQ):
        """
        R * P + t = Q --> solve R, t
        step 1)
            R * (P - P.ave) = Q - Q.ave
            as, R * P' = Q --> solve R'
        step 2) let,
            a = cos - i * sin
            b = cos + i * sin
        step 3)	diagonalize,
            R = S^-1 * J * S
            J = [[a, 0],[0, b]]
            S = [[i, 1],[-i, 1]]
        step 4) solve cos, sin
        step 5) solve t
            t = Q.ave - R * P.ave
        step X)
            take transpose for every step
        :param tP: observe points
        :param tQ: real points
        :return:
            tR:
            t: offset of observe points
        """
        tP_mean = tP.mean(axis=0)
        tQ_mean = tQ.mean(axis=0)
        tP = tP - tP_mean
        tQ = tQ - tQ_mean

        S = np.asarray([[complex(0, 1), 1], [complex(0, -1), 1]])
        A = np.dot(S, tP.transpose())
        B = np.dot(S, tQ.transpose())
        a = np.average(B[0] / A[0])
        b = np.average(B[1] / A[1])

        cos = (b + a).real
        sin = (b - a).imag
        r = math.sqrt(cos * cos + sin * sin)  # theoretically, r == 2

        cos /= r
        sin /= r

        tR = np.asarray([[cos, sin], [-sin, cos]])
        t = tQ_mean - np.dot(tP_mean, tR)

        return tR, t

    def solve_T(self, pts_det, pts_org):
        """
        T * P = Q --> solve T
        :param pts_det: detected lane points, as tP
        :param pts_org: HD map lane points, as tQ
        :return:
            T: as [[R, t], [0, 1]]
        """
        lps_tree = spatial.KDTree(pts_org[:, :2], leafsize=3)
        loop_cd = 20
        T = np.eye(3)
        model, pts1 = self.ransac(pts_det[:, :2], iterations=100, threshold=0.3)
        #pts1 = np.copy(pts_det[:, :2])
        while loop_cd > 0 and len(pts1) > 2:
            # match the nearest line points
            lane_pts, valid = self.matching(pts1, lps_tree)
            if lane_pts is None: return None

            # solve R, t
            tR, offset = self.solve_tR_t(np.array(pts1)[valid], np.array(lane_pts)[valid])
            if tR is None: return None

            # update detect points
            pts1 = np.dot(pts1, tR) + offset

            # update T
            Ti = np.eye(3)
            Ti[:2, :2] = tR.transpose()
            Ti[0, 2] = offset[0]
            Ti[1, 2] = offset[1]
            T = np.dot(Ti, T)
            loop_cd -= 1

            # if angle is small enough
            angle = abs(math.atan2(tR[0, 1], tR[0, 0]))
            if angle < 0.01: break
        return T

    def point_distance_line(self, point, line_point1, line_point2):
        vec1 = line_point1 - point
        vec2 = line_point2 - point
        distance = np.abs(np.cross(vec1, vec2)) / np.linalg.norm(line_point1 - line_point2)
        return distance

    def fix(self, pts_det, pts_org):
        lane = {}
        for i, point in enumerate(pts_org):
            if pts_org[i][2] not in lane.keys():
                lane[pts_org[i][2]] = [point]
            else:
                lane[pts_org[i][2]].append(point)
        lane_info = []
        for key in lane.keys():
            lane_info.append([len(lane[key]), key])
        lane_info = np.sort(lane_info, axis = 0)
        angle = {}
        left_lane_org = -1
        right_lane_org = -1
        for i, group_id in enumerate(lane_info):  # assume that we only 2 lane in HD map, right and left
            group_id = group_id[1]
            center = np.asarray(lane[group_id]).mean(axis=0)
            center = np.asarray([center[0], center[1], 0])

            angle[group_id] = math.atan2(center[1], center[0])
            if i > 0:
                if self.point_distance_line(lane[group_id][0][:2], lane[old_group_id][0][:2], lane[old_group_id][-1][:2]) > 1.5: # 同じ側のグループを避ける
                    right_lane_org = old_group_id if angle[group_id] > angle[old_group_id] else group_id
                    left_lane_org = group_id if angle[group_id] > angle[old_group_id] else old_group_id
                    print('right_lane_org:', right_lane_org)
                    print('left_lane_org:', left_lane_org)
                    break
                else: continue
            old_group_id = group_id

        # decide which lane from image is right lane
        # left_lane_det = pts_det[pts_det[:, 2] == True]
        # right_lane_det = pts_det[pts_det[:, 2] == False]
        print(lane.keys())

        # T_l = None
        # T_r = None

        # if left_lane_org != -1:
        #     T_l = self.solve_T(left_lane_det, np.asarray(lane[left_lane_org]))
        # if right_lane_org != -1:
        #     T_r = self.solve_T(right_lane_det, np.asarray(lane[right_lane_org]))
        # if T_l is not None:
        #     T = T_l
        # else:
        #     T = T_r

        T = self.solve_T(pts_det, pts_org)

        return T


    def run(self, lanes, frame: Frame, ipe: InitialPoseEstimator):
        self._flag = False
        if lanes is None: return False
        lane_pts_HD = lanes

        # calculate ground rect view by camera
        R_v2c = ipe.get_R_v2c()
        height = ipe.get_height()
        self.setup_perspective(R_v2c, height, frame.camera)

        printLog("LaneFix", f"trimming ground ..")
        # get ground part image, and is uv_pos in org image
        img_org, roi = trimming(frame.img, self.pts_i)
        if img_org is None: return False  # something was wrong ..
        # ---------- debug: cut ground ----------
        if DEBUG:
            tmp_img = np.copy(frame.img)
            tmp_img = cv2.rectangle(tmp_img, (roi[0], roi[1]), (roi[2], roi[3]), (20, 220, 20), 2)
            rect_pts = np.asarray(self.pts_i, np.int32)
            tmp_img = cv2.polylines(tmp_img, [rect_pts], True, [0, 100, 255], 2)
            cv2.imwrite(folder_LaneFix + f"/tr_ground/{frame.t:.3f}.jpg", tmp_img)
        # ---------------------------------------
        self.pts_i -= (roi[0], roi[1])  # offset points
        img_h, img_w, _ = img_org.shape

        # get perspective matrix
        persMat_i2g = cv2.getPerspectiveTransform(self.pts_i, self.pts_g)
        # ---------- debug: perspective ground ----------
        if DEBUG:
            dst = cv2.warpPerspective(img_org, persMat_i2g, self.size_g)
            cv2.imwrite(folder_LaneFix + f"/perspective/{frame.t:.3f}.jpg", dst)
        # -----------------------------------------------

        # create lane detector and detect
        printLog("LaneFix", f"new LaneDetector start ..")
        laneDetector = LaneDetector(persMat_i2g, (img_w, img_h), self.size_g)
        lane_pts_i = laneDetector.run(img_org)  # whatever happened, we got lane points
        if lane_pts_i is None:
            printLog("LaneFix", f"no lane detected ..")
            return False
        printLog("LaneFix", f"got detect lane points total = {len(lane_pts_i)}")
        # ---------- debug: lane detect result ----------
        if DEBUG:
            view = np.copy(frame.img)
            view[roi[1]:roi[3], roi[0]:roi[2], :] = laneDetector.orig_lane

            _, w, _ = view.shape
            view[:108, w - 212:] = (50, 0, 0)
            cv2.putText(view, "white", (w - 200, 125), cv2.FONT_HERSHEY_PLAIN, 1.2, (100, 30, 0), 1)
            cv2.putText(view, "yellow", (w - 90, 125), cv2.FONT_HERSHEY_PLAIN, 1.2, (100, 30, 0), 1)
            if laneDetector.num_white > 0:
                view[4:104, w - 208:w - 108] = laneDetector.sum_white / laneDetector.num_white
            else:
                view[4:104, w - 208:w - 108] = 0
            if laneDetector.num_yellow > 0:
                view[4:104, w - 104:w - 4] = laneDetector.sum_yellow / laneDetector.num_yellow
            else:
                view[4:104, w - 104:w - 4] = 0

            h1 = frame.img.shape[0]
            h2 = laneDetector.size_g[1]
            fx = h1 / h2 * 1.25
            fy = h1 / h2 * 0.85
            sub_view = cv2.resize(laneDetector.pers_lane, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
            h, w, _ = sub_view.shape
            view[:h, :w, :] = sub_view
            cv2.imwrite(folder_LaneFix + f"/result/{frame.t:.3f}.jpg", view)
        # -----------------------------------------------

        # re-project lane points back to ground(vehicle) 2D-system as (x-axis, z-axis)
        persMat_i2v = cv2.getPerspectiveTransform(self.pts_i, self.pts_v)
        left_mask = np.where(lane_pts_i[:, 0] < (1280 / 2), True, False)
        #lane_pts_det = perspective(persMat_i2v, lane_pts_i)  # bravo
        #lane_pts_det = np.c_[lane_pts_det, left_mask]

        #ミリ波レーダーシンボル取り込み
        # RADAR_GNSS_OFFSET = np.array([0.45, -0.37])
        RADAR_GNSS_OFFSET = np.array([-0.2, 0.5])
        mwr_symbol = frame.mwr_symbol
        mwr_symbol_xy = mwr_symbol[['M_Xrange[m]', 'M_Yrange[m]']].to_numpy()
        mwr_symbol_xy = (np.asarray([[0, 1], [-1, 0]]) @ mwr_symbol_xy.T).T
        #取付角45deg
        ## mwr_symbol_xy = (np.array([[np.cos(np.radians(45)), -np.sin(np.radians(45))],
        ##                                    [np.sin(np.radians(45)), np.cos(np.radians(45))]]) @ mwr_symbol_xy.T).T
        #取付角90deg
        mwr_symbol_xy = (np.array([[np.cos(np.radians(90)), -np.sin(np.radians(90))],
                                   [np.sin(np.radians(90)), np.cos(np.radians(90))]]) @ mwr_symbol_xy.T).T
        mwr_symbol_xy = mwr_symbol_xy + RADAR_GNSS_OFFSET
        # mwr_symbol_xy = (mwr_symbol_xy@ mwr_symbol_xy.T).T
        mwr_symbol_xy = (np.asarray([[0, 1], [-1, 0]]) @ mwr_symbol_xy.T).T
        #mwr_symbol_xy = -(np.asarray([[-1, 0], [0, -1]]) @ mwr_symbol_xy.T).T

        #mwr_symbol_IDxy_array = np.hstack((mwr_symbol_xy, mwr_symbol['ID'].to_numpy()[:, None]))
        lane_pts_det = np.hstack((mwr_symbol_xy, mwr_symbol['ID'].to_numpy()[:, None]))
        # ---------- debug: match result ----------
        if DEBUG:
            from MMSProbe.utils.Visualization import Canvas
            canvas = Canvas()
            canvas.draw_points(lanes, para=dict(
                color="crimson", marker=".", s=5, label="HD Map Lane"
            ))
            canvas.draw_points(np.concatenate((self.pts_v, [[0, 0]])), para=dict(
                marker="+", s=100, c="white", lw=1,
            ))
            canvas.draw_points(lane_pts_det, para=dict(
                color="gray", marker=".", s=10, label="Detected Lane"
            ))

            canvas.set_axis(sci_on=False, legend_on=True, legend_loc="lower left")
            canvas.save(folder_LaneFix + f"/solveT/{frame.t:.3f}_in.jpg")
            canvas.close()
        # -----------------------------------------
        # lane_pts_det[:, 0] = lane_pts_det[:, 0] + CAMERA_GNSS_OFFSET[0]
        # lane_pts_det[:, 1] = lane_pts_det[:, 1] + CAMERA_GNSS_OFFSET[1]
        # calc diff matrix
        T = self.fix(lane_pts_det, lane_pts_HD)
        printLog("LaneFix", f"solve T = {np.ravel(T)[:-3]}")
        if T is None: return False

        # ---------- debug: fixed result ----------！

        if DEBUG:
            fixed_lanes = np.asarray(np.dot(lanes[:, :2] - T[:2, 2].T, T[:2, :2]))

            from MMSProbe.utils.Visualization import Canvas
            canvas = Canvas()
            canvas.draw_points(np.concatenate((self.pts_v, [[0, 0]])), para=dict(
                marker="+", s=100, c="white", lw=1,
            ))
            canvas.draw_points(lane_pts_det, para=dict(
                color="gray", marker=".", s=10, label="Detected Lane"
            ))
            canvas.draw_points(lanes, para=dict(
                color="crimson", marker=".", s=5, label="HD Map Lane"
            ))
            canvas.draw_points(fixed_lanes, para=dict(
                color="yellow", marker=".", s=5, label="Fixed Lanes"
            ))
            canvas.set_axis(sci_on=False, legend_on=True, legend_loc="lower left")
            canvas.save(folder_LaneFix + f"/solveT/{frame.t:.3f}_out.jpg")
            canvas.close()
        #-----------------------------------------

        self._T = T
        self._flag = True
        return True

    def is_succeed(self):
        return self._flag

    def get_result_Body_Frame(self):
        dx = self._T[0, 2]
        dy = self._T[1, 2]
        dpsi = math.atan2(-self._T[0, 1], self._T[0, 0]) # -はいるぞ
        return dx, dy, dpsi

