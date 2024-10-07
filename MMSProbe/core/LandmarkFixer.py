import math
import cv2
from munkres import Munkres
import numpy as np
from scipy import spatial
import pandas as pd

from MMSProbe.core import Camera, InitialPoseEstimator, Frame, LaneDetector
from MMSProbe.utils.Common import projection, perspective, trimming, printLog
from MMSProbe.utils.Common.debug_manager import folder_LaneFix
from MMSProbe.conf import Config
from MMSProbe.utils.Math.icp import icp

PIXEL_COE = Config.laneDet_pixel_coe
SIDE_FAR = Config.laneDet_side_far
FRONT_FAR = Config.laneDet_front_far
SIDE_FAR = Config.laneDet_side_far
FRONT_FAR_HALF = FRONT_FAR / 2
LANDMARK_DISTANCE_THRESHOLD = Config.landmark_distance_threshold
ALPHA = Config.matching_alpha
BETA = Config.matching_beta
sigmoid = lambda x: 1. / (1. + np.exp(ALPHA * (BETA - x)))
RADAR_GNSS_OFFSET = np.array([0.01, 0.20]) # GNSSアンテナを原点として，そこから見たRADARの位置
DEBUG = True


# 継承したほうが良い？
class LandmarkFixer:
    def __init__(self, landmarks):
        self.size_g = (int(2 * SIDE_FAR * PIXEL_COE), 0)
        tmp = np.zeros((len(landmarks), 4))
        tmp[:, 0:2] = -1
        self._landmark_ID = dict(zip(landmarks[:, 2], tmp))
        self.success_match_N = None
        # key: landmark ID value: [symbol ID, 最後にマッチングできたループ，マッチング出来たフレーム数，マッチングをロストしたフレーム数]
        # IDがfloatになってるけどまぁいいか
        self.pts_i = None
        self.pts_g = None
        self.pts_v = None
        self._T = None

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
        tmp_i = projection(camera.mat, tmp_c)  # 地面を画像に投影する

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

#ここから大きく変更を加えた箇所


    def fix_landmark(self, landmarks, mwr_symbol_IDxy_array):
        landmark_tree = spatial.KDTree(mwr_symbol_IDxy_array[:, 0:2], balanced_tree=False)
        c_symbols = []
        c_landmark = []
        for landmark in landmarks:
            indexes = landmark_tree.query_ball_point(landmark[0:2], LANDMARK_DISTANCE_THRESHOLD)
            if len(indexes) != 0:
                c_symbols.extend(mwr_symbol_IDxy_array[indexes])
                c_landmark.append(landmark)
        c_symbols = pd.DataFrame(c_symbols).drop_duplicates().values # 複数回でたやつを消す，ちょっと良くないやり方
        print('c_symbol: ', c_symbols)
        match = None
        score = None
        if len(c_landmark) >= 1:
            c_landmark = np.asarray(c_landmark)
            c_symbols = np.asarray(c_symbols)
            # cost = spatial.distance_matrix(c_landmark[:,0:2], c_symbols[:,0:2])
            score = self.matching_score(c_landmark, c_symbols)
            m = Munkres()
            score_cp = score.copy()
            if score_cp.shape[0]>score_cp.shape[1]:
                match = []
                match_ = m.compute(score_cp.T)
                for match_element in match_:
                    new_match_element = [match_element[1], match_element[0]]
                    match.append(new_match_element)
            else:
                match = m.compute(score_cp)
        return c_landmark, c_symbols, match, score

    def matching_score(self, c_landmark, c_symbols):
        '''
        scipy distance_matrixを参照
        '''
        c_landmark_xy = c_landmark[:, 0:2]
        m, k = c_landmark_xy.shape
        c_symbols_xy = c_symbols[:, 0:2]
        n, kk = c_symbols_xy.shape

        dist = np.empty((m, n), dtype=float)
        coeff = np.empty((m, n), dtype=float)
        for i, landmark in enumerate(c_landmark_xy):
            dist[i, :] = np.sum(np.abs(c_symbols_xy - landmark) ** 2, axis=-1) ** (1. / 2)  # 分散を求める必要があるな
            for j, symbol in enumerate(c_symbols):
                if symbol[2] == self._landmark_ID[c_landmark[i][2]][0]:
                    n = self._landmark_ID[c_landmark[i][2]][2] + 1
                    n_lost = self._landmark_ID[c_landmark[i][2]][3]
                else:
                    n = self._landmark_ID[c_landmark[i][2]][2]
                    n_lost = self._landmark_ID[c_landmark[i][2]][3] + 1
                coeff[i, j] = sigmoid((n - min(n, n_lost)))
        dist = 2.5 * np.exp(-dist ** 2/25)
        score = dist * coeff
        return 1 / score

    def compute_diff_landmark(self, c_landmark, c_symbols, match, score):
        pts_symbols = []
        pts_landmark = []
        s = []
        T = np.eye(3)
        # t = np.array([0, 0])
        if match:
            j = -1
            for i, landmark in enumerate(c_landmark):
                j += 1 # たまにランドマークに対応する物標がない場合があるので，そうするとmatchのところの番号が飛んでしまう．
                if i not in np.array(match)[:, 0]:
                    j -= 1
                    continue
                pts_symbols.append(c_symbols[match[j][1], 0:2])
                pts_landmark.append(landmark[0:2])
                s.append(score[i][match[j][1]])
            pts_landmark = np.asarray(pts_landmark)
            pts_symbols = np.asarray(pts_symbols)
            if pts_landmark.shape[0] == 1:
                s = np.asarray(s)
                s = s / sum(s)
                t = (s[:, None] * (pts_symbols - pts_landmark)).sum(axis=0)
                T[:2, 2] = t
            else:
                T, distances, iterations = icp(pts_landmark, pts_symbols, tolerance=0.000001)
        return T

    def run_MWR(self, landmark_from_GNSS, landmark_from_predicted, frame: Frame, ipe: InitialPoseEstimator, loop_count):
        R_v2c = ipe.get_R_v2c()
        height = ipe.get_height()
        self.setup_perspective(R_v2c, height, frame.camera)
        for landmark in landmark_from_predicted:
            if loop_count - self._landmark_ID[landmark[2]][1] >= 40:
                self._landmark_ID[landmark[2]] = [-1, -1, 0, 0]
        mwr_symbol = frame.mwr_symbol
        mwr_symbol_xy = mwr_symbol[['M_Xrange[m]', 'M_Yrange[m]']].to_numpy()
        mwr_symbol_xy = (np.asarray([[0, 1], [-1, 0]]) @ mwr_symbol_xy.T).T
        mwr_symbol_xy = mwr_symbol_xy + RADAR_GNSS_OFFSET
        mwr_symbol_IDxy_array = np.hstack((mwr_symbol_xy, mwr_symbol['ID'].to_numpy()[:, None]))
        c_landmark_predicted, c_symbols, match, score = self.fix_landmark(landmark_from_predicted,
                                                                          mwr_symbol_IDxy_array)  # c: corresponding
        # record = []
        # for i, _ in enumerate(c_landmark_predicted):
        #     if c_symbols[match[i][1], 2] not in record:
        #         record.append(c_symbols[match[i][1], 2])
        #     else:
        #         print(1)

        c_landmark_gnss = []
        self.success_match_N = len(c_symbols)
        for landmark in c_landmark_predicted:
            if landmark[2] in landmark_from_GNSS[:, 2]:
                c_landmark_gnss.append(landmark_from_GNSS[np.where(landmark_from_GNSS[:, 2] == landmark[2])[0][0]])

        if frame.t == 1645110554.919:
            print(1)
        if match:
            j = -1
            for i, landmark in enumerate(c_landmark_gnss):
                j += 1 # たまにランドマークに対応する物標がない場合があるので，そうするとmatchのところの番号が飛んでしまう．
                if i not in np.array(match)[:, 0]:
                    j -= 1
                    continue
                landmark_id = landmark[2]
                if self._landmark_ID[landmark_id][0] == -1:
                    self._landmark_ID[landmark_id] = [c_symbols[match[j][1], 2], loop_count, 1, 0]
                elif self._landmark_ID[landmark_id][0] == c_symbols[match[j][1], 2]:
                    self._landmark_ID[landmark_id][1] = loop_count
                    self._landmark_ID[landmark_id][2] += 1
                else:
                    self._landmark_ID[landmark_id][3] += 1
                # reset
                if self._landmark_ID[landmark_id][2] == self._landmark_ID[landmark_id][3]:
                    self._landmark_ID[landmark_id] = [-1, -1, 0, 0]

        if DEBUG:
            from MMSProbe.utils.Visualization import Canvas
            canvas = Canvas()
            canvas.draw_points(np.concatenate((self.pts_v, [[0, 0]])), para=dict(
                marker="+", s=100, c="white", lw=1,
            ))

            canvas.draw_points(landmark_from_GNSS, para=dict(
                color="green", marker=".", s=20, label="HD Map Landmark"
            ))
            canvas.draw_points(landmark_from_predicted, para=dict(
                color="red", marker=".", s=20, label="HD Map Landmark"
            ))
            for i, landmark in enumerate(landmark_from_GNSS):
                canvas.draw_text(landmark[0:2], str(landmark[2]), color="yellow", rotation=90)
                for _ in self._landmark_ID.keys():
                    if self._landmark_ID[landmark[2]][1] != -1:
                        text = str(self._landmark_ID[landmark[2]])
                        canvas.draw_text(landmark[0:2], text, color="green")
            canvas.draw_points(mwr_symbol_xy, para=dict(
                color="orange", marker=".", s=20, label="MWR Symbol"
            ))
            for i, symbol in enumerate(mwr_symbol_IDxy_array):
                canvas.draw_text(mwr_symbol_IDxy_array[i, 0:2], str(mwr_symbol_IDxy_array[i, 2]))

            if len(c_symbols):
                canvas.draw_points(c_symbols[:, 0:2], para=dict(
                    color="purple", marker=".", s=30, label="Corresponding MWR Symbol"
                ))

            if match:
                j = -1
                for i, landmark in enumerate(c_landmark_gnss):
                    j += 1  # たまにランドマークに対応する物標がない場合があるので，そうするとmatchのところの番号が飛んでしまう．
                    if i not in np.array(match)[:, 0]:
                        j -= 1
                        continue
                    canvas.draw_lines(np.asarray([landmark[0:2], c_symbols[match[j][1], 0:2]], dtype=object),
                                      para=dict(color="yellow"))

            canvas.set_axis(sci_on=False, legend_on=True, legend_loc="lower left")
            canvas.save(folder_LaneFix + f"/fixMWR/{frame.t:.3f}_in.jpg")
            canvas.close()

        T = self.compute_diff_landmark(c_landmark_gnss, c_symbols, match, score)
        self._T = T

        if DEBUG:
            from MMSProbe.utils.Visualization import Canvas
            canvas = Canvas()
            canvas.draw_points(np.concatenate((self.pts_v, [[0, 0]])), para=dict(
                marker="+", s=100, c="white", lw=1,
            ))
            if match:
                fixed_landmark_from_GNSS = np.asarray(
                    np.dot(T[:2, :2], np.array(landmark_from_GNSS)[:, 0:2][:, :2].T).T + T[:2, 2])
                for i, landmark in enumerate(fixed_landmark_from_GNSS):
                    canvas.draw_text(fixed_landmark_from_GNSS[i], str(landmark_from_GNSS[i][2]))
                    canvas.draw_points(fixed_landmark_from_GNSS[i], para=dict(
                        color="green", marker=".", s=20, label="HD Map Landmark"
                    ))
                canvas.draw_points(c_symbols, para=dict(
                    color="blue", marker=".", s=50, label="Symbols"
                ))
            canvas.set_axis(sci_on=False, legend_on=True, legend_loc="lower left")
            canvas.save(folder_LaneFix + f"/fixMWR/{frame.t:.3f}_out.jpg")
            canvas.close()
        return True

    # def get_result_MWR(self, psi):
    #     t_x = self._T[0][2]
    #     t_y = self._T[1][2]
    #     cos = math.cos(psi)
    #     sin = math.sin(psi)
    #     dx = cos * t_x - sin * t_y
    #     dy = sin * t_x + cos * t_y
    #     return dx, dy

    def get_result_MWR_Body_Frame(self):
        # なんか　RQ+t=PになってたんでRP+t=Qにするため逆変換
        R = self._T[:2,:2]
        dx = self._T[0, 2]
        dy = self._T[1, 2]
        t = np.array([dx, dy])
        t = -np.dot(R.T , t)
        dpsi = math.atan2(-R.T[0, 1], R.T[0, 0])
        return t[0], t[1], dpsi, self.success_match_N
