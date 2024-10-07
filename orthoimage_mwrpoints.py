#セマンティックセグメンテーションによる道路領域or壁領域の推定→道路領域部分をオルソ化?(いずれにせよ壁部分だけ切り出すことが必要，そのために最も合理的かつ必然的な手法とは?? 各手法の短所を補い合っている必要がある)→MWR点群と重ねて高さ方向点群を削除

import cv2
import numpy as np
import glob
import csv
import os
import copy
import math
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta
from mpl_toolkits.mplot3d import Axes3D

from MMSProbe.core import *

# DEBUG = False  # debug switch

# # Shi-Tomasi corner detection para
# FEATURE_PARA = dict(maxCorners = 500, qualityLevel = 0.01, minDistance = 7, blockSize = 5)
# # Lucas-Kanade optical flow para
# FEATURE_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.3)
# LK_PARA = dict(winSize = (15, 15), maxLevel = 5, criteria = FEATURE_CRITERIA)
# TH_KEYPOINT_NUM = 12  # the min of keypoints number, officially, 5 is enough
# TH_HOMOGRAPHY_DIS = 10  # [pixel], the min distance of Homography distance
# TH_KEYPOINT_DIS = 2  # [pixel], the min distance of keypoints distance

# # 二フレーム間の回転行列を求める
# class Frame:
# 	def __init__(self, time: float, x: float, y: float, image , camera: Camera): #Camera(コンストラクタ?)の形は??
# 		self.t = time
# 		self.x = x
# 		self.y = y

# 		self.camera = camera
# 		# self.mwr_symbol = mwr_symbol
# 		self.img = camera.imread(image)
# 		self.img_mini = camera.resize(self.img)
# 		gray = cv2.cvtColor(self.img_mini, cv2.COLOR_BGR2GRAY)
# 		self.gray_mini = cv2.equalizeHist(gray)
# 		pass

# 	def __str__(self):
# 		return f"Frame({self.t:.3f}, {self.x:.3f}, {self.y:.3f}, img{self.img.shape}, img_mini{self.img_mini.shape})"

# 	def pos(self):
# 		return self.x, self.y

# 	def pose_estimate(self, other): #otherの形はインスタンス。
# 		"""
# 		:param other:2枚目の画像
# 		:return: flag, R, t
# 		"""
# 		# attention, somehow, the points return by opencv always be (N, 1, dim), be careful
		
#         # 画像から特徴点の検出
# 		pts1 = cv2.goodFeaturesToTrack(self.gray_mini, mask = None, **FEATURE_PARA)
# 		if len(pts1) < TH_KEYPOINT_NUM: return False, None, None
		
#         # 画像特徴点の追跡 最初の画像の特徴点をother.grayminiに追跡
# 		pts2, status12, _ = cv2.calcOpticalFlowPyrLK(self.gray_mini, other.gray_mini, pts1, None, **LK_PARA) #2枚の画像間の特徴点の移動ベクトルを計算 返り値：次の画像上の特徴点の新しい位置、各特徴点の追跡ステータス(1:成功、0:失敗), 各特徴点の追跡誤差
# 		ptsR, status2R, _ = cv2.calcOpticalFlowPyrLK(other.gray_mini, self.gray_mini, pts2, None, **LK_PARA)

# 		# select by opencv method
# 		# 有効な追跡点の選択　再投影誤差(status12, status2Rのことか)が一定の閾値(おそらく閾値0)以上のもののみ削除 追跡点の数が一定の閾値未満のものも採用しない
# 		valid_LK = np.logical_and(status12 > 0, status2R > 0)
# 		if np.sum(valid_LK) < TH_KEYPOINT_NUM: return False, None, None
# 		pts1 = pts1[valid_LK].reshape(-1, 2) #1枚目特徴点
# 		ptsR = ptsR[valid_LK].reshape(-1, 2) #1枚目→2枚目→1枚目に再投影された特徴点

# 		# select by matching (like re-projection)
# 		# 再投影誤差による選択　
# 		proj_drift = np.abs(pts1 - ptsR).max(1) #1枚目の特徴点と再投影された特徴点の最大誤差
# 		valid_proj = proj_drift < 1 #誤差1ピクセル以内の特徴点をプール
# 		if np.sum(valid_proj) < TH_KEYPOINT_NUM: return False, None, None
# 		pts2 = pts2[valid_LK].reshape(-1, 2)[valid_proj] #2枚目の特徴点
# 		pts1 = pts1[valid_proj] #1枚目の特徴点

# 		# select by Homography projection
# 		H, _ = cv2.findHomography(pts1, pts2, cv2.RHO) #
# 		add = np.ones((len(pts1), 1))
# 		pts1_ = np.concatenate((pts1, add), axis = 1).transpose()
# 		pts2_ = np.asarray(np.asmatrix(H) * pts1_).transpose()
# 		pts2_ = pts2_[:, :2] / np.tile(pts2_[:, 2], (2, 1)).transpose()
# 		valid_H = np.linalg.norm(pts2_ - pts2, axis = 1).ravel() < TH_HOMOGRAPHY_DIS
# 		if np.sum(valid_H) < TH_KEYPOINT_NUM: return False, None, None
# 		pts1 = pts1[valid_H]
# 		pts2 = pts2[valid_H]

# 		# select by distance (remove if not moved) 
# 		# 特徴点の移動距離による選択
# 		diff = np.abs(pts1 - pts2).max(1)
# 		valid_diff = diff > TH_KEYPOINT_DIS #一定距離以上の特徴点のみ保存
# 		if np.sum(valid_diff) < TH_KEYPOINT_NUM: return False, None, None
# 		pts1 = pts1[valid_diff]
# 		pts2 = pts2[valid_diff]

# 		# ---------- debug ----------
# 		if DEBUG:
# 			folder_frame = ""
# 			print(f"save image \'lk_match/{self.t:.3f}.jpg\'")
# 			dst = np.copy(self.img_mini)
# 			for p1, p2 in zip(pts1, pts2):
# 				p1 = tuple(p1.astype(int))
# 				p2 = tuple(p2.astype(int))
# 				dst = cv2.circle(dst, p1, 1, (139, 0, 0), 2)
# 				dst = cv2.line(dst, p1, p2, (214, 112, 218), 2)
# 			cv2.imwrite(folder_frame + f"/lk_match/{self.t:.3f}.jpg", dst)
# 		# ---------------------------

# 		mat = self.camera.mat_mini
# 		# looks like RANSAC performs good here
# 		E, _ = cv2.findEssentialMat(pts1, pts2, mat, method = cv2.RANSAC, prob = 0.99, threshold = 1.0)
# 		ret, R, t, _ = cv2.recoverPose(E, pts1, pts2, mat)  # the results are c_other to c_this
# 		if ret > 0:  # successes, and adjust to our calculation system
# 			return True, R.T, t.ravel() * -1
# 		return False, None, None, pts1, pts2
	



# # # from MMSProbe.utils.Common import printLog
# # from MMSProbe.conf import Config

# # # ========== constant from Config
# # INIT_CAM_PARA = Config.camera_para
# # INIT_CAM_DIST = Config.undistort_dist
# # FLAG_UNDISTORT = Config.undistort_flag
# # IMG_SIZE = Config.image_size
# # IMG_RESIZE_W = Config.image_resize_width
# # RESIZE_RATE = IMG_RESIZE_W / IMG_SIZE[0]
# # IMG_RESIZE_H = int(RESIZE_RATE * IMG_SIZE[1])

# # # カメラパラメータ あとでキャリブレーションもする。いったん保留。
# # class Camera:
# # 	def __init__(self):
# # 		"""
# # 		for fast performance, origin image will be resized in to 'mini' size
# # 		"""
# # 		fx, fy, cx, cy = INIT_CAM_PARA
# # 		self.imgSize = w, h = IMG_SIZE
# # 		self.imgSize_mini = (IMG_RESIZE_W, IMG_RESIZE_H)

# # 		# image undistort part
# # 		mat = np.asarray([[fx, 0., cx],
# # 		                  [0., fy, cy],
# # 		                  [0., 0., 1.]], dtype = np.float32)
# # 		self._dist = None  # dist coefficient, only used for undistort
# # 		self._oldMat = None  # initiated camera matrix
# # 		self.mat = None  # new camera matrix after optimal
# # 		self.mat_mini = None  # camera matrix for mini size

# # 		if FLAG_UNDISTORT:
# # 			dist = np.zeros(14)
# # 			dist[:len(INIT_CAM_DIST)] = INIT_CAM_DIST
# # 			new_mat, _ = cv2.getOptimalNewCameraMatrix(mat, dist, (w, h), 0)
# # 			self._dist = dist
# # 			self._oldMat = mat
# # 			self.mat = new_mat
# # 			self.mat_mini = new_mat * RESIZE_RATE
# # 		else:
# # 			self.mat = mat
# # 			self.mat_mini = mat * RESIZE_RATE
# # 		print("Camera")
# # 		pass


# class InitialPoseEstimator:
# 	def __init__(self):
# 		self._v2c = np.asmatrix("0 0 1; 0 -1 0; 1 0 0", np.float32)  # base matrix v->c
# 		self._c2v = np.asmatrix("0 0 1; 0 -1 0; 1 0 0", np.float32)  # base matrix c->v
# 		self._R_v2c = Euler_Rotation(INIT_SYS_ROT, order = (0, 2, 1)) * self._v2c # カメラ行列

# 	def update_data(self, R, t):
# 		v, a = Rotation_AxisAngle(R)
# 		# mathematically explanation
# 		# give a > 1.0 [deg] => math.exp(-a * a * 25000) ~ 0
# 		# when vehicle is turning, vector t shall be unreliable
# 		x_sample = t * math.exp(-a * a * 20000)  # todo: find best weight
# 		# similarly, when vehicle is turning, vector v shall be reliable
# 		y_sample = v * (math.exp(a) - 1)  # todo: find best weight
# 		self.sample_pts_x = np.vstack((self.sample_pts_x, x_sample))
# 		self.sample_pts_y = np.vstack((self.sample_pts_y, y_sample))
# 		self.trigger_cd -= 1
# 		pass

# 	def update_R_v2c(self):
# 		"""
# 		calculate axis by fix points distance
# 		:return:
# 		"""
# 		if self.trigger_cd > 0: return False

# 		def fix_points(P, v):
# 			"""
# 			in short, let outliers disappear
# 			:return:
# 			"""
# 			K = np.dot(P, v)
# 			S = np.tile(K, (3, 1)).transpose() * v - P
# 			D = np.linalg.norm(S, axis = 1)
# 			K = 1 - np.exp(-D * D * 20)
# 			P_ = np.tile(K, (3, 1)).transpose() * S + P
# 			return P_

# 		R_v2c_tmp = self._R_v2c

# 		if self.is_pose_stable:
# 			vx = np.asarray(R_v2c_tmp[0]).ravel()
# 			vy = np.asarray(R_v2c_tmp[1]).ravel()
# 			pts_x = fix_points(self.sample_pts_x, vx)
# 			pts_y = fix_points(self.sample_pts_y, vy)
# 		else:
# 			pts_x = np.copy(self.sample_pts_x)
# 			pts_y = np.copy(self.sample_pts_y)

# 		err_th = 2e-4
# 		loop_th = 20
# 		for _ in range(loop_th):
# 			# solve vector axis-x
# 			x_zz = np.dot(pts_x[:, 2], pts_x[:, 2])
# 			if x_zz > 0:
# 				x_xz = np.dot(pts_x[:, 0], pts_x[:, 2]) / x_zz
# 				x_yz = np.dot(pts_x[:, 1], pts_x[:, 2]) / x_zz
# 				r = math.sqrt(x_xz * x_xz + x_yz * x_yz + 1)
# 				vx = np.asarray([x_xz, x_yz, 1.]) / r
# 			else: vx = [0., 0., 1.]
# 			# solve vector axis-y
# 			y_yy = np.dot(pts_y[:, 1], pts_y[:, 1])
# 			if y_yy > 0:
# 				y_xy = np.dot(pts_y[:, 0], pts_y[:, 1]) / y_yy
# 				y_zy = np.dot(pts_y[:, 2], pts_y[:, 1]) / y_yy
# 				r = math.sqrt(y_xy * y_xy + y_zy * y_zy + 1)
# 				vy = np.asarray([y_xy, 1., y_zy]) / -r
# 			else: vy = [0., -1., 0.]
# 			# update axis
# 			# vx -= np.dot(vx, vy) * vy
# 			# vx /= np.linalg.norm(vx)
# 			vy -= np.dot(vy, vx) * vx
# 			vy /= np.linalg.norm(vy)
# 			vz = np.cross(vx, vy)
# 			vz /= np.linalg.norm(vz)
# 			R_v2c_new = np.asmatrix([
# 				[vx[0], vx[1], vx[2]],
# 				[vy[0], vy[1], vy[2]],
# 				[vz[0], vz[1], vz[2]],
# 			])
# 			_, a = Rotation_AxisAngle(R_v2c_new.T * R_v2c_tmp)
# 			R_v2c_tmp = R_v2c_new
# 			if a < err_th: break
# 			pts_x = fix_points(self.sample_pts_x, vx)
# 			pts_y = fix_points(self.sample_pts_y, vy)

# 		R_c2c_ = R_v2c_tmp.transpose() * self._R_v2c
# 		_, a = Rotation_AxisAngle(R_c2c_)
# 		self.change_angle = self.change_angle * 0.5 + a
# 		quat_old = Rotation_Quaternion(self._R_v2c)
# 		quat_new = Rotation_Quaternion(R_v2c_tmp)
# 		quat_mid = Quat.slerp(quat_old, quat_new, 0.8)
# 		self._R_v2c = Quaternion_Rotation(quat_mid)
# 		# ---------- debug ----------
# 		self.log_angle.append(a)
# 		self.sys_rot_list.append(self.recover_sys_rot())
# 		# ---------------------------
# 		if self.change_angle > TH_STABLE_ANGLE: return False
# 		self.is_pose_stable = True

# 		return True

# 	def update(self, R, t):
# 		self.update_data(R, t)
# 		self.update_R_v2c()

def extract_number(file_path):
    # ファイル名を取得
    file_name = os.path.basename(file_path)
    # 拡張子を除いたファイル名を取得
    name, _ = os.path.splitext(file_name)
    # 数字として取得
    return float(name)

#画像の取得時刻のaqlocフレームを探し, 速度を算出
def synchro_aqloc(input_file, image_pre_time, image_curr_time):
	pre_min_diff = float("inf")
	curr_min_diff = float("inf")
	jst = timezone(timedelta(hours=9))
	with open(input_file, "r", encoding="utf-8-sig") as f:
		csv_reader = csv.reader(f)
		for row in csv_reader:
			aqloc_time = float(row[0])+90000
			# print(f"image_pre:{image_pre_time}")
			# print(f"aqloc_time:{aqloc_time}")
			# print(f"image_curr:{image_curr_time}")
			pre_diff = abs(aqloc_time - image_pre_time)
			curr_diff = abs(aqloc_time - image_curr_time)
			if pre_diff < pre_min_diff:
				pre_min_diff = pre_diff
				pre_timerow = row
				# print(f"premindiff:{pre_min_diff}")
				# print(f"prediff:{pre_diff}")
			if curr_diff < curr_min_diff:
				curr_min_diff = curr_diff
				curr_timerow = row
				# print(f"curr_min_diff:{curr_min_diff}")
				# print(f"currdiff:{curr_diff}")
	
	delta_t = float(curr_timerow[0])-float(pre_timerow[0])
	pre_x = float(pre_timerow[2])
	pre_y = float(pre_timerow[1])
	pre_theta = float(pre_timerow[3])

	curr_x = float(curr_timerow[2])
	curr_y = float(curr_timerow[1])
	curr_theta = float(curr_timerow[3])
	distance = math.sqrt((curr_x - pre_x)**2 + (curr_y - pre_y)**2)
	# print(f"delta_t:{delta_t}")
	# print(f"pre_x:{pre_x}")
	# print(f"curr_x:{curr_x}")
	# print(f"pre_y:{pre_y}")
	# print(f"curr_y:{curr_y}")
	# print(f"distance:{distance}")
	# v = distance/delta_t

	return distance, curr_x, curr_y, curr_theta


def rotation_matrix_to_euler_angles(R): #zyx回転行列：(x→y→z軸(roll, pitch, yaw)の順番で回転を行う行列.)に適応するようの関数
    """
    回転行列 R からオイラー角 (x, y, z) を算出する関数
    R: 3x3の回転行列
    戻り値: (roll, pitch, yaw) のタプル
    """
    # assert R.shape == (3, 3), "R must be a 3x3 matrix"
    
    # y軸回りの回転 (pitch)
    yaw = np.degrees(np.arcsin(-R[2, 0]))
    
    # if np.cos(pitch) > 1e-6:  # pitchが90度でない場合
    pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))  # x軸回りの回転
    roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))   # z軸回りの回転
    # else:  # pitchが90度の場合
    #     roll = np.arctan2(-R[0, 1], R[1, 1])  # x軸回りの回転
    #     yaw = 0

    return roll, pitch, yaw


def rotation_vector_to_euler_angles(rotation_matrix):
    # 回転行列からオイラー角を計算
    sy = np.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])
    
    singular = sy < 1e-6  # 特異点のチェック
    if not singular:
        # print(rotation_matrix[2, 1], rotation_matrix[2, 2])
        x_angle = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y_angle = np.arctan2(-rotation_matrix[2, 0], sy)
        z_angle = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x_angle = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y_angle = np.arctan2(-rotation_matrix[2, 0], sy)
        z_angle = 0

    # ラジアンを度に変換
    x_angle = np.degrees(x_angle)
    y_angle = np.degrees(y_angle)
    z_angle = np.degrees(z_angle)

    return [x_angle, y_angle, z_angle]

def cal_angle(points_3d_scaled, pts2):
	#モーションステレオから車両姿勢の推定		
	#世界座標
	points = []
	for point_3d_x, point_3d_y, point_3d_z in zip(points_3d_scaled[0], points_3d_scaled[1], points_3d_scaled[2]):
		points.append([point_3d_x,point_3d_y, point_3d_z])
	points = np.array(points)
	
	if int(points.shape[0]) < 12:
		print("points num under 12")
		return None

	#画像座標
	img_points = np.array(pts2)

	print(points.shape, img_points.shape)

	# カメラの内部パラメータ（例: カメラ行列）
	camera_matrix = np.array([[616.5263, 0, 651.1589],
							[0, 617.2315, 376.0408],
							[0, 0, 1]], dtype=np.float32)

	# # 歪み係数（例: 0に設定）
	dist_coeffs = np.zeros((4, 1))  # [k1, k2, p1, p2]

	# PnP法による姿勢推定
	success, rotation_vector, translation_vector = cv2.solvePnP(points, img_points, camera_matrix, dist_coeffs)

	if success:
		# 回転ベクトルを行列に変換
		rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
		
		# 結果の表示
		print("Rotation Vector:\n", rotation_vector)
		print("Translation Vector:\n", translation_vector)
		print("Rotation Matrix:\n", rotation_matrix)
		pnpangle = rotation_vector_to_euler_angles(rotation_matrix)
		print("オイラー角 (度):", pnpangle)

		return pnpangle
	else:
		print("姿勢推定に失敗しました。")

		return None


# # 関数test用 例: 90度回転の回転行列
# angle = np.radians(90)  # 90度をラジアンに変換
# R = np.array([ #roll
#     [np.cos(angle), -np.sin(angle), 0],
#     [np.sin(angle), np.cos(angle), 0],
#     [0, 0, 1]
# ])

# R = np.array([ #yaw
#     [np.cos(angle), 0, np.sin(angle)],
# 	[0,1,0],
#     [-np.sin(angle), 0, np.cos(angle)]
# ])

# R = np.array([ #pitch
#     [1, 0, 0],
# 	[0, np.cos(angle), -np.sin(angle)],
#     [0, np.sin(angle), np.cos(angle)]
# ])

# roll, pitch, yaw = rotation_matrix_to_euler_angles(R)
# print("Roll (x-axis):", np.degrees(roll))
# print("Pitch (y-axis):", np.degrees(pitch))
# print("Yaw (z-axis):", np.degrees(yaw))
# exit()

## mwr+画像の複合
CAMERA = Camera() #新しくつくる、オルソ化機能, カメラパラの推定保存機能＆画像座標から世界座標を算出する機能をつける, オルソ化画像上の特徴点座標の計算機能, 
vo = VisualOdometry()
ipe = InitialPoseEstimator()
debug = False
debug_rpy = False

# img_dir = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\image0909\image0909-4\image0909-4/" #1枚目
# img_dir = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\image0907\image0907_1_2_correct/"
img_dir = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\image0907\image0907_1_2/"
# img_dir = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\image0907\image0907_2_2/"
mwr_file = ""
aqloc_file = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\aqloc\20240907_1_latlon19.csv"
# aqloc_file = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\aqloc\20240907_2_latlon19.csv"
img_files = glob.glob(img_dir + "*.jpg")
curr_img = None
curr_R = None
x = 0 #temp
y = 0 #temp
wcoord_list = [[], [], []]
camera_rpy = [[], [], [], []]
pnpangle_list = []

img_files = sorted(img_files, key=extract_number)

for id, img_file in enumerate(img_files):
	if id < 29:
		continue
	print(f"processing img{id+1} {img_file} ...")

	img = cv2.imread(img_file)

	#連続する画像1, 画像2から特徴点、対応点を抽出
	
	#画像撮影時の時間の読み込み
	time = img_file.split("\\")[-1].split(".")[0]
	time = float(time.split("_")[-1])

	if curr_img is None:
		curr_img = img
		curr_time = time
		continue
	else:
		pre_img = curr_img
		curr_img = img
		pre_time = curr_time
		curr_time = time 
	
	frame_1 = Frame(pre_time, x, y, pre_img, CAMERA) #変数からmwr_symbolを削除。必要???
	frame_2 = Frame(curr_time, x, y, curr_img, CAMERA)
	

	flag, R, t, pts1, pts2 = frame_1.pose_estimate(frame_2)	#画像の成功フラッグ、1→2のカメラの回転、1→2のカメラの並進, 特徴点1, 特徴点2
	if flag == False:
		print("There are no matching points")
		continue
	
	if curr_R is None:
		t = t[:, np.newaxis]
		curr_R = R
		curr_t = t
		continue
	else:
		t = t[:, np.newaxis]
		pre_R = curr_R
		curr_R = R
		pre_t = curr_t	
		curr_t = t



	#奥行情報の算出
	#2つのカメラ投影行列を構築
	# ipe.update(pre_R, pre_t)
	# R_v2c_pre = ipe.get_R_v2c() #1フレーム目カメラ外部パラメータ行列:カメラの回転、並進を表す
	# ipe.update(pre_R, pre_t)
	# R_v2c_curr = ipe.get_R_v2c() #2フレーム目カメラ外部パラメータ行列:カメラの回転、並進を表す
	R_v2c_pre = np.hstack((pre_R, pre_t))
	R_v2c_curr= np.hstack((curr_R, curr_t))

	# # 特徴点の座標を正規化
	# ## やり方1
	# K = CAMERA.mat #カメラ内部パラメータ
	# pts1_norm = cv2.undistortPoints(np.expand_dims(pts1, axis=1), K, None) #1フレーム目特徴点
	# pts2_norm = cv2.undistortPoints(np.expand_dims(pts2, axis=1), K, None) #2フレーム目特徴点

	# # 三角測量による3D再構成
	# points_4d_hom = cv2.triangulatePoints(R_v2c_pre, R_v2c_curr, pts1_norm, pts2_norm) #三角測量、同時座標系で出力
	# points_3d = points_4d_hom[:3] / points_4d_hom[3]  # 同次座標をデカルト座標に変換 ,4次元目のスケールファクターで除す, [[x1, x2, ...], [y1, y2, ...], [z1, z2, ...]] 

	## やり方2 　やり方1とそんなに変わらん
	# 内部パラメータ行列（例として）
	K = CAMERA.mat #カメラ内部パラメータ

	# 投影行列の構築
	P1 = np.dot(K , R_v2c_pre)
	P2 = np.dot(K, R_v2c_curr)

	# 画像の一部分のみの対応点の抽出
	# pt1_list = []
	# pt2_list = []
	# for pt1, pt2 in zip(pts1, pts2):
	# 	if pt1[0] <= 600: #550
	# 		if pt1[0] >= 500:
	# 			if pt2[0] <= 600: #550
	# 				if pt2[0] >= 500:
	# 					pt1_list.append(pt1)
	# 					pt2_list.append(pt2)
	# pts1 = np.array(pt1_list)
	# pts2 = np.array(pt2_list)

	distance, currpos_w_x, currpos_w_y, curr_theta = synchro_aqloc(aqloc_file, pre_time, curr_time) #二フレーム間の速度算出
	
	# 三角測量の実行 
	pts1_2 = pts1[:, np.newaxis, :]
	pts2_2 = pts2[:, np.newaxis, :]
	
	points_4d_hom = cv2.triangulatePoints(P1, P2, pts1_2, pts2_2)	
	points_3d = points_4d_hom[:3] / points_4d_hom[3]  # 同次座標をデカルト座標に変換 ,4次元目のスケールファクターで除す, [[x1, x2, ...], [y1, y2, ...], [z1, z2, ...]] 
	# カメラ間の実際の距離（基線長）
	baseline = distance
	# baseline = math.sqrt((curr_t[0, 0])**2 + (curr_t[1, 0])**2 + (curr_t[2, 0])**2)

	# スケールの調整
	points_3d_scaled_pre = points_3d * baseline

	#カメラ座標系→世界座標系に座標系を変換する(xyz座標系順番に記述すると, points_3dの3つ目インデックスが左向き, 1つめインデックスが後ろ向き, 2つめインデックスが上向き　→　1つ目インデックスが右向き, 2つめインデックスが前向き, 3つめインデックスが上向き)
	points_3d_scaled = [-points_3d_scaled_pre[2], -points_3d_scaled_pre[0], points_3d_scaled_pre[1]]

	#世界座標への変換
	# sin_theta = math.cos(curr_theta)
	# cos_theta = math.sin(curr_theta)
	# w_x = currpos_w_x + (points_3d_scaled[0]) * cos_theta - (points_3d_scaled[1]) * sin_theta #調整をかける, x 
	# w_y = currpos_w_y + (points_3d_scaled[0]) * sin_theta + (points_3d_scaled[1]) * cos_theta #調整をかける, y
	sin_theta = math.sin(-curr_theta)
	cos_theta = math.cos(-curr_theta)
	w_x = currpos_w_x + (points_3d_scaled[0]) * cos_theta - (points_3d_scaled[1]) * sin_theta #調整をかける, x 
	w_y = currpos_w_y + (points_3d_scaled[0]) * sin_theta + (points_3d_scaled[1]) * cos_theta #調整をかける, y
	w_z = 0 + points_3d_scaled[2]
	# w_x = points_3d_scaled[0]+currpos_w_x
	# w_y = points_3d_scaled[1]+currpos_w_y
	# w_z = points_3d_scaled[2]+0 #9系のz座標は???????????????????????????????????????????????????????????????????????????????????????
	for x, y, z in zip(w_x, w_y, w_z):
		wcoord_list[0].append(x)
		wcoord_list[1].append(y)
		wcoord_list[2].append(z)

	# #モーションステレオから車両姿勢の推定		
	# #世界座標
	# points = []
	# for point_3d_x, point_3d_y, point_3d_z in zip(points_3d_scaled[0], points_3d_scaled[1], points_3d_scaled[2]):
	# 	points.append([point_3d_x,point_3d_y, point_3d_z])
	# points = np.array(points)

	# #画像座標
	# img_points = np.array(pts2)

	# print(points.shape, img_points.shape)

	# # カメラの内部パラメータ（例: カメラ行列）
	# camera_matrix = np.array([[616.5263, 0, 651.1589],
	# 						[0, 617.2315, 376.0408],
	# 						[0, 0, 1]], dtype=np.float32)

	# # # 歪み係数（例: 0に設定）
	# dist_coeffs = np.zeros((4, 1))  # [k1, k2, p1, p2]

	# # PnP法による姿勢推定
	# success, rotation_vector, translation_vector = cv2.solvePnP(points, img_points, camera_matrix, dist_coeffs)

	# if success:
	# 	# 回転ベクトルを行列に変換
	# 	rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
		
	# 	# 結果の表示
	# 	print("Rotation Vector:\n", rotation_vector)
	# 	print("Translation Vector:\n", translation_vector)
	# 	print("Rotation Matrix:\n", rotation_matrix)
	# 	pnpangle = rotation_vector_to_euler_angles(rotation_matrix)
	# 	print("オイラー角 (度):", pnpangle)
	# 	pnpangle_list.append(pnpangle)
	# else:
	# 	print("姿勢推定に失敗しました。")
			
	# モーションステレオから車両姿勢の推定	
	pnpangle = cal_angle(points_3d_scaled, pts2)
	pnpangle_list.append(pnpangle[0], pnpangle_list[1], pnpangle_list[2])


	if debug_rpy:
		print(f"matcing points at image 1:{pts1}")
		print(f"matcing points at image 2:{pts2}")
		print(f"points_3d_scaled:{points_3d_scaled}")
		print(f"R_v2c_pre:{pre_R}")
		print(f"R_v2c_curr:{curr_R}")
		print(f"distance:{baseline}")
		print(f"len:{len(points_3d_scaled[0])}")
		print()
		
		roll, pitch, yaw = rotation_matrix_to_euler_angles(R_v2c_curr)

		print(f"roll:{roll}")
		print(f"pitch:{pitch}")
		print(f"yaw:{yaw}")

		camera_rpy[0].append(roll)
		camera_rpy[1].append(pitch)
		camera_rpy[2].append(yaw)
		camera_rpy[3].append(time)

	if debug:
		print(f"matcing points at image 1:{pts1}")
		print(f"matcing points at image 2:{pts2}")
		print(f"points_3d_scaled:{points_3d_scaled}")
		print(f"R_v2c_pre:{pre_R}")
		print(f"R_v2c_curr:{curr_R}")
		print(f"distance:{baseline}")
		print(f"len:{len(points_3d_scaled[0])}")

		# 対応点を描画
		copy_pre_img = copy.deepcopy(pre_img)
		copy_curr_img = copy.deepcopy(curr_img)
		for idx, (pt1, pt2) in enumerate(zip(pts1, pts2)):
			x1, y1 = int(pt1[0]), int(pt1[1])
			cv2.circle(copy_pre_img, (x1, y1), 10, (0,0,255), -1)
			x2, y2 = int(pt2[0]), int(pt2[1])
			cv2.circle(copy_curr_img, (x2,y2), 10, (0,0,255), -1)
		
		cv2.imshow("pre_"+str(img_file), copy_pre_img) #対応点をプロットした画像
		cv2.imshow("curr", copy_curr_img)
		cv2.waitKey(0)
		# cv2.imwrite(r"C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\imagepoint\pre_img_"+str(id)+".jpg", copy_pre_img)
		# cv2.imwrite(r"C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\imagepoint\curr_img_"+str(id)+".jpg", copy_curr_img)

		for idx, (pt1, pt2) in enumerate(zip(pts1, pts2)):
			copy_pre_img2 = copy.deepcopy(pre_img)
			copy_curr_img2 = copy.deepcopy(curr_img)
			x1, y1 = int(pt1[0]), int(pt1[1])
			cv2.circle(copy_pre_img2, (x1, y1), 10, (0,0,255), -1)
			x2, y2 = int(pt2[0]), int(pt2[1])
			cv2.circle(copy_curr_img2, (x2, y2), 10, (0,0,255), -1)
			cv2.imshow("image1", copy_pre_img2)
			cv2.imshow("image2", copy_curr_img2)
			cv2.waitKey(0)

			# x, y, z座標をそれぞれのリストに分解
			x = points_3d_scaled[0]
			y = points_3d_scaled[1]
			z = points_3d_scaled[2]
			x_now = points_3d_scaled[0][idx]
			y_now = points_3d_scaled[1][idx]
			z_now = points_3d_scaled[2][idx]
			print(f"x_now, y_now, z_now: {x_now, y_now, z_now}")

			# 3Dプロットの設定
			fig = plt.figure()
			ax = fig.add_subplot(111, projection='3d')

			# 点をプロット
			ax.scatter(x, y, z, c='r', marker='o')
			ax.scatter(x_now, y_now, z_now, c="b", s=100)			
			# ax.scatter(-z, -x, y, c='r', marker='o')
			# ax.scatter(-z_now, -x_now, y_now, c="b", s=100)

			# 各軸の最大値を設定
			# ax.set_xlim(-80, -10)
			# ax.set_ylim(-20, 50)
			# ax.set_zlim(-90, -20)

			# ax.set_xlim(-100, 10)
			# ax.set_ylim(-40, 60)
			# ax.set_zlim(0, 200)

			# ax.set_xlim(-5, 4)
			# ax.set_ylim(-2, 4)
			# ax.set_zlim(-2, 5)

			# ラベルの設定
			ax.set_xlabel('X軸')
			ax.set_ylabel('Y軸')
			ax.set_zlabel('Z軸')

			# グラフを表示
			plt.show()

			# 2Dプロットの設定
			fig2, axes = plt.subplots(2, 2, tight_layout=True)

			# 1つ目のプロット
			# axes[0, 0].plot(x, y)
			axes[0, 0].scatter(x, y, c="r")
			axes[0, 0].scatter(x_now, y_now, c="b", s=100)
			axes[0, 0].set_xlabel('x (m)')
			axes[0, 0].set_ylabel('y (m)')
			axes[0, 0].set_xticks(np.arange(-40, 40, 5)) # x軸に100ずつ目盛り
			axes[0, 0].set_yticks(np.arange(-10, 10, 1)) 
			axes[0, 0].set_xlim(-30, 30)
			axes[0, 0].set_ylim(-10, 10) #軸の下限値と上限値設定
			axes[0, 0].grid(True)

			# 2つ目のプロット
			# axes[0, 1].plot(x, z)
			axes[0, 1].scatter(x, z, c="r")
			axes[0, 1].scatter(x_now, z_now, c="b", s=100)
			axes[0, 1].set_xlabel('x (m)')
			axes[0, 1].set_ylabel('z (m)')
			axes[0, 1].set_xticks(np.arange(-40, 40, 5))
			axes[0, 1].set_yticks(np.arange(-10, 10, 1))  # x軸に100ずつ目盛り
			axes[0, 1].set_xlim(-30, 30)
			axes[0, 1].set_ylim(-10, 10) #軸の下限値と上限値設定
			axes[0, 1].grid(True)

			# 3つ目のプロット
			# axes[1, 1].plot(y, z)
			axes[1, 1].scatter(y, z, c="r")
			axes[1, 1].scatter(y_now, z_now, c="b", s=100)
			axes[1, 1].set_xlabel('y (m)')
			axes[1, 1].set_ylabel('z (m)')
			axes[1, 1].set_xticks(np.arange(-20, 20, 5))
			axes[1, 1].set_yticks(np.arange(-10, 10, 1))  # x軸に100ずつ目盛り
			axes[1, 1].set_xlim(-20, 20)
			axes[1, 1].set_ylim(-10, 10) #軸の下限値と上限値設定
			axes[1, 1].grid(True)

			# 4つ目のプロットは空白のままにする
			axes[1, 0].axis('off')
			plt.show()

		
		print(f"pre frame matching points num:{len(pts1)}")
		print(f"curr frame matching points num:{len(pts2)}")
		# # x = points_3d_scaled[0]
		# y = points_3d_scaled[1]
		# z = points_3d_scaled[2]
		# fig = plt.figure()
		# ax = fig.add_subplot(111, projection='3d')
		# ax.scatter(x, y, z, c='r', marker='o')
		# # ax.set_xlim(-100, 10)
		# # ax.set_ylim(-40, 60)
		# # ax.set_zlim(0, 200)
		# ax.set_xlabel('X軸')
		# ax.set_ylabel('Y軸')
		# ax.set_zlabel('Z軸')
		# plt.show()
		# exit()

	#オルソ化する calc_wc_gen_ortho.pyを使う

	#オルソ化画像上の特徴点座標の計算 calc_wc_gen_ortho.pyを使う, 投影 過去コード使える??

	#オルソ化画像上の特徴点座標を世界座標に変換する計算 calc_wc_gen_ortho.pyを使う。
	
	# #世界座標を元にmwr点群とオルソ化点群を重ねる calc_wc_gen_ortho.pyを使う。		
	# with open(mwr_file, newline='', encoding='utf-8') as csvfile:
	# 	csv_reader = csv.reader(csvfile)
	# 	for row in csv_reader:
	# 		time = row[0] #何列目??
	# 		# x = row[]
	# 		# y = row[]

	# 		#時刻同期。同一時刻にとれたフレームを見つける. 
	# 		if time == cameratimeframe+3 or time == cameratimeframe-3:
	# 			continue
	
# 世界座標に変換した点群をCSVファイルに書き込む
with open(r'C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\imagepoint\w_coordinates_fromimage.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)

    # 各列を一行として書き込む
    for x, y, z in zip(wcoord_list[0], wcoord_list[1], wcoord_list[2]):
        if z >= 50:
            continue
        elif z <= -50:
            continue
        else:
            csvwriter.writerow([x, y, z])


#カメラ姿勢をcsvに書き込む.
with open(r'C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\imagepoint\yawfromms.csv', 'w', newline='') as csvfile:
	csvwriter = csv.writer(csvfile)

	# 各列を一行として書き込む
	for roll, pitch, yaw in zip(pnpangle_list[0], pnpangle_list[1], pnpangle_list[2]):
		csvwriter.writerow([roll, pitch, yaw])

if debug_rpy:
	#カメラ姿勢をcsvに書き込む.
	with open(r'C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\imagepoint\cameraroll_pitch_yaw.csv', 'w', newline='') as csvfile:
		csvwriter = csv.writer(csvfile)

		# 各列を一行として書き込む
		for roll, pitch, yaw, t in zip(camera_rpy[0], camera_rpy[1], camera_rpy[2], camera_rpy[3]):
			csvwriter.writerow([roll, pitch, yaw, t])

# 車両走行軌跡の読み込み
with open(r'C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\aqloc\20240907_1_latlon19.csv', 'r', encoding='utf-8_sig') as file:
# with open(r'C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\aqloc\122705_ichimill19.csv', 'r', encoding='utf-8_sig') as file:
    reader = csv.reader(file)
    
    # 配列を用意
    trajectory_x = []
    trajectory_y = []
    # 行を1つずつ読み込む
    for i, row in enumerate(reader):
        if float(row[0]) < 90640:
            continue
        elif float(row[0]) > 90726:
            continue
        trajectory_y.append(float(row[1]))
        trajectory_x.append(float(row[2]))

x = wcoord_list[0]
y = wcoord_list[1]
z = wcoord_list[2]
print(f"length x, y, z:{len(x), len(y), len(z)}")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.plot(trajectory_x, trajectory_y, marker='o', label='走行軌跡')
ax.scatter(x, y, z, c='r', marker='o')
ax.set_ylim(-32730, -32700)
ax.set_xlim(-10150, -10120)
ax.set_zlim(-60, 60)
ax.set_xlabel('X軸')
ax.set_ylabel('Y軸')
ax.set_zlabel('Z軸')
plt.show()



