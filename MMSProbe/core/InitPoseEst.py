import math
import os

import numpy as np
from MMSProbe.utils.Math import Euler_Rotation, Rotation_AxisAngle, Rotation_Euler, Rotation_Quaternion, Quat, \
	Quaternion_Rotation
from MMSProbe.utils.Common.print_log import printLog
from MMSProbe.conf import Config

# ========== constant from Config
INIT_SYS_ROT = Config.ipe_init_rot_v2c
INIT_SYS_H = Config.ipe_init_sys_height
PATH_SAMPLE_PTS = Config.ipe_path_sample_pts
TH_STABLE_ANGLE = Config.th_pose_stable_angle

class InitialPoseEstimator:
	def __init__(self):
		self._v2c = np.asmatrix("0 0 1; 0 -1 0; 1 0 0", np.float32)  # base matrix v->c
		self._c2v = np.asmatrix("0 0 1; 0 -1 0; 1 0 0", np.float32)  # base matrix c->v
		self._R_v2c = Euler_Rotation(INIT_SYS_ROT, order = (0, 2, 1)) * self._v2c
		self._height = INIT_SYS_H
		printLog("IPE", f"set init rot = {INIT_SYS_ROT}")
		printLog("IPE", f"set init height = {INIT_SYS_H}")
		# flag paras
		self.is_pose_stable = False
		self.trigger_cd = 100  # count down for least time to be stable
		self.change_angle = 0

		# storage para
		self.sample_pts_x = np.zeros((0, 3))
		self.sample_pts_y = np.zeros((0, 3))
		printLog("IPE", f"sample points path \'{PATH_SAMPLE_PTS}\'")
		if PATH_SAMPLE_PTS is not None:
			self.read_sample_points(PATH_SAMPLE_PTS)
		printLog("IPE", f"sample points total = {len(self.sample_pts_x)}")
		self.log_angle = []
		self.sys_rot_list = []
		pass

	def __str__(self):
		tmp_str = "current initial pose is "
		if not self.is_pose_stable: tmp_str += "not "
		tmp_str += "stable ..\n"
		tmp_str += f"system initial rot = {np.rad2deg(self.recover_sys_rot()).round(3)}\n"
		tmp_str += f"system initial height = {self._height:.3f}\n"
		tmp_str += f"system change angle = {np.rad2deg(self.change_angle):.3f}"
		return tmp_str

	def read_sample_points(self, path: str):
		"""
		format: X_x, X_y, X_z, Y_x, Y_y, Y_z, ...
		:param path:
		:return:
		"""
		if not os.path.exists(path):
			printLog("IPE", f"can not read sample points, no such file @ \'{path}\'")
			return self
		pts_x = []
		pts_y = []
		fr = open(path, "r")
		for line in fr:
			row = line.split(",")
			pts_x.append(row[0:3])
			pts_y.append(row[3:6])
		fr.close()
		if len(pts_x) < 2:
			printLog("IPE", f"can not read sample points, empty file")
			return self  # empty file
		pts_x = np.asarray(pts_x, np.float32)
		pts_y = np.asarray(pts_y, np.float32)
		self.sample_pts_x = np.concatenate([self.sample_pts_x, pts_x])
		self.sample_pts_y = np.concatenate([self.sample_pts_y, pts_y])
		return self

	def write_sample_points(self, path: str):
		"""
		format: X_x, X_y, X_z, Y_x, Y_y, Y_z, ...
		:param path:
		:return:
		"""
		fw = open(path, "w")
		for X, Y in zip(self.sample_pts_x, self.sample_pts_y):
			X_x, X_y, X_z = X
			Y_x, Y_y, Y_z = Y
			fw.write(f"{X_x:.9f},{X_y:.9f},{X_z:.9f},{Y_x:.9f},{Y_y:.9f},{Y_z:.9f},\n")
		fw.close()

	def update_data(self, R, t):
		v, a = Rotation_AxisAngle(R)
		# mathematically explanation
		# give a > 1.0 [deg] => math.exp(-a * a * 25000) ~ 0
		# when vehicle is turning, vector t shall be unreliable
		x_sample = t * math.exp(-a * a * 20000)  # todo: find best weight
		# similarly, when vehicle is turning, vector v shall be reliable
		y_sample = v * (math.exp(a) - 1)  # todo: find best weight
		self.sample_pts_x = np.vstack((self.sample_pts_x, x_sample))
		self.sample_pts_y = np.vstack((self.sample_pts_y, y_sample))
		self.trigger_cd -= 1
		pass

	def update_R_v2c(self):
		"""
		calculate axis by fix points distance
		:return:
		"""
		if self.trigger_cd > 0: return False

		def fix_points(P, v):
			"""
			in short, let outliers disappear
			:return:
			"""
			K = np.dot(P, v)
			S = np.tile(K, (3, 1)).transpose() * v - P
			D = np.linalg.norm(S, axis = 1)
			K = 1 - np.exp(-D * D * 20)
			P_ = np.tile(K, (3, 1)).transpose() * S + P
			return P_

		R_v2c_tmp = self._R_v2c

		if self.is_pose_stable:
			vx = np.asarray(R_v2c_tmp[0]).ravel()
			vy = np.asarray(R_v2c_tmp[1]).ravel()
			pts_x = fix_points(self.sample_pts_x, vx)
			pts_y = fix_points(self.sample_pts_y, vy)
		else:
			pts_x = np.copy(self.sample_pts_x)
			pts_y = np.copy(self.sample_pts_y)

		err_th = 2e-4
		loop_th = 20
		for _ in range(loop_th):
			# solve vector axis-x
			x_zz = np.dot(pts_x[:, 2], pts_x[:, 2])
			if x_zz > 0:
				x_xz = np.dot(pts_x[:, 0], pts_x[:, 2]) / x_zz
				x_yz = np.dot(pts_x[:, 1], pts_x[:, 2]) / x_zz
				r = math.sqrt(x_xz * x_xz + x_yz * x_yz + 1)
				vx = np.asarray([x_xz, x_yz, 1.]) / r
			else: vx = [0., 0., 1.]
			# solve vector axis-y
			y_yy = np.dot(pts_y[:, 1], pts_y[:, 1])
			if y_yy > 0:
				y_xy = np.dot(pts_y[:, 0], pts_y[:, 1]) / y_yy
				y_zy = np.dot(pts_y[:, 2], pts_y[:, 1]) / y_yy
				r = math.sqrt(y_xy * y_xy + y_zy * y_zy + 1)
				vy = np.asarray([y_xy, 1., y_zy]) / -r
			else: vy = [0., -1., 0.]
			# update axis
			# vx -= np.dot(vx, vy) * vy
			# vx /= np.linalg.norm(vx)
			vy -= np.dot(vy, vx) * vx
			vy /= np.linalg.norm(vy)
			vz = np.cross(vx, vy)
			vz /= np.linalg.norm(vz)
			R_v2c_new = np.asmatrix([
				[vx[0], vx[1], vx[2]],
				[vy[0], vy[1], vy[2]],
				[vz[0], vz[1], vz[2]],
			])
			_, a = Rotation_AxisAngle(R_v2c_new.T * R_v2c_tmp)
			R_v2c_tmp = R_v2c_new
			if a < err_th: break
			pts_x = fix_points(self.sample_pts_x, vx)
			pts_y = fix_points(self.sample_pts_y, vy)

		R_c2c_ = R_v2c_tmp.transpose() * self._R_v2c
		_, a = Rotation_AxisAngle(R_c2c_)
		self.change_angle = self.change_angle * 0.5 + a
		quat_old = Rotation_Quaternion(self._R_v2c)
		quat_new = Rotation_Quaternion(R_v2c_tmp)
		quat_mid = Quat.slerp(quat_old, quat_new, 0.8)
		self._R_v2c = Quaternion_Rotation(quat_mid)
		# ---------- debug ----------
		self.log_angle.append(a)
		self.sys_rot_list.append(self.recover_sys_rot())
		# ---------------------------
		if self.change_angle > TH_STABLE_ANGLE: return False
		self.is_pose_stable = True

		return True

	def update(self, R, t):
		self.update_data(R, t)
		self.update_R_v2c()

	def is_good(self):
		return self.is_pose_stable

	def is_not_good(self):
		return not self.is_good()

	def get_R_v2c(self):
		return self._R_v2c

	def get_height(self):
		return self._height

	def recover_sys_rot(self):
		return Rotation_Euler(self._R_v2c * self._c2v, order = (0, 2, 1))