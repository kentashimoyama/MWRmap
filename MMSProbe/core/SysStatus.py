import math

import numpy as np
from datetime import datetime, timezone
from MMSProbe.utils.Common import printLog, unwrap_angle, folder_Status

TIME_FORMAT = "%y%m%d%H%M%S%f"
TIME_ZONE_INFO = timezone.utc

TH_GPS_DRIFT_DIS2 = 0.04 ** 2  # [m]
TH_TURNING_ANGLE = 0.0872665  # [rad/s], about 10 [deg/s]
TH_STRAIGHT_ANGLE = 0.0524  # [rad/s], about 3 [deg/s]
TH_MIX_ANGLE = 0.0698  # [rad/s], about 4 [deg/s]
TH_GPS_DRIFT_MAX_DIS = 1.4  # [m]
TH_GPS_JUMP_TIME = 1.0  # [s]
TH_GPS_JUMP_DIS2 = 16 ** 2  # [m]
# TH_GPS_JUMP_DIFF_PSI = 0.3501  # [rad] ~ 20 [deg]
TH_GPS_JUMP_DIFF_PSI = 1

# K = P[0] + P[1] / (1 + exp(-P[2] * (|a| - P[3])))
LOGIC_TRUST_PARA_GPS = (1, -0.9, 63, 0.0873)  # set inflection range (3, 7) [deg/s], decrease
LOGIC_TRUST_PARA_VO = (0, 0.8, 42, 0.1222)  # set inflection range (4, 10) [deg/s], increase
LOGIC_TRUST_PARA_FIX = (1, -1, 63, 0.0698)  # set inflection range (2, 6) [deg], decrease
LOGIC_TRUST_GPS_DRIFT = (1, -1, 2.2, 1.0)  # integral k * dx from [0, inf] ~ 1.4 [m]

class SysStatus:
	def __init__(self):
		# storage value for gps
		self.bias = np.zeros(6)
		self.dX_lane = np.zeros(3)
		self.dX_MWR = np.zeros(3)

		self.frame_prev = None
		self.frame_curr = None
		self.frame_next = None

		# state -> (t:0, x:1, y:2, psi:3, v:4, psid:5)
		self.state_prev = None
		self.state_curr = None
		self.state_next = None

		# storage value for ukf
		self.state_covar = None

		pass


	def set_state_next(self, t, x, y):
		self.state_next = np.zeros(6)
		self.state_next[0] = t
		self.state_next[1] = x
		self.state_next[2] = y

	def init_state(self):
		self.set_state_prev()

	def set_state_prev(self):
		"""
		(x, y) = F(t)
		         dt_1^2 * (F_2 - F_0) - dt_2^2 * (F_1 - F_0)
 		dF_0 = ----------------------------------------------- + o(2)
 		                dt_1^2 * dt_2 - dt_2^2 * dt_1
 		psi = atan2(dy, dx)
 		v = (dx^2 + dy^2)^0.5
		:return:
		"""
		dt_1 = self.frame_curr.t - self.frame_prev.t
		dt_2 = self.frame_next.t - self.frame_prev.t

		F_0 = np.asarray(self.frame_prev.pos())
		F_1 = np.asarray(self.frame_curr.pos())
		F_2 = np.asarray(self.frame_next.pos())

		dt_12 = dt_1 * dt_1
		dt_22 = dt_2 * dt_2
		den = dt_12 * dt_2 - dt_22 * dt_1
		mol = dt_12 * (F_2 - F_0) - dt_22 * (F_1 - F_0)
		dF_0 = mol / den

		dx, dy = dF_0
		self.state_prev[3] = math.atan2(dy, dx)
		self.state_prev[4] = math.sqrt(dx * dx + dy * dy)
		pass

	def set_state_curr(self):
		"""
		(x, y) = F(t)
		         dt_l^2 * (F_r - F_0) + dt_r^2 * (F_0 - F_l)
 		dF_0 = ----------------------------------------------- + o(2)
 		                dt_l^2 * dt_r + dt_r^2 * dt_l
 		psi = atan2(dy, dx)
 		v = (dx^2 + dy^2)^0.5
		:return:
		"""
		dt_l = self.frame_curr.t - self.frame_prev.t
		dt_r = self.frame_next.t - self.frame_curr.t

		F_l = np.asarray(self.frame_prev.pos())
		F_0 = np.asarray(self.frame_curr.pos())
		F_r = np.asarray(self.frame_next.pos())

		dt_l2 = dt_l * dt_l
		dt_r2 = dt_r * dt_r
		den = dt_l2 * dt_r + dt_r2 * dt_l
		mol = dt_l2 * (F_r - F_0) + dt_r2 * (F_0 - F_l)
		dF_0 = mol / den

		dx, dy = dF_0
		psi = math.atan2(dy, dx)
		print(dx)
		print(dy)
		print(psi)
		print(psi)
		self.state_curr[3] = unwrap_angle(psi, self.state_prev[3])
		self.state_curr[4] = math.sqrt(dx * dx + dy * dy)
		self.state_curr[5] = (self.state_curr[3] - self.state_prev[3])/(self.state_curr[0] - self.state_prev[0])

	def update_state_curr(self, x, y, psi, v):
		x0, y0 = self.state_prev[1:3]
		psi = unwrap_angle(psi, 0)
		self.state_curr[1:5] = x + x0, y + y0, psi, v

	def mix_angle_vo2gps(self, dpsi_vo):
		dpsi_gps = unwrap_angle(self.state_curr[3] - self.state_prev[3], 0)
		dt = self.state_curr[0] - self.state_prev[0]
		if abs(dpsi_gps - dpsi_vo) / dt > TH_MIX_ANGLE: return
		k_gps = self.sigmoid(LOGIC_TRUST_PARA_GPS, dpsi_gps / dt)
		k_vo = self.sigmoid(LOGIC_TRUST_PARA_VO, dpsi_vo / dt)
		dpsi = self.mix_angle(dpsi_gps, k_gps, dpsi_vo, k_vo)
		log_info = f"mix angle vo2gps {np.rad2deg([dpsi_gps, dpsi_vo]).round(4)}, "
		log_info += f"[{100 * k_gps:.1f}% : {100 * k_vo:.1f}%] -> {np.rad2deg(dpsi):.4f}"
		printLog("Status", log_info)
		self.state_curr[3] = self.state_prev[3] + dpsi

	@staticmethod
	def sigmoid(para, a):
		return para[0] + para[1] / (1 + math.exp(-para[2] * (abs(a) - para[3])))

	@staticmethod
	def mix_angle(angle1, k1, angle2, k2):
		"""
		:param angle1:
		:param k1:
		:param angle2:
		:param k2:
		:return:
		"""
		return (angle1 * k1 + angle2 * k2) / (k1 + k2)

	def is_sys_stop(self):
		# check gps distance
		dx = self.frame_curr.x - self.frame_prev.x
		dy = self.frame_curr.y - self.frame_prev.y
		printLog("Status", f"gps drift = {math.sqrt(dx * dx + dy * dy):.4f}")
		if dx * dx + dy * dy < TH_GPS_DRIFT_DIS2: return True
		# check frame difference
		# return self.frame_curr.is_same(self.frame_prev)
		return False

	def is_sys_teleport(self):
		# check gps time
		dt = self.frame_curr.t - self.frame_prev.t
		if dt > TH_GPS_JUMP_TIME: return True
		# check gps distance
		dx = self.frame_curr.x - self.frame_prev.x
		dy = self.frame_curr.y - self.frame_prev.y
		printLog("Status", f"gps drift = {math.sqrt(dx * dx + dy * dy):.4f}")
		if dx * dx + dy * dy > TH_GPS_JUMP_DIS2: return True
		return False

	def is_impossible_state(self):
		dpsi = self.state_curr[3] - self.state_prev[3]
		if abs(dpsi) > TH_GPS_JUMP_DIFF_PSI: return True
		return False

	def is_turning(self):
		dt = self.state_curr[0] - self.state_prev[0]
		dpsi = self.state_curr[3] - self.state_prev[3]
		dpsi = unwrap_angle(dpsi, 0) / dt
		printLog("Status", f"detect system dpsi = {np.rad2deg(dpsi):.3f}")
		if abs(dpsi) > TH_TURNING_ANGLE: return True
		return False

	# def is_straight(self):
	# 	dt = self.state_curr[0] - self.state_prev[0]
	# 	dpsi = self.state_curr[3] - self.state_prev[3]
	# 	dpsi = unwrap_angle(dpsi, 0) / dt
	# 	printLog("Status", f"detect system dpsi = {np.rad2deg(dpsi):.3f}")
	# 	if dpsi < TH_STRAIGHT_ANGLE: return True
	# 	return False

	def trans_body2global(self, bias_fusion, psi):
		'''
		バイアスをb'からgに変換するやつ
		'''
		t_x = bias_fusion[0]
		t_y = bias_fusion[1]
		print('psi', psi)
		cos = math.cos(psi)
		sin = math.sin(psi)
		dx = cos * t_x - sin * t_y
		dy = sin * t_x + cos * t_y
		return np.array([dx, dy, bias_fusion[2]])

	def fusion_bias_GNSS(self, bias_raw, cov, v, psi, psid):
		'''
		https://qiita.com/ReoNagai/items/b75a1eb6ee91118fb05f
		'''
		# cov_lane_inv = np.linalg.inv(cov[:3, :3])
		# cov_MWR_inv = np.linalg.inv(cov[3:, 3:])
		# cov_inv = np.linalg.inv(cov_lane_inv + cov_MWR_inv)
		# # bias = cov_inv @ (cov_lane_inv @ bias_raw[:3] + cov_MWR_inv @ bias_raw[3:])
		# bias_MWR = bias_raw[3:] # MWR
		# bias_lane = bias_raw[:3]  # lane
		# print('bias lane', bias_lane)
		bias_fusion = self.trans_body2global(bias_raw, psi)
		# print('bias fusion', bias)
		measurement = self.state_curr[1:4] + bias_fusion # これは合ってるはず
		measurement = np.append(measurement, v)
		return np.append(measurement, psid), bias_fusion, bias_raw


	def __str__(self):
		def state_str(s):
			if s is None: return "None"
			t, x, y, psi, v, psid = s
			return f"[{t:.3f}, {x:.3f}, {y:.3f}, {np.rad2deg(psi):.3f}, {v:.3f}]"

		tmp_str = "system status log:\n"
		tmp_str += f"state_prev = {state_str(self.state_prev)}\n"
		tmp_str += f"state_curr = {state_str(self.state_curr)}\n"
		tmp_str += f"state_next = {state_str(self.state_next)}\n"
		tmp_str += f"frame_prev = {self.frame_prev}\n"
		tmp_str += f"frame_curr = {self.frame_curr}\n"
		tmp_str += f"frame_next = {self.frame_next}\n"
		tmp_str += f"state_covar = {self.state_covar}"

		return tmp_str
