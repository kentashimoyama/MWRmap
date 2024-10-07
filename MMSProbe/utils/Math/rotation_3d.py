import math

import numpy as np

from MMSProbe.utils.Math.Quaternion import Quat

"""
- memo:
	. type of system
		1. Euler
			rot: (roll, pitch, yaw) radians
			order: (list of axis ids) roll -> pitch -> yaw
				=> rotation order is necessary
		2. Rotation
			mat: 3x3 rotation matrix
				=> most useful for calculation
		3. Quaternion
			quat: normalized Quaternion
				=> main used in interpolation
				=> need to normalize for return
		4. AxisAngle
			vector: (vx, vy, vz) normalized
			angle: radians
				=> Warming: this part of code still unstable ..
	. basic conversion roles
+--------------------------------------------+
|                                            |
|       ┌─────────> Euler o─────────┐        |
|       │                           │        |
|       v                           v        |
|    Rotation <───────────────> Quaternion   |
|       ^                           ^        |
|       │                           │        |
|       └───────o AxisAngle <───────┘        |
|                                            |
+--------------------------------------------+
- support: "┌", "└", "┐", "┘", "─", "│", "├", "┤", "┬", "┴", "┼",

author@Kai3645 left: 
	For fastest performers, input parameters logic will not be checked.
	(ex. quat was not be normalized)
"""


# --------------------> conversion <--------------------

def Euler_Rotation(rot: np.ndarray, *, order: tuple = (0, 1, 2)):
	"""
	default = Rz(yaw) * Ry(pitch) * Rx(roll)
	:param: rot: (roll, pitch, yaw) radians
	:param: order: (list of axis ids) roll -> pitch -> yaw
	:return: mat: 3x3 rotation matrix
	"""

	def AngleAxis(angle, idx: int):
		"""
		:param: angle: radians
		:param: idx: 0->x, 1->y, 2->z
		:return: 3x3 rotation matrix
		"""
		cos = math.cos(angle)
		sin = math.sin(angle)
		row = (idx + 1) % 3
		clm = (idx + 2) % 3
		mat = np.identity(3)
		mat[row, row] = cos
		mat[row, clm] = -sin
		mat[clm, row] = sin
		mat[clm, clm] = cos
		return np.asmatrix(mat)

	return AngleAxis(rot[2], order[2]) * AngleAxis(rot[1], order[1]) * AngleAxis(rot[0], order[0])


def Euler_Quaternion(rot: np.ndarray, *, order: tuple = (0, 1, 2)):
	"""
	:param: rot: (roll, pitch, yaw) radians
	:param: order: (list of axis ids) roll -> pitch -> yaw
	:return: quat: normalized Quaternion
	"""

	def AngleAxis(angle, idx: int):
		"""
		:param: angle: radians
		:param: idx: 0->x, 1->y, 2->z
		:return: Quaternion
		"""
		angle *= 0.5
		v = np.zeros(3)
		v[idx] = math.sin(angle)
		w = math.cos(angle)
		return Quat.init_by_wv(w, v)

	quat = AngleAxis(rot[2], order[2]) * AngleAxis(rot[1], order[1]) * AngleAxis(rot[0], order[0])
	return abs(quat).normalization()


def Rotation_Euler(mat: np.matrix, *, order: tuple = (0, 1, 2)):
	"""
	:param: mat: 3x3 rotation matrix
	:param: order: (list of axis ids) roll -> pitch -> yaw
	:return: rot: (roll, pitch, yaw) radians
	"""

	def norm(x: float, y: float):
		return math.sqrt(x * x + y * y)

	sign = np.asmatrix("0 -1 1; 1 0 -1; -1, 1, 0")[order[2], order[0]]

	return np.asarray([
		math.atan2(-sign * mat[order[2], order[1]], mat[order[2], order[2]]),
		math.atan2(sign * mat[order[2], order[0]], norm(mat[order[2], order[1]],
		                                                mat[order[2], order[2]])),
		math.atan2(-sign * mat[order[1], order[0]], mat[order[0], order[0]]),
	], np.float64)


def Rotation_Quaternion(mat: np.matrix):
	"""
	:param: mat: 3x3 rotation matrix
	:return: quat: normalized Quaternion
	"""
	w = math.sqrt(max(0, 1 + mat[0, 0] + mat[1, 1] + mat[2, 2])) * 0.5
	x = math.sqrt(max(0, 1 + mat[0, 0] - mat[1, 1] - mat[2, 2])) * 0.5
	y = math.sqrt(max(0, 1 - mat[0, 0] + mat[1, 1] - mat[2, 2])) * 0.5
	z = math.sqrt(max(0, 1 - mat[0, 0] - mat[1, 1] + mat[2, 2])) * 0.5
	if mat[2, 1] - mat[1, 2] < 0: x = -x
	if mat[0, 2] - mat[2, 0] < 0: y = -y
	if mat[1, 0] - mat[0, 1] < 0: z = -z
	return abs(Quat(w, x, y, z)).normalization()


def Quaternion_Rotation(quat: Quat):
	"""
	:param: quat: normalized Quaternion
	:return: mat: 3x3 rotation matrix
	"""
	mat = np.eye(3)
	mat[0, 0] -= (quat.y * quat.y + quat.z * quat.z) * 2
	mat[1, 1] -= (quat.x * quat.x + quat.z * quat.z) * 2
	mat[2, 2] -= (quat.x * quat.x + quat.y * quat.y) * 2
	mat[0, 1] = (quat.x * quat.y - quat.w * quat.z) * 2
	mat[1, 0] = (quat.x * quat.y + quat.w * quat.z) * 2
	mat[0, 2] = (quat.x * quat.z + quat.w * quat.y) * 2
	mat[2, 0] = (quat.x * quat.z - quat.w * quat.y) * 2
	mat[1, 2] = (quat.y * quat.z - quat.w * quat.x) * 2
	mat[2, 1] = (quat.y * quat.z + quat.w * quat.x) * 2
	return np.asmatrix(mat)


def Quaternion_AxisAngle(quat: Quat):
	"""
	when w == 1 (no rotation happened), return (0, 0, 1), 0 as default
	:param quat: normalized Quaternion
	:return:
		vector: (vx, vy, vz) normalized
		angle: radians
	"""
	if quat.w == 1: return np.asarray((0., 0., 1.)), 0.
	angle = 2 * math.acos(quat.w)
	den = math.sqrt(1 - quat.w * quat.w)
	vector = np.asarray([quat.x / den, quat.y / den, quat.z / den])
	vector /= np.linalg.norm(vector)
	return vector, angle


def AxisAngle_Rotation(vector: np.ndarray, angle: float):
	"""
	:param: vector: (vx, vy, vz) normalized
	:param: angle: radians
	:return: mat: 3x3 rotation matrix
	"""
	c = math.cos(angle)
	s = math.sin(angle)
	# vector /= np.linalg.norm(vector)
	sign = np.asmatrix([
		[0, -vector[2], vector[1]],
		[vector[2], 0, -vector[0]],
		[-vector[1], vector[0], 0]
	], np.float64)
	x = np.asmatrix(vector)
	return c * np.eye(3) + (1 - c) * x.transpose() * x + s * sign


def AxisAngle_Quaternion(vector: np.ndarray, angle: float):
	"""
	:param vector: (vx, vy, vz) normalized
	:param angle: radians
	:return: quat: normalized Quaternion
	"""
	a = angle * 0.5
	v = vector * math.sin(a)
	w = math.cos(a)
	return abs(Quat.init_by_wv(w, v)).normalization()


# indirect
def Euler_AxisAngle(rot: np.ndarray, *, order: tuple = (0, 1, 2)):
	"""
	Euler -> Quaternion -> AxisAngle
	:param rot: (roll, pitch, yaw) radians
	:param order: (list of axis ids) roll -> pitch -> yaw
	:return:
		vector: (vx, vy, vz) normalized
		angle: radians
	"""
	quat = Euler_Quaternion(rot, order = order)
	return Quaternion_AxisAngle(quat)


# indirect
def Rotation_AxisAngle(mat: np.matrix):
	"""
	Rotation -> Quaternion -> AxisAngle
	:param mat: 3x3 rotation matrix
	:return:
		vector: (vx, vy, vz) normalized
		angle: radians
	"""
	quat = Rotation_Quaternion(mat)
	return Quaternion_AxisAngle(quat)


# indirect
def Quaternion_Euler(quat: Quat, *, order: tuple = (0, 1, 2)):
	"""
	Quaternion -> Rotation -> Euler
	:param: quat: normalized Quaternion
	:param: order: (list of axis ids) roll -> pitch -> yaw
	:return: rot: (roll, pitch, yaw) radians
	"""
	mat = Quaternion_Rotation(quat)
	return Rotation_Euler(mat, order = order)


# indirect
def AxisAngle_Euler(vector: np.ndarray, angle: float, *, order: tuple = (0, 1, 2)):
	"""
	AxisAngle -> Rotation -> Euler
	:param vector: (vx, vy, vz) normalized
	:param angle: radians
	:param order: (list of axis ids) roll -> pitch -> yaw
	:return: rot: (roll, pitch, yaw) radians
	"""
	mat = AxisAngle_Rotation(vector, angle)
	return Rotation_Euler(mat, order = order)


# --------------------> interpolation <--------------------

def Euler_Interpolate(Rots, idxes1, idxes2, t, *, order: tuple = (0, 1, 2)):
	"""
	:param: Rot: basic Euler array [rad]
	:param: idxes1: left indexes integer id-array
	:param: idxes2: right indexes integer id-array
	:param: t: required scaling 1d-array in [0, 1]
	:param: order: roll -> pitch -> yaw
	:return:
	"""
	Qs = np.asarray([Euler_Quaternion(r, order = order) for r in Rots])
	Qs = list(map(Quat.slerp, Qs[idxes1], Qs[idxes2], t))
	return np.asarray([Quaternion_Euler(q, order = order) for q in Qs])

# todo: 1. add analyze funcs
# todo: 2. add test funcs for debug
