import numpy as np

from MMSProbe.utils.Math.rotation_3d import Euler_Rotation, Rotation_Euler


class CoordSys:
	@staticmethod
	def conv(mat: np.matrix, pos):
		"""
		pos^ = Mat * pos^
		:param mat: 4x4 matrix
		:param pos: (x, y, z) * N
		:return: (x, y, z) * N
		"""
		pos = np.atleast_2d(pos)
		length = len(pos)

		add = np.ones((length, 1))
		pos = np.concatenate((pos, add), axis = 1)
		pos = np.asarray(pos * mat.transpose())[:, :3]
		if length > 1: return pos
		return pos.ravel()

	@staticmethod
	def convMat(pos, rot, *, order: tuple = (0, 1, 2)):
		"""
		:param pos: (x, y, z)
		:param rot: (roll, pitch, yaw)
		:param order: (list of axis ids) roll -> pitch -> yaw
		:return: 4x4 matrix
		"""
		rot = np.asarray(rot, np.float64)
		mat = np.asmatrix(np.eye(4))
		mat[:3, :3] = Euler_Rotation(rot, order = order)
		mat[:3, 3] = np.asmatrix(pos, np.float64).transpose()
		return mat

	@staticmethod
	def deconvMat(mat, *, order: tuple = (0, 1, 2)):
		"""
		:param mat: 4x4 matrix
		:param order: (list of axis ids) roll -> pitch -> yaw
		:return:
			pos: (x, y, z),
			rot: (roll, pitch, yaw)
		"""
		pos = np.asarray(mat[:3, 3], np.float64).ravel()
		rot = Rotation_Euler(mat[:3, :3], order = order)
		return pos, rot
