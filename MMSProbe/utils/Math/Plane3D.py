import numpy as np

from MMSProbe.utils.Common import num2str
from MMSProbe.utils.Math.Matrix import gaussian_elimination
from MMSProbe.utils.Math.plane_fitting import fit_plane


class Plane3D:
	"""
	F(X) = (x, y, z) * v^t = d
	"""

	def __init__(self, vector, distance):
		"""
		:param vector: vertical vector to plane
		:param distance: zero to plane
		"""
		self.v = np.array(vector, np.float64)
		self.d = float(distance)

	def __abs__(self):
		if self.d < 0:
			self.v *= -1
			self.d *= -1
		return self

	def __str__(self):
		tmp_str = "Plane3D object \n"
		tmp_str += f"    vector = ({num2str(self.v, 4)}) \n"
		tmp_str += f"    distance = {num2str(self.d, 4)} \n"
		return tmp_str

	# ---------- initial functions ----------
	@classmethod
	def init_pv(cls, point, vector):
		"""
		:param point:
		:param vector: vertical vector to plane
		:return:
		"""
		p = np.array(point, np.float64)
		v = np.array(vector, np.float64)
		v /= np.linalg.norm(v)
		return abs(cls(v, np.dot(p, v)))

	@classmethod
	def init_pn(cls, points):
		"""
		:param points: (dim, dim) points
		:return:
		"""
		points = np.atleast_2d(points)
		N, dim = points.shape
		X = points[:dim] - points[0]
		v = gaussian_elimination(X)
		v /= np.linalg.norm(v)
		return abs(cls(v, np.dot(points[0], v)))

	# ---------- fitting functions ----------
	@classmethod
	def fitting(cls, points, method: str = "lsm"):
		"""
		:param points:
		:param method: {"lsm", "pca", "ransac", ..}
		:return:
		"""
		flag, v, d = fit_plane(points, fit_method = method)
		if flag: return cls(v, d)
		return cls.init_pn(points)
