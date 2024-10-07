import numpy as np

from MMSProbe.utils.Common import num2str

class Line3D:
	"""
	(x, y, z) = f(t) = p + t * v
	for easily using, p and v are mutually perpendicular
	"""

	def __init__(self, point, vector):
		self.p = np.array(point, np.float64)
		self.v = np.array(vector, np.float64)
		self.normalize()

	def normalize(self):
		self.v /= np.linalg.norm(self.v)
		self.p -= np.dot(self.p, self.v) * self.v
		return self

	def __str__(self):
		tmp_str = "Line3D object \n"
		tmp_str += f"    point = ({num2str(self.p, 4)}) \n"
		tmp_str += f"    vector = ({num2str(self.v, 4)}) \n"
		return tmp_str

	# ---------- initial functions ----------
	@classmethod
	def init_p2(cls, point1, point2):
		p1 = np.asarray(point1)
		p2 = np.asarray(point2)
		line = cls(p1, p2 - p1)
		return line.normalize()

	# ---------- fitting functions ----------
	@classmethod
	def fitting(cls, points, method: str):
		"""
		:param points:
		:param method: {"none", ..}
		:return:
		"""
		# todo:
		pass
