import math

import cv2
import numpy as np
import scipy.stats

from MMSProbe.utils.Common import errPrint, num2int
from MMSProbe.utils.Math import CoordSys, AxisAngle_Rotation
from MMSProbe.utils.Visualization.Color import Color
from MMSProbe.utils.Visualization.Las import Las


# todo: need to be upgrade ..
class PCModel(Las):
	def __init__(self, count: int = 0):
		super().__init__(count)

	def save_to_las(self, path, *, format_id = 3, **kwargs):
		super().save_to_las(path, format_id = format_id, **kwargs)

	def conv(self, mat):
		self.points[:, :] = CoordSys.conv(mat, self.points)
		return self

	def convCopy(self, mat):
		model = self.copy()
		return model.conv(mat)

	def add_model(self, model):
		if model is None:
			errPrint(">> pcm: addition failed .. ")
			return self
		return self.merge(model)

	def add_point(self, point, color, **kwargs):
		model = self.new_point(point, color, **kwargs)
		return self.add_model(model)

	def add_line(self, p1, p2, color, **kwargs):
		model = self.new_line(p1, p2, color, **kwargs)
		return self.add_model(model)

	def add_dash(self, p1, p2, color, **kwargs):
		model = self.new_dash(p1, p2, color, **kwargs)
		return self.add_model(model)

	@classmethod
	def new_point(cls, point, color, **kwargs):
		pos = np.atleast_2d(point)
		length = pos.shape[0]
		if length < 1:
			errPrint(">> pcm: empty point .. ")
			return None
		model = cls(length)
		model.points[:, :] = point
		model.colors[:, :] = color
		model.times[:] = kwargs.get("time", 0)
		model.intensity[:] = kwargs.get("intensity", 0)
		model.pt_src_id[:] = kwargs.get("pt_src_id", 0)
		model.classification[:] = kwargs.get("classification", 0)
		model.center[:] = kwargs.get("center", (0, 0, 0))
		return model

	@classmethod
	def new_line(cls, p1, p2, color, **kwargs):
		density = kwargs.get("density", 200)
		group_id = p1[3]
		p1 = np.asarray(p1[:3], np.float64)
		p2 = np.asarray(p2[:3], np.float64)

		dx = p2 - p1
		num = int(round(np.linalg.norm(dx) * density) + 1)
		dx = dx / num

		w = np.tile(np.arange(num + 1), (3, 1))
		points = p1 + dx * w.transpose()
		points = np.c_[points, int(group_id) * np.ones((points.shape[0], 1))]
		return cls.new_point(points, color, **kwargs)

	@classmethod
	def new_dash(cls, p1, p2, color, **kwargs):
		density = kwargs.get("density", 200)
		dash_space = int(kwargs.get("dash_space", 20))

		p1 = np.asarray(p1, np.float64)
		p2 = np.asarray(p2, np.float64)

		dx = p2 - p1
		num = int(round(np.linalg.norm(dx) * density) + 2)
		dx = dx / (num - 1)
		sam_a = int(dash_space * 2)
		sam_b = dash_space // 2

		tmp_num = (num + sam_b) % sam_a
		count = int(num // sam_a * dash_space)
		count += int((1 - tmp_num // dash_space) * (tmp_num + dash_space + sam_b) % dash_space)
		count += int((tmp_num // dash_space) * sam_b)

		points = np.zeros((count + 1, 3), np.float64)
		i = 0
		j = 0
		while i < num:
			points[j, :] = p1 + dx * i
			j += 1
			i += 1
			if i % sam_a == sam_b:
				i += dash_space
		return cls.new_point(points, color, **kwargs)

	@classmethod
	def new_axis(cls, **kwargs):
		scale3d = kwargs.get("scale3d", (1.5, 1.5, 1.5))
		center_color = kwargs.get("center_color", None)
		zero = np.zeros(3)

		if center_color is None: center_color = Color.plotColor("white")
		model = cls.new_point(zero, color = center_color, **kwargs)
		model.add_line(zero, (scale3d[0], 0, 0), color = Color.axis(0), **kwargs)
		model.add_line(zero, (0, scale3d[1], 0), color = Color.axis(1), **kwargs)
		model.add_line(zero, (0, 0, scale3d[2]), color = Color.axis(2), **kwargs)
		return model

	@classmethod
	def new_cross(cls, pos, color, **kwargs):
		dx = np.eye(3) * kwargs.get("cross_scale", 0.2)
		model = cls()
		for i in range(3):
			model.add_line(pos - dx[i], pos + dx[i], color, **kwargs)
		return model

	@classmethod
	def normal(cls, mean, covar, color, count: int = 3000):
		normal = scipy.stats.multivariate_normal(mean, covar)
		return cls.new_point(normal.rvs(count), color)

	@classmethod
	def image(cls, img, img_para, **kwargs):
		new_size = kwargs.get("new_size", None)
		scale = kwargs.get("scale", 1.5)

		if new_size is not None:
			img = cv2.resize(img, tuple(new_size), interpolation = cv2.INTER_AREA)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		h, w = img.shape[:2]
		fx, fy, cx, cy = img_para

		x_idx, y_idx = np.mgrid[:w, :h]
		x_idx = x_idx.ravel()
		y_idx = y_idx.ravel()

		model = cls(w * h)
		model.points[:, 0] = (x_idx - cx) / fx
		model.points[:, 1] = (y_idx - cy) / fy
		model.points[:, 2] = 1
		model.points *= scale
		model.colors[:, :] = img[y_idx, x_idx]

		return model

	@classmethod
	def cameraBox(cls, img_size, img_para, **kwargs):
		"""
		@param img_size: (w, h)
		@param img_para: (fx, fy, cx, cy)
		@param kwargs:
		@return:
		"""
		scale = kwargs.get("scale", 1.5)
		width = kwargs.get("new_img_width", 800)
		axis_on = kwargs.get("axis_on", True)

		center_color = kwargs.get("center_color", None)
		if center_color is None: center_color = Color.plotColor("white")
		stand_color = kwargs.get("stand_color", None)
		if stand_color is None: stand_color = Color.plotColor("gray")
		frame_color = kwargs.get("frame_color", None)

		img = kwargs.get("img", None)

		# draw axis
		if axis_on: model = cls.new_axis(scale3d = (scale / 2.5, scale / 2.5, scale / 1.4), **kwargs)
		else: model = PCModel.new_point((0, 0, 0), color = center_color, **kwargs)

		# draw image
		img_size = np.array(img_size, np.float64)
		img_para = np.array(img_para, np.float64)
		k = width / img_size[0]
		img_para *= k
		img_size *= k
		img_size = num2int(img_size)
		if img is not None: model.add_model(cls.image(img, img_para, new_size = img_size))

		# draw frame
		w, h = img_size
		fx, fy, cx, cy = img_para

		x_idx = np.asarray([0, 1, 1, 0], int) * (w + 1)
		y_idx = np.asarray([0, 0, 1, 1], int) * (h + 1)

		vertex = np.zeros((4, 3))
		vertex[:, 0] = (x_idx - cx) / fx
		vertex[:, 1] = (y_idx - cy) / fy
		vertex[:, 2] = 1
		vertex *= scale
		if frame_color is not None:
			for i in range(4):
				j = (i + 1) % 4
				model.add_line(vertex[i], vertex[j], frame_color, **kwargs)

		for pos in vertex:
			model.add_line((0, 0, 0), pos, stand_color, **kwargs)

		return model

	@classmethod
	def line_loop(cls, points, color, **kwargs):
		model = cls()
		for i, p in enumerate(points, -1):
			model.add_line(points[i], p, color, **kwargs)
		return model

	@classmethod
	def circle(cls, pos, vector, color, **kwargs):
		density = kwargs.get("density", 200)
		r = kwargs.get("radius", 1)
		vector /= np.linalg.norm(vector)
		E = np.eye(3)
		tmp_v = E[np.argmin(np.abs(np.dot(E, vector)))]
		vi = np.cross(tmp_v, vector)
		vi *= r / np.linalg.norm(vi)
		vi = np.asmatrix(vi).transpose()
		# x0 = pos + vi
		num = int(round(2 * math.pi * r * density) + 1)
		a = 2 * math.pi / num * np.arange(num + 1)

		points = np.asarray([np.asarray(
			AxisAngle_Rotation(vector, ai) * vi
		).ravel() for ai in a])
		points += pos
		return cls.new_point(points, color, **kwargs)
