import math
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import scipy.interpolate
import scipy.stats
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.ticker import ScalarFormatter


from MMSProbe.utils.Math.interpolate import spline_1d
from MMSProbe.utils.Visualization.Color import Color

NUM_ERROR = 1e-9

def histogram(x, delta, normalize: bool = True):
	"""
	:param x:
	:param delta:
	:param normalize:
	:return:
	"""
	lim_L = int(min(x) / delta) - 2
	lim_R = int(max(x) / delta) + 2
	length = lim_R - lim_L

	tmp_data = x - lim_L * delta
	vote_L = ((tmp_data + NUM_ERROR) / delta).astype(int)
	vote_R = ((tmp_data - NUM_ERROR) / delta).astype(int)
	vote = np.concatenate((vote_L, vote_R))
	index, count = np.unique(vote, return_counts = True)

	weight = len(x) * delta * 2
	x_dst = (np.arange(length) + lim_L + 0.5) * delta
	y_dst = np.zeros(length)
	y_dst[index] = count
	if normalize: y_dst /= weight
	return x_dst, y_dst


class Canvas:
	def __init__(self, **kwargs):
		rcParams["lines.linewidth"] = kwargs.get("linewidth", 0.75)
		if kwargs.get("dark_mode", True): plt.style.use('dark_background')

		size = kwargs.get("fig_size", (11, 8))
		edge = kwargs.get("fig_edge", (0.65, 0.25, 0.32, 0.45))
		rect = (
			edge[0] / size[0],
			edge[3] / size[1],
			max(size[0] - edge[0] - edge[1], 0) / size[0],
			max(size[1] - edge[2] - edge[3], 0) / size[1],
		)

		self.fig = plt.figure(figsize = size)
		self.ax = self.fig.add_axes(rect)
		self.layer = 10

	def layer_update(self, step: int = 1):
		num = self.layer
		self.layer += step
		return num

	def set_axis(self, **kwargs):
		font_size_units = kwargs.get("font_size_units", 10)
		font_size_ticks = kwargs.get("font_size_ticks", 12)
		font_size_title = kwargs.get("font_size_title", 13)

		if kwargs.get("sci_on", True):
			self.ax.xaxis.set_major_formatter(ScalarFormatter(useMathText = True))
			self.ax.xaxis.offsetText.set_fontsize(font_size_units)
			self.ax.yaxis.set_major_formatter(ScalarFormatter(useMathText = True))
			self.ax.yaxis.offsetText.set_fontsize(font_size_units)
			self.ax.ticklabel_format(style = "sci", axis = "both", scilimits = (0, 0))
		elif kwargs.get("decimal", None) is not None:
			str_format = "%." + str(kwargs["decimal"]) + "f"
			self.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter(str_format))
			self.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(str_format))

		if kwargs.get("equal_axis", True): self.ax.axis("equal")
		if kwargs.get("grid_on", True): self.ax.grid(alpha = 0.2, zorder = 0)
		self.ax.tick_params(direction = 'in')

		xticks = kwargs.get("xticks", None)
		if xticks is not None: self.ax.set_xticks(xticks)
		self.ax.set_xlim(kwargs.get("xlim", None))
		yticks = kwargs.get("yticks", None)
		if yticks is not None: self.ax.set_yticks(yticks)
		self.ax.set_ylim(kwargs.get("ylim", None))
		if kwargs.get("legend_on", False):
			self.ax.legend(loc = kwargs.get("legend_loc", "upper right"), fontsize = font_size_ticks,
			               facecolor = None, framealpha = 0, edgecolor = None)
		self.ax.set_xlabel(kwargs.get("xlabel", None), fontsize = font_size_ticks)
		self.ax.set_ylabel(kwargs.get("ylabel", None), fontsize = font_size_ticks)
		self.ax.set_title(kwargs.get("title", None), size = font_size_title)

	def save(self, path: str, dpi: int = 250):
		self.fig.savefig(path, dpi = dpi)

	@staticmethod
	def show():
		plt.show()

	@staticmethod
	def close():
		plt.close()

	# draw funcs
	def draw_points(self, points, *, para: dict = None):
		"""
		based on matplotlib.pyplot.scatter
		https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html
		:param points:
		:param para:
		:return: None
		"""
		pos = np.atleast_2d(points)
		if para is None: para = dict()
		para["zorder"] = para.get("zorder", self.layer_update())
		self.ax.scatter(pos[:, 0], pos[:, 1], **para)

	def draw_lines(self, points, *, para: dict = None):
		"""
		based on matplotlib.lines.Line2D
		https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.lines.Line2D.html?highlight=line2d#matplotlib.lines.Line2D
		:param points:
		:param para:
		:return: None
		"""
		pos = np.atleast_2d(points)
		if len(pos) < 2: return
		if para is None: para = dict()
		para["color"] = para.get("color", Color.rand() / 255)
		para["zorder"] = para.get("zorder", self.layer_update())
		line = Line2D(pos[:, 0], pos[:, 1], **para)
		self.ax.add_line(line)

	def draw_text(self, point, s, color='white', rotation=0):
		plt.text(point[0], point[1], s, color=color, rotation=rotation)


	# def draw_lines_p2p(self, points1, points2, *, para: dict = None):
	# 	"""
	# 	based on matplotlib.lines.Line2D
	# 	https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.lines.Line2D.html?highlight=line2d#matplotlib.lines.Line2D
	# 	:param points1:
	# 	:param points2:
	# 	:param para:
	# 	:return:
	# 	"""
	# 	points1 = np.atleast_2d(points1)
	# 	points2 = np.atleast_2d(points2)
	# 	if para is None: para = dict()
	# 	para["color"] = para.get("color", Color.rand() / 255)
	# 	para["zorder"] = para.get("zorder", self.layer_update())
	# 	for (p1, p2) in zip(points1, points2):
	# 		line = Line2D((p1[0], p2[0]), (p1[1], p2[1]), **para)
	# 		self.ax.add_line(line)
	# 	pass

	def draw_ray_v(self, pos, vector, length: float, *, para: dict = None, **kwargs):
		"""
		based on matplotlib.lines.Line2D
		https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.lines.Line2D.html?highlight=line2d#matplotlib.lines.Line2D
		:param pos:
		:param vector:
		:param length:
		:param para:
		:param kwargs:
			double_side: bool default = False
		:return:
		"""
		pos = p1 = np.array(pos, np.float64).ravel()
		v = np.array(vector, np.float64).ravel()
		v *= length / np.linalg.norm(v)
		if kwargs.get("double_side", False):
			p1 = pos - v
		p2 = pos + v
		if para is None: para = dict()
		para["color"] = para.get("color", Color.rand() / 255)
		para["zorder"] = para.get("zorder", self.layer_update())
		line = Line2D((p1[0], p2[0]), (p1[1], p2[1]), **para)
		self.ax.add_line(line)

	def draw_ray_a(self, pos, angle, length: float, *, para: dict = None, **kwargs):
		"""
		based on matplotlib.lines.Line2D
		https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.lines.Line2D.html?highlight=line2d#matplotlib.lines.Line2D
		:param pos:
		:param angle: [rad]
		:param length:
		:param para:
		:param kwargs:
			double_side: bool default = False
		:return:
		"""
		vector = (math.cos(angle), math.sin(angle))
		self.draw_ray_v(pos, vector, length, para = para, **kwargs)

	def draw_ellipse_a(self, pos, angle, a, b, *, para: dict = None):
		"""
		based on matplotlib.patches.Ellipse
		https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.patches.Ellipse.html?highlight=ellipse#matplotlib.patches.Ellipse
		:param pos:
		:param angle: width direction [rad]
		:param a: width / 2
		:param b: height / 2
		:param para:
		:return:
		"""
		if para is None: para = dict()
		para["fill"] = para.get("fill", False)
		para["color"] = para.get("color", Color.rand() / 255)
		para["zorder"] = para.get("zorder", self.layer_update())
		ellipse = Ellipse(pos, a, b, angle, **para)
		self.ax.add_patch(ellipse)

	def draw_ellipse_v(self, pos, vector, a, b, *, para: dict = None):
		"""
		based on matplotlib.patches.Ellipse
		https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.patches.Ellipse.html?highlight=ellipse#matplotlib.patches.Ellipse
		:param pos:
		:param vector: width direction
		:param a: width / 2
		:param b: height / 2
		:param para:
		:return:
		"""
		angle = np.rad2deg(math.atan2(vector[1], vector[0]))
		self.draw_ellipse_a(pos, angle, a, b, para = para)

	def draw_covariance(self, mean, covar, rate: float = 1, *, para: dict = None):
		"""
		:param mean:
		:param covar:
		:param rate:
		:param para: ellipse para
		:return:
			sigma: sigma array
			w: base vectors
		"""
		mean = np.array(mean, np.float64).ravel()
		covar = np.asmatrix(covar)

		lam, w = np.linalg.eig(covar)
		idxes = lam.argsort()[::-1]
		lam = lam[idxes]
		w = w[:, idxes]

		sigma = np.sqrt(lam)
		a, b = sigma * rate * 2
		vector = np.asarray(w[:, 0]).ravel()

		if para is None: para = dict()
		para["color"] = para.get("color", Color.rand() / 255)
		para["zorder"] = para.get("zorder", self.layer_update())
		self.draw_ellipse_v(mean, vector, a, b, para = para)
		return sigma, w

	# graph funcs
	def histogram(self, data, delta: float = 0, force_delta = False, ignore_rate: float = 0, fit_type: str = "norm",
	              *, step_para: dict = None, fit_para: dict = None, cubic_para: dict = None,
	              **kwargs):
		"""
		:param data: 1d-array
		:param delta:
		:param force_delta:
		:param ignore_rate: to avoid system err, some max & min part of data will be ignore
		:param fit_type: ("norm", "beta", "gamma", "pareto", "rayleigh")
		:param step_para:
		:param fit_para:
		:param cubic_para:
		:param kwargs:
		:return:
		"""
		title = kwargs.get("title", "")

		data = np.array(data, np.float64)
		length = len(data)

		idx_ignore = int(length * ignore_rate / 2)
		idxes = data.argsort()
		fit_data = data[idxes[idx_ignore: length - idx_ignore]]

		dist = getattr(scipy.stats, "norm")
		mu, sigma = dist.fit(fit_data)

		if not force_delta and delta < sigma / 2: delta = sigma / 2

		x, y = histogram(data, delta, normalize = True)

		if step_para is None: step_para = dict()
		step_para["where"] = step_para.get("where", "mid")
		step_para["label"] = step_para.get("label", title + ".step")
		step_para["zorder"] = step_para.get("zorder", self.layer_update())
		self.ax.step(x, y, **step_para)

		T = np.linspace(x[0], x[-1], num = 500)

		dist = getattr(scipy.stats, fit_type)
		param = dist.fit(fit_data)
		if kwargs.get("show_fit", True):
			pdf_fitted = dist.pdf(T, *param[:-2], loc = param[-2], scale = param[-1])

			if fit_para is None: fit_para = dict()
			fit_para["label"] = fit_para.get("label", title + "." + fit_type)
			fit_para["zorder"] = fit_para.get("zorder", self.layer_update())
			self.ax.plot(T, pdf_fitted, **fit_para)

		if kwargs.get("show_spline", False):
			func = spline_1d(x, y, 3)

			if cubic_para is None: cubic_para = dict()
			cubic_para["dashes"] = cubic_para.get("dashes", [4, 6, 13, 6])
			cubic_para["label"] = cubic_para.get("label", title + ".spline")
			cubic_para["zorder"] = cubic_para.get("zorder", self.layer_update())
			self.ax.plot(T, func(T), **cubic_para)

		return param

	pass
