import math
from itertools import combinations

import numpy as np

from MMSProbe.utils.Common.print_log import errPrint
from MMSProbe.utils.Math.Matrix import gaussian_elimination
from MMSProbe.utils.Math.statistic_basic import PCA, LPP

NUM_ERROR = 1e-9

def distance2plane(points: np.ndarray, v: np.ndarray, d):
	"""
	distance = x * v^t - d
	:param points:
	:param v:
	:param d:
	:return:
	"""
	X = np.asmatrix(points)
	Vt = np.asmatrix(v).transpose()
	return np.asarray(X * Vt - d).ravel()


def distance_error(points: np.ndarray, v: np.ndarray, d):
	"""
	d' = x * v^t - d
	err = (d' * d'^t) / N
	:param points:
	:param v:
	:param d:
	:return:
	"""
	N, dim = points.shape
	D = distance2plane(points, v, d)
	return np.sum(D * D) / N


def distance_filter(D: np.ndarray, threshold = 1):
	D2 = D * D
	th = float(np.median(D2)) * 2
	th = max(th, threshold) * 10
	k = np.exp(-D2 / th) - 1
	return k * D


def fix_points(points: np.ndarray, v: np.ndarray, d):
	"""
	d' = x * v^t - d
	k = f(d') - 1
	x_i += (k_i * d'_i) * v_i
	:param points:
	:param v:
	:param d:
	:return:
	"""
	N, dim = points.shape
	D = distance2plane(points, v, d)
	D_ = distance_filter(D)
	k_ = np.tile(D_, (dim, 1)).transpose()
	return points + k_ * v


def fit_plane_(points: np.ndarray):
	N, dim = points.shape
	v = gaussian_elimination(points[1:dim] - points[0])
	if v is None: return False, np.zeros(dim), 0
	v /= np.linalg.norm(v)
	return True, v, np.dot(v, points[0])


def fit_plane_Random(points: np.ndarray):
	N, dim = points.shape
	idxes = np.random.choice(np.arange(N), dim, replace = False)
	return fit_plane_(points[idxes])


def fit_plane_RANSAC(points: np.ndarray, iter_num):
	N, dim = points.shape
	random_indexes = (np.random.random(N)).argsort()
	points = points[random_indexes]

	# solve C(num, dim) < iter_num
	lnp = math.log(iter_num) + math.log((np.arange(2, dim + 1)).prod())
	num = min(int(math.exp(lnp / dim) + (0.5 * dim) - 0.5), N)
	rate = N / num
	idxes_pool = list(combinations(range(num), dim))
	idxes_pool = (np.asarray(idxes_pool) * rate).astype(int)

	best_v = np.zeros(dim)
	best_d = 0
	best_flag = False
	inlier_num = 0
	dis_th = 1
	for idxes in idxes_pool:
		flag, v, d = fit_plane_(points[idxes])
		if not flag: continue
		D = np.abs(distance2plane(points, v, d))
		count = np.sum(D < dis_th)

		if count > inlier_num:
			best_v = v
			best_d = d
			best_flag = flag
			dis_th = min(dis_th, np.median(D) * 1.414)
			inlier_num = np.sum(D < dis_th)
	if not best_flag: return False, np.zeros(dim), 0
	return True, best_v, best_d


def fit_plane_LSM(points: np.ndarray):
	"""
	Least Square Method
	N-dim points -> m-dim plane
	fitting function
	f(x) = Vx = d
	:param points: (n, dim) array
	:return: flag, (V, d)*m array
	"""
	N, dim = points.shape

	ave = np.mean(points, axis = 0)
	X = points - ave
	weights = np.sum(X * X, axis = 0)

	order = np.arange(dim)
	flag_idx = np.argmin(weights)
	order = order[order != flag_idx]

	X_ = np.asmatrix(np.zeros((N, dim)))
	X_[:, :-1] = X[:, order]
	X_[:, -1] = -1
	M = X_.transpose() * X_
	b = np.asmatrix(-X[:, flag_idx]) * X_
	V_ = gaussian_elimination(M, b)

	V = np.ones(dim + 1)
	V[order] = V_[:-1]
	V[-1] = V_[-1]
	V /= np.linalg.norm(V)
	return True, V[:-1], V[-1] + np.dot(V[:-1], ave)


def fit_plane_PCA(points: np.ndarray):
	N, dim = points.shape
	ave, s, u = PCA(points, ccr_th = 1)
	if ave is None: return False, np.zeros(dim), 0
	v = np.asarray(u[:, -1]).ravel()
	v /= np.linalg.norm(v)
	return True, v, np.dot(v, ave)


def fit_plane_LPP(points: np.ndarray):
	N, dim = points.shape
	ave, s, u = LPP(points, ccr_th = 1)
	if ave is None: return False, np.zeros(dim), 0
	v = np.asarray(u[:, -1]).ravel()
	v /= np.linalg.norm(v)
	return True, v, np.dot(v, ave)


def fit_plane(points, fit_method: str = "none", com_method: str = "none", **kwargs):
	"""
	N-dim points -> (N-1)-dim plane
	fitting function
	f(x) = v*x = d
	:param points: (n, dim) array
	:param fit_method:
		'none': Least Square Method
		'fast': use top 3 points only
		'random': use random 3 points
		'ransac': non repeating random selecting
		'pca': maybe the best
		'lpp': do not fit well
	:param com_method: ("none", "fpd") compensate method
		'none': no compensation
		'fpd': fix points distance to plant  # todo: need to rename
	:param kwargs:
	:return:
		flag: True if succeed
		v:
		d:
	"""
	points = np.atleast_2d(points)
	N, dim = points.shape
	if N < dim:
		errPrint(f">> fitting err, need more {dim}-dim points .. ")
		return np.zeros(dim), 0

	fit_method = fit_method.lower()
	fit_func = fit_plane_LSM
	if fit_method == "none": fit_func = fit_plane_LSM
	elif fit_method == "lsm": fit_func = fit_plane_LSM
	elif fit_method == "pca": fit_func = fit_plane_PCA
	elif fit_method == "lpp": fit_func = fit_plane_LPP
	elif fit_method == "ransac":
		iterate_num = kwargs.get("iterate_num", 2000)
		return fit_plane_RANSAC(points, iterate_num)
	elif fit_method == "random": return fit_plane_Random(points)
	elif fit_method == "fast": return fit_plane_(points)
	else: errPrint("err, unknown plane fitting method .. ")

	com_method = com_method.lower()
	if com_method == "none": return fit_func(points)
	if com_method == "fpd":
		err_th = kwargs.get("dis_err_th", NUM_ERROR * 1000)
		loop_th = kwargs.get("loop_th", 50)
		flag, v, d = fit_func(points)
		dis_err_ = distance_error(points, v, d)
		if fit_method == "lpp": fit_func = fit_plane_LPP
		while flag:
			points_ = fix_points(points, v, d)
			flag, v, d = fit_func(points_)
			loop_th -= 1
			if loop_th < 0: return flag, v, d
			dis_err = distance_error(points, v, d)
			if abs(dis_err_ - dis_err) < err_th: return flag, v, d
			dis_err_ = dis_err
		return flag, v, d

	errPrint("err, unknown compensate method .. ")
	return fit_func(points)


if __name__ == '__main__':
	import numpy as np
	from scipy.stats import multivariate_normal

	from Core.Basic import num2str
	from Core.Visualization import KaisCanvas


	def init_line(num):
		p0 = np.random.random(2) - 2
		v = np.random.random(2)
		v /= np.linalg.norm(v)

		d = np.arange(num) * 0.5

		err = (np.random.random((num, 2)) - 0.5)
		points = np.zeros((num, 2))
		for i, (e, r) in enumerate(zip(err, d)):
			points[i, :] = p0 + v * r + e

		print(f"true v = {num2str(v, 4)}")
		return points


	def init_covar(num):
		a = np.deg2rad(30)
		c = math.cos(a)
		s = math.sin(a)
		mat = np.asmatrix([[c, -s], [s, c]])
		covar = mat * np.diag((6, 1)) * mat.transpose()
		mean = (3, 4)
		return multivariate_normal(mean, covar).rvs(num), mean, covar


	def cross(v):
		a, b = v[:2]
		dst = np.zeros(len(v))
		dst[0] = -b
		dst[1] = a
		dst -= np.dot(v, dst) * v
		dst /= np.linalg.norm(dst)
		return dst


	def main():
		canvas = KaisCanvas()
		canvas.draw_lines_p2p((0, 0), (5, 0), para = dict(color = "crimson", lw = 1.2))
		canvas.draw_lines_p2p((0, 0), (0, 4), para = dict(color = "green", lw = 1.2))

		# points = np.random.random((10, 2))
		# points[1:, :] = points[0]
		# points, c, m = init_covar(1000)
		points = init_line(100)
		points[-1, :] = (50, -10)
		# points[-1, :] = (10, -10)
		# points[-2, :] = (-10, 10)

		canvas.draw_points(points, para = dict(
			marker = ".", s = 10, c = "yellow", alpha = 0.3, zorder = 20,
		))
		# canvas.draw_covariance(c, m, rate = 3)

		# ------------------------------ working ------------------------------
		# N, dim = points.shape

		# ------------------------------ working ------------------------------

		# flag, v, d = fit_plane(points, fit_method = "fast", com_method = "fpd")
		# p = v * d
		# vec = cross(v)
		# canvas.draw_ray_v(p, vec, 9, double_side = True, para = dict(label = "fast"))

		# flag, v, d = fit_plane(points, fit_method = "lsm", com_method = "none")
		# p = v * d
		# vec = cross(v)
		# canvas.draw_ray_v(p, vec, 9, double_side = True, para = dict(label = "lsm"))

		# flag, v, d = fit_plane(points, fit_method = "pca", com_method = "none")
		# p = v * d
		# vec = cross(v)
		# canvas.draw_ray_v(p, vec, 9, double_side = True, para = dict(label = "pca"))

		# flag, v, d = fit_plane(points, fit_method = "lpp", com_method = "none")
		# p = v * d
		# vec = cross(v)
		# canvas.draw_ray_v(p, vec, 9, double_side = True, para = dict(label = "lpp"))

		# flag, v, d = fit_plane(points, fit_method = "random")
		# p = v * d
		# vec = cross(v)
		# canvas.draw_ray_v(p, vec, 9, double_side = True, para = dict(label = "rand"))

		flag, v, d = fit_plane(points, fit_method = "ransac")
		p = v * d
		vec = cross(v)
		canvas.draw_ray_v(p, vec, 9, double_side = True, para = dict(label = "ransac"))

		canvas.set_axis(equal_axis = True, legend_on = True)
		canvas.save("/home/kai/PycharmProjects/pyCenter/diary_output/d_20201210/out.png")
		canvas.close()
		pass


	if __name__ == '__main__':
		main()
