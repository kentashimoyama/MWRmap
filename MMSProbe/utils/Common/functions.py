import numpy as np
import os
import math
from MMSProbe.utils.Common.debug_manager import folder_Main, folder_Status


def str2folder(path: str):
	# ensure path end with "/"
	if len(path) < 1: return None
	if path[-1] != "/": return path + "/"
	return path


def num2str(x, decimal: int, *, separator: str = ", "):
	f = f"%.{str(decimal)}f"  # LOL :D
	num = np.array(x).ravel()
	tmp_str = f % num[0]
	for i in num[1:]: tmp_str += separator + f % i
	return tmp_str

def num2int(x):
	return np.array(x, np.float64).round().astype(int)

def unwrap_angle(a, ref):
	"""
	wrap angle near to reference angle
	:param a: angle [rad]
	:param ref: reference angle [rad]
	:return: fixed angle [rad]
	"""
	pi2 = math.pi * 2
	diff = a - ref
	fix = int(abs(diff) / pi2 + 0.5) * pi2
	if diff > 0: return a - fix
	return a + fix

def perspective(mat: np.ndarray, pts):  # 2D -> 2D
	"""
	change perspective as,
		P_j = func(persMat_i2j, P_i)
	persMat calculated by,
		persMat_i2j = cv2.getPerspectiveTransform(P_i, P_j)
	:param mat: perspective 'matrix', better with suffix
	:param pts: (N, 2) array in image, renamed to 'src'
	:return: (N, 2) array in image
	"""
	# mat = np.asmatrix(persMat)
	src = np.atleast_2d(pts)
	add = np.ones((len(src), 1))
	src = np.concatenate((src, add), axis = 1)
	dst = src.dot(mat.T)
	dst[:, 0] = dst[:, 0] / dst[:, 2]
	dst[:, 1] = dst[:, 1] / dst[:, 2]
	return np.asarray(dst[:, :2], np.float32)  # unbelievable

def projection(mat: np.ndarray, pts):  # 3D -> 2D
	"""
	project 3D points in camera view into image
	:param mat: camera 'matrix'
	:param pts: (N, 3) array in camera view, renamed to 'src'
	:return: (N, 2) array in image
	"""
	src = np.atleast_2d(pts)
	dst = np.zeros((len(src), 2), dtype = np.float32)
	dst[:, 0] = src[:, 0] / src[:, 2] * mat[0, 0] + mat[0, 2]
	dst[:, 1] = src[:, 1] / src[:, 2] * mat[1, 1] + mat[1, 2]
	return dst

def trimming(image, pts):
	"""
	cut image by given points
	:param image: src
	:param pts:
	:return: dst, (w1, h1, w2, h2)
	"""
	shape = image.shape
	w1 = int(max(0, np.min(pts[:, 0])))
	h1 = int(max(0, np.min(pts[:, 1])))
	w2 = int(min(shape[1], np.max(pts[:, 0])))
	h2 = int(min(shape[0], np.max(pts[:, 1])))
	if w1 >= w2 or h1 >= h2:  # empty image, todo: warming, maybe not enough
		return None, (w1, h1, w2, h2)
	return image[h1:h2, w1:w2, :], (w1, h1, w2, h2)

def merge_covar(x, P, y, Q):
	"""
	:param x: 1st mean
	:param P: 1st covariance
	:param y: 2nd mean
	:param Q: 2nd covariance
	:return:
		z: result mean
		R: result covariance
	"""
	S = P + Q
	K = P * np.linalg.inv(S)
	Dx = K * np.asmatrix(y - x).transpose()
	z = x + np.asarray(Dx).transpose().ravel()
	R = P - K * S * K.transpose()
	return z, R

def argSample(a, nums: tuple = (3, 3, 3)):
	"""
	when u want to display a very long data-array,
	this func picks sample index at [start, body, end]
	:param a:
	:param nums: of rows to display
	:return
		start:
		body:
		enc:
	"""
	length = len(a)
	idxes = np.arange(length)
	if nums[0] + nums[2] > length:
		return idxes[:nums[0]], idxes[nums[0]:nums[0]], idxes[nums[0]:]

	mid = length - nums[2]
	body = np.random.choice(idxes[nums[0]:mid], nums[1], replace = False)
	body.sort()
	return idxes[:nums[0]], body, idxes[mid:]

def result_analyze(mapLane, draw_mapLane: bool):
	from MMSProbe.utils.Visualization import Canvas
	gps_raw_path = folder_Main + "gps_raw.csv"
	gps_raw = []
	fr = open(gps_raw_path, "r")
	for line in fr:
		row = np.asarray(line.split(",")[:3], np.float64)
		gps_raw.append(row)
	fr.close()
	gps_raw = np.asarray(gps_raw)

	# -------------------- state results --------------------
	result_path = folder_Main + "result.csv"
	gps_log_path = folder_Main + "gps_log.csv"

	data_ukf = []
	fr = open(result_path, "r")
	next(fr)
	for line in fr:
		row = np.asarray(line.split(",")[:5], np.float64)
		data_ukf.append(row)
	fr.close()
	data_gps = []
	fr = open(gps_log_path, "r")
	next(fr)
	for line in fr:
		row = np.asarray(line.split(",")[:5], np.float64)
		data_gps.append(row)
	fr.close()

	data_gps = np.asarray(data_gps)  # (t:0, x:1, y:2, psi:3, v:4)
	data_ukf = np.asarray(data_ukf)  # (t:0, x:1, y:2, psi:3, v:4)

	# vehicle traject
	canvas = Canvas()
	if draw_mapLane:
		canvas.draw_lines_p2p(mapLane.pts1, mapLane.pts2, para = dict(color = "white"))
	for d_gps in data_gps:
		t, x, y, psi, v = d_gps
		canvas.draw_ray_a((x, y), psi, 0.6, para = dict(color = "violet"))
	for d_ukf in data_ukf:
		t, x, y, psi, v = d_ukf
		canvas.draw_ray_a((x, y), psi, 0.6, para = dict(color = "yellow"))
	canvas.draw_points(gps_raw[:, 1:3], para = dict(
		color = "cyan", marker = "x", s = 20, label = "gps", lw = 1.1
	))
	canvas.draw_points(data_gps[:, 1:3], para = dict(
		color = "crimson", marker = "+", s = 40, label = "obs", lw = 1.1
	))
	canvas.draw_points(data_ukf[:, 1:3], para = dict(
		color = "white", marker = ".", s = 16, label = "ukf",
	))
	canvas.set_axis(equal_axis = True, sci_on = False, legend_on = True)
	canvas.show()
	canvas.save(folder_Main + "traject.pdf")
	canvas.close()

	# plot psi
	canvas = Canvas()
	psi_gps = [data_gps[0, 3]]
	psi_ukf = [data_ukf[0, 3]]
	for a in data_gps[1:, 3]: psi_gps.append(unwrap_angle(a, psi_gps[-1]))
	for a in data_ukf[1:, 3]: psi_ukf.append(unwrap_angle(a, psi_ukf[-1]))

	canvas.ax.plot(data_gps[:, 0], np.rad2deg(psi_gps), label = "psi_gps", color = "gray")
	canvas.ax.plot(data_ukf[:, 0], np.rad2deg(psi_ukf), label = "psi_ukf", color = "orange")
	canvas.set_axis(equal_axis = False, legend_on = True, legend_loc = "best")
	canvas.show()
	canvas.close()

	# plot dpsi
	canvas = Canvas()
	dpsi_gps = (data_gps[1:, 3] - data_gps[:-1, 3]) / (data_gps[1:, 0] - data_gps[:-1, 0])
	dpsi_ukf = (data_ukf[1:, 3] - data_ukf[:-1, 3]) / (data_ukf[1:, 0] - data_ukf[:-1, 0])
	dpsi_gps = [unwrap_angle(a, 0) for a in dpsi_gps]
	dpsi_ukf = [unwrap_angle(a, 0) for a in dpsi_ukf]
	canvas.ax.plot(data_gps[1:, 0], np.rad2deg(dpsi_gps), label = "dpsi_gps", color = "gray")
	canvas.ax.plot(data_ukf[1:, 0], np.rad2deg(dpsi_ukf), label = "dpsi_ukf", color = "orange")
	canvas.set_axis(equal_axis = False, sci_on = False, legend_on = True, legend_loc = "best")
	canvas.show()
	canvas.close()

	# plot speed
	canvas = Canvas()
	canvas.ax.plot(data_gps[:, 0], data_gps[:, 4], label = "v_gps", color = "gray")
	canvas.ax.plot(data_ukf[:, 0], data_ukf[:, 4], label = "v_ukf", color = "orange")
	canvas.set_axis(equal_axis = False, sci_on = False, legend_on = True)
	canvas.show()
	canvas.close()

	# -------------------- gps bias --------------------
	data_bias_path = folder_Status + "gps_bias_log.txt"

	data_bias = [[0, 0, 0]]
	fr = open(data_bias_path, "r")
	for line in fr:
		row = np.asarray(line.split(",")[:3], np.float64)
		data_bias.append(row)
	fr.close()
	data_bias = np.asarray(data_bias)  # (t:0, dx:1, dy:2)

	# gps drift plot
	canvas = Canvas()
	canvas.draw_lines(data_bias[:, 1:3], para = dict(color = "orange", label = "gps_drift"))
	canvas.draw_points([[0, 0]], para = dict(color = "white", marker = "+", s = 100, lw = 1))
	canvas.set_axis(legend_on = True, sci_on = False)
	canvas.show()
	canvas.close()

	# gps drift distance plot
	canvas = Canvas()
	canvas.ax.plot(np.linalg.norm(data_bias[:, 1:3], axis = 1), label = "gps_drift_dis")
	canvas.set_axis(equal_axis = False, legend_on = True)
	canvas.show()
	canvas.close()
	pass
