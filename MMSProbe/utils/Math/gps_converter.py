import numpy as np
import math
from pyproj import Transformer


# Geometry
PROJ_FROM = 4326  # WGS84_EPSG
# PROJ_TO = 6670  # JPRCS2_EPSG
PROJ_TO = 6674  # JPRCS9_EPSG
#関東6677
#vison6674

GPS_CONVERTER = Transformer.from_proj(PROJ_FROM, PROJ_TO)


def LatLon2Pos(lat, lon):
	return GPS_CONVERTER.transform(lat, lon)


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
	# ---------- debug: ROI of the ground ----------
	# from Core.Visualization import KaisCanvas
	# canvas = KaisCanvas()
	#
	# h, w = image.shape[:2]
	# canvas.draw_lines(np.asarray([[0, 0], [w, 0], [w, h], [0, h], [0, 0]]),
	#                   para = dict(color = "white"))
	# canvas.draw_lines_p2p(pts[0], pts[1], para = dict(color = "green"))
	# canvas.draw_lines_p2p(pts[1], pts[2], para = dict(color = "green"))
	# canvas.draw_lines_p2p(pts[2], pts[3], para = dict(color = "green"))
	# canvas.draw_lines_p2p(pts[3], pts[0], para = dict(color = "green"))
	# canvas.draw_lines(np.asarray([[w1, h1], [w2, h1], [w2, h2], [w1, h2], [w1, h1]]),
	#                   para = dict(color = "yellow"))
	# canvas.set_axis()
	# canvas.show()
	# ----------------------------------------------
	if w1 >= w2 or h1 >= h2:  # empty image, todo: warming, maybe not enough
		return None, (w1, h1, w2, h2)
	return image[h1:h2, w1:w2, :], (w1, h1, w2, h2)
