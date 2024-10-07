import numpy as np

NUM_ERROR = 1e-9

def in_triangle(points, vertex3):
	"""
	:param points: nx2 number
	:param vertex3: 3x2 number
	:return: valid if in triangle
	"""
	points = np.atleast_2d(points)
	length = len(points)
	vertex3 = np.array(vertex3, np.float64)

	mat = np.asmatrix(vertex3[1:] - vertex3[0])
	inv_mat = np.linalg.inv(mat.transpose())
	points -= vertex3[0]
	points = np.asmatrix(points).transpose()

	K = np.asarray(inv_mat * points)
	valid = np.logical_and(K[0, :] > -NUM_ERROR, K[1, :] > -NUM_ERROR)
	valid = np.logical_and(valid, K[0, :] + K[1, :] - 1 < NUM_ERROR)
	if length > 1: return valid
	return valid[0]


def in_rectangle(points, vertex4):
	"""
	:param points: nx2 number
	:param vertex4: 4x2 number (loop like)
	:return: valid if in rectangle
	"""
	vertex4 = np.array(vertex4, np.float64)
	valid_1 = in_triangle(points, vertex4[[0, 1, 2]])
	valid_2 = in_triangle(points, vertex4[[2, 3, 0]])
	return np.logical_and(valid_1, valid_2)


def intersect_line2rect(p0, p1, c0, c1):
	"""
	:param p0:
	:param p1:
	:param c0: left bottom corner
	:param c1: right top corner
	:return:
	"""
	edge = np.asarray([[0, 1, 0], [0, 1, -1],
	                   [1, 0, 0], [1, 0, -1]], np.float64)
	w, h = float(c1[0] - c0[0]), float(c1[1] - c0[1])
	p0 = np.asarray(p0, np.float64) - c0
	p1 = np.asarray(p1, np.float64) - c0
	v = p1 - p0
	line = np.asarray([v[1] / h, -v[0] / w, np.cross(v, p0) / w / h])
	temp = [np.cross(line, e) for e in edge]
	points = np.asarray([[p[0] / p[2], p[1] / p[2]] for p in temp])
	points = np.unique(points, axis = 0)
	valid = np.logical_and(points >= 0, points <= 1)
	valid = np.logical_and(valid[:, 0], valid[:, 1])
	if np.sum(valid) != 2: return None
	return points[valid, :] * [[w, h], [w, h]] + c0
