import numpy as np

NUM_ERROR = 1e-9

from MMSProbe.utils.Math.Line3D import Line3D
from MMSProbe.utils.Math.Plane3D import Plane3D


def is_clockwise(vertex3, vector):
	vertex3 = np.asarray(vertex3, np.float64)
	vector = np.asarray(vector, np.float64)
	v0 = vertex3[1] - vertex3[0]
	v1 = vertex3[2] - vertex3[0]
	s = np.cross(v0, v1)
	if np.dot(s, vector) > 0: return True
	return False


is_CW = is_clockwise


def is_counterclockwise(vertex3, vector):
	return not is_clockwise(vertex3, vector)


is_CCW = is_counterclockwise


def intersect_line2plane(line: Line3D, plane: Plane3D):
	dis = np.dot(line.p, plane.v) - plane.d
	if dis == 0: return line.p
	vv = np.dot(line.v, plane.v)
	if abs(vv) < NUM_ERROR: return None
	return line.p - dis / vv * line.v


def distance_p2line(point, line: Line3D):
	vec = np.asarray(point) - line.p
	vec -= np.dot(vec, line.v) * line.v
	return np.linalg.norm(vec)


def distance_p2plane(point, plane: Plane3D):
	dis = np.dot(np.asarray(point), plane.v) - plane.d
	return abs(dis)


def line_division(p1, p2, density):
	"""
	:param p1: dim array
 	:param p2: dim array
	:param density: points per meters
	:return:
	"""
	dim = len(p1)
	p1 = np.asarray(p1, np.float64)
	p2 = np.asarray(p2, np.float64)
	dx = np.asarray(p2 - p1)

	num = int(round(np.linalg.norm(dx) * density))
	w = np.tile(np.linspace(0, 1, num + 2), (dim, 1))
	return dx * w.transpose() + p1
