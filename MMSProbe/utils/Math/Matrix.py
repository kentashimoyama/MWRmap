import numpy as np
from scipy.linalg import lu

from MMSProbe.utils.Common import errPrint


def gaussian_elimination(A: np.ndarray, b: np.ndarray = None):
	"""
	solve "Ax = b" by gaussian elimination
	:param A: (N, N) array
	:param b: (N) array
	:return:
		x: one of the answer
	"""
	A = np.atleast_2d(A)
	N, dim = A.shape
	if b is None: b = np.zeros(N)
	if N > dim:
		errPrint(f">> matrix err, N {N} > dim {dim} .. ")
		return None

	T = np.zeros((dim, dim + 1))
	T[:N, :dim] = A
	T[:N, -1] = b

	x = np.ones(dim)
	for i in reversed(range(dim)):
		T = lu(T, permute_l = True)[1]
		if np.sum(T[-1, :-1] != 0) > 0:
			x[i] = T[-1, -1] / T[-1, i]
		elif T[-1, -1] != 0: return None
		T[:-1, -1] -= T[:-1, i] * x[i]
		T = np.delete(T[:-1], i, axis = 1)
	return x