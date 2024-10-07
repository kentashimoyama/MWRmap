import numpy as np

from MMSProbe.utils.Common import errPrint
NUM_ERROR = 1e-9

def CumulativeContributionRatio(w):
	ccr = np.copy(w)
	for i in range(1, len(w)):
		ccr[i] += ccr[i - 1]
	return ccr / ccr[-1]


CCR = CumulativeContributionRatio


def PrincipalComponentAnalysis(X, ccr_th: float = 1):
	"""
	:param X: (N, dim) data
	:param ccr_th: ccr threshold
	:return:
		ave:
		s:
		u:
	"""
	X = np.atleast_2d(X)
	N, dim = X.shape

	if N < dim:
		errPrint(f">> err, data num {N} < dim {dim} ..")
		return None, None, None

	ave = np.mean(X, axis = 0)
	X = np.asmatrix(X - ave)
	V = X.transpose() * X / (N - 1)
	u, s, _ = np.linalg.svd(V)

	if ccr_th == 1: return ave, s, u
	ccr = CumulativeContributionRatio(s)
	valid = ccr < ccr_th
	return ave, s[valid], u[:, valid]


PCA = PrincipalComponentAnalysis


def LocalityPreservingProjection(X, th: float = -1, ccr_th: float = 1):
	"""
	:param X: (N, dim) data
	:param th: safety reciprocal coe
	:param ccr_th: ccr threshold
	:return:
	"""
	X = np.atleast_2d(X)
	N, dim = X.shape

	if N < dim:
		errPrint(f">> err, data num {N} < dim {dim} ..")
		return None, None, None

	ave = np.mean(X, axis = 0)
	X = np.asmatrix(X - ave)

	tmp = np.arange(N)
	idx1, idx2 = np.meshgrid(tmp, tmp)
	idx1 = idx1.reshape(-1)
	idx2 = idx2.reshape(-1)

	dX = np.asarray(X[idx1] - X[idx2])
	dS = np.sum(dX * dX, axis = 1)
	if th <= NUM_ERROR:
		th = max(float(np.median(dS)) * 2, 1) * 3
	A = np.exp(dS / -th)
	A = A.reshape((N, N))
	D = np.diag(np.sum(A, axis = 1))
	L = np.asmatrix(D - A)
	V = np.linalg.inv(X.transpose() * D * X) * X.transpose() * L * X
	u, s, _ = np.linalg.svd(V)

	if ccr_th == 1: return ave, s, u
	CCRs = CumulativeContributionRatio(s)
	valid = CCRs < ccr_th
	return ave, s[valid], u[:, :valid]


LPP = LocalityPreservingProjection


def LinearDiscriminantAnalysis(X, classIndexes, ccr_th: float = 1):
	"""
	:param X: (N, dim) data
	:param classIndexes: (M, Ni) class indexes
	:param ccr_th: ccr threshold
	:return:
	"""
	X = np.atleast_2d(X)
	N, dim = X.shape

	if N < dim:
		errPrint(f">> err, data num {N} < dim {dim} ..")
		return None, None, None

	ave = np.mean(X, axis = 0)
	X = np.asmatrix(X - ave)

	Sw = np.asmatrix(np.zeros((dim, dim)))
	Sb = np.asmatrix(np.zeros((dim, dim)))
	for idxes in classIndexes:
		Xi = X[idxes]
		Ni, _ = Xi.shape
		ave_i = np.mean(Xi, axis = 0)
		Xi = Xi - ave_i
		Sw += Xi.transpose() * Xi
		delta_ave = ave_i - ave
		Sb += delta_ave * delta_ave.transpose() * Ni
	Vw = Sw / N
	Vb = Sb / N
	V = np.linalg.inv(Vw) * Vb
	u, s, _ = np.linalg.svd(V)

	if ccr_th == 1: return ave, s, u
	CCRs = CumulativeContributionRatio(s)
	valid = CCRs < ccr_th
	return ave, s[valid], u[:, valid]


LDA = FDA = LinearDiscriminantAnalysis


# def merge_covariance(x: np.ndarray, P: np.matrix, y: np.ndarray, Q: np.matrix):
def merge_covariance(x: np.ndarray, P, y: np.ndarray, Q):
	S = P + Q
	K = P * np.linalg.inv(S)
	Dx = K * np.asmatrix(y - x).transpose()
	z = x + np.asarray(Dx).transpose().ravel()
	R = P - K * S * K.transpose()
	return z, R
