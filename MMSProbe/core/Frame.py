import cv2
import numpy as np

from MMSProbe.core.Camera import Camera
from MMSProbe.utils.Common import printLog

from MMSProbe.utils.Common.debug_manager import folder_frame

DEBUG = True  # debug switch

# Shi-Tomasi corner detection para
FEATURE_PARA = dict(maxCorners = 500, qualityLevel = 0.01, minDistance = 7, blockSize = 5)
# Lucas-Kanade optical flow para
FEATURE_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.3)
LK_PARA = dict(winSize = (15, 15), maxLevel = 5, criteria = FEATURE_CRITERIA)
TH_KEYPOINT_NUM = 12  # the min of keypoints number, officially, 5 is enough
TH_HOMOGRAPHY_DIS = 10  # [pixel], the min distance of Homography distance
TH_KEYPOINT_DIS = 2  # [pixel], the min distance of keypoints distance


class Frame:
	def __init__(self, time: float, x: float, y: float, image , camera: Camera):
		self.t = time
		self.x = x
		self.y = y

		self.camera = camera
		# self.mwr_symbol = mwr_symbol
		self.img = camera.imread(image)
		self.img_mini = camera.resize(self.img)
		gray = cv2.cvtColor(self.img_mini, cv2.COLOR_BGR2GRAY)
		self.gray_mini = cv2.equalizeHist(gray)
		pass

	def __str__(self):
		return f"Frame({self.t:.3f}, {self.x:.3f}, {self.y:.3f}, img{self.img.shape}, img_mini{self.img_mini.shape})"

	def pos(self):
		return self.x, self.y

	def pose_estimate(self, other):
		"""
		:param other:
		:return: flag, R, t
		"""
		# attention, somehow, the points return by opencv always be (N, 1, dim), be careful
		pts1 = cv2.goodFeaturesToTrack(self.gray_mini, mask = None, **FEATURE_PARA)
		if len(pts1) < TH_KEYPOINT_NUM: return False, None, None, None, None
		pts2, status12, _ = cv2.calcOpticalFlowPyrLK(self.gray_mini, other.gray_mini, pts1, None, **LK_PARA)
		ptsR, status2R, _ = cv2.calcOpticalFlowPyrLK(other.gray_mini, self.gray_mini, pts2, None, **LK_PARA)

		# select by opencv method
		valid_LK = np.logical_and(status12 > 0, status2R > 0)
		if np.sum(valid_LK) < TH_KEYPOINT_NUM: return False, None, None, None, None
		pts1 = pts1[valid_LK].reshape(-1, 2)
		ptsR = ptsR[valid_LK].reshape(-1, 2)

		# select by matching (like re-projection)
		proj_drift = np.abs(pts1 - ptsR).max(1)
		valid_proj = proj_drift < 1
		if np.sum(valid_proj) < TH_KEYPOINT_NUM: return False, None, None, None, None
		pts2 = pts2[valid_LK].reshape(-1, 2)[valid_proj]
		pts1 = pts1[valid_proj]

		# select by Homography projection
		H, _ = cv2.findHomography(pts1, pts2, cv2.RHO)
		add = np.ones((len(pts1), 1))
		pts1_ = np.concatenate((pts1, add), axis = 1).transpose()
		pts2_ = np.asarray(np.asmatrix(H) * pts1_).transpose()
		pts2_ = pts2_[:, :2] / np.tile(pts2_[:, 2], (2, 1)).transpose()
		valid_H = np.linalg.norm(pts2_ - pts2, axis = 1).ravel() < TH_HOMOGRAPHY_DIS
		if np.sum(valid_H) < TH_KEYPOINT_NUM: return False, None, None, None, None
		pts1 = pts1[valid_H]
		pts2 = pts2[valid_H]

		# select by distance (remove if not moved)
		diff = np.abs(pts1 - pts2).max(1)
		valid_diff = diff > TH_KEYPOINT_DIS
		if np.sum(valid_diff) < TH_KEYPOINT_NUM: return False, None, None, None, None
		pts1 = pts1[valid_diff]
		pts2 = pts2[valid_diff]

		# ---------- debug ----------
		if DEBUG:
			printLog("Frame", f"save image \'lk_match/{self.t:.3f}.jpg\'")
			dst = np.copy(self.img_mini)
			for p1, p2 in zip(pts1, pts2):
				p1 = tuple(p1.astype(int))
				p2 = tuple(p2.astype(int))
				dst = cv2.circle(dst, p1, 1, (139, 0, 0), 2)
				dst = cv2.line(dst, p1, p2, (214, 112, 218), 2)
			# cv2.imwrite(folder_frame + f"/lk_match/{self.t:.3f}.jpg", dst)
			cv2.imwrite(f"C:/Users/kenta shimoyama/Documents/amanolab/melco/generate_mvp/imagepoint/{self.t:.3f}.jpg", dst)
		# ---------------------------

		mat = self.camera.mat_mini
		# looks like RANSAC performs good here
		E, _ = cv2.findEssentialMat(pts1, pts2, mat, method = cv2.RANSAC, prob = 0.99, threshold = 1.0)
		ret, R, t, _ = cv2.recoverPose(E, pts1, pts2, mat)  # the results are c_other to c_this
		if ret > 0:  # successes, and adjust to our calculation system
			return True, R.T, t.ravel() * -1, pts1, pts2
		return False, None, None, None, None