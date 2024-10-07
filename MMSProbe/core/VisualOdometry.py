from MMSProbe.utils.Math import Rotation_Euler
from MMSProbe.core.Frame import Frame


class VisualOdometry:
	def __init__(self):
		self._R = None
		self._t = None
		self._flag = False

	def run(self, frame1: Frame, frame2: Frame):
		"""
		todo: calc by 3 frames
		mainly work: calculate transition matrix from frame1 to frame2
		:param frame1: basic frame, pose already known
		:param frame2: new frame, pose to be calculated
		:return:
			flag, whether succeed (, but not used)
		"""
		flag, R, t = frame1.pose_estimate(frame2)
		if flag and t[2] < 0:  # camera should always go forward
			self._flag = False
			return False
		if flag and abs(t[2]) < 0.77:  # camera should always face to front
			self._flag = False
			return False
		self._flag = flag
		self._R = R
		self._t = t
		return flag

	def is_succeed(self):
		return self._flag

	def get_result(self):
		return self._R, self._t

	def solve_pose(self, R_v2c):
		"""
		:param R_v2c:
		:return:
			rot: from basic frame to new frame
		"""
		R_v_2v = R_v2c * self._R * R_v2c.transpose()
		rot = Rotation_Euler(R_v_2v, order = (0, 2, 1))
		return rot
