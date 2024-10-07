import cv2
import numpy as np

from MMSProbe.utils.Common import printLog
from MMSProbe.conf import Config

# ========== constant from Config
INIT_CAM_PARA = Config.camera_para
INIT_CAM_DIST = Config.undistort_dist
FLAG_UNDISTORT = Config.undistort_flag
IMG_SIZE = Config.image_size
IMG_RESIZE_W = Config.image_resize_width
RESIZE_RATE = IMG_RESIZE_W / IMG_SIZE[0]
IMG_RESIZE_H = int(RESIZE_RATE * IMG_SIZE[1])


class Camera:
	def __init__(self):
		"""
		for fast performance, origin image will be resized in to 'mini' size
		"""
		fx, fy, cx, cy = INIT_CAM_PARA
		self.imgSize = w, h = IMG_SIZE
		self.imgSize_mini = (IMG_RESIZE_W, IMG_RESIZE_H)

		# image undistort part
		mat = np.asarray([[fx, 0., cx],
		                  [0., fy, cy],
		                  [0., 0., 1.]], dtype = np.float32)
		self._dist = None  # dist coefficient, only used for undistort
		self._oldMat = None  # initiated camera matrix
		self.mat = None  # new camera matrix after optimal
		self.mat_mini = None  # camera matrix for mini size

		if FLAG_UNDISTORT:
			dist = np.zeros(14)
			dist[:len(INIT_CAM_DIST)] = INIT_CAM_DIST
			new_mat, _ = cv2.getOptimalNewCameraMatrix(mat, dist, (w, h), 0)
			self._dist = dist
			self._oldMat = mat
			self.mat = new_mat
			self.mat_mini = new_mat * RESIZE_RATE
		else:
			self.mat = mat
			self.mat_mini = mat * RESIZE_RATE
		printLog("Camera", str(self))
		pass

	def __str__(self):
		tmp_str = f"<Camera at {hex(id(self))}>\n"
		tmp_str += f"image size = {self.imgSize}\n"
		tmp_str += f"image size mini = {self.imgSize_mini}\n"
		tmp_str += f"camera mat = \n{self.mat}\n"
		tmp_str += f"camera mat mini = \n{self.mat_mini}\n"
		tmp_str += f"do undistort = {FLAG_UNDISTORT}\n"
		tmp_str += f"undistort dist = {self._dist}\n"
		return tmp_str

	def imread(self, image):  # read image from '*.raw'
		img = image
		if not FLAG_UNDISTORT: return img
		return self.undistort(img)

	# def undistort(self, image):
	# 	img_gpu_src = cv2.cuda_GpuMat()
	# 	img_gpu_src.upload(image)
	# 	img_gpu_dst = cv2.cuda_GpuMat()
	# 	img_gpu_dst = cv2.cuda.undistort(img_gpu_src, self._oldMat, self._dist, newCameraMatrix=self.mat)
	# 	return img_gpu_dst.download()

	def undistort(self, image):
		return cv2.undistort(image, self._oldMat, self._dist, newCameraMatrix = self.mat)

	def resize(self, image):
		return cv2.resize(image, self.imgSize_mini, interpolation = cv2.INTER_LINEAR_EXACT)
