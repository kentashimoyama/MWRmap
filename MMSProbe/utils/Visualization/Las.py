"""
lass version = 1.2
"""

import laspy  # todo: current require laspy < 2.0.0
import numpy as np

from MMSProbe.utils.Common import errPrint, num2str, argSample, colInfo
from MMSProbe.utils.Visualization.Color import Color


class Las:
	def __init__(self, count: int):
		self.times = np.zeros(count, np.float64)
		self.points = np.zeros((count, 4), np.float64) # (x, y, z, group_id)
		self.colors = np.zeros((count, 3), np.uint16)
		self.intensity = np.zeros(count, np.uint16)
		self.pt_src_id = np.zeros(count, np.uint16)
		self.classification = np.zeros(count, np.uint8)
		self.center = np.zeros(3, np.float64)

	@classmethod
	def init_by_rand(cls, count: int):
		myLas = cls(count)
		myLas.times[:] = np.random.random(count)
		myLas.points[:, :] = np.random.random((count, 3))
		myLas.colors[:, :] = Color.rand(length = count)
		myLas.intensity[:] = np.random.randint(0, 200, count)
		return myLas

	def save_to_las(self, path: str, *, format_id: int, **kwargs):
		if self.length() < 1:
			errPrint(">> err, empty las data .. ")
			return

		head = laspy.header.Header()
		head.data_format_id = format_id
		fw = laspy.file.File(path, mode = "w", header = head)
		fw.header.scale = kwargs.get("las_scale", (0.001, 0.001, 0.001))
		fw.header.offset = kwargs.get("map_center", self.center)

		self.points[:, :3] += self.center
		fw.x = self.points[:, 0]
		fw.y = self.points[:, 1]
		fw.z = self.points[:, 2]
		fw.intensity = self.intensity
		fw.classification = self.classification
		fw.pt_src_id = self.pt_src_id

		if format_id > 1:
			fw.red = self.colors[:, 0]
			fw.green = self.colors[:, 1]
			fw.blue = self.colors[:, 2]

		if format_id % 2 > 0:
			fw.gps_time = self.times

		fw.close()

	def swap_center(self, new_center):
		new_center = np.asarray(new_center, np.float64)
		self.points[:, :3] += self.center - new_center
		self.center = new_center
		return self

	def update_center(self):
		delta_center = np.median(self.points, axis = 0).round()
		self.points -= delta_center
		self.center += delta_center
		return self

	@classmethod
	def init_by_las(cls, path: str, *, new_center = None):
		fr = laspy.file.File(path, mode = "r")
		count = fr.header.point_records_count

		myLas = cls(count)
		myLas.center[:] = fr.header.offset
		myLas.points[:, 0] = fr.x
		myLas.points[:, 1] = fr.y
		myLas.points[:, 2] = fr.z
		myLas.points -= myLas.center
		myLas.intensity[:] = fr.intensity
		myLas.classification[:] = fr.classification
		myLas.pt_src_id[:] = fr.pt_src_id

		format_id = fr.header.data_format_id

		if format_id > 1:
			myLas.colors[:, 0] = fr.red
			myLas.colors[:, 1] = fr.green
			myLas.colors[:, 2] = fr.blue

		if format_id % 2 > 0:
			myLas.times[:] = fr.gps_time

		if new_center is None: return myLas
		return myLas.swap_center(new_center)

	def save_to_csv(self, path: str):
		fw = open(path, "w")
		fw.write("cx,cy,cz\n")
		fw.write(num2str(self.center, 4) + "\n")
		fw.write("time,x,y,z,intensity,classification,pt_src_id,r,g,b,\n")
		for i in range(len(self.times)):
			tmp_str = f"{self.times[i]:.4f},"
			tmp_str += num2str(self.points[i], 4)
			tmp_str += f",{self.intensity[i]}"
			tmp_str += f",{self.classification[i]}"
			tmp_str += f",{self.pt_src_id[i]},"
			tmp_str += num2str(self.colors[i], 0)
			fw.write(tmp_str + "\n")
		fw.close()

	@classmethod
	def init_by_csv(cls, path: str):
		# todo: update when needed
		pass

	def __len__(self):
		return len(self.times)

	def length(self):
		return len(self)

	def copy(self):
		myLas = self.__class__(self.length())
		myLas.times[:] = self.times
		myLas.points[:, :] = self.points
		myLas.colors[:, :] = self.colors
		myLas.intensity[:] = self.intensity
		myLas.classification[:] = self.classification
		myLas.pt_src_id[:] = self.pt_src_id
		myLas.center[:] = self.center
		return myLas

	def merge(self, tmpLas):
		tmpLas.swap_center(self.center)
		self.times = np.concatenate((self.times, tmpLas.times))
		self.points = np.concatenate((self.points, tmpLas.points))
		self.colors = np.concatenate((self.colors, tmpLas.colors))
		self.intensity = np.concatenate((self.intensity, tmpLas.intensity))
		self.classification = np.concatenate((self.classification, tmpLas.classification))
		self.pt_src_id = np.concatenate((self.pt_src_id, tmpLas.pt_src_id))
		return self


class LasInfo:
	def __init__(self, path: str):
		fr = laspy.file.File(path, mode = "r")
		head = fr.header

		# header info
		self.header_format = head.header_format
		self.file_signature = head.file_signature
		self.file_source_id = head.file_source_id
		self.global_encoding = head.global_encoding
		self.project_id = head.project_id
		self.version = head.version
		self.system_id = head.system_id
		self.software_id = head.software_id
		self.date = head.date
		self.header_size = head.header_size
		self.vlrs = head.vlrs
		self.data_format_id = head.data_format_id
		self.data_record_length = head.data_record_length
		self.point_records_count = head.point_records_count
		self.point_return_count = head.point_return_count
		self.scale = head.scale
		self.offset = head.offset
		self.max = head.max
		self.min = head.min

		# point format
		self.point_format = fr.point_format
		idxes1, idxes2, idxes3 = argSample(fr.points, (6, 6, 6))
		self.points1 = fr.points[idxes1]
		self.points2 = fr.points[idxes2]
		self.points3 = fr.points[idxes3]

		fr.close()

	def __repr__(self):
		tmp_str = ">> las file info \n"
		tmp_str += "-------------------------------\n"
		tmp_str += ">> las head format \n"
		tmp_str += "- - - - - - - - - - - - - - - -\n"
		for tmp in self.header_format:
			tmp_str += "   | " + tmp.name + "\n"
		tmp_str += "-------------------------------\n"
		tmp_str += ">> las head data \n"
		tmp_str += "- - - - - - - - - - - - - - - -\n"
		tmp_str += "file_signature = " + str(self.file_signature) + "\n"
		tmp_str += "file_source_id = " + str(self.file_source_id) + "\n"
		tmp_str += "global_encoding = " + str(self.global_encoding) + "\n"
		tmp_str += "project_id = " + str(self.project_id) + "\n"
		tmp_str += "version = " + str(self.version) + "\n"
		tmp_str += "system_id = " + str(self.system_id) + "\n"
		tmp_str += "software_id = " + str(self.software_id) + "\n"
		tmp_str += "date = " + str(self.date) + "\n"
		tmp_str += "header_size = " + str(self.header_size) + "\n"
		tmp_str += "num_variable_len_recs = " + str(self.vlrs) + "\n"
		tmp_str += "data_format_id = " + str(self.data_format_id) + "\n"
		tmp_str += "data_record_length = " + str(self.data_record_length) + "\n"
		tmp_str += "point_records_count = " + str(self.point_records_count) + "\n"
		tmp_str += "point_return_count = " + str(self.point_return_count) + "\n"
		tmp_str += "scale = " + num2str(self.scale, 4) + "\n"
		tmp_str += "offset = " + num2str(self.offset, 4) + "\n"
		tmp_str += "max = " + num2str(self.max, 4) + "\n"
		tmp_str += "min = " + num2str(self.min, 4) + "\n"
		tmp_str += "-------------------------------\n"
		tmp_str += ">> las point format \n"
		tmp_str += "- - - - - - - - - - - - - - - -\n"
		for tmp in self.point_format: tmp_str += "   | " + tmp.name + "\n"
		tmp_str += "-------------------------------\n"
		tmp_str += ">> las point sample \n"
		tmp_str += "- - - - - - - - - - - - - - - -\n"
		for p in self.points1: tmp_str += str(p) + "\n"
		tmp_str += "...... \n"
		for p in self.points2: tmp_str += str(p) + "\n"
		tmp_str += "...... \n"
		for p in self.points3: tmp_str += str(p) + "\n"
		tmp_str += "===============================\n"
		return colInfo(tmp_str, "yellow")
