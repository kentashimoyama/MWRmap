from datetime import datetime, timezone
from MMSProbe.utils.IO.DataIO import DataSocket
from MMSProbe.utils.Common import str2folder
from MMSProbe.utils.Common.os_functions import listdir
import numpy as np

TIME_FORMAT = "%y%m%d%H%M%S%f"
TIME_ZONE_INFO = timezone.utc


def str2timestamp(time_str: str):
	t = datetime.strptime(time_str, TIME_FORMAT)
	t = t.replace(tzinfo = TIME_ZONE_INFO)
	return round(t.timestamp(), 6)


def timestamp2str(t: float):
	dt = datetime.fromtimestamp(t, TIME_ZONE_INFO)
	return dt.strftime(TIME_FORMAT)

class DataFlow:
	def __init__(self, root: str, folder: str):
		# todo: what if given root only and combine multi_folders
		root = str2folder(root)
		folder = str2folder(folder)

		GPSData = dict()
		fr = open(root + folder + "GPSData.csv", "r")
		for line in fr:
			row = line[:-1].split(",")
			GPSData[row[0]] = np.asarray(row[1:4], float)
		fr.close()

		self._index = 0
		self._time_strs = []
		self._lats = []
		self._lons = []
		self._paths = []
		self._length = 0

		files = listdir(root + folder, pattern = "*.raw")
		for i, filename in enumerate(files):
			time_id = filename.split(".")[0]
			lat, lon, alt = GPSData[time_id]
			self._time_strs.append(folder[:6] + time_id)
			self._lats.append(lat)
			self._lons.append(lon)
			self._paths.append(root + folder + filename)
			self._length += 1
		pass

	def pop(self):
		i = self._index
		time_str = self._time_strs[i]
		t = str2timestamp(time_str)
		lat = self._lats[i]
		lon = self._lons[i]
		path = self._paths[i]
		self._index += 1
		return t, lat, lon, path

	def reset(self):
		self._index = 0

	def __len__(self):
		return self._length

	def is_end(self):
		return self._index >= len(self)

	def is_not_end(self):
		return not self.is_end()

	def set_index(self, i):
		self._index = i

	pass

class DataFlowRealTime:
	def __init__(self, conn_str = "tcp://localhost:5000", channel=''):
		# todo: Initial process
		self._DS = DataSocket(conn_str, channel)
		self._index = 0
		self._time_strs = []
		self._lats = []
		self._lons = []
		self._paths = []
		self._length = 0
		self._frame_number = 0
		self._image = None
		self._flag = 0
		self._mwr_symbol = None

	def update(self):
		self._frame = self._DS.get_frame()
		self._frame_number = self._frame.frame_number
		self._lats.append(self._frame.gps_latitude)
		self._lons.append(self._frame.gps_longitude)
		time_stamp = datetime.fromtimestamp(self._frame.gps_timestamp/1000).strftime(TIME_FORMAT)
		self._time_strs.append(time_stamp[:15])
		self._length += 1
		self._paths.append(False)
		self._image = self._frame.image
		self._flag = self._frame.flag
		self._mwr_symbol = self._frame.mwr_symbol

	def pop(self):
		# todo: query
		self.update()

		i = self._index
		time_str = self._time_strs[i]
		t = str2timestamp(time_str)
		frame_number = self._frame_number
		lat = self._lats[i]
		lon = self._lons[i]
		image = self._image
		path = self._paths[i]
		mwr_symbol = self._mwr_symbol
		self._index += 1

		return t, lat, lon, image, frame_number, mwr_symbol


	def reset(self):
		self._index = 0

	def __len__(self):
		return self._length

	def is_end(self):
		if self._flag:
			return True
		else:
			return False

	def is_not_end(self):
		return not self.is_end()

	def set_index(self, i):
		self._index = i

	pass


if __name__ == '__main__':
	dataFlow = DataFlow()

	loop_count = 0
	while True:
		data = dataFlow.pop()
		print(f"loop {loop_count}", end=" >> ")
		loop_count += 1
