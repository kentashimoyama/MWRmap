import numpy as np
import os
from fnmatch import fnmatch
from MMSProbe.utils.Common.print_log import errPrint, sysPrint

def mkdir(local_path: str, folder_name: str):
	if not os.path.exists(local_path):
		errPrint(f">> mkdir: can not reach local path .. \"{local_path}\"")
		return None
	local_path = os.path.join(local_path, folder_name)
	if os.path.exists(local_path):
		sysPrint(f">> mkdir: folder exist .. \"{folder_name}\"")
		return local_path
	os.mkdir(local_path)
	sysPrint(f">> mkdir: create folder \"{folder_name}\"")
	return local_path

def listdir(local_path: str, *, pattern = None):
	"""
	:param local_path:
	:param pattern:
	:return: file_names:
	"""
	ignore_list = (
		".*",
		"_*",
	)
	file_names = np.asarray(os.listdir(local_path))
	invalid = np.full(len(file_names), False)
	for ig in ignore_list:
		sub_invalid = [fnmatch(n, ig) for n in file_names]
		invalid = np.logical_or(invalid, sub_invalid)
	file_names = file_names[np.logical_not(invalid)]
	if pattern is None:
		file_names.sort()
		return file_names
	valid = [fnmatch(n, pattern) for n in file_names]
	file_names = file_names[valid]
	file_names.sort()
	return file_names