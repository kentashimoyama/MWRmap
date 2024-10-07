from datetime import datetime

from MMSProbe.utils.Common.os_functions import mkdir


def time_info():
	return datetime.now().strftime("%H%M%S.%f")


debug_root = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\motionstereo_result"
debug_root = mkdir(debug_root, datetime.now().strftime("debug_%Y%m%d%H%M"))

# class Frame
folder_frame = mkdir(debug_root, "Frame")
mkdir(folder_frame, "lk_match")

# class MapLane
folder_MapLane = mkdir(debug_root, "MapLane")

# class LaneFixer
folder_LaneFix = mkdir(debug_root, "LaneFix")
mkdir(folder_LaneFix, "tr_ground")
mkdir(folder_LaneFix, "perspective")
mkdir(folder_LaneFix, "result")
mkdir(folder_LaneFix, "solveT")
mkdir(folder_LaneFix, "fixMWR")

# class LaneDetector
# folder_LaneDetect = mkdir(debug_root, "LaneDetector")
# mkdir(folder_LaneDetect, "any")
# mkdir(folder_LaneDetect, "tmp")
# mkdir(folder_LaneDetect, "orig_lane")
# mkdir(folder_LaneDetect, "ld_with_ipe")

# class InitPoseEst
folder_IPE = mkdir(debug_root, "InitialPose")

# class KalmanFilter
folder_KF = mkdir(debug_root, "KalmanFilter")
mkdir(folder_KF, "update")

# class Status
folder_Status = mkdir(debug_root, "Status")

# Main file
folder_Main = mkdir(debug_root, "Main")
mkdir(folder_Main, "image_out")
sys_log_path = folder_Main + "sys_log.txt"
