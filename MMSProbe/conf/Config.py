# todo: add an io for file system
# worming, should not import any project files

class Config:
	# attention, emphasize float value please ..
	# todo: move unchangeable paras into its own file

	# camera info
	# camera_para = (627.1458, 626.1273, 657.2204, 381.8784)  # (fx, fy, cx, cy) << for no distort
	camera_para = (616.5263, 617.2315, 651.1589, 376.0408)  # (fx, fy, cx, cy) << for distort
	image_size = (1280, 720)
	undistort_flag = True  # if image undistort needed
	undistort_dist = (-0.00581053, -0.00275994, -0.01471530, 0.02229724,
	                  0.01747018, 0.00414806, 0.00542942, 0.00993107,
	                  -0.02302616, 0.00131315, 0.00866005, 0.00088481,
	                  0.01059735, 0.05109549)  # undistort paras
	image_resize_width = 1280  # [pixel], resize image for vo, better choose from (500, )
	camera_z_far = 50  # [m], better choose from (30, 60)
	camera_z_near = 1  # [m], better choose from (1, 4)
	camera_side_far = 10  # [m], better choose from (5, )

	# vehicle info
	vehicle_length = 1  # ignore: what if unknown

	# initial data
	ipe_init_sys_height = 1.00  # [m], the height of camera, also used as system height
	ipe_init_rot_v2c = (0., 0., 0.)  # [rad], rotation from vehicle to camera
	# ipe_init_rot_v2c = (-0.04266028, 0.30625996, 0.10308432)  # [rad], rotation from vehicle to camera
	ipe_path_sample_pts = None
	th_pose_stable_angle = 0.007  # [rad], about 0.4 [deg]

	# threshold data
	# th_gps_drift_dis = 0.05  # [m^2]
	# th_turning_angle = 0.175  # [rad], about 10 [deg]
	# th_straight_angle = 0.035  # [rad], about 2 [deg]

	# lane detect para
	# lane_line_width_real = 0.04  # [m]
	# lane_width_basic_real = 3.0  # [m]
	# laneDet_front_far = 10  # [m], better choose from (20, 40)
	# laneDet_side_far = 2.5  # [m], better choose from (2, 6)
	# laneDet_pixel_coe = 60  # [pixel/m], better choose from (10, 60)
	# laneDet_line_div_density = 50  # [/m]
	# laneDet_scan_length = 0.5  # [m]
	# laneDet_lane_gap_length = 1.5  # [m]
	# laneDet_lane_min_length = 4.0  # [m], better choose from (6, 10)
	# laneDet_lane_image_min_length = 40  # [pixel]
	# laneDet_lane_basic_length = 2.0  # [m]
	lane_line_width_real = 0.05  # [m]
	lane_width_basic_real = 3.0  # [m]
	laneDet_front_far = 15  # [m], better choose from (20, 40)
	laneDet_side_far = 2.5  # [m], better choose from (2, 6)
	laneDet_pixel_coe = 50  # [pixel/m], better choose from (10, 60)
	laneDet_line_div_density = 50  # [/m]
	laneDet_scan_length = 0.5  # [m]
	laneDet_lane_gap_length = 1.5  # [m]
	laneDet_lane_min_length = 4.0  # [m], better choose from (6, 10)
	laneDet_lane_image_min_length = 40  # [pixel]
	laneDet_lane_basic_length = 2.0  # [m]

	matching_alpha = 0.5
	matching_beta = 1
	landmark_distance_threshold = 2  # [m]
	landmark_front_far = 25  # [m]
	landmark_side_far = 10  # [m]