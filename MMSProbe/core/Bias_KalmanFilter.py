import numpy as np
from filterpy.kalman import KalmanFilter

rho = 1
std_x_lane = 30
# std_x_lane = 15
std_y_lane = 14
# std_y_lane = 7
std_yaw_lane = 0.0872665
std_x_MWR = 7
std_y_MWR = 10
std_yaw_MWR = 0.0872665


class BiasKF:
    def __init__(self):
        self.kf = KalmanFilter(dim_x=3, dim_z=3)

        self.kf.x = np.zeros(3)  # x:[lane x, lane y, lane yaw, mwr x, mwr y, mwr z]
        self.kf.F = np.diag([rho, rho, rho])  # 一次マルコフ過程係数，よくわからないのでとりあえず適当
        self.kf.H = np.eye(3, 3)
        self.kf.P = np.diag([2, 2, 0.17 * 6])
        # self.kf.P = np.diag([1, 1, 0.17 * 3])

        self.kf.R = np.diag(
            [std_x_lane ** 2, std_y_lane ** 2, std_yaw_lane ** 2])

        self.kf.Q = np.diag([0.5, 0.5, 0.0872665]) # Prediction Covariance

    def predict(self):
        self.kf.predict()
        return True

    def update(self, z):
        self.kf.update(z)
        return True

    def get_bias(self):
        return self.kf.x

    def get_cov(self):
        return self.kf.P
