import numpy as np
import math
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
from MMSProbe.conf import Config

VEHICLE_LENGTH = Config.vehicle_length

# kalman filter error para
KF_ERR2_X = 1 ** 2  # [m^2]
KF_ERR2_Y = 1 ** 2  # [m^2]
KF_ERR2_PSI = 0.174533  ** 2  # [rad^2], 20 [deg]
KF_ERR2_V = 0.05 ** 2  # [m^2/s^2]
KF_ERR2_PSID = 0.0872665 ** 2


def normalize_angle(x):
    x = x % (2 * np.pi)  # force in range [0, 2 pi)
    if x > np.pi:  # move to [-pi, pi)
        x -= 2 * np.pi
    return x


def residual_h(a, b):
    y = a - b
    y[2] = normalize_angle(y[2])
    return y


def residual_x(a, b):
    y = a - b
    y[2] = normalize_angle(y[2])
    y[4] = normalize_angle(y[4])
    return y


def state_mean(sigmas, Wm):
    x = np.zeros(5)

    sum_sin = np.sum(np.dot(np.sin(sigmas[:, 2]), Wm))
    sum_cos = np.sum(np.dot(np.cos(sigmas[:, 2]), Wm))
    x[0] = np.sum(np.dot(sigmas[:, 0], Wm))
    x[1] = np.sum(np.dot(sigmas[:, 1], Wm))
    x[2] = math.atan2(sum_sin, sum_cos)
    x[3] = np.sum(np.dot(sigmas[:, 3], Wm))

    sum_sin_d = np.sum(np.dot(np.sin(sigmas[:, 4]), Wm))
    sum_cos_d = np.sum(np.dot(np.cos(sigmas[:, 4]), Wm))
    x[4] = math.atan2(sum_sin_d, sum_cos_d)
    return x


def z_mean(sigmas, Wm):
    x = np.zeros(4)

    sum_sin = np.sum(np.dot(np.sin(sigmas[:, 2]), Wm))
    sum_cos = np.sum(np.dot(np.cos(sigmas[:, 2]), Wm))
    x[0] = np.sum(np.dot(sigmas[:, 0], Wm))
    x[1] = np.sum(np.dot(sigmas[:, 1], Wm))
    x[2] = math.atan2(sum_sin, sum_cos)
    x[3] = np.sum(np.dot(sigmas[:, 3], Wm))
    return x


def CTRV(x, dt):
    x_pre, y_pre, yaw_pre, v_pre, yawd = x
    if np.abs(yawd) > 10:
        x_p = x_pre + (v_pre / yawd) * (np.sin(yaw_pre + yawd * dt) - np.sin(yaw_pre))
        y_p = y_pre + (v_pre / yawd) * (-np.cos(yaw_pre + yawd * dt) + np.cos(yaw_pre))
    else:
        x_p = x_pre + v_pre * np.cos(yaw_pre) * dt
        y_p = y_pre + v_pre * np.sin(yaw_pre) * dt
    v_p = v_pre
    yaw_p = yaw_pre + yawd * dt
    yawd_p = yawd

    return np.array([x_p, y_p, yaw_p, v_p, yawd_p])


def Hx(x):
    H = np.array([[1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1]])
    return H @ x


class CarUKF:
    def __init__(self):
        points = MerweScaledSigmaPoints(n=5, alpha=.00001, beta=2, kappa=0,
                                        subtract=None)
        self.ukf = UKF(dim_x=5, dim_z=5, dt=None, fx=CTRV, hx=Hx, points=points, x_mean_fn=None,
                       z_mean_fn=None, residual_x=None,
                       residual_z=None)
        self.ukf.x = None
        self.ukf.P = np.diag([3, 3, 0.349066, 2, 0.349066])

        self.ukf.Q = np.diag([10, 10, 0.0872665, 0.1, 0.0872665])  # Prediction Covariance
        self.ukf.R = np.diag([KF_ERR2_X,  # Measurement Covariance
                              KF_ERR2_Y,
                              KF_ERR2_PSI,
                              KF_ERR2_V,
                              KF_ERR2_PSID])

    def predict(self, dt):
        self.ukf.predict(dt)
        return True

    def update(self, z):
        self.ukf.update(z)
        return True

    def get_bias(self):
        return self.ukf.x

    def get_cov(self):
        return self.ukf.P
