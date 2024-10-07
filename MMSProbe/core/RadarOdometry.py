import numpy as np
import pandas as pd
import glob
import os
import csv
import argparse
from tqdm import tqdm
from scipy.optimize import curve_fit

MAX_ITER = 500  # RANSACのiteration数
THRESHOLD = 0.05  # 速度推定時各点からフィッティング曲線の間の差の閾値
MAX_VELOCITY = 1.3  # 最大速度
MAX_ACCELERATION = 10  # 最大加速度
VEHICLE_LENGTH = 0.5
MAX_ANGULAR_VELOCITY = 1
MAX_ANGULAR_ACCELERATION = 3.5
ANGLULAR_VELOCITY_BIAS = 0  # 角速度バイアス


def fit_func(x, a, b):
    return a * np.cos(x - b)


def calculate_velocity(angle, velocity):
    max_inlier = 0
    best_popt = None
    # max_fitting_score = 0
    N = angle.shape[0]
    for i in range(MAX_ITER):
        n_inlier = 0
        # fitting_score = 0

        sample = np.random.choice(range(N), 3, replace=0)
        angle_sample = angle[sample]
        velocity_sample = velocity[sample]
        popt, _ = curve_fit(fit_func, angle_sample, velocity_sample, maxfev=10000)
        for j in range(N):
            diff = np.abs(velocity[j] - fit_func(angle[j], *popt))
            if diff < THRESHOLD:
                n_inlier += 1
                # fitting_score -= diff
        if n_inlier > max_inlier:
            # if fitting_score > max_fitting_score:
            best_popt = popt
            max_inlier = n_inlier
    return best_popt


class RadarOdometry:
    def __init__(self, time_init):
        Frame_init = {'time': time_init, 'time increment': 0., 'popts': np.array([0, 0]), 'position': np.array([0, 0]),
                      'x': 0., 'y': 0., 'pose': np.pi / 2, 'velocity': 0., 'angular velocity': 0.}
        self._data = [Frame_init]

        pass

    def run(self, MWRData: pd.DataFrame, t_current):
        Frame_data = {}
        if MWRData['numObj'][0] < 3:
            return 1
        x = MWRData['x'].to_numpy()
        y = MWRData['y'].replace(0, 1e-9).to_numpy()
        angle = np.arctan(x / y)
        velocity = MWRData['doppler'].to_numpy()

        Frame_data['time'] = t_current
        t_increment = t_current - self._data[-1]['time']
        Frame_data['time increment'] = t_increment  # 時間増量

        best_popt = calculate_velocity(angle, velocity)
        if best_popt is not None:
            if best_popt[0] > 0:  # 角度で
                best_popt[0] = -best_popt[0]
                best_popt[1] = best_popt[1] - np.pi
            acceleration = (best_popt[0] - self._data[-1]['popts'][0]) / t_increment
            if np.abs(acceleration) > MAX_ACCELERATION:
                best_popt = self._data[-1]['popts']
            if best_popt[0] > MAX_VELOCITY or best_popt[0] < -MAX_VELOCITY:
                best_popt = self._data[-1]['popts']
            Frame_data['popts'] = best_popt
            Frame_data['velocity'] = -best_popt[0]
        else:
            Frame_data['popts'] = self._data[-1]['popts']
            Frame_data['velocity'] = self._data[-1]['velocity']

        v_s = -Frame_data['velocity']
        angular_velocity = np.sin(Frame_data['popts'][1]) / VEHICLE_LENGTH * v_s
        angular_acceleration = (angular_velocity - self._data[-1]['angular velocity']) / t_increment  # 角加速度判定
        # if np.abs(angular_acceleration) > MAX_ANGULAR_ACCELERATION:
        #     angular_velocity = self._data[-1]['angular velocity']
        # if angular_velocity > MAX_ANGULAR_VELOCITY or angular_velocity < -MAX_ANGULAR_VELOCITY: # 角速度判定
        #     angular_velocity = self._data[-1]['angular velocity']
        Frame_data['angular velocity'] = angular_velocity - ANGLULAR_VELOCITY_BIAS

        # 位置姿勢変化分
        if t_increment < 1:
            pos_pre = self._data[-1]['position']
            pose_pre = self._data[-1]['pose']
            # pos_increment = Frame_data['popts'] * t_increment
            x_increment = Frame_data['velocity'] * np.cos(pose_pre) * t_increment  # 今の速度で今の位置を推定？
            y_increment = Frame_data['velocity'] * np.sin(pose_pre) * t_increment
            pos_increment = np.array([x_increment, y_increment])
            pose_increment = Frame_data['angular velocity'] * t_increment if np.abs(
                Frame_data['angular velocity']) > 0.25 else 0
        else:
            pos_increment = np.array([0, 0])
            pose_increment = 0.
            x_increment = 0.
            y_increment = 0.
        x = self._data[-1]['x'] + x_increment
        y = self._data[-1]['y'] + y_increment
        pos = self._data[-1]['position'] + pos_increment
        pose = self._data[-1]['pose'] + pose_increment
        Frame_data['position'] = pos
        Frame_data['pose'] = pose
        Frame_data['x'] = x
        Frame_data['y'] = y
        self._data.append(Frame_data)
        return True

    def get_data(self):
        return self._data


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-t', '--init_time', type=str, help='initial time', required=True)
    # parser.add_argument('-i', '--input_folder', type=str, help='input folder', required=True)
    # parser.add_argument('-o', '--output_folder', type=str, help='output folder', required=True)
    #
    # args = parser.parse_args()
    # init_time = args.init_time
    # folder = args.input_folder
    # output_folder = args.output_folder

    init_time = '1671776159.5958698'
    folder = r'E:\MELCO\data\20221223/'
    output_folder = r'E:\workspace\MMSProbe_2022\data\MWR_results\20221223/'

    path = os.path.join(folder, str(init_time), '*.csv')

    files = glob.glob(path)
    files.sort()
    RO = RadarOdometry(float(init_time))

    # for i, file in enumerate(tqdm(files[2079:2171])):
    for i, file in enumerate(tqdm(files)):
        df = pd.read_csv(file)
        name = os.path.basename(file).split('.')
        time = float(name[0] + '.' + name[1])
        RO.run(df, time)
    df = pd.DataFrame(RO.get_data())
    # df.to_csv(r'E:\workspace\MMSProbe_2022\data\MWR_results/20221123/1669193390.9330957.csv')
    # print(df)

    field_name = ['time', 'time increment', 'popts', 'position', 'x', 'y', 'pose', 'velocity', 'angular velocity']
    output_path = os.path.join(output_folder, str(init_time) + '.csv')
    with open(output_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_name)
        writer.writeheader()
        writer.writerows(RO.get_data())
