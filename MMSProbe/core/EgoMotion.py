import time
import os
import datetime
from scipy.optimize import curve_fit
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import glob

MAX_ITER = 500
THRESHOLD = 0.05
is_MELCO = 0

def fit_func(x, a, b):
    return a * np.cos(x-b)

def calculate_velocity(angle, velocity):
    max_inlier = 0
    best_popt = None
    #max_fitting_score = 0
    N = angle.shape[0]
    for i in range(MAX_ITER):
        n_inlier = 0
        #fitting_score = 0

        sample = np.random.choice(range(N), 4, replace=0)
        angle_sample = angle[sample]
        velocity_sample = velocity[sample]
        popt, _ = curve_fit(fit_func, angle_sample, velocity_sample, maxfev=5000)
        for j in range(N):
            diff = np.abs(velocity[j]-fit_func(angle[j], *popt))
            if diff < THRESHOLD:
                n_inlier += 1
                # fitting_score -= diff
        if n_inlier > max_inlier:
            # if fitting_score > max_fitting_score:
                best_popt = popt
                max_inlier = n_inlier
    return best_popt

if __name__ == '__main__':
    MELCOfile = r'/Volumes/Untitled/waseda_test_20221024/waseda_indoor_1/fusion_result.csv'
    folder = r'E:\MELCO\data\vison\IWR1443\1669780881.6607308/*.csv'
    output_folder = r'E:\workspace\MMSProbe_2022\data\MWR_results\vison/' + str(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    os.mkdir(output_folder)
    files = glob.glob(folder)
    files.sort()
    popts = []

    if is_MELCO:
        MWRData = dict()
        df_milliwave = pd.read_csv(MELCOfile, sep=',', encoding='utf-8')
        df_milliwave = df_milliwave[['UnixTime', 'M_Xrange[m]', 'M_Yrange[m]', 'M_Xvelocity[m/s]', 'M_Yvelocity[m/s]', 'ID']]
        MWR_time_list = sorted(list(set(df_milliwave['UnixTime'])))
        MWR_dict = {}
        for MWR_time in MWR_time_list:
            MWR_dict[MWR_time] = df_milliwave[df_milliwave['UnixTime'] == MWR_time]

        for i, MWR_time in enumerate(MWR_dict.keys()):
            df = MWR_dict[MWR_time]
            if len(df) < 4: continue
            x = df['M_Xrange[m]'].to_numpy()
            y = df['M_Yrange[m]'].replace(0, 1e-9).to_numpy()
            angle = np.arctan(x / y)

            v_x = df['M_Xvelocity[m/s]'].to_numpy()
            v_y = df['M_Yvelocity[m/s]'].to_numpy()
            velocity = np.sqrt(v_x**2 + v_y**2)
            sign = x * v_x/np.abs(x * v_x)
            velocity *= sign

            best_popt = calculate_velocity(angle, velocity)

            popts.append(best_popt)

            fig, p = plt.subplots(1, 1, sharex=True)
            p.plot(angle, velocity, "o", markersize=5, markerfacecolor="dodgerblue", markeredgewidth=0.0,
                   fillstyle="full")
            x = np.linspace(-1.5, 1.5, 300)
            p.plot(x, fit_func(np.array(x), *best_popt), alpha=0.5, color="crimson")
            p.grid(linewidth=0.5)
            fig.savefig("/Users/minamoto/Workspace/MELCO/MMSProbe_2022/test/RadarOdometry_turn/fit_best" + str(i) + ".png",
                        dpi=200, bbox_inches="tight")
            plt.close()
        result_a = []
        result_b = []
        for popt in popts:
            result_a.append(fit_func(0, *popt))
            # result_a.append(popt[0])
        plt.plot(result_a)
        plt.ylim(-1.5, 1.5)
        plt.savefig("/Users/minamoto/Workspace/MELCO/MMSProbe_2022/test/RadarOdometry/a.png", dpi=200,
                    bbox_inches="tight")

    else:
        for i, file in enumerate(files):
            df = pd.read_csv(file)
            if df['numObj'][0]<4: continue
            x = df['x'].to_numpy()
            y = df['y'].replace(0, 1e-9).to_numpy()
            angle = np.arctan(x / y)
            velocity = df['doppler'].to_numpy()
            best_popt = calculate_velocity(angle, velocity)
            if best_popt is None: continue
            popts.append(best_popt)

            fig, axes = plt.subplots(1, 2)
            axes[0].plot(angle, velocity, "o", markersize=5, markerfacecolor="dodgerblue", markeredgewidth=0.0,
                         fillstyle="full")
            abscissa = np.linspace(-1.5, 1.5, 300)
            if best_popt is not None:
                axes[0].plot(abscissa, fit_func(np.array(abscissa), *best_popt), alpha=0.5, color="crimson")
            axes[0].grid(linewidth=0.5)
            axes[0].set_xlim(-1.6, 1.6)
            axes[0].set_ylim(-1, 1)
            axes[0].set_title(str(file))
            #axes[0].set_aspect(1)
            # plt.savefig(output_folder + "/fit_best" + str(i) + ".png",
            #             dpi=200, bbox_inches="tight")
            # plt.close()
            axes[1].scatter(x, y, s=2)
            axes[1].set_xlim(-20, 20)
            axes[1].set_ylim(-5, 30)
            axes[1].set_aspect(1)
            plt.savefig(output_folder + "/xy" + str(i).zfill(4) + ".png",
                        dpi=200, bbox_inches="tight")
            plt.close('all')
        result_a = []
        result_b = []
        for popt in popts:
            result_a.append(fit_func(0, *popt))
            # result_a.append(popt[0])
        plt.plot(result_a)
        plt.ylim(-1.5, 1.5)
        plt.savefig(output_folder + "/a.png", dpi=200,
                    bbox_inches="tight")
        #     fig, p = plt.subplots(1, 1, sharex=True)
        #     p.plot(angle, velocity, "o", markersize=5, markerfacecolor="dodgerblue", markeredgewidth=0.0, fillstyle="full")
        #     x = np.linspace(-1.5, 1.5, 300)
        #     p.plot(x, fit_func(np.array(x), *best_popt), alpha=0.5, color="crimson")
        #     p.grid(linewidth=0.5)
        #     fig.savefig("/Users/minamoto/Workspace/MELCO/MMSProbe_2022/test/MELCORadar/fit_best" + str(i) + ".png",
        #                 dpi=200, bbox_inches="tight")
        #     plt.close()
        #
        # result_a = []
        # result_b = []
        # for popt in popts:
        #     result_a.append(fit_func(0, *popt))
        #     # result_a.append(popt[0])
        # plt.plot(result_a)
        # plt.ylim(-1.5,1.5)
        # plt.savefig("/Users/minamoto/Workspace/MELCO/MMSProbe_2022/test/MELCORadar/a.png", dpi=200, bbox_inches="tight")
