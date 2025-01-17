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
import shutil
from config import Set_config

class MwrProcessing():
    def __init__(self):
        self.MAX_ITER = 500
        self.THRESHOLD = 0.05
        sc = Set_config()
        self.home_dir = sc.home_dir
        # self.amp_th = 65
        self.amp_th = sc.amp_th

    def fit_func(self, x, a, b):
        return a * np.cos(x-b)

    def calculate_velocity(self, angle, velocity):
        max_inlier = 0
        best_popt = None
        #max_fitting_score = 0
        N = angle.shape[0]
        velocity.index = range(0, N) #velocityデータフレームのインデックスを0から振り直し
        for i in range(self.MAX_ITER):
            #print(f"calculate parameter... {i+1}/{MAX_ITER}")
            n_inlier = 0
            inliers = []
            #fitting_score = 0

            sample = np.random.choice(range(N), 4, replace=0)
            angle_sample = angle[sample]
            velocity_sample = velocity[sample]
            popt, _ = curve_fit(self.fit_func, angle_sample, velocity_sample, maxfev=5000)
            for j in range(N):
                diff = np.abs(velocity[j]-self.fit_func(angle[j], *popt))
                if diff < self.THRESHOLD:
                    n_inlier += 1
                    inliers.append(j)
                    # fitting_score -= diff
            if n_inlier > max_inlier:
                # if fitting_score > max_fitting_score:
                    best_popt = popt
                    max_inlier = n_inlier
                    best_inliers = inliers
        if inliers == []:
            best_inliers = None
        print(f" best_popt:{best_popt}, best_inliers:{best_inliers}")
        return best_popt, best_inliers

    #amplitude閾値
    def write_amp_hist(self, file): #6mwr1 th:>80に設定
        df = pd.read_csv(file)
        amp = df["Amplitude"]
        plt.hist(amp, bins=300, edgecolor="black")
        plt.title("Histgram of Amplitude")
        plt.xlabel("Amplitude dB")
        plt.ylabel("Frequency")
        plt.xlim(left=40, right=130)
        plt.savefig(r"C:\Users\kenta shimoyama\Documents\amanolab\melco\簡易地図生成\mwr\hist.png")
        plt.close()

    def amp_th(self, file, th):
        index_list = []
        df = pd.read_csv(file)
        amp = df["Amplitude"]
        amp_list = list(amp)
        for i ,j in enumerate(amp_list):
            print(i)
            if j > th:
                index_list.append(i)
        inlier_df = df.iloc[index_list]

        return inlier_df

    def azimath_th(self, file):
        index_list = []
        df = pd.read_csv(file) 
        df.columns = ["X_cor", "Y_cor", "Amplitude", "UnixTime", "Xrange", "Yrange", "Velocity[m/s]"]
        x = df['Xrange'].to_numpy()
        y = df['Yrange'].replace(0, 1e-9).to_numpy()
        angle = np.degrees(np.arctan(x / y))
        for i, j in enumerate(angle):
            print(i)
            if abs(j) < 30:
                index_list.append(i)
        inlier_df = df.iloc[index_list]
        
        return inlier_df

    def amp_azimath_th(self, file, th):
        index_list = []
        if os.path.getsize(file) > 0:
            df = pd.read_csv(file)
        else:
            print("ファイルが空です。処理をスキップします。")
            return pd.DataFrame()
        df.columns = ["X_cor", "Y_cor", "Amplitude", "UnixTime", "Xrange", "Yrange", "Velocity[m/s]"]
        amp = df["Amplitude"]
        amp_list = list(amp)
        for i ,j in enumerate(amp_list):
            print(f"processing amp ... {i+1}")
            if j < th: #<
                index_list.append(i)
        inlier_amp_df = df.iloc[index_list]

        index_list = []
        x = inlier_amp_df['Xrange'].to_numpy()
        y = inlier_amp_df['Yrange'].replace(0, 1e-9).to_numpy()
        angle = np.degrees(np.arctan(x / y))
        for i, j in enumerate(angle):
            print(f"processing azi ... {i}")
            if abs(j) < 30:
                index_list.append(i)
        inlier_ampazi_df = inlier_amp_df.iloc[index_list]

        return inlier_ampazi_df
    
    def main(self, dir, is_MELCO, cosfitting):
        MELCOdir = self.home_dir + "\groundpoint_filtered/" + dir
        filename_list = os.listdir(MELCOdir)
        output_directory = self.home_dir + "\\amp_azimuth_filtered\\" + dir + "\\"
        if os.path.exists(output_directory) and os.path.isdir(output_directory):
            shutil.rmtree(output_directory)  # ディレクトリを削除
            print(f"{output_directory} は削除されました。")
        else:
            print(f"{output_directory} は存在しません。")

        for filename in filename_list:
            MELCOfile = MELCOdir + "/" + filename

            # #write amplitude histgram
            # write_amp_hist(MELCOfile)
            # #amplitude ourlier removing
            # ampth = 80
            # inlier_amp_df = amp_th(MELCOfile, ampth)
            # inlier_amp_df.to_csv(MELCOfile.split(".")[0]+"_"+str(ampth)+"inlier.csv", mode='w', header=True, index=False)
            # #azimath outlier removing
            # inlier_azi_df = azimath_th(MELCOfile)
            # inlier_azi_df.to_csv(MELCOfile.split(".")[0]+"_aziinlier.csv", mode='w', header=True, index=False)
            
            ampth = self.amp_th #65
            # ampth_list = [85, 90, 95, 100, 105, 110, 115, 120]
            # for ampth in ampth_list:
            inlier_ampazi_df = self.amp_azimath_th(MELCOfile, ampth)
            if cosfitting:
                #MELCO
                #output_folder = r'E:\workspace\MMSProbe_2022\data\MWR_results\vison/' + str(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
                output_folder = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\mwr+aqloc\fig/" + str(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
                fig_output_folder = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\mwr+aqloc\fit_best/" + str(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
                os.mkdir(output_folder)
                os.mkdir(fig_output_folder)
                outcsv_name = MELCOfile.split(".")[0] + "_"+str(ampth)+"azi_cosinlier.csv"
                #not MELCO
                folder = r'E:\MELCO\data\vison\IWR1443\1669780881.6607308/*.csv'
                files = glob.glob(folder)
                files.sort()
                popts = []

                if is_MELCO:
                    MWRData = dict()
                    #df_milliwave = pd.read_csv(MELCOfile, sep=',', encoding='utf-8')
                    #df_milliwave = df_milliwave[['FrameID','SymbolID', 'UnixTime', 'Xrange', 'Yrange', 'Amplitude', 'Velocity[m/s]']]
                    MWR_time_list = sorted(list(set(inlier_ampazi_df['UnixTime'])))
                    MWR_dict = {}
                    inlier_df=None
                    for MWR_time in MWR_time_list:
                        MWR_dict[MWR_time] = inlier_ampazi_df[inlier_ampazi_df['UnixTime'] == MWR_time]

                    for i, MWR_time in enumerate(MWR_dict.keys()):
                        print(f"execute frame{i+1}")
                        df = MWR_dict[MWR_time]
                        if len(df) < 4: continue
                        x = df['Xrange'].to_numpy()
                        y = df['Yrange'].replace(0, 1e-9).to_numpy()
                        angle = np.arctan(x / y) 

                        # v_x = df['M_Xvelocity[m/s]'].to_numpy()
                        # v_y = df['M_Yvelocity[m/s]'].to_numpy()
                        # velocity = np.sqrt(v_x**2 + v_y**2)
                        # sign = x * v_x/np.abs(x * v_x)
                        velocity = df["Velocity[m/s]"]

                        best_popt, best_inliers = self.calculate_velocity(angle, velocity) #cosフィッティングのためのransacパラメータ

                        if best_inliers is not None:
                            inlier_df = df.iloc[best_inliers]
                            x_inlier = inlier_df['Xrange'].to_numpy()
                            y_inlier = inlier_df['Yrange'].replace(0, 1e-9).to_numpy()
                            angle_inlier = np.arctan(x_inlier / y_inlier) 
                            velocity_inlier = inlier_df["Velocity[m/s]"]
                            fig2, p2 = plt.subplots(1, 1, sharex=True) #フレーム内のうちインライアの点
                            p2.plot(angle_inlier, velocity_inlier, "o", markersize=5, markerfacecolor="dodgerblue", markeredgewidth=0.0,
                                fillstyle="full")
                            x = np.linspace(-1.5, 1.5, 300)
                            p2.plot(x, self.fit_func(np.array(x), *best_popt), alpha=0.5, color="crimson")
                            p2.grid(linewidth=0.5)
                            p2.set_ylim(-7.5, 7.5)
                            fig2.savefig(fig_output_folder + "/fit2_best" + str(i) + ".png",
                                    dpi=200, bbox_inches="tight")


                        fig, p = plt.subplots(1, 1, sharex=True) #フレーム内の全ての点             
                        p.plot(angle, velocity, "o", markersize=5, markerfacecolor="dodgerblue", markeredgewidth=0.0,
                            fillstyle="full")
                        if best_popt is not None:
                            popts.append(best_popt)
                            x = np.linspace(-1.5, 1.5, 300)
                            p.plot(x, self.fit_func(np.array(x), *best_popt), alpha=0.5, color="crimson")
                            p.grid(linewidth=0.5)
                        p.set_ylim(-7.5, 7.5)
                        fig.savefig(fig_output_folder + "/fit_best" + str(i) + ".png",
                                    dpi=200, bbox_inches="tight")
                        
                        plt.close()

                        if os.path.exists(outcsv_name):
                            # ファイルが存在する場合は追記モードでヘッダーなし
                            if inlier_df is not None:
                                inlier_df.to_csv(outcsv_name, mode='a', header=False, index=False)
                                inlier_df = None
                        else:
                            # ファイルが存在しない場合は新規作成でヘッダー付き
                            if inlier_df is not None:
                                inlier_df.to_csv(outcsv_name, mode='w', header=False, index=False)
                                inlier_df = None
                                    

                    

                    result_a = []
                    result_b = []
                    for popt in popts:
                        result_a.append(self.fit_func(0, *popt))
                        # result_a.append(popt[0])
                    plt.plot(result_a)
                    plt.ylim(-1.5, 1.5)
                    plt.savefig(output_folder+"/param.png", dpi=200,
                                bbox_inches="tight")

                else:
                    for i, file in enumerate(files):
                        df = pd.read_csv(file)
                        if df['numObj'][0]<4: continue
                        x = df['x'].to_numpy()
                        y = df['y'].replace(0, 1e-9).to_numpy()
                        angle = np.arctan(x / y)
                        velocity = df['doppler'].to_numpy()
                        best_popt = self.calculate_velocity(angle, velocity)
                        if best_popt is None: continue
                        popts.append(best_popt)

                        fig, axes = plt.subplots(1, 2)
                        axes[0].plot(angle, velocity, "o", markersize=5, markerfacecolor="dodgerblue", markeredgewidth=0.0,
                                    fillstyle="full")
                        abscissa = np.linspace(-1.5, 1.5, 300)
                        if best_popt is not None:
                            axes[0].plot(abscissa, self.fit_func(np.array(abscissa), *best_popt), alpha=0.5, color="crimson")
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
                        result_a.append(self.fit_func(0, *popt))
                        # result_a.append(popt[0])
                    plt.plot(result_a)
                    plt.ylim(-1.5, 1.5)
                    plt.savefig(output_folder + "/a.png", dpi=200,
                                bbox_inches="tight")
                    #     fig, p = plt.subplots(1, 1, sharex=True)
                    #     p.plot(angle, velocity, "o", markersize=5, markerfacecolor="dodgerblue", markeredgewidth=0.0, fillstyle="full")
                    #     x = np.linspace(-1.5, 1.5, 300)
                    #     p.plot(x, self.fit_func(np.array(x), *best_popt), alpha=0.5, color="crimson")
                    #     p.grid(linewidth=0.5)
                    #     fig.savefig("/Users/minamoto/Workspace/MELCO/MMSProbe_2022/test/MELCORadar/fit_best" + str(i) + ".png",
                    #                 dpi=200, bbox_inches="tight")
                    #     plt.close()
                    #
                    # result_a = []
                    # result_b = []
                    # for popt in popts:
                    #     result_a.append(self.fit_func(0, *popt))
                    #     # result_a.append(popt[0])
                    # plt.plot(result_a)
                    # plt.ylim(-1.5,1.5)
                    # plt.savefig("/Users/minamoto/Workspace/MELCO/MMSProbe_2022/test/MELCORadar/a.png", dpi=200, bbox_inches="tight")


            else:
                output_filename = output_directory + filename.split(".")[0]+"_aziamp"+str(ampth)+".csv"
                os.makedirs(output_directory, exist_ok=True)
                inlier_ampazi_df.to_csv(output_filename, mode='w', header=False, index=False)



if __name__ == '__main__':
    mp = MwrProcessing()
    cosfitting = False
    is_MELCO = 1 #MELCO製:1 not MELCO製(TI製?):0
    # MELCOdir = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\temp_using\mwr+aqloc/"
    # file_list = ["symbols_0907_1left_extracted_2.csv", "symbols_0907_1right_extracted_2.csv", "symbols_0907_2left_extracted_2.csv", "symbols_0907_2right_extracted_2.csv", "symbols_0907_3left_extracted_2.csv", "symbols_0907_3right_extracted_2.csv"]
    # file_list = ["symbols_0907_1left_extracted_2.csv", "symbols_0907_2left_extracted_2.csv", "symbols_0907_3left_extracted_2.csv", "symbols_0907_1right_extracted_2.csv", "symbols_0907_2right_extracted_2.csv", "symbols_0907_3right_extracted_2.csv"]
    # file_list = ["symbols_vison_left_extracted_15.csv", "symbols_vison_left_extracted_20.csv", "symbols_vison_left_extracted_30.csv", "symbols_vison_left_extracted_35.csv", "symbols_vison_right_extracted_15.csv", "symbols_vison_right_extracted_20.csv", "symbols_vison_right_extracted_30.csv", "symbols_vison_right_extracted_35.csv"]
    dir_list = ["symbols_0907_1left", "symbols_0907_2left", "symbols_0907_3left", "symbols_0907_1right", "symbols_0907_2right", "symbols_0907_3right"]

    for dir in dir_list:
        mp.main(dir, is_MELCO, cosfitting)

        # MELCOfile = MELCOdir + file

        # # #write amplitude histgram
        # # write_amp_hist(MELCOfile)
        # # #amplitude ourlier removing
        # # ampth = 80
        # # inlier_amp_df = amp_th(MELCOfile, ampth)
        # # inlier_amp_df.to_csv(MELCOfile.split(".")[0]+"_"+str(ampth)+"inlier.csv", mode='w', header=True, index=False)
        # # #azimath outlier removing
        # # inlier_azi_df = azimath_th(MELCOfile)
        # # inlier_azi_df.to_csv(MELCOfile.split(".")[0]+"_aziinlier.csv", mode='w', header=True, index=False)
        
        # ampth = 65
        # # ampth_list = [85, 90, 95, 100, 105, 110, 115, 120]
        # # for ampth in ampth_list:
        # inlier_ampazi_df = self.amp_azimath_th(MELCOfile, ampth)
        # if cosfitting:
        #     #MELCO
        #     #output_folder = r'E:\workspace\MMSProbe_2022\data\MWR_results\vison/' + str(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
        #     output_folder = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\mwr+aqloc\fig/" + str(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
        #     fig_output_folder = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\mwr+aqloc\fit_best/" + str(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
        #     os.mkdir(output_folder)
        #     os.mkdir(fig_output_folder)
        #     outcsv_name = MELCOfile.split(".")[0] + "_"+str(ampth)+"azi_cosinlier.csv"
        #     #not MELCO
        #     folder = r'E:\MELCO\data\vison\IWR1443\1669780881.6607308/*.csv'
        #     files = glob.glob(folder)
        #     files.sort()
        #     popts = []

        #     if is_MELCO:
        #         MWRData = dict()
        #         #df_milliwave = pd.read_csv(MELCOfile, sep=',', encoding='utf-8')
        #         #df_milliwave = df_milliwave[['FrameID','SymbolID', 'UnixTime', 'Xrange', 'Yrange', 'Amplitude', 'Velocity[m/s]']]
        #         MWR_time_list = sorted(list(set(inlier_ampazi_df['UnixTime'])))
        #         MWR_dict = {}
        #         inlier_df=None
        #         for MWR_time in MWR_time_list:
        #             MWR_dict[MWR_time] = inlier_ampazi_df[inlier_ampazi_df['UnixTime'] == MWR_time]

        #         for i, MWR_time in enumerate(MWR_dict.keys()):
        #             print(f"execute frame{i+1}")
        #             df = MWR_dict[MWR_time]
        #             if len(df) < 4: continue
        #             x = df['Xrange'].to_numpy()
        #             y = df['Yrange'].replace(0, 1e-9).to_numpy()
        #             angle = np.arctan(x / y) 

        #             # v_x = df['M_Xvelocity[m/s]'].to_numpy()
        #             # v_y = df['M_Yvelocity[m/s]'].to_numpy()
        #             # velocity = np.sqrt(v_x**2 + v_y**2)
        #             # sign = x * v_x/np.abs(x * v_x)
        #             velocity = df["Velocity[m/s]"]

        #             best_popt, best_inliers = self.calculate_velocity(angle, velocity) #cosフィッティングのためのransacパラメータ

        #             if best_inliers is not None:
        #                 inlier_df = df.iloc[best_inliers]
        #                 x_inlier = inlier_df['Xrange'].to_numpy()
        #                 y_inlier = inlier_df['Yrange'].replace(0, 1e-9).to_numpy()
        #                 angle_inlier = np.arctan(x_inlier / y_inlier) 
        #                 velocity_inlier = inlier_df["Velocity[m/s]"]
        #                 fig2, p2 = plt.subplots(1, 1, sharex=True) #フレーム内のうちインライアの点
        #                 p2.plot(angle_inlier, velocity_inlier, "o", markersize=5, markerfacecolor="dodgerblue", markeredgewidth=0.0,
        #                     fillstyle="full")
        #                 x = np.linspace(-1.5, 1.5, 300)
        #                 p2.plot(x, self.fit_func(np.array(x), *best_popt), alpha=0.5, color="crimson")
        #                 p2.grid(linewidth=0.5)
        #                 p2.set_ylim(-7.5, 7.5)
        #                 fig2.savefig(fig_output_folder + "/fit2_best" + str(i) + ".png",
        #                         dpi=200, bbox_inches="tight")


        #             fig, p = plt.subplots(1, 1, sharex=True) #フレーム内の全ての点             
        #             p.plot(angle, velocity, "o", markersize=5, markerfacecolor="dodgerblue", markeredgewidth=0.0,
        #                 fillstyle="full")
        #             if best_popt is not None:
        #                 popts.append(best_popt)
        #                 x = np.linspace(-1.5, 1.5, 300)
        #                 p.plot(x, self.fit_func(np.array(x), *best_popt), alpha=0.5, color="crimson")
        #                 p.grid(linewidth=0.5)
        #             p.set_ylim(-7.5, 7.5)
        #             fig.savefig(fig_output_folder + "/fit_best" + str(i) + ".png",
        #                         dpi=200, bbox_inches="tight")
                    
        #             plt.close()

        #             if os.path.exists(outcsv_name):
        #                 # ファイルが存在する場合は追記モードでヘッダーなし
        #                 if inlier_df is not None:
        #                     inlier_df.to_csv(outcsv_name, mode='a', header=False, index=False)
        #                     inlier_df = None
        #             else:
        #                 # ファイルが存在しない場合は新規作成でヘッダー付き
        #                 if inlier_df is not None:
        #                     inlier_df.to_csv(outcsv_name, mode='w', header=False, index=False)
        #                     inlier_df = None
                                

                

        #         result_a = []
        #         result_b = []
        #         for popt in popts:
        #             result_a.append(self.fit_func(0, *popt))
        #             # result_a.append(popt[0])
        #         plt.plot(result_a)
        #         plt.ylim(-1.5, 1.5)
        #         plt.savefig(output_folder+"/param.png", dpi=200,
        #                     bbox_inches="tight")

        #     else:
        #         for i, file in enumerate(files):
        #             df = pd.read_csv(file)
        #             if df['numObj'][0]<4: continue
        #             x = df['x'].to_numpy()
        #             y = df['y'].replace(0, 1e-9).to_numpy()
        #             angle = np.arctan(x / y)
        #             velocity = df['doppler'].to_numpy()
        #             best_popt = calculate_velocity(angle, velocity)
        #             if best_popt is None: continue
        #             popts.append(best_popt)

        #             fig, axes = plt.subplots(1, 2)
        #             axes[0].plot(angle, velocity, "o", markersize=5, markerfacecolor="dodgerblue", markeredgewidth=0.0,
        #                         fillstyle="full")
        #             abscissa = np.linspace(-1.5, 1.5, 300)
        #             if best_popt is not None:
        #                 axes[0].plot(abscissa, self.fit_func(np.array(abscissa), *best_popt), alpha=0.5, color="crimson")
        #             axes[0].grid(linewidth=0.5)
        #             axes[0].set_xlim(-1.6, 1.6)
        #             axes[0].set_ylim(-1, 1)
        #             axes[0].set_title(str(file))
        #             #axes[0].set_aspect(1)
        #             # plt.savefig(output_folder + "/fit_best" + str(i) + ".png",
        #             #             dpi=200, bbox_inches="tight")
        #             # plt.close()
        #             axes[1].scatter(x, y, s=2)
        #             axes[1].set_xlim(-20, 20)
        #             axes[1].set_ylim(-5, 30)
        #             axes[1].set_aspect(1)
        #             plt.savefig(output_folder + "/xy" + str(i).zfill(4) + ".png",
        #                         dpi=200, bbox_inches="tight")
        #             plt.close('all')
        #         result_a = []
        #         result_b = []
        #         for popt in popts:
        #             result_a.append(self.fit_func(0, *popt))
        #             # result_a.append(popt[0])
        #         plt.plot(result_a)
        #         plt.ylim(-1.5, 1.5)
        #         plt.savefig(output_folder + "/a.png", dpi=200,
        #                     bbox_inches="tight")
        #         #     fig, p = plt.subplots(1, 1, sharex=True)
        #         #     p.plot(angle, velocity, "o", markersize=5, markerfacecolor="dodgerblue", markeredgewidth=0.0, fillstyle="full")
        #         #     x = np.linspace(-1.5, 1.5, 300)
        #         #     p.plot(x, self.fit_func(np.array(x), *best_popt), alpha=0.5, color="crimson")
        #         #     p.grid(linewidth=0.5)
        #         #     fig.savefig("/Users/minamoto/Workspace/MELCO/MMSProbe_2022/test/MELCORadar/fit_best" + str(i) + ".png",
        #         #                 dpi=200, bbox_inches="tight")
        #         #     plt.close()
        #         #
        #         # result_a = []
        #         # result_b = []
        #         # for popt in popts:
        #         #     result_a.append(self.fit_func(0, *popt))
        #         #     # result_a.append(popt[0])
        #         # plt.plot(result_a)
        #         # plt.ylim(-1.5,1.5)
        #         # plt.savefig("/Users/minamoto/Workspace/MELCO/MMSProbe_2022/test/MELCORadar/a.png", dpi=200, bbox_inches="tight")


        # else:
        #     inlier_ampazi_df.to_csv(MELCOfile.split(".")[0]+"_"+str(ampth)+"azi_inlier.csv", mode='w', header=True, index=False)
