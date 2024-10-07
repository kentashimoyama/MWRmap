import pandas as pd
import numpy as np
from datetime import datetime, timezone
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

MMS_TIME2UTC = 0
MMS_TIME_DELAY = 0
# MMS_TIME_DELAY = -9.3 + 9 * 60 * 60
GNSS_MMS_OFFSET = np.array([0, 0])  # MMSを原点として，GNSSアンテナの位置


class ErrorAnalyzer:
    def __init__(self):
        df_GPS = pd.read_csv(r'C:\workspace\MELCO\data\results\debug_202312271707/Maingps_log.csv',
                             header=None, usecols=[0, 1, 2, 3], encoding='utf-8')
        self.GPS_data = df_GPS[[0, 1, 2, 3]]
        self.GPS_poses = self.GPS_data.to_numpy().tolist()
        self.GPS_time = df_GPS[[0]].to_numpy()[:,0].tolist()

        # 今回はいらない
        # df_MWR = pd.read_csv(r'E:\workspace\MMSProbe_2022\data\thesis_result_決定版\220217060807_物標/Mainresult.csv',
        #                           header=None, usecols=[0, 1, 2, 3], encoding='utf-8')
        # self.MWR_data = df_MWR[[0, 1, 2, 3]]
        # self.MWR_poses = self.MWR_data.to_numpy().tolist()
        # self.MWR_time = df_MWR[[0]].to_numpy()[:,0].tolist()

        # 今回はいらない
        # df_lane = pd.read_csv(r'C:\workspace\MELCO\data\results\debug_202312271707/Mainresult.csv',
        #                           header=None, usecols=[0, 1, 2, 3], encoding='utf-8')
        # # df_lane = pd.read_csv(r'E:\workspace\MMSProbe_2022\data\thesis_result_決定版\220217060807_区画線/Mainresult.csv',
        # #     header=None, usecols=[0, 1, 2, 3], encoding='utf-8')
        # self.lane_data = df_lane[[0, 1, 2, 3]]
        # self.lane_poses = self.lane_data.to_numpy().tolist()
        # self.lane_time = df_lane[[0]].to_numpy()[:,0].tolist()

        # df_MWRlane = pd.read_csv(r'E:\workspace\MMSProbe_2022\data\results\debug_202301071825/Mainresult.csv',
        #                           header=None, usecols=[0, 1, 2, 3], encoding='utf-8')
        df_MWRlane = pd.read_csv(r'C:\workspace\MELCO\data\results\debug_202312271707/Mainresult.csv', header=None, usecols=[0, 1, 2, 3], encoding='utf-8')
        self.MWRlane_data = df_MWRlane[[0, 1, 2, 3]]
        self.MWRlane_poses = self.MWRlane_data.to_numpy().tolist()
        self.MWRlane_time = df_MWRlane[[0]].to_numpy()[:,0].tolist()

        self.MMS_TimePose_dict = {}
        date_utc = datetime.strptime("20220217", "%Y%m%d")
        date_utc = date_utc.replace(tzinfo=timezone.utc)
        weekTime = (date_utc.weekday() + 1) % 7 * 86400
        df_MMS = pd.read_csv(
            r'C:\workspace\MELCO\data\20231227_KIKUICHO\AQLOC/122701_ichimill19hokan.csv',
            header=None)
        time_MMS = df_MMS[0]
        time_MMS = date_utc.timestamp() - weekTime + time_MMS + MMS_TIME2UTC + MMS_TIME_DELAY
        for i, time in tqdm(enumerate(time_MMS)):
            angle_GT = df_MMS.iloc[i][3]
            try:
                angle_GT = (-float(angle_GT) + 630) % 360
            except ValueError:
                # もし変換できない場合は、エラーメッセージを表示して何らかの対処を行う
                print(f"Could not convert string to float: {angle_GT}")
                # 例外が発生した場合の処理を追加する（例: デフォルトの値を代入するなど）
                angle_GT = 0.0
            angle_GT = np.deg2rad(angle_GT)

            self.MMS_TimePose_dict[time] = [time, df_MMS.iloc[i][2], df_MMS.iloc[i][1], angle_GT]
        self.time_MMS_list = list(self.MMS_TimePose_dict.keys())
        self.i_flag = 0
        self.GT_Pose = self.find_GT_Pose(self.MWRlane_data.iloc[0][0])
        self.GT_Poses = [self.GT_Pose]
        self.GT_color = "green"
        self.time = None

    def find_GT_Pose(self, time_MMSProbe):
        # 三次元のオイラー角は線形補間しちゃだめなんだけど一次元ならOK？と信じる
        for i, time in enumerate(self.time_MMS_list[self.i_flag:]):
            if time <= time_MMSProbe < self.time_MMS_list[self.i_flag + i + 1]:

                self.i_flag += i
                alpha = (time_MMSProbe - time) / (self.time_MMS_list[self.i_flag + 1] - time)
                GT_TimePose = (1 - alpha) * np.asarray(self.MMS_TimePose_dict[self.time_MMS_list[self.i_flag]]) \
                              + alpha * np.asarray(self.MMS_TimePose_dict[self.time_MMS_list[self.i_flag + 1]])
                return GT_TimePose[1:4].tolist()

    def calculate_error(self, Poses):
        self.i_flag = 0
        x_errors = []
        y_errors = []
        yaw_errors = []
        try:
            for pose in Poses:
                if not((pose[0] in self.GPS_time) and (pose[0] in self.lane_time) and (pose[0] in self.MWR_time)
                       and (pose[0] in self.MWRlane_time)):
                    print('False')
                    continue
                GT_Pose = self.find_GT_Pose(pose[0])
                R = np.array([[np.cos(GT_Pose[2]), -np.sin(GT_Pose[2])],
                              [np.sin(GT_Pose[2]), np.cos(GT_Pose[2])]])
                # GT_Pose[:2] = GT_Pose[:2] + R @ MMSProbe_pose
                yaw_vector = np.array([np.cos(GT_Pose[2]), np.sin(GT_Pose[2])])

                x_error = np.dot((np.asarray(pose[1:3]) - np.asarray(GT_Pose[0:2])),yaw_vector) - 0.2 # 内積で進行方向の誤差を出す, GNSSとMMS軌跡中心20 cm
                x_errors.append(x_error)
                y_error = np.cross((np.asarray(pose[1:3]) - np.asarray(GT_Pose[0:2])),yaw_vector) - 0.34 # 外積で横方向の誤差を出す, GNSSとMMS軌跡中心340 cm
                y_errors.append(y_error)
                if np.abs(np.rad2deg(pose[3]) - np.rad2deg(GT_Pose[2]))>180:
                    yaw_error = np.rad2deg(pose[3]) - np.rad2deg(GT_Pose[2]) + 360
                else:
                    yaw_error = np.rad2deg(pose[3]) - np.rad2deg(GT_Pose[2])
                yaw_errors.append(yaw_error)
        except TypeError:
            print(pose[0])
        return x_errors, y_errors, yaw_errors


    def run(self): #test 1-2 343 500 #test 2-1 280 420
        # inittime = 1645110257.982  # 303
        # step = 157

        inittime = 1703689442.38   # 各csv微妙に行数違うので時間で探す.この時間以降は多分おｋ
        step = 140                        #807

        GPSstart = self.GPS_time.index(inittime)
        GPSend = GPSstart + step
        print('GPSstart:',GPSstart,'GPSend:',GPSend)

        lanestart = self.lane_time.index(inittime)
        laneend = lanestart + step
        print('lanestart:',lanestart,'laneend:',laneend)

        MWRstart = self.MWR_time.index(inittime)
        MWRend = MWRstart + step
        print('MWRstart:',MWRstart,'MWRend:',MWRend)

        MWRlanestart = self.MWRlane_time.index(inittime)
        MWRlaneend = MWRlanestart + step
        print('MWRlanestart:',MWRlanestart,'MWRlaneend:',MWRlaneend)

        GPS_x_errors, GPS_y_errors, GPS_yaw_errors = self.calculate_error(self.GPS_poses[GPSstart:GPSend])
        lane_x_errors, lane_y_errors, lane_yaw_errors = self.calculate_error(self.lane_poses[lanestart:laneend])
        MWR_x_errors, MWR_y_errors, MWR_yaw_errors = self.calculate_error(self.MWR_poses[MWRstart:MWRend])
        MWRlane_x_errors, MWRlane_y_errors, MWRlane_yaw_errors = self.calculate_error(self.MWRlane_poses[MWRlanestart:MWRlaneend])
        print(len(GPS_x_errors), len(lane_x_errors),len(MWR_x_errors), len(MWRlane_x_errors))
        GPS_mse_x = mean_squared_error(np.asarray(GPS_x_errors), np.zeros(len(GPS_x_errors)))
        GPS_mse_y = mean_squared_error(np.asarray(GPS_y_errors), np.zeros(len(GPS_y_errors)))
        GPS_mse_yaw = mean_squared_error(np.asarray(GPS_yaw_errors), np.zeros(len(GPS_yaw_errors)))
        print('GPS_mse_x', GPS_mse_x)
        print('GPS_mse_y', GPS_mse_y)
        print('GPS_mse_yaw', GPS_mse_yaw)
        print(np.sqrt(GPS_mse_x), np.sqrt(GPS_mse_y), np.sqrt(GPS_mse_yaw))

        lane_mse_x = mean_squared_error(np.asarray(lane_x_errors), np.zeros(len(lane_x_errors)))
        lane_mse_y = mean_squared_error(np.asarray(lane_y_errors), np.zeros(len(lane_y_errors)))
        lane_mse_yaw = mean_squared_error(np.asarray(lane_yaw_errors), np.zeros(len(lane_yaw_errors)))
        print('lane_mse_x', lane_mse_x)
        print('lane_mse_y', lane_mse_y)
        print('lane_mse_yaw', lane_mse_yaw)
        print(np.sqrt(lane_mse_x), np.sqrt(lane_mse_y), np.sqrt(lane_mse_yaw))

        MWR_mse_x = mean_squared_error(np.asarray(MWR_x_errors), np.zeros(len(MWR_x_errors)))
        MWR_mse_y = mean_squared_error(np.asarray(MWR_y_errors), np.zeros(len(MWR_y_errors)))
        MWR_mse_yaw = mean_squared_error(np.asarray(MWR_yaw_errors), np.zeros(len(MWR_yaw_errors)))
        print('MWR_mse_x', MWR_mse_x)
        print('MWR_mse_y', MWR_mse_y)
        print('MWR_mse_yaw', MWR_mse_yaw)
        print(np.sqrt(MWR_mse_x), np.sqrt(MWR_mse_y), np.sqrt(MWR_mse_yaw))

        MWRlane_mse_x = mean_squared_error(np.asarray(MWRlane_x_errors), np.zeros(len(MWRlane_x_errors)))
        MWRlane_mse_y = mean_squared_error(np.asarray(MWRlane_y_errors), np.zeros(len(MWRlane_y_errors)))
        MWRlane_mse_yaw = mean_squared_error(np.asarray(MWRlane_yaw_errors), np.zeros(len(MWRlane_yaw_errors)))
        print('MWRlane_mse_x', MWRlane_mse_x)
        print('MWRlane_mse_y', MWRlane_mse_y)
        print('MWRlane_mse_yaw', MWRlane_mse_yaw)
        print(np.sqrt(MWRlane_mse_x), np.sqrt(MWRlane_mse_y), np.sqrt(MWRlane_mse_yaw))

        # mse_x = mean_squared_error(np.asarray(GT_Poses)[:, 0], np.asarray(self.MMSProbe_poses)[:, 1])
        # mse_y = mean_squared_error(np.asarray(GT_Poses)[:, 1], np.asarray(self.MMSProbe_poses)[:, 2])
        # mse_yaw = mean_squared_error(np.asarray(GT_Poses)[:, 2], np.asarray(self.MMSProbe_poses)[:, 3])

        from matplotlib import pyplot as plt
        plt.figure(figsize=(15, 7.5)) #################################
        # plt.figure(figsize=(15, 9.75))
        plt.hlines([0], -10, 200, "Green", linestyles='dashed')

        plt.plot(GPS_x_errors, label='GNSS', color="red")############################
        plt.plot(lane_x_errors, label='Lane', color="orange")
        plt.plot(MWR_x_errors, label='Target', color="royalblue")
        plt.plot(MWRlane_x_errors, label='Target+Lane', color="black")

        plt.legend(fontsize=15)
        plt.xlim(0, step)
        plt.ylim(-1.25, 1.25) # for x errors##########################################
        # plt.ylim(-0.5, 2.75)  # for y errors
        # plt.ylim(-25, 25)  # for yaw errors
        plt.tick_params(direction="in")
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel("Items", fontsize=16)
        plt.ylabel("Errors  m", fontsize=16)#######################################
        # plt.ylabel("Errors  deg", fontsize=16)
        plt.savefig(r'C:\workspace\MELCO\data\results/debug_202312271707/x_errors.svg')##############################
        plt.show()
        ##################################################################################################################################################
        # plt.figure(figsize=(15, 7.5)) #################################
        plt.figure(figsize=(15, 9.75))
        plt.hlines([0], -10, 200, "Green", linestyles='dashed')

        plt.plot(GPS_y_errors, label='GNSS', color="red")############################
        plt.plot(lane_y_errors, label='Lane', color="orange")
        plt.plot(MWR_y_errors, label='Target', color="royalblue")
        plt.plot(MWRlane_y_errors, label='Target+Lane', color="black")

        plt.legend(fontsize=15)
        plt.xlim(0, step)
        # plt.ylim(-1.25, 1.25) # for x errors##########################################
        plt.ylim(-0.5, 2.75)  # for y errors
        # plt.ylim(-25, 25)  # for yaw errors
        plt.tick_params(direction="in")
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel("Items", fontsize=16)
        plt.ylabel("Errors  m", fontsize=16)#######################################
        # plt.ylabel("Errors  deg", fontsize=16)
        plt.savefig(r'C:\workspace\MELCO\data\results/debug_202312271707/y_errors.svg')##############################
        # plt.plot(yaw_errors)

        ##################################################################################################################################################
        plt.figure(figsize=(15, 7.5)) #################################
        # plt.figure(figsize=(15, 9.75))
        plt.hlines([0], -10, 200, "Green", linestyles='dashed')

        plt.plot(GPS_yaw_errors, label='GNSS', color="red")############################
        plt.plot(lane_yaw_errors, label='Lane', color="orange")
        plt.plot(MWR_yaw_errors, label='Target', color="royalblue")
        plt.plot(MWRlane_yaw_errors, label='Target+Lane', color="black")

        plt.legend(fontsize=15)
        plt.xlim(0, step)
        # plt.ylim(-1.25, 1.25) # for x errors##########################################
        # plt.ylim(-0.5, 2.75)  # for y errors
        plt.ylim(-25, 25)  # for yaw errors
        plt.tick_params(direction="in")
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel("Items", fontsize=16)
        # plt.ylabel("Errors  m", fontsize=16)#######################################
        plt.ylabel("Errors  deg", fontsize=16)
        plt.savefig(r'C:\workspace\MELCO\data\results/debug_202312271707/yaw_errors.svg')##############################
        # plt.plot(yaw_errors)
        plt.show()

if __name__ == '__main__':
    ea = ErrorAnalyzer()
    ea.run()
