import numpy as np
from MMSProbe.utils.IO.LaneLine import MapLane
from MMSProbe.utils.IO.Landmark import Landmark

mapLane = MapLane.read_points(r"E:\workspace\MMSProbe_2022/data/KTF/MERGED_roadLine.csv")
landmark = Landmark.read_points(r'E:\workspace\MMSProbe_2022/data/KTF/MERGED_landmark.csv')
mask = np.random.choice(range(len(mapLane.points)), 10000)
lane = mapLane.points[mask]

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as anm
import pandas as pd
from datetime import datetime, timezone
from tqdm import tqdm

MMS_TIME2UTC = 0
MMS_TIME_DELAY = -9.3 + 9*60*60
GNSS_MMS_OFFSET = np.array([0.2, -0.34])  # MMSを原点として，GNSSアンテナの位置
class World:  ### fig:world_init_add_timespan (1-6行目)
    def __init__(self, debug=False):  # time_span, time_intervalを追加
        self.objects = []
        self.debug = debug

    def append(self, obj):  # オブジェクトを登録するための関数
        self.objects.append(obj)

    def draw(self):  ### fig:world_draw_with_timespan (11, 22-36行目)
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)  # サブプロットを準備
        ax.scatter(landmark.landmarks[:, 0], landmark.landmarks[:, 1], s=30, marker="D", color="lightslategray")
        ax.scatter(lane[:, 0], lane[:, 1], s=6, marker=".", color="gray")
        ax.set_aspect('equal')  # 縦横比を座標の値と一致させる
        ax.set_xlabel("X  m", fontsize=14)  # X軸にラベルを表示
        ax.set_ylabel("Y  m", fontsize=14)  # 同じくY軸に
        # ax.set_xlim(-28360, -28290)
        # ax.set_ylim(-72975, -72950)
        ax.set_xlim(-28350, -28300)
        ax.set_ylim(-72970, -72950)
        ax.tick_params(direction="in", pad=5)
        ax.grid(alpha=0.4)
        elems = []

        if self.debug:
            for i in range(1, 1000): self.one_step(i, elems, ax)
        else:
            self.ani = anm.FuncAnimation(fig, self.one_step, fargs=(elems, ax), frames=1000, interval=100, repeat=False)
            plt.show()

    def one_step(self, i, elems, ax):
        ax.legend()
        if i == 139: plt.savefig(r'E:\workspace\MMSProbe_2022\data\thesis_result_決定版\pic/220217060303.svg')
        while elems: elems.pop().remove()
        for obj in self.objects:
            obj.draw(ax, elems)
            if hasattr(obj, "one_step"): obj.one_step(i)

class MMSProbe:
    def __init__(self):
        # 303
        GPSstart = 337
        GPSend = 494

        lanestart = 344
        laneend = 501

        MWRstart = 336
        MWRend = 493

        MWRlanestart = 343
        MWRlaneend = 500

        # # 807
        # GPSstart = 278
        # GPSend = 418
        #
        # lanestart = 279
        # laneend = 419
        #
        # MWRstart = 266
        # MWRend = 406
        #
        # MWRlanestart = 279
        # MWRlaneend = 419


        self.GPS_color = "red"
        df_GPS = pd.read_csv(r'E:\workspace\MMSProbe_2022\data\thesis_result_決定版\220217060303_物標+区画線/Maingps_log.csv',
                             header=None, usecols=[0, 1, 2], encoding='utf-8')
        df_GPS = df_GPS[GPSstart:GPSend]
        self.GPS_data = df_GPS[[0, 1, 2]]
        self.GPS_pose = [self.GPS_data.iloc[0][1], self.GPS_data.iloc[0][2]]
        self.GPS_poses = [self.GPS_pose]

        self.MWR_color = "royalblue"
        # df_MMSProbe = pd.read_csv(r'E:\workspace\MMSProbe_2021-20211109\data\results\debug_202208050553\Mainresult.csv',
        #                         header=None,usecols=[0, 1, 2], encoding='utf-8')
        df_MWR = pd.read_csv(r'E:\workspace\MMSProbe_2022\data\thesis_result_決定版\220217060303_物標/Mainresult.csv',
                                  header=None, usecols=[0, 1, 2], encoding='utf-8')
        df_MWR = df_MWR[MWRstart:MWRend]
        self.MWR_data = df_MWR[[0, 1, 2]]
        self.MWR_pose = [self.MWR_data.iloc[0][1], self.MWR_data.iloc[0][2]]
        self.MWR_poses = [self.MWR_pose]

        self.lane_color = "orange"
        # df_MMSProbe = pd.read_csv(r'E:\workspace\MMSProbe_2021-20211109\data\results\debug_202208050553\Mainresult.csv',
        #                         header=None,usecols=[0, 1, 2], encoding='utf-8')
        df_lane = pd.read_csv(r'E:\workspace\MMSProbe_2022\data\thesis_result_決定版\220217060303_区画線/Mainresult.csv',
                                  header=None, usecols=[0, 1, 2], encoding='utf-8')
        df_lane = df_lane[lanestart:laneend]
        self.lane_data = df_lane[[0, 1, 2]]
        self.lane_pose = [self.lane_data.iloc[0][1], self.lane_data.iloc[0][2]]
        self.lane_poses = [self.lane_pose]

        self.MWRlane_color = "black"
        # df_MMSProbe = pd.read_csv(r'E:\workspace\MMSProbe_2021-20211109\data\results\debug_202208050553\Mainresult.csv',
        #                         header=None,usecols=[0, 1, 2], encoding='utf-8')
        df_MWRlane = pd.read_csv(r'E:\workspace\MMSProbe_2022\data\thesis_result_決定版\220217060303_物標+区画線/Mainresult.csv',
                                  header=None, usecols=[0, 1, 2], encoding='utf-8')
        df_MWRlane = df_MWRlane[MWRlanestart:MWRlaneend]
        self.MWRlane_data = df_MWRlane[[0, 1, 2]]
        self.MWRlane_pose = [self.MWRlane_data.iloc[0][1], self.MWRlane_data.iloc[0][2]]
        self.MWRlane_poses = [self.MWRlane_pose]

        # self.MMSProbe_MWR_color = "red"
        # # df_MMSProbe_MWR = pd.read_csv(r'E:\workspace\MMSProbe_2022\data\results\debug_202208050339\Mainresult.csv',
        # #                         header=None,usecols=[0, 1, 2], encoding='utf-8')debug_202210130325_ONLY_MWR
        # df_MMSProbe_MWR = pd.read_csv(r'E:\workspace\MMSProbe_2022\data\results\debug_202210130440_only_MWR\Maingps_raw.csv',
        #                         header=None,usecols=[0, 1, 2], encoding='utf-8')
        # self.MMSProbe_MWR_data = df_MMSProbe_MWR[[0, 1, 2]]
        # self.MMSProbe_MWR_pose = [self.MMSProbe_MWR_data.iloc[0][1], self.MMSProbe_MWR_data.iloc[0][2]]
        # self.MMSProbe_MWR_poses = [self.MMSProbe_MWR_pose]

        self.MMS_TimePose_dict = {}
        date_utc = datetime.strptime("20220217", "%Y%m%d")
        date_utc = date_utc.replace(tzinfo=timezone.utc)
        weekTime = (date_utc.weekday() + 1) % 7 * 86400
        df_MMS = pd.read_csv(r'E:\MELCO\data\20220414\複合実験時ＭＭＳデータ\1105_202202171423\Output\output_202202171423.n_1105_10Hz_PosAttMgn.csv',header=None)
        time_MMS = df_MMS[0]
        time_MMS = date_utc.timestamp() - weekTime + time_MMS + MMS_TIME2UTC + MMS_TIME_DELAY
        for i, time in tqdm(enumerate(time_MMS)):
            angle_GT = df_MMS.iloc[i][9]
            angle_GT = (-angle_GT + 630) % 360
            angle_GT = np.deg2rad(angle_GT)
            self.MMS_TimePose_dict[time] = [time, df_MMS.iloc[i][2], df_MMS.iloc[i][1], angle_GT]
        self.time_MMS_list = list(self.MMS_TimePose_dict.keys())
        self.i_flag = 0
        self.GT_Pose = self.find_GT_Pose(self.MWRlane_data.iloc[0][0])
        self.GT_Poses = [self.GT_Pose]
        self.GT_color = "green"
        self.time = None



    def draw(self, ax, elems):
        self.GPS_poses.append(self.GPS_pose)
        self.MWR_poses.append(self.MWR_pose)
        self.lane_poses.append(self.lane_pose)
        self.MWRlane_poses.append(self.MWRlane_pose)
        self.GT_Poses.append(self.GT_Pose)

        elems += ax.plot([e[0] for e in self.GPS_poses], [e[1] for e in self.GPS_poses],label='GNSS', linewidth=2.5,
                         color=self.GPS_color)
        elems += ax.plot([e[0] for e in self.lane_poses], [e[1] for e in self.lane_poses],label='Lane', linewidth=2.5,
                         color=self.lane_color)
        elems += ax.plot([e[0] for e in self.MWR_poses], [e[1] for e in self.MWR_poses],label='Target', linewidth=2.5,
                         color=self.MWR_color)
        elems += ax.plot([e[0] for e in self.MWRlane_poses], [e[1] for e in self.MWRlane_poses],label='Target+Lane', linewidth=2.5,
                         color=self.MWRlane_color)
        elems += ax.plot([e[0] for e in self.GT_Poses], [e[1] for e in self.GT_Poses],label='Ground Truth', linewidth=2.5,
                         color=self.GT_color)
        # elems += ax.plot([e[0] for e in self.GT_Poses], [e[1] for e in self.GT_Poses],label='Ground Truth',marker='.',markersize=6,
        #                  color=self.GT_color)
        # elems.append(ax.text(-28360, -72950, self.time, fontsize=14))
    def one_step(self, i):
        self.GPS_pose = [self.GPS_data.iloc[i][1], self.GPS_data.iloc[i][2]]
        self.MWR_pose = [self.MWR_data.iloc[i][1], self.MWR_data.iloc[i][2]]
        self.lane_pose = [self.lane_data.iloc[i][1], self.lane_data.iloc[i][2]]
        self.MWRlane_pose = [self.MWRlane_data.iloc[i][1], self.MWRlane_data.iloc[i][2]]
        time_MWRlane = self.MWRlane_data.iloc[i][0]
        self.GT_Pose = self.find_GT_Pose(time_MWRlane)
        print(self.GT_Pose)
        self.time = self.MWRlane_data.iloc[i][0]

    def find_GT_Pose(self, time_MMSProbe):
        # 三次元のオイラー角は線形補間しちゃだめなんだけど一次元ならOK？と信じる
        for i, time in enumerate(self.time_MMS_list[self.i_flag:]):
            if time <= time_MMSProbe < self.time_MMS_list[self.i_flag + i + 1]:
                self.i_flag += i
                alpha = (time_MMSProbe - time) / (self.time_MMS_list[self.i_flag + 1] - time)
                GT_TimePose = (1 - alpha) * np.asarray(self.MMS_TimePose_dict[self.time_MMS_list[self.i_flag]]) \
                                + alpha * np.asarray(self.MMS_TimePose_dict[self.time_MMS_list[self.i_flag + 1]])
                angel = GT_TimePose[3]
                R = np.asarray([[np.cos(angel), -np.sin(angel)],
                               [np.sin(angel), np.cos(angel)]])
                GT_TimePose[1:3] = GT_TimePose[1:3] + R @ GNSS_MMS_OFFSET
                return GT_TimePose[1:4].tolist()

if __name__ == '__main__':
    world = World()  ### fig:class_world3
    MMS = MMSProbe()
    world.append(MMS)
    world.draw()


