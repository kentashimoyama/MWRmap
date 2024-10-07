# カメラパラメータからカメラのロールピッチよーを時系列的に可視化する。

import matplotlib.pyplot as plt
import numpy as np
import csv
import math

# CSVファイルを読み込む
with open(r'C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\imagepoint\cameraroll_pitch_yaw.csv', 'r', encoding='utf-8_sig') as file:
    reader = csv.reader(file)
    
    # 配列を用意
    times = []
    yaws = []
    rolls = []
    pitchs = []
    # 行を1つずつ読み込む
    for i, row in enumerate(reader):
        if float(row[3]) < 180640:
            continue
        elif float(row[3]) > 180726:
            continue
        times.append(float(row[3]))
        yaws.append(float(row[0]))
        rolls.append(float(row[1])) 
        pitchs.append(float(row[2])) 


# plt.figure(figsize=(20, 20))
# plt.plot(times, yaws)
# plt.xlabel('time (s)')
# plt.ylabel('yaw (deg)')
# interval = 60  # 60秒ごと
# xticks = times[::interval]  # 一定間隔で時間を取得
# plt.xticks(xticks) 
# plt.yticks(np.arange(float(min(yaws)), float(max(yaws)), 5)) 
# plt.grid(True)
# plt.show()

fig, axes = plt.subplots(2, 2, tight_layout=True)

# 1つ目のプロット
axes[0, 0].plot(times, yaws)
axes[0, 0].set_xlabel('time (s)')
axes[0, 0].set_ylabel('yaw (deg)')
# axes[0, 0].set_xticks(np.arange(float(min(times)), float(max(times)), 10)) # x軸に100ずつ目盛り
interval = 60  # 60秒ごと
xticks = times[::interval]  # 一定間隔で時間を取得
axes[0, 0].set_xticks(xticks) 
# axes[0, 0].set_ylim(float(min(yaws)), 40) #軸の下限値と上限値設定
axes[0, 0].set_xticks(np.arange(float(min(times)), float(max(times)), 10)) 
axes[0, 0].set_yticks(np.arange(float(min(yaws)), float(max(yaws)), 1)) 
axes[0, 0].grid(True)

# 2つ目のプロット
axes[0, 1].plot(times, pitchs)
axes[0, 1].set_xlabel('time (s)')
axes[0, 1].set_ylabel('pitch (deg)')
# axes[0, 1].set_xticks(np.arange(float(min(times)), float(max(times)), 10))
interval = 60  # 60秒ごと
xticks = times[::interval]  # 一定間隔で時間を取得
axes[0, 1].set_xticks(xticks) 
# axes[0, 1].set_ylim(float(min(pitchs)), 10)
axes[0, 1].set_xticks(np.arange(float(min(times)), float(max(times)), 10))
axes[0, 1].set_yticks(np.arange(float(min(pitchs)), float(max(pitchs)), 1)) # x軸に100ずつ目盛り
axes[0, 1].grid(True)

# 3つ目のプロット
axes[1, 1].plot(times, rolls)
axes[1, 1].set_xlabel('time (s)')
axes[1, 1].set_ylabel('roll (deg)')
# axes[1, 1].set_xticks(np.arange(float(min(times)), float(max(times)), 10))
interval = 60  # 60秒ごと
xticks = times[::interval]  # 一定間隔で時間を取得
axes[1, 1].set_xticks(xticks) 
# axes[1, 1].set_ylim(170, float(max(rolls)))
axes[1, 1].set_xticks(np.arange(float(min(times)), float(max(times)), 10))
axes[1, 1].set_yticks(np.arange(float(min(rolls)), float(max(rolls)), 1)) # x軸に100ずつ目盛り
axes[1, 1].grid(True)

# 4つ目のプロットは空白のままにする
axes[1, 0].axis('off')

plt.show()