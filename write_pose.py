# aqlocの時系列位置情報から、yaw角を推定し時系列で可視化

import matplotlib.pyplot as plt
import numpy as np
import csv
import math

# CSVファイルを読み込む
with open(r'C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\aqloc\20240907_1_latlon19.csv', 'r', encoding='utf-8_sig') as file:
    reader = csv.reader(file)
    
    # 配列を用意
    times = []
    yaws = []
    rolls = []
    pitchs = []
    # 行を1つずつ読み込む
    for i, row in enumerate(reader):
        if float(row[0]) < 90640:
            continue
        elif float(row[0]) > 90726:
            continue
        times.append(float(row[0]))
        yaws.append(np.rad2deg(float(row[3]))%180) #yaw角-179.99999～179.9999°　真北が0°, pitch角-179.99999～179.9999°　局地座標系の水平が0°, roll角-179.99999～179.9999°　局地座標系の水平が0°
        # rolls.append(np.rad2deg(float(row[4]))%180) #yaw角-179.99999～179.9999°　真北が0°, pitch角-179.99999～179.9999°　局地座標系の水平が0°, roll角-179.99999～179.9999°　局地座標系の水平が0°
        # pitchs.append(np.rad2deg(float(row[5]))%180) #yaw角-179.99999～179.9999°　真北が0°, pitch角-179.99999～179.9999°　局地座標系の水平が0°, roll角-179.99999～179.9999°　局地座標系の水平が0°


plt.figure(figsize=(20, 20))
plt.plot(times, yaws)
plt.xlabel('time (s)')
plt.ylabel('yaw (deg)')
interval = 60  # 60秒ごと
xticks = times[::interval]  # 一定間隔で時間を取得
plt.xticks(xticks) 
plt.yticks(np.arange(float(min(yaws)), float(max(yaws)), 5)) 
plt.grid(True)
plt.show()

# fig, axes = plt.subplots(2, 2, tight_layout=True)

# # 1つ目のプロット
# axes[0, 0].plot(times, yaws)
# axes[0, 0].set_xlabel('time (s)')
# axes[0, 0].set_ylabel('yaw (deg)')
# # axes[0, 0].set_xticks(np.arange(float(min(times)), float(max(times)), 10)) # x軸に100ずつ目盛り
# interval = 60  # 60秒ごと
# xticks = times[::interval]  # 一定間隔で時間を取得
# axes[0, 0].set_xticks(xticks) 
# # axes[0, 0].set_ylim(float(min(yaws)), 40) #軸の下限値と上限値設定
# axes[0, 0].set_yticks(np.arange(float(min(yaws)), float(max(yaws)), 1)) 
# axes[0, 0].grid(True)

# # 2つ目のプロット
# axes[0, 1].plot(times, pitchs)
# axes[0, 1].set_xlabel('time (s)')
# axes[0, 1].set_ylabel('pitch (deg)')
# # axes[0, 1].set_xticks(np.arange(float(min(times)), float(max(times)), 10))
# interval = 60  # 60秒ごと
# xticks = times[::interval]  # 一定間隔で時間を取得
# axes[0, 1].set_xticks(xticks) 
# # axes[0, 1].set_ylim(float(min(pitchs)), 10)
# axes[0, 1].set_yticks(np.arange(float(min(pitchs)), float(max(pitchs)), 10)) # x軸に100ずつ目盛り
# axes[0, 1].grid(True)

# # 3つ目のプロット
# axes[1, 1].plot(times, rolls)
# axes[1, 1].set_xlabel('time (s)')
# axes[1, 1].set_ylabel('roll (deg)')
# # axes[1, 1].set_xticks(np.arange(float(min(times)), float(max(times)), 10))
# interval = 60  # 60秒ごと
# xticks = times[::interval]  # 一定間隔で時間を取得
# axes[1, 1].set_xticks(xticks) 
# # axes[1, 1].set_ylim(170, float(max(rolls)))
# axes[1, 1].set_yticks(np.arange(float(min(rolls)), float(max(rolls)), 10)) # x軸に100ずつ目盛り
# axes[1, 1].grid(True)

# 4つ目のプロットは空白のままにする
axes[1, 0].axis('off')

plt.show()