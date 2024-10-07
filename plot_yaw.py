# aqloc位置情報から求めたyaw角と車両走行軌跡を描画。

import matplotlib.pyplot as plt
import numpy as np
import csv
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import FancyArrowPatch

with open(r'C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\aqloc\20240907_1_latlon19.csv', 'r', encoding='utf-8_sig') as file:
# with open(r'C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\aqloc\122705_ichimill19.csv', 'r', encoding='utf-8_sig') as file:
    reader = csv.reader(file)
    
    # 配列を用意
    trajectory_x = []
    trajectory_y = []
    yaws = []
    # 行を1つずつ読み込む
    for i, row in enumerate(reader):
        # if float(row[0]) < 90611:
        #     continue
        # elif float(row[0]) > 90730:
        #     continue
        trajectory_y.append(float(row[1]))
        trajectory_x.append(float(row[2]))
        # yaws.append(np.rad2deg(float(row[3]))%180)
        # if row[3] == "NaN":
        #     yaws.append("")
        yaws.append(float(row[3]))

# 車両の姿勢（ラジアン）
# angles = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])  # 各ポイントでの角度

# プロットの設定
plt.figure(figsize=(8, 8))
plt.plot(trajectory_x, trajectory_y, marker='o', label='走行軌跡')

# 矢印の描画
for i in range(len(trajectory_x)):
    # 矢印の始点
    x = trajectory_x[i]
    y = trajectory_y[i]

    arrow_len = 1

    # 矢印の終点を計算
    if yaws[i] == "":
        continue
    end_x = x + arrow_len * np.sin(yaws[i])
    end_y = y + arrow_len * np.cos(yaws[i])
    # print(yaws[i])
    # print(x, y)
    # print(arrow_len * np.sin(yaws[i]), arrow_len * np.cos(yaws[i]))
    # print(end_x, end_y)
    
    # 矢印を描画
    # FancyArrowPatchを使用して矢印を描画
    arrow = FancyArrowPatch((x, y), (end_x, end_y),
                         mutation_scale=15,  # 矢印のサイズ
                         color='red',
                         arrowstyle='->')
    plt.gca().add_patch(arrow)

    

# グラフの設定
plt.ylim(-32730, -32700)
plt.xlim(-10150, -10120)
# plt.ylim(-32740, -32698)
# plt.xlim(-10145, -10115)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('trajectory and vehicle atitude')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.grid(True)
plt.legend()
plt.show()
