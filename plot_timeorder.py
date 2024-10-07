import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# CSVファイルを読み込む

a = "wall_xsort_symbols_0717_tennis_1_withoutlier_left_extracted_2_80azi_inlier.csv"
#a = "wall_timesort_symbols_0717_tennis_1_withoutlier_right_extracted_2_80azi_inlier.csv"

df = pd.read_csv(r"C:\Users\kenta shimoyama\Documents\amanolab\melco\簡易地図生成\mwr+aqloc/"+a, header=None)

# データを時系列でソート
df = df.sort_values(by=3)
df = df[[1, 0]]
df.columns = ["x", "y"]

# プロットの初期設定
fig, ax = plt.subplots()
scat = ax.scatter([], [], s=10)

# 軸の範囲を設定（必要に応じて変更）
ax.set_xlim(df["x"].min() - 1, df["x"].max() + 1)
ax.set_ylim(df["y"].min() - 1, df["y"].max() + 1)

def init():
    scat.set_offsets(np.empty((0, 2)))
    return scat,

def update(frame):
    # 指定したフレームまでのデータをプロット
    current_data = df.iloc[:frame]
    scat.set_offsets(current_data[['x', 'y']].values)
    return scat,

# アニメーションの設定
ani = animation.FuncAnimation(fig, update,interval=100, frames=len(df), init_func=init, blit=True, repeat=False)

# プロットを表示
plt.show()

#アニメーションの保存
# ani.save(r"C:\Users\kenta shimoyama\Documents\amanolab\melco\簡易地図生成\mwr+aqloc/"+a.split(".")[0]+".gif", writer="imagemagick")

