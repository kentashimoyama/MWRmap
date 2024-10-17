import csv
import os
import numpy as np
import matplotlib.pyplot as plt

# 直線のフィット関数
def fit_line(points):
    x = points[:, 0]
    y = points[:, 1]
    A = np.vstack([x, np.ones(len(x))]).T
    k, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return k, b

# 点と直線の距離を計算
def distance_point_to_line(point, line):
    k, b = line
    x, y = point
    d = abs(k * x - y + b) / np.sqrt(k ** 2 + 1)
    return d

def ransac(points, iterations=100, threshold=0.2): #詳細はracsacアルゴリズムで検索　
    best_model = None
    best_inliers = []
    for i in range(iterations):
        if len(points) <= 3: #点の数3個以下
            return 0, []

        if len(points) >= 5: #点の数5個以上 #2, 3つのシンボルをランダムに選択
            maybe_inliers = np.random.choice(len(points), 3, replace=False)
        else: #点の数4個
            maybe_inliers = np.random.choice(len(points), 2, replace=False) 
        maybe_model = fit_line(points[maybe_inliers]) #直線を作成
        
        #tennis court pole reference 喜久井町のみ
        x_values = [-10122.88941, -10127.43841, -10131.28541, -10134.92941, -10138.66841, -10143.88741]
        y_values = [-32697.68842, -32698.86942, -32703.57142, -32708.26742, -32712.79242, -32719.01442]
        # plt.plot(x_values, y_values, color='red', lw=4,  label='reference')

        #mwr
        x = points[maybe_inliers][:, 0]
        y = points[maybe_inliers][:, 1]
        plt.scatter(x, y, color='black', s=50, label='MWR symbols')

        #tennis court pole generated straitline map 喜久井町のみ
        x_line2 = np.linspace(-10145.12348, -10124.86966, 100) #100分割
        y_line2 = 1.240033538 * x_line2 - 20140.42631 #線形方程式に100分割したx座標を代入
        plt.plot(x_line2, y_line2, color='orange', lw=3)

        #k, b
        x_line = np.linspace(min(x), max(x), 100)  # 100分割
        y_line = maybe_model[0] * x_line + maybe_model[1]  # 線形方程式に100分割したx座標を代入
        plt.plot(x_line, y_line, color='yellow', lw=3)

        plt.xlabel('X  m')
        plt.ylabel('Y  m')
        plt.legend()
        # plt.axis('equal')
        plt.xlim(-10145, -10140)  
        plt.ylim(-32720, -32715)
        print(maybe_model[0])
        plt.show()
    
        also_inliers = []
        for index, point in enumerate(points):
            if index not in maybe_inliers:
                if distance_point_to_line(point, maybe_model) < threshold: #直線と各シンボルの距離計算，閾値いないの場合，inlier
                    also_inliers.append(point) 
        if len(also_inliers) > len(best_inliers):
            best_model = fit_line(np.array(also_inliers)) #最もinlierが多いものが最適な直線
            best_inliers = also_inliers

    # best_modelがNoneの場合の例外処理を追加
    if best_model is None:
        return 0, []

    return best_model, best_inliers






# データの読み込み
folder = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\temp_using\frame_mwr+aqloc\filteramp65azi30\symbols_vison_right/"
#folder = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\temp_using\frame_mwr+aqloc\filteramp65azi30\symbols_vison_left/"
#csv_file_path = 'C:\workspace\MELCO\data/20240123_VISON\MWR/240123_104400632_VISON_001/symbols80kabe.csv'
csv_file_path = folder + "points_frame10.csv"
output_csv_file = r'C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\temp_using\wall/'


##以下毎フレームごとの直線近似を描画する用
#tennis court pole reference 喜久井町のみ
x_values = [-10122.88941, -10127.43841, -10131.28541, -10134.92941, -10138.66841, -10143.88741]
y_values = [-32697.68842, -32698.86942, -32703.57142, -32708.26742, -32712.79242, -32719.01442]
# plt.plot(x_values, y_values, color='red', lw=4,  label='reference')
#mwr
with open(csv_file_path, 'r') as file:
    reader = csv.reader(file)
    data = np.array([list(map(float, row)) for row in reader])
#data = data[0:15]
points = data[:, [1, 0]] #X,Yを抽出???
x = points[:, 0]
y = points[:, 1]
plt.scatter(x, y, color='black', s=50, label='MWR symbols')

#tennis court pole generated straitline map 喜久井町のみ
x_line2 = np.linspace(-10145.12348, -10124.86966, 100) #100分割
y_line2 = 1.240033538 * x_line2 - 20140.42631 #線形方程式に100分割したx座標を代入
# plt.plot(x_line2, y_line2, color='orange', lw=3)

plt.xlabel('X  m')
plt.ylabel('Y  m')
plt.legend()
# plt.axis('equal')
# plt.xlim(-10145, -10140)  
# plt.ylim(-32720, -32715)
plt.show()
##



with open(csv_file_path, 'r') as file:
    reader = csv.reader(file)
    data = np.array([list(map(float, row)) for row in reader])

# RANSACで直線近似
iterations = 100
threshold = 0.1

for i in range(0, len(data), 15): #15行おきにデータを抽出
    subset_dataall = data[i:i+15]
    subset_data = subset_dataall[:, [1, 0]] #X,Yを抽出???
    model, inliers = ransac(subset_data, iterations=iterations, threshold=threshold)
    if model == 0:
        continue
    inliers = np.array(inliers)
    # print(f"model:{model}")
    # print(f"inliers:{inliers}")
    x_min = sorted(inliers[:, 0])[0]
    x_max = sorted(inliers[:, 0])[-1]
    print([model[0], model[1], x_min, x_max])

    # 結果をCSVファイルに追記
    with open(output_csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([model[0], model[1], x_min, x_max]) #傾き,y切片 xmin, xmax

    print(f"CSVファイルに追記しました。({i+1}-{i+15}行)")

print("処理が完了しました。")
