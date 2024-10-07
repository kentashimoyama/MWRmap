import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import statistics
import math
import os


# # CSVファイルの読み込み
# inp_file = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\簡易地図生成\mwr+aqloc\wall_timesort_symbols_0717_tennis_1_withoutlier_right_extracted_2_80azi_inlier.csv"
# outp_file = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\簡易地図生成\mwr+aqloc\wall_xsort_symbols_0717_tennis_1_withoutlier_right_extracted_2_80azi_inlier.csv"

# # データフレームに読み込む
# df = pd.read_csv(inp_file, header=None)

# # 0列目を基準にソート
# df_sorted = df.sort_values(by=0)

# # ソートされたデータフレームを新しいCSVファイルに保存
# df_sorted.to_csv(outp_file, index=False, header=False)

# print(f"ファイル '{outp_file}' にソートされたデータを保存しました。")
# exit()


# csv_list = []
# input_file = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\簡易地図生成\mwr+aqloc\wall_timesort_symbols_0717_tennis_1_withoutlier_right_extracted_2_80azi_inlier.csv"
# output_csv_file = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\簡易地図生成\mwr+aqloc\bbbbbbwall_timesort_symbols_0717_tennis_1_withoutlier_right_extracted_2_80azi_inlier.csv"

# df = pd.read_csv(input_file)
# csv_list = np.array(df)
# csv_list = csv_list[0:381,:]
# csv_list = csv_list[csv_list[:, 3].argsort()]

# with open(output_csv_file, 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(csv_list)
# exit()

#フレーム毎傾きと切片をすべて出力
# input_file = r'C:\Users\kenta shimoyama\Documents\amanolab\melco\簡易地図生成\wall\wall_symbols_0717_tennis_1_withoutlier_right_extracted_2_80azi_inlier.csv'
# #全てのフレームの近似直線を描画
# csv_list = []
# with open(input_file, 'r') as csv_input:
#     reader = csv.reader(csv_input)
#     for i, row in enumerate(reader): #[y切片，傾き，x最小値，x最大値]
#         print(f"{i+1}行目が完了")
#         csv_list.append(row)
#         b = float(row[0])
#         k = float(row[1]) #y=kx+b
#         x_min = float(row[2])
#         x_max = float(row[3])
#         x_line2 = np.linspace(x_min, x_max, 100) #100分割 
#         y_line2 = k * x_line2 + b #線形方程式に100分割したx座標を代入
#         plt.plot(x_line2, y_line2, color='orange', lw=1)
#     plt.show()

# exit()

# #trim average straight map
# def gen_straight_map(input_file):
#     csv_list = []
#     with open(input_file, 'r') as csv_input:
#         reader = csv.reader(csv_input)
#         for i, row in enumerate(reader): #[傾き，y切片, x最小値，x最大値]
#             print(f"{i+1}行目が完了")
#             csv_list.append(row)

#     csv_arr = np.array(csv_list)
#     csv_arr = csv_arr.astype(float)

#     arr_sorted_k = csv_arr[np.argsort(csv_arr[:, 0])]  # 傾きでソート
#     arr_sorted_b = csv_arr[np.argsort(csv_arr[:, 1])]  # y切片でソート
#     le = len(arr_sorted_b[:, 0])

#     a = math.ceil((le * 0.1) - 1)
#     c = math.floor(le - (le * 0.1))
#     arr_trimed_k = arr_sorted_k[a:c, :]
#     arr_trimed_b = arr_sorted_b[a:c, :]

#     k = statistics.mean(list(arr_trimed_k[:, 0]))  # 傾きの平均
#     b = statistics.mean(list(arr_trimed_b[:, 1]))  # y切片の平均

#     list_trimed_x_min_k = list(arr_trimed_k[:, 2])
#     list_trimed_x_min_b = list(arr_trimed_b[:, 2])
#     list_trimed_x_max_k = list(arr_trimed_k[:, 3])
#     list_trimed_x_max_b = list(arr_trimed_b[:, 3])
    
#     list_trimed_x_min_k.extend(list_trimed_x_min_b)
#     list_trimed_x_max_k.extend(list_trimed_x_max_b)
    
#     x_min = min(list_trimed_x_min_k)
#     x_max = max(list_trimed_x_max_k)

#     x_line = np.linspace(x_min, x_max, 100)  # 100分割
#     y_line = k * x_line + b  # 線形方程式に100分割したx座標を代入
#     print(k, b)   

#     # arr_sorted_b = csv_arr[np.argsort(csv_arr[:, 0])]
#     # arr_sorted_k = csv_arr[np.argsort(csv_arr[:, 1])]
#     # le = len(arr_sorted_b[:, 0])

#     # # arr_sorted_b_list = list(arr_sorted_b[:, 0])
#     # # arr_sorted_k_list = list(arr_sorted_k[:, 1])

#     # #arr_sorted_b = list(sorted(csv_arr[:, 0]))
#     # #arr_sorted_k = list(sorted(csv_arr[:, 1]))

#     # a = int((le*0.1)-1)
#     # b = int(le-(le*0.1))
#     # arr_trimed_b = arr_sorted_b[a:b, :]
#     # arr_trimed_k = arr_sorted_k[a:b, :]

#     # b = statistics.mean(list(arr_trimed_b[:, 1]))
#     # k = statistics.mean(list(arr_trimed_k[:, 0]))

#     # list_trimed_x_min_b = list(arr_trimed_b[:, 2])
#     # list_trimed_x_min_k = list(arr_trimed_k[:, 2])
#     # list_trimed_x_max_b = list(arr_trimed_b[:, 3])
#     # list_trimed_x_max_k = list(arr_trimed_k[:, 3])
    
#     # list_trimed_x_min_b.extend(list_trimed_x_min_k)
#     # list_trimed_x_max_b.extend(list_trimed_x_max_k)
    
#     # x_min = min(list_trimed_x_min_b)
#     # x_max = max(list_trimed_x_max_b)

#     # x_line = np.linspace(x_min, x_max, 100) #100分割 
#     # y_line = k * x_line + b #線形方程式に100分割したx座標を代入 
#     # print(k, b)   

#     return x_line, y_line


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

def exe_ransac(input_file):
    iterations = 100
    threshold = 0.1
    with open(input_file, 'r') as csv_input:
        reader = csv.reader(csv_input)
        data = np.array([list(map(float, row)) for row in reader])
    subset_data = data[:, [1, 0]] #X,Yの座標を反転
    #print(subset_data)
    model, inliers = ransac(subset_data, iterations=iterations, threshold=threshold)
    inliers = np.array(inliers)
    x_min = sorted(inliers[:, 0])[0]
    x_max = sorted(inliers[:, 0])[-1]
    k = model[0]
    b = model[1]

    x_line = np.linspace(x_min, x_max, 100) #100分割 
    y_line = k * x_line + b #線形方程式に100分割したx座標を代入    

    return x_line, y_line

#時間で分割
def exe_ransac_timeseg(input_file, ts):
    iterations = 100
    threshold = 0.1
    with open(input_file, 'r') as csv_input:
        reader = csv.reader(csv_input)
        data = np.array([list(map(float, row)) for row in reader])
    subset_data = data[:, [1, 0, 3]] #X,Yの座標を反転, 時刻t
    basetime_id = 0  
    data_list = []
    line_list = []
    for i, j in enumerate(subset_data):
        basetime = subset_data[basetime_id][2]
        if ts > j[2] - basetime:
            data_list.append(j)
            if i+1 == len(subset_data):
                data_array = np.array(data_list)[:, [0, 1]]
                model, inliers = ransac(data_array, iterations=iterations, threshold=threshold)
                inliers = np.array(inliers)
                if len(inliers) != 0:
                    x_min = sorted(inliers[:, 0])[0]
                    x_max = sorted(inliers[:, 0])[-1]
                    k = model[0]
                    b = model[1]
                    x_line = np.linspace(x_min, x_max, 100) #100分割 
                    y_line = k * x_line + b #線形方程式に100分割したx座標を代入    
                    line_list.append([x_line, y_line])

                return line_list
            
        else:
            basetime_id = i

            data_array = np.array(data_list)[:, [0, 1]]
            model, inliers = ransac(data_array, iterations=iterations, threshold=threshold)
            inliers = np.array(inliers)
            x_min = sorted(inliers[:, 0])[0]
            x_max = sorted(inliers[:, 0])[-1]
            k = model[0]
            b = model[1]
            x_line = np.linspace(x_min, x_max, 100) #100分割 
            y_line = k * x_line + b #線形方程式に100分割したx座標を代入    
            line_list.append([x_line, y_line])

            data_list = []



#x座標で分割
def exe_ransac_xseg(input_file, xs):
    iterations = 100
    threshold = 0.1
    with open(input_file, 'r') as csv_input:
        reader = csv.reader(csv_input)
        data = np.array([list(map(float, row)) for row in reader])
    subset_data = data[:, [1, 0]] #X,Yの座標を反転
    basex_id = 0  
    data_list = []
    line_list = []
    for i, j in enumerate(subset_data):
        basex = subset_data[basex_id][1]
        if xs > abs(j[1] - basex):
            data_list.append(j)
            if i+1 == len(subset_data):
                data_array = np.array(data_list)
                model, inliers = ransac(data_array, iterations=iterations, threshold=threshold)
                inliers = np.array(inliers)
                if len(inliers) != 0:
                    x_min = sorted(inliers[:, 0])[0]
                    x_max = sorted(inliers[:, 0])[-1]
                    k = model[0]
                    b = model[1]
                    x_line = np.linspace(x_min, x_max, 100) #100分割 
                    y_line = k * x_line + b #線形方程式に100分割したx座標を代入    
                    line_list.append([x_line, y_line])

                return line_list
            
        else:
            basex_id = i

            data_array = np.array(data_list)
            model, inliers = ransac(data_array, iterations=iterations, threshold=threshold)
            inliers = np.array(inliers)
            x_min = sorted(inliers[:, 0])[0]
            x_max = sorted(inliers[:, 0])[-1]
            k = model[0]
            b = model[1]
            x_line = np.linspace(x_min, x_max, 100) #100分割 
            y_line = k * x_line + b #線形方程式に100分割したx座標を代入    
            line_list.append([x_line, y_line])

            data_list = []

#MWR
#csv_file_path = 'C:\workspace\MELCO\data/20240123_VISON\MWR/240125_142932747_vison012501r/symbols80.csv'  # ファイルパスを適切なものに置き換えてください
#csv_file_path = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\簡易地図生成\mwr+aqloc\testwall_symbols_tennis_extracted.csv" 
#csv_file_path = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\簡易地図生成\mwr+aqloc\wall_timesort_symbols_0717_tennis_1_withoutlier_right_extracted_2_80azi_inlier.csv"
csv_file_path = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\簡易地図生成\mwr+aqloc\wall_xsort_symbols_0717_tennis_1_withoutlier_right_extracted_2_80azi_inlier.csv"
#csv_file_path = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\簡易地図生成\mwr+aqloc\wall_symbols_0717_tennis_1_withoutlier_left_extracted_2_80azi_inlier.csv"
data = pd.read_csv(csv_file_path, header=None)

#MWR直線近似用
#input_file = r'C:\Users\kenta shimoyama\Documents\amanolab\melco\簡易地図生成\wall\testwall_symbols_tennis_extracted.csv'
#input_file = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\簡易地図生成\mwr+aqloc\testwall_symbols_tennis_extracted.csv" #大和田+地面除去
#input_file = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\簡易地図生成\mwr+aqloc\wall_symbols_0717_tennis_1_withoutlier_left_extracted_2_80azi_inlier.csv"
#input_file = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\簡易地図生成\mwr+aqloc\wall_timesort_symbols_0717_tennis_1_withoutlier_right_extracted_2_80azi_inlier.csv"
input_file = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\簡易地図生成\mwr+aqloc\wall_xsort_symbols_0717_tennis_1_withoutlier_right_extracted_2_80azi_inlier.csv"
output_file = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\簡易地図生成\wall\wall_xseg_symbols_0717_tennis_1_withoutlier_right_extracted_2_80azi_inlier.csv"
#output_file = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\簡易地図生成\wall\wall_timeseg_symbols_0717_tennis_1_withoutlier_right_extracted_2_80azi_inlier.csv"
#output_file = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\簡易地図生成\wall\testwall_symbols_tennis_extracted.csv"
#output_file = r"C:\Users\kenta shimoyama\Documents\amanolab\melco\簡易地図生成\wall\wall_symbols_0717_tennis_1_withoutlier_left_extracted_2_80azi_inlier.csv"

#AQLOC
#csv_file_path2 = 'C:\workspace\MELCO\data/20240123_VISON\AQLOC/012501_19.csv'
csv_file_path2 = r'C:\Users\kenta shimoyama\Documents\amanolab\melco\簡易地図生成\aqloc\0717_tennis_1_latlon19.csv'
data2 = pd.read_csv(csv_file_path2, header=None)





# x, yのデータを取得
x = data.iloc[:, 1]  # 2列目をxに対応
y = data.iloc[:, 0]  # 1列目をyに対応

x2 = data2.iloc[:, 2]  # 1列目をxに対応
y2 = data2.iloc[:, 1]  # 0列目をyに対応


plt.scatter(x, y, color='black', s=0.4, label='MWR symbols')
plt.scatter(x2, y2, color='red', s=3, label='Trajectory')

#tennis court pole reference 喜久井町のみ
x_values = [-10122.88941, -10127.43841, -10131.28541, -10134.92941, -10138.66841, -10143.88741]
y_values = [-32697.68842, -32698.86942, -32703.57142, -32708.26742, -32712.79242, -32719.01442]
plt.plot(x_values, y_values, color='red', lw=4,  label='reference')
#plt.plot(x_values2, y_values2, color='red', lw=4)

#tennis court pole generated straitline map 喜久井町のみ
x_line2 = np.linspace(-10145.12348, -10124.86966, 100) #100分割
y_line2 = 1.240033538 * x_line2 - 20140.42631 #線形方程式に100分割したx座標を代入
plt.plot(x_line2, y_line2, color='orange', lw=3)



#直線地図を生成し描画
##時間で区切る
# ts = 8
# line_list = exe_ransac_timeseg(input_file, ts)
# print(len(line_list))
# for i in line_list:
#     x_line, y_line = i
#     plt.plot(x_line, y_line, color='yellow', lw=3)

##x座標で区切る
xs = 2
line_list = exe_ransac_xseg(input_file, xs)
print(len(line_list))
for i in line_list:
    x_line, y_line = i
    plt.plot(x_line, y_line, color='yellow', lw=3)

# 結果をCSVファイルに追記
if os.path.exists(output_file):
    os.remove(output_file)
with open(output_file, 'w', newline='') as file:
    for l in line_list:
        x_line, y_line = l
        for i, j in enumerate(x_line):
            writer = csv.writer(file)
            writer.writerow([x_line[i], y_line[i]]) #x座標, y座標

# # 全部一気に
# x_line, y_line = exe_ransac(input_file)
# #x_line, y_line = gen_straight_map(input_file)
# plt.plot(x_line, y_line, color='yellow', lw=3)
# # 結果をCSVファイルに追記
# if os.path.exists(output_file):
#     os.remove(output_file)
# with open(output_file, 'w', newline='') as file:
#     for i, j in enumerate(x_line):
#         writer = csv.writer(file)
#         writer.writerow([x_line[i], y_line[i]]) #x座標, y座標



#original reference???
# y_values = [-32699.33842, -32700.51942, -32705.22142, -32709.91742, -32714.44242, -32720.66442]
# x_values = [-10122.86071, -10127.40971, -10131.25671, -10134.90071, -10138.63971, -10143.85871]
# x_values2 = [-10126.0074, -10116.8466]
# y_values2 = [-32739.1864, -32719.9844]

#control building wall reference
# x_values2 = [-10126.0361, -10116.8753]
# y_values2 = [-32737.5364, -32718.3344]



#vison wall
# x1_line = np.linspace(47415.86818, 47454.66952, 100)
# y1_line = 0.30503694146117 * x1_line - 185063.980842072
# plt.plot(x1_line, y1_line, color='orange', lw=3, label='approximate straight line')

# x2_line = np.linspace(47403.66981, 47416.60928, 100)
# y2_line = -2.77363066721389 * x2_line - 39108.254781585
# plt.plot(x2_line, y2_line, color='orange', lw=3)

# x3_line = np.linspace(47375, 47402.64904, 100)#47395.16571
# y3_line = 0.335158633618352 * x3_line -186516.835124765
# plt.plot(x3_line, y3_line, color='orange', lw=3)

# x4_line = np.linspace(47413.80116, 47420.3257, 100)
# y4_line = -3.65065533563694 * x4_line + 2492.29579128973
# plt.plot(x4_line, y4_line, color='orange', lw=3)

# x5_line = np.linspace(47415.94283, 47426.58557, 100)
# y5_line = 0.281261979677639 * x5_line -183943.124271905
# plt.plot(x5_line, y5_line, color='orange', lw=3)

#control building wall
# x_line = np.linspace(-10124.88774, -10115.96426, 100)
# y_line = 1.70431953608241 * x_line - 15477.4698422945
# plt.plot(x_line, y_line, color='orange', lw=3, label='approximate straight line')

#全フレームの近似直線を統合した直線地図を描画
# input_file = r'C:\Users\shimo\PycharmProjects\MELCO引継ぎ用\簡易地図生成\wall\kabe.csv' #壁付近のデータのみ収集
# x_line, y_line = gen_straight_map(input_file)
# plt.plot(x_line, y_line, color='black', lw=3)







#plt.title('approximate straight line')
# plt.xlim(-10155, -10120)  # x軸の範囲を0から5に設定
# plt.ylim(-32723, -32695) # y軸の範囲を0から20に設定
# plt.xlim(-32720, -32715)  
# plt.ylim(-10200, -10100)

plt.xlabel('X  m')
plt.ylabel('Y  m')
plt.legend()

# plt.axis('equal')
plt.show()
    