import pandas as pd
import numpy as np
import numpy as np
#import open3d as o3d
import pandas as pd
import matplotlib.pyplot as plt
#from pclpy import pcl

debug = False

def filter_ground_and_vegetation(points, voxel_size=0.05, threshold=5):
    # グリッドのサイズを設定
    grid_size = voxel_size

    # 点群の最小値と最大値を求める
    min_x, min_y, _, _, _, _, _ = np.min(points, axis=0)
    max_x, max_y, _, _, _, _, _ = np.max(points, axis=0)

    # グリッドの範囲を計算
    grid_min_x = min_x
    grid_min_y = min_y
    grid_max_x = max_x
    grid_max_y = max_y

    # グリッドの数を計算
    grid_num_x = int(np.ceil((grid_max_x - grid_min_x) / grid_size)) #少数第二位を切り上げ
    grid_num_y = int(np.ceil((grid_max_y - grid_min_y) / grid_size))

    # グリッド内の点の数を計算
    grid_count = np.zeros((grid_num_x, grid_num_y)) #各gridに含まれる点の数を保存する配列
    points_ingrid = np.empty((grid_num_x, grid_num_y), dtype=object) #各gridの点の座標を格納する配列
    for i in range(grid_num_x):
        for j in range(grid_num_y):
            points_ingrid[i, j] = [] #配列をlistで初期化

    for num, point in enumerate(points):
        print(f"generating grid ...  {num+1}/{points.shape[0]}")
        x, y, amp, t, xrange, yrange, velocity = point
        grid_x = int((x - grid_min_x) // grid_size) #x方向grid index. 1つめのgridのインデックスは0．
        grid_y = int((y - grid_min_y) // grid_size) #y方向grid index. 1つめのgridのインデックスは0．
        if 0 <= grid_x < grid_num_x and 0 <= grid_y < grid_num_y:
            grid_count[grid_x, grid_y] += 1 #gridを表す配列に各gridの点の個数を保存．
            # points_ingrid[grid_x, grid_y].append([x, y]) #x, y座標を保存. 2次元配列の1要素が1グリッドにあたる． [[ [[x,y], [x,y]] ], [ [] ], [ [[x,y]] ]]
            points_ingrid[grid_x, grid_y].append([x, y, amp, t, xrange, yrange, velocity]) #x, y座標を保存. 2次元配列の1要素が1グリッドにあたる． [[ [[x,y,amp,t], [x,y,amp,t]] ], [ [] ], [ [[x,y,amp,t]] ]]


    indices = np.where(grid_count >= threshold) #閾値より大きい??　小さい??  gridのindexのみ抽出．出力：[[行index1, 行index2, ...], [列index1, 列index2, ...]]
    #print(points_ingrid.shape)

    filtered_points_list = points_ingrid[list(indices[0]), list(indices[1])] #1次元numpy配列で出力される. 抽出されたgridに入っている点の座標 [　[[x, y, amp, t], [x,y,amp,t]], [x,y,amp,t], [[x, y,amp,t], [x,y,amp,t],　...　] 
    print(f"extracted grids: {filtered_points_list.shape[0]}")

    filtered_ps_list = []
    for i in range(filtered_points_list.shape[0]):
        for filtered_p_list in filtered_points_list[i]:
            filtered_ps_list.append(filtered_p_list)
    filtered_points = np.array(filtered_ps_list)
    filtered_points = filtered_points[filtered_points[:, 3].argsort()] #unixtimeの列を基準にソート

    # 直線を描画 open3dを使用。
    #line_set = grid_line(grid_min_x, grid_min_y ,grid_max_x, grid_max_y, grid_num_x, grid_num_y, grid_size)

    return filtered_points

def amp_azimath_th(df, th):
    index_list = []
    df.columns = ["X_cor", "Y_cor", "Amplitude", "UnixTime", "Xrange", "Yrange", "Velocity[m/s]"]
    amp = df["Amplitude"]
    amp_list = list(amp)
    for i ,j in enumerate(amp_list):
        print(f"processing amp ... {i+1}")
        if j > th:
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

def sep_8sec(input_file):
    data = pd.read_csv(input_file)

    # unixtimeを取得
    unixtime = data.iloc[:, 3].values  # 3列目のunixtimeを取得

    # 8秒ごとにデータを分割するための配列を初期化
    split_data = []

    # データを8秒ごとに分割
    start_time = unixtime[0]
    end_time = start_time + 8

    current_chunk = []

    for index, time in enumerate(unixtime):
        if time < end_time:
            current_chunk.append(data.iloc[index].values)  # 現在のデータを追加
        else:
            split_data.append(np.array(current_chunk))  # 現在のチャンクを配列に変換して追加
            current_chunk = [data.iloc[index].values]  # 新しいチャンクを開始
            start_time = time
            end_time = start_time + 8

    # 最後のチャンクを追加（残っている場合）
    if current_chunk:
        split_data.append(np.array(current_chunk))

    if debug:
        # 結果の確認
        for i, chunk in enumerate(split_data):
            print(f"Chunk {i}:")
            print(chunk)
    
    return split_data

# CSVファイルを読み込む
input_file_list = [r"C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\temp_using\mwr+aqloc/symbols_vison_right.csv", r"C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\temp_using\mwr+aqloc/symbols_vison_left.csv"]
for input_file in input_file_list:
    split_data = sep_8sec(input_file)
    filtered_points_all = []
    
    for i, data in enumerate(split_data):
        print(f"processing {i}/{len(split_data)} ...")
        # 0列目と1列目のデータを抽出する
        df = pd.DataFrame(data)
        extracted_columns = df.iloc[:, [0, 1]] #dataframe型 x, y

        # 抽出したデータを表示
        #print(extracted_columns)
        # world_pos = np.array(extracted_columns) #[[x, y], [x, y], [x, y]...]
        world_pos = np.array(df)  #[[x, y, amp, t, Xrange, Yrange, Velocity[m/s]], [x, y, amp, t, , Xrange, Yrange, Velocity[m/s]], [x, y, amp, t, , Xrange, Yrange, Velocity[m/s]]...]

        # print(world_pos)

        #open3dによる元点群の可視化
        # pcd_origin = o3d.geometry.PointCloud()
        # pcd_origin.points = o3d.utility.Vector3dVector(world_pos)
        # pcd_origin.paint_uniform_color([1, 0, 0])
        #o3d.visualization.draw_geometries([pcd])

        # 点群のフィルタリング
        th = 2
        filtered_points = filter_ground_and_vegetation(world_pos, voxel_size=0.05, threshold=th) #gridあたり何点以上 大体閾値は2
        
        # フィルタリング結果を表示
        print(f"extracted points: {filtered_points.shape[0]}")

        #　保存
        filtered_points_df = pd.DataFrame(filtered_points)
        # filtered_points_df.to_csv(r"C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\temp_using\frame_mwr+aqloc\filter\" + input_file.split("/")[1].split(".")[0] + "/points_frame" + str(i+1) + ".csv", mode='w', header=False, index=False)
        filtered_points_df.to_csv(
        r"C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\temp_using\frame_mwr+aqloc\filter\\" + 
        input_file.split("/")[1].split(".")[0] + 
        "\\points_frame" + str(i + 1) + ".csv", 
        mode='w', 
        header=False, 
        index=False
        )
        # 縦に連結
        if i == 0:
            filtered_points_all = filtered_points
        else:
            filtered_points_all = np.vstack((filtered_points_all, filtered_points))

        # amp aziによるフィルタリング
        ampth = 65
        ampazifiltered_points_df = amp_azimath_th(pd.DataFrame(filtered_points), ampth)

        # 保存
        # ampazifiltered_points_df.to_csv(r"C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\temp_using\frame_mwr+aqloc\filterampazi\" + input_file.split("/")[1].split(".")[0] + "/points_frame" + str(i+1) + ".csv", mode='w', header=False, index=False)
        ampazifiltered_points_df.to_csv(
        r"C:\Users\kenta shimoyama\Documents\amanolab\melco\generate_mvp\temp_using\frame_mwr+aqloc\filterampazi\\" + 
        input_file.split("/")[1].split(".")[0] + 
        "\\points_frame" + str(i + 1) + ".csv", 
        mode='w', 
        header=False, 
        index=False
        )

        ampazifiltered_points_array = ampazifiltered_points_df.iloc[:, :].to_numpy()

        # 縦に連結
        if i == 0:
            ampazifiltered_points_array_all = ampazifiltered_points_array
        else:
            ampazifiltered_points_array_all = np.vstack((ampazifiltered_points_array_all, ampazifiltered_points_array))

    # print(filtered_points_all)
    extracted_mwr_df = pd.DataFrame(filtered_points_all)
    extracted_mwr_df.to_csv(input_file.split(".")[0]+"_extracted_"+str(th)+".csv"  , index=False, header=False)

    ampth = 80
    ampazifiltered_points_array_all_df = pd.DataFrame(ampazifiltered_points_array_all)
    inlier_ampazi_df = amp_azimath_th(ampazifiltered_points_array_all_df, ampth)
    inlier_ampazi_df.to_csv(input_file.split(".")[0]+"_extracted_"+str(th)+"_"+str(ampth)+"azi_inlier.csv", mode='w', header=True, index=False)

        

        
